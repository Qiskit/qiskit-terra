// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::BitType;
use hashbrown::HashMap;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

/// Private wrapper for Python-side Bit instances that implements
/// [Hash] and [Eq], allowing them to be used in Rust hash-based
/// sets and maps.
///
/// Python's `hash()` is called on the wrapped Bit instance during
/// construction and returned from Rust's [Hash] trait impl.
/// The impl of [PartialEq] first compares the native Py pointers
/// to determine equality. If these are not equal, only then does
/// it call `repr()` on both sides, which has a significant
/// performance advantage.
#[derive(Clone, Debug)]
struct BitAsKey {
    /// Python's `hash()` of the wrapped instance.
    hash: isize,
    /// The wrapped instance.
    bit: PyObject,
}

impl BitAsKey {
    pub fn new(bit: &Bound<PyAny>) -> Self {
        BitAsKey {
            // This really shouldn't fail, but if it does,
            // we'll just use 0.
            hash: bit.hash().unwrap_or(0),
            bit: bit.clone().unbind(),
        }
    }
}

impl Hash for BitAsKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_isize(self.hash);
    }
}

impl PartialEq for BitAsKey {
    fn eq(&self, other: &Self) -> bool {
        self.bit.is(&other.bit)
            || Python::with_gil(|py| {
                self.bit
                    .bind(py)
                    .repr()
                    .unwrap()
                    .eq(other.bit.bind(py).repr().unwrap())
                    .unwrap()
            })
    }
}

impl Eq for BitAsKey {}

#[derive(Clone, Debug)]
pub struct BitData<T> {
    /// The public field name (i.e. `qubits` or `clbits`).
    description: String,
    /// Registered Python bits.
    bits: Vec<PyObject>,
    /// Maps Python bits to native type.
    indices: HashMap<BitAsKey, T>,
    /// The bits registered, cached as a PyList.
    cached: Py<PyList>,
}

impl<T> BitData<T>
where
    T: From<BitType> + Copy,
    BitType: From<T>,
{
    pub fn new(py: Python<'_>, description: String) -> Self {
        BitData {
            description,
            bits: Vec::new(),
            indices: HashMap::new(),
            cached: PyList::empty_bound(py).unbind(),
        }
    }

    pub fn with_capacity(py: Python<'_>, description: String, capacity: usize) -> Self {
        BitData {
            description,
            bits: Vec::with_capacity(capacity),
            indices: HashMap::with_capacity(capacity),
            cached: PyList::empty_bound(py).unbind(),
        }
    }

    /// Gets the number of bits.
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Gets a reference to the underlying vector of Python bits.
    #[inline]
    pub fn bits(&self) -> &Vec<PyObject> {
        &self.bits
    }

    /// Gets a reference to the cached Python list, maintained by
    /// this instance.
    #[inline]
    pub fn cached(&self) -> &Py<PyList> {
        &self.cached
    }

    /// Finds the native bit index of the given Python bit.
    #[inline]
    pub fn find(&self, bit: &Bound<PyAny>) -> Option<T> {
        self.indices.get(&BitAsKey::new(bit)).copied()
    }

    /// Map the provided Python bits to their native indices.
    /// An error is returned if any bit is not registered.
    pub fn map_bits<'py>(
        &self,
        bits: impl IntoIterator<Item = Bound<'py, PyAny>>,
    ) -> PyResult<impl Iterator<Item = T>> {
        let v: Result<Vec<_>, _> = bits
            .into_iter()
            .map(|b| {
                self.indices
                    .get(&BitAsKey::new(&b))
                    .copied()
                    .ok_or_else(|| {
                        PyKeyError::new_err(format!(
                            "Bit {:?} has not been added to this circuit.",
                            b
                        ))
                    })
            })
            .collect();
        v.map(|x| x.into_iter())
    }

    /// Map the provided native indices to the corresponding Python
    /// bit instances.
    /// Panics if any of the indices are out of range.
    pub fn map_indices(&self, bits: &[T]) -> impl ExactSizeIterator<Item = &Py<PyAny>> {
        let v: Vec<_> = bits.iter().map(|i| self.get(*i).unwrap()).collect();
        v.into_iter()
    }

    /// Gets the Python bit corresponding to the given native
    /// bit index.
    #[inline]
    pub fn get(&self, index: T) -> Option<&PyObject> {
        self.bits.get(<BitType as From<T>>::from(index) as usize)
    }

    /// Adds a new Python bit.
    pub fn add(&mut self, py: Python, bit: &Bound<PyAny>, strict: bool) -> PyResult<T> {
        if self.bits.len() != self.cached.bind(bit.py()).len() {
            return Err(PyRuntimeError::new_err(
            format!("This circuit's {} list has become out of sync with the circuit data. Did something modify it?", self.description)
            ));
        }
        let idx: BitType = self.bits.len().try_into().map_err(|_| {
            PyRuntimeError::new_err(format!(
                "The number of {} in the circuit has exceeded the maximum capacity",
                self.description
            ))
        })?;
        if self
            .indices
            .try_insert(BitAsKey::new(bit), idx.into())
            .is_ok()
        {
            self.bits.push(bit.into_py(py));
            self.cached.bind(py).append(bit)?;
        } else if strict {
            return Err(PyValueError::new_err(format!(
                "Existing bit {:?} cannot be re-added in strict mode.",
                bit
            )));
        }
        Ok(idx.into())
    }

    pub fn remove_indices<I>(&mut self, py: Python, indices: I) -> PyResult<()>
    where
        I: IntoIterator<Item = T>,
    {
        let mut indices_sorted: Vec<usize> = indices
            .into_iter()
            .map(|i| <BitType as From<T>>::from(i) as usize)
            .collect();
        indices_sorted.sort();

        for index in indices_sorted.into_iter().rev() {
            self.cached.bind(py).del_item(index)?;
            let bit = self.bits.remove(index);
            self.indices.remove(&BitAsKey::new(bit.bind(py)));
        }
        // Update indices.
        for (i, bit) in self.bits.iter().enumerate() {
            self.indices
                .insert(BitAsKey::new(bit.bind(py)), (i as BitType).into());
        }
        Ok(())
    }

    /// Called during Python garbage collection, only!.
    /// Note: INVALIDATES THIS INSTANCE.
    pub fn dispose(&mut self) {
        self.indices.clear();
        self.bits.clear();
    }
}

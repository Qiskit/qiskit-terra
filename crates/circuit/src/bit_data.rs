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

use crate::bit::BitInfo;
use crate::imports::{CLASSICAL_REGISTER, QUANTUM_REGISTER, REGISTER};
use crate::register::{Register, RegisterAsKey};
use crate::{BitType, ToPyBit};
use hashbrown::HashMap;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

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
pub(crate) struct BitAsKey {
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
                    .as_any()
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

#[derive(Clone, Debug)]
pub struct NewBitData<T: From<BitType>, R: Register + Hash + Eq> {
    /// The public field name (i.e. `qubits` or `clbits`).
    description: String,
    /// Registered Python bits.
    bits: Vec<OnceLock<PyObject>>,
    /// Maps Python bits to native type.
    indices: HashMap<BitAsKey, T>,
    /// Maps Register keys to indices
    reg_keys: HashMap<RegisterAsKey, u32>,
    /// Mapping between bit index and its register info
    bit_info: Vec<Option<BitInfo>>,
    /// Registers in the circuit
    registry: Vec<R>,
    /// Registers in Python
    registers: Vec<OnceLock<PyObject>>,
    /// Cached Python bits
    cached_py_bits: OnceLock<Py<PyList>>,
    /// Cached Python registers
    cached_py_regs: OnceLock<Py<PyList>>,
}

impl<T, R> NewBitData<T, R>
where
    T: From<BitType> + Copy + Debug + ToPyBit,
    R: Register<Bit = T>
        + Hash
        + Eq
        + From<(usize, Option<String>)>
        + for<'a> From<&'a [T]>
        + for<'a> From<(&'a [T], String)>,
    BitType: From<T>,
{
    pub fn new(description: String) -> Self {
        NewBitData {
            description,
            bits: Vec::new(),
            indices: HashMap::new(),
            bit_info: Vec::new(),
            registry: Vec::new(),
            registers: Vec::new(),
            cached_py_bits: OnceLock::new(),
            cached_py_regs: OnceLock::new(),
            reg_keys: HashMap::new(),
        }
    }

    pub fn with_capacity(description: String, bit_capacity: usize, reg_capacity: usize) -> Self {
        NewBitData {
            description,
            bits: Vec::with_capacity(bit_capacity),
            indices: HashMap::with_capacity(bit_capacity),
            bit_info: Vec::with_capacity(bit_capacity),
            registry: Vec::with_capacity(reg_capacity),
            registers: Vec::with_capacity(reg_capacity),
            cached_py_bits: OnceLock::new(),
            cached_py_regs: OnceLock::new(),
            reg_keys: HashMap::with_capacity(reg_capacity),
        }
    }

    /// Gets the number of bits.
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Gets the number of registers.
    pub fn len_regs(&self) -> usize {
        self.registry.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Adds a register onto the [BitData] of the circuit.
    pub fn add_register(
        &mut self,
        name: Option<String>,
        size: Option<usize>,
        bits: Option<&[T]>,
    ) -> u32 {
        match (size, bits) {
            (None, None) => panic!("You should at least provide either a size or the bit indices."),
            (None, Some(bits)) => {
                let reg: R = if let Some(name) = name {
                    (bits, name).into()
                } else {
                    bits.into()
                };
                let idx = self.registry.len().try_into().unwrap_or_else(|_| {
                    panic!(
                        "The {} registry in this circuit has reached its maximum capacity.",
                        self.description
                    )
                });
                // Add register info cancel if any qubit is duplicated
                for (bit_idx, bit) in bits.iter().enumerate() {
                    let bit_info = &mut self.bit_info[BitType::from(*bit) as usize];
                    if bit_info.is_some() {
                        panic!(
                            "The bit {:?} is currently assigned to another register.",
                            bit
                        )
                    } else {
                        *bit_info = Some(BitInfo::new(
                            idx,
                            bit_idx.try_into().unwrap_or_else(|_| {
                                panic!(
                                    "The current register exceeds its capacity limit. Bits {}",
                                    reg.len()
                                )
                            }),
                        ))
                    }
                }
                self.reg_keys.insert(reg.as_key().clone(), idx);
                self.registry.push(reg);
                self.registers.push(OnceLock::new());
                idx
            }
            (Some(size), None) => {
                let bits: Vec<T> = (0..size).map(|_| self.add_bit()).collect();
                let reg: R = if let Some(name) = name {
                    (bits.as_slice(), name).into()
                } else {
                    bits.as_slice().into()
                };
                let idx = self.registry.len().try_into().unwrap_or_else(|_| {
                    panic!(
                        "The {} registry in this circuit has reached its maximum capacity.",
                        self.description
                    )
                });
                self.reg_keys.insert(reg.as_key().clone(), idx);
                self.registry.push(reg);
                self.registers.push(OnceLock::new());
                idx
            }
            (Some(_), Some(_)) => {
                panic!("You should only provide either a size or the bit indices, not both.")
            }
        }
    }

    /// Adds a bit index into the circuit's [BitData].
    ///
    /// _**Note:** You cannot add bits to registers once they are added._
    pub fn add_bit(&mut self) -> T {
        let idx: BitType = self.bits.len().try_into().unwrap_or_else(|_| {
            panic!(
                "The number of {} in the circuit has exceeded the maximum capacity",
                self.description
            )
        });
        self.bit_info.push(None);
        self.bits.push(OnceLock::new());
        idx.into()
    }

    /// Retrieves the register info of a bit. Will panic if the index is out of range.
    pub fn get_bit_info(&self, index: T) -> Option<&BitInfo> {
        self.bit_info[BitType::from(index) as usize].as_ref()
    }

    /// Retrieves a register by its index within the circuit
    #[inline]
    pub fn get_register(&self, index: u32) -> Option<&R> {
        self.registry.get(index as usize)
    }

    #[inline]
    pub fn get_register_by_key(&self, key: &RegisterAsKey) -> Option<&R> {
        self.reg_keys
            .get(key)
            .and_then(|idx| self.get_register(*idx))
    }

    /// Checks if a register is in the circuit
    #[inline]
    pub fn contains_register(&self, reg: &R) -> bool {
        self.contains_register_by_key(reg.as_key())
    }

    #[inline]
    pub fn contains_register_by_key(&self, reg: &RegisterAsKey) -> bool {
        self.reg_keys.contains_key(reg)
    }

    // =======================
    //        PyMethods
    // =======================

    /// Finds the native bit index of the given Python bit.
    #[inline]
    pub fn py_find_bit(&self, bit: &Bound<PyAny>) -> Option<T> {
        self.indices.get(&BitAsKey::new(bit)).copied()
    }

    /// Gets a reference to the cached Python list, maintained by
    /// this instance.
    #[inline]
    pub fn py_cached_bits(&self, py: Python) -> &Py<PyList> {
        self
            .cached_py_bits
            .get_or_init(|| PyList::empty_bound(py).into())
    }

    /// Gets a reference to the underlying vector of Python bits.
    #[inline]
    pub fn py_bits(&self, py: Python) -> PyResult<Vec<&PyObject>> {
        (0..self.len())
            .map(|idx| {
                self.py_get_bit(py, (idx as u32).into())
                    .map(|bit| bit.unwrap())
            })
            .collect::<PyResult<_>>()
    }

    /// Gets a reference to the underlying vector of Python registers.
    #[inline]
    pub fn py_registers(&self, py: Python) -> PyResult<Vec<&PyObject>> {
        (0..self.len_regs() as u32)
            .map(|idx| self.py_get_register(py, idx).map(|reg| reg.unwrap()))
            .collect::<PyResult<_>>()
    }

    /// Map the provided Python bits to their native indices.
    /// An error is returned if any bit is not registered.
    pub fn py_map_bits<'py>(
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
    pub fn py_map_indices(
        &mut self,
        py: Python,
        bits: &[T],
    ) -> PyResult<impl ExactSizeIterator<Item = &Py<PyAny>>> {
        let v: Vec<_> = bits
            .iter()
            .map(|i| -> PyResult<&PyObject> { Ok(self.py_get_bit(py, *i)?.unwrap()) })
            .collect::<PyResult<_>>()?;
        Ok(v.into_iter())
    }

    /// Gets the Python bit corresponding to the given native
    /// bit index.
    #[inline]
    pub fn py_get_bit(&self, py: Python, index: T) -> PyResult<Option<&PyObject>> {
        /*
        For this method we want to make sure a couple of things are done first:

        - First off, the method needs mutable access to all of `BitData` for a couple
          of reasons, but mainly because it needs to be able to initialize the `OnceCell`
          for both the `Bit` instance as well as the register.

          There is a problem with this as it could cause two mutable references to `BitData`.
          How do we solve this? I guess we solved it LOL
         */
        let index_as_usize = BitType::from(index) as usize;
        // First check if the cell is in range if not, return none
        if self.bits.get(index_as_usize).is_none() {
            Ok(None)
        }
        // If the bit has an assigned register, check if it has been initialized.
        else if let Some(bit_info) = self.bit_info[index_as_usize] {
            // If it is not initalized and has a register, initialize the register
            // and retrieve it from there the first time
            if self.bits[index_as_usize].get().is_none() {
                // A register index is guaranteed to exist in the instance of `BitData`.
                let py_reg = self.py_get_register(py, bit_info.register_index())?;
                let res = py_reg.unwrap().bind(py).get_item(bit_info.index())?;
                self.bits[index_as_usize]
                    .set(res.into())
                    .map_err(|_| PyRuntimeError::new_err("Could not set the OnceCell correctly"))?;
                return Ok(self.bits[index_as_usize].get());
            }
            // If it is initialized, just retrieve.
            else {
                return Ok(self.bits[index_as_usize].get());
            }
        } else if let Some(bit) = self.bits[index_as_usize].get() {
            Ok(Some(bit))
        } else {
            self.bits[index_as_usize]
                .set(T::to_py_bit(py)?)
                .map_err(|_| PyRuntimeError::new_err("Could not set the OnceCell correctly"))?;
            Ok(self.bits[index_as_usize].get())
        }
    }

    /// Retrieves a register instance from Python based on the rust description.
    pub fn py_get_register(&self, py: Python, index: u32) -> PyResult<Option<&PyObject>> {
        let index_as_usize = index as usize;
        // First check if the cell is in range if not, return none
        if self.registers.get(index_as_usize).is_none() {
            Ok(None)
        } else if self.registers[index_as_usize].get().is_none() {
            let register = &self.registry[index as usize];
            // Decide the register type based on its key
            let reg_as_key = register.as_key();
            let reg_type = match reg_as_key {
                RegisterAsKey::Register(_) => REGISTER.get_bound(py),
                RegisterAsKey::Quantum(_) => QUANTUM_REGISTER.get_bound(py),
                RegisterAsKey::Classical(_) => CLASSICAL_REGISTER.get_bound(py),
            };
            // Check if any indices have been initialized, if such is the case
            // Treat the rest of indices as new `Bits``
            if register
                .bits()
                .any(|bit| self.bits[BitType::from(bit) as usize].get().is_some())
            {
                let bits: Vec<PyObject> = register
                    .bits()
                    .map(|bit| -> PyResult<PyObject> {
                        if let Some(bit_obj) = self.bits[BitType::from(bit) as usize].get() {
                            Ok(bit_obj.clone_ref(py))
                        } else {
                            T::to_py_bit(py)
                        }
                    })
                    .collect::<PyResult<_>>()?;

                // Extract kwargs
                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("name", register.name())?;
                kwargs.set_item("bits", bits)?;

                // Create register and assign to OnceCell
                let reg = reg_type.call((), Some(&kwargs))?;
                self.registers[index_as_usize]
                    .set(reg.into())
                    .map_err(|_| PyRuntimeError::new_err("Could not set the OnceCell correctly"))?;
                Ok(self.registers[index_as_usize].get())
            } else {
                let reg = reg_type.call1((register.len(), register.name()))?;
                self.registers[index_as_usize]
                    .set(reg.into())
                    .map_err(|_| PyRuntimeError::new_err("Could not set the OnceCell correctly"))?;
                Ok(self.registers[index_as_usize].get())
            }
        } else {
            Ok(self.registers[index_as_usize].get())
        }
    }

    /// Adds a new Python bit.
    ///
    /// _**Note:** If this Bit has register information, it will not be reflected unless
    /// the Register is also added._
    pub fn py_add_bit(&mut self, bit: &Bound<PyAny>, strict: bool) -> PyResult<T> {
        let py: Python<'_> = bit.py();

        if self.bits.len()
            != self
                .cached_py_bits
                .get_or_init(|| PyList::empty_bound(py).into())
                .bind(bit.py())
                .len()
        {
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
            self.bit_info.push(None);
            self.bits.push(bit.into_py(py).into());
            self.cached_py_bits
                .get_or_init(|| PyList::empty_bound(py).into())
                .bind(py)
                .append(bit)?;
            // self.cached.bind(py).append(bit)?;
        } else if strict {
            return Err(PyValueError::new_err(format!(
                "Existing bit {:?} cannot be re-added in strict mode.",
                bit
            )));
        }
        Ok(idx.into())
    }

    pub fn py_add_register(&mut self, register: &Bound<PyAny>) -> PyResult<u32> {
        let py = register.py();
        if self.registers.len()
            != self
                .cached_py_regs
                .get_or_init(|| PyList::empty_bound(py).into())
                .bind(py)
                .len()
        {
            return Err(PyRuntimeError::new_err(
            format!("This circuit's {} list has become out of sync with the circuit data. Did something modify it?", self.description)
            ));
        }

        // let index: u32 = self.registers.len().try_into().map_err(|_| {
        //     PyRuntimeError::new_err(format!(
        //         "The number of {} registers in the circuit has exceeded the maximum capacity",
        //         self.description
        //     ))
        // })?;

        let bits: Vec<T> = register
            .iter()?
            .map(|bit| -> PyResult<T> {
                let bit = bit?;
                if let Some(idx) = self.indices.get(&BitAsKey::new(&bit)) {
                    Ok(*idx)
                } else {
                    self.py_add_bit(&bit, true)
                }
            })
            .collect::<PyResult<_>>()?;

        let name: String = register.getattr("name")?.extract()?;
        self.registers.push(register.clone().unbind().into());
        self.cached_py_regs
            .get_or_init(|| PyList::empty_bound(py).into())
            .bind(py)
            .append(register)?;
        Ok(self.add_register(Some(name), None, Some(&bits)))
    }

    pub fn py_remove_bit_indices<I>(&mut self, py: Python, indices: I) -> PyResult<()>
    where
        I: IntoIterator<Item = T>,
    {
        let mut indices_sorted: Vec<usize> = indices
            .into_iter()
            .map(|i| <BitType as From<T>>::from(i) as usize)
            .collect();
        indices_sorted.sort();

        for index in indices_sorted.into_iter().rev() {
            // self.cached.bind(py).del_item(index)?;
            let bit = self
                .py_get_bit(py, (index as BitType).into())?
                .unwrap()
                .clone_ref(py);
            self.indices.remove(&BitAsKey::new(bit.bind(py)));
            self.bits.remove(index);
            self.bit_info.remove(index);
        }
        // Update indices.
        for i in 0..self.bits.len() {
            let bit = self
                .py_get_bit(py, (i as BitType).into())?
                .unwrap()
                .clone_ref(py);
            self.indices
                .insert(BitAsKey::new(bit.bind(py)), (i as BitType).into());
        }
        Ok(())
    }

    pub(crate) fn py_bits_raw(&self) -> &[OnceLock<PyObject>] {
        &self.bits
    }

    pub(crate) fn py_bits_cached_raw(&self) -> Option<&Py<PyList>> {
        self.cached_py_bits.get()
    }

    pub(crate) fn py_regs_raw(&self) -> &[OnceLock<PyObject>] {
        &self.bits
    }

    pub(crate) fn py_regs_cached_raw(&self) -> Option<&Py<PyList>> {
        self.cached_py_bits.get()
    }

    /// Called during Python garbage collection, only!.
    /// Note: INVALIDATES THIS INSTANCE.
    pub fn dispose(&mut self) {
        self.indices.clear();
        self.bits.clear();
        self.registers.clear();
        self.bit_info.clear();
        self.registry.clear();
    }

    /// To convert [BitData] into [NewBitData]. If the structure the original comes from contains register
    /// info. Make sure to add it manually after.
    pub fn from_bit_data(py: Python, bit_data: &BitData<T>) -> Self {
        Self {
            description: bit_data.description.clone(),
            bits: bit_data
                .bits
                .iter()
                .map(|bit| bit.clone_ref(py).into())
                .collect(),
            indices: bit_data.indices.clone(),
            reg_keys: HashMap::new(),
            bit_info: (0..bit_data.len()).map(|_| None).collect(),
            registry: Vec::new(),
            registers: Vec::new(),
            cached_py_bits: bit_data.cached().clone_ref(py).into(),
            cached_py_regs: OnceLock::new(),
        }
    }
}

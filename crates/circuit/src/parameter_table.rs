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

use std::cell::OnceCell;

use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use thiserror::Error;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyList, PySet};
use pyo3::{import_exception, intern, PyTraverseError, PyVisit};

use crate::imports::UUID;

import_exception!(qiskit.circuit, CircuitError);

#[derive(Error, Debug)]
pub enum ParameterTableError {
    #[error("parameter '{0:?}' is not tracked in the table")]
    ParameterNotTracked(ParameterUuid),
    #[error("usage {0:?} is not tracked by the table")]
    UsageNotTracked(ParameterUse),
}
impl From<ParameterTableError> for PyErr {
    fn from(value: ParameterTableError) -> PyErr {
        CircuitError::new_err(value.to_string())
    }
}

/// A single use of a symbolic parameter.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum ParameterUse {
    Index { instruction: usize, parameter: u32 },
    GlobalPhase,
}

/// Rust-space extra information that a `ParameterVectorElement` has.  This is used most heavily
/// during sorting; vector elements are sorted by their parent name, and index within that.
#[derive(Clone, Debug)]
struct VectorElement {
    vector_uuid: VectorUuid,
    index: usize,
}

/// Tracked data tied to each parameter's UUID in the table.
#[derive(Clone, Debug)]
pub struct ParameterInfo {
    uses: HashSet<ParameterUse>,
    name: PyBackedStr,
    element: Option<VectorElement>,
    object: Py<PyAny>,
}

/// Rust-space information on a Python `ParameterVector` and its uses in the table.
#[derive(Clone, Debug)]
struct VectorInfo {
    name: PyBackedStr,
    /// Number of elements of the vector tracked within the parameter table.
    refcount: usize,
}

/// Type-safe UUID for a symbolic parameter.  This does not track the name of the `Parameter`; it
/// can't be used alone to reconstruct a Python instance.  That tracking remains only withing the
/// `ParameterTable`.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct ParameterUuid(u128);
impl ParameterUuid {
    /// Extract a UUID from a Python-space `Parameter` object. This assumes that the object is known
    /// to be a parameter.
    pub fn from_parameter(ob: &Bound<PyAny>) -> PyResult<Self> {
        ob.getattr(intern!(ob.py(), "_uuid"))?.extract()
    }
}

/// This implementation of `FromPyObject` is for the UUID itself, which is what the `ParameterUuid`
/// struct actually encapsulates.
impl<'py> FromPyObject<'py> for ParameterUuid {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if ob.is_exact_instance(UUID.get_bound(ob.py())) {
            ob.getattr(intern!(ob.py(), "int"))?.extract().map(Self)
        } else {
            Err(PyTypeError::new_err("not a UUID"))
        }
    }
}

/// Type-safe UUID for a parameter vector.  This is just used internally for tracking.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
struct VectorUuid(u128);
impl VectorUuid {
    /// Extract a UUID from a Python-space `ParameterVector` object. This assumes that the object is
    /// the correct type.
    fn from_vector(ob: &Bound<PyAny>) -> PyResult<Self> {
        ob.getattr(intern!(ob.py(), "_root_uuid"))?.extract()
    }
}
impl<'py> FromPyObject<'py> for VectorUuid {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if ob.is_exact_instance(UUID.get_bound(ob.py())) {
            ob.getattr(intern!(ob.py(), "int"))?.extract().map(Self)
        } else {
            Err(PyTypeError::new_err("not a UUID"))
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct ParameterTable {
    /// Mapping of the parameter key (its UUID) to the information on it tracked by this table.
    by_uuid: HashMap<ParameterUuid, ParameterInfo>,
    /// Mapping of the parameter names to the UUID that represents them.  Since we always get
    /// parameters in as Python-space objects, we use the string object extracted from Python space.
    by_name: HashMap<PyBackedStr, ParameterUuid>,
    /// Additional information on any `ParameterVector` instances that have elements in the circuit.
    vectors: HashMap<VectorUuid, VectorInfo>,
    /// Cache of the sort order of the parameters.  This is lexicographical for most parameters,
    /// except elements of a `ParameterVector` are sorted within the vector by numerical index.  We
    /// calculate this on demand and cache it.
    ///
    /// Any method that adds or removes a parameter needs to invalidate this.
    order_cache: OnceCell<Vec<ParameterUuid>>,
    /// Cache of a Python-space list of the parameter objects, in order.  We only generate this
    /// specifically when asked.
    ///
    /// Any method that adds or removes a parameter needs to invalidate this.
    py_parameters_cache: OnceCell<Py<PyList>>,
}

impl ParameterTable {
    pub fn new() -> Self {
        Default::default()
    }

    /// Get the number of parameters tracked by the table.
    pub fn num_parameters(&self) -> usize {
        self.by_uuid.len()
    }

    /// Add a new usage of a parameter coming in from Python space, optionally adding a first usage
    /// to it.
    ///
    /// The no-use form is useful when doing parameter assignments from Rust space, where the
    /// replacement is itself parametric; the replacement can be extracted once, then subsequent
    /// lookups and updates done without interaction with Python.
    pub fn track(
        &mut self,
        param_ob: &Bound<PyAny>,
        usage: Option<ParameterUse>,
    ) -> PyResult<ParameterUuid> {
        let py = param_ob.py();
        let uuid = ParameterUuid::from_parameter(param_ob)?;
        match self.by_uuid.entry(uuid) {
            Entry::Occupied(mut entry) => {
                if let Some(usage) = usage {
                    entry.get_mut().uses.insert(usage);
                }
            }
            Entry::Vacant(entry) => {
                let py_name_attr = intern!(py, "name");
                let name = param_ob.getattr(py_name_attr)?.extract::<PyBackedStr>()?;
                if self.by_name.contains_key(&name) {
                    return Err(CircuitError::new_err(format!(
                        "name conflict adding parameter '{}'",
                        &name
                    )));
                }
                let element = if let Ok(vector) = param_ob.getattr(intern!(py, "vector")) {
                    let vector_uuid = VectorUuid::from_vector(&vector)?;
                    match self.vectors.entry(vector_uuid) {
                        Entry::Occupied(mut entry) => entry.get_mut().refcount += 1,
                        Entry::Vacant(entry) => {
                            entry.insert(VectorInfo {
                                name: vector.getattr(py_name_attr)?.extract()?,
                                refcount: 1,
                            });
                        }
                    }
                    Some(VectorElement {
                        vector_uuid,
                        index: param_ob.getattr(intern!(py, "index"))?.extract()?,
                    })
                } else {
                    None
                };
                self.by_name.insert(name.clone(), uuid);
                let mut uses = HashSet::new();
                if let Some(usage) = usage {
                    uses.insert_unique_unchecked(usage);
                };
                entry.insert(ParameterInfo {
                    name,
                    uses,
                    element,
                    object: param_ob.clone().unbind(),
                });
                self.invalidate_cache();
            }
        }
        Ok(uuid)
    }

    /// Untrack one use of a single Python-space `Parameter` object from the table, discarding all
    /// other tracking of that `Parameter` if this was the last usage of it.
    pub fn untrack(&mut self, param_ob: &Bound<PyAny>, usage: ParameterUse) -> PyResult<()> {
        self.remove_use(ParameterUuid::from_parameter(param_ob)?, usage)
            .map_err(PyErr::from)
    }

    /// Lookup the Python parameter object by name.
    pub fn py_parameter_by_name(&self, name: &PyBackedStr) -> Option<&Py<PyAny>> {
        self.by_name
            .get(name)
            .map(|uuid| &self.by_uuid[uuid].object)
    }

    /// Lookup the Python parameter object by uuid.
    pub fn py_parameter_by_uuid(&self, uuid: ParameterUuid) -> Option<&Py<PyAny>> {
        self.by_uuid.get(&uuid).map(|param| &param.object)
    }

    /// Get the (maybe cached) Python list of the sorted `Parameter` objects.
    pub fn py_parameters<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        self.py_parameters_cache
            .get_or_init(|| {
                PyList::new_bound(
                    py,
                    self.order_cache
                        .get_or_init(|| self.sorted_order())
                        .iter()
                        .map(|uuid| self.by_uuid[uuid].object.bind(py).clone()),
                )
                .unbind()
            })
            .bind(py)
            .clone()
    }

    /// Get a Python set of all tracked `Parameter` objects.
    pub fn py_parameters_unsorted<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        PySet::new_bound(py, self.by_uuid.values().map(|info| &info.object))
    }

    /// Get the sorted order of the `ParameterTable`.  This does not access the cache.
    fn sorted_order(&self) -> Vec<ParameterUuid> {
        let mut out = self.by_uuid.keys().copied().collect::<Vec<_>>();
        out.sort_unstable_by_key(|uuid| {
            let info = &self.by_uuid[uuid];
            if let Some(vec) = info.element.as_ref() {
                (&self.vectors[&vec.vector_uuid].name, vec.index)
            } else {
                (&info.name, 0)
            }
        });
        out
    }

    /// Add a use of a parameter to the table.
    pub fn add_use(
        &mut self,
        uuid: ParameterUuid,
        usage: ParameterUse,
    ) -> Result<(), ParameterTableError> {
        self.by_uuid
            .get_mut(&uuid)
            .ok_or(ParameterTableError::ParameterNotTracked(uuid))?
            .uses
            .insert(usage);
        Ok(())
    }

    /// Return a use of a parameter.
    ///
    /// If the last use a parameter is discarded, the parameter is untracked.
    pub fn remove_use(
        &mut self,
        uuid: ParameterUuid,
        usage: ParameterUse,
    ) -> Result<(), ParameterTableError> {
        let Entry::Occupied(mut entry) = self.by_uuid.entry(uuid) else {
            return Err(ParameterTableError::ParameterNotTracked(uuid));
        };
        let info = entry.get_mut();
        if !info.uses.remove(&usage) {
            return Err(ParameterTableError::UsageNotTracked(usage));
        }
        if info.uses.is_empty() {
            self.by_name.remove(&info.name);
            if let Some(vec) = info.element.as_ref() {
                let Entry::Occupied(mut vec_entry) = self.vectors.entry(vec.vector_uuid) else {
                    unreachable!()
                };
                vec_entry.get_mut().refcount -= 1;
                if vec_entry.get().refcount == 0 {
                    vec_entry.remove_entry();
                }
            }
            entry.remove_entry();
            self.invalidate_cache();
        }
        Ok(())
    }

    /// Remove a parameter from the table, returning the tracked uses of it.
    pub fn pop(
        &mut self,
        uuid: ParameterUuid,
    ) -> Result<HashSet<ParameterUse>, ParameterTableError> {
        let info = self
            .by_uuid
            .remove(&uuid)
            .ok_or(ParameterTableError::ParameterNotTracked(uuid))?;
        self.by_name
            .remove(&info.name)
            .expect("each parameter should be tracked by both UUID and name");
        if let Some(element) = info.element {
            self.vectors
                .entry(element.vector_uuid)
                .and_replace_entry_with(|_k, mut vector_info| {
                    vector_info.refcount -= 1;
                    (vector_info.refcount > 0).then_some(vector_info)
                });
        }
        self.invalidate_cache();
        Ok(info.uses)
    }

    /// Clear this table, yielding the Python parameter objects and their uses in sorted order.
    ///
    /// The clearing effect is eager and not dependent on the iteration.
    pub fn drain_ordered(
        &mut self,
    ) -> impl ExactSizeIterator<Item = (Py<PyAny>, HashSet<ParameterUse>)> {
        let order = self
            .order_cache
            .take()
            .unwrap_or_else(|| self.sorted_order());
        let by_uuid = ::std::mem::take(&mut self.by_uuid);
        self.by_name.clear();
        self.vectors.clear();
        self.py_parameters_cache.take();
        ParameterTableDrain {
            order: order.into_iter(),
            by_uuid,
        }
    }

    /// Empty this `ParameterTable` of all its contents.  This does not affect the capacities of the
    /// internal data storage.
    pub fn clear(&mut self) {
        self.by_uuid.clear();
        self.by_name.clear();
        self.vectors.clear();
        self.invalidate_cache();
    }

    fn invalidate_cache(&mut self) {
        self.order_cache.take();
        self.py_parameters_cache.take();
    }

    /// Expose the tracked data for a given parameter as directly as possible to Python space.
    ///
    /// This is only really intended for use in testing.
    pub(crate) fn _py_raw_entry(&self, param: Bound<PyAny>) -> PyResult<Py<PySet>> {
        let py = param.py();
        let uuid = ParameterUuid::from_parameter(&param)?;
        let info = self
            .by_uuid
            .get(&uuid)
            .ok_or(ParameterTableError::ParameterNotTracked(uuid))?;
        // PyO3's `PySet::new_bound` only accepts iterables of references.
        let out = PySet::empty_bound(py)?;
        for usage in info.uses.iter() {
            match usage {
                ParameterUse::GlobalPhase => out.add((py.None(), py.None()))?,
                ParameterUse::Index {
                    instruction,
                    parameter,
                } => out.add((*instruction, *parameter))?,
            }
        }
        Ok(out.unbind())
    }

    /// Accept traversal of this object by the Python garbage collector.
    ///
    /// This is not a pyclass, so it's up to our owner to delegate their own traversal to us.
    pub fn py_gc_traverse(&self, visit: &PyVisit) -> Result<(), PyTraverseError> {
        for info in self.by_uuid.values() {
            visit.call(&info.object)?
        }
        // We don't need to / can't visit the `PyBackedStr` stores.
        if let Some(list) = self.py_parameters_cache.get() {
            visit.call(list)?
        }
        Ok(())
    }
}

struct ParameterTableDrain {
    order: ::std::vec::IntoIter<ParameterUuid>,
    by_uuid: HashMap<ParameterUuid, ParameterInfo>,
}
impl Iterator for ParameterTableDrain {
    type Item = (Py<PyAny>, HashSet<ParameterUse>);

    fn next(&mut self) -> Option<Self::Item> {
        self.order.next().map(|uuid| {
            let info = self
                .by_uuid
                .remove(&uuid)
                .expect("tracked UUIDs should be consistent");
            (info.object, info.uses)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.order.size_hint()
    }
}
impl ExactSizeIterator for ParameterTableDrain {}
impl ::std::iter::FusedIterator for ParameterTableDrain {}

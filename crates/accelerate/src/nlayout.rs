// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::types::PyList;

use hashbrown::HashMap;

/// A newtype for the different categories of qubits used within layouts.  This is to enforce
/// significantly more type safety when dealing with mixtures of physical and virtual qubits, as we
/// typically are when dealing with layouts.  In Rust space, `NLayout` only works in terms of the
/// correct newtype, meaning that it's not possible to accidentally pass the wrong type of qubit to
/// a lookup.  We can't enforce the same rules on integers in Python space without runtime
/// overhead, so we just allow conversion to and from any valid `PyLong`.
macro_rules! qubit_newtype {
    ($id: ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $id(pub u32);

        impl $id {
            #[inline]
            pub fn new(val: u32) -> Self {
                Self(val)
            }
            #[inline]
            pub fn index(&self) -> usize {
                self.0 as usize
            }
        }

        impl pyo3::IntoPy<PyObject> for $id {
            fn into_py(self, py: Python<'_>) -> PyObject {
                self.0.into_py(py)
            }
        }
        impl pyo3::ToPyObject for $id {
            fn to_object(&self, py: Python<'_>) -> PyObject {
                self.0.to_object(py)
            }
        }

        impl pyo3::FromPyObject<'_> for $id {
            fn extract(ob: &PyAny) -> PyResult<Self> {
                Ok(Self(ob.extract()?))
            }
        }

        unsafe impl numpy::Element for $id {
            const IS_COPY: bool = true;

            fn get_dtype_bound(py: Python<'_>) -> Bound<'_, numpy::PyArrayDescr> {
                u32::get_dtype_bound(py)
            }
        }
    };
}

qubit_newtype!(PhysicalQubit);
impl PhysicalQubit {
    /// Get the virtual qubit that currently corresponds to this index of physical qubit in the
    /// given layout.
    pub fn to_virt(self, layout: &NLayout) -> VirtualQubit {
        layout.phys_to_virt[self.index()]
    }
}

qubit_newtype!(VirtualQubit);
impl VirtualQubit {
    /// Get the physical qubit that currently corresponds to this index of virtual qubit in the
    /// given layout.
    pub fn to_phys(self, layout: &NLayout) -> PhysicalQubit {
        layout.virt_to_phys[self.index()]
    }
}

/// An unsigned integer Vector based layout class
///
/// This class tracks the layout (or mapping between virtual qubits in the the
/// circuit and physical qubits on the physical device) efficiently
///
/// Args:
///     qubit_indices (dict): A dictionary mapping the virtual qubit index in the circuit to the
///         physical qubit index on the coupling graph.
///     logical_qubits (int): The number of logical qubits in the layout
///     physical_qubits (int): The number of physical qubits in the layout
#[pyclass(module = "qiskit._accelerate.nlayout")]
#[derive(Clone, Debug)]
pub struct NLayout {
    virt_to_phys: Vec<PhysicalQubit>,
    phys_to_virt: Vec<VirtualQubit>,
}

#[pymethods]
impl NLayout {
    #[new]
    fn new(
        qubit_indices: HashMap<VirtualQubit, PhysicalQubit>,
        virtual_qubits: usize,
        physical_qubits: usize,
    ) -> Self {
        let mut res = NLayout {
            virt_to_phys: vec![PhysicalQubit(u32::MAX); virtual_qubits],
            phys_to_virt: vec![VirtualQubit(u32::MAX); physical_qubits],
        };
        for (virt, phys) in qubit_indices {
            res.virt_to_phys[virt.index()] = phys;
            res.phys_to_virt[phys.index()] = virt;
        }
        res
    }

    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok((
            py.get_type_bound::<Self>()
                .getattr("from_virtual_to_physical")?,
            (self.virt_to_phys.to_object(py),),
        )
            .into_py(py))
    }

    /// Return the layout mapping.
    ///
    /// .. note::
    ///
    ///     This copies the data from Rust to Python and has linear overhead based on the number of
    ///     qubits.
    ///
    /// Returns:
    ///     list: A list of 2 element lists in the form ``[(virtual_qubit, physical_qubit), ...]``,
    ///     where the virtual qubit is the index in the qubit index in the circuit.
    ///
    #[pyo3(text_signature = "(self, /)")]
    fn layout_mapping(&self, py: Python<'_>) -> Py<PyList> {
        PyList::new_bound(py, self.iter_virtual()).into()
    }

    /// Get physical bit from virtual bit
    #[pyo3(text_signature = "(self, virtual, /)")]
    pub fn virtual_to_physical(&self, r#virtual: VirtualQubit) -> PhysicalQubit {
        self.virt_to_phys[r#virtual.index()]
    }

    /// Get virtual bit from physical bit
    #[pyo3(text_signature = "(self, physical, /)")]
    pub fn physical_to_virtual(&self, physical: PhysicalQubit) -> VirtualQubit {
        self.phys_to_virt[physical.index()]
    }

    /// Swap the specified virtual qubits
    #[pyo3(text_signature = "(self, bit_a, bit_b, /)")]
    pub fn swap_virtual(&mut self, bit_a: VirtualQubit, bit_b: VirtualQubit) {
        self.virt_to_phys.swap(bit_a.index(), bit_b.index());
        self.phys_to_virt[self.virt_to_phys[bit_a.index()].index()] = bit_a;
        self.phys_to_virt[self.virt_to_phys[bit_b.index()].index()] = bit_b;
    }

    /// Swap the specified physical qubits
    #[pyo3(text_signature = "(self, bit_a, bit_b, /)")]
    pub fn swap_physical(&mut self, bit_a: PhysicalQubit, bit_b: PhysicalQubit) {
        self.phys_to_virt.swap(bit_a.index(), bit_b.index());
        self.virt_to_phys[self.phys_to_virt[bit_a.index()].index()] = bit_a;
        self.virt_to_phys[self.phys_to_virt[bit_b.index()].index()] = bit_b;
    }

    pub fn copy(&self) -> NLayout {
        self.clone()
    }

    #[staticmethod]
    pub fn generate_trivial_layout(num_qubits: u32) -> Self {
        NLayout {
            virt_to_phys: (0..num_qubits).map(PhysicalQubit).collect(),
            phys_to_virt: (0..num_qubits).map(VirtualQubit).collect(),
        }
    }

    #[staticmethod]
    pub fn from_virtual_to_physical(virt_to_phys: Vec<PhysicalQubit>) -> PyResult<Self> {
        let mut phys_to_virt = vec![VirtualQubit(u32::MAX); virt_to_phys.len()];
        for (virt, phys) in virt_to_phys.iter().enumerate() {
            phys_to_virt[phys.index()] = VirtualQubit(virt.try_into()?);
        }
        Ok(NLayout {
            virt_to_phys,
            phys_to_virt,
        })
    }
}

impl NLayout {
    /// Iterator of `(VirtualQubit, PhysicalQubit)` pairs, in order of the `VirtualQubit` indices.
    pub fn iter_virtual(
        &'_ self,
    ) -> impl ExactSizeIterator<Item = (VirtualQubit, PhysicalQubit)> + '_ {
        self.virt_to_phys
            .iter()
            .enumerate()
            .map(|(v, p)| (VirtualQubit::new(v as u32), *p))
    }
    /// Iterator of `(PhysicalQubit, VirtualQubit)` pairs, in order of the `PhysicalQubit` indices.
    pub fn iter_physical(
        &'_ self,
    ) -> impl ExactSizeIterator<Item = (PhysicalQubit, VirtualQubit)> + '_ {
        self.phys_to_virt
            .iter()
            .enumerate()
            .map(|(p, v)| (PhysicalQubit::new(p as u32), *v))
    }
}

pub fn nlayout(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<NLayout>()?;
    Ok(())
}

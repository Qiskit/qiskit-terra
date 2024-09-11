// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

pub mod bit_data;
pub mod circuit_data;
pub mod circuit_instruction;
pub mod converters;
pub mod dag_circuit;
pub mod dag_node;
mod dot_utils;
mod error;
pub mod gate_matrix;
pub mod imports;
mod interner;
pub mod operations;
pub mod packed_instruction;
pub mod parameter_table;
pub mod slice;
pub mod util;

mod rustworkx_core_vnext;

use pyo3::prelude::*;
use pyo3::types::{PySequence, PyTuple};

pub type BitType = u32;
#[derive(Copy, Clone, Debug, Hash, Ord, PartialOrd, Eq, PartialEq, FromPyObject)]
pub struct Qubit(pub BitType);
#[derive(Copy, Clone, Debug, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct Clbit(pub BitType);

pub struct TupleLikeArg<'py> {
    value: Bound<'py, PyTuple>,
}

impl<'py> FromPyObject<'py> for TupleLikeArg<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let value = match ob.downcast::<PySequence>() {
            Ok(seq) => seq.to_tuple()?,
            Err(_) => PyTuple::new_bound(
                ob.py(),
                ob.iter()?
                    .map(|o| Ok(o?.unbind()))
                    .collect::<PyResult<Vec<PyObject>>>()?,
            ),
        };
        Ok(TupleLikeArg { value })
    }
}

impl From<BitType> for Qubit {
    fn from(value: BitType) -> Self {
        Qubit(value)
    }
}

impl From<Qubit> for BitType {
    fn from(value: Qubit) -> Self {
        value.0
    }
}

impl From<BitType> for Clbit {
    fn from(value: BitType) -> Self {
        Clbit(value)
    }
}

impl From<Clbit> for BitType {
    fn from(value: Clbit) -> Self {
        value.0
    }
}

pub fn circuit(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<circuit_data::CircuitData>()?;
    m.add_class::<circuit_instruction::CircuitInstruction>()?;
    m.add_class::<dag_circuit::DAGCircuit>()?;
    m.add_class::<dag_node::DAGNode>()?;
    m.add_class::<dag_node::DAGInNode>()?;
    m.add_class::<dag_node::DAGOutNode>()?;
    m.add_class::<dag_node::DAGOpNode>()?;
    m.add_class::<operations::StandardGate>()?;
    Ok(())
}

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

mod heuristic;
mod layer;
mod layout;
mod neighbor_table;
mod route;
pub mod sabre_dag;
pub mod swap_map;

use hashbrown::HashMap;
use numpy::{IntoPyArray, ToPyArray};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::IntoPyObjectExt;
use pyo3::Python;

use crate::nlayout::PhysicalQubit;
use neighbor_table::NeighborTable;
use sabre_dag::SabreDAG;
use swap_map::SwapMap;

/// A container for Sabre mapping results.
#[pyclass(module = "qiskit._accelerate.sabre")]
#[derive(Clone, Debug)]
pub struct SabreResult {
    #[pyo3(get)]
    pub map: SwapMap,
    pub node_order: Vec<usize>,
    #[pyo3(get)]
    pub node_block_results: NodeBlockResults,
}

#[pymethods]
impl SabreResult {
    #[getter]
    fn node_order(&self, py: Python) -> PyObject {
        self.node_order.to_pyarray(py).into_any().unbind()
    }
}

#[pyclass(mapping, module = "qiskit._accelerate.sabre")]
#[derive(Clone, Debug)]
pub struct NodeBlockResults {
    pub results: HashMap<usize, Vec<BlockResult>>,
}

#[pymethods]
impl NodeBlockResults {
    // Mapping Protocol
    pub fn __len__(&self) -> usize {
        self.results.len()
    }

    pub fn __contains__(&self, object: usize) -> bool {
        self.results.contains_key(&object)
    }

    pub fn __getitem__(&self, py: Python, object: usize) -> PyResult<PyObject> {
        match self.results.get(&object) {
            Some(val) => Ok(val
                .iter()
                .map(|x| x.clone().into_py_any(py))
                .collect::<PyResult<Vec<_>>>()?
                .into_pyarray(py)
                .into_any()
                .unbind()),
            None => Err(PyIndexError::new_err(format!(
                "Node index {object} has no block results",
            ))),
        }
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.results))
    }
}

#[pyclass(module = "qiskit._accelerate.sabre")]
#[derive(Clone, Debug)]
pub struct BlockResult {
    #[pyo3(get)]
    pub result: SabreResult,
    pub swap_epilogue: Vec<[PhysicalQubit; 2]>,
}

#[pymethods]
impl BlockResult {
    #[getter]
    fn swap_epilogue(&self, py: Python) -> PyResult<PyObject> {
        Ok(self
            .swap_epilogue
            .iter()
            .map(|x| x.into_py_any(py))
            .collect::<PyResult<Vec<_>>>()?
            .into_pyarray(py)
            .into_any()
            .unbind())
    }
}

pub fn sabre(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(route::sabre_routing))?;
    m.add_wrapped(wrap_pyfunction!(layout::sabre_layout_and_routing))?;
    m.add_class::<heuristic::SetScaling>()?;
    m.add_class::<heuristic::Heuristic>()?;
    m.add_class::<heuristic::BasicHeuristic>()?;
    m.add_class::<heuristic::LookaheadHeuristic>()?;
    m.add_class::<heuristic::DecayHeuristic>()?;
    m.add_class::<NeighborTable>()?;
    m.add_class::<SabreDAG>()?;
    m.add_class::<SwapMap>()?;
    m.add_class::<BlockResult>()?;
    m.add_class::<NodeBlockResults>()?;
    m.add_class::<SabreResult>()?;
    Ok(())
}

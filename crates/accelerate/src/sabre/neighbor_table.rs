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

use crate::getenv_use_multiple_threads;
use ndarray::prelude::*;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use rustworkx_core::petgraph::prelude::*;
use smallvec::SmallVec;

use crate::nlayout::PhysicalQubit;

/// A simple container that contains a vector of vectors representing
/// neighbors of each node in the coupling map
///
/// This object is typically created once from the adjacency matrix of
/// a coupling map, for example::
///
///     neigh_table = NeighborTable(rustworkx.adjacency_matrix(coupling_map.graph))
///
/// and used solely to represent neighbors of each node in qiskit-terra's rust
/// module.
#[pyclass(module = "qiskit._accelerate.sabre")]
#[derive(Clone, Debug)]
pub struct NeighborTable {
    // The choice of 4 `PhysicalQubit`s in the stack-allocated region is because a) this causes the
    // `SmallVec<T>` to be the same width as a `Vec` on 64-bit systems (three machine words == 24
    // bytes); b) the majority of coupling maps we're likely to encounter have a degree of 3 (heavy
    // hex) or 4 (grid / heavy square).
    neighbors: Vec<SmallVec<[PhysicalQubit; 4]>>,
}

impl NeighborTable {
    /// Regenerate a Rust-space coupling graph from the table.
    pub fn coupling_graph(&self) -> DiGraph<(), ()> {
        DiGraph::from_edges(self.neighbors.iter().enumerate().flat_map(|(u, targets)| {
            targets
                .iter()
                .map(move |v| (NodeIndex::new(u), NodeIndex::new(v.index())))
        }))
    }

    pub fn num_qubits(&self) -> usize {
        self.neighbors.len()
    }
}

impl std::ops::Index<PhysicalQubit> for NeighborTable {
    type Output = [PhysicalQubit];

    fn index(&self, index: PhysicalQubit) -> &Self::Output {
        &self.neighbors[index.index()]
    }
}

#[pymethods]
impl NeighborTable {
    #[new]
    #[pyo3(signature = (adjacency_matrix=None))]
    pub fn new(adjacency_matrix: Option<PyReadonlyArray2<f64>>) -> PyResult<Self> {
        let run_in_parallel = getenv_use_multiple_threads();
        let neighbors = match adjacency_matrix {
            Some(adjacency_matrix) => {
                let adj_mat = adjacency_matrix.as_array();
                let build_neighbors =
                    |row: ArrayView1<f64>| -> PyResult<SmallVec<[PhysicalQubit; 4]>> {
                        row.iter()
                            .enumerate()
                            .filter_map(|(row_index, value)| {
                                if *value == 0. {
                                    None
                                } else {
                                    Some(match row_index.try_into() {
                                        Ok(index) => Ok(PhysicalQubit::new(index)),
                                        Err(err) => Err(err.into()),
                                    })
                                }
                            })
                            .collect()
                    };
                if run_in_parallel {
                    adj_mat
                        .axis_iter(Axis(0))
                        .into_par_iter()
                        .map(build_neighbors)
                        .collect::<PyResult<_>>()?
                } else {
                    adj_mat
                        .axis_iter(Axis(0))
                        .map(build_neighbors)
                        .collect::<PyResult<_>>()?
                }
            }
            None => Vec::new(),
        };
        Ok(NeighborTable { neighbors })
    }

    fn __getstate__(&self, py: Python<'_>) -> Py<PyList> {
        PyList::new_bound(
            py,
            self.neighbors
                .iter()
                .map(|v| PyList::new_bound(py, v.iter()).to_object(py)),
        )
        .into()
    }

    fn __setstate__(&mut self, state: &Bound<PyList>) -> PyResult<()> {
        self.neighbors = state
            .iter()
            .map(|v| {
                v.downcast::<PyList>()?
                    .iter()
                    .map(|b| b.extract())
                    .collect::<PyResult<_>>()
            })
            .collect::<PyResult<_>>()?;
        Ok(())
    }
}

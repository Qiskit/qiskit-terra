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

use numpy::PyReadonlyArray2;
use pyo3::{
    prelude::*,
    pyfunction,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult,
};
use qiskit_circuit::{circuit_data::CircuitData, operations::Param};

pub(crate) mod cz_depth_lnn;

/// Synthesis of a CZ circuit for linear nearest neighbor (LNN) connectivity,
/// based on Maslov and Roetteler.
///
///  Note that this method *reverts* the order of qubits in the circuit,
///  and returns a circuit containing :class:`.CXGate`\s and phase gates
/// (:class:`.SGate`, :class:`.SdgGate` or :class:`.ZGate`).
///
/// References:
///     1. Dmitri Maslov, Martin Roetteler,
///        *Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations*,
///        `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
#[pyfunction]
#[pyo3(signature = (mat))]
fn synth_cz_depth_line_mr(py: Python, mat: PyReadonlyArray2<bool>) -> PyResult<CircuitData> {
    let view = mat.as_array();
    let (num_qubits, lnn_gates) = cz_depth_lnn::synth_cz_depth_line_mr_inner(view);
    CircuitData::from_standard_gates(py, num_qubits as u32, lnn_gates, Param::Float(0.0))
}

pub fn linear_phase(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(synth_cz_depth_line_mr))?;
    Ok(())
}

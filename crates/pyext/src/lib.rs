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

use pyo3::prelude::*;

use qiskit_accelerate::{
    check_map::check_map_mod, circuit_library::circuit_library,
    commutation_analysis::commutation_analysis, commutation_cancellation::commutation_cancellation,
    commutation_checker::commutation_checker, convert_2q_block_matrix::convert_2q_block_matrix,
    dense_layout::dense_layout, error_map::error_map,
    euler_one_qubit_decomposer::euler_one_qubit_decomposer, filter_op_nodes::filter_op_nodes_mod,
    gate_direction::gate_direction, inverse_cancellation::inverse_cancellation_mod,
    isometry::isometry, nlayout::nlayout, optimize_1q_gates::optimize_1q_gates,
    pauli_exp_val::pauli_expval,
    remove_diagonal_gates_before_measure::remove_diagonal_gates_before_measure, results::results,
    sabre::sabre, sampled_exp_val::sampled_exp_val, sparse_pauli_op::sparse_pauli_op,
    split_2q_unitaries::split_2q_unitaries_mod, star_prerouting::star_prerouting,
    stochastic_swap::stochastic_swap, synthesis::synthesis, target_transpiler::target,
    two_qubit_decompose::two_qubit_decompose, uc_gate::uc_gate, utils::utils,
    vf2_layout::vf2_layout,
};

#[inline(always)]
#[doc(hidden)]
fn add_submodule<F>(m: &Bound<PyModule>, constructor: F, name: &str) -> PyResult<()>
where
    F: FnOnce(&Bound<PyModule>) -> PyResult<()>,
{
    let new_mod = PyModule::new_bound(m.py(), name)?;
    constructor(&new_mod)?;
    m.add_submodule(&new_mod)
}

#[pymodule]
fn _accelerate(m: &Bound<PyModule>) -> PyResult<()> {
    add_submodule(m, qiskit_circuit::circuit, "circuit")?;
    add_submodule(m, qiskit_circuit::converters::converters, "converters")?;
    add_submodule(m, qiskit_qasm2::qasm2, "qasm2")?;
    add_submodule(m, qiskit_qasm3::qasm3, "qasm3")?;
    add_submodule(m, circuit_library, "circuit_library")?;
    add_submodule(m, check_map_mod, "check_map")?;
    add_submodule(m, convert_2q_block_matrix, "convert_2q_block_matrix")?;
    add_submodule(m, dense_layout, "dense_layout")?;
    add_submodule(m, error_map, "error_map")?;
    add_submodule(m, euler_one_qubit_decomposer, "euler_one_qubit_decomposer")?;
    add_submodule(m, inverse_cancellation_mod, "inverse_cancellation")?;
    add_submodule(m, filter_op_nodes_mod, "filter_op_nodes")?;
    add_submodule(m, isometry, "isometry")?;
    add_submodule(m, nlayout, "nlayout")?;
    add_submodule(m, optimize_1q_gates, "optimize_1q_gates")?;
    add_submodule(m, pauli_expval, "pauli_expval")?;
    add_submodule(m, synthesis, "synthesis")?;
    add_submodule(
        m,
        remove_diagonal_gates_before_measure,
        "remove_diagonal_gates_before_measure",
    )?;
    add_submodule(m, results, "results")?;
    add_submodule(m, sabre, "sabre")?;
    add_submodule(m, sampled_exp_val, "sampled_exp_val")?;
    add_submodule(m, sparse_pauli_op, "sparse_pauli_op")?;
    add_submodule(m, split_2q_unitaries_mod, "split_2q_unitaries")?;
    add_submodule(m, star_prerouting, "star_prerouting")?;
    add_submodule(m, stochastic_swap, "stochastic_swap")?;
    add_submodule(m, target, "target")?;
    add_submodule(m, two_qubit_decompose, "two_qubit_decompose")?;
    add_submodule(m, uc_gate, "uc_gate")?;
    add_submodule(m, utils, "utils")?;
    add_submodule(m, vf2_layout, "vf2_layout")?;
    add_submodule(m, gate_direction, "gate_direction")?;
    add_submodule(m, commutation_checker, "commutation_checker")?;
    add_submodule(m, commutation_analysis, "commutation_analysis")?;
    add_submodule(m, commutation_cancellation, "commutation_cancellation")?;
    Ok(())
}

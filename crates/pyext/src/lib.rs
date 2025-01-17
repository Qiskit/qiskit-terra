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

// Formatting is off here so every module import can just be a single line in sorted order, to help
// avoid merge conflicts as modules are added.
#[rustfmt::skip]
#[pymodule]
fn _accelerate(m: &Bound<PyModule>) -> PyResult<()> {
    add_submodule(m, ::qiskit_accelerate::barrier_before_final_measurement::barrier_before_final_measurements_mod, "barrier_before_final_measurement")?;
    add_submodule(m, ::qiskit_accelerate::basis::basis, "basis")?;
    add_submodule(m, ::qiskit_accelerate::check_map::check_map_mod, "check_map")?;
    add_submodule(m, ::qiskit_accelerate::circuit_library::circuit_library, "circuit_library")?;
    add_submodule(m, ::qiskit_accelerate::commutation_analysis::commutation_analysis, "commutation_analysis")?;
    add_submodule(m, ::qiskit_accelerate::commutation_cancellation::commutation_cancellation, "commutation_cancellation")?;
    add_submodule(m, ::qiskit_accelerate::commutation_checker::commutation_checker, "commutation_checker")?;
    add_submodule(m, ::qiskit_accelerate::consolidate_blocks::consolidate_blocks_mod, "consolidate_blocks")?;
    add_submodule(m, ::qiskit_accelerate::dense_layout::dense_layout, "dense_layout")?;
    add_submodule(m, ::qiskit_accelerate::equivalence::equivalence, "equivalence")?;
    add_submodule(m, ::qiskit_accelerate::error_map::error_map, "error_map")?;
    add_submodule(m, ::qiskit_accelerate::elide_permutations::elide_permutations, "elide_permutations")?;
    add_submodule(m, ::qiskit_accelerate::euler_one_qubit_decomposer::euler_one_qubit_decomposer, "euler_one_qubit_decomposer")?;
    add_submodule(m, ::qiskit_accelerate::filter_op_nodes::filter_op_nodes_mod, "filter_op_nodes")?;
    add_submodule(m, ::qiskit_accelerate::gate_direction::gate_direction, "gate_direction")?;
    add_submodule(m, ::qiskit_accelerate::gates_in_basis::gates_in_basis, "gates_in_basis")?;
    add_submodule(m, ::qiskit_accelerate::inverse_cancellation::inverse_cancellation_mod, "inverse_cancellation")?;
    add_submodule(m, ::qiskit_accelerate::isometry::isometry, "isometry")?;
    add_submodule(m, ::qiskit_accelerate::nlayout::nlayout, "nlayout")?;
    add_submodule(m, ::qiskit_accelerate::optimize_1q_gates::optimize_1q_gates, "optimize_1q_gates")?;
    add_submodule(m, ::qiskit_accelerate::pauli_exp_val::pauli_expval, "pauli_expval")?;
    add_submodule(m, ::qiskit_accelerate::high_level_synthesis::high_level_synthesis_mod, "high_level_synthesis")?;
    add_submodule(m, ::qiskit_accelerate::remove_diagonal_gates_before_measure::remove_diagonal_gates_before_measure, "remove_diagonal_gates_before_measure")?;
    add_submodule(m, ::qiskit_accelerate::remove_identity_equiv::remove_identity_equiv_mod, "remove_identity_equiv")?;
    add_submodule(m, ::qiskit_accelerate::results::results, "results")?;
    add_submodule(m, ::qiskit_accelerate::sabre::sabre, "sabre")?;
    add_submodule(m, ::qiskit_accelerate::sampled_exp_val::sampled_exp_val, "sampled_exp_val")?;
    add_submodule(m, ::qiskit_accelerate::sparse_observable::sparse_observable, "sparse_observable")?;
    add_submodule(m, ::qiskit_accelerate::sparse_pauli_op::sparse_pauli_op, "sparse_pauli_op")?;
    add_submodule(m, ::qiskit_accelerate::split_2q_unitaries::split_2q_unitaries_mod, "split_2q_unitaries")?;
    add_submodule(m, ::qiskit_accelerate::star_prerouting::star_prerouting, "star_prerouting")?;
    add_submodule(m, ::qiskit_accelerate::stochastic_swap::stochastic_swap, "stochastic_swap")?;
    add_submodule(m, ::qiskit_accelerate::synthesis::synthesis, "synthesis")?;
    add_submodule(m, ::qiskit_accelerate::target_transpiler::target, "target")?;
    add_submodule(m, ::qiskit_accelerate::twirling::twirling, "twirling")?;
    add_submodule(m, ::qiskit_accelerate::two_qubit_decompose::two_qubit_decompose, "two_qubit_decompose")?;
    add_submodule(m, ::qiskit_accelerate::unitary_synthesis::unitary_synthesis, "unitary_synthesis")?;
    add_submodule(m, ::qiskit_accelerate::uc_gate::uc_gate, "uc_gate")?;
    add_submodule(m, ::qiskit_accelerate::utils::utils, "utils")?;
    add_submodule(m, ::qiskit_accelerate::vf2_layout::vf2_layout, "vf2_layout")?;
    add_submodule(m, ::qiskit_circuit::circuit, "circuit")?;
    add_submodule(m, ::qiskit_circuit::converters::converters, "converters")?;
    add_submodule(m, ::qiskit_qasm2::qasm2, "qasm2")?;
    add_submodule(m, ::qiskit_qasm3::qasm3, "qasm3")?;
    Ok(())
}

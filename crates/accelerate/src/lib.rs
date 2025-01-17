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

// This stylistic lint suppression should be in `Cargo.toml`, but we can't do that until we're at an
// MSRV of 1.74 or greater.
#![allow(clippy::comparison_chain)]

use std::env;

use pyo3::import_exception;

pub mod barrier_before_final_measurement;
pub mod basis;
pub mod check_map;
pub mod circuit_library;
pub mod commutation_analysis;
pub mod commutation_cancellation;
pub mod commutation_checker;
pub mod consolidate_blocks;
pub mod convert_2q_block_matrix;
pub mod dense_layout;
pub mod edge_collections;
pub mod elide_permutations;
pub mod equivalence;
pub mod error_map;
pub mod euler_one_qubit_decomposer;
pub mod filter_op_nodes;
pub mod gate_direction;
pub mod gates_in_basis;
pub mod high_level_synthesis;
pub mod inverse_cancellation;
pub mod isometry;
pub mod nlayout;
pub mod optimize_1q_gates;
pub mod pauli_exp_val;
pub mod remove_diagonal_gates_before_measure;
pub mod remove_identity_equiv;
pub mod results;
pub mod sabre;
pub mod sampled_exp_val;
pub mod sparse_observable;
pub mod sparse_pauli_op;
pub mod split_2q_unitaries;
pub mod star_prerouting;
pub mod stochastic_swap;
pub mod synthesis;
pub mod target_transpiler;
pub mod twirling;
pub mod two_qubit_decompose;
pub mod uc_gate;
pub mod unitary_synthesis;
pub mod utils;
pub mod vf2_layout;

mod rayon_ext;
#[cfg(test)]
mod test;
mod unitary_compose;

#[inline]
pub fn getenv_use_multiple_threads() -> bool {
    let parallel_context = env::var("QISKIT_IN_PARALLEL")
        .unwrap_or_else(|_| "FALSE".to_string())
        .to_uppercase()
        == "TRUE";
    let force_threads = env::var("QISKIT_FORCE_THREADS")
        .unwrap_or_else(|_| "FALSE".to_string())
        .to_uppercase()
        == "TRUE";
    !parallel_context || force_threads
}

import_exception!(qiskit.exceptions, QiskitError);
import_exception!(qiskit.circuit.exceptions, CircuitError);

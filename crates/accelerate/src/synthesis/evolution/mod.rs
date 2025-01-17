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

mod pauli_network;

use pyo3::prelude::*;
use pyo3::types::PyList;

use qiskit_circuit::circuit_data::CircuitData;

use crate::synthesis::evolution::pauli_network::pauli_network_synthesis_inner;

/// Calls Rustiq's pauli network synthesis algorithm and returns the
/// Qiskit circuit data with Clifford gates and rotations.
///
/// # Arguments
///
/// * py: a GIL handle, needed to add and negate rotation parameters in Python space.
/// * num_qubits: total number of qubits.
/// * pauli_network: pauli network represented in sparse format. It's a list
///     of triples such as `[("XX", [0, 3], theta), ("ZZ", [0, 1], 0.1)]`.
/// * optimize_count: if `true`, Rustiq's synthesis algorithms aims to optimize
///     the 2-qubit gate count; and if `false`, then the 2-qubit depth.
/// * preserve_order: whether the order of paulis should be preserved, up to
///     commutativity. If the order is not preserved, the returned circuit will
///     generally not be equivalent to the given pauli network.
/// * upto_clifford: if `true`, the final Clifford operator is not synthesized
///     and the returned circuit will generally not be equivalent to the given
///     pauli network. In addition, the argument `upto_phase` would be ignored.
/// * upto_phase: if `true`, the global phase of the returned circuit may differ
///     from the global phase of the given pauli network. The argument is considered
///     to be `true` when `upto_clifford` is `true`.
/// * resynth_clifford_method: describes the strategy to synthesize the final
///     Clifford operator. If `0` a naive approach is used, which doubles the number
///     of gates but preserves the global phase of the circuit. If `1`, the Clifford is
///     resynthesized using Qiskit's greedy Clifford synthesis algorithm. If `2`, it
///     is resynthesized by Rustiq itself. If `upto_phase` is `false`, the naive
///     approach is used, as neither synthesis method preserves the global phase.
///
/// If `preserve_order` is `true` and both `upto_clifford` and `upto_phase` are `false`,
/// the returned circuit is equivalent to the given pauli network.
#[pyfunction]
#[pyo3(signature = (num_qubits, pauli_network, optimize_count=true, preserve_order=true, upto_clifford=false, upto_phase=false, resynth_clifford_method=1))]
#[allow(clippy::too_many_arguments)]
pub fn pauli_network_synthesis(
    py: Python,
    num_qubits: usize,
    pauli_network: &Bound<PyList>,
    optimize_count: bool,
    preserve_order: bool,
    upto_clifford: bool,
    upto_phase: bool,
    resynth_clifford_method: usize,
) -> PyResult<CircuitData> {
    pauli_network_synthesis_inner(
        py,
        num_qubits,
        pauli_network,
        optimize_count,
        preserve_order,
        upto_clifford,
        upto_phase,
        resynth_clifford_method,
    )
}

pub fn evolution(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pauli_network_synthesis, m)?)?;
    Ok(())
}

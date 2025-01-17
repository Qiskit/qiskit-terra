# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Synthesis for multiple-control, multiple-target Gate."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, Gate
from qiskit._accelerate.synthesis.multi_controlled import mcmt_v_chain


def synth_mcmt_vchain(
    gate: Gate, num_ctrl_qubits: int, num_target_qubits: int, ctrl_state: int | None = None
) -> QuantumCircuit:
    """Synthesize MCMT using a V-chain.

    This uses a chain of CCX gates, using ``num_ctrl_qubits - 1`` auxiliary qubits.

    For example, a 3-control and 2-target H gate will be synthesized as::

        q_0: ──■────────────────────────■──
               │                        │
        q_1: ──■────────────────────────■──
               │                        │
        q_2: ──┼────■──────────────■────┼──
               │    │  ┌───┐       │    │
        q_3: ──┼────┼──┤ H ├───────┼────┼──
               │    │  └─┬─┘┌───┐  │    │
        q_4: ──┼────┼────┼──┤ H ├──┼────┼──
             ┌─┴─┐  │    │  └─┬─┘  │  ┌─┴─┐
        q_5: ┤ X ├──■────┼────┼────■──┤ X ├
             └───┘┌─┴─┐  │    │  ┌─┴─┐└───┘
        q_6: ─────┤ X ├──■────■──┤ X ├─────
                  └───┘          └───┘

    """
    if gate.num_qubits != 1:
        raise ValueError("Only single qubit gates are supported as input.")

    circ = QuantumCircuit._from_circuit_data(
        mcmt_v_chain(gate.control(), num_ctrl_qubits, num_target_qubits, ctrl_state)
    )
    return circ

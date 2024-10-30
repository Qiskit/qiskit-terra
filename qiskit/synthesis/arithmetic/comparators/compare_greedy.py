# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Integer comparator based on 2s complement."""

from qiskit.circuit import QuantumCircuit


def synth_integer_comparator_greedy(
    num_state_qubits: int, value: int, geq: bool = True
) -> QuantumCircuit:
    """Implement an integer comparison based on value-by-value comparison."""
    circuit = QuantumCircuit(num_state_qubits + 1)

    if value <= 0:  # condition always satisfied for non-positive values
        if geq:  # otherwise the condition is never satisfied
            circuit.x(num_state_qubits)

        return circuit

    # make sure to always choose the comparison where we have to place less than
    # (2 ** n)/2 MCX gates
    if (value < 2 ** (num_state_qubits - 1) and geq) or (
        value > 2 ** (num_state_qubits - 1) and not geq
    ):
        geq = not geq
        circuit.x(num_state_qubits)

    if geq:
        accepted_values = range(value, 2**num_state_qubits)
    else:
        accepted_values = range(0, value)

    for accepted_value in accepted_values:
        circuit.mcx(list(range(num_state_qubits)), num_state_qubits, ctrl_state=accepted_value)

    return circuit

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Diagonal matrix circuit."""

from __future__ import annotations
import cmath
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError

_EPS = 1e-10


class Diagonal(QuantumCircuit):
    r"""Diagonal circuit.

    Circuit symbol:

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤0          ├
             │           │
        q_1: ┤1 Diagonal ├
             │           │
        q_2: ┤2          ├
             └───────────┘

    Matrix form:

    .. math::
        \text{DiagonalGate}\ q_0, q_1, .., q_{n-1} =
            \begin{pmatrix}
                D[0]    & 0         & \dots     & 0 \\
                0       & D[1]      & \dots     & 0 \\
                \vdots  & \vdots    & \ddots    & 0 \\
                0       & 0         & \dots     & D[n-1]
            \end{pmatrix}

    Diagonal gates are useful as representations of Boolean functions,
    as they can map from {0,1}^2**n to {0,1}^2**n space. For example a phase
    oracle can be seen as a diagonal gate with {+1, -1} on the diagonals. Such
    an oracle will induce a +1 or -1 phase on the amplitude of any corresponding
    basis state.

    Diagonal gates appear in many classically hard oracular problems such as
    Forrelation or Hidden Shift circuits.

    Diagonal gates are represented and simulated more efficiently than a dense
    2**n x 2**n unitary matrix.

    The reference implementation is via the method described in
    Theorem 7 of [1]. The code is based on Emanuel Malvetti's semester thesis
    at ETH in 2018, supervised by Raban Iten and Prof. Renato Renner.

    **Reference:**

    [1] Shende et al., Synthesis of Quantum Logic Circuits, 2009
    `arXiv:0406176 <https://arxiv.org/pdf/quant-ph/0406176.pdf>`_
    """

    def __init__(self, diag_general: list[complex] | np.ndarray | dict) -> None:
        """Create a new Diagonal circuit.

        Args:
            diag: list or np.ndarray of 2^k diagonal entries or
                dict of the diagonal entries which are not 1.Where k is number of qubits

        Raises:
            CircuitError: if the diagonal entries are in bad format;
                if absolute value of any entry is not 1.
        """
        diag = _dict_to_list(diag_general)
        if not isinstance(diag, (list, np.ndarray, dict)):
            raise CircuitError("Diagonal entries must be list , np.ndarray or dict.")
        num_qubits = np.log2(len(diag))
        if num_qubits < 1 or not num_qubits.is_integer():
            raise CircuitError("The number of diagonal entries is not a positive power of 2.")
        if not np.allclose(np.abs(diag), 1, atol=_EPS):
            raise CircuitError("A diagonal element does not have absolute value one.")

        num_qubits = int(num_qubits)

        circuit = QuantumCircuit(num_qubits, name="Diagonal")

        # Since the diagonal is a unitary, all its entries have absolute value
        # one and the diagonal is fully specified by the phases of its entries.
        diag_phases = [cmath.phase(z) for z in diag]
        n = len(diag)
        while n >= 2:
            angles_rz = []
            for i in range(0, n, 2):
                diag_phases[i // 2], rz_angle = _extract_rz(diag_phases[i], diag_phases[i + 1])
                angles_rz.append(rz_angle)
            num_act_qubits = int(np.log2(n))
            ctrl_qubits = list(range(num_qubits - num_act_qubits + 1, num_qubits))
            target_qubit = num_qubits - num_act_qubits
            circuit.ucrz(angles_rz, ctrl_qubits, target_qubit)
            n //= 2
        circuit.global_phase += diag_phases[0]

        super().__init__(num_qubits, name="Diagonal")
        self.append(circuit.to_gate(), self.qubits)


# extract a Rz rotation (angle given by first output) such that exp(j*phase)*Rz(z_angle)
# is equal to the diagonal matrix with entires exp(1j*ph1) and exp(1j*ph2)


def _extract_rz(phi1, phi2):
    phase = (phi1 + phi2) / 2.0
    z_angle = phi2 - phi1
    return phase, z_angle


# Returns size of Diagonal or 2^k
def _near_exp_two(n: int) -> int:
    i = 1
    while i <= n:
        i *= 2
    return i


# Convert dictionary to list
def _dict_to_list(d: list[complex] | np.ndarray | dict) -> list:
    if isinstance(d, dict):
        diag_list = [1] * _near_exp_two(max(d))
        for i in range(len(diag_list)):
            if i in d:
                diag_list[i] = d[i]
        return diag_list
    return d

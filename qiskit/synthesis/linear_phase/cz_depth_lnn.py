# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Synthesis of an n-qubit circuit containing only CZ gates for
linear nearest neighbor (LNN) connectivity, using CX and phase (S, Sdg or Z) gates.
The two-qubit depth of the circuit is bounded by 2*n+2.
This algorithm reverts the order of qubits.

References:
    [1]: Dmitri Maslov, Martin Roetteler,
         Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations,
         `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.permutation.permutation_reverse_lnn import (
    _append_cx_stage1,
    _append_cx_stage2,
)


def _odd_pattern1(n):
    """A pattern denoted by Pj in [1] for odd number of qubits:
    [n-2, n-4, n-4, ..., 3, 3, 1, 1, 0, 0, 2, 2, ..., n-3, n-3]
    """
    pat = []
    pat.append(n - 2)
    for i in range((n - 3) // 2):
        pat.append(n - 2 * i - 4)
        pat.append(n - 2 * i - 4)
    for i in range((n - 1) // 2):
        pat.append(2 * i)
        pat.append(2 * i)
    return pat


def _odd_pattern2(n):
    """A pattern denoted by Pk in [1] for odd number of qubits:
    [2, 2, 4, 4, ..., n-1, n-1, n-2, n-2, n-4, n-4, ..., 5, 5, 3, 3, 1]
    """
    pat = []
    for i in range((n - 1) // 2):
        pat.append(2 * i + 2)
        pat.append(2 * i + 2)
    for i in range((n - 3) // 2):
        pat.append(n - 2 * i - 2)
        pat.append(n - 2 * i - 2)
    pat.append(1)
    return pat


def _even_pattern1(n):
    """A pattern denoted by Pj in [1] for even number of qubits:
    [n-1, n-3, n-3, n-5, n-5, ..., 1, 1, 0, 0, 2, 2, ..., n-4, n-4, n-2]
    """
    pat = []
    pat.append(n - 1)
    for i in range((n - 2) // 2):
        pat.append(n - 2 * i - 3)
        pat.append(n - 2 * i - 3)
    for i in range((n - 2) // 2):
        pat.append(2 * i)
        pat.append(2 * i)
    pat.append(n - 2)
    return pat


def _even_pattern2(n):
    """A pattern denoted by Pk in [1] for even number of qubits:
    [2, 2, 4, 4, ..., n-2, n-2, n-1, n-1, ..., 3, 3, 1, 1]
    """
    pat = []
    for i in range((n - 2) // 2):
        pat.append(2 * (i + 1))
        pat.append(2 * (i + 1))
    for i in range(n // 2):
        pat.append(n - 2 * i - 1)
        pat.append(n - 2 * i - 1)
    return pat


def _create_patterns(n):
    """Creating the patterns for the phase layers."""
    if (n % 2) == 0:
        pat1 = _even_pattern1(n)
        pat2 = _even_pattern2(n)
    else:
        pat1 = _odd_pattern1(n)
        pat2 = _odd_pattern2(n)
    pats = {}

    layer = 0
    for i in range(n):
        pats[(0, i)] = (i, i)

    if (n % 2) == 0:
        ind1 = (2 * n - 4) // 2
    else:
        ind1 = (2 * n - 4) // 2 - 1
    ind2 = 0
    while layer < (n // 2):
        for i in range(n):
            pats[(layer + 1, i)] = (pat1[ind1 + i], pat2[ind2 + i])
        layer += 1
        ind1 -= 2
        ind2 += 2
    return pats


def synth_cz_depth_line_mr(mat: np.ndarray) -> QuantumCircuit:
    r"""Synthesis of a CZ circuit for linear nearest neighbour (LNN) connectivity,
    based on Maslov and Roetteler.

    Note that this method *reverts* the order of qubits in the circuit,
    and returns a circuit containing :class:`.CXGate`\s and phase gates
    (:class:`.SGate`, :class:`.SdgGate` or :class:`.ZGate`).

    Args:
        mat: an upper-diagonal matrix representing the CZ circuit.
            ``mat[i][j]=1 for i<j`` represents a ``cz(i,j)`` gate

    Returns:
        A circuit implementation of the CZ circuit of depth :math:`2n+2` for LNN
        connectivity.

    References:
        1. Dmitri Maslov, Martin Roetteler,
           *Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations*,
           `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
    """
    num_qubits = mat.shape[0]
    pats = _create_patterns(num_qubits)
    patlist = []
    # s_gates[i] = 0, 1, 2 or 3 for a gate id, sdg, z or s on qubit i respectively
    s_gates = np.zeros(num_qubits)

    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if mat[i][j]:  # CZ(i,j) gate
                s_gates[i] += 2  # qc.z[i]
                s_gates[j] += 2  # qc.z[j]
                patlist.append((i, j - 1))
                patlist.append((i, j))
                patlist.append((i + 1, j - 1))
                patlist.append((i + 1, j))

    for i in range((num_qubits + 1) // 2):
        for j in range(num_qubits):
            if pats[(i, j)] in patlist:
                patcnt = patlist.count(pats[(i, j)])
                for _ in range(patcnt):
                    s_gates[j] += 1  # qc.sdg[j]
        # Add phase gates: s, sdg or z
        for j in range(num_qubits):
            if s_gates[j] % 4 == 1:
                qc.sdg(j)
            elif s_gates[j] % 4 == 2:
                qc.z(j)
            elif s_gates[j] % 4 == 3:
                qc.s(j)
        qc = _append_cx_stage1(qc, num_qubits)
        qc = _append_cx_stage2(qc, num_qubits)
        s_gates = np.zeros(num_qubits)

    if (num_qubits % 2) == 0:
        i = num_qubits // 2
        for j in range(num_qubits):
            if pats[(i, j)] in patlist and pats[(i, j)][0] != pats[(i, j)][1]:
                patcnt = patlist.count(pats[(i, j)])
                for _ in range(patcnt):
                    s_gates[j] += 1  # qc.sdg[j]
        # Add phase gates: s, sdg or z
        for j in range(num_qubits):
            if s_gates[j] % 4 == 1:
                qc.sdg(j)
            elif s_gates[j] % 4 == 2:
                qc.z(j)
            elif s_gates[j] % 4 == 3:
                qc.s(j)
        qc = _append_cx_stage1(qc, num_qubits)

    return qc

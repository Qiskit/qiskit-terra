# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Given -CZ-CX- transformation (a layer consisting only CNOT gates
    followed by a layer consisting only CZ gates)
Return a depth-5n circuit implementation of the -CZ-CX- transformation over LNN.

Args:
    mat_z: n*n symmetric binary matrix representing a -CZ- circuit
    mat_x: n*n invertable binary matrix representing a -CX- transformation

Output:
    QuantumCircuit: :class:`.QuantumCircuit` object containing a depth-5n circuit to implement -CZ-CX-

References:
    [1] S. A. Kutin, D. P. Moulton, and L. M. Smithline, "Computation at a distance," 2007.
    [2] D. Maslov and W. Yang, "CNOT circuits need little help to implement arbitrary
        Hadamard-free Clifford transformations they generate," 2022.
"""

from copy import deepcopy
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import calc_inverse_matrix
from qiskit._accelerate.synthesis.linear import optimize_cx_circ_depth_5n_line


def _initialize_phase_schedule(mat_z):
    """
    Given a CZ layer (represented as an n*n CZ matrix Mz)
    Return a schedule of phase gates implementing Mz in a SWAP-only netwrok
    (c.f. Alg 1, [2])
    """
    n = len(mat_z)
    phase_schedule = np.zeros((n, n), dtype=int)
    for i, j in zip(*np.where(mat_z)):
        if i >= j:
            continue

        phase_schedule[i, j] = 3
        phase_schedule[i, i] += 1
        phase_schedule[j, j] += 1

    return phase_schedule


def _shuffle(labels, odd):
    """
    Args:
        labels : a list of indices
        odd : a boolean indicating whether this layer is odd or even,
    Shuffle the indices in labels by swapping adjacent elements
    (c.f. Fig.2, [2])
    """
    swapped = [v for p in zip(labels[1::2], labels[::2]) for v in p]
    return swapped + labels[-1:] if odd else swapped


def _make_seq(n):
    """
    Given the width of the circuit n,
    Return the labels of the boxes in order from left to right, top to bottom
    (c.f. Fig.2, [2])
    """
    seq = []
    wire_labels = list(range(n - 1, -1, -1))

    for i in range(n):
        wire_labels_new = (
            _shuffle(wire_labels, n % 2)
            if i % 2 == 0
            else wire_labels[0:1] + _shuffle(wire_labels[1:], (n + 1) % 2)
        )
        seq += [
            (min(i), max(i)) for i in zip(wire_labels[::2], wire_labels_new[::2]) if i[0] != i[1]
        ]
        wire_labels = wire_labels_new

    return seq


def _swap_plus(instructions, seq):
    """
    Given CX instructions (c.f. Thm 7.1, [1]) and the labels of all boxes,
    Return a list of labels of the boxes that is SWAP+ in descending order
        * Assumes the instruction gives gates in the order from top to bottom,
          from left to right
        * SWAP+ is defined in section 3.A. of [2]. Note the northwest
          diagonalization procedure of [1] consists exactly n layers of boxes,
          each being either a SWAP or a SWAP+. That is, each northwest
          diagonalization circuit can be uniquely represented by which of its
          n(n-1)/2 boxes are SWAP+ and which are SWAP.
    """
    instr = deepcopy(instructions)
    swap_plus = set()
    for i, j in reversed(seq):
        cnot_1 = instr.pop()
        instr.pop()

        if instr == [] or instr[-1] != cnot_1:
            # Only two CNOTs on same set of controls -> this box is SWAP+
            swap_plus.add((i, j))
        else:
            instr.pop()
    return swap_plus


def _update_phase_schedule(n, phase_schedule, swap_plus):
    """
    Given phase_schedule initialized to induce a CZ circuit in SWAP-only network and list of SWAP+ boxes
    Update phase_schedule for each SWAP+ according to Algorithm 2, [2]
    """
    layer_order = list(range(n))[-3::-2] + list(range(n))[-2::-2][::-1]
    order_comp = np.argsort(layer_order[::-1])

    # Go through each box by descending layer order

    for i in layer_order:
        for j in range(i + 1, n):
            if (i, j) not in swap_plus:
                continue
            # we need to correct for the effected linear functions:

            # We first correct type 1 and type 2 by switching
            # the phase applied to c_j and c_i+c_j
            phase_schedule[j, j], phase_schedule[i, j] = phase_schedule[i, j], phase_schedule[j, j]

            # Then, we go through all the boxes that permutes j BEFORE box(i,j) and update:

            for k in range(n):  # all boxes that permutes j
                if k in (i, j):
                    continue
                if (
                    order_comp[min(k, j)] < order_comp[i]
                    and phase_schedule[min(k, j), max(k, j)] % 4 != 0
                ):
                    phase = phase_schedule[min(k, j), max(k, j)]
                    phase_schedule[min(k, j), max(k, j)] = 0

                    # Step 1, apply phase to c_i, c_j, c_k
                    for l_s in (i, j, k):
                        phase_schedule[l_s, l_s] = (phase_schedule[l_s, l_s] + phase * 3) % 4

                    # Step 2, apply phase to c_i+ c_j, c_i+c_k, c_j+c_k:
                    for l1, l2 in [(i, j), (i, k), (j, k)]:
                        ls = min(l1, l2)
                        lb = max(l1, l2)
                        phase_schedule[ls, lb] = (phase_schedule[ls, lb] + phase * 3) % 4
    return phase_schedule


def _apply_phase_to_nw_circuit(n, phase_schedule, seq, swap_plus):
    """
    Given
        Width of the circuit (int n)
        A CZ circuit, represented by the n*n phase schedule phase_schedule
        A CX circuit, represented by box-labels (seq) and whether the box is SWAP+ (swap_plus)
            *   This circuit corresponds to the CX tranformation that tranforms a matrix to
                a NW matrix (c.f. Prop.7.4, [1])
            *   SWAP+ is defined in section 3.A. of [2].
            *   As previously noted, the northwest diagonalization procedure of [1] consists
                of exactly n layers of boxes, each being either a SWAP or a SWAP+. That is,
                each northwest diagonalization circuit can be uniquely represented by which
                of its n(n-1)/2 boxes are SWAP+ and which are SWAP.
    Return a QuantumCircuit that computes the phase schedule S inside CX
    """
    cir = QuantumCircuit(n)

    wires = list(zip(range(n), range(1, n)))
    wires = wires[::2] + wires[1::2]

    for i, (j, k) in zip(range(len(seq) - 1, -1, -1), reversed(seq)):
        w1, w2 = wires[i % (n - 1)]

        p = phase_schedule[j, k]

        if (j, k) not in swap_plus:
            cir.cx(w1, w2)

        cir.cx(w2, w1)

        if p % 4 == 0:
            pass
        elif p % 4 == 1:
            cir.sdg(w2)
        elif p % 4 == 2:
            cir.z(w2)
        else:
            cir.s(w2)

        cir.cx(w1, w2)

    for i in range(n):
        p = phase_schedule[n - 1 - i, n - 1 - i]
        if p % 4 == 0:
            continue
        if p % 4 == 1:
            cir.sdg(i)
        elif p % 4 == 2:
            cir.z(i)
        else:
            cir.s(i)

    return cir


def synth_cx_cz_depth_line_my(mat_x: np.ndarray, mat_z: np.ndarray) -> QuantumCircuit:
    """
    Joint synthesis of a -CZ-CX- circuit for linear nearest neighbor (LNN) connectivity,
    with 2-qubit depth at most 5n, based on Maslov and Yang.
    This method computes the CZ circuit inside the CX circuit via phase gate insertions.

    Args:
        mat_z : a boolean symmetric matrix representing a CZ circuit.
            ``mat_z[i][j]=1`` represents a ``cz(i,j)`` gate

        mat_x : a boolean invertible matrix representing a CX circuit.

    Returns:
        A circuit implementation of a CX circuit following a CZ circuit,
        denoted as a -CZ-CX- circuit,in two-qubit depth at most ``5n``, for LNN connectivity.

    References:
        1. Kutin, S., Moulton, D. P., Smithline, L.,
           *Computation at a distance*, Chicago J. Theor. Comput. Sci., vol. 2007, (2007),
           `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_
        2. Dmitri Maslov, Willers Yang, *CNOT circuits need little help to implement arbitrary
           Hadamard-free Clifford transformations they generate*,
           `arXiv:2210.16195 <https://arxiv.org/abs/2210.16195>`_.
    """

    # First, find circuits implementing mat_x by Proposition 7.3 and Proposition 7.4 of [1]

    n = len(mat_x)
    mat_x = calc_inverse_matrix(mat_x)

    cx_instructions_rows_m2nw, cx_instructions_rows_nw2id = optimize_cx_circ_depth_5n_line(mat_x)

    # Meanwhile, also build the -CZ- circuit via Phase gate insertions as per Algorithm 2 [2]
    phase_schedule = _initialize_phase_schedule(mat_z)
    seq = _make_seq(n)
    swap_plus = _swap_plus(cx_instructions_rows_nw2id, seq)

    _update_phase_schedule(n, phase_schedule, swap_plus)

    qc = _apply_phase_to_nw_circuit(n, phase_schedule, seq, swap_plus)

    for i, j in reversed(cx_instructions_rows_m2nw):
        qc.cx(i, j)

    return qc

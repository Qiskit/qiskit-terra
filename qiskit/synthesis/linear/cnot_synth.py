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

"""
Implementation of the GraySynth algorithm for synthesizing CNOT-Phase
circuits with efficient CNOT cost, and the Patel-Hayes-Markov algorithm
for optimal synthesis of linear (CNOT-only) reversible circuits.
"""

from __future__ import annotations
import copy
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CXGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import QiskitError


def synth_cnot_count_full_pmh(
    state: list[list[bool]] | np.ndarray[bool], section_size: int = 2, use_dag: bool = False
) -> QuantumCircuit | DAGCircuit:
    """
    Synthesize linear reversible circuits for all-to-all architecture
    using Patel, Markov and Hayes method.

    This function is an implementation of the Patel, Markov and Hayes algorithm from [1]
    for optimal synthesis of linear reversible circuits for all-to-all architecture,
    as specified by an :math:`n \\times n` matrix.

    Args:
        state: :math:`n \\times n` boolean invertible matrix, describing
            the state of the input circuit
        section_size: The size of each section, used in the
            Patel–Markov–Hayes algorithm [1].
        use_dag: If true a :class:`.DAGCircuit` is returned instead of a
                :class:`QuantumCircuit` when this class is called.

    Returns:
        QuantumCircuit: a CX-only circuit implementing the linear transformation.

    Raises:
        QiskitError: when variable ``state`` isn't of type ``numpy.ndarray``

    References:
        1. Patel, Ketan N., Igor L. Markov, and John P. Hayes,
           *Optimal synthesis of linear reversible circuits*,
           Quantum Information & Computation 8.3 (2008): 282-294.
           `arXiv:quant-ph/0302002 [quant-ph] <https://arxiv.org/abs/quant-ph/0302002>`_
    """
    if not isinstance(state, (list, np.ndarray)):
        raise QiskitError(
            "state should be of type list or numpy.ndarray, "
            "but was of the type {}".format(type(state))
        )
    state = np.array(state)
    # Synthesize lower triangular part
    [state, circuit_l] = _lwr_cnot_synth(state, section_size)
    state = np.transpose(state)
    # Synthesize upper triangular part
    [state, circuit_u] = _lwr_cnot_synth(state, section_size)
    circuit_l.reverse()
    for i in circuit_u:
        i.reverse()

    # Convert the list into a circuit of C-NOT gates
    if use_dag:
        qreg = QuantumRegister(state.shape[0])
        circ = DAGCircuit()
        circ.add_qreg(qreg)
        for i in circuit_u + circuit_l:
            circ.apply_operation_back(CXGate(), (qreg[i[0]], qreg[i[1]]), check=False)
    else:
        circ = QuantumCircuit(state.shape[0])
        for i in circuit_u + circuit_l:
            circ.cx(i[0], i[1])

    return circ


def _lwr_cnot_synth(state, section_size):
    """
    This function is a helper function of the algorithm for optimal synthesis
    of linear reversible circuits (the Patel–Markov–Hayes algorithm). It works
    like gaussian elimination, except that it works a lot faster, and requires
    fewer steps (and therefore fewer CNOTs). It takes the matrix "state" and
    splits it into sections of size section_size. Then it eliminates all non-zero
    sub-rows within each section, which are the same as a non-zero sub-row
    above. Once this has been done, it continues with normal gaussian elimination.
    The benefit is that with small section sizes (m), most of the sub-rows will
    be cleared in the first step, resulting in a factor m fewer row row operations
    during Gaussian elimination.

    The algorithm is described in detail in the following paper
    "Optimal synthesis of linear reversible circuits."
    Patel, Ketan N., Igor L. Markov, and John P. Hayes.
    Quantum Information & Computation 8.3 (2008): 282-294.

    Note:
    This implementation tweaks the Patel, Markov, and Hayes algorithm by adding
    a "back reduce" which adds rows below the pivot row with a high degree of
    overlap back to it. The intuition is to avoid a high-weight pivot row
    increasing the weight of lower rows.

    Args:
        state (ndarray): n x n matrix, describing a linear quantum circuit
        section_size (int): the section size the matrix columns are divided into

    Returns:
        numpy.matrix: n by n matrix, describing the state of the output circuit
        list: a k by 2 list of C-NOT operations that need to be applied
    """
    circuit = []
    num_qubits = state.shape[0]
    cutoff = 1

    # Iterate over column sections
    for sec in range(1, int(np.floor(num_qubits / section_size) + 1)):
        # Remove duplicate sub-rows in section sec
        patt = {}
        for row in range((sec - 1) * section_size, num_qubits):
            sub_row_patt = copy.deepcopy(state[row, (sec - 1) * section_size : sec * section_size])
            if np.sum(sub_row_patt) == 0:
                continue
            if str(sub_row_patt) not in patt:
                patt[str(sub_row_patt)] = row
            else:
                state[row, :] ^= state[patt[str(sub_row_patt)], :]
                circuit.append([patt[str(sub_row_patt)], row])
        # Use gaussian elimination for remaining entries in column section
        for col in range((sec - 1) * section_size, sec * section_size):
            # Check if 1 on diagonal
            diag_one = 1
            if state[col, col] == 0:
                diag_one = 0
            # Remove ones in rows below column col
            for row in range(col + 1, num_qubits):
                if state[row, col] == 1:
                    if diag_one == 0:
                        state[col, :] ^= state[row, :]
                        circuit.append([row, col])
                        diag_one = 1
                    state[row, :] ^= state[col, :]
                    circuit.append([col, row])
                # Back reduce the pivot row using the current row
                if sum(state[col, :] & state[row, :]) > cutoff:
                    state[col, :] ^= state[row, :]
                    circuit.append([row, col])
    return [state, circuit]

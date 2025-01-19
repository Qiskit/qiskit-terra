# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Optimize the synthesis of an n-qubit circuit contains only CX gates for
linear nearest neighbor (LNN) connectivity.
The depth of the circuit is bounded by 5*n, while the gate count is approximately 2.5*n^2

References:
    [1]: Kutin, S., Moulton, D. P., Smithline, L. (2007).
         Computation at a Distance.
         `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_.
"""

from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import check_invertible_binary_matrix
from qiskit._accelerate.synthesis.linear import optimize_cx_circ_depth_5n_line


def synth_cnot_depth_line_kms(mat: np.ndarray[bool]) -> QuantumCircuit:
    """
    Synthesize linear reversible circuit for linear nearest-neighbor architectures using
    Kutin, Moulton, Smithline method.

    Synthesis algorithm for linear reversible circuits from [1], section 7.
    This algorithm synthesizes any linear reversible circuit of :math:`n` qubits over
    a linear nearest-neighbor architecture using CX gates with depth at most :math:`5n`.

    Args:
        mat: A boolean invertible matrix.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``mat`` is not invertible.

    References:
        1. Kutin, S., Moulton, D. P., Smithline, L.,
           *Computation at a distance*, Chicago J. Theor. Comput. Sci., vol. 2007, (2007),
           `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_
    """
    if not check_invertible_binary_matrix(mat):
        raise QiskitError("The input matrix is not invertible.")

    # Returns the quantum circuit constructed from the instructions
    # that we got in _optimize_cx_circ_depth_5n_line
    num_qubits = len(mat)
    cx_inst = optimize_cx_circ_depth_5n_line(mat)
    qc = QuantumCircuit(num_qubits)
    for pair in cx_inst[0]:
        qc.cx(pair[0], pair[1])
    for pair in cx_inst[1]:
        qc.cx(pair[0], pair[1])
    return qc

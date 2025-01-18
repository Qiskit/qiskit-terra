# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum Shannon Decomposition.

Method is described in arXiv:quant-ph/0406176.
"""
from __future__ import annotations
import math
from typing import Callable
import scipy
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.synthesis.two_qubit import (
    TwoQubitBasisDecomposer,
    two_qubit_decompose,
)
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.circuit.library.generalized_gates.uc_pauli_rot import UCPauliRotGate, _EPS
from qiskit.circuit.library.generalized_gates.ucry import UCRYGate
from qiskit.circuit.library.generalized_gates.ucrz import UCRZGate
from qiskit._accelerate.two_qubit_decompose import two_qubit_decompose_up_to_diagonal


def qs_decomposition(
    mat: np.ndarray,
    opt_a1: bool = True,
    opt_a2: bool = True,
    decomposer_1q: Callable[[np.ndarray], QuantumCircuit] | None = None,
    decomposer_2q: Callable[[np.ndarray], QuantumCircuit] | None = None,
    *,
    _depth=0,
):
    r"""
    Decomposes a unitary matrix into one and two qubit gates using Quantum Shannon Decomposition,

    This decomposition is described in Shende et al. [1].

    .. code-block:: text

          ┌───┐               ┌───┐     ┌───┐     ┌───┐
         ─┤   ├─       ───────┤ Rz├─────┤ Ry├─────┤ Rz├─────
          │   │    ≃     ┌───┐└─┬─┘┌───┐└─┬─┘┌───┐└─┬─┘┌───┐
        /─┤   ├─       /─┤   ├──□──┤   ├──□──┤   ├──□──┤   ├
          └───┘          └───┘     └───┘     └───┘     └───┘

    The number of :class:`.CXGate`\ s generated with the decomposition without optimizations is:

    .. math::

        \frac{9}{16} 4^n - \frac{3}{2} 2^n

    If ``opt_a1 = True``, the default, the CX count is reduced by:

    .. math::

        \frac{1}{3} 4^{n - 2} - 1.

    If ``opt_a2 = True``, the default, the CX count is reduced by:

    .. math::

        4^{n-2} - 1.

    Args:
        mat: unitary matrix to decompose
        opt_a1: whether to try optimization A.1 from Shende et al. [1].
            This should eliminate 1 ``cx`` per call.
            If ``True``, :class:`.CZGate`\s are left in the output.
            If desired these can be further decomposed to :class:`.CXGate`\s.
        opt_a2: whether to try optimization A.2 from Shende et al. [1].
            This decomposes two qubit unitaries into a diagonal gate and
            a two cx unitary and reduces overall cx count by :math:`4^{n-2} - 1`.
        decomposer_1q: optional 1Q decomposer. If None, uses
            :class:`~qiskit.synthesis.OneQubitEulerDecomposer`.
        decomposer_2q: optional 2Q decomposer. If None, uses
            :class:`~qiskit.synthesis.TwoQubitBasisDecomposer`.

    Returns:
        QuantumCircuit: Decomposed quantum circuit.

    References:
        1. Shende, Bullock, Markov, *Synthesis of Quantum Logic Circuits*,
           `arXiv:0406176 [quant-ph] <https://arxiv.org/abs/quant-ph/0406176>`_
    """
    #  _depth (int): Internal use parameter to track recursion depth.
    dim = mat.shape[0]
    nqubits = dim.bit_length() - 1
    
    if np.allclose(np.identity(dim), mat):
        return QuantumCircuit(nqubits)
    if dim == 2:
        if decomposer_1q is None:
            decomposer_1q = one_qubit_decompose.OneQubitEulerDecomposer()
        circ = decomposer_1q(mat)
    elif dim == 4:
        if decomposer_2q is None:
            if opt_a2 and _depth > 0:
                from qiskit.circuit.library.generalized_gates.unitary import (
                    UnitaryGate,
                )  # pylint: disable=cyclic-import

                def decomp_2q(mat):
                    ugate = UnitaryGate(mat)
                    qc = QuantumCircuit(2, name="qsd2q")
                    qc.append(ugate, [0, 1])
                    return qc

                decomposer_2q = decomp_2q
            else:
                decomposer_2q = TwoQubitBasisDecomposer(CXGate())
        circ = decomposer_2q(mat)
    else:
        # check whether matrix is equivalent to block diagonal wrt ctrl_index
        if opt_a2 is False:
            for ctrl_index in range(nqubits):
                um00, um11, um01, um10 = _extract_multiplex_blocks(mat, ctrl_index)
                if _off_diagonals_are_zero(um01, um10):
                    decirc = _demultiplex(um00, um11, opt_a1=opt_a1, opt_a2=opt_a2,
                                          _depth=_depth, _ctrl_index=ctrl_index)
                    return decirc
        qr = QuantumRegister(nqubits)
        circ = QuantumCircuit(qr)
        dim_o2 = dim // 2
        # perform cosine-sine decomposition
        (u1, u2), vtheta, (v1h, v2h) = scipy.linalg.cossin(mat, separate=True, p=dim_o2, q=dim_o2)
        # left circ
        left_circ = _demultiplex(v1h, v2h, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth)
        circ.append(left_circ.to_instruction(), qr)
        # middle circ
        if opt_a1:
            nangles = len(vtheta)
            half_size = nangles // 2
            # get UCG in terms of CZ
            circ_cz = _get_ucry_cz(nqubits, (2 * vtheta).tolist())
            circ.append(circ_cz.to_instruction(), range(nqubits))
            # merge final cz with right-side generic multiplexer
            u2[:, half_size:] = np.negative(u2[:, half_size:])
        else:
            ucry = UCRYGate((2 * vtheta).tolist())
            circ.append(ucry, [qr[-1]] + qr[:-1])
        # right circ
        right_circ = _demultiplex(u1, u2, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth)
        circ.append(right_circ.to_instruction(), qr)

    if opt_a2 and _depth == 0 and dim > 4:
        return _apply_a2(circ)
    return circ


def _demultiplex(um0, um1, opt_a1=False, opt_a2=False, *, _depth=0, _ctrl_index=0):
    """Decompose a generic multiplexer.

          ────□────
           ┌──┴──┐
         /─┤     ├─
           └─────┘

    represented by the block diagonal matrix

            ┏         ┓
            ┃ um0     ┃
            ┃     um1 ┃
            ┗         ┛

    to
               ┌───┐
        ───────┤ Rz├──────
          ┌───┐└─┬─┘┌───┐
        /─┤ w ├──□──┤ v ├─
          └───┘     └───┘

    where v and w are general unitaries determined from decomposition.

    Args:
       um0 (ndarray): applied if MSB is 0
       um1 (ndarray): applied if MSB is 1
       opt_a1 (bool): whether to try optimization A.1 from Shende. This should eliminate 1 cnot
          per call. If True CZ gates are left in the output. If desired these can be further decomposed
       opt_a2 (bool): whether to try  optimization A.2 from Shende. This decomposes two qubit
          unitaries into a diagonal gate and a two cx unitary and reduces overall cx count by
          4^(n-2) - 1.
       _depth (int): This is an internal variable to track the recursion depth.
       _ctrl_index (int): The index wrt which um0 and um1 are controlled.

    Returns:
        QuantumCircuit: decomposed circuit
    """
    dim = um0.shape[0] + um1.shape[0]  # these should be same dimension
    nqubits = dim.bit_length() - 1
    layout = list(range(nqubits))
    if _ctrl_index:
        rindex = np.arange(nqubits)[::-1][_ctrl_index]
        pop_element = layout.pop(rindex)
        layout = layout + [pop_element]
        print(f"ctrl_index = {_ctrl_index}")
    
    um0um1 = um0 @ um1.T.conjugate()
    if is_hermitian_matrix(um0um1):
        eigvals, vmat = scipy.linalg.eigh(um0um1)
    else:
        evals, vmat = scipy.linalg.schur(um0um1, output="complex")
        eigvals = evals.diagonal()
    dvals = np.emath.sqrt(eigvals)
    dmat = np.diag(dvals)
    wmat = dmat @ vmat.T.conjugate() @ um1

    qr = [Qubit() for _ in range(nqubits)]
    circ = QuantumCircuit(qr)

    # left gate
    left_gate = qs_decomposition(
        wmat, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth + 1
    ).to_instruction()
    circ.append(left_gate, layout[:nqubits-1])

    # multiplexed Rz
    angles = 2 * np.angle(np.conj(dvals))
    ucrz = UCRZGate(angles.tolist())
    circ.append(ucrz, [layout[-1]] + layout[:nqubits-1])

    # right gate
    right_gate = qs_decomposition(
        vmat, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth + 1
    ).to_instruction()
    circ.append(right_gate, layout[:nqubits - 1])

    return circ


def _get_ucry_cz(nqubits, angles):
    """
    Get uniformly controlled Ry gate in CZ-Ry as in UCPauliRotGate.
    """
    nangles = len(angles)
    qc = QuantumCircuit(nqubits)
    q_controls = qc.qubits[:-1]
    q_target = qc.qubits[-1]
    if not q_controls:
        if np.abs(angles[0]) > _EPS:
            qc.ry(angles[0], q_target)
    else:
        angles = angles.copy()
        UCPauliRotGate._dec_uc_rotations(angles, 0, len(angles), False)
        for i, angle in enumerate(angles):
            if np.abs(angle) > _EPS:
                qc.ry(angle, q_target)
            if not i == len(angles) - 1:
                binary_rep = np.binary_repr(i + 1)
                q_contr_index = len(binary_rep) - len(binary_rep.rstrip("0"))
            else:
                # Handle special case:
                q_contr_index = len(q_controls) - 1
            # leave off last CZ for merging with adjacent UCG
            if i < nangles - 1:
                qc.cz(q_controls[q_contr_index], q_target)
    return qc


def _apply_a2(circ):
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
    from qiskit.transpiler.passes.synthesis import HighLevelSynthesis

    decomposer = two_qubit_decompose_up_to_diagonal
    hls = HighLevelSynthesis(basis_gates=["u", "cx", "qsd2q"], qubits_initially_zero=False)
    ccirc = hls(circ)
    ind2q = []
    # collect 2q instrs
    for i, instruction in enumerate(ccirc.data):
        if instruction.operation.name == "qsd2q":
            ind2q.append(i)
    if len(ind2q) == 0:
        return ccirc
    elif len(ind2q) == 1:
        # No neighbors to merge diagonal into; revert name
        ccirc.data[ind2q[0]].operation.name = "Unitary"
        return ccirc
    # rolling over diagonals
    ind2 = None  # lint
    for ind1, ind2 in zip(ind2q[0:-1:], ind2q[1::]):
        # get neighboring 2q gates separated by controls
        instr1 = ccirc.data[ind1]
        mat1 = Operator(instr1.operation).data
        instr2 = ccirc.data[ind2]
        mat2 = Operator(instr2.operation).data
        # rollover
        dmat, qc2cx_data = decomposer(mat1)
        qc2cx = QuantumCircuit._from_circuit_data(qc2cx_data)
        ccirc.data[ind1] = instr1.replace(operation=qc2cx.to_gate())
        mat2 = mat2 @ dmat
        ccirc.data[ind2] = instr2.replace(UnitaryGate(mat2))
    qc3 = two_qubit_decompose.two_qubit_cnot_decompose(mat2)
    ccirc.data[ind2] = ccirc.data[ind2].replace(operation=qc3.to_gate())
    return ccirc

def _extract_multiplex_blocks(U, k):
    """
    A block diagonal gate is represented as 
    """
    dim = U.shape[0]
    nqubits = dim.bit_length() - 1
    halfdim = dim // 2

    U_tensor = U.reshape((2,) * nqubits + (2,) * nqubits)

    # Move qubit k to top
    if k != 0:
        U_tensor = np.moveaxis(U_tensor, k,   0)
        U_tensor = np.moveaxis(U_tensor, k+nqubits, nqubits)

    # reshape for extraction
    U_4d = U_tensor.reshape(2, halfdim, 2, halfdim)
    # block for qubit k = |0>
    um00 = U_4d[0, :, 0, :]
    # block for qubit k = |1>
    um11 = U_4d[1, :, 1, :]
    # off diagonal blocks
    um01 = U_4d[0, :, 1, :]
    um10 = U_4d[1, :, 0, :]
    return um00, um11, um01, um10

def _off_diagonals_are_zero(um01, um10, atol=1e-12):
    """
    Checks whether off-diagonal blocks are zero.
    Args:
       um01 (ndarray): upper right block
       um10 (ndarray): lower left block
       atol (float): absolute tolerance
    Returns:
       bool: whether both blocks are zero within tolerance
    """
    return (np.allclose(um01, 0, atol=atol) and
            np.allclose(um10, 0, atol=atol))

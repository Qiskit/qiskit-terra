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

# pylint: disable=invalid-name

"""
Expand 2-qubit Unitary operators into an equivalent
decomposition over SU(2)+fixed 2q basis gate, using the KAK method.

May be exact or approximate expansion. In either case uses the minimal
number of basis applications.

Method is described in Appendix B of Cross, A. W., Bishop, L. S., Sheldon, S., Nation, P. D. &
Gambetta, J. M. Validating quantum computers using randomized model circuits.
arXiv:1811.12926 [quant-ph] (2018).
"""
from __future__ import annotations
import cmath
import math
import io
import base64
import warnings
from typing import Optional, Type, TYPE_CHECKING

import logging

import numpy as np

from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate, CircuitInstruction
from qiskit.circuit.library.standard_gates import (
    CXGate,
    U3Gate,
    U2Gate,
    U1Gate,
    UGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    SXGate,
    XGate,
    RGate,
)
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.one_qubit.one_qubit_decompose import (
    OneQubitEulerDecomposer,
    DEFAULT_ATOL,
)
from qiskit.utils.deprecation import deprecate_func
from qiskit._accelerate import two_qubit_decompose

if TYPE_CHECKING:
    from qiskit.dagcircuit.dagcircuit import DAGCircuit, DAGOpNode

logger = logging.getLogger(__name__)


GATE_NAME_MAP = {
    "cx": CXGate,
    "rx": RXGate,
    "sx": SXGate,
    "x": XGate,
    "rz": RZGate,
    "u": UGate,
    "p": PhaseGate,
    "u1": U1Gate,
    "u2": U2Gate,
    "u3": U3Gate,
    "ry": RYGate,
    "r": RGate,
}


def decompose_two_qubit_product_gate(special_unitary_matrix: np.ndarray):
    r"""Decompose :math:`U = U_l \otimes U_r` where :math:`U \in SU(4)`,
    and :math:`U_l,~U_r \in SU(2)`.

    Args:
        special_unitary_matrix: special unitary matrix to decompose
    Raises:
        QiskitError: if decomposition isn't possible.
    """
    special_unitary_matrix = np.asarray(special_unitary_matrix, dtype=complex)
    # extract the right component
    R = special_unitary_matrix[:2, :2].copy()
    detR = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
    if abs(detR) < 0.1:
        R = special_unitary_matrix[2:, :2].copy()
        detR = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
    if abs(detR) < 0.1:
        raise QiskitError("decompose_two_qubit_product_gate: unable to decompose: detR < 0.1")
    R /= np.sqrt(detR)

    # extract the left component
    temp = np.kron(np.eye(2), R.T.conj())
    temp = special_unitary_matrix.dot(temp)
    L = temp[::2, ::2]
    detL = L[0, 0] * L[1, 1] - L[0, 1] * L[1, 0]
    if abs(detL) < 0.9:
        raise QiskitError("decompose_two_qubit_product_gate: unable to decompose: detL < 0.9")
    L /= np.sqrt(detL)
    phase = cmath.phase(detL) / 2

    temp = np.kron(L, R)
    deviation = abs(abs(temp.conj().T.dot(special_unitary_matrix).trace()) - 4)
    if deviation > 1.0e-13:
        raise QiskitError(
            "decompose_two_qubit_product_gate: decomposition failed: "
            f"deviation too large: {deviation}"
        )

    return L, R, phase


_ipx = np.array([[0, 1j], [1j, 0]], dtype=complex)
_ipy = np.array([[0, 1], [-1, 0]], dtype=complex)
_ipz = np.array([[1j, 0], [0, -1j]], dtype=complex)
_id = np.array([[1, 0], [0, 1]], dtype=complex)


class TwoQubitWeylDecomposition:
    r"""Two-qubit Weyl decomposition.

    Decompose two-qubit unitary

    .. math::

        U = ({K_1}^l \otimes {K_1}^r) e^{(i a XX + i b YY + i c ZZ)} ({K_2}^l \otimes {K_2}^r)

    where

    .. math::

        U \in U(4),~
        {K_1}^l, {K_1}^r, {K_2}^l, {K_2}^r \in SU(2)

    and we stay in the "Weyl Chamber"

    .. math::

        \pi /4 \geq a \geq b \geq |c|

    This class avoids some problems of numerical instability near high-symmetry loci within the Weyl
    chamber. If there is a high-symmetry gate "nearby" (in terms of the requested average gate fidelity),
    then it return a canonicalized decomposition of that high-symmetry gate.

    References:
        1. Cross, A. W., Bishop, L. S., Sheldon, S., Nation, P. D. & Gambetta, J. M.,
           *Validating quantum computers using randomized model circuits*,
           `arXiv:1811.12926 [quant-ph] <https://arxiv.org/abs/1811.12926>`_
        2. B. Kraus, J. I. Cirac, *Optimal Creation of Entanglement Using a Two-Qubit Gate*,
           `arXiv:0011050 [quant-ph] <https://arxiv.org/abs/quant-ph/0011050>`_
        3. B. Drury, P. J. Love, *Constructive Quantum Shannon Decomposition from Cartan
           Involutions*, `arXiv:0806.4015 [quant-ph] <https://arxiv.org/abs/0806.4015>`_

    """

    # The parameters of the decomposition:
    a: float
    b: float
    c: float
    global_phase: float
    K1l: np.ndarray
    K2l: np.ndarray
    K1r: np.ndarray
    K2r: np.ndarray

    unitary_matrix: np.ndarray  # The unitary that was input
    requested_fidelity: Optional[float]  # None means no automatic specialization
    calculated_fidelity: float  # Fidelity after specialization

    _specializations = two_qubit_decompose.Specialization

    def __init__(
        self,
        unitary_matrix: np.ndarray,
        fidelity: float | None = 1.0 - 1.0e-9,
        *,
        _specialization: two_qubit_decompose.Specialization | None = None,
    ):
        unitary_matrix = np.asarray(unitary_matrix, dtype=complex)
        self._inner_decomposition = two_qubit_decompose.TwoQubitWeylDecomposition(
            unitary_matrix, fidelity=fidelity, _specialization=_specialization
        )
        self.a = self._inner_decomposition.a
        self.b = self._inner_decomposition.b
        self.c = self._inner_decomposition.c
        self.global_phase = self._inner_decomposition.global_phase
        self.K1l = self._inner_decomposition.K1l
        self.K1r = self._inner_decomposition.K1r
        self.K2l = self._inner_decomposition.K2l
        self.K2r = self._inner_decomposition.K2r
        self.unitary_matrix = unitary_matrix
        self.requested_fidelity = fidelity
        self.calculated_fidelity = self._inner_decomposition.calculated_fidelity
        if logger.isEnabledFor(logging.DEBUG):
            actual_fidelity = self.actual_fidelity()
            logger.debug(
                "Requested fidelity: %s calculated fidelity: %s actual fidelity %s",
                self.requested_fidelity,
                self.calculated_fidelity,
                actual_fidelity,
            )
            if abs(self.calculated_fidelity - actual_fidelity) > 1.0e-12:
                logger.warning(
                    "Requested fidelity different from actual by %s",
                    self.calculated_fidelity - actual_fidelity,
                )

    @deprecate_func(since="1.1.0", removal_timeline="in the 2.0.0 release")
    def specialize(self):
        """Make changes to the decomposition to comply with any specializations.

        This method will always raise a ``NotImplementedError`` because
        there are no specializations to comply with in the current implementation.
        """
        raise NotImplementedError

    def circuit(
        self, *, euler_basis: str | None = None, simplify: bool = False, atol: float = DEFAULT_ATOL
    ) -> QuantumCircuit:
        """Returns Weyl decomposition in circuit form."""
        circuit_data = self._inner_decomposition.circuit(
            euler_basis=euler_basis, simplify=simplify, atol=atol
        )
        return QuantumCircuit._from_circuit_data(circuit_data, add_regs=True)

    def actual_fidelity(self, **kwargs) -> float:
        """Calculates the actual fidelity of the decomposed circuit to the input unitary."""
        circ = self.circuit(**kwargs)
        trace = np.trace(Operator(circ).data.T.conj() @ self.unitary_matrix)
        return trace_to_fid(trace)

    def __repr__(self):
        """Represent with enough precision to allow copy-paste debugging of all corner cases"""
        prefix = f"{type(self).__qualname__}.from_bytes("
        with io.BytesIO() as f:
            np.save(f, self.unitary_matrix, allow_pickle=False)
            b64 = base64.encodebytes(f.getvalue()).splitlines()
        b64ascii = [repr(x) for x in b64]
        b64ascii[-1] += ","
        pretty = [f"# {x.rstrip()}" for x in str(self).splitlines()]
        indent = "\n" + " " * 4
        specialization_variant = str(self._inner_decomposition.specialization).split(".")[1]
        specialization_repr = f"{type(self).__qualname__}._specializations.{specialization_variant}"
        lines = (
            [prefix]
            + pretty
            + b64ascii
            + [
                f"requested_fidelity={self.requested_fidelity},",
                f"_specialization={specialization_repr},",
                f"calculated_fidelity={self.calculated_fidelity},",
                f"actual_fidelity={self.actual_fidelity()},",
                f"abc={(self.a, self.b, self.c)})",
            ]
        )
        return indent.join(lines)

    @classmethod
    def from_bytes(
        cls,
        bytes_in: bytes,
        *,
        requested_fidelity: float,
        _specialization: two_qubit_decompose.Specialization | None = None,
        **kwargs,
    ) -> "TwoQubitWeylDecomposition":
        """Decode bytes into :class:`.TwoQubitWeylDecomposition`."""
        # Used by __repr__
        del kwargs  # Unused (just for display)
        b64 = base64.decodebytes(bytes_in)
        with io.BytesIO(b64) as f:
            arr = np.load(f, allow_pickle=False)
        return cls(arr, fidelity=requested_fidelity, _specialization=_specialization)

    def __str__(self):
        specialization = str(self._inner_decomposition.specialization).split(".")[1]
        pre = f"{self.__class__.__name__} [specialization={specialization}] (\n\t"
        circ_indent = "\n\t".join(self.circuit(simplify=True).draw("text").lines(-1))
        return f"{pre}{circ_indent}\n)"


class TwoQubitControlledUDecomposer:
    r"""Decompose two-qubit unitary in terms of a desired
    :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}`
    gate that is locally equivalent to an :class:`.RXXGate`."""

    def __init__(self, rxx_equivalent_gate: Type[Gate]):
        r"""Initialize the KAK decomposition.

        Args:
            rxx_equivalent_gate: Gate that is locally equivalent to an :class:`.RXXGate`:
            :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}` gate.
        Raises:
            QiskitError: If the gate is not locally equivalent to an :class:`.RXXGate`.
        """
        atol = DEFAULT_ATOL

        scales, test_angles, scale = [], [0.2, 0.3, np.pi / 2], None

        for test_angle in test_angles:
            # Check that gate takes a single angle parameter
            try:
                rxx_equivalent_gate(test_angle, label="foo")
            except TypeError as _:
                raise QiskitError("Equivalent gate needs to take exactly 1 angle parameter.") from _
            decomp = TwoQubitWeylDecomposition(rxx_equivalent_gate(test_angle))

            circ = QuantumCircuit(2)
            circ.rxx(test_angle, 0, 1)
            decomposer_rxx = TwoQubitWeylDecomposition(
                Operator(circ).data,
                fidelity=None,
                _specialization=two_qubit_decompose.Specialization.ControlledEquiv,
            )

            circ = QuantumCircuit(2)
            circ.append(rxx_equivalent_gate(test_angle), qargs=[0, 1])
            decomposer_equiv = TwoQubitWeylDecomposition(
                Operator(circ).data,
                fidelity=None,
                _specialization=two_qubit_decompose.Specialization.ControlledEquiv,
            )

            scale = decomposer_rxx.a / decomposer_equiv.a

            if abs(decomp.a * 2 - test_angle / scale) > atol:
                raise QiskitError(
                    f"{rxx_equivalent_gate.__name__} is not equivalent to an RXXGate."
                )

            scales.append(scale)

        # Check that all three tested angles give the same scale
        if not np.allclose(scales, [scale] * len(test_angles)):
            raise QiskitError(
                f"Cannot initialize {self.__class__.__name__}: with gate {rxx_equivalent_gate}. "
                "Inconsistent scaling parameters in checks."
            )

        self.scale = scales[0]

        self.rxx_equivalent_gate = rxx_equivalent_gate

    def __call__(self, unitary, *, atol=DEFAULT_ATOL) -> QuantumCircuit:
        """Returns the Weyl decomposition in circuit form.

        Note: atol ist passed to OneQubitEulerDecomposer.
        """

        # pylint: disable=attribute-defined-outside-init
        self.decomposer = TwoQubitWeylDecomposition(unitary)

        oneq_decompose = OneQubitEulerDecomposer("ZYZ")
        c1l, c1r, c2l, c2r = (
            oneq_decompose(k, atol=atol)
            for k in (
                self.decomposer.K1l,
                self.decomposer.K1r,
                self.decomposer.K2l,
                self.decomposer.K2r,
            )
        )
        circ = QuantumCircuit(2, global_phase=self.decomposer.global_phase)
        circ.compose(c2r, [0], inplace=True)
        circ.compose(c2l, [1], inplace=True)
        self._weyl_gate(circ)
        circ.compose(c1r, [0], inplace=True)
        circ.compose(c1l, [1], inplace=True)
        return circ

    def _to_rxx_gate(self, angle: float) -> QuantumCircuit:
        """
        Takes an angle and returns the circuit equivalent to an RXXGate with the
        RXX equivalent gate as the two-qubit unitary.

        Args:
            angle: Rotation angle (in this case one of the Weyl parameters a, b, or c)

        Returns:
            Circuit: Circuit equivalent to an RXXGate.

        Raises:
            QiskitError: If the circuit is not equivalent to an RXXGate.
        """

        # The user-provided RXXGate equivalent gate may be locally equivalent to the RXXGate
        # but with some scaling in the rotation angle. For example, RXXGate(angle) has Weyl
        # parameters (angle, 0, 0) for angle in [0, pi/2] but the user provided gate, i.e.
        # :code:`self.rxx_equivalent_gate(angle)` might produce the Weyl parameters
        # (scale * angle, 0, 0) where scale != 1. This is the case for the CPhaseGate.

        circ = QuantumCircuit(2)
        circ.append(self.rxx_equivalent_gate(self.scale * angle), qargs=[0, 1])
        decomposer_inv = TwoQubitWeylDecomposition(Operator(circ).data)

        oneq_decompose = OneQubitEulerDecomposer("ZYZ")

        # Express the RXXGate in terms of the user-provided RXXGate equivalent gate.
        rxx_circ = QuantumCircuit(2, global_phase=-decomposer_inv.global_phase)
        rxx_circ.compose(oneq_decompose(decomposer_inv.K2r).inverse(), inplace=True, qubits=[0])
        rxx_circ.compose(oneq_decompose(decomposer_inv.K2l).inverse(), inplace=True, qubits=[1])
        rxx_circ.compose(circ, inplace=True)
        rxx_circ.compose(oneq_decompose(decomposer_inv.K1r).inverse(), inplace=True, qubits=[0])
        rxx_circ.compose(oneq_decompose(decomposer_inv.K1l).inverse(), inplace=True, qubits=[1])

        return rxx_circ

    def _weyl_gate(self, circ: QuantumCircuit, atol=1.0e-13):
        """Appends U_d(a, b, c) to the circuit."""

        circ_rxx = self._to_rxx_gate(-2 * self.decomposer.a)
        circ.compose(circ_rxx, inplace=True)

        # translate the RYYGate(b) into a circuit based on the desired Ctrl-U gate.
        if abs(self.decomposer.b) > atol:
            circ_ryy = QuantumCircuit(2)
            circ_ryy.sdg(0)
            circ_ryy.sdg(1)
            circ_ryy.compose(self._to_rxx_gate(-2 * self.decomposer.b), inplace=True)
            circ_ryy.s(0)
            circ_ryy.s(1)
            circ.compose(circ_ryy, inplace=True)

        # translate the RZZGate(c) into a circuit based on the desired Ctrl-U gate.
        if abs(self.decomposer.c) > atol:
            # Since the Weyl chamber is here defined as a > b > |c| we may have
            # negative c. This will cause issues in _to_rxx_gate
            # as TwoQubitWeylControlledEquiv will map (c, 0, 0) to (|c|, 0, 0).
            # We therefore produce RZZGate(|c|) and append its inverse to the
            # circuit if c < 0.
            gamma, invert = -2 * self.decomposer.c, False
            if gamma > 0:
                gamma *= -1
                invert = True

            circ_rzz = QuantumCircuit(2)
            circ_rzz.h(0)
            circ_rzz.h(1)
            circ_rzz.compose(self._to_rxx_gate(gamma), inplace=True)
            circ_rzz.h(0)
            circ_rzz.h(1)

            if invert:
                circ.compose(circ_rzz.inverse(), inplace=True)
            else:
                circ.compose(circ_rzz, inplace=True)

        return circ


def Ud(a, b, c):
    r"""Generates the array :math:`e^{(i a XX + i b YY + i c ZZ)}`"""
    return np.array(
        [
            [cmath.exp(1j * c) * math.cos(a - b), 0, 0, 1j * cmath.exp(1j * c) * math.sin(a - b)],
            [0, cmath.exp(-1j * c) * math.cos(a + b), 1j * cmath.exp(-1j * c) * math.sin(a + b), 0],
            [0, 1j * cmath.exp(-1j * c) * math.sin(a + b), cmath.exp(-1j * c) * math.cos(a + b), 0],
            [1j * cmath.exp(1j * c) * math.sin(a - b), 0, 0, cmath.exp(1j * c) * math.cos(a - b)],
        ],
        dtype=complex,
    )


def trace_to_fid(trace):
    r"""Average gate fidelity is

    .. math::

        \bar{F} = \frac{d + |\mathrm{Tr} (U_\text{target} \cdot U^{\dag})|^2}{d(d+1)}

    M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)"""
    return (4 + abs(trace) ** 2) / 20


def rz_array(theta):
    """Return numpy array for Rz(theta).

    Rz(theta) = diag(exp(-i*theta/2),exp(i*theta/2))
    """
    return np.array(
        [[cmath.exp(-1j * theta / 2.0), 0], [0, cmath.exp(1j * theta / 2.0)]], dtype=complex
    )


class TwoQubitBasisDecomposer:
    """A class for decomposing 2-qubit unitaries into minimal number of uses of a 2-qubit
    basis gate.

    Args:
        gate: Two-qubit gate to be used in the KAK decomposition.
        basis_fidelity: Fidelity to be assumed for applications of KAK Gate. Defaults to ``1.0``.
        euler_basis: Basis string to be provided to :class:`.OneQubitEulerDecomposer` for 1Q synthesis.
            Valid options are [``'ZYZ'``, ``'ZXZ'``, ``'XYX'``, ``'U'``, ``'U3'``, ``'U1X'``,
            ``'PSX'``, ``'ZSX'``, ``'RR'``].
        pulse_optimize: If ``True``, try to do decomposition which minimizes
            local unitaries in between entangling gates. This will raise an exception if an
            optimal decomposition is not implemented. Currently, only [{CX, SX, RZ}] is known.
            If ``False``, don't attempt optimization. If ``None``, attempt optimization but don't raise
            if unknown.


    .. automethod:: __call__
    """

    def __init__(
        self,
        gate: Gate,
        basis_fidelity: float = 1.0,
        euler_basis: str = "U",
        pulse_optimize: bool | None = None,
    ):
        self.gate = gate
        self.basis_fidelity = basis_fidelity
        self.pulse_optimize = pulse_optimize
        # Use cx as gate name for pulse optimal decomposition detection
        # otherwise use USER_GATE as a unique key to support custom gates
        # including parameterized gates like UnitaryGate.
        if isinstance(gate, CXGate):
            gate_name = "cx"
        else:
            gate_name = "USER_GATE"

        self._inner_decomposer = two_qubit_decompose.TwoQubitBasisDecomposer(
            gate_name,
            Operator(gate).data,
            basis_fidelity=basis_fidelity,
            euler_basis=euler_basis,
            pulse_optimize=pulse_optimize,
        )
        self.is_supercontrolled = self._inner_decomposer.super_controlled
        if not self.is_supercontrolled:
            warnings.warn(
                "Only know how to decompose properly for a supercontrolled basis gate.",
                stacklevel=2,
            )

    def num_basis_gates(self, unitary):
        """Computes the number of basis gates needed in
        a decomposition of input unitary
        """
        unitary = np.asarray(unitary, dtype=complex)
        return self._inner_decomposer.num_basis_gates(unitary)

    @staticmethod
    def decomp0(target):
        r"""
        Decompose target :math:`\sim U_d(x, y, z)` with :math:`0` uses of the basis gate.
        Result :math:`U_r` has trace:

        .. math::

            \Big\vert\text{Tr}(U_r\cdot U_\text{target}^{\dag})\Big\vert =
            4\Big\vert (\cos(x)\cos(y)\cos(z)+ j \sin(x)\sin(y)\sin(z)\Big\vert

        which is optimal for all targets and bases
        """

        return two_qubit_decompose.TwoQubitBasisDecomposer.decomp0(target)

    def decomp1(self, target):
        r"""Decompose target :math:`\sim U_d(x, y, z)` with :math:`1` use of the basis gate
        :math:`\sim U_d(a, b, c)`.
        Result :math:`U_r` has trace:

        .. math::

            \Big\vert\text{Tr}(U_r \cdot U_\text{target}^{\dag})\Big\vert =
            4\Big\vert \cos(x-a)\cos(y-b)\cos(z-c) + j \sin(x-a)\sin(y-b)\sin(z-c)\Big\vert

        which is optimal for all targets and bases with ``z==0`` or ``c==0``.
        """
        return self._inner_decomposer.decomp1(target)

    def decomp2_supercontrolled(self, target):
        r"""
        Decompose target :math:`\sim U_d(x, y, z)` with :math:`2` uses of the basis gate.

        For supercontrolled basis :math:`\sim U_d(\pi/4, b, 0)`, all b, result :math:`U_r` has trace

        .. math::

            \Big\vert\text{Tr}(U_r \cdot U_\text{target}^\dag) \Big\vert = 4\cos(z)

        which is the optimal approximation for basis of CNOT-class :math:`\sim U_d(\pi/4, 0, 0)`
        or DCNOT-class :math:`\sim U_d(\pi/4, \pi/4, 0)` and any target. It may
        be sub-optimal for :math:`b \neq 0` (i.e. there exists an exact decomposition for any target
        using :math:`B \sim U_d(\pi/4, \pi/8, 0)`, but it may not be this decomposition).
        This is an exact decomposition for supercontrolled basis and target :math:`\sim U_d(x, y, 0)`.
        No guarantees for non-supercontrolled basis.
        """
        return self._inner_decomposer.decomp2_supercontrolled(target)

    def decomp3_supercontrolled(self, target):
        r"""
        Decompose target with :math:`3` uses of the basis.
        This is an exact decomposition for supercontrolled basis :math:`\sim U_d(\pi/4, b, 0)`, all b,
        and any target. No guarantees for non-supercontrolled basis.
        """
        return self._inner_decomposer.decomp3_supercontrolled(target)

    def __call__(
        self,
        unitary: Operator | np.ndarray,
        basis_fidelity: float | None = None,
        approximate: bool = True,
        use_dag: bool = False,
        *,
        _num_basis_uses: int | None = None,
    ) -> QuantumCircuit | DAGCircuit:
        r"""Decompose a two-qubit ``unitary`` over fixed basis and :math:`SU(2)` using the best
        approximation given that each basis application has a finite ``basis_fidelity``.

        Args:
            unitary (Operator or ndarray): :math:`4 \times 4` unitary to synthesize.
            basis_fidelity (float or None): Fidelity to be assumed for applications of KAK Gate.
                If given, overrides ``basis_fidelity`` given at init.
            approximate (bool): Approximates if basis fidelities are less than 1.0.
            use_dag (bool): If true a :class:`.DAGCircuit` is returned instead of a
                :class:`QuantumCircuit` when this class is called.
            _num_basis_uses (int): force a particular approximation by passing a number in [0, 3].

        Returns:
            QuantumCircuit: Synthesized quantum circuit.

        Raises:
            QiskitError: if ``pulse_optimize`` is True but we don't know how to do it.
        """

        if use_dag:
            from qiskit.dagcircuit.dagcircuit import DAGCircuit
            from qiskit.dagcircuit.dagnode import DAGOpNode

            sequence = self._inner_decomposer(
                np.asarray(unitary, dtype=complex),
                basis_fidelity,
                approximate,
                _num_basis_uses=_num_basis_uses,
            )
            q = QuantumRegister(2)

            dag = DAGCircuit()
            dag.global_phase = sequence.global_phase
            dag.add_qreg(q)
            for gate, params, qubits in sequence:
                if gate is None:
                    dag.apply_operation_back(self.gate, tuple(q[x] for x in qubits), check=False)
                else:
                    op = CircuitInstruction.from_standard(
                        gate, qubits=tuple(q[x] for x in qubits), params=params
                    )
                    node = DAGOpNode.from_instruction(op)
                    dag._apply_op_node_back(node)
            return dag
        else:
            if getattr(self.gate, "_standard_gate", None):
                circ_data = self._inner_decomposer.to_circuit(
                    np.asarray(unitary, dtype=complex),
                    self.gate,
                    basis_fidelity,
                    approximate,
                    _num_basis_uses=_num_basis_uses,
                )
                return QuantumCircuit._from_circuit_data(circ_data, add_regs=True)
            else:
                sequence = self._inner_decomposer(
                    np.asarray(unitary, dtype=complex),
                    basis_fidelity,
                    approximate,
                    _num_basis_uses=_num_basis_uses,
                )
                q = QuantumRegister(2)
                circ = QuantumCircuit(q, global_phase=sequence.global_phase)
                for gate, params, qubits in sequence:
                    if gate is None:
                        circ._append(self.gate, qargs=tuple(q[x] for x in qubits))
                    else:
                        inst = CircuitInstruction.from_standard(
                            gate, qubits=tuple(q[x] for x in qubits), params=params
                        )
                        circ._append(inst)
                return circ

    def traces(self, target):
        r"""
        Give the expected traces :math:`\Big\vert\text{Tr}(U \cdot U_\text{target}^{\dag})\Big\vert`
        for a different number of basis gates.
        """
        return self._inner_decomposer.traces(target._inner_decomposition)


# This weird duplicated lazy structure is for backwards compatibility; Qiskit has historically
# always made ``two_qubit_cnot_decompose`` available publicly immediately on import, but it's quite
# expensive to construct, and we want to defer the object's creation until it's actually used.  We
# only need to pass through the public methods that take `self` as a parameter.  Using `__getattr__`
# doesn't work because it is only called if the normal resolution methods fail.  Using
# `__getattribute__` is too messy for a simple one-off use object.


class _LazyTwoQubitCXDecomposer(TwoQubitBasisDecomposer):
    __slots__ = ("_inner",)

    def __init__(self):  # pylint: disable=super-init-not-called
        self._inner = None

    def _load(self):
        if self._inner is None:
            self._inner = TwoQubitBasisDecomposer(CXGate())

    def __call__(self, *args, **kwargs) -> QuantumCircuit:
        self._load()
        return self._inner(*args, **kwargs)

    def traces(self, target):
        self._load()
        return self._inner.traces(target)

    def decomp1(self, target):
        self._load()
        return self._inner.decomp1(target)

    def decomp2_supercontrolled(self, target):
        self._load()
        return self._inner.decomp2_supercontrolled(target)

    def decomp3_supercontrolled(self, target):
        self._load()
        return self._inner.decomp3_supercontrolled(target)

    def num_basis_gates(self, unitary):
        self._load()
        return self._inner.num_basis_gates(unitary)


two_qubit_cnot_decompose = _LazyTwoQubitCXDecomposer()
"""
This is an instance of :class:`.TwoQubitBasisDecomposer` that always uses
``cx`` as the KAK gate for the basis decomposition. You can use this function
as a quick access to ``cx``-based 2-qubit decompositions.

Args:
    unitary (Operator or np.ndarray): The 4x4 unitary to synthesize.
    basis_fidelity (float or None): If given the assumed fidelity for applications of :class:`.CXGate`.
    approximate (bool): If ``True`` approximate if ``basis_fidelity`` is less than 1.0.

Returns:
    QuantumCircuit: The synthesized circuit of the input unitary.
"""

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""X, CX, CCX and multi-controlled X gates."""
from __future__ import annotations
from typing import Optional, Union, Type
from math import pi
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int, with_gate_array, with_controlled_gate_array
from qiskit._accelerate.circuit import StandardGate
from qiskit.utils.deprecation import deprecate_func

_X_ARRAY = [[0, 1], [1, 0]]
_SX_ARRAY = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]


@with_gate_array(_X_ARRAY)
class XGate(SingletonGate):
    r"""The single-qubit Pauli-X gate (:math:`\sigma_x`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.x` method.

    **Matrix Representation:**

    .. math::

        X = \begin{pmatrix}
                0 & 1 \\
                1 & 0
            \end{pmatrix}

    **Circuit symbol:**

    .. code-block:: text

             ┌───┐
        q_0: ┤ X ├
             └───┘

    Equivalent to a :math:`\pi` radian rotation about the X axis.

    .. note::

        A global phase difference exists between the definitions of
        :math:`RX(\pi)` and :math:`X`.

        .. math::

            RX(\pi) = \begin{pmatrix}
                        0 & -i \\
                        -i & 0
                      \end{pmatrix}
                    = -i X

    The gate is equivalent to a classical bit flip.

    .. math::

        |0\rangle \rightarrow |1\rangle \\
        |1\rangle \rightarrow |0\rangle
    """

    _standard_gate = StandardGate.XGate

    def __init__(self, label: Optional[str] = None, *, duration=None, unit="dt"):
        """Create new X gate."""
        super().__init__("x", 1, [], label=label, duration=duration, unit=unit)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate x a { u3(pi,0,pi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(pi, 0, pi), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        annotated: bool = False,
    ):
        """Return a (multi-)controlled-X gate.

        One control returns a CX gate. Two controls returns a CCX gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g. ``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated:
            gate = MCXGate(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                _base_label=self.label,
            )
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted X gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            XGate: inverse gate (self-inverse).
        """
        return XGate()  # self-inverse

    def __eq__(self, other):
        return isinstance(other, XGate)


@with_controlled_gate_array(_X_ARRAY, num_ctrl_qubits=1)
class CXGate(SingletonControlledGate):
    r"""Controlled-X gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cx` and
    :meth:`~qiskit.circuit.QuantumCircuit.cnot` methods.

    **Circuit symbol:**

    .. code-block:: text

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ X ├
             └───┘

    **Matrix representation:**

    .. math::

        CX\ q_0, q_1 =
            I \otimes |0\rangle\langle0| + X \otimes |1\rangle\langle1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌───┐
            q_0: ┤ X ├
                 └─┬─┘
            q_1: ──■──

        .. math::

            CX\ q_1, q_0 =
                |0 \rangle\langle 0| \otimes I + |1 \rangle\langle 1| \otimes X =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1 \\
                    0 & 0 & 1 & 0
                \end{pmatrix}


    In the computational basis, this gate flips the target qubit
    if the control qubit is in the :math:`|1\rangle` state.
    In this sense it is similar to a classical XOR gate.

    .. math::
        `|a, b\rangle \rightarrow |a, a \oplus b\rangle`
    """

    _standard_gate = StandardGate.CXGate

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create new CX gate."""
        super().__init__(
            "cx",
            2,
            [],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=XGate(label=_base_label),
            _base_label=_base_label,
            duration=duration,
            unit=unit,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        annotated: bool = False,
    ):
        """Return a controlled-X gate with more control lines.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g. ``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated:
            ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
            new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
            gate = MCXGate(
                num_ctrl_qubits=num_ctrl_qubits + 1,
                label=label,
                ctrl_state=new_ctrl_state,
                _base_label=self.label,
            )
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        """Return inverted CX gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CXGate: inverse gate (self-inverse).
        """
        return CXGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __eq__(self, other):
        return isinstance(other, CXGate) and self.ctrl_state == other.ctrl_state


@with_controlled_gate_array(_X_ARRAY, num_ctrl_qubits=2, cached_states=(3,))
class CCXGate(SingletonControlledGate):
    r"""CCX gate, also known as Toffoli gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ccx` and
    :meth:`~qiskit.circuit.QuantumCircuit.toffoli` methods.

    **Circuit symbol:**

    .. code-block:: text

        q_0: ──■──
               │
        q_1: ──■──
             ┌─┴─┐
        q_2: ┤ X ├
             └───┘

    **Matrix representation:**

    .. math::

        CCX q_0, q_1, q_2 =
            I \otimes I \otimes |0 \rangle \langle 0| + CX \otimes |1 \rangle \langle 1| =
           \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_2 and q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌───┐
            q_0: ┤ X ├
                 └─┬─┘
            q_1: ──■──
                   │
            q_2: ──■──

        .. math::

            CCX\ q_2, q_1, q_0 =
                |0 \rangle \langle 0| \otimes I \otimes I + |1 \rangle \langle 1| \otimes CX =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
                \end{pmatrix}

    """

    _standard_gate = StandardGate.CCXGate

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create new CCX gate."""
        super().__init__(
            "ccx",
            3,
            [],
            num_ctrl_qubits=2,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=XGate(label=_base_label),
            duration=duration,
            unit=unit,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=2)

    def _define(self):
        """
        gate ccx a,b,c
        {
        h c; cx b,c; tdg c; cx a,c;
        t c; cx b,c; tdg c; cx a,c;
        t b; t c; h c; cx a,b;
        t a; tdg b; cx a,b;}
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .t import TGate, TdgGate

        #                                                        ┌───┐
        # q_0: ───────────────────■─────────────────────■────■───┤ T ├───■──
        #                         │             ┌───┐   │  ┌─┴─┐┌┴───┴┐┌─┴─┐
        # q_1: ───────■───────────┼─────────■───┤ T ├───┼──┤ X ├┤ Tdg ├┤ X ├
        #      ┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌┴───┴┐┌─┴─┐├───┤└┬───┬┘└───┘
        # q_2: ┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├─┤ H ├──────
        #      └───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘ └───┘
        q = QuantumRegister(3, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (HGate(), [q[2]], []),
            (CXGate(), [q[1], q[2]], []),
            (TdgGate(), [q[2]], []),
            (CXGate(), [q[0], q[2]], []),
            (TGate(), [q[2]], []),
            (CXGate(), [q[1], q[2]], []),
            (TdgGate(), [q[2]], []),
            (CXGate(), [q[0], q[2]], []),
            (TGate(), [q[1]], []),
            (TGate(), [q[2]], []),
            (HGate(), [q[2]], []),
            (CXGate(), [q[0], q[1]], []),
            (TGate(), [q[0]], []),
            (TdgGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        annotated: bool = False,
    ):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g. ``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated:
            ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
            new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
            gate = MCXGate(
                num_ctrl_qubits=num_ctrl_qubits + 2,
                label=label,
                ctrl_state=new_ctrl_state,
                _base_label=self.label,
            )
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        """Return an inverted CCX gate (also a CCX).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CCXGate: inverse gate (self-inverse).
        """
        return CCXGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __eq__(self, other):
        return isinstance(other, CCXGate) and self.ctrl_state == other.ctrl_state


@with_gate_array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1j],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1j, 0, 0, 0, 0],
    ]
)
class RCCXGate(SingletonGate):
    """The simplified Toffoli gate, also referred to as Margolus gate.

    The simplified Toffoli gate implements the Toffoli gate up to relative phases.
    This implementation requires three CX gates which is the minimal amount possible,
    as shown in https://arxiv.org/abs/quant-ph/0312225.
    Note, that the simplified Toffoli is not equivalent to the Toffoli. But can be used in places
    where the Toffoli gate is uncomputed again.

    This concrete implementation is from https://arxiv.org/abs/1508.03273, the dashed box
    of Fig. 3.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rccx` method.
    """

    _standard_gate = StandardGate.RCCXGate

    def __init__(self, label: Optional[str] = None, *, duration=None, unit="dt"):
        """Create a new simplified CCX gate."""
        super().__init__("rccx", 3, [], label=label, duration=duration, unit=unit)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate rccx a,b,c
        { u2(0,pi) c;
          u1(pi/4) c;
          cx b, c;
          u1(-pi/4) c;
          cx a, c;
          u1(pi/4) c;
          cx b, c;
          u1(-pi/4) c;
          u2(0,pi) c;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        from .u2 import U2Gate

        q = QuantumRegister(3, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U2Gate(0, pi), [q[2]], []),  # H gate
            (U1Gate(pi / 4), [q[2]], []),  # T gate
            (CXGate(), [q[1], q[2]], []),
            (U1Gate(-pi / 4), [q[2]], []),  # inverse T gate
            (CXGate(), [q[0], q[2]], []),
            (U1Gate(pi / 4), [q[2]], []),
            (CXGate(), [q[1], q[2]], []),
            (U1Gate(-pi / 4), [q[2]], []),  # inverse T gate
            (U2Gate(0, pi), [q[2]], []),  # H gate
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def __eq__(self, other):
        return isinstance(other, RCCXGate)


@with_controlled_gate_array(_SX_ARRAY, num_ctrl_qubits=3, cached_states=(7,))
class C3SXGate(SingletonControlledGate):
    """The 3-qubit controlled sqrt-X gate.

    This implementation is based on Page 17 of [1].

    References:
        [1] Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
    """

    _standard_gate = StandardGate.C3SXGate

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create a new 3-qubit controlled sqrt-X gate.

        Args:
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g. ``'110'``), or ``None``. If ``None``, use all 1s.
        """
        from .sx import SXGate

        super().__init__(
            "c3sx",
            4,
            [],
            num_ctrl_qubits=3,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=SXGate(label=_base_label),
            duration=duration,
            unit=unit,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=3)

    def _define(self):
        """
        gate c3sqrtx a,b,c,d
        {
            h d; cu1(pi/8) a,d; h d;
            cx a,b;
            h d; cu1(-pi/8) b,d; h d;
            cx a,b;
            h d; cu1(pi/8) b,d; h d;
            cx b,c;
            h d; cu1(-pi/8) c,d; h d;
            cx a,c;
            h d; cu1(pi/8) c,d; h d;
            cx b,c;
            h d; cu1(-pi/8) c,d; h d;
            cx a,c;
            h d; cu1(pi/8) c,d; h d;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import CU1Gate
        from .h import HGate

        angle = numpy.pi / 8
        q = QuantumRegister(4, name="q")
        rules = [
            (HGate(), [q[3]], []),
            (CU1Gate(angle), [q[0], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(-angle), [q[1], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(angle), [q[1], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[1], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(-angle), [q[2], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(angle), [q[2], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[1], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(-angle), [q[2], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(angle), [q[2], q[3]], []),
            (HGate(), [q[3]], []),
        ]
        qc = QuantumCircuit(q)
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def __eq__(self, other):
        return isinstance(other, C3SXGate) and self.ctrl_state == other.ctrl_state


@with_controlled_gate_array(_X_ARRAY, num_ctrl_qubits=3, cached_states=(7,))
class C3XGate(SingletonControlledGate):
    r"""The X gate controlled on 3 qubits.

    This implementation uses :math:`\sqrt{T}` and 14 CNOT gates.
    """

    _standard_gate = StandardGate.C3XGate

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        _base_label=None,
        duration=None,
        unit="dt",
    ):
        """Create a new 3-qubit controlled X gate."""
        super().__init__(
            "mcx",
            4,
            [],
            num_ctrl_qubits=3,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=XGate(label=_base_label),
            duration=duration,
            unit=unit,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=3)

    # seems like open controls not happening?
    def _define(self):
        """
        gate c3x a,b,c,d
        {
            h d;
            p(pi/8) a;
            p(pi/8) b;
            p(pi/8) c;
            p(pi/8) d;
            cx a, b;
            p(-pi/8) b;
            cx a, b;
            cx b, c;
            p(-pi/8) c;
            cx a, c;
            p(pi/8) c;
            cx b, c;
            p(-pi/8) c;
            cx a, c;
            cx c, d;
            p(-pi/8) d;
            cx b, d;
            p(pi/8) d;
            cx c, d;
            p(-pi/8) d;
            cx a, d;
            p(pi/8) d;
            cx c, d;
            p(-pi/8) d;
            cx b, d;
            p(pi/8) d;
            cx c, d;
            p(-pi/8) d;
            cx a, d;
            h d;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        q = QuantumRegister(4, name="q")
        qc = QuantumCircuit(q, name=self.name)
        qc.h(3)
        qc.p(pi / 8, [0, 1, 2, 3])
        qc.cx(0, 1)
        qc.p(-pi / 8, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.p(-pi / 8, 2)
        qc.cx(0, 2)
        qc.p(pi / 8, 2)
        qc.cx(1, 2)
        qc.p(-pi / 8, 2)
        qc.cx(0, 2)
        qc.cx(2, 3)
        qc.p(-pi / 8, 3)
        qc.cx(1, 3)
        qc.p(pi / 8, 3)
        qc.cx(2, 3)
        qc.p(-pi / 8, 3)
        qc.cx(0, 3)
        qc.p(pi / 8, 3)
        qc.cx(2, 3)
        qc.p(-pi / 8, 3)
        qc.cx(1, 3)
        qc.p(pi / 8, 3)
        qc.cx(2, 3)
        qc.p(-pi / 8, 3)
        qc.cx(0, 3)
        qc.h(3)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        annotated: bool = False,
    ):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g. ``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated:
            ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
            new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
            gate = MCXGate(
                num_ctrl_qubits=num_ctrl_qubits + 3,
                label=label,
                ctrl_state=new_ctrl_state,
                _base_label=self.label,
            )
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        """Invert this gate. The C3X is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            C3XGate: inverse gate (self-inverse).
        """
        return C3XGate(ctrl_state=self.ctrl_state)

    def __eq__(self, other):
        return isinstance(other, C3XGate) and self.ctrl_state == other.ctrl_state


@with_gate_array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1j, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
class RC3XGate(SingletonGate):
    """The simplified 3-controlled Toffoli gate.

    The simplified Toffoli gate implements the Toffoli gate up to relative phases.
    Note, that the simplified Toffoli is not equivalent to the Toffoli. But can be used in places
    where the Toffoli gate is uncomputed again.

    This concrete implementation is from https://arxiv.org/abs/1508.03273, the complete circuit
    of Fig. 4.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rcccx` method.
    """

    _standard_gate = StandardGate.RC3XGate

    def __init__(self, label: Optional[str] = None, *, duration=None, unit="dt"):
        """Create a new RC3X gate."""
        super().__init__("rcccx", 4, [], label=label, duration=duration, unit=unit)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate rc3x a,b,c,d
        { u2(0,pi) d;
          u1(pi/4) d;
          cx c,d;
          u1(-pi/4) d;
          u2(0,pi) d;
          cx a,d;
          u1(pi/4) d;
          cx b,d;
          u1(-pi/4) d;
          cx a,d;
          u1(pi/4) d;
          cx b,d;
          u1(-pi/4) d;
          u2(0,pi) d;
          u1(pi/4) d;
          cx c,d;
          u1(-pi/4) d;
          u2(0,pi) d;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        from .u2 import U2Gate

        q = QuantumRegister(4, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U2Gate(0, pi), [q[3]], []),  # H gate
            (U1Gate(pi / 4), [q[3]], []),  # T gate
            (CXGate(), [q[2], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),  # inverse T gate
            (U2Gate(0, pi), [q[3]], []),
            (CXGate(), [q[0], q[3]], []),
            (U1Gate(pi / 4), [q[3]], []),
            (CXGate(), [q[1], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),
            (CXGate(), [q[0], q[3]], []),
            (U1Gate(pi / 4), [q[3]], []),
            (CXGate(), [q[1], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),
            (U2Gate(0, pi), [q[3]], []),
            (U1Gate(pi / 4), [q[3]], []),
            (CXGate(), [q[2], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),
            (U2Gate(0, pi), [q[3]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def __eq__(self, other):
        return isinstance(other, RC3XGate)


@with_controlled_gate_array(_X_ARRAY, num_ctrl_qubits=4, cached_states=(15,))
class C4XGate(SingletonControlledGate):
    """The 4-qubit controlled X gate.

    This implementation is based on Page 21, Lemma 7.5, of [1], with the use
    of the relative phase version of c3x, the rc3x [2].

    References:
        1. Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
        2. Maslov, 2015. https://arxiv.org/abs/1508.03273
    """

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create a new 4-qubit controlled X gate."""
        if unit is None:
            unit = "dt"
        super().__init__(
            "mcx",
            5,
            [],
            num_ctrl_qubits=4,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=XGate(label=_base_label),
            duration=duration,
            unit=unit,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=4)

    # seems like open controls not happening?
    def _define(self):
        """
        gate c3sqrtx a,b,c,d
        {
            h d; cu1(pi/8) a,d; h d;
            cx a,b;
            h d; cu1(-pi/8) b,d; h d;
            cx a,b;
            h d; cu1(pi/8) b,d; h d;
            cx b,c;
            h d; cu1(-pi/8) c,d; h d;
            cx a,c;
            h d; cu1(pi/8) c,d; h d;
            cx b,c;
            h d; cu1(-pi/8) c,d; h d;
            cx a,c;
            h d; cu1(pi/8) c,d; h d;
        }
        gate c4x a,b,c,d,e
        {
            h e; cu1(pi/2) d,e; h e;
            rc3x a,b,c,d;
            h e; cu1(-pi/2) d,e; h e;
            rc3x a,b,c,d;
            c3sqrtx a,b,c,e;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import CU1Gate
        from .h import HGate

        q = QuantumRegister(5, name="q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (HGate(), [q[4]], []),
            (CU1Gate(numpy.pi / 2), [q[3], q[4]], []),
            (HGate(), [q[4]], []),
            (RC3XGate(), [q[0], q[1], q[2], q[3]], []),
            (HGate(), [q[4]], []),
            (CU1Gate(-numpy.pi / 2), [q[3], q[4]], []),
            (HGate(), [q[4]], []),
            (RC3XGate().inverse(), [q[0], q[1], q[2], q[3]], []),
            (C3SXGate(), [q[0], q[1], q[2], q[4]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        annotated: bool = False,
    ):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g. ``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated:
            ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
            new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
            gate = MCXGate(
                num_ctrl_qubits=num_ctrl_qubits + 4,
                label=label,
                ctrl_state=new_ctrl_state,
                _base_label=self.label,
            )
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        """Invert this gate. The C4X is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            C4XGate: inverse gate (self-inverse).
        """
        return C4XGate(ctrl_state=self.ctrl_state)

    def __eq__(self, other):
        return isinstance(other, C4XGate) and self.ctrl_state == other.ctrl_state


class MCXGate(ControlledGate):
    """The general, multi-controlled X gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.mcx` method.
    """

    def __new__(
        cls,
        num_ctrl_qubits: Optional[int] = None,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create a new MCX instance.

        Depending on the number of controls and which mode of the MCX, this creates an
        explicit CX, CCX, C3X or C4X instance or a generic MCX gate.
        """
        # The CXGate and CCXGate will be implemented for all modes of the MCX, and
        # the C3XGate and C4XGate are handled in the gate definition.
        explicit: dict[int, Type[ControlledGate]] = {1: CXGate, 2: CCXGate}
        gate_class = explicit.get(num_ctrl_qubits, None)
        if gate_class is not None:
            gate = gate_class.__new__(
                gate_class, label=label, ctrl_state=ctrl_state, _base_label=_base_label
            )
            # if __new__ does not return the same type as cls, init is not called
            gate.__init__(
                label=label,
                ctrl_state=ctrl_state,
                _base_label=_base_label,
                duration=duration,
                unit=unit,
            )
            return gate
        return super().__new__(cls)

    def __init__(
        self,
        num_ctrl_qubits: int,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _name="mcx",
        _base_label=None,
    ):
        """Create new MCX gate."""
        num_ancilla_qubits = self.__class__.get_num_ancilla_qubits(num_ctrl_qubits)
        super().__init__(
            _name,
            num_ctrl_qubits + 1 + num_ancilla_qubits,
            [],
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=XGate(label=_base_label),
        )

    def inverse(self, annotated: bool = False):
        """Invert this gate. The MCX is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            MCXGate: inverse gate (self-inverse).
        """
        return MCXGate(num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state)

    @staticmethod
    @deprecate_func(
        additional_msg=(
            "For an MCXGate it is no longer possible to know the number of ancilla qubits "
            "that would be eventually used by the transpiler when the gate is created. "
            "Instead, it is recommended to use MCXGate and let HighLevelSynthesis choose "
            "the best synthesis method depending on the number of ancilla qubits available. "
            "However, if a specific synthesis method using a specific number of ancilla "
            "qubits is require, one can create a custom gate by calling the corresponding "
            "synthesis function directly."
        ),
        since="1.3",
        pending=True,
    )
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str = "noancilla") -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        This staticmethod might be necessary to check the number of ancillas before
        creating the gate, or to use the number of ancillas in the initialization.
        """
        if mode == "noancilla":
            return 0
        if mode in ["recursion", "advanced"]:
            return int(num_ctrl_qubits > 4)
        if mode[:7] == "v-chain" or mode[:5] == "basic":
            return max(0, num_ctrl_qubits - 2)
        raise AttributeError(f"Unsupported mode ({mode}) specified!")

    def _define(self):
        """This definition is based on MCPhaseGate implementation."""
        # pylint: disable=cyclic-import
        from qiskit.synthesis.multi_controlled import synth_mcx_noaux_v24

        qc = synth_mcx_noaux_v24(self.num_ctrl_qubits)
        self.definition = qc

    @property
    def num_ancilla_qubits(self):
        """The number of ancilla qubits."""
        return self.__class__.get_num_ancilla_qubits(self.num_ctrl_qubits)

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        annotated: bool = False,
    ):
        """Return a multi-controlled-X gate with more control lines.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g. ``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated and ctrl_state is None:
            # use __class__ so this works for derived classes
            gate = self.__class__(
                self.num_ctrl_qubits + num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                _base_label=self.label,
            )
        else:
            gate = super().control(num_ctrl_qubits, label=label, ctrl_state=ctrl_state)
        return gate


class MCXGrayCode(MCXGate):
    r"""Implement the multi-controlled X gate using the Gray code.

    This delegates the implementation to the MCU1 gate, since :math:`X = H \cdot U1(\pi) \cdot H`.
    """

    def __new__(
        cls,
        num_ctrl_qubits: Optional[int] = None,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create a new MCXGrayCode instance"""
        # if 1 to 4 control qubits, create explicit gates
        explicit = {1: CXGate, 2: CCXGate, 3: C3XGate, 4: C4XGate}
        gate_class = explicit.get(num_ctrl_qubits, None)
        if gate_class is not None:
            gate = gate_class.__new__(
                gate_class,
                label=label,
                ctrl_state=ctrl_state,
                _base_label=_base_label,
                duration=duration,
                unit=unit,
            )
            # if __new__ does not return the same type as cls, init is not called
            gate.__init__(
                label=label,
                ctrl_state=ctrl_state,
                duration=duration,
                unit=unit,
            )
            return gate
        return super().__new__(cls)

    @deprecate_func(
        additional_msg=(
            "It is recommended to use MCXGate and let HighLevelSynthesis choose "
            "the best synthesis method depending on the number of ancilla qubits available. "
            "If this specific synthesis method is required, one can specify it using the "
            "high-level-synthesis plugin `gray_code` for MCX gates, or, alternatively, "
            "one can use synth_mcx_gray_code to construct the gate directly."
        ),
        since="1.3",
        pending=True,
    )
    def __init__(
        self,
        num_ctrl_qubits: int,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        super().__init__(num_ctrl_qubits, label=label, ctrl_state=ctrl_state, _name="mcx_gray")

    def inverse(self, annotated: bool = False):
        """Invert this gate. The MCX is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            MCXGrayCode: inverse gate (self-inverse).
        """
        return MCXGrayCode(num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state)

    def _define(self):
        """Define the MCX gate using the Gray code."""
        # pylint: disable=cyclic-import
        from qiskit.synthesis.multi_controlled import synth_mcx_gray_code

        qc = synth_mcx_gray_code(self.num_ctrl_qubits)
        self.definition = qc


class MCXRecursive(MCXGate):
    """Implement the multi-controlled X gate using recursion.

    Using a single clean ancilla qubit, the multi-controlled X gate is split into
    four sub-registers, each one of them uses the V-chain method.

    The method is based on Lemma 9 of [2], first shown in Lemma 7.3 of [1].

    References:
        1. Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
        2. Iten et al., 2015. https://arxiv.org/abs/1501.06911
    """

    @deprecate_func(
        additional_msg=(
            "It is recommended to use MCXGate and let HighLevelSynthesis choose "
            "the best synthesis method depending on the number of ancilla qubits available. "
            "If this specific synthesis method is required, one can specify it using the "
            "high-level-synthesis plugin '1_clean_b95' for MCX gates, or, alternatively, "
            "one can use synth_mcx_1_clean to construct the gate directly."
        ),
        since="1.3",
        pending=True,
    )
    def __init__(
        self,
        num_ctrl_qubits: int,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        super().__init__(
            num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            _name="mcx_recursive",
            duration=duration,
            unit=unit,
            _base_label=None,
        )

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str = "recursion"):
        """Get the number of required ancilla qubits."""
        return MCXGate.get_num_ancilla_qubits(num_ctrl_qubits, mode)

    def inverse(self, annotated: bool = False):
        """Invert this gate. The MCX is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            MCXRecursive: inverse gate (self-inverse).
        """
        return MCXRecursive(num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state)

    def _define(self):
        """Define the MCX gate using recursion."""

        # pylint: disable=cyclic-import
        from qiskit.synthesis.multi_controlled import synth_mcx_1_clean_b95

        qc = synth_mcx_1_clean_b95(self.num_ctrl_qubits)
        self.definition = qc


class MCXVChain(MCXGate):
    """Implement the multi-controlled X gate using a V-chain of CX gates."""

    def __new__(
        cls,
        num_ctrl_qubits: Optional[int] = None,
        dirty_ancillas: bool = False,  # pylint: disable=unused-argument
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
        relative_phase: bool = False,  # pylint: disable=unused-argument
        action_only: bool = False,  # pylint: disable=unused-argument
    ):
        """Create a new MCX instance.

        This must be defined anew to include the additional argument ``dirty_ancillas``.
        """
        return super().__new__(
            cls,
            num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            _base_label=_base_label,
            duration=duration,
            unit=unit,
        )

    @deprecate_func(
        additional_msg=(
            "It is recommended to use MCXGate and let HighLevelSynthesis choose "
            "the best synthesis method depending on the number of ancilla qubits available. "
            "If this specific synthesis method is required, one can specify it using the "
            "high-level-synthesis plugins `n_clean_m15` (using clean ancillas) or "
            "`n_dirty_i15` (using dirty ancillas) for MCX gates. Alternatively, one can "
            "use synth_mcx_n_dirty_i15 and synth_mcx_n_clean_m15 to construct the gate directly."
        ),
        since="1.3",
        pending=True,
    )
    def __init__(
        self,
        num_ctrl_qubits: int,
        dirty_ancillas: bool = False,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
        relative_phase: bool = False,
        action_only: bool = False,
    ):
        """
        Args:
            dirty_ancillas: when set to ``True``, the method applies an optimized multicontrolled-X gate
                up to a relative phase using dirty ancillary qubits with the properties of lemmas 7 and 8
                from arXiv:1501.06911, with at most 8*k - 6 CNOT gates.
                For k within the range {1, ..., ceil(n/2)}. And for n representing the total number of
                qubits.
            relative_phase: when set to ``True``, the method applies the optimized multicontrolled-X gate
                up to a relative phase, in a way that, by lemma 7 of arXiv:1501.06911, the relative
                phases of the ``action part`` cancel out with the phases of the ``reset part``.

            action_only: when set to ``True``, the method applies only the action part of lemma 8
                from arXiv:1501.06911.

        """
        super().__init__(
            num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            _name="mcx_vchain",
            _base_label=_base_label,
            duration=duration,
            unit=unit,
        )
        self._dirty_ancillas = dirty_ancillas
        self._relative_phase = relative_phase
        self._action_only = action_only
        super().__init__(num_ctrl_qubits, label=label, ctrl_state=ctrl_state, _name="mcx_vchain")

    def inverse(self, annotated: bool = False):
        """Invert this gate. The MCX is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            MCXVChain: inverse gate (self-inverse).
        """
        return MCXVChain(
            num_ctrl_qubits=self.num_ctrl_qubits,
            dirty_ancillas=self._dirty_ancillas,
            ctrl_state=self.ctrl_state,
            relative_phase=self._relative_phase,
            action_only=self._action_only,
        )

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str = "v-chain"):
        """Get the number of required ancilla qubits."""
        return MCXGate.get_num_ancilla_qubits(num_ctrl_qubits, mode)

    def _define(self):
        """Define the MCX gate using a V-chain of CX gates."""

        if self._dirty_ancillas:
            # pylint: disable=cyclic-import
            from qiskit.synthesis.multi_controlled import synth_mcx_n_dirty_i15

            qc = synth_mcx_n_dirty_i15(
                self.num_ctrl_qubits,
                self._relative_phase,
                self._action_only,
            )

        else:  # use clean ancillas
            # pylint: disable=cyclic-import
            from qiskit.synthesis.multi_controlled import synth_mcx_n_clean_m15

            qc = synth_mcx_n_clean_m15(self.num_ctrl_qubits)

        self.definition = qc

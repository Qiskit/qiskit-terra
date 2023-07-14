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
from math import ceil, pi
import numpy as np
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _compute_control_matrix, _ctrl_state_to_int
from .h import HGate
from .t import TGate, TdgGate
from .u1 import U1Gate
from .u2 import U2Gate
from .sx import SXGate


class XGate(Gate):
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

    .. parsed-literal::

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

    def __init__(self, label: Optional[str] = None):
        """Create new X gate."""
        super().__init__("x", 1, [], label=label)

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
    ):
        """Return a (multi-)controlled-X gate.

        One control returns a CX gate. Two controls returns a CCX gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        r"""Return inverted X gate (itself)."""
        return XGate()  # self-inverse

    def __array__(self, dtype=None):
        """Return a np.array for the X gate."""
        return np.array([[0, 1], [1, 0]], dtype=dtype)


class CXGate(ControlledGate):
    r"""Controlled-X gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cx` and
    :meth:`~qiskit.circuit.QuantumCircuit.cnot` methods.

    **Circuit symbol:**

    .. parsed-literal::

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

        .. parsed-literal::
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

    def __init__(self, label: Optional[str] = None, ctrl_state: Optional[Union[str, int]] = None):
        """Create new CX gate."""
        super().__init__(
            "cx", 2, [], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=XGate()
        )

    def _define_qasm3(self):
        from qiskit.qasm3.ast import (
            Constant,
            Identifier,
            Integer,
            QuantumBlock,
            QuantumGateModifier,
            QuantumGateModifierName,
            QuantumGateSignature,
            QuantumGateDefinition,
            QuantumGateCall,
        )

        control, target = Identifier("c"), Identifier("t")
        call = QuantumGateCall(
            Identifier("U"),
            [control, target],
            parameters=[Constant.PI, Integer(0), Constant.PI],
            modifiers=[QuantumGateModifier(QuantumGateModifierName.CTRL)],
        )
        return QuantumGateDefinition(
            QuantumGateSignature(Identifier("cx"), [control, target]),
            QuantumBlock([call]),
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Return a controlled-X gate with more control lines.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 1, label=label, ctrl_state=new_ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        """Return inverted CX gate (itself)."""
        return CXGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __array__(self, dtype=None):
        """Return a np.array for the CX gate."""
        if self.ctrl_state:
            return np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=dtype)
        else:
            return np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=dtype)


class CCXGate(ControlledGate):
    r"""CCX gate, also known as Toffoli gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ccx` and
    :meth:`~qiskit.circuit.QuantumCircuit.toffoli` methods.

    **Circuit symbol:**

    .. parsed-literal::

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

        .. parsed-literal::
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

    def __init__(self, label: Optional[str] = None, ctrl_state: Optional[Union[str, int]] = None):
        """Create new CCX gate."""
        super().__init__(
            "ccx", 3, [], num_ctrl_qubits=2, label=label, ctrl_state=ctrl_state, base_gate=XGate()
        )

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
    ):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 2, label=label, ctrl_state=new_ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        """Return an inverted CCX gate (also a CCX)."""
        return CCXGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __array__(self, dtype=None):
        """Return a np.array for the CCX gate."""
        mat = _compute_control_matrix(
            self.base_gate.to_matrix(), self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )
        if dtype:
            return np.asarray(mat, dtype=dtype)
        return mat


class RCCXGate(Gate):
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

    def __init__(self, label: Optional[str] = None):
        """Create a new simplified CCX gate."""
        super().__init__("rccx", 3, [], label=label)

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

    def __array__(self, dtype=None):
        """Return a np.array for the simplified CCX gate."""
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1j],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1j, 0, 0, 0, 0],
            ],
            dtype=dtype,
        )


class C3SXGate(ControlledGate):
    """The 3-qubit controlled sqrt-X gate.

    This implementation is based on Page 17 of [1].

    References:
        [1] Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
    """

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Create a new 3-qubit controlled sqrt-X gate.

        Args:
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.
        """
        super().__init__(
            "c3sx", 4, [], num_ctrl_qubits=3, label=label, ctrl_state=ctrl_state, base_gate=SXGate()
        )

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

        angle = np.pi / 8
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

    def qasm(self):
        # Gross hack to override the Qiskit name with the name this gate has in Terra's version of
        # 'qelib1.inc'.  In general, the larger exporter mechanism should know about this to do the
        # mapping itself, but right now that's not possible without a complete rewrite of the OQ2
        # exporter code (low priority), or we would need to modify 'qelib1.inc' which would be
        # needlessly disruptive this late in OQ2's lifecycle.  The current OQ2 exporter _always_
        # outputs the `include 'qelib1.inc' line.  ---Jake, 2022-11-21.
        old_name = self.name
        self.name = "c3sqrtx"
        try:
            return super().qasm()
        finally:
            self.name = old_name


class C3XGate(ControlledGate):
    r"""The X gate controlled on 3 qubits.

    This implementation uses :math:`\sqrt{T}` and 14 CNOT gates.
    """

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Create a new 3-qubit controlled X gate."""
        super().__init__(
            "mcx", 4, [], num_ctrl_qubits=3, label=label, ctrl_state=ctrl_state, base_gate=XGate()
        )

    # seems like open controls not hapening?
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
    ):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 3, label=label, ctrl_state=new_ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        """Invert this gate. The C4X is its own inverse."""
        return C3XGate(ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        """Return a np.array for the C4X gate."""
        mat = _compute_control_matrix(
            self.base_gate.to_matrix(), self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )
        if dtype:
            return np.asarray(mat, dtype=dtype)
        return mat


class RC3XGate(Gate):
    """The simplified 3-controlled Toffoli gate.

    The simplified Toffoli gate implements the Toffoli gate up to relative phases.
    Note, that the simplified Toffoli is not equivalent to the Toffoli. But can be used in places
    where the Toffoli gate is uncomputed again.

    This concrete implementation is from https://arxiv.org/abs/1508.03273, the complete circuit
    of Fig. 4.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rcccx` method.
    """

    def __init__(self, label: Optional[str] = None):
        """Create a new RC3X gate."""
        super().__init__("rcccx", 4, [], label=label)

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

    def __array__(self, dtype=None):
        """Return a np.array for the RC3X gate."""
        return np.array(
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
            ],
            dtype=dtype,
        )


class C4XGate(ControlledGate):
    """The 4-qubit controlled X gate.

    This implementation is based on Page 21, Lemma 7.5, of [1], with the use
    of the relative phase version of c3x, the rc3x [2].

    References:
        [1] Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
        [2] Maslov, 2015. https://arxiv.org/abs/1508.03273
    """

    def __init__(self, label: Optional[str] = None, ctrl_state: Optional[Union[str, int]] = None):
        """Create a new 4-qubit controlled X gate."""
        super().__init__(
            "mcx", 5, [], num_ctrl_qubits=4, label=label, ctrl_state=ctrl_state, base_gate=XGate()
        )

    # seems like open controls not hapening?
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

        q = QuantumRegister(5, name="q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (HGate(), [q[4]], []),
            (CU1Gate(np.pi / 2), [q[3], q[4]], []),
            (HGate(), [q[4]], []),
            (RC3XGate(), [q[0], q[1], q[2], q[3]], []),
            (HGate(), [q[4]], []),
            (CU1Gate(-np.pi / 2), [q[3], q[4]], []),
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
    ):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 4, label=label, ctrl_state=new_ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        """Invert this gate. The C4X is its own inverse."""
        return C4XGate(ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        """Return a np.array for the C4X gate."""
        mat = _compute_control_matrix(
            self.base_gate.to_matrix(), self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )
        if dtype:
            return np.asarray(mat, dtype=dtype)
        return mat


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
    ):
        """Create a new MCX instance.

        Depending on the number of controls and which mode of the MCX, this creates an
        explicit CX, CCX, C3X or C4X instance or a generic MCX gate.
        """
        # The CXGate and CCXGate will be implemented for all modes of the MCX, and
        # the C3XGate and C4XGate will be implemented in the MCXGrayCode class.
        explicit: dict[int, Type[ControlledGate]] = {1: CXGate, 2: CCXGate}
        if num_ctrl_qubits in explicit:
            gate_class = explicit[num_ctrl_qubits]
            gate = gate_class.__new__(gate_class, label=label, ctrl_state=ctrl_state)
            # if __new__ does not return the same type as cls, init is not called
            gate.__init__(label=label, ctrl_state=ctrl_state)
            return gate
        return super().__new__(cls)

    def __init__(
        self,
        num_ctrl_qubits: int,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        _name="mcx",
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
            base_gate=XGate(),
        )

    def inverse(self):
        """Invert this gate. The MCX is its own inverse."""
        return MCXGate(num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state)

    @staticmethod
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
        """The standard definition used the Gray code implementation."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        q = QuantumRegister(self.num_qubits, name="q")
        qc = QuantumCircuit(q)
        qc._append(MCXGrayCode(self.num_ctrl_qubits), q[:], [])
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
    ):
        """Return a multi-controlled-X gate with more control lines.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if ctrl_state is None:
            # use __class__ so this works for derived classes
            gate = self.__class__(
                self.num_ctrl_qubits + num_ctrl_qubits, label=label, ctrl_state=ctrl_state
            )
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits, label=label, ctrl_state=ctrl_state)


class MCXGrayCode(MCXGate):
    r"""Implement the multi-controlled X gate using the Gray code.

    This delegates the implementation to the MCU1 gate, since :math:`X = H \cdot U1(\pi) \cdot H`.
    """

    def __new__(
        cls,
        num_ctrl_qubits: Optional[int] = None,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Create a new MCXGrayCode instance"""
        # if 1 to 4 control qubits, create explicit gates
        explicit = {1: CXGate, 2: CCXGate, 3: C3XGate, 4: C4XGate}
        if num_ctrl_qubits in explicit:
            gate_class = explicit[num_ctrl_qubits]
            gate = gate_class.__new__(gate_class, label=label, ctrl_state=ctrl_state)
            # if __new__ does not return the same type as cls, init is not called
            gate.__init__(label=label, ctrl_state=ctrl_state)
            return gate
        return super().__new__(cls)

    def __init__(
        self,
        num_ctrl_qubits: int,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        super().__init__(num_ctrl_qubits, label=label, ctrl_state=ctrl_state, _name="mcx_gray")

    def inverse(self):
        """Invert this gate. The MCX is its own inverse."""
        return MCXGrayCode(num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state)

    def _define(self):
        """Define the MCX gate using the Gray code."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import MCU1Gate

        q = QuantumRegister(self.num_qubits, name="q")
        qc = QuantumCircuit(q, name=self.name)
        qc._append(HGate(), [q[-1]], [])
        qc._append(MCU1Gate(np.pi, num_ctrl_qubits=self.num_ctrl_qubits), q[:], [])
        qc._append(HGate(), [q[-1]], [])
        self.definition = qc


class MCXRecursive(MCXGate):
    """Implement the multi-controlled X gate using recursion.

    Using a single ancilla qubit, the multi-controlled X gate is recursively split onto
    four sub-registers. This is done until we reach the 3- or 4-controlled X gate since
    for these we have a concrete implementation that do not require ancillas.
    """

    def __init__(
        self,
        num_ctrl_qubits: int,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        super().__init__(num_ctrl_qubits, label=label, ctrl_state=ctrl_state, _name="mcx_recursive")

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str = "recursion"):
        """Get the number of required ancilla qubits."""
        return MCXGate.get_num_ancilla_qubits(num_ctrl_qubits, mode)

    def inverse(self):
        """Invert this gate. The MCX is its own inverse."""
        return MCXRecursive(num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state)

    def _define(self):
        """Define the MCX gate using recursion."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        q = QuantumRegister(self.num_qubits, name="q")
        qc = QuantumCircuit(q, name=self.name)
        if self.num_qubits == 4:
            qc._append(C3XGate(), q[:], [])
            self.definition = qc
        elif self.num_qubits == 5:
            qc._append(C4XGate(), q[:], [])
            self.definition = qc
        else:
            for instr, qargs, cargs in self._recurse(q[:-1], q_ancilla=q[-1]):
                qc._append(instr, qargs, cargs)
            self.definition = qc

    def _recurse(self, q, q_ancilla=None):
        # recursion stop
        if len(q) == 4:
            return [(C3XGate(), q[:], [])]
        if len(q) == 5:
            return [(C4XGate(), q[:], [])]
        if len(q) < 4:
            raise AttributeError("Something went wrong in the recursion, have less than 4 qubits.")

        # recurse
        num_ctrl_qubits = len(q) - 1
        middle = ceil(num_ctrl_qubits / 2)
        first_half = [*q[:middle], q_ancilla]
        second_half = [*q[middle:num_ctrl_qubits], q_ancilla, q[num_ctrl_qubits]]

        rule = []
        rule += self._recurse(first_half, q_ancilla=q[middle])
        rule += self._recurse(second_half, q_ancilla=q[middle - 1])
        rule += self._recurse(first_half, q_ancilla=q[middle])
        rule += self._recurse(second_half, q_ancilla=q[middle - 1])

        return rule


class MCXVChain(MCXGate):
    """Implement the multi-controlled X gate using a V-chain of CX gates."""

    def __new__(
        cls,
        num_ctrl_qubits: Optional[int] = None,
        dirty_ancillas: bool = False,  # pylint: disable=unused-argument
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Create a new MCX instance.

        This must be defined anew to include the additional argument ``dirty_ancillas``.
        """
        return super().__new__(cls, num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def __init__(
        self,
        num_ctrl_qubits: int,
        dirty_ancillas: bool = False,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        super().__init__(num_ctrl_qubits, label=label, ctrl_state=ctrl_state, _name="mcx_vchain")
        self._dirty_ancillas = dirty_ancillas

    def inverse(self):
        """Invert this gate. The MCX is its own inverse."""
        return MCXVChain(
            num_ctrl_qubits=self.num_ctrl_qubits,
            dirty_ancillas=self._dirty_ancillas,
            ctrl_state=self.ctrl_state,
        )

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str = "v-chain"):
        """Get the number of required ancilla qubits."""
        return MCXGate.get_num_ancilla_qubits(num_ctrl_qubits, mode)

    def _define(self):
        """Define the MCX gate using a V-chain of CX gates."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        q = QuantumRegister(self.num_qubits, name="q")
        qc = QuantumCircuit(q, name=self.name)
        q_controls = q[: self.num_ctrl_qubits]
        q_target = q[self.num_ctrl_qubits]
        q_ancillas = q[self.num_ctrl_qubits + 1 :]

        definition = []

        if self._dirty_ancillas:
            i = self.num_ctrl_qubits - 3
            ancilla_pre_rule = [
                (U2Gate(0, np.pi), [q_target], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U1Gate(-np.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(np.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U1Gate(-np.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(np.pi / 4), [q_ancillas[i]], []),
            ]
            for inst in ancilla_pre_rule:
                definition.append(inst)

            for j in reversed(range(2, self.num_ctrl_qubits - 1)):
                definition.append(
                    (RCCXGate(), [q_controls[j], q_ancillas[i - 1], q_ancillas[i]], [])
                )
                i -= 1

        definition.append((RCCXGate(), [q_controls[0], q_controls[1], q_ancillas[0]], []))
        i = 0
        for j in range(2, self.num_ctrl_qubits - 1):
            definition.append((RCCXGate(), [q_controls[j], q_ancillas[i], q_ancillas[i + 1]], []))
            i += 1

        if self._dirty_ancillas:
            ancilla_post_rule = [
                (U1Gate(-np.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(np.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U1Gate(-np.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(np.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U2Gate(0, np.pi), [q_target], []),
            ]
            for inst in ancilla_post_rule:
                definition.append(inst)
        else:
            definition.append((CCXGate(), [q_controls[-1], q_ancillas[i], q_target], []))

        for j in reversed(range(2, self.num_ctrl_qubits - 1)):
            definition.append((RCCXGate(), [q_controls[j], q_ancillas[i - 1], q_ancillas[i]], []))
            i -= 1
        definition.append((RCCXGate(), [q_controls[0], q_controls[1], q_ancillas[i]], []))

        if self._dirty_ancillas:
            for i, j in enumerate(list(range(2, self.num_ctrl_qubits - 1))):
                definition.append(
                    (RCCXGate(), [q_controls[j], q_ancillas[i], q_ancillas[i + 1]], [])
                )

        for instr, qargs, cargs in definition:
            qc._append(instr, qargs, cargs)
        self.definition = qc

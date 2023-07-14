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

"""U1 Gate."""
from __future__ import annotations
from cmath import exp
import numpy as np
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int


class U1Gate(Gate):
    r"""Single-qubit rotation about the Z axis.

    This is a diagonal gate. It can be implemented virtually in hardware
    via framechanges (i.e. at zero error and duration).

    .. warning::

       This gate is deprecated. Instead, the following replacements should be used

       .. math::

           U1(\lambda) = P(\lambda)= U(0,0,\lambda)

       .. code-block:: python

          circuit = QuantumCircuit(1)
          circuit.p(lambda, 0) # or circuit.u(0, 0, lambda)




    **Circuit symbol:**

    .. parsed-literal::

             ┌───────┐
        q_0: ┤ U1(λ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        U1(\lambda) =
            \begin{pmatrix}
                1 & 0 \\
                0 & e^{i\lambda}
            \end{pmatrix}

    **Examples:**

        .. math::

            U1(\lambda = \pi) = Z

        .. math::

            U1(\lambda = \pi/2) = S

        .. math::

            U1(\lambda = \pi/4) = T

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.RZGate`:
        This gate is equivalent to RZ up to a phase factor.

            .. math::

                U1(\lambda) = e^{i{\lambda}/2} RZ(\lambda)

        :class:`~qiskit.circuit.library.standard_gates.U3Gate`:
        U3 is a generalization of U2 that covers all single-qubit rotations,
        using two X90 pulses.

        Reference for virtual Z gate implementation:
        `1612.00858 <https://arxiv.org/abs/1612.00858>`_
    """

    def __init__(self, theta: ParameterValueType, label: str | None = None):
        """Create new U1 gate."""
        super().__init__("u1", 1, [theta], label=label)

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate  # pylint: disable=cyclic-import

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(0, 0, self.params[0]), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ):
        """Return a (multi-)controlled-U1 gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CU1Gate(self.params[0], label=label, ctrl_state=ctrl_state)
        elif ctrl_state is None and num_ctrl_qubits > 1:
            gate = MCU1Gate(self.params[0], num_ctrl_qubits, label=label)
        else:
            return super().control(
                num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state
            )
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        r"""Return inverted U1 gate (:math:`U1(\lambda)^{\dagger} = U1(-\lambda)`)"""
        return U1Gate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a np.array for the U1 gate."""
        lam = float(self.params[0])
        return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=dtype)


class CU1Gate(ControlledGate):
    r"""Controlled-U1 gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the control state.

    **Circuit symbol:**

    .. parsed-literal::


        q_0: ─■──
              │λ
        q_1: ─■──


    **Matrix representation:**

    .. math::

        CU1(\lambda) =
            I \otimes |0\rangle\langle 0| + U1 \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\lambda}
            \end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CRZGate`:
        Due to the global phase difference in the matrix definitions
        of U1 and RZ, CU1 and CRZ are different gates with a relative
        phase difference.
    """

    def __init__(
        self,
        theta: ParameterValueType,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ):
        """Create new CU1 gate."""
        super().__init__(
            "cu1",
            2,
            [theta],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=U1Gate(theta),
        )

    def _define(self):
        """
        gate cu1(lambda) a,b
        { u1(lambda/2) a; cx a,b;
          u1(-lambda/2) b; cx a,b;
          u1(lambda/2) b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate  # pylint: disable=cyclic-import

        #      ┌─────────┐
        # q_0: ┤ U1(λ/2) ├──■────────────────■─────────────
        #      └─────────┘┌─┴─┐┌──────────┐┌─┴─┐┌─────────┐
        # q_1: ───────────┤ X ├┤ U1(-λ/2) ├┤ X ├┤ U1(λ/2) ├
        #                 └───┘└──────────┘└───┘└─────────┘
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U1Gate(self.params[0] / 2), [q[0]], []),
            (CXGate(), [q[0], q[1]], []),
            (U1Gate(-self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U1Gate(self.params[0] / 2), [q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
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
        if ctrl_state is None:
            gate = MCU1Gate(self.params[0], num_ctrl_qubits=num_ctrl_qubits + 1, label=label)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted CU1 gate (:math:`CU1(\lambda)^{\dagger} = CU1(-\lambda)`)"""
        return CU1Gate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        """Return a np.array for the CU1 gate."""
        eith = exp(1j * float(self.params[0]))
        if self.ctrl_state:
            return np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, eith]], dtype=dtype
            )
        else:
            return np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, eith, 0], [0, 0, 0, 1]], dtype=dtype
            )


class MCU1Gate(ControlledGate):
    r"""Multi-controlled-U1 gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the state of the control qubits.

    **Circuit symbol:**

    .. parsed-literal::

            q_0: ────■────
                     │
                     .
                     │
        q_(n-1): ────■────
                 ┌───┴───┐
            q_n: ┤ U1(λ) ├
                 └───────┘

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CU1Gate`:
        The singly-controlled-version of this gate.
    """

    def __init__(
        self,
        lam: ParameterValueType,
        num_ctrl_qubits: int,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ):
        """Create new MCU1 gate."""
        super().__init__(
            "mcu1",
            num_ctrl_qubits + 1,
            [lam],
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=U1Gate(lam),
        )

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)

        if self.num_ctrl_qubits == 0:
            definition = U1Gate(self.params[0]).definition
        if self.num_ctrl_qubits == 1:
            definition = CU1Gate(self.params[0]).definition
        else:
            from .u3 import _gray_code_chain

            scaled_lam = self.params[0] / (2 ** (self.num_ctrl_qubits - 1))
            bottom_gate = CU1Gate(scaled_lam)
            definition = _gray_code_chain(q, self.num_ctrl_qubits, bottom_gate)
        for instr, qargs, cargs in definition:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
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
        gate = MCU1Gate(
            self.params[0],
            num_ctrl_qubits=num_ctrl_qubits + self.num_ctrl_qubits,
            label=label,
            ctrl_state=new_ctrl_state,
        )
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        r"""Return inverted MCU1 gate (:math:`MCU1(\lambda)^{\dagger} = MCU1(-\lambda)`)"""
        return MCU1Gate(-self.params[0], self.num_ctrl_qubits)

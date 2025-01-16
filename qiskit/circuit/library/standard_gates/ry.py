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

"""Rotation around the Y axis."""

from __future__ import annotations

import math
from math import pi
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit._accelerate.circuit import StandardGate


class RYGate(Gate):
    r"""Single-qubit rotation about the Y axis.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ry` method.

    **Circuit symbol:**

    .. code-block:: text

             ┌───────┐
        q_0: ┤ Ry(ϴ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        RY(\theta) = \exp\left(-i \rotationangle Y\right) =
            \begin{pmatrix}
                \cos\left(\rotationangle\right) & -\sin\left(\rotationangle\right) \\
                \sin\left(\rotationangle\right) & \cos\left(\rotationangle\right)
            \end{pmatrix}
    """

    _standard_gate = StandardGate.RYGate

    def __init__(
        self,
        theta: ParameterValueType,
        label: Optional[str] = None,
    ):
        """Create new RY gate."""
        super().__init__("ry", 1, [theta], label=label)

    def _define(self):
        """
        gate ry(theta) a { r(theta, pi/2) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .r import RGate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RGate(self.params[0], pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a (multi-)controlled-RY gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate. If ``None``, this is set to ``True`` if
                the gate contains free parameters and more than one control qubit, in which
                case it cannot yet be synthesized. Otherwise it is set to ``False``.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        # deliberately capture annotated in [None, False] here
        if not annotated and num_ctrl_qubits == 1:
            gate = CRYGate(self.params[0], label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverse RY gate.

        :math:`RY(\lambda)^{\dagger} = RY(-\lambda)`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RYGate` with an inverted parameter value.

        Returns:
            RYGate: inverse gate.
        """
        return RYGate(-self.params[0])

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the RY gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        cos = math.cos(self.params[0] / 2)
        sin = math.sin(self.params[0] / 2)
        return numpy.array([[cos, -sin], [sin, cos]], dtype=dtype)

    def power(self, exponent: float, annotated: bool = False):
        (theta,) = self.params
        return RYGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, RYGate):
            return self._compare_parameters(other)
        return False


class CRYGate(ControlledGate):
    r"""Controlled-RY gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cry` method.

    **Circuit symbol:**

    .. code-block:: text

        q_0: ────■────
             ┌───┴───┐
        q_1: ┤ Ry(ϴ) ├
             └───────┘

    **Matrix representation:**

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        CRY(\theta)\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| + RY(\theta) \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0         & 0 & 0 \\
                0 & \cos\left(\rotationangle\right) & 0 & -\sin\left(\rotationangle\right) \\
                0 & 0         & 1 & 0 \\
                0 & \sin\left(\rotationangle\right) & 0 & \cos\left(\rotationangle\right)
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌───────┐
            q_0: ┤ Ry(ϴ) ├
                 └───┬───┘
            q_1: ────■────

        .. math::

            \newcommand{\rotationangle}{\frac{\theta}{2}}

            CRY(\theta)\ q_1, q_0 =
            |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes RY(\theta) =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\left(\rotationangle\right) & -\sin\left(\rotationangle\right) \\
                    0 & 0 & \sin\left(\rotationangle\right) & \cos\left(\rotationangle\right)
                \end{pmatrix}
    """

    _standard_gate = StandardGate.CRYGate

    def __init__(
        self,
        theta: ParameterValueType,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        _base_label=None,
    ):
        """Create new CRY gate."""
        super().__init__(
            "cry",
            2,
            [theta],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=RYGate(theta, label=_base_label),
        )

    def _define(self):
        """
        gate cry(lambda) a,b
        { u3(lambda/2,0,0) b; cx a,b;
          u3(-lambda/2,0,0) b; cx a,b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate

        # q_0: ─────────────■───────────────■──
        #      ┌─────────┐┌─┴─┐┌─────────┐┌─┴─┐
        # q_1: ┤ Ry(λ/2) ├┤ X ├┤ Ry(λ/2) ├┤ X ├
        #      └─────────┘└───┘└─────────┘└───┘
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (RYGate(self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RYGate(-self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Return inverse CRY gate (i.e. with the negative rotation angle)

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CRYGate` with an inverted parameter value.

        Returns:
            CRYGate: inverse gate.
        ."""
        return CRYGate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the CRY gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        half_theta = float(self.params[0]) / 2
        cos = math.cos(half_theta)
        sin = math.sin(half_theta)
        if self.ctrl_state:
            return numpy.array(
                [[1, 0, 0, 0], [0, cos, 0, -sin], [0, 0, 1, 0], [0, sin, 0, cos]], dtype=dtype
            )
        else:
            return numpy.array(
                [[cos, 0, -sin, 0], [0, 1, 0, 0], [sin, 0, cos, 0], [0, 0, 0, 1]], dtype=dtype
            )

    def __eq__(self, other):
        if isinstance(other, CRYGate):
            return self._compare_parameters(other) and self.ctrl_state == other.ctrl_state
        return False

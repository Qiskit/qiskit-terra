# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit XX-rotation gate."""

from __future__ import annotations

import math
from typing import Optional
import numpy
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType, ParameterExpression
from qiskit._accelerate.circuit import StandardGate


class RXXGate(Gate):
    r"""A parametric 2-qubit :math:`X \otimes X` interaction (rotation about XX).

    This gate is symmetric, and is maximally entangling at :math:`\theta = \pi/2`.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rxx` method.

    **Circuit Symbol:**

    .. code-block:: text

             ┌─────────┐
        q_0: ┤1        ├
             │  Rxx(ϴ) │
        q_1: ┤0        ├
             └─────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        R_{XX}(\theta) = \exp\left(-i \rotationangle X{\otimes}X\right) =
            \begin{pmatrix}
                \cos\left(\rotationangle\right) & 0 & 0 & -i\sin\left(\rotationangle\right) \\
                0 & \cos\left(\rotationangle\right) & -i\sin\left(\rotationangle\right) & 0 \\
                0 & -i\sin\left(\rotationangle\right) & \cos\left(\rotationangle\right) & 0 \\
                -i\sin\left(\rotationangle\right) & 0 & 0 & \cos\left(\rotationangle\right)
            \end{pmatrix}

    **Examples:**

        .. math::

            R_{XX}(\theta = 0) = I

        .. math::

            R_{XX}(\theta = \pi) = -i X \otimes X

        .. math::

            R_{XX}\left(\theta = \frac{\pi}{2}\right) = \frac{1}{\sqrt{2}}
                                    \begin{pmatrix}
                                        1  & 0  & 0  & -i \\
                                        0  & 1  & -i & 0 \\
                                        0  & -i & 1  & 0 \\
                                        -i & 0  & 0  & 1
                                    \end{pmatrix}
    """

    _standard_gate = StandardGate.RXXGate

    def __init__(
        self, theta: ParameterValueType, label: Optional[str] = None, *, duration=None, unit="dt"
    ):
        """Create new RXX gate."""
        super().__init__("rxx", 2, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        from .h import HGate
        from .rz import RZGate

        #      ┌───┐                   ┌───┐
        # q_0: ┤ H ├──■─────────────■──┤ H ├
        #      ├───┤┌─┴─┐┌───────┐┌─┴─┐├───┤
        # q_1: ┤ H ├┤ X ├┤ Rz(0) ├┤ X ├┤ H ├
        #      └───┘└───┘└───────┘└───┘└───┘
        theta = self.params[0]
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (HGate(), [q[0]], []),
            (HGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RZGate(theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[1]], []),
            (HGate(), [q[0]], []),
        ]
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
        """Return a (multi-)controlled-RXX gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate. If ``None``, this is set to ``True`` if
                the gate contains free parameters, in which case it cannot
                yet be synthesized.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if annotated is None:
            annotated = any(isinstance(p, ParameterExpression) for p in self.params)

        gate = super().control(
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            annotated=annotated,
        )
        return gate

    def inverse(self, annotated: bool = False):
        """Return inverse RXX gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RXXGate` with an inverted parameter value.

        Returns:
            RXXGate: inverse gate.
        """
        return RXXGate(-self.params[0])

    def __array__(self, dtype=None, copy=None):
        """Return a Numpy.array for the RXX gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta2 = float(self.params[0]) / 2
        cos = math.cos(theta2)
        isin = 1j * math.sin(theta2)
        return numpy.array(
            [[cos, 0, 0, -isin], [0, cos, -isin, 0], [0, -isin, cos, 0], [-isin, 0, 0, cos]],
            dtype=dtype,
        )

    def power(self, exponent: float, annotated: bool = False):
        (theta,) = self.params
        return RXXGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, RXXGate):
            return self._compare_parameters(other)
        return False

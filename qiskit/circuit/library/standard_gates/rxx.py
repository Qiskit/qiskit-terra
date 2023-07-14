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
import math
from typing import Optional
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType


class RXXGate(Gate):
    r"""A parametric 2-qubit :math:`X \otimes X` interaction (rotation about XX).

    This gate is symmetric, and is maximally entangling at :math:`\theta = \pi/2`.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rxx` method.

    **Circuit Symbol:**

    .. parsed-literal::

             ┌─────────┐
        q_0: ┤1        ├
             │  Rxx(ϴ) │
        q_1: ┤0        ├
             └─────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{XX}(\theta) = \exp\left(-i \th X{\otimes}X\right) =
            \begin{pmatrix}
                \cos\left(\th\right)   & 0           & 0           & -i\sin\left(\th\right) \\
                0           & \cos\left(\th\right)   & -i\sin\left(\th\right) & 0 \\
                0           & -i\sin\left(\th\right) & \cos\left(\th\right)   & 0 \\
                -i\sin\left(\th\right) & 0           & 0           & \cos\left(\th\right)
            \end{pmatrix}

    **Examples:**

        .. math::

            R_{XX}(\theta = 0) = I

        .. math::

            R_{XX}(\theta = \pi) = i X \otimes X

        .. math::

            R_{XX}\left(\theta = \frac{\pi}{2}\right) = \frac{1}{\sqrt{2}}
                                    \begin{pmatrix}
                                        1  & 0  & 0  & -i \\
                                        0  & 1  & -i & 0 \\
                                        0  & -i & 1  & 0 \\
                                        -i & 0  & 0  & 1
                                    \end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str] = None):
        """Create new RXX gate."""
        super().__init__("rxx", 2, [theta], label=label)

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

    def inverse(self):
        """Return inverse RXX gate (i.e. with the negative rotation angle)."""
        return RXXGate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a np.array for the RXX gate."""
        theta2 = float(self.params[0]) / 2
        cos = math.cos(theta2)
        isin = 1j * math.sin(theta2)
        return np.array(
            [[cos, 0, 0, -isin], [0, cos, -isin, 0], [0, -isin, cos, 0], [-isin, 0, 0, cos]],
            dtype=dtype,
        )

    def power(self, exponent: float):
        """Raise gate to a power."""
        (theta,) = self.params
        return RXXGate(exponent * theta)

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit XX+YY gate."""
import math
from cmath import exp
from math import pi
from typing import Optional
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType


class XXPlusYYGate(Gate):
    r"""XX+YY interaction gate.

    A 2-qubit parameterized XX+YY interaction, also known as an XY gate. Its action is to induce
    a coherent rotation by some angle between :math:`|01\rangle` and :math:`|10\rangle`.

    **Circuit Symbol:**

    .. parsed-literal::

             ┌───────────────┐
        q_0: ┤0              ├
             │  (XX+YY)(θ,β) │
        q_1: ┤1              ├
             └───────────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{XX+YY}(\theta, \beta)\ q_0, q_1 =
          RZ_0(-\beta) \cdot \exp\left(-i \frac{\theta}{2} \frac{XX+YY}{2}\right) \cdot RZ_0(\beta) =
            \begin{pmatrix}
                1 & 0                     & 0                    & 0  \\
                0 & \cos\left(\th\right)             & -i\sin\left(\th\right)e^{-i\beta} & 0  \\
                0 & -i\sin\left(\th\right)e^{i\beta} & \cos\left(\th\right)            & 0  \\
                0 & 0                     & 0                    & 1
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In the above example we apply the gate
        on (q_0, q_1) which results in adding the (optional) phase defined
        by :math:`beta` on q_0. Instead, if we apply it on (q_1, q_0), the
        phase is added on q_1. If :math:`beta` is set to its default value
        of :math:`0`, the gate is equivalent in big and little endian.

        .. parsed-literal::

                 ┌───────────────┐
            q_0: ┤1              ├
                 │  (XX+YY)(θ,β) │
            q_1: ┤0              ├
                 └───────────────┘

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{XX+YY}(\theta, \beta)\ q_0, q_1 =
          RZ_1(-\beta) \cdot \exp\left(-i \frac{\theta}{2} \frac{XX+YY}{2}\right) \cdot RZ_1(\beta) =
            \begin{pmatrix}
                1 & 0                     & 0                    & 0  \\
                0 & \cos\left(\th\right)             & -i\sin\left(\th\right)e^{i\beta} & 0  \\
                0 & -i\sin\left(\th\right)e^{-i\beta} & \cos\left(\th\right)            & 0  \\
                0 & 0                     & 0                    & 1
            \end{pmatrix}
    """

    def __init__(
        self,
        theta: ParameterValueType,
        beta: ParameterValueType = 0,
        label: Optional[str] = "(XX+YY)",
    ):
        """Create new XX+YY gate.

        Args:
            theta: The rotation angle.
            beta: The phase angle.
            label: The label of the gate.
        """
        super().__init__("xx_plus_yy", 2, [theta, beta], label=label)

    def _define(self):
        """
        gate xx_plus_yy(theta, beta) a, b {
            rz(beta) b;
            rz(-pi/2) a;
            sx a;
            rz(pi/2) a;
            s b;
            cx a, b;
            ry(theta/2) a;
            ry(theta/2) b;
            cx a, b;
            sdg b;
            rz(-pi/2) a;
            sxdg a;
            rz(pi/2) a;
            rz(-beta) b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        from .s import SGate, SdgGate
        from .sx import SXGate, SXdgGate
        from .rz import RZGate
        from .ry import RYGate

        theta = self.params[0]
        beta = self.params[1]
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (RZGate(beta), [q[0]], []),
            (RZGate(-pi / 2), [q[1]], []),
            (SXGate(), [q[1]], []),
            (RZGate(pi / 2), [q[1]], []),
            (SGate(), [q[0]], []),
            (CXGate(), [q[1], q[0]], []),
            (RYGate(-theta / 2), [q[1]], []),
            (RYGate(-theta / 2), [q[0]], []),
            (CXGate(), [q[1], q[0]], []),
            (SdgGate(), [q[0]], []),
            (RZGate(-pi / 2), [q[1]], []),
            (SXdgGate(), [q[1]], []),
            (RZGate(pi / 2), [q[1]], []),
            (RZGate(-beta), [q[0]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse XX+YY gate (i.e. with the negative rotation angle and same phase angle)."""
        return XXPlusYYGate(-self.params[0], self.params[1])

    def __array__(self, dtype=complex):
        """Return a np.array for the XX+YY gate."""
        import numpy as np

        half_theta = float(self.params[0]) / 2
        beta = float(self.params[1])
        cos = math.cos(half_theta)
        sin = math.sin(half_theta)
        return np.array(
            [
                [1, 0, 0, 0],
                [0, cos, -1j * sin * exp(-1j * beta), 0],
                [0, -1j * sin * exp(1j * beta), cos, 0],
                [0, 0, 0, 1],
            ],
            dtype=dtype,
        )

    def power(self, exponent: float):
        """Raise gate to a power."""
        theta, beta = self.params
        return XXPlusYYGate(exponent * theta, beta)

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rotation around an axis in x-y plane."""

import math
from cmath import exp
from math import pi
from typing import Optional
import numpy
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit._accelerate.circuit import StandardGate


class RGate(Gate):
    r"""Rotation θ around the cos(φ)x + sin(φ)y axis.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.r` method.

    **Circuit symbol:**

    .. code-block:: text

             ┌──────┐
        q_0: ┤ R(ϴ) ├
             └──────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        R(\theta, \phi) = e^{-i \rotationangle \left(\cos{\phi} x + \sin{\phi} y\right)} =
            \begin{pmatrix}
                \cos\left(\rotationangle\right) & -i e^{-i \phi} \sin\left(\rotationangle\right) \\
                -i e^{i \phi} \sin\left(\rotationangle\right) & \cos\left(\rotationangle\right)
            \end{pmatrix}
    """

    _standard_gate = StandardGate.RGate

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        label: Optional[str] = None,
        *,
        duration=None,
        unit="dt",
    ):
        """Create new r single-qubit gate."""
        super().__init__("r", 1, [theta, phi], label=label, duration=duration, unit=unit)

    def _define(self):
        """
        gate r(θ, φ) a {u3(θ, φ - π/2, -φ + π/2) a;}
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        theta = self.params[0]
        phi = self.params[1]
        rules = [(U3Gate(theta, phi - pi / 2, -phi + pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Invert this gate as: :math:`r(θ, φ)^dagger = r(-θ, φ)`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RGate` with an inverted parameter value.

        Returns:
            RGate: inverse gate.
        """
        return RGate(-self.params[0], self.params[1])

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the R gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta, phi = float(self.params[0]), float(self.params[1])
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        exp_m = exp(-1j * phi)
        exp_p = exp(1j * phi)
        return numpy.array([[cos, -1j * exp_m * sin], [-1j * exp_p * sin, cos]], dtype=dtype)

    def power(self, exponent: float, annotated: bool = False):
        theta, phi = self.params
        return RGate(exponent * theta, phi)

    def __eq__(self, other):
        if isinstance(other, RGate):
            return self._compare_parameters(other)
        return False

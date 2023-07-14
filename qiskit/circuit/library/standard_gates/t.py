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

"""T and Tdg gate."""
import math
from math import pi
from typing import Optional

import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.quantumregister import QuantumRegister


class TGate(Gate):
    r"""Single qubit T gate (Z**0.25).

    It induces a :math:`\pi/4` phase, and is sometimes called the pi/8 gate
    (because of how the RZ(\pi/4) matrix looks like).

    This is a non-Clifford gate and a fourth-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.t` method.

    **Matrix Representation:**

    .. math::

        T = \begin{pmatrix}
                1 & 0 \\
                0 & e^{i\pi/4}
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ T ├
             └───┘

    Equivalent to a :math:`\pi/4` radian rotation about the Z axis.
    """

    def __init__(self, label: Optional[str] = None):
        """Create new T gate."""
        super().__init__("t", 1, [], label=label)

    def _define(self):
        """
        gate t a { u1(pi/4) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        from .u1 import U1Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(pi / 4), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse T gate (i.e. Tdg)."""
        return TdgGate()

    def __array__(self, dtype=None):
        """Return a np.array for the T gate."""
        return np.array([[1, 0], [0, (1 + 1j) / np.sqrt(2)]], dtype=dtype)

    def power(self, exponent: float):
        """Raise gate to a power."""
        return PhaseGate(0.25 * np.pi * exponent)


class TdgGate(Gate):
    r"""Single qubit T-adjoint gate (~Z**0.25).

    It induces a :math:`-\pi/4` phase.

    This is a non-Clifford gate and a fourth-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.tdg` method.

    **Matrix Representation:**

    .. math::

        Tdg = \begin{pmatrix}
                1 & 0 \\
                0 & e^{-i\pi/4}
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────┐
        q_0: ┤ Tdg ├
             └─────┘

    Equivalent to a :math:`-\pi/4` radian rotation about the Z axis.
    """

    def __init__(self, label: Optional[str] = None):
        """Create new Tdg gate."""
        super().__init__("tdg", 1, [], label=label)

    def _define(self):
        """
        gate tdg a { u1(pi/4) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        from .u1 import U1Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(-pi / 4), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse Tdg gate (i.e. T)."""
        return TGate()

    def __array__(self, dtype=None):
        """Return a np.array for the inverse T gate."""
        return np.array([[1, 0], [0, (1 - 1j) / math.sqrt(2)]], dtype=dtype)

    def power(self, exponent: float):
        """Raise gate to a power."""
        return PhaseGate(-0.25 * np.pi * exponent)

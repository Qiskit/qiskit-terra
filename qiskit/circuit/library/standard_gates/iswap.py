# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""iSWAP gate."""

from typing import Optional

import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister

from .xx_plus_yy import XXPlusYYGate


class iSwapGate(Gate):
    r"""iSWAP gate.

    A 2-qubit XX+YY interaction.
    This is a Clifford and symmetric gate. Its action is to swap two qubit
    states and phase the :math:`|01\rangle` and :math:`|10\rangle`
    amplitudes by i.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.iswap` method.

    **Circuit Symbol:**

    .. parsed-literal::

        q_0: ─⊗─
              │
        q_1: ─⊗─

    **Reference Implementation:**

    .. parsed-literal::

             ┌───┐┌───┐     ┌───┐
        q_0: ┤ S ├┤ H ├──■──┤ X ├─────
             ├───┤└───┘┌─┴─┐└─┬─┘┌───┐
        q_1: ┤ S ├─────┤ X ├──■──┤ H ├
             └───┘     └───┘     └───┘

    **Matrix Representation:**

    .. math::

        iSWAP = R_{XX+YY}\left(-\frac{\pi}{2}\right)
          = \exp\left(i \frac{\pi}{4} \left(X{\otimes}X+Y{\otimes}Y\right)\right) =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & i & 0 \\
                0 & i & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    This gate is equivalent to a SWAP up to a diagonal.

    .. math::

         iSWAP =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}
         .  \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & i & 0 & 0 \\
                0 & 0 & i & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}
    """

    def __init__(self, label: Optional[str] = None):
        """Create new iSwap gate."""
        super().__init__("iswap", 2, [], label=label)

    def _define(self):
        """
        gate iswap a,b {
            s q[0];
            s q[1];
            h q[0];
            cx q[0],q[1];
            cx q[1],q[0];
            h q[1];
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        from .h import HGate
        from .s import SGate
        from .x import CXGate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (SGate(), [q[0]], []),
            (SGate(), [q[1]], []),
            (HGate(), [q[0]], []),
            (CXGate(), [q[0], q[1]], []),
            (CXGate(), [q[1], q[0]], []),
            (HGate(), [q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def __array__(self, dtype=None):
        """Return a numpy.array for the iSWAP gate."""
        return np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=dtype)

    def power(self, exponent: float):
        """Raise gate to a power."""
        return XXPlusYYGate(-np.pi * exponent)

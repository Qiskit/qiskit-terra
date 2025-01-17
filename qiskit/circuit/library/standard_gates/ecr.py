# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit ZX-rotation gate."""
from math import sqrt
import numpy as np

from qiskit.circuit._utils import with_gate_array
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit._accelerate.circuit import StandardGate
from .rzx import RZXGate
from .x import XGate


@with_gate_array(
    sqrt(0.5) * np.array([[0, 1, 0, 1.0j], [1, 0, -1.0j, 0], [0, 1.0j, 0, 1], [-1.0j, 0, 1, 0]])
)
class ECRGate(SingletonGate):
    r"""An echoed cross-resonance gate.

    This gate is maximally entangling and is equivalent to a CNOT up to
    single-qubit pre-rotations. The echoing procedure mitigates some
    unwanted terms (terms other than ZX) to cancel in an experiment.
    More specifically, this gate implements :math:`\frac{1}{\sqrt{2}}(IX-XY)`.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ecr` method.

    **Circuit Symbol:**

    .. code-block:: text

             ┌─────────┐            ┌────────────┐┌────────┐┌─────────────┐
        q_0: ┤0        ├       q_0: ┤0           ├┤ RX(pi) ├┤0            ├
             │   ECR   │   =        │  RZX(pi/4) │└────────┘│  RZX(-pi/4) │
        q_1: ┤1        ├       q_1: ┤1           ├──────────┤1            ├
             └─────────┘            └────────────┘          └─────────────┘

    **Matrix Representation:**

    .. math::

        ECR\ q_0, q_1 = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                0   & 1   &  0  & i \\
                1   & 0   &  -i & 0 \\
                0   & i   &  0  & 1 \\
                -i  & 0   &  1  & 0
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In the above example we apply the gate
        on (q_0, q_1) which results in the :math:`X \otimes Z` tensor order.
        Instead, if we apply it on (q_1, q_0), the matrix will
        be :math:`Z \otimes X`:

        .. code-block:: text

                 ┌─────────┐
            q_0: ┤1        ├
                 │   ECR   │
            q_1: ┤0        ├
                 └─────────┘

        .. math::

            ECR\ q_0, q_1 = \frac{1}{\sqrt{2}}
                \begin{pmatrix}
                    0   & 0   &  1  & i \\
                    0   & 0   &  i  & 1 \\
                    1   & -i  &  0  & 0 \\
                    -i  & 1   &  0  & 0
                \end{pmatrix}
    """

    _standard_gate = StandardGate.ECRGate

    def __init__(self, label=None, *, duration=None, unit="dt"):
        """Create new ECR gate."""
        super().__init__("ecr", 2, [], label=label, duration=duration, unit=unit)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate ecr a, b { rzx(pi/4) a, b; x a; rzx(-pi/4) a, b;}
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (RZXGate(np.pi / 4), [q[0], q[1]], []),
            (XGate(), [q[0]], []),
            (RZXGate(-np.pi / 4), [q[0], q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Return inverse ECR gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            ECRGate: inverse gate (self-inverse).
        """
        return ECRGate()  # self-inverse

    def __eq__(self, other):
        return isinstance(other, ECRGate)

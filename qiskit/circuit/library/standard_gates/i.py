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

"""Identity gate."""

from typing import Optional
from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit.circuit._utils import with_gate_array
from qiskit._accelerate.circuit import StandardGate


@with_gate_array([[1, 0], [0, 1]])
class IGate(SingletonGate):
    r"""Identity gate.

    Identity gate corresponds to a single-qubit gate wait cycle,
    and should not be optimized or unrolled (it is an opaque gate).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.i` and
    :meth:`~qiskit.circuit.QuantumCircuit.id` methods.

    **Matrix Representation:**

    .. math::

        I = \begin{pmatrix}
                1 & 0 \\
                0 & 1
            \end{pmatrix}

    **Circuit symbol:**

    .. code-block:: text

             ┌───┐
        q_0: ┤ I ├
             └───┘
    """

    _standard_gate = StandardGate.IGate

    def __init__(self, label: Optional[str] = None):
        """Create new Identity gate."""
        super().__init__("id", 1, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def inverse(self, annotated: bool = False):
        """Returne the inverse gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            IGate: inverse gate (self-inverse).
        ."""
        return IGate()  # self-inverse

    def power(self, exponent: float, annotated: bool = False):
        return IGate()

    def __eq__(self, other):
        return isinstance(other, IGate)

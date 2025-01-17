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

"""Barrier instruction.

Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
with the :meth:`~qiskit.circuit.QuantumCircuit.barrier` method.
"""

from __future__ import annotations

from qiskit.exceptions import QiskitError
from qiskit.utils import deprecate_func
from .instruction import Instruction


class Barrier(Instruction):
    """A directive for circuit compilation to separate pieces of a circuit so that any optimizations
    or re-writes are constrained to only act between barriers.

    This will also appear in visualizations as a visual marker.
    """

    _directive = True

    def __init__(self, num_qubits: int, label: str | None = None):
        """
        Args:
            num_qubits: the number of qubits for the barrier.
            label: the optional label of this barrier.
        """
        self._label = label
        super().__init__("barrier", num_qubits, 0, [], label=label)

    def inverse(self, annotated: bool = False):
        """Special case. Return self."""
        return Barrier(self.num_qubits)

    @deprecate_func(since="1.3.0", removal_timeline="in 2.0.0")
    def c_if(self, classical, val):
        raise QiskitError("Barriers are compiler directives and cannot be conditional.")

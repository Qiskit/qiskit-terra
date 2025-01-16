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

"""
Quantum measurement in the computational basis.
"""

from qiskit.circuit.singleton import SingletonInstruction, stdlib_singleton_key
from qiskit.circuit.exceptions import CircuitError


class Measure(SingletonInstruction):
    """Quantum measurement in the computational basis."""

    def __init__(self, label=None, *, duration=None, unit="dt"):
        """
        Args:
            label: optional string label for this instruction.
        """
        super().__init__("measure", 1, 1, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def broadcast_arguments(self, qargs, cargs):
        qarg = qargs[0]
        carg = cargs[0]

        if len(carg) == len(qarg):
            for qarg, carg in zip(qarg, carg):
                yield [qarg], [carg]
        elif len(qarg) == 1 and carg:
            for each_carg in carg:
                yield qarg, [each_carg]
        else:
            raise CircuitError("register size error")

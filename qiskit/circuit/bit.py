# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum bit and Classical bit objects.
"""


class Bit:
    """Implement a generic bit.

    .. note::
        This class should not be instantiated directly. This is just a superclass
        for :class:`~.Clbit` and :class:`~.circuit.Qubit`.
    """

    def __repr__(self):
        """Return the official string representing the bit."""
        return object.__repr__(self)

    def __copy__(self):
        # Bits are immutable.
        return self

    def __deepcopy__(self, memo=None):
        # Bits are immutable.
        return self

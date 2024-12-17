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

# pylint: disable=bad-docstring-quotes

"""
Quantum register reference object.
"""
import itertools

# Over-specific import to avoid cyclic imports.
from .register import Register
from .bit import Bit


class Qubit(Bit):
    """Implement a quantum bit."""


class QuantumRegister(Register):
    """Implement a quantum register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = "q"
    bit_type = Qubit


class AncillaQubit(Qubit):
    """A qubit used as ancillary qubit."""

    __slots__ = ()

    pass


class AncillaRegister(QuantumRegister):
    """Implement an ancilla register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = "a"
    bit_type = AncillaQubit

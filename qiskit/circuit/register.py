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

# pylint: disable=not-callable

"""
Base register reference object.
"""

from __future__ import annotations
import itertools
import numpy as np

from qiskit.circuit.exceptions import CircuitError


class Register:
    """Implement a generic register.

    .. note::
        This class should not be instantiated directly. This is just a superclass
        for :class:`~.ClassicalRegister` and :class:`~.QuantumRegister`.

    """

    __slots__ = ["_name", "_size", "_bits", "_bit_indices", "_hash", "_repr"]

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = "reg"
    bit_type = None

    def __init__(self, size: int | None = None, name: str | None = None, bits=None):
        """Create a new generic register.

        Either the ``size`` or the ``bits`` argument must be provided. If
        ``size`` is not None, the register will be pre-populated with bits of the
        correct type.

        Args:
            size (int): Optional. The number of bits to include in the register.
            name (str): Optional. The name of the register. If not provided, a
               unique name will be auto-generated from the register type.
            bits (list[Bit]): Optional. A list of Bit() instances to be used to
               populate the register.

        Raises:
            CircuitError: if both the ``size`` and ``bits`` arguments are
                provided, or if neither are.
            CircuitError: if ``size`` is not valid.
            CircuitError: if ``name`` is not a valid name according to the
                OpenQASM spec.
            CircuitError: if ``bits`` contained duplicated bits.
            CircuitError: if ``bits`` contained bits of an incorrect type.
        """

        if (size, bits) == (None, None) or (size is not None and bits is not None):
            raise CircuitError(
                "Exactly one of the size or bits arguments can be "
                f"provided. Provided size={size} bits={bits}."
            )

        # validate (or cast) size
        if bits is not None:
            size = len(bits)

        try:
            valid_size = size == int(size)
        except (ValueError, TypeError):
            valid_size = False

        if not valid_size:
            raise CircuitError(
                f"Register size must be an integer. ({type(size).__name__} '{size}' was provided)"
            )
        size = int(size)  # cast to int

        if size < 0:
            raise CircuitError(
                f"Register size must be non-negative ({type(size).__name__} '{size}' was provided)"
            )

        # validate (or cast) name
        if name is None:
            name = f"{self.prefix}{next(self.instances_counter)}"
        else:
            try:
                name = str(name)
            except Exception as ex:
                raise CircuitError(
                    "The circuit name should be castable to a string "
                    "(or None for autogenerate a name)."
                ) from ex

        self._name = name
        self._size = size

        self._hash = hash((type(self), self._name, self._size))
        self._repr = f"{self.__class__.__qualname__}({self.size}, '{self.name}')"
        if bits is not None:
            # check duplicated bits
            if self._size != len(set(bits)):
                raise CircuitError(f"Register bits must not be duplicated. bits={bits}")
            # pylint: disable=isinstance-second-argument-not-valid-type
            if any(not isinstance(bit, self.bit_type) for bit in bits):
                raise CircuitError(f"Provided bits did not all match register type. bits={bits}")
            self._bits = list(bits)
            self._bit_indices = {bit: idx for idx, bit in enumerate(self._bits)}
        else:
            self._bits = [self.bit_type() for idx in range(size)]

            # Since the hash of Bits created by the line above will depend upon
            # the hash of self, which is not guaranteed to have been initialized
            # first on deep-copying or on pickling, so defer populating _bit_indices
            # until first access.
            self._bit_indices = None

    @property
    def name(self):
        """Get the register name."""
        return self._name

    @property
    def size(self):
        """Get the register size."""
        return self._size

    def __repr__(self):
        """Return the official string representing the register."""
        return self._repr

    def __len__(self):
        """Return register size."""
        return self._size

    def __getitem__(self, key):
        """
        Arg:
            bit_type (Qubit or Clbit): a constructor type return element/s.
            key (int or slice or list): index of the bit to be retrieved.

        Returns:
            Qubit or Clbit or list(Qubit) or list(Clbit): a Qubit or Clbit instance if
            key is int. If key is a slice, returns a list of these instances.

        Raises:
            CircuitError: if the `key` is not an integer or not in the range `(0, self.size)`.
        """
        if not isinstance(key, (int, np.integer, slice, list)):
            raise CircuitError("expected integer or slice index into register")
        if isinstance(key, slice):
            return self._bits[key]
        elif isinstance(key, list):  # list of qubit indices
            if max(key) < len(self):
                return [self._bits[idx] for idx in key]
            else:
                raise CircuitError("register index out of range")
        else:
            return self._bits[key]

    def __iter__(self):
        for idx in range(self._size):
            yield self._bits[idx]

    def __contains__(self, bit):
        if self._bit_indices is None:
            self._bit_indices = {bit: idx for idx, bit in enumerate(self._bits)}

        return bit in self._bit_indices

    def index(self, bit):
        """Find the index of the provided bit within this register."""
        if self._bit_indices is None:
            self._bit_indices = {bit: idx for idx, bit in enumerate(self._bits)}

        try:
            return self._bit_indices[bit]
        except KeyError as err:
            raise ValueError(f"Bit {bit} not found in Register {self}.") from err

    def __eq__(self, other):
        """Two Registers are the same if they are of the same type
        (i.e. quantum/classical), and have the same name and size. Additionally,
        if either Register contains new-style bits, the bits in both registers
        will be checked for pairwise equality. If two registers are equal,
        they will have behave identically when specified as circuit args.

        Args:
            other (Register): other Register

        Returns:
            bool: `self` and `other` are equal.
        """
        if self is other:
            return True

        res = False
        if type(self) is type(other) and self._repr == other._repr:
            res = True
        return res

    def __hash__(self):
        """Make object hashable, based on the name and size to hash."""
        return self._hash

    def __getstate__(self):
        # Specifically exclude _bit_indices from pickled state as bit hashes
        # can in general depend on the hash of their containing register,
        # which may not have yet been initialized.
        return self._name, self._size, self._hash, self._repr, self._bits

    def __setstate__(self, state):
        self._name, self._size, self._hash, self._repr, self._bits = state
        self._bit_indices = None

# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Serialize objects such as quantum circuit while keeping the `Bit` references consistent between
(de)serializations."""

from io import BytesIO

import dill

from qiskit.circuit import Bit


class QiskitPickler(dill.Pickler):
    """Custom serialization class that cache `Bit` references."""

    def __init__(self, file=None, bit_cache=None):
        """
        Arg:
            file (file): pickle an object to `file`
            bit_cache (dict): a dictionary to be populated with mappings of `id` to instances of `Bit`
        """
        super().__init__(file)
        if bit_cache is None:
            self.bit_cache = {}
        else:
            self.bit_cache = bit_cache

    def get_qubit_cache(self):
        """Get the `Bit` cache, i.e. a dictionary mapping from a `Bit` id to its `Bit` instance."""
        return self.bit_cache

    def persistent_id(self, obj):
        """
        Arg:
            obj (Bit): a `Bit` instance to be serialized

        Returns:
            The persistent id used during serialization
        """
        if isinstance(obj, Bit):
            self.bit_cache[id(obj)] = obj
            return ("QBit", id(obj))
        else:
            return None


class QiskitUnpickler(dill.Unpickler):
    """Custom deserialization class that with consistent `Bit` references."""

    def __init__(self, file=None, bit_cache=None):
        """
        Arg:
            file (file): pickle an object to `file`
            bit_cache (dict): a dictionary with mappings of `id` to instances of `Bit`
        """
        super().__init__(file)
        if bit_cache is None:
            self.bit_cache = {}
        else:
            self.bit_cache = bit_cache

    def get_qubit_cache(self):
        """Get the `Bit` cache, i.e. a dictionary mapping from a `Bit` id to its `Bit` instance."""
        return self.bit_cache

    def persistent_load(self, pid):
        """
        Arg:
            pid (tuple): the persistent id of a `Bit` specified by persistend_id

        Returns:
            A deserialized `Bit`

        Raises:
            dill.UnpicklingError: if the object to be deserialized is not supported.
        """
        type_tag, qubit_id = pid
        if type_tag == "QBit":
            return self.bit_cache[qubit_id]
        else:
            raise dill.UnpicklingError("unsupported persistent object")


def dumps(obj, bit_cache):
    """
    Arg:
        file (file): pickle an object to `file`
        bit_cache (dict): a dictionary to be populated with mappings of `id` to instances of `Bit`

    Returns:
        The serialized object.
    """
    file = BytesIO()
    QiskitPickler(file, bit_cache).dump(obj)
    return file.getvalue()


def loads(obj_bin, bit_cache):
    """
    Arg:
        obj_bin (bytes): the bytes of a serialized object
        bit_cache (dict): a dictionary with mappings of `id` to instances of `Bit`

    Returns:
        The deserialized object.
    """
    file = BytesIO(obj_bin)
    return QiskitUnpickler(file, bit_cache).load()

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
.. _pulse-channels:

=======================================
Channels (:mod:`qiskit.pulse.channels`)
=======================================

Pulse is meant to be agnostic to the underlying hardware implementation, while still allowing
low-level control. Therefore, our signal channels are  *virtual* hardware channels. The backend
which executes our programs is responsible for mapping these virtual channels to the proper
physical channel within the quantum control hardware.

Channels are characterized by their type and their index.  Channels include:

* transmit channels, which should subclass ``PulseChannel``
* receive channels, such as :class:`AcquireChannel`
* non-signal "channels" such as :class:`SnapshotChannel`, :class:`MemorySlot` and
  :class:`RegisterChannel`.

Novel channel types can often utilize the :class:`ControlChannel`, but if this is not sufficient,
new channel types can be created. Then, they must be supported in the PulseQobj schema and the
assembler.  Channels are characterized by their type and their index. See each channel type below to
learn more.

.. autosummary::
   :toctree: ../stubs/

   DriveChannel
   MeasureChannel
   AcquireChannel
   ControlChannel
   RegisterSlot
   MemorySlot
   SnapshotChannel

All channels are children of the same abstract base class:

.. autoclass:: Channel
"""
from __future__ import annotations
from abc import ABCMeta
from typing import Any

import numpy as np

from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
from qiskit.utils.deprecate_pulse import deprecate_pulse_func


class Channel(metaclass=ABCMeta):
    """Base class of channels. Channels provide a Qiskit-side label for typical quantum control
    hardware signal channels. The final label -> physical channel mapping is the responsibility
    of the hardware backend. For instance, ``DriveChannel(0)`` holds instructions which the backend
    should map to the signal line driving gate operations on the qubit labeled (indexed) 0.

    When serialized channels are identified by their serialized name ``<prefix><index>``.
    The type of the channel is interpreted from the prefix,
    and the index often (but not always) maps to the qubit index.
    All concrete channel classes must have a ``prefix`` class attribute
    (and instances of that class have an index attribute). Base classes which have
    ``prefix`` set to ``None`` are prevented from being instantiated.

    To implement a new channel inherit from :class:`Channel` and provide a unique string identifier
    for the ``prefix`` class attribute.
    """

    prefix: str | None = None
    """A shorthand string prefix for characterizing the channel type."""

    # pylint: disable=unused-argument
    def __new__(cls, *args, **kwargs):
        if cls.prefix is None:
            raise NotImplementedError(
                "Cannot instantiate abstract channel. "
                "See Channel documentation for more information."
            )

        return super().__new__(cls)

    @deprecate_pulse_func
    def __init__(self, index: int):
        """Channel class.

        Args:
            index: Index of channel.
        """
        self._validate_index(index)
        self._index = index

    @property
    def index(self) -> int | ParameterExpression:
        """Return the index of this channel. The index is a label for a control signal line
        typically mapped trivially to a qubit index. For instance, ``DriveChannel(0)`` labels
        the signal line driving the qubit labeled with index 0.
        """
        return self._index

    def _validate_index(self, index: Any) -> None:
        """Raise a PulseError if the channel index is invalid, namely, if it's not a positive
        integer.

        Raises:
            PulseError: If ``index`` is not a nonnegative integer.
        """
        if isinstance(index, ParameterExpression) and index.parameters:
            # Parameters are unbound
            return
        elif isinstance(index, ParameterExpression):
            index = float(index)
            if index.is_integer():
                index = int(index)

        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError("Channel index must be a nonnegative integer")

    @property
    def parameters(self) -> set[Parameter]:
        """Parameters which determine the channel index."""
        if isinstance(self.index, ParameterExpression):
            return self.index.parameters
        return set()

    def is_parameterized(self) -> bool:
        """Return True iff the channel is parameterized."""
        return isinstance(self.index, ParameterExpression)

    @property
    def name(self) -> str:
        """Return the shorthand alias for this channel, which is based on its type and index."""
        return f"{self.__class__.prefix}{self._index}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self._index})"

    def __eq__(self, other: object) -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type
        and the same index.

        Args:
            other: The channel to compare to this channel.

        Returns:
            True iff equal.
        """
        if not isinstance(other, Channel):
            return NotImplemented
        return type(self) is type(other) and self._index == other._index

    def __hash__(self):
        return hash((type(self), self._index))


class PulseChannel(Channel, metaclass=ABCMeta):
    """Base class of transmit Channels. Pulses can be played on these channels."""

    pass


class ClassicalIOChannel(Channel, metaclass=ABCMeta):
    """Base class of classical IO channels. These cannot have instructions scheduled on them."""

    pass


class DriveChannel(PulseChannel):
    """Drive channels transmit signals to qubits which enact gate operations."""

    prefix = "d"


class MeasureChannel(PulseChannel):
    """Measure channels transmit measurement stimulus pulses for readout."""

    prefix = "m"


class ControlChannel(PulseChannel):
    """Control channels provide supplementary control over the qubit to the drive channel.
    These are often associated with multi-qubit gate operations. They may not map trivially
    to a particular qubit index.
    """

    prefix = "u"


class AcquireChannel(Channel):
    """Acquire channels are used to collect data."""

    prefix = "a"


class SnapshotChannel(ClassicalIOChannel):
    """Snapshot channels are used to specify instructions for simulators."""

    prefix = "s"

    def __init__(self):
        """Create new snapshot channel."""
        super().__init__(0)


class MemorySlot(ClassicalIOChannel):
    """Memory slot channels represent classical memory storage."""

    prefix = "m"


class RegisterSlot(ClassicalIOChannel):
    """Classical resister slot channels represent classical registers (low-latency classical
    memory).
    """

    prefix = "c"

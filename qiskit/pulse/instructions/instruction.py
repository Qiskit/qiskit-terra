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
``Instruction`` s are single operations within a :py:class:`~qiskit.pulse.Schedule`, and can be
used the same way as :py:class:`~qiskit.pulse.Schedule` s.

For example::

    duration = 10
    channel = DriveChannel(0)
    sched = Schedule()
    sched += Delay(duration, channel)  # Delay is a specific subclass of Instruction
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable

from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError
from qiskit.utils.deprecate_pulse import deprecate_pulse_func


# pylint: disable=bad-docstring-quotes


class Instruction(ABC):
    """The smallest schedulable unit: a single instruction. It has a fixed duration and specified
    channels.
    """

    @deprecate_pulse_func
    def __init__(
        self,
        operands: tuple,
        name: str | None = None,
    ):
        """Instruction initializer.

        Args:
            operands: The argument list.
            name: Optional display name for this instruction.
        """
        self._operands = operands
        self._name = name
        self._validate()

    def _validate(self):
        """Called after initialization to validate instruction data.

        Raises:
            PulseError: If the input ``channels`` are not all of type :class:`Channel`.
        """
        for channel in self.channels:
            if not isinstance(channel, Channel):
                raise PulseError(f"Expected a channel, got {channel} instead.")

    @property
    def name(self) -> str:
        """Name of this instruction."""
        return self._name

    @property
    def id(self) -> int:  # pylint: disable=invalid-name
        """Unique identifier for this instruction."""
        return id(self)

    @property
    def operands(self) -> tuple:
        """Return instruction operands."""
        return self._operands

    @property
    @abstractmethod
    def channels(self) -> tuple[Channel, ...]:
        """Returns the channels that this schedule uses."""
        raise NotImplementedError

    @property
    def start_time(self) -> int:
        """Relative begin time of this instruction."""
        return 0

    @property
    def stop_time(self) -> int:
        """Relative end time of this instruction."""
        return self.duration

    @property
    def duration(self) -> int | ParameterExpression:
        """Duration of this instruction."""
        raise NotImplementedError

    @property
    def _children(self) -> tuple["Instruction", ...]:
        """Instruction has no child nodes."""
        return ()

    @property
    def instructions(self) -> tuple[tuple[int, "Instruction"], ...]:
        """Iterable for getting instructions from Schedule tree."""
        return tuple(self._instructions())

    def ch_duration(self, *channels: Channel) -> int:
        """Return duration of the supplied channels in this Instruction.

        Args:
            *channels: Supplied channels
        """
        return self.ch_stop_time(*channels)

    def ch_start_time(self, *channels: Channel) -> int:
        # pylint: disable=unused-argument
        """Return minimum start time for supplied channels.

        Args:
            *channels: Supplied channels
        """
        return 0

    def ch_stop_time(self, *channels: Channel) -> int:
        """Return maximum start time for supplied channels.

        Args:
            *channels: Supplied channels
        """
        if any(chan in self.channels for chan in channels):
            return self.duration
        return 0

    def _instructions(self, time: int = 0) -> Iterable[tuple[int, "Instruction"]]:
        """Iterable for flattening Schedule tree.

        Args:
            time: Shifted time of this node due to parent

        Yields:
            Tuple[int, Union['Schedule, 'Instruction']]: Tuple of the form
                (start_time, instruction).
        """
        yield (time, self)

    def shift(self, time: int, name: str | None = None):
        """Return a new schedule shifted forward by `time`.

        Args:
            time: Time to shift by
            name: Name of the new schedule. Defaults to name of self

        Returns:
            Schedule: The shifted schedule.
        """
        from qiskit.pulse.schedule import Schedule

        if name is None:
            name = self.name
        return Schedule((time, self), name=name)

    def insert(self, start_time: int, schedule, name: str | None = None):
        """Return a new :class:`~qiskit.pulse.Schedule` with ``schedule`` inserted within
        ``self`` at ``start_time``.

        Args:
            start_time: Time to insert the schedule schedule
            schedule (Union['Schedule', 'Instruction']): Schedule or instruction to insert
            name: Name of the new schedule. Defaults to name of self

        Returns:
            Schedule: A new schedule with ``schedule`` inserted with this instruction at t=0.
        """
        from qiskit.pulse.schedule import Schedule

        if name is None:
            name = self.name
        return Schedule(self, (start_time, schedule), name=name)

    def append(self, schedule, name: str | None = None):
        """Return a new :class:`~qiskit.pulse.Schedule` with ``schedule`` inserted at the
        maximum time over all channels shared between ``self`` and ``schedule``.

        Args:
            schedule (Union['Schedule', 'Instruction']): Schedule or instruction to be appended
            name: Name of the new schedule. Defaults to name of self

        Returns:
            Schedule: A new schedule with ``schedule`` a this instruction at t=0.
        """
        common_channels = set(self.channels) & set(schedule.channels)
        time = self.ch_stop_time(*common_channels)
        return self.insert(time, schedule, name=name)

    @property
    def parameters(self) -> set:
        """Parameters which determine the instruction behavior."""

        def _get_parameters_recursive(obj):
            params = set()
            if hasattr(obj, "parameters"):
                for param in obj.parameters:
                    if isinstance(param, Parameter):
                        params.add(param)
                    else:
                        params |= _get_parameters_recursive(param)
            return params

        parameters = set()
        for op in self.operands:
            parameters |= _get_parameters_recursive(op)
        return parameters

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return any(self.parameters)

    def __eq__(self, other: object) -> bool:
        """Check if this Instruction is equal to the `other` instruction.

        Equality is determined by the instruction sharing the same operands and channels.
        """
        if not isinstance(other, Instruction):
            return NotImplemented
        return isinstance(other, type(self)) and self.operands == other.operands

    def __hash__(self) -> int:
        return hash((type(self), self.operands, self.name))

    def __add__(self, other):
        """Return a new schedule with `other` inserted within `self` at `start_time`.

        Args:
            other (Union['Schedule', 'Instruction']): Schedule or instruction to be appended

        Returns:
            Schedule: A new schedule with ``schedule`` appended after this instruction at t=0.
        """
        return self.append(other)

    def __or__(self, other):
        """Return a new schedule which is the union of `self` and `other`.

        Args:
            other (Union['Schedule', 'Instruction']): Schedule or instruction to union with

        Returns:
            Schedule: A new schedule with ``schedule`` inserted with this instruction at t=0
        """
        return self.insert(0, other)

    def __lshift__(self, time: int):
        """Return a new schedule which is shifted forward by `time`.

        Returns:
            Schedule: The shifted schedule
        """
        return self.shift(time)

    def __repr__(self) -> str:
        operands = ", ".join(str(op) for op in self.operands)
        name_repr = f", name='{self.name}'" if self.name else ""
        return f"{self.__class__.__name__}({operands}{name_repr})"

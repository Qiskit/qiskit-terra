# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An instruction for blocking time on a channel; useful for scheduling alignment."""
from __future__ import annotations

from qiskit.circuit import ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.utils.deprecate_pulse import deprecate_pulse_func


class Delay(Instruction):
    """A blocking instruction with no other effect. The delay is used for aligning and scheduling
    other instructions.

    Example:

        To schedule an instruction at time = 10, on a channel assigned to the variable ``channel``,
        the following could be used::

            sched = Schedule(name="Delay instruction example")
            sched += Delay(10, channel)
            sched += Gaussian(duration, amp, sigma, channel)

        The ``channel`` will output no signal from time=0 up until time=10.
    """

    @deprecate_pulse_func
    def __init__(
        self,
        duration: int | ParameterExpression,
        channel: Channel,
        name: str | None = None,
    ):
        """Create a new delay instruction.

        No other instruction may be scheduled within a ``Delay``.

        Args:
            duration: Length of time of the delay in terms of dt.
            channel: The channel that will have the delay.
            name: Name of the delay for display purposes.
        """
        super().__init__(operands=(duration, channel), name=name)

    @property
    def channel(self) -> Channel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self.operands[1]

    @property
    def channels(self) -> tuple[Channel]:
        """Returns the channels that this schedule uses."""
        return (self.channel,)

    @property
    def duration(self) -> int | ParameterExpression:
        """Duration of this instruction."""
        return self.operands[0]

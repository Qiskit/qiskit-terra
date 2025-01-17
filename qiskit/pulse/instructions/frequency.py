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

"""Frequency instructions module. These instructions allow the user to manipulate
the frequency of a channel.
"""
from typing import Optional, Union, Tuple

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.pulse.exceptions import PulseError
from qiskit.utils.deprecate_pulse import deprecate_pulse_func


class SetFrequency(Instruction):
    r"""Set the channel frequency. This instruction operates on ``PulseChannel`` s.
    A ``PulseChannel`` creates pulses of the form

    .. math::
        Re[\exp(i 2\pi f jdt + \phi) d_j].

    Here, :math:`f` is the frequency of the channel. The instruction ``SetFrequency`` allows
    the user to set the value of :math:`f`. All pulses that are played on a channel
    after SetFrequency has been called will have the corresponding frequency.

    The duration of SetFrequency is 0.
    """

    @deprecate_pulse_func
    def __init__(
        self,
        frequency: Union[float, ParameterExpression],
        channel: PulseChannel,
        name: Optional[str] = None,
    ):
        """Creates a new set channel frequency instruction.

        Args:
            frequency: New frequency of the channel in Hz.
            channel: The channel this instruction operates on.
            name: Name of this set channel frequency instruction.
        """
        super().__init__(operands=(frequency, channel), name=name)

    def _validate(self):
        """Called after initialization to validate instruction data.

        Raises:
            PulseError: If the input ``channel`` is not type :class:`PulseChannel`.
        """
        if not isinstance(self.channel, PulseChannel):
            raise PulseError(f"Expected a pulse channel, got {self.channel} instead.")

    @property
    def frequency(self) -> Union[float, ParameterExpression]:
        """New frequency."""
        return self.operands[0]

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self.operands[1]

    @property
    def channels(self) -> Tuple[PulseChannel]:
        """Returns the channels that this schedule uses."""
        return (self.channel,)

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return 0


class ShiftFrequency(Instruction):
    """Shift the channel frequency away from the current frequency."""

    @deprecate_pulse_func
    def __init__(
        self,
        frequency: Union[float, ParameterExpression],
        channel: PulseChannel,
        name: Optional[str] = None,
    ):
        """Creates a new shift frequency instruction.

        Args:
            frequency: Frequency shift of the channel in Hz.
            channel: The channel this instruction operates on.
            name: Name of this set channel frequency instruction.
        """
        super().__init__(operands=(frequency, channel), name=name)

    def _validate(self):
        """Called after initialization to validate instruction data.

        Raises:
            PulseError: If the input ``channel`` is not type :class:`PulseChannel`.
        """
        if not isinstance(self.channel, PulseChannel):
            raise PulseError(f"Expected a pulse channel, got {self.channel} instead.")

    @property
    def frequency(self) -> Union[float, ParameterExpression]:
        """Frequency shift from the set frequency."""
        return self.operands[0]

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self.operands[1]

    @property
    def channels(self) -> Tuple[PulseChannel]:
        """Returns the channels that this schedule uses."""
        return (self.channel,)

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return 0

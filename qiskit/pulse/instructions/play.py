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

"""An instruction to transmit a given pulse on a ``PulseChannel`` (i.e., those which support
transmitted pulses, such as ``DriveChannel``).
"""
from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.pulse.library.pulse import Pulse
from qiskit.utils.deprecate_pulse import deprecate_pulse_func


class Play(Instruction):
    """This instruction is responsible for applying a pulse on a channel.

    The pulse specifies the exact time dynamics of the output signal envelope for a limited
    time. The output is modulated by a phase and frequency which are controlled by separate
    instructions. The pulse duration must be fixed, and is implicitly given in terms of the
    cycle time, dt, of the backend.
    """

    @deprecate_pulse_func
    def __init__(self, pulse: Pulse, channel: PulseChannel, name: str | None = None):
        """Create a new pulse instruction.

        Args:
            pulse: A pulse waveform description, such as
                   :py:class:`~qiskit.pulse.library.Waveform`.
            channel: The channel to which the pulse is applied.
            name: Name of the instruction for display purposes. Defaults to ``pulse.name``.
        """
        if name is None:
            name = pulse.name
        super().__init__(operands=(pulse, channel), name=name)

    def _validate(self):
        """Called after initialization to validate instruction data.

        Raises:
            PulseError: If pulse is not a Pulse type.
            PulseError: If the input ``channel`` is not type :class:`PulseChannel`.
        """
        if not isinstance(self.pulse, Pulse):
            raise PulseError("The `pulse` argument to `Play` must be of type `library.Pulse`.")

        if not isinstance(self.channel, PulseChannel):
            raise PulseError(f"Expected a pulse channel, got {self.channel} instead.")

    @property
    def pulse(self) -> Pulse:
        """A description of the samples that will be played."""
        return self.operands[0]

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self.operands[1]

    @property
    def channels(self) -> tuple[PulseChannel]:
        """Returns the channels that this schedule uses."""
        return (self.channel,)

    @property
    def duration(self) -> int | ParameterExpression:
        """Duration of this instruction."""
        return self.pulse.duration

    @property
    def parameters(self) -> set[Parameter]:
        """Parameters which determine the instruction behavior."""
        parameters: set[Parameter] = set()

        # Note that Pulse.parameters returns dict rather than set for convention.
        # We need special handling for Play instruction.
        for pulse_param_expr in self.pulse.parameters.values():
            if isinstance(pulse_param_expr, ParameterExpression):
                parameters = parameters | pulse_param_expr.parameters

        if self.channel.is_parameterized():
            parameters = parameters | self.channel.parameters

        return parameters

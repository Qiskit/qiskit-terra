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

"""The Acquire instruction is used to trigger the qubit measurement unit and provide
some metadata for the acquisition process, for example, where to store classified readout data.
"""
from __future__ import annotations
from qiskit.circuit import ParameterExpression
from qiskit.pulse.channels import MemorySlot, RegisterSlot, AcquireChannel
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.utils.deprecate_pulse import deprecate_pulse_func


class Acquire(Instruction):
    """The Acquire instruction is used to trigger the ADC associated with a particular qubit;
    e.g. instantiated with AcquireChannel(0), the Acquire command will trigger data collection
    for the channel associated with qubit 0 readout. This instruction also provides acquisition
    metadata:

     * the number of cycles during which to acquire (in terms of dt),

     * the register slot to store classified, intermediary readout results,

     * the memory slot to return classified results,

     * the kernel to integrate raw data for each shot, and

     * the discriminator to classify kerneled IQ points.
    """

    @deprecate_pulse_func
    def __init__(
        self,
        duration: int | ParameterExpression,
        channel: AcquireChannel,
        mem_slot: MemorySlot | None = None,
        reg_slot: RegisterSlot | None = None,
        kernel: Kernel | None = None,
        discriminator: Discriminator | None = None,
        name: str | None = None,
    ):
        """Create a new Acquire instruction.

        Args:
            duration: Length of time to acquire data in terms of dt.
            channel: The channel that will acquire data.
            mem_slot: The classical memory slot in which to store the classified readout result.
            reg_slot: The fast-access register slot in which to store the classified readout
                      result for fast feedback.
            kernel: A ``Kernel`` for integrating raw data.
            discriminator: A ``Discriminator`` for discriminating kerneled IQ data into 0/1
                           results.
            name: Name of the instruction for display purposes.
        """
        super().__init__(
            operands=(duration, channel, mem_slot, reg_slot, kernel, discriminator),
            name=name,
        )

    def _validate(self):
        """Called after initialization to validate instruction data.

        Raises:
            PulseError: If the input ``channel`` is not type :class:`AcquireChannel`.
            PulseError: If the input ``mem_slot`` is not type :class:`MemorySlot`.
            PulseError: If the input ``reg_slot`` is not type :class:`RegisterSlot`.
            PulseError: When memory slot and register slot are both empty.
        """
        if not isinstance(self.channel, AcquireChannel):
            raise PulseError(f"Expected an acquire channel, got {self.channel} instead.")

        if self.mem_slot and not isinstance(self.mem_slot, MemorySlot):
            raise PulseError(f"Expected a memory slot, got {self.mem_slot} instead.")

        if self.reg_slot and not isinstance(self.reg_slot, RegisterSlot):
            raise PulseError(f"Expected a register slot, got {self.reg_slot} instead.")

        if self.mem_slot is None and self.reg_slot is None:
            raise PulseError("Neither MemorySlots nor RegisterSlots were supplied.")

    @property
    def channel(self) -> AcquireChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self.operands[1]

    @property
    def channels(self) -> tuple[AcquireChannel | MemorySlot | RegisterSlot, ...]:
        """Returns the channels that this schedule uses."""
        return tuple(self.operands[ind] for ind in (1, 2, 3) if self.operands[ind] is not None)

    @property
    def duration(self) -> int | ParameterExpression:
        """Duration of this instruction."""
        return self.operands[0]

    @property
    def kernel(self) -> Kernel:
        """Return kernel settings."""
        return self._operands[4]

    @property
    def discriminator(self) -> Discriminator:
        """Return discrimination settings."""
        return self._operands[5]

    @property
    def acquire(self) -> AcquireChannel:
        """Acquire channel to acquire data. The ``AcquireChannel`` index maps trivially to
        qubit index.
        """
        return self.channel

    @property
    def mem_slot(self) -> MemorySlot:
        """The classical memory slot which will store the classified readout result."""
        return self.operands[2]

    @property
    def reg_slot(self) -> RegisterSlot:
        """The fast-access register slot which will store the classified readout result for
        fast-feedback computation.
        """
        return self.operands[3]

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return isinstance(self.duration, ParameterExpression) or super().is_parameterized()

    def __repr__(self) -> str:
        mem_slot_repr = str(self.mem_slot) if self.mem_slot else ""
        reg_slot_repr = str(self.reg_slot) if self.reg_slot else ""
        kernel_repr = str(self.kernel) if self.kernel else ""
        discriminator_repr = str(self.discriminator) if self.discriminator else ""
        return (
            f"{self.__class__.__name__}({self.duration}, {str(self.channel)}, "
            f"{mem_slot_repr}, {reg_slot_repr}, {kernel_repr}, {discriminator_repr})"
        )

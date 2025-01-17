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

# pylint: disable=invalid-name, missing-function-docstring

"""Helper class used to convert a pulse instruction into PulseQobjInstruction."""

import hashlib
import re
import warnings
from enum import Enum
from functools import singledispatchmethod
from typing import Union, List, Iterator, Optional
import numpy as np

from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse import channels, instructions, library
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import QiskitError
from qiskit.pulse.parser import parse_string_expr
from qiskit.pulse.schedule import Schedule
from qiskit.qobj import QobjMeasurementOption, PulseLibraryItem, PulseQobjInstruction
from qiskit.qobj.utils import MeasLevel
from qiskit.utils import deprecate_func
from qiskit.utils.deprecate_pulse import deprecate_pulse_dependency


class ParametricPulseShapes(Enum):
    """Map the assembled pulse names to the pulse module waveforms.

    The enum name is the transport layer name for pulse shapes, the
    value is its mapping to the OpenPulse Command in Qiskit.
    """

    gaussian = "Gaussian"
    gaussian_square = "GaussianSquare"
    gaussian_square_drag = "GaussianSquareDrag"
    gaussian_square_echo = "gaussian_square_echo"
    drag = "Drag"
    constant = "Constant"

    @classmethod
    def from_instance(
        cls,
        instance: library.SymbolicPulse,
    ) -> "ParametricPulseShapes":
        """Get Qobj name from the pulse class instance.

        Args:
            instance: SymbolicPulse class.

        Returns:
            Qobj name.

        Raises:
            QiskitError: When pulse instance is not recognizable type.
        """
        if isinstance(instance, library.SymbolicPulse):
            return cls(instance.pulse_type)

        raise QiskitError(f"'{instance}' is not valid pulse type.")

    @classmethod
    def to_type(cls, name: str) -> library.SymbolicPulse:
        """Get symbolic pulse class from the name.

        Args:
            name: Qobj name of the pulse.

        Returns:
            Corresponding class.
        """
        return getattr(library, cls[name].value)


class InstructionToQobjConverter:
    """Converts Qiskit Pulse in-memory representation into Qobj data.

    This converter converts the Qiskit Pulse in-memory representation into
    the transfer layer format to submit the data from client to the server.

    The transfer layer format must be the text representation that conforms to
    the `OpenPulse specification<https://arxiv.org/abs/1809.03452>`__.
    Extension to the OpenPulse can be achieved by subclassing this this with
    extra methods corresponding to each augmented instruction. For example,

    .. plot::
       :include-source:
       :nofigs:

        class MyConverter(InstructionToQobjConverter):

            def _convert_NewInstruction(self, instruction, time_offset):
                command_dict = {
                    'name': 'new_inst',
                    't0': time_offset + instruction.start_time,
                    'param1': instruction.param1,
                    'param2': instruction.param2
                }
                return self._qobj_model(**command_dict)

    where ``NewInstruction`` must be a class name of Qiskit Pulse instruction.
    """

    @deprecate_func(
        since="1.2",
        removal_timeline="in the 2.0 release",
        additional_msg="The `Qobj` class and related functionality are part of the deprecated "
        "`BackendV1` workflow,  and no longer necessary for `BackendV2`. If a user "
        "workflow requires `Qobj` it likely relies on deprecated functionality and "
        "should be updated to use `BackendV2`.",
    )
    def __init__(
        self,
        qobj_model: PulseQobjInstruction,
        **run_config,
    ):
        """Create new converter.

        Args:
             qobj_model: Transfer layer data schema.
             run_config: Run configuration.
        """
        self._qobj_model = qobj_model
        self._run_config = run_config

    def __call__(
        self,
        shift: int,
        instruction: Union[instructions.Instruction, List[instructions.Acquire]],
    ) -> PulseQobjInstruction:
        """Convert Qiskit in-memory representation to Qobj instruction.

        Args:
            instruction: Instruction data in Qiskit Pulse.

        Returns:
            Qobj instruction data.

        Raises:
            QiskitError: When list of instruction is provided except for Acquire.
        """
        if isinstance(instruction, list):
            if all(isinstance(inst, instructions.Acquire) for inst in instruction):
                return self._convert_bundled_acquire(
                    instruction_bundle=instruction,
                    time_offset=shift,
                )
            raise QiskitError("Bundle of instruction is not supported except for Acquire.")

        return self._convert_instruction(instruction, shift)

    @singledispatchmethod
    def _convert_instruction(
        self,
        instruction,
        time_offset: int,
    ) -> PulseQobjInstruction:
        raise QiskitError(
            f"Pulse Qobj doesn't support {instruction.__class__.__name__}. "
            "This instruction cannot be submitted with Qobj."
        )

    @_convert_instruction.register(instructions.Acquire)
    def _convert_acquire(
        self,
        instruction,
        time_offset: int,
    ) -> PulseQobjInstruction:
        """Return converted `Acquire`.

        Args:
            instruction: Qiskit Pulse acquire instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        meas_level = self._run_config.get("meas_level", 2)
        mem_slot = []
        if instruction.mem_slot:
            mem_slot = [instruction.mem_slot.index]

        command_dict = {
            "name": "acquire",
            "t0": time_offset + instruction.start_time,
            "duration": instruction.duration,
            "qubits": [instruction.channel.index],
            "memory_slot": mem_slot,
        }
        if meas_level == MeasLevel.CLASSIFIED:
            # setup discriminators
            if instruction.discriminator:
                command_dict.update(
                    {
                        "discriminators": [
                            QobjMeasurementOption(
                                name=instruction.discriminator.name,
                                params=instruction.discriminator.params,
                            )
                        ]
                    }
                )
            # setup register_slots
            if instruction.reg_slot:
                command_dict.update({"register_slot": [instruction.reg_slot.index]})
        if meas_level in [MeasLevel.KERNELED, MeasLevel.CLASSIFIED]:
            # setup kernels
            if instruction.kernel:
                command_dict.update(
                    {
                        "kernels": [
                            QobjMeasurementOption(
                                name=instruction.kernel.name, params=instruction.kernel.params
                            )
                        ]
                    }
                )
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.SetFrequency)
    def _convert_set_frequency(
        self,
        instruction,
        time_offset: int,
    ) -> PulseQobjInstruction:
        """Return converted `SetFrequency`.

        Args:
            instruction: Qiskit Pulse set frequency instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        command_dict = {
            "name": "setf",
            "t0": time_offset + instruction.start_time,
            "ch": instruction.channel.name,
            "frequency": instruction.frequency / 10**9,
        }
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.ShiftFrequency)
    def _convert_shift_frequency(
        self,
        instruction,
        time_offset: int,
    ) -> PulseQobjInstruction:
        """Return converted `ShiftFrequency`.

        Args:
            instruction: Qiskit Pulse shift frequency instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        command_dict = {
            "name": "shiftf",
            "t0": time_offset + instruction.start_time,
            "ch": instruction.channel.name,
            "frequency": instruction.frequency / 10**9,
        }
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.SetPhase)
    def _convert_set_phase(
        self,
        instruction,
        time_offset: int,
    ) -> PulseQobjInstruction:
        """Return converted `SetPhase`.

        Args:
            instruction: Qiskit Pulse set phase instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        command_dict = {
            "name": "setp",
            "t0": time_offset + instruction.start_time,
            "ch": instruction.channel.name,
            "phase": instruction.phase,
        }
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.ShiftPhase)
    def _convert_shift_phase(
        self,
        instruction,
        time_offset: int,
    ) -> PulseQobjInstruction:
        """Return converted `ShiftPhase`.

        Args:
            instruction: Qiskit Pulse shift phase instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        command_dict = {
            "name": "fc",
            "t0": time_offset + instruction.start_time,
            "ch": instruction.channel.name,
            "phase": instruction.phase,
        }
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.Delay)
    def _convert_delay(
        self,
        instruction,
        time_offset: int,
    ) -> PulseQobjInstruction:
        """Return converted `Delay`.

        Args:
            instruction: Qiskit Pulse delay instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        command_dict = {
            "name": "delay",
            "t0": time_offset + instruction.start_time,
            "ch": instruction.channel.name,
            "duration": instruction.duration,
        }
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.Play)
    def _convert_play(
        self,
        instruction,
        time_offset: int,
    ) -> PulseQobjInstruction:
        """Return converted `Play`.

        Args:
            instruction: Qiskit Pulse play instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        if isinstance(instruction.pulse, library.SymbolicPulse):
            params = dict(instruction.pulse.parameters)
            # IBM backends expect "amp" to be the complex amplitude
            if "amp" in params and "angle" in params:
                params["amp"] = complex(params["amp"] * np.exp(1j * params["angle"]))
                del params["angle"]

            command_dict = {
                "name": "parametric_pulse",
                "pulse_shape": ParametricPulseShapes.from_instance(instruction.pulse).name,
                "t0": time_offset + instruction.start_time,
                "ch": instruction.channel.name,
                "parameters": params,
            }
        else:
            command_dict = {
                "name": instruction.name,
                "t0": time_offset + instruction.start_time,
                "ch": instruction.channel.name,
            }

        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.Snapshot)
    def _convert_snapshot(
        self,
        instruction,
        time_offset: int,
    ) -> PulseQobjInstruction:
        """Return converted `Snapshot`.

        Args:
            time_offset: Offset time.
            instruction: Qiskit Pulse snapshot instruction.

        Returns:
            Qobj instruction data.
        """
        command_dict = {
            "name": "snapshot",
            "t0": time_offset + instruction.start_time,
            "label": instruction.label,
            "type": instruction.type,
        }
        return self._qobj_model(**command_dict)

    def _convert_bundled_acquire(
        self,
        instruction_bundle: List[instructions.Acquire],
        time_offset: int,
    ) -> PulseQobjInstruction:
        """Return converted list of parallel `Acquire` instructions.

        Args:
            instruction_bundle: List of Qiskit Pulse acquire instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.

        Raises:
            QiskitError: When instructions are not aligned.
            QiskitError: When instructions have different duration.
            QiskitError: When discriminator or kernel is missing in a part of instructions.
        """
        meas_level = self._run_config.get("meas_level", 2)

        t0 = instruction_bundle[0].start_time
        duration = instruction_bundle[0].duration
        memory_slots = []
        register_slots = []
        qubits = []
        discriminators = []
        kernels = []

        for instruction in instruction_bundle:
            qubits.append(instruction.channel.index)

            if instruction.start_time != t0:
                raise QiskitError(
                    "The supplied acquire instructions have different starting times. "
                    "Something has gone wrong calling this code. Please report this "
                    "issue."
                )

            if instruction.duration != duration:
                raise QiskitError(
                    "Acquire instructions beginning at the same time must have same duration."
                )

            if instruction.mem_slot:
                memory_slots.append(instruction.mem_slot.index)

            if meas_level == MeasLevel.CLASSIFIED:
                # setup discriminators
                if instruction.discriminator:
                    discriminators.append(
                        QobjMeasurementOption(
                            name=instruction.discriminator.name,
                            params=instruction.discriminator.params,
                        )
                    )
                # setup register_slots
                if instruction.reg_slot:
                    register_slots.append(instruction.reg_slot.index)

            if meas_level in [MeasLevel.KERNELED, MeasLevel.CLASSIFIED]:
                # setup kernels
                if instruction.kernel:
                    kernels.append(
                        QobjMeasurementOption(
                            name=instruction.kernel.name, params=instruction.kernel.params
                        )
                    )
        command_dict = {
            "name": "acquire",
            "t0": time_offset + t0,
            "duration": duration,
            "qubits": qubits,
        }
        if memory_slots:
            command_dict["memory_slot"] = memory_slots

        if register_slots:
            command_dict["register_slot"] = register_slots

        if discriminators:
            num_discriminators = len(discriminators)
            if num_discriminators == len(qubits) or num_discriminators == 1:
                command_dict["discriminators"] = discriminators
            else:
                raise QiskitError(
                    "A discriminator must be supplied for every acquisition or a single "
                    "discriminator for all acquisitions."
                )

        if kernels:
            num_kernels = len(kernels)
            if num_kernels == len(qubits) or num_kernels == 1:
                command_dict["kernels"] = kernels
            else:
                raise QiskitError(
                    "A kernel must be supplied for every acquisition or a single "
                    "kernel for all acquisitions."
                )

        return self._qobj_model(**command_dict)


class QobjToInstructionConverter:
    """Converts Qobj data into Qiskit Pulse in-memory representation.

    This converter converts data from transfer layer into the in-memory representation of
    the front-end of Qiskit Pulse.

    The transfer layer format must be the text representation that conforms to
    the `OpenPulse specification<https://arxiv.org/abs/1809.03452>`__.
    Extension to the OpenPulse can be achieved by subclassing this this with
    extra methods corresponding to each augmented instruction. For example,

    .. plot::
       :include-source:
       :nofigs:

        class MyConverter(QobjToInstructionConverter):

            def get_supported_instructions(self):
                instructions = super().get_supported_instructions()
                instructions += ["new_inst"]

                return instructions

            def _convert_new_inst(self, instruction):
                return NewInstruction(...)

    where ``NewInstruction`` must be a subclass of :class:`~qiskit.pulse.instructions.Instruction`.
    """

    __chan_regex__ = re.compile(r"([a-zA-Z]+)(\d+)")

    @deprecate_pulse_dependency
    def __init__(
        self,
        pulse_library: Optional[List[PulseLibraryItem]] = None,
        **run_config,
    ):
        """Create new converter.

        Args:
            pulse_library: Pulse library in Qobj format.
             run_config: Run configuration.
        """
        pulse_library_dict = {}
        for lib_item in pulse_library:
            pulse_library_dict[lib_item.name] = lib_item.samples

        self._pulse_library = pulse_library_dict
        self._run_config = run_config

    def __call__(self, instruction: PulseQobjInstruction) -> Schedule:
        """Convert Qobj instruction to Qiskit in-memory representation.

        Args:
            instruction: Instruction data in Qobj format.

        Returns:
            Scheduled Qiskit Pulse instruction in Schedule format.
        """
        schedule = Schedule()
        for inst in self._get_sequences(instruction):
            schedule.insert(instruction.t0, inst, inplace=True)
        return schedule

    def _get_sequences(
        self,
        instruction: PulseQobjInstruction,
    ) -> Iterator[instructions.Instruction]:
        """A method to iterate over pulse instructions without creating Schedule.

        .. note::

            This is internal fast-path function, and callers other than this converter class
            might directly use this method to generate schedule from multiple
            Qobj instructions. Because __call__ always returns a schedule with the time offset
            parsed instruction, composing multiple Qobj instructions to create
            a gate schedule is somewhat inefficient due to composing overhead of schedules.
            Directly combining instructions with this method is much performant.

        Args:
            instruction: Instruction data in Qobj format.

        Yields:
            Qiskit Pulse instructions.

        :meta public:
        """
        try:
            method = getattr(self, f"_convert_{instruction.name}")
        except AttributeError:
            method = self._convert_generic

        yield from method(instruction)

    def get_supported_instructions(self) -> List[str]:
        """Retrun a list of supported instructions."""
        return [
            "acquire",
            "setp",
            "fc",
            "setf",
            "shiftf",
            "delay",
            "parametric_pulse",
            "snapshot",
        ]

    def get_channel(self, channel: str) -> channels.PulseChannel:
        """Parse and retrieve channel from ch string.

        Args:
            channel: String identifier of pulse instruction channel.

        Returns:
            Matched channel object.

        Raises:
            QiskitError: Is raised if valid channel is not matched
        """
        match = self.__chan_regex__.match(channel)
        if match:
            prefix, index = match.group(1), int(match.group(2))

            if prefix == channels.DriveChannel.prefix:
                return channels.DriveChannel(index)
            elif prefix == channels.MeasureChannel.prefix:
                return channels.MeasureChannel(index)
            elif prefix == channels.ControlChannel.prefix:
                return channels.ControlChannel(index)

        raise QiskitError(f"Channel {channel} is not valid")

    @staticmethod
    def disassemble_value(value_expr: Union[float, str]) -> Union[float, ParameterExpression]:
        """A helper function to format instruction operand.

        If parameter in string representation is specified, this method parses the
        input string and generates Qiskit ParameterExpression object.

        Args:
            value_expr: Operand value in Qobj.

        Returns:
            Parsed operand value. ParameterExpression object is returned if value is not number.
        """
        if isinstance(value_expr, str):
            str_expr = parse_string_expr(value_expr, partial_binding=False)
            value_expr = str_expr(**{pname: Parameter(pname) for pname in str_expr.params})

        return value_expr

    def _convert_acquire(
        self,
        instruction: PulseQobjInstruction,
    ) -> Iterator[instructions.Instruction]:
        """Return converted `Acquire` instruction.

        Args:
            instruction: Acquire qobj

        Yields:
            Qiskit Pulse acquire instructions
        """
        duration = instruction.duration
        qubits = instruction.qubits
        acquire_channels = [channels.AcquireChannel(qubit) for qubit in qubits]

        mem_slots = [channels.MemorySlot(instruction.memory_slot[i]) for i in range(len(qubits))]

        if hasattr(instruction, "register_slot"):
            register_slots = [
                channels.RegisterSlot(instruction.register_slot[i]) for i in range(len(qubits))
            ]
        else:
            register_slots = [None] * len(qubits)

        discriminators = (
            instruction.discriminators if hasattr(instruction, "discriminators") else None
        )
        if not isinstance(discriminators, list):
            discriminators = [discriminators]
        if any(discriminators[i] != discriminators[0] for i in range(len(discriminators))):
            warnings.warn(
                "Can currently only support one discriminator per acquire. Defaulting "
                "to first discriminator entry."
            )
        discriminator = discriminators[0]
        if discriminator:
            discriminator = Discriminator(name=discriminators[0].name, **discriminators[0].params)

        kernels = instruction.kernels if hasattr(instruction, "kernels") else None
        if not isinstance(kernels, list):
            kernels = [kernels]
        if any(kernels[0] != kernels[i] for i in range(len(kernels))):
            warnings.warn(
                "Can currently only support one kernel per acquire. Defaulting to first "
                "kernel entry."
            )
        kernel = kernels[0]
        if kernel:
            kernel = Kernel(name=kernels[0].name, **kernels[0].params)

        for acquire_channel, mem_slot, reg_slot in zip(acquire_channels, mem_slots, register_slots):
            yield instructions.Acquire(
                duration,
                acquire_channel,
                mem_slot=mem_slot,
                reg_slot=reg_slot,
                kernel=kernel,
                discriminator=discriminator,
            )

    def _convert_setp(
        self,
        instruction: PulseQobjInstruction,
    ) -> Iterator[instructions.Instruction]:
        """Return converted `SetPhase` instruction.

        Args:
            instruction: SetPhase qobj instruction

        Yields:
            Qiskit Pulse set phase instructions
        """
        channel = self.get_channel(instruction.ch)
        phase = self.disassemble_value(instruction.phase)

        yield instructions.SetPhase(phase, channel)

    def _convert_fc(
        self,
        instruction: PulseQobjInstruction,
    ) -> Iterator[instructions.Instruction]:
        """Return converted `ShiftPhase` instruction.

        Args:
            instruction: ShiftPhase qobj instruction

        Yields:
            Qiskit Pulse shift phase schedule instructions
        """
        channel = self.get_channel(instruction.ch)
        phase = self.disassemble_value(instruction.phase)

        yield instructions.ShiftPhase(phase, channel)

    def _convert_setf(
        self,
        instruction: PulseQobjInstruction,
    ) -> Iterator[instructions.Instruction]:
        """Return converted `SetFrequencyInstruction` instruction.

        .. note::

            We assume frequency value is expressed in string with "GHz".
            Operand value is thus scaled by a factor of 10^9.

        Args:
            instruction: SetFrequency qobj instruction

        Yields:
            Qiskit Pulse set frequency instructions
        """
        channel = self.get_channel(instruction.ch)
        frequency = self.disassemble_value(instruction.frequency) * 10**9

        yield instructions.SetFrequency(frequency, channel)

    def _convert_shiftf(
        self,
        instruction: PulseQobjInstruction,
    ) -> Iterator[instructions.Instruction]:
        """Return converted `ShiftFrequency` instruction.

        .. note::

            We assume frequency value is expressed in string with "GHz".
            Operand value is thus scaled by a factor of 10^9.

        Args:
            instruction: ShiftFrequency qobj instruction

        Yields:
            Qiskit Pulse shift frequency schedule instructions
        """
        channel = self.get_channel(instruction.ch)
        frequency = self.disassemble_value(instruction.frequency) * 10**9

        yield instructions.ShiftFrequency(frequency, channel)

    def _convert_delay(
        self,
        instruction: PulseQobjInstruction,
    ) -> Iterator[instructions.Instruction]:
        """Return converted `Delay` instruction.

        Args:
            instruction: Delay qobj instruction

        Yields:
            Qiskit Pulse delay instructions
        """
        channel = self.get_channel(instruction.ch)
        duration = instruction.duration

        yield instructions.Delay(duration, channel)

    def _convert_parametric_pulse(
        self,
        instruction: PulseQobjInstruction,
    ) -> Iterator[instructions.Instruction]:
        """Return converted `Play` instruction with parametric pulse operand.

        .. note::

            If parametric pulse label is not provided by the backend, this method naively generates
            a pulse name based on the pulse shape and bound parameters. This pulse name is formatted
            to, for example, `gaussian_a4e3`, here the last four digits are a part of
            the hash string generated based on the pulse shape and the parameters.
            Because we are using a truncated hash for readability,
            there may be a small risk of pulse name collision with other pulses.
            Basically the parametric pulse name is used just for visualization purpose and
            the pulse module should not have dependency on the parametric pulse names.

        Args:
            instruction: Play qobj instruction with parametric pulse

        Yields:
            Qiskit Pulse play schedule instructions
        """
        channel = self.get_channel(instruction.ch)

        try:
            pulse_name = instruction.label
        except AttributeError:
            sorted_params = sorted(instruction.parameters.items(), key=lambda x: x[0])
            base_str = f"{instruction.pulse_shape}_{str(sorted_params)}"
            short_pulse_id = hashlib.md5(base_str.encode("utf-8")).hexdigest()[:4]
            pulse_name = f"{instruction.pulse_shape}_{short_pulse_id}"
        params = dict(instruction.parameters)
        if "amp" in params and isinstance(params["amp"], complex):
            params["angle"] = np.angle(params["amp"])
            params["amp"] = np.abs(params["amp"])
        pulse = ParametricPulseShapes.to_type(instruction.pulse_shape)(**params, name=pulse_name)

        yield instructions.Play(pulse, channel)

    def _convert_snapshot(
        self,
        instruction: PulseQobjInstruction,
    ) -> Iterator[instructions.Instruction]:
        """Return converted `Snapshot` instruction.

        Args:
            instruction: Snapshot qobj instruction

        Yields:
            Qiskit Pulse snapshot instructions
        """
        yield instructions.Snapshot(instruction.label, instruction.type)

    def _convert_generic(
        self,
        instruction: PulseQobjInstruction,
    ) -> Iterator[instructions.Instruction]:
        """Convert generic pulse instruction.

        Args:
            instruction: Generic qobj instruction

        Yields:
            Qiskit Pulse generic instructions

        Raises:
            QiskitError: When instruction name not found.
        """
        if instruction.name in self._pulse_library:
            waveform = library.Waveform(
                samples=self._pulse_library[instruction.name],
                name=instruction.name,
            )
            channel = self.get_channel(instruction.ch)

            yield instructions.Play(waveform, channel)
        else:
            if qubits := getattr(instruction, "qubits", None):
                msg = f"qubits {qubits}"
            else:
                msg = f"channel {instruction.ch}"
            raise QiskitError(
                f"Instruction {instruction.name} on {msg} is not found "
                "in Qiskit namespace. This instruction cannot be deserialized."
            )

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Basic rescheduling functions which take schedule or instructions and return new schedules."""
from __future__ import annotations
import typing
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Type

import numpy as np

from qiskit.pulse import channels as chans, exceptions, instructions
from qiskit.pulse.channels import ClassicalIOChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.exceptions import UnassignedDurationError
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock, ScheduleComponent

if typing.TYPE_CHECKING:
    from qiskit.pulse.library import Pulse  # pylint: disable=cyclic-import


def block_to_schedule(block: ScheduleBlock) -> Schedule:
    """Convert ``ScheduleBlock`` to ``Schedule``.

    Args:
        block: A ``ScheduleBlock`` to convert.

    Returns:
        Scheduled pulse program.

    Raises:
        UnassignedDurationError: When any instruction duration is not assigned.
        PulseError: When the alignment context duration is shorter than the schedule duration.

    .. note:: This transform may insert barriers in between contexts.
    """
    if not block.is_schedulable():
        raise UnassignedDurationError(
            "All instruction durations should be assigned before creating `Schedule`."
            "Please check `.parameters` to find unassigned parameter objects."
        )

    schedule = Schedule.initialize_from(block)

    for op_data in block.blocks:
        if isinstance(op_data, ScheduleBlock):
            context_schedule = block_to_schedule(op_data)
            if hasattr(op_data.alignment_context, "duration"):
                # context may have local scope duration, e.g. EquispacedAlignment for 1000 dt
                post_buffer = op_data.alignment_context.duration - context_schedule.duration
                if post_buffer < 0:
                    raise PulseError(
                        f"ScheduleBlock {op_data.name} has longer duration than "
                        "the specified context duration "
                        f"{context_schedule.duration} > {op_data.duration}."
                    )
            else:
                post_buffer = 0
            schedule.append(context_schedule, inplace=True)

            # prevent interruption by following instructions.
            # padding with delay instructions is no longer necessary, thanks to alignment context.
            if post_buffer > 0:
                context_boundary = instructions.RelativeBarrier(*op_data.channels)
                schedule.append(context_boundary.shift(post_buffer), inplace=True)
        else:
            schedule.append(op_data, inplace=True)

    # transform with defined policy
    return block.alignment_context.align(schedule)


def compress_pulses(schedules: list[Schedule]) -> list[Schedule]:
    """Optimization pass to replace identical pulses.

    Args:
        schedules: Schedules to compress.

    Returns:
        Compressed schedules.
    """
    existing_pulses: list[Pulse] = []
    new_schedules = []

    for schedule in schedules:
        new_schedule = Schedule.initialize_from(schedule)

        for time, inst in schedule.instructions:
            if isinstance(inst, instructions.Play):
                if inst.pulse in existing_pulses:
                    idx = existing_pulses.index(inst.pulse)
                    identical_pulse = existing_pulses[idx]
                    new_schedule.insert(
                        time,
                        instructions.Play(identical_pulse, inst.channel, inst.name),
                        inplace=True,
                    )
                else:
                    existing_pulses.append(inst.pulse)
                    new_schedule.insert(time, inst, inplace=True)
            else:
                new_schedule.insert(time, inst, inplace=True)

        new_schedules.append(new_schedule)

    return new_schedules


def flatten(program: Schedule) -> Schedule:
    """Flatten (inline) any called nodes into a Schedule tree with no nested children.

    Args:
        program: Pulse program to remove nested structure.

    Returns:
        Flatten pulse program.

    Raises:
        PulseError: When invalid data format is given.
    """
    if isinstance(program, Schedule):
        flat_sched = Schedule.initialize_from(program)
        for time, inst in program.instructions:
            flat_sched.insert(time, inst, inplace=True)
        return flat_sched
    else:
        raise PulseError(f"Invalid input program {program.__class__.__name__} is specified.")


def inline_subroutines(program: Schedule | ScheduleBlock) -> Schedule | ScheduleBlock:
    """Recursively remove call instructions and inline the respective subroutine instructions.

    Assigned parameter values, which are stored in the parameter table, are also applied.
    The subroutine is copied before the parameter assignment to avoid mutation problem.

    Args:
        program: A program which may contain the subroutine, i.e. ``Call`` instruction.

    Returns:
        A schedule without subroutine.

    Raises:
        PulseError: When input program is not valid data format.
    """
    if isinstance(program, Schedule):
        return _inline_schedule(program)
    elif isinstance(program, ScheduleBlock):
        return _inline_block(program)
    else:
        raise PulseError(f"Invalid program {program.__class__.__name__} is specified.")


def _inline_schedule(schedule: Schedule) -> Schedule:
    """A helper function to inline subroutine of schedule.

    .. note:: If subroutine is ``ScheduleBlock`` it is converted into Schedule to get ``t0``.
    """
    ret_schedule = Schedule.initialize_from(schedule)
    for t0, inst in schedule.children:
        # note that schedule.instructions unintentionally flatten the nested schedule.
        # this should be performed by another transformer node.
        if isinstance(inst, Schedule):
            # recursively inline the program
            inline_schedule = _inline_schedule(inst)
            ret_schedule.insert(t0, inline_schedule, inplace=True)
        else:
            ret_schedule.insert(t0, inst, inplace=True)
    return ret_schedule


def _inline_block(block: ScheduleBlock) -> ScheduleBlock:
    """A helper function to inline subroutine of schedule block.

    .. note:: If subroutine is ``Schedule`` the function raises an error.
    """
    ret_block = ScheduleBlock.initialize_from(block)
    for inst in block.blocks:
        if isinstance(inst, ScheduleBlock):
            # recursively inline the program
            inline_block = _inline_block(inst)
            ret_block.append(inline_block, inplace=True)
        else:
            ret_block.append(inst, inplace=True)
    return ret_block


def remove_directives(schedule: Schedule) -> Schedule:
    """Remove directives.

    Args:
        schedule: A schedule to remove compiler directives.

    Returns:
        A schedule without directives.
    """
    return schedule.exclude(instruction_types=[directives.Directive])


def remove_trivial_barriers(schedule: Schedule) -> Schedule:
    """Remove trivial barriers with 0 or 1 channels.

    Args:
        schedule: A schedule to remove trivial barriers.

    Returns:
        schedule: A schedule without trivial barriers
    """

    def filter_func(inst):
        return isinstance(inst[1], directives.RelativeBarrier) and len(inst[1].channels) < 2

    return schedule.exclude(filter_func)


def align_measures(
    schedules: Iterable[ScheduleComponent],
    inst_map: InstructionScheduleMap | None = None,
    cal_gate: str = "u3",
    max_calibration_duration: int | None = None,
    align_time: int | None = None,
    align_all: bool | None = True,
) -> list[Schedule]:
    """Return new schedules where measurements occur at the same physical time.

    This transformation will align the first :class:`.Acquire` on
    every channel to occur at the same time.

    Minimum measurement wait time (to allow for calibration pulses) is enforced
    and may be set with ``max_calibration_duration``.

    By default only instructions containing a :class:`.AcquireChannel` or :class:`.MeasureChannel`
    will be shifted. If you wish to keep the relative timing of all instructions in the schedule set
    ``align_all=True``.

    This method assumes that ``MeasureChannel(i)`` and ``AcquireChannel(i)``
    correspond to the same qubit and the acquire/play instructions
    should be shifted together on these channels.

    .. plot::
       :include-source:
       :nofigs:
       :context: reset

        from qiskit import pulse
        from qiskit.pulse import transforms

        d0 = pulse.DriveChannel(0)
        m0 = pulse.MeasureChannel(0)
        a0 = pulse.AcquireChannel(0)
        mem0 = pulse.MemorySlot(0)

        sched = pulse.Schedule()
        sched.append(pulse.Play(pulse.Constant(10, 0.5), d0), inplace=True)
        sched.append(pulse.Play(pulse.Constant(10, 1.), m0).shift(sched.duration), inplace=True)
        sched.append(pulse.Acquire(20, a0, mem0).shift(sched.duration), inplace=True)

        sched_shifted = sched << 20

        aligned_sched, aligned_sched_shifted = transforms.align_measures([sched, sched_shifted])

        assert aligned_sched == aligned_sched_shifted

    If it is desired to only shift acquisition and measurement stimulus instructions
    set the flag ``align_all=False``:

    .. plot::
       :include-source:
       :nofigs:
       :context:

        aligned_sched, aligned_sched_shifted = transforms.align_measures(
            [sched, sched_shifted],
            align_all=False,
        )

        assert aligned_sched != aligned_sched_shifted


    Args:
        schedules: Collection of schedules to be aligned together
        inst_map: Mapping of circuit operations to pulse schedules
        cal_gate: The name of the gate to inspect for the calibration time
        max_calibration_duration: If provided, inst_map and cal_gate will be ignored
        align_time: If provided, this will be used as final align time.
        align_all: Shift all instructions in the schedule such that they maintain
            their relative alignment with the shifted acquisition instruction.
            If ``False`` only the acquisition and measurement pulse instructions
            will be shifted.
    Returns:
        The input list of schedules transformed to have their measurements aligned.

    Raises:
        PulseError: If the provided alignment time is negative.
    """

    def get_first_acquire_times(schedules):
        """Return a list of first acquire times for each schedule."""
        acquire_times = []
        for schedule in schedules:
            visited_channels = set()
            qubit_first_acquire_times: dict[int, int] = defaultdict(lambda: None)

            for time, inst in schedule.instructions:
                if isinstance(inst, instructions.Acquire) and inst.channel not in visited_channels:
                    visited_channels.add(inst.channel)
                    qubit_first_acquire_times[inst.channel.index] = time

            acquire_times.append(qubit_first_acquire_times)
        return acquire_times

    def get_max_calibration_duration(inst_map, cal_gate):
        """Return the time needed to allow for readout discrimination calibration pulses."""
        # TODO (qiskit-terra #5472): fix behavior of this.
        max_calibration_duration = 0
        for qubits in inst_map.qubits_with_instruction(cal_gate):
            cmd = inst_map.get(cal_gate, qubits, np.pi, 0, np.pi)
            max_calibration_duration = max(cmd.duration, max_calibration_duration)
        return max_calibration_duration

    if align_time is not None and align_time < 0:
        raise exceptions.PulseError("Align time cannot be negative.")

    first_acquire_times = get_first_acquire_times(schedules)
    # Extract the maximum acquire in every schedule across all acquires in the schedule.
    # If there are no acquires in the schedule default to 0.
    max_acquire_times = [max(0, *times.values()) for times in first_acquire_times]
    if align_time is None:
        if max_calibration_duration is None:
            if inst_map:
                max_calibration_duration = get_max_calibration_duration(inst_map, cal_gate)
            else:
                max_calibration_duration = 0
        align_time = max(max_calibration_duration, *max_acquire_times)

    # Shift acquires according to the new scheduled time
    new_schedules = []
    for sched_idx, schedule in enumerate(schedules):
        new_schedule = Schedule.initialize_from(schedule)
        stop_time = schedule.stop_time

        if align_all:
            if first_acquire_times[sched_idx]:
                shift = align_time - max_acquire_times[sched_idx]
            else:
                shift = align_time - stop_time
        else:
            shift = 0

        for time, inst in schedule.instructions:
            measurement_channels = {
                chan.index
                for chan in inst.channels
                if isinstance(chan, (chans.MeasureChannel, chans.AcquireChannel))
            }
            if measurement_channels:
                sched_first_acquire_times = first_acquire_times[sched_idx]
                max_start_time = max(
                    sched_first_acquire_times[chan]
                    for chan in measurement_channels
                    if chan in sched_first_acquire_times
                )
                shift = align_time - max_start_time

            if shift < 0:
                warnings.warn(
                    "The provided alignment time is scheduling an acquire instruction "
                    "earlier than it was scheduled for in the original Schedule. "
                    "This may result in an instruction being scheduled before t=0 and "
                    "an error being raised."
                )
            new_schedule.insert(time + shift, inst, inplace=True)

        new_schedules.append(new_schedule)

    return new_schedules


def add_implicit_acquires(schedule: ScheduleComponent, meas_map: list[list[int]]) -> Schedule:
    """Return a new schedule with implicit acquires from the measurement mapping replaced by
    explicit ones.

    .. warning:: Since new acquires are being added, Memory Slots will be set to match the
                 qubit index. This may overwrite your specification.

    Args:
        schedule: Schedule to be aligned.
        meas_map: List of lists of qubits that are measured together.

    Returns:
        A ``Schedule`` with the additional acquisition instructions.
    """
    new_schedule = Schedule.initialize_from(schedule)
    acquire_map = {}

    for time, inst in schedule.instructions:
        if isinstance(inst, instructions.Acquire):
            if inst.mem_slot and inst.mem_slot.index != inst.channel.index:
                warnings.warn(
                    "One of your acquires was mapped to a memory slot which didn't match"
                    " the qubit index. I'm relabeling them to match."
                )

            # Get the label of all qubits that are measured with the qubit(s) in this instruction
            all_qubits = []
            for sublist in meas_map:
                if inst.channel.index in sublist:
                    all_qubits.extend(sublist)
            # Replace the old acquire instruction by a new one explicitly acquiring all qubits in
            # the measurement group.
            for i in all_qubits:
                explicit_inst = instructions.Acquire(
                    inst.duration,
                    chans.AcquireChannel(i),
                    mem_slot=chans.MemorySlot(i),
                    kernel=inst.kernel,
                    discriminator=inst.discriminator,
                )
                if time not in acquire_map:
                    new_schedule.insert(time, explicit_inst, inplace=True)
                    acquire_map = {time: {i}}
                elif i not in acquire_map[time]:
                    new_schedule.insert(time, explicit_inst, inplace=True)
                    acquire_map[time].add(i)
        else:
            new_schedule.insert(time, inst, inplace=True)

    return new_schedule


def pad(
    schedule: Schedule,
    channels: Iterable[chans.Channel] | None = None,
    until: int | None = None,
    inplace: bool = False,
    pad_with: Type[instructions.Instruction] | None = None,
) -> Schedule:
    """Pad the input Schedule with ``Delay``s on all unoccupied timeslots until
    ``schedule.duration`` or ``until`` if not ``None``.

    Args:
        schedule: Schedule to pad.
        channels: Channels to pad. Defaults to all channels in
            ``schedule`` if not provided. If the supplied channel is not a member
            of ``schedule`` it will be added.
        until: Time to pad until. Defaults to ``schedule.duration`` if not provided.
        inplace: Pad this schedule by mutating rather than returning a new schedule.
        pad_with: Pulse ``Instruction`` subclass to be used for padding.
            Default to :class:`~qiskit.pulse.instructions.Delay` instruction.

    Returns:
        The padded schedule.

    Raises:
        PulseError: When non pulse instruction is set to `pad_with`.
    """
    until = until or schedule.duration
    channels = channels or schedule.channels

    if pad_with:
        if issubclass(pad_with, instructions.Instruction):
            pad_cls = pad_with
        else:
            raise PulseError(
                f"'{pad_with.__class__.__name__}' is not valid pulse instruction to pad with."
            )
    else:
        pad_cls = instructions.Delay

    for channel in channels:
        if isinstance(channel, ClassicalIOChannel):
            continue

        if channel not in schedule.channels:
            schedule = schedule.insert(0, instructions.Delay(until, channel), inplace=inplace)
            continue

        prev_time = 0
        timeslots = iter(schedule.timeslots[channel])
        to_pad = []
        while prev_time < until:
            try:
                t0, t1 = next(timeslots)
            except StopIteration:
                to_pad.append((prev_time, until - prev_time))
                break
            if prev_time < t0:
                to_pad.append((prev_time, min(t0, until) - prev_time))
            prev_time = t1
        for t0, duration in to_pad:
            schedule = schedule.insert(t0, pad_cls(duration, channel), inplace=inplace)

    return schedule

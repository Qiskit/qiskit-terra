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

"""Module for common pulse programming macros."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from qiskit.pulse import channels, exceptions, instructions, utils
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.schedule import Schedule
from qiskit.providers.backend import BackendV2


if TYPE_CHECKING:
    from qiskit.transpiler import Target


def measure(
    qubits: Sequence[int],
    backend=None,
    inst_map: InstructionScheduleMap | None = None,
    meas_map: list[list[int]] | dict[int, list[int]] | None = None,
    qubit_mem_slots: dict[int, int] | None = None,
    measure_name: str = "measure",
) -> Schedule:
    """Return a schedule which measures the requested qubits according to the given
    instruction mapping and measure map, or by using the defaults provided by the backend.

    .. note::
        This function internally dispatches schedule generation logic depending on input backend model.
        For the :class:`.BackendV1`, it considers conventional :class:`.InstructionScheduleMap`
        and utilizes the backend calibration defined for a group of qubits in the `meas_map`.
        For the :class:`.BackendV2`, it assembles calibrations of single qubit measurement
        defined in the backend target to build a composite measurement schedule for `qubits`.

    By default, the measurement results for each qubit are trivially mapped to the qubit
    index. This behavior is overridden by qubit_mem_slots. For instance, to measure
    qubit 0 into MemorySlot(1), qubit_mem_slots can be provided as {0: 1}.

    Args:
        qubits: List of qubits to be measured.
        backend (Union[Backend, BaseBackend]): A backend instance, which contains
            hardware-specific data required for scheduling.
        inst_map: Mapping of circuit operations to pulse schedules. If None, defaults to the
                  ``instruction_schedule_map`` of ``backend``.
        meas_map: List of sets of qubits that must be measured together. If None, defaults to
                  the ``meas_map`` of ``backend``.
        qubit_mem_slots: Mapping of measured qubit index to classical bit index.
        measure_name: Name of the measurement schedule.

    Returns:
        A measurement schedule corresponding to the inputs provided.
    """

    # backend is V2.
    if isinstance(backend, BackendV2):

        return _measure_v2(
            qubits=qubits,
            target=backend.target,
            meas_map=meas_map or backend.meas_map,
            qubit_mem_slots=qubit_mem_slots or dict(zip(qubits, range(len(qubits)))),
            measure_name=measure_name,
        )
    # backend is V1 or backend is None.
    else:
        try:
            return _measure_v1(
                qubits=qubits,
                inst_map=inst_map or backend.defaults().instruction_schedule_map,
                meas_map=meas_map or backend.configuration().meas_map,
                qubit_mem_slots=qubit_mem_slots,
                measure_name=measure_name,
            )
        except AttributeError as ex:
            raise exceptions.PulseError(
                "inst_map or meas_map, and backend cannot be None simultaneously"
            ) from ex


def _measure_v1(
    qubits: Sequence[int],
    inst_map: InstructionScheduleMap,
    meas_map: list[list[int]] | dict[int, list[int]],
    qubit_mem_slots: dict[int, int] | None = None,
    measure_name: str = "measure",
) -> Schedule:
    """Return a schedule which measures the requested qubits according to the given
    instruction mapping and measure map, or by using the defaults provided by the backendV1.

    Args:
        qubits: List of qubits to be measured.
        backend (Union[Backend, BaseBackend]): A backend instance, which contains
            hardware-specific data required for scheduling.
        inst_map: Mapping of circuit operations to pulse schedules. If None, defaults to the
                  ``instruction_schedule_map`` of ``backend``.
        meas_map: List of sets of qubits that must be measured together. If None, defaults to
                  the ``meas_map`` of ``backend``.
        qubit_mem_slots: Mapping of measured qubit index to classical bit index.
        measure_name: Name of the measurement schedule.
    Returns:
        A measurement schedule corresponding to the inputs provided.
    Raises:
        PulseError: If both ``inst_map`` or ``meas_map``, and ``backend`` is None.
    """

    schedule = Schedule(name=f"Default measurement schedule for qubits {qubits}")

    if isinstance(meas_map, list):
        meas_map = utils.format_meas_map(meas_map)

    measure_groups = set()
    for qubit in qubits:
        measure_groups.add(tuple(meas_map[qubit]))
    for measure_group_qubits in measure_groups:

        unused_mem_slots = (
            set()
            if qubit_mem_slots is None
            else set(measure_group_qubits) - set(qubit_mem_slots.values())
        )

        try:
            default_sched = inst_map.get(measure_name, measure_group_qubits)
        except exceptions.PulseError as ex:
            raise exceptions.PulseError(
                f"We could not find a default measurement schedule called '{measure_name}'. "
                "Please provide another name using the 'measure_name' keyword "
                "argument. For assistance, the instructions which are defined are: "
                f"{inst_map.instructions}"
            ) from ex
        for time, inst in default_sched.instructions:
            if inst.channel.index not in qubits:
                continue
            if qubit_mem_slots and isinstance(inst, instructions.Acquire):
                if inst.channel.index in qubit_mem_slots:
                    mem_slot = channels.MemorySlot(qubit_mem_slots[inst.channel.index])
                else:
                    mem_slot = channels.MemorySlot(unused_mem_slots.pop())
                inst = instructions.Acquire(inst.duration, inst.channel, mem_slot=mem_slot)
            # Measurement pulses should only be added if its qubit was measured by the user
            schedule = schedule.insert(time, inst)

    return schedule


def _measure_v2(
    qubits: Sequence[int],
    target: Target,
    meas_map: list[list[int]] | dict[int, list[int]],
    qubit_mem_slots: dict[int, int],
    measure_name: str = "measure",
) -> Schedule:
    """Return a schedule which measures the requested qubits according to the given
    target and measure map, or by using the defaults provided by the backendV2.

    Args:
        qubits: List of qubits to be measured.
        target: The :class:`~.Target` representing the target backend.
        meas_map: List of sets of qubits that must be measured together.
        qubit_mem_slots: Mapping of measured qubit index to classical bit index.
        measure_name: Name of the measurement schedule.

    Returns:
        A measurement schedule corresponding to the inputs provided.
    """
    schedule = Schedule(name=f"Default measurement schedule for qubits {qubits}")

    if isinstance(meas_map, list):
        meas_map = utils.format_meas_map(meas_map)
    meas_group = set()
    for qubit in qubits:
        meas_group |= set(meas_map[qubit])
    meas_group = sorted(meas_group)

    meas_group_set = set(range(max(meas_group) + 1))
    unassigned_qubit_indices = sorted(set(meas_group) - qubit_mem_slots.keys())
    unassigned_reg_indices = sorted(meas_group_set - set(qubit_mem_slots.values()), reverse=True)
    if set(qubit_mem_slots.values()).issubset(meas_group_set):
        for qubit in unassigned_qubit_indices:
            qubit_mem_slots[qubit] = unassigned_reg_indices.pop()

    for measure_qubit in meas_group:
        try:
            if measure_qubit in qubits:
                default_sched = target._get_calibration(measure_name, (measure_qubit,)).filter(
                    channels=[
                        channels.MeasureChannel(measure_qubit),
                        channels.AcquireChannel(measure_qubit),
                    ]
                )
                schedule += _schedule_remapping_memory_slot(default_sched, qubit_mem_slots)
        except KeyError as ex:
            raise exceptions.PulseError(
                f"We could not find a default measurement schedule called '{measure_name}'. "
                "Please provide another name using the 'measure_name' keyword "
                "argument. For assistance, the instructions which are defined are: "
                f"{target.instructions}"
            ) from ex
    return schedule


def measure_all(backend) -> Schedule:
    """
    Return a Schedule which measures all qubits of the given backend.

    Args:
        backend (Union[Backend, BaseBackend]): A backend instance, which contains
            hardware-specific data required for scheduling.

    Returns:
        A schedule corresponding to the inputs provided.
    """
    # backend is V2.
    if isinstance(backend, BackendV2):
        qubits = list(range(backend.num_qubits))
    else:
        qubits = list(range(backend.configuration().n_qubits))
    return measure(qubits=qubits, backend=backend)


def _schedule_remapping_memory_slot(
    schedule: Schedule, qubit_mem_slots: dict[int, int]
) -> Schedule:
    """
    A helper function to overwrite MemorySlot index of :class:`.Acquire` instruction.

    Args:
        schedule: A measurement schedule.
        qubit_mem_slots: Mapping of measured qubit index to classical bit index.

    Returns:
        A measurement schedule with new memory slot index.
    """
    new_schedule = Schedule()
    for t0, inst in schedule.instructions:
        if isinstance(inst, instructions.Acquire):
            qubit_index = inst.channel.index
            reg_index = qubit_mem_slots.get(qubit_index, qubit_index)
            new_schedule.insert(
                t0,
                instructions.Acquire(
                    inst.duration,
                    channels.AcquireChannel(qubit_index),
                    mem_slot=channels.MemorySlot(reg_index),
                ),
                inplace=True,
            )
        else:
            new_schedule.insert(t0, inst, inplace=True)
    return new_schedule

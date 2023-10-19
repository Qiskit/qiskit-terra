# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Lower gates to schedules. The relative timing within gates is respected. This
module handles the translation, but does not handle timing.
"""
from collections import namedtuple
from typing import Dict, List

from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.circuit.duration import convert_durations_to_dt
from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse import Schedule
from qiskit.pulse import instructions as pulse_inst
from qiskit.pulse.channels import AcquireChannel, MemorySlot, DriveChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.macros import measure
from qiskit.scheduler.config import ScheduleConfig

CircuitPulseDef = namedtuple(
    "CircuitPulseDef",
    ["schedule", "qubits"],  # The schedule which implements the quantum circuit command
)  # The labels of the qubits involved in the command according to the circuit


def lower_gates(circuit: QuantumCircuit, schedule_config: ScheduleConfig) -> List[CircuitPulseDef]:
    """
    Return a list of Schedules and the qubits they operate on, for each element encountered in the
    input circuit.

    Without concern for the final schedule, extract and return a list of Schedules and the qubits
    they operate on, for each element encountered in the input circuit. Measures are grouped when
    possible, so ``qc.measure(q0, c0)`` or ``qc.measure(q1, c1)`` will generate a synchronous
    measurement pulse.

    Args:
        circuit: The quantum circuit to translate.
        schedule_config: Backend specific parameters used for building the Schedule.

    Returns:
        A list of CircuitPulseDefs: the pulse definition for each circuit element.

    Raises:
        QiskitError: If circuit uses a command that isn't defined in config.inst_map.
    """
    from qiskit.pulse.transforms.base_transforms import target_qobj_transform

    circ_pulse_defs = []

    inst_map = schedule_config.inst_map
    qubit_mem_slots = {}  # Map measured qubit index to classical bit index

    # convert the unit of durations from SI to dt before lowering
    circuit = convert_durations_to_dt(circuit, dt_in_sec=schedule_config.dt, inplace=False)

    def get_measure_schedule(qubit_mem_slots: Dict[int, int]) -> CircuitPulseDef:
        """Create a schedule to measure the qubits queued for measuring."""
        sched = Schedule()
        # Exclude acquisition on these qubits, since they are handled by the user calibrations
        acquire_excludes = {}
        if Measure().name in circuit.calibrations.keys():
            qubits = tuple(sorted(qubit_mem_slots.keys()))
            params = ()
            for qubit in qubits:
                try:
                    meas_q = circuit.calibrations[Measure().name][((qubit,), params)]
                    meas_q = target_qobj_transform(meas_q)
                    acquire_q = meas_q.filter(channels=[AcquireChannel(qubit)])
                    mem_slot_index = [
                        chan.index for chan in acquire_q.channels if isinstance(chan, MemorySlot)
                    ][0]
                    if mem_slot_index != qubit_mem_slots[qubit]:
                        raise KeyError(
                            "The measurement calibration is not defined on "
                            "the requested classical bits"
                        )
                    sched |= meas_q
                    del qubit_mem_slots[qubit]
                    acquire_excludes[qubit] = mem_slot_index
                except KeyError:
                    pass

        if qubit_mem_slots:
            qubits = list(qubit_mem_slots.keys())
            qubit_mem_slots.update(acquire_excludes)
            meas_sched = measure(
                qubits=qubits,
                backend=schedule_config.backend,
                inst_map=inst_map,
                meas_map=schedule_config.meas_map,
                qubit_mem_slots=qubit_mem_slots,
            )
            meas_sched = target_qobj_transform(meas_sched)
            meas_sched = meas_sched.exclude(
                channels=[AcquireChannel(qubit) for qubit in acquire_excludes]
            )
            sched |= meas_sched
        qubit_mem_slots.clear()
        return CircuitPulseDef(
            schedule=sched,
            qubits=[chan.index for chan in sched.channels if isinstance(chan, AcquireChannel)],
        )

    qubit_indices = {bit: idx for idx, bit in enumerate(circuit.qubits)}
    clbit_indices = {bit: idx for idx, bit in enumerate(circuit.clbits)}

    for instruction in circuit.data:
        inst_qubits = [qubit_indices[qubit] for qubit in instruction.qubits]

        if any(q in qubit_mem_slots for q in inst_qubits):
            # If we are operating on a qubit that was scheduled to be measured, process that first
            circ_pulse_defs.append(get_measure_schedule(qubit_mem_slots))

        if isinstance(instruction.operation, Barrier):
            circ_pulse_defs.append(
                CircuitPulseDef(schedule=instruction.operation, qubits=inst_qubits)
            )
        elif isinstance(instruction.operation, Delay):
            sched = Schedule(name=instruction.operation.name)
            for qubit in inst_qubits:
                for channel in [DriveChannel]:
                    sched += pulse_inst.Delay(
                        duration=instruction.operation.duration, channel=channel(qubit)
                    )
            circ_pulse_defs.append(CircuitPulseDef(schedule=sched, qubits=inst_qubits))
        elif isinstance(instruction.operation, Measure):
            if len(inst_qubits) != 1 and len(instruction.clbits) != 1:
                raise QiskitError(
                    "Qubit '{}' or classical bit '{}' errored because the "
                    "circuit Measure instruction only takes one of "
                    "each.".format(inst_qubits, instruction.clbits)
                )
            qubit_mem_slots[inst_qubits[0]] = clbit_indices[instruction.clbits[0]]
        else:
            try:
                gate_cals = circuit.calibrations[instruction.operation.name]
                schedule = gate_cals[
                    (
                        tuple(inst_qubits),
                        tuple(
                            p if getattr(p, "parameters", None) else float(p)
                            for p in instruction.operation.params
                        ),
                    )
                ]
                schedule = target_qobj_transform(schedule)
                circ_pulse_defs.append(CircuitPulseDef(schedule=schedule, qubits=inst_qubits))
                continue
            except KeyError:
                pass  # Calibration not defined for this operation

            try:
                schedule = inst_map.get(
                    instruction.operation, inst_qubits, *instruction.operation.params
                )
                schedule = target_qobj_transform(schedule)
                circ_pulse_defs.append(CircuitPulseDef(schedule=schedule, qubits=inst_qubits))
            except PulseError as ex:
                raise QiskitError(
                    f"Operation '{instruction.operation.name}' on qubit(s) {inst_qubits} "
                    "not supported by the backend command definition. Did you remember to "
                    "transpile your input circuit for the same backend?"
                ) from ex

    if qubit_mem_slots:
        circ_pulse_defs.append(get_measure_schedule(qubit_mem_slots))

    return circ_pulse_defs

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

"""
Mapping a scheduled QuantumCircuit to a pulse Schedule.
"""
from collections import defaultdict

from typing import Optional, Union
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse.schedule import Schedule
from qiskit.pulse.transforms import pad
from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.lowering import lower_gates
from qiskit.providers import BackendV1, BackendV2
from qiskit.utils.deprecate_pulse import deprecate_pulse_dependency


@deprecate_pulse_dependency(moving_to_dynamics=True)
def sequence(
    scheduled_circuit: QuantumCircuit,
    schedule_config: ScheduleConfig,
    backend: Optional[Union[BackendV1, BackendV2]] = None,
) -> Schedule:
    """
    Return the pulse Schedule which implements the input scheduled circuit.

    Assume all measurements are done at once at the last of the circuit.
    Schedules according to the command definition given by the schedule_config.

    Args:
        scheduled_circuit: The scheduled quantum circuit to translate.
        schedule_config: Backend specific parameters used for building the Schedule.
        backend: A backend used to build the Schedule, the backend could be BackendV1
                 or BackendV2

    Returns:
        A schedule corresponding to the input ``circuit``.

    Raises:
        QiskitError: If invalid scheduled circuit is supplied.
    """
    circ_pulse_defs = lower_gates(scheduled_circuit, schedule_config, backend)

    # find the measurement start time (assume measurement once)
    def _meas_start_time():
        _qubit_time_available = defaultdict(int)
        for instruction in scheduled_circuit.data:
            if isinstance(instruction.operation, Measure):
                return _qubit_time_available[instruction.qubits[0]]
            for q in instruction.qubits:
                _qubit_time_available[q] += instruction.operation.duration
        return None

    meas_time = _meas_start_time()

    # restore start times
    qubit_time_available = {}
    start_times = []
    out_circ_pulse_defs = []
    for circ_pulse_def in circ_pulse_defs:
        active_qubits = [q for q in circ_pulse_def.qubits if q in qubit_time_available]

        start_time = max((qubit_time_available[q] for q in active_qubits), default=0)

        for q in active_qubits:
            if qubit_time_available[q] != start_time:
                # print(q, ":", qubit_time_available[q], "!=", start_time)
                raise QiskitError("Invalid scheduled circuit.")

        stop_time = start_time
        if not isinstance(circ_pulse_def.schedule, Barrier):
            stop_time += circ_pulse_def.schedule.duration

        delay_overlaps_meas = False
        for q in circ_pulse_def.qubits:
            qubit_time_available[q] = stop_time
            if (
                meas_time is not None
                and circ_pulse_def.schedule.name == "delay"
                and stop_time > meas_time
            ):
                qubit_time_available[q] = meas_time
                delay_overlaps_meas = True
        # skip to delays overlapping measures and barriers
        if not delay_overlaps_meas and not isinstance(circ_pulse_def.schedule, Barrier):
            start_times.append(start_time)
            out_circ_pulse_defs.append(circ_pulse_def)

    timed_schedules = [(time, cpd.schedule) for time, cpd in zip(start_times, out_circ_pulse_defs)]
    sched = Schedule(*timed_schedules, name=scheduled_circuit.name)
    return pad(sched)

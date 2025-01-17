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
Mapping a scheduled ``QuantumCircuit`` to a pulse ``Schedule``.
"""
from typing import List, Optional, Union

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.pulse import InstructionScheduleMap, Schedule
from qiskit.scheduler import ScheduleConfig
from qiskit.scheduler.sequence import sequence as _sequence
from qiskit.utils.deprecate_pulse import deprecate_pulse_dependency


@deprecate_pulse_dependency(moving_to_dynamics=True)
def sequence(
    scheduled_circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Optional[Backend] = None,
    inst_map: Optional[InstructionScheduleMap] = None,
    meas_map: Optional[List[List[int]]] = None,
    dt: Optional[float] = None,
) -> Union[Schedule, List[Schedule]]:
    """
    Schedule a scheduled circuit to a pulse ``Schedule``, using the backend.

    Args:
        scheduled_circuits: Scheduled circuit(s) to be translated
        backend: A backend instance, which contains hardware-specific data required for scheduling
        inst_map: Mapping of circuit operations to pulse schedules. If ``None``, defaults to the
                  ``backend``\'s ``instruction_schedule_map``
        meas_map: List of sets of qubits that must be measured together. If ``None``, defaults to
                  the ``backend``\'s ``meas_map``
        dt: The output sample rate of backend control electronics. For scheduled circuits
            which contain time information, dt is required. If not provided, it will be
            obtained from the backend configuration

    Returns:
        A pulse ``Schedule`` that implements the input circuit

    Raises:
        QiskitError: If ``inst_map`` and ``meas_map`` are not passed and ``backend`` is not passed
    """
    if inst_map is None:
        if backend is None:
            raise QiskitError("Must supply either a backend or inst_map for sequencing.")
        inst_map = backend.defaults().instruction_schedule_map
    if meas_map is None:
        if backend is None:
            raise QiskitError("Must supply either a backend or a meas_map for sequencing.")
        meas_map = backend.configuration().meas_map
    if dt is None:
        if backend is None:
            raise QiskitError("Must supply either a backend or a dt for sequencing.")
        dt = backend.configuration().dt

    schedule_config = ScheduleConfig(inst_map=inst_map, meas_map=meas_map, dt=dt)
    circuits = scheduled_circuits if isinstance(scheduled_circuits, list) else [scheduled_circuits]
    schedules = [_sequence(circuit, schedule_config) for circuit in circuits]
    return schedules[0] if len(schedules) == 1 else schedules

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""QuantumCircuit to Pulse scheduler."""
from typing import Optional, Union

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError

from qiskit.pulse.schedule import Schedule
from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.methods import as_soon_as_possible, as_late_as_possible
from qiskit.providers import BackendV1, BackendV2
from qiskit.utils.deprecate_pulse import deprecate_pulse_dependency


@deprecate_pulse_dependency(moving_to_dynamics=True)
def schedule_circuit(
    circuit: QuantumCircuit,
    schedule_config: ScheduleConfig,
    method: Optional[str] = None,
    backend: Optional[Union[BackendV1, BackendV2]] = None,
) -> Schedule:
    """
    Basic scheduling pass from a circuit to a pulse Schedule, using the backend. If no method is
    specified, then a basic, as late as possible scheduling pass is performed, i.e. pulses are
    scheduled to occur as late as possible.

    Supported methods:

        * ``'as_soon_as_possible'``: Schedule pulses greedily, as early as possible on a
          qubit resource. (alias: ``'asap'``)
        * ``'as_late_as_possible'``: Schedule pulses late-- keep qubits in the ground state when
          possible. (alias: ``'alap'``)

    Args:
        circuit: The quantum circuit to translate.
        schedule_config: Backend specific parameters used for building the Schedule.
        method: The scheduling pass method to use.
        backend: A backend used to build the Schedule, the backend could be BackendV1
                 or BackendV2.

    Returns:
        Schedule corresponding to the input circuit.

    Raises:
        QiskitError: If method isn't recognized.
    """
    methods = {
        "as_soon_as_possible": as_soon_as_possible,
        "asap": as_soon_as_possible,
        "as_late_as_possible": as_late_as_possible,
        "alap": as_late_as_possible,
    }
    if method is None:
        method = "as_late_as_possible"
    try:
        return methods[method](circuit, schedule_config, backend)
    except KeyError as ex:
        raise QiskitError(f"Scheduling method {method} isn't recognized.") from ex

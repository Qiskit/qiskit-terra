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

"""Scheduling container classes."""

from typing import List

from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.utils import format_meas_map
from qiskit.utils.deprecate_pulse import deprecate_pulse_dependency


class ScheduleConfig:
    """Configuration for pulse scheduling."""

    @deprecate_pulse_dependency(moving_to_dynamics=True)
    def __init__(self, inst_map: InstructionScheduleMap, meas_map: List[List[int]], dt: float):
        """
        Container for information needed to schedule a QuantumCircuit into a pulse Schedule.

        Args:
            inst_map: The schedule definition of all gates supported on a backend.
            meas_map: A list of groups of qubits which have to be measured together.
            dt: Sample duration.
        """
        self.inst_map = inst_map
        self.meas_map = format_meas_map(meas_map)
        self.dt = dt

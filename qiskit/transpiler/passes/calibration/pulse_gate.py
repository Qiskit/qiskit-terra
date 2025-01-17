# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Instruction schedule map reference pass."""

from typing import List, Union

from qiskit.circuit import Instruction as CircuitInst
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.transpiler.target import Target
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils.deprecate_pulse import deprecate_pulse_dependency

from .base_builder import CalibrationBuilder


class PulseGates(CalibrationBuilder):
    """Pulse gate adding pass.

    This pass adds gate calibrations from the supplied ``InstructionScheduleMap``
    to a quantum circuit.

    This pass checks each DAG circuit node and acquires a corresponding schedule from
    the instruction schedule map object that may be provided by the target backend.
    Because this map is a mutable object, the end-user can provide a configured backend to
    execute the circuit with customized gate implementations.

    This mapping object returns a schedule with "publisher" metadata which is an integer Enum
    value representing who created the gate schedule.
    If the gate schedule is provided by end-users, this pass attaches the schedule to
    the DAG circuit as a calibration.

    This pass allows users to easily override quantum circuit with custom gate definitions
    without directly dealing with those schedules.

    References
        * [1] OpenQASM 3: A broader and deeper quantum assembly language
          https://arxiv.org/abs/2104.14722
    """

    @deprecate_pulse_dependency
    def __init__(
        self,
        inst_map: InstructionScheduleMap = None,
        target: Target = None,
    ):
        """Create new pass.

        Args:
            inst_map: Instruction schedule map that user may override.
            target: The :class:`~.Target` representing the target backend, if both
                ``inst_map`` and ``target`` are specified then it updates instructions
                in the ``target`` with ``inst_map``.
        """
        super().__init__()

        if inst_map is None and target is None:
            raise TranspilerError("inst_map and target cannot be None simulataneously.")

        if target is None:
            target = Target()
            target.update_from_instruction_schedule_map(inst_map)
        self.target = target

    def supported(self, node_op: CircuitInst, qubits: List) -> bool:
        """Determine if a given node supports the calibration.

        Args:
            node_op: Target instruction object.
            qubits: Integer qubit indices to check.

        Returns:
            Return ``True`` is calibration can be provided.
        """
        return self.target._has_calibration(node_op.name, tuple(qubits))

    def get_calibration(self, node_op: CircuitInst, qubits: List) -> Union[Schedule, ScheduleBlock]:
        """Gets the calibrated schedule for the given instruction and qubits.

        Args:
            node_op: Target instruction object.
            qubits: Integer qubit indices to check.

        Returns:
            Return Schedule of target gate instruction.

        Raises:
            TranspilerError: When node is parameterized and calibration is raw schedule object.
        """
        return self.target._get_calibration(node_op.name, tuple(qubits), *node_op.params)

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unify time unit in circuit for scheduling and following passes."""
from typing import Set

from qiskit.circuit import Delay
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.target import Target


class TimeUnitConversion(TransformationPass):
    """Choose a time unit to be used in the following time-aware passes,
    and make all circuit time units consistent with that.

    This pass will add a :attr:`.Instruction.duration` metadata to each op whose duration is known
    which will be used by subsequent scheduling passes for scheduling.

    If ``dt`` (in seconds) is known to transpiler, the unit ``'dt'`` is chosen. Otherwise,
    the unit to be selected depends on what units are used in delays and instruction durations:

    * ``'s'``: if they are all in SI units.
    * ``'dt'``: if they are all in the unit ``'dt'``.
    * raise error: if they are a mix of SI units and ``'dt'``.
    """

    def __init__(self, inst_durations: InstructionDurations = None, target: Target = None):
        """TimeUnitAnalysis initializer.

        Args:
            inst_durations (InstructionDurations): A dictionary of durations of instructions.
            target: The :class:`~.Target` representing the target backend, if both
                  ``inst_durations`` and ``target`` are specified then this argument will take
                  precedence and ``inst_durations`` will be ignored.


        """
        super().__init__()
        self.inst_durations = inst_durations or InstructionDurations()
        if target is not None:
            self.inst_durations = target.durations()
        self._durations_provided = inst_durations is not None or target is not None

    def run(self, dag: DAGCircuit):
        """Run the TimeUnitAnalysis pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to be checked.

        Returns:
            DAGCircuit: DAG with consistent timing and op nodes annotated with duration.

        Raises:
            TranspilerError: if the units are not unifiable
        """

        inst_durations = self._update_inst_durations(dag)

        # Choose unit
        if inst_durations.dt is not None:
            time_unit = "dt"
        else:
            # Check what units are used in delays and other instructions: dt or SI or mixed
            units_delay = self._units_used_in_delays(dag)
            if self._unified(units_delay) == "mixed":
                raise TranspilerError(
                    "Fail to unify time units in delays. SI units "
                    "and dt unit must not be mixed when dt is not supplied."
                )
            units_other = inst_durations.units_used()
            if self._unified(units_other) == "mixed":
                raise TranspilerError(
                    "Fail to unify time units in instruction_durations. SI units "
                    "and dt unit must not be mixed when dt is not supplied."
                )

            unified_unit = self._unified(units_delay | units_other)
            if unified_unit == "SI":
                time_unit = "s"
            elif unified_unit == "dt":
                time_unit = "dt"
            else:
                raise TranspilerError(
                    "Fail to unify time units. SI units "
                    "and dt unit must not be mixed when dt is not supplied."
                )

        # Make units consistent
        for node in dag.op_nodes():
            try:
                duration = inst_durations.get(
                    node.op, [dag.find_bit(qarg).index for qarg in node.qargs], unit=time_unit
                )
            except TranspilerError:
                continue
            op = node.op.to_mutable()
            op.duration = duration
            op.unit = time_unit
            dag.substitute_node(node, op, propagate_condition=False)

        self.property_set["time_unit"] = time_unit
        return dag

    def _update_inst_durations(self, dag):
        """Update instruction durations with circuit information. If the dag contains gate
        calibrations and no instruction durations were provided through the target or as a
        standalone input, the circuit calibration durations will be used.
        The priority order for instruction durations is: target > standalone > circuit.
        """
        circ_durations = InstructionDurations()

        if dag.calibrations:
            cal_durations = []
            for gate, gate_cals in dag.calibrations.items():
                for (qubits, parameters), schedule in gate_cals.items():
                    cal_durations.append((gate, qubits, parameters, schedule.duration))
            circ_durations.update(cal_durations, circ_durations.dt)

        if self._durations_provided:
            circ_durations.update(self.inst_durations, getattr(self.inst_durations, "dt", None))

        return circ_durations

    @staticmethod
    def _units_used_in_delays(dag: DAGCircuit) -> Set[str]:
        units_used = set()
        for node in dag.op_nodes(op=Delay):
            units_used.add(node.op.unit)
        return units_used

    @staticmethod
    def _unified(unit_set: Set[str]) -> str:
        if not unit_set:
            return "dt"

        if len(unit_set) == 1 and "dt" in unit_set:
            return "dt"

        all_si = True
        for unit in unit_set:
            if not unit.endswith("s"):
                all_si = False
                break

        if all_si:
            return "SI"

        return "mixed"

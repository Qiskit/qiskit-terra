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

"""Base circuit scheduling pass."""

import warnings

from qiskit.circuit import Delay, Gate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes.scheduling.time_unit_conversion import TimeUnitConversion
from qiskit.transpiler.target import Target


class BaseScheduler(AnalysisPass):
    """Base scheduler pass."""

    CONDITIONAL_SUPPORTED = (Gate, Delay)

    def __init__(self, durations: InstructionDurations = None, target: Target = None):
        """Scheduler initializer.

        Args:
            durations: Durations of instructions to be used in scheduling
            target: The :class:`~.Target` representing the target backend, if both
                  ``durations`` and this are specified then this argument will take
                  precedence and ``durations`` will be ignored.

        """
        super().__init__()
        self.durations = durations
        if target is not None:
            self.durations = target.durations()

        # Ensure op node durations are attached and in consistent unit
        self.requires.append(TimeUnitConversion(inst_durations=durations, target=target))

        # Initialize timeslot
        if "node_start_time" in self.property_set:
            warnings.warn(
                "This circuit has been already scheduled. "
                "The output of previous scheduling pass will be overridden.",
                UserWarning,
            )
        self.property_set["node_start_time"] = {}

    @staticmethod
    def _get_node_duration(
        node: DAGOpNode,
        dag: DAGCircuit,
    ) -> int:
        """A helper method to get duration from node or calibration."""
        indices = [dag.find_bit(qarg).index for qarg in node.qargs]

        if dag.has_calibration_for(node):
            # If node has calibration, this value should be the highest priority
            cal_key = tuple(indices), tuple(float(p) for p in node.op.params)
            duration = dag.calibrations[node.op.name][cal_key].duration

            # Note that node duration is updated (but this is analysis pass)
            op = node.op.to_mutable()
            op.duration = duration
            dag.substitute_node(node, op, propagate_condition=False)
        else:
            duration = node.duration

        if isinstance(duration, ParameterExpression):
            raise TranspilerError(
                f"Parameterized duration ({duration}) "
                f"of {node.op.name} on qubits {indices} is not bounded."
            )
        if duration is None:
            raise TranspilerError(f"Duration of {node.op.name} on qubits {indices} is not found.")

        return duration

    def run(self, dag: DAGCircuit):
        raise NotImplementedError

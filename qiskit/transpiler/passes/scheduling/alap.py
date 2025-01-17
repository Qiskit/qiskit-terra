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

"""ALAP Scheduling."""

from qiskit.circuit import Delay, Qubit, Measure
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.scheduling.base_scheduler import BaseSchedulerTransform
from qiskit.utils.deprecation import deprecate_func


class ALAPSchedule(BaseSchedulerTransform):
    """ALAP Scheduling pass, which schedules the **stop** time of instructions as late as possible.

    See :class:`~qiskit.transpiler.passes.scheduling.base_scheduler.BaseSchedulerTransform` for the
    detailed behavior of the control flow operation, i.e. ``c_if``.
    """

    @deprecate_func(
        additional_msg=(
            "Instead, use :class:`~.ALAPScheduleAnalysis`, which is an "
            "analysis pass that requires a padding pass to later modify the circuit."
        ),
        since="1.1.0",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, dag):
        """Run the ALAPSchedule pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to schedule.

        Returns:
            DAGCircuit: A scheduled DAG.

        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
            TranspilerError: if conditional bit is added to non-supported instruction.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("ALAP schedule runs on physical circuits only")

        time_unit = self.property_set["time_unit"]
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        idle_before = {q: 0 for q in dag.qubits + dag.clbits}
        for node in reversed(list(dag.topological_op_nodes())):
            op_duration = self._get_node_duration(node, dag)

            # compute t0, t1: instruction interval, note that
            # t0: start time of instruction
            # t1: end time of instruction

            # since this is alap scheduling, node is scheduled in reversed topological ordering
            # and nodes are packed from the very end of the circuit.
            # the physical meaning of t0 and t1 is flipped here.
            if isinstance(node.op, self.CONDITIONAL_SUPPORTED):
                t0q = max(idle_before[q] for q in node.qargs)
                if node.op.condition_bits:
                    # conditional is bit tricky due to conditional_latency
                    t0c = max(idle_before[c] for c in node.op.condition_bits)
                    # Assume following case (t0c > t0q):
                    #
                    #                |t0q
                    # Q ░░░░░░░░░░░░░▒▒▒
                    # C ░░░░░░░░▒▒▒▒▒▒▒▒
                    #           |t0c
                    #
                    # In this case, there is no actual clbit read before gate.
                    #
                    #             |t0q' = t0c - conditional_latency
                    # Q ░░░░░░░░▒▒▒░░▒▒▒
                    # C ░░░░░░▒▒▒▒▒▒▒▒▒▒
                    #         |t1c' = t0c + conditional_latency
                    #
                    # rather than naively doing
                    #
                    #        |t1q' = t0c + duration
                    # Q ░░░░░▒▒▒░░░░░▒▒▒
                    # C ░░▒▒░░░░▒▒▒▒▒▒▒▒
                    #     |t1c' = t0c + duration + conditional_latency
                    #
                    t0 = max(t0q, t0c - op_duration)
                    t1 = t0 + op_duration
                    for clbit in node.op.condition_bits:
                        idle_before[clbit] = t1 + self.conditional_latency
                else:
                    t0 = t0q
                    t1 = t0 + op_duration
            else:
                if node.op.condition_bits:
                    raise TranspilerError(
                        f"Conditional instruction {node.op.name} is not supported in ALAP scheduler."
                    )

                if isinstance(node.op, Measure):
                    # clbit time is always right (alap) justified
                    t0 = max(idle_before[bit] for bit in node.qargs + node.cargs)
                    t1 = t0 + op_duration
                    #
                    #        |t1 = t0 + duration
                    # Q ░░░░░▒▒▒▒▒▒▒▒▒▒▒
                    # C ░░░░░░░░░▒▒▒▒▒▒▒
                    #            |t0 + (duration - clbit_write_latency)
                    #
                    for clbit in node.cargs:
                        idle_before[clbit] = t0 + (op_duration - self.clbit_write_latency)
                else:
                    # It happens to be directives such as barrier
                    t0 = max(idle_before[bit] for bit in node.qargs + node.cargs)
                    t1 = t0 + op_duration

            for bit in node.qargs:
                delta = t0 - idle_before[bit]
                if delta > 0 and self._delay_supported(dag.find_bit(bit).index):
                    new_dag.apply_operation_front(Delay(delta, time_unit), [bit], [], check=False)
                idle_before[bit] = t1

            new_dag.apply_operation_front(node.op, node.qargs, node.cargs, check=False)

        circuit_duration = max(idle_before.values())
        for bit, before in idle_before.items():
            delta = circuit_duration - before
            if not (delta > 0 and isinstance(bit, Qubit)):
                continue
            if self._delay_supported(dag.find_bit(bit).index):
                new_dag.apply_operation_front(Delay(delta, time_unit), [bit], [], check=False)

        new_dag.name = dag.name
        new_dag.metadata = dag.metadata
        new_dag._calibrations_prop = dag._calibrations_prop

        # set circuit duration and unit to indicate it is scheduled
        new_dag.duration = circuit_duration
        new_dag.unit = time_unit

        return new_dag

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

"""ASAP Scheduling."""
from qiskit.circuit import Measure
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.transpiler.passes.scheduling.scheduling.base_scheduler import BaseScheduler


class ASAPScheduleAnalysis(BaseScheduler):
    """ASAP Scheduling pass, which schedules the start time of instructions as early as possible.

    See the :ref:`transpiler-scheduling-description` section in the :mod:`qiskit.transpiler`
    module documentation for the detailed behavior of the control flow
    operation, i.e. ``c_if``.
    """

    def run(self, dag):
        """Run the ASAPSchedule pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to schedule.

        Returns:
            DAGCircuit: A scheduled DAG.

        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
            TranspilerError: if conditional bit is added to non-supported instruction.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("ASAP schedule runs on physical circuits only")

        conditional_latency = self.property_set.get("conditional_latency", 0)
        clbit_write_latency = self.property_set.get("clbit_write_latency", 0)

        node_start_time = {}
        idle_after = {q: 0 for q in dag.qubits + dag.clbits}
        for node in dag.topological_op_nodes():
            op_duration = self._get_node_duration(node, dag)

            # compute t0, t1: instruction interval, note that
            # t0: start time of instruction
            # t1: end time of instruction
            if isinstance(node.op, self.CONDITIONAL_SUPPORTED):
                t0q = max(idle_after[q] for q in node.qargs)
                if node.op.condition_bits:
                    # conditional is bit tricky due to conditional_latency
                    t0c = max(idle_after[bit] for bit in node.op.condition_bits)
                    if t0q > t0c:
                        # This is situation something like below
                        #
                        #           |t0q
                        # Q ▒▒▒▒▒▒▒▒▒░░
                        # C ▒▒▒░░░░░░░░
                        #     |t0c
                        #
                        # In this case, you can insert readout access before tq0
                        #
                        #           |t0q
                        # Q ▒▒▒▒▒▒▒▒▒▒▒
                        # C ▒▒▒░░░▒▒░░░
                        #         |t0q - conditional_latency
                        #
                        t0c = max(t0q - conditional_latency, t0c)
                    t1c = t0c + conditional_latency
                    for bit in node.op.condition_bits:
                        # Lock clbit until state is read
                        idle_after[bit] = t1c
                    # It starts after register read access
                    t0 = max(t0q, t1c)
                else:
                    t0 = t0q
                t1 = t0 + op_duration
            else:
                if node.op.condition_bits:
                    raise TranspilerError(
                        f"Conditional instruction {node.op.name} is not supported in ASAP scheduler."
                    )

                if isinstance(node.op, Measure):
                    # measure instruction handling is bit tricky due to clbit_write_latency
                    t0q = max(idle_after[q] for q in node.qargs)
                    t0c = max(idle_after[c] for c in node.cargs)
                    # Assume following case (t0c > t0q)
                    #
                    #       |t0q
                    # Q ▒▒▒▒░░░░░░░░░░░░
                    # C ▒▒▒▒▒▒▒▒░░░░░░░░
                    #           |t0c
                    #
                    # In this case, there is no actual clbit access until clbit_write_latency.
                    # The node t0 can be push backward by this amount.
                    #
                    #         |t0q' = t0c - clbit_write_latency
                    # Q ▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒
                    # C ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                    #           |t0c' = t0c
                    #
                    # rather than naively doing
                    #
                    #           |t0q' = t0c
                    # Q ▒▒▒▒░░░░▒▒▒▒▒▒▒▒
                    # C ▒▒▒▒▒▒▒▒░░░▒▒▒▒▒
                    #              |t0c' = t0c + clbit_write_latency
                    #
                    t0 = max(t0q, t0c - clbit_write_latency)
                    t1 = t0 + op_duration
                    for clbit in node.cargs:
                        idle_after[clbit] = t1
                else:
                    # It happens to be directives such as barrier
                    t0 = max(idle_after[bit] for bit in node.qargs + node.cargs)
                    t1 = t0 + op_duration

            for bit in node.qargs:
                idle_after[bit] = t1

            node_start_time[node] = t0

        self.property_set["node_start_time"] = node_start_time

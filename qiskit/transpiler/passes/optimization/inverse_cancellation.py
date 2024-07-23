# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A generic InverseCancellation pass for any set of gate-inverse pairs.
"""
from __future__ import annotations

from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class InverseCancellation(TransformationPass):
    """Cancel specific Gates which are inverses of each other when they occur back-to-
    back."""

    def __init__(self, gates_to_cancel: list[Gate | tuple[Gate, Gate]]):
        """Initialize InverseCancellation pass.

        Args:
            gates_to_cancel: List describing the gates to cancel. Each element of the
                list is either a single gate or a pair of gates. If a single gate, then
                it should be self-inverse. If a pair of gates, then the gates in the
                pair should be inverses of each other.

        Raises:
            TranspilerError: Input is not a self-inverse gate or a pair of inverse gates.
        """

        for gates in gates_to_cancel:
            if isinstance(gates, Gate):
                if gates != gates.inverse():
                    raise TranspilerError(f"Gate {gates.name} is not self-inverse")
            elif isinstance(gates, tuple):
                if len(gates) != 2:
                    raise TranspilerError(
                        f"Too many or too few inputs: {gates}. Only two are allowed."
                    )
                if gates[0] != gates[1].inverse():
                    raise TranspilerError(
                        f"Gate {gates[0].name} and {gates[1].name} are not inverse."
                    )
            else:
                raise TranspilerError(
                    f"InverseCancellation pass does not take input type {type(gates)}. Input must be"
                    " a Gate."
                )

        self.self_inverse_gates = []
        self.inverse_gate_pairs = []
        self.self_inverse_gate_names = set()
        self.inverse_gate_pairs_names: set[str] = set()

        for gates in gates_to_cancel:
            if isinstance(gates, Gate):
                self.self_inverse_gates.append(gates)
                self.self_inverse_gate_names.add(gates.name)
            else:
                self.inverse_gate_pairs.append(gates)
                self.inverse_gate_pairs_names.update(x.name for x in gates)

        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the InverseCancellation pass on `dag`.

        Args:
            dag: the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        if self.self_inverse_gates:
            dag = self._run_on_self_inverse(dag)
        if self.inverse_gate_pairs:
            dag = self._run_on_inverse_pairs(dag)
        return dag

    def _run_on_self_inverse(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run self-inverse gates on `dag`.

        Args:
            dag: the directed acyclic graph to run on.
            self_inverse_gates: list of gates who cancel themeselves in pairs

        Returns:
            DAGCircuit: Transformed DAG.
        """
        op_counts = dag.count_ops()
        if not self.self_inverse_gate_names.intersection(op_counts):
            return dag
        # Sets of gate runs by name, for instance: [{(H 0, H 0), (H 1, H 1)}, {(X 0, X 0}]
        for gate in self.self_inverse_gates:
            gate_name = gate.name
            gate_count = op_counts.get(gate_name, 0)
            if gate_count <= 1:
                continue
            gate_runs = dag.collect_runs([gate_name])
            for gate_cancel_run in gate_runs:
                partitions = []
                chunk = []
                max_index = len(gate_cancel_run) - 1
                for i, cancel_gate in enumerate(gate_cancel_run):
                    if cancel_gate.op == gate:
                        chunk.append(cancel_gate)
                    else:
                        if chunk:
                            partitions.append(chunk)
                            chunk = []
                        continue
                    if i == max_index or cancel_gate.qargs != gate_cancel_run[i + 1].qargs:
                        partitions.append(chunk)
                        chunk = []
                # Remove an even number of gates from each chunk
                for chunk in partitions:
                    if len(chunk) % 2 == 0:
                        dag.remove_op_node(chunk[0])
                    for node in chunk[1:]:
                        dag.remove_op_node(node)
        return dag

    def _run_on_inverse_pairs(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Run inverse gate pairs on `dag`.

        Args:
            dag: the directed acyclic graph to run on.
            inverse_gate_pairs: list of gates with inverse angles that cancel each other.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        op_counts = dag.count_ops()
        if not self.inverse_gate_pairs_names.intersection(op_counts):
            return dag

        for pair in self.inverse_gate_pairs:
            gate_0_name = pair[0].name
            gate_1_name = pair[1].name
            if gate_0_name not in op_counts or gate_1_name not in op_counts:
                continue
            gate_cancel_runs = dag.collect_runs([gate_0_name, gate_1_name])
            for dag_nodes in gate_cancel_runs:
                i = 0
                while i < len(dag_nodes) - 1:
                    if (
                        dag_nodes[i].qargs == dag_nodes[i + 1].qargs
                        and dag_nodes[i].op == pair[0]
                        and dag_nodes[i + 1].op == pair[1]
                    ):
                        dag.remove_op_node(dag_nodes[i])
                        dag.remove_op_node(dag_nodes[i + 1])
                        i = i + 2
                    elif (
                        dag_nodes[i].qargs == dag_nodes[i + 1].qargs
                        and dag_nodes[i].op == pair[1]
                        and dag_nodes[i + 1].op == pair[0]
                    ):
                        dag.remove_op_node(dag_nodes[i])
                        dag.remove_op_node(dag_nodes[i + 1])
                        i = i + 2
                    else:
                        i = i + 1
        return dag

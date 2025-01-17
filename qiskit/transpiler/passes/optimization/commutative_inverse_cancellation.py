# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Cancel pairs of inverse gates exploiting commutation relations."""
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc

from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.transpiler.basepasses import TransformationPass


class CommutativeInverseCancellation(TransformationPass):
    """Cancel pairs of inverse gates exploiting commutation relations."""

    def __init__(self, matrix_based: bool = False, max_qubits: int = 4):
        """
        Args:
            matrix_based: If ``True``, uses matrix representations to check whether two
                operations are inverse of each other. This makes the checks more powerful,
                and, in addition, allows canceling pairs of operations that are inverse up to a
                phase, while updating the global phase of the circuit accordingly.
                Generally this leads to more reductions at the expense of increased runtime.
            max_qubits: Limits the number of qubits in matrix-based commutativity and
                inverse checks.
        """
        self._matrix_based = matrix_based
        self._max_qubits = max_qubits
        self.comm_checker = scc
        super().__init__()

    def _skip_node(self, node):
        """Returns True if we should skip this node for the analysis."""
        if not isinstance(node, DAGOpNode):
            return True

        # We are currently taking an over-conservative approach with respect to which nodes
        # can be inverses of which other nodes, and do not allow reductions for barriers, measures,
        # conditional gates or parameterized gates. Possibly both this and commutativity
        # checking can be extended to cover additional cases.
        if getattr(node.op, "_directive", False) or node.name in {"measure", "reset", "delay"}:
            return True
        if getattr(node, "condition", None):
            return True
        if node.op.is_parameterized():
            return True
        return False

    def _check_inverse(self, node1, node2):
        """Checks whether op1 and op2 are inverse up to a phase, that is whether
        ``op2 = e^{i * d} op1^{-1})`` for some phase difference ``d``.
        If this is the case, we can replace ``op2 * op1`` by `e^{i * d} I``.
        The input to this function is a pair of DAG nodes.
        The output is a tuple representing whether the two nodes
        are inverse up to a phase and that phase difference.
        """
        phase_difference = 0
        if not self._matrix_based:
            is_inverse = node1.op.inverse() == node2.op
        elif len(node2.qargs) > self._max_qubits:
            is_inverse = False
        else:
            mat1 = Operator(node1.op.inverse()).data
            mat2 = Operator(node2.op).data
            props = {}
            is_inverse = matrix_equal(mat1, mat2, ignore_phase=True, props=props)
            if is_inverse:
                # mat2 = e^{i * phase_difference} mat1
                phase_difference = props["phase_difference"]
        return is_inverse, phase_difference

    def run(self, dag: DAGCircuit):
        """
        Run the CommutativeInverseCancellation pass on `dag`.

        Args:
            dag: the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        topo_sorted_nodes = list(dag.topological_op_nodes())

        circ_size = len(topo_sorted_nodes)

        removed = [False for _ in range(circ_size)]

        phase_update = 0

        for idx1 in range(0, circ_size):
            if self._skip_node(topo_sorted_nodes[idx1]):
                continue

            matched_idx2 = -1

            for idx2 in range(idx1 - 1, -1, -1):
                if removed[idx2]:
                    continue

                if (
                    not self._skip_node(topo_sorted_nodes[idx2])
                    and topo_sorted_nodes[idx2].qargs == topo_sorted_nodes[idx1].qargs
                    and topo_sorted_nodes[idx2].cargs == topo_sorted_nodes[idx1].cargs
                ):
                    is_inverse, phase = self._check_inverse(
                        topo_sorted_nodes[idx1], topo_sorted_nodes[idx2]
                    )
                    if is_inverse:
                        phase_update += phase
                        matched_idx2 = idx2
                        break

                if not self.comm_checker.commute_nodes(
                    topo_sorted_nodes[idx1],
                    topo_sorted_nodes[idx2],
                    max_num_qubits=self._max_qubits,
                ):
                    break

            if matched_idx2 != -1:
                removed[idx1] = True
                removed[matched_idx2] = True

        for idx in range(circ_size):
            if removed[idx]:
                dag.remove_op_node(topo_sorted_nodes[idx])

        if phase_update != 0:
            dag.global_phase += phase_update

        return dag

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Recursively expands 3q+ gates until the circuit only contains 2q or 1q gates."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.exceptions import QiskitError
from qiskit.circuit import ControlFlowOp
from qiskit.converters.circuit_to_dag import circuit_to_dag


class Unroll3qOrMore(TransformationPass):
    """Recursively expands 3q+ gates until the circuit only contains 2q or 1q gates."""

    def __init__(self, target=None, basis_gates=None):
        """Initialize the Unroll3qOrMore pass

        Args:
            target (Target): The target object representing the compilation
                target. If specified any multi-qubit instructions in the
                circuit when the pass is run that are supported by the target
                device will be left in place. If both this and ``basis_gates``
                are specified only the target will be checked.
            basis_gates (list): A list of basis gate names that the target
                device supports. If specified any gate names in the circuit
                which are present in this list will not be unrolled. If both
                this and ``target`` are specified only the target will be used
                for checking which gates are supported.
        """
        super().__init__()
        self.target = target
        self.basis_gates = None
        if basis_gates is not None:
            self.basis_gates = set(basis_gates)

    def run(self, dag):
        """Run the Unroll3qOrMore pass on `dag`.

        Args:
            dag(DAGCircuit): input dag
        Returns:
            DAGCircuit: output dag with maximum node degrees of 2
        Raises:
            QiskitError: if a 3q+ gate is not decomposable
        """
        for node in dag.multi_qubit_ops():
            if dag._has_calibration_for(node):
                continue

            if isinstance(node.op, ControlFlowOp):
                dag.substitute_node(
                    node, control_flow.map_blocks(self.run, node.op), propagate_condition=False
                )
                continue

            if self.target is not None:
                # Treat target instructions as global since this pass can be run
                # prior to layout and routing we don't have physical qubits from
                # the circuit yet
                if node.name in self.target:
                    continue
            elif self.basis_gates is not None and node.name in self.basis_gates:
                continue

            # TODO: allow choosing other possible decompositions
            rule = node.op.definition.data
            if not rule:
                if rule == []:  # empty node
                    dag.remove_op_node(node)
                    continue
                raise QiskitError(
                    "Cannot unroll all 3q or more gates. "
                    f"No rule to expand instruction {node.op.name}."
                )
            decomposition = circuit_to_dag(node.op.definition, copy_operations=False)
            decomposition = self.run(decomposition)  # recursively unroll
            dag.substitute_node_with_dag(node, decomposition)
        return dag

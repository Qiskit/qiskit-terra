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

"""Replace conditional instructions with equivalent :class:`.IfElseOp` objects."""

from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit import (
    CircuitInstruction,
    ClassicalRegister,
    Clbit,
    IfElseOp,
    QuantumCircuit,
)
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.utils import deprecate_func


class ConvertConditionsToIfOps(TransformationPass):
    """Convert instructions whose ``condition`` attribute is set to a non-``None`` value into the
    equivalent single-statement :class:`.IfElseBlock`.

    This is a simple pass aimed at easing the conversion from the old style of using
    :meth:`.InstructionSet.c_if` into the new style of using more complex conditional logic."""

    @deprecate_func(since="1.3.0", removal_timeline="in Qiskit 2.0.0")
    def __init__(self):
        super().__init__()

    def _run_inner(self, dag):
        """Run the pass on one :class:`.DAGCircuit`, mutating it.  Returns ``True`` if the circuit
        was modified and ``False`` if not."""
        modified = False
        for node in dag.op_nodes():
            if node.is_control_flow():
                modified_blocks = False
                new_dags = []
                for block in node.op.blocks:
                    new_dag = circuit_to_dag(block)
                    modified_blocks |= self._run_inner(new_dag)
                    new_dags.append(new_dag)
                if not modified_blocks:
                    continue
                dag.substitute_node(
                    node,
                    node.op.replace_blocks(dag_to_circuit(block) for block in new_dags),
                    inplace=True,
                )
            elif node.condition is None:
                continue
            else:
                target, value = node.op.condition
                clbits = list(node.cargs)
                condition_clbits = [target] if isinstance(target, Clbit) else list(target)
                clbits_set = set(clbits)
                clbits += [bit for bit in condition_clbits if bit not in clbits_set]
                block_body = QuantumCircuit(list(node.qargs) + clbits)
                if isinstance(target, ClassicalRegister):
                    block_body.add_register(target)
                new_op = node.op.copy()
                new_op.condition = None
                block_body._append(CircuitInstruction(new_op, node.qargs, node.cargs))
                # Despite only being a node-for-node replacement, control-flow ops contain the
                # condition bits in their cargs, which requires slightly different handling in the
                # DAGCircuit methods right now.
                replacement = DAGCircuit()
                replacement.add_qubits(block_body.qubits)
                replacement.add_clbits(block_body.clbits)
                if isinstance(target, ClassicalRegister):
                    replacement.add_creg(target)
                replacement.apply_operation_back(
                    IfElseOp((target, value), block_body),
                    block_body.qubits,
                    block_body.clbits,
                    check=False,
                )
                wire_map = {bit: bit for bit in block_body.qubits + block_body.clbits}
                dag.substitute_node_with_dag(node, replacement, wire_map, propagate_condition=False)
            modified = True
        return modified

    def run(self, dag):
        self._run_inner(dag)
        return dag

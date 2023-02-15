# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Replace Swap gate in case of qubit in zero state with dynamic protocol."""

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister, Reset
from qiskit.circuit.library import CXGate, HGate, Measure, SwapGate
from qiskit.circuit.controlflow.if_else import IfElseOp
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGNode, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass


class ReplaceSwapWithZeroState(TransformationPass):
    """Replace Swap gate in case of qubit in zero state with
    dynamic protocol."""

    def run(self, dag):
        """Run the ReplaceSwapWithZeroState pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        swaps = dag.op_nodes(SwapGate)
        for swap in swaps:
            for node in dag.predecessors(swap):
                if isinstance(node, DAGInNode):
                    zero_qubit = node.wire.index
                    for qarg in swap.qargs:
                        if qarg.index != zero_qubit:
                            data_qubit = qarg.index

                    self._replace_circuit(dag, node, zero_qubit, data_qubit)

                elif isinstance(node, DAGOpNode):
                    if isinstance(node.op, Reset):
                        zero_qubit = node.qargs[0].index
                        for qarg in swap.qargs:
                            if qarg.index != zero_qubit:
                                data_qubit = qarg.index

                        self._replace_circuit(dag, node, zero_qubit, data_qubit)

        return dag

    @staticmethod
    def _replace_circuit(node: DAGNode, dag: DAGCircuit, zero_qubit, data_qubit):
        mini_dag = DAGCircuit()
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(1)
        mini_dag.add_qreg(qreg)
        mini_dag.add_creg(creg)

        mini_dag.apply_operation_back(CXGate(), [qreg[data_qubit], qreg[zero_qubit]])
        mini_dag.apply_operation_back(HGate(), [qreg[data_qubit]])
        mini_dag.apply_operation_back(Measure(), [qreg[data_qubit]], [creg[0]])

        true_body = QuantumCircuit(qreg)
        true_body.z(0)
        true_body.x(1)

        mini_dag.apply_operation_back(
            IfElseOp((creg[0], 1), true_body), [qreg[zero_qubit], qreg[data_qubit]], [creg[0]]
        )
        dag.substitute_node_with_dag(
            swap, mini_dag, wires=[qreg[zero_qubit], qreg[data_qubit], creg[0]]
        )

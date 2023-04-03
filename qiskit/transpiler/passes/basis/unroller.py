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

"""Unroll a circuit to a given basis."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.exceptions import QiskitError
from qiskit.circuit import ControlledGate, ControlFlowOp
from qiskit.converters.circuit_to_dag import circuit_to_dag


class Unroller(TransformationPass):
    """Unroll a circuit to a given basis.

    Unroll (expand) non-basis, non-opaque instructions recursively
    to a desired basis, using decomposition rules defined for each instruction.
    """

    def __init__(self, basis=None, target=None):
        """Unroller initializer.

        Args:
            basis (list[str] or None): Target basis names to unroll to, e.g. `['u3', 'cx']` . If
                None, does not unroll any gate.
            target (Target):  The :class:`~.Target` representing the target backend, if both
                ``basis`` and this are specified then this argument will take
                precedence and ``basis`` will be ignored.
        """
        super().__init__()
        self.basis = basis
        self.target = target

    def run(self, dag):
        """Run the Unroller pass on `dag`.

        Args:
            dag (DAGCircuit): input dag

        Raises:
            QiskitError: if unable to unroll given the basis due to undefined
            decomposition rules (such as a bad basis) or excessive recursion.

        Returns:
            DAGCircuit: output unrolled dag
        """
        if self.basis is None and self.target is None:
            return dag
        qubit_mapping = {}
        if self.target is not None:
            qubit_mapping = {bit: index for index, bit in enumerate(dag.qubits)}
        # Walk through the DAG and expand each non-basis node
        basic_insts = ["measure", "reset", "barrier", "snapshot", "delay"]
        for node in dag.op_nodes():
            if getattr(node.op, "_directive", False):
                continue

            run_qubits = None
            if self.target is not None:
                run_qubits = tuple(qubit_mapping[x] for x in node.qargs)
                if (
                    self.target.instruction_supported(node.op.name, qargs=run_qubits)
                    or node.op.name == "barrier"
                ):
                    print("blue")
                    if isinstance(node.op, ControlledGate) and node.op._open_ctrl:
                        pass
                    else:
                        continue
            else:
                if node.name in basic_insts:
                    # TODO: this is legacy behavior.Basis_insts should be removed that these
                    #  instructions should be part of the device-reported basis. Currently, no
                    #  backend reports "measure", for example.
                    continue

                if node.name in self.basis:  # If already a base, ignore.
                    if isinstance(node.op, ControlledGate) and node.op._open_ctrl:
                        pass
                    else:
                        continue

            if isinstance(node.op, ControlFlowOp):
                node.op = control_flow.map_blocks(self.run, node.op)
                continue

            try:
                phase = node.op.definition.global_phase
                rule = node.op.definition.data
            except (TypeError, AttributeError) as err:
                raise QiskitError(
                    f"Error decomposing node of instruction '{node.name}'. "
                    f"Unable to define instruction '{node.name}' in the given basis."
                ) from err

            # Isometry gates definitions can have widths smaller than that of the
            # original gate, in which case substitute_node will raise. Fall back
            # to substitute_node_with_dag if an the width of the definition is
            # different that the width of the node.
            while rule and len(rule) == 1 and len(node.qargs) == len(rule[0].qubits) == 1:
                if self.target is not None:
                    if self.target.instruction_supported(rule[0].operation.name, run_qubits):
                        dag.global_phase += phase
                        dag.substitute_node(node, rule[0].operation, inplace=True)
                        break
                else:
                    if rule[0].operation.name in self.basis:
                        dag.global_phase += phase
                        dag.substitute_node(node, rule[0].operation, inplace=True)
                        break
                try:
                    phase += rule[0].operation.definition.global_phase
                    rule = rule[0].operation.definition.data
                except (TypeError, AttributeError) as err:
                    raise QiskitError(
                        f"Error decomposing node of instruction '{node.name}'. "
                        f"Unable to define instruction '{rule[0].operation.name}' in the basis."
                    ) from err

            else:
                if not rule:
                    if rule == []:  # empty node
                        dag.remove_op_node(node)
                        dag.global_phase += phase
                        continue
                    # opaque node
                    raise QiskitError(
                        "Cannot unroll the circuit to the given basis, %s. "
                        "No rule to expand instruction %s." % (str(self.basis), node.op.name)
                    )
                decomposition = circuit_to_dag(node.op.definition)
                unrolled_dag = self.run(decomposition)  # recursively unroll ops
                dag.substitute_node_with_dag(node, unrolled_dag)
        return dag

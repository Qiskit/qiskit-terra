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

"""Cancel the redundant (self-adjoint) gates through commutation relations."""

from collections import defaultdict
import numpy as np

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes.optimization.commutation_analysis import CommutationAnalysis
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGOutNode
from qiskit.circuit.commutation_library import CommutationChecker, StandardGateCommutations
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.rx import RXGate
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES


_CUTOFF_PRECISION = 1e-5


class CommutativeCancellation(TransformationPass):
    """Cancel the redundant (self-adjoint) gates through commutation relations.

    Pass for cancelling self-inverse gates/rotations. The cancellation utilizes
    the commutation relations in the circuit. Gates considered include::

        H, X, Y, Z, CX, CY, CZ
    """

    def __init__(self, basis_gates=None, target=None):
        """
        CommutativeCancellation initializer.

        Args:
            basis_gates (list[str]): Basis gates to consider, e.g.
                ``['u3', 'cx']``. For the effects of this pass, the basis is
                the set intersection between the ``basis_gates`` parameter
                and the gates in the dag.
            target (Target): The :class:`~.Target` representing the target backend, if both
                ``basis_gates`` and ``target`` are specified then this argument will take
                precedence and ``basis_gates`` will be ignored.
        """
        super().__init__()

        if target is not None:
            self.basis = set(target.operation_names)
        elif basis_gates is not None:
            self.basis = set(basis_gates)
        else:
            self.basis = set()

        self._var_z_map = {"rz": RZGate, "p": PhaseGate, "u1": U1Gate}

        self._z_rotations = {"p", "z", "u1", "rz", "t", "s", "sdg", "tdg"}
        self._x_rotations = {"x", "rx", "sx", "sxdg"}
        self._gates = {"cx", "cy", "cz", "h", "y"}  # Now the gates supported are hard-coded

        # build a commutation checker restricted to the gates we cancel -- the others we
        # do not have to investigate, which allows to save time
        commutation_checker = CommutationChecker(
            StandardGateCommutations, gates=self._gates | self._z_rotations | self._x_rotations
        )

        self.requires.append(CommutationAnalysis(_commutation_checker=commutation_checker))

    def run(self, dag):
        """Run the CommutativeCancellation pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        # For RZ we do extra gymnastics: we prioritize the Z-rotation that is already in the
        # circuit, if there is one.
        var_z_map = {"rz": RZGate, "p": PhaseGate, "u1": U1Gate}
        z_var_gates = [gate for gate in dag.count_ops().keys() if gate in var_z_map]
        if len(z_var_gates) > 0:
            # prioritize z gates in circuit
            var_z_gate = var_z_map[z_var_gates[0]]
        else:
            z_var_gates = [gate for gate in self.basis if gate in var_z_map]
            if len(z_var_gates) > 0:
                var_z_gate = var_z_map[z_var_gates[0]]
            else:
                var_z_gate = None  # no Z-rotation in circuit or in the basis!

        # Gate sets to be cancelled
        cancellation_sets = defaultdict(lambda: [])

        # Traverse each qubit to generate the cancel dictionaries
        # Cancel dictionaries:
        #  - For 1-qubit gates the key is (gate_type, qubit_id, commutation_set_id),
        #    the value is the list of gates that share the same gate type, qubit, commutation set.
        #  - For 2qbit gates the key: (gate_type, first_qbit, sec_qbit, first commutation_set_id,
        #    sec_commutation_set_id), the value is the list gates that share the same gate type,
        #    qubits and commutation sets.
        for wire in dag.qubits:
            wire_commutation_set = self.property_set["commutation_set"][wire]

            for com_set_idx, com_set in enumerate(wire_commutation_set):
                if isinstance(com_set[0], (DAGInNode, DAGOutNode)):
                    continue
                for node in com_set:
                    num_qargs = len(node.qargs)
                    if any(isinstance(p, ParameterExpression) for p in node.params):
                        continue  # no support for cancellation of parameterized gates
                    if num_qargs == 1 and node.name in self._gates:
                        cancellation_sets[(node.name, wire, com_set_idx)].append(node)
                    if num_qargs == 1 and node.name in self._z_rotations:
                        cancellation_sets[("z_rotation", wire, com_set_idx)].append(node)
                    if num_qargs == 1 and node.name in self._x_rotations:
                        cancellation_sets[("x_rotation", wire, com_set_idx)].append(node)
                    # Don't deal with Y rotation, because Y rotation doesn't commute with CNOT, so
                    # it should be dealt with by optimized1qgate pass
                    elif num_qargs == 2 and node.qargs[0] == wire:
                        second_qarg = node.qargs[1]
                        q2_key = (
                            node.name,
                            wire,
                            second_qarg,
                            com_set_idx,
                            self.property_set["commutation_set"][(node, second_qarg)],
                        )
                        cancellation_sets[q2_key].append(node)

        for cancel_set_key in cancellation_sets:
            if cancel_set_key[0] == "z_rotation" and var_z_gate is None:
                continue  # we would like to cancel, but our hands are bound

            set_len = len(cancellation_sets[cancel_set_key])
            if set_len > 1 and cancel_set_key[0] in self._gates:
                gates_to_cancel = cancellation_sets[cancel_set_key]
                for c_node in gates_to_cancel[: (set_len // 2) * 2]:
                    dag.remove_op_node(c_node)

            elif set_len > 1 and cancel_set_key[0] in ["z_rotation", "x_rotation"]:
                run = cancellation_sets[cancel_set_key]
                run_qarg = run[0].qargs[0]
                total_angle = 0.0  # lambda
                total_phase = 0.0
                for current_node in run:
                    if (
                        current_node.condition is not None
                        or len(current_node.qargs) != 1
                        or current_node.qargs[0] != run_qarg
                    ):
                        raise RuntimeError("internal error")

                    if current_node.name in ["p", "u1", "rz", "rx"]:
                        current_angle = float(current_node.params[0])
                    elif current_node.name in ["z", "x"]:
                        current_angle = np.pi
                    elif current_node.name == "t":
                        current_angle = np.pi / 4
                    elif current_node.name == "tdg":
                        current_angle = -np.pi / 4
                    elif current_node.name in ["s", "sx"]:
                        current_angle = np.pi / 2
                    elif current_node.name in ["sdg", "sxdg"]:
                        current_angle = -np.pi / 2
                    else:
                        raise RuntimeError(
                            f"Angle for operation {current_node.name} is not defined"
                        )

                    # Compose gates
                    total_angle = current_angle + total_angle
                    if current_node.definition:
                        total_phase += current_node.definition.global_phase

                # Replace the data of the first node in the run
                if cancel_set_key[0] == "z_rotation":
                    new_op = var_z_gate(total_angle)
                elif cancel_set_key[0] == "x_rotation":
                    new_op = RXGate(total_angle)
                else:
                    raise RuntimeError("impossible case")

                new_op_phase = 0
                if np.mod(total_angle, (2 * np.pi)) > _CUTOFF_PRECISION:
                    new_qarg = QuantumRegister(1, "q")
                    new_dag = DAGCircuit()
                    new_dag.add_qreg(new_qarg)
                    new_dag.apply_operation_back(new_op, [new_qarg[0]])
                    dag.substitute_node_with_dag(run[0], new_dag)
                    if new_op.definition:
                        new_op_phase = new_op.definition.global_phase

                dag.global_phase = total_phase - new_op_phase

                # Delete the other nodes in the run
                for current_node in run[1:]:
                    dag.remove_op_node(current_node)

                if np.mod(total_angle, (2 * np.pi)) < _CUTOFF_PRECISION:
                    dag.remove_op_node(run[0])

        dag = self._handle_control_flow_ops(dag)

        return dag

    def _handle_control_flow_ops(self, dag):
        """
        This is similar to transpiler/passes/utils/control_flow.py except that the
        commutation analysis is redone for the control flow blocks.
        """

        pass_manager = PassManager([CommutationAnalysis(), self])
        for node in dag.op_nodes():
            if node.name not in CONTROL_FLOW_OP_NAMES:
                continue
            mapped_blocks = []
            for block in node.op.blocks:
                new_circ = pass_manager.run(block)
                mapped_blocks.append(new_circ)
            dag.substitute_node(
                node, node.op.replace_blocks(mapped_blocks), propagate_condition=False
            )
        return dag

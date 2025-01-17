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
from __future__ import annotations

from qiskit.circuit.commutation_checker import CommutationChecker
from qiskit.circuit.commutation_library import standard_gates_commutations
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library.generalized_gates.pauli import PauliGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import Pauli
from qiskit.transpiler.basepasses import TransformationPass

commutator = CommutationChecker(standard_gates_commutations)


class LightCone(TransformationPass):
    """Remove the gates that do not affect the outcome of a measurement on a circuit.

    Pass for computing the light-cone of an observable or measurement. The Pass can
    handle either a Pauli operator one would like to measure or a measurement on a set
    of qubits.
    """

    def __init__(self, observable: Pauli | None = None):
        """
        LightCone initializer.

        Args:
            observable: If None the lightcone will be computed for the set
            of measurements in the circuit.
            If a Pauli operator is specified, the lightcone will correspond to
            the reduced circuit with the
            same expectation value for the observable.
        """
        super().__init__()
        self.observable = observable

    @staticmethod
    def _find_measurement_qubits(dag: DAGCircuit):
        """For now, this method considers only final circuit measurements:
        mid-circuit measurements are discarded.
        """
        qubits_measured = set()

        # Iterate over the DAG nodes in topological order
        for node in dag.topological_nodes():
            # Check if the node is a measurement operation
            if getattr(node, "name", False) == "measure":
                qubits = set(node.qargs)

                # Check if these qubits are used in any subsequent operations
                is_final_measurement = True
                for subsequent_node in dag.successors(node):
                    if isinstance(subsequent_node, DAGOpNode):
                        is_final_measurement = False
                        break

                qubits_measured |= qubits if is_final_measurement else set()
        if not qubits_measured:
            raise ValueError("No measurements found in the circuit.")
        return qubits_measured

    def _get_initial_lightcone(
        self, dag: DAGCircuit
    ) -> tuple[set[int], tuple[Instruction, list[int]]]:
        """Returns the initial lightcone.
        If obervable is `None`, the lightcone is the set of measured qubits.
        If a Pauli observable is provided, the qubit corresponding to
        the non-trivial Paulis define the lightcone.
        """
        if self.observable is None:
            qubits_measured = self._find_measurement_qubits(dag)
            light_cone = qubits_measured
            light_cone_ops = [(PauliGate("Z"), [qubit_index]) for qubit_index in light_cone]
        else:
            pauli_string = str(self.observable)
            # Check if the size of the observable matches the number of qubits in the circuit
            if len(pauli_string) != len(dag.qubits):
                raise ValueError(
                    "Observable size does not match the number of qubits in the circuit."
                )
            stripped_pauli_string = pauli_string.replace("I", "")
            if len(stripped_pauli_string) == 0:
                raise ValueError("Observable is the identity operator.")
            light_cone = [dag.qubits[i] for i, p in enumerate(pauli_string) if p != "I"]
            light_cone_ops = [(PauliGate(stripped_pauli_string), light_cone)]

        return set(light_cone), light_cone_ops

    def run(self, dag: DAGCircuit):
        """Run the LightCone pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        # Get the initial light cone and operations
        light_cone, light_cone_ops = self._get_initial_lightcone(dag)

        #  Initialize a new, empty DAG
        new_dag = DAGCircuit()

        new_dag.add_qreg(*dag.qregs.values())

        if dag.cregs:  # Add classical registers if they exist
            new_dag.add_creg(*dag.cregs.values())

        # Iterate over the nodes in reverse topological order
        for node in reversed(list(dag.topological_op_nodes())):
            # Check if the node belongs to the light cone
            if light_cone.intersection(node.qargs):
                # Check commutation with all previous operations
                commutes_bool = True
                for op in light_cone_ops:
                    commute_bool = commutator.commute(op[0], op[1], [], node.op, node.qargs, [])
                    if not commute_bool:
                        # If the current node does not commute, update the light cone
                        light_cone.update(node.qargs)
                        light_cone_ops.append((node.op, node.qargs))
                        commutes_bool = False
                        break

                # If the node is in the light cone and commutes with previous ops,
                # add it to the new DAG at the front
                if not commutes_bool:
                    new_dag.apply_operation_front(node.op, node.qargs, node.cargs)
        return new_dag

# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Remove any swap gates in the circuit by pushing it through into a qubit permutation."""

import logging

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.library.generalized_gates import PermutationGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout

logger = logging.getLogger(__name__)


class ElidePermutations(TransformationPass):
    r"""Remove permutation operations from a pre-layout circuit

    This pass is intended to be run before a layout (mapping virtual qubits
    to physical qubits) is set during the transpilation pipeline. This
    pass iterates over the :class:`~.DAGCircuit` and when a :class:`~.SwapGate`
    or :class:`~.PermutationGate` are encountered it permutes the virtual qubits in
    the circuit and removes the swap gate. This will effectively remove any
    :class:`~SwapGate`\s or :class:`~PermutationGate` in the circuit prior to running
    layout. If this pass is run after a layout has been set it will become a no-op
    (and log a warning) as this optimization is not sound after physical qubits are
    selected and there are connectivity constraints to adhere to.

    For tracking purposes this pass sets 3 values in the property set if there
    are any :class:`~.SwapGate` or :class:`~.PermutationGate` objects in the circuit
    and the pass isn't a no-op.

    * ``original_layout``: The trivial :class:`~.Layout` for the input to this pass being run
    * ``original_qubit_indices``: The mapping of qubit objects to positional indices for the state
        of the circuit as input to this pass.
    * ``virtual_permutation_layout``: A :class:`~.Layout` object mapping input qubits to the output
        state after eliding permutations.

    These three properties are needed for the transpiler to track the permutations in the out
    :attr:`.QuantumCircuit.layout` attribute. The elision of permutations is equivalent to a
    ``final_layout`` set by routing and all three of these attributes are needed in the case
    """

    def run(self, dag):
        """Run the ElidePermutations pass on ``dag``.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        if self.property_set["layout"] is not None:
            logger.warning(
                "ElidePermutations is not valid after a layout has been set. This indicates "
                "an invalid pass manager construction."
            )
            return dag

        op_count = dag.count_ops(recurse=False)
        if op_count.get("swap", 0) == 0 and op_count.get("permutation", 0) == 0:
            return dag

        new_dag = dag.copy_empty_like()
        qubit_mapping = list(range(len(dag.qubits)))

        def _apply_mapping(qargs):
            return tuple(dag.qubits[qubit_mapping[dag.find_bit(qubit).index]] for qubit in qargs)

        for node in dag.topological_op_nodes():
            if not isinstance(node.op, (SwapGate, PermutationGate)):
                new_dag.apply_operation_back(
                    node.op, _apply_mapping(node.qargs), node.cargs, check=False
                )
            elif getattr(node.op, "condition", None) is not None:
                new_dag.apply_operation_back(
                    node.op, _apply_mapping(node.qargs), node.cargs, check=False
                )
            elif isinstance(node.op, SwapGate):
                index_0 = dag.find_bit(node.qargs[0]).index
                index_1 = dag.find_bit(node.qargs[1]).index
                qubit_mapping[index_1], qubit_mapping[index_0] = (
                    qubit_mapping[index_0],
                    qubit_mapping[index_1],
                )
            elif isinstance(node.op, PermutationGate):
                starting_indices = [qubit_mapping[dag.find_bit(qarg).index] for qarg in node.qargs]
                pattern = node.op.params[0]
                pattern_indices = [qubit_mapping[idx] for idx in pattern]
                for i, j in zip(starting_indices, pattern_indices):
                    qubit_mapping[i] = j
        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = input_qubit_mapping

        new_layout = Layout({dag.qubits[out]: idx for idx, out in enumerate(qubit_mapping)})
        if current_layout := self.property_set["virtual_permutation_layout"]:
            self.property_set["virtual_permutation_layout"] = new_layout.compose(
                current_layout.inverse(dag.qubits, dag.qubits), dag.qubits
            )
        else:
            self.property_set["virtual_permutation_layout"] = new_layout

        new_dag._final_permutation = dag._final_permutation.compose_with_permutation(
            qubit_mapping, front=True
        )

        return new_dag

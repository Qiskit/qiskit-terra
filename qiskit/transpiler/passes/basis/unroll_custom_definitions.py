# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unrolls instructions with custom definitions."""

from qiskit.exceptions import QiskitError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.circuit import ControlledGate, ControlFlowOp
from qiskit.converters.circuit_to_dag import circuit_to_dag


class UnrollCustomDefinitions(TransformationPass):
    """Unrolls instructions with custom definitions."""

    def __init__(self, equivalence_library, basis_gates=None, target=None):
        """Unrolls instructions with custom definitions.

        Args:
            equivalence_library (EquivalenceLibrary): The equivalence library
                which will be used by the BasisTranslator pass. (Instructions in
                this library will not be unrolled by this pass.)
            basis_gates (Optional[list[str]]): Target basis names to unroll to, e.g. `['u3', 'cx']`.
                Ignored if ``target`` is also specified.
            target (Optional[Target]): The :class:`~.Target` object corresponding to the compilation
                target. When specified, any argument specified for ``basis_gates`` is ignored.
        """

        super().__init__()
        self._equiv_lib = equivalence_library
        self._basis_gates = basis_gates
        self._target = target

    def run(self, dag):
        """Run the UnrollCustomDefinitions pass on `dag`.

        Args:
            dag (DAGCircuit): input dag

        Raises:
            QiskitError: if unable to unroll given the basis due to undefined
            decomposition rules (such as a bad basis) or excessive recursion.

        Returns:
            DAGCircuit: output unrolled dag
        """

        if self._basis_gates is None and self._target is None:
            return dag

        if self._target is None:
            basic_insts = {"measure", "reset", "barrier", "snapshot", "delay"}
            device_insts = basic_insts | set(self._basis_gates)
        qubit_mapping = {bit: idx for idx, bit in enumerate(dag.qubits)}

        for node in dag.op_nodes():
            if isinstance(node.op, ControlFlowOp):
                node.op = control_flow.map_blocks(self.run, node.op)
                continue

            if getattr(node.op, "_directive", False):
                continue

            if dag.has_calibration_for(node):
                continue

            controlled_gate_open_ctrl = isinstance(node.op, ControlledGate) and node.op._open_ctrl
            if not controlled_gate_open_ctrl:
                inst_supported = (
                    self._target.instruction_supported(
                        operation_name=node.op.name,
                        qargs=tuple(qubit_mapping[x] for x in node.qargs),
                    )
                    if self._target is not None
                    else node.name in device_insts
                )
                if inst_supported or self._equiv_lib.has_entry(node.op):
                    continue
            try:
                rule = node.op.definition.data
            except TypeError as err:
                raise QiskitError(f"Error decomposing node {node.name}.") from err
            except AttributeError:
                # definition is None
                rule = None

            if not rule:
                if rule == []:
                    dag.remove_op_node(node)
                    continue

                # opaque node
                raise QiskitError(
                    "Cannot unroll the circuit to the given basis, %s. "
                    "Instruction %s not found in equivalence library "
                    "and no rule found to expand." % (str(self._basis_gates), node.op.name)
                )
            decomposition = circuit_to_dag(node.op.definition)
            unrolled_dag = UnrollCustomDefinitions(
                self._equiv_lib, self._basis_gates, target=self._target
            ).run(decomposition)
            dag.substitute_node_with_dag(node, unrolled_dag)

        return dag

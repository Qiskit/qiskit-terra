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

"""Check if a DAG circuit is already mapped to a coupling map."""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.circuit.controlflow import ControlFlowOp


class CheckMap(AnalysisPass):
    """Check if a DAG circuit is already mapped to a coupling map.

    Check if a DAGCircuit is mapped to `coupling_map` by checking that all
    2-qubit interactions are laid out to be on adjacent qubits in the global coupling
    map of the device, setting the property ``is_swap_mapped`` to ``True`` or ``False``
    accordingly. Note this does not validate directionality of the connectivity between
    qubits. If you need to check gates are implemented in a native direction
    for a target use the :class:`~.CheckGateDirection` pass instead.
    """

    def __init__(self, coupling_map=None, target=None):
        """CheckMap initializer.

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
            target (Target): A target representing the target backend, if both
                ``coupling_map`` and this are specified then this argument will take
                precedence and ``coupling_map`` will be ignored.
        """
        super().__init__()
        if coupling_map is None and target is None:
            self.qargs = None
        else:
            self.qargs = set()
            if target is not None:
                if target.qargs is not None:
                    for edge in target.qargs:
                        if len(edge) == 2:
                            self.qargs.add(edge)
                            self.qargs.add((edge[1], edge[0]))
            else:
                for edge in coupling_map.get_edges():
                    self.qargs.add(edge)
                    self.qargs.add((edge[1], edge[0]))

    def run(self, dag):
        """Run the CheckMap pass on `dag`.

        If `dag` is mapped to `coupling_map`, the property
        `is_swap_mapped` is set to True (or to False otherwise).

        Args:
            dag (DAGCircuit): DAG to map.
        """
        from qiskit.converters import circuit_to_dag

        self.property_set["is_swap_mapped"] = True

        if not self.qargs:
            return

        qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        for node in dag.op_nodes(include_directives=False):
            is_controlflow_op = isinstance(node.op, ControlFlowOp)
            if len(node.qargs) == 2 and not is_controlflow_op:
                if dag.has_calibration_for(node):
                    continue
                physical_q0 = qubit_indices[node.qargs[0]]
                physical_q1 = qubit_indices[node.qargs[1]]
                if (physical_q0, physical_q1) not in self.qargs:
                    self.property_set["check_map_msg"] = "{}({}, {}) failed".format(
                        node.name,
                        physical_q0,
                        physical_q1,
                    )
                    self.property_set["is_swap_mapped"] = False
                    return
            elif is_controlflow_op:
                order = [qubit_indices[bit] for bit in node.qargs]
                for block in node.op.blocks:
                    dag_block = circuit_to_dag(block)
                    mapped_dag = dag.copy_empty_like()
                    mapped_dag.compose(dag_block, qubits=order)
                    self.run(mapped_dag)
                    if not self.property_set["is_swap_mapped"]:
                        return

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper function for converting a dag circuit to a dag dependency"""
from qiskit.dagcircuit.dagdependency_v2 import _DAGDependencyV2
from .circuit_conversion_data import CircuitConversionData


def _dag_to_dagdependency_v2_with_data(dag, *, create_conversion_data=False):
    """Build a ``_DAGDependencyV2`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.
        create_conversion_data (bool): whether to construct mappings
            between nodes in the input and the output circuits.

    Returns:
        A tuple consisting of the DAGDependency representation of the input
        circuit and the additional data. This data stores mappings between
        nodes in the input and the output circuits when ``create_conversion_data``
        is ``True`` and ``None`` otherwise.

    """
    conversion_data = CircuitConversionData() if create_conversion_data else None

    dagdependency = _DAGDependencyV2()
    dagdependency.name = dag.name
    dagdependency.metadata = dag.metadata
    dagdependency.global_phase = dag.global_phase
    dagdependency.calibrations = dag._calibrations_prop

    dagdependency.add_qubits(dag.qubits)
    dagdependency.add_clbits(dag.clbits)

    for register in dag.qregs.values():
        dagdependency.add_qreg(register)

    for register in dag.cregs.values():
        dagdependency.add_creg(register)

    for in_node in dag.topological_op_nodes():
        out_node = dagdependency.apply_operation_back(
            in_node.op.copy(), in_node.qargs, in_node.cargs
        )
        if create_conversion_data:
            conversion_data.store_mapping(in_node, out_node)

    return dagdependency, conversion_data


def _dag_to_dagdependency_v2(dag):
    """Build a ``_DAGDependencyV2`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.

    Return:
        _DAGDependencyV2: the DAG representing the input circuit as a dag dependency.
    """
    dagdependency, _ = _dag_to_dagdependency_v2_with_data(dag)
    return dagdependency

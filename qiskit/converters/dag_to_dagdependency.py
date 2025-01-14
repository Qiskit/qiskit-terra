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
from qiskit.dagcircuit.dagdependency import DAGDependency
from .circuit_conversion_data import CircuitConversionData


def dag_to_dagdependency_with_data(
    dag, create_preds_and_succs=True, *, create_conversion_data=False
):
    """Build a ``DAGDependency`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.
        create_preds_and_succs (bool): whether to construct lists of
            predecessors and successors for every node.
        create_conversion_data (bool): whether to construct mappings
            between nodes in the input and the output circuits.

    Returns:
        A tuple consisting of the DAGDependency representation of the input
        circuit and the additional data. This data stores mappings between
        nodes in the input and the output circuits when ``create_conversion_data``
        is ``True`` and ``None`` otherwise.

    """
    conversion_data = CircuitConversionData() if create_conversion_data else None

    dagdependency = DAGDependency()
    dagdependency.name = dag.name
    dagdependency.metadata = dag.metadata

    dagdependency.add_qubits(dag.qubits)
    dagdependency.add_clbits(dag.clbits)

    for register in dag.qregs.values():
        dagdependency.add_qreg(register)

    for register in dag.cregs.values():
        dagdependency.add_creg(register)

    for from_node in dag.topological_op_nodes():
        # Get arguments for classical control (if any)
        inst = from_node.op.copy()
        to_node = dagdependency.add_op_node(inst, from_node.qargs, from_node.cargs)
        if create_conversion_data:
            conversion_data.store_mapping(from_node, to_node)

    if create_preds_and_succs:
        dagdependency._add_predecessors()
        dagdependency._add_successors()

    # copy metadata
    dagdependency.global_phase = dag.global_phase
    dagdependency._calibrations_prop = dag._calibrations_prop

    return dagdependency, conversion_data


def dag_to_dagdependency(dag, create_preds_and_succs=True):
    """Build a ``DAGDependency`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.
        create_preds_and_succs (bool): whether to construct lists of
            predecessors and successors for every node.

    Return:
        DAGDependency: the DAG representing the input circuit as a dag dependency.
    """
    dagdependency, _ = dag_to_dagdependency_with_data(
        dag, create_preds_and_succs=create_preds_and_succs
    )
    return dagdependency

# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module contains common utils for disjoint coupling maps."""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar
import uuid

import rustworkx as rx
from qiskit.dagcircuit import DAGOpNode

from qiskit.circuit import Qubit, Barrier, Clbit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOutNode
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.target import Target
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.layout import vf2_utils

T = TypeVar("T")


def run_pass_over_connected_components(
    dag: DAGCircuit,
    components_source: Target | CouplingMap,
    run_func: Callable[[DAGCircuit, CouplingMap], T],
) -> list[T]:
    """Run a transpiler pass inner function over mapped components."""
    if isinstance(components_source, Target):
        coupling_map = components_source.build_coupling_map(filter_idle_qubits=True)
    else:
        coupling_map = components_source
    cmap_components = coupling_map.connected_components()
    # If graph is connected we only need to run the pass once
    if len(cmap_components) == 1:
        if dag.num_qubits() > cmap_components[0].size():
            raise TranspilerError(
                "A connected component of the DAGCircuit is too large for any of the connected "
                "components in the coupling map."
            )
        return [run_func(dag, cmap_components[0])]
    dag_components = separate_dag(dag)
    mapped_components = map_components(dag_components, cmap_components)
    out_component_pairs = []
    for cmap_index, dags in mapped_components.items():
        # Take the first dag from the mapped dag components and then
        # compose it with any other dag components that are operating on the
        # same coupling map connected component. This results in a subcircuit
        # of possibly disjoint circuit components which we will run the layout
        # pass on.
        out_dag = dag_components[dags.pop()]
        for dag_index in dags:
            dag = dag_components[dag_index]
            out_dag.add_qubits(dag.qubits)
            out_dag.add_clbits(dag.clbits)
            for qreg in dag.qregs:
                out_dag.add_qreg(qreg)
            for creg in dag.cregs:
                out_dag.add_creg(creg)
            out_dag.compose(dag, qubits=dag.qubits, clbits=dag.clbits)
        out_component_pairs.append((out_dag, cmap_components[cmap_index]))
    res = [run_func(out_dag, cmap) for out_dag, cmap in out_component_pairs]
    return res


def map_components(
    dag_components: list[DAGCircuit], cmap_components: list[CouplingMap]
) -> dict[int, list[int]]:
    """Returns a map where the key is the index of each connected component in cmap_components and
    the value is a list of indices from dag_components which should be placed onto it."""
    free_qubits = {index: len(cmap.graph) for index, cmap in enumerate(cmap_components)}
    out_mapping = defaultdict(list)

    for dag_index, dag in sorted(
        enumerate(dag_components), key=lambda x: x[1].num_qubits(), reverse=True
    ):
        for cmap_index in sorted(
            range(len(cmap_components)), key=lambda index: free_qubits[index], reverse=True
        ):
            # TODO: Improve heuristic to involve connectivity and estimate
            # swap cost
            if dag.num_qubits() <= free_qubits[cmap_index]:
                out_mapping[cmap_index].append(dag_index)
                free_qubits[cmap_index] -= dag.num_qubits()
                break
        else:
            raise TranspilerError(
                "A connected component of the DAGCircuit is too large for any of the connected "
                "components in the coupling map."
            )
    return out_mapping


def split_barriers(dag: DAGCircuit):
    """Mutate an input dag to split barriers into single qubit barriers."""
    for node in dag.op_nodes(Barrier):
        num_qubits = len(node.qargs)
        if num_qubits == 1:
            continue
        if node.op.label:
            barrier_uuid = f"{node.op.label}_uuid={uuid.uuid4()}"
        else:
            barrier_uuid = f"_none_uuid={uuid.uuid4()}"
        split_dag = DAGCircuit()
        split_dag.add_qubits([Qubit() for _ in range(num_qubits)])
        for i in range(num_qubits):
            split_dag.apply_operation_back(
                Barrier(1, label=barrier_uuid),
                qargs=(split_dag.qubits[i],),
                check=False,
            )
        dag.substitute_node_with_dag(node, split_dag)


def combine_barriers(dag: DAGCircuit, retain_uuid: bool = True):
    """Mutate input dag to combine barriers with UUID labels into a single barrier."""
    qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
    uuid_map: dict[uuid.UUID, DAGOpNode] = {}
    for node in dag.op_nodes(Barrier):
        if node.op.label:
            if "_uuid=" in node.op.label:
                barrier_uuid = node.op.label
            else:
                continue
            if barrier_uuid in uuid_map:
                other_node = uuid_map[barrier_uuid]
                num_qubits = len(other_node.qargs) + len(node.qargs)
                new_op = Barrier(num_qubits, label=barrier_uuid)
                new_node = dag.replace_block_with_op([node, other_node], new_op, qubit_indices)
                uuid_map[barrier_uuid] = new_node
            else:
                uuid_map[barrier_uuid] = node
    if not retain_uuid:
        for node in dag.op_nodes(Barrier):
            if isinstance(node.op.label, str) and node.op.label.startswith("_none_uuid="):
                node.op.label = None
            elif isinstance(node.op.label, str) and "_uuid=" in node.op.label:
                original_label = "_uuid=".join(node.op.label.split("_uuid=")[:-1])
                node.op.label = original_label


def require_layout_isolated_to_component(dag: DAGCircuit, components_source: Target | CouplingMap):
    """
    Check that the layout of the dag does not require connectivity across connected components
    in the CouplingMap

    Args:
        dag: DAGCircuit to check.
        components_source: Target to check against.

    Raises:
        TranspilerError: Chosen layout is not valid for the target disjoint connectivity.
    """
    if isinstance(components_source, Target):
        coupling_map = components_source.build_coupling_map(filter_idle_qubits=True)
    else:
        coupling_map = components_source
    component_sets = [set(x.graph.nodes()) for x in coupling_map.connected_components()]
    for inst in dag.two_qubit_ops():
        component_index = None
        for i, component_set in enumerate(component_sets):
            if dag.find_bit(inst.qargs[0]).index in component_set:
                component_index = i
                break
        if dag.find_bit(inst.qargs[1]).index not in component_sets[component_index]:
            raise TranspilerError(
                "The circuit has an invalid layout as two qubits need to interact in disconnected "
                "components of the coupling map. The physical qubit "
                f"{dag.find_bit(inst.qargs[1]).index} needs to interact with the "
                f"qubit {dag.find_bit(inst.qargs[0]).index} and they belong to different components"
            )


def separate_dag(dag: DAGCircuit) -> list[DAGCircuit]:
    """Separate a dag circuit into it's connected components."""
    # Split barriers into single qubit barriers before splitting connected components
    split_barriers(dag)
    im_graph, _, qubit_map, __ = vf2_utils.build_interaction_graph(dag)
    connected_components = rx.weakly_connected_components(im_graph)
    component_qubits = []
    for component in connected_components:
        component_qubits.append({qubit_map[x] for x in component})

    qubits = set(dag.qubits)

    decomposed_dags = []
    for dag_qubits in component_qubits:
        new_dag = dag.copy_empty_like()
        new_dag.remove_qubits(*qubits - dag_qubits)
        new_dag.global_phase = 0
        for node in dag.topological_op_nodes():
            if dag_qubits.issuperset(node.qargs):
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs, check=False)
        idle_clbits = []
        for bit, node in new_dag.input_map.items():
            succ_node = next(new_dag.successors(node))
            if isinstance(succ_node, DAGOutNode) and isinstance(succ_node.wire, Clbit):
                idle_clbits.append(bit)
        new_dag.remove_clbits(*idle_clbits)
        combine_barriers(new_dag)
        decomposed_dags.append(new_dag)
    # Reverse split barriers on input dag to avoid leaking out internal transformations as
    # part of splitting
    combine_barriers(dag, retain_uuid=False)
    return decomposed_dags

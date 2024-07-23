# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module contains common utils for vf2 layout passes."""
from __future__ import annotations

from collections import defaultdict
import statistics
import random

import numpy as np
from rustworkx import PyDiGraph, PyGraph, connected_components

from qiskit.circuit import ControlFlowOp, ForLoopOp
from qiskit.converters import circuit_to_dag
from qiskit._accelerate import vf2_layout
from qiskit._accelerate.nlayout import NLayout
from qiskit._accelerate.error_map import ErrorMap


def build_interaction_graph(dag, strict_direction=True):
    """Build an interaction graph from a dag."""
    im_graph = PyDiGraph(multigraph=False) if strict_direction else PyGraph(multigraph=False)
    im_graph_node_map = {}
    reverse_im_graph_node_map = {}

    class MultiQEncountered(Exception):
        """Used to singal an error-status return from the DAG visitor."""

    def _visit(dag, weight, wire_map):
        for node in dag.op_nodes(include_directives=False):
            if isinstance(node.op, ControlFlowOp):
                if isinstance(node.op, ForLoopOp):
                    inner_weight = len(node.op.params[0]) * weight
                else:
                    inner_weight = weight
                for block in node.op.blocks:
                    inner_wire_map = {
                        inner: wire_map[outer] for outer, inner in zip(node.qargs, block.qubits)
                    }
                    _visit(circuit_to_dag(block), inner_weight, inner_wire_map)
                continue
            len_args = len(node.qargs)
            qargs = [wire_map[q] for q in node.qargs]
            if len_args == 1:
                if qargs[0] not in im_graph_node_map:
                    weights: dict[str, int] = defaultdict(int)
                    weights[node.name] += weight
                    im_graph_node_map[qargs[0]] = im_graph.add_node(weights)
                    reverse_im_graph_node_map[im_graph_node_map[qargs[0]]] = qargs[0]
                else:
                    im_graph[im_graph_node_map[qargs[0]]][node.op.name] += weight
            if len_args == 2:
                if qargs[0] not in im_graph_node_map:
                    im_graph_node_map[qargs[0]] = im_graph.add_node(defaultdict(int))
                    reverse_im_graph_node_map[im_graph_node_map[qargs[0]]] = qargs[0]
                if qargs[1] not in im_graph_node_map:
                    im_graph_node_map[qargs[1]] = im_graph.add_node(defaultdict(int))
                    reverse_im_graph_node_map[im_graph_node_map[qargs[1]]] = qargs[1]
                edge = (im_graph_node_map[qargs[0]], im_graph_node_map[qargs[1]])
                if im_graph.has_edge(*edge):
                    im_graph.get_edge_data(*edge)[node.name] += weight
                else:
                    weights = defaultdict(int)
                    weights[node.name] += weight
                    im_graph.add_edge(*edge, weights)
            if len_args > 2:
                raise MultiQEncountered()

    try:
        _visit(dag, 1, {bit: bit for bit in dag.qubits})
    except MultiQEncountered:
        return None
    # Remove components with no 2q interactions from interaction graph
    # these will be evaluated separately independently of scoring isomorphic
    # mappings. This is not done for strict direction because for post layout
    # we need to factor in local operation constraints when evaluating a graph
    free_nodes = {}
    if not strict_direction:
        conn_comp = connected_components(im_graph)
        for comp in conn_comp:
            if len(comp) == 1:
                index = comp.pop()
                free_nodes[index] = im_graph[index]
                im_graph.remove_node(index)

    return im_graph, im_graph_node_map, reverse_im_graph_node_map, free_nodes


def build_edge_list(im_graph):
    """Generate an edge list for scoring."""
    return vf2_layout.EdgeList(
        [((edge[0], edge[1]), sum(edge[2].values())) for edge in im_graph.edge_index_map().values()]
    )


def build_bit_list(im_graph, bit_map):
    """Generate a bit list for scoring."""
    bit_list = np.zeros(len(im_graph), dtype=np.int32)
    for node_index in bit_map.values():
        try:
            bit_list[node_index] = sum(im_graph[node_index].values())
        # If node_index not in im_graph that means there was a standalone
        # node we will score/sort separately outside the vf2 mapping, so we
        # can skip the hole
        except IndexError:
            pass
    return bit_list


def score_layout(
    avg_error_map,
    layout_mapping,
    bit_map,
    _reverse_bit_map,
    im_graph,
    strict_direction=False,
    run_in_parallel=False,
    edge_list=None,
    bit_list=None,
):
    """Score a layout given an average error map."""
    if layout_mapping:
        size = max(max(layout_mapping), max(layout_mapping.values()))
    else:
        size = 0
    nlayout = NLayout(layout_mapping, size + 1, size + 1)
    if bit_list is None:
        bit_list = build_bit_list(im_graph, bit_map)
    if edge_list is None:
        edge_list = build_edge_list(im_graph)
    return vf2_layout.score_layout(
        bit_list, edge_list, avg_error_map, nlayout, strict_direction, run_in_parallel
    )


def build_average_error_map(target, properties, coupling_map):
    """Build an average error map used for scoring layouts pre-basis translation."""
    num_qubits = 0
    if target is not None and target.qargs is not None:
        num_qubits = target.num_qubits
        avg_map = ErrorMap(len(target.qargs))
    elif coupling_map is not None:
        num_qubits = coupling_map.size()
        avg_map = ErrorMap(num_qubits + coupling_map.graph.num_edges())
    else:
        # If coupling map is not defined almost certainly we don't have any
        # data to build an error map, but just in case initialize an empty
        # object
        avg_map = ErrorMap(0)
    built = False
    if target is not None and target.qargs is not None:
        for qargs in target.qargs:
            if qargs is None:
                continue
            qarg_error = 0.0
            count = 0
            for op in target.operation_names_for_qargs(qargs):
                inst_props = target[op].get(qargs, None)
                if inst_props is not None and inst_props.error is not None:
                    count += 1
                    qarg_error += inst_props.error
            if count > 0:
                if len(qargs) == 1:
                    qargs = (qargs[0], qargs[0])
                avg_map.add_error(qargs, qarg_error / count)
                built = True
    elif properties is not None:
        errors = defaultdict(list)
        for qubit in range(len(properties.qubits)):
            errors[(qubit,)].append(properties.readout_error(qubit))
        for gate in properties.gates:
            qubits = tuple(gate.qubits)
            for param in gate.parameters:
                if param.name == "gate_error":
                    errors[qubits].append(param.value)
        for k, v in errors.items():
            if len(k) == 1:
                qargs = (k[0], k[0])
            else:
                qargs = k
            # If the properties payload contains an index outside the number of qubits
            # the properties are invalid for the given input. This normally happens either
            # with a malconstructed properties payload or if the faulty qubits feature of
            # BackendV1/BackendPropeties is being used. In such cases we map noise characteristics
            # so we should just treat the mapping as an ideal case.
            if qargs[0] >= num_qubits or qargs[1] >= num_qubits:
                continue
            avg_map.add_error(qargs, statistics.mean(v))
            built = True
    # if there are no error rates in the target we should fallback to using the degree heuristic
    # used for a coupling map. To do this we can build the coupling map from the target before
    # running the fallback heuristic
    if not built and target is not None and coupling_map is None:
        coupling_map = target.build_coupling_map()
    if not built and coupling_map is not None:
        for qubit in range(num_qubits):
            avg_map.add_error(
                (qubit, qubit),
                (coupling_map.graph.out_degree(qubit) + coupling_map.graph.in_degree(qubit))
                / num_qubits,
            )
        for edge in coupling_map.graph.edge_list():
            avg_map.add_error(edge, (avg_map[edge[0], edge[0]] + avg_map[edge[1], edge[1]]) / 2)
            built = True
    if built:
        return avg_map
    else:
        return None


def shuffle_coupling_graph(coupling_map, seed, strict_direction=True):
    """Create a shuffled coupling graph from a coupling map."""
    if strict_direction:
        cm_graph = coupling_map.graph
    else:
        cm_graph = coupling_map.graph.to_undirected(multigraph=False)
    cm_nodes = list(cm_graph.node_indexes())
    if seed != -1:
        random.Random(seed).shuffle(cm_nodes)
        shuffled_cm_graph = type(cm_graph)()
        shuffled_cm_graph.add_nodes_from(cm_nodes)
        new_edges = [(cm_nodes[edge[0]], cm_nodes[edge[1]]) for edge in cm_graph.edge_list()]
        shuffled_cm_graph.add_edges_from_no_data(new_edges)
        cm_nodes = [k for k, v in sorted(enumerate(cm_nodes), key=lambda item: item[1])]
        cm_graph = shuffled_cm_graph
    return cm_graph, cm_nodes


def map_free_qubits(
    free_nodes, partial_layout, num_physical_qubits, reverse_bit_map, avg_error_map
):
    """Add any free nodes to a layout."""
    if not free_nodes:
        return partial_layout
    if avg_error_map is not None:
        free_qubits = sorted(
            set(range(num_physical_qubits)) - partial_layout.get_physical_bits().keys(),
            key=lambda bit: avg_error_map.get((bit, bit), 1.0),
        )
    # If no error map is available this means there is no scoring heuristic available for this
    # backend and we can just randomly pick a free qubit
    else:
        free_qubits = list(
            set(range(num_physical_qubits)) - partial_layout.get_physical_bits().keys()
        )
    for im_index in sorted(free_nodes, key=lambda x: sum(free_nodes[x].values())):
        if not free_qubits:
            return None
        selected_qubit = free_qubits.pop(0)
        partial_layout.add(reverse_bit_map[im_index], selected_qubit)
    return partial_layout

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""VF2PostLayout pass to find a layout after transpile using subgraph isomorphism"""
from __future__ import annotations

from enum import Enum
import logging
import inspect
import itertools
import time

from rustworkx import PyDiGraph, vf2_mapping, PyGraph

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.transpiler.passes.layout import vf2_utils


logger = logging.getLogger(__name__)


class VF2PostLayoutStopReason(Enum):
    """Stop reasons for VF2PostLayout pass."""

    SOLUTION_FOUND = "solution found"
    NO_BETTER_SOLUTION_FOUND = "no better solution found"
    NO_SOLUTION_FOUND = "nonexistent solution"
    MORE_THAN_2Q = ">2q gates in basis"


def _target_match(node_a, node_b):
    # Node A is the set of operations in the target. Node B is the count dict
    # of operations on the node or edge in the circuit.
    if isinstance(node_a, set):
        return node_a.issuperset(node_b.keys())
    # Node A is the count dict of operations on the node or edge in the circuit
    # Node B is the set of operations in the target on the same qubit(s).
    else:
        return set(node_a).issubset(node_b)


class VF2PostLayout(AnalysisPass):
    """A pass for improving an existing Layout after transpilation of a circuit onto a
    Coupling graph, as a subgraph isomorphism problem, solved by VF2++.

    Unlike the :class:`~.VF2Layout` transpiler pass which is designed to find an
    initial layout for a circuit early in the transpilation pipeline this transpiler
    pass is designed to try and find a better layout after transpilation is complete.
    The initial layout phase of the transpiler doesn't have as much information available
    as we do after transpilation. This pass is designed to be paired in a similar pipeline
    as the layout passes. This pass will strip any idle wires from the circuit, use VF2
    to find a subgraph in the coupling graph for the circuit to run on with better fidelity
    and then update the circuit layout to use the new qubits. The algorithm used in this
    pass is described in `arXiv:2209.15512 <https://arxiv.org/abs/2209.15512>`__.

    If a solution is found that means there is a lower error layout available for the
    circuit. If a solution is found the layout will be set in the property set as
    ``property_set['post_layout']``. However, if no solution or no better solution is found, no
    ``property_set['post_layout']`` is set. The stopping reason is
    set in ``property_set['VF2PostLayout_stop_reason']`` in all the cases and will be
    one of the values enumerated in ``VF2PostLayoutStopReason`` which has the
    following values:

        * ``"solution found"``: If a solution was found.
        * ``"no better solution found"``: If the initial layout of the circuit is the best solution.
        * ``"nonexistent solution"``: If no solution was found.
        * ``">2q gates in basis"``: If VF2PostLayout can't work with the basis of the circuit.

    By default, this pass will construct a heuristic scoring map based on
    the error rates in the provided ``target`` (or ``properties`` if ``target``
    is not provided). However, analysis passes can be run prior to this pass
    and set ``vf2_avg_error_map`` in the property set with a :class:`~.ErrorMap`
    instance. If a value is ``NaN`` that is treated as an ideal edge
    For example if an error map is created as::

        from qiskit.transpiler.passes.layout.vf2_utils import ErrorMap

        error_map = ErrorMap(3)
        error_map.add_error((0, 0), 0.0024)
        error_map.add_error((0, 1), 0.01)
        error_map.add_error((1, 1), 0.0032)

    that represents the error map for a 2 qubit target, where the avg 1q error
    rate is ``0.0024`` on qubit 0 and ``0.0032`` on qubit 1. Then the avg 2q
    error rate for gates that operate on (0, 1) is 0.01 and (1, 0) is not
    supported by the target. This will be used for scoring if it's set as the
    ``vf2_avg_error_map`` key in the property set when :class:`~.VF2PostLayout`
    is run.
    """

    def __init__(
        self,
        target=None,
        coupling_map=None,
        properties=None,
        seed=None,
        call_limit=None,
        time_limit=None,
        strict_direction=True,
        max_trials=0,
    ):
        """Initialize a ``VF2PostLayout`` pass instance

        Args:
            target (Target): A target representing the backend device to run ``VF2PostLayout`` on.
                If specified it will supersede a set value for ``properties`` and
                ``coupling_map``.
            coupling_map (CouplingMap): Directed graph representing a coupling map.
            properties (BackendProperties): The backend properties for the backend. If
                :meth:`~qiskit.providers.models.BackendProperties.readout_error` is available
                it is used to score the layout.
            seed (int): Sets the seed of the PRNG. -1 Means no node shuffling.
            call_limit (int): The number of state visits to attempt in each execution of
                VF2.
            time_limit (float): The total time limit in seconds to run ``VF2PostLayout``
            strict_direction (bool): Whether the pass is configured to follow
                the strict direction in the coupling graph. If this is set to
                false, the pass will treat any edge in the coupling graph as
                a weak edge and the interaction graph will be undirected. For
                the purposes of evaluating layouts the avg error rate for
                each qubit and 2q link will be used. This enables the pass to be
                run prior to basis translation and work with any 1q and 2q operations.
                However, if ``strict_direction=True`` the pass expects the input
                :class:`~.DAGCircuit` object to :meth:`~.VF2PostLayout.run` to be in
                the target set of instructions.
            max_trials (int): The maximum number of trials to run VF2 to find
                a layout. A value of ``0`` (the default) means 'unlimited'.

        Raises:
            TypeError: At runtime, if neither ``coupling_map`` or ``target`` are provided.
        """
        super().__init__()
        self.target = target
        self.coupling_map = coupling_map
        self.properties = properties
        self.call_limit = call_limit
        self.time_limit = time_limit
        self.max_trials = max_trials
        self.seed = seed
        self.strict_direction = strict_direction
        self.avg_error_map = None

    def run(self, dag):
        """run the layout method"""
        if self.target is None and (self.coupling_map is None or self.properties is None):
            raise TranspilerError(
                "A target must be specified or a coupling map and properties must be provided"
            )
        if not self.strict_direction:
            self.avg_error_map = self.property_set["vf2_avg_error_map"]
            if self.avg_error_map is None:
                self.avg_error_map = vf2_utils.build_average_error_map(
                    self.target, self.properties, self.coupling_map
                )

        result = vf2_utils.build_interaction_graph(dag, self.strict_direction)
        if result is None:
            self.property_set["VF2PostLayout_stop_reason"] = VF2PostLayoutStopReason.MORE_THAN_2Q
            return
        im_graph, im_graph_node_map, reverse_im_graph_node_map, free_nodes = result
        scoring_bit_list = vf2_utils.build_bit_list(im_graph, im_graph_node_map)
        scoring_edge_list = vf2_utils.build_edge_list(im_graph)

        if self.target is not None:
            # If qargs is None then target is global and ideal so no
            # scoring is needed
            if self.target.qargs is None:
                return
            if self.strict_direction:
                cm_graph = PyDiGraph(multigraph=False)
            else:
                cm_graph = PyGraph(multigraph=False)
            # If None is present in qargs there are globally defined ideal operations
            # we should add these to all entries based on the number of qubits, so we
            # treat that as a valid operation even if there is no scoring for the
            # strict direction case
            global_ops: dict[int, list] | None = None
            if None in self.target.qargs:
                global_ops = {1: [], 2: []}
                for op in self.target.operation_names_for_qargs(None):
                    operation = self.target.operation_for_name(op)
                    # If operation is a class this is a variable width ideal instruction
                    # so we treat it as available on both 1 and 2 qubits
                    if inspect.isclass(operation):
                        global_ops[1].append(op)
                        global_ops[2].append(op)
                    else:
                        num_qubits = operation.num_qubits
                        if num_qubits in global_ops:
                            global_ops[num_qubits].append(op)
            op_names = []
            for i in range(self.target.num_qubits):
                try:
                    entry = set(self.target.operation_names_for_qargs((i,)))
                except KeyError:
                    entry = set()
                if global_ops is not None:
                    entry.update(global_ops[1])
                op_names.append(entry)
            cm_graph.add_nodes_from(op_names)
            for qargs in self.target.qargs:
                len_args = len(qargs)
                # If qargs == 1 we already populated it and if qargs > 2 there are no instructions
                # using those in the circuit because we'd have already returned by this point
                if len_args == 2:
                    ops = set(self.target.operation_names_for_qargs(qargs))
                    if global_ops is not None:
                        ops.update(global_ops[2])
                    cm_graph.add_edge(qargs[0], qargs[1], ops)
            cm_nodes = list(cm_graph.node_indexes())
            # Filter qubits without any supported operations. If they
            # don't support any operations, they're not valid for layout selection.
            # This is only needed in the undirected case because in strict direction
            # mode the node matcher will not match since none of the circuit ops
            # will match the cmap ops.
            if not self.strict_direction:
                has_operations = set(itertools.chain.from_iterable(self.target.qargs))
                to_remove = set(cm_graph.node_indices()).difference(has_operations)
                if to_remove:
                    cm_graph.remove_nodes_from(list(to_remove))
        else:
            cm_graph, cm_nodes = vf2_utils.shuffle_coupling_graph(
                self.coupling_map, self.seed, self.strict_direction
            )

        logger.debug("Running VF2 to find post transpile mappings")
        if self.target and self.strict_direction:
            mappings = vf2_mapping(
                cm_graph,
                im_graph,
                node_matcher=_target_match,
                edge_matcher=_target_match,
                subgraph=True,
                id_order=False,
                induced=False,
                call_limit=self.call_limit,
            )
        else:
            mappings = vf2_mapping(
                cm_graph,
                im_graph,
                subgraph=True,
                id_order=False,
                induced=False,
                call_limit=self.call_limit,
            )
        chosen_layout = None
        try:
            if self.strict_direction:
                initial_layout = Layout({bit: index for index, bit in enumerate(dag.qubits)})
                chosen_layout_score = self._score_layout(
                    initial_layout,
                    im_graph_node_map,
                    reverse_im_graph_node_map,
                    im_graph,
                )
            else:
                initial_layout = {
                    im_graph_node_map[bit]: index
                    for index, bit in enumerate(dag.qubits)
                    if bit in im_graph_node_map
                }
                chosen_layout_score = vf2_utils.score_layout(
                    self.avg_error_map,
                    initial_layout,
                    im_graph_node_map,
                    reverse_im_graph_node_map,
                    im_graph,
                    self.strict_direction,
                    edge_list=scoring_edge_list,
                    bit_list=scoring_bit_list,
                )
            chosen_layout = initial_layout
            stop_reason = VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND
        # Circuit not in basis so we have nothing to compare against return here
        except KeyError:
            self.property_set["VF2PostLayout_stop_reason"] = (
                VF2PostLayoutStopReason.NO_SOLUTION_FOUND
            )
            return

        logger.debug("Initial layout has score %s", chosen_layout_score)

        start_time = time.time()
        trials = 0
        for mapping in mappings:
            trials += 1
            logger.debug("Running trial: %s", trials)
            layout_mapping = {im_i: cm_nodes[cm_i] for cm_i, im_i in mapping.items()}
            if self.strict_direction:
                layout = Layout(
                    {reverse_im_graph_node_map[k]: v for k, v in layout_mapping.items()}
                )
                layout_score = self._score_layout(
                    layout, im_graph_node_map, reverse_im_graph_node_map, im_graph
                )
            else:
                layout_score = vf2_utils.score_layout(
                    self.avg_error_map,
                    layout_mapping,
                    im_graph_node_map,
                    reverse_im_graph_node_map,
                    im_graph,
                    self.strict_direction,
                    edge_list=scoring_edge_list,
                    bit_list=scoring_bit_list,
                )
            logger.debug("Trial %s has score %s", trials, layout_score)
            if layout_score < chosen_layout_score:
                layout = Layout(
                    {reverse_im_graph_node_map[k]: v for k, v in layout_mapping.items()}
                )
                logger.debug(
                    "Found layout %s has a lower score (%s) than previous best %s (%s)",
                    layout,
                    layout_score,
                    chosen_layout,
                    chosen_layout_score,
                )
                chosen_layout = layout
                chosen_layout_score = layout_score
                stop_reason = VF2PostLayoutStopReason.SOLUTION_FOUND

            if self.max_trials and trials >= self.max_trials:
                logger.debug("Trial %s is >= configured max trials %s", trials, self.max_trials)
                break

            elapsed_time = time.time() - start_time
            if self.time_limit is not None and elapsed_time >= self.time_limit:
                logger.debug(
                    "VFPostLayout has taken %s which exceeds configured max time: %s",
                    elapsed_time,
                    self.time_limit,
                )
                break
        if stop_reason == VF2PostLayoutStopReason.SOLUTION_FOUND:
            chosen_layout = vf2_utils.map_free_qubits(
                free_nodes,
                chosen_layout,
                cm_graph.num_nodes(),
                reverse_im_graph_node_map,
                self.avg_error_map,
            )
            existing_layout = self.property_set["layout"]
            # If any ancillas in initial layout map them back to the final layout output
            if existing_layout is not None and len(existing_layout) > len(chosen_layout):
                virtual_bits = chosen_layout.get_virtual_bits()
                used_bits = set(virtual_bits.values())
                num_qubits = len(cm_graph)
                for bit in dag.qubits:
                    if len(chosen_layout) == len(existing_layout):
                        break
                    if bit not in virtual_bits:
                        for i in range(num_qubits):
                            if i not in used_bits:
                                used_bits.add(i)
                                chosen_layout.add(bit, i)
                                break
            self.property_set["post_layout"] = chosen_layout
        else:
            if chosen_layout is None:
                stop_reason = VF2PostLayoutStopReason.NO_SOLUTION_FOUND
            # else the initial layout is optimal -> don't set post_layout, return 'no better solution'
        self.property_set["VF2PostLayout_stop_reason"] = stop_reason

    def _score_layout(self, layout, bit_map, reverse_bit_map, im_graph):
        bits = layout.get_virtual_bits()
        fidelity = 1
        if self.target is not None:
            for bit, node_index in bit_map.items():
                gate_counts = im_graph[node_index]
                for gate, count in gate_counts.items():
                    if self.target[gate] is not None and None not in self.target[gate]:
                        props = self.target[gate][(bits[bit],)]
                        if props is not None and props.error is not None:
                            fidelity *= (1 - props.error) ** count

            for edge in im_graph.edge_index_map().values():
                qargs = (bits[reverse_bit_map[edge[0]]], bits[reverse_bit_map[edge[1]]])
                gate_counts = edge[2]
                for gate, count in gate_counts.items():
                    if self.target[gate] is not None and None not in self.target[gate]:
                        props = self.target[gate][qargs]
                        if props is not None and props.error is not None:
                            fidelity *= (1 - props.error) ** count
        else:
            for bit, node_index in bit_map.items():
                gate_counts = im_graph[node_index]
                for gate, count in gate_counts.items():
                    if gate == "measure":
                        try:
                            fidelity *= (1 - self.properties.readout_error(bits[bit])) ** count
                        except BackendPropertyError:
                            pass
                    else:
                        try:
                            fidelity *= (1 - self.properties.gate_error(gate, bits[bit])) ** count
                        except BackendPropertyError:
                            pass
            for edge in im_graph.edge_index_map().values():
                qargs = (bits[reverse_bit_map[edge[0]]], bits[reverse_bit_map[edge[1]]])
                gate_counts = edge[2]
                for gate, count in gate_counts.items():
                    try:
                        fidelity *= (1 - self.properties.gate_error(gate, qargs)) ** count
                    except BackendPropertyError:
                        pass
        return 1 - fidelity

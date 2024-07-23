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

"""Creating Sabre starting layouts."""
from __future__ import annotations

import itertools

from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler import CouplingMap, Target, AnalysisPass, TranspilerError, Layout
from qiskit.transpiler.passes.layout.vf2_layout import VF2Layout
from qiskit._accelerate.error_map import ErrorMap


class SabrePreLayout(AnalysisPass):
    """Choose a starting layout to use for additional Sabre layout trials.

    Property Set Values Written
    ---------------------------

    ``sabre_starting_layouts`` (``list[Layout]``)
        An optional list of :class:`~.Layout` objects to use for additional Sabre layout trials.

    """

    def __init__(
        self,
        coupling_map,
        max_distance=2,
        error_rate=0.1,
        max_trials_vf2=100,
        call_limit_vf2=None,
        improve_layout=True,
    ):
        """SabrePreLayout initializer.

        The pass works by augmenting the coupling map with more and more "extra" edges
        until VF2 succeeds to find a perfect graph isomorphism. More precisely, the
        augmented coupling map contains edges between nodes that are within a given
        distance ``d`` in the original coupling map, and the value of ``d`` is increased
        until an isomorphism is found.

        Intuitively, a better layout involves fewer extra edges. The pass also optionally
        minimizes the number of extra edges involved in the layout until a local minimum
        is found. This involves removing extra edges and running VF2 to see if an
        isomorphism still exists.

        Args:
            coupling_map (Union[CouplingMap, Target]): directed graph representing the
                original coupling map or a target modelling the backend (including its
                connectivity).
            max_distance (int): the maximum distance to consider for augmented coupling maps.
            error_rate (float): the error rate to assign to the "extra" edges. A non-zero
                error rate prioritizes VF2 to choose original edges over extra edges.
            max_trials_vf2 (int): specifies the maximum number of VF2 trials. A larger number
                allows VF2 to explore more layouts, eventually choosing the one with the smallest
                error rate.
            call_limit_vf2 (int): limits each call to VF2 by bounding the number of VF2 state visits.
            improve_layout (bool): whether to improve the layout by minimizing the number of
                extra edges involved. This might be time-consuming as this requires additional
                VF2 calls.

        Raises:
            TranspilerError: At runtime, if neither ``coupling_map`` or ``target`` are provided.
        """

        self.max_distance = max_distance
        self.error_rate = error_rate
        self.max_trials_vf2 = max_trials_vf2
        self.call_limit_vf2 = call_limit_vf2
        self.improve_layout = improve_layout

        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map

        super().__init__()

    def run(self, dag):
        """Run the SabrePreLayout pass on `dag`.

        The discovered starting layout is written to the property set
        value ``sabre_starting_layouts``.

        Args:
            dag (DAGCircuit): DAG to create starting layout for.
        """

        if self.coupling_map is None:
            raise TranspilerError(
                "SabrePreLayout requires coupling_map to be used with either"
                "CouplingMap or a Target."
            )

        starting_layout = None
        cur_distance = 1
        while cur_distance <= self.max_distance:
            augmented_map, augmented_error_map = self._add_extra_edges(cur_distance)
            pass_ = VF2Layout(
                augmented_map,
                seed=0,
                max_trials=self.max_trials_vf2,
                call_limit=self.call_limit_vf2,
            )
            pass_.property_set["vf2_avg_error_map"] = augmented_error_map
            pass_.run(dag)

            if "layout" in pass_.property_set:
                starting_layout = pass_.property_set["layout"]
                break

            cur_distance += 1

        if cur_distance > 1 and starting_layout is not None:
            # optionally improve starting layout
            if self.improve_layout:
                starting_layout = self._minimize_extra_edges(dag, starting_layout)
            # write discovered layout into the property set
            if "sabre_starting_layouts" not in self.property_set:
                self.property_set["sabre_starting_layouts"] = [starting_layout]
            else:
                self.property_set["sabre_starting_layouts"].append(starting_layout)

    def _add_extra_edges(self, distance):
        """Augments the coupling map with extra edges that connect nodes ``distance``
        apart in the original graph. The extra edges are assigned errors allowing VF2
        to prioritize real edges over extra edges.
        """
        nq = len(self.coupling_map.graph)
        augmented_coupling_map = CouplingMap()
        augmented_coupling_map.graph = self.coupling_map.graph.copy()
        augmented_error_map = ErrorMap(nq)

        for x, y in itertools.combinations(self.coupling_map.graph.node_indices(), 2):
            d = self.coupling_map.distance(x, y)
            if 1 < d <= distance:
                error_rate = 1 - ((1 - self.error_rate) ** d)
                augmented_coupling_map.add_edge(x, y)
                augmented_error_map.add_error((x, y), error_rate)
                augmented_coupling_map.add_edge(y, x)
                augmented_error_map.add_error((y, x), error_rate)

        return augmented_coupling_map, augmented_error_map

    def _get_extra_edges_used(self, dag: DAGCircuit, layout: Layout) -> set[tuple[int, int]]:
        """Returns the set of extra edges involved in the layout."""
        extra_edges_used = set()
        virtual_bits = layout.get_virtual_bits()
        for node in dag.two_qubit_ops():
            p0 = virtual_bits[node.qargs[0]]
            p1 = virtual_bits[node.qargs[1]]
            if self.coupling_map.distance(p0, p1) > 1:
                extra_edge = (p0, p1) if p0 < p1 else (p1, p0)
                extra_edges_used.add(extra_edge)
        return extra_edges_used

    def _find_layout(self, dag, edges):
        """Checks if there is a layout for a given set of edges."""
        cm = CouplingMap(edges)
        pass_ = VF2Layout(cm, seed=0, max_trials=1, call_limit=self.call_limit_vf2)
        pass_.run(dag)
        return pass_.property_set.get("layout", None)

    def _minimize_extra_edges(self, dag, starting_layout):
        """Minimizes the set of extra edges involved in the layout. This iteratively
        removes extra edges from the coupling map and uses VF2 to check if a layout
        still exists. This is reasonably efficiently as it only looks for a local
        minimum.
        """
        # compute the set of edges in the original coupling map
        real_edges = []
        for x, y in itertools.combinations(self.coupling_map.graph.node_indices(), 2):
            d = self.coupling_map.distance(x, y)
            if d == 1:
                real_edges.append((x, y))

        best_layout = starting_layout

        # keeps the set of "necessary" extra edges: without a necessary edge
        # a layout no longer exists
        extra_edges_necessary: list[tuple[int, int]] = []

        extra_edges_unprocessed_set = self._get_extra_edges_used(dag, starting_layout)

        while extra_edges_unprocessed_set:
            # choose some unprocessed edge
            edge_chosen = next(iter(extra_edges_unprocessed_set))
            extra_edges_unprocessed_set.remove(edge_chosen)

            # check if a layout still exists without this edge
            layout = self._find_layout(
                dag, real_edges + extra_edges_necessary + list(extra_edges_unprocessed_set)
            )

            if layout is None:
                # without this edge the layout either does not exist or is too hard to find
                extra_edges_necessary.append(edge_chosen)

            else:
                # this edge is not necessary, furthermore we can trim the set of edges to examine based
                # in the edges involved in the layout.
                extra_edges_unprocessed_set = self._get_extra_edges_used(dag, layout).difference(
                    set(extra_edges_necessary)
                )
                best_layout = layout

        return best_layout

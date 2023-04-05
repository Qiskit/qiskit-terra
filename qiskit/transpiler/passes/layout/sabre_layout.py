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

"""Layout selection using the SABRE bidirectional search approach from Li et al.
"""

import copy
import logging
import numpy as np
import rustworkx as rx

from qiskit.converters import dag_to_circuit
from qiskit.transpiler.passes.layout.set_layout import SetLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.apply_layout import ApplyLayout
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit._accelerate.nlayout import NLayout
from qiskit._accelerate.sabre_layout import sabre_layout_and_routing
from qiskit._accelerate.sabre_swap import (
    Heuristic,
    NeighborTable,
)
from qiskit.transpiler.passes.routing.sabre_swap import process_swaps, apply_gate
from qiskit.transpiler.target import Target
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.tools.parallel import CPU_COUNT

logger = logging.getLogger(__name__)


class SabreLayout(TransformationPass):
    """Choose a Layout via iterative bidirectional routing of the input circuit.

    Starting with a random initial `Layout`, the algorithm does a full routing
    of the circuit (via the `routing_pass` method) to end up with a
    `final_layout`. This final_layout is then used as the initial_layout for
    routing the reverse circuit. The algorithm iterates a number of times until
    it finds an initial_layout that reduces full routing cost.

    This method exploits the reversibility of quantum circuits, and tries to
    include global circuit information in the choice of initial_layout.

    By default this pass will run both layout and routing and will transform the
    circuit so that the layout is applied to the input dag (meaning that the output
    circuit will have ancilla qubits allocated for unused qubits on the coupling map
    and the qubits will be reordered to match the mapped physical qubits) and then
    routing will be applied (inserting :class:`~.SwapGate`s to account for limited
    connectivity). This is unlike most other layout passes which are :class:`~.AnalysisPass`
    objects and just find an initial layout and set that on the property set. This is
    done because by default the pass will run parallel seed trials with different random
    seeds for selecting the random initial layout and then selecting the routed output
    which results in the least number of swap gates needed.

    You can use the ``routing_pass`` argument to have this pass operate as a typical
    layout pass. When specified this will use the specified routing pass to select an
    initial layout only and will not run multiple seed trials.

    **References:**

    [1] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(
        self,
        coupling_map,
        routing_pass=None,
        seed=None,
        max_iterations=3,
        swap_trials=None,
        layout_trials=None,
        skip_routing=False,
    ):
        """SabreLayout initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): directed graph representing a coupling map.
            routing_pass (BasePass): the routing pass to use while iterating.
                If specified this pass operates as an :class:`~.AnalysisPass` and
                will only populate the ``layout`` field in the property set and
                the input dag is returned unmodified. This argument is mutually
                exclusive with the ``swap_trials`` and the ``layout_trials``
                arguments and if this is specified at the same time as either
                argument an error will be raised.
            seed (int): seed for setting a random first trial layout.
            max_iterations (int): number of forward-backward iterations.
            swap_trials (int): The number of trials to run of
                :class:`~.SabreSwap` for each iteration. This is equivalent to
                the ``trials`` argument on :class:`~.SabreSwap`. If this is not
                specified (and ``routing_pass`` isn't set) by default the number
                of physical CPUs on your local system will be used. For
                reproducibility between environments it is best to set this
                to an explicit number because the output will potentially depend
                on the number of trials run. This option is mutually exclusive
                with the ``routing_pass`` argument and an error will be raised
                if both are used.
            layout_trials (int): The number of random seed trials to run
                layout with. When > 1 the trial that resuls in the output with
                the fewest swap gates will be selected. If this is not specified
                (and ``routing_pass`` is not set) then the number of local
                physical CPUs will be used as the default value. This option is
                mutually exclusive with the ``routing_pass`` argument and an error
                will be raised if both are used.
            skip_routing (bool): If this is set ``True`` and ``routing_pass`` is not used
                then routing will not be applied to the output circuit.  Only the layout
                will be returned in the property set. This is a tradeoff to run custom
                routing with multiple layout trials, as using this option will cause
                SabreLayout to run the routing stage internally but not use that result.

        Raises:
            TranspilerError: If both ``routing_pass`` and ``swap_trials`` or
            both ``routing_pass`` and ``layout_trials`` are specified
        """
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        self._neighbor_table = None
        if routing_pass is not None and (swap_trials is not None or layout_trials is not None):
            raise TranspilerError("Both routing_pass and swap_trials can't be set at the same time")
        self.routing_pass = routing_pass
        self.seed = seed
        self.max_iterations = max_iterations
        self.trials = swap_trials
        if swap_trials is None:
            self.swap_trials = CPU_COUNT
        else:
            self.swap_trials = swap_trials
        if layout_trials is None:
            self.layout_trials = CPU_COUNT
        else:
            self.layout_trials = layout_trials
        self.skip_routing = skip_routing
        if self.coupling_map is not None:
            if not self.coupling_map.is_symmetric:
                # deepcopy is needed here to avoid modifications updating
                # shared references in passes which require directional
                # constraints
                self.coupling_map = copy.deepcopy(self.coupling_map)
                self.coupling_map.make_symmetric()
            self._neighbor_table = NeighborTable(rx.adjacency_matrix(self.coupling_map.graph))

    def run(self, dag):
        """Run the SabreLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Returns:
           DAGCircuit: The output dag if swap mapping was run
            (otherwise the input dag is returned unmodified).

        Raises:
            TranspilerError: if dag wider than self.coupling_map
        """
        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")
        if not self.coupling_map.is_connected():
            raise TranspilerError(
                "Coupling Map is disjoint, this pass can't be used with a disconnected coupling "
                "map."
            )
        # Choose a random initial_layout.
        if self.routing_pass is not None:
            if self.seed is None:
                seed = np.random.randint(0, np.iinfo(np.int32).max)
            else:
                seed = self.seed
            rng = np.random.default_rng(seed)

            physical_qubits = rng.choice(self.coupling_map.size(), len(dag.qubits), replace=False)
            physical_qubits = rng.permutation(physical_qubits)
            initial_layout = Layout({q: dag.qubits[i] for i, q in enumerate(physical_qubits)})

            self.routing_pass.fake_run = True

            # Do forward-backward iterations.
            circ = dag_to_circuit(dag)
            rev_circ = circ.reverse_ops()
            for _ in range(self.max_iterations):
                for _ in ("forward", "backward"):
                    pm = self._layout_and_route_passmanager(initial_layout)
                    new_circ = pm.run(circ)

                    # Update initial layout and reverse the unmapped circuit.
                    pass_final_layout = pm.property_set["final_layout"]
                    final_layout = self._compose_layouts(
                        initial_layout, pass_final_layout, new_circ.qregs
                    )
                    initial_layout = final_layout
                    circ, rev_circ = rev_circ, circ

                # Diagnostics
                logger.info("new initial layout")
                logger.info(initial_layout)

            for qreg in dag.qregs.values():
                initial_layout.add_register(qreg)
            self.property_set["layout"] = initial_layout
            self.routing_pass.fake_run = False
            return dag

        dist_matrix = self.coupling_map.distance_matrix
        print('original distance matrix')
        print(dist_matrix)

        if self.target:
            dist_matrix = rx.digraph_floyd_warshall_numpy(
                self.coupling_map.graph, weight_fn=lambda x: self._swap_cost(x)
            )
        print('new distance matrix')
        print(dist_matrix)

        original_qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        original_clbit_indices = {bit: index for index, bit in enumerate(dag.clbits)}

        dag_list = []
        for node in dag.topological_op_nodes():
            cargs = {original_clbit_indices[x] for x in node.cargs}
            if node.op.condition is not None:
                for clbit in dag._bits_in_condition(node.op.condition):
                    cargs.add(original_clbit_indices[clbit])

            dag_list.append(
                (
                    node._node_id,
                    [original_qubit_indices[x] for x in node.qargs],
                    cargs,
                )
            )
        ((initial_layout, final_layout), swap_map, gate_order) = sabre_layout_and_routing(
            len(dag.clbits),
            dag_list,
            self._neighbor_table,
            dist_matrix,
            Heuristic.Decay,
            self.max_iterations,
            self.swap_trials,
            self.layout_trials,
            self.seed,
        )
        # Apply initial layout selected.
        original_dag = dag
        layout_dict = {}
        num_qubits = len(dag.qubits)
        for k, v in initial_layout.layout_mapping():
            if k < num_qubits:
                layout_dict[dag.qubits[k]] = v
        initital_layout = Layout(layout_dict)
        self.property_set["layout"] = initital_layout
        # If skip_routing is set then return the layout in the property set
        # and throwaway the extra work we did to compute the swap map
        if self.skip_routing:
            return dag
        # After this point the pass is no longer an analysis pass and the
        # output circuit returned is transformed with the layout applied
        # and swaps inserted
        dag = self._apply_layout_no_pass_manager(dag)
        # Apply sabre swap ontop of circuit with sabre layout
        final_layout_mapping = final_layout.layout_mapping()
        self.property_set["final_layout"] = Layout(
            {dag.qubits[k]: v for (k, v) in final_layout_mapping}
        )
        mapped_dag = dag.copy_empty_like()
        canonical_register = dag.qregs["q"]
        qubit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}
        original_layout = NLayout.generate_trivial_layout(self.coupling_map.size())
        for node_id in gate_order:
            node = original_dag._multi_graph[node_id]
            process_swaps(
                swap_map,
                node,
                mapped_dag,
                original_layout,
                canonical_register,
                False,
                qubit_indices,
            )
            apply_gate(mapped_dag, node, original_layout, canonical_register, False, layout_dict)
        return mapped_dag

    def _apply_layout_no_pass_manager(self, dag):
        """Apply and embed a layout into a dagcircuit without using a ``PassManager`` to
        avoid circuit<->dag conversion.
        """
        ancilla_pass = FullAncillaAllocation(self.coupling_map)
        ancilla_pass.property_set = self.property_set
        dag = ancilla_pass.run(dag)
        enlarge_pass = EnlargeWithAncilla()
        enlarge_pass.property_set = ancilla_pass.property_set
        dag = enlarge_pass.run(dag)
        apply_pass = ApplyLayout()
        apply_pass.property_set = enlarge_pass.property_set
        dag = apply_pass.run(dag)
        return dag

    def _layout_and_route_passmanager(self, initial_layout):
        """Return a passmanager for a full layout and routing.

        We use a factory to remove potential statefulness of passes.
        """
        layout_and_route = [
            SetLayout(initial_layout),
            FullAncillaAllocation(self.coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
            self.routing_pass,
        ]
        pm = PassManager(layout_and_route)
        return pm

    def _compose_layouts(self, initial_layout, pass_final_layout, qregs):
        """Return the real final_layout resulting from the composition
        of an initial_layout with the final_layout reported by a pass.

        The routing passes internally start with a trivial layout, as the
        layout gets applied to the circuit prior to running them. So the
        "final_layout" they report must be amended to account for the actual
        initial_layout that was selected.
        """
        trivial_layout = Layout.generate_trivial_layout(*qregs)
        qubit_map = Layout.combine_into_edge_map(initial_layout, trivial_layout)
        final_layout = {v: pass_final_layout._v2p[qubit_map[v]] for v in initial_layout._v2p}
        return Layout(final_layout)

    def _swap_cost(self, instructions):
        """Get the minimum cost of a SWAP on a pair of physical qubits given a list
        of available instructions on those qubits.

        TODO: this function only considers certain known basis for which Qiskit knows
        analytical decompositions. This can be generalized to arbitrary basis using the
        tools of monodromy polytopes.

        Args:
            instructions (list[dict]): mappings of gate name to properties for that gate.

        Return:
            float: minimum cost of performing SWAP using the available physical instructions.
        """
        swap_cost = np.inf
        for key, value in instructions.items():
            weyl = TwoQubitWeylDecomposition(self.target.operation_from_name(key))
            a, b, c = weyl.a, weyl.b, weyl.c
            if a == b == c == np.pi / 4:  # 1 x SWAP
                gates_per_swap = 1
            elif b == c == 0:  # k x controlled XX(theta)
                gates_per_swap = 3 * (np.pi / 2 / (2 * a))
            elif a == np.pi / 4 and c == 0:  # 3 x supercontrolled
                gates_per_swap = 3
            elif a == b == abs(c) == np.pi / 8:  # 2 x sqrt(SWAP)
                gates_per_swap = 2
            elif a == b == np.pi / 8 and c == 0:  # 2 x sqrt(iSWAP)
                gates_per_swap = 3
            elif (a, b, c) == (np.pi / 4, np.pi / 8, 0):  # 2 x B
                gates_per_swap = 2

            if gates_per_swap * value.error < swap_cost:
                swap_cost = gates_per_swap * value.error
        return swap_cost

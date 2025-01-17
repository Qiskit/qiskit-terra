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
import dataclasses
import logging
import functools
import time

import numpy as np
import rustworkx as rx

from qiskit.converters import dag_to_circuit
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes.layout.set_layout import SetLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.apply_layout import ApplyLayout
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit._accelerate.nlayout import NLayout
from qiskit._accelerate.sabre import sabre_layout_and_routing, Heuristic, NeighborTable, SetScaling
from qiskit.transpiler.passes.routing.sabre_swap import _build_sabre_dag, _apply_sabre_result
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.utils.parallel import CPU_COUNT

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

    By default, this pass will run both layout and routing and will transform the
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

    In addition to starting with a random initial `Layout` the pass can also take in
    an additional list of starting layouts which will be used for additional
    trials. If the ``sabre_starting_layouts`` is present in the property set
    when this pass is run, that will be used for additional trials. There will still
    be ``layout_trials`` of full random starting layouts run and the contents of
    ``sabre_starting_layouts`` will be run in addition to those. The output which results
    in the lowest amount of swap gates (whether from the random trials or the property
    set starting point) will be used. The value for this property set field should be a
    list of :class:`.Layout` objects representing the starting layouts to use. If a
    virtual qubit is missing from an :class:`.Layout` object in the list a random qubit
    will be selected.

    Property Set Fields Read
    ------------------------

    ``sabre_starting_layouts`` (``list[Layout]``)
        An optional list of :class:`~.Layout` objects to use for additional layout trials. This is
        in addition to the full random trials specified with the ``layout_trials`` argument.

    Property Set Values Written
    ---------------------------

    ``layout`` (:class:`.Layout`)
        The chosen initial mapping of virtual to physical qubits, including the ancilla allocation.

    ``final_layout`` (:class:`.Layout`)
        A permutation of how swaps have been applied to the input qubits at the end of the circuit.

    **References:**

    [1] Henry Zou and Matthew Treinish and Kevin Hartman and Alexander Ivrii and Jake Lishman.
    "LightSABRE: A Lightweight and Enhanced SABRE Algorithm"
    `arXiv:2409.08368 <https://doi.org/10.48550/arXiv.2409.08368>`__
    [2] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
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
                layout with. When > 1 the trial that results in the output with
                the fewest swap gates will be selected. If this is not specified
                (and ``routing_pass`` is not set) then the number of local
                physical CPUs will be used as the default value. This option is
                mutually exclusive with the ``routing_pass`` argument and an error
                will be raised if both are used. An additional 3 or 4 trials
                depending on the ``coupling_map`` value are run with common layouts
                on top of the random trial count specified by this value.
            skip_routing (bool): If this is set ``True`` and ``routing_pass`` is not used
                then routing will not be applied to the output circuit.  Only the layout
                will be set in the property set. This is a tradeoff to run custom
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
                # deepcopy is needed here if we don't own the coupling map (i.e. we were passed it
                # directly) to avoid modifications updating shared references in passes which
                # require directional constraints
                if isinstance(coupling_map, CouplingMap):
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

        # Choose a random initial_layout.
        if self.routing_pass is not None:
            if not self.coupling_map.is_connected():
                raise TranspilerError(
                    "The routing_pass argument cannot be used with disjoint coupling maps."
                )
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
        # Combined
        if self.target is not None:
            # This is a special case SABRE only works with a bidirectional coupling graph
            # which we explicitly can't create from the target. So do this manually here
            # to avoid altering the shared state with the unfiltered indices.
            target = self.target.build_coupling_map(filter_idle_qubits=True)
            target.make_symmetric()
        else:
            target = self.coupling_map
        inner_run = self._inner_run
        if "sabre_starting_layouts" in self.property_set:
            inner_run = functools.partial(
                self._inner_run, starting_layouts=self.property_set["sabre_starting_layouts"]
            )
        components = disjoint_utils.run_pass_over_connected_components(dag, target, inner_run)
        self.property_set["layout"] = Layout(
            {
                component.dag.qubits[logic]: component.coupling_map.graph[phys]
                for component in components
                for logic, phys in component.initial_layout.layout_mapping()
                # A physical component of the coupling map might be wider than the DAG that we're
                # laying out onto it.  We shouldn't include these implicit ancillas right now as the
                # ancilla-allocation pass will run on the whole map in one go.
                if logic < len(component.dag.qubits)
            }
        )

        # Add the existing registers to the layout
        for qreg in dag.qregs.values():
            self.property_set["layout"].add_register(qreg)

        # If skip_routing is set then return the layout in the property set
        # and throwaway the extra work we did to compute the swap map.
        # We also skip routing here if there is more than one connected
        # component we ran layout on. We can only reliably route the full dag
        # in this case if there is any dependency between the components
        # (typically shared classical data or barriers).
        if self.skip_routing or len(components) > 1:
            return dag

        # At this point, we become a transformation pass, and apply the layout and the routing to
        # the DAG directly.  This includes filling in the `property_set` data of the embed passes.

        dag = self._ancilla_allocation_no_pass_manager(dag)
        # The ancilla-allocation pass has expanded this since we set it above.
        full_initial_layout = self.property_set["layout"]

        # Set up a physical DAG to apply the Sabre result onto.  We do not need to run the
        # `ApplyLayout` transpiler pass (which usually does this step), because we're about to apply
        # the layout and routing together as part of resolving the Sabre result.
        physical_qubits = QuantumRegister(self.coupling_map.size(), "q")
        mapped_dag = DAGCircuit()
        mapped_dag.add_qreg(physical_qubits)
        mapped_dag.add_clbits(dag.clbits)
        for creg in dag.cregs.values():
            mapped_dag.add_creg(creg)
        for var in dag.iter_input_vars():
            mapped_dag.add_input_var(var)
        for var in dag.iter_captured_vars():
            mapped_dag.add_captured_var(var)
        for var in dag.iter_declared_vars():
            mapped_dag.add_declared_var(var)
        mapped_dag.global_phase = dag.global_phase
        self.property_set["original_qubit_indices"] = {
            bit: index for index, bit in enumerate(dag.qubits)
        }
        final_layout = Layout(
            {
                mapped_dag.qubits[
                    component.coupling_map.graph[initial]
                ]: component.coupling_map.graph[final]
                for component in components
                for initial, final in enumerate(component.final_permutation)
            }
        )
        if self.property_set["final_layout"] is None:
            self.property_set["final_layout"] = final_layout
        else:
            self.property_set["final_layout"] = final_layout.compose(
                self.property_set["final_layout"], dag.qubits
            )
        for component in components:
            # Sabre routing still returns all its swaps as on virtual qubits, so we need to expand
            # each component DAG with the virtual ancillas that were allocated to it, so the layout
            # application can succeed.  This is the last thing we do with the component DAG, so it's
            # ok for us to modify it.
            component_size = component.coupling_map.size()
            dag_size = component.dag.num_qubits()
            if component_size > dag_size:
                used_physical = {full_initial_layout[logic] for logic in component.dag.qubits}
                component.dag.add_qubits(
                    [
                        full_initial_layout[component.coupling_map.graph[phys]]
                        for phys in range(component.dag.num_qubits(), component_size)
                        if component.coupling_map.graph[phys] not in used_physical
                    ]
                )
            mapped_dag = _apply_sabre_result(
                mapped_dag,
                component.dag,
                component.sabre_result,
                component.initial_layout,
                [
                    mapped_dag.qubits[component.coupling_map.graph[phys]]
                    for phys in range(component_size)
                ],
                component.circuit_to_dag_dict,
            )
        disjoint_utils.combine_barriers(mapped_dag, retain_uuid=False)
        return mapped_dag

    def _inner_run(self, dag, coupling_map, starting_layouts=None):
        if not coupling_map.is_symmetric:
            # deepcopy is needed here to avoid modifications updating
            # shared references in passes which require directional
            # constraints
            coupling_map = copy.deepcopy(coupling_map)
            coupling_map.make_symmetric()
        neighbor_table = NeighborTable(rx.adjacency_matrix(coupling_map.graph))
        dist_matrix = coupling_map.distance_matrix
        original_qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        partial_layouts = []
        if starting_layouts is not None:
            coupling_map_reverse_mapping = {
                coupling_map.graph[x]: x for x in coupling_map.graph.node_indices()
            }
            for layout in starting_layouts:
                virtual_bits = layout.get_virtual_bits()
                out_layout = [None] * len(dag.qubits)
                for bit, phys in virtual_bits.items():
                    pos = original_qubit_indices.get(bit, None)
                    if pos is None:
                        continue
                    out_layout[pos] = coupling_map_reverse_mapping[phys]
                partial_layouts.append(out_layout)

        sabre_dag, circuit_to_dag_dict = _build_sabre_dag(
            dag,
            coupling_map.size(),
            original_qubit_indices,
        )
        heuristic = (
            Heuristic(attempt_limit=10 * coupling_map.size())
            .with_basic(1.0, SetScaling.Size)
            .with_lookahead(0.5, 20, SetScaling.Size)
            .with_decay(0.001, 5)
        )
        sabre_start = time.perf_counter()
        (initial_layout, final_permutation, sabre_result) = sabre_layout_and_routing(
            sabre_dag,
            neighbor_table,
            dist_matrix,
            heuristic,
            self.max_iterations,
            self.swap_trials,
            self.layout_trials,
            self.seed,
            partial_layouts,
        )
        sabre_stop = time.perf_counter()
        logger.debug(
            "Sabre layout algorithm execution for a connected component complete in: %s sec.",
            sabre_stop - sabre_start,
        )
        return _DisjointComponent(
            dag,
            coupling_map,
            initial_layout,
            final_permutation,
            sabre_result,
            circuit_to_dag_dict,
        )

    def _ancilla_allocation_no_pass_manager(self, dag):
        """Run the ancilla-allocation and -enlargement passes on the DAG chained onto our
        ``property_set``, skipping the DAG-to-circuit conversion cost of using a ``PassManager``."""
        ancilla_pass = FullAncillaAllocation(self.coupling_map)
        ancilla_pass.property_set = self.property_set
        dag = ancilla_pass.run(dag)
        enlarge_pass = EnlargeWithAncilla()
        enlarge_pass.property_set = ancilla_pass.property_set
        dag = enlarge_pass.run(dag)
        self.property_set = enlarge_pass.property_set
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
        ``"final_layout"`` they report must be amended to account for the actual
        initial_layout that was selected.
        """
        trivial_layout = Layout.generate_trivial_layout(*qregs)
        qubit_map = Layout.combine_into_edge_map(initial_layout, trivial_layout)
        final_layout = {v: pass_final_layout._v2p[qubit_map[v]] for v in initial_layout._v2p}
        return Layout(final_layout)


@dataclasses.dataclass
class _DisjointComponent:
    __slots__ = (
        "dag",
        "coupling_map",
        "initial_layout",
        "final_permutation",
        "sabre_result",
        "circuit_to_dag_dict",
    )

    dag: DAGCircuit
    coupling_map: CouplingMap
    initial_layout: NLayout
    final_permutation: "list[int]"
    sabre_result: "tuple[SwapMap, Sequence[int], NodeBlockResults]"
    circuit_to_dag_dict: "dict[int, DAGCircuit]"

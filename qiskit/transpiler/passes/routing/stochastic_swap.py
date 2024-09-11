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

"""Map a DAGCircuit onto a ``coupling_map`` adding swap gates."""

import itertools
import logging
from math import inf
import numpy as np

from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit.classical import expr, types
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target
from qiskit.circuit import (
    Clbit,
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    SwitchCaseOp,
    ControlFlowOp,
    CASE_DEFAULT,
)
from qiskit._accelerate import stochastic_swap as stochastic_swap_rs
from qiskit._accelerate import nlayout
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.utils import deprecate_func

from .utils import get_swap_map_dag

logger = logging.getLogger(__name__)


class StochasticSwap(TransformationPass):
    """Map a DAGCircuit onto a `coupling_map` adding swap gates.

    Uses a randomized algorithm.

    Notes:
        1. Measurements may occur and be followed by swaps that result in repeated
           measurement of the same qubit. Near-term experiments cannot implement
           these circuits, so some care is required when using this mapper
           with experimental backend targets.

        2. We do not use the fact that the input state is zero to simplify
           the circuit.
    """

    @deprecate_func(
        since="1.3",
        removal_timeline="in the 2.0 release",
        additional_msg="The StochasticSwap transpilation pass is a suboptimal "
        "routing algorithm and has been superseded by the SabreSwap pass.",
    )
    def __init__(self, coupling_map, trials=20, seed=None, fake_run=False, initial_layout=None):
        """StochasticSwap initializer.

        The coupling map is a connected graph

        If these are not satisfied, the behavior is undefined.

        Args:
            coupling_map (Union[CouplingMap, Target]): Directed graph representing a coupling
                map.
            trials (int): maximum number of iterations to attempt
            seed (int): seed for random number generator
            fake_run (bool): if true, it will only pretend to do routing, i.e., no
                swap is effectively added.
            initial_layout (Layout): starting layout at beginning of pass.
        """
        super().__init__()

        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        self.trials = trials
        self.seed = seed
        self.rng = None
        self.fake_run = fake_run
        self.qregs = None
        self.initial_layout = initial_layout
        self._int_to_qubit = None

    def run(self, dag):
        """Run the StochasticSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG, or if the coupling_map=None
        """

        if self.coupling_map is None:
            raise TranspilerError("StochasticSwap cannot run with coupling_map=None")

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("StochasticSwap runs on physical circuits only")

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")
        disjoint_utils.require_layout_isolated_to_component(
            dag, self.coupling_map if self.target is None else self.target
        )

        self.rng = np.random.default_rng(self.seed)

        canonical_register = dag.qregs["q"]
        if self.initial_layout is None:
            self.initial_layout = Layout.generate_trivial_layout(canonical_register)
        # Qubit indices are used to assign an integer to each virtual qubit during the routing: it's
        # a mapping of {virtual: virtual}, for converting between Python and Rust forms.
        self._int_to_qubit = tuple(dag.qubits)

        self.qregs = dag.qregs
        logger.debug("StochasticSwap rng seeded with seed=%s", self.seed)
        self.coupling_map.compute_distance_matrix()
        new_dag = self._mapper(dag, self.coupling_map, trials=self.trials)
        return new_dag

    def _layer_permutation(self, dag, layer_partition, layout, qubit_subset, coupling, trials):
        """Find a swap circuit that implements a permutation for this layer.

        The goal is to swap qubits such that qubits in the same two-qubit gates
        are adjacent.

        Based on S. Bravyi's algorithm.

        Args:
            layer_partition (list): The layer_partition is a list of (qu)bit
                lists and each qubit is a tuple (qreg, index).
            layout (Layout): The layout is a Layout object mapping virtual
                qubits in the input circuit to physical qubits in the coupling
                graph. It reflects the current positions of the data.
            qubit_subset (list): The qubit_subset is the set of qubits in
                the coupling graph that we have chosen to map into, as tuples
                (Register, index).
            coupling (CouplingMap): Directed graph representing a coupling map.
                This coupling map should be one that was provided to the
                stochastic mapper.
            trials (int): Number of attempts the randomized algorithm makes.

        Returns:
            Tuple: success_flag, best_circuit, best_depth, best_layout

        If success_flag is True, then best_circuit contains a DAGCircuit with
        the swap circuit, best_depth contains the depth of the swap circuit,
        and best_layout contains the new positions of the data qubits after the
        swap circuit has been applied.

        Raises:
            TranspilerError: if anything went wrong.
        """
        logger.debug("layer_permutation: layer_partition = %s", layer_partition)
        logger.debug("layer_permutation: layout = %s", layout.get_virtual_bits())
        logger.debug("layer_permutation: qubit_subset = %s", qubit_subset)
        logger.debug("layer_permutation: trials = %s", trials)

        # The input dag is on a flat canonical register
        canonical_register = QuantumRegister(len(layout), "q")

        gates = []  # list of lists of tuples [[(register, index), ...], ...]
        for gate_args in layer_partition:
            if len(gate_args) > 2:
                raise TranspilerError("Layer contains > 2-qubit gates")
            if len(gate_args) == 2:
                gates.append(tuple(gate_args))
        logger.debug("layer_permutation: gates = %s", gates)

        # Can we already apply the gates? If so, there is no work to do.
        # Accessing via private attributes to avoid overhead from __getitem__
        # and to optimize performance of the distance matrix access
        dist = sum(coupling._dist_matrix[layout._v2p[g[0]], layout._v2p[g[1]]] for g in gates)
        logger.debug("layer_permutation: distance = %s", dist)
        if dist == len(gates):
            logger.debug("layer_permutation: nothing to do")
            circ = DAGCircuit()
            circ.add_qreg(canonical_register)
            return True, circ, 0, layout

        # Begin loop over trials of randomized algorithm
        num_qubits = len(layout)
        best_depth = inf  # initialize best depth
        best_edges = None  # best edges found
        best_circuit = None  # initialize best swap circuit
        best_layout = None  # initialize best final layout

        cdist2 = coupling._dist_matrix**2
        int_qubit_subset = np.fromiter(
            (dag.find_bit(bit).index for bit in qubit_subset),
            dtype=np.uint32,
            count=len(qubit_subset),
        )

        int_gates = np.fromiter(
            (dag.find_bit(bit).index for gate in gates for bit in gate),
            dtype=np.uint32,
            count=2 * len(gates),
        )

        layout_mapping = {dag.find_bit(k).index: v for k, v in layout.get_virtual_bits().items()}
        int_layout = nlayout.NLayout(layout_mapping, num_qubits, coupling.size())

        trial_circuit = DAGCircuit()  # SWAP circuit for slice of swaps in this trial
        trial_circuit.add_qubits(list(layout.get_virtual_bits()))

        edges = np.asarray(coupling.get_edges(), dtype=np.uint32).ravel()
        cdist = coupling._dist_matrix
        best_edges, best_layout, best_depth = stochastic_swap_rs.swap_trials(
            trials,
            num_qubits,
            int_layout,
            int_qubit_subset,
            int_gates,
            cdist,
            cdist2,
            edges,
            seed=self.seed,
        )
        # If we have no best circuit for this layer, all of the trials have failed
        if best_layout is None:
            logger.debug("layer_permutation: failed!")
            return False, None, None, None

        edges = best_edges.edges()
        for idx in range(len(edges) // 2):
            swap_src = self._int_to_qubit[edges[2 * idx]]
            swap_tgt = self._int_to_qubit[edges[2 * idx + 1]]
            trial_circuit.apply_operation_back(SwapGate(), (swap_src, swap_tgt), (), check=False)
        best_circuit = trial_circuit

        # Otherwise, we return our result for this layer
        logger.debug("layer_permutation: success!")
        layout_mapping = best_layout.layout_mapping()

        best_lay = Layout({best_circuit.qubits[k]: v for (k, v) in layout_mapping})
        return True, best_circuit, best_depth, best_lay

    def _layer_update(self, dag, layer, best_layout, best_depth, best_circuit):
        """Add swaps followed by the now mapped layer from the original circuit.

        Args:
            dag (DAGCircuit): The DAGCircuit object that the _mapper method is building
            layer (DAGCircuit): A DAGCircuit layer from the original circuit
            best_layout (Layout): layout returned from _layer_permutation
            best_depth (int): depth returned from _layer_permutation
            best_circuit (DAGCircuit): swap circuit returned from _layer_permutation
        """
        logger.debug("layer_update: layout = %s", best_layout)
        logger.debug("layer_update: self.initial_layout = %s", self.initial_layout)

        # Output any swaps
        if best_depth > 0:
            logger.debug("layer_update: there are swaps in this layer, depth %d", best_depth)
            dag.compose(best_circuit, qubits=list(best_circuit.qubits), inline_captures=True)
        else:
            logger.debug("layer_update: there are no swaps in this layer")
        # Output this layer
        dag.compose(
            layer["graph"], qubits=best_layout.reorder_bits(dag.qubits), inline_captures=True
        )

    def _mapper(self, circuit_graph, coupling_graph, trials=20):
        """Map a DAGCircuit onto a CouplingMap using swap gates.

        Args:
            circuit_graph (DAGCircuit): input DAG circuit
            coupling_graph (CouplingMap): coupling graph to map onto
            trials (int): number of trials.

        Returns:
            DAGCircuit: object containing a circuit equivalent to
                circuit_graph that respects couplings in coupling_graph

        Raises:
            TranspilerError: if there was any error during the mapping
                or with the parameters.
        """
        # Schedule the input circuit by calling layers()
        layerlist = list(circuit_graph.layers())
        logger.debug("schedule:")
        for i, v in enumerate(layerlist):
            logger.debug("    %d: %s", i, v["partition"])

        qubit_subset = self.initial_layout.get_virtual_bits().keys()

        # Find swap circuit to precede each layer of input circuit
        layout = self.initial_layout.copy()

        # Construct an empty DAGCircuit with the same set of
        # qregs and cregs as the input circuit
        dagcircuit_output = None
        if not self.fake_run:
            dagcircuit_output = circuit_graph.copy_empty_like()

        logger.debug("layout = %s", layout)

        # Iterate over layers
        for i, layer in enumerate(layerlist):
            # First try and compute a route for the entire layer in one go.
            if not layer["graph"].op_nodes(op=ControlFlowOp):
                success_flag, best_circuit, best_depth, best_layout = self._layer_permutation(
                    circuit_graph, layer["partition"], layout, qubit_subset, coupling_graph, trials
                )

                logger.debug("mapper: layer %d", i)
                logger.debug("mapper: success_flag=%s,best_depth=%s", success_flag, str(best_depth))
                if success_flag:
                    layout = best_layout

                    # Update the DAG
                    if not self.fake_run:
                        self._layer_update(
                            dagcircuit_output, layer, best_layout, best_depth, best_circuit
                        )
                    continue

            # If we're here, we need to go through every gate in the layer serially.
            logger.debug("mapper: failed, layer %d, retrying sequentially", i)
            # Go through each gate in the layer
            for j, serial_layer in enumerate(layer["graph"].serial_layers()):
                layer_dag = serial_layer["graph"]
                # layer_dag has only one operation
                op_node = layer_dag.op_nodes()[0]
                if isinstance(op_node.op, ControlFlowOp):
                    layout = self._controlflow_layer_update(
                        dagcircuit_output, layer_dag, layout, circuit_graph
                    )
                else:
                    (success_flag, best_circuit, best_depth, best_layout) = self._layer_permutation(
                        circuit_graph,
                        serial_layer["partition"],
                        layout,
                        qubit_subset,
                        coupling_graph,
                        trials,
                    )
                    logger.debug("mapper: layer %d, sublayer %d", i, j)
                    logger.debug(
                        "mapper: success_flag=%s,best_depth=%s,", success_flag, str(best_depth)
                    )

                    # Give up if we fail again
                    if not success_flag:
                        raise TranspilerError(f"swap mapper failed: layer {i}, sublayer {j}")

                    # Update the record of qubit positions
                    # for each inner iteration
                    layout = best_layout
                    # Update the DAG
                    if not self.fake_run:
                        self._layer_update(
                            dagcircuit_output,
                            serial_layer,
                            best_layout,
                            best_depth,
                            best_circuit,
                        )

        # This is the final edgemap. We might use it to correctly replace
        # any measurements that needed to be removed earlier.
        logger.debug("mapper: self.initial_layout = %s", self.initial_layout)
        logger.debug("mapper: layout = %s", layout)
        if self.property_set["final_layout"] is None:
            self.property_set["final_layout"] = layout
        else:
            self.property_set["final_layout"] = layout.compose(
                self.property_set["final_layout"], circuit_graph.qubits
            )

        if self.fake_run:
            return circuit_graph
        return dagcircuit_output

    def _controlflow_layer_update(self, dagcircuit_output, layer_dag, current_layout, root_dag):
        """
        Updates the new dagcircuit with a routed control flow operation.

        Args:
           dagcircuit_output (DAGCircuit): dagcircuit that is being built with routed operations.
           layer_dag (DAGCircuit): layer to route containing a single controlflow operation.
           current_layout (Layout): current layout coming into this layer.
           root_dag (DAGCircuit): root dag of pass

        Returns:
           Layout: updated layout after this layer has been routed.

        Raises:
            TranspilerError: if layer_dag does not contain a recognized ControlFlowOp.

        """
        node = layer_dag.op_nodes()[0]
        if not isinstance(node.op, (IfElseOp, ForLoopOp, WhileLoopOp, SwitchCaseOp)):
            raise TranspilerError(f"unsupported control flow operation: {node}")
        # For each block, expand it up be the full width of the containing DAG so we can be certain
        # that it is routable, then route it within that.  When we recombine later, we'll reduce all
        # these blocks down to remove any qubits that are idle.
        block_dags = []
        block_layouts = []
        for block in node.op.blocks:
            inner_pass = self._recursive_pass(current_layout)
            block_dags.append(inner_pass.run(_dag_from_block(block, node, root_dag)))
            block_layouts.append(inner_pass.property_set["final_layout"].copy())

        # Determine what layout we need to go towards.  For some blocks (such as `for`), we must
        # guarantee that the final layout is the same as the initial or the loop won't work.
        if _controlflow_exhaustive_acyclic(node.op):
            # We heuristically just choose to use the layout of whatever the deepest block is, to
            # avoid extending the total depth by too much.
            final_layout = max(
                zip(block_layouts, block_dags), key=lambda x: x[1].depth(recurse=True)
            )[0]
        else:
            final_layout = current_layout
        if self.fake_run:
            return final_layout

        # Add swaps to the end of each block to make sure they all have the same layout at the end.
        # Adding these swaps can cause fewer wires to be idle than we expect (if we have to swap
        # across unused qubits), so we track that at this point too.
        idle_qubits = set(root_dag.qubits)
        for layout, updated_dag_block in zip(block_layouts, block_dags):
            swap_dag, swap_qubits = get_swap_map_dag(
                root_dag, self.coupling_map, layout, final_layout, seed=self._new_seed()
            )
            if swap_dag.size(recurse=False):
                updated_dag_block.compose(swap_dag, qubits=swap_qubits, inline_captures=True)
            idle_qubits &= set(updated_dag_block.idle_wires())

        # Now for each block, expand it to be full width over all active wires (all blocks of a
        # control-flow operation need to have equal input wires), and convert it to circuit form.
        block_circuits = []
        for updated_dag_block in block_dags:
            updated_dag_block.remove_qubits(*idle_qubits)
            block_circuits.append(dag_to_circuit(updated_dag_block))

        new_op = node.op.replace_blocks(block_circuits)
        new_qargs = block_circuits[0].qubits
        dagcircuit_output.apply_operation_back(new_op, new_qargs, node.cargs, check=False)
        return final_layout

    def _new_seed(self):
        """Get a seed for a new RNG instance."""
        return self.rng.integers(0x7FFF_FFFF_FFFF_FFFF)

    def _recursive_pass(self, initial_layout):
        """Get a new instance of this class to handle a recursive call for a control-flow block.

        Each pass starts with its own new seed, determined deterministically from our own."""
        return self.__class__(
            self.coupling_map,
            # This doesn't cause an exponential explosion of the trials because we only generate a
            # recursive pass instance for control-flow operations, while the trial multiplicity is
            # only for non-control-flow layers.
            trials=self.trials,
            seed=self._new_seed(),
            fake_run=self.fake_run,
            initial_layout=initial_layout,
        )


def _controlflow_exhaustive_acyclic(operation: ControlFlowOp):
    """Return True if the entire control-flow operation represents a block that is guaranteed to be
    entered, and does not cycle back to the initial layout."""
    if isinstance(operation, IfElseOp):
        return len(operation.blocks) == 2
    if isinstance(operation, SwitchCaseOp):
        cases = operation.cases()
        if isinstance(operation.target, expr.Expr):
            type_ = operation.target.type
            if type_.kind is types.Bool:
                max_matches = 2
            elif type_.kind is types.Uint:
                max_matches = 1 << type_.width
            else:
                raise RuntimeError(f"unhandled target type: '{type_}'")
        else:
            max_matches = 2 if isinstance(operation.target, Clbit) else 1 << len(operation.target)
        return CASE_DEFAULT in cases or len(cases) == max_matches
    return False


def _dag_from_block(block, node, root_dag):
    """Get a :class:`DAGCircuit` that represents the :class:`.QuantumCircuit` ``block`` embedded
    within the ``root_dag`` for full-width routing purposes.  This means that all the qubits are in
    the output DAG, but only the necessary clbits and classical registers are."""
    out = DAGCircuit()
    # The pass already ensured that `root_dag` has only a single quantum register with everything.
    for qreg in root_dag.qregs.values():
        out.add_qreg(qreg)
    # For clbits, we need to take more care.  Nested control-flow might need registers to exist for
    # conditions on inner blocks.  `DAGCircuit.substitute_node_with_dag` handles this register
    # mapping when required, so we use that with a dummy block that pretends to act on all variables
    # in the DAG.
    out.add_clbits(node.cargs)
    for var in block.iter_input_vars():
        out.add_input_var(var)
    for var in block.iter_captured_vars():
        out.add_captured_var(var)
    for var in block.iter_declared_vars():
        out.add_declared_var(var)

    dummy = out.apply_operation_back(
        IfElseOp(expr.lift(True), block.copy_empty_like(vars_mode="captures")),
        node.qargs,
        node.cargs,
        check=False,
    )
    wire_map = dict(itertools.chain(zip(block.qubits, node.qargs), zip(block.clbits, node.cargs)))
    out.substitute_node_with_dag(dummy, circuit_to_dag(block), wires=wire_map)
    return out

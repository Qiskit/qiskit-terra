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

"""Search for star connectivity patterns and replace them with."""
from typing import Iterable, Union, Optional, List, Tuple
from math import floor, log10

from qiskit.circuit import SwitchCaseOp, Clbit, ClassicalRegister, Barrier
from qiskit.circuit.controlflow import condition_resources, node_resources
from qiskit.dagcircuit import DAGOpNode, DAGDepNode, DAGDependency, DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes.routing.sabre_swap import _build_sabre_dag, _apply_sabre_result

from qiskit._accelerate import star_prerouting
from qiskit._accelerate.nlayout import NLayout


class StarBlock:
    """Defines blocks representing star-shaped pieces of a circuit."""

    def __init__(self, nodes=None, center=None, num2q=0):
        self.center = center
        self.num2q = num2q
        self.nodes = [] if nodes is None else nodes

    def get_nodes(self):
        """Returns the list of nodes used in the block."""
        return self.nodes

    def append_node(self, node):
        """
        If node can be added to block while keeping the block star-shaped, and
        return True. Otherwise, does not add node to block and returns False.
        """

        added = False

        if len(node.qargs) == 1:
            self.nodes.append(node)
            added = True
        elif self.center is None:
            self.center = set(node.qargs)
            self.nodes.append(node)
            self.num2q += 1
            added = True
        elif isinstance(self.center, set):
            if node.qargs[0] in self.center:
                self.center = node.qargs[0]
                self.nodes.append(node)
                self.num2q += 1
                added = True
            elif node.qargs[1] in self.center:
                self.center = node.qargs[1]
                self.nodes.append(node)
                self.num2q += 1
                added = True
        else:
            if self.center in node.qargs:
                self.nodes.append(node)
                self.num2q += 1
                added = True

        return added

    def size(self):
        """
        Returns the number of two-qubit quantum gates in this block.
        """
        return self.num2q


class StarPreRouting(TransformationPass):
    """Run star to linear pre-routing

    This pass is a logical optimization pass that rewrites any
    solely 2q gate star connectivity subcircuit as a linear connectivity
    equivalent with swaps.

    For example:

      .. plot::
         :alt: Circuit diagram output by the previous code.
         :include-source:

         from qiskit.circuit import QuantumCircuit
         from qiskit.transpiler.passes import StarPreRouting

         qc = QuantumCircuit(10)
         qc.h(0)
         qc.cx(0, range(1, 5))
         qc.h(9)
         qc.cx(9, range(8, 4, -1))
         qc.measure_all()
         StarPreRouting()(qc).draw("mpl")

    This pass was inspired by a similar pass described in Section IV of:
    C. Campbell et al., "Superstaq: Deep Optimization of Quantum Programs,"
    2023 IEEE International Conference on Quantum Computing and Engineering (QCE),
    Bellevue, WA, USA, 2023, pp. 1020-1032, doi: 10.1109/QCE57702.2023.00116.
    """

    def __init__(self):
        """StarPreRouting"""

        self._pending_nodes: Optional[list[Union[DAGOpNode, DAGDepNode]]] = None
        self._in_degree: Optional[dict[Union[DAGOpNode, DAGDepNode], int]] = None
        super().__init__()

    def _setup_in_degrees(self, dag):
        """For an efficient implementation, for every node we keep the number of its
        unprocessed immediate predecessors (called ``_in_degree``). This ``_in_degree``
        is set up at the start and updated throughout the algorithm.
        A node is leaf (or input) node iff its ``_in_degree`` is 0.
        When a node is (marked as) collected, the ``_in_degree`` of each of its immediate
        successor is updated by subtracting 1.
        Additionally, ``_pending_nodes`` explicitly keeps the list of nodes whose
        ``_in_degree`` is 0.
        """
        self._pending_nodes = []
        self._in_degree = {}
        for node in self._op_nodes(dag):
            deg = len(self._direct_preds(dag, node))
            self._in_degree[node] = deg
            if deg == 0:
                self._pending_nodes.append(node)

    def _op_nodes(self, dag) -> Iterable[Union[DAGOpNode, DAGDepNode]]:
        """Returns DAG nodes."""
        if not isinstance(dag, DAGDependency):
            return dag.op_nodes()
        else:
            return dag.get_nodes()

    def _direct_preds(self, dag, node):
        """Returns direct predecessors of a node. This function takes into account the
        direction of collecting blocks, that is node's predecessors when collecting
        backwards are the direct successors of a node in the DAG.
        """
        if not isinstance(dag, DAGDependency):
            return [pred for pred in dag.predecessors(node) if isinstance(pred, DAGOpNode)]
        else:
            return [dag.get_node(pred_id) for pred_id in dag.direct_predecessors(node.node_id)]

    def _direct_succs(self, dag, node):
        """Returns direct successors of a node. This function takes into account the
        direction of collecting blocks, that is node's successors when collecting
        backwards are the direct predecessors of a node in the DAG.
        """
        if not isinstance(dag, DAGDependency):
            return [succ for succ in dag.successors(node) if isinstance(succ, DAGOpNode)]
        else:
            return [dag.get_node(succ_id) for succ_id in dag.direct_successors(node.node_id)]

    def _have_uncollected_nodes(self):
        """Returns whether there are uncollected (pending) nodes"""
        return len(self._pending_nodes) > 0

    def collect_matching_block(self, dag, filter_fn):
        """Iteratively collects the largest block of input nodes (that is, nodes with
        ``_in_degree`` equal to 0) that match a given filtering function.
        Examples of this include collecting blocks of swap gates,
        blocks of linear gates (CXs and SWAPs), blocks of Clifford gates, blocks of single-qubit gates,
        blocks of two-qubit gates, etc.  Here 'iteratively' means that once a node is collected,
        the ``_in_degree`` of each of its immediate successor is decreased by 1, allowing more nodes
        to become input and to be eligible for collecting into the current block.
        Returns the block of collected nodes.
        """
        unprocessed_pending_nodes = self._pending_nodes
        self._pending_nodes = []

        current_block = StarBlock()

        # Iteratively process unprocessed_pending_nodes:
        # - any node that does not match filter_fn is added to pending_nodes
        # - any node that match filter_fn is added to the current_block,
        #   and some of its successors may be moved to unprocessed_pending_nodes.
        while unprocessed_pending_nodes:
            new_pending_nodes = []
            for node in unprocessed_pending_nodes:
                added = filter_fn(node) and current_block.append_node(node)
                if added:
                    # update the _in_degree of node's successors
                    for suc in self._direct_succs(dag, node):
                        self._in_degree[suc] -= 1
                        if self._in_degree[suc] == 0:
                            new_pending_nodes.append(suc)
                else:
                    self._pending_nodes.append(node)
            unprocessed_pending_nodes = new_pending_nodes

        return current_block

    def collect_all_matching_blocks(
        self,
        dag,
        min_block_size=2,
    ):
        """Collects all blocks that match a given filtering function filter_fn.
        This iteratively finds the largest block that does not match filter_fn,
        then the largest block that matches filter_fn, and so on, until no more uncollected
        nodes remain. Intuitively, finding larger blocks of non-matching nodes helps to
        find larger blocks of matching nodes later on. The option ``min_block_size``
        specifies the minimum number of gates in the block for the block to be collected.

        By default, blocks are collected in the direction from the inputs towards the outputs
        of the circuit. The option ``collect_from_back`` allows to change this direction,
        that is collect blocks from the outputs towards the inputs of the circuit.

        Returns the list of matching blocks only.
        """

        def filter_fn(node):
            """Specifies which nodes can be collected into star blocks."""
            return (
                len(node.qargs) <= 2
                and len(node.cargs) == 0
                and getattr(node, "condition", None) is None
                and not isinstance(node.op, Barrier)
            )

        def not_filter_fn(node):
            """Returns the opposite of filter_fn."""
            return not filter_fn(node)

        # Note: the collection direction must be specified before setting in-degrees
        self._setup_in_degrees(dag)

        # Iteratively collect non-matching and matching blocks.
        matching_blocks: list[StarBlock] = []
        processing_order = []
        while self._have_uncollected_nodes():
            self.collect_matching_block(dag, filter_fn=not_filter_fn)
            matching_block = self.collect_matching_block(dag, filter_fn=filter_fn)
            if matching_block.size() >= min_block_size:
                matching_blocks.append(matching_block)
            processing_order.append(matching_block)

        processing_order = [n for p in processing_order for n in p.nodes]

        return matching_blocks, processing_order

    def run(self, dag):
        # Extract StarBlocks from DAGCircuit / DAGDependency / DAGDependencyV2
        star_blocks, processing_order = self.determine_star_blocks_processing(dag, min_block_size=2)

        if not star_blocks:
            return dag

        if all(b.size() < 3 for b in star_blocks):
            # we only process blocks with less than 3 two-qubit gates in this pre-routing pass
            # if they occur in a collection of larger stars, otherwise we consider them to be 'lines'
            return dag

        # Create a new DAGCircuit / DAGDependency / DAGDependencyV2, replacing each
        # star block by a linear sequence of gates
        new_dag, qubit_mapping = self.star_preroute(dag, star_blocks, processing_order)

        # Fix output permutation -- copied from ElidePermutations
        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = input_qubit_mapping

        new_layout = Layout({dag.qubits[out]: idx for idx, out in enumerate(qubit_mapping)})
        if current_layout := self.property_set["virtual_permutation_layout"]:
            self.property_set["virtual_permutation_layout"] = new_layout.compose(
                current_layout.inverse(dag.qubits, dag.qubits), dag.qubits
            )
        else:
            self.property_set["virtual_permutation_layout"] = new_layout

        return new_dag

    def determine_star_blocks_processing(
        self, dag: Union[DAGCircuit, DAGDependency], min_block_size: int
    ) -> Tuple[List[StarBlock], Union[List[DAGOpNode], List[DAGDepNode]]]:
        """Returns star blocks in dag and the processing order of nodes within these star blocks
        Args:
            dag (DAGCircuit or DAGDependency): a dag on which star blocks should be determined.
            min_block_size (int): minimum number of two-qubit gates in a star block.

        Returns:
            List[StarBlock]: a list of star blocks in the given dag
            Union[List[DAGOpNode], List[DAGDepNode]]: a list of operations specifying processing order
        """
        blocks, processing_order = self.collect_all_matching_blocks(
            dag, min_block_size=min_block_size
        )
        return blocks, processing_order

    def star_preroute(self, dag, blocks, processing_order):
        """Returns star blocks in dag and the processing order of nodes within these star blocks
        Args:
            dag (DAGCircuit or DAGDependency): a dag on which star prerouting should be performed.
            blocks (List[StarBlock]): a list of star blocks in the given dag.
            processing_order (Union[List[DAGOpNode], List[DAGDepNode]]): a list of operations specifying
            processing order

        Returns:
            new_dag: a dag specifying the pre-routed circuit
            qubit_mapping: the final qubit mapping after pre-routing
        """
        # Convert the DAG to a SabreDAG
        num_qubits = len(dag.qubits)
        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)
        qubit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}
        layout_mapping = {qubit_indices[k]: v for k, v in current_layout.get_virtual_bits().items()}
        initial_layout = NLayout(layout_mapping, num_qubits, num_qubits)
        sabre_dag, circuit_to_dag_dict = _build_sabre_dag(dag, num_qubits, qubit_indices)

        # Extract the nodes from the blocks for the Rust representation
        rust_blocks = [
            (block.center is not None, _extract_nodes(block.get_nodes(), dag)) for block in blocks
        ]

        # Determine the processing order of the nodes in the DAG for the Rust representation
        int_digits = floor(log10(len(processing_order))) + 1
        processing_order_index_map = {
            node: f"a{index:0{int_digits}}" for index, node in enumerate(processing_order)
        }

        def tie_breaker_key(node):
            return processing_order_index_map.get(node, node.sort_key)

        rust_processing_order = _extract_nodes(dag.topological_op_nodes(key=tie_breaker_key), dag)

        # Run the star prerouting algorithm to obtain the new DAG and qubit mapping
        *sabre_result, qubit_mapping = star_prerouting.star_preroute(
            sabre_dag, rust_blocks, rust_processing_order
        )

        res_dag = _apply_sabre_result(
            dag.copy_empty_like(),
            dag,
            sabre_result,
            initial_layout,
            dag.qubits,
            circuit_to_dag_dict,
        )

        return res_dag, qubit_mapping


def _extract_nodes(nodes, dag):
    """Extract and format node information for Rust representation used in SabreDAG.

    Each node is represented as a tuple containing:
    - Node ID (int): The unique identifier of the node in the DAG.
    - Qubit indices (list of int): Indices of qubits involved in the node's operation.
    - Classical bit indices (set of int): Indices of classical bits involved in the node's operation.
    - Directive flag (bool): Indicates whether the operation is a directive (True) or not (False).

    Args:
        nodes (list[DAGOpNode]): List of DAGOpNode objects to extract information from.
        dag (DAGCircuit): DAGCircuit object containing the circuit structure.

    Returns:
        list of tuples: Each tuple contains information about a node in the format described above.
    """
    extracted_node_info = []
    for node in nodes:
        qubit_indices = [dag.find_bit(qubit).index for qubit in node.qargs]
        classical_bit_indices = set()

        if node.condition is not None:
            classical_bit_indices.update(condition_resources(node.op.condition).clbits)

        if isinstance(node.op, SwitchCaseOp):
            switch_case_target = node.op.target
            if isinstance(switch_case_target, Clbit):
                classical_bit_indices.add(switch_case_target)
            elif isinstance(switch_case_target, ClassicalRegister):
                classical_bit_indices.update(switch_case_target)
            else:  # Assume target is an expression involving classical bits
                classical_bit_indices.update(node_resources(switch_case_target).clbits)

        is_directive = getattr(node.op, "_directive", False)
        extracted_node_info.append(
            (node._node_id, qubit_indices, classical_bit_indices, is_directive)
        )

    return extracted_node_info

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test random circuit generation utility."""
from collections import defaultdict
import rustworkx as rx
import numpy as np
import ddt
from qiskit.circuit import QuantumCircuit, ClassicalRegister, Clbit
from qiskit.circuit import Measure
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.random import random_circuit
from qiskit.circuit.random.utils import random_circuit_from_graph
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestCircuitRandom(QiskitTestCase):
    """Testing qiskit.circuit.random"""

    def test_simple_random(self):
        """Test creating a simple random circuit."""
        circ = random_circuit(num_qubits=5, depth=4)
        self.assertIsInstance(circ, QuantumCircuit)
        self.assertEqual(circ.width(), 5)
        self.assertEqual(circ.depth(), 4)

    def test_random_depth_0(self):
        """Test random depth 0 circuit."""
        circ = random_circuit(num_qubits=1, depth=0)
        self.assertEqual(circ.width(), 1)
        self.assertEqual(circ.depth(), 0)

    def test_random_measure(self):
        """Test random circuit with final measurement."""
        num_qubits = depth = 3
        circ = random_circuit(num_qubits, depth, measure=True)
        self.assertEqual(circ.width(), 2 * num_qubits)
        dag = circuit_to_dag(circ)
        for nd in list(dag.topological_op_nodes())[-num_qubits:]:
            self.assertIsInstance(nd.op, Measure)

    def test_random_circuit_conditional_reset(self):
        """Test generating random circuits with conditional and reset."""
        num_qubits = 1
        depth = 100
        with self.assertWarns(DeprecationWarning):
            circ = random_circuit(num_qubits, depth, conditional=True, reset=True, seed=5)
        self.assertEqual(circ.width(), 2 * num_qubits)
        self.assertIn("reset", circ.count_ops())

    def test_large_conditional(self):
        """Test that conditions do not fail with large conditionals.  Regression test of gh-6994."""
        # The main test is that this call actually returns without raising an exception.
        with self.assertWarns(DeprecationWarning):
            circ = random_circuit(64, 2, conditional=True, seed=0)
        # Test that at least one instruction had a condition generated.  It's possible that this
        # fails due to very bad luck with the random seed - if so, change the seed to ensure that a
        # condition _is_ generated, because we need to test that generation doesn't error.
        conditions = (getattr(instruction.operation, "_condition", None) for instruction in circ)
        conditions = [x for x in conditions if x is not None]
        self.assertNotEqual(conditions, [])
        for register, value in conditions:
            self.assertIsInstance(register, (ClassicalRegister, Clbit))
            # Condition values always have to be Python bigints (of which `bool` is a subclass), not
            # any of Numpy's fixed-width types, for example.
            self.assertIsInstance(value, int)

    def test_random_mid_circuit_measure_conditional(self):
        """Test random circuit with mid-circuit measurements for conditionals."""
        num_qubits = depth = 2
        with self.assertWarns(DeprecationWarning):
            circ = random_circuit(num_qubits, depth, conditional=True, seed=17)
        self.assertEqual(circ.width(), 2 * num_qubits)
        op_names = [instruction.operation.name for instruction in circ]
        self.assertEqual(6, len(op_names))

        # Before a condition, there needs to be measurement in all the qubits.
        # The last operation `U` gate is a conditional operation. This is
        # particular to the given seed, hence the circuit generated.
        start_idx = len(op_names) - 3
        self.assertEqual(["measure"] * num_qubits, op_names[start_idx : start_idx + num_qubits])
        conditions = [
            bool(getattr(instruction.operation, "_condition", None)) for instruction in circ
        ]
        self.assertEqual([False, False, False, False, False, True], conditions)

    def test_random_circuit_num_operand_distribution(self):
        """Test that num_operand_distribution argument generates gates in correct proportion"""
        num_qubits = 50
        depth = 300
        num_op_dist = {2: 0.25, 3: 0.25, 1: 0.25, 4: 0.25}
        circ = random_circuit(
            num_qubits, depth, num_operand_distribution=num_op_dist, seed=123456789
        )
        total_gates = circ.size()
        self.assertEqual(circ.width(), num_qubits)
        self.assertEqual(circ.depth(), depth)
        gate_qubits = [instruction.operation.num_qubits for instruction in circ]
        gate_type_counter = np.bincount(gate_qubits, minlength=5)
        for gate_type, prob in sorted(num_op_dist.items()):
            self.assertAlmostEqual(prob, gate_type_counter[gate_type] / total_gates, delta=0.1)

    def test_random_circuit_2and3_qubit_gates_only(self):
        """
        Test that the generated random circuit only has 2 and 3 qubit gates,
        while disallowing 1-qubit and 4-qubit gates if
        num_operand_distribution = {2: some_prob, 3: some_prob}
        """
        num_qubits = 10
        depth = 200
        num_op_dist = {2: 0.5, 3: 0.5}
        circ = random_circuit(num_qubits, depth, num_operand_distribution=num_op_dist, seed=200)
        total_gates = circ.size()
        gate_qubits = [instruction.operation.num_qubits for instruction in circ]
        gate_type_counter = np.bincount(gate_qubits, minlength=5)
        # Testing that the distribution of 2 and 3 qubit gate matches with given distribution
        for gate_type, prob in sorted(num_op_dist.items()):
            self.assertAlmostEqual(prob, gate_type_counter[gate_type] / total_gates, delta=0.1)
        # Testing that there are no 1-qubit gate and 4-qubit in the generated random circuit
        self.assertEqual(gate_type_counter[1], 0.0)
        self.assertEqual(gate_type_counter[4], 0.0)

    def test_random_circuit_3and4_qubit_gates_only(self):
        """
        Test that the generated random circuit only has 3 and 4 qubit gates,
        while disallowing 1-qubit and 2-qubit gates if
        num_operand_distribution = {3: some_prob, 4: some_prob}
        """
        num_qubits = 10
        depth = 200
        num_op_dist = {3: 0.5, 4: 0.5}
        circ = random_circuit(
            num_qubits, depth, num_operand_distribution=num_op_dist, seed=11111111
        )
        total_gates = circ.size()
        gate_qubits = [instruction.operation.num_qubits for instruction in circ]
        gate_type_counter = np.bincount(gate_qubits, minlength=5)
        # Testing that the distribution of 3 and 4 qubit gate matches with given distribution
        for gate_type, prob in sorted(num_op_dist.items()):
            self.assertAlmostEqual(prob, gate_type_counter[gate_type] / total_gates, delta=0.1)
        # Testing that there are no 1-qubit gate and 2-qubit in the generated random circuit
        self.assertEqual(gate_type_counter[1], 0.0)
        self.assertEqual(gate_type_counter[2], 0.0)

    def test_random_circuit_with_zero_distribution(self):
        """
        Test that the generated random circuit only has 3 and 4 qubit gates,
        while disallowing 1-qubit and 2-qubit gates if
        num_operand_distribution = {1: 0.0, 2: 0.0, 3: some_prob, 4: some_prob}
        """
        num_qubits = 10
        depth = 200
        num_op_dist = {1: 0.0, 2: 0.0, 3: 0.5, 4: 0.5}
        circ = random_circuit(num_qubits, depth, num_operand_distribution=num_op_dist, seed=12)
        total_gates = circ.size()
        gate_qubits = [instruction.operation.num_qubits for instruction in circ]
        gate_type_counter = np.bincount(gate_qubits, minlength=5)
        # Testing that the distribution of 3 and 4 qubit gate matches with given distribution
        for gate_type, prob in sorted(num_op_dist.items()):
            self.assertAlmostEqual(prob, gate_type_counter[gate_type] / total_gates, delta=0.1)
        # Testing that there are no 1-qubit gate and 2-qubit in the generated random circuit
        self.assertEqual(gate_type_counter[1], 0.0)
        self.assertEqual(gate_type_counter[2], 0.0)


def incomplete_graph(n_nodes):
    # pylint: disable=missing-function-docstring
    pydi_graph = rx.generators.directed_complete_graph(n_nodes)
    pydi_graph.remove_edge(1, 3)
    return pydi_graph


def digraph_with_no_edges(n_nodes):
    # pylint: disable=missing-function-docstring
    graph = rx.PyDiGraph()
    graph.add_nodes_from(range(n_nodes))
    return graph


test_cases = (
    (rx.generators.directed_cycle_graph(5), 1550),
    (rx.generators.directed_mesh_graph(4), 87978),
    # The (almost) fully connected graph.
    (incomplete_graph(4), 154),
    (rx.generators.directed_heavy_hex_graph(3), 458),
    # Sparse connected graph
    (rx.generators.directed_path_graph(10), 458),
    # A graph with no edges, should yeild a circuit with no edges,
    # this means there would be no 2Q gates on that circuit.
    (digraph_with_no_edges(10), 0),
)


@ddt.ddt
class TestRandomCircuitFromGraph(QiskitTestCase):
    """Testing random_circuit_from_graph from
    qiskit.circuit.random.utils.py"""

    @ddt.data(*test_cases)
    @ddt.unpack
    def test_simple_random(self, inter_graph, seed):
        """Test creating a simple random circuit."""

        n_nodes = inter_graph.num_nodes()
        circ = random_circuit_from_graph(
            interaction_graph=inter_graph, min_2q_gate_per_edge=1, seed=seed
        )

        self.assertIsInstance(circ, QuantumCircuit)
        self.assertEqual(circ.width(), n_nodes)

    @ddt.data(*test_cases)
    @ddt.unpack
    def test_min_times_qubit_pair_usage(self, inter_graph, seed):
        """the `min_2q_gate_per_edge` parameter specifies how often each qubit-pair must at
        least be used in a two-qubit gate before the circuit is returned"""

        freq = 1
        qc = random_circuit_from_graph(
            interaction_graph=inter_graph, min_2q_gate_per_edge=freq, seed=seed
        )
        dag = circuit_to_dag(qc)
        count_register = defaultdict(int)

        for wire in dag.wires:
            for node in dag.nodes_on_wire(wire, only_ops=True):
                if node.op.name == "measure" or node.op.num_qubits < 2:
                    continue
                q_args = node.qargs
                count_register[(q_args[0]._index, q_args[1]._index)] += 1

        for occurence in count_register.values():
            self.assertLessEqual(freq, occurence)

    @ddt.data(*test_cases)
    @ddt.unpack
    def test_random_measure(self, inter_graph, seed):
        """Test random circuit with final measurement."""

        qc = random_circuit_from_graph(
            interaction_graph=inter_graph, min_2q_gate_per_edge=1, measure=True, seed=seed
        )
        self.assertIn("measure", qc.count_ops())

    @ddt.data(*test_cases)
    @ddt.unpack
    def test_random_circuit_conditional_reset(self, inter_graph, seed):
        """Test generating random circuits with conditional and reset."""

        with self.assertWarns(DeprecationWarning):
            qc = random_circuit_from_graph(
                interaction_graph=inter_graph,
                min_2q_gate_per_edge=2,
                conditional=True,
                reset=True,
                seed=seed,  # Do not change the seed or the args.
                insert_1q_oper=True,
                prob_conditional=0.41,
                prob_reset=0.50,
            )

        self.assertIn("reset", qc.count_ops())

        # Now, checking for conditionals
        conditions = []
        for instr in qc:
            cond = getattr(instr.operation, "_condition", None)
            if not cond is None:
                conditions.append(cond)

        # See if conditionals are present.
        self.assertNotEqual(conditions, [])

    @ddt.data(*test_cases)
    @ddt.unpack
    def test_2q_gates_applied_to_edges_from_interaction_graph(self, inter_graph, seed):
        """Test 2Q gates are applied to the qubit-pairs given by the interaction graph supplied"""
        with self.assertWarns(DeprecationWarning):
            qc = random_circuit_from_graph(
                interaction_graph=inter_graph,
                min_2q_gate_per_edge=2,
                measure=True,
                conditional=True,
                reset=True,
                insert_1q_oper=True,
                seed=seed,  # Do not change the seed or args
                prob_conditional=0.41,
                prob_reset=0.50,
            )
        dag = circuit_to_dag(qc)

        cp_mp = set()
        edge_list = set(inter_graph.edge_list())

        for wire in dag.wires:
            for dag_op_node in dag.nodes_on_wire(wire, only_ops=True):
                if dag_op_node.op.num_qubits == 2:
                    control, target = dag_op_node.qargs
                    control_idx = control._index
                    target_idx = target._index
                    cp_mp.update({(control_idx, target_idx)})

        # make sure every qubit-pair from the circuit actually present in the edge_list
        for cp in cp_mp:
            self.assertTrue(cp in edge_list)

    def test_2q_gates_excluded_edges_with_zero_weight(self):
        """Test 2Q gates are not applied to the qubit-pairs given by the interaction graph
        whose weight is zero"""

        num_qubits = 7
        pydi_graph = rx.PyDiGraph()
        pydi_graph.add_nodes_from(range(num_qubits))
        cp_map = [(0, 1, 10), (1, 2, 11), (2, 3, 0), (3, 4, 9), (4, 5, 12), (5, 6, 13)]
        pydi_graph.add_edges_from(cp_map)

        qc = random_circuit_from_graph(
            interaction_graph=pydi_graph,
            min_2q_gate_per_edge=1,
        )
        dag = circuit_to_dag(qc)

        ckt_cp_mp = set()
        for wire in dag.wires:
            for dag_op_node in dag.nodes_on_wire(wire, only_ops=True):
                if dag_op_node.op.num_qubits == 2:
                    control, target = dag_op_node.qargs
                    control_idx = control._index
                    target_idx = target._index
                    ckt_cp_mp.update({(control_idx, target_idx)})

        # make sure qubit-pair with zero weight is not present in the edge_list from
        # the circuit.
        self.assertFalse((2, 3) in ckt_cp_mp)

    def test_edges_weight_with_some_None_raises(self):
        """Test if the function raises ValueError, if some of the edge
        weights are None, but not all."""

        pydi_graph = rx.PyDiGraph()
        pydi_graph.add_nodes_from(range(5))
        cp_mp = [(0, 1, None), (1, 2, 54), (2, 3, 23), (3, 4, 32)]

        pydi_graph.add_edges_from(cp_mp)
        with self.assertRaisesRegex(ValueError, ".getting selected is."):
            _ = random_circuit_from_graph(
                interaction_graph=pydi_graph,
                min_2q_gate_per_edge=1,
            )

    def test_max_operands_not_between_1_2_raises(self):
        """Test if the function raises CircuitError when max_operands is not 1 or 2"""

        pydi_graph = rx.PyDiGraph()
        pydi_graph.add_nodes_from(range(10))
        with self.assertRaisesRegex(CircuitError, ".function is intended to only work on."):
            _ = random_circuit_from_graph(
                interaction_graph=pydi_graph,
                min_2q_gate_per_edge=2,
                max_operands=3,  # This would fail
            )

    def test_negative_edge_weight_raises(self):
        """Test if negative edge weights raises ValueError"""

        pydi_graph = rx.PyDiGraph()
        pydi_graph.add_nodes_from(range(5))
        cp_mp = [(0, 1, -10), (1, 2, 54), (2, 3, 23), (3, 4, 32)]

        pydi_graph.add_edges_from(cp_mp)
        with self.assertRaisesRegex(ValueError, ".probability."):
            _ = random_circuit_from_graph(interaction_graph=pydi_graph, min_2q_gate_per_edge=1)

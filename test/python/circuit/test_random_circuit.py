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
import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister, Clbit
from qiskit.circuit import Measure
from qiskit.circuit.random import random_circuit
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
            circ = random_circuit(num_qubits, depth, conditional=True, seed=16)
        self.assertEqual(circ.width(), 2 * num_qubits)
        op_names = [instruction.operation.name for instruction in circ]
        # Before a condition, there needs to be measurement in all the qubits.
        self.assertEqual(4, len(op_names))
        self.assertEqual(["measure"] * num_qubits, op_names[1 : 1 + num_qubits])
        conditions = [
            bool(getattr(instruction.operation, "_condition", None)) for instruction in circ
        ]
        self.assertEqual([False, False, False, True], conditions)

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

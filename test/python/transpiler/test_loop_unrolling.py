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

"""Tests for loop unrolling passes"""

from math import pi
from ddt import ddt, data

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.optimization import UnrollForLoops, ForLoopBodyOptimizer
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.test import QiskitTestCase
from qiskit.quantum_info import Operator


@ddt
class TestUnrollForLoop(QiskitTestCase):
    """Tests for unrolling for loops"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._pass = UnrollForLoops()

    @data(1, 2, 3)
    def test_only_for_loop(self, nloops):
        """Test unrolling circuit with only a for-loop"""
        body = QuantumCircuit(1)
        body.rx(0.1, 0)
        circ = QuantumCircuit(1)
        circ.for_loop(range(nloops), None, body.copy(), [0], [])
        ccirc = self._pass(circ)
        self.assertEqual(Operator(body.repeat(nloops)), Operator(ccirc))

    @data(1, 2, 3)
    def test_parametrized_for_loop(self, nloops):
        """Test unrolling parameterized circuits."""
        body = QuantumCircuit(1)
        theta = Parameter("θ")
        body.rx(theta + pi / 2, 0)
        circ = QuantumCircuit(1)
        circ.for_loop(range(nloops), theta, body.copy(), [0], [])
        ccirc = self._pass(circ)

        expected = QuantumCircuit(1)
        for i in range(nloops):
            expected.rx(i, 0)
        self.assertEqual(Operator(expected), Operator(ccirc))

    def test_for_loop_compound(self):
        """Test unrolling for-loop amidst other operations"""
        body = QuantumCircuit(1)
        body.rx(pi, 0)
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.for_loop(range(2), None, body, [0], [])
        circ.y(0)

        expected = QuantumCircuit(1)
        expected.x(0)
        expected.rx(pi, 0)
        expected.rx(pi, 0)
        expected.y(0)
        self.assertEqual(expected, self._pass(circ))


@ddt
class TestForLoopBodyOptimizer(QiskitTestCase):
    """Test optimization of for loops."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._pass = UnrollForLoops()

    def test_pre_opt_1(self):
        """Test pre-optimization to pre_opt_limit=1"""
        pre_opt_limit = 1
        circ = QuantumCircuit(2)
        circ.rx(0.1, 0)
        circ.rx(0.2, 0)
        circ.x(0)
        # define for loop
        body = QuantumCircuit(1)
        body.rx(pi / 2, 0)
        circ.for_loop(range(2), None, body, [0], [])
        basis_gates = ["x", "rx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        pm_opt = PassManager(_opt)
        _pass = ForLoopBodyOptimizer(pm_opt, pre_opt_limit=pre_opt_limit)
        ccirc = _pass(circ)

        expected = QuantumCircuit(2)
        expected.rx(0.1, 0)
        expected.rx(0.2, 0)
        expected.rx(3 * pi / 2, 0)
        expected.for_loop(range(1), None, body, [0], [])

        self.assertEqual(ccirc, expected)

    def test_pre_opt_2(self):
        """Test pre-optimization to pre_opt_limit=2"""
        pre_opt_limit = 2
        circ = QuantumCircuit(2)
        circ.rx(0.1, 0)
        circ.rx(0.2, 0)
        circ.x(0)
        # define for loop
        body = QuantumCircuit(1)
        body.rx(pi / 2, 0)
        circ.for_loop(range(2), None, body, [0], [])
        basis_gates = ["x", "rx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        pm_opt = PassManager(_opt)
        _pass = ForLoopBodyOptimizer(pm_opt, pre_opt_limit=pre_opt_limit)
        ccirc = _pass(circ)

        expected = QuantumCircuit(2)
        expected.rx(0.1, 0)
        expected.rx(pi / 2 + pi + 0.2, 0)
        expected.for_loop(range(1), None, body, [0], [])

        self.assertEqual(ccirc, expected)

    def test_pre_opt_cnot(self):
        """
        Test that the circuit,
                ┌─────────┐┌───┐┌───┐┌───────────┐
           q_0: ┤ Rx(0.1) ├┤ X ├┤ X ├┤0          ├
                └─────────┘└─┬─┘└───┘│  For_loop │
           q_1: ─────────────■───────┤1          ├
                                     └───────────┘
        where the body of the for-loop is
                ┌─────────┐
           q_0: ┤ Rx(π/2) ├──■──
                └─────────┘┌─┴─┐
           q_1: ───────────┤ X ├
                           └───┘
        becomes,
                ┌────────────┐┌───┐     ┌───────────┐
           q_0: ┤ Rx(4.8124) ├┤ X ├──■──┤0          ├
                └────────────┘└─┬─┘┌─┴─┐│  For_loop │
           q_1: ────────────────■──┤ X ├┤1          ├
                                   └───┘└───────────┘
        """
        pre_opt_limit = 3
        circ = QuantumCircuit(2)
        circ.rx(0.1, 0)
        circ.cx(1, 0)
        circ.x(0)
        # define for loop
        body = QuantumCircuit(2)
        body.rx(pi / 2, 0)
        body.cx(0, 1)
        circ.for_loop(range(2), None, body, [0, 1], [])
        basis_gates = ["x", "rx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        pm_opt = PassManager(_opt)
        _pass = ForLoopBodyOptimizer(pm_opt, pre_opt_limit=pre_opt_limit)
        ccirc = _pass(circ)

        expected = QuantumCircuit(2)
        expected.rx(0.1 + pi + pi / 2, 0)
        expected.cx(1, 0)
        expected.cx(0, 1)
        expected.for_loop(range(1), None, body, [0, 1], [])
        self.assertEqual(ccirc, expected)

    def test_pre_opt_breadth(self):
        """
        Test that the circuit,
             ┌─────────┐┌───┐┌───────────┐
        q_0: ┤ Rx(0.1) ├┤ X ├┤0          ├
             ├─────────┤├───┤│  For_loop │
        q_1: ┤ Rx(0.2) ├┤ X ├┤1          ├
             └──┬───┬──┘└─┬─┘└───┬───┬───┘
        q_2: ───┤ Z ├─────■──────┤ Z ├────
                └───┘            └───┘
        with for loop body,
             ┌─────────┐
        q_0: ┤ Rx(π/2) ├──■─────────────
             └─────────┘┌─┴─┐┌─────────┐
        q_1: ───────────┤ X ├┤ Rx(0.3) ├
                        └───┘└─────────┘

        Where this circuit considers operations on qubits outside
        """
        pre_opt_limit = 10
        circ = QuantumCircuit(3)
        circ.rx(0.1, 0)
        circ.x(0)
        circ.z(2)
        circ.rx(0.2, 1)
        circ.cx(2, 1)
        circ.z(2)
        # define for loop
        body = QuantumCircuit(2)
        body.rx(pi / 2, 0)
        body.cx(0, 1)
        body.rx(0.3, 1)
        circ.for_loop(range(3), None, body, [0, 1], [])
        basis_gates = ["x", "rx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        pm_opt = PassManager(_opt)
        _pass = ForLoopBodyOptimizer(pm_opt, pre_opt_limit=pre_opt_limit)
        ccirc = _pass(circ)

        expected = QuantumCircuit(3)
        expected.rx(0.1 + pi + pi / 2, 0)
        expected.rx(0.2 + 0.3, 1)
        expected.z(2)
        expected.cx(2, 1)
        expected.cx(0, 1)
        expected.z(2)
        expected.for_loop(range(2), None, body, [0, 1], [])
        self.assertEqual(ccirc, expected)

    def test_post_opt(self):
        """Test optimization with operations after for-loop"""
        circ = QuantumCircuit(2)
        # define for loop
        body = QuantumCircuit(1)
        body.rx(pi / 2, 0)
        circ.for_loop(range(2), None, body, [0], [])
        circ.rx(0.1, 0)
        circ.rx(0.2, 0)
        circ.x(0)
        basis_gates = ["x", "rx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        pm_opt = PassManager(_opt)
        _pass = ForLoopBodyOptimizer(pm_opt, post_opt_limit=3)
        ccirc = _pass(circ)

        expected = QuantumCircuit(2)
        expected.for_loop(range(1), None, body, [0], [])
        expected.rx(3 * pi / 2 + 0.1 + 0.2, 0)
        self.assertEqual(ccirc, expected)

    def test_loop_jamming_basic_even(self):
        """basic loop jaming test with an even number of loops"""
        circ = QuantumCircuit(2)
        body = QuantumCircuit(2)
        body.rx(0.1, 0)
        body.cx(0, 1)
        body.rx(0.1, 0)
        circ.for_loop(range(4), None, body, [0, 1], [])
        basis_gates = ["x", "rx", "cx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        pm_opt = PassManager(_opt)
        _pass = ForLoopBodyOptimizer(pm_opt, jamming_limit=4)
        ccirc = _pass(circ)

        expected = QuantumCircuit(2)
        expected_body = QuantumCircuit(2)
        expected_body.rx(0.1, 0)
        expected_body.cx(0, 1)
        expected_body.rx(0.2, 0)
        expected_body.cx(0, 1)
        expected_body.rx(0.1, 0)
        expected.for_loop(range(2), None, expected_body, [0, 1], [])
        self.assertEqual(ccirc, expected)

    def test_loop_jamming_basic_odd(self):
        """basic loop jaming test with an odd number of loops"""
        circ = QuantumCircuit(2)
        body = QuantumCircuit(2)
        body.rx(0.1, 0)
        body.cx(0, 1)
        body.rx(0.1, 0)
        circ.for_loop(range(5), None, body, [0, 1], [])
        basis_gates = ["x", "rx", "cx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        pm_opt = PassManager(_opt)
        _pass = ForLoopBodyOptimizer(pm_opt, jamming_limit=4)
        ccirc = _pass(circ)

        expected = QuantumCircuit(2)
        expected_body = QuantumCircuit(2)
        expected_body.rx(0.1, 0)
        expected_body.cx(0, 1)
        expected_body.rx(0.2, 0)
        expected_body.cx(0, 1)
        expected_body.rx(0.1, 0)
        expected.for_loop(range(2), None, expected_body, [0, 1], [])
        expected.rx(0.1, 0)
        expected.cx(0, 1)
        expected.rx(0.1, 0)
        self.assertEqual(ccirc, expected)

    def test_loop_jamming_limit_too_small(self):
        """basic loop jaming test with jamming limit set < 2"""
        basis_gates = ["x", "rx", "cx"]
        _opt = [
            Optimize1qGatesDecomposition(basis=basis_gates),
            CommutativeCancellation(basis_gates=basis_gates),
        ]
        pm_opt = PassManager(_opt)
        with self.assertRaises(TranspilerError):
            _pass = ForLoopBodyOptimizer(pm_opt, jamming_limit=1)

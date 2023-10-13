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

"""Tests for pass cancelling 2 consecutive CNOTs on the same qubits."""
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Clbit, Qubit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CXCancellation
from qiskit.test import QiskitTestCase


class TestCXCancellation(QiskitTestCase):
    """Test the CXCancellation pass."""

    def test_pass_cx_cancellation(self):
        """Test the cx cancellation pass.

        It should cancel consecutive cx pairs on same qubits.
        """
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[1], qr[0])

        pass_manager = PassManager()
        pass_manager.append(CXCancellation())
        out_circuit = pass_manager.run(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr[0])
        expected.h(qr[0])

        self.assertEqual(out_circuit, expected)

    def test_pass_cx_cancellation_intermixed_ops(self):
        """Cancellation shouldn't be affected by the order of ops on different qubits."""
        qr = QuantumRegister(4)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])

        pass_manager = PassManager()
        pass_manager.append(CXCancellation())
        out_circuit = pass_manager.run(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr[0])
        expected.h(qr[1])

        self.assertEqual(out_circuit, expected)

    def test_pass_cx_cancellation_chained_cx(self):
        """Include a test were not all operations can be cancelled."""

        #       ┌───┐
        # q0_0: ┤ H ├──■─────────■───────
        #       ├───┤┌─┴─┐     ┌─┴─┐
        # q0_1: ┤ H ├┤ X ├──■──┤ X ├─────
        #       └───┘└───┘┌─┴─┐└───┘
        # q0_2: ──────────┤ X ├──■────■──
        #                 └───┘┌─┴─┐┌─┴─┐
        # q0_3: ───────────────┤ X ├┤ X ├
        #                      └───┘└───┘
        qr = QuantumRegister(4)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[2], qr[3])

        pass_manager = PassManager()
        pass_manager.append(CXCancellation())
        out_circuit = pass_manager.run(circuit)

        #       ┌───┐
        # q0_0: ┤ H ├──■─────────■──
        #       ├───┤┌─┴─┐     ┌─┴─┐
        # q0_1: ┤ H ├┤ X ├──■──┤ X ├
        #       └───┘└───┘┌─┴─┐└───┘
        # q0_2: ──────────┤ X ├─────
        #                 └───┘
        # q0_3: ────────────────────
        expected = QuantumCircuit(qr)
        expected.h(qr[0])
        expected.h(qr[1])
        expected.cx(qr[0], qr[1])
        expected.cx(qr[1], qr[2])
        expected.cx(qr[0], qr[1])

        self.assertEqual(out_circuit, expected)

    def test_swapped_cx(self):
        """Test that CX isn't cancelled if there are intermediary ops."""
        qr = QuantumRegister(4)
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])
        circuit.swap(qr[1], qr[2])
        circuit.cx(qr[1], qr[0])

        pass_manager = PassManager()
        pass_manager.append(CXCancellation())
        out_circuit = pass_manager.run(circuit)
        self.assertEqual(out_circuit, circuit)

    def test_inverted_cx(self):
        """Test that CX order dependence is respected."""
        qr = QuantumRegister(4)
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[1])

        pass_manager = PassManager()
        pass_manager.append(CXCancellation())
        out_circuit = pass_manager.run(circuit)
        self.assertEqual(out_circuit, circuit)

    def test_if_else(self):
        """Test that the pass recurses in a simple if-else."""
        pass_ = CXCancellation()

        inner_test = QuantumCircuit(4, 1)
        inner_test.cx(0, 1)
        inner_test.cx(0, 1)
        inner_test.cx(2, 3)

        inner_expected = QuantumCircuit(4, 1)
        inner_expected.cx(2, 3)

        test = QuantumCircuit(4, 1)
        test.h(0)
        test.measure(0, 0)
        test.if_else((0, True), inner_test.copy(), inner_test.copy(), range(4), [0])

        expected = QuantumCircuit(4, 1)
        expected.h(0)
        expected.measure(0, 0)
        expected.if_else((0, True), inner_expected, inner_expected, range(4), [0])

        self.assertEqual(pass_(test), expected)

    def test_nested_control_flow(self):
        """Test that collection recurses into nested control flow."""
        pass_ = CXCancellation()
        qubits = [Qubit() for _ in [None] * 4]
        clbit = Clbit()

        inner_test = QuantumCircuit(qubits, [clbit])
        inner_test.cx(0, 1)
        inner_test.cx(0, 1)
        inner_test.cx(2, 3)

        inner_expected = QuantumCircuit(qubits, [clbit])
        inner_expected.cx(2, 3)

        true_body = QuantumCircuit(qubits, [clbit])
        true_body.while_loop((clbit, True), inner_test.copy(), [0, 1, 2, 3], [0])

        test = QuantumCircuit(qubits, [clbit])
        test.for_loop(range(2), None, inner_test.copy(), [0, 1, 2, 3], [0])
        test.if_else((clbit, True), true_body, None, [0, 1, 2, 3], [0])

        expected_if_body = QuantumCircuit(qubits, [clbit])
        expected_if_body.while_loop((clbit, True), inner_expected, [0, 1, 2, 3], [0])
        expected = QuantumCircuit(qubits, [clbit])
        expected.for_loop(range(2), None, inner_expected, [0, 1, 2, 3], [0])
        expected.if_else((clbit, True), expected_if_body, None, [0, 1, 2, 3], [0])

        self.assertEqual(pass_(test), expected)

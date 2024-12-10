# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Testing InverseCancellation
"""

import unittest
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import InverseCancellation
from qiskit.transpiler import PassManager
from qiskit.circuit import Clbit, Qubit
from qiskit.circuit.library import (
    RXGate,
    HGate,
    CXGate,
    PhaseGate,
    XGate,
    TGate,
    TdgGate,
    CZGate,
    RZGate,
)
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestInverseCancellation(QiskitTestCase):
    """Test the InverseCancellation transpiler pass."""

    def test_basic_self_inverse(self):
        """Test that a single self-inverse gate as input can be cancelled."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        pass_ = InverseCancellation([HGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("h", gates_after)

    def test_odd_number_self_inverse(self):
        """Test that an odd number of self-inverse gates leaves one gate remaining."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        pass_ = InverseCancellation([HGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn("h", gates_after)
        self.assertEqual(gates_after["h"], 1)

    def test_basic_cx_self_inverse(self):
        """Test that a single self-inverse cx gate as input can be cancelled."""
        qc = QuantumCircuit(2, 2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        pass_ = InverseCancellation([CXGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("cx", gates_after)

    def test_basic_gate_inverse(self):
        """Test that a basic pair of gate inverse can be cancelled."""
        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi / 4, 0)
        qc.rx(-np.pi / 4, 0)
        pass_ = InverseCancellation([(RXGate(np.pi / 4), RXGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("rx", gates_after)

    def test_non_inverse_do_not_cancel(self):
        """Test that non-inverse gate pairs do not cancel."""
        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi / 4, 0)
        qc.rx(np.pi / 4, 0)
        pass_ = InverseCancellation([(RXGate(np.pi / 4), RXGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn("rx", gates_after)
        self.assertEqual(gates_after["rx"], 2)

    def test_non_consecutive_gates(self):
        """Test that only consecutive gates cancel."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.h(0)
        pass_ = InverseCancellation([HGate(), CXGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("cx", gates_after)
        self.assertEqual(gates_after["h"], 2)

    def test_gate_inverse_phase_gate(self):
        """Test that an inverse pair of a PhaseGate can be cancelled."""
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("p", gates_after)

    def test_self_inverse_on_different_qubits(self):
        """Test that self_inverse gates cancel on the correct qubits."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(1)
        qc.h(0)
        qc.h(1)
        pass_ = InverseCancellation([HGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("h", gates_after)

    def test_non_inverse_raise_error(self):
        """Test that non-inverse gate inputs raise an error."""
        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi / 2, 0)
        qc.rx(np.pi / 4, 0)
        with self.assertRaises(TranspilerError):
            InverseCancellation([RXGate(0.5)])

    def test_non_gate_inverse_raise_error(self):
        """Test that non-inverse gate inputs raise an error."""
        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi / 4, 0)
        qc.rx(np.pi / 4, 0)
        with self.assertRaises(TranspilerError):
            InverseCancellation([(RXGate(np.pi / 4))])

    def test_string_gate_error(self):
        """Test that when gate is passed as a string an error is raised."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        with self.assertRaises(TranspilerError):
            InverseCancellation(["h"])

    def test_consecutive_self_inverse_h_x_gate(self):
        """Test that only consecutive self-inverse gates cancel."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        qc.x(0)
        qc.x(0)
        qc.h(0)
        pass_ = InverseCancellation([HGate(), XGate()])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("x", gates_after)
        self.assertEqual(gates_after["h"], 2)

    def test_inverse_with_different_names(self):
        """Test that inverse gates that have different names."""
        qc = QuantumCircuit(2, 2)
        qc.t(0)
        qc.tdg(0)
        pass_ = InverseCancellation([(TGate(), TdgGate())])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("t", gates_after)
        self.assertNotIn("tdg", gates_after)

    def test_three_alternating_inverse_gates(self):
        """Test that inverse cancellation works correctly for alternating sequences
        of inverse gates of odd-length."""
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn("p", gates_after)
        self.assertEqual(gates_after["p"], 1)

    def test_four_alternating_inverse_gates(self):
        """Test that inverse cancellation works correctly for alternating sequences
        of inverse gates of even-length."""
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertNotIn("p", gates_after)

    def test_five_alternating_inverse_gates(self):
        """Test that inverse cancellation works correctly for alternating sequences
        of inverse gates of odd-length."""
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn("p", gates_after)
        self.assertEqual(gates_after["p"], 1)

    def test_sequence_of_inverse_gates_1(self):
        """Test that inverse cancellation works correctly for more general sequences
        of inverse gates. In this test two pairs of inverse gates are supposed to
        cancel out."""
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn("p", gates_after)
        self.assertEqual(gates_after["p"], 1)

    def test_sequence_of_inverse_gates_2(self):
        """Test that inverse cancellation works correctly for more general sequences
        of inverse gates. In this test, in theory three pairs of inverse gates can
        cancel out, but in practice only two pairs are back-to-back."""
        qc = QuantumCircuit(2, 2)
        qc.p(np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(-np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        qc.p(np.pi / 4, 0)
        pass_ = InverseCancellation([(PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4))])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn("p", gates_after)
        self.assertEqual(gates_after["p"] % 2, 1)

    def test_cx_do_not_wrongly_cancel(self):
        """Test that CX(0,1) and CX(1, 0) do not cancel out, when (CX, CX) is passed
        as an inverse pair."""
        qc = QuantumCircuit(2, 0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        pass_ = InverseCancellation([(CXGate(), CXGate())])
        pm = PassManager(pass_)
        new_circ = pm.run(qc)
        gates_after = new_circ.count_ops()
        self.assertIn("cx", gates_after)
        self.assertEqual(gates_after["cx"], 2)

    def test_no_gates_to_cancel(self):
        """Test when there are no gates to cancel."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        inverse_pass = InverseCancellation([HGate()])
        new_circ = inverse_pass(qc)
        self.assertEqual(qc, new_circ)

    def test_some_cancel_rules_to_cancel(self):
        """Test when there are some gates to cancel."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.h(0)
        qc.h(0)
        inverse_pass = InverseCancellation([HGate(), CXGate(), CZGate()])
        new_circ = inverse_pass(qc)
        self.assertNotIn("h", new_circ.count_ops())

    def test_no_inverse_pairs(self):
        """Test when there are no inverse pairs to cancel."""
        qc = QuantumCircuit(1)
        qc.s(0)
        qc.sdg(0)
        inverse_pass = InverseCancellation([(TGate(), TdgGate())])
        new_circ = inverse_pass(qc)
        self.assertEqual(qc, new_circ)

    def test_some_inverse_pairs(self):
        """Test when there are some but not all inverse pairs to cancel."""
        qc = QuantumCircuit(1)
        qc.s(0)
        qc.sdg(0)
        qc.t(0)
        qc.tdg(0)
        inverse_pass = InverseCancellation([(TGate(), TdgGate())])
        new_circ = inverse_pass(qc)
        self.assertNotIn("t", new_circ.count_ops())
        self.assertNotIn("tdg", new_circ.count_ops())

    def test_some_inverse_and_cancelled(self):
        """Test when there are some but not all pairs to cancel."""
        qc = QuantumCircuit(2)
        qc.s(0)
        qc.sdg(0)
        qc.t(0)
        qc.tdg(0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.h(0)
        qc.h(0)
        inverse_pass = InverseCancellation([HGate(), CXGate(), CZGate(), (TGate(), TdgGate())])
        new_circ = inverse_pass(qc)
        self.assertNotIn("h", new_circ.count_ops())
        self.assertNotIn("t", new_circ.count_ops())
        self.assertNotIn("tdg", new_circ.count_ops())

    def test_half_of_an_inverse_pair(self):
        """Test that half of an inverse pair doesn't do anything."""
        qc = QuantumCircuit(1)
        qc.t(0)
        qc.t(0)
        qc.t(0)
        inverse_pass = InverseCancellation([(TGate(), TdgGate())])
        new_circ = inverse_pass(qc)
        self.assertEqual(new_circ, qc)

    def test_parameterized_self_inverse(self):
        """Test that a parameterized self inverse gate cancels correctly."""
        qc = QuantumCircuit(1)
        qc.rz(0, 0)
        qc.rz(0, 0)
        inverse_pass = InverseCancellation([RZGate(0)])
        new_circ = inverse_pass(qc)
        self.assertEqual(new_circ, QuantumCircuit(1))

    def test_parameterized_self_inverse_not_equal_parameter_1(self):
        """Test that a parameterized self inverse gate doesn't cancel incorrectly.
        This test, checks three gates with the same name but the middle one has a
        different parameter."""
        qc = QuantumCircuit(1)
        qc.rz(0, 0)
        qc.rz(3.14159, 0)
        qc.rz(0, 0)
        inverse_pass = InverseCancellation([RZGate(0)])
        new_circ = inverse_pass(qc)
        self.assertEqual(new_circ, qc)

    def test_parameterized_self_inverse_not_equal_parameter_2(self):
        """Test that a parameterized self inverse gate doesn't cancel incorrectly.
        This test, checks two gates with the same name but different parameters."""
        qc = QuantumCircuit(1)
        qc.rz(0, 0)
        qc.rz(3.14159, 0)
        inverse_pass = InverseCancellation([RZGate(0)])
        new_circ = inverse_pass(qc)
        self.assertEqual(qc, new_circ)

    def test_controlled_gate_open_control_does_not_cancel(self):
        """Test that a controlled gate with an open control doesn't cancel."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1, ctrl_state=0)
        inverse_pass = InverseCancellation([CXGate()])
        new_circ = inverse_pass(qc)
        self.assertEqual(new_circ, qc)

    def test_backwards_pair(self):
        """Test a backwards inverse pair works."""
        qc = QuantumCircuit(1)
        qc.tdg(0)
        qc.t(0)
        inverse_pass = InverseCancellation([(TGate(), TdgGate())])
        new_circ = inverse_pass(qc)
        self.assertEqual(new_circ, QuantumCircuit(1))

    def test_if_else(self):
        """Test that the pass recurses in a simple if-else."""
        pass_ = InverseCancellation([CXGate()])

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
        pass_ = InverseCancellation([CXGate()])
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


class TestCXCancellation(QiskitTestCase):
    """Test the former CXCancellation pass, which it was superseded by InverseCancellation.
    See: https://github.com/Qiskit/qiskit/pull/13426"""

    def test_pass_cx_cancellation(self):
        """Test the cx cancellation.
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
        pass_manager.append(InverseCancellation([CXGate()]))
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
        pass_manager.append(InverseCancellation([CXGate()]))
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
        pass_manager.append(InverseCancellation([CXGate()]))
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
        pass_manager.append(InverseCancellation([CXGate()]))
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
        pass_manager.append(InverseCancellation([CXGate()]))
        out_circuit = pass_manager.run(circuit)
        self.assertEqual(out_circuit, circuit)

    def test_if_else(self):
        """Test that the pass recurses in a simple if-else."""
        pass_ = InverseCancellation([CXGate()])
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
        pass_ = InverseCancellation([CXGate()])
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


if __name__ == "__main__":
    unittest.main()

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring

"""Test operations on the builder interfaces for control flow in dynamic QuantumCircuits."""

import copy
import math

import ddt

from qiskit.circuit import (
    ClassicalRegister,
    Clbit,
    Measure,
    Parameter,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
    Store,
)
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import ForLoopOp, IfElseOp, WhileLoopOp, SwitchCaseOp, CASE_DEFAULT
from qiskit.circuit.controlflow.if_else import IfElsePlaceholder
from qiskit.circuit.exceptions import CircuitError
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from test.utils._canonical import canonicalize_control_flow  # pylint: disable=wrong-import-order


class SentinelException(Exception):
    """An exception that we know was raised deliberately."""


@ddt.ddt
class TestControlFlowBuilders(QiskitTestCase):
    """Test that the control-flow builder interfaces work, and manage resources correctly."""

    def test_if_simple(self):
        """Test a simple if statement builds correctly, in the midst of other instructions."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        test = QuantumCircuit(qubits, clbits)
        test.h(0)
        test.measure(0, 0)
        with test.if_test((clbits[0], 0)):
            test.x(0)
        test.h(0)
        test.measure(0, 1)
        with test.if_test((clbits[1], 0)):
            test.h(1)
            test.cx(1, 0)

        if_true0 = QuantumCircuit([qubits[0], clbits[0]])
        if_true0.x(qubits[0])

        if_true1 = QuantumCircuit([qubits[0], qubits[1], clbits[1]])
        if_true1.h(qubits[1])
        if_true1.cx(qubits[1], qubits[0])

        expected = QuantumCircuit(qubits, clbits)
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[0])
        expected.if_test((clbits[0], 0), if_true0, [qubits[0]], [clbits[0]])
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[1])
        expected.if_test((clbits[1], 0), if_true1, [qubits[0], qubits[1]], [clbits[1]])

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_if_register(self):
        """Test a simple if statement builds correctly, when using a register as the condition.
        This requires the builder to unpack all the bits from the register to use as resources."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)

        test = QuantumCircuit(qr, cr)
        test.measure(qr, cr)
        with test.if_test((cr, 0)):
            test.x(0)

        if_true0 = QuantumCircuit([qr[0]], cr)
        if_true0.x(qr[0])

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr, cr)
        expected.if_test((cr, 0), if_true0, [qr[0]], cr)

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_if_expr(self):
        """Test a simple if statement builds correctly, when using an expression as the condition.
        This requires the builder to unpack all the bits from contained registers to use as
        resources."""
        qr = QuantumRegister(2)
        cr1 = ClassicalRegister(2)
        cr2 = ClassicalRegister(2)

        test = QuantumCircuit(qr, cr1, cr2)
        test.measure(qr, cr1)
        test.measure(qr, cr2)
        with test.if_test(expr.less(cr1, expr.bit_not(cr2))):
            test.x(0)

        if_true0 = QuantumCircuit([qr[0]], cr1, cr2)
        if_true0.x(qr[0])

        expected = QuantumCircuit(qr, cr1, cr2)
        expected.measure(qr, cr1)
        expected.measure(qr, cr2)
        expected.if_test(
            expr.less(cr1, expr.bit_not(cr2)), if_true0, [qr[0]], list(cr1) + list(cr2)
        )

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_register_condition_in_nested_block(self):
        """Test that nested blocks can use registers of the outermost circuits as conditions, and
        they get propagated through all the blocks."""

        qr = QuantumRegister(2)
        clbits = [Clbit(), Clbit(), Clbit()]
        cr1 = ClassicalRegister(3)
        # Try aliased classical registers as well, to catch potential overlap bugs.
        cr2 = ClassicalRegister(bits=clbits[:2])
        cr3 = ClassicalRegister(bits=clbits[1:])
        cr4 = ClassicalRegister(bits=clbits)

        with self.subTest("for/if"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.for_loop(range(3)):
                with test.if_test((cr1, 0)):
                    test.x(0)
                with test.if_test((cr2, 0)):
                    test.y(0)
                with test.if_test((cr3, 0)):
                    test.z(0)

            true_body1 = QuantumCircuit([qr[0]], cr1)
            true_body1.x(0)
            true_body2 = QuantumCircuit([qr[0]], cr2)
            true_body2.y(0)
            true_body3 = QuantumCircuit([qr[0]], cr3)
            true_body3.z(0)

            for_body = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3)  # but not cr4.
            for_body.if_test((cr1, 0), true_body1, [qr[0]], cr1)
            for_body.if_test((cr2, 0), true_body2, [qr[0]], cr2)
            for_body.if_test((cr3, 0), true_body3, [qr[0]], cr3)

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.for_loop(range(3), None, for_body, [qr[0]], clbits + list(cr1))

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("for/while"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.for_loop(range(3)):
                with test.while_loop((cr1, 0)):
                    test.x(0)
                with test.while_loop((cr2, 0)):
                    test.y(0)
                with test.while_loop((cr3, 0)):
                    test.z(0)

            while_body1 = QuantumCircuit([qr[0]], cr1)
            while_body1.x(0)
            while_body2 = QuantumCircuit([qr[0]], cr2)
            while_body2.y(0)
            while_body3 = QuantumCircuit([qr[0]], cr3)
            while_body3.z(0)

            for_body = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3)
            for_body.while_loop((cr1, 0), while_body1, [qr[0]], cr1)
            for_body.while_loop((cr2, 0), while_body2, [qr[0]], cr2)
            for_body.while_loop((cr3, 0), while_body3, [qr[0]], cr3)

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.for_loop(range(3), None, for_body, [qr[0]], clbits + list(cr1))

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("if/c_if"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.if_test((cr1, 0)):
                with self.assertWarns(DeprecationWarning):
                    test.x(0).c_if(cr2, 0)
                with self.assertWarns(DeprecationWarning):
                    test.z(0).c_if(cr3, 0)

            true_body = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3)
            with self.assertWarns(DeprecationWarning):
                true_body.x(0).c_if(cr2, 0)
            with self.assertWarns(DeprecationWarning):
                true_body.z(0).c_if(cr3, 0)

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.if_test((cr1, 0), true_body, [qr[0]], clbits + list(cr1))

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("while/else/c_if"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.while_loop((cr1, 0)):
                with test.if_test((cr2, 0)) as else_:
                    with self.assertWarns(DeprecationWarning):
                        test.x(0).c_if(cr3, 0)
                with else_:
                    with self.assertWarns(DeprecationWarning):
                        test.z(0).c_if(cr4, 0)

            true_body = QuantumCircuit([qr[0]], cr2, cr3, cr4)
            with self.assertWarns(DeprecationWarning):
                true_body.x(0).c_if(cr3, 0)
            false_body = QuantumCircuit([qr[0]], cr2, cr3, cr4)
            with self.assertWarns(DeprecationWarning):
                false_body.z(0).c_if(cr4, 0)

            while_body = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3, cr4)
            while_body.if_else((cr2, 0), true_body, false_body, [qr[0]], clbits)

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.while_loop((cr1, 0), while_body, [qr[0]], clbits + list(cr1))

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("switch/if"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.switch(cr1) as case_:
                with case_(0):
                    with test.if_test((cr2, 0)):
                        test.x(0)
                with case_(1, 2):
                    with test.if_test((cr3, 0)):
                        test.y(0)
                with case_(case_.DEFAULT):
                    with test.if_test((cr4, 0)):
                        test.z(0)

            true_body1 = QuantumCircuit([qr[0]], cr2)
            true_body1.x(0)
            case_body1 = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3, cr4)
            case_body1.if_test((cr2, 0), true_body1, [qr[0]], cr2)

            true_body2 = QuantumCircuit([qr[0]], cr3)
            true_body2.y(0)
            case_body2 = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3, cr4)
            case_body2.if_test((cr3, 0), true_body2, [qr[0]], cr3)

            true_body3 = QuantumCircuit([qr[0]], cr4)
            true_body3.z(0)
            case_body3 = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3, cr4)
            case_body3.if_test((cr4, 0), true_body3, [qr[0]], cr4)

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.switch(
                cr1,
                [
                    (0, case_body1),
                    ((1, 2), case_body2),
                    (CASE_DEFAULT, case_body3),
                ],
                [qr[0]],
                clbits + list(cr1),
            )

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("switch/while"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.switch(cr1) as case_:
                with case_(0):
                    with test.while_loop((cr2, 0)):
                        test.x(0)
                with case_(1, 2):
                    with test.while_loop((cr3, 0)):
                        test.y(0)
                with case_(case_.DEFAULT):
                    with test.while_loop((cr4, 0)):
                        test.z(0)

            loop_body1 = QuantumCircuit([qr[0]], cr2)
            loop_body1.x(0)
            case_body1 = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3, cr4)
            case_body1.while_loop((cr2, 0), loop_body1, [qr[0]], cr2)

            loop_body2 = QuantumCircuit([qr[0]], cr3)
            loop_body2.y(0)
            case_body2 = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3, cr4)
            case_body2.while_loop((cr3, 0), loop_body2, [qr[0]], cr3)

            loop_body3 = QuantumCircuit([qr[0]], cr4)
            loop_body3.z(0)
            case_body3 = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3, cr4)
            case_body3.while_loop((cr4, 0), loop_body3, [qr[0]], cr4)

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.switch(
                cr1,
                [
                    (0, case_body1),
                    ((1, 2), case_body2),
                    (CASE_DEFAULT, case_body3),
                ],
                [qr[0]],
                clbits + list(cr1),
            )

    def test_expr_condition_in_nested_block(self):
        """Test that nested blocks can use expressions with registers of the outermost circuits as
        conditions, and they get propagated through all the blocks."""

        qr = QuantumRegister(2)
        clbits = [Clbit(), Clbit(), Clbit()]
        cr1 = ClassicalRegister(3)
        # Try aliased classical registers as well, to catch potential overlap bugs.
        cr2 = ClassicalRegister(bits=clbits[:2])
        cr3 = ClassicalRegister(bits=clbits[1:])
        cr4 = ClassicalRegister(bits=clbits)

        with self.subTest("for/if"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.for_loop(range(3)):
                with test.if_test(expr.equal(cr1, 0)):
                    test.x(0)
                with test.if_test(expr.less(expr.bit_or(cr2, 1), 2)):
                    test.y(0)
                with test.if_test(expr.cast(expr.bit_not(cr3), types.Bool())):
                    test.z(0)

            true_body1 = QuantumCircuit([qr[0]], cr1)
            true_body1.x(0)
            true_body2 = QuantumCircuit([qr[0]], cr2)
            true_body2.y(0)
            true_body3 = QuantumCircuit([qr[0]], cr3)
            true_body3.z(0)

            for_body = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3)  # but not cr4.
            for_body.if_test(expr.equal(cr1, 0), true_body1, [qr[0]], cr1)
            for_body.if_test(expr.less(expr.bit_or(cr2, 1), 2), true_body2, [qr[0]], cr2)
            for_body.if_test(expr.cast(expr.bit_not(cr3), types.Bool()), true_body3, [qr[0]], cr3)

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.for_loop(range(3), None, for_body, [qr[0]], clbits + list(cr1))

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("for/while"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.for_loop(range(3)):
                with test.while_loop(expr.equal(cr1, 0)):
                    test.x(0)
                with test.while_loop(expr.less(expr.bit_or(cr2, 1), 2)):
                    test.y(0)
                with test.while_loop(expr.cast(expr.bit_not(cr3), types.Bool())):
                    test.z(0)

            while_body1 = QuantumCircuit([qr[0]], cr1)
            while_body1.x(0)
            while_body2 = QuantumCircuit([qr[0]], cr2)
            while_body2.y(0)
            while_body3 = QuantumCircuit([qr[0]], cr3)
            while_body3.z(0)

            for_body = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3)
            for_body.while_loop(expr.equal(cr1, 0), while_body1, [qr[0]], cr1)
            for_body.while_loop(expr.less(expr.bit_or(cr2, 1), 2), while_body2, [qr[0]], cr2)
            for_body.while_loop(
                expr.cast(expr.bit_not(cr3), types.Bool()), while_body3, [qr[0]], cr3
            )

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.for_loop(range(3), None, for_body, [qr[0]], clbits + list(cr1))

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("switch/if"):
            test = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            with test.switch(expr.lift(cr1)) as case_:
                with case_(0):
                    with test.if_test(expr.less(expr.bit_or(cr2, 1), 2)):
                        test.x(0)
                with case_(1, 2):
                    with test.if_test(expr.cast(expr.bit_not(cr3), types.Bool())):
                        test.y(0)
                with case_(case_.DEFAULT):
                    with test.if_test(expr.not_equal(cr4, 0)):
                        test.z(0)

            true_body1 = QuantumCircuit([qr[0]], cr2)
            true_body1.x(0)
            case_body1 = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3, cr4)
            case_body1.if_test(expr.less(expr.bit_or(cr2, 1), 2), true_body1, [qr[0]], cr2)

            true_body2 = QuantumCircuit([qr[0]], cr3)
            true_body2.y(0)
            case_body2 = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3, cr4)
            case_body2.if_test(expr.cast(expr.bit_not(cr3), types.Bool()), true_body2, [qr[0]], cr3)

            true_body3 = QuantumCircuit([qr[0]], cr4)
            true_body3.z(0)
            case_body3 = QuantumCircuit([qr[0]], clbits, cr1, cr2, cr3, cr4)
            case_body3.if_test(expr.not_equal(cr4, 0), true_body3, [qr[0]], cr4)

            expected = QuantumCircuit(qr, clbits, cr1, cr2, cr3, cr4)
            expected.switch(
                expr.lift(cr1),
                [
                    (0, case_body1),
                    ((1, 2), case_body2),
                    (CASE_DEFAULT, case_body3),
                ],
                [qr[0]],
                clbits + list(cr1),
            )

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_if_else_simple(self):
        """Test a simple if/else statement builds correctly, in the midst of other instructions.
        This test has paired if and else blocks the same natural width."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        test = QuantumCircuit(qubits, clbits)
        test.h(0)
        test.measure(0, 0)
        with test.if_test((clbits[0], 0)) as else_:
            test.x(0)
        with else_:
            test.z(0)
        test.h(0)
        test.measure(0, 1)
        with test.if_test((clbits[1], 0)) as else_:
            test.h(1)
            test.cx(1, 0)
        with else_:
            test.h(0)
            test.h(1)

        # Both the if and else blocks in this circuit are the same natural width to begin with.
        if_true0 = QuantumCircuit([qubits[0], clbits[0]])
        if_true0.x(qubits[0])
        if_false0 = QuantumCircuit([qubits[0], clbits[0]])
        if_false0.z(qubits[0])

        if_true1 = QuantumCircuit([qubits[0], qubits[1], clbits[1]])
        if_true1.h(qubits[1])
        if_true1.cx(qubits[1], qubits[0])
        if_false1 = QuantumCircuit([qubits[0], qubits[1], clbits[1]])
        if_false1.h(qubits[0])
        if_false1.h(qubits[1])

        expected = QuantumCircuit(qubits, clbits)
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[0])
        expected.if_else((clbits[0], 0), if_true0, if_false0, [qubits[0]], [clbits[0]])
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[1])
        expected.if_else((clbits[1], 0), if_true1, if_false1, [qubits[0], qubits[1]], [clbits[1]])

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_if_else_expr_simple(self):
        """Test a simple if/else statement builds correctly, in the midst of other instructions.
        This test has paired if and else blocks the same natural width."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        test = QuantumCircuit(qubits, clbits)
        test.h(0)
        test.measure(0, 0)
        with test.if_test(expr.lift(clbits[0])) as else_:
            test.x(0)
        with else_:
            test.z(0)
        test.h(0)
        test.measure(0, 1)
        with test.if_test(expr.logic_not(clbits[1])) as else_:
            test.h(1)
            test.cx(1, 0)
        with else_:
            test.h(0)
            test.h(1)

        # Both the if and else blocks in this circuit are the same natural width to begin with.
        if_true0 = QuantumCircuit([qubits[0], clbits[0]])
        if_true0.x(qubits[0])
        if_false0 = QuantumCircuit([qubits[0], clbits[0]])
        if_false0.z(qubits[0])

        if_true1 = QuantumCircuit([qubits[0], qubits[1], clbits[1]])
        if_true1.h(qubits[1])
        if_true1.cx(qubits[1], qubits[0])
        if_false1 = QuantumCircuit([qubits[0], qubits[1], clbits[1]])
        if_false1.h(qubits[0])
        if_false1.h(qubits[1])

        expected = QuantumCircuit(qubits, clbits)
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[0])
        expected.if_else(expr.lift(clbits[0]), if_true0, if_false0, [qubits[0]], [clbits[0]])
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[1])
        expected.if_else(
            expr.logic_not(clbits[1]), if_true1, if_false1, [qubits[0], qubits[1]], [clbits[1]]
        )

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_if_else_resources_expand_true_superset_false(self):
        """Test that the resources of the if and else bodies come out correctly if the true body
        needs a superset of the resources of the false body."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        test = QuantumCircuit(qubits, clbits)
        test.h(0)
        test.measure(0, 0)
        with test.if_test((clbits[0], 0)) as else_:
            test.x(0)
            test.measure(1, 1)
        with else_:
            test.z(0)

        if_true0 = QuantumCircuit(qubits, clbits)
        if_true0.x(qubits[0])
        if_true0.measure(qubits[1], clbits[1])
        # The false body doesn't actually use qubits[1] or clbits[1], but it still needs to contain
        # them so the bodies match.
        if_false0 = QuantumCircuit(qubits, clbits)
        if_false0.z(qubits[0])

        expected = QuantumCircuit(qubits, clbits)
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[0])
        expected.if_else((clbits[0], 0), if_true0, if_false0, qubits, clbits)

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_if_else_resources_expand_false_superset_true(self):
        """Test that the resources of the if and else bodies come out correctly if the false body
        needs a superset of the resources of the true body.  This requires that the manager
        correctly adds resources to the true body after it has been created."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        test = QuantumCircuit(qubits, clbits)
        test.h(0)
        test.measure(0, 0)
        with test.if_test((clbits[0], 0)) as else_:
            test.x(0)
        with else_:
            test.z(0)
            test.measure(1, 1)

        # The true body doesn't actually use qubits[1] or clbits[1], but it still needs to contain
        # them so the bodies match.
        if_true0 = QuantumCircuit(qubits, clbits)
        if_true0.x(qubits[0])
        if_false0 = QuantumCircuit(qubits, clbits)
        if_false0.z(qubits[0])
        if_false0.measure(qubits[1], clbits[1])

        expected = QuantumCircuit(qubits, clbits)
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[0])
        expected.if_else((clbits[0], 0), if_true0, if_false0, qubits, clbits)

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_if_else_resources_expand_true_false_symmetric_difference(self):
        """Test that the resources of the if and else bodies come out correctly if the sets of
        resources for the true body and the false body have some overlap, but neither is a subset of
        the other.  This tests that the flow of resources from block to block is simultaneously
        bidirectional."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        test = QuantumCircuit(qubits, clbits)
        test.h(0)
        test.measure(0, 0)
        with test.if_test((clbits[0], 0)) as else_:
            test.x(0)
        with else_:
            test.z(1)

        # The true body doesn't use qubits[1] and the false body doesn't use qubits[0].
        if_true0 = QuantumCircuit(qubits, [clbits[0]])
        if_true0.x(qubits[0])
        if_false0 = QuantumCircuit(qubits, [clbits[0]])
        if_false0.z(qubits[1])

        expected = QuantumCircuit(qubits, clbits)
        expected.h(qubits[0])
        expected.measure(qubits[0], clbits[0])
        expected.if_else((clbits[0], 0), if_true0, if_false0, qubits, [clbits[0]])

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_if_else_empty_branches(self):
        """Test that the context managers can cope with a body being empty."""
        qubits = [Qubit()]
        clbits = [Clbit()]

        cond = (clbits[0], 0)
        test = QuantumCircuit(qubits, clbits)
        # Sole empty if.
        with test.if_test(cond):
            pass
        # Normal if with an empty else body.
        with test.if_test(cond) as else_:
            test.x(0)
        with else_:
            pass
        # Empty if with a normal else body.
        with test.if_test(cond) as else_:
            pass
        with else_:
            test.x(0)
        # Both empty.
        with test.if_test(cond) as else_:
            pass
        with else_:
            pass

        empty_with_qubit = QuantumCircuit([qubits[0], clbits[0]])
        empty = QuantumCircuit([clbits[0]])
        only_x = QuantumCircuit([qubits[0], clbits[0]])
        only_x.x(qubits[0])

        expected = QuantumCircuit(qubits, clbits)
        expected.if_test(cond, empty, [], [clbits[0]])
        expected.if_else(cond, only_x, empty_with_qubit, [qubits[0]], [clbits[0]])
        expected.if_else(cond, empty_with_qubit, only_x, [qubits[0]], [clbits[0]])
        expected.if_else(cond, empty, empty, [], [clbits[0]])

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_if_else_tracks_registers(self):
        """Test that classical registers used in both branches of if statements are tracked
        correctly."""
        qr = QuantumRegister(2)
        cr = [ClassicalRegister(2) for _ in [None] * 4]

        test = QuantumCircuit(qr, *cr)
        with test.if_test((cr[0], 0)) as else_:
            with self.assertWarns(DeprecationWarning):
                test.h(0).c_if(cr[1], 0)
            # Test repetition.
            with self.assertWarns(DeprecationWarning):
                test.h(0).c_if(cr[1], 0)
        with else_:
            with self.assertWarns(DeprecationWarning):
                test.h(0).c_if(cr[2], 0)

        true_body = QuantumCircuit([qr[0]], cr[0], cr[1], cr[2])
        with self.assertWarns(DeprecationWarning):
            true_body.h(qr[0]).c_if(cr[1], 0)
        with self.assertWarns(DeprecationWarning):
            true_body.h(qr[0]).c_if(cr[1], 0)
        false_body = QuantumCircuit([qr[0]], cr[0], cr[1], cr[2])
        with self.assertWarns(DeprecationWarning):
            false_body.h(qr[0]).c_if(cr[2], 0)

        expected = QuantumCircuit(qr, *cr)
        expected.if_else(
            (cr[0], 0), true_body, false_body, [qr[0]], list(cr[0]) + list(cr[1]) + list(cr[2])
        )

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_if_else_nested(self):
        """Test that the if and else context managers can be nested, and don't interfere with each
        other."""
        qubits = [Qubit(), Qubit(), Qubit()]
        clbits = [Clbit(), Clbit(), Clbit()]

        outer_cond = (clbits[0], 0)
        inner_cond = (clbits[2], 1)

        with self.subTest("if (if) else"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(outer_cond) as else_:
                with test.if_test(inner_cond):
                    test.h(0)
            with else_:
                test.h(1)

            inner_true = QuantumCircuit([qubits[0], clbits[2]])
            inner_true.h(qubits[0])

            outer_true = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[2]])
            outer_true.if_test(inner_cond, inner_true, [qubits[0]], [clbits[2]])
            outer_false = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[2]])
            outer_false.h(qubits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.if_else(
                outer_cond, outer_true, outer_false, [qubits[0], qubits[1]], [clbits[0], clbits[2]]
            )
            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("if (if else) else"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(outer_cond) as outer_else:
                with test.if_test(inner_cond) as inner_else:
                    test.h(0)
                with inner_else:
                    test.h(2)
            with outer_else:
                test.h(1)

            inner_true = QuantumCircuit([qubits[0], qubits[2], clbits[2]])
            inner_true.h(qubits[0])
            inner_false = QuantumCircuit([qubits[0], qubits[2], clbits[2]])
            inner_false.h(qubits[2])

            outer_true = QuantumCircuit(qubits, [clbits[0], clbits[2]])
            outer_true.if_else(
                inner_cond, inner_true, inner_false, [qubits[0], qubits[2]], [clbits[2]]
            )
            outer_false = QuantumCircuit(qubits, [clbits[0], clbits[2]])
            outer_false.h(qubits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.if_else(outer_cond, outer_true, outer_false, qubits, [clbits[0], clbits[2]])
            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_if_else_expr_nested(self):
        """Test that the if and else context managers can be nested, and don't interfere with each
        other."""
        qubits = [Qubit(), Qubit(), Qubit()]
        clbits = [Clbit(), Clbit(), Clbit()]
        cr = ClassicalRegister(2, "c")

        outer_cond = expr.logic_not(clbits[0])
        inner_cond = expr.logic_and(clbits[2], expr.greater(cr, 1))

        with self.subTest("if (if) else"):
            test = QuantumCircuit(qubits, clbits, cr)
            with test.if_test(outer_cond) as else_:
                with test.if_test(inner_cond):
                    test.h(0)
            with else_:
                test.h(1)

            inner_true = QuantumCircuit([qubits[0], clbits[2]], cr)
            inner_true.h(qubits[0])

            outer_true = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[2]], cr)
            outer_true.if_test(inner_cond, inner_true, [qubits[0]], [clbits[2]] + list(cr))
            outer_false = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[2]], cr)
            outer_false.h(qubits[1])

            expected = QuantumCircuit(qubits, clbits, cr)
            expected.if_else(
                outer_cond,
                outer_true,
                outer_false,
                [qubits[0], qubits[1]],
                [clbits[0], clbits[2]] + list(cr),
            )
            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("if (if else) else"):
            test = QuantumCircuit(qubits, clbits, cr)
            with test.if_test(outer_cond) as outer_else:
                with test.if_test(inner_cond) as inner_else:
                    test.h(0)
                with inner_else:
                    test.h(2)
            with outer_else:
                test.h(1)

            inner_true = QuantumCircuit([qubits[0], qubits[2], clbits[2]], cr)
            inner_true.h(qubits[0])
            inner_false = QuantumCircuit([qubits[0], qubits[2], clbits[2]], cr)
            inner_false.h(qubits[2])

            outer_true = QuantumCircuit(qubits, [clbits[0], clbits[2]], cr)
            outer_true.if_else(
                inner_cond, inner_true, inner_false, [qubits[0], qubits[2]], [clbits[2]] + list(cr)
            )
            outer_false = QuantumCircuit(qubits, [clbits[0], clbits[2]], cr)
            outer_false.h(qubits[1])

            expected = QuantumCircuit(qubits, clbits, cr)
            expected.if_else(
                outer_cond, outer_true, outer_false, qubits, [clbits[0], clbits[2]] + list(cr)
            )
            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_switch_simple(self):
        """Individual labels switch test."""
        qubits = [Qubit(), Qubit(), Qubit()]
        creg = ClassicalRegister(2)
        test = QuantumCircuit(qubits, creg)
        with test.switch(creg) as case:
            with case(0):
                test.x(0)
            with case(1):
                test.x(2)
            with case(2):
                test.h(0)
            with case(3):
                test.h(2)

        body0 = QuantumCircuit([qubits[0], qubits[2]], creg)
        body0.x(qubits[0])
        body1 = QuantumCircuit([qubits[0], qubits[2]], creg)
        body1.x(qubits[2])
        body2 = QuantumCircuit([qubits[0], qubits[2]], creg)
        body2.h(qubits[0])
        body3 = QuantumCircuit([qubits[0], qubits[2]], creg)
        body3.h(qubits[2])
        expected = QuantumCircuit(qubits, creg)
        expected.switch(
            creg,
            [(0, body0), (1, body1), (2, body2), (3, body3)],
            [qubits[0], qubits[2]],
            list(creg),
        )

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_switch_expr_simple(self):
        """Individual labels switch test."""
        qubits = [Qubit(), Qubit(), Qubit()]
        creg = ClassicalRegister(2)
        test = QuantumCircuit(qubits, creg)
        with test.switch(expr.bit_and(creg, 2)) as case:
            with case(0):
                test.x(0)
            with case(1):
                test.x(2)
            with case(2):
                test.h(0)
            with case(3):
                test.h(2)

        body0 = QuantumCircuit([qubits[0], qubits[2]], creg)
        body0.x(qubits[0])
        body1 = QuantumCircuit([qubits[0], qubits[2]], creg)
        body1.x(qubits[2])
        body2 = QuantumCircuit([qubits[0], qubits[2]], creg)
        body2.h(qubits[0])
        body3 = QuantumCircuit([qubits[0], qubits[2]], creg)
        body3.h(qubits[2])
        expected = QuantumCircuit(qubits, creg)
        expected.switch(
            expr.bit_and(creg, 2),
            [(0, body0), (1, body1), (2, body2), (3, body3)],
            [qubits[0], qubits[2]],
            list(creg),
        )

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_switch_nested(self):
        """Individual labels switch test."""
        qubits = [Qubit(), Qubit(), Qubit()]
        cr1 = ClassicalRegister(2, "c1")
        cr2 = ClassicalRegister(2, "c2")
        cr3 = ClassicalRegister(3, "c3")
        loose = Clbit()
        test = QuantumCircuit(qubits, cr1, cr2, cr3, [loose])
        with test.switch(cr1) as case_outer:
            with case_outer(0), test.switch(loose) as case_inner, case_inner(False):
                test.x(0)
            with case_outer(1), test.switch(cr2) as case_inner, case_inner(0):
                test.x(1)

        body0_0 = QuantumCircuit([qubits[0]], [loose])
        body0_0.x(qubits[0])
        body0 = QuantumCircuit([qubits[0], qubits[1]], cr1, cr2, [loose])
        body0.switch(
            loose,
            [(False, body0_0)],
            [qubits[0]],
            [loose],
        )

        body1_0 = QuantumCircuit([qubits[1]], cr2)
        body1_0.x(qubits[1])
        body1 = QuantumCircuit([qubits[0], qubits[1]], cr1, cr2, [loose])
        body1.switch(
            cr2,
            [(0, body1_0)],
            [qubits[1]],
            list(cr2),
        )

        expected = QuantumCircuit(qubits, cr1, cr2, cr3, [loose])
        expected.switch(
            cr1,
            [(0, body0), (1, body1)],
            [qubits[0], qubits[1]],
            list(cr1) + list(cr2) + [loose],
        )

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_switch_expr_nested(self):
        """Individual labels switch test."""
        qubits = [Qubit(), Qubit(), Qubit()]
        cr1 = ClassicalRegister(2, "c1")
        cr2 = ClassicalRegister(2, "c2")
        cr3 = ClassicalRegister(3, "c3")
        loose = Clbit()
        test = QuantumCircuit(qubits, cr1, cr2, cr3, [loose])
        with test.switch(expr.bit_and(cr1, 2)) as case_outer:
            with case_outer(0), test.switch(expr.lift(loose)) as case_inner, case_inner(False):
                test.x(0)
            with case_outer(1), test.switch(expr.bit_and(cr2, 2)) as case_inner, case_inner(0):
                test.x(1)

        body0_0 = QuantumCircuit([qubits[0]], [loose])
        body0_0.x(qubits[0])
        body0 = QuantumCircuit([qubits[0], qubits[1]], cr1, cr2, [loose])
        body0.switch(
            expr.lift(loose),
            [(False, body0_0)],
            [qubits[0]],
            [loose],
        )

        body1_0 = QuantumCircuit([qubits[1]], cr2)
        body1_0.x(qubits[1])
        body1 = QuantumCircuit([qubits[0], qubits[1]], cr1, cr2, [loose])
        body1.switch(
            expr.bit_and(cr2, 2),
            [(0, body1_0)],
            [qubits[1]],
            list(cr2),
        )

        expected = QuantumCircuit(qubits, cr1, cr2, cr3, [loose])
        expected.switch(
            expr.bit_and(cr1, 2),
            [(0, body0), (1, body1)],
            [qubits[0], qubits[1]],
            list(cr1) + list(cr2) + [loose],
        )

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_switch_several_labels(self):
        """Several labels pointing to the same body."""
        qubits = [Qubit(), Qubit(), Qubit()]
        creg = ClassicalRegister(2)
        test = QuantumCircuit(qubits, creg)
        with test.switch(creg) as case:
            with case(0, 1):
                test.x(0)
            with case(2):
                test.h(0)

        body0 = QuantumCircuit([qubits[0]], creg)
        body0.x(qubits[0])
        body1 = QuantumCircuit([qubits[0]], creg)
        body1.h(qubits[0])
        expected = QuantumCircuit(qubits, creg)
        expected.switch(creg, [((0, 1), body0), (2, body1)], [qubits[0]], list(creg))

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_switch_default(self):
        """Allow a default case."""
        qubits = [Qubit(), Qubit(), Qubit()]
        creg = ClassicalRegister(2)
        test = QuantumCircuit(qubits, creg)
        with test.switch(creg) as case:
            with case(case.DEFAULT):
                test.x(0)
            # Additional test that the exposed `case.DEFAULT` object is referentially identical to
            # the general `CASE_DEFAULT` as well, to avoid subtle equality bugs.
            self.assertIs(case.DEFAULT, CASE_DEFAULT)

        body = QuantumCircuit([qubits[0]], creg)
        body.x(qubits[0])
        expected = QuantumCircuit(qubits, creg)
        expected.switch(creg, [(CASE_DEFAULT, body)], [qubits[0]], list(creg))

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_break_continue_expand_to_match_arguments_simple(self):
        """Test that ``break`` and ``continue`` statements expand to include all resources in the
        containing loop for simple cases with unconditional breaks."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.break_loop()
                test.h(0)
                test.continue_loop()
                test.measure(1, 0)

            body = QuantumCircuit(qubits, [clbits[0]])
            body.break_loop()
            body.h(qubits[0])
            body.continue_loop()
            body.measure(qubits[1], clbits[0])

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, body, qubits, [clbits[0]])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("while"):
            cond = (clbits[0], 0)
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond):
                test.break_loop()
                test.h(0)
                test.continue_loop()
                test.measure(1, 0)

            body = QuantumCircuit(qubits, [clbits[0]])
            body.break_loop()
            body.h(qubits[0])
            body.continue_loop()
            body.measure(qubits[1], clbits[0])

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond, body, qubits, [clbits[0]])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    @ddt.data(QuantumCircuit.break_loop, QuantumCircuit.continue_loop)
    def test_break_continue_accept_c_if(self, loop_operation):
        """Test that ``break`` and ``continue`` statements accept :meth:`.Instruction.c_if` calls,
        and that these propagate through correctly."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.h(0)
                with self.assertWarns(DeprecationWarning):
                    loop_operation(test).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], [clbits[1]])
            body.h(qubits[0])
            with self.assertWarns(DeprecationWarning):
                loop_operation(body).c_if(clbits[1], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, body, [qubits[0]], [clbits[1]])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("while"):
            cond = (clbits[0], 0)
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond):
                test.h(0)
                with self.assertWarns(DeprecationWarning):
                    loop_operation(test).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], clbits)
            body.h(qubits[0])
            with self.assertWarns(DeprecationWarning):
                loop_operation(body).c_if(clbits[1], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond, body, [qubits[0]], clbits)

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    @ddt.data(QuantumCircuit.break_loop, QuantumCircuit.continue_loop)
    def test_break_continue_only_expand_to_nearest_loop(self, loop_operation):
        """Test that a ``break`` or ``continue`` nested in more than one loop only expands as far as
        the inner loop scope, not further."""
        qubits = [Qubit(), Qubit(), Qubit()]
        clbits = [Clbit(), Clbit(), Clbit()]
        cond_inner = (clbits[0], 0)
        cond_outer = (clbits[1], 0)

        with self.subTest("for for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.measure(1, 1)
                with test.for_loop(range(2)):
                    test.h(0)
                    loop_operation(test)
                loop_operation(test)

            inner_body = QuantumCircuit([qubits[0]])
            inner_body.h(qubits[0])
            loop_operation(inner_body)

            outer_body = QuantumCircuit([qubits[0], qubits[1], clbits[1]])
            outer_body.measure(qubits[1], clbits[1])
            outer_body.for_loop(range(2), None, inner_body, [qubits[0]], [])
            loop_operation(outer_body)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, outer_body, [qubits[0], qubits[1]], [clbits[1]])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("for while"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.measure(1, 1)
                with test.while_loop(cond_inner):
                    test.h(0)
                    loop_operation(test)
                loop_operation(test)

            inner_body = QuantumCircuit([qubits[0], clbits[0]])
            inner_body.h(qubits[0])
            loop_operation(inner_body)

            outer_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            outer_body.measure(qubits[1], clbits[1])
            outer_body.while_loop(cond_inner, inner_body, [qubits[0]], [clbits[0]])
            loop_operation(outer_body)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(
                range(2), None, outer_body, [qubits[0], qubits[1]], [clbits[0], clbits[1]]
            )

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("while for"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_outer):
                test.measure(1, 1)
                with test.for_loop(range(2)):
                    test.h(0)
                    loop_operation(test)
                loop_operation(test)

            inner_body = QuantumCircuit([qubits[0]])
            inner_body.h(qubits[0])
            loop_operation(inner_body)

            outer_body = QuantumCircuit([qubits[0], qubits[1], clbits[1]])
            outer_body.measure(qubits[1], clbits[1])
            outer_body.for_loop(range(2), None, inner_body, [qubits[0]], [])
            loop_operation(outer_body)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond_outer, outer_body, [qubits[0], qubits[1]], [clbits[1]])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("while while"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_outer):
                test.measure(1, 1)
                with test.while_loop(cond_inner):
                    test.h(0)
                    loop_operation(test)
                loop_operation(test)

            inner_body = QuantumCircuit([qubits[0], clbits[0]])
            inner_body.h(qubits[0])
            loop_operation(inner_body)

            outer_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            outer_body.measure(qubits[1], clbits[1])
            outer_body.while_loop(cond_inner, inner_body, [qubits[0]], [clbits[0]])
            loop_operation(outer_body)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(
                cond_outer, outer_body, [qubits[0], qubits[1]], [clbits[0], clbits[1]]
            )

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("for (for, for)"):
            # This test is specifically to check that multiple inner loops with different numbers of
            # variables to each ``break`` still expand correctly.
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.measure(1, 1)
                with test.for_loop(range(2)):
                    test.h(0)
                    loop_operation(test)
                with test.for_loop(range(2)):
                    test.h(2)
                    loop_operation(test)
                loop_operation(test)

            inner_body1 = QuantumCircuit([qubits[0]])
            inner_body1.h(qubits[0])
            loop_operation(inner_body1)

            inner_body2 = QuantumCircuit([qubits[2]])
            inner_body2.h(qubits[2])
            loop_operation(inner_body2)

            outer_body = QuantumCircuit([qubits[0], qubits[1], qubits[2], clbits[1]])
            outer_body.measure(qubits[1], clbits[1])
            outer_body.for_loop(range(2), None, inner_body1, [qubits[0]], [])
            outer_body.for_loop(range(2), None, inner_body2, [qubits[2]], [])
            loop_operation(outer_body)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, outer_body, qubits, [clbits[1]])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    @ddt.data(QuantumCircuit.break_loop, QuantumCircuit.continue_loop)
    def test_break_continue_nested_in_if(self, loop_operation):
        """Test that ``break`` and ``continue`` work correctly when inside an ``if`` block within a
        loop.  This includes testing that multiple different ``if`` statements with and without
        ``break`` expand to the correct number of arguments.

        This is a very important case; it requires that the :obj:`.IfElseOp` is not built until the
        loop builds, and that the width expands to include everything that the loop knows about, not
        just the inner context.  We test both ``if`` and ``if/else`` paths separately, because the
        chaining of the context managers allows lots of possibilities for super weird edge cases.

        There are several tests that build up in complexity to aid debugging if something goes
        wrong; the aim is that there will be more information available depending on which of the
        subtests pass and which fail.
        """
        qubits = [Qubit(), Qubit(), Qubit()]
        clbits = [Clbit(), Clbit(), Clbit(), Clbit()]
        cond_inner = (clbits[0], 0)
        cond_outer = (clbits[1], 0)

        with self.subTest("for/if"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                with test.if_test(cond_inner):
                    loop_operation(test)
                # The second empty `if` is to test that only blocks that _need_ to expand to be the
                # full width of the loop do so.
                with test.if_test(cond_inner):
                    pass
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(2, 0)

            true_body1 = QuantumCircuit([qubits[0], clbits[0], clbits[2]])
            loop_operation(true_body1)

            true_body2 = QuantumCircuit([clbits[0]])

            loop_body = QuantumCircuit([qubits[0], clbits[0], clbits[2]])
            loop_body.if_test(cond_inner, true_body1, [qubits[0]], [clbits[0], clbits[2]])
            loop_body.if_test(cond_inner, true_body2, [], [clbits[0]])
            with self.assertWarns(DeprecationWarning):
                loop_body.h(qubits[0]).c_if(clbits[2], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, loop_body, [qubits[0]], [clbits[0], clbits[2]])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("for/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                with test.if_test(cond_inner) as else_:
                    test.h(1)
                with else_:
                    loop_operation(test)
                with test.if_test(cond_inner) as else_:
                    pass
                with else_:
                    pass
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(2, 0)

            true_body1 = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[2]])
            true_body1.h(qubits[1])
            false_body1 = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[2]])
            loop_operation(false_body1)

            true_body2 = QuantumCircuit([clbits[0]])
            false_body2 = QuantumCircuit([clbits[0]])

            loop_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[2]])
            loop_body.if_else(
                cond_inner, true_body1, false_body1, [qubits[0], qubits[1]], [clbits[0], clbits[2]]
            )
            loop_body.if_else(cond_inner, true_body2, false_body2, [], [clbits[0]])
            with self.assertWarns(DeprecationWarning):
                loop_body.h(qubits[0]).c_if(clbits[2], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(
                range(2), None, loop_body, [qubits[0], qubits[1]], [clbits[0], clbits[2]]
            )

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("while/if"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_outer):
                with test.if_test(cond_inner):
                    loop_operation(test)
                with test.if_test(cond_inner):
                    pass
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(2, 0)

            true_body1 = QuantumCircuit([qubits[0], clbits[0], clbits[1], clbits[2]])
            loop_operation(true_body1)

            true_body2 = QuantumCircuit([clbits[0]])

            loop_body = QuantumCircuit([qubits[0], clbits[0], clbits[1], clbits[2]])
            loop_body.if_test(
                cond_inner, true_body1, [qubits[0]], [clbits[0], clbits[1], clbits[2]]
            )
            loop_body.if_test(cond_inner, true_body2, [], [clbits[0]])
            with self.assertWarns(DeprecationWarning):
                loop_body.h(qubits[0]).c_if(clbits[2], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(
                cond_outer, loop_body, [qubits[0]], [clbits[0], clbits[1], clbits[2]]
            )

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("while/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_outer):
                with test.if_test(cond_inner) as else_:
                    test.h(1)
                with else_:
                    loop_operation(test)
                with test.if_test(cond_inner) as else_:
                    pass
                with else_:
                    pass
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(2, 0)

            true_body1 = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1], clbits[2]])
            true_body1.h(qubits[1])
            false_body1 = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1], clbits[2]])
            loop_operation(false_body1)

            true_body2 = QuantumCircuit([clbits[0]])
            false_body2 = QuantumCircuit([clbits[0]])

            loop_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1], clbits[2]])
            loop_body.if_else(
                cond_inner,
                true_body1,
                false_body1,
                [qubits[0], qubits[1]],
                [clbits[0], clbits[1], clbits[2]],
            )
            loop_body.if_else(cond_inner, true_body2, false_body2, [], [clbits[0]])
            with self.assertWarns(DeprecationWarning):
                loop_body.h(qubits[0]).c_if(clbits[2], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(
                cond_outer, loop_body, [qubits[0], qubits[1]], [clbits[0], clbits[1], clbits[2]]
            )

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    @ddt.data(QuantumCircuit.break_loop, QuantumCircuit.continue_loop)
    def test_break_continue_nested_in_switch(self, loop_operation):
        """Similar to the nested-in-if case, we have to ensure that `break` and `continue` inside a
        `switch` expand in size to the containing loop."""
        qubits = [Qubit(), Qubit(), Qubit()]
        clbits = [Clbit(), Clbit(), Clbit(), Clbit()]

        test = QuantumCircuit(qubits, clbits)
        with test.for_loop(range(2)):
            with test.switch(clbits[0]) as case:
                with case(0):
                    loop_operation(test)
                with case(1):
                    pass
            # The second empty `switch` is to test that only blocks that _need_ to expand to be the
            # full width of the loop do so.
            with test.switch(clbits[0]) as case:
                with case(case.DEFAULT):
                    pass
            with self.assertWarns(DeprecationWarning):
                test.h(0).c_if(clbits[2], 0)

        body0 = QuantumCircuit([qubits[0], clbits[0], clbits[2]])
        loop_operation(body0)
        body1 = QuantumCircuit([qubits[0], clbits[0], clbits[2]])

        body2 = QuantumCircuit([clbits[0]])

        loop_body = QuantumCircuit([qubits[0], clbits[0], clbits[2]])
        loop_body.switch(clbits[0], [(0, body0), (1, body1)], [qubits[0]], [clbits[0], clbits[2]])
        loop_body.switch(clbits[0], [(CASE_DEFAULT, body2)], [], [clbits[0]])
        with self.assertWarns(DeprecationWarning):
            loop_body.h(qubits[0]).c_if(clbits[2], 0)

        expected = QuantumCircuit(qubits, clbits)
        expected.for_loop(range(2), None, loop_body, [qubits[0]], [clbits[0], clbits[2]])

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    @ddt.data(QuantumCircuit.break_loop, QuantumCircuit.continue_loop)
    def test_break_continue_nested_in_multiple_switch(self, loop_operation):
        """Similar to the nested-in-if case, we have to ensure that `break` and `continue` inside
        more than one `switch` in a loop expand in size to the containing loop."""
        qubits = [Qubit(), Qubit(), Qubit()]
        clbits = [Clbit(), Clbit(), Clbit()]
        test = QuantumCircuit(qubits, clbits)
        with test.for_loop(range(2)):
            test.measure(1, 1)
            with test.switch(1) as case:
                with case(False):
                    test.h(0)
                    loop_operation(test)
                with case(True):
                    pass
            with test.switch(1) as case:
                with case(False):
                    pass
                with case(True):
                    test.h(2)
                    loop_operation(test)
            loop_operation(test)

        case1_f = QuantumCircuit([qubits[0], qubits[1], qubits[2], clbits[1]])
        case1_f.h(qubits[0])
        loop_operation(case1_f)
        case1_t = QuantumCircuit([qubits[0], qubits[1], qubits[2], clbits[1]])

        case2_f = QuantumCircuit([qubits[0], qubits[1], qubits[2], clbits[1]])
        case2_t = QuantumCircuit([qubits[0], qubits[1], qubits[2], clbits[1]])
        case2_t.h(qubits[2])
        loop_operation(case2_t)

        body = QuantumCircuit([qubits[0], qubits[1], qubits[2], clbits[1]])
        body.measure(qubits[1], clbits[1])
        body.switch(clbits[1], [(False, case1_f), (True, case1_t)], body.qubits, body.clbits)
        body.switch(clbits[1], [(False, case2_f), (True, case2_t)], body.qubits, body.clbits)
        loop_operation(body)

        expected = QuantumCircuit(qubits, clbits)
        expected.for_loop(range(2), None, body, qubits, [clbits[1]])

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    @ddt.data(QuantumCircuit.break_loop, QuantumCircuit.continue_loop)
    def test_break_continue_deeply_nested(self, loop_operation):
        """Test that ``break`` and ``continue`` work correctly when inside more than one block
        within a loop.  This includes testing that multiple different statements with and without
        ``break`` expand to the correct number of arguments.

        These are the deepest tests, hitting all parts of the deferred builder scopes.  We test
        ``if``, ``if/else`` and ``switch`` paths at various levels of the scoping to try and account
        for as many weird edge cases with the deferred behavior as possible.  We try to make sure,
        particularly in the most complicated examples, that there are resources added before and
        after every single scope, to try and catch all possibilities of where resources may be
        missed.

        There are several tests that build up in complexity to aid debugging if something goes
        wrong; the aim is that there will be more information available depending on which of the
        subtests pass and which fail.
        """
        # These are deliberately more than is absolutely needed so we can detect if extra resources
        # are being erroneously included as well.
        qubits = [Qubit() for _ in [None] * 20]
        clbits = [Clbit() for _ in [None] * 20]
        cond_inner = (clbits[0], 0)
        cond_outer = (clbits[1], 0)
        cond_loop = (clbits[2], 0)

        with self.subTest("for/if/if"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                # outer true 1
                with test.if_test(cond_outer):
                    # inner true 1
                    with test.if_test(cond_inner):
                        loop_operation(test)
                    # inner true 2
                    with test.if_test(cond_inner):
                        with self.assertWarns(DeprecationWarning):
                            test.h(0).c_if(3, 0)
                    with self.assertWarns(DeprecationWarning):
                        test.h(1).c_if(4, 0)
                # outer true 2
                with test.if_test(cond_outer):
                    with self.assertWarns(DeprecationWarning):
                        test.h(2).c_if(5, 0)
                with self.assertWarns(DeprecationWarning):
                    test.h(3).c_if(6, 0)

            inner_true_body1 = QuantumCircuit(qubits[:4], clbits[:2], clbits[3:7])
            loop_operation(inner_true_body1)

            inner_true_body2 = QuantumCircuit([qubits[0], clbits[0], clbits[3]])
            with self.assertWarns(DeprecationWarning):
                inner_true_body2.h(qubits[0]).c_if(clbits[3], 0)

            outer_true_body1 = QuantumCircuit(qubits[:4], clbits[:2], clbits[3:7])
            outer_true_body1.if_test(
                cond_inner, inner_true_body1, qubits[:4], clbits[:2] + clbits[3:7]
            )
            outer_true_body1.if_test(
                cond_inner, inner_true_body2, [qubits[0]], [clbits[0], clbits[3]]
            )
            with self.assertWarns(DeprecationWarning):
                outer_true_body1.h(qubits[1]).c_if(clbits[4], 0)

            outer_true_body2 = QuantumCircuit([qubits[2], clbits[1], clbits[5]])
            with self.assertWarns(DeprecationWarning):
                outer_true_body2.h(qubits[2]).c_if(clbits[5], 0)

            loop_body = QuantumCircuit(qubits[:4], clbits[:2] + clbits[3:7])
            loop_body.if_test(cond_outer, outer_true_body1, qubits[:4], clbits[:2] + clbits[3:7])
            loop_body.if_test(cond_outer, outer_true_body2, [qubits[2]], [clbits[1], clbits[5]])
            with self.assertWarns(DeprecationWarning):
                loop_body.h(qubits[3]).c_if(clbits[6], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, loop_body, qubits[:4], clbits[:2] + clbits[3:7])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("for/if/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                # outer 1
                with test.if_test(cond_outer):
                    # inner 1
                    with test.if_test(cond_inner) as inner1_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(0).c_if(3, 0)
                    with inner1_else:
                        with self.assertWarns(DeprecationWarning):
                            loop_operation(test).c_if(4, 0)
                    # inner 2
                    with test.if_test(cond_inner) as inner2_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(1).c_if(5, 0)
                    with inner2_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(2).c_if(6, 0)
                    with self.assertWarns(DeprecationWarning):
                        test.h(3).c_if(7, 0)
                # outer 2
                with test.if_test(cond_outer) as outer2_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(4).c_if(8, 0)
                with outer2_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(5).c_if(9, 0)
                with self.assertWarns(DeprecationWarning):
                    test.h(6).c_if(10, 0)

            inner1_true = QuantumCircuit(qubits[:7], clbits[:2], clbits[3:11])
            with self.assertWarns(DeprecationWarning):
                inner1_true.h(qubits[0]).c_if(clbits[3], 0)
            inner1_false = QuantumCircuit(qubits[:7], clbits[:2], clbits[3:11])
            with self.assertWarns(DeprecationWarning):
                loop_operation(inner1_false).c_if(clbits[4], 0)

            inner2_true = QuantumCircuit([qubits[1], qubits[2], clbits[0], clbits[5], clbits[6]])
            with self.assertWarns(DeprecationWarning):
                inner2_true.h(qubits[1]).c_if(clbits[5], 0)
            inner2_false = QuantumCircuit([qubits[1], qubits[2], clbits[0], clbits[5], clbits[6]])
            with self.assertWarns(DeprecationWarning):
                inner2_false.h(qubits[2]).c_if(clbits[6], 0)

            outer1_true = QuantumCircuit(qubits[:7], clbits[:2], clbits[3:11])
            outer1_true.if_else(
                cond_inner, inner1_true, inner1_false, qubits[:7], clbits[:2] + clbits[3:11]
            )
            outer1_true.if_else(
                cond_inner,
                inner2_true,
                inner2_false,
                qubits[1:3],
                [clbits[0], clbits[5], clbits[6]],
            )
            with self.assertWarns(DeprecationWarning):
                outer1_true.h(qubits[3]).c_if(clbits[7], 0)

            outer2_true = QuantumCircuit([qubits[4], qubits[5], clbits[1], clbits[8], clbits[9]])
            with self.assertWarns(DeprecationWarning):
                outer2_true.h(qubits[4]).c_if(clbits[8], 0)
            outer2_false = QuantumCircuit([qubits[4], qubits[5], clbits[1], clbits[8], clbits[9]])
            with self.assertWarns(DeprecationWarning):
                outer2_false.h(qubits[5]).c_if(clbits[9], 0)

            loop_body = QuantumCircuit(qubits[:7], clbits[:2], clbits[3:11])
            loop_body.if_test(cond_outer, outer1_true, qubits[:7], clbits[:2] + clbits[3:11])
            loop_body.if_else(
                cond_outer,
                outer2_true,
                outer2_false,
                qubits[4:6],
                [clbits[1], clbits[8], clbits[9]],
            )
            with self.assertWarns(DeprecationWarning):
                loop_body.h(qubits[6]).c_if(clbits[10], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, loop_body, qubits[:7], clbits[:2] + clbits[3:11])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("for/else/else/switch"):
            # Look on my works, ye Mighty, and despair!

            # (but also hopefully this is less hubristic pretension and more a useful stress test)

            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(3, 0)

                # outer 1
                with test.if_test(cond_outer) as outer1_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(1).c_if(4, 0)
                with outer1_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(2).c_if(5, 0)

                # outer 2 (nesting the inner condition in the 'if')
                with test.if_test(cond_outer) as outer2_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(3).c_if(6, 0)

                    # inner 21
                    with test.if_test(cond_inner) as inner21_else:
                        loop_operation(test)
                    with inner21_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(4).c_if(7, 0)

                    # inner 22
                    with test.if_test(cond_inner) as inner22_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(5).c_if(8, 0)
                    with inner22_else:
                        loop_operation(test)

                    # inner 23
                    with test.switch(cond_inner[0]) as inner23_case:
                        with inner23_case(True):
                            with self.assertWarns(DeprecationWarning):
                                test.h(5).c_if(8, 0)
                        with inner23_case(False):
                            loop_operation(test)

                    with self.assertWarns(DeprecationWarning):
                        test.h(6).c_if(9, 0)
                with outer2_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(7).c_if(10, 0)

                    # inner 24
                    with test.if_test(cond_inner) as inner24_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(8).c_if(11, 0)
                    with inner24_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(9).c_if(12, 0)

                # outer 3 (nesting the inner condition in an 'else' branch)
                with test.if_test(cond_outer) as outer3_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(10).c_if(13, 0)
                with outer3_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(11).c_if(14, 0)

                    # inner 31
                    with test.if_test(cond_inner) as inner31_else:
                        loop_operation(test)
                    with inner31_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(12).c_if(15, 0)

                    # inner 32
                    with test.if_test(cond_inner) as inner32_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(13).c_if(16, 0)
                    with inner32_else:
                        loop_operation(test)

                    # inner 33
                    with test.if_test(cond_inner) as inner33_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(14).c_if(17, 0)
                    with inner33_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(15).c_if(18, 0)

                with self.assertWarns(DeprecationWarning):
                    test.h(16).c_if(19, 0)
            # End of test "for" loop.

            # No `clbits[2]` here because that's only used in `cond_loop`, for while loops.
            loop_qubits = qubits[:17]
            loop_clbits = clbits[:2] + clbits[3:20]
            loop_bits = loop_qubits + loop_clbits

            outer1_true = QuantumCircuit([qubits[1], qubits[2], clbits[1], clbits[4], clbits[5]])
            with self.assertWarns(DeprecationWarning):
                outer1_true.h(qubits[1]).c_if(clbits[4], 0)
            outer1_false = QuantumCircuit([qubits[1], qubits[2], clbits[1], clbits[4], clbits[5]])
            with self.assertWarns(DeprecationWarning):
                outer1_false.h(qubits[2]).c_if(clbits[5], 0)

            inner21_true = QuantumCircuit(loop_bits)
            loop_operation(inner21_true)
            inner21_false = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                inner21_false.h(qubits[4]).c_if(clbits[7], 0)

            inner22_true = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                inner22_true.h(qubits[5]).c_if(clbits[8], 0)
            inner22_false = QuantumCircuit(loop_bits)
            loop_operation(inner22_false)

            inner23_true = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                inner23_true.h(qubits[5]).c_if(clbits[8], 0)
            inner23_false = QuantumCircuit(loop_bits)
            loop_operation(inner23_false)

            inner24_true = QuantumCircuit(qubits[8:10], [clbits[0], clbits[11], clbits[12]])
            with self.assertWarns(DeprecationWarning):
                inner24_true.h(qubits[8]).c_if(clbits[11], 0)
            inner24_false = QuantumCircuit(qubits[8:10], [clbits[0], clbits[11], clbits[12]])
            with self.assertWarns(DeprecationWarning):
                inner24_false.h(qubits[9]).c_if(clbits[12], 0)

            outer2_true = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                outer2_true.h(qubits[3]).c_if(clbits[6], 0)
            outer2_true.if_else(cond_inner, inner21_true, inner21_false, loop_qubits, loop_clbits)
            outer2_true.if_else(cond_inner, inner22_true, inner22_false, loop_qubits, loop_clbits)
            outer2_true.switch(
                cond_inner[0],
                [(True, inner23_true), (False, inner23_false)],
                loop_qubits,
                loop_clbits,
            )
            with self.assertWarns(DeprecationWarning):
                outer2_true.h(qubits[6]).c_if(clbits[9], 0)
            outer2_false = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                outer2_false.h(qubits[7]).c_if(clbits[10], 0)
            outer2_false.if_else(
                cond_inner,
                inner24_true,
                inner24_false,
                [qubits[8], qubits[9]],
                [clbits[0], clbits[11], clbits[12]],
            )

            inner31_true = QuantumCircuit(loop_bits)
            loop_operation(inner31_true)
            inner31_false = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                inner31_false.h(qubits[12]).c_if(clbits[15], 0)

            inner32_true = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                inner32_true.h(qubits[13]).c_if(clbits[16], 0)
            inner32_false = QuantumCircuit(loop_bits)
            loop_operation(inner32_false)

            inner33_true = QuantumCircuit(qubits[14:16], [clbits[0], clbits[17], clbits[18]])
            with self.assertWarns(DeprecationWarning):
                inner33_true.h(qubits[14]).c_if(clbits[17], 0)
            inner33_false = QuantumCircuit(qubits[14:16], [clbits[0], clbits[17], clbits[18]])
            with self.assertWarns(DeprecationWarning):
                inner33_false.h(qubits[15]).c_if(clbits[18], 0)

            outer3_true = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                outer3_true.h(qubits[10]).c_if(clbits[13], 0)
            outer3_false = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                outer3_false.h(qubits[11]).c_if(clbits[14], 0)
            outer3_false.if_else(cond_inner, inner31_true, inner31_false, loop_qubits, loop_clbits)
            outer3_false.if_else(cond_inner, inner32_true, inner32_false, loop_qubits, loop_clbits)
            outer3_false.if_else(
                cond_inner,
                inner33_true,
                inner33_false,
                qubits[14:16],
                [clbits[0], clbits[17], clbits[18]],
            )

            loop_body = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                loop_body.h(qubits[0]).c_if(clbits[3], 0)
            loop_body.if_else(
                cond_outer,
                outer1_true,
                outer1_false,
                qubits[1:3],
                [clbits[1], clbits[4], clbits[5]],
            )
            loop_body.if_else(cond_outer, outer2_true, outer2_false, loop_qubits, loop_clbits)
            loop_body.if_else(cond_outer, outer3_true, outer3_false, loop_qubits, loop_clbits)
            with self.assertWarns(DeprecationWarning):
                loop_body.h(qubits[16]).c_if(clbits[19], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, loop_body, loop_qubits, loop_clbits)

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        # And now we repeat everything for "while" loops...  Trying to parameterize the test over
        # 'for/while' mostly just ends up in it being a bit illegible, because so many parameters
        # vary in the explicit construction form.  These tests are just copies of the above tests,
        # but with `while_loop(cond_loop)` instead of `for_loop(range(2))`, and the corresponding
        # clbit ranges updated to include `clbits[2]` from the condition.

        with self.subTest("while/if/if"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_loop):
                # outer true 1
                with test.if_test(cond_outer):
                    # inner true 1
                    with test.if_test(cond_inner):
                        loop_operation(test)
                    # inner true 2
                    with test.if_test(cond_inner):
                        with self.assertWarns(DeprecationWarning):
                            test.h(0).c_if(3, 0)
                    with self.assertWarns(DeprecationWarning):
                        test.h(1).c_if(4, 0)
                # outer true 2
                with test.if_test(cond_outer):
                    with self.assertWarns(DeprecationWarning):
                        test.h(2).c_if(5, 0)
                with self.assertWarns(DeprecationWarning):
                    test.h(3).c_if(6, 0)

            inner_true_body1 = QuantumCircuit(qubits[:4], clbits[:7])
            loop_operation(inner_true_body1)

            inner_true_body2 = QuantumCircuit([qubits[0], clbits[0], clbits[3]])
            with self.assertWarns(DeprecationWarning):
                inner_true_body2.h(qubits[0]).c_if(clbits[3], 0)

            outer_true_body1 = QuantumCircuit(qubits[:4], clbits[:7])
            outer_true_body1.if_test(cond_inner, inner_true_body1, qubits[:4], clbits[:7])
            outer_true_body1.if_test(
                cond_inner, inner_true_body2, [qubits[0]], [clbits[0], clbits[3]]
            )
            with self.assertWarns(DeprecationWarning):
                outer_true_body1.h(qubits[1]).c_if(clbits[4], 0)

            outer_true_body2 = QuantumCircuit([qubits[2], clbits[1], clbits[5]])
            with self.assertWarns(DeprecationWarning):
                outer_true_body2.h(qubits[2]).c_if(clbits[5], 0)

            loop_body = QuantumCircuit(qubits[:4], clbits[:7])
            loop_body.if_test(cond_outer, outer_true_body1, qubits[:4], clbits[:7])
            loop_body.if_test(cond_outer, outer_true_body2, [qubits[2]], [clbits[1], clbits[5]])
            with self.assertWarns(DeprecationWarning):
                loop_body.h(qubits[3]).c_if(clbits[6], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond_loop, loop_body, qubits[:4], clbits[:7])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("while/if/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_loop):
                # outer 1
                with test.if_test(cond_outer):
                    # inner 1
                    with test.if_test(cond_inner) as inner1_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(0).c_if(3, 0)
                    with inner1_else:
                        with self.assertWarns(DeprecationWarning):
                            loop_operation(test).c_if(4, 0)
                    # inner 2
                    with test.if_test(cond_inner) as inner2_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(1).c_if(5, 0)
                    with inner2_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(2).c_if(6, 0)
                    with self.assertWarns(DeprecationWarning):
                        test.h(3).c_if(7, 0)
                # outer 2
                with test.if_test(cond_outer) as outer2_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(4).c_if(8, 0)
                with outer2_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(5).c_if(9, 0)
                with self.assertWarns(DeprecationWarning):
                    test.h(6).c_if(10, 0)

            inner1_true = QuantumCircuit(qubits[:7], clbits[:11])
            with self.assertWarns(DeprecationWarning):
                inner1_true.h(qubits[0]).c_if(clbits[3], 0)
            inner1_false = QuantumCircuit(qubits[:7], clbits[:11])
            with self.assertWarns(DeprecationWarning):
                loop_operation(inner1_false).c_if(clbits[4], 0)

            inner2_true = QuantumCircuit([qubits[1], qubits[2], clbits[0], clbits[5], clbits[6]])
            with self.assertWarns(DeprecationWarning):
                inner2_true.h(qubits[1]).c_if(clbits[5], 0)
            inner2_false = QuantumCircuit([qubits[1], qubits[2], clbits[0], clbits[5], clbits[6]])
            with self.assertWarns(DeprecationWarning):
                inner2_false.h(qubits[2]).c_if(clbits[6], 0)

            outer1_true = QuantumCircuit(qubits[:7], clbits[:11])
            outer1_true.if_else(cond_inner, inner1_true, inner1_false, qubits[:7], clbits[:11])
            outer1_true.if_else(
                cond_inner,
                inner2_true,
                inner2_false,
                qubits[1:3],
                [clbits[0], clbits[5], clbits[6]],
            )
            with self.assertWarns(DeprecationWarning):
                outer1_true.h(qubits[3]).c_if(clbits[7], 0)

            outer2_true = QuantumCircuit([qubits[4], qubits[5], clbits[1], clbits[8], clbits[9]])
            with self.assertWarns(DeprecationWarning):
                outer2_true.h(qubits[4]).c_if(clbits[8], 0)
            outer2_false = QuantumCircuit([qubits[4], qubits[5], clbits[1], clbits[8], clbits[9]])
            with self.assertWarns(DeprecationWarning):
                outer2_false.h(qubits[5]).c_if(clbits[9], 0)

            loop_body = QuantumCircuit(qubits[:7], clbits[:11])
            loop_body.if_test(cond_outer, outer1_true, qubits[:7], clbits[:11])
            loop_body.if_else(
                cond_outer,
                outer2_true,
                outer2_false,
                qubits[4:6],
                [clbits[1], clbits[8], clbits[9]],
            )
            with self.assertWarns(DeprecationWarning):
                loop_body.h(qubits[6]).c_if(clbits[10], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond_loop, loop_body, qubits[:7], clbits[:11])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("while/else/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond_loop):
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(3, 0)

                # outer 1
                with test.if_test(cond_outer) as outer1_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(1).c_if(4, 0)
                with outer1_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(2).c_if(5, 0)

                # outer 2 (nesting the inner condition in the 'if')
                with test.if_test(cond_outer) as outer2_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(3).c_if(6, 0)

                    # inner 21
                    with test.if_test(cond_inner) as inner21_else:
                        loop_operation(test)
                    with inner21_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(4).c_if(7, 0)

                    # inner 22
                    with test.if_test(cond_inner) as inner22_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(5).c_if(8, 0)
                    with inner22_else:
                        loop_operation(test)

                    with self.assertWarns(DeprecationWarning):
                        test.h(6).c_if(9, 0)
                with outer2_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(7).c_if(10, 0)

                    # inner 23
                    with test.if_test(cond_inner) as inner23_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(8).c_if(11, 0)
                    with inner23_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(9).c_if(12, 0)

                # outer 3 (nesting the inner condition in an 'else' branch)
                with test.if_test(cond_outer) as outer3_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(10).c_if(13, 0)
                with outer3_else:
                    with self.assertWarns(DeprecationWarning):
                        test.h(11).c_if(14, 0)

                    # inner 31
                    with test.if_test(cond_inner) as inner31_else:
                        loop_operation(test)
                    with inner31_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(12).c_if(15, 0)

                    # inner 32
                    with test.if_test(cond_inner) as inner32_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(13).c_if(16, 0)
                    with inner32_else:
                        loop_operation(test)

                    # inner 33
                    with test.if_test(cond_inner) as inner33_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(14).c_if(17, 0)
                    with inner33_else:
                        with self.assertWarns(DeprecationWarning):
                            test.h(15).c_if(18, 0)
                with self.assertWarns(DeprecationWarning):
                    test.h(16).c_if(19, 0)
            # End of test "for" loop.

            # No `clbits[2]` here because that's only used in `cond_loop`, for while loops.
            loop_qubits = qubits[:17]
            loop_clbits = clbits[:20]
            loop_bits = loop_qubits + loop_clbits

            outer1_true = QuantumCircuit([qubits[1], qubits[2], clbits[1], clbits[4], clbits[5]])
            with self.assertWarns(DeprecationWarning):
                outer1_true.h(qubits[1]).c_if(clbits[4], 0)
            outer1_false = QuantumCircuit([qubits[1], qubits[2], clbits[1], clbits[4], clbits[5]])
            with self.assertWarns(DeprecationWarning):
                outer1_false.h(qubits[2]).c_if(clbits[5], 0)

            inner21_true = QuantumCircuit(loop_bits)
            loop_operation(inner21_true)
            inner21_false = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                inner21_false.h(qubits[4]).c_if(clbits[7], 0)

            inner22_true = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                inner22_true.h(qubits[5]).c_if(clbits[8], 0)
            inner22_false = QuantumCircuit(loop_bits)
            loop_operation(inner22_false)

            inner23_true = QuantumCircuit(qubits[8:10], [clbits[0], clbits[11], clbits[12]])
            with self.assertWarns(DeprecationWarning):
                inner23_true.h(qubits[8]).c_if(clbits[11], 0)
            inner23_false = QuantumCircuit(qubits[8:10], [clbits[0], clbits[11], clbits[12]])
            with self.assertWarns(DeprecationWarning):
                inner23_false.h(qubits[9]).c_if(clbits[12], 0)

            outer2_true = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                outer2_true.h(qubits[3]).c_if(clbits[6], 0)
            outer2_true.if_else(cond_inner, inner21_true, inner21_false, loop_qubits, loop_clbits)
            outer2_true.if_else(cond_inner, inner22_true, inner22_false, loop_qubits, loop_clbits)
            with self.assertWarns(DeprecationWarning):
                outer2_true.h(qubits[6]).c_if(clbits[9], 0)
            outer2_false = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                outer2_false.h(qubits[7]).c_if(clbits[10], 0)
            outer2_false.if_else(
                cond_inner,
                inner23_true,
                inner23_false,
                [qubits[8], qubits[9]],
                [clbits[0], clbits[11], clbits[12]],
            )

            inner31_true = QuantumCircuit(loop_bits)
            loop_operation(inner31_true)
            inner31_false = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                inner31_false.h(qubits[12]).c_if(clbits[15], 0)

            inner32_true = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                inner32_true.h(qubits[13]).c_if(clbits[16], 0)
            inner32_false = QuantumCircuit(loop_bits)
            loop_operation(inner32_false)

            inner33_true = QuantumCircuit(qubits[14:16], [clbits[0], clbits[17], clbits[18]])
            with self.assertWarns(DeprecationWarning):
                inner33_true.h(qubits[14]).c_if(clbits[17], 0)
            inner33_false = QuantumCircuit(qubits[14:16], [clbits[0], clbits[17], clbits[18]])
            with self.assertWarns(DeprecationWarning):
                inner33_false.h(qubits[15]).c_if(clbits[18], 0)

            outer3_true = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                outer3_true.h(qubits[10]).c_if(clbits[13], 0)
            outer3_false = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                outer3_false.h(qubits[11]).c_if(clbits[14], 0)
            outer3_false.if_else(cond_inner, inner31_true, inner31_false, loop_qubits, loop_clbits)
            outer3_false.if_else(cond_inner, inner32_true, inner32_false, loop_qubits, loop_clbits)
            outer3_false.if_else(
                cond_inner,
                inner33_true,
                inner33_false,
                qubits[14:16],
                [clbits[0], clbits[17], clbits[18]],
            )

            loop_body = QuantumCircuit(loop_bits)
            with self.assertWarns(DeprecationWarning):
                loop_body.h(qubits[0]).c_if(clbits[3], 0)
            loop_body.if_else(
                cond_outer,
                outer1_true,
                outer1_false,
                qubits[1:3],
                [clbits[1], clbits[4], clbits[5]],
            )
            loop_body.if_else(cond_outer, outer2_true, outer2_false, loop_qubits, loop_clbits)
            loop_body.if_else(cond_outer, outer3_true, outer3_false, loop_qubits, loop_clbits)
            with self.assertWarns(DeprecationWarning):
                loop_body.h(qubits[16]).c_if(clbits[19], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond_loop, loop_body, loop_qubits, loop_clbits)

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("if/while/if/switch"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond_outer):  # outer_t
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(3, 0)
                with test.while_loop(cond_loop):  # loop
                    with self.assertWarns(DeprecationWarning):
                        test.h(1).c_if(4, 0)
                    with test.if_test(cond_inner):  # inner_t
                        with self.assertWarns(DeprecationWarning):
                            test.h(2).c_if(5, 0)
                        with test.switch(5) as case_:
                            with case_(False):  # case_f
                                with self.assertWarns(DeprecationWarning):
                                    test.h(3).c_if(6, 0)
                            with case_(True):  # case_t
                                loop_operation(test)
                        with self.assertWarns(DeprecationWarning):
                            test.h(4).c_if(7, 0)
                    # exit inner_t
                    with self.assertWarns(DeprecationWarning):
                        test.h(5).c_if(8, 0)
                # exit loop
                with self.assertWarns(DeprecationWarning):
                    test.h(6).c_if(9, 0)
            # exit outer_t
            with self.assertWarns(DeprecationWarning):
                test.h(7).c_if(10, 0)

            case_f = QuantumCircuit(qubits[1:6], [clbits[0], clbits[2]] + clbits[4:9])
            with self.assertWarns(DeprecationWarning):
                case_f.h(qubits[3]).c_if(clbits[6], 0)
            case_t = QuantumCircuit(qubits[1:6], [clbits[0], clbits[2]] + clbits[4:9])
            loop_operation(case_t)

            inner_t = QuantumCircuit(qubits[1:6], [clbits[0], clbits[2]] + clbits[4:9])
            with self.assertWarns(DeprecationWarning):
                inner_t.h(qubits[2]).c_if(clbits[5], 0)
            inner_t.switch(
                clbits[5],
                [(False, case_f), (True, case_t)],
                qubits[1:6],
                [clbits[0], clbits[2]] + clbits[4:9],
            )
            with self.assertWarns(DeprecationWarning):
                inner_t.h(qubits[4]).c_if(clbits[7], 0)

            loop = QuantumCircuit(qubits[1:6], [clbits[0], clbits[2]] + clbits[4:9])
            with self.assertWarns(DeprecationWarning):
                loop.h(qubits[1]).c_if(clbits[4], 0)
            loop.if_test(cond_inner, inner_t, qubits[1:6], [clbits[0], clbits[2]] + clbits[4:9])
            with self.assertWarns(DeprecationWarning):
                loop.h(qubits[5]).c_if(clbits[8], 0)

            outer_t = QuantumCircuit(qubits[:7], clbits[:10])
            with self.assertWarns(DeprecationWarning):
                outer_t.h(qubits[0]).c_if(clbits[3], 0)
            outer_t.while_loop(cond_loop, loop, qubits[1:6], [clbits[0], clbits[2]] + clbits[4:9])
            with self.assertWarns(DeprecationWarning):
                outer_t.h(qubits[6]).c_if(clbits[9], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.if_test(cond_outer, outer_t, qubits[:7], clbits[:10])
            with self.assertWarns(DeprecationWarning):
                expected.h(qubits[7]).c_if(clbits[10], 0)

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("switch/for/switch/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.switch(0) as case_outer:
                with case_outer(False):  # outer_case_f
                    with self.assertWarns(DeprecationWarning):
                        test.h(0).c_if(3, 0)
                    with test.for_loop(range(2)):  # loop
                        with self.assertWarns(DeprecationWarning):
                            test.h(1).c_if(4, 0)
                        with test.switch(1) as case_inner:
                            with case_inner(False):  # inner_case_f
                                with self.assertWarns(DeprecationWarning):
                                    test.h(2).c_if(5, 0)
                                with test.if_test((2, True)) as else_:  # if_t
                                    with self.assertWarns(DeprecationWarning):
                                        test.h(3).c_if(6, 0)
                                with else_:  # if_f
                                    loop_operation(test)
                                with self.assertWarns(DeprecationWarning):
                                    test.h(4).c_if(7, 0)
                            with case_inner(True):  # inner_case_t
                                loop_operation(test)
                        with self.assertWarns(DeprecationWarning):
                            test.h(5).c_if(8, 0)
                    # exit loop1
                    with self.assertWarns(DeprecationWarning):
                        test.h(6).c_if(9, 0)
                with case_outer(True):  # outer_case_t
                    with self.assertWarns(DeprecationWarning):
                        test.h(7).c_if(10, 0)
            with self.assertWarns(DeprecationWarning):
                test.h(8).c_if(11, 0)

            if_t = QuantumCircuit(qubits[1:6], clbits[1:3] + clbits[4:9])
            with self.assertWarns(DeprecationWarning):
                if_t.h(qubits[3]).c_if(clbits[6], 0)
            if_f = QuantumCircuit(qubits[1:6], clbits[1:3] + clbits[4:9])
            loop_operation(if_f)

            inner_case_f = QuantumCircuit(qubits[1:6], clbits[1:3] + clbits[4:9])
            with self.assertWarns(DeprecationWarning):
                inner_case_f.h(qubits[2]).c_if(clbits[5], 0)
            inner_case_f.if_else(
                (clbits[2], True), if_t, if_f, qubits[1:6], clbits[1:3] + clbits[4:9]
            )
            with self.assertWarns(DeprecationWarning):
                inner_case_f.h(qubits[4]).c_if(clbits[7], 0)

            inner_case_t = QuantumCircuit(qubits[1:6], clbits[1:3] + clbits[4:9])
            loop_operation(inner_case_t)

            loop = QuantumCircuit(qubits[1:6], clbits[1:3] + clbits[4:9])
            with self.assertWarns(DeprecationWarning):
                loop.h(qubits[1]).c_if(clbits[4], 0)
            loop.switch(
                clbits[1],
                [(False, inner_case_f), (True, inner_case_t)],
                qubits[1:6],
                clbits[1:3] + clbits[4:9],
            )
            with self.assertWarns(DeprecationWarning):
                loop.h(qubits[5]).c_if(clbits[8], 0)

            outer_case_f = QuantumCircuit(qubits[:8], clbits[:11])
            with self.assertWarns(DeprecationWarning):
                outer_case_f.h(qubits[0]).c_if(clbits[3], 0)
            outer_case_f.for_loop(range(2), None, loop, qubits[1:6], clbits[1:3] + clbits[4:9])
            with self.assertWarns(DeprecationWarning):
                outer_case_f.h(qubits[6]).c_if(clbits[9], 0)

            outer_case_t = QuantumCircuit(qubits[:8], clbits[:11])
            with self.assertWarns(DeprecationWarning):
                outer_case_t.h(qubits[7]).c_if(clbits[10], 0)

            expected = QuantumCircuit(qubits, clbits)
            expected.switch(
                clbits[0], [(False, outer_case_f), (True, outer_case_t)], qubits[:8], clbits[:11]
            )
            with self.assertWarns(DeprecationWarning):
                expected.h(qubits[8]).c_if(clbits[11], 0)

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_for_handles_iterables_correctly(self):
        """Test that the ``indexset`` in ``for`` loops is handled the way we expect.  In general,
        this means all iterables are consumed into a tuple on first access, except for ``range``
        which is passed through as-is."""
        bits = [Qubit(), Clbit()]
        expected_indices = (3, 9, 1)

        with self.subTest("list"):
            test = QuantumCircuit(bits)
            with test.for_loop(list(expected_indices)):
                pass
            instruction = test.data[-1].operation
            self.assertIsInstance(instruction, ForLoopOp)
            indices, _, _ = instruction.params
            self.assertEqual(indices, expected_indices)

        with self.subTest("tuple"):
            test = QuantumCircuit(bits)
            with test.for_loop(tuple(expected_indices)):
                pass
            instruction = test.data[-1].operation
            self.assertIsInstance(instruction, ForLoopOp)
            indices, _, _ = instruction.params
            self.assertEqual(indices, expected_indices)

        with self.subTest("consumable"):

            def consumable():
                yield from expected_indices

            test = QuantumCircuit(bits)
            with test.for_loop(consumable()):
                pass
            instruction = test.data[-1].operation
            self.assertIsInstance(instruction, ForLoopOp)
            indices, _, _ = instruction.params
            self.assertEqual(indices, expected_indices)

        with self.subTest("range"):
            range_indices = range(0, 8, 2)

            test = QuantumCircuit(bits)
            with test.for_loop(range_indices):
                pass
            instruction = test.data[-1].operation
            self.assertIsInstance(instruction, ForLoopOp)
            indices, _, _ = instruction.params
            self.assertEqual(indices, range_indices)

    def test_for_returns_a_given_parameter(self):
        """Test that the ``for``-loop manager returns the parameter that we gave it."""
        parameter = Parameter("x")
        test = QuantumCircuit(1, 1)
        with test.for_loop((0, 1), parameter) as test_parameter:
            pass
        self.assertIs(test_parameter, parameter)

    def test_for_binds_parameter_to_op(self):
        """Test that the ``for`` manager binds a parameter to the resulting :obj:`.ForLoopOp` if a
        user-generated one is given, or if a generated parameter is used.  Generated parameters that
        are not used should not be bound."""
        parameter = Parameter("x")

        with self.subTest("passed and used"):
            circuit = QuantumCircuit(1, 1)
            with circuit.for_loop((0, 0.5 * math.pi), parameter) as received_parameter:
                circuit.rx(received_parameter, 0)
            self.assertIs(parameter, received_parameter)
            instruction = circuit.data[-1].operation
            self.assertIsInstance(instruction, ForLoopOp)
            _, bound_parameter, _ = instruction.params
            self.assertIs(bound_parameter, parameter)

        with self.subTest("passed and unused"):
            circuit = QuantumCircuit(1, 1)
            with circuit.for_loop((0, 0.5 * math.pi), parameter) as received_parameter:
                circuit.x(0)
            self.assertIs(parameter, received_parameter)
            instruction = circuit.data[-1].operation
            self.assertIsInstance(instruction, ForLoopOp)
            _, bound_parameter, _ = instruction.params
            self.assertIs(parameter, received_parameter)

        with self.subTest("generated and used"):
            circuit = QuantumCircuit(1, 1)
            with circuit.for_loop((0, 0.5 * math.pi)) as received_parameter:
                circuit.rx(received_parameter, 0)
            self.assertIsInstance(received_parameter, Parameter)
            instruction = circuit.data[-1].operation
            self.assertIsInstance(instruction, ForLoopOp)
            _, bound_parameter, _ = instruction.params
            self.assertIs(bound_parameter, received_parameter)

        with self.subTest("generated and used in deferred-build if"):
            circuit = QuantumCircuit(1, 1)
            with circuit.for_loop((0, 0.5 * math.pi)) as received_parameter:
                with circuit.if_test((0, 0)):
                    circuit.rx(received_parameter, 0)
                    circuit.break_loop()
            self.assertIsInstance(received_parameter, Parameter)
            instruction = circuit.data[-1].operation
            self.assertIsInstance(instruction, ForLoopOp)
            _, bound_parameter, _ = instruction.params
            self.assertIs(bound_parameter, received_parameter)

        with self.subTest("generated and used in deferred-build else"):
            circuit = QuantumCircuit(1, 1)
            with circuit.for_loop((0, 0.5 * math.pi)) as received_parameter:
                with circuit.if_test((0, 0)) as else_:
                    pass
                with else_:
                    circuit.rx(received_parameter, 0)
                    circuit.break_loop()
            self.assertIsInstance(received_parameter, Parameter)
            instruction = circuit.data[-1].operation
            self.assertIsInstance(instruction, ForLoopOp)
            _, bound_parameter, _ = instruction.params
            self.assertIs(bound_parameter, received_parameter)

    def test_for_does_not_bind_generated_parameter_if_unused(self):
        """Test that the ``for`` manager does not bind a generated parameter into the resulting
        :obj:`.ForLoopOp` if the parameter was not used."""
        test = QuantumCircuit(1, 1)
        with test.for_loop(range(2)) as generated_parameter:
            pass
        instruction = test.data[-1].operation
        self.assertIsInstance(instruction, ForLoopOp)
        _, bound_parameter, _ = instruction.params
        self.assertIsNot(generated_parameter, None)
        self.assertIs(bound_parameter, None)

    def test_for_allocates_parameters(self):
        """Test that the ``for``-loop manager allocates a parameter if it is given ``None``, and
        that it always allocates new parameters."""
        test = QuantumCircuit(1, 1)
        with test.for_loop(range(2)) as outer_parameter:
            with test.for_loop(range(2)) as inner_parameter:
                pass
        with test.for_loop(range(2)) as final_parameter:
            pass
        self.assertIsInstance(outer_parameter, Parameter)
        self.assertIsInstance(inner_parameter, Parameter)
        self.assertIsInstance(final_parameter, Parameter)
        self.assertNotEqual(outer_parameter, inner_parameter)
        self.assertNotEqual(outer_parameter, final_parameter)
        self.assertNotEqual(inner_parameter, final_parameter)

    def test_access_of_resources_from_direct_append(self):
        """Test that direct calls to :obj:`.QuantumCircuit.append` within a builder block still
        collect all the relevant resources."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]
        cond = (clbits[0], 0)

        with self.subTest("if"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond):
                test.append(Measure(), [qubits[1]], [clbits[1]])

            true_body = QuantumCircuit([qubits[1]], clbits)
            true_body.measure(qubits[1], clbits[1])
            expected = QuantumCircuit(qubits, clbits)
            expected.if_test(cond, true_body, [qubits[1]], clbits)

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("else"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                test.append(Measure(), [qubits[1]], [clbits[1]])

            true_body = QuantumCircuit([qubits[1]], clbits)
            false_body = QuantumCircuit([qubits[1]], clbits)
            false_body.measure(qubits[1], clbits[1])
            expected = QuantumCircuit(qubits, clbits)
            expected.if_else(cond, true_body, false_body, [qubits[1]], clbits)

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.append(Measure(), [qubits[1]], [clbits[1]])

            body = QuantumCircuit([qubits[1]], [clbits[1]])
            body.measure(qubits[1], clbits[1])
            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, body, [qubits[1]], [clbits[1]])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("while"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond):
                test.append(Measure(), [qubits[1]], [clbits[1]])

            body = QuantumCircuit([qubits[1]], clbits)
            body.measure(qubits[1], clbits[1])
            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond, body, [qubits[1]], clbits)

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("switch"):
            test = QuantumCircuit(qubits, clbits)
            with test.switch(cond[0]) as case:
                with case(0):
                    test.append(Measure(), [qubits[1]], [clbits[1]])

            body = QuantumCircuit([qubits[1]], clbits)
            body.measure(qubits[1], clbits[1])
            expected = QuantumCircuit(qubits, clbits)
            expected.switch(cond[0], [(0, body)], [qubits[1]], clbits)

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_access_of_clbit_from_c_if(self):
        """Test that resources added from a call to :meth:`.InstructionSet.c_if` propagate through
        the context managers correctly."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]
        bits = qubits + clbits
        cond = (clbits[0], 0)

        with self.subTest("if"):
            test = QuantumCircuit(bits)
            with test.if_test(cond):
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], clbits)
            with self.assertWarns(DeprecationWarning):
                body.h(qubits[0]).c_if(clbits[1], 0)
            expected = QuantumCircuit(bits)
            expected.if_test(cond, body, [qubits[0]], clbits)

        with self.subTest("else"):
            test = QuantumCircuit(bits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(1, 0)

            true_body = QuantumCircuit([qubits[0]], clbits)
            false_body = QuantumCircuit([qubits[0]], clbits)
            with self.assertWarns(DeprecationWarning):
                false_body.h(qubits[0]).c_if(clbits[1], 0)
            expected = QuantumCircuit(bits)
            expected.if_else(cond, true_body, false_body, [qubits[0]], clbits)

        with self.subTest("for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], clbits)
            with self.assertWarns(DeprecationWarning):
                body.h(qubits[0]).c_if(clbits[1], 0)
            expected = QuantumCircuit(bits)
            expected.for_loop(range(2), None, body, [qubits[0]], clbits)

        with self.subTest("while"):
            test = QuantumCircuit(bits)
            with test.while_loop(cond):
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], clbits)
            with self.assertWarns(DeprecationWarning):
                body.h(qubits[0]).c_if(clbits[1], 0)
            expected = QuantumCircuit(bits)
            expected.while_loop(cond, body, [qubits[0]], clbits)

        with self.subTest("switch"):
            test = QuantumCircuit(bits)
            with test.switch(cond[0]) as case, case(False):
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], clbits)
            with self.assertWarns(DeprecationWarning):
                body.h(qubits[0]).c_if(clbits[1], 0)
            expected = QuantumCircuit(bits)
            expected.switch(cond[0], [(False, body)], [qubits[0]], clbits)

        with self.subTest("if inside for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.if_test(cond):
                    with self.assertWarns(DeprecationWarning):
                        test.h(0).c_if(1, 0)

            true_body = QuantumCircuit([qubits[0]], clbits)
            with self.assertWarns(DeprecationWarning):
                true_body.h(qubits[0]).c_if(clbits[1], 0)
            body = QuantumCircuit([qubits[0]], clbits)
            body.if_test(cond, body, [qubits[0]], clbits)
            expected = QuantumCircuit(bits)
            expected.for_loop(range(2), None, body, [qubits[0]], clbits)

        with self.subTest("switch inside for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)), test.switch(cond[0]) as case, case(False):
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], clbits)
            with self.assertWarns(DeprecationWarning):
                body.h(qubits[0]).c_if(clbits[1], 0)
            body = QuantumCircuit([qubits[0]], clbits)
            body.switch(cond[0], [(False, body)], [qubits[0]], clbits)
            expected = QuantumCircuit(bits)
            expected.for_loop(range(2), None, body, [qubits[0]], clbits)

    def test_access_of_classicalregister_from_c_if(self):
        """Test that resources added from a call to :meth:`.InstructionSet.c_if` propagate through
        the context managers correctly."""
        qubits = [Qubit(), Qubit()]
        creg = ClassicalRegister(2)
        clbits = [Clbit()]
        all_clbits = list(clbits) + list(creg)
        cond = (clbits[0], 0)

        with self.subTest("if"):
            test = QuantumCircuit(qubits, clbits, creg)
            with test.if_test(cond):
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(creg, 0)

            body = QuantumCircuit([qubits[0]], clbits, creg)
            with self.assertWarns(DeprecationWarning):
                body.h(qubits[0]).c_if(creg, 0)
            expected = QuantumCircuit(qubits, clbits, creg)
            expected.if_test(cond, body, [qubits[0]], all_clbits)

        with self.subTest("else"):
            test = QuantumCircuit(qubits, clbits, creg)
            with test.if_test(cond) as else_:
                pass
            with else_:
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(1, 0)

            true_body = QuantumCircuit([qubits[0]], clbits, creg)
            false_body = QuantumCircuit([qubits[0]], clbits, creg)
            with self.assertWarns(DeprecationWarning):
                false_body.h(qubits[0]).c_if(creg, 0)
            expected = QuantumCircuit(qubits, clbits, creg)
            expected.if_else(cond, true_body, false_body, [qubits[0]], all_clbits)

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits, creg)
            with test.for_loop(range(2)):
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(1, 0)

            body = QuantumCircuit([qubits[0]], clbits, creg)
            with self.assertWarns(DeprecationWarning):
                body.h(qubits[0]).c_if(creg, 0)
            expected = QuantumCircuit(qubits, clbits, creg)
            expected.for_loop(range(2), None, body, [qubits[0]], all_clbits)

        with self.subTest("while"):
            test = QuantumCircuit(qubits, clbits, creg)
            with test.while_loop(cond):
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(creg, 0)

            body = QuantumCircuit([qubits[0]], clbits, creg)
            with self.assertWarns(DeprecationWarning):
                body.h(qubits[0]).c_if(creg, 0)
            expected = QuantumCircuit(qubits, clbits, creg)
            expected.while_loop(cond, body, [qubits[0]], all_clbits)

        with self.subTest("switch"):
            test = QuantumCircuit(qubits, clbits, creg)
            with test.switch(cond[0]) as case, case(False):
                with self.assertWarns(DeprecationWarning):
                    test.h(0).c_if(creg, 0)

            body = QuantumCircuit([qubits[0]], clbits, creg)
            with self.assertWarns(DeprecationWarning):
                body.h(qubits[0]).c_if(creg, 0)
            expected = QuantumCircuit(qubits, clbits, creg)
            expected.switch(cond[0], [(False, body)], [qubits[0]], all_clbits)

        with self.subTest("if inside for"):
            test = QuantumCircuit(qubits, clbits, creg)
            with test.for_loop(range(2)):
                with test.if_test(cond):
                    with self.assertWarns(DeprecationWarning):
                        test.h(0).c_if(creg, 0)

            true_body = QuantumCircuit([qubits[0]], clbits, creg)
            with self.assertWarns(DeprecationWarning):
                true_body.h(qubits[0]).c_if(creg, 0)
            body = QuantumCircuit([qubits[0]], clbits, creg)
            body.if_test(cond, body, [qubits[0]], all_clbits)
            expected = QuantumCircuit(qubits, clbits, creg)
            expected.for_loop(range(2), None, body, [qubits[0]], all_clbits)

        with self.subTest("switch inside for"):
            test = QuantumCircuit(qubits, clbits, creg)
            with test.for_loop(range(2)):
                with test.switch(cond[0]) as case, case(False):
                    with self.assertWarns(DeprecationWarning):
                        test.h(0).c_if(creg, 0)

            case = QuantumCircuit([qubits[0]], clbits, creg)
            with self.assertWarns(DeprecationWarning):
                case.h(qubits[0]).c_if(creg, 0)
            body = QuantumCircuit([qubits[0]], clbits, creg)
            body.switch(cond[0], [(False, case)], [qubits[0]], all_clbits)
            expected = QuantumCircuit(qubits, clbits, creg)
            expected.for_loop(range(2), None, body, [qubits[0]], all_clbits)

    def test_accept_broadcast_gates(self):
        """Test that the context managers accept gates that are broadcast during their addition to
        the scope."""
        qubits = [Qubit(), Qubit(), Qubit()]
        clbits = [Clbit(), Clbit(), Clbit()]
        cond = (clbits[0], 0)

        with self.subTest("if"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond):
                test.measure([0, 1], [0, 1])

            body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            body.measure(qubits[0], clbits[0])
            body.measure(qubits[1], clbits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.if_test(cond, body, [qubits[0], qubits[1]], [clbits[0], clbits[1]])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("else"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                test.measure([0, 1], [0, 1])

            true_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            false_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            false_body.measure(qubits[0], clbits[0])
            false_body.measure(qubits[1], clbits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.if_else(
                cond, true_body, false_body, [qubits[0], qubits[1]], [clbits[0], clbits[1]]
            )

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                test.measure([0, 1], [0, 1])

            body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            body.measure(qubits[0], clbits[0])
            body.measure(qubits[1], clbits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(range(2), None, body, [qubits[0], qubits[1]], [clbits[0], clbits[1]])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("while"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond):
                test.measure([0, 1], [0, 1])

            body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            body.measure(qubits[0], clbits[0])
            body.measure(qubits[1], clbits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.while_loop(cond, body, [qubits[0], qubits[1]], [clbits[0], clbits[1]])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("switch"):
            test = QuantumCircuit(qubits, clbits)
            with test.switch(cond[0]) as case, case(True):
                test.measure([0, 1], [0, 1])

            body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            body.measure(qubits[0], clbits[0])
            body.measure(qubits[1], clbits[1])

            expected = QuantumCircuit(qubits, clbits)
            expected.switch(cond[0], [(True, body)], [qubits[0], qubits[1]], [clbits[0], clbits[1]])

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("if inside for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)):
                with test.if_test(cond):
                    test.measure([0, 1], [0, 1])

            true_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            true_body.measure(qubits[0], clbits[0])
            true_body.measure(qubits[1], clbits[1])

            for_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            for_body.if_test(cond, true_body, [qubits[0], qubits[1]], [clbits[0], clbits[1]])

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(
                range(2), None, for_body, [qubits[0], qubits[1]], [clbits[0], clbits[1]]
            )

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

        with self.subTest("switch inside for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(2)), test.switch(cond[0]) as case, case(True):
                test.measure([0, 1], [0, 1])

            case_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            case_body.measure(qubits[0], clbits[0])
            case_body.measure(qubits[1], clbits[1])

            for_body = QuantumCircuit([qubits[0], qubits[1], clbits[0], clbits[1]])
            for_body.switch(cond[0], [(True, body)], [qubits[0], qubits[1]], [clbits[0], clbits[1]])

            expected = QuantumCircuit(qubits, clbits)
            expected.for_loop(
                range(2), None, for_body, [qubits[0], qubits[1]], [clbits[0], clbits[1]]
            )

            self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_labels_propagated_to_instruction(self):
        """Test that labels given to the circuit-builder interface are passed through."""
        bits = [Qubit(), Clbit()]
        cond = (bits[1], 0)
        label = "sentinel_label"

        with self.subTest("if"):
            test = QuantumCircuit(bits)
            with test.if_test(cond, label=label):
                pass
            instruction = test.data[-1].operation
            self.assertIsInstance(instruction, IfElseOp)
            self.assertEqual(instruction.label, label)

        with self.subTest("if else"):
            test = QuantumCircuit(bits)
            with test.if_test(cond, label=label) as else_:
                pass
            with else_:
                pass
            instruction = test.data[-1].operation
            self.assertIsInstance(instruction, IfElseOp)
            self.assertEqual(instruction.label, label)

        with self.subTest("for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2), label=label):
                pass
            instruction = test.data[-1].operation
            self.assertIsInstance(instruction, ForLoopOp)
            self.assertEqual(instruction.label, label)

        with self.subTest("while"):
            test = QuantumCircuit(bits)
            with test.while_loop(cond, label=label):
                pass
            instruction = test.data[-1].operation
            self.assertIsInstance(instruction, WhileLoopOp)
            self.assertEqual(instruction.label, label)

        with self.subTest("switch"):
            test = QuantumCircuit(bits)
            with test.switch(cond[0], label=label) as case:
                with case(False):
                    pass
            instruction = test.data[-1].operation
            self.assertIsInstance(instruction, SwitchCaseOp)
            self.assertEqual(instruction.label, label)

        # The tests of blocks inside 'for' are to ensure we're hitting the paths where the scope is
        # built lazily at the completion of the 'for'.
        with self.subTest("if inside for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.if_test(cond, label=label):
                    # Use break to ensure that we're triggering the lazy building of 'if'.
                    test.break_loop()

            instruction = test.data[-1].operation.blocks[0].data[-1].operation
            self.assertIsInstance(instruction, IfElseOp)
            self.assertEqual(instruction.label, label)

        with self.subTest("else inside for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.if_test(cond, label=label) as else_:
                    # Use break to ensure that we're triggering the lazy building of 'if'.
                    test.break_loop()
                with else_:
                    test.break_loop()

            instruction = test.data[-1].operation.blocks[0].data[-1].operation
            self.assertIsInstance(instruction, IfElseOp)
            self.assertEqual(instruction.label, label)

        with self.subTest("switch inside for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.switch(cond[0], label=label) as case:
                    with case(False):
                        # Use break to ensure that we're triggering the lazy building
                        test.break_loop()

            instruction = test.data[-1].operation.blocks[0].data[-1].operation
            self.assertIsInstance(instruction, SwitchCaseOp)
            self.assertEqual(instruction.label, label)

    def test_copy_of_circuits(self):
        """Test that various methods of copying a circuit made with the builder interface works."""
        test = QuantumCircuit(5, 5)
        cond = (test.clbits[2], False)
        with test.if_test(cond) as else_:
            test.cx(0, 1)
        with else_:
            test.h(2)
        with test.for_loop(range(5)):
            with test.if_test(cond):
                test.x(3)
        with test.while_loop(cond):
            test.measure(0, 4)
        self.assertEqual(test, test.copy())
        self.assertEqual(test, copy.copy(test))
        self.assertEqual(test, copy.deepcopy(test))

    def test_copy_of_instructions(self):
        """Test that various methods of copying the actual instructions created by the builder
        interface work."""
        qubits = [Qubit() for _ in [None] * 3]
        clbits = [Clbit() for _ in [None] * 3]
        cond = (clbits[1], False)

        with self.subTest("if"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond):
                test.cx(0, 1)
                test.measure(2, 2)
            if_instruction = test.data[0].operation
            self.assertEqual(if_instruction, if_instruction.copy())
            self.assertEqual(if_instruction, copy.copy(if_instruction))
            self.assertEqual(if_instruction, copy.deepcopy(if_instruction))

        with self.subTest("if/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond) as else_:
                test.cx(0, 1)
                test.measure(2, 2)
            with else_:
                test.cx(1, 0)
                test.measure(2, 2)
            if_instruction = test.data[0].operation
            self.assertEqual(if_instruction, if_instruction.copy())
            self.assertEqual(if_instruction, copy.copy(if_instruction))
            self.assertEqual(if_instruction, copy.deepcopy(if_instruction))

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(4)):
                test.cx(0, 1)
                test.measure(2, 2)
            for_instruction = test.data[0].operation
            self.assertEqual(for_instruction, for_instruction.copy())
            self.assertEqual(for_instruction, copy.copy(for_instruction))
            self.assertEqual(for_instruction, copy.deepcopy(for_instruction))

        with self.subTest("while"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond):
                test.cx(0, 1)
                test.measure(2, 2)
            while_instruction = test.data[0].operation
            self.assertEqual(while_instruction, while_instruction.copy())
            self.assertEqual(while_instruction, copy.copy(while_instruction))
            self.assertEqual(while_instruction, copy.deepcopy(while_instruction))

        with self.subTest("switch"):
            creg = ClassicalRegister(4)
            test = QuantumCircuit(qubits, creg)
            with test.switch(creg) as case:
                with case(0):
                    test.h(0)
                with case(1, 2, 3):
                    test.z(1)
                with case(case.DEFAULT):
                    test.cx(0, 1)
                    test.measure(2, 2)
            switch_instruction = test.data[0].operation
            self.assertEqual(switch_instruction, switch_instruction.copy())
            self.assertEqual(switch_instruction, copy.copy(switch_instruction))
            self.assertEqual(switch_instruction, copy.deepcopy(switch_instruction))

    def test_copy_of_instruction_parameters(self):
        """Test that various methods of copying the parameters inside instructions created by the
        builder interface work.  Regression test of gh-7367."""
        qubits = [Qubit() for _ in [None] * 3]
        clbits = [Clbit() for _ in [None] * 3]
        cond = (clbits[1], False)

        with self.subTest("if"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond):
                test.cx(0, 1)
                test.measure(2, 2)
            if_instruction = test.data[0].operation
            (true_body,) = if_instruction.blocks
            self.assertEqual(true_body, true_body.copy())
            self.assertEqual(true_body, copy.copy(true_body))
            self.assertEqual(true_body, copy.deepcopy(true_body))

        with self.subTest("if/else"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond) as else_:
                test.cx(0, 1)
                test.measure(2, 2)
            with else_:
                test.cx(1, 0)
                test.measure(2, 2)
            if_instruction = test.data[0].operation
            true_body, false_body = if_instruction.blocks
            self.assertEqual(true_body, true_body.copy())
            self.assertEqual(true_body, copy.copy(true_body))
            self.assertEqual(true_body, copy.deepcopy(true_body))
            self.assertEqual(false_body, false_body.copy())
            self.assertEqual(false_body, copy.copy(false_body))
            self.assertEqual(false_body, copy.deepcopy(false_body))

        with self.subTest("for"):
            test = QuantumCircuit(qubits, clbits)
            with test.for_loop(range(4)):
                test.cx(0, 1)
                test.measure(2, 2)
            for_instruction = test.data[0].operation
            (for_body,) = for_instruction.blocks
            self.assertEqual(for_body, for_body.copy())
            self.assertEqual(for_body, copy.copy(for_body))
            self.assertEqual(for_body, copy.deepcopy(for_body))

        with self.subTest("while"):
            test = QuantumCircuit(qubits, clbits)
            with test.while_loop(cond):
                test.cx(0, 1)
                test.measure(2, 2)
            while_instruction = test.data[0].operation
            (while_body,) = while_instruction.blocks
            self.assertEqual(while_body, while_body.copy())
            self.assertEqual(while_body, copy.copy(while_body))
            self.assertEqual(while_body, copy.deepcopy(while_body))

        with self.subTest("switch"):
            test = QuantumCircuit(qubits, clbits)
            with test.switch(cond[0]) as case, case(0):
                test.cx(0, 1)
                test.measure(2, 2)
            case_instruction = test.data[0].operation
            (case_body,) = case_instruction.blocks
            self.assertEqual(case_body, case_body.copy())
            self.assertEqual(case_body, copy.copy(case_body))
            self.assertEqual(case_body, copy.deepcopy(case_body))

    def test_inplace_compose_within_builder(self):
        """Test that QuantumCircuit.compose used in-place works as expected within control-flow
        scopes."""
        inner = QuantumCircuit(1)
        inner.x(0)

        base = QuantumCircuit(1, 1)
        base.h(0)
        base.measure(0, 0)

        with self.subTest("if"):
            outer = base.copy()
            with outer.if_test((outer.clbits[0], 1)):
                outer.compose(inner, inplace=True)

            expected = base.copy()
            with expected.if_test((expected.clbits[0], 1)):
                expected.x(0)

            self.assertEqual(canonicalize_control_flow(outer), canonicalize_control_flow(expected))

        with self.subTest("else"):
            outer = base.copy()
            with outer.if_test((outer.clbits[0], 1)) as else_:
                outer.compose(inner, inplace=True)
            with else_:
                outer.compose(inner, inplace=True)

            expected = base.copy()
            with expected.if_test((expected.clbits[0], 1)) as else_:
                expected.x(0)
            with else_:
                expected.x(0)

            self.assertEqual(canonicalize_control_flow(outer), canonicalize_control_flow(expected))

        with self.subTest("for"):
            outer = base.copy()
            with outer.for_loop(range(3)):
                outer.compose(inner, inplace=True)

            expected = base.copy()
            with expected.for_loop(range(3)):
                expected.x(0)

            self.assertEqual(canonicalize_control_flow(outer), canonicalize_control_flow(expected))

        with self.subTest("while"):
            outer = base.copy()
            with outer.while_loop((outer.clbits[0], 0)):
                outer.compose(inner, inplace=True)

            expected = base.copy()
            with expected.while_loop((outer.clbits[0], 0)):
                expected.x(0)

            self.assertEqual(canonicalize_control_flow(outer), canonicalize_control_flow(expected))

        with self.subTest("switch"):
            outer = base.copy()
            with outer.switch(outer.clbits[0]) as case, case(False):
                outer.compose(inner, inplace=True)

            expected = base.copy()
            with expected.switch(outer.clbits[0]) as case, case(False):
                expected.x(0)

            self.assertEqual(canonicalize_control_flow(outer), canonicalize_control_flow(expected))

    def test_global_phase_of_blocks(self):
        """It should be possible to set a global phase of a scope independently of the containing
        scope and other sibling scopes."""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        qc = QuantumCircuit(qr, cr, global_phase=math.pi)

        with qc.if_test((qc.clbits[0], False)):
            # This scope's phase shouldn't be affected by the outer scope.
            self.assertEqual(qc.global_phase, 0.0)
            qc.global_phase += math.pi / 2
            self.assertEqual(qc.global_phase, math.pi / 2)
        # Back outside the scope, the phase shouldn't have changed...
        self.assertEqual(qc.global_phase, math.pi)
        # ... but we still should be able to see the phase in the built block definition.
        self.assertEqual(qc.data[-1].operation.blocks[0].global_phase, math.pi / 2)

        with qc.while_loop((qc.clbits[1], False)):
            self.assertEqual(qc.global_phase, 0.0)
            qc.global_phase = 1 * math.pi / 7
            with qc.for_loop(range(3)):
                self.assertEqual(qc.global_phase, 0.0)
                qc.global_phase = 2 * math.pi / 7

            with qc.if_test((qc.clbits[2], False)) as else_:
                self.assertEqual(qc.global_phase, 0.0)
                qc.global_phase = 3 * math.pi / 7
            with else_:
                self.assertEqual(qc.global_phase, 0.0)
                qc.global_phase = 4 * math.pi / 7

            with qc.switch(cr) as case:
                with case(0):
                    self.assertEqual(qc.global_phase, 0.0)
                    qc.global_phase = 5 * math.pi / 7
                with case(case.DEFAULT):
                    self.assertEqual(qc.global_phase, 0.0)
                    qc.global_phase = 6 * math.pi / 7

        while_body = qc.data[-1].operation.blocks[0]
        for_body = while_body.data[0].operation.blocks[0]
        if_body, else_body = while_body.data[1].operation.blocks
        case_0_body, case_default_body = while_body.data[2].operation.blocks

        # The setter should respect exact floating-point equality since the values are in the
        # interval [0, pi).
        self.assertEqual(
            [
                while_body.global_phase,
                for_body.global_phase,
                if_body.global_phase,
                else_body.global_phase,
                case_0_body.global_phase,
                case_default_body.global_phase,
            ],
            [i * math.pi / 7 for i in range(1, 7)],
        )

    def test_can_capture_input(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        base = QuantumCircuit(inputs=[a, b])
        with base.for_loop(range(3)):
            base.store(a, expr.lift(True))
        self.assertEqual(set(base.data[-1].operation.blocks[0].iter_captured_vars()), {a})

    def test_can_capture_declared(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        base = QuantumCircuit(declarations=[(a, expr.lift(False)), (b, expr.lift(True))])
        with base.if_test(expr.lift(False)):
            base.store(a, expr.lift(True))
        self.assertEqual(set(base.data[-1].operation.blocks[0].iter_captured_vars()), {a})

    def test_can_capture_capture(self):
        # It's a bit wild to be manually building an outer circuit that's intended to be a subblock,
        # but be using the control-flow builder interface internally, but eh, it should work.
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        base = QuantumCircuit(captures=[a, b])
        with base.while_loop(expr.lift(False)):
            base.store(a, expr.lift(True))
        self.assertEqual(set(base.data[-1].operation.blocks[0].iter_captured_vars()), {a})

    def test_can_capture_from_nested(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        c = expr.Var.new("c", types.Bool())
        base = QuantumCircuit(inputs=[a, b])
        with base.switch(expr.lift(False)) as case, case(case.DEFAULT):
            base.add_var(c, expr.lift(False))
            with base.if_test(expr.lift(False)):
                base.store(a, c)
        outer_block = base.data[-1].operation.blocks[0]
        inner_block = outer_block.data[-1].operation.blocks[0]
        self.assertEqual(set(inner_block.iter_captured_vars()), {a, c})

        # The containing block should have captured it as well, despite not using it explicitly.
        self.assertEqual(set(outer_block.iter_captured_vars()), {a})
        self.assertEqual(set(outer_block.iter_declared_vars()), {c})

    def test_can_manually_capture(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        base = QuantumCircuit(inputs=[a, b])
        with base.while_loop(expr.lift(False)):
            # Why do this?  Who knows, but it clearly has a well-defined meaning.
            base.add_capture(a)
        self.assertEqual(set(base.data[-1].operation.blocks[0].iter_captured_vars()), {a})

    def test_later_blocks_do_not_inherit_captures(self):
        """Neither 'if' nor 'switch' should have later blocks inherit the captures from the earlier
        blocks, and the earlier blocks shouldn't be affected by later ones."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        c = expr.Var.new("c", types.Bool())

        base = QuantumCircuit(inputs=[a, b, c])
        with base.if_test(expr.lift(False)) as else_:
            base.store(a, expr.lift(False))
        with else_:
            base.store(b, expr.lift(False))
        blocks = base.data[-1].operation.blocks
        self.assertEqual(set(blocks[0].iter_captured_vars()), {a})
        self.assertEqual(set(blocks[1].iter_captured_vars()), {b})

        base = QuantumCircuit(inputs=[a, b, c])
        with base.switch(expr.lift(False)) as case:
            with case(0):
                base.store(a, expr.lift(False))
            with case(case.DEFAULT):
                base.store(b, expr.lift(False))
        blocks = base.data[-1].operation.blocks
        self.assertEqual(set(blocks[0].iter_captured_vars()), {a})
        self.assertEqual(set(blocks[1].iter_captured_vars()), {b})

    def test_blocks_have_independent_declarations(self):
        """The blocks of if and switch should be separate scopes for declarations."""
        b1 = expr.Var.new("b", types.Bool())
        b2 = expr.Var.new("b", types.Bool())
        self.assertNotEqual(b1, b2)

        base = QuantumCircuit()
        with base.if_test(expr.lift(False)) as else_:
            base.add_var(b1, expr.lift(False))
        with else_:
            base.add_var(b2, expr.lift(False))
        blocks = base.data[-1].operation.blocks
        self.assertEqual(set(blocks[0].iter_declared_vars()), {b1})
        self.assertEqual(set(blocks[1].iter_declared_vars()), {b2})

        base = QuantumCircuit()
        with base.switch(expr.lift(False)) as case:
            with case(0):
                base.add_var(b1, expr.lift(False))
            with case(case.DEFAULT):
                base.add_var(b2, expr.lift(False))
        blocks = base.data[-1].operation.blocks
        self.assertEqual(set(blocks[0].iter_declared_vars()), {b1})
        self.assertEqual(set(blocks[1].iter_declared_vars()), {b2})

    def test_can_shadow_outer_name(self):
        outer = expr.Var.new("a", types.Bool())
        inner = expr.Var.new("a", types.Bool())
        base = QuantumCircuit(inputs=[outer])
        with base.if_test(expr.lift(False)):
            base.add_var(inner, expr.lift(True))
        block = base.data[-1].operation.blocks[0]
        self.assertEqual(set(block.iter_declared_vars()), {inner})
        self.assertEqual(set(block.iter_captured_vars()), set())

    def test_iterators_run_over_scope(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        c = expr.Var.new("c", types.Bool())
        d = expr.Var.new("d", types.Bool())

        base = QuantumCircuit(inputs=[a, b, c])
        self.assertEqual(set(base.iter_input_vars()), {a, b, c})
        self.assertEqual(set(base.iter_declared_vars()), set())
        self.assertEqual(set(base.iter_captured_vars()), set())

        with base.switch(expr.lift(3)) as case:
            with case(0):
                # Nothing here.
                self.assertEqual(set(base.iter_vars()), set())
                self.assertEqual(set(base.iter_input_vars()), set())
                self.assertEqual(set(base.iter_declared_vars()), set())
                self.assertEqual(set(base.iter_captured_vars()), set())

                # Capture a variable.
                base.store(a, expr.lift(False))
                self.assertEqual(set(base.iter_captured_vars()), {a})

                # Declare a variable.
                base.add_var(d, expr.lift(False))
                self.assertEqual(set(base.iter_declared_vars()), {d})
                self.assertEqual(set(base.iter_vars()), {a, d})

            with case(1):
                # We should have reset.
                self.assertEqual(set(base.iter_vars()), set())
                self.assertEqual(set(base.iter_input_vars()), set())
                self.assertEqual(set(base.iter_declared_vars()), set())
                self.assertEqual(set(base.iter_captured_vars()), set())

                # Capture a variable.
                base.store(b, expr.lift(False))
                self.assertEqual(set(base.iter_captured_vars()), {b})

                # Capture some more in another scope.
                with base.while_loop(expr.lift(False)):
                    self.assertEqual(set(base.iter_vars()), set())
                    base.store(c, expr.lift(False))
                    self.assertEqual(set(base.iter_captured_vars()), {c})

                self.assertEqual(set(base.iter_captured_vars()), {b, c})
                self.assertEqual(set(base.iter_vars()), {b, c})
        # And back to the outer scope.
        self.assertEqual(set(base.iter_input_vars()), {a, b, c})
        self.assertEqual(set(base.iter_declared_vars()), set())
        self.assertEqual(set(base.iter_captured_vars()), set())

    def test_get_var_respects_scope(self):
        outer = expr.Var.new("a", types.Bool())
        inner = expr.Var.new("a", types.Bool())
        base = QuantumCircuit(inputs=[outer])
        self.assertEqual(base.get_var("a"), outer)
        with base.if_test(expr.lift(False)) as else_:
            # Before we've done anything, getting the variable should get the outer one.
            self.assertEqual(base.get_var("a"), outer)

            # If we shadow it, we should get the shadowed one after.
            base.add_var(inner, expr.lift(False))
            self.assertEqual(base.get_var("a"), inner)
        with else_:
            # In a new scope, we should see the outer one again.
            self.assertEqual(base.get_var("a"), outer)
            # ... until we shadow it.
            base.add_var(inner, expr.lift(False))
            self.assertEqual(base.get_var("a"), inner)
        self.assertEqual(base.get_var("a"), outer)

    def test_has_var_respects_scope(self):
        outer = expr.Var.new("a", types.Bool())
        inner = expr.Var.new("a", types.Bool())
        base = QuantumCircuit(inputs=[outer])
        self.assertEqual(base.get_var("a"), outer)
        with base.if_test(expr.lift(False)) as else_:
            self.assertFalse(base.has_var("b"))

            # Before we've done anything, we should see the outer one.
            self.assertTrue(base.has_var("a"))
            self.assertTrue(base.has_var(outer))
            self.assertFalse(base.has_var(inner))

            # If we shadow it, we should see the shadowed one after.
            base.add_var(inner, expr.lift(False))
            self.assertTrue(base.has_var("a"))
            self.assertFalse(base.has_var(outer))
            self.assertTrue(base.has_var(inner))
        with else_:
            # In a new scope, we should see the outer one again.
            self.assertTrue(base.has_var("a"))
            self.assertTrue(base.has_var(outer))
            self.assertFalse(base.has_var(inner))

            # ... until we shadow it.
            base.add_var(inner, expr.lift(False))
            self.assertTrue(base.has_var("a"))
            self.assertFalse(base.has_var(outer))
            self.assertTrue(base.has_var(inner))

        self.assertTrue(base.has_var("a"))
        self.assertTrue(base.has_var(outer))
        self.assertFalse(base.has_var(inner))

    def test_store_to_clbit_captures_bit(self):
        base = QuantumCircuit(1, 2)
        with base.if_test(expr.lift(False)):
            base.store(expr.lift(base.clbits[0]), expr.lift(True))

        expected = QuantumCircuit(1, 2)
        body = QuantumCircuit([expected.clbits[0]])
        body.store(expr.lift(expected.clbits[0]), expr.lift(True))
        expected.if_test(expr.lift(False), body, [], [0])

        self.assertEqual(base, expected)

    def test_store_to_register_captures_register(self):
        cr1 = ClassicalRegister(2, "cr1")
        cr2 = ClassicalRegister(2, "cr2")
        base = QuantumCircuit(cr1, cr2)
        with base.if_test(expr.lift(False)):
            base.store(expr.lift(cr1), expr.lift(3))

        body = QuantumCircuit(cr1)
        body.store(expr.lift(cr1), expr.lift(3))
        expected = QuantumCircuit(cr1, cr2)
        expected.if_test(expr.lift(False), body, [], cr1[:])

        self.assertEqual(base, expected)

    def test_rebuild_captures_variables_in_blocks(self):
        """Test that when the separate blocks of a statement cause it to require a full rebuild of
        the circuit objects during builder resolution, the variables are all moved over
        correctly."""

        a = expr.Var.new("🐍🐍🐍", types.Uint(8))

        qc = QuantumCircuit(3, 1, inputs=[a])
        qc.measure(0, 0)
        b_outer = qc.add_var("b", False)
        with qc.switch(a) as case:
            with case(0):
                qc.cx(1, 2)
                qc.store(b_outer, True)
            with case(1):
                qc.store(qc.clbits[0], False)
            with case(2):
                # Explicit shadowing.
                b_inner = qc.add_var("b", True)
            with case(3):
                qc.store(a, expr.lift(1, a.type))
            with case(case.DEFAULT):
                qc.cx(2, 1)

        # (inputs, captures, declares) for each block of the `switch`.
        expected = [
            ([], [b_outer], []),
            ([], [], []),
            ([], [], [b_inner]),
            ([], [a], []),
            ([], [], []),
        ]
        actual = [
            (
                list(block.iter_input_vars()),
                list(block.iter_captured_vars()),
                list(block.iter_declared_vars()),
            )
            for block in qc.data[-1].operation.blocks
        ]
        self.assertEqual(expected, actual)


@ddt.ddt
class TestControlFlowBuildersFailurePaths(QiskitTestCase):
    """Tests for the failure paths of the control-flow builders."""

    def test_if_rejects_break_continue_if_not_in_loop(self):
        """Test that the ``if`` and ``else`` context managers raise a suitable exception if you try
        to use a ``break`` or ``continue`` within them without being inside a loop.  This is for
        safety; without the loop context, the context manager will cause the wrong resources to be
        assigned to the ``break``, so if you want to make a manual loop, you have to use manual
        ``if`` as well.  That way the onus is on you."""
        qubits = [Qubit()]
        clbits = [Clbit()]
        cond = (clbits[0], 0)

        message = r"The current builder scope cannot take a '.*' because it is not in a loop\."

        with self.subTest("if break"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond):
                with self.assertRaisesRegex(CircuitError, message):
                    test.break_loop()

        with self.subTest("if continue"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond):
                with self.assertRaisesRegex(CircuitError, message):
                    test.continue_loop()

        with self.subTest("else break"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                with self.assertRaisesRegex(CircuitError, message):
                    test.break_loop()

        with self.subTest("else continue"):
            test = QuantumCircuit(qubits, clbits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                with self.assertRaisesRegex(CircuitError, message):
                    test.continue_loop()

    def test_for_rejects_reentry(self):
        """Test that the ``for``-loop context manager rejects attempts to re-enter it.  Since it
        holds some forms of state during execution (the loop variable, which may be generated), we
        can't safely re-enter it and get the expected behavior."""

        for_manager = QuantumCircuit(2, 2).for_loop(range(2))
        with for_manager:
            pass
        with self.assertRaisesRegex(
            CircuitError, r"A for-loop context manager cannot be re-entered."
        ):
            with for_manager:
                pass

    def test_cannot_enter_else_context_incorrectly(self):
        """Test that various forms of using an 'else_' context manager incorrectly raise
        exceptions."""
        bits = [Qubit(), Clbit()]
        cond = (bits[1], 0)

        with self.subTest("not the next instruction"):
            test = QuantumCircuit(bits)
            with test.if_test(cond) as else_:
                pass
            test.h(0)
            with self.assertRaisesRegex(CircuitError, "The 'if' block is not the most recent"):
                with else_:
                    test.h(0)

        with self.subTest("inside the attached if"):
            test = QuantumCircuit(bits)
            with test.if_test(cond) as else_:
                with self.assertRaisesRegex(
                    CircuitError, r"Cannot attach an 'else' branch to an incomplete 'if' block\."
                ):
                    with else_:
                        test.h(0)

        with self.subTest("inner else"):
            test = QuantumCircuit(bits)
            with test.if_test(cond) as else1:
                with test.if_test(cond):
                    pass
                with self.assertRaisesRegex(
                    CircuitError, r"Cannot attach an 'else' branch to an incomplete 'if' block\."
                ):
                    with else1:
                        test.h(0)

        with self.subTest("reused else"):
            test = QuantumCircuit(bits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                pass
            with self.assertRaisesRegex(CircuitError, r"Cannot re-use an 'else' context\."):
                with else_:
                    pass

        with self.subTest("else from an inner block"):
            test = QuantumCircuit(bits)
            with test.if_test(cond):
                with test.if_test(cond) as else_:
                    pass
            with self.assertRaisesRegex(CircuitError, "The 'if' block is not the most recent"):
                with else_:
                    pass

    def test_if_placeholder_rejects_c_if(self):
        """Test that the :obj:`.IfElsePlaceholder" class rejects attempts to use
        :meth:`.Instruction.c_if` on it.

        It *should* be the case that you need to use private methods to get access to one of these
        placeholder objects at all, because they're appended to a scope at the exit of a context
        manager, so not returned from a method call. Just in case, here's a test that it correctly
        rejects the dangerous method that can overwrite ``condition``.
        """
        bits = [Qubit(), Clbit()]

        with self.subTest("if"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.if_test((bits[1], 0)):
                    test.break_loop()
                # These tests need to be done before the 'for' context exits so we don't trigger the
                # "can't add conditions from out-of-scope" handling.
                placeholder = test._peek_previous_instruction_in_scope().operation
                self.assertIsInstance(placeholder, IfElsePlaceholder)
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"IfElseOp cannot be classically controlled through Instruction\.c_if",
                ):
                    with self.assertWarns(DeprecationWarning):
                        placeholder.c_if(bits[1], 0)

        with self.subTest("else"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.if_test((bits[1], 0)) as else_:
                    pass
                with else_:
                    test.break_loop()
                # These tests need to be done before the 'for' context exits so we don't trigger the
                # "can't add conditions from out-of-scope" handling.
                placeholder = test._peek_previous_instruction_in_scope().operation
                self.assertIsInstance(placeholder, IfElsePlaceholder)
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"IfElseOp cannot be classically controlled through Instruction\.c_if",
                ):
                    with self.assertWarns(DeprecationWarning):
                        placeholder.c_if(bits[1], 0)

    def test_switch_rejects_operations_outside_cases(self):
        """It shouldn't be permissible to try and put instructions inside a switch but outside a
        case."""
        circuit = QuantumCircuit(1, 1)
        with circuit.switch(0) as case:
            with case(0):
                pass
            with self.assertRaisesRegex(CircuitError, r"Cannot have instructions outside a case"):
                circuit.x(0)

    def test_switch_rejects_entering_case_after_close(self):
        """It shouldn't be possible to enter a case within another case."""
        circuit = QuantumCircuit(1, 1)
        with circuit.switch(0) as case, case(0):
            pass
        with self.assertRaisesRegex(CircuitError, r"Cannot add .* to a completed switch"), case(1):
            pass

    def test_switch_rejects_reentering_case(self):
        """It shouldn't be possible to enter a case within another case."""
        circuit = QuantumCircuit(1, 1)
        with (
            circuit.switch(0) as case,
            case(0),
            self.assertRaisesRegex(CircuitError, r"Cannot enter more than one case at once"),
            case(1),
        ):
            pass

    @ddt.data("1", 1.0, None, (1, 2))
    def test_switch_rejects_bad_case_value(self, value):
        """Only well-typed values should be accepted."""
        circuit = QuantumCircuit(1, 1)
        with circuit.switch(0) as case:
            with case(0):
                pass
            with self.assertRaisesRegex(CircuitError, "Case values must be"), case(value):
                pass

    def test_case_rejects_duplicate_labels(self):
        """Using duplicates in the same `case` should raise an error."""
        circuit = QuantumCircuit(1, 2)
        with circuit.switch(circuit.cregs[0]) as case:
            with case(0):
                pass
            with self.assertRaisesRegex(CircuitError, "duplicate"), case(1, 1):
                pass
            with self.assertRaisesRegex(CircuitError, "duplicate"), case(1, 2, 3, 1):
                pass

    def test_switch_rejects_duplicate_labels(self):
        """Using duplicates in different `case`s should raise an error."""
        circuit = QuantumCircuit(1, 2)
        with circuit.switch(circuit.cregs[0]) as case:
            with case(0):
                pass
            with case(1):
                pass
            with self.assertRaisesRegex(CircuitError, "duplicate"), case(1):
                pass

    def test_switch_accepts_label_after_failure(self):
        """If one case causes an exception that's caught, subsequent cases should still be possible
        using labels that were "used" by the failing case."""
        qreg = QuantumRegister(1, "q")
        creg = ClassicalRegister(2, "c")

        test = QuantumCircuit(qreg, creg)
        with test.switch(creg) as case:
            with case(0):
                pass
            # assertRaises here is an extra test that the exception is propagated through the
            # context manager, and acts as an `except` clause for the exception so control will
            # continue beyond.
            with self.assertRaises(SentinelException), case(1):
                raise SentinelException
            with case(1):
                test.x(0)

        expected = QuantumCircuit(qreg, creg)
        with expected.switch(creg) as case:
            with case(0):
                pass
            with case(1):
                expected.x(0)
        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_reject_c_if_from_outside_scope(self):
        """Test that the context managers reject :meth:`.InstructionSet.c_if` calls if they occur
        after their scope has completed."""
        bits = [Qubit(), Clbit()]
        cond = (bits[1], 0)

        with self.subTest("if"):
            test = QuantumCircuit(bits)
            with test.if_test(cond):
                instructions = test.h(0)
            with self.assertRaisesRegex(
                CircuitError, r"Cannot add resources after the scope has been built\."
            ):
                with self.assertWarns(DeprecationWarning):
                    instructions.c_if(*cond)

        with self.subTest("else"):
            test = QuantumCircuit(bits)
            with test.if_test(cond) as else_:
                pass
            with else_:
                instructions = test.h(0)
            with self.assertRaisesRegex(
                CircuitError, r"Cannot add resources after the scope has been built\."
            ):
                with self.assertWarns(DeprecationWarning):
                    instructions.c_if(*cond)

        with self.subTest("for"):
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                instructions = test.h(0)
            with self.assertRaisesRegex(
                CircuitError, r"Cannot add resources after the scope has been built\."
            ):
                with self.assertWarns(DeprecationWarning):
                    instructions.c_if(*cond)

        with self.subTest("while"):
            test = QuantumCircuit(bits)
            with test.while_loop(cond):
                instructions = test.h(0)
            with self.assertRaisesRegex(
                CircuitError, r"Cannot add resources after the scope has been built\."
            ):
                with self.assertWarns(DeprecationWarning):
                    instructions.c_if(*cond)

        with self.subTest("switch"):
            test = QuantumCircuit(bits)
            with test.switch(bits[1]) as case, case(0):
                instructions = test.h(0)
            with self.assertRaisesRegex(
                CircuitError, r"Cannot add resources after the scope has been built\."
            ):
                with self.assertWarns(DeprecationWarning):
                    instructions.c_if(*cond)

        with self.subTest("if inside for"):
            # As a side-effect of how the lazy building of 'if' statements works, we actually
            # *could* add a condition to the gate after the 'if' block as long as we were still
            # within the 'for' loop.  It should actually manage the resource correctly as well, but
            # it's "undefined behavior" than something we specifically want to forbid or allow.
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.if_test(cond):
                    instructions = test.h(0)
            with self.assertRaisesRegex(
                CircuitError, r"Cannot add resources after the scope has been built\."
            ):
                with self.assertWarns(DeprecationWarning):
                    instructions.c_if(*cond)

        with self.subTest("switch inside for"):
            # `switch` has the same lazy building as `if`, so is subject to the same considerations
            # as the above subtest.
            test = QuantumCircuit(bits)
            with test.for_loop(range(2)):
                with test.switch(bits[1]) as case:
                    with case(0):
                        instructions = test.h(0)
            with self.assertRaisesRegex(
                CircuitError, r"Cannot add resources after the scope has been built\."
            ):
                with self.assertWarns(DeprecationWarning):
                    instructions.c_if(*cond)

    def test_raising_inside_context_manager_leave_circuit_usable(self):
        """Test that if we leave a builder by raising some sort of exception, the circuit is left in
        a usable state, and extra resources have not been added to the circuit."""

        x, y = Parameter("x"), Parameter("y")

        with self.subTest("for"):
            test = QuantumCircuit(1, 1)
            test.h(0)
            with self.assertRaises(SentinelException):
                with test.for_loop(range(2), x) as bound_x:
                    test.x(0)
                    test.rx(bound_x, 0)
                    test.ry(y, 0)
                    raise SentinelException
            test.z(0)

            expected = QuantumCircuit(1, 1)
            expected.h(0)
            expected.z(0)

            self.assertEqual(test, expected)
            # We don't want _either_ the loop variable or the loose variable to be in the circuit.
            self.assertEqual(set(), set(test.parameters))

        with self.subTest("while"):
            bits = [Qubit(), Clbit()]
            test = QuantumCircuit(bits)
            test.h(0)
            with self.assertRaises(SentinelException):
                with test.while_loop((bits[1], 0)):
                    test.x(0)
                    test.rx(x, 0)
                    raise SentinelException
            test.z(0)

            expected = QuantumCircuit(bits)
            expected.h(0)
            expected.z(0)

            self.assertEqual(test, expected)
            self.assertEqual(set(), set(test.parameters))

        with self.subTest("if"):
            bits = [Qubit(), Clbit()]
            test = QuantumCircuit(bits)
            test.h(0)
            with self.assertRaises(SentinelException):
                with test.if_test((bits[1], 0)):
                    test.x(0)
                    test.rx(x, 0)
                    raise SentinelException
            test.z(0)

            expected = QuantumCircuit(bits)
            expected.h(0)
            expected.z(0)

            self.assertEqual(test, expected)
            self.assertEqual(set(), set(test.parameters))

        with self.subTest("else"):
            bits = [Qubit(), Clbit()]
            test = QuantumCircuit(bits)
            test.h(0)
            with test.if_test((bits[1], 0)) as else_:
                test.rx(x, 0)
            with self.assertRaises(SentinelException):
                with else_:
                    test.x(0)
                    test.rx(y, 0)
                    raise SentinelException
            test.z(0)

            # Note that we expect the "else" manager to restore the "if" block if something errors
            # out during "else" block.
            true_body = QuantumCircuit(bits)
            true_body.rx(x, 0)
            expected = QuantumCircuit(bits)
            expected.h(0)
            expected.if_test((bits[1], 0), true_body, [0], [0])
            expected.z(0)

            self.assertEqual(test, expected)
            self.assertEqual({x}, set(test.parameters))

        with self.subTest("switch"):
            test = QuantumCircuit(1, 1)
            with self.assertRaises(SentinelException), test.switch(0) as case:
                with case(False):
                    pass
                with case(True):
                    pass
                raise SentinelException
            test.h(0)
            expected = QuantumCircuit(1, 1)
            expected.h(0)
            self.assertEqual(test, expected)

    def test_can_reuse_else_manager_after_exception(self):
        """Test that the "else" context manager is usable after a first attempt to construct it
        raises an exception.  Normally you cannot re-enter an "else" block, but we want the user to
        be able to recover from errors if they so try."""
        bits = [Qubit(), Clbit()]
        test = QuantumCircuit(bits)
        test.h(0)
        with test.if_test((bits[1], 0)) as else_:
            test.x(0)
        with self.assertRaises(SentinelException):
            with else_:
                test.y(0)
                raise SentinelException
        with else_:
            test.h(0)
        test.z(0)

        true_body = QuantumCircuit(bits)
        true_body.x(0)
        false_body = QuantumCircuit(bits)
        false_body.h(0)
        expected = QuantumCircuit(bits)
        expected.h(0)
        expected.if_else((bits[1], 0), true_body, false_body, [0], [0])
        expected.z(0)

        self.assertEqual(test, expected)

    @ddt.data((None, [0]), ([0], None), ([0], [0]))
    def test_context_managers_reject_passing_qubits(self, resources):
        """Test that the context-manager forms of the control-flow circuit methods raise exceptions
        if they are given explicit qubits or clbits."""
        test = QuantumCircuit(1, 1)
        qubits, clbits = resources
        with self.subTest("for"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'for_loop' as a context manager, you cannot pass qubits or clbits\.",
            ):
                test.for_loop(range(2), None, body=None, qubits=qubits, clbits=clbits)
        with self.subTest("while"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'while_loop' as a context manager, you cannot pass qubits or clbits\.",
            ):
                test.while_loop((test.clbits[0], 0), body=None, qubits=qubits, clbits=clbits)
        with self.subTest("if"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'if_test' as a context manager, you cannot pass qubits or clbits\.",
            ):
                test.if_test((test.clbits[0], 0), true_body=None, qubits=qubits, clbits=clbits)
        with self.subTest("switch"):
            with self.assertRaisesRegex(CircuitError, r"When using 'switch' as a context manager"):
                test.switch(test.clbits[0], cases=None, qubits=qubits, clbits=clbits)

    @ddt.data((None, [0]), ([0], None), (None, None))
    def test_non_context_manager_calling_states_reject_missing_resources(self, resources):
        """Test that the non-context-manager forms of the control-flow circuit methods raise
        exceptions if they are not given explicit qubits or clbits."""
        test = QuantumCircuit(1, 1)
        body = QuantumCircuit(1, 1)
        qubits, clbits = resources
        with self.subTest("for"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'for_loop' with a body, you must pass qubits and clbits\.",
            ):
                test.for_loop(range(2), None, body=body, qubits=qubits, clbits=clbits)
        with self.subTest("while"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'while_loop' with a body, you must pass qubits and clbits\.",
            ):
                test.while_loop((test.clbits[0], 0), body=body, qubits=qubits, clbits=clbits)
        with self.subTest("if"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'if_test' with a body, you must pass qubits and clbits\.",
            ):
                test.if_test((test.clbits[0], 0), true_body=body, qubits=qubits, clbits=clbits)
        with self.subTest("switch"):
            with self.assertRaisesRegex(
                CircuitError,
                r"When using 'switch' with cases, you must pass qubits and clbits\.",
            ):
                test.switch(test.clbits[0], [(False, body)], qubits=qubits, clbits=clbits)

    def test_compose_front_inplace_invalid_within_builder(self):
        """Test that `QuantumCircuit.compose` raises a sensible error when called within a
        control-flow builder block."""
        inner = QuantumCircuit(1)
        inner.x(0)

        outer = QuantumCircuit(1, 1)
        outer.measure(0, 0)
        outer.compose(inner, front=True, inplace=True)
        with outer.if_test((outer.clbits[0], 1)):
            with self.assertRaisesRegex(CircuitError, r"Cannot compose to the front.*"):
                outer.compose(inner, front=True, inplace=True)

    def test_compose_new_invalid_within_builder(self):
        """Test that `QuantumCircuit.compose` raises a sensible error when called within a
        control-flow builder block if trying to emit a new circuit."""
        inner = QuantumCircuit(1)
        inner.x(0)

        outer = QuantumCircuit(1, 1)
        outer.measure(0, 0)
        with outer.if_test((outer.clbits[0], 1)):
            with self.assertRaisesRegex(CircuitError, r"Cannot emit a new composed circuit.*"):
                outer.compose(inner, inplace=False)

    def test_cannot_capture_variable_not_in_scope(self):
        a = expr.Var.new("a", types.Bool())

        base = QuantumCircuit(1, 1)
        with base.if_test((0, True)) as else_, self.assertRaisesRegex(CircuitError, "not in scope"):
            base.store(a, expr.lift(False))
        with else_, self.assertRaisesRegex(CircuitError, "not in scope"):
            base.store(a, expr.lift(False))

        base.add_input(a)
        with base.while_loop((0, True)), self.assertRaisesRegex(CircuitError, "not in scope"):
            base.store(expr.Var.new("a", types.Bool()), expr.lift(False))

        with base.for_loop(range(3)):
            with base.switch(base.clbits[0]) as case, case(0):
                with self.assertRaisesRegex(CircuitError, "not in scope"):
                    base.store(expr.Var.new("a", types.Bool()), expr.lift(False))

    def test_cannot_add_existing_variable(self):
        a = expr.Var.new("a", types.Bool())
        base = QuantumCircuit()
        with base.if_test(expr.lift(False)) as else_:
            base.add_var(a, expr.lift(False))
            with self.assertRaisesRegex(CircuitError, "already present"):
                base.add_var(a, expr.lift(False))
        with else_:
            base.add_var(a, expr.lift(False))
            with self.assertRaisesRegex(CircuitError, "already present"):
                base.add_var(a, expr.lift(False))

    def test_cannot_shadow_in_same_scope(self):
        a = expr.Var.new("a", types.Bool())
        base = QuantumCircuit()
        with base.switch(expr.lift(3)) as case:
            with case(0):
                base.add_var(a, expr.lift(False))
                with self.assertRaisesRegex(CircuitError, "its name shadows"):
                    base.add_var(a.name, expr.lift(False))
            with case(case.DEFAULT):
                base.add_var(a, expr.lift(False))
                with self.assertRaisesRegex(CircuitError, "its name shadows"):
                    base.add_var(a.name, expr.lift(False))

    def test_cannot_shadow_captured_variable(self):
        """It shouldn't be possible to shadow a variable that has already been captured into the
        block."""
        outer = expr.Var.new("a", types.Bool())
        inner = expr.Var.new("a", types.Bool())

        base = QuantumCircuit(inputs=[outer])
        with base.while_loop(expr.lift(True)):
            # Capture the outer.
            base.store(outer, expr.lift(True))
            # Attempt to shadow it.
            with self.assertRaisesRegex(CircuitError, "its name shadows"):
                base.add_var(inner, expr.lift(False))

    def test_cannot_use_outer_variable_after_shadow(self):
        """If we've shadowed a variable, the outer one shouldn't be visible to us for use."""
        outer = expr.Var.new("a", types.Bool())
        inner = expr.Var.new("a", types.Bool())

        base = QuantumCircuit(inputs=[outer])
        with base.for_loop(range(3)):
            # Shadow the outer.
            base.add_var(inner, expr.lift(False))
            with self.assertRaisesRegex(CircuitError, "cannot use.*shadowed"):
                base.store(outer, expr.lift(True))

    def test_cannot_use_beyond_outer_shadow(self):
        outer = expr.Var.new("a", types.Bool())
        inner = expr.Var.new("a", types.Bool())
        base = QuantumCircuit(inputs=[outer])
        with base.while_loop(expr.lift(True)):
            # Shadow 'outer'
            base.add_var(inner, expr.lift(True))
            with base.switch(expr.lift(3)) as case, case(0):
                with self.assertRaisesRegex(CircuitError, "not in scope"):
                    # Attempt to access the shadowed variable.
                    base.store(outer, expr.lift(False))

    def test_exception_during_initialisation_does_not_add_variable(self):
        uint_var = expr.Var.new("a", types.Uint(16))
        bool_expr = expr.Value(False, types.Bool())
        with self.assertRaises(CircuitError):
            Store(uint_var, bool_expr)
        base = QuantumCircuit()
        with base.while_loop(expr.lift(False)):
            # Should succeed.
            b = base.add_var("b", expr.lift(False))
            try:
                base.add_var(uint_var, bool_expr)
            except CircuitError:
                pass
            # Should succeed.
            c = base.add_var("c", expr.lift(False))
            local_vars = set(base.iter_vars())
        self.assertEqual(local_vars, {b, c})

    def test_cannot_use_old_var_not_in_circuit(self):
        base = QuantumCircuit()
        with base.if_test(expr.lift(False)) as else_:
            with self.assertRaisesRegex(CircuitError, "not present"):
                base.store(expr.lift(Clbit()), expr.lift(False))
        with else_:
            with self.assertRaisesRegex(CircuitError, "not present"):
                with base.if_test(expr.equal(ClassicalRegister(2, "c"), 3)):
                    pass

    def test_cannot_add_input_in_scope(self):
        base = QuantumCircuit()
        with base.for_loop(range(3)):
            with self.assertRaisesRegex(CircuitError, "cannot add an input variable"):
                base.add_input("a", types.Bool())

    def test_cannot_add_uninitialized_in_scope(self):
        base = QuantumCircuit()
        with base.for_loop(range(3)):
            with self.assertRaisesRegex(CircuitError, "cannot add an uninitialized variable"):
                base.add_uninitialized_var(expr.Var.new("a", types.Bool()))

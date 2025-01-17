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

"""Parser Test."""

from qiskit.pulse.parser import parse_string_expr
from qiskit.pulse.exceptions import PulseError
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestInstructionToQobjConverter(QiskitTestCase):
    """Expression parser test."""

    def test_valid_expression1(self):
        """Parsing valid expression."""

        expr = "1+1*2*3.2+8*cos(0)**2"
        parsed_expr = parse_string_expr(expr)

        self.assertEqual(parsed_expr.params, [])
        self.assertEqual(parsed_expr(), 15.4 + 0j)

    def test_valid_expression2(self):
        """Parsing valid expression."""

        expr = "pi*2"
        parsed_expr = parse_string_expr(expr)

        self.assertEqual(parsed_expr.params, [])
        self.assertEqual(parsed_expr(), 6.283185307179586 + 0j)

    def test_valid_expression3(self):
        """Parsing valid expression."""

        expr = "-P1*cos(P2)"
        parsed_expr = parse_string_expr(expr)

        self.assertEqual(parsed_expr.params, ["P1", "P2"])
        self.assertEqual(parsed_expr(P1=1.0, P2=2.0), 0.4161468365471424 + 0j)

    def test_valid_expression4(self):
        """Parsing valid expression."""

        expr = "-P1*P2*P3"
        parsed_expr = parse_string_expr(expr)

        self.assertEqual(parsed_expr.params, ["P1", "P2", "P3"])
        self.assertEqual(parsed_expr(P1=1.0, P2=2.0, P3=3.0), -6.0 + 0j)

    def test_valid_expression5(self):
        """Parsing valid expression."""

        expr = "-(P1)"
        parsed_expr = parse_string_expr(expr)

        self.assertEqual(parsed_expr.params, ["P1"])
        self.assertEqual(parsed_expr(P1=1.0), -1.0 + 0j)

    def test_valid_expression6(self):
        """Parsing valid expression."""

        expr = "-1.*P1"
        parsed_expr = parse_string_expr(expr)

        self.assertEqual(parsed_expr.params, ["P1"])
        self.assertEqual(parsed_expr(P1=1.0), -1.0 + 0j)

    def test_valid_expression7(self):
        """Parsing valid expression."""

        expr = "-1.*P1*P2"
        parsed_expr = parse_string_expr(expr)

        self.assertEqual(parsed_expr.params, ["P1", "P2"])
        self.assertEqual(parsed_expr(P1=1.0, P2=2.0), -2.0 + 0j)

    def test_valid_expression8(self):
        """Parsing valid expression."""

        expr = "P3-P2*(4+P1)"
        parsed_expr = parse_string_expr(expr)

        self.assertEqual(parsed_expr.params, ["P1", "P2", "P3"])
        self.assertEqual(parsed_expr(P1=1, P2=2, P3=3), -7.0 + 0j)

    def test_invalid_expressions1(self):
        """Parsing invalid expressions."""

        expr = "2***2"
        with self.assertRaises(PulseError):
            parsed_expr = parse_string_expr(expr)
            parsed_expr()

    def test_invalid_expressions2(self):
        """Parsing invalid expressions."""

        expr = "avdfd*3"
        with self.assertRaises(PulseError):
            parsed_expr = parse_string_expr(expr)
            parsed_expr()

    def test_invalid_expressions3(self):
        """Parsing invalid expressions."""

        expr = "Cos(1+2)"
        with self.assertRaises(PulseError):
            parsed_expr = parse_string_expr(expr)
            parsed_expr()

    def test_invalid_expressions4(self):
        """Parsing invalid expressions."""

        expr = "hello_world"
        with self.assertRaises(PulseError):
            parsed_expr = parse_string_expr(expr)
            parsed_expr()

    def test_invalid_expressions5(self):
        """Parsing invalid expressions."""

        expr = "1.1.1.1"
        with self.assertRaises(PulseError):
            parsed_expr = parse_string_expr(expr)
            parsed_expr()

    def test_invalid_expressions6(self):
        """Parsing invalid expressions."""

        expr = "abc.1"
        with self.assertRaises(PulseError):
            parsed_expr = parse_string_expr(expr)
            parsed_expr()

    def test_malicious_expressions1(self):
        """Parsing malicious expressions."""

        expr = '__import__("sys").stdout.write("unsafe input.")'
        with self.assertRaises(PulseError):
            parsed_expr = parse_string_expr(expr)
            parsed_expr()

    def test_malicious_expressions2(self):
        """Parsing malicious expressions."""

        expr = "INSERT INTO students VALUES (?,?)"
        with self.assertRaises(PulseError):
            parsed_expr = parse_string_expr(expr)
            parsed_expr()

    def test_malicious_expressions3(self):
        """Parsing malicious expressions."""

        expr = "import math"
        with self.assertRaises(PulseError):
            parsed_expr = parse_string_expr(expr)
            parsed_expr()

    def test_malicious_expressions4(self):
        """Parsing malicious expressions."""

        expr = "complex"
        with self.assertRaises(PulseError):
            parsed_expr = parse_string_expr(expr)
            parsed_expr()

    def test_malicious_expressions5(self):
        """Parsing malicious expressions."""

        expr = "print(1.0)"
        with self.assertRaises(PulseError):
            parsed_expr = parse_string_expr(expr)
            parsed_expr()

    def test_malicious_expressions6(self):
        """Parsing malicious expressions."""

        expr = 'eval("()._" + "_class_" + "_._" +  "_bases_" + "_[0]")'
        with self.assertRaises(PulseError):
            parsed_expr = parse_string_expr(expr)
            parsed_expr()

    def test_partial_binding(self):
        """Test partial binding of parameters."""

        expr = "P1 * P2 + P3 / P4 - P5"

        parsed_expr = parse_string_expr(expr, partial_binding=True)
        self.assertEqual(parsed_expr.params, ["P1", "P2", "P3", "P4", "P5"])

        bound_three = parsed_expr(P1=1, P2=2, P3=3)
        self.assertEqual(bound_three.params, ["P4", "P5"])

        self.assertEqual(bound_three(P4=4, P5=5), -2.25)
        self.assertEqual(bound_three(4, 5), -2.25)

        bound_four = bound_three(P4=4)
        self.assertEqual(bound_four.params, ["P5"])
        self.assertEqual(bound_four(P5=5), -2.25)
        self.assertEqual(bound_four(5), -2.25)

        bound_four_new = bound_three(P4=40)
        self.assertEqual(bound_four_new.params, ["P5"])
        self.assertEqual(bound_four_new(P5=5), -2.925)
        self.assertEqual(bound_four_new(5), -2.925)

    def test_argument_duplication(self):
        """Test duplication of *args and **kwargs."""

        expr = "P1+P2"
        parsed_expr = parse_string_expr(expr, partial_binding=True)

        with self.assertRaises(PulseError):
            parsed_expr(1, P1=1)

        self.assertEqual(parsed_expr(1, P2=2), 3.0)

    def test_unexpected_argument(self):
        """Test unexpected argument error."""
        expr = "P1+P2"
        parsed_expr = parse_string_expr(expr, partial_binding=True)

        with self.assertRaises(PulseError):
            parsed_expr(1, 2, P3=3)

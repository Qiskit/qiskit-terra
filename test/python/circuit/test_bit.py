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

# pylint: disable=missing-function-docstring

"""Test library of quantum circuits."""
import copy
from unittest import mock

from qiskit.circuit import bit
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestBitClass(QiskitTestCase):
    """Test library of boolean logic quantum circuits."""

    def test_bit_eq_invalid_type_comparison(self):
        orig_reg = mock.MagicMock()
        orig_reg.size = 3
        test_bit = bit.Bit()
        self.assertNotEqual(test_bit, 3.14)


class TestNewStyleBit(QiskitTestCase):
    """Test behavior of new-style bits."""

    def test_bits_do_not_require_registers(self):
        """Verify we can create a bit outside the context of a register."""
        self.assertIsInstance(bit.Bit(), bit.Bit)

    def test_new_style_bit_deepcopy(self):
        """Verify deep-copies of bits are the same instance."""
        bit1 = bit.Bit()
        bit2 = copy.deepcopy(bit1)

        self.assertIs(bit1, bit2)

    def test_new_style_bit_copy(self):
        """Verify copies of bits are the same instance."""
        bit1 = bit.Bit()
        bit2 = copy.copy(bit1)

        self.assertIs(bit1, bit2)

    def test_new_style_bit_equality(self):
        """Verify bits instances are equal only to themselves."""
        bit1 = bit.Bit()
        bit2 = bit.Bit()

        self.assertEqual(bit1, bit1)
        self.assertNotEqual(bit1, bit2)
        self.assertNotEqual(bit1, 3.14)

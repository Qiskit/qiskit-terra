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

"""Tests the layout object"""

import copy
import unittest
import numpy as np

from qiskit.circuit import QuantumRegister, Qubit
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.exceptions import LayoutError
from qiskit.test import QiskitTestCase


class LayoutTest(QiskitTestCase):
    """Test the methods in the layout object."""

    def setUp(self):
        super().setUp()
        self.qr = QuantumRegister(3, "qr")

    def test_default_layout(self):
        """Static method generate_trivial_layout creates a Layout"""
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(2, "qr1")
        layout = Layout.generate_trivial_layout(qr0, qr1)

        self.assertEqual(layout[qr0[0]], 0)
        self.assertEqual(layout[qr0[1]], 1)
        self.assertEqual(layout[qr0[2]], 2)
        self.assertEqual(layout[qr1[0]], 3)
        self.assertEqual(layout[qr1[1]], 4)

    def test_layout_from_dict(self):
        """Constructor from a dict"""
        layout = Layout({self.qr[0]: 0, self.qr[1]: 1, self.qr[2]: 2})

        self.assertEqual(layout[self.qr[0]], 0)
        self.assertEqual(layout[self.qr[1]], 1)
        self.assertEqual(layout[self.qr[2]], 2)
        self.assertEqual(layout[0], self.qr[0])
        self.assertEqual(layout[1], self.qr[1])
        self.assertEqual(layout[2], self.qr[2])

    def test_layout_from_dict_hole(self):
        """Constructor from a dict with a hole"""
        qr0 = QuantumRegister(2)
        qr1 = QuantumRegister(2)

        layout = Layout({qr0[0]: 0, qr1[0]: 1, qr1[1]: 3, qr0[1]: 4})

        self.assertEqual(layout[qr0[0]], 0)
        self.assertEqual(layout[qr1[0]], 1)
        with self.assertRaises(KeyError):
            _ = layout[None]
        self.assertEqual(layout[qr1[1]], 3)
        self.assertEqual(layout[qr0[1]], 4)

        self.assertEqual(layout[0], qr0[0])
        self.assertEqual(layout[1], qr1[0])
        self.assertEqual(layout[3], qr1[1])
        self.assertEqual(layout[4], qr0[1])

    def test_layout_set(self):
        """Setter"""
        layout = Layout()
        layout[self.qr[0]] = 0
        self.assertEqual(layout[self.qr[0]], 0)
        self.assertEqual(layout[0], self.qr[0])

    def test_layout_del(self):
        """Deleter"""
        layout = Layout()
        layout[self.qr[0]] = 0
        del layout[self.qr[0]]
        self.assertTrue(self.qr[0] not in layout)

    def test_layout_avoid_dangling_physical(self):
        """No dangling pointers for physical qubits."""
        layout = Layout({self.qr[0]: 0})
        self.assertEqual(layout[0], self.qr[0])
        layout[self.qr[0]] = 1
        with self.assertRaises(KeyError):
            _ = layout[0]

    def test_layout_avoid_dangling_virtual(self):
        """No dangling pointers for virtual qubits."""
        layout = Layout({self.qr[0]: 0})
        self.assertEqual(layout[0], self.qr[0])
        layout[0] = self.qr[1]
        with self.assertRaises(KeyError):
            _ = layout[self.qr[0]]

    def test_layout_len(self):
        """Length of the layout is the amount of physical bits"""
        layout = Layout()
        self.assertEqual(len(layout), 0)
        layout.add(self.qr[2])
        self.assertEqual(len(layout), 1)
        layout.add(self.qr[1], 3)
        self.assertEqual(len(layout), 2)

    def test_layout_len_with_idle(self):
        """Length of the layout is the amount of physical bits"""
        layout = Layout()
        self.assertEqual(len(layout), 0)
        layout.add(self.qr[2])
        self.assertEqual(len(layout), 1)
        layout.add(self.qr[1], 3)
        self.assertEqual(len(layout), 2)

    def test_layout_get_bits(self):
        """Get the map from the (qu)bits view"""
        layout_dict = {self.qr[0]: 0, self.qr[1]: 1, self.qr[2]: 2}
        layout = Layout(layout_dict)
        self.assertDictEqual(layout_dict, layout.get_virtual_bits())

    def test_layout_get_physical_bits(self):
        """Get the map from the physical bits view"""
        layout = Layout({self.qr[0]: 0, self.qr[1]: 1, self.qr[2]: 2})
        self.assertDictEqual(
            layout.get_physical_bits(), {0: self.qr[0], 1: self.qr[1], 2: self.qr[2]}
        )

    def test_layout_add(self):
        """add() method"""
        layout = Layout()
        layout[self.qr[0]] = 0
        layout.add(self.qr[1])

        self.assertEqual(layout[self.qr[1]], 1)

    def test_layout_add_register(self):
        """add_register() method"""
        layout = Layout()
        qr0 = QuantumRegister(2, "q0")
        qr1 = QuantumRegister(1, "qr1")
        layout.add_register(qr0)
        layout.add_register(qr1)

        self.assertEqual(layout[qr0[0]], 0)
        self.assertEqual(layout[qr0[1]], 1)
        self.assertEqual(layout[qr1[0]], 2)
        self.assertIn(qr0, layout.get_registers())
        self.assertIn(qr1, layout.get_registers())

    def test_physical_keyerror(self):
        """When asking for an unexistant physical qubit, KeyError"""
        layout = Layout()
        layout[self.qr[0]] = 1

        with self.assertRaises(KeyError):
            _ = layout[0]

    def test_virtual_keyerror(self):
        """When asking for an unexistant virtual qubit, KeyError"""
        layout = Layout()
        layout[self.qr[0]] = 1

        with self.assertRaises(KeyError):
            _ = layout[self.qr[1]]

    def test_layout_swap(self):
        """swap() method"""
        layout = Layout()
        layout.add(self.qr[0])
        layout.add(self.qr[1])
        layout.swap(0, 1)
        self.assertDictEqual(layout.get_virtual_bits(), {self.qr[0]: 1, self.qr[1]: 0})

    def test_layout_swap_error(self):
        """swap() method error"""
        layout = Layout()
        layout.add(self.qr[0])
        layout.add(self.qr[1])
        with self.assertRaises(LayoutError):
            layout.swap(0, self.qr[0])

    def test_layout_combine(self):
        """combine_into_edge_map() method"""
        layout = Layout()
        layout.add(self.qr[0])
        layout.add(self.qr[1])
        another_layout = Layout()
        another_layout.add(self.qr[1])
        another_layout.add(self.qr[0])

        edge_map = layout.combine_into_edge_map(another_layout)
        self.assertDictEqual(edge_map, {self.qr[0]: self.qr[1], self.qr[1]: self.qr[0]})

    def test_layout_combine_bigger(self):
        """combine_into_edge_map() method with another_layout is bigger"""
        layout = Layout()
        layout.add(self.qr[0])
        layout.add(self.qr[1])
        another_layout = Layout()
        another_layout.add(self.qr[1])
        another_layout.add(self.qr[0])
        another_layout.add(self.qr[2])

        edge_map = layout.combine_into_edge_map(another_layout)
        self.assertDictEqual(edge_map, {self.qr[0]: self.qr[1], self.qr[1]: self.qr[0]})

    def test_set_virtual_without_physical(self):
        """When adding a virtual without care in which physical is going"""
        layout = Layout()
        layout.add(self.qr[1], 2)
        layout.add(self.qr[0])

        self.assertDictEqual(layout.get_virtual_bits(), {self.qr[0]: 0, self.qr[1]: 2})

    def test_layout_combine_smaller(self):
        """combine_into_edge_map() method with another_layout is smaller and raises an Error"""
        layout = Layout()
        layout.add(self.qr[0])
        layout.add(self.qr[1])
        layout.add(self.qr[2])
        another_layout = Layout()
        another_layout.add(self.qr[1])
        another_layout.add(self.qr[0])

        with self.assertRaises(LayoutError):
            _ = layout.combine_into_edge_map(another_layout)

    def test_copy(self):
        """Test copy methods return equivalent layouts."""
        layout = Layout()
        layout.add(self.qr[0])
        layout.add(self.qr[1])

        layout_dict_copy = layout.copy()
        self.assertTrue(isinstance(layout_dict_copy, Layout))
        self.assertDictEqual(layout.get_physical_bits(), layout_dict_copy.get_physical_bits())
        self.assertDictEqual(layout.get_virtual_bits(), layout_dict_copy.get_virtual_bits())

        layout_copy_copy = copy.copy(layout)
        self.assertTrue(isinstance(layout_copy_copy, Layout))
        self.assertDictEqual(layout.get_physical_bits(), layout_dict_copy.get_physical_bits())
        self.assertDictEqual(layout.get_virtual_bits(), layout_dict_copy.get_virtual_bits())

        layout_copy_deepcopy = copy.deepcopy(layout)
        self.assertTrue(isinstance(layout_copy_deepcopy, Layout))
        self.assertDictEqual(layout.get_physical_bits(), layout_dict_copy.get_physical_bits())
        self.assertDictEqual(layout.get_virtual_bits(), layout_dict_copy.get_virtual_bits())

    def test_layout_error_str_key(self):
        """Layout does not work with strings"""
        layout = Layout()

        with self.assertRaises(LayoutError):
            layout["a_string"] = 3

        with self.assertRaises(LayoutError):
            layout[2] = "a_string"

    def test_layout_error_when_same_type(self):
        """Layout does not work when key and value are the same type"""
        layout = Layout()

        with self.assertRaises(LayoutError):
            layout[self.qr[0]] = [self.qr[1]]

        with self.assertRaises(LayoutError):
            layout[0] = 1

    def test_layout_repr(self):
        """Layout repr reproduces layout"""
        qr = QuantumRegister(5, "qr")
        layout = Layout(
            {
                qr[0]: 2,
                qr[1]: 4,
                qr[2]: 3,
                qr[3]: 0,
                qr[4]: 1,
            }
        )

        repr_layout = eval(  # pylint: disable=eval-used
            layout.__repr__(),
            {
                "Qubit": Qubit,
                "QuantumRegister": QuantumRegister,
                "Layout": Layout,
            },
        )
        self.assertDictEqual(layout._p2v, repr_layout._p2v)
        self.assertDictEqual(layout._v2p, repr_layout._v2p)

    def test_layout_repr_with_holes(self):
        """A non-bijective Layout repr reproduces layout"""
        qr = QuantumRegister(5, "qr")
        layout = Layout({qr[0]: 0, qr[1]: 3, qr[2]: 4, qr[3]: 5, qr[4]: 6})

        repr_layout = eval(  # pylint: disable=eval-used
            layout.__repr__(),
            {
                "Qubit": Qubit,
                "QuantumRegister": QuantumRegister,
                "Layout": Layout,
            },
        )
        self.assertDictEqual(layout._p2v, repr_layout._p2v)
        self.assertDictEqual(layout._v2p, repr_layout._v2p)

    def test_layout_from_intlist(self):
        """Create a layout from a list of integers.
        virtual  physical
         q1_0  ->  4
         q2_0  ->  5
         q2_1  ->  6
         q3_0  ->  8
         q3_1  ->  9
         q3_2  ->  10
        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        qr3 = QuantumRegister(3, "qr3")
        intlist_layout = [4, 5, 6, 8, 9, 10]
        layout = Layout.from_intlist(intlist_layout, qr1, qr2, qr3)

        expected = Layout({4: qr1[0], 5: qr2[0], 6: qr2[1], 8: qr3[0], 9: qr3[1], 10: qr3[2]})
        self.assertDictEqual(layout._p2v, expected._p2v)
        self.assertDictEqual(layout._v2p, expected._v2p)

    def test_layout_from_intlist_numpy(self):
        """Create a layout from a list of numpy integers. See #3097"""
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        qr3 = QuantumRegister(3, "qr3")
        intlist_layout = np.array([0, 1, 2, 3, 4, 5])
        layout = Layout.from_intlist(intlist_layout, qr1, qr2, qr3)

        expected = Layout.generate_trivial_layout(qr1, qr2, qr3)
        self.assertDictEqual(layout._p2v, expected._p2v)
        self.assertDictEqual(layout._v2p, expected._v2p)

    def test_layout_from_intlist_short(self):
        """Raise if the intlist is longer that your quantum register.
        virtual  physical
         q1_0  ->  4
         q2_0  ->  5
         q2_1  ->  6
         None  ->  8
         None  ->  9
         None  ->  10
        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")

        intlist_layout = [4, 5, 6, 8, 9, 10]
        with self.assertRaises(LayoutError):
            _ = Layout.from_intlist(intlist_layout, qr1, qr2)

    def test_layout_from_intlist_long(self):
        """If the intlist is shorter that your quantum register, fail.
        virtual  physical
         q1_0  ->  4
         q2_0  ->  5
         q2_1  ->  6
         q3_0  ->  8
         q3_1  ->  ?
         q3_2  ->  ?
        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        qr3 = QuantumRegister(3, "qr3")
        intlist_layout = [4, 5, 6, 8]

        with self.assertRaises(LayoutError):
            _ = Layout.from_intlist(intlist_layout, qr1, qr2, qr3)

    def test_layout_from_intlist_duplicated(self):
        """If the intlist contains duplicated ints, fail.
        virtual  physical
         q1_0  ->  4
         q2_0  ->  6 -- This is
         q2_1  ->  6 -- not allowed
        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        intlist_layout = [4, 6, 6]

        with self.assertRaises(LayoutError):
            _ = Layout.from_intlist(intlist_layout, qr1, qr2)

    def test_layout_from_tuplelist(self):
        """Create a Layout from list of tuples
        virtual  physical
         q1_0  ->  3
         q2_0  ->  5
         q2_1  ->  7
        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        tuplelist_layout = [None, None, None, qr1[0], None, qr2[0], None, qr2[1]]

        layout = Layout.from_qubit_list(tuplelist_layout)

        expected = Layout(
            {
                3: qr1[0],
                5: qr2[0],
                7: qr2[1],
            }
        )
        self.assertDictEqual(layout._p2v, expected._p2v)
        self.assertDictEqual(layout._v2p, expected._v2p)

    def test_layout_contains(self):
        """Verify Layouts support __contains__."""
        qr = QuantumRegister(2, "qr")
        layout = Layout()
        layout.add(qr[0], 0)

        self.assertIn(qr[0], layout)
        self.assertIn(0, layout)
        self.assertNotIn(qr[1], layout)
        self.assertNotIn(1, layout)


if __name__ == "__main__":
    unittest.main()

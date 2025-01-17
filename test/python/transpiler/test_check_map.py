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

"""Test the Check Map pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import CXGate
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler import CouplingMap, Target
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestCheckMapCX(QiskitTestCase):
    """Tests the CheckMap pass with CX gates"""

    def test_trivial_nop_map(self):
        """Trivial map in a circuit without entanglement
        qr0:---[H]---

        qr1:---[H]---

        qr2:---[H]---

        CouplingMap map: None
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        coupling = CouplingMap()
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set["is_swap_mapped"])

    def test_trivial_nop_map_target(self):
        """Trivial map in a circuit without entanglement
        qr0:---[H]---

        qr1:---[H]---

        qr2:---[H]---

        CouplingMap map: None
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        target = Target()
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(target)
        pass_.run(dag)
        self.assertTrue(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_true(self):
        """Mapped is easy to check
        qr0:--(+)-[H]-(+)-
               |       |
        qr1:---.-------|--
                       |
        qr2:-----------.--

        CouplingMap map: [1]--[0]--[2]
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])
        coupling = CouplingMap([[0, 1], [0, 2]])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertTrue(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_false(self):
        """Needs [0]-[1] in a [0]--[2]--[1]
        qr0:--(+)--
               |
        qr1:---.---

        CouplingMap map: [0]--[2]--[1]
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        coupling = CouplingMap([[0, 2], [2, 1]])
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(coupling)
        pass_.run(dag)

        self.assertFalse(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_false_target(self):
        """Needs [0]-[1] in a [0]--[2]--[1]
        qr0:--(+)--
               |
        qr1:---.---

        CouplingMap map: [0]--[2]--[1]
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        target = Target(num_qubits=2)
        target.add_instruction(CXGate(), {(0, 2): None, (2, 1): None})
        dag = circuit_to_dag(circuit)

        pass_ = CheckMap(target)
        pass_.run(dag)

        self.assertFalse(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_cf_true(self):
        """Check control flow blocks are mapped."""
        num_qubits = 3
        coupling = CouplingMap.from_line(num_qubits)
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(qr, cr)
        true_body.swap(0, 1)
        true_body.cx(2, 1)
        circuit.if_else((cr[0], 0), true_body, None, qr, cr)
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_cf_false(self):
        """Check control flow blocks are not mapped."""
        num_qubits = 3
        coupling = CouplingMap.from_line(num_qubits)
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(qr)
        true_body.cx(0, 2)
        circuit.if_else((cr[0], 0), true_body, None, qr, [])
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertFalse(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_cf_layout_change_false(self):
        """Check control flow blocks with layout change are not mapped."""
        num_qubits = 3
        coupling = CouplingMap.from_line(num_qubits)
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(qr, cr)
        true_body.cx(1, 2)
        circuit.if_else((cr[0], 0), true_body, None, qr[[1, 0, 2]], cr)
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertFalse(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_cf_layout_change_true(self):
        """Check control flow blocks with layout change are mapped."""
        num_qubits = 3
        coupling = CouplingMap.from_line(num_qubits)
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(qr)
        true_body.cx(0, 2)
        circuit.if_else((cr[0], 0), true_body, None, qr[[1, 0, 2]], [])
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set["is_swap_mapped"])

    def test_swap_mapped_cf_different_bits(self):
        """Check control flow blocks with layout change are mapped."""
        num_qubits = 3
        coupling = CouplingMap.from_line(num_qubits)
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(3, 1)
        true_body.cx(0, 2)
        circuit.if_else((cr[0], 0), true_body, None, qr[[1, 0, 2]], [cr[0]])
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set["is_swap_mapped"])

    def test_disjoint_controlflow_bits(self):
        """test control flow on with different registers"""
        num_qubits = 4
        coupling = CouplingMap.from_line(num_qubits)
        qr1 = QuantumRegister(4, "qr")
        qr2 = QuantumRegister(3, "qrif")
        cr = ClassicalRegister(3)
        circuit = QuantumCircuit(qr1, cr)
        true_body = QuantumCircuit(qr2, [cr[0]])
        true_body.cx(0, 2)
        circuit.if_else((cr[0], 0), true_body, None, qr1[[1, 0, 2]], [cr[0]])
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set["is_swap_mapped"])

    def test_nested_controlflow_true(self):
        """Test nested controlflow with true evaluation."""
        num_qubits = 4
        coupling = CouplingMap.from_line(num_qubits)
        qr1 = QuantumRegister(4, "qr")
        qr2 = QuantumRegister(3, "qrif")
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(1)
        circuit = QuantumCircuit(qr1, cr1)
        true_body = QuantumCircuit(qr2, cr2)
        for_body = QuantumCircuit(3)
        for_body.cx(0, 2)
        true_body.for_loop(range(5), body=for_body, qubits=qr2, clbits=[])
        circuit.if_else((cr1[0], 0), true_body, None, qr1[[1, 0, 2]], cr1)
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertTrue(pass_.property_set["is_swap_mapped"])

    def test_nested_controlflow_false(self):
        """Test nested controlflow with true evaluation."""
        num_qubits = 4
        coupling = CouplingMap.from_line(num_qubits)
        qr1 = QuantumRegister(4, "qr")
        qr2 = QuantumRegister(3, "qrif")
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(1)
        circuit = QuantumCircuit(qr1, cr1)
        true_body = QuantumCircuit(qr2, cr2)
        for_body = QuantumCircuit(3)
        for_body.cx(0, 2)
        true_body.for_loop(range(5), body=for_body, qubits=qr2, clbits=[])
        circuit.if_else((cr1[0], 0), true_body, None, qr1[[0, 1, 2]], cr1)
        dag = circuit_to_dag(circuit)
        pass_ = CheckMap(coupling)
        pass_.run(dag)
        self.assertFalse(pass_.property_set["is_swap_mapped"])

    def test_nested_conditional_unusual_bit_order(self):
        """Test that `CheckMap` succeeds when inner conditional blocks have clbits that are involved
        in their own (nested conditionals), and the binding order is not the same as the
        bit-definition order.  See gh-10394."""
        qr = QuantumRegister(2, "q")
        cr1 = ClassicalRegister(2, "c1")
        cr2 = ClassicalRegister(2, "c2")

        # Note that the bits here are not in the same order as in the outer circuit object, but they
        # are the same as the binding order in the `if_test`, so everything maps `{x: x}` and it
        # should all be fine.  This kind of thing is a staple of the control-flow builders.
        inner_order = [cr2[0], cr1[0], cr2[1], cr1[1]]
        inner = QuantumCircuit(qr, inner_order, cr1, cr2)
        with self.assertWarns(DeprecationWarning):
            inner.cx(0, 1).c_if(cr2, 3)

        outer = QuantumCircuit(qr, cr1, cr2)
        outer.if_test((cr1, 3), inner, outer.qubits, inner_order)

        pass_ = CheckMap(CouplingMap.from_line(2))
        pass_(outer)
        self.assertTrue(pass_.property_set["is_swap_mapped"])


if __name__ == "__main__":
    unittest.main()

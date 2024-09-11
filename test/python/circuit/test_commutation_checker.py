# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test commutation checker class ."""

import unittest

import numpy as np

from qiskit import ClassicalRegister
from qiskit.circuit import (
    QuantumRegister,
    Parameter,
    Qubit,
    AnnotatedOperation,
    InverseModifier,
    ControlModifier,
    Gate,
)
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit.library import (
    ZGate,
    XGate,
    CXGate,
    CCXGate,
    MCXGate,
    RZGate,
    Measure,
    Barrier,
    Reset,
    LinearFunction,
    SGate,
    RXXGate,
)
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class NewGateCX(Gate):
    """A dummy class containing an cx gate unknown to the commutation checker's library."""

    def __init__(self):
        super().__init__("new_cx", 2, [])

    def to_matrix(self):
        return np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)


class TestCommutationChecker(QiskitTestCase):
    """Test CommutationChecker class."""

    def test_simple_gates(self):
        """Check simple commutation relations between gates, experimenting with
        different orders of gates, different orders of qubits, different sets of
        qubits over which gates are defined, and so on."""

        # should commute
        self.assertTrue(scc.commute(ZGate(), [0], [], CXGate(), [0, 1], []))
        # should not commute
        self.assertFalse(scc.commute(ZGate(), [1], [], CXGate(), [0, 1], []))
        # should not commute
        self.assertFalse(scc.commute(XGate(), [0], [], CXGate(), [0, 1], []))
        # should commute
        self.assertTrue(scc.commute(XGate(), [1], [], CXGate(), [0, 1], []))
        # should not commute
        self.assertFalse(scc.commute(XGate(), [1], [], CXGate(), [1, 0], []))
        # should commute
        self.assertTrue(scc.commute(XGate(), [0], [], CXGate(), [1, 0], []))
        # should commute
        self.assertTrue(scc.commute(CXGate(), [1, 0], [], XGate(), [0], []))
        # should not commute
        self.assertFalse(scc.commute(CXGate(), [1, 0], [], XGate(), [1], []))
        # should commute
        self.assertTrue(scc.commute(CXGate(), [1, 0], [], CXGate(), [1, 0], []))
        # should not commute
        self.assertFalse(scc.commute(CXGate(), [1, 0], [], CXGate(), [0, 1], []))
        # should commute
        self.assertTrue(scc.commute(CXGate(), [1, 0], [], CXGate(), [1, 2], []))
        # should not commute
        self.assertFalse(scc.commute(CXGate(), [1, 0], [], CXGate(), [2, 1], []))
        # should commute
        self.assertTrue(scc.commute(CXGate(), [1, 0], [], CXGate(), [2, 3], []))
        self.assertTrue(scc.commute(XGate(), [2], [], CCXGate(), [0, 1, 2], []))
        self.assertFalse(scc.commute(CCXGate(), [0, 1, 2], [], CCXGate(), [0, 2, 1], []))

    def test_passing_quantum_registers(self):
        """Check that passing QuantumRegisters works correctly."""
        qr = QuantumRegister(4)
        # should commute
        self.assertTrue(scc.commute(CXGate(), [qr[1], qr[0]], [], CXGate(), [qr[1], qr[2]], []))
        # should not commute
        self.assertFalse(scc.commute(CXGate(), [qr[0], qr[1]], [], CXGate(), [qr[1], qr[2]], []))

    def test_standard_gates_commutations(self):
        """Check that commutativity checker uses standard gates commutations as expected."""
        scc.clear_cached_commutations()
        self.assertTrue(scc.commute(ZGate(), [0], [], CXGate(), [0, 1], []))
        self.assertEqual(scc.num_cached_entries(), 0)

    def test_caching_positive_results(self):
        """Check that hashing positive results in commutativity checker works as expected."""
        scc.clear_cached_commutations()
        self.assertTrue(scc.commute(ZGate(), [0], [], NewGateCX(), [0, 1], []))
        self.assertGreater(scc.num_cached_entries(), 0)

    def test_caching_lookup_with_non_overlapping_qubits(self):
        """Check that commutation lookup with non-overlapping qubits works as expected."""
        scc.clear_cached_commutations()
        self.assertTrue(scc.commute(CXGate(), [0, 2], [], CXGate(), [0, 1], []))
        self.assertFalse(scc.commute(CXGate(), [0, 1], [], CXGate(), [1, 2], []))
        self.assertEqual(scc.num_cached_entries(), 0)

    def test_caching_store_and_lookup_with_non_overlapping_qubits(self):
        """Check that commutations storing and lookup with non-overlapping qubits works as expected."""
        scc_lenm = scc.num_cached_entries()
        self.assertTrue(scc.commute(NewGateCX(), [0, 2], [], CXGate(), [0, 1], []))
        self.assertFalse(scc.commute(NewGateCX(), [0, 1], [], CXGate(), [1, 2], []))
        self.assertTrue(scc.commute(NewGateCX(), [1, 4], [], CXGate(), [1, 6], []))
        self.assertFalse(scc.commute(NewGateCX(), [5, 3], [], CXGate(), [3, 1], []))
        self.assertEqual(scc.num_cached_entries(), scc_lenm + 2)

    def test_caching_negative_results(self):
        """Check that hashing negative results in commutativity checker works as expected."""
        scc.clear_cached_commutations()
        self.assertFalse(scc.commute(XGate(), [0], [], NewGateCX(), [0, 1], []))
        self.assertGreater(scc.num_cached_entries(), 0)

    def test_caching_different_qubit_sets(self):
        """Check that hashing same commutativity results over different qubit sets works as expected."""
        scc.clear_cached_commutations()
        # All the following should be cached in the same way
        # though each relation gets cached twice: (A, B) and (B, A)
        scc.commute(XGate(), [0], [], NewGateCX(), [0, 1], [])
        scc.commute(XGate(), [10], [], NewGateCX(), [10, 20], [])
        scc.commute(XGate(), [10], [], NewGateCX(), [10, 5], [])
        scc.commute(XGate(), [5], [], NewGateCX(), [5, 7], [])
        self.assertEqual(scc.num_cached_entries(), 1)

    def test_cache_with_param_gates(self):
        """Check commutativity between (non-parameterized) gates with parameters."""
        scc.clear_cached_commutations()

        self.assertTrue(scc.commute(RZGate(0), [0], [], XGate(), [0], []))
        self.assertFalse(scc.commute(RZGate(np.pi / 2), [0], [], XGate(), [0], []))
        self.assertTrue(scc.commute(RZGate(np.pi / 2), [0], [], RZGate(0), [0], []))

        self.assertFalse(scc.commute(RZGate(np.pi / 2), [1], [], XGate(), [1], []))
        self.assertEqual(scc.num_cached_entries(), 3)

    def test_gates_with_parameters(self):
        """Check commutativity between (non-parameterized) gates with parameters."""
        self.assertTrue(scc.commute(RZGate(0), [0], [], XGate(), [0], []))
        self.assertFalse(scc.commute(RZGate(np.pi / 2), [0], [], XGate(), [0], []))
        self.assertTrue(scc.commute(RZGate(np.pi / 2), [0], [], RZGate(0), [0], []))

    def test_parameterized_gates(self):
        """Check commutativity between parameterized gates, both with free and with
        bound parameters."""
        # gate that has parameters but is not considered parameterized
        rz_gate = RZGate(np.pi / 2)
        self.assertEqual(len(rz_gate.params), 1)
        self.assertFalse(rz_gate.is_parameterized())

        # gate that has parameters and is considered parameterized
        rz_gate_theta = RZGate(Parameter("Theta"))
        rz_gate_phi = RZGate(Parameter("Phi"))
        self.assertEqual(len(rz_gate_theta.params), 1)
        self.assertTrue(rz_gate_theta.is_parameterized())

        # gate that has no parameters and is not considered parameterized
        cx_gate = CXGate()
        self.assertEqual(len(cx_gate.params), 0)
        self.assertFalse(cx_gate.is_parameterized())

        # We should detect that these gates commute
        self.assertTrue(scc.commute(rz_gate, [0], [], cx_gate, [0, 1], []))

        # We should detect that these gates commute
        self.assertTrue(scc.commute(rz_gate, [0], [], rz_gate, [0], []))

        # We should detect that parameterized gates over disjoint qubit subsets commute
        self.assertTrue(scc.commute(rz_gate_theta, [0], [], rz_gate_theta, [1], []))

        # We should detect that parameterized gates over disjoint qubit subsets commute
        self.assertTrue(scc.commute(rz_gate_theta, [0], [], rz_gate_phi, [1], []))

        # We should detect that parameterized gates over disjoint qubit subsets commute
        self.assertTrue(scc.commute(rz_gate_theta, [2], [], cx_gate, [1, 3], []))

        # However, for now commutativity checker should return False when checking
        # commutativity between a parameterized gate and some other gate, when
        # the two gates are over intersecting qubit subsets.
        # This check should be changed if commutativity checker is extended to
        # handle parameterized gates better.
        self.assertFalse(scc.commute(rz_gate_theta, [0], [], cx_gate, [0, 1], []))

        self.assertFalse(scc.commute(rz_gate_theta, [0], [], rz_gate, [0], []))

    def test_measure(self):
        """Check commutativity involving measures."""
        # Measure is over qubit 0, while gate is over a disjoint subset of qubits
        # We should be able to swap these.
        self.assertTrue(scc.commute(Measure(), [0], [0], CXGate(), [1, 2], []))

        # Measure and gate have intersecting set of qubits
        # We should not be able to swap these.
        self.assertFalse(scc.commute(Measure(), [0], [0], CXGate(), [0, 2], []))

        # Measures over different qubits and clbits
        self.assertTrue(scc.commute(Measure(), [0], [0], Measure(), [1], [1]))

        # Measures over different qubits but same classical bit
        # We should not be able to swap these.
        self.assertFalse(scc.commute(Measure(), [0], [0], Measure(), [1], [0]))

        # Measures over same qubits but different classical bit
        # ToDo: can we swap these?
        # Currently checker takes the safe approach and returns False.
        self.assertFalse(scc.commute(Measure(), [0], [0], Measure(), [0], [1]))

    def test_barrier(self):
        """Check commutativity involving barriers."""
        # A gate should not commute with a barrier
        # (at least if these are over intersecting qubit sets).
        self.assertFalse(scc.commute(Barrier(4), [0, 1, 2, 3], [], CXGate(), [1, 2], []))

        # Does it even make sense to have a barrier over a subset of qubits?
        # Though in this case, it probably makes sense to say that barrier and gate can be swapped.
        self.assertTrue(scc.commute(Barrier(4), [0, 1, 2, 3], [], CXGate(), [5, 6], []))

    def test_reset(self):
        """Check commutativity involving resets."""
        # A gate should not commute with reset when the qubits intersect.
        self.assertFalse(scc.commute(Reset(), [0], [], CXGate(), [0, 2], []))

        # A gate should commute with reset when the qubits are disjoint.
        self.assertTrue(scc.commute(Reset(), [0], [], CXGate(), [1, 2], []))

    def test_conditional_gates(self):
        """Check commutativity involving conditional gates."""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(2)

        # Currently, in all cases commutativity checker should returns False.
        # This is definitely suboptimal.
        self.assertFalse(
            scc.commute(CXGate().c_if(cr[0], 0), [qr[0], qr[1]], [], XGate(), [qr[2]], [])
        )
        self.assertFalse(
            scc.commute(CXGate().c_if(cr[0], 0), [qr[0], qr[1]], [], XGate(), [qr[1]], [])
        )
        self.assertFalse(
            scc.commute(
                CXGate().c_if(cr[0], 0),
                [qr[0], qr[1]],
                [],
                CXGate().c_if(cr[0], 0),
                [qr[0], qr[1]],
                [],
            )
        )
        self.assertFalse(
            scc.commute(XGate().c_if(cr[0], 0), [qr[0]], [], XGate().c_if(cr[0], 1), [qr[0]], [])
        )
        self.assertFalse(scc.commute(XGate().c_if(cr[0], 0), [qr[0]], [], XGate(), [qr[0]], []))

    def test_complex_gates(self):
        """Check commutativity involving more complex gates."""
        lf1 = LinearFunction([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        lf2 = LinearFunction([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        # lf1 is equivalent to swap(0, 1), and lf2 to swap(1, 2).
        # These do not commute.
        self.assertFalse(scc.commute(lf1, [0, 1, 2], [], lf2, [0, 1, 2], []))

        lf3 = LinearFunction([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        lf4 = LinearFunction([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        # lf3 is permutation 1->2, 2->3, 3->1.
        # lf3 is the inverse permutation 1->3, 2->1, 3->2.
        # These commute.
        self.assertTrue(scc.commute(lf3, [0, 1, 2], [], lf4, [0, 1, 2], []))

    def test_equal_annotated_operations_commute(self):
        """Check commutativity involving the same annotated operation."""
        op1 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(1)])
        op2 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(1)])
        # the same, so true
        self.assertTrue(scc.commute(op1, [0, 1], [], op2, [0, 1], []))

    def test_annotated_operations_commute_with_unannotated(self):
        """Check commutativity involving annotated operations and unannotated operations."""
        op1 = AnnotatedOperation(SGate(), [InverseModifier(), ControlModifier(1)])
        op2 = AnnotatedOperation(ZGate(), [InverseModifier()])
        op3 = ZGate()
        # all true
        self.assertTrue(scc.commute(op1, [0, 1], [], op2, [1], []))
        self.assertTrue(scc.commute(op1, [0, 1], [], op3, [1], []))
        self.assertTrue(scc.commute(op2, [1], [], op3, [1], []))

    def test_utf8_gate_names(self):
        """Check compatibility of non-ascii quantum gate names."""
        g0 = RXXGate(1.234).to_mutable()
        g0.name = "すみません"

        g1 = RXXGate(2.234).to_mutable()
        g1.name = "ok_0"

        self.assertTrue(scc.commute(g0, [0, 1], [], g1, [1, 0], []))

    def test_annotated_operations_no_commute(self):
        """Check non-commutativity involving annotated operations."""
        op1 = AnnotatedOperation(XGate(), [InverseModifier(), ControlModifier(1)])
        op2 = AnnotatedOperation(XGate(), [InverseModifier()])
        # false
        self.assertFalse(scc.commute(op1, [0, 1], [], op2, [0], []))

    def test_c7x_gate(self):
        """Test wide gate works correctly."""
        qargs = [Qubit() for _ in [None] * 8]
        res = scc.commute(XGate(), qargs[:1], [], XGate().control(7), qargs, [])
        self.assertFalse(res)

    def test_wide_gates_over_nondisjoint_qubits(self):
        """Test that checking wide gates does not lead to memory problems."""
        self.assertFalse(scc.commute(MCXGate(29), list(range(30)), [], XGate(), [0], []))

    def test_wide_gates_over_disjoint_qubits(self):
        """Test that wide gates still commute when they are over disjoint sets of qubits."""
        self.assertTrue(scc.commute(MCXGate(29), list(range(30)), [], XGate(), [30], []))
        self.assertTrue(scc.commute(XGate(), [30], [], MCXGate(29), list(range(30)), []))

    def test_serialization(self):
        """Test that the commutation checker is correctly serialized"""
        import pickle

        scc.clear_cached_commutations()
        self.assertTrue(scc.commute(ZGate(), [0], [], NewGateCX(), [0, 1], []))
        cc2 = pickle.loads(pickle.dumps(scc))
        self.assertEqual(cc2.num_cached_entries(), 1)
        dop1 = DAGOpNode(ZGate(), qargs=[0], cargs=[])
        dop2 = DAGOpNode(NewGateCX(), qargs=[0, 1], cargs=[])
        cc2.commute_nodes(dop1, dop2)
        dop1 = DAGOpNode(ZGate(), qargs=[0], cargs=[])
        dop2 = DAGOpNode(CXGate(), qargs=[0, 1], cargs=[])
        cc2.commute_nodes(dop1, dop2)
        self.assertEqual(cc2.num_cached_entries(), 1)


if __name__ == "__main__":
    unittest.main()

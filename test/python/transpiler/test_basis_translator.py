# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test the BasisTranslator pass"""

import os

from numpy import pi
import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import transpile
from qiskit.test import QiskitTestCase
from qiskit.circuit import Gate, Parameter, EquivalenceLibrary, Qubit, Clbit
from qiskit.circuit.library import (
    U1Gate,
    U2Gate,
    U3Gate,
    CU1Gate,
    CU3Gate,
    UGate,
    RZGate,
    XGate,
    SXGate,
    CXGate,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit, circuit_to_instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator
from qiskit.transpiler.target import Target, InstructionProperties
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.basis import BasisTranslator, UnrollCustomDefinitions



from qiskit.circuit.library.standard_gates.equivalence_library import (
    StandardEquivalenceLibrary as std_eqlib,
)


class OneQubitZeroParamGate(Gate):
    """Mock one qubit zero param gate."""

    def __init__(self, name="1q0p"):
        super().__init__(name, 1, [])


class OneQubitOneParamGate(Gate):
    """Mock one qubit one param gate."""

    def __init__(self, theta, name="1q1p"):
        super().__init__(name, 1, [theta])


class OneQubitOneParamPrimeGate(Gate):
    """Mock one qubit one param gate."""

    def __init__(self, alpha):
        super().__init__("1q1p_prime", 1, [alpha])


class OneQubitTwoParamGate(Gate):
    """Mock one qubit two param gate."""

    def __init__(self, phi, lam, name="1q2p"):
        super().__init__(name, 1, [phi, lam])


class TwoQubitZeroParamGate(Gate):
    """Mock one qubit zero param gate."""

    def __init__(self, name="2q0p"):
        super().__init__(name, 2, [])


class VariadicZeroParamGate(Gate):
    """Mock variadic zero param gate."""

    def __init__(self, num_qubits, name="vq0p"):
        super().__init__(name, num_qubits, [])


class TestBasisTranslator(QiskitTestCase):
    """Test the BasisTranslator pass."""

    def test_circ_in_basis_no_op(self):
        """Verify we don't change a circuit already in the target basis."""
        eq_lib = EquivalenceLibrary()
        qc = QuantumCircuit(1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = circuit_to_dag(qc)

        pass_ = BasisTranslator(eq_lib, ["1q0p"])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected)

    def test_raise_if_target_basis_unreachable(self):
        """Verify we raise if the circuit cannot be transformed to the target."""
        eq_lib = EquivalenceLibrary()

        qc = QuantumCircuit(1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        pass_ = BasisTranslator(eq_lib, ["1q1p"])

        with self.assertRaises(TranspilerError):
            pass_.run(dag)

    def test_single_substitution(self):
        """Verify we correctly unroll gates through a single equivalence."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitOneParamGate(pi), [0])

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(1)
        expected.append(OneQubitOneParamGate(pi), [0])
        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ["1q1p"])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)

    def test_double_substitution(self):
        """Verify we correctly unroll gates through multiple equivalences."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitOneParamGate(pi), [0])

        eq_lib.add_equivalence(gate, equiv)

        theta = Parameter("theta")
        gate = OneQubitOneParamGate(theta)
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitTwoParamGate(theta, pi / 2), [0])

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(1)
        expected.append(OneQubitTwoParamGate(pi, pi / 2), [0])
        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ["1q2p"])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)

    def test_single_substitution_with_global_phase(self):
        """Verify we correctly unroll gates through a single equivalence with global phase."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1, global_phase=0.2)
        equiv.append(OneQubitOneParamGate(pi), [0])

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1, global_phase=0.1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(1, global_phase=0.1 + 0.2)
        expected.append(OneQubitOneParamGate(pi), [0])
        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ["1q1p"])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)

    def test_single_two_gate_substitution_with_global_phase(self):
        """Verify we correctly unroll gates through a single equivalence with global phase."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1, global_phase=0.2)
        equiv.append(OneQubitOneParamGate(pi), [0])
        equiv.append(OneQubitOneParamGate(pi), [0])

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1, global_phase=0.1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(1, global_phase=0.1 + 0.2)
        expected.append(OneQubitOneParamGate(pi), [0])
        expected.append(OneQubitOneParamGate(pi), [0])
        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ["1q1p"])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)

    def test_two_substitutions_with_global_phase(self):
        """Verify we correctly unroll gates through a single equivalences with global phase."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1, global_phase=0.2)
        equiv.append(OneQubitOneParamGate(pi), [0])

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1, global_phase=0.1)
        qc.append(OneQubitZeroParamGate(), [0])
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(1, global_phase=0.1 + 2 * 0.2)
        expected.append(OneQubitOneParamGate(pi), [0])
        expected.append(OneQubitOneParamGate(pi), [0])
        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ["1q1p"])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)

    def test_two_single_two_gate_substitutions_with_global_phase(self):
        """Verify we correctly unroll gates through a single equivalence with global phase."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1, global_phase=0.2)
        equiv.append(OneQubitOneParamGate(pi), [0])
        equiv.append(OneQubitOneParamGate(pi), [0])

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1, global_phase=0.1)
        qc.append(OneQubitZeroParamGate(), [0])
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(1, global_phase=0.1 + 2 * 0.2)
        expected.append(OneQubitOneParamGate(pi), [0])
        expected.append(OneQubitOneParamGate(pi), [0])
        expected.append(OneQubitOneParamGate(pi), [0])
        expected.append(OneQubitOneParamGate(pi), [0])

        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ["1q1p"])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)

    def test_double_substitution_with_global_phase(self):
        """Verify we correctly unroll gates through multiple equivalences with global phase."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1, global_phase=0.2)
        equiv.append(OneQubitOneParamGate(pi), [0])

        eq_lib.add_equivalence(gate, equiv)

        theta = Parameter("theta")
        gate = OneQubitOneParamGate(theta)
        equiv = QuantumCircuit(1, global_phase=0.4)
        equiv.append(OneQubitTwoParamGate(theta, pi / 2), [0])

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1, global_phase=0.1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(1, global_phase=0.1 + 0.2 + 0.4)
        expected.append(OneQubitTwoParamGate(pi, pi / 2), [0])
        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ["1q2p"])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)

    def test_multiple_variadic(self):
        """Verify circuit with multiple instances of variadic gate."""
        eq_lib = EquivalenceLibrary()

        # e.g. MSGate
        oneq_gate = VariadicZeroParamGate(1)
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitZeroParamGate(), [0])
        eq_lib.add_equivalence(oneq_gate, equiv)

        twoq_gate = VariadicZeroParamGate(2)
        equiv = QuantumCircuit(2)
        equiv.append(TwoQubitZeroParamGate(), [0, 1])
        eq_lib.add_equivalence(twoq_gate, equiv)

        qc = QuantumCircuit(2)
        qc.append(VariadicZeroParamGate(1), [0])
        qc.append(VariadicZeroParamGate(2), [0, 1])

        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(2)
        expected.append(OneQubitZeroParamGate(), [0])
        expected.append(TwoQubitZeroParamGate(), [0, 1])

        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ["1q0p", "2q0p"])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)

    def test_diamond_path(self):
        """Verify we find a path when there are multiple paths to the target basis."""
        eq_lib = EquivalenceLibrary()

        # Path 1: 1q0p -> 1q1p(pi) -> 1q2p(pi, pi/2)

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitOneParamGate(pi), [0])

        eq_lib.add_equivalence(gate, equiv)

        theta = Parameter("theta")
        gate = OneQubitOneParamGate(theta)
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitTwoParamGate(theta, pi / 2), [0])

        eq_lib.add_equivalence(gate, equiv)

        # Path 2: 1q0p -> 1q1p_prime(pi/2) -> 1q2p(2 * pi/2, pi/2)

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitOneParamPrimeGate(pi / 2), [0])

        eq_lib.add_equivalence(gate, equiv)

        alpha = Parameter("alpha")
        gate = OneQubitOneParamPrimeGate(alpha)
        equiv = QuantumCircuit(1)
        equiv.append(OneQubitTwoParamGate(2 * alpha, pi / 2), [0])

        eq_lib.add_equivalence(gate, equiv)

        qc = QuantumCircuit(1)
        qc.append(OneQubitZeroParamGate(), [0])
        dag = circuit_to_dag(qc)

        expected = QuantumCircuit(1)
        expected.append(OneQubitTwoParamGate(pi, pi / 2), [0])
        expected_dag = circuit_to_dag(expected)

        pass_ = BasisTranslator(eq_lib, ["1q2p"])
        actual = pass_.run(dag)

        self.assertEqual(actual, expected_dag)

    def test_if_else(self):
        """Test a simple if-else with parameters."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]
        alpha = Parameter("alpha")
        beta = Parameter("beta")
        gate = OneQubitOneParamGate(alpha)
        equiv = QuantumCircuit([qubits[0]])
        equiv.append(OneQubitZeroParamGate(name="1q0p_2"), [qubits[0]])
        equiv.append(OneQubitOneParamGate(alpha, name="1q1p_2"), [qubits[0]])

        eq_lib = EquivalenceLibrary()
        eq_lib.add_equivalence(gate, equiv)

        circ = QuantumCircuit(qubits, clbits)
        circ.append(OneQubitOneParamGate(beta), [qubits[0]])
        circ.measure(qubits[0], clbits[1])
        with circ.if_test((clbits[1], 0)) as else_:
            circ.append(OneQubitOneParamGate(alpha), [qubits[0]])
            circ.append(TwoQubitZeroParamGate(), qubits)
        with else_:
            circ.append(TwoQubitZeroParamGate(), [qubits[1], qubits[0]])
        dag = circuit_to_dag(circ)
        dag_translated = BasisTranslator(eq_lib, ["if_else", "1q0p_2", "1q1p_2", "2q0p"]).run(dag)

        expected = QuantumCircuit(qubits, clbits)
        expected.append(OneQubitZeroParamGate(name="1q0p_2"), [qubits[0]])
        expected.append(OneQubitOneParamGate(beta, name="1q1p_2"), [qubits[0]])
        expected.measure(qubits[0], clbits[1])
        with expected.if_test((clbits[1], 0)) as else_:
            expected.append(OneQubitZeroParamGate(name="1q0p_2"), [qubits[0]])
            expected.append(OneQubitOneParamGate(alpha, name="1q1p_2"), [qubits[0]])
            expected.append(TwoQubitZeroParamGate(), qubits)
        with else_:
            expected.append(TwoQubitZeroParamGate(), [qubits[1], qubits[0]])
        dag_expected = circuit_to_dag(expected)
        self.assertEqual(dag_translated, dag_expected)

    def test_nested_loop(self):
        """Test a simple if-else with parameters."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]
        cr = ClassicalRegister(bits=clbits)
        index1 = Parameter("index1")
        alpha = Parameter("alpha")

        gate = OneQubitOneParamGate(alpha)
        equiv = QuantumCircuit([qubits[0]])
        equiv.append(OneQubitZeroParamGate(name="1q0p_2"), [qubits[0]])
        equiv.append(OneQubitOneParamGate(alpha, name="1q1p_2"), [qubits[0]])

        eq_lib = EquivalenceLibrary()
        eq_lib.add_equivalence(gate, equiv)

        circ = QuantumCircuit(qubits, cr)
        with circ.for_loop(range(3), loop_parameter=index1) as ind:
            with circ.while_loop((cr, 0)):
                circ.append(OneQubitOneParamGate(alpha * ind), [qubits[0]])
        dag = circuit_to_dag(circ)
        dag_translated = BasisTranslator(
            eq_lib, ["if_else", "for_loop", "while_loop", "1q0p_2", "1q1p_2"]
        ).run(dag)

        expected = QuantumCircuit(qubits, cr)
        with expected.for_loop(range(3), loop_parameter=index1) as ind:
            with expected.while_loop((cr, 0)):
                expected.append(OneQubitZeroParamGate(name="1q0p_2"), [qubits[0]])
                expected.append(OneQubitOneParamGate(alpha * ind, name="1q1p_2"), [qubits[0]])
        dag_expected = circuit_to_dag(expected)
        self.assertEqual(dag_translated, dag_expected)

    def test_different_bits(self):
        """Test that the basis translator correctly works when the inner blocks of control-flow
        operations are not over the same bits as the outer blocks."""
        base = QuantumCircuit([Qubit() for _ in [None] * 4], [Clbit()])
        for_body = QuantumCircuit([Qubit(), Qubit()])
        for_body.h(0)
        for_body.cz(0, 1)
        base.for_loop((1,), None, for_body, [1, 2], [])

        while_body = QuantumCircuit([Qubit(), Qubit(), Clbit()])
        while_body.cz(0, 1)

        true_body = QuantumCircuit([Qubit(), Qubit(), Clbit()])
        true_body.measure(0, 0)
        true_body.while_loop((0, True), while_body, [0, 1], [0])
        false_body = QuantumCircuit([Qubit(), Qubit(), Clbit()])
        false_body.cz(0, 1)
        base.if_else((0, True), true_body, false_body, [0, 3], [0])

        basis = {"rz", "sx", "cx", "for_loop", "if_else", "while_loop", "measure"}
        out = BasisTranslator(std_eqlib, basis).run(circuit_to_dag(base))
        self.assertEqual(set(out.count_ops(recurse=True)), basis)


class TestUnrollerCompatability(QiskitTestCase):
    """Tests backward compatability with the Unroller pass.

    Duplicate of TestUnroller from test.python.transpiler.test_unroller with
    Unroller replaced by UnrollCustomDefinitions -> BasisTranslator.
    """

    def test_basic_unroll(self):
        """Test decompose a single H into u2."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)
        pass_ = UnrollCustomDefinitions(std_eqlib, ["u2"])
        dag = pass_.run(dag)
        pass_ = BasisTranslator(std_eqlib, ["u2"])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0].name, "u2")

    def test_unroll_toffoli(self):
        """Test unroll toffoli on multi regs to h, t, tdg, cx."""
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(1, "qr2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = UnrollCustomDefinitions(std_eqlib, ["h", "t", "tdg", "cx"])
        dag = pass_.run(dag)
        pass_ = BasisTranslator(std_eqlib, ["h", "t", "tdg", "cx"])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            self.assertIn(node.name, ["h", "t", "tdg", "cx"])

    def test_unroll_1q_chain_conditional(self):
        """Test unroll chain of 1-qubit gates interrupted by conditional."""

        #     ┌───┐┌─────┐┌───┐┌───┐┌─────────┐┌─────────┐┌─────────┐┌─┐ ┌───┐  ┌───┐ »
        # qr: ┤ H ├┤ Tdg ├┤ Z ├┤ T ├┤ Ry(0.5) ├┤ Rz(0.3) ├┤ Rx(0.1) ├┤M├─┤ X ├──┤ Y ├─»
        #     └───┘└─────┘└───┘└───┘└─────────┘└─────────┘└─────────┘└╥┘ └─╥─┘  └─╥─┘ »
        #                                                             ║ ┌──╨──┐┌──╨──┐»
        # cr: 1/══════════════════════════════════════════════════════╩═╡ 0x1 ╞╡ 0x1 ╞»
        #                                                             0 └─────┘└─────┘»
        # «       ┌───┐
        # «  qr: ─┤ Z ├─
        # «       └─╥─┘
        # «      ┌──╨──┐
        # «cr: 1/╡ 0x1 ╞
        # «      └─────┘
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.tdg(qr)
        circuit.z(qr)
        circuit.t(qr)
        circuit.ry(0.5, qr)
        circuit.rz(0.3, qr)
        circuit.rx(0.1, qr)
        circuit.measure(qr, cr)
        circuit.x(qr).c_if(cr, 1)
        circuit.y(qr).c_if(cr, 1)
        circuit.z(qr).c_if(cr, 1)
        dag = circuit_to_dag(circuit)
        pass_ = UnrollCustomDefinitions(std_eqlib, ["u1", "u2", "u3"])
        dag = pass_.run(dag)

        pass_ = BasisTranslator(std_eqlib, ["u1", "u2", "u3"])
        unrolled_dag = pass_.run(dag)

        # Pick up -1 * 0.3 / 2 global phase for one RZ -> U1.
        #
        # global phase: 6.1332
        #     ┌─────────┐┌──────────┐┌───────┐┌─────────┐┌─────────────┐┌─────────┐»
        # qr: ┤ U2(0,π) ├┤ U1(-π/4) ├┤ U1(π) ├┤ U1(π/4) ├┤ U3(0.5,0,0) ├┤ U1(0.3) ├»
        #     └─────────┘└──────────┘└───────┘└─────────┘└─────────────┘└─────────┘»
        # cr: 1/═══════════════════════════════════════════════════════════════════»
        #                                                                          »
        # «      ┌──────────────────┐┌─┐┌───────────┐┌───────────────┐┌───────┐
        # «  qr: ┤ U3(0.1,-π/2,π/2) ├┤M├┤ U3(π,0,π) ├┤ U3(π,π/2,π/2) ├┤ U1(π) ├
        # «      └──────────────────┘└╥┘└─────╥─────┘└───────╥───────┘└───╥───┘
        # «                           ║    ┌──╨──┐        ┌──╨──┐      ┌──╨──┐
        # «cr: 1/═════════════════════╩════╡ 0x1 ╞════════╡ 0x1 ╞══════╡ 0x1 ╞═
        # «                           0    └─────┘        └─────┘      └─────┘
        ref_circuit = QuantumCircuit(qr, cr, global_phase=-0.3 / 2)
        ref_circuit.append(U2Gate(0, pi), [qr[0]])
        ref_circuit.append(U1Gate(-pi / 4), [qr[0]])
        ref_circuit.append(U1Gate(pi), [qr[0]])
        ref_circuit.append(U1Gate(pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(0.5, 0, 0), [qr[0]])
        ref_circuit.append(U1Gate(0.3), [qr[0]])
        ref_circuit.append(U3Gate(0.1, -pi / 2, pi / 2), [qr[0]])
        ref_circuit.measure(qr[0], cr[0])
        ref_circuit.append(U3Gate(pi, 0, pi), [qr[0]]).c_if(cr, 1)
        ref_circuit.append(U3Gate(pi, pi / 2, pi / 2), [qr[0]]).c_if(cr, 1)
        ref_circuit.append(U1Gate(pi), [qr[0]]).c_if(cr, 1)
        ref_dag = circuit_to_dag(ref_circuit)

        self.assertEqual(unrolled_dag, ref_dag)

    def test_unroll_no_basis(self):
        """Test when a given gate has no decompositions."""
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        dag = circuit_to_dag(circuit)
        pass_ = UnrollCustomDefinitions(std_eqlib, [])
        dag = pass_.run(dag)

        pass_ = BasisTranslator(std_eqlib, [])

        with self.assertRaises(QiskitError):
            pass_.run(dag)

    def test_unroll_all_instructions(self):
        """Test unrolling a circuit containing all standard instructions."""

        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.crx(0.5, qr[1], qr[2])
        circuit.cry(0.5, qr[1], qr[2])
        circuit.ccx(qr[0], qr[1], qr[2])
        circuit.ch(qr[0], qr[2])
        circuit.crz(0.5, qr[1], qr[2])
        circuit.cswap(qr[1], qr[0], qr[2])
        circuit.append(CU1Gate(0.1), [qr[0], qr[2]])
        circuit.append(CU3Gate(0.2, 0.1, 0.0), [qr[1], qr[2]])
        circuit.cx(qr[1], qr[0])
        circuit.cy(qr[1], qr[2])
        circuit.cz(qr[2], qr[0])
        circuit.h(qr[1])
        circuit.i(qr[0])
        circuit.rx(0.1, qr[0])
        circuit.ry(0.2, qr[1])
        circuit.rz(0.3, qr[2])
        circuit.rzz(0.6, qr[1], qr[0])
        circuit.s(qr[0])
        circuit.sdg(qr[1])
        circuit.swap(qr[1], qr[2])
        circuit.t(qr[2])
        circuit.tdg(qr[0])
        circuit.append(U1Gate(0.1), [qr[1]])
        circuit.append(U2Gate(0.2, -0.1), [qr[0]])
        circuit.append(U3Gate(0.3, 0.0, -0.1), [qr[2]])
        circuit.x(qr[2])
        circuit.y(qr[1])
        circuit.z(qr[0])
        # circuit.snapshot('0')
        # circuit.measure(qr, cr)
        dag = circuit_to_dag(circuit)
        pass_ = UnrollCustomDefinitions(std_eqlib, ["u3", "cx", "id"])
        dag = pass_.run(dag)

        pass_ = BasisTranslator(std_eqlib, ["u3", "cx", "id"])
        unrolled_dag = pass_.run(dag)

        ref_circuit = QuantumCircuit(qr, cr)
        ref_circuit.append(U3Gate(0, 0, pi / 2), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(-0.25, 0, 0), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0.25, -pi / 2, 0), [qr[2]])
        ref_circuit.append(U3Gate(0.25, 0, 0), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(-0.25, 0, 0), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(pi / 2, 0, pi), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, -pi / 4), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, pi / 4), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, pi / 4), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, -pi / 4), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.cx(qr[0], qr[1])
        ref_circuit.append(U3Gate(0, 0, pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, -pi / 4), [qr[1]])
        ref_circuit.cx(qr[0], qr[1])
        ref_circuit.append(U3Gate(0, 0, pi / 4), [qr[2]])
        ref_circuit.append(U3Gate(pi / 2, 0, pi), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, pi / 2), [qr[2]])
        ref_circuit.append(U3Gate(pi / 2, 0, pi), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, pi / 4), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, -pi / 4), [qr[2]])
        ref_circuit.append(U3Gate(pi / 2, 0, pi), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, -pi / 2), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, 0.25), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, -0.25), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.append(U3Gate(pi / 2, 0, pi), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, -pi / 4), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, pi / 4), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, -pi / 4), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(0, 0, -pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, pi / 4), [qr[1]])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(0, 0, 0.05), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, pi / 4), [qr[2]])
        ref_circuit.append(U3Gate(pi / 2, 0, pi), [qr[2]])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.append(U3Gate(0, 0, 0.05), [qr[0]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, -0.05), [qr[2]])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.append(U3Gate(0, 0, 0.05), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, -0.05), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(-0.1, 0, -0.05), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(pi / 2, 0, pi), [qr[0]])
        ref_circuit.append(U3Gate(0.1, 0.1, 0), [qr[2]])
        ref_circuit.append(U3Gate(0, 0, -pi / 2), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(pi / 2, 0, pi), [qr[1]])
        ref_circuit.append(U3Gate(0.2, 0, 0), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, pi / 2), [qr[2]])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.append(U3Gate(pi / 2, 0, pi), [qr[0]])
        ref_circuit.i(qr[0])
        ref_circuit.append(U3Gate(0.1, -pi / 2, pi / 2), [qr[0]])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(0, 0, 0.6), [qr[0]])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.append(U3Gate(0, 0, pi / 2), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, -pi / 4), [qr[0]])
        ref_circuit.append(U3Gate(pi / 2, 0.2, -0.1), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, pi), [qr[0]])
        ref_circuit.append(U3Gate(0, 0, -pi / 2), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, 0.3), [qr[2]])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[2], qr[1])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.append(U3Gate(0, 0, 0.1), [qr[1]])
        ref_circuit.append(U3Gate(pi, pi / 2, pi / 2), [qr[1]])
        ref_circuit.append(U3Gate(0, 0, pi / 4), [qr[2]])
        ref_circuit.append(U3Gate(0.3, 0.0, -0.1), [qr[2]])
        ref_circuit.append(U3Gate(pi, 0, pi), [qr[2]])
        # ref_circuit.snapshot('0')
        # ref_circuit.measure(qr, cr)
        # ref_dag = circuit_to_dag(ref_circuit)

        self.assertTrue(Operator(dag_to_circuit(unrolled_dag)).equiv(ref_circuit))

    def test_simple_unroll_parameterized_without_expressions(self):
        """Verify unrolling parameterized gates without expressions."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")

        qc.rz(theta, qr[0])
        dag = circuit_to_dag(qc)

        pass_ = UnrollCustomDefinitions(std_eqlib, ["u1", "cx"])
        dag = pass_.run(dag)

        unrolled_dag = BasisTranslator(std_eqlib, ["u1", "cx"]).run(dag)

        expected = QuantumCircuit(qr, global_phase=-theta / 2)
        expected.append(U1Gate(theta), [qr[0]])

        self.assertEqual(circuit_to_dag(expected), unrolled_dag)

    def test_simple_unroll_parameterized_with_expressions(self):
        """Verify unrolling parameterized gates with expressions."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_ = theta + phi

        qc.rz(sum_, qr[0])
        dag = circuit_to_dag(qc)
        pass_ = UnrollCustomDefinitions(std_eqlib, ["p", "cx"])
        dag = pass_.run(dag)

        unrolled_dag = BasisTranslator(std_eqlib, ["p", "cx"]).run(dag)

        expected = QuantumCircuit(qr, global_phase=-sum_ / 2)
        expected.p(sum_, qr[0])

        self.assertEqual(circuit_to_dag(expected), unrolled_dag)

    def test_definition_unroll_parameterized(self):
        """Verify that unrolling complex gates with parameters does not raise."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")

        qc.cp(theta, qr[1], qr[0])
        qc.cp(theta * theta, qr[0], qr[1])
        dag = circuit_to_dag(qc)
        pass_ = UnrollCustomDefinitions(std_eqlib, ["p", "cx"])
        dag = pass_.run(dag)

        out_dag = BasisTranslator(std_eqlib, ["p", "cx"]).run(dag)

        self.assertEqual(out_dag.count_ops(), {"p": 6, "cx": 4})

    def test_unrolling_parameterized_composite_gates(self):
        """Verify unrolling circuits with parameterized composite gates."""
        mock_sel = EquivalenceLibrary(base=std_eqlib)

        qr1 = QuantumRegister(2)
        subqc = QuantumCircuit(qr1)

        theta = Parameter("theta")

        subqc.rz(theta, qr1[0])
        subqc.cx(qr1[0], qr1[1])
        subqc.rz(theta, qr1[1])

        # Expanding across register with shared parameter
        qr2 = QuantumRegister(4)
        qc = QuantumCircuit(qr2)

        sub_instr = circuit_to_instruction(subqc, equivalence_library=mock_sel)
        qc.append(sub_instr, [qr2[0], qr2[1]])
        qc.append(sub_instr, [qr2[2], qr2[3]])

        dag = circuit_to_dag(qc)
        pass_ = UnrollCustomDefinitions(mock_sel, ["p", "cx"])
        dag = pass_.run(dag)

        out_dag = BasisTranslator(mock_sel, ["p", "cx"]).run(dag)

        # Pick up -1 * theta / 2 global phase four twice (once for each RZ -> P
        # in each of the two sub_instr instructions).
        expected = QuantumCircuit(qr2, global_phase=-1 * 4 * theta / 2.0)
        expected.p(theta, qr2[0])
        expected.p(theta, qr2[2])
        expected.cx(qr2[0], qr2[1])
        expected.cx(qr2[2], qr2[3])
        expected.p(theta, qr2[1])
        expected.p(theta, qr2[3])

        self.assertEqual(circuit_to_dag(expected), out_dag)

        # Expanding across register with shared parameter
        qc = QuantumCircuit(qr2)

        phi = Parameter("phi")
        gamma = Parameter("gamma")

        sub_instr = circuit_to_instruction(subqc, {theta: phi}, mock_sel)
        qc.append(sub_instr, [qr2[0], qr2[1]])
        sub_instr = circuit_to_instruction(subqc, {theta: gamma}, mock_sel)
        qc.append(sub_instr, [qr2[2], qr2[3]])

        dag = circuit_to_dag(qc)
        pass_ = UnrollCustomDefinitions(mock_sel, ["p", "cx"])
        dag = pass_.run(dag)

        out_dag = BasisTranslator(mock_sel, ["p", "cx"]).run(dag)

        expected = QuantumCircuit(qr2, global_phase=-1 * (2 * phi + 2 * gamma) / 2.0)
        expected.p(phi, qr2[0])
        expected.p(gamma, qr2[2])
        expected.cx(qr2[0], qr2[1])
        expected.cx(qr2[2], qr2[3])
        expected.p(phi, qr2[1])
        expected.p(gamma, qr2[3])

        self.assertEqual(circuit_to_dag(expected), out_dag)


class TestBasisExamples(QiskitTestCase):
    """Test example circuits targeting example bases over the StandardEquivalenceLibrary."""

    def test_cx_bell_to_cz(self):
        """Verify we can translate a CX bell circuit to CZ,RX,RZ."""
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        in_dag = circuit_to_dag(bell)
        out_dag = BasisTranslator(std_eqlib, ["cz", "rx", "rz"]).run(in_dag)

        self.assertTrue(set(out_dag.count_ops()).issubset(["cz", "rx", "rz"]))
        self.assertEqual(Operator(bell), Operator(dag_to_circuit(out_dag)))

    def test_cx_bell_to_iswap(self):
        """Verify we can translate a CX bell to iSwap,U3."""
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        in_dag = circuit_to_dag(bell)
        out_dag = BasisTranslator(std_eqlib, ["iswap", "u"]).run(in_dag)

        self.assertTrue(set(out_dag.count_ops()).issubset(["iswap", "u"]))
        self.assertEqual(Operator(bell), Operator(dag_to_circuit(out_dag)))

    def test_cx_bell_to_ecr(self):
        """Verify we can translate a CX bell to ECR,U."""
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        in_dag = circuit_to_dag(bell)
        out_dag = BasisTranslator(std_eqlib, ["ecr", "u"]).run(in_dag)

        #         ┌────────────┐   ┌─────────────┐┌──────┐┌──────────┐
        # q_0: ───┤ U(π/2,0,π) ├───┤ U(0,0,-π/2) ├┤0     ├┤ U(π,0,π) ├
        #      ┌──┴────────────┴──┐└─────────────┘│  Ecr │└──────────┘
        # q_1: ┤ U(-π/2,-π/2,π/2) ├───────────────┤1     ├────────────
        #      └──────────────────┘               └──────┘

        qr = QuantumRegister(2, "q")
        expected = QuantumCircuit(2)
        expected.u(pi / 2, 0, pi, qr[0])
        expected.u(0, 0, -pi / 2, qr[0])
        expected.u(-pi / 2, -pi / 2, pi / 2, qr[1])
        expected.ecr(0, 1)
        expected.u(pi, 0, pi, qr[0])
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out_dag, expected_dag)

    def test_cx_bell_to_cp(self):
        """Verify we can translate a CX bell to CP,U."""
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        in_dag = circuit_to_dag(bell)
        out_dag = BasisTranslator(std_eqlib, ["cp", "u"]).run(in_dag)

        self.assertTrue(set(out_dag.count_ops()).issubset(["cp", "u"]))
        self.assertEqual(Operator(bell), Operator(dag_to_circuit(out_dag)))

        qr = QuantumRegister(2, "q")
        expected = QuantumCircuit(qr)
        expected.u(pi / 2, 0, pi, 0)
        expected.u(pi / 2, 0, pi, 1)
        expected.cp(pi, 0, 1)
        expected.u(pi / 2, 0, pi, 1)
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out_dag, expected_dag)

    def test_cx_bell_to_crz(self):
        """Verify we can translate a CX bell to CRZ,U."""
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)

        in_dag = circuit_to_dag(bell)
        out_dag = BasisTranslator(std_eqlib, ["crz", "u"]).run(in_dag)

        self.assertTrue(set(out_dag.count_ops()).issubset(["crz", "u"]))
        self.assertEqual(Operator(bell), Operator(dag_to_circuit(out_dag)))

        qr = QuantumRegister(2, "q")
        expected = QuantumCircuit(qr)
        expected.u(pi / 2, 0, pi, 0)
        expected.u(0, 0, pi / 2, 0)
        expected.u(pi / 2, 0, pi, 1)
        expected.crz(pi, 0, 1)
        expected.u(pi / 2, 0, pi, 1)
        expected_dag = circuit_to_dag(expected)

        self.assertEqual(out_dag, expected_dag)

    def test_global_phase(self):
        """Verify global phase preserved in basis translation"""
        circ = QuantumCircuit(1)
        gate_angle = pi / 5
        circ_angle = pi / 3
        circ.rz(gate_angle, 0)
        circ.global_phase = circ_angle
        in_dag = circuit_to_dag(circ)
        out_dag = BasisTranslator(std_eqlib, ["p"]).run(in_dag)

        qr = QuantumRegister(1, "q")
        expected = QuantumCircuit(qr)
        expected.p(gate_angle, qr)
        expected.global_phase = circ_angle - gate_angle / 2
        expected_dag = circuit_to_dag(expected)
        self.assertEqual(out_dag, expected_dag)
        self.assertAlmostEqual(
            float(out_dag.global_phase), float(expected_dag.global_phase), places=14
        )
        self.assertEqual(Operator(dag_to_circuit(out_dag)), Operator(expected))

    def test_condition_set_substitute_node(self):
        """Verify condition is set in BasisTranslator on substitute_node"""

        #      ┌───┐         ┌───┐
        # q_0: ┤ H ├──■──────┤ H ├─
        #      └───┘┌─┴─┐┌─┐ └─╥─┘
        # q_1: ─────┤ X ├┤M├───╫───
        #           └───┘└╥┘┌──╨──┐
        # c: 2/═══════════╩═╡ 0x1 ╞
        #                 1 └─────┘
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circ = QuantumCircuit(qr, cr)
        circ.h(0)
        circ.cx(0, 1)
        circ.measure(1, 1)
        circ.h(0).c_if(cr, 1)
        circ_transpiled = transpile(circ, optimization_level=3, basis_gates=["cx", "id", "u"])

        #      ┌────────────┐        ┌────────────┐
        # q_0: ┤ U(π/2,0,π) ├──■─────┤ U(π/2,0,π) ├
        #      └────────────┘┌─┴─┐┌─┐└─────╥──────┘
        # q_1: ──────────────┤ X ├┤M├──────╫───────
        #                    └───┘└╥┘   ┌──╨──┐
        # c: 2/════════════════════╩════╡ 0x1 ╞════
        #                          1    └─────┘
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        expected = QuantumCircuit(qr, cr)
        expected.u(pi / 2, 0, pi, 0)
        expected.cx(0, 1)
        expected.measure(1, 1)
        expected.u(pi / 2, 0, pi, 0).c_if(cr, 1)

        self.assertEqual(circ_transpiled, expected)

    def test_skip_target_basis_equivalences_1(self):
        """Test that BasisTranslator skips gates in the target_basis - #6085"""
        circ = QuantumCircuit()
        qasm_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "qasm",
            "TestBasisTranslator_skip_target.qasm",
        )
        circ = circ.from_qasm_file(qasm_file)
        circ_transpiled = transpile(
            circ,
            basis_gates=["id", "rz", "sx", "x", "cx"],
            seed_transpiler=42,
        )
        self.assertEqual(circ_transpiled.count_ops(), {"cx": 91, "rz": 66, "sx": 22})


class TestBasisTranslatorWithTarget(QiskitTestCase):
    """Test the basis translator when running with a Target."""

    def setUp(self):
        super().setUp()
        self.target = Target()

        # U gate in qubit 0.
        self.theta = Parameter("theta")
        self.phi = Parameter("phi")
        self.lam = Parameter("lambda")
        u_props = {
            (0,): InstructionProperties(duration=5.23e-8, error=0.00038115),
        }
        self.target.add_instruction(UGate(self.theta, self.phi, self.lam), u_props)

        # Rz gate in qubit 1.
        rz_props = {
            (1,): InstructionProperties(duration=0.0, error=0),
        }
        self.target.add_instruction(RZGate(self.phi), rz_props)

        # X gate in qubit 1.
        x_props = {
            (1,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00020056469709026198
            ),
        }
        self.target.add_instruction(XGate(), x_props)

        # SX gate in qubit 1.
        sx_props = {
            (1,): InstructionProperties(
                duration=3.5555555555555554e-08, error=0.00020056469709026198
            ),
        }
        self.target.add_instruction(SXGate(), sx_props)

        cx_props = {
            (0, 1): InstructionProperties(duration=5.23e-7, error=0.00098115),
            (1, 0): InstructionProperties(duration=4.52e-7, error=0.00132115),
        }
        self.target.add_instruction(CXGate(), cx_props)

    def test_2q_with_non_global_1q(self):
        """Test translation works with a 2q gate on an non-global 1q basis."""
        qc = QuantumCircuit(2)
        qc.cz(0, 1)

        bt_pass = BasisTranslator(std_eqlib, target_basis=None, target=self.target)
        output = bt_pass(qc)
        # We need a second run of BasisTranslator to correct gates outside of
        # the target basis. This is a known isssue, see:
        #  https://qiskit.org/documentation/release_notes.html#release-notes-0-19-0-known-issues
        output = bt_pass(output)
        expected = QuantumCircuit(2)
        expected.rz(pi, 1)
        expected.sx(1)
        expected.rz(3 * pi / 2, 1)
        expected.sx(1)
        expected.rz(3 * pi, 1)
        expected.cx(0, 1)
        expected.rz(pi, 1)
        expected.sx(1)
        expected.rz(3 * pi / 2, 1)
        expected.sx(1)
        expected.rz(3 * pi, 1)
        self.assertEqual(output, expected)

if __name__ == "__main__":
    # TestBasisTranslator().test_single_substitution()
    unittest.main()
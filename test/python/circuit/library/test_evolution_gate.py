# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the evolution gate."""

from itertools import permutations

import unittest
import numpy as np
import scipy
from ddt import ddt, data, unpack

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import PauliEvolutionGate, HamiltonianGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter, MatrixExponential, QDrift
from qiskit.synthesis.evolution.product_formula import reorder_paulis
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli, Statevector
from qiskit.transpiler.passes import HLSConfig, HighLevelSynthesis
from test import QiskitTestCase  # pylint: disable=wrong-import-order

X = SparsePauliOp("X")
Y = SparsePauliOp("Y")
Z = SparsePauliOp("Z")
I = SparsePauliOp("I")


@ddt
class TestEvolutionGate(QiskitTestCase):
    """Test the evolution gate."""

    def setUp(self):
        super().setUp()
        # fix random seed for reproducibility (used in QDrift)
        self.seed = 2

    def assertSuzukiTrotterIsCorrect(self, gate):
        """Assert the Suzuki Trotter evolution is correct."""
        op = gate.operator
        time = gate.time
        synthesis = gate.synthesis

        exact_suzuki = SuzukiTrotter(
            reps=synthesis.reps, order=synthesis.order, atomic_evolution=exact_atomic_evolution
        )
        exact_gate = PauliEvolutionGate(op, time, synthesis=exact_suzuki)

        self.assertTrue(Operator(gate).equiv(exact_gate))

    def test_matrix_decomposition(self):
        """Test the default decomposition."""
        op = (X ^ X ^ X) + (Y ^ Y ^ Y) + (Z ^ Z ^ Z)
        time = 0.123

        matrix = op.to_matrix()
        evolved = scipy.linalg.expm(-1j * time * matrix)

        evo_gate = PauliEvolutionGate(op, time, synthesis=MatrixExponential())

        self.assertTrue(Operator(evo_gate).equiv(evolved))

    def test_reorder_paulis_invariant(self):
        """
        Tests that reorder_paulis is deterministic and does not depend on the
        order of the terms of the input operator.
        """
        terms = [
            (I ^ I ^ X ^ X),
            (I ^ I ^ Z ^ Z),
            (I ^ Y ^ Y ^ I),
            (X ^ I ^ I ^ I),
            (X ^ X ^ I ^ I),
            (Y ^ I ^ I ^ Y),
        ]
        results = []
        for seed, tms in enumerate(permutations(terms)):
            np.random.seed(seed)
            op = reorder_paulis(SparsePauliOp(sum(tms)).to_sparse_list())
            results.append([(t[0], t[1]) for t in op])
            np.random.seed(seed + 42)
            op = reorder_paulis(SparsePauliOp(sum(tms)).to_sparse_list())
            results.append([(t[0], t[1]) for t in op])

        for lst in results[1:]:
            self.assertListEqual(lst, results[0])

    def test_lie_trotter(self):
        """Test constructing the circuit with Lie Trotter decomposition."""
        op = (X ^ X ^ X) + (Y ^ Y ^ Y) + (Z ^ Z ^ Z)
        time = 0.123
        reps = 4
        evo_gate = PauliEvolutionGate(op, time, synthesis=LieTrotter(reps=reps))
        decomposed = evo_gate.definition.decompose()

        self.assertEqual(decomposed.count_ops()["cx"], reps * 3 * 4)
        self.assertSuzukiTrotterIsCorrect(evo_gate)

    def test_basis_change(self):
        """Test the basis change is correctly implemented."""
        op = I ^ Y  # use a string for which we do not have a basis gate
        time = 0.321
        evo_gate = PauliEvolutionGate(op, time)
        self.assertSuzukiTrotterIsCorrect(evo_gate)

    def test_rzx_order(self):
        """Test ZX and XZ is mapped onto the correct qubits."""

        for op, indices in zip([X ^ Z, Z ^ X], [(0, 1), (1, 0)]):
            with self.subTest(op=op, indices=indices):
                evo_gate = PauliEvolutionGate(op)
                decomposed = evo_gate.definition.decompose()

                #           ┌───┐┌───────┐┌───┐
                # q_0: ─────┤ X ├┤ Rz(2) ├┤ X ├─────
                #      ┌───┐└─┬─┘└───────┘└─┬─┘┌───┐
                # q_1: ┤ H ├──■─────────────■──┤ H ├
                #      └───┘                   └───┘
                ref = QuantumCircuit(2)
                ref.h(indices[1])
                ref.cx(indices[1], indices[0])
                ref.rz(2.0, indices[0])
                ref.cx(indices[1], indices[0])
                ref.h(indices[1])

                # don't use circuit equality since RZX here decomposes with RZ on the bottom
                self.assertTrue(Operator(decomposed).equiv(ref))

    def test_suzuki_trotter(self):
        """Test constructing the circuit with Lie Trotter decomposition."""
        op = (X ^ X ^ X) + (Y ^ Y ^ Y) + (Z ^ Z ^ Z)
        time = 0.123
        reps = 4
        for order in [2, 4, 6]:
            if order == 2:
                expected_cx = reps * 5 * 4
            elif order % 2 == 0:
                # recurse (order - 2) / 2 times, base case has 5 blocks with 4 CX each
                expected_cx = reps * 5 ** ((order - 2) / 2) * 5 * 4
            else:
                # recurse (order - 1) / 2 times, base case has 3 blocks with 4 CX each
                expected_cx = reps * 5 ** ((order - 1) / 2) * 3 * 4

            with self.subTest(order=order):
                evo_gate = PauliEvolutionGate(
                    op, time, synthesis=SuzukiTrotter(order=order, reps=reps)
                )
                decomposed = evo_gate.definition.decompose()
                self.assertEqual(decomposed.count_ops()["cx"], expected_cx)
                self.assertSuzukiTrotterIsCorrect(evo_gate)

    def test_suzuki_trotter_manual_no_reorder(self):
        """Test the evolution circuit of Suzuki Trotter against a manually constructed circuit."""
        op = X + Y
        time = 0.1
        reps = 1
        evo_gate = PauliEvolutionGate(
            op, time, synthesis=SuzukiTrotter(order=4, reps=reps, preserve_order=True)
        )

        # manually construct expected evolution
        expected = QuantumCircuit(1)
        p_4 = 1 / (4 - 4 ** (1 / 3))  # coefficient for reduced time from Suzuki paper
        for _ in range(2):
            # factor of 1/2 already included with factor 1/2 in rotation gate
            expected.rx(p_4 * time, 0)
            expected.ry(2 * p_4 * time, 0)
            expected.rx(p_4 * time, 0)

        expected.rx((1 - 4 * p_4) * time, 0)
        expected.ry(2 * (1 - 4 * p_4) * time, 0)
        expected.rx((1 - 4 * p_4) * time, 0)

        for _ in range(2):
            expected.rx(p_4 * time, 0)
            expected.ry(2 * p_4 * time, 0)
            expected.rx(p_4 * time, 0)

        self.assertEqual(evo_gate.definition, expected)
        self.assertSuzukiTrotterIsCorrect(evo_gate)

    @data(True, False)
    def test_suzuki_trotter_manual(self, use_plugin):
        """Test the evolution circuit of Suzuki Trotter against a manually constructed circuit."""
        op = (X ^ X ^ I ^ I) + (I ^ Y ^ Y ^ I) + (I ^ I ^ Z ^ Z)
        time, reps = 0.1, 1

        synthesis = SuzukiTrotter(order=2, reps=reps)
        if use_plugin:
            hls_config = HLSConfig(PauliEvolution=[("default", {"preserve_order": False})])
        else:
            synthesis.preserve_order = False
            hls_config = None

        evo_gate = PauliEvolutionGate(op, time, synthesis=synthesis)
        circuit = QuantumCircuit(op.num_qubits)
        circuit.append(evo_gate, circuit.qubits)

        if use_plugin:
            decomposed = HighLevelSynthesis(hls_config=hls_config)(circuit)
        else:
            decomposed = circuit.decompose()

        expected = QuantumCircuit(4)
        expected.rzz(time, 0, 1)
        expected.rxx(time, 2, 3)
        expected.ryy(2 * time, 1, 2)
        expected.rxx(time, 2, 3)
        expected.rzz(time, 0, 1)
        self.assertEqual(decomposed, expected)

    def test_suzuki_trotter_plugin(self):
        """Test setting options via the plugin."""

    @data(
        (X + Y, 0.5, 1, [(Pauli("X"), 0.5), (Pauli("X"), 0.5)]),
        (X, 0.238, 2, [(Pauli("X"), 0.238)]),
    )
    @unpack
    def test_qdrift_manual(self, op, time, reps, sampled_ops):
        """Test the evolution circuit of Suzuki Trotter against a manually constructed circuit."""
        qdrift = QDrift(reps=reps, seed=self.seed)
        evo_gate = PauliEvolutionGate(op, time, synthesis=qdrift)
        evo_gate.definition.decompose()

        # manually construct expected evolution
        expected = QuantumCircuit(1)
        for pauli in sampled_ops:
            if pauli[0].to_label() == "X":
                expected.rx(2 * pauli[1], 0)
            elif pauli[0].to_label() == "Y":
                expected.ry(2 * pauli[1], 0)

        self.assertTrue(Operator(evo_gate.definition).equiv(expected))

    def test_qdrift_evolution(self):
        """Test QDrift on an example."""
        op = 0.1 * (Z ^ Z) - 3.2 * (X ^ I) - 1.0 * (I ^ X) + 0.2 * (X ^ X)
        reps = 20
        time = 0.12
        num_samples = 300
        qdrift_energy = []

        def energy(evo):
            return Statevector(evo).expectation_value(op.to_matrix())

        for i in range(num_samples):
            qdrift = PauliEvolutionGate(
                op, time=time, synthesis=QDrift(reps=reps, seed=self.seed + i)
            ).definition

            qdrift_energy.append(energy(qdrift))

        exact = scipy.linalg.expm(-1j * time * op.to_matrix()).dot(np.eye(4)[0, :])

        self.assertAlmostEqual(energy(exact), np.average(qdrift_energy), places=2)

    @data(True, False)
    def test_passing_grouped_paulis(self, wrap):
        """Test passing a list of already grouped Paulis."""
        grouped_ops = [(X ^ Y) + (Y ^ X), (Z ^ I) + (Z ^ Z) + (I ^ Z), (X ^ X)]
        evo_gate = PauliEvolutionGate(grouped_ops, time=0.12, synthesis=LieTrotter(wrap=wrap))
        if wrap:
            decomposed = evo_gate.definition.decompose()
        else:
            decomposed = evo_gate.definition

        self.assertEqual(decomposed.count_ops()["rz"], 4)
        self.assertEqual(decomposed.count_ops()["rzz"], 1)
        self.assertEqual(decomposed.count_ops()["rxx"], 1)

    def test_list_from_grouped_paulis(self):
        """Test getting a string representation from grouped Paulis."""
        grouped_ops = [(X ^ Y) + (Y ^ X), (Z ^ I) + (Z ^ Z) + (I ^ Z), (X ^ X)]
        evo_gate = PauliEvolutionGate(grouped_ops, time=0.12, synthesis=LieTrotter())

        pauli_strings = []
        for op in evo_gate.operator:
            if isinstance(op, SparsePauliOp):
                pauli_strings.append(op.to_list())
            else:
                pauli_strings.append([(str(op), 1 + 0j)])

        expected = [
            [("XY", 1 + 0j), ("YX", 1 + 0j)],
            [("ZI", 1 + 0j), ("ZZ", 1 + 0j), ("IZ", 1 + 0j)],
            [("XX", 1 + 0j)],
        ]
        self.assertListEqual(pauli_strings, expected)

    def test_dag_conversion(self):
        """Test constructing a circuit with evolutions yields a DAG with evolution blocks."""
        time = Parameter("t")
        evo = PauliEvolutionGate((Z ^ Z) + (X ^ X), time=time)

        circuit = QuantumCircuit(2)
        circuit.h(circuit.qubits)
        circuit.append(evo, circuit.qubits)
        circuit.cx(0, 1)

        dag = circuit_to_dag(circuit)

        expected_ops = {"HGate", "CXGate", "PauliEvolutionGate"}
        ops = {node.op.base_class.__name__ for node in dag.op_nodes()}

        self.assertEqual(ops, expected_ops)

    @data("chain", "fountain")
    def test_cnot_chain_options(self, option):
        """Test selecting different kinds of CNOT chains."""

        op = Z ^ Z ^ Z
        synthesis = LieTrotter(reps=1, cx_structure=option)
        evo = PauliEvolutionGate(op, synthesis=synthesis)

        expected = QuantumCircuit(3)
        if option == "chain":
            expected.cx(2, 1)
            expected.cx(1, 0)
        else:
            expected.cx(1, 0)
            expected.cx(2, 0)

        expected.rz(2, 0)

        if option == "chain":
            expected.cx(1, 0)
            expected.cx(2, 1)
        else:
            expected.cx(2, 0)
            expected.cx(1, 0)

        self.assertEqual(expected, evo.definition)
        self.assertSuzukiTrotterIsCorrect(evo)

    @data(
        Pauli("XI"),
        SparsePauliOp(Pauli("XI")),
    )
    def test_different_input_types(self, op):
        """Test all different supported input types and that they yield the same."""
        expected = QuantumCircuit(2)
        expected.rx(4, 1)

        with self.subTest(msg="plain"):
            evo = PauliEvolutionGate(op, time=2, synthesis=LieTrotter())
            self.assertEqual(evo.definition, expected)

        with self.subTest(msg="wrapped in list"):
            evo = PauliEvolutionGate([op], time=2, synthesis=LieTrotter())
            self.assertEqual(evo.definition, expected)

    def test_pauliop_coefficients_respected(self):
        """Test that global ``PauliOp`` coefficients are being taken care of."""
        evo = PauliEvolutionGate(5 * (Z ^ I), time=1, synthesis=LieTrotter())
        circuit = evo.definition.decompose()
        rz_angle = circuit.data[0].operation.params[0]
        self.assertEqual(rz_angle, 10)
        self.assertSuzukiTrotterIsCorrect(evo)

    def test_paulisumop_coefficients_respected(self):
        """Test that global ``PauliSumOp`` coefficients are being taken care of."""
        evo = PauliEvolutionGate(5 * (2 * X + 3 * Y - Z), time=1, synthesis=LieTrotter())
        circuit = evo.definition.decompose()
        rz_angles = [
            circuit.data[0].operation.params[0],  # X
            circuit.data[1].operation.params[0],  # Y
            circuit.data[2].operation.params[0],  # Z
        ]
        self.assertListEqual(rz_angles, [20, 30, -10])
        self.assertSuzukiTrotterIsCorrect(evo)

    def test_lie_trotter_two_qubit_correct_order(self):
        """Test that evolutions on two qubit operators are in the right order.

        Regression test of Qiskit/qiskit-terra#7544.
        """
        operator = I ^ Z ^ Z
        time = 0.5
        lie_trotter = PauliEvolutionGate(operator, time, synthesis=LieTrotter())

        self.assertSuzukiTrotterIsCorrect(lie_trotter)

    def test_lie_trotter_reordered_manual(self):
        """Test the evolution circuit of Lie Trotter against a manually constructed circuit."""
        op = (X ^ I ^ I ^ I) + (X ^ X ^ I ^ I) + (I ^ Y ^ Y ^ I) + (I ^ I ^ Z ^ Z)
        time, reps = 0.1, 1
        evo_gate = PauliEvolutionGate(
            op,
            time,
            synthesis=LieTrotter(reps=reps, preserve_order=False),
        )
        # manually construct expected evolution
        expected = QuantumCircuit(4)
        expected.rxx(2 * time, 2, 3)
        expected.rzz(2 * time, 0, 1)
        expected.rx(2 * time, 3)
        expected.ryy(2 * time, 1, 2)
        self.assertEqual(evo_gate.definition, expected)

    def test_complex_op_raises(self):
        """Test an operator with complex coefficient raises an error."""
        with self.assertRaises(ValueError):
            _ = PauliEvolutionGate(Pauli("iZ"))

    def test_paramtrized_op_raises(self):
        """Test an operator with parametrized coefficient raises an error."""
        with self.assertRaises(ValueError):
            _ = PauliEvolutionGate(SparsePauliOp("Z", np.array(Parameter("t"))))

    @data(LieTrotter, MatrixExponential)
    def test_inverse(self, synth_cls):
        """Test calculating the inverse is correct."""
        evo = PauliEvolutionGate(X + Y, time=0.12, synthesis=synth_cls())

        circuit = QuantumCircuit(1)
        circuit.append(evo, circuit.qubits)
        circuit.append(evo.inverse(), circuit.qubits)

        self.assertTrue(Operator(circuit).equiv(np.identity(2**circuit.num_qubits)))

    def test_labels_and_name(self):
        """Test the name and labels are correct."""
        operators = [X, (X + Y), ((I ^ Z) + (Z ^ I) - 0.2 * (X ^ X))]

        # note: the labels do not show coefficients!
        expected_labels = ["X", "(X + Y)", "(IZ + ZI + XX)"]
        for op, label in zip(operators, expected_labels):
            with self.subTest(op=op, label=label):
                evo = PauliEvolutionGate(op)
                self.assertEqual(evo.name, "PauliEvolution")
                self.assertEqual(evo.label, f"exp(-it {label})")

    def test_atomic_evolution(self):
        """Test a custom atomic_evolution."""

        def atomic_evolution(pauli, time):
            if isinstance(pauli, SparsePauliOp):
                if len(pauli.paulis) != 1:
                    raise ValueError("Unsupported input.")
                time *= np.real(pauli.coeffs[0])
                pauli = pauli.paulis[0]

            cliff = diagonalizing_clifford(pauli)
            chain = cnot_chain(pauli)

            target = None
            for i, pauli_i in enumerate(reversed(pauli.to_label())):
                if pauli_i != "I":
                    target = i
                    break

            definition = QuantumCircuit(pauli.num_qubits)
            definition.compose(cliff, inplace=True)
            definition.compose(chain, inplace=True)
            definition.rz(2 * time, target)
            definition.compose(chain.inverse(), inplace=True)
            definition.compose(cliff.inverse(), inplace=True)

            return definition

        op = (X ^ X ^ X) + (Y ^ Y ^ Y) + (Z ^ Z ^ Z)
        time = 0.123
        reps = 4
        with self.assertWarns(PendingDeprecationWarning):
            evo_gate = PauliEvolutionGate(
                op,
                time,
                synthesis=LieTrotter(reps=reps, atomic_evolution=atomic_evolution),
            )
        decomposed = evo_gate.definition.decompose()
        self.assertEqual(decomposed.count_ops()["cx"], reps * 3 * 4)

    def test_all_identity(self):
        """Test circuit with all identity Paulis works correctly."""
        evo = PauliEvolutionGate(I ^ I, time=1).definition
        expected = QuantumCircuit(2, global_phase=-1)
        self.assertEqual(expected, evo)

    def test_global_phase(self):
        """Test a circuit with parameterized global phase terms.

        Regression test of #13625.
        """
        pauli = (X ^ X) + (I ^ I) + (I ^ X)
        time = Parameter("t")
        evo = PauliEvolutionGate(pauli, time=time)

        expected = QuantumCircuit(2, global_phase=-time)
        expected.rxx(2 * time, 0, 1)
        expected.rx(2 * time, 0)

        with self.subTest(msg="check circuit"):
            self.assertEqual(expected, evo.definition)

        # since all terms in the Pauli operator commute, we can compare to an
        # exact matrix exponential
        time_value = 1.76123
        bound = evo.definition.assign_parameters([time_value])
        exact = scipy.linalg.expm(-1j * time_value * pauli.to_matrix())
        with self.subTest(msg="check correctness"):
            self.assertEqual(Operator(exact), Operator(bound))

    def test_sympify_is_real(self):
        """Test converting the parameters to sympy is real.

        Regression test of #13642, where the parameters in the Pauli evolution had a spurious
        zero complex part. Even though this is not noticable upon binding or printing the parameter,
        it does affect the output of Parameter.sympify.
        """
        time = Parameter("t")
        evo = PauliEvolutionGate(Z, time=time)

        angle = evo.definition.data[0].operation.params[0]
        expected = (2.0 * time).sympify()
        self.assertEqual(expected, angle.sympify())


def exact_atomic_evolution(circuit, pauli, time):
    """An exact atomic evolution for Suzuki-Trotter.

    Note that the Pauli has a x2 coefficient already, hence we evolve for time/2.
    """
    circuit.append(HamiltonianGate(pauli.to_matrix(), time / 2), circuit.qubits)


def diagonalizing_clifford(pauli: Pauli) -> QuantumCircuit:
    """Get the clifford circuit to diagonalize the Pauli operator."""
    cliff = QuantumCircuit(pauli.num_qubits)
    for i, pauli_i in enumerate(reversed(pauli.to_label())):
        if pauli_i == "Y":
            cliff.sx(i)
        elif pauli_i == "X":
            cliff.h(i)

    return cliff


def cnot_chain(pauli: Pauli) -> QuantumCircuit:
    """CX chain.

    For example, for the Pauli with the label 'XYZIX'.

    .. parsed-literal::

                       ┌───┐
        q_0: ──────────┤ X ├
                       └─┬─┘
        q_1: ────────────┼──
                  ┌───┐  │
        q_2: ─────┤ X ├──■──
             ┌───┐└─┬─┘
        q_3: ┤ X ├──■───────
             └─┬─┘
        q_4: ──■────────────

    """

    chain = QuantumCircuit(pauli.num_qubits)
    control, target = None, None

    # iterate over the Pauli's and add CNOTs
    for i, pauli_i in enumerate(pauli.to_label()):
        i = pauli.num_qubits - i - 1
        if pauli_i != "I":
            if control is None:
                control = i
            else:
                target = i

        if control is not None and target is not None:
            chain.cx(control, target)
            control = i
            target = None

    return chain


if __name__ == "__main__":
    unittest.main()

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

"""Test the evolved operator ansatz."""

from ddt import ddt, data
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator, Pauli

from qiskit.circuit.library import HamiltonianGate
from qiskit.circuit.library.n_local import (
    EvolvedOperatorAnsatz,
    evolved_operator_ansatz,
    hamiltonian_variational_ansatz,
)
from qiskit.synthesis.evolution import MatrixExponential
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestEvolvedOperatorAnsatz(QiskitTestCase):
    """Test the evolved operator ansatz."""

    @data(True, False)
    def test_evolved_op_ansatz(self, use_function):
        """Test the default evolution."""
        num_qubits = 3
        ops = [Pauli("Z" * num_qubits), Pauli("Y" * num_qubits), Pauli("X" * num_qubits)]

        if use_function:
            evo = evolved_operator_ansatz(ops, 2)
        else:
            evo = EvolvedOperatorAnsatz(ops, 2)

        parameters = evo.parameters

        reference = QuantumCircuit(num_qubits)
        strings = ["z" * num_qubits, "y" * num_qubits, "x" * num_qubits] * 2
        for string, time in zip(strings, parameters):
            reference.compose(evolve(string, time), inplace=True)

        if not use_function:
            evo = evo.decompose().decompose()

        self.assertEqual(evo, reference)

    @data(True, False)
    def test_custom_evolution(self, use_function):
        """Test using another evolution than the default (e.g. matrix evolution)."""
        op = SparsePauliOp(["ZIX"])
        matrix = np.array(op)
        evolution = MatrixExponential()

        if use_function:
            evo = evolved_operator_ansatz(op, evolution=evolution)
        else:
            evo = EvolvedOperatorAnsatz(op, evolution=evolution)

        parameters = evo.parameters

        reference = QuantumCircuit(3)
        reference.append(HamiltonianGate(matrix, parameters[0]), [0, 1, 2])

        if not use_function:
            evo = evo.decompose().decompose()

        self.assertEqual(evo, reference)

    def test_changing_operators(self):
        """Test rebuilding after the operators changed."""
        ops = [Pauli("X"), Pauli("Y"), Pauli("Z")]
        evo = EvolvedOperatorAnsatz(ops)
        evo.operators = [Pauli("X"), Pauli("Y")]
        parameters = evo.parameters

        reference = QuantumCircuit(1)
        reference.rx(2 * parameters[0], 0)
        reference.ry(2 * parameters[1], 0)

        self.assertEqual(evo.decompose(reps=2), reference)

    @data(True, False)
    def test_invalid_reps(self, use_function):
        """Test setting an invalid number of reps."""
        with self.assertRaises(ValueError):
            if use_function:
                _ = evolved_operator_ansatz(Pauli("X"), reps=-1)
            else:
                _ = EvolvedOperatorAnsatz(Pauli("X"), reps=-1)

    def test_insert_barriers_circuit(self):
        """Test using insert_barriers."""
        evo = EvolvedOperatorAnsatz(Pauli("Z"), reps=4, insert_barriers=True)
        ref = QuantumCircuit(1)
        for parameter in evo.parameters:
            ref.rz(2.0 * parameter, 0)
            ref.barrier()

        self.assertEqual(evo.decompose(reps=2), ref)

    def test_insert_barriers(self):
        """Test using insert_barriers."""
        evo = evolved_operator_ansatz(Pauli("Z"), reps=4, insert_barriers=True)
        ref = QuantumCircuit(1)
        for i, parameter in enumerate(evo.parameters):
            ref.rz(2.0 * parameter, 0)
            if i < evo.num_parameters - 1:
                ref.barrier()

        self.assertEqual(evo, ref)

    def test_empty_build_fails(self):
        """Test setting no operators to evolve raises the appropriate error."""
        evo = EvolvedOperatorAnsatz()
        with self.assertRaises(ValueError):
            _ = evo.draw()

    @data(True, False)
    def test_empty_operator_list(self, use_function):
        """Test setting an empty list of operators to be equal to an empty circuit."""
        if use_function:
            evo = evolved_operator_ansatz([])
        else:
            evo = EvolvedOperatorAnsatz([])

        self.assertEqual(evo, QuantumCircuit())

    @data(True, False)
    def test_matrix_operator(self, use_function):
        """Test passing a quantum_info.Operator uses the HamiltonianGate."""
        unitary = Operator([[0, 1], [1, 0]])

        if use_function:
            evo = evolved_operator_ansatz(unitary, reps=3)
        else:
            evo = EvolvedOperatorAnsatz(unitary, reps=3).decompose()

        self.assertEqual(evo.count_ops()["hamiltonian"], 3)

    def test_flattened(self):
        """Test flatten option is actually flattened."""
        num_qubits = 3
        ops = [Pauli("Z" * num_qubits), Pauli("Y" * num_qubits), Pauli("X" * num_qubits)]
        evo = EvolvedOperatorAnsatz(ops, reps=3, flatten=True)
        self.assertNotIn("hamiltonian", evo.count_ops())
        self.assertNotIn("EvolvedOps", evo.count_ops())
        self.assertNotIn("PauliEvolution", evo.count_ops())

    def test_flattening(self):
        """Test ``flatten`` on the function."""
        operators = [Operator.from_label("X"), SparsePauliOp(["Z"])]

        with self.subTest(flatten=None):
            evo = evolved_operator_ansatz(operators, flatten=None)
            ops = evo.count_ops()
            self.assertIn("hamiltonian", ops)
            self.assertNotIn("PauliEvolution", ops)

        with self.subTest(flatten=True):
            # check we get a warning when trying to flatten a HamiltonianGate,
            # which has an unbound param and cannot be flattened
            with self.assertWarnsRegex(UserWarning, "Cannot flatten"):
                evo = evolved_operator_ansatz(operators, flatten=True)

            ops = evo.count_ops()
            self.assertIn("hamiltonian", ops)
            self.assertNotIn("PauliEvolution", ops)

        with self.subTest(flatten=False):
            evo = evolved_operator_ansatz(operators, flatten=False)
            ops = evo.count_ops()
            self.assertIn("hamiltonian", ops)
            self.assertIn("PauliEvolution", ops)


class TestHamiltonianVariationalAnsatz(QiskitTestCase):
    """Test the hamiltonian_variational_ansatz function.

    This is essentially already tested by the evolved_operator_ansatz, we just need
    to test the additional commuting functionality.
    """

    def test_detect_commutation(self):
        """Test the operator is split into commuting terms."""
        hamiltonian = SparsePauliOp(["XII", "ZZI", "IXI", "IZZ", "IIX"])
        circuit = hamiltonian_variational_ansatz(hamiltonian)

        # this Hamiltonian should be split into 2 commuting groups, hence we get 2 parameters
        self.assertEqual(2, circuit.num_parameters)

    def test_evolution_with_identity(self):
        """Test a Hamiltonian containing an identity term.

        Regression test of #13644.
        """
        hamiltonian = SparsePauliOp(["III", "IZZ", "IXI"])
        ansatz = hamiltonian_variational_ansatz(hamiltonian, reps=1)
        bound = ansatz.assign_parameters([1, 1])  # we have two non-commuting groups, hence 2 params

        expected = QuantumCircuit(3, global_phase=-1)
        expected.rzz(2, 0, 1)
        expected.rx(2, 1)

        self.assertEqual(expected, bound)


def evolve(pauli_string, time):
    """Get the reference evolution circuit for a single Pauli string."""

    num_qubits = len(pauli_string)
    forward = QuantumCircuit(num_qubits)
    for i, pauli in enumerate(pauli_string):
        if pauli == "x":
            forward.h(i)
        elif pauli == "y":
            forward.sx(i)

    for i in range(1, num_qubits):
        forward.cx(num_qubits - i, num_qubits - i - 1)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(forward, inplace=True)
    circuit.rz(2 * time, 0)
    circuit.compose(forward.inverse(), inplace=True)

    return circuit

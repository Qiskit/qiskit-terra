# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the functional Pauli rotations."""

import unittest
from collections import defaultdict
import numpy as np
from ddt import ddt, data, unpack

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    LinearPauliRotations,
    PolynomialPauliRotations,
    PiecewiseLinearPauliRotations,
    PiecewiseLinearPauliRotationsGate,
)
from qiskit.quantum_info import Statevector
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestFunctionalPauliRotations(QiskitTestCase):
    """Test the functional Pauli rotations."""

    def assertFunctionIsCorrect(self, function_circuit, reference, num_ancilla_qubits=None):
        """Assert that ``function_circuit`` implements the reference function ``reference``."""
        if isinstance(function_circuit, QuantumCircuit):
            num_ancilla_qubits = function_circuit.num_ancillas

        num_state_qubits = function_circuit.num_qubits - 1 - num_ancilla_qubits

        circuit = QuantumCircuit(num_state_qubits + 1 + num_ancilla_qubits)
        circuit.h(list(range(num_state_qubits)))
        circuit.compose(function_circuit, list(range(function_circuit.num_qubits)), inplace=True)

        tqc = transpile(circuit, basis_gates=["u", "cx"])
        statevector = Statevector(tqc)
        probabilities = defaultdict(float)
        for i, statevector_amplitude in enumerate(statevector):
            i = bin(i)[2:].zfill(circuit.num_qubits)[num_ancilla_qubits:]
            probabilities[i] += np.real(np.abs(statevector_amplitude) ** 2)

        unrolled_probabilities = []
        unrolled_expectations = []
        for i, probability in probabilities.items():
            x, last_qubit = int(i[1:], 2), i[0]
            if last_qubit == "0":
                expected_amplitude = np.cos(reference(x)) / np.sqrt(2**num_state_qubits)
            else:
                expected_amplitude = np.sin(reference(x)) / np.sqrt(2**num_state_qubits)

            unrolled_probabilities += [probability]
            unrolled_expectations += [np.real(np.abs(expected_amplitude) ** 2)]

        np.testing.assert_almost_equal(unrolled_probabilities, unrolled_expectations)

    @data(
        ([1, 0.1], 3),
        ([0, 0.4, 2], 2),
        ([1, 0.5, 0.2, -0.2, 0.4, 2.5], 5),
    )
    @unpack
    def test_polynomial_function(self, coeffs, num_state_qubits):
        """Test the polynomial rotation."""

        def poly(x):
            res = sum(coeff * x**i for i, coeff in enumerate(coeffs))
            return res

        polynome = PolynomialPauliRotations(num_state_qubits, [2 * coeff for coeff in coeffs])
        self.assertFunctionIsCorrect(polynome, poly)

    def test_polynomial_rotations_mutability(self):
        """Test the mutability of the linear rotations circuit."""

        polynomial_rotations = PolynomialPauliRotations()

        with self.subTest(msg="missing number of state qubits"):
            with self.assertRaises(AttributeError):  # no state qubits set
                _ = str(polynomial_rotations.draw())

        with self.subTest(msg="default setup, just setting number of state qubits"):
            polynomial_rotations.num_state_qubits = 2
            self.assertFunctionIsCorrect(polynomial_rotations, lambda x: x / 2)

        with self.subTest(msg="setting non-default values"):
            polynomial_rotations.coeffs = [0, 1.2 * 2, 0.4 * 2]
            self.assertFunctionIsCorrect(polynomial_rotations, lambda x: 1.2 * x + 0.4 * x**2)

        with self.subTest(msg="changing of all values"):
            polynomial_rotations.num_state_qubits = 4
            polynomial_rotations.coeffs = [1 * 2, 0, 0, -0.5 * 2]
            self.assertFunctionIsCorrect(polynomial_rotations, lambda x: 1 - 0.5 * x**3)

    @data(
        (2, 0.1, 0),
        (4, -2, 2),
        (1, 0, 0),
    )
    @unpack
    def test_linear_function(self, num_state_qubits, slope, offset):
        """Test the linear rotation arithmetic circuit."""

        def linear(x):
            return offset + slope * x

        linear_rotation = LinearPauliRotations(num_state_qubits, slope * 2, offset * 2)
        self.assertFunctionIsCorrect(linear_rotation, linear)

    def test_linear_rotations_mutability(self):
        """Test the mutability of the linear rotations circuit."""

        linear_rotation = LinearPauliRotations()

        with self.subTest(msg="missing number of state qubits"):
            with self.assertRaises(AttributeError):  # no state qubits set
                _ = str(linear_rotation.draw())

        with self.subTest(msg="default setup, just setting number of state qubits"):
            linear_rotation.num_state_qubits = 2
            self.assertFunctionIsCorrect(linear_rotation, lambda x: x / 2)

        with self.subTest(msg="setting non-default values"):
            linear_rotation.slope = -2.3 * 2
            linear_rotation.offset = 1 * 2
            self.assertFunctionIsCorrect(linear_rotation, lambda x: 1 - 2.3 * x)

        with self.subTest(msg="changing all values"):
            linear_rotation.num_state_qubits = 4
            linear_rotation.slope = 0.2 * 2
            linear_rotation.offset = 0.1 * 2
            self.assertFunctionIsCorrect(linear_rotation, lambda x: 0.1 + 0.2 * x)

    @data(
        (1, [0], [1], [0]),
        (2, [0, 2], [-0.5, 1], [2, 1]),
        (3, [0, 2, 5], [1, 0, -1], [0, 2, 2]),
        (2, [1, 2], [1, -1], [2, 1]),
        (3, [0, 1], [1, 0], [0, 1]),
    )
    @unpack
    def test_piecewise_linear_function(self, num_state_qubits, breakpoints, slopes, offsets):
        """Test the piecewise linear rotations."""

        def pw_linear(x):
            for i, point in enumerate(reversed(breakpoints)):
                if x >= point:
                    return offsets[-(i + 1)] + slopes[-(i + 1)] * (x - point)
            return 0

        for use_gate in [False, True]:
            constructor = (
                PiecewiseLinearPauliRotationsGate if use_gate else PiecewiseLinearPauliRotations
            )

            if use_gate:
                # ancilla for the comparator bit and the integer comparator itself
                num_ancillas = int(len(breakpoints) > 1)  # + num_state_qubits - 1
                # num_ancillas = 0
            else:
                num_ancillas = None  # automatically deducted

            with self.subTest(use_gate=use_gate):
                pw_linear_rotations = constructor(
                    num_state_qubits,
                    breakpoints,
                    [2 * slope for slope in slopes],
                    [2 * offset for offset in offsets],
                )

                self.assertFunctionIsCorrect(pw_linear_rotations, pw_linear, num_ancillas)

    def test_piecewise_linear_rotations_mutability(self):
        """Test the mutability of the linear rotations circuit."""

        pw_linear_rotations = PiecewiseLinearPauliRotations()

        with self.subTest(msg="missing number of state qubits"):
            with self.assertRaises(AttributeError):  # no state qubits set
                _ = str(pw_linear_rotations.draw())

        with self.subTest(msg="default setup, just setting number of state qubits"):
            pw_linear_rotations.num_state_qubits = 2
            self.assertFunctionIsCorrect(pw_linear_rotations, lambda x: x / 2)

        with self.subTest(msg="setting non-default values"):
            pw_linear_rotations.breakpoints = [0, 2]
            pw_linear_rotations.slopes = [-1 * 2, 1 * 2]
            pw_linear_rotations.offsets = [0, -1.2 * 2]
            self.assertFunctionIsCorrect(
                pw_linear_rotations, lambda x: -1.2 + (x - 2) if x >= 2 else -x
            )

        with self.subTest(msg="changing all values"):
            pw_linear_rotations.num_state_qubits = 4
            pw_linear_rotations.breakpoints = [1, 3, 6]
            pw_linear_rotations.slopes = [-1 * 2, 1 * 2, -0.2 * 2]
            pw_linear_rotations.offsets = [0, -1.2 * 2, 2 * 2]

            def pw_linear(x):
                if x >= 6:
                    return 2 - 0.2 * (x - 6)
                if x >= 3:
                    return -1.2 + (x - 3)
                if x >= 1:
                    return -(x - 1)
                return 0

            self.assertFunctionIsCorrect(pw_linear_rotations, pw_linear)


if __name__ == "__main__":
    unittest.main()

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the EchoRZXWeylDecomposition pass"""

import unittest
from math import pi
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler.passes.optimization.echo_rzx_weyl_decomposition import (
    EchoRZXWeylDecomposition,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers.fake_provider import Fake27QPulseV1
import qiskit.quantum_info as qi
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestEchoRZXWeylDecomposition(QiskitTestCase):
    """Tests the EchoRZXWeylDecomposition pass."""

    def setUp(self):
        super().setUp()
        # TODO once https://github.com/Qiskit/qiskit/issues/12759 is fixed, replace with
        # backend = GenericBackendV2(num_qubits=27, calibrate_instructions=True,
        # control_flow=True, seed=42)
        # self.inst_map = backend.instruction_schedule_map
        with self.assertWarns(DeprecationWarning):
            self.backend = Fake27QPulseV1()
        self.inst_map = self.backend.defaults().instruction_schedule_map

    def assertRZXgates(self, unitary_circuit, after):
        """Check the number of rzx gates"""
        alpha = TwoQubitWeylDecomposition(unitary_circuit).a
        beta = TwoQubitWeylDecomposition(unitary_circuit).b
        gamma = TwoQubitWeylDecomposition(unitary_circuit).c

        expected_rzx_number = 0
        if not alpha == 0:
            expected_rzx_number += 2
        if not beta == 0:
            expected_rzx_number += 2
        if not gamma == 0:
            expected_rzx_number += 2

        circuit_rzx_number = QuantumCircuit.count_ops(after)["rzx"]

        self.assertEqual(expected_rzx_number, circuit_rzx_number)

    @staticmethod
    def count_gate_number(gate, circuit):
        """Count the number of a specific gate type in a circuit"""
        if gate not in QuantumCircuit.count_ops(circuit):
            gate_number = 0
        else:
            gate_number = QuantumCircuit.count_ops(circuit)[gate]
        return gate_number

    def test_rzx_number_native_weyl_decomposition(self):
        """Check the number of RZX gates for a hardware-native cx"""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])

        unitary_circuit = qi.Operator(circuit).data
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The entire Qiskit Pulse package",
        ):
            after = EchoRZXWeylDecomposition(self.inst_map)(circuit)

        unitary_after = qi.Operator(after).data

        self.assertTrue(np.allclose(unitary_circuit, unitary_after))

        # check whether the after circuit has the correct number of rzx gates.
        self.assertRZXgates(unitary_circuit, after)

    def test_h_number_non_native_weyl_decomposition_1(self):
        """Check the number of added Hadamard gates for a native and non-native rzz gate"""
        theta = pi / 11
        qr = QuantumRegister(2, "qr")
        # rzz gate in native direction
        circuit = QuantumCircuit(qr)
        circuit.rzz(theta, qr[0], qr[1])

        # rzz gate in non-native direction
        circuit_non_native = QuantumCircuit(qr)
        circuit_non_native.rzz(theta, qr[1], qr[0])

        dag = circuit_to_dag(circuit)
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The entire Qiskit Pulse package",
        ):
            pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after = dag_to_circuit(pass_.run(dag))

        dag_non_native = circuit_to_dag(circuit_non_native)
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The entire Qiskit Pulse package",
        ):
            pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after_non_native = dag_to_circuit(pass_.run(dag_non_native))

        circuit_rzx_number = self.count_gate_number("rzx", after)

        circuit_h_number = self.count_gate_number("h", after)
        circuit_non_native_h_number = self.count_gate_number("h", after_non_native)

        # for each pair of rzx gates four hadamard gates have to be added in
        # the case of a non-hardware-native directed gate.
        self.assertEqual(
            (circuit_rzx_number / 2) * 4, circuit_non_native_h_number - circuit_h_number
        )

    def test_h_number_non_native_weyl_decomposition_2(self):
        """Check the number of added Hadamard gates for a swap gate"""
        qr = QuantumRegister(2, "qr")
        # swap gate in native direction.
        circuit = QuantumCircuit(qr)
        circuit.swap(qr[0], qr[1])

        # swap gate in non-native direction.
        circuit_non_native = QuantumCircuit(qr)
        circuit_non_native.swap(qr[1], qr[0])

        dag = circuit_to_dag(circuit)
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The entire Qiskit Pulse package",
        ):
            pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after = dag_to_circuit(pass_.run(dag))

        dag_non_native = circuit_to_dag(circuit_non_native)
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The entire Qiskit Pulse package",
        ):
            pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after_non_native = dag_to_circuit(pass_.run(dag_non_native))

        circuit_rzx_number = self.count_gate_number("rzx", after)

        circuit_h_number = self.count_gate_number("h", after)
        circuit_non_native_h_number = self.count_gate_number("h", after_non_native)

        # for each pair of rzx gates four hadamard gates have to be added in
        # the case of a non-hardware-native directed gate.
        self.assertEqual(
            (circuit_rzx_number / 2) * 4, circuit_non_native_h_number - circuit_h_number
        )

    def test_weyl_decomposition_gate_angles(self):
        """Check the number and angles of the RZX gates for different gates"""
        thetas = [pi / 9, 2.1, -0.2]

        qr = QuantumRegister(2, "qr")
        circuit_rxx = QuantumCircuit(qr)
        circuit_rxx.rxx(thetas[0], qr[1], qr[0])

        circuit_ryy = QuantumCircuit(qr)
        circuit_ryy.ryy(thetas[1], qr[0], qr[1])

        circuit_rzz = QuantumCircuit(qr)
        circuit_rzz.rzz(thetas[2], qr[1], qr[0])

        circuits = [circuit_rxx, circuit_ryy, circuit_rzz]

        for circuit in circuits:

            unitary_circuit = qi.Operator(circuit).data

            dag = circuit_to_dag(circuit)
            with self.assertWarnsRegex(
                DeprecationWarning,
                expected_regex="The entire Qiskit Pulse package",
            ):
                pass_ = EchoRZXWeylDecomposition(self.inst_map)
            after = dag_to_circuit(pass_.run(dag))
            dag_after = circuit_to_dag(after)

            unitary_after = qi.Operator(after).data

            # check whether the unitaries are equivalent.
            self.assertTrue(np.allclose(unitary_circuit, unitary_after))

            # check whether the after circuit has the correct number of rzx gates.
            self.assertRZXgates(unitary_circuit, after)

            alpha = TwoQubitWeylDecomposition(unitary_circuit).a

            rzx_angles = []
            for node in dag_after.two_qubit_ops():
                if node.name == "rzx":
                    rzx_angle = node.op.params[0]
                    # check whether the absolute values of the RZX gate angles
                    # are equivalent to the corresponding Weyl parameter.
                    self.assertAlmostEqual(np.abs(rzx_angle), alpha)
                    rzx_angles.append(rzx_angle)

            # check whether the angles of every RZX gate pair of an echoed RZX gate
            # have opposite signs.
            for idx in range(1, len(rzx_angles), 2):
                self.assertAlmostEqual(rzx_angles[idx - 1], -rzx_angles[idx])

    def test_weyl_unitaries_random_circuit(self):
        """Weyl decomposition for a random two-qubit circuit."""
        theta = pi / 9
        epsilon = 5
        delta = -1
        eta = 0.2
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)

        # random two-qubit circuit.
        circuit.rzx(theta, 0, 1)
        circuit.rzz(epsilon, 0, 1)
        circuit.rz(eta, 0)
        circuit.swap(1, 0)
        circuit.h(0)
        circuit.rzz(delta, 1, 0)
        circuit.swap(0, 1)
        circuit.cx(1, 0)
        circuit.swap(0, 1)
        circuit.h(1)
        circuit.rxx(theta, 0, 1)
        circuit.ryy(theta, 1, 0)
        circuit.ecr(0, 1)

        unitary_circuit = qi.Operator(circuit).data

        dag = circuit_to_dag(circuit)
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The entire Qiskit Pulse package",
        ):
            pass_ = EchoRZXWeylDecomposition(self.inst_map)
        after = dag_to_circuit(pass_.run(dag))

        unitary_after = qi.Operator(after).data

        self.assertTrue(np.allclose(unitary_circuit, unitary_after))


if __name__ == "__main__":
    unittest.main()

# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the DropNegligible transpiler pass."""

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit, QuantumRegister, Gate
from qiskit.circuit.library import (
    CPhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    XXMinusYYGate,
    XXPlusYYGate,
    GlobalPhaseGate,
)
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes import RemoveIdentityEquivalent
from qiskit.transpiler.target import Target, InstructionProperties

from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestDropNegligible(QiskitTestCase):
    """Test the DropNegligible pass."""

    def test_drops_negligible_gates(self):
        """Test that negligible gates are dropped."""
        qubits = QuantumRegister(2)
        circuit = QuantumCircuit(qubits)
        a, b = qubits
        circuit.append(CPhaseGate(1e-5), [a, b])
        circuit.append(CPhaseGate(1e-8), [a, b])
        circuit.append(RXGate(1e-5), [a])
        circuit.append(RXGate(1e-8), [a])
        circuit.append(RYGate(1e-5), [a])
        circuit.append(RYGate(1e-8), [a])
        circuit.append(RZGate(1e-5), [a])
        circuit.append(RZGate(1e-8), [a])
        circuit.append(RXXGate(1e-5), [a, b])
        circuit.append(RXXGate(1e-8), [a, b])
        circuit.append(RYYGate(1e-5), [a, b])
        circuit.append(RYYGate(1e-8), [a, b])
        circuit.append(RZZGate(1e-5), [a, b])
        circuit.append(RZZGate(1e-8), [a, b])
        circuit.append(XXPlusYYGate(1e-5, 1e-8), [a, b])
        circuit.append(XXPlusYYGate(1e-8, 1e-8), [a, b])
        circuit.append(XXMinusYYGate(1e-5, 1e-8), [a, b])
        circuit.append(XXMinusYYGate(1e-8, 1e-8), [a, b])
        transpiled = RemoveIdentityEquivalent()(circuit)
        self.assertEqual(circuit.count_ops()["cp"], 2)
        self.assertEqual(transpiled.count_ops()["cp"], 1)
        self.assertEqual(circuit.count_ops()["rx"], 2)
        self.assertEqual(transpiled.count_ops()["rx"], 1)
        self.assertEqual(circuit.count_ops()["ry"], 2)
        self.assertEqual(transpiled.count_ops()["ry"], 1)
        self.assertEqual(circuit.count_ops()["rz"], 2)
        self.assertEqual(transpiled.count_ops()["rz"], 1)
        self.assertEqual(circuit.count_ops()["rxx"], 2)
        self.assertEqual(transpiled.count_ops()["rxx"], 1)
        self.assertEqual(circuit.count_ops()["ryy"], 2)
        self.assertEqual(transpiled.count_ops()["ryy"], 1)
        self.assertEqual(circuit.count_ops()["rzz"], 2)
        self.assertEqual(transpiled.count_ops()["rzz"], 1)
        self.assertEqual(circuit.count_ops()["xx_plus_yy"], 2)
        self.assertEqual(transpiled.count_ops()["xx_plus_yy"], 1)
        self.assertEqual(circuit.count_ops()["xx_minus_yy"], 2)
        self.assertEqual(transpiled.count_ops()["xx_minus_yy"], 1)
        np.testing.assert_allclose(
            np.array(Operator(circuit)), np.array(Operator(transpiled)), atol=1e-7
        )

    def test_handles_parameters(self):
        """Test that gates with parameters are ignored gracefully."""
        qubits = QuantumRegister(2)
        circuit = QuantumCircuit(qubits)
        a, b = qubits
        theta = Parameter("theta")
        circuit.append(CPhaseGate(theta), [a, b])
        circuit.append(CPhaseGate(1e-5), [a, b])
        circuit.append(CPhaseGate(1e-8), [a, b])
        transpiled = RemoveIdentityEquivalent()(circuit)
        self.assertEqual(circuit.count_ops()["cp"], 3)
        self.assertEqual(transpiled.count_ops()["cp"], 2)

    def test_approximation_degree(self):
        """Test that approximation degree handled correctly."""
        qubits = QuantumRegister(2)
        circuit = QuantumCircuit(qubits)
        a, b = qubits
        circuit.append(CPhaseGate(1e-4), [a, b])
        # fidelity 0.9999850001249996 which is above the threshold and not excluded
        # so 1e-2 is the only gate remaining
        circuit.append(CPhaseGate(1e-2), [a, b])
        circuit.append(CPhaseGate(1e-20), [a, b])
        transpiled = RemoveIdentityEquivalent(approximation_degree=0.9999999)(circuit)
        self.assertEqual(circuit.count_ops()["cp"], 3)
        self.assertEqual(transpiled.count_ops()["cp"], 1)
        self.assertEqual(transpiled.data[0].operation.params[0], 1e-2)

    def test_target_approx_none(self):
        """Test error rate with target."""

        target = Target()
        props = {(0, 1): InstructionProperties(error=1e-10)}
        target.add_instruction(CPhaseGate(Parameter("theta")), props)
        circuit = QuantumCircuit(2)
        circuit.append(CPhaseGate(1e-4), [0, 1])
        circuit.append(CPhaseGate(1e-2), [0, 1])
        circuit.append(CPhaseGate(1e-20), [0, 1])
        transpiled = RemoveIdentityEquivalent(approximation_degree=None, target=target)(circuit)
        self.assertEqual(circuit.count_ops()["cp"], 3)
        self.assertEqual(transpiled.count_ops()["cp"], 2)

    def test_target_approx_approx_degree(self):
        """Test error rate with target."""

        target = Target()
        props = {(0, 1): InstructionProperties(error=1e-10)}
        target.add_instruction(CPhaseGate(Parameter("theta")), props)
        circuit = QuantumCircuit(2)
        circuit.append(CPhaseGate(1e-4), [0, 1])
        circuit.append(CPhaseGate(1e-2), [0, 1])
        circuit.append(CPhaseGate(1e-20), [0, 1])
        transpiled = RemoveIdentityEquivalent(approximation_degree=0.9999999, target=target)(
            circuit
        )
        self.assertEqual(circuit.count_ops()["cp"], 3)
        self.assertEqual(transpiled.count_ops()["cp"], 2)

    def test_custom_gate_no_matrix(self):
        """Test that opaque gates are ignored."""

        class CustomOpaqueGate(Gate):
            """Custom opaque gate."""

            def __init__(self):
                super().__init__("opaque", 2, [])

        qc = QuantumCircuit(3)
        qc.append(CustomOpaqueGate(), [0, 1])
        transpiled = RemoveIdentityEquivalent()(qc)
        self.assertEqual(qc, transpiled)

    def test_custom_gate_identity_matrix(self):
        """Test that custom gates with matrix are evaluated."""

        class CustomGate(Gate):
            """Custom gate."""

            def __init__(self):
                super().__init__("custom", 3, [])

            def to_matrix(self):
                return np.eye(8, dtype=complex)

        qc = QuantumCircuit(3)
        qc.append(CustomGate(), [0, 1, 2])
        transpiled = RemoveIdentityEquivalent()(qc)
        expected = QuantumCircuit(3)
        self.assertEqual(expected, transpiled)

    def test_global_phase_ignored(self):
        """Test that global phase gate isn't considered."""

        qc = QuantumCircuit(1)
        qc.id(0)
        qc.append(GlobalPhaseGate(0))
        transpiled = RemoveIdentityEquivalent()(qc)
        expected = QuantumCircuit(1)
        expected.append(GlobalPhaseGate(0))
        self.assertEqual(transpiled, expected)

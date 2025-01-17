# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the LightCone pass"""

import unittest

import numpy as np

from qiskit.circuit import ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Operator, Pauli
from qiskit.transpiler.passes import RemoveFinalMeasurements
from qiskit.transpiler.passes.optimization.light_cone import LightCone
from qiskit.transpiler.passmanager import PassManager
from test import QiskitTestCase  # pylint: disable=wrong-import-order

# Missing cases:
#   - Rotational gates with notable angles (0, np.pi);
#   - Light-cone with parametrized gates, see below and issue #12790.


class TestLightConePass(QiskitTestCase):
    """Test the LightCone pass."""

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.num_qubits = 4

    def test_parametrized_Z_observable(self):
        """Test the LightCone pass with a single Z observable."""

        qc = RealAmplitudes(self.num_qubits, entanglement="pairwise", reps=3).decompose()
        pm = PassManager([LightCone(Pauli("I" * (self.num_qubits - 1) + "Z"))])
        new_circuit = pm.run(qc)

        qr = QuantumRegister(self.num_qubits)
        expected = QuantumCircuit(qr)
        theta = ParameterVector("θ", self.num_qubits**2)

        expected.ry(theta[0], qr[0])
        expected.ry(theta[1], qr[1])
        expected.cx(qr[0], qr[1])
        expected.ry(theta[4], qr[0])
        expected.ry(theta[2], qr[2])
        expected.ry(theta[3], qr[3])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[1], qr[2])
        expected.ry(theta[5], qr[1])
        expected.cx(qr[0], qr[1])
        expected.ry(theta[6], qr[2])
        expected.ry(theta[7], qr[3])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[1], qr[2])
        expected.ry(theta[10], qr[2])
        expected.ry(theta[11], qr[3])
        expected.cx(qr[2], qr[3])
        expected.ry(theta[15], qr[3])

        params = np.random.rand(11)
        new_circuit.assign_parameters(params, inplace=True)
        expected.assign_parameters(params, inplace=True)

        self.assertEqual(Operator(expected), Operator(new_circuit))

    def test_parametrized_doubleZ_observable(self):
        """Test the LightCone pass with a double Z observable."""

        qc = RealAmplitudes(self.num_qubits, entanglement="pairwise", reps=3).decompose()
        pm = PassManager([LightCone(Pauli("Z" + "I" * (self.num_qubits - 2) + "Z"))])
        new_circuit = pm.run(qc)

        qr = QuantumRegister(self.num_qubits)
        expected = QuantumCircuit(qr)
        theta = ParameterVector("θ", self.num_qubits**2)

        expected.ry(theta[0], qr[0])
        expected.ry(theta[1], qr[1])
        expected.cx(qr[0], qr[1])
        expected.ry(theta[4], qr[0])
        expected.ry(theta[2], qr[2])
        expected.ry(theta[3], qr[3])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[1], qr[2])
        expected.ry(theta[5], qr[1])
        expected.cx(qr[0], qr[1])
        expected.ry(theta[8], qr[0])
        expected.ry(theta[6], qr[2])
        expected.ry(theta[7], qr[3])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[1], qr[2])
        expected.ry(theta[9], qr[1])
        expected.cx(qr[0], qr[1])
        expected.ry(theta[12], qr[0])
        expected.ry(theta[10], qr[2])
        expected.ry(theta[11], qr[3])
        expected.cx(qr[2], qr[3])
        expected.ry(theta[15], qr[3])

        params = np.random.rand(14)
        new_circuit.assign_parameters(params, inplace=True)
        expected.assign_parameters(params, inplace=True)

        self.assertEqual(Operator(expected), Operator(new_circuit))

    def test_parametrized_measurement(self):
        """Test the LightCone pass with measurements."""
        qc = RealAmplitudes(self.num_qubits, entanglement="pairwise", reps=3).decompose()
        qc.add_register(ClassicalRegister(2))
        qc.measure(1, 0)
        qc.measure(3, 1)
        # Remove final measurements for the equivalence check with `Operator`
        pm = PassManager([LightCone(), RemoveFinalMeasurements()])
        new_circuit = pm.run(qc)

        qr = QuantumRegister(self.num_qubits)
        expected = QuantumCircuit(qr)

        theta = ParameterVector("θ", self.num_qubits**2)

        expected.ry(theta[0], qr[0])
        expected.ry(theta[1], qr[1])
        expected.cx(qr[0], qr[1])
        expected.ry(theta[4], qr[0])
        expected.ry(theta[2], qr[2])
        expected.ry(theta[3], qr[3])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[1], qr[2])
        expected.ry(theta[5], qr[1])
        expected.cx(qr[0], qr[1])
        expected.ry(theta[8], qr[0])
        expected.ry(theta[6], qr[2])
        expected.ry(theta[7], qr[3])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[1], qr[2])
        expected.ry(theta[9], qr[1])
        expected.cx(qr[0], qr[1])
        expected.ry(theta[10], qr[2])
        expected.ry(theta[11], qr[3])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[1], qr[2])
        expected.ry(theta[13], qr[1])
        expected.ry(theta[15], qr[3])

        params = np.random.rand(14)
        new_circuit.assign_parameters(params, inplace=True)
        expected.assign_parameters(params, inplace=True)

        self.assertEqual(Operator(expected), Operator(new_circuit))

    def test_bounded_complex_observable(self):
        """Test the LightCone pass with a weight-two Y observable.
        For now, this test uses a bounded circuit; after fixing
        https://github.com/Qiskit/qiskit/issues/12790 the parametrised version can be used.
        For that, it suffices to move the binding before comparison instead of having it
        before the `PassManager` run.
        """

        qc = RealAmplitudes(self.num_qubits, entanglement="pairwise", reps=3).decompose()
        pm = PassManager([LightCone(Pauli("Y" + ("I" * (self.num_qubits - 2)) + "Y"))])
        params = np.random.rand(16)
        qc.assign_parameters(params, inplace=True)
        new_circuit = pm.run(qc)

        qr = QuantumRegister(self.num_qubits)
        expected = QuantumCircuit(qr)

        expected.ry(params[0], qr[0])
        expected.ry(params[1], qr[1])
        expected.ry(params[2], qr[2])
        expected.ry(params[3], qr[3])
        expected.cx(qr[0], qr[1])
        expected.cx(qr[2], qr[3])
        expected.ry(params[4], qr[0])
        expected.cx(qr[1], qr[2])
        expected.ry(params[5], qr[1])
        expected.ry(params[6], qr[2])
        expected.ry(params[7], qr[3])
        expected.cx(qr[0], qr[1])
        expected.cx(qr[2], qr[3])
        expected.ry(params[8], qr[0])
        expected.cx(qr[1], qr[2])
        expected.ry(params[9], qr[1])
        expected.ry(params[10], qr[2])
        expected.ry(params[11], qr[3])
        expected.cx(qr[0], qr[1])
        expected.cx(qr[2], qr[3])

        self.assertEqual(Operator(expected), Operator(new_circuit))

    def test_all_commuting(self):
        """Test the LightCone pass for a circuit that fully commutes with an observable."""

        qr = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(qr)
        pm = PassManager([LightCone(Pauli("Z" + "I" * (self.num_qubits - 1)))])

        qc.s(qr[0])
        qc.z(qr[0])
        qc.h(qr[2])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[2], qr[3])

        new_circuit = pm.run(qc)

        self.assertEqual(sum(new_circuit.count_ops().values()), 0)

    def test_commuting_block(self):
        """Test the LightCone pass for a commuting block: currently gates are checked
        one by one and commuting blocks are thus ignored.
        """

        qr = QuantumRegister(self.num_qubits + 1)
        qc = QuantumCircuit(qr)
        pm = PassManager(
            [LightCone(Pauli("I" * (self.num_qubits - 2) + "X" + "I" * (self.num_qubits - 2)))]
        )

        qc.cx(qr[2], qr[1])
        qc.cx(qr[3], qr[4])
        qc.cx(qr[1], qr[0])
        qc.cx(qr[2], qr[3])
        qc.cx(qr[2], qr[1])
        qc.x(qr[1])
        qc.cx(qr[2], qr[1])

        qr = QuantumRegister(self.num_qubits + 1)
        expected = QuantumCircuit(qr)

        expected.cx(qr[2], qr[1])
        expected.cx(qr[3], qr[4])
        expected.cx(qr[1], qr[0])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[2], qr[1])
        expected.cx(qr[2], qr[1])

        new_circuit = pm.run(qc)

        self.assertEqual(Operator(expected), Operator(new_circuit))


if __name__ == "__main__":
    unittest.main()

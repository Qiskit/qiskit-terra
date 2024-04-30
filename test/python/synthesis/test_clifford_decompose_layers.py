# This code is part of Qiskit.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Tests for Clifford synthesis methods."""

import unittest
from ddt import ddt

import numpy as np

from qiskit.converters import dag_to_circuit
from qiskit.quantum_info.operators import Clifford
from qiskit.quantum_info import random_clifford
from qiskit.synthesis.clifford import synth_clifford_layers, synth_clifford_depth_lnn
from qiskit.synthesis.linear.linear_circuits_utils import check_lnn_connectivity

from test import combine  # pylint: disable=wrong-import-order
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestCliffordDecomposeLayers(QiskitTestCase):
    """Tests for clifford advanced decomposition functions."""

    @combine(num_qubits=[4, 5, 6, 7])
    def test_decompose_clifford(self, num_qubits):
        """Create layer decomposition for a Clifford U, and check that it
        results in an equivalent Clifford."""
        rng = np.random.default_rng(1234)
        samples = 10
        for use_dag in [True, False]:
            with self.subTest(use_dag=use_dag):
                for _ in range(samples):
                    cliff = random_clifford(num_qubits, seed=rng)
                    if use_dag:
                        circ = dag_to_circuit(
                            synth_clifford_layers(cliff, validate=True, use_dag=True)
                        )
                    else:
                        circ = synth_clifford_layers(cliff, validate=True)
                    cliff_cliff = Clifford(circ)
                    self.assertEqual(cliff, cliff_cliff)
                    # Verify the layered structure
                    self.assertEqual(circ.data[0].operation.name, "S2")
                    self.assertEqual(circ.data[1].operation.name, "CZ")
                    self.assertEqual(circ.data[2].operation.name, "CX_dg")
                    self.assertEqual(circ.data[3].operation.name, "H2")
                    self.assertEqual(circ.data[4].operation.name, "S1")
                    self.assertEqual(circ.data[5].operation.name, "CZ")
                    self.assertEqual(circ.data[6].operation.name, "H1")
                    self.assertEqual(circ.data[7].operation.name, "Pauli")

    @combine(num_qubits=[4, 5, 6, 7])
    def test_decompose_lnn_depth(self, num_qubits):
        """Test layered decomposition for linear-nearest-neighbour (LNN) connectivity."""
        rng = np.random.default_rng(1234)
        samples = 10
        for use_dag in [True, False]:
            with self.subTest(use_dag=use_dag):
                for _ in range(samples):
                    cliff = random_clifford(num_qubits, seed=rng)
                    if use_dag:
                        circ = dag_to_circuit(synth_clifford_depth_lnn(cliff, use_dag=True))
                    else:
                        circ = synth_clifford_depth_lnn(cliff)

                    # Check that the Clifford circuit 2-qubit depth is bounded by 7*n+2
                    depth2q = (circ.decompose()).depth(
                        filter_function=lambda x: x.operation.num_qubits == 2
                    )
                    self.assertTrue(depth2q <= 7 * num_qubits + 2)
                    # Check that the Clifford circuit has linear nearest neighbour connectivity
                    self.assertTrue(check_lnn_connectivity(circ.decompose()))
                    cliff_cliff = Clifford(circ)
                    self.assertEqual(cliff, cliff_cliff)
                    # Verify the layered structure
                    self.assertEqual(circ.data[0].operation.name, "S2")
                    self.assertEqual(circ.data[2].operation.name, "H2")
                    self.assertEqual(circ.data[3].operation.name, "S1")
                    self.assertEqual(circ.data[5].operation.name, "H1")
                    self.assertEqual(circ.data[6].operation.name, "Pauli")


if __name__ == "__main__":
    unittest.main()

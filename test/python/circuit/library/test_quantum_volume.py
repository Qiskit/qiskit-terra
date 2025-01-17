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

"""Test library of quantum volume circuits."""

import unittest

from test.utils.base import QiskitTestCase
from qiskit.circuit.library import QuantumVolume
from qiskit.circuit.library.quantum_volume import quantum_volume


class TestQuantumVolumeLibrary(QiskitTestCase):
    """Test library of quantum volume quantum circuits."""

    def test_qv_seed_reproducibility(self):
        """Test qv circuit."""
        left = QuantumVolume(4, 4, seed=28, classical_permutation=False)
        right = QuantumVolume(4, 4, seed=28, classical_permutation=False)
        self.assertEqual(left, right)

        left = QuantumVolume(4, 4, seed=3, classical_permutation=True)
        right = QuantumVolume(4, 4, seed=3, classical_permutation=True)
        self.assertEqual(left, right)

        left = QuantumVolume(4, 4, seed=2024, flatten=True)
        right = QuantumVolume(4, 4, seed=2024, flatten=True)
        self.assertEqual(left, right)

    def test_qv_function_seed_reproducibility(self):
        """Test qv circuit."""
        left = quantum_volume(10, 10, seed=128)
        right = quantum_volume(10, 10, seed=128)
        self.assertEqual(left, right)

        left = quantum_volume(10, 10, seed=256)
        right = quantum_volume(10, 10, seed=256)
        self.assertEqual(left, right)

        left = quantum_volume(10, 10, seed=4196)
        right = quantum_volume(10, 10, seed=4196)
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()

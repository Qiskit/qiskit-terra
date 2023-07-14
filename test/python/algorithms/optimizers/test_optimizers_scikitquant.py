# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test of scikit-quant optimizers."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

from ddt import ddt, data, unpack

import numpy as np
from qiskit import BasicAer
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.opflow import PauliSumOp
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import BOBYQA, SNOBFIT, IMFIL


@ddt
class TestOptimizers(QiskitAlgorithmsTestCase):
    """Test scikit-quant optimizers."""

    def setUp(self):
        """Set the problem."""
        super().setUp()
        algorithm_globals.random_seed = 50
        with self.assertWarns(DeprecationWarning):
            self.qubit_op = PauliSumOp.from_list(
                [
                    ("II", -1.052373245772859),
                    ("IZ", 0.39793742484318045),
                    ("ZI", -0.39793742484318045),
                    ("ZZ", -0.01128010425623538),
                    ("XX", 0.18093119978423156),
                ]
            )

    def _optimize(self, optimizer):
        """launch vqe"""
        with self.assertWarns(DeprecationWarning):
            qe = QuantumInstance(
                BasicAer.get_backend("statevector_simulator"),
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )
        with self.assertWarns(DeprecationWarning):
            vqe = VQE(ansatz=RealAmplitudes(), optimizer=optimizer, quantum_instance=qe)
            result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)

        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=1)

    def test_bobyqa(self):
        """BOBYQA optimizer test."""
        try:
            optimizer = BOBYQA(maxiter=150)
            self._optimize(optimizer)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    @unittest.skipIf(
        tuple(map(int, np.__version__.split("."))) >= (1, 24, 0),
        "scikit's SnobFit currently incompatible with NumPy 1.24.0.",
    )
    def test_snobfit(self):
        """SNOBFIT optimizer test."""
        try:
            optimizer = SNOBFIT(maxiter=100, maxfail=100, maxmp=20)
            self._optimize(optimizer)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    @unittest.skipIf(
        tuple(map(int, np.__version__.split("."))) >= (1, 24, 0),
        "scikit's SnobFit currently incompatible with NumPy 1.24.0.",
    )
    @data((None,), ([(-1, 1), (None, None)],))
    @unpack
    def test_snobfit_missing_bounds(self, bounds):
        """SNOBFIT optimizer test with missing bounds."""
        try:
            optimizer = SNOBFIT()
            with self.assertRaises(ValueError):
                optimizer.minimize(
                    fun=lambda _: 1,  # using dummy function (never called)
                    x0=[0.1, 0.1],  # dummy initial point
                    bounds=bounds,
                )
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_imfil(self):
        """IMFIL test."""
        try:
            optimizer = IMFIL(maxiter=100)
            self._optimize(optimizer)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))


if __name__ == "__main__":
    unittest.main()

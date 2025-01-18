# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for BackendEstimator."""

import unittest
from unittest.mock import patch
from multiprocessing import Manager
import numpy as np
from ddt import ddt

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import BackendEstimator, EstimatorResult
from qiskit.providers.fake_provider import Fake7QPulseV1, GenericBackendV2
from qiskit.providers.backend_compat import BackendV2Converter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.utils import optionals
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from test import combine  # pylint: disable=wrong-import-order
from test.python.transpiler._dummy_passes import DummyAP  # pylint: disable=wrong-import-order


BACKENDS = [
    Fake7QPulseV1(),
    BackendV2Converter(Fake7QPulseV1()),
    GenericBackendV2(num_qubits=5, seed=42),
]


class CallbackPass(DummyAP):
    """A dummy analysis pass that calls a callback when executed"""

    def __init__(self, message, callback):
        super().__init__()
        self.message = message
        self.callback = callback

    def run(self, dag):
        self.callback(self.message)


@ddt
class TestBackendEstimator(QiskitTestCase):
    """Test Estimator"""

    def setUp(self):
        super().setUp()
        self._rng = np.random.default_rng(12)
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = SparsePauliOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        self.expvals = -1.0284380963435145, -1.284366511861733

        self.psi = (RealAmplitudes(num_qubits=2, reps=2), RealAmplitudes(num_qubits=2, reps=3))
        self.params = tuple(psi.parameters for psi in self.psi)
        self.hamiltonian = (
            SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)]),
            SparsePauliOp.from_list([("IZ", 1)]),
            SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)]),
        )
        self.theta = (
            [0, 1, 1, 2, 3, 5],
            [0, 1, 1, 2, 3, 5, 8, 13],
            [1, 2, 3, 4, 5, 6],
        )

    @combine(backend=BACKENDS)
    def test_estimator_run(self, backend):
        """Test Estimator.run()"""
        psi1, psi2 = self.psi
        hamiltonian1, hamiltonian2, hamiltonian3 = self.hamiltonian
        theta1, theta2, theta3 = self.theta
        with self.assertWarns(DeprecationWarning):
            estimator = BackendEstimator(backend=backend)
        estimator.set_options(seed_simulator=123)

        # Specify the circuit and observable by indices.
        # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
        with self.assertWarns(DeprecationWarning):
            job = estimator.run([psi1], [hamiltonian1], [theta1])
            result = job.result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1.5555572817900956], rtol=0.5, atol=0.2)

        # Objects can be passed instead of indices.
        # Note that passing objects has an overhead
        # since the corresponding indices need to be searched.
        # User can append a circuit and observable.
        # calculate [ <psi2(theta2)|H1|psi2(theta2)> ]
        with self.assertWarns(DeprecationWarning):
            result2 = estimator.run([psi2], [hamiltonian1], [theta2]).result()
        np.testing.assert_allclose(result2.values, [2.97797666], rtol=0.5, atol=0.2)

        # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
        with self.assertWarns(DeprecationWarning):
            result3 = estimator.run(
                [psi1, psi1], [hamiltonian2, hamiltonian3], [theta1] * 2
            ).result()
        np.testing.assert_allclose(result3.values, [-0.551653, 0.07535239], rtol=0.5, atol=0.2)

        # calculate [ <psi1(theta1)|H1|psi1(theta1)>,
        #             <psi2(theta2)|H2|psi2(theta2)>,
        #             <psi1(theta3)|H3|psi1(theta3)> ]
        with self.assertWarns(DeprecationWarning):
            result4 = estimator.run(
                [psi1, psi2, psi1],
                [hamiltonian1, hamiltonian2, hamiltonian3],
                [theta1, theta2, theta3],
            ).result()
        np.testing.assert_allclose(
            result4.values, [1.55555728, 0.17849238, -1.08766318], rtol=0.5, atol=0.2
        )

    @combine(backend=BACKENDS)
    def test_estimator_run_no_params(self, backend):
        """test for estimator without parameters"""
        circuit = self.ansatz.assign_parameters([0, 1, 1, 2, 3, 5])
        with self.assertWarns(DeprecationWarning):
            est = BackendEstimator(backend=backend)
            est.set_options(seed_simulator=123)
            result = est.run([circuit], [self.observable]).result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.284366511861733], rtol=0.05)

    @combine(backend=BACKENDS, creg=[True, False])
    def test_run_1qubit(self, backend, creg):
        """Test for 1-qubit cases"""
        qc = QuantumCircuit(1, 1) if creg else QuantumCircuit(1)
        qc2 = QuantumCircuit(1, 1) if creg else QuantumCircuit(1)
        qc2.x(0)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("Z", 1)])

        with self.assertWarns(DeprecationWarning):
            est = BackendEstimator(backend=backend)
            est.set_options(seed_simulator=123)
            result = est.run([qc], [op], [[]]).result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        with self.assertWarns(DeprecationWarning):
            result = est.run([qc], [op2], [[]]).result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        with self.assertWarns(DeprecationWarning):
            result = est.run([qc2], [op], [[]]).result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        with self.assertWarns(DeprecationWarning):
            result = est.run([qc2], [op2], [[]]).result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1], rtol=0.1)

    @combine(backend=BACKENDS, creg=[True, False])
    def test_run_2qubits(self, backend, creg):
        """Test for 2-qubit cases (to check endian)"""
        backend.set_options(seed_simulator=123)
        qc = QuantumCircuit(2, 1) if creg else QuantumCircuit(2)
        qc2 = QuantumCircuit(2, 1) if creg else QuantumCircuit(2, 1)
        qc2.x(0)

        op = SparsePauliOp.from_list([("II", 1)])
        op2 = SparsePauliOp.from_list([("ZI", 1)])
        op3 = SparsePauliOp.from_list([("IZ", 1)])

        with self.assertWarns(DeprecationWarning):
            est = BackendEstimator(backend=backend)
            result = est.run([qc], [op], [[]]).result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        with self.assertWarns(DeprecationWarning):
            result = est.run([qc2], [op], [[]]).result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        with self.assertWarns(DeprecationWarning):
            result = est.run([qc], [op2], [[]]).result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        with self.assertWarns(DeprecationWarning):
            result = est.run([qc2], [op2], [[]]).result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        with self.assertWarns(DeprecationWarning):
            result = est.run([qc], [op3], [[]]).result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [1], rtol=0.1)

        with self.assertWarns(DeprecationWarning):
            result = est.run([qc2], [op3], [[]]).result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1], rtol=0.1)

    @combine(backend=BACKENDS)
    def test_run_errors(self, backend):
        """Test for errors"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        with self.assertWarns(DeprecationWarning):
            est = BackendEstimator(backend=backend)
            est.set_options(seed_simulator=123)
            with self.assertRaises(ValueError):
                est.run([qc], [op2], [[]]).result()
            with self.assertRaises(ValueError):
                est.run([qc], [op], [[1e4]]).result()
            with self.assertRaises(ValueError):
                est.run([qc2], [op2], [[1, 2]]).result()
            with self.assertRaises(ValueError):
                est.run([qc, qc2], [op2], [[1]]).result()
            with self.assertRaises(ValueError):
                est.run([qc], [op, op2], [[1]]).result()

    @combine(backend=BACKENDS)
    def test_run_numpy_params(self, backend):
        """Test for numpy array as parameter values"""
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = self._rng.random((k, qc.num_parameters))
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        with self.assertWarns(DeprecationWarning):
            estimator = BackendEstimator(backend=backend)
            estimator.set_options(seed_simulator=123)

            target = estimator.run([qc] * k, [op] * k, params_list).result()

        with self.subTest("ndarrary"):
            with self.assertWarns(DeprecationWarning):
                result = estimator.run([qc] * k, [op] * k, params_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)

        with self.subTest("list of ndarray"):
            with self.assertWarns(DeprecationWarning):
                result = estimator.run([qc] * k, [op] * k, params_list_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)

    @combine(backend=BACKENDS)
    def test_run_with_shots_option(self, backend):
        """test with shots option."""
        with self.assertWarns(DeprecationWarning):
            est = BackendEstimator(backend=backend)
            result = est.run(
                [self.ansatz],
                [self.observable],
                parameter_values=[[0, 1, 1, 2, 3, 5]],
                shots=1024,
                seed_simulator=15,
            ).result()
            self.assertIsInstance(result, EstimatorResult)
        np.testing.assert_allclose(result.values, [-1.307397243478641], rtol=0.1)

    @combine(backend=BACKENDS)
    def test_options(self, backend):
        """Test for options"""
        with self.subTest("init"):
            with self.assertWarns(DeprecationWarning):
                estimator = BackendEstimator(backend=backend, options={"shots": 3000})
            self.assertEqual(estimator.options.get("shots"), 3000)
        with self.subTest("set_options"):
            estimator.set_options(shots=1024, seed_simulator=15)
            self.assertEqual(estimator.options.get("shots"), 1024)
            self.assertEqual(estimator.options.get("seed_simulator"), 15)
        with self.subTest("run"):
            with self.assertWarns(DeprecationWarning):
                result = estimator.run(
                    [self.ansatz],
                    [self.observable],
                    parameter_values=[[0, 1, 1, 2, 3, 5]],
                ).result()
                self.assertIsInstance(result, EstimatorResult)
            np.testing.assert_allclose(result.values, [-1.307397243478641], rtol=0.1)

    def test_job_size_limit_v2(self):
        """Test BackendEstimator respects job size limit"""

        class FakeBackendLimitedCircuits(GenericBackendV2):
            """Generic backend V2 with job size limit."""

            @property
            def max_circuits(self):
                return 1

        backend = FakeBackendLimitedCircuits(num_qubits=5, seed=42)
        backend.set_options(seed_simulator=123)
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = self._rng.random((k, qc.num_parameters))
        params_list = params_array.tolist()
        with self.assertWarns(DeprecationWarning):
            estimator = BackendEstimator(backend=backend)
        with patch.object(backend, "run") as run_mock:
            with self.assertWarns(DeprecationWarning):
                estimator.run([qc] * k, [op] * k, params_list).result()
        self.assertEqual(run_mock.call_count, 10)

    def test_job_size_limit_v1(self):
        """Test BackendEstimator respects job size limit
        REMOVE ONCE Fake7QPulseV1 GETS REMOVED"""
        with self.assertWarns(DeprecationWarning):
            backend = Fake7QPulseV1()
        config = backend.configuration()
        config.max_experiments = 1
        backend._configuration = config
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = self._rng.random((k, qc.num_parameters))
        params_list = params_array.tolist()
        with self.assertWarns(DeprecationWarning):
            estimator = BackendEstimator(backend=backend)
        estimator.set_options(seed_simulator=123)
        with patch.object(backend, "run") as run_mock:
            with self.assertWarns(DeprecationWarning):
                estimator.run([qc] * k, [op] * k, params_list).result()
        self.assertEqual(run_mock.call_count, 10)

    def test_no_max_circuits(self):
        """Test BackendEstimator works with BackendV1 and no max_experiments set.
        REMOVE ONCE Fake7QPulseV1 GETS REMOVED"""
        with self.assertWarns(DeprecationWarning):
            backend = Fake7QPulseV1()
        config = backend.configuration()
        del config.max_experiments
        backend._configuration = config
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = self._rng.random((k, qc.num_parameters))
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        with self.assertWarns(DeprecationWarning):
            estimator = BackendEstimator(backend=backend)
            estimator.set_options(seed_simulator=123)
            target = estimator.run([qc] * k, [op] * k, params_list).result()
        with self.subTest("ndarrary"):
            with self.assertWarns(DeprecationWarning):
                result = estimator.run([qc] * k, [op] * k, params_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)

        with self.subTest("list of ndarray"):
            with self.assertWarns(DeprecationWarning):
                result = estimator.run([qc] * k, [op] * k, params_list_array).result()
            self.assertEqual(len(result.metadata), k)
            np.testing.assert_allclose(result.values, target.values, rtol=0.2, atol=0.2)

    def test_bound_pass_manager(self):
        """Test bound pass manager."""

        qc = QuantumCircuit(2)
        op = SparsePauliOp.from_list([("II", 1)])

        with self.subTest("Test single circuit"):
            messages = []

            def callback(msg):
                messages.append(msg)

            bound_counter = CallbackPass("bound_pass_manager", callback)
            bound_pass = PassManager(bound_counter)
            with self.assertWarns(DeprecationWarning):
                estimator = BackendEstimator(
                    backend=GenericBackendV2(num_qubits=5, seed=42), bound_pass_manager=bound_pass
                )
                _ = estimator.run(qc, op).result()
            expected = [
                "bound_pass_manager",
            ]
            self.assertEqual(messages, expected)

        with self.subTest("Test circuit batch"):
            with Manager() as manager:
                # The multiprocessing manager is used to share data
                # between different processes. Pass Managers parallelize
                # execution for batches of circuits, so this is necessary
                # to keep track of the callback calls for num_circuits > 1
                messages = manager.list()

                def callback(msg):  # pylint: disable=function-redefined
                    messages.append(msg)

                bound_counter = CallbackPass("bound_pass_manager", callback)
                bound_pass = PassManager(bound_counter)
                with self.assertWarns(DeprecationWarning):
                    estimator = BackendEstimator(
                        backend=GenericBackendV2(num_qubits=5, seed=42),
                        bound_pass_manager=bound_pass,
                    )
                    _ = estimator.run([qc, qc], [op, op]).result()
                expected = [
                    "bound_pass_manager",
                    "bound_pass_manager",
                ]
                self.assertEqual(list(messages), expected)

    @combine(backend=BACKENDS)
    def test_layout(self, backend):
        """Test layout for split transpilation."""
        with self.subTest("initial layout test"):
            qc = QuantumCircuit(3)
            qc.x(0)
            qc.cx(0, 1)
            qc.cx(0, 2)
            op = SparsePauliOp("IZI")
            backend.set_options(seed_simulator=15)
            with self.assertWarns(DeprecationWarning):
                estimator = BackendEstimator(backend)
                estimator.set_transpile_options(seed_transpiler=15, optimization_level=1)
                value = estimator.run(qc, op, shots=10000).result().values[0]
            if optionals.HAS_AER:
                ref_value = -0.9954 if isinstance(backend, GenericBackendV2) else -0.934
            else:
                ref_value = -1
            self.assertEqual(value, ref_value)

        with self.subTest("final layout test"):
            qc = QuantumCircuit(3)
            qc.x(0)
            qc.cx(0, 1)
            qc.cx(0, 2)
            op = SparsePauliOp("IZI")
            with self.assertWarns(DeprecationWarning):
                estimator = BackendEstimator(backend)
                estimator.set_transpile_options(
                    initial_layout=[0, 1, 2], seed_transpiler=15, optimization_level=1
                )
                estimator.set_options(seed_simulator=15)
                value = estimator.run(qc, op, shots=10000).result().values[0]
            if optionals.HAS_AER:
                ref_value = -0.9954 if isinstance(backend, GenericBackendV2) else -0.8902
            else:
                ref_value = -1
            self.assertEqual(value, ref_value)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_circuit_with_measurement(self):
        """Test estimator with a dynamic circuit"""
        from qiskit_aer import AerSimulator

        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure_all()
        observable = SparsePauliOp("ZZ")

        backend = AerSimulator()
        backend.set_options(seed_simulator=15)
        with self.assertWarns(DeprecationWarning):
            estimator = BackendEstimator(backend, skip_transpilation=True)
            estimator.set_transpile_options(seed_transpiler=15)
            result = estimator.run(bell, observable).result()
        self.assertAlmostEqual(result.values[0], 1, places=1)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_dynamic_circuit(self):
        """Test estimator with a dynamic circuit"""
        from qiskit_aer import AerSimulator

        qc = QuantumCircuit(2, 1)
        with qc.for_loop(range(5)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(1, 0)
            with self.assertWarns(DeprecationWarning):
                qc.break_loop().c_if(0, True)

        observable = SparsePauliOp("IZ")

        backend = AerSimulator()
        backend.set_options(seed_simulator=15)
        with self.assertWarns(DeprecationWarning):
            estimator = BackendEstimator(backend, skip_transpilation=True)
            estimator.set_transpile_options(seed_transpiler=15)
            result = estimator.run(qc, observable).result()
        self.assertAlmostEqual(result.values[0], 0, places=1)


if __name__ == "__main__":
    unittest.main()

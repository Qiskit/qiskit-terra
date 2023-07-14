# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=no-name-in-module


"""Test Operator construction, including OpPrimitives and singletons."""

import unittest
from test.python.opflow import QiskitOpflowTestCase
from ddt import ddt, data
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.utils import QuantumInstance
from qiskit.opflow import StateFn, Zero, One, H, X, I, Z, Plus, Minus, CircuitSampler, ListOp
from qiskit.opflow.exceptions import OpflowError
from qiskit.quantum_info import Statevector


@ddt
class TestStateOpMeasEvals(QiskitOpflowTestCase):
    """Tests of evals of Meas-Operator-StateFn combos."""

    def test_statefn_overlaps(self):
        """state functions overlaps test"""
        wf = (4 * StateFn({"101010": 0.5, "111111": 0.3})) + ((3 + 0.1j) * (Zero ^ 6))
        wf_vec = StateFn(wf.to_matrix())
        self.assertAlmostEqual(wf.adjoint().eval(wf), 14.45)
        self.assertAlmostEqual(wf_vec.adjoint().eval(wf_vec), 14.45)
        self.assertAlmostEqual(wf_vec.adjoint().eval(wf), 14.45)
        self.assertAlmostEqual(wf.adjoint().eval(wf_vec), 14.45)

    def test_wf_evals_x(self):
        """wf evals x test"""
        qbits = 4

        wf = ((Zero ^ qbits) + (One ^ qbits)) * (1 / 2**0.5)
        # Note: wf = Plus^qbits fails because TensoredOp can't handle it.
        wf_vec = StateFn(wf.to_matrix())
        op = X ^ qbits
        # op = I^6
        self.assertAlmostEqual(wf.adjoint().eval(op.eval(wf)), 1)
        self.assertAlmostEqual(wf_vec.adjoint().eval(op.eval(wf)), 1)
        self.assertAlmostEqual(wf.adjoint().eval(op.eval(wf_vec)), 1)
        self.assertAlmostEqual(wf_vec.adjoint().eval(op.eval(wf_vec)), 1)

        # op = (H^X^Y)^2
        op = H ^ 6
        wf = ((Zero ^ 6) + (One ^ 6)) * (1 / 2**0.5)
        wf_vec = StateFn(wf.to_matrix())
        # print(wf.adjoint().to_matrix() @ op.to_matrix() @ wf.to_matrix())
        self.assertAlmostEqual(wf.adjoint().eval(op.eval(wf)), 0.25)
        self.assertAlmostEqual(wf_vec.adjoint().eval(op.eval(wf)), 0.25)
        self.assertAlmostEqual(wf.adjoint().eval(op.eval(wf_vec)), 0.25)
        self.assertAlmostEqual(wf_vec.adjoint().eval(op.eval(wf_vec)), 0.25)

    def test_coefficients_correctly_propagated(self):
        """Test that the coefficients in SummedOp and states are correctly used."""
        try:
            from qiskit.providers.aer import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest(f"Aer doesn't appear to be installed. Error: '{str(ex)}'")
            return
        with self.subTest("zero coeff in SummedOp"):
            op = 0 * (I + Z)
            state = Plus
            self.assertEqual((~StateFn(op) @ state).eval(), 0j)

        backend = Aer.get_backend("aer_simulator")
        with self.assertWarns(DeprecationWarning):
            q_instance = QuantumInstance(backend, seed_simulator=97, seed_transpiler=97)
        op = I
        with self.subTest("zero coeff in summed StateFn and CircuitSampler"):
            with self.assertWarns(DeprecationWarning):
                state = 0 * (Plus + Minus)
                sampler = CircuitSampler(q_instance).convert(~StateFn(op) @ state)
                self.assertEqual(sampler.eval(), 0j)

        with self.subTest("coeff gets squared in CircuitSampler shot-based readout"):
            with self.assertWarns(DeprecationWarning):
                state = (Plus + Minus) / np.sqrt(2)
                sampler = CircuitSampler(q_instance).convert(~StateFn(op) @ state)
                self.assertAlmostEqual(sampler.eval(), 1 + 0j)

    def test_is_measurement_correctly_propagated(self):
        """Test if is_measurement property of StateFn is propagated to converted StateFn."""
        try:
            from qiskit.providers.aer import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest(f"Aer doesn't appear to be installed. Error: '{str(ex)}'")
            return
        backend = Aer.get_backend("aer_simulator")

        with self.assertWarns(DeprecationWarning):
            q_instance = QuantumInstance(backend)  # no seeds needed since no values are compared
            state = Plus
            sampler = CircuitSampler(q_instance).convert(~state @ state)
            self.assertTrue(sampler.oplist[0].is_measurement)

    def test_parameter_binding_on_listop(self):
        """Test passing a ListOp with differing parameters works with the circuit sampler."""
        try:
            from qiskit.providers.aer import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest(f"Aer doesn't appear to be installed. Error: '{str(ex)}'")
            return
        x, y = Parameter("x"), Parameter("y")

        circuit1 = QuantumCircuit(1)
        circuit1.p(0.2, 0)
        circuit2 = QuantumCircuit(1)
        circuit2.p(x, 0)
        circuit3 = QuantumCircuit(1)
        circuit3.p(y, 0)

        with self.assertWarns(DeprecationWarning):
            bindings = {x: -0.4, y: 0.4}
            listop = ListOp([StateFn(circuit) for circuit in [circuit1, circuit2, circuit3]])
            sampler = CircuitSampler(Aer.get_backend("aer_simulator"))
            sampled = sampler.convert(listop, params=bindings)

        self.assertTrue(all(len(op.parameters) == 0 for op in sampled.oplist))

    def test_list_op_eval_coeff_with_nonlinear_combofn(self):
        """Test evaluating a ListOp with non-linear combo function works with coefficients."""

        state = One
        op = ListOp(5 * [I], coeff=2, combo_fn=np.prod)
        expr1 = ~StateFn(op) @ state

        expr2 = ListOp(5 * [~state @ I @ state], coeff=2, combo_fn=np.prod)

        self.assertEqual(expr1.eval(), 2)  # if the coeff is propagated too far the result is 4
        self.assertEqual(expr2.eval(), 2)

    def test_single_parameter_binds(self):
        """Test passing parameter binds as a dictionary to the circuit sampler."""
        try:
            from qiskit.providers.aer import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest(f"Aer doesn't appear to be installed. Error: '{str(ex)}'")
            return

        x = Parameter("x")
        circuit = QuantumCircuit(1)
        circuit.ry(x, 0)

        with self.assertWarns(DeprecationWarning):
            expr = ~StateFn(H) @ StateFn(circuit)
            sampler = CircuitSampler(Aer.get_backend("aer_simulator_statevector"))
            res = sampler.convert(expr, params={x: 0}).eval()

        self.assertIsInstance(res, complex)

    @data("all", "last")
    def test_circuit_sampler_caching(self, caching):
        """Test caching all operators works."""
        try:
            from qiskit.providers.aer import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest(f"Aer doesn't appear to be installed. Error: '{str(ex)}'")
            return

        x = Parameter("x")
        circuit = QuantumCircuit(1)
        circuit.ry(x, 0)

        with self.assertWarns(DeprecationWarning):
            expr1 = ~StateFn(H) @ StateFn(circuit)
            expr2 = ~StateFn(X) @ StateFn(circuit)
            sampler = CircuitSampler(Aer.get_backend("aer_simulator_statevector"), caching=caching)

            res1 = sampler.convert(expr1, params={x: 0}).eval()
            res2 = sampler.convert(expr2, params={x: 0}).eval()
            res3 = sampler.convert(expr1, params={x: 0}).eval()
            res4 = sampler.convert(expr2, params={x: 0}).eval()

        self.assertEqual(res1, res3)
        self.assertEqual(res2, res4)
        if caching == "last":
            self.assertEqual(len(sampler._cached_ops.keys()), 1)
        else:
            self.assertEqual(len(sampler._cached_ops.keys()), 2)

    def test_adjoint_nonunitary_circuit_raises(self):
        """Test adjoint on a non-unitary circuit raises a OpflowError instead of CircuitError."""
        circuit = QuantumCircuit(1)
        circuit.reset(0)

        with self.assertRaises(OpflowError):
            _ = StateFn(circuit).adjoint()

    def test_evaluating_nonunitary_circuit_state(self):
        """Test evaluating a circuit works even if it contains non-unitary instruction (resets).

        TODO: allow this for (~StateFn(circuit) @ op @ StateFn(circuit)), but this requires
        refactoring how the AerPauliExpectation works, since that currently relies on
        composing with CircuitMeasurements
        """
        circuit = QuantumCircuit(1)
        circuit.initialize([0, 1], [0])

        op = Z
        res = (~StateFn(op) @ StateFn(circuit)).eval()

        self.assertAlmostEqual(-1 + 0j, res)

    def test_quantum_instance_with_backend_shots(self):
        """Test sampling a circuit where the backend has shots attached."""
        try:
            from qiskit.providers.aer import AerSimulator
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest(f"Aer doesn't appear to be installed. Error: '{str(ex)}'")

        backend = AerSimulator(shots=10)

        with self.assertWarns(DeprecationWarning):
            sampler = CircuitSampler(backend)
            res = sampler.convert(~Plus @ Plus).eval()
        self.assertAlmostEqual(res, 1 + 0j, places=2)

    def test_adjoint_vector_to_circuit_fn(self):
        """Test it is possible to adjoint a VectorStateFn that was converted to a CircuitStateFn."""

        left = StateFn([0, 1])
        left_circuit = left.to_circuit_op().primitive

        right_circuit = QuantumCircuit(1)
        right_circuit.x(0)

        circuit = left_circuit.inverse().compose(right_circuit)

        self.assertTrue(Statevector(circuit).equiv([1, 0]))


if __name__ == "__main__":
    unittest.main()

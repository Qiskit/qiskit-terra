# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for BasePrimitive."""

import json

from ddt import data, ddt, unpack
from numpy import array, float32, float64, int32, int64

from qiskit import QuantumCircuit, pulse, transpile
from qiskit.circuit.random import random_circuit
from qiskit.primitives.base import validation
from qiskit.primitives.utils import _circuit_key
from qiskit.providers.fake_provider import GenericBackendV2
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestCircuitValidation(QiskitTestCase):
    """Test circuits validation logic."""

    @data(
        (random_circuit(2, 2, seed=0), (random_circuit(2, 2, seed=0),)),
        (
            [random_circuit(2, 2, seed=0), random_circuit(2, 2, seed=1)],
            (random_circuit(2, 2, seed=0), random_circuit(2, 2, seed=1)),
        ),
    )
    @unpack
    def test_validate_circuits(self, circuits, expected):
        """Test circuits standardization."""
        self.assertEqual(validation._validate_circuits(circuits), expected)

    @data(None, "ERROR", True, 0, 1.0, 1j, [0.0])
    def test_type_error(self, circuits):
        """Test type error if invalid input."""
        with self.assertRaises(TypeError):
            validation._validate_circuits(circuits)

    @data((), [], "")
    def test_value_error(self, circuits):
        """Test value error if no circuits are provided."""
        with self.assertRaises(ValueError):
            validation._validate_circuits(circuits)


@ddt
class TestParameterValuesValidation(QiskitTestCase):
    """Test parameter_values validation logic."""

    @data(
        ((), ((),)),
        ([], ((),)),
        (0, ((0,),)),
        (1.2, ((1.2,),)),
        ((0,), ((0,),)),
        ([0], ((0,),)),
        ([1.2], ((1.2,),)),
        ((0, 1), ((0, 1),)),
        ([0, 1], ((0, 1),)),
        ([0, 1.2], ((0, 1.2),)),
        ([0.3, 1.2], ((0.3, 1.2),)),
        (((0, 1)), ((0, 1),)),
        (([0, 1]), ((0, 1),)),
        ([(0, 1)], ((0, 1),)),
        ([[0, 1]], ((0, 1),)),
        ([[0, 1.2]], ((0, 1.2),)),
        ([[0.3, 1.2]], ((0.3, 1.2),)),
        # Test for numpy dtypes
        (int32(5), ((float(int32(5)),),)),
        (int64(6), ((float(int64(6)),),)),
        (float32(3.2), ((float(float32(3.2)),),)),
        (float64(6.4), ((float(float64(6.4)),),)),
        ([int32(5), float32(3.2)], ((float(int32(5)), float(float32(3.2))),)),
    )
    @unpack
    def test_validate_parameter_values(self, _parameter_values, expected):
        """Test parameter_values standardization."""
        for parameter_values in [_parameter_values, array(_parameter_values)]:  # Numpy
            self.assertEqual(validation._validate_parameter_values(parameter_values), expected)
            self.assertEqual(
                validation._validate_parameter_values(None, default=parameter_values), expected
            )

    @data(
        "ERROR",
        ("E", "R", "R", "O", "R"),
        (["E", "R", "R"], ["O", "R"]),
        1j,
        (1j,),
        ((1j,),),
        True,
        False,
        float("inf"),
        float("-inf"),
        float("nan"),
    )
    def test_type_error(self, parameter_values):
        """Test type error if invalid input."""
        with self.assertRaises(TypeError):
            validation._validate_parameter_values(parameter_values)

    def test_value_error(self):
        """Test value error if no parameter_values or default are provided."""
        with self.assertRaises(ValueError):
            validation._validate_parameter_values(None)


class TestCircuitKey(QiskitTestCase):
    """Tests for _circuit_key function"""

    def test_different_circuits(self):
        """Test collision of quantum circuits."""

        with self.subTest("Ry circuit"):

            def test_func(n):
                qc = QuantumCircuit(1, 1, name="foo")
                qc.ry(n, 0)
                return qc

            keys = [_circuit_key(test_func(i)) for i in range(5)]
            self.assertEqual(len(keys), len(set(keys)))

        with self.subTest("pulse circuit"):

            def test_with_scheduling(n):
                with self.assertWarns(DeprecationWarning):
                    custom_gate = pulse.Schedule(name="custom_x_gate")
                    custom_gate.insert(
                        0,
                        pulse.Play(pulse.Constant(160 * n, 0.1), pulse.DriveChannel(0)),
                        inplace=True,
                    )
                    qc = QuantumCircuit(1)
                qc.x(0)
                with self.assertWarns(DeprecationWarning):
                    qc.add_calibration("x", qubits=(0,), schedule=custom_gate)

                backend = GenericBackendV2(
                    num_qubits=2, basis_gates=["id", "u1", "u2", "u3", "cx"], seed=42
                )
                return transpile(qc, backend, scheduling_method="alap", optimization_level=1)

            keys = [_circuit_key(test_with_scheduling(i)) for i in range(1, 5)]
            self.assertEqual(len(keys), len(set(keys)))

    def test_circuit_key_controlflow(self):
        """Test for a circuit with control flow."""
        qc = QuantumCircuit(2, 1)

        with qc.for_loop(range(5)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            with self.assertWarns(DeprecationWarning):
                qc.break_loop().c_if(0, True)

        self.assertIsInstance(hash(_circuit_key(qc)), int)
        self.assertIsInstance(json.dumps(_circuit_key(qc)), str)

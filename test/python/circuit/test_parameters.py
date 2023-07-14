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

"""Test circuits with variable parameters."""
import unittest
import cmath
import math
import copy
import pickle
from operator import add, mul, sub, truediv

from test import combine

import numpy as np
from ddt import data, ddt, named_data

import qiskit
import qiskit.circuit.library as circlib
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit import BasicAer, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction, Parameter, ParameterExpression, ParameterVector
from qiskit.circuit.parametertable import ParameterReferences, ParameterTable, ParameterView
from qiskit.circuit.exceptions import CircuitError
from qiskit.compiler import assemble, transpile
from qiskit.execute_function import execute
from qiskit import pulse
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeOurense
from qiskit.tools import parallel_map


def raise_if_parameter_table_invalid(circuit):
    """Validates the internal consistency of a ParameterTable and its
    containing QuantumCircuit. Intended for use in testing.

    Raises:
       CircuitError: if QuantumCircuit and ParameterTable are inconsistent.
    """

    table = circuit._parameter_table

    # Assert parameters present in circuit match those in table.
    circuit_parameters = {
        parameter
        for instruction in circuit._data
        for param in instruction.operation.params
        for parameter in param.parameters
        if isinstance(param, ParameterExpression)
    }
    table_parameters = set(table._table.keys())

    if circuit_parameters != table_parameters:
        raise CircuitError(
            "Circuit/ParameterTable Parameter mismatch. "
            "Circuit parameters: {}. "
            "Table parameters: {}.".format(circuit_parameters, table_parameters)
        )

    # Assert parameter locations in table are present in circuit.
    circuit_instructions = [instr.operation for instr in circuit._data]

    for parameter, instr_list in table.items():
        for instr, param_index in instr_list:
            if instr not in circuit_instructions:
                raise CircuitError(f"ParameterTable instruction not present in circuit: {instr}.")

            if not isinstance(instr.params[param_index], ParameterExpression):
                raise CircuitError(
                    "ParameterTable instruction does not have a "
                    "ParameterExpression at param_index {}: {}."
                    "".format(param_index, instr)
                )

            if parameter not in instr.params[param_index].parameters:
                raise CircuitError(
                    "ParameterTable instruction parameters does "
                    "not match ParameterTable key. Instruction "
                    "parameters: {} ParameterTable key: {}."
                    "".format(instr.params[param_index].parameters, parameter)
                )

    # Assert circuit has no other parameter locations other than those in table.
    for instruction in circuit._data:
        for param_index, param in enumerate(instruction.operation.params):
            if isinstance(param, ParameterExpression):
                parameters = param.parameters

                for parameter in parameters:
                    if (instruction.operation, param_index) not in table[parameter]:
                        raise CircuitError(
                            "Found parameterized instruction not "
                            "present in table. Instruction: {} "
                            "param_index: {}".format(instruction.operation, param_index)
                        )


@ddt
class TestParameters(QiskitTestCase):
    """Test Parameters."""

    def test_gate(self):
        """Test instantiating gate with variable parameters"""
        theta = Parameter("θ")
        theta_gate = Gate("test", 1, params=[theta])
        self.assertEqual(theta_gate.name, "test")
        self.assertIsInstance(theta_gate.params[0], Parameter)

    def test_compile_quantum_circuit(self):
        """Test instantiating gate with variable parameters"""
        theta = Parameter("θ")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        backend = BasicAer.get_backend("qasm_simulator")
        qc_aer = transpile(qc, backend)
        self.assertIn(theta, qc_aer.parameters)

    def test_duplicate_name_on_append(self):
        """Test adding a second parameter object with the same name fails."""
        param_a = Parameter("a")
        param_a_again = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(param_a, 0)
        self.assertRaises(CircuitError, qc.rx, param_a_again, 0)

    def test_get_parameters(self):
        """Test instantiating gate with variable parameters"""
        from qiskit.circuit.library.standard_gates.rx import RXGate

        theta = Parameter("θ")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        rxg = RXGate(theta)
        qc.append(rxg, [qr[0]], [])
        vparams = qc._parameter_table
        self.assertEqual(len(vparams), 1)
        self.assertIs(theta, next(iter(vparams)))
        self.assertEqual(rxg, next(iter(vparams[theta]))[0])

    def test_get_parameters_by_index(self):
        """Test getting parameters by index"""
        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")
        v = ParameterVector("v", 3)
        qc = QuantumCircuit(1)
        qc.rx(x, 0)
        qc.rz(z, 0)
        qc.ry(y, 0)
        qc.u(*v, 0)
        self.assertEqual(x, qc.parameters[3])
        self.assertEqual(y, qc.parameters[4])
        self.assertEqual(z, qc.parameters[5])
        for i, vi in enumerate(v):
            self.assertEqual(vi, qc.parameters[i])

    def test_bind_parameters_anonymously(self):
        """Test setting parameters by insertion order anonymously"""
        phase = Parameter("phase")
        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")
        v = ParameterVector("v", 3)
        qc = QuantumCircuit(1, global_phase=phase)
        qc.rx(x, 0)
        qc.rz(z, 0)
        qc.ry(y, 0)
        qc.u(*v, 0)
        params = [0.1 * i for i in range(len(qc.parameters))]

        order = [phase] + v[:] + [x, y, z]
        param_dict = dict(zip(order, params))
        for assign_fun in ["bind_parameters", "assign_parameters"]:
            with self.subTest(assign_fun=assign_fun):
                bqc_anonymous = getattr(qc, assign_fun)(params)
                bqc_list = getattr(qc, assign_fun)(param_dict)
                self.assertEqual(bqc_anonymous, bqc_list)

    def test_bind_parameters_allow_unknown(self):
        """Test binding parameters allowing unknown parameters."""
        a = Parameter("a")
        b = Parameter("b")
        c = a.bind({a: 1, b: 1}, allow_unknown_parameters=True)
        self.assertEqual(c, a.bind({a: 1}))

    def test_bind_half_single_precision(self):
        """Test binding with 16bit and 32bit floats."""
        phase = Parameter("phase")
        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")
        v = ParameterVector("v", 3)
        for i in (np.float16, np.float32):
            with self.subTest(float_type=i):
                expr = (v[0] * (x + y + z) + phase) - (v[2] * v[1])
                params = np.array([0.1 * j for j in range(8)], dtype=i)
                order = [phase] + v[:] + [x, y, z]
                param_dict = dict(zip(order, params))
                bound_value = expr.bind(param_dict)
                self.assertAlmostEqual(float(bound_value), 0.09, delta=1e-4)

    def test_parameter_order(self):
        """Test the parameters are sorted by name but parameter vector order takes precedence.

        This means that the following set of parameters

            {a, z, x[0], x[1], x[2], x[3], x[10], x[11]}

        will be sorted as

            [a, x[0], x[1], x[2], x[3], x[10], x[11], z]

        """
        a, b, some_name, z = (Parameter(name) for name in ["a", "b", "some_name", "z"])
        x = ParameterVector("x", 12)
        a_vector = ParameterVector("a_vector", 15)

        qc = QuantumCircuit(2)
        qc.p(z, 0)
        for i, x_i in enumerate(reversed(x)):
            qc.rx(x_i, i % 2)
        qc.cry(a, 0, 1)
        qc.crz(some_name, 1, 0)
        for v_i in a_vector[::2]:
            qc.p(v_i, 0)
        for v_i in a_vector[1::2]:
            qc.p(v_i, 1)
        qc.p(b, 0)

        expected_order = [a] + a_vector[:] + [b, some_name] + x[:] + [z]
        actual_order = qc.parameters

        self.assertListEqual(expected_order, list(actual_order))

    @data(True, False)
    def test_parameter_order_compose(self, front):
        """Test the parameter order is correctly maintained upon composing circuits."""
        x = Parameter("x")
        y = Parameter("y")
        qc1 = QuantumCircuit(1)
        qc1.p(x, 0)
        qc2 = QuantumCircuit(1)
        qc2.rz(y, 0)

        order = [x, y]

        composed = qc1.compose(qc2, front=front)

        self.assertListEqual(list(composed.parameters), order)

    def test_parameter_order_append(self):
        """Test the parameter order is correctly maintained upon appending circuits."""
        x = Parameter("x")
        y = Parameter("y")
        qc1 = QuantumCircuit(1)
        qc1.p(x, 0)
        qc2 = QuantumCircuit(1)
        qc2.rz(y, 0)

        qc1.append(qc2, [0])

        self.assertListEqual(list(qc1.parameters), [x, y])

    def test_parameter_order_composing_nested_circuit(self):
        """Test the parameter order after nesting circuits and instructions."""
        x = ParameterVector("x", 5)
        inner = QuantumCircuit(1)
        inner.rx(x[0], [0])

        mid = QuantumCircuit(2)
        mid.p(x[1], 1)
        mid.append(inner, [0])
        mid.p(x[2], 0)
        mid.append(inner, [0])

        outer = QuantumCircuit(2)
        outer.compose(mid, inplace=True)
        outer.ryy(x[3], 0, 1)
        outer.compose(inner, inplace=True)
        outer.rz(x[4], 0)

        order = [x[0], x[1], x[2], x[3], x[4]]

        self.assertListEqual(list(outer.parameters), order)

    def test_is_parameterized(self):
        """Test checking if a gate is parameterized (bound/unbound)"""
        from qiskit.circuit.library.standard_gates.h import HGate
        from qiskit.circuit.library.standard_gates.rx import RXGate

        theta = Parameter("θ")
        rxg = RXGate(theta)
        self.assertTrue(rxg.is_parameterized())
        theta_bound = theta.bind({theta: 3.14})
        rxg = RXGate(theta_bound)
        self.assertFalse(rxg.is_parameterized())
        h_gate = HGate()
        self.assertFalse(h_gate.is_parameterized())

    def test_fix_variable(self):
        """Test setting a variable to a constant value"""
        theta = Parameter("θ")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u(0, theta, 0, qr)

        # test for both `bind_parameters` and `assign_parameters`
        for assign_fun in ["bind_parameters", "assign_parameters"]:
            with self.subTest(assign_fun=assign_fun):
                bqc = getattr(qc, assign_fun)({theta: 0.5})
                self.assertEqual(float(bqc.data[0].operation.params[0]), 0.5)
                self.assertEqual(float(bqc.data[1].operation.params[1]), 0.5)
                bqc = getattr(qc, assign_fun)({theta: 0.6})
                self.assertEqual(float(bqc.data[0].operation.params[0]), 0.6)
                self.assertEqual(float(bqc.data[1].operation.params[1]), 0.6)

    def test_multiple_parameters(self):
        """Test setting multiple parameters"""
        theta = Parameter("θ")
        x = Parameter("x")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u(0, theta, x, qr)
        self.assertEqual(qc.parameters, {theta, x})

    def test_multiple_named_parameters(self):
        """Test setting multiple named/keyword argument based parameters"""
        theta = Parameter(name="θ")
        x = Parameter(name="x")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u(0, theta, x, qr)
        self.assertEqual(theta.name, "θ")
        self.assertEqual(qc.parameters, {theta, x})

    @named_data(
        ["int", 2, int],
        ["float", 2.5, float],
        ["float16", np.float16(2.5), float],
        ["float32", np.float32(2.5), float],
        ["float64", np.float64(2.5), float],
    )
    def test_circuit_assignment_to_numeric(self, value, type_):
        """Test binding a numeric value to a circuit instruction"""
        x = Parameter("x")
        qc = QuantumCircuit(1)
        qc.append(Instruction("inst", 1, 0, [x]), (0,))
        qc.assign_parameters({x: value}, inplace=True)
        bound = qc.data[0].operation.params[0]
        self.assertIsInstance(bound, type_)
        self.assertEqual(bound, value)

    def test_partial_binding(self):
        """Test that binding a subset of circuit parameters returns a new parameterized circuit."""
        theta = Parameter("θ")
        x = Parameter("x")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u(0, theta, x, qr)

        # test for both `bind_parameters` and `assign_parameters`
        for assign_fun in ["bind_parameters", "assign_parameters"]:
            with self.subTest(assign_fun=assign_fun):
                pqc = getattr(qc, assign_fun)({theta: 2})

                self.assertEqual(pqc.parameters, {x})

                self.assertEqual(float(pqc.data[0].operation.params[0]), 2)
                self.assertEqual(float(pqc.data[1].operation.params[1]), 2)

    @data(True, False)
    def test_mixed_binding(self, inplace):
        """Test we can bind a mixed dict with Parameter objects and floats."""
        theta = Parameter("θ")
        x, new_x = Parameter("x"), Parameter("new_x")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u(0, theta, x, qr)

        pqc = qc.assign_parameters({theta: 2, x: new_x}, inplace=inplace)
        if inplace:
            self.assertEqual(qc.parameters, {new_x})
        else:
            self.assertEqual(pqc.parameters, {new_x})

    def test_expression_partial_binding(self):
        """Test that binding a subset of expression parameters returns a new
        parameterized circuit."""
        theta = Parameter("θ")
        phi = Parameter("phi")

        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta + phi, qr)

        # test for both `bind_parameters` and `assign_parameters`
        for assign_fun in ["bind_parameters", "assign_parameters"]:
            with self.subTest(assign_fun=assign_fun):
                pqc = getattr(qc, assign_fun)({theta: 2})

                self.assertEqual(pqc.parameters, {phi})

                self.assertTrue(isinstance(pqc.data[0].operation.params[0], ParameterExpression))
                self.assertEqual(str(pqc.data[0].operation.params[0]), "phi + 2")

                fbqc = getattr(pqc, assign_fun)({phi: 1.0})

                self.assertEqual(fbqc.parameters, set())
                self.assertIsInstance(fbqc.data[0].operation.params[0], float)
                self.assertEqual(float(fbqc.data[0].operation.params[0]), 3)

    def test_two_parameter_expression_binding(self):
        """Verify that for a circuit with parameters theta and phi that
        we can correctly assign theta to -phi.
        """
        theta = Parameter("theta")
        phi = Parameter("phi")

        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        qc.ry(phi, 0)

        self.assertEqual(len(qc._parameter_table[theta]), 1)
        self.assertEqual(len(qc._parameter_table[phi]), 1)

        qc.assign_parameters({theta: -phi}, inplace=True)

        self.assertEqual(len(qc._parameter_table[phi]), 2)

    def test_expression_partial_binding_zero(self):
        """Verify that binding remains possible even if a previous partial bind
        would reduce the expression to zero.
        """
        theta = Parameter("theta")
        phi = Parameter("phi")

        qc = QuantumCircuit(1)
        qc.p(theta * phi, 0)

        # test for both `bind_parameters` and `assign_parameters`
        for assign_fun in ["bind_parameters", "assign_parameters"]:
            with self.subTest(assign_fun=assign_fun):
                pqc = getattr(qc, assign_fun)({theta: 0})

                self.assertEqual(pqc.parameters, {phi})

                self.assertTrue(isinstance(pqc.data[0].operation.params[0], ParameterExpression))
                self.assertEqual(str(pqc.data[0].operation.params[0]), "0")

                fbqc = getattr(pqc, assign_fun)({phi: 1})

                self.assertEqual(fbqc.parameters, set())
                self.assertIsInstance(fbqc.data[0].operation.params[0], int)
                self.assertEqual(float(fbqc.data[0].operation.params[0]), 0)

    def test_raise_if_assigning_params_not_in_circuit(self):
        """Verify binding parameters which are not present in the circuit raises an error."""
        x = Parameter("x")
        y = Parameter("y")
        z = ParameterVector("z", 3)
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        # test for both `bind_parameters` and `assign_parameters`
        for assign_fun in ["bind_parameters", "assign_parameters"]:
            qc = QuantumCircuit(qr)
            with self.subTest(assign_fun=assign_fun):
                qc.p(0.1, qr[0])
                self.assertRaises(CircuitError, getattr(qc, assign_fun), {x: 1})
                qc.p(x, qr[0])
                self.assertRaises(CircuitError, getattr(qc, assign_fun), {x: 1, y: 2})
                qc.p(z[1], qr[0])
                self.assertRaises(CircuitError, getattr(qc, assign_fun), {z: [3, 4, 5]})
                self.assertRaises(CircuitError, getattr(qc, assign_fun), {"a_str": 6})
                self.assertRaises(CircuitError, getattr(qc, assign_fun), {None: 7})

    def test_gate_multiplicity_binding(self):
        """Test binding when circuit contains multiple references to same gate"""

        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        gate = RZGate(theta)
        qc.append(gate, [0], [])
        qc.append(gate, [0], [])
        # test for both `bind_parameters` and `assign_parameters`
        for assign_fun in ["bind_parameters", "assign_parameters"]:
            with self.subTest(assign_fun=assign_fun):
                qc2 = getattr(qc, assign_fun)({theta: 1.0})
                self.assertEqual(len(qc2._parameter_table), 0)
                for instruction in qc2.data:
                    self.assertEqual(float(instruction.operation.params[0]), 1.0)

    def test_calibration_assignment(self):
        """That that calibration mapping and the schedules they map are assigned together."""
        theta = Parameter("theta")
        circ = QuantumCircuit(3, 3)
        circ.append(Gate("rxt", 1, [theta]), [0])
        circ.measure(0, 0)

        rxt_q0 = pulse.Schedule(
            pulse.Play(
                pulse.library.Gaussian(duration=128, sigma=16, amp=0.2 * theta / 3.14),
                pulse.DriveChannel(0),
            )
        )

        circ.add_calibration("rxt", [0], rxt_q0, [theta])
        circ = circ.assign_parameters({theta: 3.14})

        instruction = circ.data[0]
        cal_key = (
            tuple(circ.find_bit(q).index for q in instruction.qubits),
            tuple(instruction.operation.params),
        )
        self.assertEqual(cal_key, ((0,), (3.14,)))
        # Make sure that key from instruction data matches the calibrations dictionary
        self.assertIn(cal_key, circ.calibrations["rxt"])
        sched = circ.calibrations["rxt"][cal_key]
        self.assertEqual(sched.instructions[0][1].pulse.amp, 0.2)

    def test_calibration_assignment_doesnt_mutate(self):
        """That that assignment doesn't mutate the original circuit."""
        theta = Parameter("theta")
        circ = QuantumCircuit(3, 3)
        circ.append(Gate("rxt", 1, [theta]), [0])
        circ.measure(0, 0)

        rxt_q0 = pulse.Schedule(
            pulse.Play(
                pulse.library.Gaussian(duration=128, sigma=16, amp=0.2 * theta / 3.14),
                pulse.DriveChannel(0),
            )
        )

        circ.add_calibration("rxt", [0], rxt_q0, [theta])
        circ_copy = copy.deepcopy(circ)
        assigned_circ = circ.assign_parameters({theta: 3.14})

        self.assertEqual(circ.calibrations, circ_copy.calibrations)
        self.assertNotEqual(assigned_circ.calibrations, circ.calibrations)

    def test_calibration_assignment_w_expressions(self):
        """That calibrations with multiple parameters are assigned correctly"""
        theta = Parameter("theta")
        sigma = Parameter("sigma")
        circ = QuantumCircuit(3, 3)
        circ.append(Gate("rxt", 1, [theta / 2, sigma]), [0])
        circ.measure(0, 0)

        rxt_q0 = pulse.Schedule(
            pulse.Play(
                pulse.library.Gaussian(duration=128, sigma=4 * sigma, amp=0.2 * theta / 3.14),
                pulse.DriveChannel(0),
            )
        )

        circ.add_calibration("rxt", [0], rxt_q0, [theta / 2, sigma])
        circ = circ.assign_parameters({theta: 3.14, sigma: 4})

        instruction = circ.data[0]
        cal_key = (
            tuple(circ.find_bit(q).index for q in instruction.qubits),
            tuple(instruction.operation.params),
        )
        self.assertEqual(cal_key, ((0,), (3.14 / 2, 4)))
        # Make sure that key from instruction data matches the calibrations dictionary
        self.assertIn(cal_key, circ.calibrations["rxt"])
        sched = circ.calibrations["rxt"][cal_key]
        self.assertEqual(sched.instructions[0][1].pulse.amp, 0.2)
        self.assertEqual(sched.instructions[0][1].pulse.sigma, 16)

    def test_substitution(self):
        """Test Parameter substitution (vs bind)."""
        alpha = Parameter("⍺")
        beta = Parameter("beta")
        schedule = pulse.Schedule(pulse.ShiftPhase(alpha, pulse.DriveChannel(0)))

        circ = QuantumCircuit(3, 3)
        circ.append(Gate("my_rz", 1, [alpha]), [0])
        circ.add_calibration("my_rz", [0], schedule, [alpha])

        circ = circ.assign_parameters({alpha: 2 * beta})

        circ = circ.assign_parameters({beta: 1.57})
        cal_sched = circ.calibrations["my_rz"][((0,), (3.14,))]
        self.assertEqual(float(cal_sched.instructions[0][1].phase), 3.14)

    def test_partial_assignment(self):
        """Expressions of parameters with partial assignment."""
        alpha = Parameter("⍺")
        beta = Parameter("beta")
        gamma = Parameter("γ")
        phi = Parameter("ϕ")

        with pulse.build() as my_cal:
            pulse.set_frequency(alpha + beta, pulse.DriveChannel(0))
            pulse.shift_frequency(gamma + beta, pulse.DriveChannel(0))
            pulse.set_phase(phi, pulse.DriveChannel(1))

        circ = QuantumCircuit(2, 2)
        circ.append(Gate("custom", 2, [alpha, beta, gamma, phi]), [0, 1])
        circ.add_calibration("custom", [0, 1], my_cal, [alpha, beta, gamma, phi])

        # Partial bind
        delta = 1e9
        freq = 4.5e9
        shift = 0.5e9
        phase = 3.14 / 4

        circ = circ.assign_parameters({alpha: freq - delta})
        cal_sched = list(circ.calibrations["custom"].values())[0]
        self.assertEqual(cal_sched.instructions[0][1].frequency, freq - delta + beta)

        circ = circ.assign_parameters({beta: delta})
        cal_sched = list(circ.calibrations["custom"].values())[0]
        self.assertEqual(float(cal_sched.instructions[0][1].frequency), freq)
        self.assertEqual(cal_sched.instructions[1][1].frequency, gamma + delta)

        circ = circ.assign_parameters({gamma: shift - delta})
        cal_sched = list(circ.calibrations["custom"].values())[0]
        self.assertEqual(float(cal_sched.instructions[1][1].frequency), shift)

        self.assertEqual(cal_sched.instructions[2][1].phase, phi)
        circ = circ.assign_parameters({phi: phase})
        cal_sched = list(circ.calibrations["custom"].values())[0]
        self.assertEqual(float(cal_sched.instructions[2][1].phase), phase)

    def test_circuit_generation(self):
        """Test creating a series of circuits parametrically"""
        theta = Parameter("θ")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        backend = BasicAer.get_backend("qasm_simulator")
        qc_aer = transpile(qc, backend)

        # generate list of circuits
        for assign_fun in ["bind_parameters", "assign_parameters"]:
            with self.subTest(assign_fun=assign_fun):
                circs = []
                theta_list = np.linspace(0, np.pi, 20)
                for theta_i in theta_list:
                    circs.append(getattr(qc_aer, assign_fun)({theta: theta_i}))
                qobj = assemble(circs)
                for index, theta_i in enumerate(theta_list):
                    res = float(qobj.experiments[index].instructions[0].params[0])
                    self.assertTrue(math.isclose(res, theta_i), f"{res} != {theta_i}")

    def test_circuit_composition(self):
        """Test preservation of parameters when combining circuits."""
        theta = Parameter("θ")
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc1 = QuantumCircuit(qr, cr)
        qc1.rx(theta, qr)

        phi = Parameter("phi")
        qc2 = QuantumCircuit(qr, cr)
        qc2.ry(phi, qr)
        qc2.h(qr)
        qc2.measure(qr, cr)

        qc3 = qc1.compose(qc2)
        self.assertEqual(qc3.parameters, {theta, phi})

    def test_composite_instruction(self):
        """Test preservation of parameters via parameterized instructions."""
        theta = Parameter("θ")
        qr1 = QuantumRegister(1, name="qr1")
        qc1 = QuantumCircuit(qr1)
        qc1.rx(theta, qr1)
        qc1.rz(np.pi / 2, qr1)
        qc1.ry(theta, qr1)
        gate = qc1.to_instruction()
        self.assertEqual(gate.params, [theta])

        phi = Parameter("phi")
        qr2 = QuantumRegister(3, name="qr2")
        qc2 = QuantumCircuit(qr2)
        qc2.ry(phi, qr2[0])
        qc2.h(qr2)
        qc2.append(gate, qargs=[qr2[1]])
        self.assertEqual(qc2.parameters, {theta, phi})

    def test_parameter_name_conflicts_raises(self):
        """Verify attempting to add different parameters with matching names raises an error."""
        theta1 = Parameter("theta")
        theta2 = Parameter("theta")

        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        qc.p(theta1, 0)

        self.assertRaises(CircuitError, qc.p, theta2, 0)

    def test_bind_ryrz_vector(self):
        """Test binding a list of floats to a ParameterVector"""
        qc = QuantumCircuit(4)
        depth = 4
        theta = ParameterVector("θ", length=len(qc.qubits) * depth * 2)
        theta_iter = iter(theta)
        for _ in range(depth):
            for q in qc.qubits:
                qc.ry(next(theta_iter), q)
                qc.rz(next(theta_iter), q)
            for i, q in enumerate(qc.qubits[:-1]):
                qc.cx(qc.qubits[i], qc.qubits[i + 1])
            qc.barrier()
        theta_vals = np.linspace(0, 1, len(theta)) * np.pi
        self.assertEqual(set(qc.parameters), set(theta.params))
        for assign_fun in ["bind_parameters", "assign_parameters"]:
            with self.subTest(assign_fun=assign_fun):
                bqc = getattr(qc, assign_fun)({theta: theta_vals})
                for instruction in bqc.data:
                    if hasattr(instruction.operation, "params") and instruction.operation.params:
                        self.assertIn(float(instruction.operation.params[0]), theta_vals)

    def test_compile_vector(self):
        """Test compiling a circuit with an unbound ParameterVector"""
        qc = QuantumCircuit(4)
        depth = 4
        theta = ParameterVector("θ", length=len(qc.qubits) * depth * 2)
        theta_iter = iter(theta)
        for _ in range(depth):
            for q in qc.qubits:
                qc.ry(next(theta_iter), q)
                qc.rz(next(theta_iter), q)
            for i, q in enumerate(qc.qubits[:-1]):
                qc.cx(qc.qubits[i], qc.qubits[i + 1])
            qc.barrier()
        backend = BasicAer.get_backend("qasm_simulator")
        qc_aer = transpile(qc, backend)
        for param in theta:
            self.assertIn(param, qc_aer.parameters)

    def test_instruction_ryrz_vector(self):
        """Test constructing a circuit from instructions with remapped ParameterVectors"""
        qubits = 5
        depth = 4
        ryrz = QuantumCircuit(qubits, name="ryrz")
        theta = ParameterVector("θ0", length=len(ryrz.qubits) * 2)
        theta_iter = iter(theta)
        for q in ryrz.qubits:
            ryrz.ry(next(theta_iter), q)
            ryrz.rz(next(theta_iter), q)

        cxs = QuantumCircuit(qubits - 1, name="cxs")
        for i, _ in enumerate(cxs.qubits[:-1:2]):
            cxs.cx(cxs.qubits[2 * i], cxs.qubits[2 * i + 1])

        paramvecs = []
        qc = QuantumCircuit(qubits)
        for i in range(depth):
            theta_l = ParameterVector(f"θ{i + 1}", length=len(ryrz.qubits) * 2)
            ryrz_inst = ryrz.to_instruction(parameter_map={theta: theta_l})
            paramvecs += [theta_l]
            qc.append(ryrz_inst, qargs=qc.qubits)
            qc.append(cxs, qargs=qc.qubits[1:])
            qc.append(cxs, qargs=qc.qubits[:-1])
            qc.barrier()

        backend = BasicAer.get_backend("qasm_simulator")
        qc_aer = transpile(qc, backend)
        for vec in paramvecs:
            for param in vec:
                self.assertIn(param, qc_aer.parameters)

    @data("single", "vector")
    def test_parameter_equality_through_serialization(self, ptype):
        """Verify parameters maintain their equality after serialization."""

        if ptype == "single":
            x1 = Parameter("x")
            x2 = Parameter("x")
        else:
            x1 = ParameterVector("x", 2)[0]
            x2 = ParameterVector("x", 2)[0]

        x1_p = pickle.loads(pickle.dumps(x1))
        x2_p = pickle.loads(pickle.dumps(x2))

        self.assertEqual(x1, x1_p)
        self.assertEqual(x2, x2_p)

        self.assertNotEqual(x1, x2_p)
        self.assertNotEqual(x2, x1_p)

    def test_binding_parameterized_circuits_built_in_multiproc(self):
        """Verify subcircuits built in a subprocess can still be bound."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2429

        num_processes = 4

        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)

        circuit = QuantumCircuit(qr, cr)
        parameters = [Parameter(f"x{i}") for i in range(num_processes)]

        results = parallel_map(
            _construct_circuit, parameters, task_args=(qr,), num_processes=num_processes
        )

        for qc in results:
            circuit.compose(qc, inplace=True)

        parameter_values = [{x: 1.0 for x in parameters}]

        qobj = assemble(
            circuit,
            backend=BasicAer.get_backend("qasm_simulator"),
            parameter_binds=parameter_values,
        )

        self.assertEqual(len(qobj.experiments), 1)
        self.assertEqual(len(qobj.experiments[0].instructions), 4)
        self.assertTrue(
            all(
                len(inst.params) == 1
                and isinstance(inst.params[0], float)
                and float(inst.params[0]) == 1
                for inst in qobj.experiments[0].instructions
            )
        )

    def test_transpiling_multiple_parameterized_circuits(self):
        """Verify several parameterized circuits can be transpiled at once."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2864

        qr = QuantumRegister(1)
        qc1 = QuantumCircuit(qr)
        qc2 = QuantumCircuit(qr)

        theta = Parameter("theta")

        qc1.u(theta, 0, 0, qr[0])
        qc2.u(theta, 3.14, 0, qr[0])

        circuits = [qc1, qc2]

        job = execute(
            circuits,
            BasicAer.get_backend("unitary_simulator"),
            shots=512,
            parameter_binds=[{theta: 1}],
        )

        self.assertTrue(len(job.result().results), 2)

    @data(0, 1, 2, 3)
    def test_transpile_across_optimization_levels(self, opt_level):
        """Verify parameterized circuits can be transpiled with all default pass managers."""

        qc = QuantumCircuit(5, 5)

        theta = Parameter("theta")
        phi = Parameter("phi")

        qc.rx(theta, 0)
        qc.x(0)
        for i in range(5 - 1):
            qc.rxx(phi, i, i + 1)

        qc.measure(range(5 - 1), range(5 - 1))

        transpile(qc, FakeOurense(), optimization_level=opt_level)

    def test_repeated_gates_to_dag_and_back(self):
        """Verify circuits with repeated parameterized gates can be converted
        to DAG and back, maintaining consistency of circuit._parameter_table."""

        from qiskit.converters import circuit_to_dag, dag_to_circuit

        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta = Parameter("theta")

        qc.p(theta, qr[0])

        double_qc = qc.compose(qc)
        test_qc = dag_to_circuit(circuit_to_dag(double_qc))

        for assign_fun in ["bind_parameters", "assign_parameters"]:
            with self.subTest(assign_fun=assign_fun):
                bound_test_qc = getattr(test_qc, assign_fun)({theta: 1})
                self.assertEqual(len(bound_test_qc.parameters), 0)

    def test_rebinding_instruction_copy(self):
        """Test rebinding a copied instruction does not modify the original."""

        theta = Parameter("th")

        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        instr = qc.to_instruction()

        qc1 = QuantumCircuit(1)
        qc1.append(instr, [0])

        for assign_fun in ["bind_parameters", "assign_parameters"]:
            with self.subTest(assign_fun=assign_fun):
                output1 = getattr(qc1, assign_fun)({theta: 0.1}).decompose()
                output2 = getattr(qc1, assign_fun)({theta: 0.2}).decompose()

                expected1 = QuantumCircuit(1)
                expected1.rx(0.1, 0)

                expected2 = QuantumCircuit(1)
                expected2.rx(0.2, 0)

                self.assertEqual(expected1, output1)
                self.assertEqual(expected2, output2)

    @combine(target_type=["gate", "instruction"], parameter_type=["numbers", "parameters"])
    def test_decompose_propagates_bound_parameters(self, target_type, parameter_type):
        """Verify bind-before-decompose preserves bound values."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2482
        theta = Parameter("th")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        if target_type == "gate":
            inst = qc.to_gate()
        elif target_type == "instruction":
            inst = qc.to_instruction()

        qc2 = QuantumCircuit(1)
        qc2.append(inst, [0])

        if parameter_type == "numbers":
            bound_qc2 = qc2.assign_parameters({theta: 0.5})
            expected_parameters = set()
            expected_qc2 = QuantumCircuit(1)
            expected_qc2.rx(0.5, 0)
        else:
            phi = Parameter("ph")
            bound_qc2 = qc2.assign_parameters({theta: phi})
            expected_parameters = {phi}
            expected_qc2 = QuantumCircuit(1)
            expected_qc2.rx(phi, 0)

        decomposed_qc2 = bound_qc2.decompose()

        with self.subTest(msg="testing parameters of initial circuit"):
            self.assertEqual(qc2.parameters, {theta})

        with self.subTest(msg="testing parameters of bound circuit"):
            self.assertEqual(bound_qc2.parameters, expected_parameters)

        with self.subTest(msg="testing parameters of deep decomposed bound circuit"):
            self.assertEqual(decomposed_qc2.parameters, expected_parameters)

        with self.subTest(msg="testing deep decomposed circuit"):
            self.assertEqual(decomposed_qc2, expected_qc2)

    @combine(target_type=["gate", "instruction"], parameter_type=["numbers", "parameters"])
    def test_decompose_propagates_deeply_bound_parameters(self, target_type, parameter_type):
        """Verify bind-before-decompose preserves deeply bound values."""
        theta = Parameter("th")
        qc1 = QuantumCircuit(1)
        qc1.rx(theta, 0)

        if target_type == "gate":
            inst = qc1.to_gate()
        elif target_type == "instruction":
            inst = qc1.to_instruction()

        qc2 = QuantumCircuit(1)
        qc2.append(inst, [0])

        if target_type == "gate":
            inst = qc2.to_gate()
        elif target_type == "instruction":
            inst = qc2.to_instruction()

        qc3 = QuantumCircuit(1)
        qc3.append(inst, [0])

        if parameter_type == "numbers":
            bound_qc3 = qc3.assign_parameters({theta: 0.5})
            expected_parameters = set()
            expected_qc3 = QuantumCircuit(1)
            expected_qc3.rx(0.5, 0)
        else:
            phi = Parameter("ph")
            bound_qc3 = qc3.assign_parameters({theta: phi})
            expected_parameters = {phi}
            expected_qc3 = QuantumCircuit(1)
            expected_qc3.rx(phi, 0)

        deep_decomposed_qc3 = bound_qc3.decompose().decompose()

        with self.subTest(msg="testing parameters of initial circuit"):
            self.assertEqual(qc3.parameters, {theta})

        with self.subTest(msg="testing parameters of bound circuit"):
            self.assertEqual(bound_qc3.parameters, expected_parameters)

        with self.subTest(msg="testing parameters of deep decomposed bound circuit"):
            self.assertEqual(deep_decomposed_qc3.parameters, expected_parameters)

        with self.subTest(msg="testing deep decomposed circuit"):
            self.assertEqual(deep_decomposed_qc3, expected_qc3)

    @data("gate", "instruction")
    def test_executing_parameterized_instruction_bound_early(self, target_type):
        """Verify bind-before-execute preserves bound values."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2482

        theta = Parameter("theta")

        sub_qc = QuantumCircuit(2)
        sub_qc.h(0)
        sub_qc.cx(0, 1)
        sub_qc.rz(theta, [0, 1])
        sub_qc.cx(0, 1)
        sub_qc.h(0)

        if target_type == "gate":
            sub_inst = sub_qc.to_gate()
        elif target_type == "instruction":
            sub_inst = sub_qc.to_instruction()

        unbound_qc = QuantumCircuit(2, 1)
        unbound_qc.append(sub_inst, [0, 1], [])
        unbound_qc.measure(0, 0)

        for assign_fun in ["bind_parameters", "assign_parameters"]:
            with self.subTest(assign_fun=assign_fun):
                bound_qc = getattr(unbound_qc, assign_fun)({theta: np.pi / 2})

                shots = 1024
                job = execute(bound_qc, backend=BasicAer.get_backend("qasm_simulator"), shots=shots)
                self.assertDictAlmostEqual(job.result().get_counts(), {"1": shots}, 0.05 * shots)

    def test_num_parameters(self):
        """Test the num_parameters property."""
        with self.subTest(msg="standard case"):
            theta = Parameter("θ")
            x = Parameter("x")
            qc = QuantumCircuit(1)
            qc.rx(theta, 0)
            qc.u(0, theta, x, 0)
            self.assertEqual(qc.num_parameters, 2)

        with self.subTest(msg="parameter vector"):
            params = ParameterVector("x", length=3)
            qc = QuantumCircuit(4)
            qc.rx(params[0], 2)
            qc.ry(params[1], 1)
            qc.rz(params[2], 3)
            self.assertEqual(qc.num_parameters, 3)

        with self.subTest(msg="no params"):
            qc = QuantumCircuit(1)
            qc.x(0)
            self.assertEqual(qc.num_parameters, 0)

    def test_execute_result_names(self):
        """Test unique names for list of parameter binds."""
        theta = Parameter("θ")
        reps = 5
        qc = QuantumCircuit(1, 1)
        qc.rx(theta, 0)
        qc.measure(0, 0)

        plist = [{theta: i} for i in range(reps)]
        simulator = BasicAer.get_backend("qasm_simulator")
        result = execute(qc, backend=simulator, parameter_binds=plist).result()
        result_names = {res.name for res in result.results}
        self.assertEqual(reps, len(result_names))

    def test_to_instruction_after_inverse(self):
        """Verify converting an inverse generates a valid ParameterTable"""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/4235
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.rz(theta, 0)

        inv_instr = qc.inverse().to_instruction()
        self.assertIsInstance(inv_instr, Instruction)

    def test_repeated_circuit(self):
        """Test repeating a circuit maintains the parameters."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.rz(theta, 0)
        rep = qc.repeat(3)

        self.assertEqual(rep.parameters, {theta})

    def test_copy_after_inverse(self):
        """Verify circuit.inverse generates a valid ParameterTable."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.rz(theta, 0)

        inverse = qc.inverse()
        self.assertIn(theta, inverse.parameters)
        raise_if_parameter_table_invalid(inverse)

    def test_copy_after_reverse(self):
        """Verify circuit.reverse generates a valid ParameterTable."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.rz(theta, 0)

        reverse = qc.reverse_ops()
        self.assertIn(theta, reverse.parameters)
        raise_if_parameter_table_invalid(reverse)

    def test_copy_after_dot_data_setter(self):
        """Verify setting circuit.data generates a valid ParameterTable."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.rz(theta, 0)

        qc.data = []
        self.assertEqual(qc.parameters, set())
        raise_if_parameter_table_invalid(qc)

    def test_circuit_with_ufunc(self):
        """Test construction of circuit and binding of parameters
        after we apply universal functions."""
        from math import pi

        phi = Parameter(name="phi")
        theta = Parameter(name="theta")

        qc = QuantumCircuit(2)
        qc.p(np.abs(-phi), 0)
        qc.p(np.cos(phi), 0)
        qc.p(np.sin(phi), 0)
        qc.p(np.tan(phi), 0)
        qc.rz(np.arccos(theta), 1)
        qc.rz(np.arctan(theta), 1)
        qc.rz(np.arcsin(theta), 1)

        qc.assign_parameters({phi: pi, theta: 1}, inplace=True)

        qc_ref = QuantumCircuit(2)
        qc_ref.p(pi, 0)
        qc_ref.p(-1, 0)
        qc_ref.p(0, 0)
        qc_ref.p(0, 0)
        qc_ref.rz(0, 1)
        qc_ref.rz(pi / 4, 1)
        qc_ref.rz(pi / 2, 1)

        self.assertEqual(qc, qc_ref)

    def test_compile_with_ufunc(self):
        """Test compiling of circuit with unbound parameters
        after we apply universal functions."""
        from math import pi

        theta = ParameterVector("theta", length=7)

        qc = QuantumCircuit(7)
        qc.rx(np.abs(theta[0]), 0)
        qc.rx(np.cos(theta[1]), 1)
        qc.rx(np.sin(theta[2]), 2)
        qc.rx(np.tan(theta[3]), 3)
        qc.rx(np.arccos(theta[4]), 4)
        qc.rx(np.arctan(theta[5]), 5)
        qc.rx(np.arcsin(theta[6]), 6)

        # transpile to different basis
        transpiled = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)

        for x in theta:
            self.assertIn(x, transpiled.parameters)

        bound = transpiled.bind_parameters({theta: [-1, pi, pi, pi, 1, 1, 1]})

        expected = QuantumCircuit(7)
        expected.rx(1.0, 0)
        expected.rx(-1.0, 1)
        expected.rx(0.0, 2)
        expected.rx(0.0, 3)
        expected.rx(0.0, 4)
        expected.rx(pi / 4, 5)
        expected.rx(pi / 2, 6)
        expected = transpile(expected, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)

        self.assertEqual(expected, bound)

    def test_parametervector_resize(self):
        """Test the resize method of the parameter vector."""

        vec = ParameterVector("x", 2)
        element = vec[1]  # store an entry for instancecheck later on

        with self.subTest("shorten"):
            vec.resize(1)
            self.assertEqual(len(vec), 1)
            self.assertListEqual([param.name for param in vec], _paramvec_names("x", 1))

        with self.subTest("enlargen"):
            vec.resize(3)
            self.assertEqual(len(vec), 3)
            # ensure we still have the same instance not a copy with the same name
            # this is crucial for adding parameters to circuits since we cannot use the same
            # name if the instance is not the same
            self.assertIs(element, vec[1])
            self.assertListEqual([param.name for param in vec], _paramvec_names("x", 3))

    def test_raise_if_sub_unknown_parameters(self):
        """Verify we raise if asked to sub a parameter not in self."""
        x = Parameter("x")

        y = Parameter("y")
        z = Parameter("z")

        with self.assertRaisesRegex(CircuitError, "not present"):
            x.subs({y: z})

    def test_sub_allow_unknown_parameters(self):
        """Verify we raise if asked to sub a parameter not in self."""
        x = Parameter("x")

        y = Parameter("y")
        z = Parameter("z")

        subbed = x.subs({y: z}, allow_unknown_parameters=True)
        self.assertEqual(subbed, x)


def _construct_circuit(param, qr):
    qc = QuantumCircuit(qr)
    qc.ry(param, qr[0])
    return qc


def _paramvec_names(prefix, length):
    return [f"{prefix}[{i}]" for i in range(length)]


@ddt
class TestParameterExpressions(QiskitTestCase):
    """Test expressions of Parameters."""

    supported_operations = [add, sub, mul, truediv]

    def test_compare_to_value_when_bound(self):
        """Verify expression can be compared to a fixed value
        when fully bound."""

        x = Parameter("x")
        bound_expr = x.bind({x: 2.3})
        self.assertEqual(bound_expr, 2.3)

    def test_abs_function_when_bound(self):
        """Verify expression can be used with
        abs functions when bound."""

        x = Parameter("x")
        xb_1 = x.bind({x: 2.0})
        xb_2 = x.bind({x: 3.0 + 4.0j})

        self.assertEqual(abs(xb_1), 2.0)
        self.assertEqual(abs(-xb_1), 2.0)
        self.assertEqual(abs(xb_2), 5.0)

    def test_abs_function_when_not_bound(self):
        """Verify expression can be used with
        abs functions when not bound."""

        x = Parameter("x")
        y = Parameter("y")

        self.assertEqual(abs(x), abs(-x))
        self.assertEqual(abs(x) * abs(y), abs(x * y))
        self.assertEqual(abs(x) / abs(y), abs(x / y))

    def test_cast_to_complex_when_bound(self):
        """Verify that the cast to complex works for bound objects."""
        x = Parameter("x")
        y = Parameter("y")
        bound_expr = (x + y).bind({x: 1.0, y: 1j})
        self.assertEqual(complex(bound_expr), 1 + 1j)

    def test_raise_if_cast_to_complex_when_not_fully_bound(self):
        """Verify raises if casting to complex and not fully bound."""

        x = Parameter("x")
        y = Parameter("y")
        bound_expr = (x + y).bind({x: 1j})
        with self.assertRaisesRegex(TypeError, "unbound parameters"):
            complex(bound_expr)

    def test_cast_to_float_when_bound(self):
        """Verify expression can be cast to a float when fully bound."""

        x = Parameter("x")
        bound_expr = x.bind({x: 2.3})
        self.assertEqual(float(bound_expr), 2.3)

    def test_cast_to_float_when_underlying_expression_bound(self):
        """Verify expression can be cast to a float when it still contains unbound parameters, but
        the underlying symbolic expression has a knowable value."""
        x = Parameter("x")
        expr = x - x + 2.3
        self.assertEqual(float(expr), 2.3)

    def test_cast_to_float_intermediate_complex_value(self):
        """Verify expression can be cast to a float when it is fully bound, but an intermediate part
        of the expression evaluation involved complex types.  Sympy is generally more permissive
        than symengine here, and sympy's tends to be the expected behaviour for our users."""
        x = Parameter("x")
        bound_expr = (x + 1.0 + 1.0j).bind({x: -1.0j})
        self.assertEqual(float(bound_expr), 1.0)

    def test_cast_to_float_of_complex_fails(self):
        """Test that an attempt to produce a float from a complex value fails if there is an
        imaginary part, with a sensible error message."""
        x = Parameter("x")
        bound_expr = (x + 1.0j).bind({x: 1.0})
        with self.assertRaisesRegex(TypeError, "could not cast expression to float"):
            float(bound_expr)

    def test_raise_if_cast_to_float_when_not_fully_bound(self):
        """Verify raises if casting to float and not fully bound."""

        x = Parameter("x")
        y = Parameter("y")
        bound_expr = (x + y).bind({x: 2.3})
        with self.assertRaisesRegex(TypeError, "unbound parameters"):
            float(bound_expr)

    def test_cast_to_int_when_bound(self):
        """Verify expression can be cast to an int when fully bound."""

        x = Parameter("x")
        bound_expr = x.bind({x: 2.3})
        self.assertEqual(int(bound_expr), 2)

    def test_cast_to_int_when_bound_truncates_after_evaluation(self):
        """Verify expression can be cast to an int when fully bound, but
        truncated only after evaluation."""

        x = Parameter("x")
        y = Parameter("y")
        bound_expr = (x + y).bind({x: 2.3, y: 0.8})
        self.assertEqual(int(bound_expr), 3)

    def test_cast_to_int_when_underlying_expression_bound(self):
        """Verify expression can be cast to a int when it still contains unbound parameters, but the
        underlying symbolic expression has a knowable value."""
        x = Parameter("x")
        expr = x - x + 2.3
        self.assertEqual(int(expr), 2)

    def test_raise_if_cast_to_int_when_not_fully_bound(self):
        """Verify raises if casting to int and not fully bound."""

        x = Parameter("x")
        y = Parameter("y")
        bound_expr = (x + y).bind({x: 2.3})
        with self.assertRaisesRegex(TypeError, "unbound parameters"):
            int(bound_expr)

    def test_raise_if_sub_unknown_parameters(self):
        """Verify we raise if asked to sub a parameter not in self."""
        x = Parameter("x")
        expr = x + 2

        y = Parameter("y")
        z = Parameter("z")

        with self.assertRaisesRegex(CircuitError, "not present"):
            expr.subs({y: z})

    def test_sub_allow_unknown_parameters(self):
        """Verify we raise if asked to sub a parameter not in self."""
        x = Parameter("x")
        expr = x + 2

        y = Parameter("y")
        z = Parameter("z")

        subbed = expr.subs({y: z}, allow_unknown_parameters=True)
        self.assertEqual(subbed, expr)

    def test_raise_if_subbing_in_parameter_name_conflict(self):
        """Verify we raise if substituting in conflicting parameter names."""
        x = Parameter("x")
        y_first = Parameter("y")

        expr = x + y_first

        y_second = Parameter("y")

        # Replacing an existing name is okay.
        expr.subs({y_first: y_second})

        with self.assertRaisesRegex(CircuitError, "Name conflict"):
            expr.subs({x: y_second})

    def test_expressions_of_parameter_with_constant(self):
        """Verify operating on a Parameter with a constant."""

        good_constants = [2, 1.3, 0, -1, -1.0, np.pi, 1j]

        x = Parameter("x")

        for op in self.supported_operations:
            for const in good_constants:
                expr = op(const, x)
                bound_expr = expr.bind({x: 2.3})

                self.assertEqual(complex(bound_expr), op(const, 2.3))

                # Division by zero will raise. Tested elsewhere.
                if const == 0 and op == truediv:
                    continue

                # Repeat above, swapping position of Parameter and constant.
                expr = op(x, const)
                bound_expr = expr.bind({x: 2.3})

                res = complex(bound_expr)
                expected = op(2.3, const)
                self.assertTrue(cmath.isclose(res, expected), f"{res} != {expected}")

    def test_complex_parameter_bound_to_real(self):
        """Test a complex parameter expression can be real if bound correctly."""

        x, y = Parameter("x"), Parameter("y")

        with self.subTest("simple 1j * x"):
            qc = QuantumCircuit(1)
            qc.rx(1j * x, 0)
            bound = qc.bind_parameters({x: 1j})
            ref = QuantumCircuit(1)
            ref.rx(-1, 0)
            self.assertEqual(bound, ref)

        with self.subTest("more complex expression"):
            qc = QuantumCircuit(1)
            qc.rx(0.5j * x - y * y + 2 * y, 0)
            bound = qc.bind_parameters({x: -4, y: 1j})
            ref = QuantumCircuit(1)
            ref.rx(1, 0)
            self.assertEqual(bound, ref)

    def test_complex_angle_raises_when_not_supported(self):
        """Test parameters are validated when fully bound and errors are raised accordingly."""
        x = Parameter("x")
        qc = QuantumCircuit(1)
        qc.r(x, 1j * x, 0)

        with self.subTest("binding x to 0 yields real parameters"):
            bound = qc.bind_parameters({x: 0})
            ref = QuantumCircuit(1)
            ref.r(0, 0, 0)
            self.assertEqual(bound, ref)

        with self.subTest("binding x to 1 yields complex parameters"):
            # RGate does not support complex parameters
            with self.assertRaises(CircuitError):
                bound = qc.bind_parameters({x: 1})

    def test_operating_on_a_parameter_with_a_non_float_will_raise(self):
        """Verify operations between a Parameter and a non-float will raise."""

        bad_constants = ["1", np.Inf, np.NaN, None, {}, []]

        x = Parameter("x")

        for op in self.supported_operations:
            for const in bad_constants:
                with self.subTest(op=op, const=const):
                    with self.assertRaises(TypeError):
                        _ = op(const, x)

                    with self.assertRaises(TypeError):
                        _ = op(x, const)

    def test_expressions_division_by_zero(self):
        """Verify dividing a Parameter by 0, or binding 0 as a denominator raises."""

        x = Parameter("x")

        with self.assertRaises(ZeroDivisionError):
            _ = x / 0

        with self.assertRaises(ZeroDivisionError):
            _ = x / 0.0

        expr = 2 / x

        with self.assertRaises(ZeroDivisionError):
            _ = expr.bind({x: 0})

        with self.assertRaises(ZeroDivisionError):
            _ = expr.bind({x: 0.0})

    def test_expressions_of_parameter_with_parameter(self):
        """Verify operating on two Parameters."""

        x = Parameter("x")
        y = Parameter("y")

        for op in self.supported_operations:
            expr = op(x, y)

            partially_bound_expr = expr.bind({x: 2.3})

            self.assertEqual(partially_bound_expr.parameters, {y})

            fully_bound_expr = partially_bound_expr.bind({y: -np.pi})

            self.assertEqual(fully_bound_expr.parameters, set())
            self.assertEqual(float(fully_bound_expr), op(2.3, -np.pi))

            bound_expr = expr.bind({x: 2.3, y: -np.pi})

            self.assertEqual(bound_expr.parameters, set())
            self.assertEqual(float(bound_expr), op(2.3, -np.pi))

    def test_expressions_operation_order(self):
        """Verify ParameterExpressions respect order of operations."""

        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")

        # Parenthesis before multiplication/division
        expr = (x + y) * z
        bound_expr = expr.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr), 9)

        expr = x * (y + z)
        bound_expr = expr.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr), 5)

        # Multiplication/division before addition/subtraction
        expr = x + y * z
        bound_expr = expr.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr), 7)

        expr = x * y + z
        bound_expr = expr.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr), 5)

    def test_nested_expressions(self):
        """Verify ParameterExpressions can also be the target of operations."""

        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")

        expr1 = x * y
        expr2 = expr1 + z
        bound_expr2 = expr2.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr2), 5)

    def test_negated_expression(self):
        """Verify ParameterExpressions can be negated."""

        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")

        expr1 = -x + y
        expr2 = -expr1 * (-z)
        bound_expr2 = expr2.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr2), 3)

    def test_standard_cu3(self):
        """This tests parameter negation in standard extension gate cu3."""
        from qiskit.circuit.library import CU3Gate

        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")
        qc = qiskit.QuantumCircuit(2)
        qc.append(CU3Gate(x, y, z), [0, 1])
        try:
            qc.decompose()
        except TypeError:
            self.fail("failed to decompose cu3 gate with negated parameter expression")

    def test_name_collision(self):
        """Verify Expressions of distinct Parameters of shared name raises."""

        x = Parameter("p")
        y = Parameter("p")

        # Expression of the same Parameter are valid.
        _ = x + x
        _ = x - x
        _ = x * x
        _ = x / x

        with self.assertRaises(CircuitError):
            _ = x + y
        with self.assertRaises(CircuitError):
            _ = x - y
        with self.assertRaises(CircuitError):
            _ = x * y
        with self.assertRaises(CircuitError):
            _ = x / y

    @combine(target_type=["gate", "instruction"], order=["bind-decompose", "decompose-bind"])
    def test_to_instruction_with_expression(self, target_type, order):
        """Test preservation of expressions via parameterized instructions.

                  ┌───────┐┌──────────┐┌───────────┐
        qr1_0: |0>┤ Rx(θ) ├┤ Rz(pi/2) ├┤ Ry(phi*θ) ├
                  └───────┘└──────────┘└───────────┘

                     ┌───────────┐
        qr2_0: |0>───┤ Ry(delta) ├───
                  ┌──┴───────────┴──┐
        qr2_1: |0>┤ Circuit0(phi,θ) ├
                  └─────────────────┘
        qr2_2: |0>───────────────────
        """

        theta = Parameter("θ")
        phi = Parameter("phi")
        qr1 = QuantumRegister(1, name="qr1")
        qc1 = QuantumCircuit(qr1)

        qc1.rx(theta, qr1)
        qc1.rz(np.pi / 2, qr1)
        qc1.ry(theta * phi, qr1)

        if target_type == "gate":
            gate = qc1.to_gate()
        elif target_type == "instruction":
            gate = qc1.to_instruction()

        self.assertEqual(gate.params, [phi, theta])

        delta = Parameter("delta")
        qr2 = QuantumRegister(3, name="qr2")
        qc2 = QuantumCircuit(qr2)
        qc2.ry(delta, qr2[0])
        qc2.append(gate, qargs=[qr2[1]])

        self.assertEqual(qc2.parameters, {delta, theta, phi})

        binds = {delta: 1, theta: 2, phi: 3}
        expected_qc = QuantumCircuit(qr2)
        expected_qc.rx(2, 1)
        expected_qc.rz(np.pi / 2, 1)
        expected_qc.ry(3 * 2, 1)
        expected_qc.r(1, np.pi / 2, 0)

        if order == "bind-decompose":
            decomp_bound_qc = qc2.assign_parameters(binds).decompose()
        elif order == "decompose-bind":
            decomp_bound_qc = qc2.decompose().assign_parameters(binds)

        self.assertEqual(decomp_bound_qc.parameters, set())
        self.assertEqual(decomp_bound_qc, expected_qc)

    @combine(target_type=["gate", "instruction"], order=["bind-decompose", "decompose-bind"])
    def test_to_instruction_expression_parameter_map(self, target_type, order):
        """Test preservation of expressions via instruction parameter_map."""

        theta = Parameter("θ")
        phi = Parameter("phi")
        qr1 = QuantumRegister(1, name="qr1")
        qc1 = QuantumCircuit(qr1)

        qc1.rx(theta, qr1)
        qc1.rz(np.pi / 2, qr1)
        qc1.ry(theta * phi, qr1)

        theta_p = Parameter("theta")
        phi_p = Parameter("phi")

        if target_type == "gate":
            gate = qc1.to_gate(parameter_map={theta: theta_p, phi: phi_p})
        elif target_type == "instruction":
            gate = qc1.to_instruction(parameter_map={theta: theta_p, phi: phi_p})

        self.assertListEqual(gate.params, [theta_p, phi_p])

        delta = Parameter("delta")
        qr2 = QuantumRegister(3, name="qr2")
        qc2 = QuantumCircuit(qr2)
        qc2.ry(delta, qr2[0])
        qc2.append(gate, qargs=[qr2[1]])

        self.assertListEqual(list(qc2.parameters), [delta, phi_p, theta_p])

        binds = {delta: 1, theta_p: 2, phi_p: 3}
        expected_qc = QuantumCircuit(qr2)
        expected_qc.rx(2, 1)
        expected_qc.rz(np.pi / 2, 1)
        expected_qc.ry(3 * 2, 1)
        expected_qc.r(1, np.pi / 2, 0)

        if order == "bind-decompose":
            decomp_bound_qc = qc2.assign_parameters(binds).decompose()
        elif order == "decompose-bind":
            decomp_bound_qc = qc2.decompose().assign_parameters(binds)

        self.assertEqual(decomp_bound_qc.parameters, set())
        self.assertEqual(decomp_bound_qc, expected_qc)

    def test_binding_across_broadcast_instruction(self):
        """Bind a parameter which was included via a broadcast instruction."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/3008

        theta = Parameter("θ")
        n = 5

        qc = QuantumCircuit(n, 1)

        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)

        qc.barrier()
        qc.rz(theta, range(n))
        qc.barrier()

        for i in reversed(range(n - 1)):
            qc.cx(i, i + 1)
        qc.h(0)
        qc.measure(0, 0)

        theta_range = np.linspace(0, 2 * np.pi, 128)
        circuits = [qc.assign_parameters({theta: theta_val}) for theta_val in theta_range]

        self.assertEqual(len(circuits), len(theta_range))
        for theta_val, bound_circ in zip(theta_range, circuits):
            rz_gates = [
                inst.operation for inst in bound_circ.data if isinstance(inst.operation, RZGate)
            ]

            self.assertEqual(len(rz_gates), n)
            self.assertTrue(all(float(gate.params[0]) == theta_val for gate in rz_gates))

    def test_substituting_parameter_with_simple_expression(self):
        """Substitute a simple parameter expression for a parameter."""
        x = Parameter("x")

        y = Parameter("y")
        sub_ = y / 2

        updated_expr = x.subs({x: sub_})

        expected = y / 2

        self.assertEqual(updated_expr, expected)

    def test_substituting_parameter_with_compound_expression(self):
        """Substitute a simple parameter expression for a parameter."""
        x = Parameter("x")

        y = Parameter("y")
        z = Parameter("z")
        sub_ = y * z

        updated_expr = x.subs({x: sub_})

        expected = y * z

        self.assertEqual(updated_expr, expected)

    def test_substituting_simple_with_simple_expression(self):
        """Substitute a simple parameter expression in a parameter expression."""
        x = Parameter("x")
        expr = x * x

        y = Parameter("y")
        sub_ = y / 2

        updated_expr = expr.subs({x: sub_})

        expected = y * y / 4

        self.assertEqual(updated_expr, expected)

    def test_substituting_compound_expression(self):
        """Substitute a compound parameter expression in a parameter expression."""
        x = Parameter("x")
        expr = x * x

        y = Parameter("y")
        z = Parameter("z")
        sub_ = y + z

        updated_expr = expr.subs({x: sub_})

        expected = (y + z) * (y + z)

        self.assertEqual(updated_expr, expected)

    def test_conjugate(self):
        """Test calling conjugate on a ParameterExpression."""
        x = Parameter("x")
        self.assertEqual((x.conjugate() + 1j), (x - 1j).conjugate())

    @data(
        circlib.RGate,
        circlib.RXGate,
        circlib.RYGate,
        circlib.RZGate,
        circlib.RXXGate,
        circlib.RYYGate,
        circlib.RZXGate,
        circlib.RZZGate,
        circlib.CRXGate,
        circlib.CRYGate,
        circlib.CRZGate,
        circlib.XXPlusYYGate,
    )
    def test_bound_gate_to_matrix(self, gate_class):
        """Test to_matrix works if previously free parameters are bound.

        The conversion might fail, if trigonometric functions such as cos are called on the
        parameters and the parameters are still of type ParameterExpression.
        """
        num_parameters = 2 if gate_class == circlib.RGate else 1
        params = list(range(1, 1 + num_parameters))
        free_params = ParameterVector("th", num_parameters)
        gate = gate_class(*params)
        num_qubits = gate.num_qubits

        circuit = QuantumCircuit(num_qubits)
        circuit.append(gate_class(*free_params), list(range(num_qubits)))
        bound_circuit = circuit.assign_parameters({free_params: params})

        np.testing.assert_array_almost_equal(Operator(bound_circuit).data, gate.to_matrix())

    def test_parameter_expression_grad(self):
        """Verify correctness of ParameterExpression gradients."""

        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")

        with self.subTest(msg="first order gradient"):
            expr = (x + y) * z
            self.assertEqual(expr.gradient(x), z)
            self.assertEqual(expr.gradient(z), (x + y))

        with self.subTest(msg="second order gradient"):
            expr = x * x
            self.assertEqual(expr.gradient(x), 2 * x)
            self.assertEqual(expr.gradient(x).gradient(x), 2)

    def test_bound_expression_is_real(self):
        """Test is_real on bound parameters."""
        x = Parameter("x")
        self.assertEqual(x.is_real(), None)
        self.assertEqual((1j * x).is_real(), None)

        expr = 1j * x
        bound = expr.bind({x: 2})
        self.assertEqual(bound.is_real(), False)

        bound = x.bind({x: 0 + 0j})
        self.assertEqual(bound.is_real(), True)

        bound = x.bind({x: 0 + 1j})
        self.assertEqual(bound.is_real(), False)

        bound = x.bind({x: 1 + 0j})
        self.assertEqual(bound.is_real(), True)

        bound = x.bind({x: 1 + 1j})
        self.assertEqual(bound.is_real(), False)


class TestParameterEquality(QiskitTestCase):
    """Test equality of Parameters and ParameterExpressions."""

    def test_parameter_equal_self(self):
        """Verify a parameter is equal to it self."""
        theta = Parameter("theta")
        self.assertEqual(theta, theta)

    def test_parameter_not_equal_param_of_same_name(self):
        """Verify a parameter is not equal to a Parameter of the same name."""
        theta1 = Parameter("theta")
        theta2 = Parameter("theta")
        self.assertNotEqual(theta1, theta2)

    def test_parameter_expression_equal_to_self(self):
        """Verify an expression is equal to itself."""
        theta = Parameter("theta")
        expr = 2 * theta

        self.assertEqual(expr, expr)

    def test_parameter_expression_equal_to_identical(self):
        """Verify an expression is equal an identical expression."""
        theta = Parameter("theta")
        expr1 = 2 * theta
        expr2 = 2 * theta

        self.assertEqual(expr1, expr2)

    def test_parameter_expression_equal_floats_to_ints(self):
        """Verify an expression with float and int is identical."""
        theta = Parameter("theta")
        expr1 = 2.0 * theta
        expr2 = 2 * theta

        self.assertEqual(expr1, expr2)

    def test_parameter_expression_not_equal_if_params_differ(self):
        """Verify expressions not equal if parameters are different."""
        theta1 = Parameter("theta")
        theta2 = Parameter("theta")
        expr1 = 2 * theta1
        expr2 = 2 * theta2

        self.assertNotEqual(expr1, expr2)

    def test_parameter_equal_to_identical_expression(self):
        """Verify parameters and ParameterExpressions can be equal if identical."""
        theta = Parameter("theta")
        phi = Parameter("phi")

        expr = (theta + phi).bind({phi: 0})

        self.assertEqual(expr, theta)
        self.assertEqual(theta, expr)

    def test_parameter_symbol_equal_after_ufunc(self):
        """Verfiy ParameterExpression phi
        and ParameterExpression cos(phi) have the same symbol map"""
        phi = Parameter("phi")
        cos_phi = np.cos(phi)
        self.assertEqual(phi._parameter_symbols, cos_phi._parameter_symbols)


class TestParameterReferences(QiskitTestCase):
    """Test the ParameterReferences class."""

    def test_equal_inst_diff_instance(self):
        """Different value equal instructions are treated as distinct."""

        theta = Parameter("theta")
        gate1 = RZGate(theta)
        gate2 = RZGate(theta)

        self.assertIsNot(gate1, gate2)
        self.assertEqual(gate1, gate2)

        refs = ParameterReferences(((gate1, 0), (gate2, 0)))

        # test __contains__
        self.assertIn((gate1, 0), refs)
        self.assertIn((gate2, 0), refs)

        gate_ids = {id(gate1), id(gate2)}
        self.assertEqual(gate_ids, {id(gate) for gate, _ in refs})
        self.assertTrue(all(idx == 0 for _, idx in refs))

    def test_pickle_unpickle(self):
        """Membership testing after pickle/unpickle."""

        theta = Parameter("theta")
        gate1 = RZGate(theta)
        gate2 = RZGate(theta)

        self.assertIsNot(gate1, gate2)
        self.assertEqual(gate1, gate2)

        refs = ParameterReferences(((gate1, 0), (gate2, 0)))

        to_pickle = (gate1, refs)
        pickled = pickle.dumps(to_pickle)
        (gate1_new, refs_new) = pickle.loads(pickled)

        self.assertEqual(len(refs_new), len(refs))
        self.assertNotIn((gate1, 0), refs_new)
        self.assertIn((gate1_new, 0), refs_new)

    def test_equal_inst_same_instance(self):
        """Referentially equal instructions are treated as same."""

        theta = Parameter("theta")
        gate = RZGate(theta)

        refs = ParameterReferences(((gate, 0), (gate, 0)))

        self.assertIn((gate, 0), refs)
        self.assertEqual(len(refs), 1)
        self.assertIs(next(iter(refs))[0], gate)
        self.assertEqual(next(iter(refs))[1], 0)

    def test_extend_refs(self):
        """Extending references handles duplicates."""

        theta = Parameter("theta")
        ref0 = (RZGate(theta), 0)
        ref1 = (RZGate(theta), 0)
        ref2 = (RZGate(theta), 0)

        refs = ParameterReferences((ref0,))
        refs |= ParameterReferences((ref0, ref1, ref2, ref1, ref0))

        self.assertEqual(refs, ParameterReferences((ref0, ref1, ref2)))

    def test_copy_param_refs(self):
        """Copy of parameter references is a shallow copy."""

        theta = Parameter("theta")
        ref0 = (RZGate(theta), 0)
        ref1 = (RZGate(theta), 0)
        ref2 = (RZGate(theta), 0)
        ref3 = (RZGate(theta), 0)

        refs = ParameterReferences((ref0, ref1))
        refs_copy = refs.copy()

        # Check same gate instances in copy
        gate_ids = {id(ref0[0]), id(ref1[0])}
        self.assertEqual({id(gate) for gate, _ in refs_copy}, gate_ids)

        # add new ref to original and check copy not modified
        refs.add(ref2)
        self.assertNotIn(ref2, refs_copy)
        self.assertEqual(refs_copy, ParameterReferences((ref0, ref1)))

        # add new ref to copy and check original not modified
        refs_copy.add(ref3)
        self.assertNotIn(ref3, refs)
        self.assertEqual(refs, ParameterReferences((ref0, ref1, ref2)))


class TestParameterTable(QiskitTestCase):
    """Test the ParameterTable class."""

    def test_init_param_table(self):
        """Parameter table init from mapping."""

        p1 = Parameter("theta")
        p2 = Parameter("theta")

        ref0 = (RZGate(p1), 0)
        ref1 = (RZGate(p1), 0)
        ref2 = (RZGate(p2), 0)

        mapping = {p1: ParameterReferences((ref0, ref1)), p2: ParameterReferences((ref2,))}

        table = ParameterTable(mapping)

        # make sure editing mapping doesn't change `table`
        del mapping[p1]

        self.assertEqual(table[p1], ParameterReferences((ref0, ref1)))
        self.assertEqual(table[p2], ParameterReferences((ref2,)))

    def test_set_references(self):
        """References replacement by parameter key."""

        p1 = Parameter("theta")

        ref0 = (RZGate(p1), 0)
        ref1 = (RZGate(p1), 0)

        table = ParameterTable()
        table[p1] = ParameterReferences((ref0, ref1))
        self.assertEqual(table[p1], ParameterReferences((ref0, ref1)))

        table[p1] = ParameterReferences((ref1,))
        self.assertEqual(table[p1], ParameterReferences((ref1,)))

    def test_set_references_from_iterable(self):
        """Parameter table init from iterable."""

        p1 = Parameter("theta")

        ref0 = (RZGate(p1), 0)
        ref1 = (RZGate(p1), 0)
        ref2 = (RZGate(p1), 0)

        table = ParameterTable({p1: ParameterReferences((ref0, ref1))})
        table[p1] = (ref2, ref1, ref0)

        self.assertEqual(table[p1], ParameterReferences((ref2, ref1, ref0)))


class TestParameterView(QiskitTestCase):
    """Test the ParameterView object."""

    def setUp(self):
        super().setUp()
        x, y, z = Parameter("x"), Parameter("y"), Parameter("z")
        self.params = [x, y, z]
        self.view1 = ParameterView([x, y])
        self.view2 = ParameterView([y, z])
        self.view3 = ParameterView([x])

    def test_and(self):
        """Test __and__."""
        self.assertEqual(self.view1 & self.view2, {self.params[1]})

    def test_or(self):
        """Test __or__."""
        self.assertEqual(self.view1 | self.view2, set(self.params))

    def test_xor(self):
        """Test __xor__."""
        self.assertEqual(self.view1 ^ self.view2, {self.params[0], self.params[2]})

    def test_len(self):
        """Test __len__."""
        self.assertEqual(len(self.view1), 2)

    def test_le(self):
        """Test __le__."""
        self.assertTrue(self.view1 <= self.view1)
        self.assertFalse(self.view1 <= self.view3)

    def test_lt(self):
        """Test __lt__."""
        self.assertTrue(self.view3 < self.view1)

    def test_ge(self):
        """Test __ge__."""
        self.assertTrue(self.view1 >= self.view1)
        self.assertFalse(self.view3 >= self.view1)

    def test_gt(self):
        """Test __lt__."""
        self.assertTrue(self.view1 > self.view3)

    def test_eq(self):
        """Test __eq__."""
        self.assertTrue(self.view1 == self.view1)
        self.assertFalse(self.view3 == self.view1)

    def test_ne(self):
        """Test __eq__."""
        self.assertTrue(self.view1 != self.view2)
        self.assertFalse(self.view3 != self.view3)


if __name__ == "__main__":
    unittest.main()

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

"""Test cases for the schedule block qpy loading and saving."""

import io
import struct

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit, Parameter, Gate
from qiskit.providers.fake_provider import Fake27QPulseV1, GenericBackendV2
from qiskit.exceptions import QiskitError
from qiskit.qpy import dump, load, formats, QPY_COMPATIBILITY_VERSION
from qiskit.qpy.common import QPY_VERSION
from qiskit.transpiler import PassManager, TranspileLayout
from qiskit.transpiler import passes
from qiskit.compiler import transpile
from qiskit.utils import optionals
from qiskit.qpy.formats import FILE_HEADER_V10_PACK, FILE_HEADER_V10, FILE_HEADER_V10_SIZE
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class QpyCircuitTestCase(QiskitTestCase):
    """QPY schedule testing platform."""

    def assert_roundtrip_equal(self, circuit, version=None, use_symengine=None):
        """QPY roundtrip equal test."""
        qpy_file = io.BytesIO()
        if use_symengine is None:
            dump(circuit, qpy_file, version=version)
        else:
            dump(circuit, qpy_file, version=version, use_symengine=use_symengine)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]

        self.assertEqual(circuit, new_circuit)
        self.assertEqual(circuit.layout, new_circuit.layout)
        if version is not None:
            qpy_file.seek(0)
            file_version = struct.unpack("!6sB", qpy_file.read(7))[1]
            self.assertEqual(
                version,
                file_version,
                f"Generated QPY file version {file_version} does not match request version {version}",
            )


@ddt
class TestCalibrationPasses(QpyCircuitTestCase):
    """QPY round-trip test case of transpiled circuits with pulse level optimization."""

    def setUp(self):
        super().setUp()
        # TODO remove context once https://github.com/Qiskit/qiskit/issues/12759 is fixed
        with self.assertWarns(DeprecationWarning):
            # This backend provides CX(0,1) with native ECR direction.
            self.inst_map = Fake27QPulseV1().defaults().instruction_schedule_map

    @data(0.1, 0.7, 1.5)
    def test_rzx_calibration(self, angle):
        """RZX builder calibration pass with echo."""
        with self.assertWarns(DeprecationWarning):
            pass_ = passes.RZXCalibrationBuilder(self.inst_map)
        pass_manager = PassManager(pass_)
        test_qc = QuantumCircuit(2)
        test_qc.rzx(angle, 0, 1)
        rzx_qc = pass_manager.run(test_qc)
        with self.assertWarns(DeprecationWarning):
            self.assert_roundtrip_equal(rzx_qc)

    @data(0.1, 0.7, 1.5)
    def test_rzx_calibration_echo(self, angle):
        """RZX builder calibration pass without echo."""
        with self.assertWarns(DeprecationWarning):
            pass_ = passes.RZXCalibrationBuilderNoEcho(self.inst_map)
        pass_manager = PassManager(pass_)
        test_qc = QuantumCircuit(2)
        test_qc.rzx(angle, 0, 1)
        rzx_qc = pass_manager.run(test_qc)
        with self.assertWarns(DeprecationWarning):
            self.assert_roundtrip_equal(rzx_qc)


class TestVersions(QpyCircuitTestCase):
    """Test version handling in qpy."""

    def test_invalid_qpy_version(self):
        """Test a descriptive exception is raised if QPY version is too new."""
        with io.BytesIO() as buf:
            buf.write(
                struct.pack(formats.FILE_HEADER_PACK, b"QISKIT", QPY_VERSION + 4, 42, 42, 1, 2)
            )
            buf.seek(0)
            with self.assertRaisesRegex(QiskitError, str(QPY_VERSION + 4)):
                load(buf)


@ddt
class TestLayout(QpyCircuitTestCase):
    """Test circuit serialization for layout preservation."""

    @data(0, 1, 2, 3)
    def test_transpile_layout(self, opt_level):
        """Test layout preserved after transpile."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        backend = GenericBackendV2(num_qubits=127, seed=42)
        tqc = transpile(qc, backend, optimization_level=opt_level)
        self.assert_roundtrip_equal(tqc)

    @data(0, 1, 2, 3)
    def test_transpile_with_routing(self, opt_level):
        """Test full layout with routing is preserved."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        backend = GenericBackendV2(num_qubits=127, seed=42)
        tqc = transpile(qc, backend, optimization_level=opt_level)
        self.assert_roundtrip_equal(tqc)

    @data(0, 1, 2, 3)
    def test_transpile_layout_explicit_None_final_layout(self, opt_level):
        """Test layout preserved after transpile."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        backend = GenericBackendV2(num_qubits=127, seed=42)
        tqc = transpile(qc, backend, optimization_level=opt_level)
        tqc.layout.final_layout = None
        self.assert_roundtrip_equal(tqc)

    def test_empty_layout(self):
        """Test an empty layout is preserved correctly."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        qc._layout = TranspileLayout(None, None, None)
        self.assert_roundtrip_equal(qc)

    def test_overlapping_definitions(self):
        """Test serialization of custom gates with overlapping definitions."""

        class MyParamGate(Gate):
            """Custom gate class with a parameter."""

            def __init__(self, phi):
                super().__init__("my_gate", 1, [phi])

            def _define(self):
                qc = QuantumCircuit(1)
                qc.rx(self.params[0], 0)
                self.definition = qc

        theta = Parameter("theta")
        two_theta = 2 * theta

        qc = QuantumCircuit(1)
        qc.append(MyParamGate(1.1), [0])
        qc.append(MyParamGate(1.2), [0])
        qc.append(MyParamGate(3.14159), [0])
        qc.append(MyParamGate(theta), [0])
        qc.append(MyParamGate(two_theta), [0])
        with io.BytesIO() as qpy_file:
            dump(qc, qpy_file)
            qpy_file.seek(0)
            new_circ = load(qpy_file)[0]
        # Custom gate classes are lowered to Gate to avoid arbitrary code
        # execution on deserialization. To compare circuit equality we
        # need to go instruction by instruction and check that they're
        # equivalent instead of doing a circuit equality check
        for new_inst, old_inst in zip(new_circ.data, qc.data):
            new_gate = new_inst.operation
            old_gate = old_inst.operation
            self.assertIsInstance(new_gate, Gate)
            self.assertEqual(new_gate.name, old_gate.name)
            self.assertEqual(new_gate.params, old_gate.params)
            self.assertEqual(new_gate.definition, old_gate.definition)

    @data(0, 1, 2, 3)
    def test_custom_register_name(self, opt_level):
        """Test layout preserved with custom register names."""
        qr = QuantumRegister(5, name="abc123")
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        backend = GenericBackendV2(num_qubits=127, seed=42)
        tqc = transpile(qc, backend, optimization_level=opt_level)
        self.assert_roundtrip_equal(tqc)

    @data(0, 1, 2, 3)
    def test_no_register(self, opt_level):
        """Test layout preserved with no register."""
        qubits = [Qubit(), Qubit()]
        qc = QuantumCircuit(qubits)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        backend = GenericBackendV2(num_qubits=127, seed=42)
        tqc = transpile(qc, backend, optimization_level=opt_level)
        # Manually validate to deal with qubit equality needing exact objects
        qpy_file = io.BytesIO()
        dump(tqc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(tqc, new_circuit)
        initial_layout_old = tqc.layout.initial_layout.get_physical_bits()
        initial_layout_new = new_circuit.layout.initial_layout.get_physical_bits()
        for i in initial_layout_old:
            self.assertIsInstance(initial_layout_old[i], Qubit)
            self.assertIsInstance(initial_layout_new[i], Qubit)
            if initial_layout_old[i]._register is not None:
                self.assertEqual(initial_layout_new[i], initial_layout_old[i])
            else:
                self.assertIsNone(initial_layout_new[i]._register)
                self.assertIsNone(initial_layout_old[i]._index)
                self.assertIsNone(initial_layout_new[i]._index)
        self.assertEqual(
            list(tqc.layout.input_qubit_mapping.values()),
            list(new_circuit.layout.input_qubit_mapping.values()),
        )
        self.assertEqual(tqc.layout.final_layout, new_circuit.layout.final_layout)


class TestVersionArg(QpyCircuitTestCase):
    """Test explicitly setting a qpy version in dump()."""

    def test_custom_gate_name_overlap_persists_with_minimum_version(self):
        """Assert the fix in version 11 doesn't get used if an older version is request."""

        class MyParamGate(Gate):
            """Custom gate class with a parameter."""

            def __init__(self, phi):
                super().__init__("my_gate", 1, [phi])

            def _define(self):
                qc = QuantumCircuit(1)
                qc.rx(self.params[0], 0)
                self.definition = qc

        theta = Parameter("theta")
        two_theta = 2 * theta

        qc = QuantumCircuit(1)
        qc.append(MyParamGate(1.1), [0])
        qc.append(MyParamGate(1.2), [0])
        qc.append(MyParamGate(3.14159), [0])
        qc.append(MyParamGate(theta), [0])
        qc.append(MyParamGate(two_theta), [0])
        with io.BytesIO() as qpy_file:
            dump(qc, qpy_file, version=10)
            qpy_file.seek(0)
            new_circ = load(qpy_file)[0]
        # Custom gate classes are lowered to Gate to avoid arbitrary code
        # execution on deserialization. To compare circuit equality we
        # need to go instruction by instruction and check that they're
        # equivalent instead of doing a circuit equality check
        first_gate = None
        for new_inst, old_inst in zip(new_circ.data, qc.data):
            new_gate = new_inst.operation
            old_gate = old_inst.operation
            self.assertIsInstance(new_gate, Gate)
            self.assertEqual(new_gate.name, old_gate.name)
            self.assertEqual(new_gate.params, old_gate.params)
            if first_gate is None:
                first_gate = new_gate
                continue
            # This is incorrect behavior. This test is explicitly validating
            # that the version kwarg being set to 10 causes the buggy behavior
            # on that version of qpy
            self.assertEqual(new_gate.definition, first_gate.definition)

    def test_invalid_version_value(self):
        """Assert we raise an error with an invalid version request."""
        qc = QuantumCircuit(2)
        with self.assertRaises(ValueError):
            dump(qc, io.BytesIO(), version=3)

    def test_compatibility_version_roundtrip(self):
        """Test the version is set correctly when specified."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        self.assert_roundtrip_equal(qc, version=QPY_COMPATIBILITY_VERSION)

    def test_nested_params_subs(self):
        """Test substitution works."""
        qc = QuantumCircuit(1)
        a = Parameter("a")
        b = Parameter("b")
        expr = a + b
        expr = expr.subs({b: a})
        qc.ry(expr, 0)
        self.assert_roundtrip_equal(qc)

    def test_all_the_expression_ops(self):
        """Test a circuit with an expression that uses all the ops available."""
        qc = QuantumCircuit(1)
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        d = Parameter("d")

        expression = (a + b.sin() / 4) * c**2
        final_expr = (
            (expression.cos() + d.arccos() - d.arcsin() + d.arctan() + d.tan()) / d.exp()
            + expression.gradient(a)
            + expression.log()
            - a.sin()
            - b.conjugate()
        )
        final_expr = final_expr.abs()
        final_expr = final_expr.subs({c: a})

        qc.rx(final_expr, 0)
        self.assert_roundtrip_equal(qc)

    def test_rpow(self):
        """Test rpow works as expected"""
        qc = QuantumCircuit(1)
        a = Parameter("A")
        b = Parameter("B")
        expr = 3.14159**a
        expr = expr**b
        expr = 1.2345**expr
        qc.ry(expr, 0)
        self.assert_roundtrip_equal(qc)

    def test_rsub(self):
        """Test rsub works as expected"""
        qc = QuantumCircuit(1)
        a = Parameter("A")
        b = Parameter("B")
        expr = 3.14159 - a
        expr = expr - b
        expr = 1.2345 - expr
        qc.ry(expr, 0)
        self.assert_roundtrip_equal(qc)

    def test_rdiv(self):
        """Test rdiv works as expected"""
        qc = QuantumCircuit(1)
        a = Parameter("A")
        b = Parameter("B")
        expr = 3.14159 / a
        expr = expr / b
        expr = 1.2345 / expr
        qc.ry(expr, 0)
        self.assert_roundtrip_equal(qc)


class TestUseSymengineFlag(QpyCircuitTestCase):
    """Test that the symengine flag works correctly."""

    def test_use_symengine_with_bool_like(self):
        """Test that the use_symengine flag is set correctly with a bool-like input."""
        theta = Parameter("theta")
        two_theta = 2 * theta
        qc = QuantumCircuit(1)
        qc.rx(two_theta, 0)
        qc.measure_all()
        # Assert Roundtrip works
        self.assert_roundtrip_equal(qc, use_symengine=optionals.HAS_SYMENGINE, version=10)
        # Also check the qpy symbolic expression encoding is correct in the
        # payload
        with io.BytesIO() as file_obj:
            dump(qc, file_obj, use_symengine=optionals.HAS_SYMENGINE)
            file_obj.seek(0)
            header_data = FILE_HEADER_V10._make(
                struct.unpack(
                    FILE_HEADER_V10_PACK,
                    file_obj.read(FILE_HEADER_V10_SIZE),
                )
            )
            self.assertEqual(header_data.symbolic_encoding, b"e")

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test QASM3 exporter."""

# We can't really help how long the lines output by the exporter are in some cases.
# pylint: disable=line-too-long

from io import StringIO
from math import pi
import re

from ddt import ddt, data

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import Parameter, Qubit, Clbit, Gate, Delay, Barrier, ParameterVector
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import CASE_DEFAULT
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.qasm3 import Exporter, dumps, dump, QASM3ExporterError, ExperimentalFeatures
from qiskit.qasm3.exporter import QASM3Builder
from qiskit.qasm3.printer import BasicPrinter
from qiskit.quantum_info import Pauli
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestQASM3Functions(QiskitTestCase):
    """QASM3 module - high level functions"""

    def setUp(self):
        self.circuit = QuantumCircuit(2)
        self.circuit.u(2 * pi, 3 * pi, -5 * pi, 0)
        self.expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "qubit[2] q;",
                "U(2*pi, 3*pi, -5*pi) q[0];",
                "",
            ]
        )
        super().setUp()

    def test_dumps(self):
        """Test dumps."""
        result = dumps(self.circuit)
        self.assertEqual(result, self.expected_qasm)

    def test_dump(self):
        """Test dump into an IO stream."""
        io = StringIO()
        dump(self.circuit, io)
        result = io.getvalue()
        self.assertEqual(result, self.expected_qasm)


@ddt
class TestCircuitQASM3(QiskitTestCase):
    """QASM3 exporter."""

    maxDiff = 1_000_000

    @classmethod
    def setUpClass(cls):
        # These regexes are not perfect by any means, but sufficient for simple tests on controlled
        # input circuits.  They can allow false negatives (in which case, update the regex), but to
        # be useful for the tests must _never_ have false positive matches.  We use an explicit
        # space (`\s`) or semicolon rather than the end-of-word `\b` because we want to ensure that
        # the exporter isn't putting out invalid characters as part of the identifiers.
        cls.register_regex = re.compile(
            r"^\s*(let|(qu)?bit(\[\d+\])?)\s+(?P<name>\w+)[\s;]", re.U | re.M
        )
        scalar_type_names = {
            "angle",
            "duration",
            "float",
            "int",
            "stretch",
            "uint",
        }
        cls.scalar_parameter_regex = re.compile(
            r"^\s*((input|output|const)\s+)?"  # Modifier
            rf"({'|'.join(scalar_type_names)})\s*(\[[^\]]+\])?\s+"  # Type name and designator
            r"(?P<name>\w+)[\s;]",  # Parameter name
            re.U | re.M,
        )
        super().setUpClass()

    def test_regs_conds_qasm(self):
        """Test with registers and conditionals."""
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        cr = ClassicalRegister(3, "cr")
        qc = QuantumCircuit(qr1, qr2, cr)
        qc.measure(qr1[0], cr[0])
        qc.measure(qr2[0], cr[1])
        qc.measure(qr2[1], cr[2])
        qc.x(qr2[1]).c_if(cr, 0)
        qc.y(qr1[0]).c_if(cr, 1)
        qc.z(qr1[0]).c_if(cr, 2)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "bit[3] cr;",
                "qubit[1] qr1;",
                "qubit[2] qr2;",
                "cr[0] = measure qr1[0];",
                "cr[1] = measure qr2[0];",
                "cr[2] = measure qr2[1];",
                "if (cr == 0) {",
                "  x qr2[1];",
                "}",
                "if (cr == 1) {",
                "  y qr1[0];",
                "}",
                "if (cr == 2) {",
                "  z qr1[0];",
                "}",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_registers_as_aliases(self):
        """Test that different types of alias creation and concatenation work."""
        qubits = [Qubit() for _ in [None] * 10]
        first_four = QuantumRegister(name="first_four", bits=qubits[:4])
        last_five = QuantumRegister(name="last_five", bits=qubits[5:])
        alternate = QuantumRegister(name="alternate", bits=qubits[::2])
        sporadic = QuantumRegister(name="sporadic", bits=[qubits[4], qubits[2], qubits[9]])
        qc = QuantumCircuit(qubits, first_four, last_five, alternate, sporadic)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "qubit _qubit0;",
                "qubit _qubit1;",
                "qubit _qubit2;",
                "qubit _qubit3;",
                "qubit _qubit4;",
                "qubit _qubit5;",
                "qubit _qubit6;",
                "qubit _qubit7;",
                "qubit _qubit8;",
                "qubit _qubit9;",
                "let first_four = {_qubit0, _qubit1, _qubit2, _qubit3};",
                "let last_five = {_qubit5, _qubit6, _qubit7, _qubit8, _qubit9};",
                "let alternate = {first_four[0], first_four[2], _qubit4, last_five[1], last_five[3]};",
                "let sporadic = {alternate[2], alternate[1], last_five[4]};",
                "",
            ]
        )
        self.assertEqual(Exporter(allow_aliasing=True).dumps(qc), expected_qasm)

    def test_composite_circuit(self):
        """Test with a composite circuit instruction and barriers"""
        composite_circ_qreg = QuantumRegister(2)
        composite_circ = QuantumCircuit(composite_circ_qreg, name="composite_circ")
        composite_circ.h(0)
        composite_circ.x(1)
        composite_circ.cx(0, 1)
        composite_circ_instr = composite_circ.to_gate()

        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.append(composite_circ_instr, [0, 1])
        qc.measure([0, 1], [0, 1])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "gate composite_circ _gate_q_0, _gate_q_1 {",
                "  h _gate_q_0;",
                "  x _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "}",
                "bit[2] cr;",
                "qubit[2] qr;",
                "h qr[0];",
                "cx qr[0], qr[1];",
                "barrier qr[0], qr[1];",
                "composite_circ qr[0], qr[1];",
                "cr[0] = measure qr[0];",
                "cr[1] = measure qr[1];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_custom_gate(self):
        """Test custom gates (via to_gate)."""
        composite_circ_qreg = QuantumRegister(2)
        composite_circ = QuantumCircuit(composite_circ_qreg, name="composite_circ")
        composite_circ.h(0)
        composite_circ.x(1)
        composite_circ.cx(0, 1)
        composite_circ_instr = composite_circ.to_gate()

        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.append(composite_circ_instr, [0, 1])
        qc.measure([0, 1], [0, 1])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "gate composite_circ _gate_q_0, _gate_q_1 {",
                "  h _gate_q_0;",
                "  x _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "}",
                "bit[2] cr;",
                "qubit[2] qr;",
                "h qr[0];",
                "cx qr[0], qr[1];",
                "barrier qr[0], qr[1];",
                "composite_circ qr[0], qr[1];",
                "cr[0] = measure qr[0];",
                "cr[1] = measure qr[1];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_same_composite_circuits(self):
        """Test when a composite circuit is added to the circuit multiple times."""
        composite_circ_qreg = QuantumRegister(2)
        composite_circ = QuantumCircuit(composite_circ_qreg, name="composite_circ")
        composite_circ.h(0)
        composite_circ.x(1)
        composite_circ.cx(0, 1)
        composite_circ_instr = composite_circ.to_gate()

        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.append(composite_circ_instr, [0, 1])
        qc.append(composite_circ_instr, [0, 1])
        qc.measure([0, 1], [0, 1])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "gate composite_circ _gate_q_0, _gate_q_1 {",
                "  h _gate_q_0;",
                "  x _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "}",
                "bit[2] cr;",
                "qubit[2] qr;",
                "h qr[0];",
                "cx qr[0], qr[1];",
                "barrier qr[0], qr[1];",
                "composite_circ qr[0], qr[1];",
                "composite_circ qr[0], qr[1];",
                "cr[0] = measure qr[0];",
                "cr[1] = measure qr[1];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_composite_circuits_with_same_name(self):
        """Test when multiple composite circuit instructions same name and different
        implementation."""
        my_gate = QuantumCircuit(1, name="my_gate")
        my_gate.h(0)
        my_gate_inst1 = my_gate.to_gate()

        my_gate = QuantumCircuit(1, name="my_gate")
        my_gate.x(0)
        my_gate_inst2 = my_gate.to_gate()

        my_gate = QuantumCircuit(1, name="my_gate")
        my_gate.x(0)
        my_gate_inst3 = my_gate.to_gate()

        qr = QuantumRegister(1, name="qr")
        circuit = QuantumCircuit(qr, name="circuit")
        circuit.append(my_gate_inst1, [qr[0]])
        circuit.append(my_gate_inst2, [qr[0]])
        circuit.append(my_gate_inst3, [qr[0]])
        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "gate my_gate _gate_q_0 {",
                "  h _gate_q_0;",
                "}",
                "gate my_gate_0 _gate_q_0 {",
                "  x _gate_q_0;",
                "}",
                "gate my_gate_1 _gate_q_0 {",
                "  x _gate_q_0;",
                "}",
                "qubit[1] qr;",
                "my_gate qr[0];",
                "my_gate_0 qr[0];",
                "my_gate_1 qr[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(circuit), expected_qasm)

    def test_pi_disable_constants_false(self):
        """Test pi constant (disable_constants=False)"""
        circuit = QuantumCircuit(2)
        circuit.u(2 * pi, 3 * pi, -5 * pi, 0)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "qubit[2] q;",
                "U(2*pi, 3*pi, -5*pi) q[0];",
                "",
            ]
        )
        self.assertEqual(Exporter(disable_constants=False).dumps(circuit), expected_qasm)

    def test_pi_disable_constants_true(self):
        """Test pi constant (disable_constants=True)"""
        circuit = QuantumCircuit(2)
        circuit.u(2 * pi, 3 * pi, -5 * pi, 0)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "qubit[2] q;",
                "U(6.283185307179586, 9.42477796076938, -15.707963267948966) q[0];",
                "",
            ]
        )
        self.assertEqual(Exporter(disable_constants=True).dumps(circuit), expected_qasm)

    def test_custom_gate_with_unbound_parameter(self):
        """Test custom gate with unbound parameter."""
        parameter_a = Parameter("a")

        custom = QuantumCircuit(1, name="custom")
        custom.rx(parameter_a, 0)
        circuit = QuantumCircuit(1)
        circuit.append(custom.to_gate(), [0])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "input float[64] a;",
                "gate custom(a) _gate_q_0 {",
                "  rx(a) _gate_q_0;",
                "}",
                "qubit[1] q;",
                "custom(a) q[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(circuit), expected_qasm)

    def test_custom_gate_with_bound_parameter(self):
        """Test custom gate with bound parameter."""
        parameter_a = Parameter("a")

        custom = QuantumCircuit(1)
        custom.rx(parameter_a, 0)
        custom_gate = custom.assign_parameters({parameter_a: 0.5}).to_gate()
        custom_gate.name = "custom"

        circuit = QuantumCircuit(1)
        circuit.append(custom_gate, [0])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "gate custom _gate_q_0 {",
                "  rx(0.5) _gate_q_0;",
                "}",
                "qubit[1] q;",
                "custom q[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(circuit), expected_qasm)

    def test_custom_gate_with_params_bound_main_call(self):
        """Custom gate with unbound parameters that are bound in the main circuit"""
        parameter0 = Parameter("p0")
        parameter1 = Parameter("p1")

        custom = QuantumCircuit(2, name="custom")
        custom.rz(parameter0, 0)
        custom.rz(parameter1 / 2, 1)

        qr_all_qubits = QuantumRegister(3, "q")
        qr_r = QuantumRegister(3, "r")
        circuit = QuantumCircuit(qr_all_qubits, qr_r)
        circuit.append(custom.to_gate(), [qr_all_qubits[0], qr_r[0]])

        circuit.assign_parameters({parameter0: pi, parameter1: pi / 2}, inplace=True)

        # NOTE: this isn't exactly what we want; note that the parameters in the signature are not
        # actually used.  It would be fine to change the output of the exporter to make `custom` non
        # parametric in this case.
        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "gate custom(_gate_p_0, _gate_p_1) _gate_q_0, _gate_q_1 {",
                "  rz(pi) _gate_q_0;",
                "  rz(pi/4) _gate_q_1;",
                "}",
                "qubit[3] q;",
                "qubit[3] r;",
                "custom(pi, pi/2) q[0], r[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(circuit), expected_qasm)

    def test_multiple_pauli_evolution_gates(self):
        """Pauli evolution gates should be detected as distinct."""
        vec = ParameterVector("t", 3)
        qc = QuantumCircuit(2)
        qc.append(PauliEvolutionGate(Pauli("XX"), vec[0]), [0, 1])
        qc.append(PauliEvolutionGate(Pauli("YY"), vec[1]), [0, 1])
        qc.append(PauliEvolutionGate(Pauli("ZZ"), vec[2]), [0, 1])
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
input float[64] _t_0_;
input float[64] _t_1_;
input float[64] _t_2_;
gate rxx(p0) _gate_q_0, _gate_q_1 {
  h _gate_q_0;
  h _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rz(p0) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_1;
  h _gate_q_0;
}
gate PauliEvolution(_t_0_) _gate_q_0, _gate_q_1 {
  rxx(2.0*_t_0_) _gate_q_0, _gate_q_1;
}
gate ryy(p0) _gate_q_0, _gate_q_1 {
  rx(pi/2) _gate_q_0;
  rx(pi/2) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rz(p0) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rx(-pi/2) _gate_q_0;
  rx(-pi/2) _gate_q_1;
}
gate PauliEvolution_0(_t_1_) _gate_q_0, _gate_q_1 {
  ryy(2.0*_t_1_) _gate_q_0, _gate_q_1;
}
gate rzz(p0) _gate_q_0, _gate_q_1 {
  cx _gate_q_0, _gate_q_1;
  rz(p0) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
}
gate PauliEvolution_1(_t_2_) _gate_q_0, _gate_q_1 {
  rzz(2.0*_t_2_) _gate_q_0, _gate_q_1;
}
qubit[2] q;
PauliEvolution(_t_0_) q[0], q[1];
PauliEvolution_0(_t_1_) q[0], q[1];
PauliEvolution_1(_t_2_) q[0], q[1];
"""
        self.assertEqual(dumps(qc), expected)

    def test_reused_custom_parameter(self):
        """Test reused custom gate with parameter."""
        parameter_a = Parameter("a")

        custom = QuantumCircuit(1)
        custom.rx(parameter_a, 0)

        circuit = QuantumCircuit(1)
        circuit.append(custom.assign_parameters({parameter_a: 0.5}).to_gate(), [0])
        circuit.append(custom.assign_parameters({parameter_a: 1}).to_gate(), [0])

        circuit_name_0 = "_" + circuit.data[0].operation.definition.name.replace("-", "_")
        circuit_name_1 = "_" + circuit.data[1].operation.definition.name.replace("-", "_")

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                f"gate {circuit_name_0} _gate_q_0 {{",
                "  rx(0.5) _gate_q_0;",
                "}",
                f"gate {circuit_name_1} _gate_q_0 {{",
                "  rx(1.0) _gate_q_0;",
                "}",
                "qubit[1] q;",
                f"{circuit_name_0} q[0];",
                f"{circuit_name_1} q[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(circuit), expected_qasm)

    def test_unbound_circuit(self):
        """Test with unbound parameters (turning them into inputs)."""
        qc = QuantumCircuit(1)
        theta = Parameter("θ")
        qc.rz(theta, 0)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "input float[64] θ;",
                "qubit[1] q;",
                "rz(θ) q[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_standard_parameterized_gate_called_multiple_times(self):
        """Test that a parameterized gate is called correctly if the first instance of it is
        generic."""
        x, y = Parameter("x"), Parameter("y")
        qc = QuantumCircuit(2)
        qc.rzx(x, 0, 1)
        qc.rzx(y, 0, 1)
        qc.rzx(0.5, 0, 1)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                "input float[64] x;",
                "input float[64] y;",
                "gate rzx(p0) _gate_q_0, _gate_q_1 {",
                "  h _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "  rz(p0) _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "  h _gate_q_1;",
                "}",
                "qubit[2] q;",
                "rzx(x) q[0], q[1];",
                "rzx(y) q[0], q[1];",
                "rzx(0.5) q[0], q[1];",
                "",
            ]
        )

        # Set the includes and basis gates to ensure that this gate is unknown.
        exporter = Exporter(includes=[], basis_gates=("rz", "h", "cx"))
        self.assertEqual(exporter.dumps(qc), expected_qasm)

    def test_standard_parameterized_gate_called_multiple_times_first_instance_float(self):
        """Test that a parameterized gate is called correctly even if the first instance of it is
        not generic."""
        x, y = Parameter("x"), Parameter("y")
        qc = QuantumCircuit(2)
        qc.rzx(0.5, 0, 1)
        qc.rzx(x, 0, 1)
        qc.rzx(y, 0, 1)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                "input float[64] x;",
                "input float[64] y;",
                "gate rzx(p0) _gate_q_0, _gate_q_1 {",
                "  h _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "  rz(p0) _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "  h _gate_q_1;",
                "}",
                "qubit[2] q;",
                "rzx(0.5) q[0], q[1];",
                "rzx(x) q[0], q[1];",
                "rzx(y) q[0], q[1];",
                "",
            ]
        )

        # Set the includes and basis gates to ensure that this gate is unknown.
        exporter = Exporter(includes=[], basis_gates=("rz", "h", "cx"))
        self.assertEqual(exporter.dumps(qc), expected_qasm)

    def test_gate_qasm_with_ctrl_state(self):
        """Test with open controlled gate that has ctrl_state"""
        qc = QuantumCircuit(2)
        qc.ch(0, 1, ctrl_state=0)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "gate ch_o0 _gate_q_0, _gate_q_1 {",
                "  x _gate_q_0;",
                "  ch _gate_q_0, _gate_q_1;",
                "  x _gate_q_0;",
                "}",
                "qubit[2] q;",
                "ch_o0 q[0], q[1];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_custom_gate_collision_with_stdlib(self):
        """Test a custom gate with name collision with the standard library."""
        custom = QuantumCircuit(2, name="cx")
        custom.cx(0, 1)
        custom_gate = custom.to_gate()

        qc = QuantumCircuit(2)
        qc.append(custom_gate, [0, 1])
        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "gate cx_0 _gate_q_0, _gate_q_1 {",
                "  cx _gate_q_0, _gate_q_1;",
                "}",
                "qubit[2] q;",
                "cx_0 q[0], q[1];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_no_include(self):
        """Test explicit gate declaration (no include)"""
        q = QuantumRegister(2, "q")
        circuit = QuantumCircuit(q)
        circuit.rz(pi / 2, 0)
        circuit.sx(0)
        circuit.cx(0, 1)
        expected_qasm = """\
OPENQASM 3.0;
gate u3(p0, p1, p2) _gate_q_0 {
  U(p0, p1, p2) _gate_q_0;
}
gate u1(p0) _gate_q_0 {
  u3(0, 0, p0) _gate_q_0;
}
gate rz(p0) _gate_q_0 {
  u1(p0) _gate_q_0;
}
gate sdg _gate_q_0 {
  u1(-pi/2) _gate_q_0;
}
gate u2(p0, p1) _gate_q_0 {
  u3(pi/2, p0, p1) _gate_q_0;
}
gate h _gate_q_0 {
  u2(0, pi) _gate_q_0;
}
gate sx _gate_q_0 {
  sdg _gate_q_0;
  h _gate_q_0;
  sdg _gate_q_0;
}
gate cx c, t {
  ctrl @ U(pi, 0, pi) c, t;
}
qubit[2] q;
rz(pi/2) q[0];
sx q[0];
cx q[0], q[1];
"""
        self.assertEqual(Exporter(includes=[]).dumps(circuit), expected_qasm)

    def test_teleportation(self):
        """Teleportation with physical qubits"""
        qc = QuantumCircuit(3, 2)
        qc.h(1)
        qc.cx(1, 2)
        qc.barrier()
        qc.cx(0, 1)
        qc.h(0)
        qc.barrier()
        qc.measure([0, 1], [0, 1])
        qc.barrier()
        qc.x(2).c_if(qc.clbits[1], 1)
        qc.z(2).c_if(qc.clbits[0], 1)

        transpiled = transpile(qc, initial_layout=[0, 1, 2])
        expected_qasm = """\
OPENQASM 3.0;
gate u3(p0, p1, p2) _gate_q_0 {
  U(p0, p1, p2) _gate_q_0;
}
gate u2(p0, p1) _gate_q_0 {
  u3(pi/2, p0, p1) _gate_q_0;
}
gate h _gate_q_0 {
  u2(0, pi) _gate_q_0;
}
gate cx c, t {
  ctrl @ U(pi, 0, pi) c, t;
}
gate x _gate_q_0 {
  u3(pi, 0, pi) _gate_q_0;
}
gate u1(p0) _gate_q_0 {
  u3(0, 0, p0) _gate_q_0;
}
gate z _gate_q_0 {
  u1(pi) _gate_q_0;
}
bit[2] c;
h $1;
cx $1, $2;
barrier $0, $1, $2;
cx $0, $1;
h $0;
barrier $0, $1, $2;
c[0] = measure $0;
c[1] = measure $1;
barrier $0, $1, $2;
if (c[1]) {
  x $2;
}
if (c[0]) {
  z $2;
}
"""
        self.assertEqual(Exporter(includes=[]).dumps(transpiled), expected_qasm)

    def test_basis_gates(self):
        """Teleportation with physical qubits"""
        qc = QuantumCircuit(3, 2)
        qc.h(1)
        qc.cx(1, 2)
        qc.barrier()
        qc.cx(0, 1)
        qc.h(0)
        qc.barrier()
        qc.measure([0, 1], [0, 1])
        qc.barrier()
        qc.x(2).c_if(qc.clbits[1], 1)
        qc.z(2).c_if(qc.clbits[0], 1)

        transpiled = transpile(qc, initial_layout=[0, 1, 2])
        expected_qasm = """\
OPENQASM 3.0;
gate u3(p0, p1, p2) _gate_q_0 {
  U(p0, p1, p2) _gate_q_0;
}
gate u2(p0, p1) _gate_q_0 {
  u3(pi/2, p0, p1) _gate_q_0;
}
gate h _gate_q_0 {
  u2(0, pi) _gate_q_0;
}
gate x _gate_q_0 {
  u3(pi, 0, pi) _gate_q_0;
}
bit[2] c;
h $1;
cx $1, $2;
barrier $0, $1, $2;
cx $0, $1;
h $0;
barrier $0, $1, $2;
c[0] = measure $0;
c[1] = measure $1;
barrier $0, $1, $2;
if (c[1]) {
  x $2;
}
if (c[0]) {
  z $2;
}
"""
        self.assertEqual(
            Exporter(includes=[], basis_gates=["cx", "z", "U"]).dumps(transpiled),
            expected_qasm,
        )

    def test_opaque_instruction_in_basis_gates(self):
        """Test that an instruction that is set in the basis gates is output verbatim with no
        definition."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.append(Gate("my_gate", 1, []), [0], [])

        basis_gates = ["my_gate", "x"]
        transpiled = transpile(qc, initial_layout=[0])
        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                "x $0;",
                "my_gate $0;",
                "",
            ]
        )
        self.assertEqual(
            Exporter(includes=[], basis_gates=basis_gates).dumps(transpiled), expected_qasm
        )

    def test_reset_statement(self):
        """Test that a reset statement gets output into valid OpenQASM 3.  This includes tests of reset
        operations on single qubits and in nested scopes."""
        qreg = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qreg)
        qc.reset(0)
        qc.reset([0, 1])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                "qubit[2] qr;",
                "reset qr[0];",
                "reset qr[0];",
                "reset qr[1];",
                "",
            ]
        )
        self.assertEqual(Exporter(includes=[]).dumps(qc), expected_qasm)

    def test_delay_statement(self):
        """Test that delay operations get output into valid OpenQASM 3."""
        qreg = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qreg)
        qc.delay(100, qreg[0], unit="ms")
        qc.delay(2, qreg[1], unit="ps")  # "ps" is not a valid unit in OQ3, so we need to convert.

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                "qubit[2] qr;",
                "delay[100ms] qr[0];",
                "delay[2000ns] qr[1];",
                "",
            ]
        )
        self.assertEqual(Exporter(includes=[]).dumps(qc), expected_qasm)

    def test_loose_qubits(self):
        """Test that qubits that are not in any register can be used without issue."""
        bits = [Qubit(), Qubit()]
        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(bits, qr, cr)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.h(3)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit _qubit0;",
                "qubit _qubit1;",
                "qubit[2] qr;",
                "h _qubit0;",
                "h _qubit1;",
                "h qr[0];",
                "h qr[1];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_loose_clbits(self):
        """Test that clbits that are not in any register can be used without issue."""
        qreg = QuantumRegister(1, name="qr")
        bits = [Clbit() for _ in [None] * 7]
        cr1 = ClassicalRegister(name="cr1", bits=bits[1:3])
        cr2 = ClassicalRegister(name="cr2", bits=bits[4:6])
        qc = QuantumCircuit(bits, qreg, cr1, cr2)
        qc.measure(0, 0)
        qc.measure(0, 1)
        qc.measure(0, 2)
        qc.measure(0, 3)
        qc.measure(0, 4)
        qc.measure(0, 5)
        qc.measure(0, 6)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "bit _bit0;",
                "bit _bit3;",
                "bit _bit6;",
                "bit[2] cr1;",
                "bit[2] cr2;",
                "qubit[1] qr;",
                "_bit0 = measure qr[0];",
                "cr1[0] = measure qr[0];",
                "cr1[1] = measure qr[0];",
                "_bit3 = measure qr[0];",
                "cr2[0] = measure qr[0];",
                "cr2[1] = measure qr[0];",
                "_bit6 = measure qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_classical_register_aliasing(self):
        """Test that clbits that are not in any register can be used without issue."""
        qreg = QuantumRegister(1, name="qr")
        bits = [Clbit() for _ in [None] * 7]
        cr1 = ClassicalRegister(name="cr1", bits=bits[1:3])
        cr2 = ClassicalRegister(name="cr2", bits=bits[4:6])
        # cr3 overlaps cr2, but this should be allowed in this alias form.
        cr3 = ClassicalRegister(name="cr3", bits=bits[5:])
        qc = QuantumCircuit(bits, qreg, cr1, cr2, cr3)
        qc.measure(0, 0)
        qc.measure(0, 1)
        qc.measure(0, 2)
        qc.measure(0, 3)
        qc.measure(0, 4)
        qc.measure(0, 5)
        qc.measure(0, 6)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "bit _bit0;",
                "bit _bit1;",
                "bit _bit2;",
                "bit _bit3;",
                "bit _bit4;",
                "bit _bit5;",
                "bit _bit6;",
                "let cr1 = {_bit1, _bit2};",
                "let cr2 = {_bit4, _bit5};",
                "let cr3 = {cr2[1], _bit6};",
                "qubit[1] qr;",
                "_bit0 = measure qr[0];",
                "cr1[0] = measure qr[0];",
                "cr1[1] = measure qr[0];",
                "_bit3 = measure qr[0];",
                "cr2[0] = measure qr[0];",
                "cr3[0] = measure qr[0];",
                "cr3[1] = measure qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc, allow_aliasing=True), expected_qasm)

    def test_old_alias_classical_registers_option(self):
        """Test that the ``alias_classical_registers`` option still functions during its changeover
        period."""
        qreg = QuantumRegister(1, name="qr")
        bits = [Clbit() for _ in [None] * 7]
        cr1 = ClassicalRegister(name="cr1", bits=bits[1:3])
        cr2 = ClassicalRegister(name="cr2", bits=bits[4:6])
        # cr3 overlaps cr2, but this should be allowed in this alias form.
        cr3 = ClassicalRegister(name="cr3", bits=bits[5:])
        qc = QuantumCircuit(bits, qreg, cr1, cr2, cr3)
        qc.measure(0, 0)
        qc.measure(0, 1)
        qc.measure(0, 2)
        qc.measure(0, 3)
        qc.measure(0, 4)
        qc.measure(0, 5)
        qc.measure(0, 6)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "bit _bit0;",
                "bit _bit1;",
                "bit _bit2;",
                "bit _bit3;",
                "bit _bit4;",
                "bit _bit5;",
                "bit _bit6;",
                "let cr1 = {_bit1, _bit2};",
                "let cr2 = {_bit4, _bit5};",
                "let cr3 = {cr2[1], _bit6};",
                "qubit[1] qr;",
                "_bit0 = measure qr[0];",
                "cr1[0] = measure qr[0];",
                "cr1[1] = measure qr[0];",
                "_bit3 = measure qr[0];",
                "cr2[0] = measure qr[0];",
                "cr3[0] = measure qr[0];",
                "cr3[1] = measure qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc, alias_classical_registers=True), expected_qasm)

    def test_simple_for_loop(self):
        """Test that a simple for loop outputs the expected result."""
        parameter = Parameter("my_x")
        loop_body = QuantumCircuit(1)
        loop_body.rx(parameter, 0)
        loop_body.break_loop()
        loop_body.continue_loop()

        qc = QuantumCircuit(2)
        qc.for_loop([0, 3, 4], parameter, loop_body, [1], [])
        qc.x(0)

        qr_name = qc.qregs[0].name

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                f"qubit[2] {qr_name};",
                f"for {parameter.name} in {{0, 3, 4}} {{",
                f"  rx({parameter.name}) {qr_name}[1];",
                "  break;",
                "  continue;",
                "}",
                f"x {qr_name}[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_nested_for_loop(self):
        """Test that a for loop nested inside another outputs the expected result."""
        inner_parameter = Parameter("my_x")
        outer_parameter = Parameter("my_y")

        inner_body = QuantumCircuit(2)
        inner_body.rz(inner_parameter, 0)
        inner_body.rz(outer_parameter, 1)
        inner_body.break_loop()

        outer_body = QuantumCircuit(2)
        outer_body.h(0)
        outer_body.rz(outer_parameter, 1)
        # Note we reverse the order of the bits here to test that this is traced.
        outer_body.for_loop(range(1, 5, 2), inner_parameter, inner_body, [1, 0], [])
        outer_body.continue_loop()

        qc = QuantumCircuit(2)
        qc.for_loop(range(4), outer_parameter, outer_body, [0, 1], [])
        qc.x(0)

        qr_name = qc.qregs[0].name

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                f"qubit[2] {qr_name};",
                f"for {outer_parameter.name} in [0:3] {{",
                f"  h {qr_name}[0];",
                f"  rz({outer_parameter.name}) {qr_name}[1];",
                f"  for {inner_parameter.name} in [1:2:4] {{",
                # Note the reversed bit order.
                f"    rz({inner_parameter.name}) {qr_name}[1];",
                f"    rz({outer_parameter.name}) {qr_name}[0];",
                "    break;",
                "  }",
                "  continue;",
                "}",
                f"x {qr_name}[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_regular_parameter_in_nested_for_loop(self):
        """Test that a for loop nested inside another outputs the expected result, including
        defining parameters that are used in nested loop scopes."""
        inner_parameter = Parameter("my_x")
        outer_parameter = Parameter("my_y")
        regular_parameter = Parameter("my_t")

        inner_body = QuantumCircuit(2)
        inner_body.h(0)
        inner_body.rx(regular_parameter, 1)
        inner_body.break_loop()

        outer_body = QuantumCircuit(2)
        outer_body.h(0)
        outer_body.h(1)
        # Note we reverse the order of the bits here to test that this is traced.
        outer_body.for_loop(range(1, 5, 2), inner_parameter, inner_body, [1, 0], [])
        outer_body.continue_loop()

        qc = QuantumCircuit(2)
        qc.for_loop(range(4), outer_parameter, outer_body, [0, 1], [])
        qc.x(0)

        qr_name = qc.qregs[0].name

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                # This next line will be missing until gh-7280 is fixed.
                f"input float[64] {regular_parameter.name};",
                f"qubit[2] {qr_name};",
                f"for {outer_parameter.name} in [0:3] {{",
                f"  h {qr_name}[0];",
                f"  h {qr_name}[1];",
                f"  for {inner_parameter.name} in [1:2:4] {{",
                # Note the reversed bit order.
                f"    h {qr_name}[1];",
                f"    rx({regular_parameter.name}) {qr_name}[0];",
                "    break;",
                "  }",
                "  continue;",
                "}",
                f"x {qr_name}[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_for_loop_with_no_parameter(self):
        """Test that a for loop with the parameter set to ``None`` outputs the expected result."""
        loop_body = QuantumCircuit(1)
        loop_body.h(0)

        qc = QuantumCircuit(2)
        qc.for_loop([0, 3, 4], None, loop_body, [1], [])
        qr_name = qc.qregs[0].name

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                f"qubit[2] {qr_name};",
                "for _ in {0, 3, 4} {",
                f"  h {qr_name}[1];",
                "}",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_simple_while_loop(self):
        """Test that a simple while loop works correctly."""
        loop_body = QuantumCircuit(1)
        loop_body.h(0)
        loop_body.break_loop()
        loop_body.continue_loop()

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.while_loop((cr, 0), loop_body, [1], [])
        qc.x(0)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] qr;",
                "while (cr == 0) {",
                "  h qr[1];",
                "  break;",
                "  continue;",
                "}",
                "x qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_nested_while_loop(self):
        """Test that a while loop nested inside another outputs the expected result."""
        inner_body = QuantumCircuit(2, 2)
        inner_body.measure(0, 0)
        inner_body.measure(1, 1)
        inner_body.break_loop()

        outer_body = QuantumCircuit(2, 2)
        outer_body.measure(0, 0)
        outer_body.measure(1, 1)
        # We reverse the order of the bits here to test this works, and test a single-bit condition.
        outer_body.while_loop((outer_body.clbits[0], 0), inner_body, [1, 0], [1, 0])
        outer_body.continue_loop()

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.while_loop((cr, 0), outer_body, [0, 1], [0, 1])
        qc.x(0)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] qr;",
                "while (cr == 0) {",
                "  cr[0] = measure qr[0];",
                "  cr[1] = measure qr[1];",
                # Note the reversed bits in the body.
                "  while (!cr[0]) {",
                "    cr[1] = measure qr[1];",
                "    cr[0] = measure qr[0];",
                "    break;",
                "  }",
                "  continue;",
                "}",
                "x qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_simple_if_statement(self):
        """Test that a simple if statement with no else works correctly."""
        true_body = QuantumCircuit(1)
        true_body.h(0)

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.if_test((cr, 0), true_body, [1], [])
        qc.x(0)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] qr;",
                "if (cr == 0) {",
                "  h qr[1];",
                "}",
                "x qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_simple_if_else_statement(self):
        """Test that a simple if statement with an else branch works correctly."""
        true_body = QuantumCircuit(1)
        true_body.h(0)
        false_body = QuantumCircuit(1)
        false_body.z(0)

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.if_else((cr, 0), true_body, false_body, [1], [])
        qc.x(0)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] qr;",
                "if (cr == 0) {",
                "  h qr[1];",
                "} else {",
                "  z qr[1];",
                "}",
                "x qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_nested_if_else_statement(self):
        """Test that a nested if/else statement works correctly."""
        inner_true_body = QuantumCircuit(2, 2)
        inner_true_body.measure(0, 0)
        inner_false_body = QuantumCircuit(2, 2)
        inner_false_body.measure(1, 1)

        outer_true_body = QuantumCircuit(2, 2)
        outer_true_body.if_else(
            (outer_true_body.clbits[0], 0), inner_true_body, inner_false_body, [0, 1], [0, 1]
        )
        outer_false_body = QuantumCircuit(2, 2)
        # Note the flipped bits here.
        outer_false_body.if_else(
            (outer_false_body.clbits[0], 1), inner_true_body, inner_false_body, [1, 0], [1, 0]
        )

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.if_else((cr, 0), outer_true_body, outer_false_body, [0, 1], [0, 1])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] qr;",
                "if (cr == 0) {",
                "  if (!cr[0]) {",
                "    cr[0] = measure qr[0];",
                "  } else {",
                "    cr[1] = measure qr[1];",
                "  }",
                "} else {",
                "  if (cr[0]) {",
                "    cr[1] = measure qr[1];",
                "  } else {",
                "    cr[0] = measure qr[0];",
                "  }",
                "}",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_chain_else_if(self):
        """Test the basic 'else/if' chaining logic for flattening the else scope if its content is a
        single if/else statement."""
        inner_true_body = QuantumCircuit(2, 2)
        inner_true_body.measure(0, 0)
        inner_false_body = QuantumCircuit(2, 2)
        inner_false_body.measure(1, 1)

        outer_true_body = QuantumCircuit(2, 2)
        outer_true_body.if_else(
            (outer_true_body.clbits[0], 0), inner_true_body, inner_false_body, [0, 1], [0, 1]
        )
        outer_false_body = QuantumCircuit(2, 2)
        # Note the flipped bits here.
        outer_false_body.if_else(
            (outer_false_body.clbits[0], 1), inner_true_body, inner_false_body, [1, 0], [1, 0]
        )

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.if_else((cr, 0), outer_true_body, outer_false_body, [0, 1], [0, 1])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] qr;",
                "if (cr == 0) {",
                "  if (!cr[0]) {",
                "    cr[0] = measure qr[0];",
                "  } else {",
                "    cr[1] = measure qr[1];",
                "  }",
                "} else if (cr[0]) {",
                "  cr[1] = measure qr[1];",
                "} else {",
                "  cr[0] = measure qr[0];",
                "}",
                "",
            ]
        )
        # This is not the default behavior, and it's pretty buried how you'd access it.
        builder = QASM3Builder(
            qc,
            includeslist=("stdgates.inc",),
            basis_gates=("U",),
            disable_constants=False,
            allow_aliasing=False,
        )
        stream = StringIO()
        BasicPrinter(stream, indent="  ", chain_else_if=True).visit(builder.build_program())
        self.assertEqual(stream.getvalue(), expected_qasm)

    def test_chain_else_if_does_not_chain_if_extra_instructions(self):
        """Test the basic 'else/if' chaining logic for flattening the else scope if its content is a
        single if/else statement does not cause a flattening if the 'else' block is not a single
        if/else."""
        inner_true_body = QuantumCircuit(2, 2)
        inner_true_body.measure(0, 0)
        inner_false_body = QuantumCircuit(2, 2)
        inner_false_body.measure(1, 1)

        outer_true_body = QuantumCircuit(2, 2)
        outer_true_body.if_else(
            (outer_true_body.clbits[0], 0), inner_true_body, inner_false_body, [0, 1], [0, 1]
        )
        outer_false_body = QuantumCircuit(2, 2)
        # Note the flipped bits here.
        outer_false_body.if_else(
            (outer_false_body.clbits[0], 1), inner_true_body, inner_false_body, [1, 0], [1, 0]
        )
        outer_false_body.h(0)

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.if_else((cr, 0), outer_true_body, outer_false_body, [0, 1], [0, 1])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] qr;",
                "if (cr == 0) {",
                "  if (!cr[0]) {",
                "    cr[0] = measure qr[0];",
                "  } else {",
                "    cr[1] = measure qr[1];",
                "  }",
                "} else {",
                "  if (cr[0]) {",
                "    cr[1] = measure qr[1];",
                "  } else {",
                "    cr[0] = measure qr[0];",
                "  }",
                "  h qr[0];",
                "}",
                "",
            ]
        )
        # This is not the default behavior, and it's pretty buried how you'd access it.
        builder = QASM3Builder(
            qc,
            includeslist=("stdgates.inc",),
            basis_gates=("U",),
            disable_constants=False,
            allow_aliasing=False,
        )
        stream = StringIO()
        BasicPrinter(stream, indent="  ", chain_else_if=True).visit(builder.build_program())
        self.assertEqual(stream.getvalue(), expected_qasm)

    def test_custom_gate_used_in_loop_scope(self):
        """Test that a custom gate only used within a loop scope still gets a definition at the top
        level."""
        parameter_a = Parameter("a")
        parameter_b = Parameter("b")

        custom = QuantumCircuit(1)
        custom.rx(parameter_a, 0)
        custom_gate = custom.assign_parameters({parameter_a: 0.5}).to_gate()
        custom_gate.name = "custom"

        loop_body = QuantumCircuit(1)
        loop_body.append(custom_gate, [0])

        qc = QuantumCircuit(1)
        qc.for_loop(range(2), parameter_b, loop_body, [0], [])
        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "gate custom _gate_q_0 {",
                "  rx(0.5) _gate_q_0;",
                "}",
                "qubit[1] q;",
                "for b in [0:1] {",
                "  custom q[0];",
                "}",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_custom_gate_with_hw_qubit_name(self):
        """Test that the name of a custom gate that is an OQ3 hardware qubit identifer is properly
        escaped when translated to OQ3."""
        mygate_circ = QuantumCircuit(1, name="$1")
        mygate_circ.x(0)
        mygate = mygate_circ.to_gate()
        qc = QuantumCircuit(1)
        qc.append(mygate, [0])
        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "gate __1 _gate_q_0 {",
                "  x _gate_q_0;",
                "}",
                "qubit[1] q;",
                "__1 q[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_registers_have_escaped_names(self):
        """Test that both types of register are emitted with safely escaped names if they begin with
        invalid names. Regression test of gh-9658."""
        qc = QuantumCircuit(
            QuantumRegister(2, name="q_{reg}"), ClassicalRegister(2, name="c_{reg}")
        )
        qc.measure([0, 1], [0, 1])
        out_qasm = dumps(qc)
        matches = {match_["name"] for match_ in self.register_regex.finditer(out_qasm)}
        self.assertEqual(len(matches), 2, msg=f"Observed OQ3 output:\n{out_qasm}")

    def test_parameters_have_escaped_names(self):
        """Test that parameters are emitted with safely escaped names if they begin with invalid
        names. Regression test of gh-9658."""
        qc = QuantumCircuit(1)
        qc.u(Parameter("p_{0}"), 2 * Parameter("p_?0!"), 0, 0)
        out_qasm = dumps(qc)
        matches = {match_["name"] for match_ in self.scalar_parameter_regex.finditer(out_qasm)}
        self.assertEqual(len(matches), 2, msg=f"Observed OQ3 output:\n{out_qasm}")

    def test_parameter_expression_after_naming_escape(self):
        """Test that :class:`.Parameter` instances are correctly renamed when they are used with
        :class:`.ParameterExpression` blocks, even if they have names that needed to be escaped."""
        param = Parameter("measure")  # an invalid name
        qc = QuantumCircuit(1)
        qc.u(2 * param, 0, 0, 0)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3.0;",
                'include "stdgates.inc";',
                "input float[64] measure_0;",
                "qubit[1] q;",
                "U(2*measure_0, 0, 0) q[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_parameters_and_registers_cannot_have_naming_clashes(self):
        """Test that parameters and registers are considered part of the same symbol table for the
        purposes of avoiding clashes."""
        qreg = QuantumRegister(1, "clash")
        param = Parameter("clash")
        qc = QuantumCircuit(qreg)
        qc.u(param, 0, 0, 0)

        out_qasm = dumps(qc)
        register_name = self.register_regex.search(out_qasm)
        parameter_name = self.scalar_parameter_regex.search(out_qasm)
        self.assertTrue(register_name)
        self.assertTrue(parameter_name)
        self.assertIn("clash", register_name["name"])
        self.assertIn("clash", parameter_name["name"])
        self.assertNotEqual(register_name["name"], parameter_name["name"])

    def test_parameters_and_gates_cannot_have_naming_clashes(self):
        """Test that parameters are renamed to avoid collisions with gate names."""
        qc = QuantumCircuit(QuantumRegister(1, "q"))
        qc.rz(Parameter("rz"), 0)

        out_qasm = dumps(qc)
        parameter_name = self.scalar_parameter_regex.search(out_qasm)
        self.assertTrue(parameter_name)
        self.assertIn("rz", parameter_name["name"])
        self.assertNotEqual(parameter_name["name"], "rz")

    # Not necessarily all the reserved keywords, just a sensibly-sized subset.
    @data("bit", "const", "def", "defcal", "float", "gate", "include", "int", "let", "measure")
    def test_reserved_keywords_as_names_are_escaped(self, keyword):
        """Test that reserved keywords used to name registers and parameters are escaped into
        another form when output, and the escaping cannot introduce new conflicts."""
        with self.subTest("register"):
            qreg = QuantumRegister(1, keyword)
            qc = QuantumCircuit(qreg)
            out_qasm = dumps(qc)
            register_name = self.register_regex.search(out_qasm)
            self.assertTrue(register_name, msg=f"Observed OQ3:\n{out_qasm}")
            self.assertNotEqual(keyword, register_name["name"])
        with self.subTest("parameter"):
            qc = QuantumCircuit(1)
            param = Parameter(keyword)
            qc.u(param, 0, 0, 0)
            out_qasm = dumps(qc)
            parameter_name = self.scalar_parameter_regex.search(out_qasm)
            self.assertTrue(parameter_name, msg=f"Observed OQ3:\n{out_qasm}")
            self.assertNotEqual(keyword, parameter_name["name"])

    def test_expr_condition(self):
        """Simple test that the conditions of `if`s and `while`s can be `Expr` nodes."""
        bits = [Qubit(), Clbit()]
        cr = ClassicalRegister(2, "cr")

        if_body = QuantumCircuit(1)
        if_body.x(0)

        while_body = QuantumCircuit(1)
        while_body.x(0)

        qc = QuantumCircuit(bits, cr)
        qc.if_test(expr.logic_not(qc.clbits[0]), if_body, [0], [])
        qc.while_loop(expr.equal(cr, 3), while_body, [0], [])

        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit _bit0;
bit[2] cr;
qubit _qubit0;
if (!_bit0) {
  x _qubit0;
}
while (cr == 3) {
  x _qubit0;
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_expr_nested_condition(self):
        """Simple test that the conditions of `if`s and `while`s can be `Expr` nodes when nested,
        and the mapping of inner bits to outer bits is correct."""
        bits = [Qubit(), Clbit(), Clbit()]
        cr = ClassicalRegister(2, "cr")

        inner_if_body = QuantumCircuit(1)
        inner_if_body.x(0)
        outer_if_body = QuantumCircuit(1, 1)
        outer_if_body.if_test(expr.lift(outer_if_body.clbits[0]), inner_if_body, [0], [])

        inner_while_body = QuantumCircuit(1)
        inner_while_body.x(0)
        outer_while_body = QuantumCircuit([Qubit()], cr)
        outer_while_body.while_loop(expr.equal(expr.bit_and(cr, 3), 3), inner_while_body, [0], [])

        qc = QuantumCircuit(bits, cr)
        qc.if_test(expr.logic_not(qc.clbits[0]), outer_if_body, [0], [1])
        qc.while_loop(expr.equal(cr, 3), outer_while_body, [0], cr)

        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit _bit0;
bit _bit1;
bit[2] cr;
qubit _qubit0;
if (!_bit0) {
  if (_bit1) {
    x _qubit0;
  }
}
while (cr == 3) {
  while ((cr & 3) == 3) {
    x _qubit0;
  }
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_expr_associativity_left(self):
        """Test that operations that are in the expression tree in a left-associative form are
        output to OQ3 correctly."""
        body = QuantumCircuit()

        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")
        cr3 = ClassicalRegister(3, "cr3")
        qc = QuantumCircuit(cr1, cr2, cr3)
        qc.if_test(expr.equal(expr.bit_and(expr.bit_and(cr1, cr2), cr3), 7), body.copy(), [], [])
        qc.if_test(expr.equal(expr.bit_or(expr.bit_or(cr1, cr2), cr3), 7), body.copy(), [], [])
        qc.if_test(expr.equal(expr.bit_xor(expr.bit_xor(cr1, cr2), cr3), 7), body.copy(), [], [])
        qc.if_test(
            expr.equal(expr.shift_left(expr.shift_left(cr1, cr2), cr3), 7), body.copy(), [], []
        )
        qc.if_test(
            expr.equal(expr.shift_right(expr.shift_right(cr1, cr2), cr3), 7), body.copy(), [], []
        )
        qc.if_test(
            expr.equal(expr.shift_left(expr.shift_right(cr1, cr2), cr3), 7), body.copy(), [], []
        )
        qc.if_test(expr.logic_and(expr.logic_and(cr1[0], cr1[1]), cr1[2]), body.copy(), [], [])
        qc.if_test(expr.logic_or(expr.logic_or(cr1[0], cr1[1]), cr1[2]), body.copy(), [], [])

        # Note that bitwise operations except shift have lower priority than `==` so there's extra
        # parentheses.  All these operators are left-associative in OQ3.
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[3] cr1;
bit[3] cr2;
bit[3] cr3;
if ((cr1 & cr2 & cr3) == 7) {
}
if ((cr1 | cr2 | cr3) == 7) {
}
if ((cr1 ^ cr2 ^ cr3) == 7) {
}
if (cr1 << cr2 << cr3 == 7) {
}
if (cr1 >> cr2 >> cr3 == 7) {
}
if (cr1 >> cr2 << cr3 == 7) {
}
if (cr1[0] && cr1[1] && cr1[2]) {
}
if (cr1[0] || cr1[1] || cr1[2]) {
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_expr_associativity_right(self):
        """Test that operations that are in the expression tree in a right-associative form are
        output to OQ3 correctly."""
        body = QuantumCircuit()

        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")
        cr3 = ClassicalRegister(3, "cr3")
        qc = QuantumCircuit(cr1, cr2, cr3)
        qc.if_test(expr.equal(expr.bit_and(cr1, expr.bit_and(cr2, cr3)), 7), body.copy(), [], [])
        qc.if_test(expr.equal(expr.bit_or(cr1, expr.bit_or(cr2, cr3)), 7), body.copy(), [], [])
        qc.if_test(expr.equal(expr.bit_xor(cr1, expr.bit_xor(cr2, cr3)), 7), body.copy(), [], [])
        qc.if_test(
            expr.equal(expr.shift_left(cr1, expr.shift_left(cr2, cr3)), 7), body.copy(), [], []
        )
        qc.if_test(
            expr.equal(expr.shift_right(cr1, expr.shift_right(cr2, cr3)), 7), body.copy(), [], []
        )
        qc.if_test(
            expr.equal(expr.shift_left(cr1, expr.shift_right(cr2, cr3)), 7), body.copy(), [], []
        )
        qc.if_test(expr.logic_and(cr1[0], expr.logic_and(cr1[1], cr1[2])), body.copy(), [], [])
        qc.if_test(expr.logic_or(cr1[0], expr.logic_or(cr1[1], cr1[2])), body.copy(), [], [])

        # Note that bitwise operations have lower priority than `==` so there's extra parentheses.
        # All these operators are left-associative in OQ3, so we need parentheses for them to be
        # parsed correctly.  Mathematically, they're all actually associative in general, so the
        # order doesn't _technically_ matter.
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[3] cr1;
bit[3] cr2;
bit[3] cr3;
if ((cr1 & (cr2 & cr3)) == 7) {
}
if ((cr1 | (cr2 | cr3)) == 7) {
}
if ((cr1 ^ (cr2 ^ cr3)) == 7) {
}
if (cr1 << (cr2 << cr3) == 7) {
}
if (cr1 >> (cr2 >> cr3) == 7) {
}
if (cr1 << (cr2 >> cr3) == 7) {
}
if (cr1[0] && (cr1[1] && cr1[2])) {
}
if (cr1[0] || (cr1[1] || cr1[2])) {
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_expr_binding_unary(self):
        """Test that nested unary operators don't insert unnecessary brackets."""
        body = QuantumCircuit()
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(cr)
        qc.if_test(expr.equal(expr.bit_not(expr.bit_not(cr)), 3), body.copy(), [], [])
        qc.if_test(expr.logic_not(expr.logic_not(cr[0])), body.copy(), [], [])

        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[2] cr;
if (~~cr == 3) {
}
if (!!cr[0]) {
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_expr_precedence(self):
        """Test that the precedence properties of operators are correctly output."""
        body = QuantumCircuit()
        cr = ClassicalRegister(2, "cr")
        # This tree is _completely_ inside out, so there's brackets needed round every operand.
        inside_out = expr.logic_not(
            expr.less(
                expr.bit_and(
                    expr.bit_xor(expr.bit_or(cr, cr), expr.bit_or(cr, cr)),
                    expr.bit_xor(expr.bit_or(cr, cr), expr.bit_or(cr, cr)),
                ),
                expr.bit_and(
                    expr.bit_xor(expr.bit_or(cr, cr), expr.bit_or(cr, cr)),
                    expr.bit_xor(expr.bit_or(cr, cr), expr.bit_or(cr, cr)),
                ),
            )
        )
        # This one is the other way round - the tightest-binding operations are on the inside, so no
        # brackets should be needed at all except to put in a comparison to a bitwise binary
        # operation, since those bind less tightly than anything that can cast them to a bool.
        outside_in = expr.logic_or(
            expr.logic_and(
                expr.equal(expr.bit_or(cr, cr), expr.bit_and(cr, cr)),
                expr.equal(expr.bit_and(cr, cr), expr.bit_or(cr, cr)),
            ),
            expr.logic_and(
                expr.greater(expr.bit_or(cr, cr), expr.bit_xor(cr, cr)),
                expr.less_equal(expr.bit_xor(cr, cr), expr.bit_or(cr, cr)),
            ),
        )

        # And an extra test of the logical operator order.
        logics = expr.logic_or(
            expr.logic_and(
                expr.logic_or(expr.logic_not(cr[0]), expr.logic_not(cr[0])),
                expr.logic_not(expr.logic_and(cr[0], cr[0])),
            ),
            expr.logic_and(
                expr.logic_not(expr.logic_and(cr[0], cr[0])),
                expr.logic_or(expr.logic_not(cr[0]), expr.logic_not(cr[0])),
            ),
        )

        # An extra test of the bitshifting rules, since we have to pick one or the other of
        # bitshifts vs comparisons due to the typing.  The first operand is inside out, the second
        bitshifts = expr.equal(
            expr.shift_left(expr.bit_and(expr.bit_xor(cr, cr), cr), expr.bit_or(cr, cr)),
            expr.bit_or(
                expr.bit_xor(expr.shift_right(cr, 3), expr.shift_left(cr, 4)),
                expr.shift_left(cr, 1),
            ),
        )

        qc = QuantumCircuit(cr)
        qc.if_test(inside_out, body.copy(), [], [])
        qc.if_test(outside_in, body.copy(), [], [])
        qc.if_test(logics, body.copy(), [], [])
        qc.if_test(bitshifts, body.copy(), [], [])

        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[2] cr;
if (!((((cr | cr) ^ (cr | cr)) & ((cr | cr) ^ (cr | cr)))\
 < (((cr | cr) ^ (cr | cr)) & ((cr | cr) ^ (cr | cr))))) {
}
if ((cr | cr) == (cr & cr) && (cr & cr) == (cr | cr)\
 || (cr | cr) > (cr ^ cr) && (cr ^ cr) <= (cr | cr)) {
}
if ((!cr[0] || !cr[0]) && !(cr[0] && cr[0]) || !(cr[0] && cr[0]) && (!cr[0] || !cr[0])) {
}
if (((cr ^ cr) & cr) << (cr | cr) == (cr >> 3 ^ cr << 4 | cr << 1)) {
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_no_unnecessary_cast(self):
        """This is a bit of a cross `Expr`-constructor / OQ3-exporter test.  It doesn't really
        matter whether or not the `Expr` constructor functions insert cast nodes into their output
        for the literals (at the time of writing [commit 2616602], they don't because they do some
        type inference) but the OQ3 export definitely shouldn't have them."""
        cr = ClassicalRegister(8, "cr")
        qc = QuantumCircuit(cr)
        # Note that the integer '1' has a minimum bit-width of 1, whereas the register has a width
        # of 8.  We're testing to make sure that there's no spurious cast up from `bit[1]` to
        # `bit[8]`, or anything like that, _whether or not_ the `Expr` node includes one.
        qc.if_test(expr.equal(cr, 1), QuantumCircuit(), [], [])

        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[8] cr;
if (cr == 1) {
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_var_use(self):
        """Test that input and declared vars work in simple local scopes and can be set."""
        qc = QuantumCircuit()
        a = qc.add_input("a", types.Bool())
        b = qc.add_input("b", types.Uint(8))
        qc.store(a, expr.logic_not(a))
        qc.store(b, expr.bit_and(b, 8))
        qc.add_var("c", expr.bit_not(b))
        # All inputs should come first, regardless of declaration order.
        qc.add_input("d", types.Bool())

        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
input bool a;
input uint[8] b;
input bool d;
uint[8] c;
a = !a;
b = b & 8;
c = ~b;
"""
        self.assertEqual(dumps(qc), expected)

    def test_var_use_in_scopes(self):
        """Test that usage of `Var` nodes works in capturing scopes."""
        qc = QuantumCircuit(2, 2)
        a = qc.add_input("a", types.Bool())
        b_outer = qc.add_var("b", expr.lift(5, types.Uint(16)))
        with qc.if_test(expr.logic_not(a)) as else_:
            qc.store(b_outer, expr.bit_not(b_outer))
            qc.h(0)
        with else_:
            # Shadow of the same type.
            qc.add_var("b", expr.lift(7, b_outer.type))
        with qc.while_loop(a):
            # Shadow of a different type.
            qc.add_var("b", a)
        with qc.switch(b_outer) as case:
            with case(0):
                qc.store(b_outer, expr.lift(3, b_outer.type))
            with case(case.DEFAULT):
                qc.add_var("b", expr.logic_not(a))
                qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
input bool a;
bit[2] c;
int switch_dummy;
qubit[2] q;
uint[16] b;
b = 5;
if (!a) {
  b = ~b;
  h q[0];
} else {
  uint[16] b;
  b = 7;
}
while (a) {
  bool b;
  b = a;
}
switch_dummy = b;
switch (switch_dummy) {
  case 0 {
    b = 3;
  }
  default {
    bool b;
    b = !a;
    cx q[0], q[1];
  }
}
c[0] = measure q[0];
c[1] = measure q[1];
"""
        self.assertEqual(dumps(qc), expected)

    def test_var_naming_clash_parameter(self):
        """We should support a `Var` clashing in name with a `Parameter` if `QuantumCircuit` allows
        it."""
        qc = QuantumCircuit(1)
        qc.add_var("a", False)
        qc.rx(Parameter("a"), 0)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
input float[64] a;
qubit[1] q;
bool a_0;
a_0 = false;
rx(a) q[0];
"""
        self.assertEqual(dumps(qc), expected)

    def test_var_naming_clash_register(self):
        """We should support a `Var` clashing in name with a `Register` if `QuantumCircuit` allows
        it."""
        qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "c"))
        qc.add_input("c", types.Bool())
        qc.add_var("q", False)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
input bool c_0;
bit[2] c;
qubit[2] q;
bool q_1;
q_1 = false;
"""
        self.assertEqual(dumps(qc), expected)

    def test_var_naming_clash_gate(self):
        """We should support a `Var` clashing in name with some gate if `QuantumCircuit` allows
        it."""
        qc = QuantumCircuit(2)
        qc.add_input("cx", types.Bool())
        qc.add_input("U", types.Bool())
        qc.add_var("rx", expr.lift(5, types.Uint(8)))

        qc.cx(0, 1)
        qc.u(0.5, 0.125, 0.25, 0)
        # We don't actually use `rx`, but it's still in the `stdgates` include.
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
input bool cx_0;
input bool U_1;
qubit[2] q;
uint[8] rx_2;
rx_2 = 5;
cx q[0], q[1];
U(0.5, 0.125, 0.25) q[0];
"""
        self.assertEqual(dumps(qc), expected)

    def test_unusual_conditions(self):
        """Test that special QASM constructs such as ``measure`` are correctly handled when the
        Terra instructions have old-style conditions."""
        qc = QuantumCircuit(3, 2)
        qc.h(0)
        qc.measure(0, 0)
        qc.measure(1, 1).c_if(0, True)
        qc.reset([0, 1]).c_if(0, True)
        with qc.while_loop((qc.clbits[0], True)):
            qc.break_loop().c_if(0, True)
            qc.continue_loop().c_if(0, True)
        # Terra forbids delay and barrier from being conditioned through `c_if`, but in theory they
        # should work fine in a dynamic-circuits sense (although what a conditional barrier _means_
        # is a whole other kettle of fish).
        delay = Delay(16, "dt")
        delay.condition = (qc.clbits[0], True)
        qc.append(delay, [0], [])
        barrier = Barrier(2)
        barrier.condition = (qc.clbits[0], True)
        qc.append(barrier, [0, 1], [])

        expected = """
OPENQASM 3.0;
include "stdgates.inc";
bit[2] c;
qubit[3] q;
h q[0];
c[0] = measure q[0];
if (c[0]) {
  c[1] = measure q[1];
}
if (c[0]) {
  reset q[0];
}
if (c[0]) {
  reset q[1];
}
while (c[0]) {
  if (c[0]) {
    break;
  }
  if (c[0]) {
    continue;
  }
}
if (c[0]) {
  delay[16dt] q[0];
}
if (c[0]) {
  barrier q[0], q[1];
}"""
        self.assertEqual(dumps(qc).strip(), expected.strip())

    def test_switch_clbit(self):
        """Test that a switch statement can be constructed with a bit as a condition."""
        qubit = Qubit()
        clbit = Clbit()
        case1 = QuantumCircuit([qubit, clbit])
        case1.x(0)
        case2 = QuantumCircuit([qubit, clbit])
        case2.z(0)
        circuit = QuantumCircuit([qubit, clbit])
        circuit.switch(clbit, [(True, case1), (False, case2)], [0], [0])

        test = dumps(circuit)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit _bit0;
int switch_dummy;
qubit _qubit0;
switch_dummy = _bit0;
switch (switch_dummy) {
  case 1 {
    x _qubit0;
  }
  case 0 {
    z _qubit0;
  }
}
"""
        self.assertEqual(test, expected)

    def test_switch_register(self):
        """Test that a switch statement can be constructed with a register as a
        condition."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "c")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)
        case3 = QuantumCircuit([qubit], creg)
        case3.z(0)

        circuit = QuantumCircuit([qubit], creg)
        circuit.switch(creg, [(0, case1), (1, case2), (2, case3)], [0], circuit.clbits)

        test = dumps(circuit)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[2] c;
int switch_dummy;
qubit _qubit0;
switch_dummy = c;
switch (switch_dummy) {
  case 0 {
    x _qubit0;
  }
  case 1 {
    y _qubit0;
  }
  case 2 {
    z _qubit0;
  }
}
"""
        self.assertEqual(test, expected)

    def test_switch_with_default(self):
        """Test that a switch statement can be constructed with a default case at the
        end."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "c")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)
        case3 = QuantumCircuit([qubit], creg)
        case3.z(0)

        circuit = QuantumCircuit([qubit], creg)
        circuit.switch(creg, [(0, case1), (1, case2), (CASE_DEFAULT, case3)], [0], circuit.clbits)

        test = dumps(circuit)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[2] c;
int switch_dummy;
qubit _qubit0;
switch_dummy = c;
switch (switch_dummy) {
  case 0 {
    x _qubit0;
  }
  case 1 {
    y _qubit0;
  }
  default {
    z _qubit0;
  }
}
"""
        self.assertEqual(test, expected)

    def test_switch_multiple_cases_to_same_block(self):
        """Test that it is possible to add multiple cases that apply to the same block, if they are
        given as a compound value.  This is an allowed special case of block fall-through."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "c")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        circuit = QuantumCircuit([qubit], creg)
        circuit.switch(creg, [(0, case1), ((1, 2), case2)], [0], circuit.clbits)

        test = dumps(circuit)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[2] c;
int switch_dummy;
qubit _qubit0;
switch_dummy = c;
switch (switch_dummy) {
  case 0 {
    x _qubit0;
  }
  case 1, 2 {
    y _qubit0;
  }
}
"""
        self.assertEqual(test, expected)

    def test_multiple_switches_dont_clash_on_dummy(self):
        """Test that having more than one switch statement in the circuit doesn't cause naming
        clashes in the dummy integer value used."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "switch_dummy")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        circuit = QuantumCircuit([qubit], creg)
        circuit.switch(creg, [(0, case1), ((1, 2), case2)], [0], circuit.clbits)
        circuit.switch(creg, [(0, case1), ((1, 2), case2)], [0], circuit.clbits)

        test = dumps(circuit)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[2] switch_dummy;
int switch_dummy_0;
int switch_dummy_1;
qubit _qubit0;
switch_dummy_0 = switch_dummy;
switch (switch_dummy_0) {
  case 0 {
    x _qubit0;
  }
  case 1, 2 {
    y _qubit0;
  }
}
switch_dummy_1 = switch_dummy;
switch (switch_dummy_1) {
  case 0 {
    x _qubit0;
  }
  case 1, 2 {
    y _qubit0;
  }
}
"""
        self.assertEqual(test, expected)

    def test_switch_nested_in_if(self):
        """Test that the switch statement works when in a nested scope, including the dummy
        classical variable being declared globally.  This isn't necessary in the OQ3 language, but
        it is universally valid and the IBM QSS stack prefers that.  They're our primary consumers
        of OQ3 strings, so it's best to play nicely with them."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "c")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        body = QuantumCircuit([qubit], creg)
        body.switch(creg, [(0, case1), ((1, 2), case2)], [0], body.clbits)

        circuit = QuantumCircuit([qubit], creg)
        circuit.if_else((creg, 1), body.copy(), body, [0], body.clbits)

        test = dumps(circuit)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[2] c;
int switch_dummy;
int switch_dummy_0;
qubit _qubit0;
if (c == 1) {
  switch_dummy = c;
  switch (switch_dummy) {
    case 0 {
      x _qubit0;
    }
    case 1, 2 {
      y _qubit0;
    }
  }
} else {
  switch_dummy_0 = c;
  switch (switch_dummy_0) {
    case 0 {
      x _qubit0;
    }
    case 1, 2 {
      y _qubit0;
    }
  }
}
"""
        self.assertEqual(test, expected)

    def test_switch_expr_target(self):
        """Simple test that the target of `switch` can be `Expr` nodes."""
        bits = [Qubit(), Clbit()]
        cr = ClassicalRegister(2, "cr")
        case0 = QuantumCircuit(1)
        case0.x(0)
        case1 = QuantumCircuit(1)
        case1.x(0)
        qc = QuantumCircuit(bits, cr)
        qc.switch(expr.logic_not(bits[1]), [(False, case0)], [0], [])
        qc.switch(expr.bit_and(cr, 3), [(3, case1)], [0], [])

        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit _bit0;
bit[2] cr;
int switch_dummy;
int switch_dummy_0;
qubit _qubit0;
switch_dummy = !_bit0;
switch (switch_dummy) {
  case 0 {
    x _qubit0;
  }
}
switch_dummy_0 = cr & 3;
switch (switch_dummy_0) {
  case 3 {
    x _qubit0;
  }
}
"""
        test = dumps(qc)
        self.assertEqual(test, expected)


class TestExperimentalFeatures(QiskitTestCase):
    """Tests of features that are hidden behind experimental flags."""

    maxDiff = None

    def test_switch_v1_clbit(self):
        """Test that a prerelease switch statement can be constructed with a bit as a condition."""
        qubit = Qubit()
        clbit = Clbit()
        case1 = QuantumCircuit([qubit, clbit])
        case1.x(0)
        case2 = QuantumCircuit([qubit, clbit])
        case2.z(0)
        circuit = QuantumCircuit([qubit, clbit])
        circuit.switch(clbit, [(True, case1), (False, case2)], [0], [0])

        test = dumps(circuit, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit _bit0;
int switch_dummy;
qubit _qubit0;
switch_dummy = _bit0;
switch (switch_dummy) {
  case 1: {
    x _qubit0;
  }
  break;
  case 0: {
    z _qubit0;
  }
  break;
}
"""
        self.assertEqual(test, expected)

    def test_switch_v1_register(self):
        """Test that a prerelease switch statement can be constructed with a register as a
        condition."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "c")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)
        case3 = QuantumCircuit([qubit], creg)
        case3.z(0)

        circuit = QuantumCircuit([qubit], creg)
        circuit.switch(creg, [(0, case1), (1, case2), (2, case3)], [0], circuit.clbits)

        test = dumps(circuit, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[2] c;
int switch_dummy;
qubit _qubit0;
switch_dummy = c;
switch (switch_dummy) {
  case 0: {
    x _qubit0;
  }
  break;
  case 1: {
    y _qubit0;
  }
  break;
  case 2: {
    z _qubit0;
  }
  break;
}
"""
        self.assertEqual(test, expected)

    def test_switch_v1_with_default(self):
        """Test that a prerelease switch statement can be constructed with a default case at the
        end."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "c")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)
        case3 = QuantumCircuit([qubit], creg)
        case3.z(0)

        circuit = QuantumCircuit([qubit], creg)
        circuit.switch(creg, [(0, case1), (1, case2), (CASE_DEFAULT, case3)], [0], circuit.clbits)

        test = dumps(circuit, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[2] c;
int switch_dummy;
qubit _qubit0;
switch_dummy = c;
switch (switch_dummy) {
  case 0: {
    x _qubit0;
  }
  break;
  case 1: {
    y _qubit0;
  }
  break;
  default: {
    z _qubit0;
  }
  break;
}
"""
        self.assertEqual(test, expected)

    def test_switch_v1_multiple_cases_to_same_block(self):
        """Test that it is possible to add multiple cases that apply to the same block, if they are
        given as a compound value.  This is an allowed special case of block fall-through."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "c")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        circuit = QuantumCircuit([qubit], creg)
        circuit.switch(creg, [(0, case1), ((1, 2), case2)], [0], circuit.clbits)

        test = dumps(circuit, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[2] c;
int switch_dummy;
qubit _qubit0;
switch_dummy = c;
switch (switch_dummy) {
  case 0: {
    x _qubit0;
  }
  break;
  case 1:
  case 2: {
    y _qubit0;
  }
  break;
}
"""
        self.assertEqual(test, expected)

    def test_multiple_switches_dont_clash_on_dummy(self):
        """Test that having more than one prerelease switch statement in the circuit doesn't cause
        naming clashes in the dummy integer value used."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "switch_dummy")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        circuit = QuantumCircuit([qubit], creg)
        circuit.switch(creg, [(0, case1), ((1, 2), case2)], [0], circuit.clbits)
        circuit.switch(creg, [(0, case1), ((1, 2), case2)], [0], circuit.clbits)

        test = dumps(circuit, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[2] switch_dummy;
int switch_dummy_0;
int switch_dummy_1;
qubit _qubit0;
switch_dummy_0 = switch_dummy;
switch (switch_dummy_0) {
  case 0: {
    x _qubit0;
  }
  break;
  case 1:
  case 2: {
    y _qubit0;
  }
  break;
}
switch_dummy_1 = switch_dummy;
switch (switch_dummy_1) {
  case 0: {
    x _qubit0;
  }
  break;
  case 1:
  case 2: {
    y _qubit0;
  }
  break;
}
"""
        self.assertEqual(test, expected)

    def test_switch_v1_nested_in_if(self):
        """Test that the switch statement works when in a nested scope, including the dummy
        classical variable being declared globally.  This isn't necessary in the OQ3 language, but
        it is universally valid and the IBM QSS stack prefers that.  They're our primary consumers
        of OQ3 strings, so it's best to play nicely with them."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "c")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        body = QuantumCircuit([qubit], creg)
        body.switch(creg, [(0, case1), ((1, 2), case2)], [0], body.clbits)

        circuit = QuantumCircuit([qubit], creg)
        circuit.if_else((creg, 1), body.copy(), body, [0], body.clbits)

        test = dumps(circuit, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit[2] c;
int switch_dummy;
int switch_dummy_0;
qubit _qubit0;
if (c == 1) {
  switch_dummy = c;
  switch (switch_dummy) {
    case 0: {
      x _qubit0;
    }
    break;
    case 1:
    case 2: {
      y _qubit0;
    }
    break;
  }
} else {
  switch_dummy_0 = c;
  switch (switch_dummy_0) {
    case 0: {
      x _qubit0;
    }
    break;
    case 1:
    case 2: {
      y _qubit0;
    }
    break;
  }
}
"""
        self.assertEqual(test, expected)

    def test_switch_v1_expr_target(self):
        """Simple test that the target of prerelease `switch` can be `Expr` nodes."""
        bits = [Qubit(), Clbit()]
        cr = ClassicalRegister(2, "cr")
        case0 = QuantumCircuit(1)
        case0.x(0)
        case1 = QuantumCircuit(1)
        case1.x(0)
        qc = QuantumCircuit(bits, cr)
        qc.switch(expr.logic_not(bits[1]), [(False, case0)], [0], [])
        qc.switch(expr.bit_and(cr, 3), [(3, case1)], [0], [])

        expected = """\
OPENQASM 3.0;
include "stdgates.inc";
bit _bit0;
bit[2] cr;
int switch_dummy;
int switch_dummy_0;
qubit _qubit0;
switch_dummy = !_bit0;
switch (switch_dummy) {
  case 0: {
    x _qubit0;
  }
  break;
}
switch_dummy_0 = cr & 3;
switch (switch_dummy_0) {
  case 3: {
    x _qubit0;
  }
  break;
}
"""
        test = dumps(qc, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        self.assertEqual(test, expected)


@ddt
class TestQASM3ExporterFailurePaths(QiskitTestCase):
    """Tests of the failure paths for the exporter."""

    def test_disallow_overlapping_classical_registers_if_no_aliasing(self):
        """Test that the exporter rejects circuits with a classical bit in more than one register if
        the ``alias_classical_registers`` option is set false."""
        qubits = [Qubit() for _ in [None] * 3]
        clbits = [Clbit() for _ in [None] * 5]
        registers = [ClassicalRegister(bits=clbits[:4]), ClassicalRegister(bits=clbits[1:])]
        qc = QuantumCircuit(qubits, *registers)
        exporter = Exporter(alias_classical_registers=False)
        with self.assertRaisesRegex(QASM3ExporterError, r"classical registers .* overlap"):
            exporter.dumps(qc)

    @data([1, 2, 1.1], [1j, 2])
    def test_disallow_for_loops_with_non_integers(self, indices):
        """Test that the exporter rejects ``for`` loops that include non-integer values in their
        index sets."""
        loop_body = QuantumCircuit()
        qc = QuantumCircuit(2, 2)
        qc.for_loop(indices, None, loop_body, [], [])
        exporter = Exporter()
        with self.assertRaisesRegex(
            QASM3ExporterError, r"The values in OpenQASM 3 'for' loops must all be integers.*"
        ):
            exporter.dumps(qc)

    def test_disallow_custom_subroutine_with_parameters(self):
        """Test that the exporter throws an error instead of trying to export a subroutine with
        parameters, while this is not supported."""
        subroutine = QuantumCircuit(1)
        subroutine.rx(Parameter("x"), 0)

        qc = QuantumCircuit(1)
        qc.append(subroutine.to_instruction(), [0], [])

        exporter = Exporter()
        with self.assertRaisesRegex(
            QASM3ExporterError, "non-unitary subroutine calls are not yet supported"
        ):
            exporter.dumps(qc)

    def test_disallow_opaque_instruction(self):
        """Test that the exporter throws an error instead of trying to export something into a
        ``defcal`` block, while this is not supported."""

        qc = QuantumCircuit(1)
        qc.append(Gate("opaque", 1, []), [0], [])

        exporter = Exporter()
        with self.assertRaisesRegex(
            QASM3ExporterError, "failed to export .* that has no definition"
        ):
            exporter.dumps(qc)

    def test_disallow_export_of_inner_scope(self):
        """A circuit with captures can't be a top-level OQ3 program."""
        qc = QuantumCircuit(captures=[expr.Var.new("a", types.Bool())])
        with self.assertRaisesRegex(
            QASM3ExporterError, "cannot export an inner scope.*as a top-level program"
        ):
            dumps(qc)

    def test_no_basis_gate_with_keyword(self):
        """Test that keyword cannot be used as a basis gate."""
        qc = QuantumCircuit()
        with self.assertRaisesRegex(QASM3ExporterError, "Cannot use 'reset' as a basis gate") as cm:
            dumps(qc, basis_gates=["U", "reset"])
        self.assertIsInstance(cm.exception.__cause__, QASM3ExporterError)
        self.assertRegex(cm.exception.__cause__.message, "cannot use the keyword 'reset'")

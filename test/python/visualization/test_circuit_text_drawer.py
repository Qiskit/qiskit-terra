# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""circuit_drawer with output="text" draws a circuit in ascii art"""

# Sometimes we want to test long-lined output.
# pylint: disable=line-too-long

import pathlib
import os
import tempfile
import unittest.mock
from codecs import encode
from math import pi

import numpy

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Gate, Parameter, Qubit, Clbit, Instruction, IfElseOp
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    InverseModifier,
    ControlModifier,
    PowerModifier,
)
from qiskit.quantum_info import random_clifford
from qiskit.quantum_info.operators import SuperOp
from qiskit.quantum_info.random import random_unitary
from qiskit.transpiler.layout import Layout, TranspileLayout
from qiskit.visualization.circuit.circuit_visualization import _text_circuit_drawer
from qiskit.visualization import circuit_drawer
from qiskit.visualization.circuit import text as elements
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import (
    HGate,
    U2Gate,
    U3Gate,
    XGate,
    CZGate,
    CXGate,
    ZGate,
    YGate,
    SGate,
    SXGate,
    U1Gate,
    SwapGate,
    RZZGate,
    CU3Gate,
    CU1Gate,
    CPhaseGate,
    UnitaryGate,
    HamiltonianGate,
    UCGate,
)
from qiskit.transpiler.passes import ApplyLayout
from qiskit.utils.optionals import HAS_TWEEDLEDUM
from test import QiskitTestCase  # pylint: disable=wrong-import-order

from .visualization import path_to_diagram_reference, QiskitVisualizationTestCase
from ..legacy_cmaps import YORKTOWN_CMAP

if HAS_TWEEDLEDUM:
    from qiskit.circuit.classicalfunction import classical_function
    from qiskit.circuit.classicalfunction.types import Int1


class TestTextDrawerElement(QiskitTestCase):
    """Draw each element"""

    def assertEqualElement(self, expected, element):
        """
        Asserts the top,mid,bot trio
        Args:
            expected (list[top,mid,bot]): What is expected.
            element (DrawElement): The element to check.
        """
        try:
            encode("\n".join(expected), encoding="cp437")
        except UnicodeEncodeError:
            self.fail("_text_circuit_drawer() should only use extended ascii (aka code page 437).")

        self.assertEqual(expected[0], element.top)
        self.assertEqual(expected[1], element.mid)
        self.assertEqual(expected[2], element.bot)

    def test_measure_to(self):
        """MeasureTo element."""
        element = elements.MeasureTo()
        # fmt: off
        expected = [" ║ ",
                    "═╩═",
                    "   "]
        # fmt: on
        self.assertEqualElement(expected, element)

    def test_measure_to_label(self):
        """MeasureTo element with cregbundle"""
        element = elements.MeasureTo("1")
        # fmt: off
        expected = [" ║ ",
                    "═╩═",
                    " 1 "]
        # fmt: on
        self.assertEqualElement(expected, element)

    def test_measure_from(self):
        """MeasureFrom element."""
        element = elements.MeasureFrom()
        # fmt: off
        expected = ["┌─┐",
                    "┤M├",
                    "└╥┘"]
        # fmt: on
        self.assertEqualElement(expected, element)

    def test_text_empty(self):
        """The empty circuit."""
        expected = ""
        circuit = QuantumCircuit()
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_pager(self):
        """The pager breaks the circuit when the drawing does not fit in the console."""
        expected = "\n".join(
            [
                "        ┌───┐     »",
                "q_0: |0>┤ X ├──■──»",
                "        └─┬─┘┌─┴─┐»",
                "q_1: |0>──■──┤ X ├»",
                "             └───┘»",
                " c: 0 1/══════════»",
                "                  »",
                "«     ┌─┐┌───┐     »",
                "«q_0: ┤M├┤ X ├──■──»",
                "«     └╥┘└─┬─┘┌─┴─┐»",
                "«q_1: ─╫───■──┤ X ├»",
                "«      ║      └───┘»",
                "«c: 1/═╩═══════════»",
                "«      0           »",
                "«     ┌─┐┌───┐     ",
                "«q_0: ┤M├┤ X ├──■──",
                "«     └╥┘└─┬─┘┌─┴─┐",
                "«q_1: ─╫───■──┤ X ├",
                "«      ║      └───┘",
                "«c: 1/═╩═══════════",
                "«      0           ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[1])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, fold=20)), expected
        )

    def test_text_no_pager(self):
        """The pager can be disable."""
        qr = QuantumRegister(1, "q")
        circuit = QuantumCircuit(qr)
        for _ in range(100):
            circuit.h(qr[0])
        amount_of_lines = str(
            circuit_drawer(circuit, output="text", initial_state=True, fold=-1)
        ).count("\n")
        self.assertEqual(amount_of_lines, 2)


class TestTextDrawerGatesInCircuit(QiskitTestCase):
    # pylint: disable=possibly-used-before-assignment
    """Gate by gate checks in different settings."""

    def test_text_measure_cregbundle(self):
        """The measure operator, using 3-bit-length registers with cregbundle=True."""
        expected = "\n".join(
            [
                "        ┌─┐      ",
                "q_0: |0>┤M├──────",
                "        └╥┘┌─┐   ",
                "q_1: |0>─╫─┤M├───",
                "         ║ └╥┘┌─┐",
                "q_2: |0>─╫──╫─┤M├",
                "         ║  ║ └╥┘",
                " c: 0 3/═╩══╩══╩═",
                "         0  1  2 ",
            ]
        )

        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_text_measure_cregbundle_2(self):
        """The measure operator, using 2 classical registers with cregbundle=True."""
        expected = "\n".join(
            [
                "        ┌─┐   ",
                "q_0: |0>┤M├───",
                "        └╥┘┌─┐",
                "q_1: |0>─╫─┤M├",
                "         ║ └╥┘",
                "cA: 0 1/═╩══╬═",
                "         0  ║ ",
                "cB: 0 1/════╩═",
                "            0 ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr_a = ClassicalRegister(1, "cA")
        cr_b = ClassicalRegister(1, "cB")
        circuit = QuantumCircuit(qr, cr_a, cr_b)
        circuit.measure(qr[0], cr_a[0])
        circuit.measure(qr[1], cr_b[0])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_text_measure_1(self):
        """The measure operator, using 3-bit-length registers."""
        expected = "\n".join(
            [
                "        ┌─┐      ",
                "q_0: |0>┤M├──────",
                "        └╥┘┌─┐   ",
                "q_1: |0>─╫─┤M├───",
                "         ║ └╥┘┌─┐",
                "q_2: |0>─╫──╫─┤M├",
                "         ║  ║ └╥┘",
                " c_0: 0 ═╩══╬══╬═",
                "            ║  ║ ",
                " c_1: 0 ════╩══╬═",
                "               ║ ",
                " c_2: 0 ═══════╩═",
                "                 ",
            ]
        )

        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_measure_1_reverse_bits(self):
        """The measure operator, using 3-bit-length registers, with reverse_bits"""
        expected = "\n".join(
            [
                "              ┌─┐",
                "q_2: |0>──────┤M├",
                "           ┌─┐└╥┘",
                "q_1: |0>───┤M├─╫─",
                "        ┌─┐└╥┘ ║ ",
                "q_0: |0>┤M├─╫──╫─",
                "        └╥┘ ║  ║ ",
                " c: 0 3/═╩══╩══╩═",
                "         0  1  2 ",
            ]
        )

        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, reverse_bits=True)),
            expected,
        )

    def test_text_measure_2(self):
        """The measure operator, using some registers."""
        expected = "\n".join(
            [
                "               ",
                "q1_0: |0>──────",
                "               ",
                "q1_1: |0>──────",
                "         ┌─┐   ",
                "q2_0: |0>┤M├───",
                "         └╥┘┌─┐",
                "q2_1: |0>─╫─┤M├",
                "          ║ └╥┘",
                " c1: 0 2/═╬══╬═",
                "          ║  ║ ",
                " c2: 0 2/═╩══╩═",
                "          0  1 ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        cr1 = ClassicalRegister(2, "c1")
        qr2 = QuantumRegister(2, "q2")
        cr2 = ClassicalRegister(2, "c2")
        circuit = QuantumCircuit(qr1, qr2, cr1, cr2)
        circuit.measure(qr2, cr2)
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_measure_2_reverse_bits(self):
        """The measure operator, using some registers, with reverse_bits"""
        expected = "\n".join(
            [
                "            ┌─┐",
                "q2_1: |0>───┤M├",
                "         ┌─┐└╥┘",
                "q2_0: |0>┤M├─╫─",
                "         └╥┘ ║ ",
                "q1_1: |0>─╫──╫─",
                "          ║  ║ ",
                "q1_0: |0>─╫──╫─",
                "          ║  ║ ",
                " c2: 0 2/═╩══╩═",
                "          0  1 ",
                " c1: 0 2/══════",
                "               ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        cr1 = ClassicalRegister(2, "c1")
        qr2 = QuantumRegister(2, "q2")
        cr2 = ClassicalRegister(2, "c2")
        circuit = QuantumCircuit(qr1, qr2, cr1, cr2)
        circuit.measure(qr2, cr2)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, reverse_bits=True)),
            expected,
        )

    def test_wire_order(self):
        """Test the wire_order option"""
        expected = "\n".join(
            [
                "                  ",
                "q_2: |0>──────────",
                "        ┌───┐     ",
                "q_1: |0>┤ X ├─────",
                "        ├───┤┌───┐",
                "q_3: |0>┤ H ├┤ X ├",
                "        ├───┤└─╥─┘",
                "q_0: |0>┤ H ├──╫──",
                "        └───┘  ║  ",
                " c_2: 0 ═══════o══",
                "               ║  ",
                "ca_0: 0 ═══════╬══",
                "               ║  ",
                "ca_1: 0 ═══════╬══",
                "               ║  ",
                " c_1: 0 ═══════■══",
                "               ║  ",
                " c_0: 0 ═══════o══",
                "               ║  ",
                " c_3: 0 ═══════■══",
                "              0xa ",
            ]
        )

        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        cr2 = ClassicalRegister(2, "ca")
        circuit = QuantumCircuit(qr, cr, cr2)
        circuit.h(0)
        circuit.h(3)
        circuit.x(1)
        with self.assertWarns(DeprecationWarning):
            circuit.x(3).c_if(cr, 10)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    cregbundle=False,
                    wire_order=[2, 1, 3, 0, 6, 8, 9, 5, 4, 7],
                )
            ),
            expected,
        )

    def test_text_swap(self):
        """Swap drawing."""
        expected = "\n".join(
            [
                "               ",
                "q1_0: |0>─X────",
                "          │    ",
                "q1_1: |0>─┼──X─",
                "          │  │ ",
                "q2_0: |0>─X──┼─",
                "             │ ",
                "q2_1: |0>────X─",
                "               ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        qr2 = QuantumRegister(2, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.swap(qr1, qr2)
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_swap_reverse_bits(self):
        """Swap drawing with reverse_bits."""
        expected = "\n".join(
            [
                "               ",
                "q2_1: |0>────X─",
                "             │ ",
                "q2_0: |0>─X──┼─",
                "          │  │ ",
                "q1_1: |0>─┼──X─",
                "          │    ",
                "q1_0: |0>─X────",
                "               ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        qr2 = QuantumRegister(2, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.swap(qr1, qr2)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, reverse_bits=True)),
            expected,
        )

    def test_text_reverse_bits_read_from_config(self):
        """Swap drawing with reverse_bits set in the configuration file."""
        expected_forward = "\n".join(
            [
                "            ",
                "q1_0: ─X────",
                "       │    ",
                "q1_1: ─┼──X─",
                "       │  │ ",
                "q2_0: ─X──┼─",
                "          │ ",
                "q2_1: ────X─",
                "            ",
            ]
        )
        expected_reverse = "\n".join(
            [
                "            ",
                "q2_1: ────X─",
                "          │ ",
                "q2_0: ─X──┼─",
                "       │  │ ",
                "q1_1: ─┼──X─",
                "       │    ",
                "q1_0: ─X────",
                "            ",
            ]
        )
        qr1 = QuantumRegister(2, "q1")
        qr2 = QuantumRegister(2, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.swap(qr1, qr2)

        self.assertEqual(str(circuit_drawer(circuit, output="text")), expected_forward)

        config_content = """
            [default]
            circuit_reverse_bits = true
        """
        with tempfile.TemporaryDirectory() as dir_path:
            file_path = pathlib.Path(dir_path) / "qiskit.conf"
            with open(file_path, "w") as fptr:
                fptr.write(config_content)
            with unittest.mock.patch.dict(os.environ, {"QISKIT_SETTINGS": str(file_path)}):
                test_reverse = str(circuit_drawer(circuit, output="text"))
        self.assertEqual(test_reverse, expected_reverse)

    def test_text_idle_wires_read_from_config(self):
        """Swap drawing with idle_wires set in the configuration file."""
        expected_with = "\n".join(
            [
                "      ┌───┐",
                "q1_0: ┤ H ├",
                "      └───┘",
                "q1_1: ─────",
                "      ┌───┐",
                "q2_0: ┤ H ├",
                "      └───┘",
                "q2_1: ─────",
                "           ",
            ]
        )
        expected_without = "\n".join(
            [
                "      ┌───┐",
                "q1_0: ┤ H ├",
                "      ├───┤",
                "q2_0: ┤ H ├",
                "      └───┘",
            ]
        )
        qr1 = QuantumRegister(2, "q1")
        qr2 = QuantumRegister(2, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.h(qr1[0])
        circuit.h(qr2[0])

        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                )
            ),
            expected_with,
        )

        config_content = """
            [default]
            circuit_idle_wires = false
        """
        with tempfile.TemporaryDirectory() as dir_path:
            file_path = pathlib.Path(dir_path) / "qiskit.conf"
            with open(file_path, "w") as fptr:
                fptr.write(config_content)
            with unittest.mock.patch.dict(os.environ, {"QISKIT_SETTINGS": str(file_path)}):
                test_without = str(circuit_drawer(circuit, output="text"))
        self.assertEqual(test_without, expected_without)

    def test_text_cswap(self):
        """CSwap drawing."""
        expected = "\n".join(
            [
                "                 ",
                "q_0: |0>─■──X──X─",
                "         │  │  │ ",
                "q_1: |0>─X──■──X─",
                "         │  │  │ ",
                "q_2: |0>─X──X──■─",
                "                 ",
            ]
        )

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cswap(qr[0], qr[1], qr[2])
        circuit.cswap(qr[1], qr[0], qr[2])
        circuit.cswap(qr[2], qr[1], qr[0])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_cswap_reverse_bits(self):
        """CSwap drawing with reverse_bits."""
        expected = "\n".join(
            [
                "                 ",
                "q_2: |0>─X──X──■─",
                "         │  │  │ ",
                "q_1: |0>─X──■──X─",
                "         │  │  │ ",
                "q_0: |0>─■──X──X─",
                "                 ",
            ]
        )

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cswap(qr[0], qr[1], qr[2])
        circuit.cswap(qr[1], qr[0], qr[2])
        circuit.cswap(qr[2], qr[1], qr[0])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, reverse_bits=True)),
            expected,
        )

    def test_text_cu3(self):
        """cu3 drawing."""
        expected = "\n".join(
            [
                "                           ┌─────────────────┐",
                "q_0: |0>─────────■─────────┤ U3(π/2,π/2,π/2) ├",
                "        ┌────────┴────────┐└────────┬────────┘",
                "q_1: |0>┤ U3(π/2,π/2,π/2) ├─────────┼─────────",
                "        └─────────────────┘         │         ",
                "q_2: |0>────────────────────────────■─────────",
                "                                              ",
            ]
        )

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(CU3Gate(pi / 2, pi / 2, pi / 2), [qr[0], qr[1]])
        circuit.append(CU3Gate(pi / 2, pi / 2, pi / 2), [qr[2], qr[0]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_cu3_reverse_bits(self):
        """cu3 drawing with reverse_bits"""
        expected = "\n".join(
            [
                "                                              ",
                "q_2: |0>────────────────────────────■─────────",
                "        ┌─────────────────┐         │         ",
                "q_1: |0>┤ U3(π/2,π/2,π/2) ├─────────┼─────────",
                "        └────────┬────────┘┌────────┴────────┐",
                "q_0: |0>─────────■─────────┤ U3(π/2,π/2,π/2) ├",
                "                           └─────────────────┘",
            ]
        )

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(CU3Gate(pi / 2, pi / 2, pi / 2), [qr[0], qr[1]])
        circuit.append(CU3Gate(pi / 2, pi / 2, pi / 2), [qr[2], qr[0]])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, reverse_bits=True)),
            expected,
        )

    def test_text_crz(self):
        """crz drawing."""
        expected = "\n".join(
            [
                "                   ┌─────────┐",
                "q_0: |0>─────■─────┤ Rz(π/2) ├",
                "        ┌────┴────┐└────┬────┘",
                "q_1: |0>┤ Rz(π/2) ├─────┼─────",
                "        └─────────┘     │     ",
                "q_2: |0>────────────────■─────",
                "                              ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.crz(pi / 2, qr[0], qr[1])
        circuit.crz(pi / 2, qr[2], qr[0])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_cry(self):
        """cry drawing."""
        expected = "\n".join(
            [
                "                   ┌─────────┐",
                "q_0: |0>─────■─────┤ Ry(π/2) ├",
                "        ┌────┴────┐└────┬────┘",
                "q_1: |0>┤ Ry(π/2) ├─────┼─────",
                "        └─────────┘     │     ",
                "q_2: |0>────────────────■─────",
                "                              ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cry(pi / 2, qr[0], qr[1])
        circuit.cry(pi / 2, qr[2], qr[0])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_crx(self):
        """crx drawing."""
        expected = "\n".join(
            [
                "                   ┌─────────┐",
                "q_0: |0>─────■─────┤ Rx(π/2) ├",
                "        ┌────┴────┐└────┬────┘",
                "q_1: |0>┤ Rx(π/2) ├─────┼─────",
                "        └─────────┘     │     ",
                "q_2: |0>────────────────■─────",
                "                              ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.crx(pi / 2, qr[0], qr[1])
        circuit.crx(pi / 2, qr[2], qr[0])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_cx(self):
        """cx drawing."""
        expected = "\n".join(
            [
                "             ┌───┐",
                "q_0: |0>──■──┤ X ├",
                "        ┌─┴─┐└─┬─┘",
                "q_1: |0>┤ X ├──┼──",
                "        └───┘  │  ",
                "q_2: |0>───────■──",
                "                  ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[0])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_cy(self):
        """cy drawing."""
        expected = "\n".join(
            [
                "             ┌───┐",
                "q_0: |0>──■──┤ Y ├",
                "        ┌─┴─┐└─┬─┘",
                "q_1: |0>┤ Y ├──┼──",
                "        └───┘  │  ",
                "q_2: |0>───────■──",
                "                  ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cy(qr[0], qr[1])
        circuit.cy(qr[2], qr[0])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_cz(self):
        """cz drawing."""
        expected = "\n".join(
            [
                "              ",
                "q_0: |0>─■──■─",
                "         │  │ ",
                "q_1: |0>─■──┼─",
                "            │ ",
                "q_2: |0>────■─",
                "              ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cz(qr[0], qr[1])
        circuit.cz(qr[2], qr[0])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_ch(self):
        """ch drawing."""
        expected = "\n".join(
            [
                "             ┌───┐",
                "q_0: |0>──■──┤ H ├",
                "        ┌─┴─┐└─┬─┘",
                "q_1: |0>┤ H ├──┼──",
                "        └───┘  │  ",
                "q_2: |0>───────■──",
                "                  ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.ch(qr[0], qr[1])
        circuit.ch(qr[2], qr[0])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_rzz(self):
        """rzz drawing. See #1957"""
        expected = "\n".join(
            [
                "                          ",
                "q_0: |0>─■────────────────",
                "         │ZZ(0)           ",
                "q_1: |0>─■───────■────────",
                "                 │ZZ(π/2) ",
                "q_2: |0>─────────■────────",
                "                          ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.rzz(0, qr[0], qr[1])
        circuit.rzz(pi / 2, qr[2], qr[1])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_cu1(self):
        """cu1 drawing."""
        expected = "\n".join(
            [
                "                            ",
                "q_0: |0>─■─────────■────────",
                "         │U1(π/2)  │        ",
                "q_1: |0>─■─────────┼────────",
                "                   │U1(π/2) ",
                "q_2: |0>───────────■────────",
                "                            ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(CU1Gate(pi / 2), [qr[0], qr[1]])
        circuit.append(CU1Gate(pi / 2), [qr[2], qr[0]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_cp(self):
        """cp drawing."""
        expected = "\n".join(
            [
                "                          ",
                "q_0: |0>─■────────■───────",
                "         │P(π/2)  │       ",
                "q_1: |0>─■────────┼───────",
                "                  │P(π/2) ",
                "q_2: |0>──────────■───────",
                "                          ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(CPhaseGate(pi / 2), [qr[0], qr[1]])
        circuit.append(CPhaseGate(pi / 2), [qr[2], qr[0]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_cu1_condition(self):
        """Test cu1 with condition"""
        expected = "\n".join(
            [
                "                      ",
                "q_0: ────────■────────",
                "             │U1(π/2) ",
                "q_1: ────────■────────",
                "             ║        ",
                "q_2: ────────╫────────",
                "        ┌────╨────┐   ",
                "c: 3/═══╡ c_1=0x1 ╞═══",
                "        └─────────┘   ",
            ]
        )
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.append(CU1Gate(pi / 2), [qr[0], qr[1]]).c_if(cr[1], 1)
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=False)), expected)

    def test_text_rzz_condition(self):
        """Test rzz with condition"""
        expected = "\n".join(
            [
                "                      ",
                "q_0: ────────■────────",
                "             │ZZ(π/2) ",
                "q_1: ────────■────────",
                "             ║        ",
                "q_2: ────────╫────────",
                "        ┌────╨────┐   ",
                "c: 3/═══╡ c_1=0x1 ╞═══",
                "        └─────────┘   ",
            ]
        )
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.append(RZZGate(pi / 2), [qr[0], qr[1]]).c_if(cr[1], 1)
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=False)), expected)

    def test_text_cp_condition(self):
        """Test cp with condition"""
        expected = "\n".join(
            [
                "                    ",
                "q_0: ───────■───────",
                "            │P(π/2) ",
                "q_1: ───────■───────",
                "            ║       ",
                "q_2: ───────╫───────",
                "       ┌────╨────┐  ",
                "c: 3/══╡ c_1=0x1 ╞══",
                "       └─────────┘  ",
            ]
        )
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.append(CPhaseGate(pi / 2), [qr[0], qr[1]]).c_if(cr[1], 1)
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=False)), expected)

    def test_text_cu1_reverse_bits(self):
        """cu1 drawing with reverse_bits"""
        expected = "\n".join(
            [
                "                            ",
                "q_2: |0>───────────■────────",
                "                   │        ",
                "q_1: |0>─■─────────┼────────",
                "         │U1(π/2)  │U1(π/2) ",
                "q_0: |0>─■─────────■────────",
                "                            ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(CU1Gate(pi / 2), [qr[0], qr[1]])
        circuit.append(CU1Gate(pi / 2), [qr[2], qr[0]])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, reverse_bits=True)),
            expected,
        )

    def test_text_ccx(self):
        """cx drawing."""
        expected = "\n".join(
            [
                "                  ┌───┐",
                "q_0: |0>──■────■──┤ X ├",
                "          │  ┌─┴─┐└─┬─┘",
                "q_1: |0>──■──┤ X ├──■──",
                "        ┌─┴─┐└─┬─┘  │  ",
                "q_2: |0>┤ X ├──■────■──",
                "        └───┘          ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.ccx(qr[0], qr[1], qr[2])
        circuit.ccx(qr[2], qr[0], qr[1])
        circuit.ccx(qr[2], qr[1], qr[0])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_reset(self):
        """Reset drawing."""
        expected = "\n".join(
            [
                "              ",
                "q1_0: |0>─|0>─",
                "              ",
                "q1_1: |0>─|0>─",
                "              ",
                "q2_0: |0>─────",
                "              ",
                "q2_1: |0>─|0>─",
                "              ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        qr2 = QuantumRegister(2, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.reset(qr1)
        circuit.reset(qr2[1])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_single_gate(self):
        """Single Qbit gate drawing."""
        expected = "\n".join(
            [
                "         ┌───┐",
                "q1_0: |0>┤ H ├",
                "         ├───┤",
                "q1_1: |0>┤ H ├",
                "         └───┘",
                "q2_0: |0>─────",
                "         ┌───┐",
                "q2_1: |0>┤ H ├",
                "         └───┘",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        qr2 = QuantumRegister(2, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.h(qr1)
        circuit.h(qr2[1])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_id(self):
        """Id drawing."""
        expected = "\n".join(
            [
                "         ┌───┐",
                "q1_0: |0>┤ I ├",
                "         ├───┤",
                "q1_1: |0>┤ I ├",
                "         └───┘",
                "q2_0: |0>─────",
                "         ┌───┐",
                "q2_1: |0>┤ I ├",
                "         └───┘",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        qr2 = QuantumRegister(2, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.id(qr1)
        circuit.id(qr2[1])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_barrier(self):
        """Barrier drawing."""
        expected = "\n".join(
            [
                "          ░ ",
                "q1_0: |0>─░─",
                "          ░ ",
                "q1_1: |0>─░─",
                "          ░ ",
                "q2_0: |0>───",
                "          ░ ",
                "q2_1: |0>─░─",
                "          ░ ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        qr2 = QuantumRegister(2, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.barrier(qr1)
        circuit.barrier(qr2[1])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_no_barriers(self):
        """Drawing without plotbarriers."""
        expected = "\n".join(
            [
                "         ┌───┐     ",
                "q1_0: |0>┤ H ├─────",
                "         ├───┤     ",
                "q1_1: |0>┤ H ├─────",
                "         ├───┤     ",
                "q2_0: |0>┤ H ├─────",
                "         └───┘┌───┐",
                "q2_1: |0>─────┤ H ├",
                "              └───┘",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        qr2 = QuantumRegister(2, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.h(qr1)
        circuit.barrier(qr1)
        circuit.barrier(qr2[1])
        circuit.h(qr2)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, plot_barriers=False)),
            expected,
        )

    def test_text_measure_html(self):
        """The measure operator. HTML representation."""
        expected = "\n".join(
            [
                '<pre style="word-wrap: normal;'
                "white-space: pre;"
                "background: #fff0;"
                "line-height: 1.1;"
                'font-family: &quot;Courier New&quot;,Courier,monospace">'
                "       ┌─┐",
                " q: |0>┤M├",
                "       └╥┘",
                "c: 0 1/═╩═",
                "        0 </pre>",
            ]
        )
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        self.assertEqual(
            circuit_drawer(circuit, output="text", initial_state=True)._repr_html_(), expected
        )

    def test_text_repr(self):
        """The measure operator. repr."""
        expected = "\n".join(
            [
                "       ┌─┐",
                " q: |0>┤M├",
                "       └╥┘",
                "c: 0 1/═╩═",
                "        0 ",
            ]
        )
        qr = QuantumRegister(1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        self.assertEqual(
            circuit_drawer(circuit, output="text", initial_state=True).__repr__(), expected
        )

    def test_text_justify_left(self):
        """Drawing with left justify"""
        expected = "\n".join(
            [
                "         ┌───┐   ",
                "q1_0: |0>┤ X ├───",
                "         ├───┤┌─┐",
                "q1_1: |0>┤ H ├┤M├",
                "         └───┘└╥┘",
                " c1: 0 2/══════╩═",
                "               1 ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        cr1 = ClassicalRegister(2, "c1")
        circuit = QuantumCircuit(qr1, cr1)
        circuit.x(qr1[0])
        circuit.h(qr1[1])
        circuit.measure(qr1[1], cr1[1])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, justify="left")),
            expected,
        )

    def test_text_justify_right(self):
        """Drawing with right justify"""
        expected = "\n".join(
            [
                "              ┌───┐",
                "q1_0: |0>─────┤ X ├",
                "         ┌───┐└┬─┬┘",
                "q1_1: |0>┤ H ├─┤M├─",
                "         └───┘ └╥┘ ",
                " c1: 0 2/═══════╩══",
                "                1  ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        cr1 = ClassicalRegister(2, "c1")
        circuit = QuantumCircuit(qr1, cr1)
        circuit.x(qr1[0])
        circuit.h(qr1[1])
        circuit.measure(qr1[1], cr1[1])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, justify="right")),
            expected,
        )

    def test_text_justify_none(self):
        """Drawing with none justify"""
        expected = "\n".join(
            [
                "         ┌───┐        ",
                "q1_0: |0>┤ X ├────────",
                "         └───┘┌───┐┌─┐",
                "q1_1: |0>─────┤ H ├┤M├",
                "              └───┘└╥┘",
                " c1: 0 2/═══════════╩═",
                "                    1 ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        cr1 = ClassicalRegister(2, "c1")
        circuit = QuantumCircuit(qr1, cr1)
        circuit.x(qr1[0])
        circuit.h(qr1[1])
        circuit.measure(qr1[1], cr1[1])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, justify="none")),
            expected,
        )

    def test_text_justify_left_barrier(self):
        """Left justify respects barriers"""
        expected = "\n".join(
            [
                "         ┌───┐ ░      ",
                "q1_0: |0>┤ H ├─░──────",
                "         └───┘ ░ ┌───┐",
                "q1_1: |0>──────░─┤ H ├",
                "               ░ └───┘",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        circuit = QuantumCircuit(qr1)
        circuit.h(qr1[0])
        circuit.barrier(qr1)
        circuit.h(qr1[1])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, justify="left")),
            expected,
        )

    def test_text_justify_right_barrier(self):
        """Right justify respects barriers"""
        expected = "\n".join(
            [
                "         ┌───┐ ░      ",
                "q1_0: |0>┤ H ├─░──────",
                "         └───┘ ░ ┌───┐",
                "q1_1: |0>──────░─┤ H ├",
                "               ░ └───┘",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        circuit = QuantumCircuit(qr1)
        circuit.h(qr1[0])
        circuit.barrier(qr1)
        circuit.h(qr1[1])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, justify="right")),
            expected,
        )

    def test_text_barrier_label(self):
        """Show barrier label"""
        expected = "\n".join(
            [
                "        ┌───┐ ░ ┌───┐ End Y/X ",
                "q_0: |0>┤ X ├─░─┤ Y ├────░────",
                "        ├───┤ ░ ├───┤    ░    ",
                "q_1: |0>┤ Y ├─░─┤ X ├────░────",
                "        └───┘ ░ └───┘    ░    ",
            ]
        )

        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.x(0)
        circuit.y(1)
        circuit.barrier()
        circuit.y(0)
        circuit.x(1)
        circuit.barrier(label="End Y/X")
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_overlap_cx(self):
        """Overlapping CX gates are drawn not overlapping"""
        expected = "\n".join(
            [
                "                   ",
                "q1_0: |0>──■───────",
                "           │       ",
                "q1_1: |0>──┼────■──",
                "           │  ┌─┴─┐",
                "q1_2: |0>──┼──┤ X ├",
                "         ┌─┴─┐└───┘",
                "q1_3: |0>┤ X ├─────",
                "         └───┘     ",
            ]
        )

        qr1 = QuantumRegister(4, "q1")
        circuit = QuantumCircuit(qr1)
        circuit.cx(qr1[0], qr1[3])
        circuit.cx(qr1[1], qr1[2])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, justify="left")),
            expected,
        )

    def test_text_overlap_measure(self):
        """Measure is drawn not overlapping"""
        expected = "\n".join(
            [
                "         ┌─┐     ",
                "q1_0: |0>┤M├─────",
                "         └╥┘┌───┐",
                "q1_1: |0>─╫─┤ X ├",
                "          ║ └───┘",
                " c1: 0 2/═╩══════",
                "          0      ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        cr1 = ClassicalRegister(2, "c1")
        circuit = QuantumCircuit(qr1, cr1)
        circuit.measure(qr1[0], cr1[0])
        circuit.x(qr1[1])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, justify="left")),
            expected,
        )

    def test_text_overlap_swap(self):
        """Swap is drawn in 2 separate columns"""
        expected = "\n".join(
            [
                "               ",
                "q1_0: |0>─X────",
                "          │    ",
                "q1_1: |0>─┼──X─",
                "          │  │ ",
                "q2_0: |0>─X──┼─",
                "             │ ",
                "q2_1: |0>────X─",
                "               ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        qr2 = QuantumRegister(2, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.swap(qr1, qr2)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, justify="left")),
            expected,
        )

    def test_text_justify_right_measure_resize(self):
        """Measure gate can resize if necessary"""
        expected = "\n".join(
            [
                "         ┌───┐",
                "q1_0: |0>┤ X ├",
                "         └┬─┬┘",
                "q1_1: |0>─┤M├─",
                "          └╥┘ ",
                " c1: 0 2/══╩══",
                "           1  ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        cr1 = ClassicalRegister(2, "c1")
        circuit = QuantumCircuit(qr1, cr1)
        circuit.x(qr1[0])
        circuit.measure(qr1[1], cr1[1])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, justify="right")),
            expected,
        )

    def test_text_box_length(self):
        """The length of boxes is independent of other boxes in the layer
        https://github.com/Qiskit/qiskit-terra/issues/1882"""
        expected = "\n".join(
            [
                "             ┌───┐    ┌───┐",
                "q1_0: |0>────┤ H ├────┤ H ├",
                "             └───┘    └───┘",
                "q1_1: |0>──────────────────",
                "         ┌───────────┐     ",
                "q1_2: |0>┤ Rz(1e-07) ├─────",
                "         └───────────┘     ",
            ]
        )

        qr = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.rz(0.0000001, qr[2])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_spacing_2378(self):
        """Small gates in the same layer as long gates.
        See https://github.com/Qiskit/qiskit-terra/issues/2378"""
        expected = "\n".join(
            [
                "                     ",
                "q_0: |0>──────X──────",
                "              │      ",
                "q_1: |0>──────X──────",
                "        ┌───────────┐",
                "q_2: |0>┤ Rz(11111) ├",
                "        └───────────┘",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.swap(qr[0], qr[1])
        circuit.rz(11111, qr[2])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    @unittest.skipUnless(HAS_TWEEDLEDUM, "Tweedledum is required for these tests.")
    def test_text_synth_no_registerless(self):
        """Test synthesis's label when registerless=False.
        See https://github.com/Qiskit/qiskit-terra/issues/9363"""
        expected = "\n".join(
            [
                "                ",
                "     a: |0>──■──",
                "             │  ",
                "     b: |0>──■──",
                "             │  ",
                "     c: |0>──o──",
                "           ┌─┴─┐",
                "return: |0>┤ X ├",
                "           └───┘",
            ]
        )

        @classical_function
        def grover_oracle(a: Int1, b: Int1, c: Int1) -> Int1:
            return a and b and not c

        circuit = grover_oracle.synth(registerless=False)
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)


class TestTextDrawerLabels(QiskitTestCase):
    """Gates with labels."""

    def test_label(self):
        """Test a gate with a label."""
        # fmt: off
        expected = "\n".join(["      ┌───────────┐",
                              "q: |0>┤ an H gate ├",
                              "      └───────────┘"])
        # fmt: on
        circuit = QuantumCircuit(1)
        circuit.append(HGate(label="an H gate"), [0])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_controlled_gate_with_label(self):
        """Test a controlled gate-with-a-label."""
        expected = "\n".join(
            [
                "                     ",
                "q_0: |0>──────■──────",
                "        ┌─────┴─────┐",
                "q_1: |0>┤ an H gate ├",
                "        └───────────┘",
            ]
        )
        circuit = QuantumCircuit(2)
        circuit.append(HGate(label="an H gate").control(1), [0, 1])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_label_on_controlled_gate(self):
        """Test a controlled gate with a label (as a as a whole)."""
        expected = "\n".join(
            [
                "         a controlled H gate ",
                "q_0: |0>──────────■──────────",
                "                ┌─┴─┐        ",
                "q_1: |0>────────┤ H ├────────",
                "                └───┘        ",
            ]
        )

        circuit = QuantumCircuit(2)
        circuit.append(HGate().control(1, label="a controlled H gate"), [0, 1])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_rzz_on_wide_layer(self):
        """Test a labeled gate (RZZ) in a wide layer.
        See https://github.com/Qiskit/qiskit-terra/issues/4838"""
        expected = "\n".join(
            [
                "                                               ",
                "q_0: |0>────────────────■──────────────────────",
                "                        │ZZ(π/2)               ",
                "q_1: |0>────────────────■──────────────────────",
                "        ┌─────────────────────────────────────┐",
                "q_2: |0>┤ This is a really long long long box ├",
                "        └─────────────────────────────────────┘",
            ]
        )
        circuit = QuantumCircuit(3)
        circuit.rzz(pi / 2, 0, 1)
        circuit.x(2, label="This is a really long long long box")

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_cu1_on_wide_layer(self):
        """Test a labeled gate (CU1) in a wide layer.
        See https://github.com/Qiskit/qiskit-terra/issues/4838"""
        expected = "\n".join(
            [
                "                                               ",
                "q_0: |0>────────────────■──────────────────────",
                "                        │U1(π/2)               ",
                "q_1: |0>────────────────■──────────────────────",
                "        ┌─────────────────────────────────────┐",
                "q_2: |0>┤ This is a really long long long box ├",
                "        └─────────────────────────────────────┘",
            ]
        )
        circuit = QuantumCircuit(3)
        circuit.append(CU1Gate(pi / 2), [0, 1])
        circuit.x(2, label="This is a really long long long box")

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)


class TestTextDrawerMultiQGates(QiskitTestCase):
    """Gates implying multiple qubits."""

    def test_2Qgate(self):
        """2Q no params."""
        expected = "\n".join(
            [
                "        ┌───────┐",
                "q_1: |0>┤1      ├",
                "        │  twoQ │",
                "q_0: |0>┤0      ├",
                "        └───────┘",
            ]
        )

        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)

        my_gate2 = Gate(name="twoQ", num_qubits=2, params=[], label="twoQ")
        circuit.append(my_gate2, [qr[0], qr[1]])

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, reverse_bits=True)),
            expected,
        )

    def test_2Qgate_cross_wires(self):
        """2Q no params, with cross wires"""
        expected = "\n".join(
            [
                "        ┌───────┐",
                "q_1: |0>┤0      ├",
                "        │  twoQ │",
                "q_0: |0>┤1      ├",
                "        └───────┘",
            ]
        )

        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)

        my_gate2 = Gate(name="twoQ", num_qubits=2, params=[], label="twoQ")
        circuit.append(my_gate2, [qr[1], qr[0]])

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, reverse_bits=True)),
            expected,
        )

    def test_3Qgate_cross_wires(self):
        """3Q no params, with cross wires"""
        expected = "\n".join(
            [
                "        ┌─────────┐",
                "q_2: |0>┤1        ├",
                "        │         │",
                "q_1: |0>┤0 threeQ ├",
                "        │         │",
                "q_0: |0>┤2        ├",
                "        └─────────┘",
            ]
        )

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)

        my_gate3 = Gate(name="threeQ", num_qubits=3, params=[], label="threeQ")
        circuit.append(my_gate3, [qr[1], qr[2], qr[0]])

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, reverse_bits=True)),
            expected,
        )

    def test_2Qgate_nottogether(self):
        """2Q that are not together"""
        expected = "\n".join(
            [
                "        ┌───────┐",
                "q_2: |0>┤1      ├",
                "        │       │",
                "q_1: |0>┤  twoQ ├",
                "        │       │",
                "q_0: |0>┤0      ├",
                "        └───────┘",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)

        my_gate2 = Gate(name="twoQ", num_qubits=2, params=[], label="twoQ")
        circuit.append(my_gate2, [qr[0], qr[2]])

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, reverse_bits=True)),
            expected,
        )

    def test_2Qgate_nottogether_across_4(self):
        """2Q that are 2 bits apart"""
        expected = "\n".join(
            [
                "        ┌───────┐",
                "q_3: |0>┤1      ├",
                "        │       │",
                "q_2: |0>┤       ├",
                "        │  twoQ │",
                "q_1: |0>┤       ├",
                "        │       │",
                "q_0: |0>┤0      ├",
                "        └───────┘",
            ]
        )

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)

        my_gate2 = Gate(name="twoQ", num_qubits=2, params=[], label="twoQ")
        circuit.append(my_gate2, [qr[0], qr[3]])

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, reverse_bits=True)),
            expected,
        )

    def test_unitary_nottogether_across_4(self):
        """unitary that are 2 bits apart"""
        expected = "\n".join(
            [
                "        ┌──────────┐",
                "q_0: |0>┤0         ├",
                "        │          │",
                "q_1: |0>┤          ├",
                "        │  Unitary │",
                "q_2: |0>┤          ├",
                "        │          │",
                "q_3: |0>┤1         ├",
                "        └──────────┘",
            ]
        )

        qr = QuantumRegister(4, "q")
        qc = QuantumCircuit(qr)

        qc.append(random_unitary(4, seed=42), [qr[0], qr[3]])

        self.assertEqual(str(circuit_drawer(qc, initial_state=True, output="text")), expected)

    def test_kraus(self):
        """Test Kraus.
        See https://github.com/Qiskit/qiskit-terra/pull/2238#issuecomment-487630014"""
        # fmt: off
        expected = "\n".join(["      ┌───────┐",
                              "q: |0>┤ kraus ├",
                              "      └───────┘"])
        # fmt: on
        error = SuperOp(0.75 * numpy.eye(4) + 0.25 * numpy.diag([1, -1, -1, 1]))
        qr = QuantumRegister(1, name="q")
        qc = QuantumCircuit(qr)
        qc.append(error, [qr[0]])

        self.assertEqual(str(circuit_drawer(qc, output="text", initial_state=True)), expected)

    def test_multiplexer(self):
        """Test Multiplexer.
        See https://github.com/Qiskit/qiskit-terra/pull/2238#issuecomment-487630014"""
        expected = "\n".join(
            [
                "        ┌──────────────┐",
                "q_0: |0>┤0             ├",
                "        │  Multiplexer │",
                "q_1: |0>┤1             ├",
                "        └──────────────┘",
            ]
        )

        cx_multiplexer = UCGate([numpy.eye(2), numpy.array([[0, 1], [1, 0]])])

        qr = QuantumRegister(2, name="q")
        qc = QuantumCircuit(qr)
        qc.append(cx_multiplexer, [qr[0], qr[1]])

        self.assertEqual(str(circuit_drawer(qc, output="text", initial_state=True)), expected)

    def test_label_over_name_2286(self):
        """If there is a label, it should be used instead of the name
        See https://github.com/Qiskit/qiskit-terra/issues/2286"""
        expected = "\n".join(
            [
                "        ┌───┐┌───────┐┌────────┐",
                "q_0: |0>┤ X ├┤ alt-X ├┤0       ├",
                "        └───┘└───────┘│  iswap │",
                "q_1: |0>──────────────┤1       ├",
                "                      └────────┘",
            ]
        )
        qr = QuantumRegister(2, "q")
        circ = QuantumCircuit(qr)
        circ.append(XGate(), [qr[0]])
        circ.append(XGate(label="alt-X"), [qr[0]])
        circ.append(UnitaryGate(numpy.eye(4), label="iswap"), [qr[0], qr[1]])

        self.assertEqual(str(circuit_drawer(circ, output="text", initial_state=True)), expected)

    def test_label_turns_to_box_2286(self):
        """If there is a label, non-boxes turn into boxes
        See https://github.com/Qiskit/qiskit-terra/issues/2286"""
        expected = "\n".join(
            [
                "            cz label ",
                "q_0: |0>─■─────■─────",
                "         │     │     ",
                "q_1: |0>─■─────■─────",
                "                     ",
            ]
        )
        qr = QuantumRegister(2, "q")

        circ = QuantumCircuit(qr)
        circ.append(CZGate(), [qr[0], qr[1]])
        circ.append(CZGate(label="cz label"), [qr[0], qr[1]])

        self.assertEqual(str(circuit_drawer(circ, output="text", initial_state=True)), expected)

    def test_control_gate_with_base_label_4361(self):
        """Control gate has a label and a base gate with a label
        See https://github.com/Qiskit/qiskit-terra/issues/4361"""
        expected = "\n".join(
            [
                "        ┌──────┐ my ch  ┌──────┐",
                "q_0: |0>┤ my h ├───■────┤ my h ├",
                "        └──────┘┌──┴───┐└──┬───┘",
                "q_1: |0>────────┤ my h ├───■────",
                "                └──────┘ my ch  ",
            ]
        )
        qr = QuantumRegister(2, "q")
        circ = QuantumCircuit(qr)
        hgate = HGate(label="my h")
        controlh = hgate.control(label="my ch")
        circ.append(hgate, [0])
        circ.append(controlh, [0, 1])
        circ.append(controlh, [1, 0])

        self.assertEqual(str(circuit_drawer(circ, output="text", initial_state=True)), expected)

    def test_control_gate_label_with_cond_1_low(self):
        """Control gate has a label and a conditional (compression=low)
        See https://github.com/Qiskit/qiskit-terra/issues/4361"""
        expected = "\n".join(
            [
                "         my ch  ",
                "q_0: |0>───■────",
                "           │    ",
                "        ┌──┴───┐",
                "q_1: |0>┤ my h ├",
                "        └──╥───┘",
                "        ┌──╨──┐ ",
                " c: 0 1/╡ 0x1 ╞═",
                "        └─────┘ ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        hgate = HGate(label="my h")
        with self.assertWarns(DeprecationWarning):
            controlh = hgate.control(label="my ch").c_if(cr, 1)
        circ.append(controlh, [0, 1])

        self.assertEqual(
            str(
                circuit_drawer(circ, output="text", initial_state=True, vertical_compression="low")
            ),
            expected,
        )

    def test_control_gate_label_with_cond_1_low_cregbundle(self):
        """Control gate has a label and a conditional (compression=low) with cregbundle
        See https://github.com/Qiskit/qiskit-terra/issues/4361"""
        expected = "\n".join(
            [
                "         my ch  ",
                "q_0: |0>───■────",
                "           │    ",
                "        ┌──┴───┐",
                "q_1: |0>┤ my h ├",
                "        └──╥───┘",
                "        ┌──╨──┐ ",
                " c: 0 1/╡ 0x1 ╞═",
                "        └─────┘ ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        hgate = HGate(label="my h")
        with self.assertWarns(DeprecationWarning):
            controlh = hgate.control(label="my ch").c_if(cr, 1)
        circ.append(controlh, [0, 1])

        self.assertEqual(
            str(
                circuit_drawer(
                    circ,
                    output="text",
                    initial_state=True,
                    vertical_compression="low",
                    cregbundle=True,
                )
            ),
            expected,
        )

    def test_control_gate_label_with_cond_1_med(self):
        """Control gate has a label and a conditional (compression=med)
        See https://github.com/Qiskit/qiskit-terra/issues/4361"""
        expected = "\n".join(
            [
                "         my ch  ",
                "q_0: |0>───■────",
                "        ┌──┴───┐",
                "q_1: |0>┤ my h ├",
                "        └──╥───┘",
                "   c: 0 ═══■════",
                "          0x1   ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        hgate = HGate(label="my h")
        with self.assertWarns(DeprecationWarning):
            controlh = hgate.control(label="my ch").c_if(cr, 1)
        circ.append(controlh, [0, 1])

        self.assertEqual(
            str(
                circuit_drawer(
                    circ,
                    output="text",
                    initial_state=True,
                    cregbundle=False,
                    vertical_compression="medium",
                )
            ),
            expected,
        )

    def test_control_gate_label_with_cond_1_med_cregbundle(self):
        """Control gate has a label and a conditional (compression=med) with cregbundle
        See https://github.com/Qiskit/qiskit-terra/issues/4361"""
        expected = "\n".join(
            [
                "         my ch  ",
                "q_0: |0>───■────",
                "        ┌──┴───┐",
                "q_1: |0>┤ my h ├",
                "        └──╥───┘",
                "        ┌──╨──┐ ",
                " c: 0 1/╡ 0x1 ╞═",
                "        └─────┘ ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        hgate = HGate(label="my h")
        with self.assertWarns(DeprecationWarning):
            controlh = hgate.control(label="my ch").c_if(cr, 1)
        circ.append(controlh, [0, 1])

        self.assertEqual(
            str(
                circuit_drawer(
                    circ,
                    output="text",
                    initial_state=True,
                    cregbundle=True,
                    vertical_compression="medium",
                )
            ),
            expected,
        )

    def test_control_gate_label_with_cond_1_high(self):
        """Control gate has a label and a conditional (compression=high)
        See https://github.com/Qiskit/qiskit-terra/issues/4361"""
        expected = "\n".join(
            [
                "         my ch  ",
                "q_0: |0>───■────",
                "        ┌──┴───┐",
                "q_1: |0>┤ my h ├",
                "        └──╥───┘",
                "   c: 0 ═══■════",
                "          0x1   ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        hgate = HGate(label="my h")
        with self.assertWarns(DeprecationWarning):
            controlh = hgate.control(label="my ch").c_if(cr, 1)
        circ.append(controlh, [0, 1])

        self.assertEqual(
            str(
                circuit_drawer(
                    circ,
                    output="text",
                    initial_state=True,
                    cregbundle=False,
                    vertical_compression="high",
                )
            ),
            expected,
        )

    def test_control_gate_label_with_cond_1_high_cregbundle(self):
        """Control gate has a label and a conditional (compression=high) with cregbundle
        See https://github.com/Qiskit/qiskit-terra/issues/4361"""
        expected = "\n".join(
            [
                "         my ch  ",
                "q_0: |0>───■────",
                "        ┌──┴───┐",
                "q_1: |0>┤ my h ├",
                "        ├──╨──┬┘",
                " c: 0 1/╡ 0x1 ╞═",
                "        └─────┘ ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        hgate = HGate(label="my h")
        with self.assertWarns(DeprecationWarning):
            controlh = hgate.control(label="my ch").c_if(cr, 1)
        circ.append(controlh, [0, 1])

        self.assertEqual(
            str(
                circuit_drawer(
                    circ,
                    output="text",
                    initial_state=True,
                    cregbundle=True,
                    vertical_compression="high",
                )
            ),
            expected,
        )

    def test_control_gate_label_with_cond_2_med_space(self):
        """Control gate has a label and a conditional (on label, compression=med)
        See https://github.com/Qiskit/qiskit-terra/issues/4361"""
        expected = "\n".join(
            [
                "        ┌──────┐",
                "q_0: |0>┤ my h ├",
                "        └──┬───┘",
                "q_1: |0>───■────",
                "         my ch  ",
                "        ┌──╨──┐ ",
                " c: 0 1/╡ 0x1 ╞═",
                "        └─────┘ ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        hgate = HGate(label="my h")
        with self.assertWarns(DeprecationWarning):
            controlh = hgate.control(label="my ch").c_if(cr, 1)
        circ.append(controlh, [1, 0])

        self.assertEqual(
            str(
                circuit_drawer(
                    circ, output="text", initial_state=True, vertical_compression="medium"
                )
            ),
            expected,
        )

    def test_control_gate_label_with_cond_2_med(self):
        """Control gate has a label and a conditional (on label, compression=med)
        See https://github.com/Qiskit/qiskit-terra/issues/4361"""
        expected = "\n".join(
            [
                "          ┌──────┐ ",
                "q_0: |0>──┤ my h ├─",
                "          └──┬───┘ ",
                "q_1: |0>─────■─────",
                "         my ctrl-h ",
                "             ║     ",
                "   c: 0 ═════■═════",
                "            0x1    ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        hgate = HGate(label="my h")
        with self.assertWarns(DeprecationWarning):
            controlh = hgate.control(label="my ctrl-h").c_if(cr, 1)
        circ.append(controlh, [1, 0])

        self.assertEqual(
            str(
                circuit_drawer(
                    circ,
                    output="text",
                    initial_state=True,
                    cregbundle=False,
                    vertical_compression="medium",
                )
            ),
            expected,
        )

    def test_control_gate_label_with_cond_2_med_cregbundle(self):
        """Control gate has a label and a conditional (on label, compression=med) with cregbundle
        See https://github.com/Qiskit/qiskit-terra/issues/4361"""
        expected = "\n".join(
            [
                "        ┌──────┐",
                "q_0: |0>┤ my h ├",
                "        └──┬───┘",
                "q_1: |0>───■────",
                "         my ch  ",
                "        ┌──╨──┐ ",
                " c: 0 1/╡ 0x1 ╞═",
                "        └─────┘ ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        hgate = HGate(label="my h")
        with self.assertWarns(DeprecationWarning):
            controlh = hgate.control(label="my ch").c_if(cr, 1)
        circ.append(controlh, [1, 0])

        self.assertEqual(
            str(
                circuit_drawer(
                    circ,
                    output="text",
                    initial_state=True,
                    cregbundle=True,
                    vertical_compression="medium",
                )
            ),
            expected,
        )

    def test_control_gate_label_with_cond_2_low(self):
        """Control gate has a label and a conditional (on label, compression=low)
        See https://github.com/Qiskit/qiskit-terra/issues/4361"""
        expected = "\n".join(
            [
                "        ┌──────┐",
                "q_0: |0>┤ my h ├",
                "        └──┬───┘",
                "           │    ",
                "q_1: |0>───■────",
                "         my ch  ",
                "           ║    ",
                "   c: 0 ═══■════",
                "          0x1   ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        hgate = HGate(label="my h")
        with self.assertWarns(DeprecationWarning):
            controlh = hgate.control(label="my ch").c_if(cr, 1)
        circ.append(controlh, [1, 0])

        self.assertEqual(
            str(
                circuit_drawer(
                    circ,
                    output="text",
                    initial_state=True,
                    cregbundle=False,
                    vertical_compression="low",
                )
            ),
            expected,
        )

    def test_control_gate_label_with_cond_2_low_cregbundle(self):
        """Control gate has a label and a conditional (on label, compression=low) with cregbundle
        See https://github.com/Qiskit/qiskit-terra/issues/4361"""
        expected = "\n".join(
            [
                "        ┌──────┐",
                "q_0: |0>┤ my h ├",
                "        └──┬───┘",
                "           │    ",
                "q_1: |0>───■────",
                "         my ch  ",
                "        ┌──╨──┐ ",
                " c: 0 1/╡ 0x1 ╞═",
                "        └─────┘ ",
            ]
        )

        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        hgate = HGate(label="my h")
        with self.assertWarns(DeprecationWarning):
            controlh = hgate.control(label="my ch").c_if(cr, 1)
        circ.append(controlh, [1, 0])

        self.assertEqual(
            str(
                circuit_drawer(
                    circ,
                    output="text",
                    initial_state=True,
                    cregbundle=True,
                    vertical_compression="low",
                )
            ),
            expected,
        )


class TestTextDrawerParams(QiskitTestCase):
    """Test drawing parameters."""

    def test_text_no_parameters(self):
        """Test drawing with no parameters"""
        expected = "\n".join(
            [
                "      ┌───┐",
                "q: |0>┤ X ├",
                "      └───┘",
            ]
        )

        qr = QuantumRegister(1, "q")
        circuit = QuantumCircuit(qr)
        circuit.x(0)
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_parameters_mix(self):
        """cu3 drawing with parameters"""
        expected = "\n".join(
            [
                "                            ",
                "q_0: |0>─────────■──────────",
                "        ┌────────┴─────────┐",
                "q_1: |0>┤ U(π/2,theta,π,0) ├",
                "        └──────────────────┘",
            ]
        )

        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.cu(pi / 2, Parameter("theta"), pi, 0, qr[0], qr[1])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_bound_parameters(self):
        """Bound parameters
        See: https://github.com/Qiskit/qiskit-terra/pull/3876"""
        # fmt: off
        expected = "\n".join(["       ┌────────────┐",
                              "qr: |0>┤ my_u2(π,π) ├",
                              "       └────────────┘"])
        # fmt: on
        my_u2_circuit = QuantumCircuit(1, name="my_u2")
        phi = Parameter("phi")
        lam = Parameter("lambda")
        my_u2_circuit.u(3.141592653589793, phi, lam, 0)
        my_u2 = my_u2_circuit.to_gate()
        qr = QuantumRegister(1, name="qr")
        circuit = QuantumCircuit(qr, name="circuit")
        circuit.append(my_u2, [qr[0]])
        circuit = circuit.assign_parameters({phi: 3.141592653589793, lam: 3.141592653589793})

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_pi_param_expr(self):
        """Text pi in circuit with parameter expression."""
        expected = "\n".join(
            [
                "   ┌─────────────────────┐",
                "q: ┤ Rx((π - x)*(π - y)) ├",
                "   └─────────────────────┘",
            ]
        )

        x, y = Parameter("x"), Parameter("y")
        circuit = QuantumCircuit(1)
        circuit.rx((pi - x) * (pi - y), 0)
        self.assertEqual(circuit.draw(output="text").single_string(), expected)

    def test_text_utf8(self):
        """Test that utf8 characters work in windows CI env."""
        # fmt: off
        expected = "\n".join(["   ┌──────────┐",
                              "q: ┤ U(0,φ,λ) ├",
                              "   └──────────┘"])
        # fmt: on
        phi, lam = Parameter("φ"), Parameter("λ")
        circuit = QuantumCircuit(1)
        circuit.u(0, phi, lam, 0)
        self.assertEqual(circuit.draw(output="text").single_string(), expected)

    def test_text_ndarray_parameters(self):
        """Test that if params are type ndarray, params are not displayed."""
        # fmt: off
        expected = "\n".join(["      ┌─────────┐",
                              "q: |0>┤ Unitary ├",
                              "      └─────────┘"])
        # fmt: on
        qr = QuantumRegister(1, "q")
        circuit = QuantumCircuit(qr)
        circuit.unitary(numpy.array([[0, 1], [1, 0]]), 0)
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_qc_parameters(self):
        """Test that if params are type QuantumCircuit, params are not displayed."""
        expected = "\n".join(
            [
                "        ┌───────┐",
                "q_0: |0>┤0      ├",
                "        │  name │",
                "q_1: |0>┤1      ├",
                "        └───────┘",
            ]
        )

        my_qc_param = QuantumCircuit(2)
        my_qc_param.h(0)
        my_qc_param.cx(0, 1)
        inst = Instruction("name", 2, 0, [my_qc_param])
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(inst, [0, 1])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)


class TestTextDrawerVerticalCompressionLow(QiskitTestCase):
    """Test vertical_compression='low'"""

    def test_text_conditional_1(self):
        """Conditional drawing with 1-bit-length regs."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c0[1];
        creg c1[1];
        if(c0==1) x q[0];
        if(c1==1) x q[0];
        """
        expected = "\n".join(
            [
                "      ┌───┐┌───┐",
                "q: |0>┤ X ├┤ X ├",
                "      └─╥─┘└─╥─┘",
                "        ║    ║  ",
                "c0: 0 ══■════╬══",
                "       0x1   ║  ",
                "             ║  ",
                "c1: 0 ═══════■══",
                "            0x1 ",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    cregbundle=False,
                    vertical_compression="low",
                )
            ),
            expected,
        )

    def test_text_conditional_1_bundle(self):
        """Conditional drawing with 1-bit-length regs."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c0[1];
        creg c1[1];
        if(c0==1) x q[0];
        if(c1==1) x q[0];
        """
        expected = "\n".join(
            [
                "         ┌───┐  ┌───┐ ",
                "  q: |0>─┤ X ├──┤ X ├─",
                "         └─╥─┘  └─╥─┘ ",
                "        ┌──╨──┐   ║   ",
                "c0: 0 1/╡ 0x1 ╞═══╬═══",
                "        └─────┘   ║   ",
                "               ┌──╨──┐",
                "c1: 0 1/═══════╡ 0x1 ╞",
                "               └─────┘",
            ]
        )

        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    vertical_compression="low",
                    cregbundle=True,
                )
            ),
            expected,
        )

    def test_text_conditional_reverse_bits_true(self):
        """Conditional drawing with 1-bit-length regs."""
        cr = ClassicalRegister(2, "cr")
        cr2 = ClassicalRegister(1, "cr2")
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr, cr, cr2)
        circuit.h(0)
        circuit.h(1)
        circuit.h(2)
        circuit.x(0)
        circuit.x(0)
        circuit.measure(2, 1)
        with self.assertWarns(DeprecationWarning):
            circuit.x(2).c_if(cr, 2)

        expected = "\n".join(
            [
                "         ┌───┐     ┌─┐     ┌───┐",
                "qr_2: |0>┤ H ├─────┤M├─────┤ X ├",
                "         └───┘     └╥┘     └─╥─┘",
                "         ┌───┐      ║        ║  ",
                "qr_1: |0>┤ H ├──────╫────────╫──",
                "         └───┘      ║        ║  ",
                "         ┌───┐┌───┐ ║ ┌───┐  ║  ",
                "qr_0: |0>┤ H ├┤ X ├─╫─┤ X ├──╫──",
                "         └───┘└───┘ ║ └───┘  ║  ",
                "                    ║        ║  ",
                "  cr2: 0 ═══════════╬════════╬══",
                "                    ║        ║  ",
                "                    ║        ║  ",
                " cr_1: 0 ═══════════╩════════■══",
                "                             ║  ",
                "                             ║  ",
                " cr_0: 0 ════════════════════o══",
                "                            0x2 ",
            ]
        )

        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    cregbundle=False,
                    reverse_bits=True,
                    vertical_compression="low",
                )
            ),
            expected,
        )

    def test_text_conditional_reverse_bits_false(self):
        """Conditional drawing with 1-bit-length regs."""
        cr = ClassicalRegister(2, "cr")
        cr2 = ClassicalRegister(1, "cr2")
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr, cr, cr2)
        circuit.h(0)
        circuit.h(1)
        circuit.h(2)
        circuit.x(0)
        circuit.x(0)
        circuit.measure(2, 1)
        with self.assertWarns(DeprecationWarning):
            circuit.x(2).c_if(cr, 2)

        expected = "\n".join(
            [
                "         ┌───┐┌───┐┌───┐",
                "qr_0: |0>┤ H ├┤ X ├┤ X ├",
                "         └───┘└───┘└───┘",
                "         ┌───┐          ",
                "qr_1: |0>┤ H ├──────────",
                "         └───┘          ",
                "         ┌───┐ ┌─┐ ┌───┐",
                "qr_2: |0>┤ H ├─┤M├─┤ X ├",
                "         └───┘ └╥┘ └─╥─┘",
                "                ║    ║  ",
                " cr_0: 0 ═══════╬════o══",
                "                ║    ║  ",
                "                ║    ║  ",
                " cr_1: 0 ═══════╩════■══",
                "                    0x2 ",
                "                        ",
                "  cr2: 0 ═══════════════",
                "                        ",
            ]
        )

        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    vertical_compression="low",
                    cregbundle=False,
                    reverse_bits=False,
                )
            ),
            expected,
        )

    def test_text_justify_right(self):
        """Drawing with right justify"""
        expected = "\n".join(
            [
                "              ┌───┐",
                "q1_0: |0>─────┤ X ├",
                "              └───┘",
                "         ┌───┐ ┌─┐ ",
                "q1_1: |0>┤ H ├─┤M├─",
                "         └───┘ └╥┘ ",
                "                ║  ",
                " c1: 0 2/═══════╩══",
                "                1  ",
            ]
        )

        qr1 = QuantumRegister(2, "q1")
        cr1 = ClassicalRegister(2, "c1")
        circuit = QuantumCircuit(qr1, cr1)
        circuit.x(qr1[0])
        circuit.h(qr1[1])
        circuit.measure(qr1[1], cr1[1])
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    justify="right",
                    vertical_compression="low",
                )
            ),
            expected,
        )


class TestTextDrawerVerticalCompressionMedium(QiskitTestCase):
    """Test vertical_compression='medium'"""

    def test_text_conditional_1(self):
        """Medium vertical compression avoids box overlap."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c0[1];
        creg c1[1];
        if(c0==1) x q[0];
        if(c1==1) x q[0];
        """
        expected = "\n".join(
            [
                "      ┌───┐┌───┐",
                "q: |0>┤ X ├┤ X ├",
                "      └─╥─┘└─╥─┘",
                "c0: 0 ══■════╬══",
                "       0x1   ║  ",
                "c1: 0 ═══════■══",
                "            0x1 ",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    cregbundle=False,
                    vertical_compression="medium",
                )
            ),
            expected,
        )

    def test_text_conditional_1_bundle(self):
        """Medium vertical compression avoids box overlap."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c0[1];
        creg c1[1];
        if(c0==1) x q[0];
        if(c1==1) x q[0];
        """
        expected = "\n".join(
            [
                "         ┌───┐  ┌───┐ ",
                "  q: |0>─┤ X ├──┤ X ├─",
                "         └─╥─┘  └─╥─┘ ",
                "        ┌──╨──┐   ║   ",
                "c0: 0 1/╡ 0x1 ╞═══╬═══",
                "        └─────┘┌──╨──┐",
                "c1: 0 1/═══════╡ 0x1 ╞",
                "               └─────┘",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    vertical_compression="medium",
                    cregbundle=True,
                )
            ),
            expected,
        )

    def test_text_measure_with_spaces(self):
        """Measure wire might have extra spaces
        Found while reproducing
        https://quantumcomputing.stackexchange.com/q/10194/1859"""
        qasm_string = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[3];
            measure q[0] -> c[1];
            if(c==1) x q[1];
        """
        expected = "\n".join(
            [
                "        ┌─┐     ",
                "q_0: |0>┤M├─────",
                "        └╥┘┌───┐",
                "q_1: |0>─╫─┤ X ├",
                "         ║ └─╥─┘",
                " c_0: 0 ═╬═══■══",
                "         ║   ║  ",
                " c_1: 0 ═╩═══o══",
                "             ║  ",
                " c_2: 0 ═════o══",
                "            0x1 ",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    cregbundle=False,
                    vertical_compression="medium",
                )
            ),
            expected,
        )

    def test_text_measure_with_spaces_bundle(self):
        """Measure wire might have extra spaces
        Found while reproducing
        https://quantumcomputing.stackexchange.com/q/10194/1859"""
        qasm_string = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[3];
            measure q[0] -> c[1];
            if(c==1) x q[1];
        """
        expected = "\n".join(
            [
                "        ┌─┐       ",
                "q_0: |0>┤M├───────",
                "        └╥┘ ┌───┐ ",
                "q_1: |0>─╫──┤ X ├─",
                "         ║  └─╥─┘ ",
                "         ║ ┌──╨──┐",
                " c: 0 3/═╩═╡ 0x1 ╞",
                "         1 └─────┘",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    vertical_compression="medium",
                    cregbundle=True,
                )
            ),
            expected,
        )

    def test_text_barrier_med_compress_1(self):
        """Medium vertical compression avoids connection break."""
        circuit = QuantumCircuit(4)
        circuit.cx(1, 3)
        circuit.x(1)
        circuit.barrier((2, 3), label="Bar 1")

        expected = "\n".join(
            [
                "                    ",
                "q_0: |0>────────────",
                "              ┌───┐ ",
                "q_1: |0>──■───┤ X ├─",
                "          │   └───┘ ",
                "          │   Bar 1 ",
                "q_2: |0>──┼─────░───",
                "        ┌─┴─┐   ░   ",
                "q_3: |0>┤ X ├───░───",
                "        └───┘   ░   ",
            ]
        )

        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    vertical_compression="medium",
                    cregbundle=False,
                )
            ),
            expected,
        )

    def test_text_barrier_med_compress_2(self):
        """Medium vertical compression avoids overprint."""
        circuit = QuantumCircuit(4)
        circuit.barrier((0, 1, 2), label="a")
        circuit.cx(1, 3)
        circuit.x(1)
        circuit.barrier((2, 3), label="Bar 1")

        expected = "\n".join(
            [
                "         a             ",
                "q_0: |0>─░─────────────",
                "         ░       ┌───┐ ",
                "q_1: |0>─░───■───┤ X ├─",
                "         ░   │   └───┘ ",
                "         ░   │   Bar 1 ",
                "q_2: |0>─░───┼─────░───",
                "         ░ ┌─┴─┐   ░   ",
                "q_3: |0>───┤ X ├───░───",
                "           └───┘   ░   ",
            ]
        )

        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    vertical_compression="medium",
                    cregbundle=False,
                )
            ),
            expected,
        )

    def test_text_barrier_med_compress_3(self):
        """Medium vertical compression avoids conditional connection break."""
        qr = QuantumRegister(1, "qr")
        qc1 = ClassicalRegister(3, "cr")
        qc2 = ClassicalRegister(1, "cr2")
        circuit = QuantumCircuit(qr, qc1, qc2)
        with self.assertWarns(DeprecationWarning):
            circuit.x(0).c_if(qc1, 3)
        with self.assertWarns(DeprecationWarning):
            circuit.x(0).c_if(qc2[0], 1)

        expected = "\n".join(
            [
                "        ┌───┐┌───┐",
                " qr: |0>┤ X ├┤ X ├",
                "        └─╥─┘└─╥─┘",
                "cr_0: 0 ══■════╬══",
                "          ║    ║  ",
                "cr_2: 0 ══o════╬══",
                "          ║    ║  ",
                " cr2: 0 ══╬════■══",
                "          ║       ",
                "cr_1: 0 ══■═══════",
                "         0x3      ",
            ]
        )

        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    vertical_compression="medium",
                    wire_order=[0, 1, 3, 4, 2],
                    cregbundle=False,
                )
            ),
            expected,
        )


class TestTextConditional(QiskitTestCase):
    """Gates with conditionals"""

    def test_text_conditional_1_cregbundle(self):
        """Conditional drawing with 1-bit-length regs and cregbundle."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c0[1];
        creg c1[1];
        if(c0==1) x q[0];
        if(c1==1) x q[0];
        """
        expected = "\n".join(
            [
                "         ┌───┐  ┌───┐ ",
                "  q: |0>─┤ X ├──┤ X ├─",
                "        ┌┴─╨─┴┐ └─╥─┘ ",
                "c0: 0 1/╡ 0x1 ╞═══╬═══",
                "        └─────┘┌──╨──┐",
                "c1: 0 1/═══════╡ 0x1 ╞",
                "               └─────┘",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    cregbundle=True,
                    vertical_compression="high",
                )
            ),
            expected,
        )

    def test_text_conditional_1(self):
        """Conditional drawing with 1-bit-length regs."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c0[1];
        creg c1[1];
        if(c0==1) x q[0];
        if(c1==1) x q[0];
        """
        expected = "\n".join(
            [
                "      ┌───┐┌───┐",
                "q: |0>┤ X ├┤ X ├",
                "      └─╥─┘└─╥─┘",
                "c0: 0 ══■════╬══",
                "       0x1   ║  ",
                "c1: 0 ═══════■══",
                "            0x1 ",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_2_cregbundle(self):
        """Conditional drawing with 2-bit-length regs with cregbundle"""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c0[2];
        creg c1[2];
        if(c0==2) x q[0];
        if(c1==2) x q[0];
        """
        expected = "\n".join(
            [
                "         ┌───┐  ┌───┐ ",
                "  q: |0>─┤ X ├──┤ X ├─",
                "        ┌┴─╨─┴┐ └─╥─┘ ",
                "c0: 0 2/╡ 0x2 ╞═══╬═══",
                "        └─────┘┌──╨──┐",
                "c1: 0 2/═══════╡ 0x2 ╞",
                "               └─────┘",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    cregbundle=True,
                    vertical_compression="high",
                )
            ),
            expected,
        )

    def test_text_conditional_2(self):
        """Conditional drawing with 2-bit-length regs."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c0[2];
        creg c1[2];
        if(c0==2) x q[0];
        if(c1==2) x q[0];
        """
        expected = "\n".join(
            [
                "        ┌───┐┌───┐",
                "  q: |0>┤ X ├┤ X ├",
                "        └─╥─┘└─╥─┘",
                "c0_0: 0 ══o════╬══",
                "          ║    ║  ",
                "c0_1: 0 ══■════╬══",
                "         0x2   ║  ",
                "c1_0: 0 ═══════o══",
                "               ║  ",
                "c1_1: 0 ═══════■══",
                "              0x2 ",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_3_cregbundle(self):
        """Conditional drawing with 3-bit-length regs with cregbundle."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c0[3];
        creg c1[3];
        if(c0==3) x q[0];
        if(c1==3) x q[0];
        """
        expected = "\n".join(
            [
                "         ┌───┐  ┌───┐ ",
                "  q: |0>─┤ X ├──┤ X ├─",
                "        ┌┴─╨─┴┐ └─╥─┘ ",
                "c0: 0 3/╡ 0x3 ╞═══╬═══",
                "        └─────┘┌──╨──┐",
                "c1: 0 3/═══════╡ 0x3 ╞",
                "               └─────┘",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    cregbundle=True,
                    vertical_compression="high",
                )
            ),
            expected,
        )

    def test_text_conditional_3(self):
        """Conditional drawing with 3-bit-length regs."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c0[3];
        creg c1[3];
        if(c0==3) x q[0];
        if(c1==3) x q[0];
        """
        expected = "\n".join(
            [
                "        ┌───┐┌───┐",
                "  q: |0>┤ X ├┤ X ├",
                "        └─╥─┘└─╥─┘",
                "c0_0: 0 ══■════╬══",
                "          ║    ║  ",
                "c0_1: 0 ══■════╬══",
                "          ║    ║  ",
                "c0_2: 0 ══o════╬══",
                "         0x3   ║  ",
                "c1_0: 0 ═══════■══",
                "               ║  ",
                "c1_1: 0 ═══════■══",
                "               ║  ",
                "c1_2: 0 ═══════o══",
                "              0x3 ",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_4(self):
        """Conditional drawing with 4-bit-length regs."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c0[4];
        creg c1[4];
        if(c0==4) x q[0];
        if(c1==4) x q[0];
        """
        expected = "\n".join(
            [
                "         ┌───┐  ┌───┐ ",
                "  q: |0>─┤ X ├──┤ X ├─",
                "        ┌┴─╨─┴┐ └─╥─┘ ",
                "c0: 0 4/╡ 0x4 ╞═══╬═══",
                "        └─────┘┌──╨──┐",
                "c1: 0 4/═══════╡ 0x4 ╞",
                "               └─────┘",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit, output="text", initial_state=True, vertical_compression="high"
                )
            ),
            expected,
        )

    def test_text_conditional_5(self):
        """Conditional drawing with 5-bit-length regs."""
        qasm_string = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c0[5];
        creg c1[5];
        if(c0==5) x q[0];
        if(c1==5) x q[0];
        """
        expected = "\n".join(
            [
                "        ┌───┐┌───┐",
                "  q: |0>┤ X ├┤ X ├",
                "        └─╥─┘└─╥─┘",
                "c0_0: 0 ══■════╬══",
                "          ║    ║  ",
                "c0_1: 0 ══o════╬══",
                "          ║    ║  ",
                "c0_2: 0 ══■════╬══",
                "          ║    ║  ",
                "c0_3: 0 ══o════╬══",
                "          ║    ║  ",
                "c0_4: 0 ══o════╬══",
                "         0x5   ║  ",
                "c1_0: 0 ═══════■══",
                "               ║  ",
                "c1_1: 0 ═══════o══",
                "               ║  ",
                "c1_2: 0 ═══════■══",
                "               ║  ",
                "c1_3: 0 ═══════o══",
                "               ║  ",
                "c1_4: 0 ═══════o══",
                "              0x5 ",
            ]
        )
        with self.assertWarns(DeprecationWarning):
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_cz_no_space_cregbundle(self):
        """Conditional CZ without space"""
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.cz(qr[0], qr[1]).c_if(cr, 1)

        expected = "\n".join(
            [
                "                ",
                "qr_0: |0>───■───",
                "            │   ",
                "qr_1: |0>───■───",
                "         ┌──╨──┐",
                " cr: 0 1/╡ 0x1 ╞",
                "         └─────┘",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_text_conditional_cz_no_space(self):
        """Conditional CZ without space"""
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.cz(qr[0], qr[1]).c_if(cr, 1)

        expected = "\n".join(
            [
                "              ",
                "qr_0: |0>──■──",
                "           │  ",
                "qr_1: |0>──■──",
                "           ║  ",
                "   cr: 0 ══■══",
                "          0x1 ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_cz_cregbundle(self):
        """Conditional CZ with a wire in the middle"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.cz(qr[0], qr[1]).c_if(cr, 1)

        expected = "\n".join(
            [
                "                ",
                "qr_0: |0>───■───",
                "            │   ",
                "qr_1: |0>───■───",
                "            ║   ",
                "qr_2: |0>───╫───",
                "         ┌──╨──┐",
                " cr: 0 1/╡ 0x1 ╞",
                "         └─────┘",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_text_conditional_cz(self):
        """Conditional CZ with a wire in the middle"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.cz(qr[0], qr[1]).c_if(cr, 1)

        expected = "\n".join(
            [
                "              ",
                "qr_0: |0>──■──",
                "           │  ",
                "qr_1: |0>──■──",
                "           ║  ",
                "qr_2: |0>──╫──",
                "           ║  ",
                "   cr: 0 ══■══",
                "          0x1 ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_cx_ct_cregbundle(self):
        """Conditional CX (control-target) with a wire in the middle"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.cx(qr[0], qr[1]).c_if(cr, 1)

        expected = "\n".join(
            [
                "                ",
                "qr_0: |0>───■───",
                "          ┌─┴─┐ ",
                "qr_1: |0>─┤ X ├─",
                "          └─╥─┘ ",
                "qr_2: |0>───╫───",
                "         ┌──╨──┐",
                " cr: 0 1/╡ 0x1 ╞",
                "         └─────┘",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_text_conditional_cx_ct(self):
        """Conditional CX (control-target) with a wire in the middle"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.cx(qr[0], qr[1]).c_if(cr, 1)

        expected = "\n".join(
            [
                "              ",
                "qr_0: |0>──■──",
                "         ┌─┴─┐",
                "qr_1: |0>┤ X ├",
                "         └─╥─┘",
                "qr_2: |0>──╫──",
                "           ║  ",
                "   cr: 0 ══■══",
                "          0x1 ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_cx_tc_cregbundle(self):
        """Conditional CX (target-control) with a wire in the middle with cregbundle."""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.cx(qr[1], qr[0]).c_if(cr, 1)

        expected = "\n".join(
            [
                "          ┌───┐ ",
                "qr_0: |0>─┤ X ├─",
                "          └─┬─┘ ",
                "qr_1: |0>───■───",
                "            ║   ",
                "qr_2: |0>───╫───",
                "         ┌──╨──┐",
                " cr: 0 1/╡ 0x1 ╞",
                "         └─────┘",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_text_conditional_cx_tc(self):
        """Conditional CX (target-control) with a wire in the middle"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.cx(qr[1], qr[0]).c_if(cr, 1)

        expected = "\n".join(
            [
                "         ┌───┐",
                "qr_0: |0>┤ X ├",
                "         └─┬─┘",
                "qr_1: |0>──■──",
                "           ║  ",
                "qr_2: |0>──╫──",
                "           ║  ",
                "   cr: 0 ══■══",
                "          0x1 ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_cu3_ct_cregbundle(self):
        """Conditional Cu3 (control-target) with a wire in the middle with cregbundle"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.append(CU3Gate(pi / 2, pi / 2, pi / 2), [qr[0], qr[1]]).c_if(cr, 1)

        expected = "\n".join(
            [
                "                            ",
                "qr_0: |0>─────────■─────────",
                "         ┌────────┴────────┐",
                "qr_1: |0>┤ U3(π/2,π/2,π/2) ├",
                "         └────────╥────────┘",
                "qr_2: |0>─────────╫─────────",
                "               ┌──╨──┐      ",
                " cr: 0 1/══════╡ 0x1 ╞══════",
                "               └─────┘      ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_text_conditional_cu3_ct(self):
        """Conditional Cu3 (control-target) with a wire in the middle"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.append(CU3Gate(pi / 2, pi / 2, pi / 2), [qr[0], qr[1]]).c_if(cr, 1)

        expected = "\n".join(
            [
                "                            ",
                "qr_0: |0>─────────■─────────",
                "         ┌────────┴────────┐",
                "qr_1: |0>┤ U3(π/2,π/2,π/2) ├",
                "         └────────╥────────┘",
                "qr_2: |0>─────────╫─────────",
                "                  ║         ",
                "   cr: 0 ═════════■═════════",
                "                 0x1        ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_cu3_tc_cregbundle(self):
        """Conditional Cu3 (target-control) with a wire in the middle with cregbundle"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.append(CU3Gate(pi / 2, pi / 2, pi / 2), [qr[1], qr[0]]).c_if(cr, 1)

        expected = "\n".join(
            [
                "         ┌─────────────────┐",
                "qr_0: |0>┤ U3(π/2,π/2,π/2) ├",
                "         └────────┬────────┘",
                "qr_1: |0>─────────■─────────",
                "                  ║         ",
                "qr_2: |0>─────────╫─────────",
                "               ┌──╨──┐      ",
                " cr: 0 1/══════╡ 0x1 ╞══════",
                "               └─────┘      ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_text_conditional_cu3_tc(self):
        """Conditional Cu3 (target-control) with a wire in the middle"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.append(CU3Gate(pi / 2, pi / 2, pi / 2), [qr[1], qr[0]]).c_if(cr, 1)

        expected = "\n".join(
            [
                "         ┌─────────────────┐",
                "qr_0: |0>┤ U3(π/2,π/2,π/2) ├",
                "         └────────┬────────┘",
                "qr_1: |0>─────────■─────────",
                "                  ║         ",
                "qr_2: |0>─────────╫─────────",
                "                  ║         ",
                "   cr: 0 ═════════■═════════",
                "                 0x1        ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_ccx_cregbundle(self):
        """Conditional CCX with a wire in the middle with cregbundle"""
        qr = QuantumRegister(4, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.ccx(qr[0], qr[1], qr[2]).c_if(cr, 1)

        expected = "\n".join(
            [
                "                ",
                "qr_0: |0>───■───",
                "            │   ",
                "qr_1: |0>───■───",
                "          ┌─┴─┐ ",
                "qr_2: |0>─┤ X ├─",
                "          └─╥─┘ ",
                "qr_3: |0>───╫───",
                "         ┌──╨──┐",
                " cr: 0 1/╡ 0x1 ╞",
                "         └─────┘",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_text_conditional_ccx(self):
        """Conditional CCX with a wire in the middle"""
        qr = QuantumRegister(4, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.ccx(qr[0], qr[1], qr[2]).c_if(cr, 1)

        expected = "\n".join(
            [
                "              ",
                "qr_0: |0>──■──",
                "           │  ",
                "qr_1: |0>──■──",
                "         ┌─┴─┐",
                "qr_2: |0>┤ X ├",
                "         └─╥─┘",
                "qr_3: |0>──╫──",
                "           ║  ",
                "   cr: 0 ══■══",
                "          0x1 ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_ccx_no_space_cregbundle(self):
        """Conditional CCX without space with cregbundle"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.ccx(qr[0], qr[1], qr[2]).c_if(cr, 1)

        expected = "\n".join(
            [
                "                ",
                "qr_0: |0>───■───",
                "            │   ",
                "qr_1: |0>───■───",
                "          ┌─┴─┐ ",
                "qr_2: |0>─┤ X ├─",
                "         ┌┴─╨─┴┐",
                " cr: 0 1/╡ 0x1 ╞",
                "         └─────┘",
            ]
        )

        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    cregbundle=True,
                    vertical_compression="high",
                )
            ),
            expected,
        )

    def test_text_conditional_ccx_no_space(self):
        """Conditional CCX without space"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.ccx(qr[0], qr[1], qr[2]).c_if(cr, 1)

        expected = "\n".join(
            [
                "              ",
                "qr_0: |0>──■──",
                "           │  ",
                "qr_1: |0>──■──",
                "         ┌─┴─┐",
                "qr_2: |0>┤ X ├",
                "         └─╥─┘",
                "   cr: 0 ══■══",
                "          0x1 ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_h_cregbundle(self):
        """Conditional H with a wire in the middle with cregbundle"""
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[0]).c_if(cr, 1)

        expected = "\n".join(
            [
                "          ┌───┐ ",
                "qr_0: |0>─┤ H ├─",
                "          └─╥─┘ ",
                "qr_1: |0>───╫───",
                "         ┌──╨──┐",
                " cr: 0 1/╡ 0x1 ╞",
                "         └─────┘",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_text_conditional_h(self):
        """Conditional H with a wire in the middle"""
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[0]).c_if(cr, 1)

        expected = "\n".join(
            [
                "         ┌───┐",
                "qr_0: |0>┤ H ├",
                "         └─╥─┘",
                "qr_1: |0>──╫──",
                "           ║  ",
                "   cr: 0 ══■══",
                "          0x1 ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_swap_cregbundle(self):
        """Conditional SWAP with cregbundle"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.swap(qr[0], qr[1]).c_if(cr, 1)

        expected = "\n".join(
            [
                "                ",
                "qr_0: |0>───X───",
                "            │   ",
                "qr_1: |0>───X───",
                "            ║   ",
                "qr_2: |0>───╫───",
                "         ┌──╨──┐",
                " cr: 0 1/╡ 0x1 ╞",
                "         └─────┘",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_text_conditional_swap(self):
        """Conditional SWAP"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.swap(qr[0], qr[1]).c_if(cr, 1)

        expected = "\n".join(
            [
                "              ",
                "qr_0: |0>──X──",
                "           │  ",
                "qr_1: |0>──X──",
                "           ║  ",
                "qr_2: |0>──╫──",
                "           ║  ",
                "   cr: 0 ══■══",
                "          0x1 ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_conditional_cswap_cregbundle(self):
        """Conditional CSwap with cregbundle"""
        qr = QuantumRegister(4, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.cswap(qr[0], qr[1], qr[2]).c_if(cr, 1)

        expected = "\n".join(
            [
                "                ",
                "qr_0: |0>───■───",
                "            │   ",
                "qr_1: |0>───X───",
                "            │   ",
                "qr_2: |0>───X───",
                "            ║   ",
                "qr_3: |0>───╫───",
                "         ┌──╨──┐",
                " cr: 0 1/╡ 0x1 ╞",
                "         └─────┘",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_text_conditional_cswap(self):
        """Conditional CSwap"""
        qr = QuantumRegister(4, "qr")
        cr = ClassicalRegister(1, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.cswap(qr[0], qr[1], qr[2]).c_if(cr, 1)

        expected = "\n".join(
            [
                "              ",
                "qr_0: |0>──■──",
                "           │  ",
                "qr_1: |0>──X──",
                "           │  ",
                "qr_2: |0>──X──",
                "           ║  ",
                "qr_3: |0>──╫──",
                "           ║  ",
                "   cr: 0 ══■══",
                "          0x1 ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_conditional_reset_cregbundle(self):
        """Reset drawing with cregbundle."""
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")

        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.reset(qr[0]).c_if(cr, 1)

        expected = "\n".join(
            [
                "                ",
                "qr_0: |0>──|0>──",
                "            ║   ",
                "qr_1: |0>───╫───",
                "         ┌──╨──┐",
                " cr: 0 1/╡ 0x1 ╞",
                "         └─────┘",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
            expected,
        )

    def test_conditional_reset(self):
        """Reset drawing."""
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")

        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.reset(qr[0]).c_if(cr, 1)

        expected = "\n".join(
            [
                "              ",
                "qr_0: |0>─|0>─",
                "           ║  ",
                "qr_1: |0>──╫──",
                "           ║  ",
                "   cr: 0 ══■══",
                "          0x1 ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_conditional_multiplexer_cregbundle(self):
        """Test Multiplexer with cregbundle."""
        cx_multiplexer = UCGate([numpy.eye(2), numpy.array([[0, 1], [1, 0]])])
        qr = QuantumRegister(3, name="qr")
        cr = ClassicalRegister(1, "cr")
        qc = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            qc.append(cx_multiplexer.c_if(cr, 1), [qr[0], qr[1]])

        expected = "\n".join(
            [
                "         ┌──────────────┐",
                "qr_0: |0>┤0             ├",
                "         │  Multiplexer │",
                "qr_1: |0>┤1             ├",
                "         └──────╥───────┘",
                "qr_2: |0>───────╫────────",
                "             ┌──╨──┐     ",
                " cr: 0 1/════╡ 0x1 ╞═════",
                "             └─────┘     ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(qc, output="text", initial_state=True, cregbundle=True)), expected
        )

    def test_conditional_multiplexer(self):
        """Test Multiplexer."""
        cx_multiplexer = UCGate([numpy.eye(2), numpy.array([[0, 1], [1, 0]])])
        qr = QuantumRegister(3, name="qr")
        cr = ClassicalRegister(1, "cr")
        qc = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            qc.append(cx_multiplexer.c_if(cr, 1), [qr[0], qr[1]])

        expected = "\n".join(
            [
                "         ┌──────────────┐",
                "qr_0: |0>┤0             ├",
                "         │  Multiplexer │",
                "qr_1: |0>┤1             ├",
                "         └──────╥───────┘",
                "qr_2: |0>───────╫────────",
                "                ║        ",
                "   cr: 0 ═══════■════════",
                "               0x1       ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(qc, output="text", initial_state=True, cregbundle=False)), expected
        )

    def test_text_conditional_measure_cregbundle(self):
        """Conditional with measure on same clbit with cregbundle"""
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[0])
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[1]).c_if(cr, 1)

        expected = "\n".join(
            [
                "         ┌───┐┌─┐       ",
                "qr_0: |0>┤ H ├┤M├───────",
                "         └───┘└╥┘ ┌───┐ ",
                "qr_1: |0>──────╫──┤ H ├─",
                "               ║ ┌┴─╨─┴┐",
                " cr: 0 2/══════╩═╡ 0x1 ╞",
                "               0 └─────┘",
            ]
        )

        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    cregbundle=True,
                    vertical_compression="high",
                )
            ),
            expected,
        )

    def test_text_conditional_measure(self):
        """Conditional with measure on same clbit"""
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[0])
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[1]).c_if(cr, 1)

        expected = "\n".join(
            [
                "         ┌───┐┌─┐     ",
                "qr_0: |0>┤ H ├┤M├─────",
                "         └───┘└╥┘┌───┐",
                "qr_1: |0>──────╫─┤ H ├",
                "               ║ └─╥─┘",
                " cr_0: 0 ══════╩═══■══",
                "                   ║  ",
                " cr_1: 0 ══════════o══",
                "                  0x1 ",
            ]
        )
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_bit_conditional(self):
        """Test bit conditions on gates"""

        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[0]).c_if(cr[0], 1)
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[1]).c_if(cr[1], 0)

        expected = "\n".join(
            [
                "         ┌───┐     ",
                "qr_0: |0>┤ H ├─────",
                "         └─╥─┘┌───┐",
                "qr_1: |0>──╫──┤ H ├",
                "           ║  └─╥─┘",
                " cr_0: 0 ══■════╬══",
                "                ║  ",
                " cr_1: 0 ═══════o══",
                "                   ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=False)),
            expected,
        )

    def test_text_bit_conditional_cregbundle(self):
        """Test bit conditions on gates when cregbundle=True"""

        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[0]).c_if(cr[0], 1)
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[1]).c_if(cr[1], 0)

        expected = "\n".join(
            [
                "            ┌───┐                ",
                "qr_0: |0>───┤ H ├────────────────",
                "            └─╥─┘       ┌───┐    ",
                "qr_1: |0>─────╫─────────┤ H ├────",
                "              ║         └─╥─┘    ",
                "         ┌────╨─────┐┌────╨─────┐",
                " cr: 0 2/╡ cr_0=0x1 ╞╡ cr_1=0x0 ╞",
                "         └──────────┘└──────────┘",
            ]
        )

        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    initial_state=True,
                    cregbundle=True,
                    vertical_compression="medium",
                )
            ),
            expected,
        )

    def test_text_condition_measure_bits_true(self):
        """Condition and measure on single bits cregbundle true"""

        bits = [Qubit(), Qubit(), Clbit(), Clbit()]
        cr = ClassicalRegister(2, "cr")
        crx = ClassicalRegister(3, "cs")
        circuit = QuantumCircuit(bits, cr, [Clbit()], crx)
        with self.assertWarns(DeprecationWarning):
            circuit.x(0).c_if(crx[1], 0)
        circuit.measure(0, bits[3])

        expected = "\n".join(
            [
                "         ┌───┐    ┌─┐",
                "   0: ───┤ X ├────┤M├",
                "         └─╥─┘    └╥┘",
                "   1: ─────╫───────╫─",
                "           ║       ║ ",
                "   0: ═════╬═══════╬═",
                "           ║       ║ ",
                "   1: ═════╬═══════╩═",
                "           ║         ",
                "cr: 2/═════╬═════════",
                "           ║         ",
                "   4: ═════╬═════════",
                "      ┌────╨─────┐   ",
                "cs: 3/╡ cs_1=0x0 ╞═══",
                "      └──────────┘   ",
            ]
        )
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", cregbundle=True, initial_state=False)),
            expected,
        )

    def test_text_condition_measure_bits_false(self):
        """Condition and measure on single bits cregbundle false"""

        bits = [Qubit(), Qubit(), Clbit(), Clbit()]
        cr = ClassicalRegister(2, "cr")
        crx = ClassicalRegister(3, "cs")
        circuit = QuantumCircuit(bits, cr, [Clbit()], crx)
        with self.assertWarns(DeprecationWarning):
            circuit.x(0).c_if(crx[1], 0)
        circuit.measure(0, bits[3])

        expected = "\n".join(
            [
                "      ┌───┐┌─┐",
                "   0: ┤ X ├┤M├",
                "      └─╥─┘└╥┘",
                "   1: ──╫───╫─",
                "        ║   ║ ",
                "   0: ══╬═══╬═",
                "        ║   ║ ",
                "   1: ══╬═══╩═",
                "        ║     ",
                "cr_0: ══╬═════",
                "        ║     ",
                "cr_1: ══╬═════",
                "        ║     ",
                "   4: ══╬═════",
                "        ║     ",
                "cs_0: ══╬═════",
                "        ║     ",
                "cs_1: ══o═════",
                "              ",
                "cs_2: ════════",
                "              ",
            ]
        )
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", cregbundle=False, initial_state=False)),
            expected,
        )

    def test_text_conditional_reverse_bits_1(self):
        """Classical condition on 2q2c circuit with cregbundle=False and reverse bits"""
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[0])
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[1]).c_if(cr, 1)

        expected = "\n".join(
            [
                "                 ┌───┐",
                "qr_1: |0>────────┤ H ├",
                "         ┌───┐┌─┐└─╥─┘",
                "qr_0: |0>┤ H ├┤M├──╫──",
                "         └───┘└╥┘  ║  ",
                " cr_1: 0 ══════╬═══o══",
                "               ║   ║  ",
                " cr_0: 0 ══════╩═══■══",
                "                  0x1 ",
            ]
        )

        self.assertEqual(
            str(
                circuit_drawer(
                    circuit, output="text", initial_state=True, cregbundle=False, reverse_bits=True
                )
            ),
            expected,
        )

    def test_text_conditional_reverse_bits_2(self):
        """Classical condition on 3q3c circuit with cergbundle=False and reverse bits"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[0]).c_if(cr, 6)
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[1]).c_if(cr, 1)
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[2]).c_if(cr, 2)
        with self.assertWarns(DeprecationWarning):
            circuit.cx(0, 1).c_if(cr, 3)

        expected = "\n".join(
            [
                "                   ┌───┐     ",
                "qr_2: |0>──────────┤ H ├─────",
                "              ┌───┐└─╥─┘┌───┐",
                "qr_1: |0>─────┤ H ├──╫──┤ X ├",
                "         ┌───┐└─╥─┘  ║  └─┬─┘",
                "qr_0: |0>┤ H ├──╫────╫────■──",
                "         └─╥─┘  ║    ║    ║  ",
                " cr_2: 0 ══■════o════o════o══",
                "           ║    ║    ║    ║  ",
                " cr_1: 0 ══■════o════■════■══",
                "           ║    ║    ║    ║  ",
                " cr_0: 0 ══o════■════o════■══",
                "          0x6  0x1  0x2  0x3 ",
            ]
        )

        self.assertEqual(
            str(
                circuit_drawer(
                    circuit, output="text", initial_state=True, cregbundle=False, reverse_bits=True
                )
            ),
            expected,
        )

    def test_text_condition_bits_reverse(self):
        """Condition and measure on single bits cregbundle true and reverse_bits true"""

        bits = [Qubit(), Qubit(), Clbit(), Clbit()]
        cr = ClassicalRegister(2, "cr")
        crx = ClassicalRegister(3, "cs")
        circuit = QuantumCircuit(bits, cr, [Clbit()], crx)
        with self.assertWarns(DeprecationWarning):
            circuit.x(0).c_if(bits[3], 0)

        expected = "\n".join(
            [
                "           ",
                "   1: ─────",
                "      ┌───┐",
                "   0: ┤ X ├",
                "      └─╥─┘",
                "cs: 3/══╬══",
                "        ║  ",
                "   4: ══╬══",
                "        ║  ",
                "cr: 2/══╬══",
                "        ║  ",
                "   1: ══o══",
                "           ",
                "   0: ═════",
                "           ",
            ]
        )
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit, output="text", cregbundle=True, initial_state=False, reverse_bits=True
                )
            ),
            expected,
        )


class TestTextIdleWires(QiskitTestCase):
    """The idle_wires option"""

    def test_text_h(self):
        """Remove QuWires."""
        # fmt: off
        expected = "\n".join(["         ┌───┐",
                              "q1_1: |0>┤ H ├",
                              "         └───┘"])
        # fmt: on
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr1)
        circuit.h(qr1[1])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, idle_wires=False)),
            expected,
        )

    def test_text_measure(self):
        """Remove QuWires and ClWires."""
        expected = "\n".join(
            [
                "         ┌─┐   ",
                "q2_0: |0>┤M├───",
                "         └╥┘┌─┐",
                "q2_1: |0>─╫─┤M├",
                "          ║ └╥┘",
                " c2: 0 2/═╩══╩═",
                "          0  1 ",
            ]
        )
        qr1 = QuantumRegister(2, "q1")
        cr1 = ClassicalRegister(2, "c1")
        qr2 = QuantumRegister(2, "q2")
        cr2 = ClassicalRegister(2, "c2")
        circuit = QuantumCircuit(qr1, qr2, cr1, cr2)
        circuit.measure(qr2, cr2)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, idle_wires=False)),
            expected,
        )

    def test_text_empty_circuit(self):
        """Remove everything in an empty circuit."""
        expected = ""
        circuit = QuantumCircuit()
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, idle_wires=False)),
            expected,
        )

    def test_text_barrier(self):
        """idle_wires should ignore barrier
        See https://github.com/Qiskit/qiskit-terra/issues/4391"""
        # fmt: off
        expected = "\n".join(["         ┌───┐ ░ ",
                              "qr_1: |0>┤ H ├─░─",
                              "         └───┘ ░ "])
        # fmt: on
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[1])
        circuit.barrier(qr[1], qr[2])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, idle_wires=False)),
            expected,
        )

    def test_text_barrier_delay(self):
        """idle_wires should ignore delay"""
        # fmt: off
        expected = "\n".join(["         ┌───┐ ░  ",
                              "qr_1: |0>┤ H ├─░──",
                              "         └───┘ ░  "])
        # fmt: on
        qr = QuantumRegister(4, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[1])
        circuit.barrier()
        circuit.delay(100, qr[2])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, idle_wires=False)),
            expected,
        )

    def test_does_not_mutate_circuit(self):
        """Using 'idle_wires=False' should not mutate the circuit.  Regression test of gh-8739."""
        circuit = QuantumCircuit(1)
        before_qubits = circuit.num_qubits
        circuit.draw(idle_wires=False)
        self.assertEqual(circuit.num_qubits, before_qubits)


class TestTextNonRational(QiskitTestCase):
    """non-rational numbers are correctly represented"""

    def test_text_pifrac(self):
        """u drawing with -5pi/8 fraction"""
        # fmt: off
        expected = "\n".join(
            ["      ┌──────────────┐",
             "q: |0>┤ U(π,-5π/8,0) ├",
             "      └──────────────┘"]
        )
        # fmt: on
        qr = QuantumRegister(1, "q")
        circuit = QuantumCircuit(qr)
        circuit.u(pi, -5 * pi / 8, 0, qr[0])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_complex(self):
        """Complex numbers show up in the text
        See https://github.com/Qiskit/qiskit-terra/issues/3640"""
        expected = "\n".join(
            [
                "     ┌────────────────────────────────────┐",
                "q_0: ┤0                                   ├",
                "     │  Initialize(0.5+0.1j,0,0,0.86023j) │",
                "q_1: ┤1                                   ├",
                "     └────────────────────────────────────┘",
            ]
        )
        ket = numpy.array([0.5 + 0.1 * 1j, 0, 0, 0.8602325267042626 * 1j])
        circuit = QuantumCircuit(2)
        circuit.initialize(ket, [0, 1])
        self.assertEqual(circuit.draw(output="text").single_string(), expected)

    def test_text_complex_pireal(self):
        """Complex numbers including pi show up in the text
        See https://github.com/Qiskit/qiskit-terra/issues/3640"""
        expected = "\n".join(
            [
                "        ┌────────────────────────────────┐",
                "q_0: |0>┤0                               ├",
                "        │  Initialize(π/10,0,0,0.94937j) │",
                "q_1: |0>┤1                               ├",
                "        └────────────────────────────────┘",
            ]
        )
        ket = numpy.array([0.1 * numpy.pi, 0, 0, 0.9493702944526474 * 1j])
        circuit = QuantumCircuit(2)
        circuit.initialize(ket, [0, 1])
        self.assertEqual(circuit.draw(output="text", initial_state=True).single_string(), expected)

    def test_text_complex_piimaginary(self):
        """Complex numbers including pi show up in the text

        See https://github.com/Qiskit/qiskit-terra/issues/3640"""
        expected = "\n".join(
            [
                "        ┌────────────────────────────────┐",
                "q_0: |0>┤0                               ├",
                "        │  Initialize(0.94937,0,0,π/10j) │",
                "q_1: |0>┤1                               ├",
                "        └────────────────────────────────┘",
            ]
        )
        ket = numpy.array([0.9493702944526474, 0, 0, 0.1 * numpy.pi * 1j])
        circuit = QuantumCircuit(2)
        circuit.initialize(ket, [0, 1])
        self.assertEqual(circuit.draw(output="text", initial_state=True).single_string(), expected)


class TestTextInstructionWithBothWires(QiskitTestCase):
    """Composite instructions with both kind of wires
    See https://github.com/Qiskit/qiskit-terra/issues/2973"""

    def test_text_all_1q_1c(self):
        """Test q0-c0 in q0-c0"""
        expected = "\n".join(
            [
                "       ┌───────┐",
                "qr: |0>┤0      ├",
                "       │  name │",
                " cr: 0 ╡0      ╞",
                "       └───────┘",
            ]
        )

        qr1 = QuantumRegister(1, "qr")
        cr1 = ClassicalRegister(1, "cr")
        inst = QuantumCircuit(qr1, cr1, name="name").to_instruction()
        circuit = QuantumCircuit(qr1, cr1)
        circuit.append(inst, qr1[:], cr1[:])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_all_2q_2c(self):
        """Test q0-q1-c0-c1 in q0-q1-c0-c1"""
        expected = "\n".join(
            [
                "         ┌───────┐",
                "qr_0: |0>┤0      ├",
                "         │       │",
                "qr_1: |0>┤1      ├",
                "         │  name │",
                " cr_0: 0 ╡0      ╞",
                "         │       │",
                " cr_1: 0 ╡1      ╞",
                "         └───────┘",
            ]
        )

        qr2 = QuantumRegister(2, "qr")
        cr2 = ClassicalRegister(2, "cr")
        inst = QuantumCircuit(qr2, cr2, name="name").to_instruction()
        circuit = QuantumCircuit(qr2, cr2)
        circuit.append(inst, qr2[:], cr2[:])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_all_2q_2c_cregbundle(self):
        """Test q0-q1-c0-c1 in q0-q1-c0-c1. Ignore cregbundle=True"""
        expected = "\n".join(
            [
                "         ┌───────┐",
                "qr_0: |0>┤0      ├",
                "         │       │",
                "qr_1: |0>┤1      ├",
                "         │  name │",
                " cr_0: 0 ╡0      ╞",
                "         │       │",
                " cr_1: 0 ╡1      ╞",
                "         └───────┘",
            ]
        )

        qr2 = QuantumRegister(2, "qr")
        cr2 = ClassicalRegister(2, "cr")
        inst = QuantumCircuit(qr2, cr2, name="name").to_instruction()
        circuit = QuantumCircuit(qr2, cr2)
        circuit.append(inst, qr2[:], cr2[:])
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(
                str(circuit_drawer(circuit, output="text", initial_state=True, cregbundle=True)),
                expected,
            )

    def test_text_4q_2c(self):
        """Test q1-q2-q3-q4-c1-c2 in q0-q1-q2-q3-q4-q5-c0-c1-c2-c3-c4-c5"""
        expected = "\n".join(
            [
                "                 ",
                "q_0: |0>─────────",
                "        ┌───────┐",
                "q_1: |0>┤0      ├",
                "        │       │",
                "q_2: |0>┤1      ├",
                "        │       │",
                "q_3: |0>┤2      ├",
                "        │       │",
                "q_4: |0>┤3      ├",
                "        │  name │",
                "q_5: |0>┤       ├",
                "        │       │",
                " c_0: 0 ╡       ╞",
                "        │       │",
                " c_1: 0 ╡0      ╞",
                "        │       │",
                " c_2: 0 ╡1      ╞",
                "        └───────┘",
                " c_3: 0 ═════════",
                "                 ",
                " c_4: 0 ═════════",
                "                 ",
                " c_5: 0 ═════════",
                "                 ",
            ]
        )

        qr4 = QuantumRegister(4)
        cr4 = ClassicalRegister(2)
        inst = QuantumCircuit(qr4, cr4, name="name").to_instruction()
        qr6 = QuantumRegister(6, "q")
        cr6 = ClassicalRegister(6, "c")
        circuit = QuantumCircuit(qr6, cr6)
        circuit.append(inst, qr6[1:5], cr6[1:3])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_2q_1c(self):
        """Test q0-c0 in q0-q1-c0
        See https://github.com/Qiskit/qiskit-terra/issues/4066"""
        expected = "\n".join(
            [
                "        ┌───────┐",
                "q_0: |0>┤0      ├",
                "        │       │",
                "q_1: |0>┤  Name ├",
                "        │       │",
                "   c: 0 ╡0      ╞",
                "        └───────┘",
            ]
        )

        qr = QuantumRegister(2, name="q")
        cr = ClassicalRegister(1, name="c")
        circuit = QuantumCircuit(qr, cr)
        inst = QuantumCircuit(1, 1, name="Name").to_instruction()
        circuit.append(inst, [qr[0]], [cr[0]])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_3q_3c_qlabels_inverted(self):
        """Test q3-q0-q1-c0-c1-c_10 in q0-q1-q2-q3-c0-c1-c2-c_10-c_11
        See https://github.com/Qiskit/qiskit-terra/issues/6178"""
        expected = "\n".join(
            [
                "        ┌───────┐",
                "q_0: |0>┤1      ├",
                "        │       │",
                "q_1: |0>┤2      ├",
                "        │       │",
                "q_2: |0>┤       ├",
                "        │       │",
                "q_3: |0>┤0      ├",
                "        │  Name │",
                " c_0: 0 ╡0      ╞",
                "        │       │",
                " c_1: 0 ╡1      ╞",
                "        │       │",
                " c_2: 0 ╡       ╞",
                "        │       │",
                "c1_0: 0 ╡2      ╞",
                "        └───────┘",
                "c1_1: 0 ═════════",
                "                 ",
            ]
        )

        qr = QuantumRegister(4, name="q")
        cr = ClassicalRegister(3, name="c")
        cr1 = ClassicalRegister(2, name="c1")
        circuit = QuantumCircuit(qr, cr, cr1)
        inst = QuantumCircuit(3, 3, name="Name").to_instruction()
        circuit.append(inst, [qr[3], qr[0], qr[1]], [cr[0], cr[1], cr1[0]])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_3q_3c_clabels_inverted(self):
        """Test q0-q1-q3-c_11-c0-c_10 in q0-q1-q2-q3-c0-c1-c2-c_10-c_11
        See https://github.com/Qiskit/qiskit-terra/issues/6178"""
        expected = "\n".join(
            [
                "        ┌───────┐",
                "q_0: |0>┤0      ├",
                "        │       │",
                "q_1: |0>┤1      ├",
                "        │       │",
                "q_2: |0>┤       ├",
                "        │       │",
                "q_3: |0>┤2      ├",
                "        │       │",
                " c_0: 0 ╡1 Name ╞",
                "        │       │",
                " c_1: 0 ╡       ╞",
                "        │       │",
                " c_2: 0 ╡       ╞",
                "        │       │",
                "c1_0: 0 ╡2      ╞",
                "        │       │",
                "c1_1: 0 ╡0      ╞",
                "        └───────┘",
            ]
        )

        qr = QuantumRegister(4, name="q")
        cr = ClassicalRegister(3, name="c")
        cr1 = ClassicalRegister(2, name="c1")
        circuit = QuantumCircuit(qr, cr, cr1)
        inst = QuantumCircuit(3, 3, name="Name").to_instruction()
        circuit.append(inst, [qr[0], qr[1], qr[3]], [cr1[1], cr[0], cr1[0]])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_3q_3c_qclabels_inverted(self):
        """Test q3-q1-q2-c_11-c0-c_10 in q0-q1-q2-q3-c0-c1-c2-c_10-c_11
        See https://github.com/Qiskit/qiskit-terra/issues/6178"""
        expected = "\n".join(
            [
                "                 ",
                "q_0: |0>─────────",
                "        ┌───────┐",
                "q_1: |0>┤1      ├",
                "        │       │",
                "q_2: |0>┤2      ├",
                "        │       │",
                "q_3: |0>┤0      ├",
                "        │       │",
                " c_0: 0 ╡1      ╞",
                "        │  Name │",
                " c_1: 0 ╡       ╞",
                "        │       │",
                " c_2: 0 ╡       ╞",
                "        │       │",
                "c1_0: 0 ╡2      ╞",
                "        │       │",
                "c1_1: 0 ╡0      ╞",
                "        └───────┘",
            ]
        )

        qr = QuantumRegister(4, name="q")
        cr = ClassicalRegister(3, name="c")
        cr1 = ClassicalRegister(2, name="c1")
        circuit = QuantumCircuit(qr, cr, cr1)
        inst = QuantumCircuit(3, 3, name="Name").to_instruction()
        circuit.append(inst, [qr[3], qr[1], qr[2]], [cr1[1], cr[0], cr1[0]])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)


class TestTextDrawerAppendedLargeInstructions(QiskitTestCase):
    """Composite instructions with more than 10 qubits
    See https://github.com/Qiskit/qiskit-terra/pull/4095"""

    def test_text_11q(self):
        """Test q0-...-q10 in q0-...-q10"""
        expected = "\n".join(
            [
                "         ┌────────┐",
                " q_0: |0>┤0       ├",
                "         │        │",
                " q_1: |0>┤1       ├",
                "         │        │",
                " q_2: |0>┤2       ├",
                "         │        │",
                " q_3: |0>┤3       ├",
                "         │        │",
                " q_4: |0>┤4       ├",
                "         │        │",
                " q_5: |0>┤5  Name ├",
                "         │        │",
                " q_6: |0>┤6       ├",
                "         │        │",
                " q_7: |0>┤7       ├",
                "         │        │",
                " q_8: |0>┤8       ├",
                "         │        │",
                " q_9: |0>┤9       ├",
                "         │        │",
                "q_10: |0>┤10      ├",
                "         └────────┘",
            ]
        )

        qr = QuantumRegister(11, "q")
        circuit = QuantumCircuit(qr)
        inst = QuantumCircuit(11, name="Name").to_instruction()
        circuit.append(inst, qr)

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_text_11q_1c(self):
        """Test q0-...-q10-c0 in q0-...-q10-c0"""
        expected = "\n".join(
            [
                "         ┌────────┐",
                " q_0: |0>┤0       ├",
                "         │        │",
                " q_1: |0>┤1       ├",
                "         │        │",
                " q_2: |0>┤2       ├",
                "         │        │",
                " q_3: |0>┤3       ├",
                "         │        │",
                " q_4: |0>┤4       ├",
                "         │        │",
                " q_5: |0>┤5       ├",
                "         │   Name │",
                " q_6: |0>┤6       ├",
                "         │        │",
                " q_7: |0>┤7       ├",
                "         │        │",
                " q_8: |0>┤8       ├",
                "         │        │",
                " q_9: |0>┤9       ├",
                "         │        │",
                "q_10: |0>┤10      ├",
                "         │        │",
                "    c: 0 ╡0       ╞",
                "         └────────┘",
            ]
        )

        qr = QuantumRegister(11, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        inst = QuantumCircuit(11, 1, name="Name").to_instruction()
        circuit.append(inst, qr, cr)

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)


class TestTextControlledGate(QiskitTestCase):
    """Test controlled gates"""

    def test_cch_bot(self):
        """Controlled CH (bottom)"""
        expected = "\n".join(
            [
                "             ",
                "q_0: |0>──■──",
                "          │  ",
                "q_1: |0>──■──",
                "        ┌─┴─┐",
                "q_2: |0>┤ H ├",
                "        └───┘",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(HGate().control(2), [qr[0], qr[1], qr[2]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_cch_mid(self):
        """Controlled CH (middle)"""
        expected = "\n".join(
            [
                "             ",
                "q_0: |0>──■──",
                "        ┌─┴─┐",
                "q_1: |0>┤ H ├",
                "        └─┬─┘",
                "q_2: |0>──■──",
                "             ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(HGate().control(2), [qr[0], qr[2], qr[1]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_cch_top(self):
        """Controlled CH"""
        expected = "\n".join(
            [
                "        ┌───┐",
                "q_0: |0>┤ H ├",
                "        └─┬─┘",
                "q_1: |0>──■──",
                "          │  ",
                "q_2: |0>──■──",
                "             ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(HGate().control(2), [qr[2], qr[1], qr[0]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_c3h(self):
        """Controlled Controlled CH"""
        expected = "\n".join(
            [
                "             ",
                "q_0: |0>──■──",
                "          │  ",
                "q_1: |0>──■──",
                "          │  ",
                "q_2: |0>──■──",
                "        ┌─┴─┐",
                "q_3: |0>┤ H ├",
                "        └───┘",
            ]
        )
        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(HGate().control(3), [qr[0], qr[1], qr[2], qr[3]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_c3h_middle(self):
        """Controlled Controlled CH (middle)"""
        expected = "\n".join(
            [
                "             ",
                "q_0: |0>──■──",
                "        ┌─┴─┐",
                "q_1: |0>┤ H ├",
                "        └─┬─┘",
                "q_2: |0>──■──",
                "          │  ",
                "q_3: |0>──■──",
                "             ",
            ]
        )
        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(HGate().control(3), [qr[0], qr[3], qr[2], qr[1]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_c3u2(self):
        """Controlled Controlled U2"""
        expected = "\n".join(
            [
                "                       ",
                "q_0: |0>───────■───────",
                "        ┌──────┴──────┐",
                "q_1: |0>┤ U2(π,-5π/8) ├",
                "        └──────┬──────┘",
                "q_2: |0>───────■───────",
                "               │       ",
                "q_3: |0>───────■───────",
                "                       ",
            ]
        )
        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(U2Gate(pi, -5 * pi / 8).control(3), [qr[0], qr[3], qr[2], qr[1]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_controlled_composite_gate_edge(self):
        """Controlled composite gates (edge)
        See: https://github.com/Qiskit/qiskit-terra/issues/3546"""
        expected = "\n".join(
            [
                "        ┌──────┐",
                "q_0: |0>┤0     ├",
                "        │      │",
                "q_1: |0>■      ├",
                "        │  ghz │",
                "q_2: |0>┤1     ├",
                "        │      │",
                "q_3: |0>┤2     ├",
                "        └──────┘",
            ]
        )
        ghz_circuit = QuantumCircuit(3, name="ghz")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        cghz = ghz.control(1)
        circuit = QuantumCircuit(4)
        circuit.append(cghz, [1, 0, 2, 3])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_controlled_composite_gate_top(self):
        """Controlled composite gates (top)"""
        expected = "\n".join(
            [
                "                ",
                "q_0: |0>───■────",
                "        ┌──┴───┐",
                "q_1: |0>┤0     ├",
                "        │      │",
                "q_2: |0>┤2 ghz ├",
                "        │      │",
                "q_3: |0>┤1     ├",
                "        └──────┘",
            ]
        )
        ghz_circuit = QuantumCircuit(3, name="ghz")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        cghz = ghz.control(1)
        circuit = QuantumCircuit(4)
        circuit.append(cghz, [0, 1, 3, 2])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_controlled_composite_gate_bot(self):
        """Controlled composite gates (bottom)"""
        expected = "\n".join(
            [
                "        ┌──────┐",
                "q_0: |0>┤1     ├",
                "        │      │",
                "q_1: |0>┤0 ghz ├",
                "        │      │",
                "q_2: |0>┤2     ├",
                "        └──┬───┘",
                "q_3: |0>───■────",
                "                ",
            ]
        )
        ghz_circuit = QuantumCircuit(3, name="ghz")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        cghz = ghz.control(1)
        circuit = QuantumCircuit(4)
        circuit.append(cghz, [3, 1, 0, 2])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_controlled_composite_gate_top_bot(self):
        """Controlled composite gates (top and bottom)"""
        expected = "\n".join(
            [
                "                ",
                "q_0: |0>───■────",
                "        ┌──┴───┐",
                "q_1: |0>┤0     ├",
                "        │      │",
                "q_2: |0>┤1 ghz ├",
                "        │      │",
                "q_3: |0>┤2     ├",
                "        └──┬───┘",
                "q_4: |0>───■────",
                "                ",
            ]
        )
        ghz_circuit = QuantumCircuit(3, name="ghz")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        ccghz = ghz.control(2)
        circuit = QuantumCircuit(5)
        circuit.append(ccghz, [4, 0, 1, 2, 3])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_controlled_composite_gate_all(self):
        """Controlled composite gates (top, bot, and edge)"""
        expected = "\n".join(
            [
                "                ",
                "q_0: |0>───■────",
                "        ┌──┴───┐",
                "q_1: |0>┤0     ├",
                "        │      │",
                "q_2: |0>■      ├",
                "        │  ghz │",
                "q_3: |0>┤1     ├",
                "        │      │",
                "q_4: |0>┤2     ├",
                "        └──┬───┘",
                "q_5: |0>───■────",
                "                ",
            ]
        )
        ghz_circuit = QuantumCircuit(3, name="ghz")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        ccghz = ghz.control(3)
        circuit = QuantumCircuit(6)
        circuit.append(ccghz, [0, 2, 5, 1, 3, 4])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_controlled_composite_gate_even_label(self):
        """Controlled composite gates (top and bottom) with a even label length"""
        expected = "\n".join(
            [
                "                 ",
                "q_0: |0>────■────",
                "        ┌───┴───┐",
                "q_1: |0>┤0      ├",
                "        │       │",
                "q_2: |0>┤1 cghz ├",
                "        │       │",
                "q_3: |0>┤2      ├",
                "        └───┬───┘",
                "q_4: |0>────■────",
                "                 ",
            ]
        )
        ghz_circuit = QuantumCircuit(3, name="cghz")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        ccghz = ghz.control(2)
        circuit = QuantumCircuit(5)
        circuit.append(ccghz, [4, 0, 1, 2, 3])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)


class TestTextOpenControlledGate(QiskitTestCase):
    """Test open controlled gates"""

    def test_ch_bot(self):
        """Open controlled H (bottom)"""
        # fmt: off
        expected = "\n".join(
            ["             ",
             "q_0: |0>──o──",
             "        ┌─┴─┐",
             "q_1: |0>┤ H ├",
             "        └───┘"]
        )
        # fmt: on
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(HGate().control(1, ctrl_state=0), [qr[0], qr[1]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_cz_bot(self):
        """Open controlled Z (bottom)"""
        # fmt: off
        expected = "\n".join(["           ",
                              "q_0: |0>─o─",
                              "         │ ",
                              "q_1: |0>─■─",
                              "           "])
        # fmt: on
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(ZGate().control(1, ctrl_state=0), [qr[0], qr[1]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_ccz_bot(self):
        """Closed-Open controlled Z (bottom)"""
        expected = "\n".join(
            [
                "           ",
                "q_0: |0>─■─",
                "         │ ",
                "q_1: |0>─o─",
                "         │ ",
                "q_2: |0>─■─",
                "           ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(ZGate().control(2, ctrl_state="01"), [qr[0], qr[1], qr[2]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_cccz_conditional(self):
        """Closed-Open controlled Z (with conditional)"""
        expected = "\n".join(
            [
                "               ",
                "q_0: |0>───■───",
                "           │   ",
                "q_1: |0>───o───",
                "           │   ",
                "q_2: |0>───■───",
                "           │   ",
                "q_3: |0>───■───",
                "        ┌──╨──┐",
                " c: 0 1/╡ 0x1 ╞",
                "        └─────┘",
            ]
        )
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            circuit.append(
                ZGate().control(3, ctrl_state="101").c_if(cr, 1), [qr[0], qr[1], qr[2], qr[3]]
            )
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_cch_bot(self):
        """Controlled CH (bottom)"""
        expected = "\n".join(
            [
                "             ",
                "q_0: |0>──o──",
                "          │  ",
                "q_1: |0>──■──",
                "        ┌─┴─┐",
                "q_2: |0>┤ H ├",
                "        └───┘",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(HGate().control(2, ctrl_state="10"), [qr[0], qr[1], qr[2]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_cch_mid(self):
        """Controlled CH (middle)"""
        expected = "\n".join(
            [
                "             ",
                "q_0: |0>──o──",
                "        ┌─┴─┐",
                "q_1: |0>┤ H ├",
                "        └─┬─┘",
                "q_2: |0>──■──",
                "             ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(HGate().control(2, ctrl_state="10"), [qr[0], qr[2], qr[1]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_cch_top(self):
        """Controlled CH"""
        expected = "\n".join(
            [
                "        ┌───┐",
                "q_0: |0>┤ H ├",
                "        └─┬─┘",
                "q_1: |0>──o──",
                "          │  ",
                "q_2: |0>──■──",
                "             ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(HGate().control(2, ctrl_state="10"), [qr[1], qr[2], qr[0]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_c3h(self):
        """Controlled Controlled CH"""
        expected = "\n".join(
            [
                "             ",
                "q_0: |0>──o──",
                "          │  ",
                "q_1: |0>──o──",
                "          │  ",
                "q_2: |0>──■──",
                "        ┌─┴─┐",
                "q_3: |0>┤ H ├",
                "        └───┘",
            ]
        )
        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(HGate().control(3, ctrl_state="100"), [qr[0], qr[1], qr[2], qr[3]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_c3h_middle(self):
        """Controlled Controlled CH (middle)"""
        expected = "\n".join(
            [
                "             ",
                "q_0: |0>──o──",
                "        ┌─┴─┐",
                "q_1: |0>┤ H ├",
                "        └─┬─┘",
                "q_2: |0>──o──",
                "          │  ",
                "q_3: |0>──■──",
                "             ",
            ]
        )
        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(HGate().control(3, ctrl_state="010"), [qr[0], qr[3], qr[2], qr[1]])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_c3u2(self):
        """Controlled Controlled U2"""
        expected = "\n".join(
            [
                "                       ",
                "q_0: |0>───────o───────",
                "        ┌──────┴──────┐",
                "q_1: |0>┤ U2(π,-5π/8) ├",
                "        └──────┬──────┘",
                "q_2: |0>───────■───────",
                "               │       ",
                "q_3: |0>───────o───────",
                "                       ",
            ]
        )
        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(
            U2Gate(pi, -5 * pi / 8).control(3, ctrl_state="100"), [qr[0], qr[3], qr[2], qr[1]]
        )
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_controlled_composite_gate_edge(self):
        """Controlled composite gates (edge)
        See: https://github.com/Qiskit/qiskit-terra/issues/3546"""
        expected = "\n".join(
            [
                "        ┌──────┐",
                "q_0: |0>┤0     ├",
                "        │      │",
                "q_1: |0>o      ├",
                "        │  ghz │",
                "q_2: |0>┤1     ├",
                "        │      │",
                "q_3: |0>┤2     ├",
                "        └──────┘",
            ]
        )
        ghz_circuit = QuantumCircuit(3, name="ghz")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        cghz = ghz.control(1, ctrl_state="0")
        circuit = QuantumCircuit(4)
        circuit.append(cghz, [1, 0, 2, 3])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_controlled_composite_gate_top(self):
        """Controlled composite gates (top)"""
        expected = "\n".join(
            [
                "                ",
                "q_0: |0>───o────",
                "        ┌──┴───┐",
                "q_1: |0>┤0     ├",
                "        │      │",
                "q_2: |0>┤2 ghz ├",
                "        │      │",
                "q_3: |0>┤1     ├",
                "        └──────┘",
            ]
        )
        ghz_circuit = QuantumCircuit(3, name="ghz")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        cghz = ghz.control(1, ctrl_state="0")
        circuit = QuantumCircuit(4)
        circuit.append(cghz, [0, 1, 3, 2])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_controlled_composite_gate_bot(self):
        """Controlled composite gates (bottom)"""
        expected = "\n".join(
            [
                "        ┌──────┐",
                "q_0: |0>┤1     ├",
                "        │      │",
                "q_1: |0>┤0 ghz ├",
                "        │      │",
                "q_2: |0>┤2     ├",
                "        └──┬───┘",
                "q_3: |0>───o────",
                "                ",
            ]
        )
        ghz_circuit = QuantumCircuit(3, name="ghz")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        cghz = ghz.control(1, ctrl_state="0")
        circuit = QuantumCircuit(4)
        circuit.append(cghz, [3, 1, 0, 2])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_controlled_composite_gate_top_bot(self):
        """Controlled composite gates (top and bottom)"""
        expected = "\n".join(
            [
                "                ",
                "q_0: |0>───o────",
                "        ┌──┴───┐",
                "q_1: |0>┤0     ├",
                "        │      │",
                "q_2: |0>┤1 ghz ├",
                "        │      │",
                "q_3: |0>┤2     ├",
                "        └──┬───┘",
                "q_4: |0>───■────",
                "                ",
            ]
        )
        ghz_circuit = QuantumCircuit(3, name="ghz")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        ccghz = ghz.control(2, ctrl_state="01")
        circuit = QuantumCircuit(5)
        circuit.append(ccghz, [4, 0, 1, 2, 3])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_controlled_composite_gate_all(self):
        """Controlled composite gates (top, bot, and edge)"""
        expected = "\n".join(
            [
                "                ",
                "q_0: |0>───o────",
                "        ┌──┴───┐",
                "q_1: |0>┤0     ├",
                "        │      │",
                "q_2: |0>o      ├",
                "        │  ghz │",
                "q_3: |0>┤1     ├",
                "        │      │",
                "q_4: |0>┤2     ├",
                "        └──┬───┘",
                "q_5: |0>───o────",
                "                ",
            ]
        )
        ghz_circuit = QuantumCircuit(3, name="ghz")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        ccghz = ghz.control(3, ctrl_state="000")
        circuit = QuantumCircuit(6)
        circuit.append(ccghz, [0, 2, 5, 1, 3, 4])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_open_controlled_x(self):
        """Controlled X gates.
        See https://github.com/Qiskit/qiskit-terra/issues/4180"""
        expected = "\n".join(
            [
                "                                  ",
                "qr_0: |0>──o────o────o────o────■──",
                "         ┌─┴─┐  │    │    │    │  ",
                "qr_1: |0>┤ X ├──o────■────■────o──",
                "         └───┘┌─┴─┐┌─┴─┐  │    │  ",
                "qr_2: |0>─────┤ X ├┤ X ├──o────o──",
                "              └───┘└───┘┌─┴─┐┌─┴─┐",
                "qr_3: |0>───────────────┤ X ├┤ X ├",
                "                        └───┘└─┬─┘",
                "qr_4: |0>──────────────────────■──",
                "                                  ",
            ]
        )
        qreg = QuantumRegister(5, "qr")
        circuit = QuantumCircuit(qreg)
        control1 = XGate().control(1, ctrl_state="0")
        circuit.append(control1, [0, 1])
        control2 = XGate().control(2, ctrl_state="00")
        circuit.append(control2, [0, 1, 2])
        control2_2 = XGate().control(2, ctrl_state="10")
        circuit.append(control2_2, [0, 1, 2])
        control3 = XGate().control(3, ctrl_state="010")
        circuit.append(control3, [0, 1, 2, 3])
        control3 = XGate().control(4, ctrl_state="0101")
        circuit.append(control3, [0, 1, 4, 2, 3])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_open_controlled_y(self):
        """Controlled Y gates.
        See https://github.com/Qiskit/qiskit-terra/issues/4180"""
        expected = "\n".join(
            [
                "                                  ",
                "qr_0: |0>──o────o────o────o────■──",
                "         ┌─┴─┐  │    │    │    │  ",
                "qr_1: |0>┤ Y ├──o────■────■────o──",
                "         └───┘┌─┴─┐┌─┴─┐  │    │  ",
                "qr_2: |0>─────┤ Y ├┤ Y ├──o────o──",
                "              └───┘└───┘┌─┴─┐┌─┴─┐",
                "qr_3: |0>───────────────┤ Y ├┤ Y ├",
                "                        └───┘└─┬─┘",
                "qr_4: |0>──────────────────────■──",
                "                                  ",
            ]
        )
        qreg = QuantumRegister(5, "qr")
        circuit = QuantumCircuit(qreg)
        control1 = YGate().control(1, ctrl_state="0")
        circuit.append(control1, [0, 1])
        control2 = YGate().control(2, ctrl_state="00")
        circuit.append(control2, [0, 1, 2])
        control2_2 = YGate().control(2, ctrl_state="10")
        circuit.append(control2_2, [0, 1, 2])
        control3 = YGate().control(3, ctrl_state="010")
        circuit.append(control3, [0, 1, 2, 3])
        control3 = YGate().control(4, ctrl_state="0101")
        circuit.append(control3, [0, 1, 4, 2, 3])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_open_controlled_z(self):
        """Controlled Z gates."""
        expected = "\n".join(
            [
                "                        ",
                "qr_0: |0>─o──o──o──o──■─",
                "          │  │  │  │  │ ",
                "qr_1: |0>─■──o──■──■──o─",
                "             │  │  │  │ ",
                "qr_2: |0>────■──■──o──o─",
                "                   │  │ ",
                "qr_3: |0>──────────■──■─",
                "                      │ ",
                "qr_4: |0>─────────────■─",
                "                        ",
            ]
        )
        qreg = QuantumRegister(5, "qr")
        circuit = QuantumCircuit(qreg)
        control1 = ZGate().control(1, ctrl_state="0")
        circuit.append(control1, [0, 1])
        control2 = ZGate().control(2, ctrl_state="00")
        circuit.append(control2, [0, 1, 2])
        control2_2 = ZGate().control(2, ctrl_state="10")
        circuit.append(control2_2, [0, 1, 2])
        control3 = ZGate().control(3, ctrl_state="010")
        circuit.append(control3, [0, 1, 2, 3])
        control3 = ZGate().control(4, ctrl_state="0101")
        circuit.append(control3, [0, 1, 4, 2, 3])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_open_controlled_u1(self):
        """Controlled U1 gates."""
        expected = "\n".join(
            [
                "                                                           ",
                "qr_0: |0>─o─────────o─────────o─────────o─────────■────────",
                "          │U1(0.1)  │         │         │         │        ",
                "qr_1: |0>─■─────────o─────────■─────────■─────────o────────",
                "                    │U1(0.2)  │U1(0.3)  │         │        ",
                "qr_2: |0>───────────■─────────■─────────o─────────o────────",
                "                                        │U1(0.4)  │        ",
                "qr_3: |0>───────────────────────────────■─────────■────────",
                "                                                  │U1(0.5) ",
                "qr_4: |0>─────────────────────────────────────────■────────",
                "                                                           ",
            ]
        )
        qreg = QuantumRegister(5, "qr")
        circuit = QuantumCircuit(qreg)
        control1 = U1Gate(0.1).control(1, ctrl_state="0")
        circuit.append(control1, [0, 1])
        control2 = U1Gate(0.2).control(2, ctrl_state="00")
        circuit.append(control2, [0, 1, 2])
        control2_2 = U1Gate(0.3).control(2, ctrl_state="10")
        circuit.append(control2_2, [0, 1, 2])
        control3 = U1Gate(0.4).control(3, ctrl_state="010")
        circuit.append(control3, [0, 1, 2, 3])
        control3 = U1Gate(0.5).control(4, ctrl_state="0101")
        circuit.append(control3, [0, 1, 4, 2, 3])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_open_controlled_swap(self):
        """Controlled SWAP gates."""
        expected = "\n".join(
            [
                "                     ",
                "qr_0: |0>─o──o──o──o─",
                "          │  │  │  │ ",
                "qr_1: |0>─X──o──■──■─",
                "          │  │  │  │ ",
                "qr_2: |0>─X──X──X──o─",
                "             │  │  │ ",
                "qr_3: |0>────X──X──X─",
                "                   │ ",
                "qr_4: |0>──────────X─",
                "                     ",
            ]
        )
        qreg = QuantumRegister(5, "qr")
        circuit = QuantumCircuit(qreg)
        control1 = SwapGate().control(1, ctrl_state="0")
        circuit.append(control1, [0, 1, 2])
        control2 = SwapGate().control(2, ctrl_state="00")
        circuit.append(control2, [0, 1, 2, 3])
        control2_2 = SwapGate().control(2, ctrl_state="10")
        circuit.append(control2_2, [0, 1, 2, 3])
        control3 = SwapGate().control(3, ctrl_state="010")
        circuit.append(control3, [0, 1, 2, 3, 4])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_open_controlled_rzz(self):
        """Controlled RZZ gates."""
        expected = "\n".join(
            [
                "                                         ",
                "qr_0: |0>─o───────o───────o───────o──────",
                "          │       │       │       │      ",
                "qr_1: |0>─■───────o───────■───────■──────",
                "          │ZZ(1)  │       │       │      ",
                "qr_2: |0>─■───────■───────■───────o──────",
                "                  │ZZ(1)  │ZZ(1)  │      ",
                "qr_3: |0>─────────■───────■───────■──────",
                "                                  │ZZ(1) ",
                "qr_4: |0>─────────────────────────■──────",
                "                                         ",
            ]
        )
        qreg = QuantumRegister(5, "qr")
        circuit = QuantumCircuit(qreg)
        control1 = RZZGate(1).control(1, ctrl_state="0")
        circuit.append(control1, [0, 1, 2])
        control2 = RZZGate(1).control(2, ctrl_state="00")
        circuit.append(control2, [0, 1, 2, 3])
        control2_2 = RZZGate(1).control(2, ctrl_state="10")
        circuit.append(control2_2, [0, 1, 2, 3])
        control3 = RZZGate(1).control(3, ctrl_state="010")
        circuit.append(control3, [0, 1, 2, 3, 4])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_open_out_of_order(self):
        """Out of order CXs
        See: https://github.com/Qiskit/qiskit-terra/issues/4052#issuecomment-613736911"""
        expected = "\n".join(
            [
                "             ",
                "q_0: |0>──■──",
                "          │  ",
                "q_1: |0>──■──",
                "        ┌─┴─┐",
                "q_2: |0>┤ X ├",
                "        └─┬─┘",
                "q_3: |0>──o──",
                "             ",
                "q_4: |0>─────",
                "             ",
            ]
        )
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(XGate().control(3, ctrl_state="101"), [qr[0], qr[3], qr[1], qr[2]])

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)


class TestTextWithLayout(QiskitTestCase):
    """The with_layout option"""

    def test_with_no_layout(self):
        """A circuit without layout"""
        expected = "\n".join(
            [
                "             ",
                "q_0: |0>─────",
                "        ┌───┐",
                "q_1: |0>┤ H ├",
                "        └───┘",
                "q_2: |0>─────",
                "             ",
            ]
        )
        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[1])
        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_mixed_layout(self):
        """With a mixed layout."""
        expected = "\n".join(
            [
                "                  ┌───┐",
                "      v_0 -> 0 |0>┤ H ├",
                "                  └───┘",
                "ancilla_1 -> 1 |0>─────",
                "                       ",
                "ancilla_0 -> 2 |0>─────",
                "                  ┌───┐",
                "      v_1 -> 3 |0>┤ H ├",
                "                  └───┘",
            ]
        )
        qr = QuantumRegister(2, "v")
        ancilla = QuantumRegister(2, "ancilla")
        circuit = QuantumCircuit(qr, ancilla)
        circuit.h(qr)

        pass_ = ApplyLayout()
        pass_.property_set["layout"] = Layout({qr[0]: 0, ancilla[1]: 1, ancilla[0]: 2, qr[1]: 3})
        circuit_with_layout = pass_(circuit)

        self.assertEqual(
            str(circuit_drawer(circuit_with_layout, output="text", initial_state=True)), expected
        )

    def test_partial_layout(self):
        """With a partial layout.
        See: https://github.com/Qiskit/qiskit-terra/issues/4757"""
        expected = "\n".join(
            [
                "            ┌───┐",
                "v_0 -> 0 |0>┤ H ├",
                "            └───┘",
                "       1 |0>─────",
                "                 ",
                "       2 |0>─────",
                "            ┌───┐",
                "v_1 -> 3 |0>┤ H ├",
                "            └───┘",
            ]
        )
        qr = QuantumRegister(2, "v")
        pqr = QuantumRegister(4, "physical")
        circuit = QuantumCircuit(pqr)
        circuit.h(0)
        circuit.h(3)
        circuit._layout = TranspileLayout(
            Layout({0: qr[0], 1: None, 2: None, 3: qr[1]}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        circuit._layout.initial_layout.add_register(qr)

        self.assertEqual(str(circuit_drawer(circuit, output="text", initial_state=True)), expected)

    def test_with_classical_regs(self):
        """Involving classical registers"""
        expected = "\n".join(
            [
                "                    ",
                "qr1_0 -> 0 |0>──────",
                "                    ",
                "qr1_1 -> 1 |0>──────",
                "              ┌─┐   ",
                "qr2_0 -> 2 |0>┤M├───",
                "              └╥┘┌─┐",
                "qr2_1 -> 3 |0>─╫─┤M├",
                "               ║ └╥┘",
                "      cr: 0 2/═╩══╩═",
                "               0  1 ",
            ]
        )
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        cr = ClassicalRegister(2, "cr")

        circuit = QuantumCircuit(qr1, qr2, cr)
        circuit.measure(qr2[0], cr[0])
        circuit.measure(qr2[1], cr[1])

        pass_ = ApplyLayout()
        pass_.property_set["layout"] = Layout({qr1[0]: 0, qr1[1]: 1, qr2[0]: 2, qr2[1]: 3})
        circuit_with_layout = pass_(circuit)

        self.assertEqual(
            str(circuit_drawer(circuit_with_layout, output="text", initial_state=True)), expected
        )

    def test_with_layout_but_disable(self):
        """With parameter without_layout=False"""
        expected = "\n".join(
            [
                "              ",
                "q_0: |0>──────",
                "              ",
                "q_1: |0>──────",
                "        ┌─┐   ",
                "q_2: |0>┤M├───",
                "        └╥┘┌─┐",
                "q_3: |0>─╫─┤M├",
                "         ║ └╥┘",
                "cr: 0 2/═╩══╩═",
                "         0  1 ",
            ]
        )
        pqr = QuantumRegister(4, "q")
        qr1 = QuantumRegister(2, "qr1")
        cr = ClassicalRegister(2, "cr")
        qr2 = QuantumRegister(2, "qr2")
        circuit = QuantumCircuit(pqr, cr)
        circuit._layout = Layout({qr1[0]: 0, qr1[1]: 1, qr2[0]: 2, qr2[1]: 3})
        circuit.measure(pqr[2], cr[0])
        circuit.measure(pqr[3], cr[1])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=True, with_layout=False)),
            expected,
        )

    def test_after_transpile(self):
        """After transpile, the drawing should include the layout"""
        expected = "\n".join(
            [
                "                 ┌─────────┐┌─────────┐┌───┐┌─────────┐┌─┐   ",
                "   userqr_0 -> 0 ┤ U2(0,π) ├┤ U2(0,π) ├┤ X ├┤ U2(0,π) ├┤M├───",
                "                 ├─────────┤├─────────┤└─┬─┘├─────────┤└╥┘┌─┐",
                "   userqr_1 -> 1 ┤ U2(0,π) ├┤ U2(0,π) ├──■──┤ U2(0,π) ├─╫─┤M├",
                "                 └─────────┘└─────────┘     └─────────┘ ║ └╥┘",
                "  ancilla_0 -> 2 ───────────────────────────────────────╫──╫─",
                "                                                        ║  ║ ",
                "  ancilla_1 -> 3 ───────────────────────────────────────╫──╫─",
                "                                                        ║  ║ ",
                "  ancilla_2 -> 4 ───────────────────────────────────────╫──╫─",
                "                                                        ║  ║ ",
                "  ancilla_3 -> 5 ───────────────────────────────────────╫──╫─",
                "                                                        ║  ║ ",
                "  ancilla_4 -> 6 ───────────────────────────────────────╫──╫─",
                "                                                        ║  ║ ",
                "  ancilla_5 -> 7 ───────────────────────────────────────╫──╫─",
                "                                                        ║  ║ ",
                "  ancilla_6 -> 8 ───────────────────────────────────────╫──╫─",
                "                                                        ║  ║ ",
                "  ancilla_7 -> 9 ───────────────────────────────────────╫──╫─",
                "                                                        ║  ║ ",
                " ancilla_8 -> 10 ───────────────────────────────────────╫──╫─",
                "                                                        ║  ║ ",
                " ancilla_9 -> 11 ───────────────────────────────────────╫──╫─",
                "                                                        ║  ║ ",
                "ancilla_10 -> 12 ───────────────────────────────────────╫──╫─",
                "                                                        ║  ║ ",
                "ancilla_11 -> 13 ───────────────────────────────────────╫──╫─",
                "                                                        ║  ║ ",
                "           c0_0: ═══════════════════════════════════════╩══╬═",
                "                                                           ║ ",
                "           c0_1: ══════════════════════════════════════════╩═",
                "                                                             ",
            ]
        )

        qr = QuantumRegister(2, "userqr")
        cr = ClassicalRegister(2, "c0")
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)
        qc.cx(qr[0], qr[1])
        qc.measure(qr, cr)

        coupling_map = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]
        qc_result = transpile(
            qc,
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            coupling_map=coupling_map,
            optimization_level=0,
            seed_transpiler=0,
        )
        self.assertEqual(qc_result.draw(output="text", cregbundle=False).single_string(), expected)


class TestTextInitialValue(QiskitTestCase):
    """Testing the initial_state parameter"""

    def setUp(self) -> None:
        super().setUp()
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        self.circuit = QuantumCircuit(qr, cr)
        self.circuit.measure(qr, cr)

    def test_draw_initial_value_default(self):
        """Text drawer (.draw) default initial_state parameter (False)."""
        expected = "\n".join(
            [
                "     ┌─┐   ",
                "q_0: ┤M├───",
                "     └╥┘┌─┐",
                "q_1: ─╫─┤M├",
                "      ║ └╥┘",
                "c_0: ═╩══╬═",
                "         ║ ",
                "c_1: ════╩═",
                "           ",
            ]
        )

        self.assertEqual(
            self.circuit.draw(output="text", cregbundle=False).single_string(), expected
        )

    def test_draw_initial_value_true(self):
        """Text drawer .draw(initial_state=True)."""
        expected = "\n".join(
            [
                "        ┌─┐   ",
                "q_0: |0>┤M├───",
                "        └╥┘┌─┐",
                "q_1: |0>─╫─┤M├",
                "         ║ └╥┘",
                " c_0: 0 ═╩══╬═",
                "            ║ ",
                " c_1: 0 ════╩═",
                "              ",
            ]
        )
        self.assertEqual(
            self.circuit.draw(output="text", initial_state=True, cregbundle=False).single_string(),
            expected,
        )

    def test_initial_value_false(self):
        """Text drawer with initial_state parameter False."""
        expected = "\n".join(
            [
                "     ┌─┐   ",
                "q_0: ┤M├───",
                "     └╥┘┌─┐",
                "q_1: ─╫─┤M├",
                "      ║ └╥┘",
                "c: 2/═╩══╩═",
                "      0  1 ",
            ]
        )

        self.assertEqual(
            str(circuit_drawer(self.circuit, output="text", initial_state=False)), expected
        )


class TestTextHamiltonianGate(QiskitTestCase):
    """Testing the Hamiltonian gate drawer"""

    def test_draw_hamiltonian_single(self):
        """Text Hamiltonian gate with single qubit."""
        # fmt: off
        expected = "\n".join(["    ┌─────────────┐",
                              "q0: ┤ Hamiltonian ├",
                              "    └─────────────┘"])
        # fmt: on
        qr = QuantumRegister(1, "q0")
        circuit = QuantumCircuit(qr)
        matrix = numpy.zeros((2, 2))
        theta = Parameter("theta")
        circuit.append(HamiltonianGate(matrix, theta), [qr[0]])
        circuit = circuit.assign_parameters({theta: 1})
        self.assertEqual(circuit.draw(output="text").single_string(), expected)

    def test_draw_hamiltonian_multi(self):
        """Text Hamiltonian gate with multiple qubits."""
        expected = "\n".join(
            [
                "      ┌──────────────┐",
                "q0_0: ┤0             ├",
                "      │  Hamiltonian │",
                "q0_1: ┤1             ├",
                "      └──────────────┘",
            ]
        )

        qr = QuantumRegister(2, "q0")
        circuit = QuantumCircuit(qr)
        matrix = numpy.zeros((4, 4))
        theta = Parameter("theta")
        circuit.append(HamiltonianGate(matrix, theta), [qr[0], qr[1]])
        circuit = circuit.assign_parameters({theta: 1})
        self.assertEqual(circuit.draw(output="text").single_string(), expected)


class TestTextPhase(QiskitTestCase):
    """Testing the drawing a circuit with phase"""

    def test_bell(self):
        """Text Bell state with phase."""
        expected = "\n".join(
            [
                "global phase: \u03C0/2",
                "     ┌───┐     ",
                "q_0: ┤ H ├──■──",
                "     └───┘┌─┴─┐",
                "q_1: ─────┤ X ├",
                "          └───┘",
            ]
        )

        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.global_phase = 3.141592653589793 / 2

        circuit.h(0)
        circuit.cx(0, 1)
        self.assertEqual(circuit.draw(output="text").single_string(), expected)

    def test_empty(self):
        """Text empty circuit (two registers) with phase."""
        # fmt: off
        expected = "\n".join(["global phase: 3",
                              "     ",
                              "q_0: ",
                              "     ",
                              "q_1: ",
                              "     "])
        # fmt: on
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.global_phase = 3

        self.assertEqual(circuit.draw(output="text").single_string(), expected)

    def test_empty_noregs(self):
        """Text empty circuit (no registers) with phase."""
        expected = "\n".join(["global phase: 4.21"])

        circuit = QuantumCircuit()
        circuit.global_phase = 4.21

        self.assertEqual(circuit.draw(output="text").single_string(), expected)

    def test_registerless_one_bit(self):
        """Text circuit with one-bit registers and registerless bits."""
        # fmt: off
        expected = "\n".join(["       ",
                              "qrx_0: ",
                              "       ",
                              "qrx_1: ",
                              "       ",
                              "    2: ",
                              "       ",
                              "    3: ",
                              "       ",
                              "  qry: ",
                              "       ",
                              "    0: ",
                              "       ",
                              "    1: ",
                              "       ",
                              "crx: 2/",
                              "       "])
        # fmt: on

        qrx = QuantumRegister(2, "qrx")
        qry = QuantumRegister(1, "qry")
        crx = ClassicalRegister(2, "crx")
        circuit = QuantumCircuit(qrx, [Qubit(), Qubit()], qry, [Clbit(), Clbit()], crx)
        self.assertEqual(circuit.draw(output="text", cregbundle=True).single_string(), expected)


class TestCircuitVisualizationImplementation(QiskitVisualizationTestCase):
    """Tests utf8 and cp437 encoding."""

    text_reference_utf8 = path_to_diagram_reference("circuit_text_ref_utf8.txt")
    text_reference_cp437 = path_to_diagram_reference("circuit_text_ref_cp437.txt")

    def sample_circuit(self):
        """Generate a sample circuit that includes the most common elements of
        quantum circuits.
        """
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.y(qr[0])
        circuit.z(qr[0])
        circuit.barrier(qr[0])
        circuit.barrier(qr[1])
        circuit.barrier(qr[2])
        circuit.h(qr[0])
        circuit.s(qr[0])
        circuit.sdg(qr[0])
        circuit.t(qr[0])
        circuit.tdg(qr[0])
        circuit.sx(qr[0])
        circuit.sxdg(qr[0])
        circuit.id(qr[0])
        circuit.reset(qr[0])
        circuit.rx(pi, qr[0])
        circuit.ry(pi, qr[0])
        circuit.rz(pi, qr[0])
        circuit.append(U1Gate(pi), [qr[0]])
        circuit.append(U2Gate(pi, pi), [qr[0]])
        circuit.append(U3Gate(pi, pi, pi), [qr[0]])
        circuit.swap(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cy(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])
        circuit.ch(qr[0], qr[1])
        circuit.append(CU1Gate(pi), [qr[0], qr[1]])
        circuit.append(CU3Gate(pi, pi, pi), [qr[0], qr[1]])
        circuit.crz(pi, qr[0], qr[1])
        circuit.cry(pi, qr[0], qr[1])
        circuit.crx(pi, qr[0], qr[1])
        circuit.ccx(qr[0], qr[1], qr[2])
        circuit.cswap(qr[0], qr[1], qr[2])
        circuit.measure(qr, cr)

        return circuit

    def test_text_drawer_utf8(self):
        """Test that text drawer handles utf8 encoding."""
        filename = "current_textplot_utf8.txt"
        qc = self.sample_circuit()
        output = _text_circuit_drawer(
            qc,
            filename=filename,
            fold=-1,
            initial_state=True,
            cregbundle=False,
            encoding="utf8",
        )
        try:
            encode(str(output), encoding="utf8")
        except UnicodeEncodeError:
            self.fail("_text_circuit_drawer() should be utf8.")
        self.assertFilesAreEqual(filename, self.text_reference_utf8, "utf8")
        os.remove(filename)

    def test_text_drawer_cp437(self):
        """Test that text drawer handles cp437 encoding."""
        filename = "current_textplot_cp437.txt"
        qc = self.sample_circuit()
        output = _text_circuit_drawer(
            qc,
            filename=filename,
            fold=-1,
            initial_state=True,
            cregbundle=False,
            encoding="cp437",
        )
        try:
            encode(str(output), encoding="cp437")
        except UnicodeEncodeError:
            self.fail("_text_circuit_drawer() should be cp437.")
        self.assertFilesAreEqual("current_textplot_cp437.txt", self.text_reference_cp437, "cp437")
        os.remove(filename)


class TestCircuitControlFlowOps(QiskitVisualizationTestCase):
    """Test ControlFlowOps."""

    def test_if_op_bundle_false(self):
        """Test an IfElseOp with if only and cregbundle false"""
        expected = "\n".join(
            [
                "      ┌────── ┌───┐      ───────┐ ",
                " q_0: ┤       ┤ H ├──■──        ├─",
                "      │ If-0  └───┘┌─┴─┐  End-0 │ ",
                " q_1: ┤       ─────┤ X ├        ├─",
                "      └──╥───      └───┘ ───────┘ ",
                " q_2: ───╫────────────────────────",
                "         ║                        ",
                " q_3: ───╫────────────────────────",
                "         ║                        ",
                "cr_0: ═══╬════════════════════════",
                "         ║                        ",
                "cr_1: ═══■════════════════════════",
                "                                  ",
            ]
        )

        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)

        with circuit.if_test((cr[1], 1)):
            circuit.h(0)
            circuit.cx(0, 1)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=False, cregbundle=False)),
            expected,
        )

    def test_if_op_bundle_true(self):
        """Test an IfElseOp with if only and cregbundle true"""
        expected = "\n".join(
            [
                "        ┌──────   ┌───┐      ───────┐ ",
                " q_0: ──┤       ──┤ H ├──■──        ├─",
                "        │ If-0    └───┘┌─┴─┐  End-0 │ ",
                " q_1: ──┤       ───────┤ X ├        ├─",
                "        └──╥───        └───┘ ───────┘ ",
                " q_2: ─────╫──────────────────────────",
                "           ║                          ",
                " q_3: ─────╫──────────────────────────",
                "      ┌────╨─────┐                    ",
                "cr: 2/╡ cr_1=0x1 ╞════════════════════",
                "      └──────────┘                    ",
            ]
        )

        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)

        with circuit.if_test((cr[1], 1)):
            circuit.h(0)
            circuit.cx(0, 1)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=False)),
            expected,
        )

    def test_if_else_with_body_specified(self):
        """Test an IfElseOp where the body is directly specified."""

        expected = "\n".join(
            [
                "      ┌───┐┌─┐             ┌────── ┌───┐     ┌─────┐ ───────┐ ┌─────┐",
                " q_0: ┤ H ├┤M├─────────────┤       ┤ Z ├─────┤ X1i ├        ├─┤ X1i ├",
                "      ├───┤└╥┘┌─┐          │       ├───┤┌───┐└──╥──┘        │ └─────┘",
                " q_1: ┤ H ├─╫─┤M├──────────┤ If-0  ┤ X ├┤ Y ├───╫───  End-0 ├────────",
                "      ├───┤ ║ └╥┘┌────────┐│       └───┘└───┘   ║           │        ",
                " q_2: ┤ X ├─╫──╫─┤ XLabel ├┤       ─────────────╫───        ├────────",
                "      └───┘ ║  ║ └───╥────┘└──╥───              ║    ───────┘        ",
                " q_3: ──────╫──╫─────╫────────╫─────────────────╫────────────────────",
                "            ║  ║     ║        ║                 ║                    ",
                "cr_0: ══════╬══╬═════o════════╬═════════════════o════════════════════",
                "            ║  ║     ║        ║                 ║                    ",
                "cr_1: ══════╩══╬═════■════════■═════════════════o════════════════════",
                "               ║     ║                          ║                    ",
                "cr_2: ═════════╩═════o══════════════════════════■════════════════════",
                "                    0x2                        0x4                   ",
            ]
        )
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(0)
        circuit.h(1)
        circuit.measure(0, 1)
        circuit.measure(1, 2)
        circuit.x(2)
        with self.assertWarns(DeprecationWarning):
            circuit.x(2, label="XLabel").c_if(cr, 2)

        qr2 = QuantumRegister(3, "qr2")
        circuit2 = QuantumCircuit(qr2, cr)
        circuit2.x(1)
        circuit2.y(1)
        circuit2.z(0)
        with self.assertWarns(DeprecationWarning):
            circuit2.x(0, label="X1i").c_if(cr, 4)

        circuit.if_else((cr[1], 1), circuit2, None, [0, 1, 2], [0, 1, 2])
        circuit.x(0, label="X1i")
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=False, cregbundle=False)),
            expected,
        )

    def test_if_op_nested_wire_order(self):
        """Test IfElseOp with nested if's and wire_order change."""
        expected = "\n".join(
            [
                "           ┌──────           ┌──────      ┌────── ┌───┐                     »",
                " q_2: ─────┤       ──────────┤       ─────┤       ┤ Z ├─────────────────────»",
                "      ┌───┐│       ┌────────┐│       ┌───┐│       └───┘┌──────      ┌────── »",
                " q_0: ┤ H ├┤       ┤ X c_if ├┤       ┤ Z ├┤       ─────┤       ──■──┤       »",
                "      └───┘│ If-0  └───╥────┘│ If-1  └───┘│ If-2       │         │  │       »",
                " q_3: ─────┤       ────╫─────┤       ─────┤       ─────┤ If-3  ──┼──┤ If-4  »",
                "           │           ║     │       ┌───┐│       ┌───┐│       ┌─┴─┐│       »",
                " q_1: ─────┤       ────╫─────┤       ┤ Y ├┤       ┤ Y ├┤       ┤ X ├┤       »",
                "           └──╥───     ║     └──╥─── └───┘└──╥─── └───┘└──╥─── └───┘└──╥─── »",
                "cr_0: ════════╬════════o════════╬════════════╬════════════╬════════════╬════»",
                "              ║        ║        ║            ║            ║            ║    »",
                "cr_1: ════════■════════o════════╬════════════■════════════╬════════════■════»",
                "                       ║        ║                         ║                 »",
                "cr_2: ═════════════════■════════■═════════════════════════■═════════════════»",
                "                      0x4                                                   »",
                "«                                ───────┐  ───────┐ ┌────────              »",
                "« q_2: ─────────────────────────        ├─        ├─┤         ─────────────»",
                "«      ┌───┐ ───────┐  ───────┐         │         │ │              ┌────── »",
                "« q_0: ┤ H ├        ├─        ├─        ├─        ├─┤         ─────┤       »",
                "«      └───┘        │         │   End-2 │   End-1 │ │ Else-0       │       »",
                "« q_3: ─────  End-4 ├─  End-3 ├─        ├─        ├─┤         ─────┤ If-1  »",
                "«      ┌───┐        │         │         │         │ │         ┌───┐│       »",
                "« q_1: ┤ X ├        ├─        ├─        ├─        ├─┤         ┤ Y ├┤       »",
                "«      └───┘ ───────┘  ───────┘  ───────┘  ───────┘ └──────── └───┘└──╥─── »",
                "«cr_0: ═══════════════════════════════════════════════════════════════╬════»",
                "«                                                                     ║    »",
                "«cr_1: ═══════════════════════════════════════════════════════════════╬════»",
                "«                                                                     ║    »",
                "«cr_2: ═══════════════════════════════════════════════════════════════■════»",
                "«                                                                          »",
                "«                               ───────┐      ",
                "« q_2: ────────────────────────        ├──────",
                "«      ┌───┐ ───────┐ ┌───────┐        │ ┌───┐",
                "« q_0: ┤ X ├        ├─┤0      ├        ├─┤ X ├",
                "«      └───┘        │ │       │  End-0 │ └───┘",
                "« q_3: ─────  End-1 ├─┤       ├        ├──────",
                "«      ┌───┐        │ │       │        │      ",
                "« q_1: ┤ X ├        ├─┤1 Inst ├        ├──────",
                "«      └───┘ ───────┘ │       │ ───────┘      ",
                "«cr_0: ═══════════════╡0      ╞═══════════════",
                "«                     │       │               ",
                "«cr_1: ═══════════════╡1      ╞═══════════════",
                "«                     └───────┘               ",
                "«cr_2: ═══════════════════════════════════════",
                "«                                             ",
            ]
        )
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)

        circuit.h(0)
        with circuit.if_test((cr[1], 1)) as _else:
            with self.assertWarns(DeprecationWarning):
                circuit.x(0, label="X c_if").c_if(cr, 4)
            with circuit.if_test((cr[2], 1)):
                circuit.z(0)
                circuit.y(1)
                with circuit.if_test((cr[1], 1)):
                    circuit.y(1)
                    circuit.z(2)
                    with circuit.if_test((cr[2], 1)):
                        circuit.cx(0, 1)
                        with circuit.if_test((cr[1], 1)):
                            circuit.h(0)
                            circuit.x(1)
        with _else:
            circuit.y(1)
            with circuit.if_test((cr[2], 1)):
                circuit.x(0)
                circuit.x(1)
            inst = QuantumCircuit(2, 2, name="Inst").to_instruction()
            circuit.append(inst, [qr[0], qr[1]], [cr[0], cr[1]])
        circuit.x(0)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit,
                    output="text",
                    fold=77,
                    initial_state=False,
                    wire_order=[2, 0, 3, 1, 4, 5, 6],
                )
            ),
            expected,
        )

    def test_while_loop(self):
        """Test WhileLoopOp."""
        expected = "\n".join(
            [
                "      ┌───┐┌─┐┌───────── ┌───┐     ┌─┐┌────── ┌───┐ ───────┐  ───────┐ ",
                " q_0: ┤ H ├┤M├┤          ┤ H ├──■──┤M├┤ If-1  ┤ X ├  End-1 ├─        ├─",
                "      └───┘└╥┘│ While-0  └───┘┌─┴─┐└╥┘└──╥─── └───┘ ───────┘   End-0 │ ",
                " q_1: ──────╫─┤          ─────┤ X ├─╫────╫───────────────────        ├─",
                "            ║ └────╥────      └───┘ ║    ║                    ───────┘ ",
                " q_2: ──────╫──────╫────────────────╫────╫─────────────────────────────",
                "            ║      ║                ║    ║                             ",
                " q_3: ──────╫──────╫────────────────╫────╫─────────────────────────────",
                "            ║      ║                ║    ║                             ",
                "cr_0: ══════╬══════o════════════════╩════╬═════════════════════════════",
                "            ║                            ║                             ",
                "cr_1: ══════╬════════════════════════════╬═════════════════════════════",
                "            ║                            ║                             ",
                "cr_2: ══════╩════════════════════════════■═════════════════════════════",
                "                                                                       ",
            ]
        )

        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)

        circuit.h(0)
        circuit.measure(0, 2)
        with circuit.while_loop((cr[0], 0)):
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.measure(0, 0)
            with circuit.if_test((cr[2], 1)):
                circuit.x(0)
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=False, cregbundle=False)),
            expected,
        )

    def test_for_loop(self):
        """Test ForLoopOp."""
        expected = "\n".join(
            [
                "      ┌───┐┌─┐┌─────────────────                 ┌─┐┌────── ┌───┐ ───────┐  ───────┐ ",
                " q_0: ┤ H ├┤M├┤                  ──■─────────────┤M├┤ If-1  ┤ Z ├  End-1 ├─        ├─",
                "      └───┘└╥┘│ For-0 (2, 4, 8)  ┌─┴─┐┌─────────┐└╥┘└──╥─── └───┘ ───────┘   End-0 │ ",
                " q_1: ──────╫─┤                  ┤ X ├┤ Rx(π/a) ├─╫────╫───────────────────        ├─",
                "            ║ └───────────────── └───┘└─────────┘ ║    ║                    ───────┘ ",
                " q_2: ──────╫─────────────────────────────────────╫────╫─────────────────────────────",
                "            ║                                     ║    ║                             ",
                " q_3: ──────╫─────────────────────────────────────╫────╫─────────────────────────────",
                "            ║                                     ║    ║                             ",
                "cr_0: ══════╬═════════════════════════════════════╩════╬═════════════════════════════",
                "            ║                                          ║                             ",
                "cr_1: ══════╬══════════════════════════════════════════╬═════════════════════════════",
                "            ║                                          ║                             ",
                "cr_2: ══════╩══════════════════════════════════════════■═════════════════════════════",
                "                                                                                     ",
            ]
        )

        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)

        a = Parameter("a")
        circuit.h(0)
        circuit.measure(0, 2)
        with circuit.for_loop((2, 4, 8), loop_parameter=a):
            circuit.cx(0, 1)
            circuit.rx(pi / a, 1)
            circuit.measure(0, 0)
            with circuit.if_test((cr[2], 1)):
                circuit.z(0)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit, output="text", fold=-1, initial_state=False, cregbundle=False
                )
            ),
            expected,
        )

    def test_switch_case(self):
        """Test SwitchCaseOp."""
        expected = "\n".join(
            [
                "      ┌───┐┌─┐      ┌────────── ┌────────────────── ┌───┐┌────────────────── »",
                " q_0: ┤ H ├┤M├──────┤           ┤                   ┤ X ├┤                   »",
                "      ├───┤└╥┘┌─┐   │ Switch-0  │ Case-0 (0, 1, 2)  ├───┤│ Case-0 (3, 4, 5)  »",
                " q_1: ┤ H ├─╫─┤M├───┤           ┤                   ┤ X ├┤                   »",
                "      ├───┤ ║ └╥┘┌─┐└────╥───── └────────────────── └───┘└────────────────── »",
                " q_2: ┤ H ├─╫──╫─┤M├─────╫───────────────────────────────────────────────────»",
                "      └───┘ ║  ║ └╥┘     ║                                                   »",
                "cr_0: ══════╩══╬══╬══════■═══════════════════════════════════════════════════»",
                "               ║  ║      ║                                                   »",
                "cr_1: ═════════╩══╬══════■═══════════════════════════════════════════════════»",
                "                  ║      ║                                                   »",
                "cr_2: ════════════╩══════■═══════════════════════════════════════════════════»",
                "                        0x7                                                  »",
                "«      ┌───┐┌───┐┌────────────────       ───────┐ ┌───┐",
                "« q_0: ┤ Y ├┤ Y ├┤                 ──■──        ├─┤ H ├",
                "«      ├───┤└───┘│ Case-0 default  ┌─┴─┐  End-0 │ └───┘",
                "« q_1: ┤ Y ├─────┤                 ┤ X ├        ├──────",
                "«      └───┘     └──────────────── └───┘ ───────┘      ",
                "« q_2: ────────────────────────────────────────────────",
                "«                                                      ",
                "«cr_0: ════════════════════════════════════════════════",
                "«                                                      ",
                "«cr_1: ════════════════════════════════════════════════",
                "«                                                      ",
                "«cr_2: ════════════════════════════════════════════════",
                "«                                                      ",
            ]
        )

        qreg = QuantumRegister(3, "q")
        creg = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qreg, creg)

        circuit.h([0, 1, 2])
        circuit.measure([0, 1, 2], [0, 1, 2])

        with circuit.switch(creg) as case:
            with case(0, 1, 2):
                circuit.x(0)
                circuit.x(1)
            with case(3, 4, 5):
                circuit.y(1)
                circuit.y(0)
                circuit.y(0)
            with case(case.DEFAULT):
                circuit.cx(0, 1)
        circuit.h(0)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit, output="text", fold=78, initial_state=False, cregbundle=False
                )
            ),
            expected,
        )

    def test_inner_wire_map_control_op(self):
        """Test that the gates inside ControlFlowOps land on correct qubits when transpiled"""
        expected = "\n".join(
            [
                "                                                                  ",
                "     qr_1 -> 0 ───────────────────────────────────────────────────",
                "                                                                  ",
                "ancilla_0 -> 1 ───────────────────────────────────────────────────",
                "               ┌────── ┌────────┐┌────── ┌───┐ ───────┐  ───────┐ ",
                "     qr_0 -> 2 ┤ If-0  ┤ Rz(-π) ├┤ If-1  ┤ X ├  End-1 ├─  End-0 ├─",
                "               └──╥─── └────────┘└──╥─── └───┘ ───────┘  ───────┘ ",
                "ancilla_1 -> 3 ───╫─────────────────╫─────────────────────────────",
                "                  ║                 ║                             ",
                "ancilla_2 -> 4 ───╫─────────────────╫─────────────────────────────",
                "                  ║                 ║                             ",
                "         cr_0: ═══o═════════════════╬═════════════════════════════",
                "                  ║                 ║                             ",
                "         cr_1: ═══■═════════════════■═════════════════════════════",
                "                 0x2                                              ",
            ]
        )

        qreg = QuantumRegister(2, "qr")
        creg = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qreg, creg)

        with qc.if_test((creg, 2)):
            qc.z(0)
            with qc.if_test((creg[1], 1)):
                qc.x(0)

        backend = GenericBackendV2(num_qubits=5, coupling_map=YORKTOWN_CMAP, seed=42)
        backend.target.add_instruction(IfElseOp, name="if_else")

        circuit = transpile(qc, backend, optimization_level=2, seed_transpiler=671_42)
        self.assertEqual(
            str(
                circuit_drawer(
                    circuit, output="text", fold=78, initial_state=False, cregbundle=False
                )
            ),
            expected,
        )

    def test_if_else_op_from_circuit_with_conditions(self):
        """Test an IfElseOp built from circuit with conditions inside the if using inner creg"""
        expected = "\n".join(
            [
                "      ┌───┐┌────── ┌────┐       ───────┐ ",
                " q_0: ┤ H ├┤       ┤ X1 ├──────        ├─",
                "      └───┘│       └─╥──┘┌────┐        │ ",
                " q_1: ─────┤ If-0  ──╫───┤ X2 ├  End-0 ├─",
                "      ┌───┐│         ║   └─╥──┘        │ ",
                " q_2: ┤ X ├┤       ──╫─────╫───        ├─",
                "      └─╥─┘└──╥───   ║     ║    ───────┘ ",
                " q_3: ──╫─────╫──────╫─────╫─────────────",
                "        ║     ║      ║     ║             ",
                "cr_0: ══╬═════╬══════o═════╬═════════════",
                "        ║     ║      ║     ║             ",
                "cr_1: ══■═════■══════o═════■═════════════",
                "                     ║                   ",
                "cr_2: ═══════════════■═══════════════════",
                "                    0x4                  ",
            ]
        )

        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(0)
        with self.assertWarns(DeprecationWarning):
            circuit.x(2).c_if(cr[1], 2)

        qr2 = QuantumRegister(3, "qr2")
        qc2 = QuantumCircuit(qr2, cr)
        with self.assertWarns(DeprecationWarning):
            qc2.x(0, label="X1").c_if(cr, 4)
        with self.assertWarns(DeprecationWarning):
            qc2.x(1, label="X2").c_if(cr[1], 1)

        circuit.if_else((cr[1], 1), qc2, None, [0, 1, 2], [0, 1, 2])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=False, cregbundle=False)),
            expected,
        )

    def test_if_with_expr(self):
        """Test an IfElseOp with an expression"""
        expected = "\n".join(
            [
                "       ┌───┐┌─────────────────────────────── ┌───┐ ───────┐ ",
                " qr_0: ┤ H ├┤ If-0 (cr1 & (cr2 & cr3)) == 3  ┤ Z ├  End-0 ├─",
                "       └───┘└───────────────╥─────────────── └───┘ ───────┘ ",
                " qr_1: ─────────────────────╫───────────────────────────────",
                "                            ║                               ",
                " qr_2: ─────────────────────╫───────────────────────────────",
                "                            ║                               ",
                " cr: 3/═════════════════════╬═══════════════════════════════",
                "                        ┌───╨────┐                          ",
                "cr1: 3/═════════════════╡ [expr] ╞══════════════════════════",
                "                        ├───╨────┤                          ",
                "cr2: 3/═════════════════╡ [expr] ╞══════════════════════════",
                "                        ├───╨────┤                          ",
                "cr3: 3/═════════════════╡ [expr] ╞══════════════════════════",
                "                        └────────┘                          ",
            ]
        )
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")
        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")
        cr3 = ClassicalRegister(3, "cr3")
        circuit = QuantumCircuit(qr, cr, cr1, cr2, cr3)

        circuit.h(0)
        with circuit.if_test(expr.equal(expr.bit_and(cr1, expr.bit_and(cr2, cr3)), 3)):
            circuit.z(0)

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=False)),
            expected,
        )

    def test_if_with_expr_nested(self):
        """Test an IfElseOp with an expression for nested"""
        expected = "\n".join(
            [
                "       ┌───┐┌─────────────────────── ┌───┐                                 ───────┐ ",
                " qr_0: ┤ H ├┤                        ┤ X ├────────────────────────────────        ├─",
                "       └───┘│ If-0 (cr2 & cr3) == 3  └───┘┌─────────────── ┌───┐ ───────┐   End-0 │ ",
                " qr_1: ─────┤                        ─────┤ If-1 cr2 == 5  ┤ Z ├  End-1 ├─        ├─",
                "            └───────────╥───────────      └───────╥─────── └───┘ ───────┘  ───────┘ ",
                " qr_2: ─────────────────╫─────────────────────────╫─────────────────────────────────",
                "                        ║                         ║                                 ",
                " cr: 3/═════════════════╬═════════════════════════╬═════════════════════════════════",
                "                        ║                         ║                                 ",
                "cr1: 3/═════════════════╬═════════════════════════╬═════════════════════════════════",
                "                    ┌───╨────┐                ┌───╨────┐                            ",
                "cr2: 3/═════════════╡ [expr] ╞════════════════╡ [expr] ╞════════════════════════════",
                "                    ├───╨────┤                └────────┘                            ",
                "cr3: 3/═════════════╡ [expr] ╞══════════════════════════════════════════════════════",
                "                    └────────┘                                                      ",
            ]
        )
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")
        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")
        cr3 = ClassicalRegister(3, "cr3")
        circuit = QuantumCircuit(qr, cr, cr1, cr2, cr3)

        circuit.h(0)
        with circuit.if_test(expr.equal(expr.bit_and(cr2, cr3), 3)):
            circuit.x(0)
            with circuit.if_test(expr.equal(cr2, 5)):
                circuit.z(1)

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=False, fold=120)),
            expected,
        )

    def test_switch_with_expression(self):
        """Test an SwitchcaseOp with an expression"""
        expected = "\n".join(
            [
                "       ┌───┐┌──────────────────────────── ┌───────────────────── ┌───┐»",
                " qr_0: ┤ H ├┤                             ┤                      ┤ X ├»",
                "       └───┘│ Switch-0 cr1 & (cr2 & cr3)  │ Case-0 (0, 1, 2, 3)  └───┘»",
                " qr_1: ─────┤                             ┤                      ─────»",
                "            └─────────────╥────────────── └─────────────────────      »",
                " qr_2: ───────────────────╫───────────────────────────────────────────»",
                "                          ║                                           »",
                " cr: 3/═══════════════════╬═══════════════════════════════════════════»",
                "                      ┌───╨────┐                                      »",
                "cr1: 3/═══════════════╡ [expr] ╞══════════════════════════════════════»",
                "                      ├───╨────┤                                      »",
                "cr2: 3/═══════════════╡ [expr] ╞══════════════════════════════════════»",
                "                      ├───╨────┤                                      »",
                "cr3: 3/═══════════════╡ [expr] ╞══════════════════════════════════════»",
                "                      └────────┘                                      »",
                "«       ┌────────────────       ───────┐ ",
                "« qr_0: ┤                 ──■──        ├─",
                "«       │ Case-0 default  ┌─┴─┐  End-0 │ ",
                "« qr_1: ┤                 ┤ X ├        ├─",
                "«       └──────────────── └───┘ ───────┘ ",
                "« qr_2: ─────────────────────────────────",
                "«                                        ",
                "« cr: 3/═════════════════════════════════",
                "«                                        ",
                "«cr1: 3/═════════════════════════════════",
                "«                                        ",
                "«cr2: 3/═════════════════════════════════",
                "«                                        ",
                "«cr3: 3/═════════════════════════════════",
                "«                                        ",
            ]
        )
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")
        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")
        cr3 = ClassicalRegister(3, "cr3")
        circuit = QuantumCircuit(qr, cr, cr1, cr2, cr3)

        circuit.h(0)
        with circuit.switch(expr.bit_and(cr1, expr.bit_and(cr2, cr3))) as case:
            with case(0, 1, 2, 3):
                circuit.x(0)
            with case(case.DEFAULT):
                circuit.cx(0, 1)

        self.assertEqual(
            str(circuit_drawer(circuit, output="text", fold=80, initial_state=False)),
            expected,
        )

    def test_nested_if_else_op_var(self):
        """Test if/else with standalone Var."""
        expected = "\n".join(
            [
                "     ┌───────── ┌────────────────       ───────┐ ┌──────────────────── ┌───┐ ───────┐  ───────┐ ",
                "q_0: ┤          ┤                 ──■──        ├─┤ If-1 c && a == 128  ┤ H ├  End-1 ├─        ├─",
                "     │ If-0 !b  │ If-1 b == c[0]  ┌─┴─┐  End-1 │ └──────────────────── └───┘ ───────┘   End-0 │ ",
                "q_1: ┤          ┤                 ┤ X ├        ├──────────────────────────────────────        ├─",
                "     └───────── └───────╥──────── └───┘ ───────┘                                       ───────┘ ",
                "                    ┌───╨────┐                                                                  ",
                "c: 2/═══════════════╡ [expr] ╞══════════════════════════════════════════════════════════════════",
                "                    └────────┘                                                                  ",
            ]
        )
        a = expr.Var.new("a", types.Uint(8))
        qc = QuantumCircuit(2, 2, inputs=[a])
        b = qc.add_var("b", False)
        qc.store(a, 128)
        with qc.if_test(expr.logic_not(b)):
            # Mix old-style and new-style.
            with qc.if_test(expr.equal(b, qc.clbits[0])):
                qc.cx(0, 1)
            c = qc.add_var("c", b)
            with qc.if_test(expr.logic_and(c, expr.equal(a, 128))):
                qc.h(0)

        actual = str(qc.draw("text", fold=-1, initial_state=False))
        self.assertEqual(actual, expected)

    def test_nested_switch_op_var(self):
        """Test switch with standalone Var."""
        expected = "\n".join(
            [
                "     ┌───────────── ┌──────────── ┌──────────── ┌────────────      »",
                "q_0: ┤              ┤             ┤             ┤             ──■──»",
                "     │ Switch-0 ~a  │ Case-0 (0)  │ Switch-1 b  │ Case-1 (2)  ┌─┴─┐»",
                "q_1: ┤              ┤             ┤             ┤             ┤ X ├»",
                "     └───────────── └──────────── └──────────── └──────────── └───┘»",
                "c: 2/══════════════════════════════════════════════════════════════»",
                "                                                                   »",
                "«     ┌──────────────── ┌───┐ ───────┐ ┌──────────────── ┌──────── ┌───┐»",
                "«q_0: ┤                 ┤ X ├        ├─┤                 ┤ If-1 c  ┤ H ├»",
                "«     │ Case-1 default  └─┬─┘  End-1 │ │ Case-0 default  └──────── └───┘»",
                "«q_1: ┤                 ──■──        ├─┤                 ───────────────»",
                "«     └────────────────       ───────┘ └────────────────                »",
                "«c: 2/══════════════════════════════════════════════════════════════════»",
                "«                                                                       »",
                "«      ───────┐  ───────┐ ",
                "«q_0:   End-1 ├─        ├─",
                "«      ───────┘   End-0 │ ",
                "«q_1: ──────────        ├─",
                "«                ───────┘ ",
                "«c: 2/════════════════════",
                "«                         ",
            ]
        )

        a = expr.Var.new("a", types.Uint(8))
        qc = QuantumCircuit(2, 2, inputs=[a])
        b = qc.add_var("b", expr.lift(5, a.type))
        with qc.switch(expr.bit_not(a)) as case:
            with case(0):
                with qc.switch(b) as case2:
                    with case2(2):
                        qc.cx(0, 1)
                    with case2(case2.DEFAULT):
                        qc.cx(1, 0)
            with case(case.DEFAULT):
                c = qc.add_var("c", expr.equal(a, b))
                with qc.if_test(c):
                    qc.h(0)
        actual = str(qc.draw("text", fold=80, initial_state=False))
        self.assertEqual(actual, expected)


class TestCircuitAnnotatedOperations(QiskitVisualizationTestCase):
    """Test AnnotatedOperations and other non-Instructions."""

    def test_annotated_operation(self):
        """Test AnnotatedOperation and other non-Instructions."""
        expected = "\n".join(
            [
                "     ┌───────────┐┌───┐                                     ",
                "q_0: ┤0          ├┤ X ├──■────■────────────■────────────────",
                "     │  Clifford │├───┤┌─┴─┐  │            │          ┌────┐",
                "q_1: ┤1          ├┤ H ├┤ S ├──■────────────o──────────┤ √X ├",
                "     └───────────┘└───┘└─┬─┘┌─┴─┐┌─────────┴─────────┐└────┘",
                "q_2: ────────────────────o──┤ X ├┤ S - Inv, Pow(3.3) ├──────",
                "                            └───┘└───────────────────┘      ",
            ]
        )
        circuit = QuantumCircuit(3)
        cliff = random_clifford(2)
        circuit.append(cliff, [0, 1])
        circuit.x(0)
        circuit.h(1)
        circuit.append(SGate().control(2, ctrl_state=1), [0, 2, 1])
        circuit.ccx(0, 1, 2)
        op1 = AnnotatedOperation(
            SGate(), [InverseModifier(), ControlModifier(2, 1), PowerModifier(3.29)]
        )
        circuit.append(op1, [0, 1, 2])
        circuit.append(SXGate(), [1])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=False)),
            expected,
        )

    def test_annotated_multi_qubit(self):
        """Test AnnotatedOperation and other non-Instructions."""
        expected = "\n".join(
            [
                "                  ",
                "q_0: ──────o──────",
                "     ┌─────┴─────┐",
                "q_1: ┤0          ├",
                "     │  Cx - Inv │",
                "q_2: ┤1          ├",
                "     └─────┬─────┘",
                "q_3: ──────■──────",
                "                  ",
            ]
        )
        gate = AnnotatedOperation(CXGate(), [ControlModifier(2, 2), InverseModifier()])
        circuit = QuantumCircuit(gate.num_qubits)
        circuit.append(gate, [0, 3, 1, 2])
        self.assertEqual(
            str(circuit_drawer(circuit, output="text", initial_state=False)),
            expected,
        )


if __name__ == "__main__":
    unittest.main()

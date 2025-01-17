# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for visualization tools."""

import unittest
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Qubit, Clbit
from qiskit.visualization.circuit import _utils
from qiskit.visualization import array_to_latex
from qiskit.utils import optionals
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestVisualizationUtils(QiskitTestCase):
    """Tests for circuit drawer utilities."""

    def setUp(self):
        super().setUp()
        self.qr1 = QuantumRegister(2, "qr1")
        self.qr2 = QuantumRegister(2, "qr2")
        self.cr1 = ClassicalRegister(2, "cr1")
        self.cr2 = ClassicalRegister(2, "cr2")

        self.circuit = QuantumCircuit(self.qr1, self.qr2, self.cr1, self.cr2)
        self.circuit.cx(self.qr2[0], self.qr2[1])
        self.circuit.measure(self.qr2[0], self.cr2[0])
        self.circuit.cx(self.qr2[1], self.qr2[0])
        self.circuit.measure(self.qr2[1], self.cr2[1])
        self.circuit.cx(self.qr1[0], self.qr1[1])
        self.circuit.measure(self.qr1[0], self.cr1[0])
        self.circuit.cx(self.qr1[1], self.qr1[0])
        self.circuit.measure(self.qr1[1], self.cr1[1])

    def test_get_layered_instructions(self):
        """_get_layered_instructions without reverse_bits"""
        (qregs, cregs, layered_ops) = _utils._get_layered_instructions(self.circuit)

        exp = [
            {("cx", (self.qr2[0], self.qr2[1]), ()), ("cx", (self.qr1[0], self.qr1[1]), ())},
            {("measure", (self.qr2[0],), (self.cr2[0],))},
            {("measure", (self.qr1[0],), (self.cr1[0],))},
            {("cx", (self.qr2[1], self.qr2[0]), ()), ("cx", (self.qr1[1], self.qr1[0]), ())},
            {("measure", (self.qr2[1],), (self.cr2[1],))},
            {("measure", (self.qr1[1],), (self.cr1[1],))},
        ]

        self.assertEqual([self.qr1[0], self.qr1[1], self.qr2[0], self.qr2[1]], qregs)
        self.assertEqual([self.cr1[0], self.cr1[1], self.cr2[0], self.cr2[1]], cregs)
        self.assertEqual(
            exp, [{(op.name, op.qargs, op.cargs) for op in ops} for ops in layered_ops]
        )

    def test_get_layered_instructions_reverse_bits(self):
        """_get_layered_instructions with reverse_bits=True"""
        (qregs, cregs, layered_ops) = _utils._get_layered_instructions(
            self.circuit, reverse_bits=True
        )

        exp = [
            {("cx", (self.qr2[0], self.qr2[1]), ()), ("cx", (self.qr1[0], self.qr1[1]), ())},
            {("measure", (self.qr2[0],), (self.cr2[0],))},
            {("measure", (self.qr1[0],), (self.cr1[0],)), ("cx", (self.qr2[1], self.qr2[0]), ())},
            {("cx", (self.qr1[1], self.qr1[0]), ())},
            {("measure", (self.qr2[1],), (self.cr2[1],))},
            {("measure", (self.qr1[1],), (self.cr1[1],))},
        ]

        self.assertEqual([self.qr2[1], self.qr2[0], self.qr1[1], self.qr1[0]], qregs)
        self.assertEqual([self.cr2[1], self.cr2[0], self.cr1[1], self.cr1[0]], cregs)
        self.assertEqual(
            exp, [{(op.name, op.qargs, op.cargs) for op in ops} for ops in layered_ops]
        )

    def test_get_layered_instructions_remove_idle_wires(self):
        """_get_layered_instructions with idle_wires=False"""
        qr1 = QuantumRegister(3, "qr1")
        qr2 = QuantumRegister(3, "qr2")
        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")

        circuit = QuantumCircuit(qr1, qr2, cr1, cr2)
        circuit.cx(qr2[0], qr2[1])
        circuit.measure(qr2[0], cr2[0])
        circuit.cx(qr2[1], qr2[0])
        circuit.measure(qr2[1], cr2[1])
        circuit.cx(qr1[0], qr1[1])
        circuit.measure(qr1[0], cr1[0])
        circuit.cx(qr1[1], qr1[0])
        circuit.measure(qr1[1], cr1[1])

        (qregs, cregs, layered_ops) = _utils._get_layered_instructions(circuit, idle_wires=False)

        exp = [
            {("cx", (qr2[0], qr2[1]), ()), ("cx", (qr1[0], qr1[1]), ())},
            {("measure", (qr2[0],), (cr2[0],))},
            {("measure", (qr1[0],), (cr1[0],))},
            {("cx", (qr2[1], qr2[0]), ()), ("cx", (qr1[1], qr1[0]), ())},
            {("measure", (qr2[1],), (cr2[1],))},
            {("measure", (qr1[1],), (cr1[1],))},
        ]

        self.assertEqual([qr1[0], qr1[1], qr2[0], qr2[1]], qregs)
        self.assertEqual([cr1[0], cr1[1], cr2[0], cr2[1]], cregs)
        self.assertEqual(
            exp, [{(op.name, op.qargs, op.cargs) for op in ops} for ops in layered_ops]
        )

    def test_get_layered_instructions_left_justification_simple(self):
        """Test _get_layered_instructions left justification simple since #2802
        q_0: |0>───────■──
                ┌───┐  │
        q_1: |0>┤ H ├──┼──
                ├───┤  │
        q_2: |0>┤ H ├──┼──
                └───┘┌─┴─┐
        q_3: |0>─────┤ X ├
                     └───┘
        """
        qc = QuantumCircuit(4)
        qc.h(1)
        qc.h(2)
        qc.cx(0, 3)

        (_, _, layered_ops) = _utils._get_layered_instructions(qc, justify="left")

        l_exp = [
            {
                ("h", (Qubit(QuantumRegister(4, "q"), 1),), ()),
                ("h", (Qubit(QuantumRegister(4, "q"), 2),), ()),
            },
            {("cx", (Qubit(QuantumRegister(4, "q"), 0), Qubit(QuantumRegister(4, "q"), 3)), ())},
        ]

        self.assertEqual(
            l_exp, [{(op.name, op.qargs, op.cargs) for op in ops} for ops in layered_ops]
        )

    def test_get_layered_instructions_right_justification_simple(self):
        """Test _get_layered_instructions right justification simple since #2802
        q_0: |0>──■───────
                  │  ┌───┐
        q_1: |0>──┼──┤ H ├
                  │  ├───┤
        q_2: |0>──┼──┤ H ├
                ┌─┴─┐└───┘
        q_3: |0>┤ X ├─────
                └───┘
        """
        qc = QuantumCircuit(4)
        qc.h(1)
        qc.h(2)
        qc.cx(0, 3)

        (_, _, layered_ops) = _utils._get_layered_instructions(qc, justify="right")

        r_exp = [
            {("cx", (Qubit(QuantumRegister(4, "q"), 0), Qubit(QuantumRegister(4, "q"), 3)), ())},
            {
                ("h", (Qubit(QuantumRegister(4, "q"), 1),), ()),
                ("h", (Qubit(QuantumRegister(4, "q"), 2),), ()),
            },
        ]

        self.assertEqual(
            r_exp, [{(op.name, op.qargs, op.cargs) for op in ops} for ops in layered_ops]
        )

    def test_get_layered_instructions_left_justification_less_simple(self):
        """Test _get_layered_instructions left justification
        less simple example since #2802
                ┌────────────┐┌───┐┌────────────┐              ┌─┐┌────────────┐┌───┐┌────────────┐
        q_0: |0>┤ U2(0,pi/1) ├┤ X ├┤ U2(0,pi/1) ├──────────────┤M├┤ U2(0,pi/1) ├┤ X ├┤ U2(0,pi/1) ├
                ├────────────┤└─┬─┘├────────────┤┌────────────┐└╥┘└────────────┘└─┬─┘├────────────┤
        q_1: |0>┤ U2(0,pi/1) ├──■──┤ U2(0,pi/1) ├┤ U2(0,pi/1) ├─╫─────────────────■──┤ U2(0,pi/1) ├
                └────────────┘     └────────────┘└────────────┘ ║                    └────────────┘
        q_2: |0>────────────────────────────────────────────────╫──────────────────────────────────
                                                                ║
        q_3: |0>────────────────────────────────────────────────╫──────────────────────────────────
                                                                ║
        q_4: |0>────────────────────────────────────────────────╫──────────────────────────────────
                                                                ║
        c1_0: 0 ════════════════════════════════════════════════╩══════════════════════════════════
        """
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[5];
        creg c1[1];
        u2(0,3.14159265358979) q[0];
        u2(0,3.14159265358979) q[1];
        cx q[1],q[0];
        u2(0,3.14159265358979) q[0];
        u2(0,3.14159265358979) q[1];
        u2(0,3.14159265358979) q[1];
        measure q[0] -> c1[0];
        u2(0,3.14159265358979) q[0];
        cx q[1],q[0];
        u2(0,3.14159265358979) q[0];
        u2(0,3.14159265358979) q[1];
        """
        qc = QuantumCircuit.from_qasm_str(qasm)

        (_, _, layered_ops) = _utils._get_layered_instructions(qc, justify="left")

        l_exp = [
            {
                ("u2", (Qubit(QuantumRegister(5, "q"), 0),), ()),
                ("u2", (Qubit(QuantumRegister(5, "q"), 1),), ()),
            },
            {("cx", (Qubit(QuantumRegister(5, "q"), 1), Qubit(QuantumRegister(5, "q"), 0)), ())},
            {
                ("u2", (Qubit(QuantumRegister(5, "q"), 0),), ()),
                ("u2", (Qubit(QuantumRegister(5, "q"), 1),), ()),
            },
            {("u2", (Qubit(QuantumRegister(5, "q"), 1),), ())},
            {
                (
                    "measure",
                    (Qubit(QuantumRegister(5, "q"), 0),),
                    (Clbit(ClassicalRegister(1, "c1"), 0),),
                )
            },
            {("u2", (Qubit(QuantumRegister(5, "q"), 0),), ())},
            {("cx", (Qubit(QuantumRegister(5, "q"), 1), Qubit(QuantumRegister(5, "q"), 0)), ())},
            {
                ("u2", (Qubit(QuantumRegister(5, "q"), 0),), ()),
                ("u2", (Qubit(QuantumRegister(5, "q"), 1),), ()),
            },
        ]

        self.assertEqual(
            l_exp, [{(op.name, op.qargs, op.cargs) for op in ops} for ops in layered_ops]
        )

    def test_get_layered_instructions_right_justification_less_simple(self):
        """Test _get_layered_instructions right justification
        less simple example since #2802
                ┌────────────┐┌───┐┌────────────┐┌─┐┌────────────┐┌───┐┌────────────┐
        q_0: |0>┤ U2(0,pi/1) ├┤ X ├┤ U2(0,pi/1) ├┤M├┤ U2(0,pi/1) ├┤ X ├┤ U2(0,pi/1) ├
                ├────────────┤└─┬─┘├────────────┤└╥┘├────────────┤└─┬─┘├────────────┤
        q_1: |0>┤ U2(0,pi/1) ├──■──┤ U2(0,pi/1) ├─╫─┤ U2(0,pi/1) ├──■──┤ U2(0,pi/1) ├
                └────────────┘     └────────────┘ ║ └────────────┘     └────────────┘
        q_2: |0>──────────────────────────────────╫──────────────────────────────────
                                                  ║
        q_3: |0>──────────────────────────────────╫──────────────────────────────────
                                                  ║
        q_4: |0>──────────────────────────────────╫──────────────────────────────────
                                                  ║
        c1_0: 0 ══════════════════════════════════╩══════════════════════════════════
        """
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[5];
        creg c1[1];
        u2(0,3.14159265358979) q[0];
        u2(0,3.14159265358979) q[1];
        cx q[1],q[0];
        u2(0,3.14159265358979) q[0];
        u2(0,3.14159265358979) q[1];
        u2(0,3.14159265358979) q[1];
        measure q[0] -> c1[0];
        u2(0,3.14159265358979) q[0];
        cx q[1],q[0];
        u2(0,3.14159265358979) q[0];
        u2(0,3.14159265358979) q[1];
        """
        qc = QuantumCircuit.from_qasm_str(qasm)

        (_, _, layered_ops) = _utils._get_layered_instructions(qc, justify="right")

        r_exp = [
            {
                ("u2", (Qubit(QuantumRegister(5, "q"), 0),), ()),
                ("u2", (Qubit(QuantumRegister(5, "q"), 1),), ()),
            },
            {("cx", (Qubit(QuantumRegister(5, "q"), 1), Qubit(QuantumRegister(5, "q"), 0)), ())},
            {
                ("u2", (Qubit(QuantumRegister(5, "q"), 0),), ()),
                ("u2", (Qubit(QuantumRegister(5, "q"), 1),), ()),
            },
            {
                (
                    "measure",
                    (Qubit(QuantumRegister(5, "q"), 0),),
                    (Clbit(ClassicalRegister(1, "c1"), 0),),
                )
            },
            {
                ("u2", (Qubit(QuantumRegister(5, "q"), 0),), ()),
                ("u2", (Qubit(QuantumRegister(5, "q"), 1),), ()),
            },
            {("cx", (Qubit(QuantumRegister(5, "q"), 1), Qubit(QuantumRegister(5, "q"), 0)), ())},
            {
                ("u2", (Qubit(QuantumRegister(5, "q"), 0),), ()),
                ("u2", (Qubit(QuantumRegister(5, "q"), 1),), ()),
            },
        ]

        self.assertEqual(
            r_exp, [{(op.name, op.qargs, op.cargs) for op in ops} for ops in layered_ops]
        )

    def test_get_layered_instructions_op_with_cargs(self):
        """Test _get_layered_instructions op with cargs right of measure
                ┌───┐┌─┐
        q_0: |0>┤ H ├┤M├─────────────
                └───┘└╥┘┌───────────┐
        q_1: |0>──────╫─┤0          ├
                      ║ │  add_circ │
         c_0: 0 ══════╩═╡0          ╞
                        └───────────┘
         c_1: 0 ═════════════════════
        """
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        qc_2 = QuantumCircuit(1, 1, name="add_circ")
        with self.assertWarns(DeprecationWarning):
            qc_2.h(0).c_if(qc_2.cregs[0], 1)
        qc_2.measure(0, 0)
        # This append calls ends up calling .to_instruction() which calls
        # .c_if() internally
        with self.assertWarns(DeprecationWarning):
            qc.append(qc_2, [1], [0])

        (_, _, layered_ops) = _utils._get_layered_instructions(qc)

        expected = [
            {("h", (Qubit(QuantumRegister(2, "q"), 0),), ())},
            {
                (
                    "measure",
                    (Qubit(QuantumRegister(2, "q"), 0),),
                    (Clbit(ClassicalRegister(2, "c"), 0),),
                )
            },
            {
                (
                    "add_circ",
                    (Qubit(QuantumRegister(2, "q"), 1),),
                    (Clbit(ClassicalRegister(2, "c"), 0),),
                )
            },
        ]

        self.assertEqual(
            expected, [{(op.name, op.qargs, op.cargs) for op in ops} for ops in layered_ops]
        )

    @unittest.skipUnless(optionals.HAS_PYLATEX, "needs pylatexenc")
    def test_generate_latex_label_nomathmode(self):
        """Test generate latex label default."""
        self.assertEqual("abc", _utils.generate_latex_label("abc"))

    @unittest.skipUnless(optionals.HAS_PYLATEX, "needs pylatexenc")
    def test_generate_latex_label_nomathmode_utf8char(self):
        """Test generate latex label utf8 characters."""
        self.assertEqual(
            "{\\ensuremath{\\iiint}}X{\\ensuremath{\\forall}}Y",
            _utils.generate_latex_label("∭X∀Y"),
        )

    @unittest.skipUnless(optionals.HAS_PYLATEX, "needs pylatexenc")
    def test_generate_latex_label_mathmode_utf8char(self):
        """Test generate latex label mathtext with utf8."""
        self.assertEqual(
            "abc_{\\ensuremath{\\iiint}}X{\\ensuremath{\\forall}}Y",
            _utils.generate_latex_label("$abc_$∭X∀Y"),
        )

    @unittest.skipUnless(optionals.HAS_PYLATEX, "needs pylatexenc")
    def test_generate_latex_label_mathmode_underscore_outside(self):
        """Test generate latex label with underscore outside mathmode."""
        self.assertEqual(
            "abc_{\\ensuremath{\\iiint}}X{\\ensuremath{\\forall}}Y",
            _utils.generate_latex_label("$abc$_∭X∀Y"),
        )

    @unittest.skipUnless(optionals.HAS_PYLATEX, "needs pylatexenc")
    def test_generate_latex_label_escaped_dollar_signs(self):
        """Test generate latex label with escaped dollarsign."""
        self.assertEqual("${\\ensuremath{\\forall}}$", _utils.generate_latex_label(r"\$∀\$"))

    @unittest.skipUnless(optionals.HAS_PYLATEX, "needs pylatexenc")
    def test_generate_latex_label_escaped_dollar_sign_in_mathmode(self):
        """Test generate latex label with escaped dollar sign in mathmode."""
        self.assertEqual(
            "a$bc_{\\ensuremath{\\iiint}}X{\\ensuremath{\\forall}}Y",
            _utils.generate_latex_label(r"$a$bc$_∭X∀Y"),
        )

    def test_array_to_latex(self):
        """Test array_to_latex produces correct latex string"""
        matrix = [
            [np.sqrt(1 / 2), 1 / 16, 1 / np.sqrt(8) + 3j, -0.5 + 0.5j],
            [1 / 3 - 1 / 3j, np.sqrt(1 / 2) * 1j, 34.3210, -9 / 2],
        ]
        matrix = np.array(matrix)
        exp_str = (
            "\\begin{bmatrix}\\frac{\\sqrt{2}}{2}&\\frac{1}{16}&"
            "\\frac{\\sqrt{2}}{4}+3i&-\\frac{1}{2}+\\frac{i}{2}\\\\"
            "\\frac{1}{3}+\\frac{i}{3}&\\frac{\\sqrt{2}i}{2}&34.321&-"
            "\\frac{9}{2}\\\\\\end{bmatrix}"
        )
        result = array_to_latex(matrix, source=True).replace(" ", "").replace("\n", "")
        self.assertEqual(exp_str, result)


if __name__ == "__main__":
    unittest.main(verbosity=2)

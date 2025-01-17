# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-class-docstring,missing-module-docstring

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ConvertConditionsToIfOps
from test.utils._canonical import canonicalize_control_flow  # pylint: disable=wrong-import-order
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestConvertConditionsToIfOps(QiskitTestCase):
    def test_simple_loose_bits(self):
        """Test that basic conversions work when operating on loose classical bits."""
        bits = [Qubit(), Qubit(), Clbit(), Clbit()]

        base = QuantumCircuit(bits)
        base.h(0)
        with self.assertWarns(DeprecationWarning):
            base.x(0).c_if(0, 1)
        with self.assertWarns(DeprecationWarning):
            base.z(1).c_if(1, 0)
        base.measure(0, 0)
        base.measure(1, 1)
        base.h(0)
        with self.assertWarns(DeprecationWarning):
            base.x(0).c_if(0, 1)
        with self.assertWarns(DeprecationWarning):
            base.cx(0, 1).c_if(1, 0)

        expected = QuantumCircuit(bits)
        expected.h(0)
        with expected.if_test((expected.clbits[0], True)):
            expected.x(0)
        with expected.if_test((expected.clbits[1], False)):
            expected.z(1)
        expected.measure(0, 0)
        expected.measure(1, 1)
        expected.h(0)
        with expected.if_test((expected.clbits[0], True)):
            expected.x(0)
        with expected.if_test((expected.clbits[1], False)):
            expected.cx(0, 1)
        expected = canonicalize_control_flow(expected)
        with self.assertWarns(DeprecationWarning):
            output = PassManager([ConvertConditionsToIfOps()]).run(base)
        self.assertEqual(output, expected)

    def test_simple_registers(self):
        """Test that basic conversions work when operating on conditions over registers."""
        registers = [QuantumRegister(2), ClassicalRegister(2), ClassicalRegister(1)]

        base = QuantumCircuit(*registers)
        base.h(0)
        with self.assertWarns(DeprecationWarning):
            base.x(0).c_if(base.cregs[0], 1)
        with self.assertWarns(DeprecationWarning):
            base.z(1).c_if(base.cregs[1], 0)
        base.measure(0, 0)
        base.measure(1, 2)
        base.h(0)
        with self.assertWarns(DeprecationWarning):
            base.x(0).c_if(base.cregs[0], 1)
        with self.assertWarns(DeprecationWarning):
            base.cx(0, 1).c_if(base.cregs[1], 0)

        expected = QuantumCircuit(*registers)
        expected.h(0)
        with expected.if_test((expected.cregs[0], 1)):
            expected.x(0)
        with expected.if_test((expected.cregs[1], 0)):
            expected.z(1)
        expected.measure(0, 0)
        expected.measure(1, 2)
        expected.h(0)
        with expected.if_test((expected.cregs[0], 1)):
            expected.x(0)
        with expected.if_test((expected.cregs[1], 0)):
            expected.cx(0, 1)
        expected = canonicalize_control_flow(expected)

        with self.assertWarns(DeprecationWarning):
            output = PassManager([ConvertConditionsToIfOps()]).run(base)
        self.assertEqual(output, expected)

    def test_nested_control_flow(self):
        """Test that the pass successfully converts instructions nested within control-flow
        blocks."""
        bits = [Clbit()]
        registers = [QuantumRegister(3), ClassicalRegister(2)]

        base = QuantumCircuit(*registers, bits)
        with self.assertWarns(DeprecationWarning):
            base.x(0).c_if(bits[0], False)
        with base.if_test((base.cregs[0], 0)) as else_:
            with self.assertWarns(DeprecationWarning):
                base.z(1).c_if(bits[0], False)
        with else_:
            with self.assertWarns(DeprecationWarning):
                base.z(1).c_if(base.cregs[0], 1)
        with base.for_loop(range(2)):
            with base.while_loop((base.cregs[0], 1)):
                with self.assertWarns(DeprecationWarning):
                    base.cx(1, 2).c_if(base.cregs[0], 1)
        base = canonicalize_control_flow(base)

        expected = QuantumCircuit(*registers, bits)
        with expected.if_test((bits[0], False)):
            expected.x(0)
        with expected.if_test((expected.cregs[0], 0)) as else_:
            with expected.if_test((bits[0], False)):
                expected.z(1)
        with else_:
            with expected.if_test((expected.cregs[0], 1)):
                expected.z(1)
        with expected.for_loop(range(2)):
            with expected.while_loop((expected.cregs[0], 1)):
                with expected.if_test((expected.cregs[0], 1)):
                    expected.cx(1, 2)
        expected = canonicalize_control_flow(expected)
        with self.assertWarns(DeprecationWarning):
            output = PassManager([ConvertConditionsToIfOps()]).run(base)
        self.assertEqual(output, expected)

    def test_no_op(self):
        """Test that the pass works when recursing into control-flow structures, but there's nothing
        that actually needs replacing."""
        bits = [Clbit()]
        registers = [QuantumRegister(3), ClassicalRegister(2)]

        base = QuantumCircuit(*registers, bits)
        base.x(0)
        with base.if_test((base.cregs[0], 0)) as else_:
            base.z(1)
        with else_:
            base.z(2)
        with base.for_loop(range(2)):
            with base.while_loop((base.cregs[0], 1)):
                base.cx(1, 2)
        base = canonicalize_control_flow(base)
        with self.assertWarns(DeprecationWarning):
            output = PassManager([ConvertConditionsToIfOps()]).run(base)
        self.assertEqual(output, base)

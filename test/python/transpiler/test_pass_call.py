# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test calling passes (passmanager-less)"""

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import ZGate
from qiskit.transpiler.passes import Unroller
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError
from qiskit.transpiler import PropertySet
from ._dummy_passes import PassD_TP_NR_NP, PassE_AP_NR_NP, PassN_AP_NR_NP


class TestPassCall(QiskitTestCase):
    """Test calling passes (passmanager-less)."""

    def assertMessageLog(self, context, messages):
        """Checks the log messages"""
        self.assertEqual([record.message for record in context.records], messages)

    def test_transformation_pass(self):
        """Call a transformation pass without a scheduler"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, name="MyCircuit")

        pass_d = PassD_TP_NR_NP(argument1=[1, 2])
        with self.assertLogs("LocalLogger", level="INFO") as cm:
            result = pass_d(circuit)

        self.assertMessageLog(cm, ["run transformation pass PassD_TP_NR_NP", "argument [1, 2]"])
        self.assertEqual(circuit, result)

    def test_analysis_pass_dict(self):
        """Call an analysis pass without a scheduler (property_set dict)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, name="MyCircuit")
        property_set = {"another_property": "another_value"}

        pass_e = PassE_AP_NR_NP("value")
        with self.assertLogs("LocalLogger", level="INFO") as cm:
            result = pass_e(circuit, property_set)

        self.assertMessageLog(cm, ["run analysis pass PassE_AP_NR_NP", "set property as value"])
        self.assertEqual(property_set, {"another_property": "another_value", "property": "value"})
        self.assertIsInstance(property_set, dict)
        self.assertEqual(circuit, result)

    def test_analysis_pass_property_set(self):
        """Call an analysis pass without a scheduler (PropertySet dict)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, name="MyCircuit")
        property_set = PropertySet({"another_property": "another_value"})

        pass_e = PassE_AP_NR_NP("value")
        with self.assertLogs("LocalLogger", level="INFO") as cm:
            result = pass_e(circuit, property_set)

        self.assertMessageLog(cm, ["run analysis pass PassE_AP_NR_NP", "set property as value"])
        self.assertEqual(
            property_set, PropertySet({"another_property": "another_value", "property": "value"})
        )
        self.assertIsInstance(property_set, PropertySet)
        self.assertEqual(circuit, result)

    def test_analysis_pass_remove_property(self):
        """Call an analysis pass that removes a property without a scheduler"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, name="MyCircuit")
        property_set = {"to remove": "value to remove", "to none": "value to none"}

        pass_e = PassN_AP_NR_NP("to remove", "to none")
        with self.assertLogs("LocalLogger", level="INFO") as cm:
            result = pass_e(circuit, property_set)

        self.assertMessageLog(
            cm,
            [
                "run analysis pass PassN_AP_NR_NP",
                "property to remove deleted",
                "property to none noned",
            ],
        )
        self.assertEqual(property_set, PropertySet({"to none": None}))
        self.assertIsInstance(property_set, dict)
        self.assertEqual(circuit, result)

    def test_error_unknown_defn_unroller_pass(self):
        """Check for proper error message when unroller cannot find the definition
        of a gate."""
        circuit = ZGate().control(2).definition
        basis = ["u1", "u2", "u3", "cx"]
        unroller = Unroller(basis)
        with self.assertRaises(QiskitError) as cm:
            unroller(circuit)
        exp_msg = "Error decomposing node of instruction 'p'. Unable to define instruction 'u' in the basis."
        self.assertEqual(exp_msg, cm.exception.message)

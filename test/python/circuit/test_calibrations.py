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

"""Test calibrations in quantum circuits."""

import unittest

from qiskit.pulse import Schedule
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RZXGate
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestCalibrations(QiskitTestCase):
    """Test composition of two circuits."""

    def test_iadd(self):
        """Test that __iadd__ keeps the calibrations."""
        qc_cal = QuantumCircuit(2)
        qc_cal.rzx(0.5, 0, 1)
        with self.assertWarns(DeprecationWarning):
            qc_cal.add_calibration(RZXGate, (0, 1), params=[0.5], schedule=Schedule())

        qc = QuantumCircuit(2)
        qc &= qc_cal

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(qc.calibrations[RZXGate], {((0, 1), (0.5,)): Schedule(name="test")})
            self.assertEqual(qc_cal.calibrations, qc.calibrations)

    def test_add(self):
        """Test that __add__ keeps the calibrations."""
        qc_cal = QuantumCircuit(2)
        qc_cal.rzx(0.5, 0, 1)
        with self.assertWarns(DeprecationWarning):
            qc_cal.add_calibration(RZXGate, (0, 1), params=[0.5], schedule=Schedule())

        qc = QuantumCircuit(2) & qc_cal

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(qc.calibrations[RZXGate], {((0, 1), (0.5,)): Schedule(name="test")})
            self.assertEqual(qc_cal.calibrations, qc.calibrations)

        qc = qc_cal & QuantumCircuit(2)

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(qc.calibrations[RZXGate], {((0, 1), (0.5,)): Schedule(name="test")})
            self.assertEqual(qc_cal.calibrations, qc.calibrations)


if __name__ == "__main__":
    unittest.main()

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring

"""Tests basic functionality of the sequence function"""
# TODO with the removal of pulses, this file can be removed too.

import unittest

from qiskit import QuantumCircuit, pulse
from qiskit.compiler import sequence, transpile, schedule
from qiskit.pulse.transforms import pad
from qiskit.providers.fake_provider import Fake127QPulseV1
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestSequence(QiskitTestCase):
    """Test sequence function."""

    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            self.backend = Fake127QPulseV1()
        self.backend.configuration().timing_constraints = {}

    def test_sequence_empty(self):
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(sequence([], self.backend), [])

    def test_transpile_and_sequence_agree_with_schedule(self):
        qc = QuantumCircuit(2, name="bell")
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            sc = transpile(qc, self.backend, scheduling_method="alap")
        with self.assertWarns(DeprecationWarning):
            actual = sequence(sc, self.backend)
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            expected = schedule(transpile(qc, self.backend), self.backend)
        with self.assertWarns(DeprecationWarning):
            # pad adds Delay which is deprecated
            self.assertEqual(actual, pad(expected))

    def test_transpile_and_sequence_agree_with_schedule_for_circuit_with_delay(self):
        qc = QuantumCircuit(1, 1, name="t2")
        qc.h(0)
        qc.delay(500, 0, unit="ns")
        qc.h(0)
        qc.measure(0, 0)
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            sc = transpile(qc, self.backend, scheduling_method="alap")
        with self.assertWarns(DeprecationWarning):
            actual = sequence(sc, self.backend)
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            expected = schedule(transpile(qc, self.backend), self.backend)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(
                actual.exclude(instruction_types=[pulse.Delay]),
                expected.exclude(instruction_types=[pulse.Delay]),
            )

    @unittest.skip("not yet determined if delays on ancilla should be removed or not")
    def test_transpile_and_sequence_agree_with_schedule_for_circuits_without_measures(self):
        qc = QuantumCircuit(2, name="bell_without_measurement")
        qc.h(0)
        qc.cx(0, 1)
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            sc = transpile(qc, self.backend, scheduling_method="alap")
        with self.assertWarns(DeprecationWarning):
            actual = sequence(sc, self.backend)
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            with self.assertWarns(DeprecationWarning):
                expected = schedule(transpile(qc, self.backend), self.backend)
        self.assertEqual(actual, pad(expected))

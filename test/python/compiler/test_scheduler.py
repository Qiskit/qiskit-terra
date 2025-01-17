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

"""Scheduler Test."""

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse import InstructionScheduleMap, Schedule
from qiskit.providers.fake_provider import FakeOpenPulse3Q, GenericBackendV2
from qiskit.compiler.scheduler import schedule
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestCircuitScheduler(QiskitTestCase):
    """Tests for scheduling."""

    def setUp(self):
        super().setUp()
        qr = QuantumRegister(2, name="q")
        cr = ClassicalRegister(2, name="c")
        self.circ = QuantumCircuit(qr, cr, name="circ")
        self.circ.cx(qr[0], qr[1])
        self.circ.measure(qr, cr)

        qr2 = QuantumRegister(2, name="q")
        cr2 = ClassicalRegister(2, name="c")
        self.circ2 = QuantumCircuit(qr2, cr2, name="circ2")
        self.circ2.cx(qr2[0], qr2[1])
        self.circ2.measure(qr2, cr2)

        with self.assertWarns(DeprecationWarning):
            self.backend = GenericBackendV2(
                3, calibrate_instructions=True, basis_gates=["cx", "u1", "u2", "u3"], seed=42
            )

    def test_instruction_map_and_backend_not_supplied(self):
        """Test instruction map and backend not supplied."""
        with self.assertRaisesRegex(
            QiskitError,
            r"Must supply either a backend or InstructionScheduleMap for scheduling passes.",
        ):
            with self.assertWarns(DeprecationWarning):
                schedule(self.circ)

    def test_instruction_map_and_backend_defaults_unavailable(self):
        """Test backend defaults unavailable when backend is provided, but instruction map is not."""
        with self.assertWarns(DeprecationWarning):
            self.backend = FakeOpenPulse3Q()
        self.backend._defaults = None
        with self.assertRaisesRegex(
            QiskitError, r"The backend defaults are unavailable. The backend may not support pulse."
        ):
            with self.assertWarns(DeprecationWarning):
                schedule(self.circ, self.backend)

    def test_measurement_map_and_backend_not_supplied(self):
        """Test measurement map and backend not supplied."""
        with self.assertRaisesRegex(
            QiskitError,
            r"Must supply either a backend or a meas_map for scheduling passes.",
        ):
            with self.assertWarns(DeprecationWarning):
                schedule(self.circ, inst_map=InstructionScheduleMap())

    def test_schedules_single_circuit(self):
        """Test scheduling of a single circuit."""
        with self.assertWarns(DeprecationWarning):
            circuit_schedule = schedule(self.circ, self.backend)

        self.assertIsInstance(circuit_schedule, Schedule)
        self.assertEqual(circuit_schedule.name, "circ")

    def test_schedules_multiple_circuits(self):
        """Test scheduling of multiple circuits."""
        self.enable_parallel_processing()

        circuits = [self.circ, self.circ2]
        with self.assertWarns(DeprecationWarning):
            circuit_schedules = schedule(circuits, self.backend, method="asap")
        self.assertEqual(len(circuit_schedules), len(circuits))

        circuit_one_schedule = circuit_schedules[0]
        circuit_two_schedule = circuit_schedules[1]

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(
                circuit_one_schedule,
                schedule(self.circ, self.backend, method="asap"),
            )

            self.assertEqual(
                circuit_two_schedule,
                schedule(self.circ2, self.backend, method="asap"),
            )

# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test pulse builder with backendV2 context utilities."""

import numpy as np

from qiskit import pulse
from qiskit.pulse import macros

from qiskit.pulse.instructions import directives
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.pulse import instructions
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings

from ..legacy_cmaps import MUMBAI_CMAP


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestBuilderV2(QiskitTestCase):
    """Test the pulse builder context with backendV2."""

    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            self.backend = GenericBackendV2(
                num_qubits=27, coupling_map=MUMBAI_CMAP, calibrate_instructions=True, seed=42
            )

    def assertScheduleEqual(self, program, target):
        """Assert an error when two pulse programs are not equal.

        .. note:: Two programs are converted into standard execution format then compared.
        """
        self.assertEqual(target_qobj_transform(program), target_qobj_transform(target))


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestContextsV2(TestBuilderV2):
    """Test builder contexts."""

    def test_phase_compensated_frequency_offset(self):
        """Test that the phase offset context properly compensates for phase
        accumulation with backendV2."""
        d0 = pulse.DriveChannel(0)
        with pulse.build(self.backend) as schedule:
            with pulse.frequency_offset(1e9, d0, compensate_phase=True):
                pulse.delay(10, d0)

        reference = pulse.Schedule()
        reference += instructions.ShiftFrequency(1e9, d0)
        reference += instructions.Delay(10, d0)
        reference += instructions.ShiftPhase(
            -2 * np.pi * ((1e9 * 10 * self.backend.target.dt) % 1), d0
        )
        reference += instructions.ShiftFrequency(-1e9, d0)
        self.assertScheduleEqual(schedule, reference)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestChannelsV2(TestBuilderV2):
    """Test builder channels."""

    def test_drive_channel(self):
        """Text context builder drive channel."""
        with pulse.build(self.backend):
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(pulse.drive_channel(0), pulse.DriveChannel(0))

    def test_measure_channel(self):
        """Text context builder measure channel."""
        with pulse.build(self.backend):
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(pulse.measure_channel(0), pulse.MeasureChannel(0))

    def test_acquire_channel(self):
        """Text context builder acquire channel."""
        with self.assertWarns(DeprecationWarning):
            with pulse.build(self.backend):
                self.assertEqual(pulse.acquire_channel(0), pulse.AcquireChannel(0))

    def test_control_channel(self):
        """Text context builder control channel."""
        with pulse.build(self.backend):
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(pulse.control_channels(0, 1)[0], pulse.ControlChannel(0))


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestDirectivesV2(TestBuilderV2):
    """Test builder directives."""

    def test_barrier_on_qubits(self):
        """Test barrier directive on qubits with backendV2.
        A part of qubits map of Mumbai
            0 -- 1 -- 4 --
                |
                |
                2
        """
        with pulse.build(self.backend) as schedule:
            with self.assertWarns(DeprecationWarning):
                pulse.barrier(0, 1)
        reference = pulse.ScheduleBlock()
        reference += directives.RelativeBarrier(
            pulse.DriveChannel(0),
            pulse.DriveChannel(1),
            pulse.MeasureChannel(0),
            pulse.MeasureChannel(1),
            pulse.ControlChannel(0),
            pulse.ControlChannel(1),
            pulse.ControlChannel(2),
            pulse.ControlChannel(3),
            pulse.ControlChannel(4),
            pulse.ControlChannel(8),
            pulse.AcquireChannel(0),
            pulse.AcquireChannel(1),
        )
        self.assertEqual(schedule, reference)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestUtilitiesV2(TestBuilderV2):
    """Test builder utilities."""

    def test_active_backend(self):
        """Test getting active builder backend."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.active_backend(), self.backend)

    def test_qubit_channels(self):
        """Test getting the qubit channels of the active builder's backend."""
        with pulse.build(self.backend):
            with self.assertWarns(DeprecationWarning):
                qubit_channels = pulse.qubit_channels(0)

        self.assertEqual(
            qubit_channels,
            {
                pulse.DriveChannel(0),
                pulse.MeasureChannel(0),
                pulse.AcquireChannel(0),
                pulse.ControlChannel(0),
                pulse.ControlChannel(1),
            },
        )

    def test_num_qubits(self):
        """Test builder utility to get number of qubits with backendV2."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.num_qubits(), 27)

    def test_samples_to_seconds(self):
        """Test samples to time with backendV2"""
        target = self.backend.target
        target.dt = 0.1
        with pulse.build(self.backend):
            time = pulse.samples_to_seconds(100)
            self.assertTrue(isinstance(time, float))
            self.assertEqual(pulse.samples_to_seconds(100), 10)

    def test_samples_to_seconds_array(self):
        """Test samples to time (array format) with backendV2."""
        target = self.backend.target
        target.dt = 0.1
        with pulse.build(self.backend):
            samples = np.array([100, 200, 300])
            times = pulse.samples_to_seconds(samples)
            self.assertTrue(np.issubdtype(times.dtype, np.floating))
            np.testing.assert_allclose(times, np.array([10, 20, 30]))

    def test_seconds_to_samples(self):
        """Test time to samples with backendV2"""
        target = self.backend.target
        target.dt = 0.1
        with pulse.build(self.backend):
            samples = pulse.seconds_to_samples(10)
            self.assertTrue(isinstance(samples, int))
            self.assertEqual(pulse.seconds_to_samples(10), 100)

    def test_seconds_to_samples_array(self):
        """Test time to samples (array format) with backendV2."""
        target = self.backend.target
        target.dt = 0.1
        with pulse.build(self.backend):
            times = np.array([10, 20, 30])
            samples = pulse.seconds_to_samples(times)
            self.assertTrue(np.issubdtype(samples.dtype, np.integer))
            np.testing.assert_allclose(pulse.seconds_to_samples(times), np.array([100, 200, 300]))


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestMacrosV2(TestBuilderV2):
    """Test builder macros with backendV2."""

    def test_macro(self):
        """Test builder macro decorator."""

        @pulse.macro
        def nested(a):
            with self.assertWarns(DeprecationWarning):
                pulse.play(pulse.Gaussian(100, a, 20), pulse.drive_channel(0))
            return a * 2

        @pulse.macro
        def test():
            with self.assertWarns(DeprecationWarning):
                pulse.play(pulse.Constant(100, 1.0), pulse.drive_channel(0))
            output = nested(0.5)
            return output

        with pulse.build(self.backend) as schedule:
            output = test()
            self.assertEqual(output, 0.5 * 2)

        reference = pulse.Schedule()
        reference += pulse.Play(pulse.Constant(100, 1.0), pulse.DriveChannel(0))
        reference += pulse.Play(pulse.Gaussian(100, 0.5, 20), pulse.DriveChannel(0))

        self.assertScheduleEqual(schedule, reference)

    def test_measure(self):
        """Test utility function - measure with backendV2."""
        with pulse.build(self.backend) as schedule:
            with self.assertWarns(DeprecationWarning):
                reg = pulse.measure(0)

        self.assertEqual(reg, pulse.MemorySlot(0))

        reference = macros.measure(qubits=[0], backend=self.backend, meas_map=self.backend.meas_map)

        self.assertScheduleEqual(schedule, reference)

    def test_measure_multi_qubits(self):
        """Test utility function - measure with multi qubits with backendV2."""
        with pulse.build(self.backend) as schedule:
            with self.assertWarns(DeprecationWarning):
                regs = pulse.measure([0, 1])

        self.assertListEqual(regs, [pulse.MemorySlot(0), pulse.MemorySlot(1)])

        reference = macros.measure(
            qubits=[0, 1], backend=self.backend, meas_map=self.backend.meas_map
        )

        self.assertScheduleEqual(schedule, reference)

    def test_measure_all(self):
        """Test utility function - measure with backendV2.."""
        with pulse.build(self.backend) as schedule:
            with self.assertWarns(DeprecationWarning):
                regs = pulse.measure_all()

        self.assertEqual(regs, [pulse.MemorySlot(i) for i in range(self.backend.num_qubits)])
        reference = macros.measure_all(self.backend)

        self.assertScheduleEqual(schedule, reference)

    def test_delay_qubit(self):
        """Test delaying on a qubit macro."""
        with pulse.build(self.backend) as schedule:
            with self.assertWarns(DeprecationWarning):
                pulse.delay_qubits(10, 0)

        d0 = pulse.DriveChannel(0)
        m0 = pulse.MeasureChannel(0)
        a0 = pulse.AcquireChannel(0)
        u0 = pulse.ControlChannel(0)
        u1 = pulse.ControlChannel(1)

        reference = pulse.Schedule()
        reference += instructions.Delay(10, d0)
        reference += instructions.Delay(10, m0)
        reference += instructions.Delay(10, a0)
        reference += instructions.Delay(10, u0)
        reference += instructions.Delay(10, u1)

        self.assertScheduleEqual(schedule, reference)

    def test_delay_qubits(self):
        """Test delaying on multiple qubits with backendV2 to make sure we don't insert delays twice."""
        with pulse.build(self.backend) as schedule:
            with self.assertWarns(DeprecationWarning):
                pulse.delay_qubits(10, 0, 1)

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        m0 = pulse.MeasureChannel(0)
        m1 = pulse.MeasureChannel(1)
        a0 = pulse.AcquireChannel(0)
        a1 = pulse.AcquireChannel(1)
        u0 = pulse.ControlChannel(0)
        u1 = pulse.ControlChannel(1)
        u2 = pulse.ControlChannel(2)
        u3 = pulse.ControlChannel(3)
        u4 = pulse.ControlChannel(4)
        u8 = pulse.ControlChannel(8)

        reference = pulse.Schedule()
        reference += instructions.Delay(10, d0)
        reference += instructions.Delay(10, d1)
        reference += instructions.Delay(10, m0)
        reference += instructions.Delay(10, m1)
        reference += instructions.Delay(10, a0)
        reference += instructions.Delay(10, a1)
        reference += instructions.Delay(10, u0)
        reference += instructions.Delay(10, u1)
        reference += instructions.Delay(10, u2)
        reference += instructions.Delay(10, u3)
        reference += instructions.Delay(10, u4)
        reference += instructions.Delay(10, u8)

        self.assertScheduleEqual(schedule, reference)

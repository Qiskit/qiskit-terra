# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for the pulse Schedule transforms."""
import unittest
from typing import List, Set

import numpy as np

from qiskit import pulse
from qiskit.pulse import (
    Play,
    Delay,
    Acquire,
    Schedule,
    Waveform,
    Drag,
    Gaussian,
    GaussianSquare,
    Constant,
)
from qiskit.pulse import transforms, instructions
from qiskit.pulse.channels import (
    MemorySlot,
    DriveChannel,
    AcquireChannel,
    RegisterSlot,
    SnapshotChannel,
)
from qiskit.pulse.instructions import directives
from qiskit.providers.fake_provider import FakeOpenPulse2Q
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestAlignMeasures(QiskitTestCase):
    """Test the helper function which aligns acquires."""

    @ignore_pulse_deprecation_warnings
    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            self.backend = FakeOpenPulse2Q()
        self.config = self.backend.configuration()
        self.inst_map = self.backend.defaults().instruction_schedule_map
        self.short_pulse = pulse.Waveform(
            samples=np.array([0.02739068], dtype=np.complex128), name="p0"
        )

    def test_align_measures(self):
        """Test that one acquire is delayed to match the time of the later acquire."""
        sched = pulse.Schedule(name="fake_experiment")
        sched.insert(0, Play(self.short_pulse, self.config.drive(0)), inplace=True)
        sched.insert(1, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)
        sched.insert(10, Acquire(5, self.config.acquire(1), MemorySlot(1)), inplace=True)
        sched.insert(10, Play(self.short_pulse, self.config.measure(0)), inplace=True)
        sched.insert(11, Play(self.short_pulse, self.config.measure(0)), inplace=True)
        sched.insert(10, Play(self.short_pulse, self.config.measure(1)), inplace=True)
        aligned = transforms.align_measures([sched])[0]
        self.assertEqual(aligned.name, "fake_experiment")

        ref = pulse.Schedule(name="fake_experiment")
        ref.insert(0, Play(self.short_pulse, self.config.drive(0)), inplace=True)
        ref.insert(10, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)
        ref.insert(10, Acquire(5, self.config.acquire(1), MemorySlot(1)), inplace=True)
        ref.insert(19, Play(self.short_pulse, self.config.measure(0)), inplace=True)
        ref.insert(20, Play(self.short_pulse, self.config.measure(0)), inplace=True)
        ref.insert(10, Play(self.short_pulse, self.config.measure(1)), inplace=True)

        self.assertEqual(aligned, ref)

        aligned = transforms.align_measures([sched], self.inst_map, align_time=20)[0]

        ref = pulse.Schedule(name="fake_experiment")
        ref.insert(10, Play(self.short_pulse, self.config.drive(0)), inplace=True)
        ref.insert(20, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)
        ref.insert(20, Acquire(5, self.config.acquire(1), MemorySlot(1)), inplace=True)
        ref.insert(29, Play(self.short_pulse, self.config.measure(0)), inplace=True)
        ref.insert(30, Play(self.short_pulse, self.config.measure(0)), inplace=True)
        ref.insert(20, Play(self.short_pulse, self.config.measure(1)), inplace=True)
        self.assertEqual(aligned, ref)

    def test_align_post_u3(self):
        """Test that acquires are scheduled no sooner than the duration of the longest X gate."""
        sched = pulse.Schedule(name="fake_experiment")
        sched = sched.insert(0, Play(self.short_pulse, self.config.drive(0)))
        sched = sched.insert(1, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        sched = transforms.align_measures([sched], self.inst_map)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, Acquire):
                self.assertEqual(time, 4)
        sched = transforms.align_measures([sched], self.inst_map, max_calibration_duration=10)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, Acquire):
                self.assertEqual(time, 10)

    def test_multi_acquire(self):
        """Test that the last acquire is aligned to if multiple acquires occur on the
        same channel."""
        sched = pulse.Schedule()
        sched.insert(0, Play(self.short_pulse, self.config.drive(0)), inplace=True)
        sched.insert(4, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)
        sched.insert(20, Acquire(5, self.config.acquire(1), MemorySlot(1)), inplace=True)
        sched.insert(10, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)
        aligned = transforms.align_measures([sched], self.inst_map)

        ref = pulse.Schedule()
        ref.insert(0, Play(self.short_pulse, self.config.drive(0)), inplace=True)
        ref.insert(20, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)
        ref.insert(20, Acquire(5, self.config.acquire(1), MemorySlot(1)), inplace=True)
        ref.insert(26, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)
        self.assertEqual(aligned[0], ref)

    def test_multiple_acquires(self):
        """Test that multiple acquires are also aligned."""
        sched = pulse.Schedule(name="fake_experiment")
        sched.insert(0, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)
        sched.insert(5, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)
        sched.insert(10, Acquire(5, self.config.acquire(1), MemorySlot(1)), inplace=True)

        ref = pulse.Schedule()
        ref.insert(10, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)
        ref.insert(15, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)
        ref.insert(10, Acquire(5, self.config.acquire(1), MemorySlot(1)), inplace=True)

        aligned = transforms.align_measures([sched], self.inst_map)[0]

        self.assertEqual(aligned, ref)

    def test_align_across_schedules(self):
        """Test that acquires are aligned together across multiple schedules."""
        sched1 = pulse.Schedule(name="fake_experiment")
        sched1 = sched1.insert(0, Play(self.short_pulse, self.config.drive(0)))
        sched1 = sched1.insert(10, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        sched2 = pulse.Schedule(name="fake_experiment")
        sched2 = sched2.insert(3, Play(self.short_pulse, self.config.drive(0)))
        sched2 = sched2.insert(25, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        schedules = transforms.align_measures([sched1, sched2], self.inst_map)
        for time, inst in schedules[0].instructions:
            if isinstance(inst, Acquire):
                self.assertEqual(time, 25)
        for time, inst in schedules[0].instructions:
            if isinstance(inst, Acquire):
                self.assertEqual(time, 25)

    def test_align_all(self):
        """Test alignment of all instructions in a schedule."""
        sched0 = pulse.Schedule()
        sched0.insert(0, Play(self.short_pulse, self.config.drive(0)), inplace=True)
        sched0.insert(10, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)

        sched1 = pulse.Schedule()
        sched1.insert(25, Play(self.short_pulse, self.config.drive(0)), inplace=True)
        sched1.insert(25, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)

        all_aligned = transforms.align_measures([sched0, sched1], self.inst_map, align_all=True)

        ref1_aligned = pulse.Schedule()
        ref1_aligned.insert(15, Play(self.short_pulse, self.config.drive(0)), inplace=True)
        ref1_aligned.insert(25, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)

        self.assertEqual(all_aligned[0], ref1_aligned)
        self.assertEqual(all_aligned[1], sched1)

        ref1_not_aligned = pulse.Schedule()
        ref1_not_aligned.insert(0, Play(self.short_pulse, self.config.drive(0)), inplace=True)
        ref1_not_aligned.insert(25, Acquire(5, self.config.acquire(0), MemorySlot(0)), inplace=True)

        all_not_aligned = transforms.align_measures(
            [sched0, sched1],
            self.inst_map,
            align_all=False,
        )
        self.assertEqual(all_not_aligned[0], ref1_not_aligned)
        self.assertEqual(all_not_aligned[1], sched1)

    def test_measurement_at_zero(self):
        """Test that acquire at t=0 works."""
        sched1 = pulse.Schedule(name="fake_experiment")
        sched1 = sched1.insert(0, Play(self.short_pulse, self.config.drive(0)))
        sched1 = sched1.insert(0, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        sched2 = pulse.Schedule(name="fake_experiment")
        sched2 = sched2.insert(0, Play(self.short_pulse, self.config.drive(0)))
        sched2 = sched2.insert(0, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        schedules = transforms.align_measures([sched1, sched2], max_calibration_duration=0)
        for time, inst in schedules[0].instructions:
            if isinstance(inst, Acquire):
                self.assertEqual(time, 0)
        for time, inst in schedules[0].instructions:
            if isinstance(inst, Acquire):
                self.assertEqual(time, 0)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestAddImplicitAcquires(QiskitTestCase):
    """Test the helper function which makes implicit acquires explicit."""

    @ignore_pulse_deprecation_warnings
    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            self.backend = FakeOpenPulse2Q()
        self.config = self.backend.configuration()
        self.short_pulse = pulse.Waveform(
            samples=np.array([0.02739068], dtype=np.complex128), name="p0"
        )
        sched = pulse.Schedule(name="fake_experiment")
        sched = sched.insert(0, Play(self.short_pulse, self.config.drive(0)))
        sched = sched.insert(5, Acquire(5, self.config.acquire(0), MemorySlot(0)))
        sched = sched.insert(5, Acquire(5, self.config.acquire(1), MemorySlot(1)))
        self.sched = sched

    def test_add_implicit(self):
        """Test that implicit acquires are made explicit according to the meas map."""
        sched = transforms.add_implicit_acquires(self.sched, [[0, 1]])
        acquired_qubits = set()
        for _, inst in sched.instructions:
            if isinstance(inst, Acquire):
                acquired_qubits.add(inst.acquire.index)
        self.assertEqual(acquired_qubits, {0, 1})

    def test_add_across_meas_map_sublists(self):
        """Test that implicit acquires in separate meas map sublists are all added."""
        sched = transforms.add_implicit_acquires(self.sched, [[0, 2], [1, 3]])
        acquired_qubits = set()
        for _, inst in sched.instructions:
            if isinstance(inst, Acquire):
                acquired_qubits.add(inst.acquire.index)
        self.assertEqual(acquired_qubits, {0, 1, 2, 3})

    def test_dont_add_all(self):
        """Test that acquires aren't added if no qubits in the sublist aren't being acquired."""
        sched = transforms.add_implicit_acquires(self.sched, [[4, 5], [0, 2], [1, 3]])
        acquired_qubits = set()
        for _, inst in sched.instructions:
            if isinstance(inst, Acquire):
                acquired_qubits.add(inst.acquire.index)
        self.assertEqual(acquired_qubits, {0, 1, 2, 3})

    def test_multiple_acquires(self):
        """Test for multiple acquires."""
        sched = pulse.Schedule()
        acq_q0 = pulse.Acquire(1200, AcquireChannel(0), MemorySlot(0))
        sched += acq_q0
        sched += acq_q0 << sched.duration
        sched = transforms.add_implicit_acquires(sched, meas_map=[[0]])
        self.assertEqual(sched.instructions, ((0, acq_q0), (2400, acq_q0)))


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestPad(QiskitTestCase):
    """Test padding of schedule with delays."""

    def test_padding_empty_schedule(self):
        """Test padding of empty schedule."""
        self.assertEqual(pulse.Schedule(), transforms.pad(pulse.Schedule()))

    def test_padding_schedule(self):
        """Test padding schedule."""
        delay = 10
        sched = (
            Delay(delay, DriveChannel(0)).shift(10)
            + Delay(delay, DriveChannel(0)).shift(10)
            + Delay(delay, DriveChannel(1)).shift(10)
        )

        ref_sched = (
            sched  # pylint: disable=unsupported-binary-operation
            | Delay(delay, DriveChannel(0))
            | Delay(delay, DriveChannel(0)).shift(20)
            | Delay(delay, DriveChannel(1))
            | Delay(  # pylint: disable=unsupported-binary-operation
                2 * delay, DriveChannel(1)
            ).shift(20)
        )

        self.assertEqual(transforms.pad(sched), ref_sched)

    def test_padding_schedule_inverse_order(self):
        """Test padding schedule is insensitive to order in which commands were added.

        This test is the same as `test_adding_schedule` but the order by channel
        in which commands were added to the schedule to be padded has been reversed.
        """
        delay = 10
        sched = (
            Delay(delay, DriveChannel(1)).shift(10)
            + Delay(delay, DriveChannel(0)).shift(10)
            + Delay(delay, DriveChannel(0)).shift(10)
        )

        ref_sched = (
            sched  # pylint: disable=unsupported-binary-operation
            | Delay(delay, DriveChannel(0))
            | Delay(delay, DriveChannel(0)).shift(20)
            | Delay(delay, DriveChannel(1))
            | Delay(  # pylint: disable=unsupported-binary-operation
                2 * delay, DriveChannel(1)
            ).shift(20)
        )

        self.assertEqual(transforms.pad(sched), ref_sched)

    def test_padding_until_less(self):
        """Test padding until time that is less than schedule duration."""
        delay = 10

        sched = Delay(delay, DriveChannel(0)).shift(10) + Delay(delay, DriveChannel(1))

        ref_sched = sched | Delay(delay, DriveChannel(0)) | Delay(5, DriveChannel(1)).shift(10)

        self.assertEqual(transforms.pad(sched, until=15), ref_sched)

    def test_padding_until_greater(self):
        """Test padding until time that is greater than schedule duration."""
        delay = 10

        sched = Delay(delay, DriveChannel(0)).shift(10) + Delay(delay, DriveChannel(1))

        ref_sched = (
            sched  # pylint: disable=unsupported-binary-operation
            | Delay(delay, DriveChannel(0))
            | Delay(30, DriveChannel(0)).shift(20)
            | Delay(40, DriveChannel(1)).shift(10)  # pylint: disable=unsupported-binary-operation
        )

        self.assertEqual(transforms.pad(sched, until=50), ref_sched)

    def test_padding_supplied_channels(self):
        """Test padding of only specified channels."""
        delay = 10
        sched = Delay(delay, DriveChannel(0)).shift(10) + Delay(delay, DriveChannel(1))

        ref_sched = sched | Delay(delay, DriveChannel(0)) | Delay(2 * delay, DriveChannel(2))

        channels = [DriveChannel(0), DriveChannel(2)]

        self.assertEqual(transforms.pad(sched, channels=channels), ref_sched)

    def test_padding_less_than_sched_duration(self):
        """Test that the until arg is respected even for less than the input schedule duration."""
        delay = 10
        sched = Delay(delay, DriveChannel(0)) + Delay(delay, DriveChannel(0)).shift(20)
        ref_sched = sched | pulse.Delay(5, DriveChannel(0)).shift(10)
        self.assertEqual(transforms.pad(sched, until=15), ref_sched)

    def test_padding_prepended_delay(self):
        """Test that there is delay before the first instruction."""
        delay = 10
        sched = Delay(delay, DriveChannel(0)).shift(10) + Delay(delay, DriveChannel(0))

        ref_sched = (
            Delay(delay, DriveChannel(0))
            + Delay(delay, DriveChannel(0))
            + Delay(delay, DriveChannel(0))
        )

        self.assertEqual(transforms.pad(sched, until=30, inplace=True), ref_sched)

    def test_pad_no_delay_on_classical_io_channels(self):
        """Test padding does not apply to classical IO channels."""
        delay = 10
        sched = (
            Delay(delay, MemorySlot(0)).shift(20)
            + Delay(delay, RegisterSlot(0)).shift(10)
            + Delay(delay, SnapshotChannel())
        )

        ref_sched = (
            Delay(delay, MemorySlot(0)).shift(20)
            + Delay(delay, RegisterSlot(0)).shift(10)
            + Delay(delay, SnapshotChannel())
        )

        self.assertEqual(transforms.pad(sched, until=15), ref_sched)


def get_pulse_ids(schedules: List[Schedule]) -> Set[int]:
    """Returns ids of pulses used in Schedules."""
    ids = set()
    for schedule in schedules:
        for _, inst in schedule.instructions:
            ids.add(inst.pulse.id)
    return ids


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestCompressTransform(QiskitTestCase):
    """Compress function test."""

    def test_with_duplicates(self):
        """Test compression of schedule."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(Waveform([0.0, 0.1]), drive_channel)
        schedule += Play(Waveform([0.0, 0.1]), drive_channel)

        compressed_schedule = transforms.compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)

        self.assertEqual(len(compressed_pulse_ids), 1)
        self.assertEqual(len(original_pulse_ids), 2)
        self.assertTrue(next(iter(compressed_pulse_ids)) in original_pulse_ids)

    def test_sample_pulse_with_clipping(self):
        """Test sample pulses with clipping."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(Waveform([0.0, 1.0]), drive_channel)
        schedule += Play(Waveform([0.0, 1.001], epsilon=1e-3), drive_channel)
        schedule += Play(Waveform([0.0, 1.0000000001]), drive_channel)

        compressed_schedule = transforms.compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)

        self.assertEqual(len(compressed_pulse_ids), 1)
        self.assertEqual(len(original_pulse_ids), 3)
        self.assertTrue(next(iter(compressed_pulse_ids)) in original_pulse_ids)

    def test_no_duplicates(self):
        """Test with no pulse duplicates."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(Waveform([0.0, 1.0]), drive_channel)
        schedule += Play(Waveform([0.0, 0.9]), drive_channel)
        schedule += Play(Waveform([0.0, 0.3]), drive_channel)

        compressed_schedule = transforms.compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), len(compressed_pulse_ids))

    def test_parametric_pulses_with_duplicates(self):
        """Test with parametric pulses."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(Gaussian(duration=25, sigma=4, amp=0.5, angle=np.pi / 2), drive_channel)
        schedule += Play(Gaussian(duration=25, sigma=4, amp=0.5, angle=np.pi / 2), drive_channel)
        schedule += Play(GaussianSquare(duration=150, amp=0.2, sigma=8, width=140), drive_channel)
        schedule += Play(GaussianSquare(duration=150, amp=0.2, sigma=8, width=140), drive_channel)
        schedule += Play(Constant(duration=150, amp=0.5, angle=0.7), drive_channel)
        schedule += Play(Constant(duration=150, amp=0.5, angle=0.7), drive_channel)
        schedule += Play(Drag(duration=25, amp=0.4, angle=-0.3, sigma=7.8, beta=4), drive_channel)
        schedule += Play(Drag(duration=25, amp=0.4, angle=-0.3, sigma=7.8, beta=4), drive_channel)

        compressed_schedule = transforms.compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), 8)
        self.assertEqual(len(compressed_pulse_ids), 4)

    def test_parametric_pulses_with_no_duplicates(self):
        """Test parametric pulses with no duplicates."""
        schedule = Schedule()
        drive_channel = DriveChannel(0)
        schedule += Play(Gaussian(duration=25, sigma=4, amp=0.5, angle=np.pi / 2), drive_channel)
        schedule += Play(Gaussian(duration=25, sigma=4, amp=0.49, angle=np.pi / 2), drive_channel)
        schedule += Play(GaussianSquare(duration=150, amp=0.2, sigma=8, width=140), drive_channel)
        schedule += Play(GaussianSquare(duration=150, amp=0.19, sigma=8, width=140), drive_channel)
        schedule += Play(Constant(duration=150, amp=0.5, angle=0.3), drive_channel)
        schedule += Play(Constant(duration=150, amp=0.51, angle=0.3), drive_channel)
        schedule += Play(Drag(duration=25, amp=0.5, angle=0.5, sigma=7.8, beta=4), drive_channel)
        schedule += Play(Drag(duration=25, amp=0.5, angle=0.51, sigma=7.8, beta=4), drive_channel)

        compressed_schedule = transforms.compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), len(compressed_pulse_ids))

    def test_with_different_channels(self):
        """Test with different channels."""
        schedule = Schedule()
        schedule += Play(Waveform([0.0, 0.1]), DriveChannel(0))
        schedule += Play(Waveform([0.0, 0.1]), DriveChannel(1))

        compressed_schedule = transforms.compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), 2)
        self.assertEqual(len(compressed_pulse_ids), 1)

    def test_sample_pulses_with_tolerance(self):
        """Test sample pulses with tolerance."""
        schedule = Schedule()
        schedule += Play(Waveform([0.0, 0.1001], epsilon=1e-3), DriveChannel(0))
        schedule += Play(Waveform([0.0, 0.1], epsilon=1e-3), DriveChannel(1))

        compressed_schedule = transforms.compress_pulses([schedule])
        original_pulse_ids = get_pulse_ids([schedule])
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), 2)
        self.assertEqual(len(compressed_pulse_ids), 1)

    def test_multiple_schedules(self):
        """Test multiple schedules."""
        schedules = []
        for _ in range(2):
            schedule = Schedule()
            drive_channel = DriveChannel(0)
            schedule += Play(Waveform([0.0, 0.1]), drive_channel)
            schedule += Play(Waveform([0.0, 0.1]), drive_channel)
            schedule += Play(Waveform([0.0, 0.2]), drive_channel)
            schedules.append(schedule)

        compressed_schedule = transforms.compress_pulses(schedules)
        original_pulse_ids = get_pulse_ids(schedules)
        compressed_pulse_ids = get_pulse_ids(compressed_schedule)
        self.assertEqual(len(original_pulse_ids), 6)
        self.assertEqual(len(compressed_pulse_ids), 2)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestAlignSequential(QiskitTestCase):
    """Test sequential alignment transform."""

    def test_align_sequential(self):
        """Test sequential alignment without a barrier."""
        context = transforms.AlignSequential()

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        schedule = pulse.Schedule()
        schedule.insert(1, instructions.Delay(3, d0), inplace=True)
        schedule.insert(4, instructions.Delay(5, d1), inplace=True)
        schedule.insert(12, instructions.Delay(7, d0), inplace=True)
        schedule = context.align(schedule)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(8, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(3, instructions.Delay(5, d1), inplace=True)

        self.assertEqual(schedule, reference)

    def test_align_sequential_with_barrier(self):
        """Test sequential alignment with a barrier."""
        context = transforms.AlignSequential()

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        schedule = pulse.Schedule()
        schedule.insert(1, instructions.Delay(3, d0), inplace=True)
        schedule.append(directives.RelativeBarrier(d0, d1), inplace=True)
        schedule.insert(4, instructions.Delay(5, d1), inplace=True)
        schedule.insert(12, instructions.Delay(7, d0), inplace=True)
        schedule = context.align(schedule)

        reference = pulse.Schedule()
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(3, directives.RelativeBarrier(d0, d1), inplace=True)
        reference.insert(3, instructions.Delay(5, d1), inplace=True)
        reference.insert(8, instructions.Delay(7, d0), inplace=True)

        self.assertEqual(schedule, reference)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestAlignLeft(QiskitTestCase):
    """Test left alignment transform."""

    def test_align_left(self):
        """Test left alignment without a barrier."""
        context = transforms.AlignLeft()

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        schedule = pulse.Schedule()
        schedule.insert(1, instructions.Delay(3, d0), inplace=True)
        schedule.insert(17, instructions.Delay(11, d2), inplace=True)

        sched_grouped = pulse.Schedule()
        sched_grouped += instructions.Delay(5, d1)
        sched_grouped += instructions.Delay(7, d0)
        schedule.append(sched_grouped, inplace=True)
        schedule = context.align(schedule)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(3, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(3, instructions.Delay(5, d1), inplace=True)
        # d2
        reference.insert(0, instructions.Delay(11, d2), inplace=True)

        self.assertEqual(schedule, reference)

    def test_align_left_with_barrier(self):
        """Test left alignment with a barrier."""
        context = transforms.AlignLeft()

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        schedule = pulse.Schedule()
        schedule.insert(1, instructions.Delay(3, d0), inplace=True)
        schedule.append(directives.RelativeBarrier(d0, d1, d2), inplace=True)
        schedule.insert(17, instructions.Delay(11, d2), inplace=True)

        sched_grouped = pulse.Schedule()
        sched_grouped += instructions.Delay(5, d1)
        sched_grouped += instructions.Delay(7, d0)
        schedule.append(sched_grouped, inplace=True)
        schedule = transforms.remove_directives(context.align(schedule))

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(3, instructions.Delay(7, d0), inplace=True)
        # d1
        reference = reference.insert(3, instructions.Delay(5, d1))
        # d2
        reference = reference.insert(3, instructions.Delay(11, d2))

        self.assertEqual(schedule, reference)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestAlignRight(QiskitTestCase):
    """Test right alignment transform."""

    def test_align_right(self):
        """Test right alignment without a barrier."""
        context = transforms.AlignRight()

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        schedule = pulse.Schedule()
        schedule.insert(1, instructions.Delay(3, d0), inplace=True)
        schedule.insert(17, instructions.Delay(11, d2), inplace=True)

        sched_grouped = pulse.Schedule()
        sched_grouped.insert(2, instructions.Delay(5, d1), inplace=True)
        sched_grouped += instructions.Delay(7, d0)

        schedule.append(sched_grouped, inplace=True)
        schedule = context.align(schedule)

        reference = pulse.Schedule()
        # d0
        reference.insert(1, instructions.Delay(3, d0), inplace=True)
        reference.insert(4, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(6, instructions.Delay(5, d1), inplace=True)
        # d2
        reference.insert(0, instructions.Delay(11, d2), inplace=True)
        self.assertEqual(schedule, reference)

    def test_align_right_with_barrier(self):
        """Test right alignment with a barrier."""
        context = transforms.AlignRight()

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        schedule = pulse.Schedule()
        schedule.insert(1, instructions.Delay(3, d0), inplace=True)
        schedule.append(directives.RelativeBarrier(d0, d1, d2), inplace=True)
        schedule.insert(17, instructions.Delay(11, d2), inplace=True)

        sched_grouped = pulse.Schedule()
        sched_grouped.insert(2, instructions.Delay(5, d1), inplace=True)
        sched_grouped += instructions.Delay(7, d0)

        schedule.append(sched_grouped, inplace=True)
        schedule = transforms.remove_directives(context.align(schedule))

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(7, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(9, instructions.Delay(5, d1), inplace=True)
        # d2
        reference.insert(3, instructions.Delay(11, d2), inplace=True)

        self.assertEqual(schedule, reference)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestAlignEquispaced(QiskitTestCase):
    """Test equispaced alignment transform."""

    def test_equispaced_with_short_duration(self):
        """Test equispaced context with duration shorter than the schedule duration."""
        context = transforms.AlignEquispaced(duration=20)

        d0 = pulse.DriveChannel(0)

        schedule = pulse.Schedule()
        for _ in range(3):
            schedule.append(Delay(10, d0), inplace=True)
        schedule = context.align(schedule)

        reference = pulse.Schedule()
        reference.insert(0, Delay(10, d0), inplace=True)
        reference.insert(10, Delay(10, d0), inplace=True)
        reference.insert(20, Delay(10, d0), inplace=True)

        self.assertEqual(schedule, reference)

    def test_equispaced_with_longer_duration(self):
        """Test equispaced context with duration longer than the schedule duration."""
        context = transforms.AlignEquispaced(duration=50)

        d0 = pulse.DriveChannel(0)

        schedule = pulse.Schedule()
        for _ in range(3):
            schedule.append(Delay(10, d0), inplace=True)
        schedule = context.align(schedule)

        reference = pulse.Schedule()
        reference.insert(0, Delay(10, d0), inplace=True)
        reference.insert(20, Delay(10, d0), inplace=True)
        reference.insert(40, Delay(10, d0), inplace=True)

        self.assertEqual(schedule, reference)

    def test_equispaced_with_multiple_channels_short_duration(self):
        """Test equispaced context with multiple channels and duration shorter than the total
        duration."""
        context = transforms.AlignEquispaced(duration=20)

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        schedule = pulse.Schedule()
        schedule.append(Delay(10, d0), inplace=True)
        schedule.append(Delay(20, d1), inplace=True)
        schedule = context.align(schedule)

        reference = pulse.Schedule()
        reference.insert(0, Delay(10, d0), inplace=True)
        reference.insert(0, Delay(20, d1), inplace=True)

        self.assertEqual(schedule, reference)

    def test_equispaced_with_multiple_channels_longer_duration(self):
        """Test equispaced context with multiple channels and duration longer than the total
        duration."""
        context = transforms.AlignEquispaced(duration=30)

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        schedule = pulse.Schedule()
        schedule.append(Delay(10, d0), inplace=True)
        schedule.append(Delay(20, d1), inplace=True)
        schedule = context.align(schedule)

        reference = pulse.Schedule()
        reference.insert(0, Delay(10, d0), inplace=True)
        reference.insert(10, Delay(20, d1), inplace=True)

        self.assertEqual(schedule, reference)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestAlignFunc(QiskitTestCase):
    """Test callback alignment transform."""

    @staticmethod
    def _position(ind):
        """Returns 0.25, 0.5, 0.75 for ind = 1, 2, 3."""
        return ind / (3 + 1)

    def test_numerical_with_short_duration(self):
        """Test numerical alignment context with duration shorter than the schedule duration."""
        context = transforms.AlignFunc(duration=20, func=self._position)

        d0 = pulse.DriveChannel(0)

        schedule = pulse.Schedule()
        for _ in range(3):
            schedule.append(Delay(10, d0), inplace=True)
        schedule = context.align(schedule)

        reference = pulse.Schedule()
        reference.insert(0, Delay(10, d0), inplace=True)
        reference.insert(10, Delay(10, d0), inplace=True)
        reference.insert(20, Delay(10, d0), inplace=True)

        self.assertEqual(schedule, reference)

    def test_numerical_with_longer_duration(self):
        """Test numerical alignment context with duration longer than the schedule duration."""
        context = transforms.AlignFunc(duration=80, func=self._position)

        d0 = pulse.DriveChannel(0)

        schedule = pulse.Schedule()
        for _ in range(3):
            schedule.append(Delay(10, d0), inplace=True)
        schedule = context.align(schedule)

        reference = pulse.Schedule()
        reference.insert(15, Delay(10, d0), inplace=True)
        reference.insert(35, Delay(10, d0), inplace=True)
        reference.insert(55, Delay(10, d0), inplace=True)

        self.assertEqual(schedule, reference)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestFlatten(QiskitTestCase):
    """Test flattening transform."""

    def test_flatten(self):
        """Test the flatten transform."""
        context_left = transforms.AlignLeft()

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        schedule = pulse.Schedule()
        schedule += instructions.Delay(3, d0)

        grouped = pulse.Schedule()
        grouped += instructions.Delay(5, d1)
        grouped += instructions.Delay(7, d0)
        # include a grouped schedule
        grouped = schedule + grouped

        # flatten the schedule inline internal groups
        flattened = transforms.flatten(grouped)

        # align all the instructions to the left after flattening
        flattened = context_left.align(flattened)
        grouped = context_left.align(grouped)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(3, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(0, instructions.Delay(5, d1), inplace=True)

        self.assertEqual(flattened, reference)
        self.assertNotEqual(grouped, reference)


class _TestDirective(directives.Directive):
    """Pulse ``RelativeBarrier`` directive."""

    def __init__(self, *channels):
        """Test directive"""
        super().__init__(operands=tuple(channels))

    @property
    def channels(self):
        return self.operands


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestRemoveDirectives(QiskitTestCase):
    """Test removing of directives."""

    def test_remove_directives(self):
        """Test that all directives are removed."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        schedule = pulse.Schedule()
        schedule += _TestDirective(d0, d1)
        schedule += instructions.Delay(3, d0)
        schedule += _TestDirective(d0, d1)
        schedule = transforms.remove_directives(schedule)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Delay(3, d0)
        self.assertEqual(schedule, reference)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestRemoveTrivialBarriers(QiskitTestCase):
    """Test scheduling transforms."""

    def test_remove_trivial_barriers(self):
        """Test that trivial barriers are properly removed."""
        schedule = pulse.Schedule()
        schedule += directives.RelativeBarrier()
        schedule += directives.RelativeBarrier(pulse.DriveChannel(0))
        schedule += directives.RelativeBarrier(pulse.DriveChannel(0), pulse.DriveChannel(1))
        schedule = transforms.remove_trivial_barriers(schedule)

        reference = pulse.Schedule()
        reference += directives.RelativeBarrier(pulse.DriveChannel(0), pulse.DriveChannel(1))
        self.assertEqual(schedule, reference)


if __name__ == "__main__":
    unittest.main()

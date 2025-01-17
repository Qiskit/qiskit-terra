# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for the pulse schedule."""
import unittest
from unittest.mock import patch

import numpy as np

from qiskit.pulse import (
    Play,
    Waveform,
    ShiftPhase,
    Instruction,
    SetFrequency,
    Acquire,
    Snapshot,
    Delay,
    library,
    Gaussian,
    Drag,
    GaussianSquare,
    Constant,
    functional_pulse,
    ShiftFrequency,
    SetPhase,
)
from qiskit.pulse.channels import (
    MemorySlot,
    RegisterSlot,
    DriveChannel,
    ControlChannel,
    AcquireChannel,
    SnapshotChannel,
    MeasureChannel,
)
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, _overlaps, _find_insertion_index
from qiskit.providers.fake_provider import FakeOpenPulse2Q
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class BaseTestSchedule(QiskitTestCase):
    """Schedule tests."""

    @ignore_pulse_deprecation_warnings
    def setUp(self):
        super().setUp()

        @functional_pulse
        def linear(duration, slope, intercept):
            x = np.linspace(0, duration - 1, duration)
            return slope * x + intercept

        self.linear = linear
        with self.assertWarns(DeprecationWarning):
            self.config = FakeOpenPulse2Q().configuration()


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestScheduleBuilding(BaseTestSchedule):
    """Test construction of schedules."""

    def test_append_an_instruction_to_empty_schedule(self):
        """Test append instructions to an empty schedule."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.append(Play(lp0, self.config.drive(0)))
        self.assertEqual(0, sched.start_time)
        self.assertEqual(3, sched.stop_time)

    def test_append_instructions_applying_to_different_channels(self):
        """Test append instructions to schedule."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.append(Play(lp0, self.config.drive(0)))
        sched = sched.append(Play(lp0, self.config.drive(1)))
        self.assertEqual(0, sched.start_time)
        # appending to separate channel so should be at same time.
        self.assertEqual(3, sched.stop_time)

    def test_insert_an_instruction_into_empty_schedule(self):
        """Test insert an instruction into an empty schedule."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.insert(10, Play(lp0, self.config.drive(0)))
        self.assertEqual(10, sched.start_time)
        self.assertEqual(13, sched.stop_time)

    def test_insert_an_instruction_before_an_existing_instruction(self):
        """Test insert an instruction before an existing instruction."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.insert(10, Play(lp0, self.config.drive(0)))
        sched = sched.insert(5, Play(lp0, self.config.drive(0)))
        self.assertEqual(5, sched.start_time)
        self.assertEqual(13, sched.stop_time)

    def test_fail_to_insert_instruction_into_occupied_timing(self):
        """Test insert an instruction before an existing instruction."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        sched = Schedule()
        sched = sched.insert(10, Play(lp0, self.config.drive(0)))
        with self.assertRaises(PulseError):
            sched.insert(11, Play(lp0, self.config.drive(0)))

    def test_can_create_valid_schedule(self):
        """Test valid schedule creation without error."""
        gp0 = library.Gaussian(duration=20, amp=0.7, sigma=3)
        gp1 = library.Gaussian(duration=20, amp=0.7, sigma=3)

        sched = Schedule()
        sched = sched.append(Play(gp0, self.config.drive(0)))
        sched = sched.insert(60, ShiftPhase(-1.57, self.config.drive(0)))
        sched = sched.insert(30, Play(gp1, self.config.drive(0)))
        sched = sched.insert(60, Play(gp0, self.config.control([0, 1])[0]))
        sched = sched.insert(80, Snapshot("label", "snap_type"))
        sched = sched.insert(90, ShiftPhase(1.57, self.config.drive(0)))
        sched = sched.insert(
            90, Acquire(10, self.config.acquire(0), MemorySlot(0), RegisterSlot(0))
        )
        self.assertEqual(0, sched.start_time)
        self.assertEqual(100, sched.stop_time)
        self.assertEqual(100, sched.duration)
        new_sched = Schedule()
        new_sched = new_sched.append(sched)
        new_sched = new_sched.append(sched)
        self.assertEqual(0, new_sched.start_time)
        self.assertEqual(200, new_sched.stop_time)
        self.assertEqual(200, new_sched.duration)
        ids = set()
        for _, inst in sched.instructions:
            self.assertFalse(inst.id in ids)
            ids.add(inst.id)

    def test_can_create_valid_schedule_with_syntax_sugar(self):
        """Test that in place operations on schedule are still immutable
        and return equivalent schedules."""
        gp0 = library.Gaussian(duration=20, amp=0.7, sigma=3)
        gp1 = library.Gaussian(duration=20, amp=0.5, sigma=3)

        sched = Schedule()
        sched += Play(gp0, self.config.drive(0))
        sched |= ShiftPhase(-1.57, self.config.drive(0)) << 60
        sched |= Play(gp1, self.config.drive(0)) << 30
        sched |= Play(gp0, self.config.control(qubits=[0, 1])[0]) << 60
        sched |= Snapshot("label", "snap_type") << 60
        sched |= ShiftPhase(1.57, self.config.drive(0)) << 90
        sched |= Acquire(10, self.config.acquire(0), MemorySlot(0)) << 90
        sched += sched

    def test_immutability(self):
        """Test that operations are immutable."""
        gp0 = library.Gaussian(duration=100, amp=0.7, sigma=3)
        gp1 = library.Gaussian(duration=20, amp=0.5, sigma=3)

        sched = Play(gp1, self.config.drive(0)) << 100
        # if schedule was mutable the next two sequences would overlap and an error
        # would be raised.
        sched.insert(0, Play(gp0, self.config.drive(0)))
        sched.insert(0, Play(gp0, self.config.drive(0)))

    def test_inplace(self):
        """Test that in place operations on schedule are still immutable."""
        gp0 = library.Gaussian(duration=100, amp=0.7, sigma=3)
        gp1 = library.Gaussian(duration=20, amp=0.5, sigma=3)

        sched = Schedule()
        sched = sched + Play(gp1, self.config.drive(0))
        sched2 = sched
        sched += Play(gp0, self.config.drive(0))
        self.assertNotEqual(sched, sched2)

    def test_empty_schedule(self):
        """Test empty schedule."""
        sched = Schedule()
        self.assertEqual(0, sched.start_time)
        self.assertEqual(0, sched.stop_time)
        self.assertEqual(0, sched.duration)
        self.assertEqual(0, len(sched))
        self.assertEqual((), sched.children)
        self.assertEqual({}, sched.timeslots)
        self.assertEqual([], list(sched.instructions))
        self.assertFalse(sched)

    def test_overlapping_schedules(self):
        """Test overlapping schedules."""

        def my_test_make_schedule(acquire: int, memoryslot: int, shift: int):
            sched1 = Acquire(acquire, AcquireChannel(0), MemorySlot(memoryslot))
            sched2 = Acquire(acquire, AcquireChannel(1), MemorySlot(memoryslot)).shift(shift)

            return Schedule(sched1, sched2)

        self.assertIsInstance(my_test_make_schedule(4, 0, 4), Schedule)
        self.assertRaisesRegex(
            PulseError, r".*MemorySlot\(0\).*overlaps .*", my_test_make_schedule, 4, 0, 2
        )
        self.assertRaisesRegex(
            PulseError, r".*MemorySlot\(1\).*overlaps .*", my_test_make_schedule, 4, 1, 0
        )

    def test_flat_instruction_sequence_returns_instructions(self):
        """Test if `flat_instruction_sequence` returns `Instruction`s."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)

        # empty schedule with empty schedule
        empty = Schedule().append(Schedule())
        for _, instr in empty.instructions:
            self.assertIsInstance(instr, Instruction)

        # normal schedule
        subsched = Schedule()
        subsched = subsched.insert(20, Play(lp0, self.config.drive(0)))  # grand child 1
        subsched = subsched.append(Play(lp0, self.config.drive(0)))  # grand child 2

        sched = Schedule()
        sched = sched.append(Play(lp0, self.config.drive(0)))  # child
        sched = sched.append(subsched)
        for _, instr in sched.instructions:
            self.assertIsInstance(instr, Instruction)

    def test_absolute_start_time_of_grandchild(self):
        """Test correct calculation of start time of grandchild of a schedule."""
        lp0 = self.linear(duration=10, slope=0.02, intercept=0.01)

        subsched = Schedule()
        subsched = subsched.insert(20, Play(lp0, self.config.drive(0)))  # grand child 1
        subsched = subsched.append(Play(lp0, self.config.drive(0)))  # grand child 2

        sched = Schedule()
        sched = sched.append(Play(lp0, self.config.drive(0)))  # child
        sched = sched.append(subsched)

        start_times = sorted(shft + instr.start_time for shft, instr in sched.instructions)
        self.assertEqual([0, 30, 40], start_times)

    def test_shift_schedule(self):
        """Test shift schedule."""
        lp0 = self.linear(duration=10, slope=0.02, intercept=0.01)

        subsched = Schedule()
        subsched = subsched.insert(20, Play(lp0, self.config.drive(0)))  # grand child 1
        subsched = subsched.append(Play(lp0, self.config.drive(0)))  # grand child 2

        sched = Schedule()
        sched = sched.append(Play(lp0, self.config.drive(0)))  # child
        sched = sched.append(subsched)

        shift = sched.shift(100)

        start_times = sorted(shft + instr.start_time for shft, instr in shift.instructions)

        self.assertEqual([100, 130, 140], start_times)

    def test_keep_original_schedule_after_attached_to_another_schedule(self):
        """Test if a schedule keeps its children after attached to another schedule."""
        children = Acquire(10, self.config.acquire(0), MemorySlot(0)).shift(20) + Acquire(
            10, self.config.acquire(0), MemorySlot(0)
        )
        self.assertEqual(2, len(list(children.instructions)))

        sched = Acquire(10, self.config.acquire(0), MemorySlot(0)).append(children)
        self.assertEqual(3, len(list(sched.instructions)))

        # add 2 instructions to children (2 instructions -> 4 instructions)
        children = children.append(Acquire(10, self.config.acquire(0), MemorySlot(0)))
        children = children.insert(100, Acquire(10, self.config.acquire(0), MemorySlot(0)))
        self.assertEqual(4, len(list(children.instructions)))
        # sched must keep 3 instructions (must not update to 5 instructions)
        self.assertEqual(3, len(list(sched.instructions)))

    @patch("qiskit.utils.is_main_process", return_value=True)
    def test_auto_naming(self, is_main_process_mock):
        """Test that a schedule gets a default name, incremented per instance"""

        del is_main_process_mock

        sched_0 = Schedule()
        sched_0_name_count = int(sched_0.name[len("sched") :])

        sched_1 = Schedule()
        sched_1_name_count = int(sched_1.name[len("sched") :])
        self.assertEqual(sched_1_name_count, sched_0_name_count + 1)

        sched_2 = Schedule()
        sched_2_name_count = int(sched_2.name[len("sched") :])
        self.assertEqual(sched_2_name_count, sched_1_name_count + 1)

    def test_name_inherited(self):
        """Test that schedule keeps name if an instruction is added."""
        gp0 = library.Gaussian(duration=100, amp=0.7, sigma=3, name="pulse_name")
        snapshot = Snapshot("snapshot_label", "state")

        sched1 = Schedule(name="test_name")
        sched2 = Schedule(name=None)
        sched3 = sched1 | sched2
        self.assertEqual(sched3.name, "test_name")

        sched_acq = Acquire(10, self.config.acquire(1), MemorySlot(1), name="acq_name") | sched1
        self.assertEqual(sched_acq.name, "acq_name")

        sched_pulse = Play(gp0, self.config.drive(0)) | sched1
        self.assertEqual(sched_pulse.name, "pulse_name")

        sched_fc = ShiftPhase(0.1, self.config.drive(0), name="fc_name") | sched1
        self.assertEqual(sched_fc.name, "fc_name")

        sched_snapshot = snapshot | sched1
        self.assertEqual(sched_snapshot.name, "snapshot_label")

    def test_schedule_with_acquire_on_single_qubit(self):
        """Test schedule with acquire on single qubit."""
        sched_single = Schedule()
        for i in range(self.config.n_qubits):
            sched_single = sched_single.insert(
                10,
                Acquire(
                    10, self.config.acquire(i), mem_slot=MemorySlot(i), reg_slot=RegisterSlot(i)
                ),
            )

        self.assertEqual(len(sched_single.instructions), 2)
        self.assertEqual(len(sched_single.channels), 6)

    def test_parametric_commands_in_sched(self):
        """Test that schedules can be built with parametric commands."""
        sched = Schedule(name="test_parametric")
        sched += Play(Gaussian(duration=25, sigma=4, amp=0.5, angle=np.pi / 2), DriveChannel(0))
        sched += Play(Drag(duration=25, amp=0.4, angle=0.5, sigma=7.8, beta=4), DriveChannel(1))
        sched += Play(Constant(duration=25, amp=1), DriveChannel(2))
        sched_duration = sched.duration
        sched += (
            Play(GaussianSquare(duration=1500, amp=0.2, sigma=8, width=140), MeasureChannel(0))
            << sched_duration
        )
        self.assertEqual(sched.duration, 1525)
        self.assertTrue("sigma" in sched.instructions[0][1].pulse.parameters)

    def test_numpy_integer_input(self):
        """Test that mixed integer duration types can build a schedule (#5754)."""
        sched = Schedule()
        sched += Delay(np.int32(25), DriveChannel(0))
        sched += Play(Constant(duration=30, amp=0.1), DriveChannel(0))
        self.assertEqual(sched.duration, 55)

    def test_negative_time_raises(self):
        """Test that a negative time will raise an error."""
        sched = Schedule()
        sched += Delay(1, DriveChannel(0))
        with self.assertRaises(PulseError):
            sched.shift(-10)

    def test_shift_float_time_raises(self):
        """Test that a floating time will raise an error with shift."""
        sched = Schedule()
        sched += Delay(1, DriveChannel(0))
        with self.assertRaises(PulseError):
            sched.shift(0.1)

    def test_insert_float_time_raises(self):
        """Test that a floating time will raise an error with insert."""
        sched = Schedule()
        sched += Delay(1, DriveChannel(0))
        with self.assertRaises(PulseError):
            sched.insert(10.1, sched)

    def test_shift_unshift(self):
        """Test shift and then unshifting of schedule"""
        reference_sched = Schedule()
        reference_sched += Delay(10, DriveChannel(0))
        shifted_sched = reference_sched.shift(10).shift(-10)
        self.assertEqual(shifted_sched, reference_sched)

    def test_duration(self):
        """Test schedule.duration."""
        reference_sched = Schedule()
        reference_sched = reference_sched.insert(10, Delay(10, DriveChannel(0)))
        reference_sched = reference_sched.insert(10, Delay(50, DriveChannel(1)))
        reference_sched = reference_sched.insert(10, ShiftPhase(0.1, DriveChannel(0)))

        reference_sched = reference_sched.insert(100, ShiftPhase(0.1, DriveChannel(1)))

        self.assertEqual(reference_sched.duration, 100)
        self.assertEqual(reference_sched.duration, 100)

    def test_ch_duration(self):
        """Test schedule.ch_duration."""
        reference_sched = Schedule()
        reference_sched = reference_sched.insert(10, Delay(10, DriveChannel(0)))
        reference_sched = reference_sched.insert(10, Delay(50, DriveChannel(1)))
        reference_sched = reference_sched.insert(10, ShiftPhase(0.1, DriveChannel(0)))

        reference_sched = reference_sched.insert(100, ShiftPhase(0.1, DriveChannel(1)))

        self.assertEqual(reference_sched.ch_duration(DriveChannel(0)), 20)
        self.assertEqual(reference_sched.ch_duration(DriveChannel(1)), 100)
        self.assertEqual(
            reference_sched.ch_duration(*reference_sched.channels), reference_sched.duration
        )

    def test_ch_start_time(self):
        """Test schedule.ch_start_time."""
        reference_sched = Schedule()
        reference_sched = reference_sched.insert(10, Delay(10, DriveChannel(0)))
        reference_sched = reference_sched.insert(10, Delay(50, DriveChannel(1)))
        reference_sched = reference_sched.insert(10, ShiftPhase(0.1, DriveChannel(0)))

        reference_sched = reference_sched.insert(100, ShiftPhase(0.1, DriveChannel(1)))

        self.assertEqual(reference_sched.ch_start_time(DriveChannel(0)), 10)
        self.assertEqual(reference_sched.ch_start_time(DriveChannel(1)), 10)

    def test_ch_stop_time(self):
        """Test schedule.ch_stop_time."""
        reference_sched = Schedule()
        reference_sched = reference_sched.insert(10, Delay(10, DriveChannel(0)))
        reference_sched = reference_sched.insert(10, Delay(50, DriveChannel(1)))
        reference_sched = reference_sched.insert(10, ShiftPhase(0.1, DriveChannel(0)))

        reference_sched = reference_sched.insert(100, ShiftPhase(0.1, DriveChannel(1)))

        self.assertEqual(reference_sched.ch_stop_time(DriveChannel(0)), 20)
        self.assertEqual(reference_sched.ch_stop_time(DriveChannel(1)), 100)

    def test_timeslots(self):
        """Test schedule.timeslots."""
        reference_sched = Schedule()
        reference_sched = reference_sched.insert(10, Delay(10, DriveChannel(0)))
        reference_sched = reference_sched.insert(10, Delay(50, DriveChannel(1)))
        reference_sched = reference_sched.insert(10, ShiftPhase(0.1, DriveChannel(0)))

        reference_sched = reference_sched.insert(100, ShiftPhase(0.1, DriveChannel(1)))

        self.assertEqual(reference_sched.timeslots[DriveChannel(0)], [(10, 10), (10, 20)])
        self.assertEqual(reference_sched.timeslots[DriveChannel(1)], [(10, 60), (100, 100)])

    def test_len(self):
        """Test __len__ method"""
        sched = Schedule()
        self.assertEqual(len(sched), 0)

        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        for j in range(1, 10):
            sched = sched.append(Play(lp0, self.config.drive(0)))
            self.assertEqual(len(sched), j)

    def test_inherit_from(self):
        """Test creating schedule with another schedule."""
        ref_metadata = {"test": "value"}
        ref_name = "test"

        base_sched = Schedule(name=ref_name, metadata=ref_metadata)
        new_sched = Schedule.initialize_from(base_sched)

        self.assertEqual(new_sched.name, ref_name)
        self.assertDictEqual(new_sched.metadata, ref_metadata)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestReplace(BaseTestSchedule):
    """Test schedule replacement."""

    def test_replace_instruction(self):
        """Test replacement of simple instruction"""
        old = Play(Constant(100, 1.0), DriveChannel(0))
        new = Play(Constant(100, 0.1), DriveChannel(0))

        sched = Schedule(old)
        new_sched = sched.replace(old, new)

        self.assertEqual(new_sched, Schedule(new))

        # test replace inplace
        sched.replace(old, new, inplace=True)
        self.assertEqual(sched, Schedule(new))

    def test_replace_schedule(self):
        """Test replacement of schedule."""

        old = Schedule(
            Delay(10, DriveChannel(0)),
            Delay(100, DriveChannel(1)),
        )
        new = Schedule(
            Play(Constant(10, 1.0), DriveChannel(0)),
            Play(Constant(100, 0.1), DriveChannel(1)),
        )
        const = Play(Constant(100, 1.0), DriveChannel(0))

        sched = Schedule()
        sched += const
        sched += old

        new_sched = sched.replace(old, new)

        ref_sched = Schedule()
        ref_sched += const
        ref_sched += new
        self.assertEqual(new_sched, ref_sched)

        # test replace inplace
        sched.replace(old, new, inplace=True)
        self.assertEqual(sched, ref_sched)

    def test_replace_fails_on_overlap(self):
        """Test that replacement fails on overlap."""
        old = Play(Constant(20, 1.0), DriveChannel(0))
        new = Play(Constant(100, 0.1), DriveChannel(0))

        sched = Schedule()
        sched += old
        sched += Delay(100, DriveChannel(0))

        with self.assertRaises(PulseError):
            sched.replace(old, new)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestDelay(BaseTestSchedule):
    """Test Delay Instruction"""

    def setUp(self):
        super().setUp()
        self.delay_time = 10

    def test_delay_drive_channel(self):
        """Test Delay on DriveChannel"""
        drive_ch = self.config.drive(0)
        pulse = Waveform(np.full(10, 0.1))
        # should pass as is an append
        sched = Delay(self.delay_time, drive_ch) + Play(pulse, drive_ch)
        self.assertIsInstance(sched, Schedule)
        pulse_instr = sched.instructions[-1]
        # assert last instruction is pulse
        self.assertIsInstance(pulse_instr[1], Play)
        # assert pulse is scheduled at time 10
        self.assertEqual(pulse_instr[0], 10)
        # should fail due to overlap
        with self.assertRaises(PulseError):
            sched = Delay(self.delay_time, drive_ch) | Play(pulse, drive_ch)

    def test_delay_measure_channel(self):
        """Test Delay on MeasureChannel"""

        measure_ch = self.config.measure(0)
        pulse = Waveform(np.full(10, 0.1))
        # should pass as is an append
        sched = Delay(self.delay_time, measure_ch) + Play(pulse, measure_ch)
        self.assertIsInstance(sched, Schedule)
        # should fail due to overlap
        with self.assertRaises(PulseError):
            sched = Delay(self.delay_time, measure_ch) | Play(pulse, measure_ch)

    def test_delay_control_channel(self):
        """Test Delay on ControlChannel"""

        control_ch = self.config.control([0, 1])[0]
        pulse = Waveform(np.full(10, 0.1))
        # should pass as is an append
        sched = Delay(self.delay_time, control_ch) + Play(pulse, control_ch)
        self.assertIsInstance(sched, Schedule)
        # should fail due to overlap
        with self.assertRaises(PulseError):
            sched = Delay(self.delay_time, control_ch) | Play(pulse, control_ch)
            self.assertIsInstance(sched, Schedule)

    def test_delay_acquire_channel(self):
        """Test Delay on DriveChannel"""

        acquire_ch = self.config.acquire(0)
        # should pass as is an append
        sched = Delay(self.delay_time, acquire_ch) + Acquire(10, acquire_ch, MemorySlot(0))
        self.assertIsInstance(sched, Schedule)
        # should fail due to overlap
        with self.assertRaises(PulseError):
            sched = Delay(self.delay_time, acquire_ch) | Acquire(10, acquire_ch, MemorySlot(0))
            self.assertIsInstance(sched, Schedule)

    def test_delay_snapshot_channel(self):
        """Test Delay on DriveChannel"""

        snapshot_ch = SnapshotChannel()
        snapshot = Snapshot(label="test")
        # should pass as is an append
        sched = Delay(self.delay_time, snapshot_ch) + snapshot
        self.assertIsInstance(sched, Schedule)
        # should fail due to overlap
        with self.assertRaises(PulseError):
            sched = Delay(self.delay_time, snapshot_ch) | snapshot << 5
            self.assertIsInstance(sched, Schedule)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestScheduleFilter(BaseTestSchedule):
    """Test Schedule filtering methods"""

    def test_filter_channels(self):
        """Test filtering over channels."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        sched = Schedule(name="fake_experiment")
        sched = sched.insert(0, Play(lp0, self.config.drive(0)))
        sched = sched.insert(10, Play(lp0, self.config.drive(1)))
        sched = sched.insert(30, ShiftPhase(-1.57, self.config.drive(0)))
        sched = sched.insert(60, Acquire(5, AcquireChannel(0), MemorySlot(0)))
        sched = sched.insert(60, Acquire(5, AcquireChannel(1), MemorySlot(1)))
        sched = sched.insert(90, Play(lp0, self.config.drive(0)))

        # split instructions for those on AcquireChannel(1) and those not
        filtered, excluded = self._filter_and_test_consistency(sched, channels=[AcquireChannel(1)])
        self.assertEqual(len(filtered.instructions), 1)
        self.assertEqual(len(excluded.instructions), 5)

        # Split schedule into the part with channels on 1 and into a part without
        channels = [AcquireChannel(1), DriveChannel(1)]
        filtered, excluded = self._filter_and_test_consistency(sched, channels=channels)
        for _, inst in filtered.instructions:
            self.assertTrue(any(chan in channels for chan in inst.channels))

        for _, inst in excluded.instructions:
            self.assertFalse(any(chan in channels for chan in inst.channels))

    def test_filter_exclude_name(self):
        """Test the name of the schedules after applying filter and exclude functions."""
        sched = Schedule(name="test-schedule")
        sched = sched.insert(10, Acquire(5, AcquireChannel(0), MemorySlot(0)))
        sched = sched.insert(10, Acquire(5, AcquireChannel(1), MemorySlot(1)))
        excluded = sched.exclude(channels=[AcquireChannel(0)])
        filtered = sched.filter(channels=[AcquireChannel(1)])

        # check if the excluded and filtered schedule have the same name as sched
        self.assertEqual(sched.name, filtered.name)
        self.assertEqual(sched.name, excluded.name)

    def test_filter_inst_types(self):
        """Test filtering on instruction types."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        sched = Schedule(name="fake_experiment")
        sched = sched.insert(0, Play(lp0, self.config.drive(0)))
        sched = sched.insert(10, Play(lp0, self.config.drive(1)))
        sched = sched.insert(30, ShiftPhase(-1.57, self.config.drive(0)))
        sched = sched.insert(40, SetFrequency(8.0, self.config.drive(0)))
        sched = sched.insert(50, ShiftFrequency(4.0e6, self.config.drive(0)))
        sched = sched.insert(55, SetPhase(3.14, self.config.drive(0)))
        for i in range(2):
            sched = sched.insert(60, Acquire(5, self.config.acquire(i), MemorySlot(i)))
        sched = sched.insert(90, Play(lp0, self.config.drive(0)))

        # test on Acquire
        only_acquire, no_acquire = self._filter_and_test_consistency(
            sched, instruction_types=[Acquire]
        )
        for _, inst in only_acquire.instructions:
            self.assertIsInstance(inst, Acquire)
        for _, inst in no_acquire.instructions:
            self.assertFalse(isinstance(inst, Acquire))

        # test two instruction types
        only_pulse_and_fc, no_pulse_and_fc = self._filter_and_test_consistency(
            sched, instruction_types=[Play, ShiftPhase]
        )
        for _, inst in only_pulse_and_fc.instructions:
            self.assertIsInstance(inst, (Play, ShiftPhase))
        for _, inst in no_pulse_and_fc.instructions:
            self.assertFalse(isinstance(inst, (Play, ShiftPhase)))
        self.assertEqual(len(only_pulse_and_fc.instructions), 4)
        self.assertEqual(len(no_pulse_and_fc.instructions), 5)

        # test on ShiftPhase
        only_fc, no_fc = self._filter_and_test_consistency(sched, instruction_types={ShiftPhase})
        self.assertEqual(len(only_fc.instructions), 1)
        self.assertEqual(len(no_fc.instructions), 8)

        # test on SetPhase
        only_setp, no_setp = self._filter_and_test_consistency(sched, instruction_types={SetPhase})
        self.assertEqual(len(only_setp.instructions), 1)
        self.assertEqual(len(no_setp.instructions), 8)

        # test on SetFrequency
        only_setf, no_setf = self._filter_and_test_consistency(
            sched, instruction_types=[SetFrequency]
        )
        for _, inst in only_setf.instructions:
            self.assertTrue(isinstance(inst, SetFrequency))
        self.assertEqual(len(only_setf.instructions), 1)
        self.assertEqual(len(no_setf.instructions), 8)

        # test on ShiftFrequency
        only_shiftf, no_shiftf = self._filter_and_test_consistency(
            sched, instruction_types=[ShiftFrequency]
        )
        for _, inst in only_shiftf.instructions:
            self.assertTrue(isinstance(inst, ShiftFrequency))
        self.assertEqual(len(only_shiftf.instructions), 1)
        self.assertEqual(len(no_shiftf.instructions), 8)

    def test_filter_intervals(self):
        """Test filtering on intervals."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        sched = Schedule(name="fake_experiment")
        sched = sched.insert(0, Play(lp0, self.config.drive(0)))
        sched = sched.insert(10, Play(lp0, self.config.drive(1)))
        sched = sched.insert(30, ShiftPhase(-1.57, self.config.drive(0)))
        for i in range(2):
            sched = sched.insert(60, Acquire(5, self.config.acquire(i), MemorySlot(i)))
        sched = sched.insert(90, Play(lp0, self.config.drive(0)))

        # split schedule into instructions occurring in (0,13), and those outside
        filtered, excluded = self._filter_and_test_consistency(sched, time_ranges=((0, 13),))
        for start_time, inst in filtered.instructions:
            self.assertTrue((start_time >= 0) and (start_time + inst.stop_time <= 13))
        for start_time, inst in excluded.instructions:
            self.assertFalse((start_time >= 0) and (start_time + inst.stop_time <= 13))
        self.assertEqual(len(filtered.instructions), 2)
        self.assertEqual(len(excluded.instructions), 4)

        # split into schedule occurring in and outside of interval (59,65)
        filtered, excluded = self._filter_and_test_consistency(sched, time_ranges=[(59, 65)])
        self.assertEqual(len(filtered.instructions), 2)
        self.assertEqual(filtered.instructions[0][0], 60)
        self.assertIsInstance(filtered.instructions[0][1], Acquire)
        self.assertEqual(len(excluded.instructions), 4)
        self.assertEqual(excluded.instructions[3][0], 90)
        self.assertIsInstance(excluded.instructions[3][1], Play)

        # split instructions based on the interval
        # (none should be, though they have some overlap with some of the instructions)
        filtered, excluded = self._filter_and_test_consistency(
            sched, time_ranges=[(0, 2), (8, 11), (61, 70)]
        )
        self.assertEqual(len(filtered.instructions), 0)
        self.assertEqual(len(excluded.instructions), 6)

        # split instructions from multiple non-overlapping intervals, specified
        # as time ranges
        filtered, excluded = self._filter_and_test_consistency(
            sched, time_ranges=[(10, 15), (63, 93)]
        )
        self.assertEqual(len(filtered.instructions), 2)
        self.assertEqual(len(excluded.instructions), 4)

        # split instructions from non-overlapping intervals, specified as Intervals
        filtered, excluded = self._filter_and_test_consistency(
            sched, intervals=[(10, 15), (63, 93)]
        )
        self.assertEqual(len(filtered.instructions), 2)
        self.assertEqual(len(excluded.instructions), 4)

    def test_filter_multiple(self):
        """Test filter composition."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        sched = Schedule(name="fake_experiment")
        sched = sched.insert(0, Play(lp0, self.config.drive(0)))
        sched = sched.insert(10, Play(lp0, self.config.drive(1)))
        sched = sched.insert(30, ShiftPhase(-1.57, self.config.drive(0)))
        for i in range(2):
            sched = sched.insert(60, Acquire(5, self.config.acquire(i), MemorySlot(i)))

        sched = sched.insert(90, Play(lp0, self.config.drive(0)))

        # split instructions with filters on channel 0, of type Play
        # occurring in the time interval (25, 100)
        filtered, excluded = self._filter_and_test_consistency(
            sched,
            channels={self.config.drive(0)},
            instruction_types=[Play],
            time_ranges=[(25, 100)],
        )
        for time, inst in filtered.instructions:
            self.assertIsInstance(inst, Play)
            self.assertTrue(all(chan.index == 0 for chan in inst.channels))
            self.assertTrue(25 <= time <= 100)
        self.assertEqual(len(excluded.instructions), 5)
        self.assertTrue(excluded.instructions[0][1].channels[0] == DriveChannel(0))
        self.assertTrue(excluded.instructions[2][0] == 30)

        # split based on Plays in the specified intervals
        filtered, excluded = self._filter_and_test_consistency(
            sched, instruction_types=[Play], time_ranges=[(25, 100), (0, 11)]
        )
        self.assertTrue(len(excluded.instructions), 3)
        for time, inst in filtered.instructions:
            self.assertIsInstance(inst, (ShiftPhase, Play))
        self.assertTrue(len(filtered.instructions), 4)
        # make sure the Play instruction is not in the intervals
        self.assertIsInstance(excluded.instructions[0][1], Play)

        # split based on Acquire in the specified intervals
        filtered, excluded = self._filter_and_test_consistency(
            sched, instruction_types=[Acquire], time_ranges=[(25, 100)]
        )
        self.assertTrue(len(excluded.instructions), 4)
        for _, inst in filtered.instructions:
            self.assertIsInstance(inst, Acquire)
        self.assertTrue(len(filtered.instructions), 2)

    def test_custom_filters(self):
        """Test custom filters."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        sched = Schedule(name="fake_experiment")
        sched = sched.insert(0, Play(lp0, self.config.drive(0)))
        sched = sched.insert(10, Play(lp0, self.config.drive(1)))
        sched = sched.insert(30, ShiftPhase(-1.57, self.config.drive(0)))

        filtered, excluded = self._filter_and_test_consistency(sched, lambda x: True)
        for i in filtered.instructions:
            self.assertTrue(i in sched.instructions)
        for i in excluded.instructions:
            self.assertFalse(i in sched.instructions)

        filtered, excluded = self._filter_and_test_consistency(sched, lambda x: False)
        self.assertEqual(len(filtered.instructions), 0)
        self.assertEqual(len(excluded.instructions), 3)

        filtered, excluded = self._filter_and_test_consistency(sched, lambda x: x[0] < 30)
        self.assertEqual(len(filtered.instructions), 2)
        self.assertEqual(len(excluded.instructions), 1)

        # multiple custom filters
        filtered, excluded = self._filter_and_test_consistency(
            sched, lambda x: x[0] > 0, lambda x: x[0] < 30
        )
        self.assertEqual(len(filtered.instructions), 1)
        self.assertEqual(len(excluded.instructions), 2)

    def test_empty_filters(self):
        """Test behavior on empty filters."""
        lp0 = self.linear(duration=3, slope=0.2, intercept=0.1)
        sched = Schedule(name="fake_experiment")
        sched = sched.insert(0, Play(lp0, self.config.drive(0)))
        sched = sched.insert(10, Play(lp0, self.config.drive(1)))
        sched = sched.insert(30, ShiftPhase(-1.57, self.config.drive(0)))
        for i in range(2):
            sched = sched.insert(60, Acquire(5, self.config.acquire(i), MemorySlot(i)))
        sched = sched.insert(90, Play(lp0, self.config.drive(0)))

        # empty channels
        filtered, excluded = self._filter_and_test_consistency(sched, channels=[])
        self.assertTrue(len(filtered.instructions) == 0)
        self.assertTrue(len(excluded.instructions) == 6)

        # empty instruction_types
        filtered, excluded = self._filter_and_test_consistency(sched, instruction_types=[])
        self.assertTrue(len(filtered.instructions) == 0)
        self.assertTrue(len(excluded.instructions) == 6)

        # empty time_ranges
        filtered, excluded = self._filter_and_test_consistency(sched, time_ranges=[])
        self.assertTrue(len(filtered.instructions) == 0)
        self.assertTrue(len(excluded.instructions) == 6)

        # empty intervals
        filtered, excluded = self._filter_and_test_consistency(sched, intervals=[])
        self.assertTrue(len(filtered.instructions) == 0)
        self.assertTrue(len(excluded.instructions) == 6)

        # empty channels with other non-empty filters
        filtered, excluded = self._filter_and_test_consistency(
            sched, channels=[], instruction_types=[Play]
        )
        self.assertTrue(len(filtered.instructions) == 0)
        self.assertTrue(len(excluded.instructions) == 6)

    def _filter_and_test_consistency(self, schedule: Schedule, *args, **kwargs):
        """
        Returns the tuple
        (schedule.filter(*args, **kwargs), schedule.exclude(*args, **kwargs)),
        including a test that schedule.filter | schedule.exclude == schedule
        """
        filtered = schedule.filter(*args, **kwargs)
        excluded = schedule.exclude(*args, **kwargs)
        self.assertEqual(filtered | excluded, schedule)
        return filtered, excluded


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestScheduleEquality(BaseTestSchedule):
    """Test equality of schedules."""

    def test_different_channels(self):
        """Test equality is False if different channels."""
        self.assertNotEqual(
            Schedule(ShiftPhase(0, DriveChannel(0))), Schedule(ShiftPhase(0, DriveChannel(1)))
        )

    def test_same_time_equal(self):
        """Test equal if instruction at same time."""

        self.assertEqual(
            Schedule((0, ShiftPhase(0, DriveChannel(1)))),
            Schedule((0, ShiftPhase(0, DriveChannel(1)))),
        )

    def test_different_time_not_equal(self):
        """Test that not equal if instruction at different time."""
        self.assertNotEqual(
            Schedule((0, ShiftPhase(0, DriveChannel(1)))),
            Schedule((1, ShiftPhase(0, DriveChannel(1)))),
        )

    def test_single_channel_out_of_order(self):
        """Test that schedule with single channel equal when out of order."""
        instructions = [
            (0, ShiftPhase(0, DriveChannel(0))),
            (15, Play(Waveform(np.ones(10)), DriveChannel(0))),
            (5, Play(Waveform(np.ones(10)), DriveChannel(0))),
        ]

        self.assertEqual(Schedule(*instructions), Schedule(*reversed(instructions)))

    def test_multiple_channels_out_of_order(self):
        """Test that schedule with multiple channels equal when out of order."""
        instructions = [
            (0, ShiftPhase(0, DriveChannel(1))),
            (1, Acquire(10, AcquireChannel(0), MemorySlot(1))),
        ]

        self.assertEqual(Schedule(*instructions), Schedule(*reversed(instructions)))

    def test_same_commands_on_two_channels_at_same_time_out_of_order(self):
        """Test that schedule with same commands on two channels at the same time equal
        when out of order."""
        sched1 = Schedule()
        sched1 = sched1.append(Delay(100, DriveChannel(1)))
        sched1 = sched1.append(Delay(100, ControlChannel(1)))
        sched2 = Schedule()
        sched2 = sched2.append(Delay(100, ControlChannel(1)))
        sched2 = sched2.append(Delay(100, DriveChannel(1)))
        self.assertEqual(sched1, sched2)

    def test_different_name_equal(self):
        """Test that names are ignored when checking equality."""

        self.assertEqual(
            Schedule((0, ShiftPhase(0, DriveChannel(1), name="fc1")), name="s1"),
            Schedule((0, ShiftPhase(0, DriveChannel(1), name="fc2")), name="s2"),
        )


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestTimingUtils(QiskitTestCase):
    """Test the Schedule helper functions."""

    def test_overlaps(self):
        """Test the `_overlaps` function."""
        a = (0, 1)
        b = (1, 4)
        c = (2, 3)
        d = (3, 5)
        self.assertFalse(_overlaps(a, b))
        self.assertFalse(_overlaps(b, a))
        self.assertFalse(_overlaps(a, d))
        self.assertTrue(_overlaps(b, c))
        self.assertTrue(_overlaps(c, b))
        self.assertTrue(_overlaps(b, d))
        self.assertTrue(_overlaps(d, b))

    def test_overlaps_zero_duration(self):
        """Test the `_overlaps` function for intervals with duration zero."""
        a = 0
        b = 1
        self.assertFalse(_overlaps((a, a), (a, a)))
        self.assertFalse(_overlaps((a, a), (a, b)))
        self.assertFalse(_overlaps((a, b), (a, a)))
        self.assertFalse(_overlaps((a, b), (b, b)))
        self.assertFalse(_overlaps((b, b), (a, b)))
        self.assertTrue(_overlaps((a, a + 2), (a + 1, a + 1)))
        self.assertTrue(_overlaps((a + 1, a + 1), (a, a + 2)))

    def test_find_insertion_index(self):
        """Test the `_find_insertion_index` function."""
        intervals = [(1, 2), (4, 5)]
        self.assertEqual(_find_insertion_index(intervals, (2, 3)), 1)
        self.assertEqual(_find_insertion_index(intervals, (3, 4)), 1)
        self.assertEqual(intervals, [(1, 2), (4, 5)])
        intervals = [(1, 2), (4, 5), (6, 7)]
        self.assertEqual(_find_insertion_index(intervals, (2, 3)), 1)
        self.assertEqual(_find_insertion_index(intervals, (0, 1)), 0)
        self.assertEqual(_find_insertion_index(intervals, (5, 6)), 2)
        self.assertEqual(_find_insertion_index(intervals, (8, 9)), 3)

        longer_intervals = [(1, 2), (2, 3), (4, 5), (5, 6), (7, 9), (11, 11)]
        self.assertEqual(_find_insertion_index(longer_intervals, (4, 4)), 2)
        self.assertEqual(_find_insertion_index(longer_intervals, (5, 5)), 3)
        self.assertEqual(_find_insertion_index(longer_intervals, (3, 4)), 2)
        self.assertEqual(_find_insertion_index(longer_intervals, (3, 4)), 2)

        # test when two identical zero duration timeslots are present
        intervals = [(0, 10), (73, 73), (73, 73), (90, 101)]
        self.assertEqual(_find_insertion_index(intervals, (42, 73)), 1)
        self.assertEqual(_find_insertion_index(intervals, (73, 81)), 3)

    def test_find_insertion_index_when_overlapping(self):
        """Test that `_find_insertion_index` raises an error when the new_interval _overlaps."""
        intervals = [(10, 20), (44, 55), (60, 61), (80, 1000)]
        with self.assertRaises(PulseError):
            _find_insertion_index(intervals, (60, 62))
        with self.assertRaises(PulseError):
            _find_insertion_index(intervals, (100, 1500))

        intervals = [(0, 1), (10, 15)]
        with self.assertRaises(PulseError):
            _find_insertion_index(intervals, (7, 13))

    def test_find_insertion_index_empty_list(self):
        """Test that the insertion index is properly found for empty lists."""
        self.assertEqual(_find_insertion_index([], (0, 1)), 0)


if __name__ == "__main__":
    unittest.main()

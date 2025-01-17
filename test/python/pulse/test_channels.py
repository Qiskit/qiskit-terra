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

"""Test cases for the pulse channel group."""

import unittest

from qiskit.pulse.channels import (
    AcquireChannel,
    Channel,
    ClassicalIOChannel,
    ControlChannel,
    DriveChannel,
    MeasureChannel,
    MemorySlot,
    PulseChannel,
    RegisterSlot,
    SnapshotChannel,
    PulseError,
)
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


class TestChannel(QiskitTestCase):
    """Test base channel."""

    def test_cannot_be_instantiated(self):
        """Test base channel cannot be instantiated."""
        with self.assertRaises(NotImplementedError):
            Channel(0)


class TestPulseChannel(QiskitTestCase):
    """Test base pulse channel."""

    def test_cannot_be_instantiated(self):
        """Test base pulse channel cannot be instantiated."""
        with self.assertRaises(NotImplementedError):
            PulseChannel(0)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestAcquireChannel(QiskitTestCase):
    """AcquireChannel tests."""

    def test_default(self):
        """Test default acquire channel."""
        acquire_channel = AcquireChannel(123)

        self.assertEqual(acquire_channel.index, 123)
        self.assertEqual(acquire_channel.name, "a123")

    def test_channel_hash(self):
        """Test hashing for acquire channel."""
        acq_channel_1 = AcquireChannel(123)
        acq_channel_2 = AcquireChannel(123)

        hash_1 = hash(acq_channel_1)
        hash_2 = hash(acq_channel_2)

        self.assertEqual(hash_1, hash_2)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestClassicalIOChannel(QiskitTestCase):
    """Test base classical IO channel."""

    def test_cannot_be_instantiated(self):
        """Test base classical IO channel cannot be instantiated."""
        with self.assertRaises(NotImplementedError):
            ClassicalIOChannel(0)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestMemorySlot(QiskitTestCase):
    """MemorySlot tests."""

    def test_default(self):
        """Test default memory slot."""
        memory_slot = MemorySlot(123)

        self.assertEqual(memory_slot.index, 123)
        self.assertEqual(memory_slot.name, "m123")
        self.assertTrue(isinstance(memory_slot, ClassicalIOChannel))

    def test_validation(self):
        """Test channel validation"""
        with self.assertRaises(PulseError):
            MemorySlot(0.5)
        with self.assertRaises(PulseError):
            MemorySlot(-1)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestRegisterSlot(QiskitTestCase):
    """RegisterSlot tests."""

    def test_default(self):
        """Test default register slot."""
        register_slot = RegisterSlot(123)

        self.assertEqual(register_slot.index, 123)
        self.assertEqual(register_slot.name, "c123")
        self.assertTrue(isinstance(register_slot, ClassicalIOChannel))

    def test_validation(self):
        """Test channel validation"""
        with self.assertRaises(PulseError):
            RegisterSlot(0.5)
        with self.assertRaises(PulseError):
            RegisterSlot(-1)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestSnapshotChannel(QiskitTestCase):
    """SnapshotChannel tests."""

    def test_default(self):
        """Test default snapshot channel."""
        snapshot_channel = SnapshotChannel()

        self.assertEqual(snapshot_channel.index, 0)
        self.assertEqual(snapshot_channel.name, "s0")
        self.assertTrue(isinstance(snapshot_channel, ClassicalIOChannel))


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestDriveChannel(QiskitTestCase):
    """DriveChannel tests."""

    def test_default(self):
        """Test default drive channel."""
        drive_channel = DriveChannel(123)

        self.assertEqual(drive_channel.index, 123)
        self.assertEqual(drive_channel.name, "d123")

    def test_validation(self):
        """Test channel validation"""
        with self.assertRaises(PulseError):
            DriveChannel(0.5)
        with self.assertRaises(PulseError):
            DriveChannel(-1)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestControlChannel(QiskitTestCase):
    """ControlChannel tests."""

    def test_default(self):
        """Test default control channel."""
        control_channel = ControlChannel(123)

        self.assertEqual(control_channel.index, 123)
        self.assertEqual(control_channel.name, "u123")

    def test_validation(self):
        """Test channel validation"""
        with self.assertRaises(PulseError):
            ControlChannel(0.5)
        with self.assertRaises(PulseError):
            ControlChannel(-1)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestMeasureChannel(QiskitTestCase):
    """MeasureChannel tests."""

    def test_default(self):
        """Test default measure channel."""
        measure_channel = MeasureChannel(123)

        self.assertEqual(measure_channel.index, 123)
        self.assertEqual(measure_channel.name, "m123")

    def test_validation(self):
        """Test channel validation"""
        with self.assertRaises(PulseError):
            MeasureChannel(0.5)
        with self.assertRaises(PulseError):
            MeasureChannel(-1)


if __name__ == "__main__":
    unittest.main()

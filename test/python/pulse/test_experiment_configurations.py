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

"""Test cases for the experimental conditions for pulse."""
import unittest
import numpy as np

from qiskit.pulse.channels import DriveChannel, MeasureChannel, AcquireChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse import LoConfig, LoRange, Kernel, Discriminator
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


class TestLoRange(QiskitTestCase):
    """Test LO LoRange."""

    def test_properties_includes_and_eq(self):
        """Test creation of LoRange. Test upper/lower bounds and includes.
        Test __eq__ for two same and different LoRange's.
        """
        lo_range_1 = LoRange(lower_bound=-0.1, upper_bound=+0.1)

        self.assertEqual(lo_range_1.lower_bound, -0.1)
        self.assertEqual(lo_range_1.upper_bound, +0.1)
        self.assertTrue(lo_range_1.includes(0.0))

        lo_range_2 = LoRange(lower_bound=-0.1, upper_bound=+0.1)
        lo_range_3 = LoRange(lower_bound=-0.2, upper_bound=+0.2)

        self.assertTrue(lo_range_1 == lo_range_2)
        self.assertFalse(lo_range_1 == lo_range_3)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestLoConfig(QiskitTestCase):
    """LoConfig tests."""

    def test_can_create_empty_user_lo_config(self):
        """Test if a LoConfig can be created without no arguments."""
        user_lo_config = LoConfig()
        self.assertEqual({}, user_lo_config.qubit_los)
        self.assertEqual({}, user_lo_config.meas_los)

    def test_can_create_valid_user_lo_config(self):
        """Test if a LoConfig can be created with valid user_los."""
        channel1 = DriveChannel(0)
        channel2 = MeasureChannel(0)
        user_lo_config = LoConfig({channel1: 1.4, channel2: 3.6})
        self.assertEqual(1.4, user_lo_config.qubit_los[channel1])
        self.assertEqual(3.6, user_lo_config.meas_los[channel2])

    def test_fail_to_create_with_out_of_range_user_lo(self):
        """Test if a LoConfig cannot be created with invalid user_los."""
        channel = DriveChannel(0)
        with self.assertRaises(PulseError):
            LoConfig({channel: 3.3}, {channel: (1.0, 2.0)})

    def test_fail_to_create_with_invalid_channel(self):
        """Test if a LoConfig cannot be created with invalid channel."""
        channel = AcquireChannel(0)
        with self.assertRaises(PulseError):
            LoConfig({channel: 1.0})

    def test_keep_dict_unchanged_after_updating_the_dict_used_in_construction(self):
        """Test if a LoConfig keeps its dictionary unchanged even after
        the dictionary used in construction is updated.
        """
        channel = DriveChannel(0)
        original = {channel: 3.4}
        user_lo_config = LoConfig(original)
        self.assertEqual(3.4, user_lo_config.qubit_los[channel])
        original[channel] = 5.6
        self.assertEqual(3.4, user_lo_config.qubit_los[channel])

    def test_get_channel_lo(self):
        """Test retrieving channel lo from LO config."""
        channel = DriveChannel(0)
        lo_config = LoConfig({channel: 1.0})
        self.assertEqual(lo_config.channel_lo(channel), 1.0)

        channel = MeasureChannel(0)
        lo_config = LoConfig({channel: 2.0})
        self.assertEqual(lo_config.channel_lo(channel), 2.0)

        with self.assertRaises(PulseError):
            lo_config.channel_lo(MeasureChannel(1))


class TestKernel(QiskitTestCase):
    """Test Kernel."""

    def test_eq(self):
        """Test if two kernels are equal."""
        kernel_a = Kernel(
            "kernel_test",
            kernel={"real": np.zeros(10), "imag": np.zeros(10)},
            bias=[0, 0],
        )
        kernel_b = Kernel(
            "kernel_test",
            kernel={"real": np.zeros(10), "imag": np.zeros(10)},
            bias=[0, 0],
        )
        self.assertTrue(kernel_a == kernel_b)

    def test_neq_name(self):
        """Test if two kernels with different names are not equal."""
        kernel_a = Kernel(
            "kernel_test",
            kernel={"real": np.zeros(10), "imag": np.zeros(10)},
            bias=[0, 0],
        )
        kernel_b = Kernel(
            "kernel_test_2",
            kernel={"real": np.zeros(10), "imag": np.zeros(10)},
            bias=[0, 0],
        )
        self.assertFalse(kernel_a == kernel_b)

    def test_neq_params(self):
        """Test if two kernels with different parameters are not equal."""
        kernel_a = Kernel(
            "kernel_test",
            kernel={"real": np.zeros(10), "imag": np.zeros(10)},
            bias=[0, 0],
        )
        kernel_b = Kernel(
            "kernel_test",
            kernel={"real": np.zeros(10), "imag": np.zeros(10)},
            bias=[1, 0],
        )
        self.assertFalse(kernel_a == kernel_b)

    def test_neq_nested_params(self):
        """Test if two kernels with different nested parameters are not equal."""
        kernel_a = Kernel(
            "kernel_test",
            kernel={"real": np.zeros(10), "imag": np.zeros(10)},
            bias=[0, 0],
        )
        kernel_b = Kernel(
            "kernel_test",
            kernel={"real": np.ones(10), "imag": np.zeros(10)},
            bias=[0, 0],
        )
        self.assertFalse(kernel_a == kernel_b)


class TestDiscriminator(QiskitTestCase):
    """Test Discriminator."""

    def test_eq(self):
        """Test if two discriminators are equal."""
        discriminator_a = Discriminator(
            "discriminator_test",
            discriminator_type="linear",
            params=[1, 0],
        )
        discriminator_b = Discriminator(
            "discriminator_test",
            discriminator_type="linear",
            params=[1, 0],
        )
        self.assertTrue(discriminator_a == discriminator_b)

    def test_neq_name(self):
        """Test if two discriminators with different names are not equal."""
        discriminator_a = Discriminator(
            "discriminator_test",
            discriminator_type="linear",
            params=[1, 0],
        )
        discriminator_b = Discriminator(
            "discriminator_test_2",
            discriminator_type="linear",
            params=[1, 0],
        )
        self.assertFalse(discriminator_a == discriminator_b)

    def test_neq_params(self):
        """Test if two discriminators with different parameters are not equal."""
        discriminator_a = Discriminator(
            "discriminator_test",
            discriminator_type="linear",
            params=[1, 0],
        )
        discriminator_b = Discriminator(
            "discriminator_test",
            discriminator_type="non-linear",
            params=[0, 0],
        )
        self.assertFalse(discriminator_a == discriminator_b)


if __name__ == "__main__":
    unittest.main()

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


"""Tests pulse function samplers."""

import numpy as np

from qiskit.pulse import library
from qiskit.pulse.library import samplers
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


def linear(times: np.ndarray, m: float, b: float = 0.1) -> np.ndarray:
    """Linear test function
    Args:
        times: Input times.
        m: Slope.
        b: Intercept
    Returns:
        np.ndarray
    """
    return m * times + b


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestSampler(QiskitTestCase):
    """Test continuous pulse function samplers."""

    def test_left_sampler(self):
        """Test left sampler."""
        m = 0.1
        b = 0.1
        duration = 2
        left_linear_pulse_fun = samplers.left(linear)
        reference = np.array([0.1, 0.2], dtype=complex)

        pulse = left_linear_pulse_fun(duration, m=m, b=b)
        self.assertIsInstance(pulse, library.Waveform)
        np.testing.assert_array_almost_equal(pulse.samples, reference)

    def test_right_sampler(self):
        """Test right sampler."""
        m = 0.1
        b = 0.1
        duration = 2
        right_linear_pulse_fun = samplers.right(linear)
        reference = np.array([0.2, 0.3], dtype=complex)

        pulse = right_linear_pulse_fun(duration, m=m, b=b)
        self.assertIsInstance(pulse, library.Waveform)
        np.testing.assert_array_almost_equal(pulse.samples, reference)

    def test_midpoint_sampler(self):
        """Test midpoint sampler."""
        m = 0.1
        b = 0.1
        duration = 2
        midpoint_linear_pulse_fun = samplers.midpoint(linear)
        reference = np.array([0.15, 0.25], dtype=complex)

        pulse = midpoint_linear_pulse_fun(duration, m=m, b=b)
        self.assertIsInstance(pulse, library.Waveform)
        np.testing.assert_array_almost_equal(pulse.samples, reference)

    def test_sampler_name(self):
        """Test that sampler setting of pulse name works."""
        m = 0.1
        b = 0.1
        duration = 2
        left_linear_pulse_fun = samplers.left(linear)

        pulse = left_linear_pulse_fun(duration, m=m, b=b, name="test")
        self.assertIsInstance(pulse, library.Waveform)
        self.assertEqual(pulse.name, "test")

    def test_default_arg_sampler(self):
        """Test that default arguments work with sampler."""
        m = 0.1
        duration = 2
        left_linear_pulse_fun = samplers.left(linear)
        reference = np.array([0.1, 0.2], dtype=complex)

        pulse = left_linear_pulse_fun(duration, m=m)
        self.assertIsInstance(pulse, library.Waveform)
        np.testing.assert_array_almost_equal(pulse.samples, reference)

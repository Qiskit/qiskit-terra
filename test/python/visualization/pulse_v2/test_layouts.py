# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for core modules of pulse drawer."""

from qiskit import pulse
from qiskit.visualization.pulse_v2 import layouts, device_info
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestChannelArrangement(QiskitTestCase):
    """Tests for channel mapping functions."""

    @ignore_pulse_deprecation_warnings
    def setUp(self) -> None:
        super().setUp()
        self.channels = [
            pulse.DriveChannel(0),
            pulse.DriveChannel(1),
            pulse.DriveChannel(2),
            pulse.MeasureChannel(1),
            pulse.MeasureChannel(2),
            pulse.AcquireChannel(1),
            pulse.AcquireChannel(2),
            pulse.ControlChannel(0),
            pulse.ControlChannel(2),
            pulse.ControlChannel(5),
        ]
        self.formatter = {"control.show_acquire_channel": True}
        self.device = device_info.OpenPulseBackendInfo(
            name="test",
            dt=1,
            channel_frequency_map={
                pulse.DriveChannel(0): 5.0e9,
                pulse.DriveChannel(1): 5.1e9,
                pulse.DriveChannel(2): 5.2e9,
                pulse.MeasureChannel(1): 7.0e9,
                pulse.MeasureChannel(1): 7.1e9,
                pulse.MeasureChannel(2): 7.2e9,
                pulse.ControlChannel(0): 5.0e9,
                pulse.ControlChannel(1): 5.1e9,
                pulse.ControlChannel(2): 5.2e9,
                pulse.ControlChannel(3): 5.3e9,
                pulse.ControlChannel(4): 5.4e9,
                pulse.ControlChannel(5): 5.5e9,
            },
            qubit_channel_map={
                0: [
                    pulse.DriveChannel(0),
                    pulse.MeasureChannel(0),
                    pulse.AcquireChannel(0),
                    pulse.ControlChannel(0),
                ],
                1: [
                    pulse.DriveChannel(1),
                    pulse.MeasureChannel(1),
                    pulse.AcquireChannel(1),
                    pulse.ControlChannel(1),
                ],
                2: [
                    pulse.DriveChannel(2),
                    pulse.MeasureChannel(2),
                    pulse.AcquireChannel(2),
                    pulse.ControlChannel(2),
                    pulse.ControlChannel(3),
                    pulse.ControlChannel(4),
                ],
                3: [
                    pulse.DriveChannel(3),
                    pulse.MeasureChannel(3),
                    pulse.AcquireChannel(3),
                    pulse.ControlChannel(5),
                ],
            },
        )

    def test_channel_type_grouped_sort(self):
        """Test channel_type_grouped_sort."""
        out_layout = layouts.channel_type_grouped_sort(
            self.channels, formatter=self.formatter, device=self.device
        )

        ref_channels = [
            [pulse.DriveChannel(0)],
            [pulse.DriveChannel(1)],
            [pulse.DriveChannel(2)],
            [pulse.ControlChannel(0)],
            [pulse.ControlChannel(2)],
            [pulse.ControlChannel(5)],
            [pulse.MeasureChannel(1)],
            [pulse.MeasureChannel(2)],
            [pulse.AcquireChannel(1)],
            [pulse.AcquireChannel(2)],
        ]
        ref_names = ["D0", "D1", "D2", "U0", "U2", "U5", "M1", "M2", "A1", "A2"]

        ref = list(zip(ref_names, ref_channels))

        self.assertListEqual(list(out_layout), ref)

    def test_channel_index_sort(self):
        """Test channel_index_grouped_sort."""
        # Add an unusual channel number to stress test the channel ordering
        self.channels.append(pulse.DriveChannel(100))
        self.channels.reverse()
        out_layout = layouts.channel_index_grouped_sort(
            self.channels, formatter=self.formatter, device=self.device
        )

        ref_channels = [
            [pulse.DriveChannel(0)],
            [pulse.ControlChannel(0)],
            [pulse.DriveChannel(1)],
            [pulse.MeasureChannel(1)],
            [pulse.AcquireChannel(1)],
            [pulse.DriveChannel(2)],
            [pulse.ControlChannel(2)],
            [pulse.MeasureChannel(2)],
            [pulse.AcquireChannel(2)],
            [pulse.ControlChannel(5)],
            [pulse.DriveChannel(100)],
        ]

        ref_names = ["D0", "U0", "D1", "M1", "A1", "D2", "U2", "M2", "A2", "U5", "D100"]

        ref = list(zip(ref_names, ref_channels))

        self.assertListEqual(list(out_layout), ref)

    def test_channel_index_sort_grouped_control(self):
        """Test channel_index_grouped_sort_u."""
        out_layout = layouts.channel_index_grouped_sort_u(
            self.channels, formatter=self.formatter, device=self.device
        )

        ref_channels = [
            [pulse.DriveChannel(0)],
            [pulse.DriveChannel(1)],
            [pulse.MeasureChannel(1)],
            [pulse.AcquireChannel(1)],
            [pulse.DriveChannel(2)],
            [pulse.MeasureChannel(2)],
            [pulse.AcquireChannel(2)],
            [pulse.ControlChannel(0)],
            [pulse.ControlChannel(2)],
            [pulse.ControlChannel(5)],
        ]

        ref_names = ["D0", "D1", "M1", "A1", "D2", "M2", "A2", "U0", "U2", "U5"]

        ref = list(zip(ref_names, ref_channels))

        self.assertListEqual(list(out_layout), ref)

    def test_channel_qubit_index_sort(self):
        """Test qubit_index_sort."""
        out_layout = layouts.qubit_index_sort(
            self.channels, formatter=self.formatter, device=self.device
        )

        ref_channels = [
            [pulse.DriveChannel(0), pulse.ControlChannel(0)],
            [pulse.DriveChannel(1), pulse.MeasureChannel(1)],
            [pulse.DriveChannel(2), pulse.MeasureChannel(2), pulse.ControlChannel(2)],
            [pulse.ControlChannel(5)],
        ]

        ref_names = ["Q0", "Q1", "Q2", "Q3"]

        ref = list(zip(ref_names, ref_channels))

        self.assertListEqual(list(out_layout), ref)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestHorizontalAxis(QiskitTestCase):
    """Tests for horizontal axis mapping functions."""

    def test_time_map_in_ns(self):
        """Test for time_map_in_ns."""
        time_window = (0, 1000)
        breaks = [(100, 200)]
        dt = 1e-9

        haxis = layouts.time_map_in_ns(time_window=time_window, axis_breaks=breaks, dt=dt)

        self.assertListEqual(list(haxis.window), [0, 900])
        self.assertListEqual(list(haxis.axis_break_pos), [100])
        ref_axis_map = {
            0.0: "0",
            180.0: "280",
            360.0: "460",
            540.0: "640",
            720.0: "820",
            900.0: "1000",
        }
        self.assertDictEqual(haxis.axis_map, ref_axis_map)
        self.assertEqual(haxis.label, "Time (ns)")

    def test_time_map_in_without_dt(self):
        """Test for time_map_in_ns when dt is not provided."""
        time_window = (0, 1000)
        breaks = [(100, 200)]
        dt = None

        haxis = layouts.time_map_in_ns(time_window=time_window, axis_breaks=breaks, dt=dt)

        self.assertListEqual(list(haxis.window), [0, 900])
        self.assertListEqual(list(haxis.axis_break_pos), [100])
        ref_axis_map = {
            0.0: "0",
            180.0: "280",
            360.0: "460",
            540.0: "640",
            720.0: "820",
            900.0: "1000",
        }
        self.assertDictEqual(haxis.axis_map, ref_axis_map)
        self.assertEqual(haxis.label, "System cycle time (dt)")


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestFigureTitle(QiskitTestCase):
    """Tests for figure title generation."""

    @ignore_pulse_deprecation_warnings
    def setUp(self) -> None:
        super().setUp()
        self.device = device_info.OpenPulseBackendInfo(name="test_backend", dt=1e-9)
        self.prog = pulse.Schedule(name="test_sched")
        self.prog.insert(
            0, pulse.Play(pulse.Constant(100, 0.1), pulse.DriveChannel(0)), inplace=True
        )

    def detail_title(self):
        """Test detail_title layout function."""
        ref_title = "Name: test_sched, Duration: 100.0 ns, Backend: test_backend"
        out = layouts.detail_title(self.prog, self.device)

        self.assertEqual(out, ref_title)

    def empty_title(self):
        """Test empty_title layout function."""
        ref_title = ""
        out = layouts.detail_title(self.prog, self.device)

        self.assertEqual(out, ref_title)

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

# pylint: disable=missing-function-docstring, unused-argument

"""Tests for core modules of pulse drawer."""

import numpy as np
from qiskit import pulse
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import core, stylesheet, device_info, drawings, types, layouts
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestChart(QiskitTestCase):
    """Tests for chart."""

    @ignore_pulse_deprecation_warnings
    def setUp(self) -> None:
        super().setUp()

        self.style = stylesheet.QiskitPulseStyle()
        self.device = device_info.OpenPulseBackendInfo(
            name="test",
            dt=1,
            channel_frequency_map={
                pulse.DriveChannel(0): 5.0,
                pulse.MeasureChannel(0): 7.0,
                pulse.ControlChannel(0): 5.0,
            },
            qubit_channel_map={
                0: [
                    pulse.DriveChannel(0),
                    pulse.MeasureChannel(0),
                    pulse.AcquireChannel(0),
                    pulse.ControlChannel(0),
                ]
            },
        )

        # objects
        self.short_pulse = drawings.LineData(
            data_type=types.WaveformType.REAL,
            xvals=[0, 0, 1, 4, 5, 5],
            yvals=[0, 0.5, 0.5, 0.5, 0.5, 0],
            channels=[pulse.DriveChannel(0)],
        )
        self.long_pulse = drawings.LineData(
            data_type=types.WaveformType.REAL,
            xvals=[8, 8, 9, 19, 20, 20],
            yvals=[0, 0.3, 0.3, 0.3, 0.3, 0],
            channels=[pulse.DriveChannel(1)],
        )
        self.abstract_hline = drawings.LineData(
            data_type=types.LineType.BASELINE,
            xvals=[types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT],
            yvals=[0, 0],
            channels=[pulse.DriveChannel(0)],
        )

    def test_add_data(self):
        """Test add data to chart."""
        fake_canvas = core.DrawerCanvas(stylesheet=self.style, device=self.device)
        chart = core.Chart(parent=fake_canvas)

        chart.add_data(self.short_pulse)
        self.assertEqual(len(chart._collections), 1)

        # the same pulse will be overwritten
        chart.add_data(self.short_pulse)
        self.assertEqual(len(chart._collections), 1)

        chart.add_data(self.long_pulse)
        self.assertEqual(len(chart._collections), 2)

    def test_bind_coordinate(self):
        """Test bind coordinate."""
        fake_canvas = core.DrawerCanvas(stylesheet=self.style, device=self.device)
        fake_canvas.formatter = {"margin.left_percent": 0.1, "margin.right_percent": 0.1}
        fake_canvas.time_range = (500, 2000)

        chart = core.Chart(parent=fake_canvas)
        chart.vmin = -0.1
        chart.vmax = 0.5

        # vertical
        vline = [types.AbstractCoordinate.BOTTOM, types.AbstractCoordinate.TOP]
        vals = chart._bind_coordinate(vline)
        np.testing.assert_array_equal(vals, np.array([-0.1, 0.5]))

        # horizontal, margin is is considered
        hline = [types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT]
        vals = chart._bind_coordinate(hline)
        np.testing.assert_array_equal(vals, np.array([350.0, 2150.0]))

    def test_truncate(self):
        """Test pulse truncation."""
        fake_canvas = core.DrawerCanvas(stylesheet=self.style, device=self.device)
        fake_canvas.formatter = {
            "margin.left_percent": 0,
            "margin.right_percent": 0,
            "axis_break.length": 20,
            "axis_break.max_length": 10,
        }
        fake_canvas.time_range = (0, 20)
        fake_canvas.time_breaks = [(5, 10)]

        chart = core.Chart(parent=fake_canvas)

        xvals = np.array([4, 5, 6, 7, 8, 9, 10, 11])
        yvals = np.array([1, 2, 3, 4, 5, 6, 7, 8])

        new_xvals, new_yvals = chart._truncate_vectors(xvals, yvals)

        ref_xvals = np.array([4.0, 5.0, 5.0, 6.0])
        ref_yvals = np.array([1.0, 2.0, 7.0, 8.0])

        np.testing.assert_array_almost_equal(new_xvals, ref_xvals)
        np.testing.assert_array_almost_equal(new_yvals, ref_yvals)

    def test_truncate_multiple(self):
        """Test pulse truncation."""
        fake_canvas = core.DrawerCanvas(stylesheet=self.style, device=self.device)
        fake_canvas.formatter = {
            "margin.left_percent": 0,
            "margin.right_percent": 0,
            "axis_break.length": 20,
            "axis_break.max_length": 10,
        }
        fake_canvas.time_range = (2, 12)
        fake_canvas.time_breaks = [(4, 7), (9, 11)]

        chart = core.Chart(parent=fake_canvas)

        xvals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        yvals = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        new_xvals, new_yvals = chart._truncate_vectors(xvals, yvals)

        ref_xvals = np.array([2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0])
        ref_yvals = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        np.testing.assert_array_almost_equal(new_xvals, ref_xvals)
        np.testing.assert_array_almost_equal(new_yvals, ref_yvals)

    def test_visible(self):
        """Test pulse truncation."""
        fake_canvas = core.DrawerCanvas(stylesheet=self.style, device=self.device)
        fake_canvas.disable_chans = {pulse.DriveChannel(0)}
        fake_canvas.disable_types = {types.WaveformType.REAL}

        chart = core.Chart(parent=fake_canvas)

        test_data = drawings.ElementaryData(
            data_type=types.WaveformType.REAL,
            xvals=np.array([0]),
            yvals=np.array([0]),
            channels=[pulse.DriveChannel(0)],
        )
        self.assertFalse(chart._check_visible(test_data))

        test_data = drawings.ElementaryData(
            data_type=types.WaveformType.IMAG,
            xvals=np.array([0]),
            yvals=np.array([0]),
            channels=[pulse.DriveChannel(0)],
        )
        self.assertFalse(chart._check_visible(test_data))

        test_data = drawings.ElementaryData(
            data_type=types.WaveformType.IMAG,
            xvals=np.array([0]),
            yvals=np.array([0]),
            channels=[pulse.DriveChannel(1)],
        )
        self.assertTrue(chart._check_visible(test_data))

    def test_update(self):
        fake_canvas = core.DrawerCanvas(stylesheet=self.style, device=self.device)
        fake_canvas.formatter = {
            "margin.left_percent": 0,
            "margin.right_percent": 0,
            "axis_break.length": 20,
            "axis_break.max_length": 10,
            "control.auto_chart_scaling": True,
            "general.vertical_resolution": 1e-6,
            "general.max_scale": 10,
            "channel_scaling.pos_spacing": 0.1,
            "channel_scaling.neg_spacing": -0.1,
        }
        fake_canvas.time_range = (0, 20)
        fake_canvas.time_breaks = [(10, 15)]

        chart = core.Chart(fake_canvas)
        chart.add_data(self.short_pulse)
        chart.add_data(self.long_pulse)
        chart.add_data(self.abstract_hline)
        chart.update()

        short_pulse = chart._output_dataset[self.short_pulse.data_key]
        xref = np.array([0.0, 0.0, 1.0, 4.0, 5.0, 5.0])
        yref = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(xref, short_pulse.xvals)
        np.testing.assert_array_almost_equal(yref, short_pulse.yvals)

        long_pulse = chart._output_dataset[self.long_pulse.data_key]
        xref = np.array([8.0, 8.0, 9.0, 10.0, 10.0, 14.0, 15.0, 15.0])
        yref = np.array([0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0])
        np.testing.assert_array_almost_equal(xref, long_pulse.xvals)
        np.testing.assert_array_almost_equal(yref, long_pulse.yvals)

        abstract_hline = chart._output_dataset[self.abstract_hline.data_key]
        xref = np.array([0.0, 10.0, 10.0, 15.0])
        yref = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(xref, abstract_hline.xvals)
        np.testing.assert_array_almost_equal(yref, abstract_hline.yvals)

        self.assertEqual(chart.vmax, 1.0)
        self.assertEqual(chart.vmin, -0.1)
        self.assertEqual(chart.scale, 2.0)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestDrawCanvas(QiskitTestCase):
    """Tests for draw canvas."""

    @ignore_pulse_deprecation_warnings
    def setUp(self) -> None:
        super().setUp()
        self.style = stylesheet.QiskitPulseStyle()
        self.device = device_info.OpenPulseBackendInfo(
            name="test",
            dt=1,
            channel_frequency_map={
                pulse.DriveChannel(0): 5.0,
                pulse.MeasureChannel(0): 7.0,
                pulse.ControlChannel(0): 5.0,
            },
            qubit_channel_map={
                0: [
                    pulse.DriveChannel(0),
                    pulse.MeasureChannel(0),
                    pulse.AcquireChannel(0),
                    pulse.ControlChannel(0),
                ]
            },
        )

        self.sched = pulse.Schedule()
        self.sched.insert(
            0,
            pulse.Play(pulse.Waveform([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), pulse.DriveChannel(0)),
            inplace=True,
        )
        self.sched.insert(
            10,
            pulse.Play(pulse.Waveform([0.5, 0.4, 0.3, 0.2, 0.1, 0.0]), pulse.DriveChannel(0)),
            inplace=True,
        )
        self.sched.insert(
            0,
            pulse.Play(pulse.Waveform([0.3, 0.3, 0.3, 0.3, 0.3, 0.3]), pulse.DriveChannel(1)),
            inplace=True,
        )

    def test_time_breaks(self):
        """Test calculating time breaks."""
        canvas = core.DrawerCanvas(stylesheet=self.style, device=self.device)
        canvas.formatter = {
            "margin.left_percent": 0,
            "margin.right_percent": 0,
            "axis_break.length": 20,
            "axis_break.max_length": 10,
        }
        canvas.layout = {"figure_title": layouts.empty_title}
        canvas.time_breaks = [(10, 40), (60, 80)]

        canvas.time_range = (0, 100)
        ref_breaks = [(10, 40), (60, 80)]
        self.assertListEqual(canvas.time_breaks, ref_breaks)

        # break too large
        canvas.time_range = (20, 30)
        with self.assertRaises(VisualizationError):
            _ = canvas.time_breaks

        # time range overlap
        canvas.time_range = (15, 100)
        ref_breaks = [(20, 40), (60, 80)]
        self.assertListEqual(canvas.time_breaks, ref_breaks)

        # time range overlap
        canvas.time_range = (30, 100)
        ref_breaks = [(60, 80)]
        self.assertListEqual(canvas.time_breaks, ref_breaks)

        # time range overlap
        canvas.time_range = (0, 70)
        ref_breaks = [(10, 40)]
        self.assertListEqual(canvas.time_breaks, ref_breaks)

        # time range no overlap
        canvas.time_range = (40, 60)
        ref_breaks = []
        self.assertListEqual(canvas.time_breaks, ref_breaks)

    def test_time_range(self):
        """Test calculating time range."""
        canvas = core.DrawerCanvas(stylesheet=self.style, device=self.device)
        canvas.formatter = {
            "margin.left_percent": 0.1,
            "margin.right_percent": 0.1,
            "axis_break.length": 20,
            "axis_break.max_length": 10,
        }
        canvas.layout = {"figure_title": layouts.empty_title}
        canvas.time_range = (0, 100)

        # no breaks
        canvas.time_breaks = []
        ref_range = [-10.0, 110.0]
        self.assertListEqual(list(canvas.time_range), ref_range)

        # with break
        canvas.time_breaks = [(20, 40)]
        ref_range = [-8.0, 108.0]
        self.assertListEqual(list(canvas.time_range), ref_range)

    def chart_channel_map(self, **kwargs):
        """Mock of chart channel mapper."""
        names = ["D0", "D1"]
        chans = [[pulse.DriveChannel(0)], [pulse.DriveChannel(1)]]

        yield from zip(names, chans)

    def generate_dummy_obj(self, data: types.PulseInstruction, **kwargs):
        dummy_obj = drawings.ElementaryData(
            data_type="test",
            xvals=np.arange(data.inst.pulse.duration),
            yvals=data.inst.pulse.samples,
            channels=[data.inst.channel],
        )
        return [dummy_obj]

    def test_load_program(self):
        """Test loading program."""
        canvas = core.DrawerCanvas(stylesheet=self.style, device=self.device)
        canvas.formatter = {
            "axis_break.length": 20,
            "axis_break.max_length": 10,
            "channel_scaling.drive": 5,
        }
        canvas.generator = {
            "waveform": [self.generate_dummy_obj],
            "frame": [],
            "chart": [],
            "snapshot": [],
            "barrier": [],
        }
        canvas.layout = {
            "chart_channel_map": self.chart_channel_map,
            "figure_title": layouts.empty_title,
        }

        canvas.load_program(self.sched)

        self.assertEqual(len(canvas.charts), 2)

        self.assertListEqual(canvas.charts[0].channels, [pulse.DriveChannel(0)])
        self.assertListEqual(canvas.charts[1].channels, [pulse.DriveChannel(1)])

        self.assertEqual(len(canvas.charts[0]._collections), 2)
        self.assertEqual(len(canvas.charts[1]._collections), 1)

        ref_scale = {pulse.DriveChannel(0): 5, pulse.DriveChannel(1): 5}
        self.assertDictEqual(canvas.chan_scales, ref_scale)

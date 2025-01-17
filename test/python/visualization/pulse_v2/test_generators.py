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

# pylint: disable=invalid-name

"""Tests for drawing object generation."""

import numpy as np

from qiskit import pulse, circuit
from qiskit.visualization.pulse_v2 import drawings, types, stylesheet, device_info
from qiskit.visualization.pulse_v2.generators import barrier, chart, frame, snapshot, waveform
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


def create_instruction(inst, phase, freq, t0, dt, is_opaque=False):
    """A helper function to create InstructionTuple."""
    frame_info = types.PhaseFreqTuple(phase=phase, freq=freq)
    return types.PulseInstruction(t0=t0, dt=dt, frame=frame_info, inst=inst, is_opaque=is_opaque)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestWaveformGenerators(QiskitTestCase):
    """Tests for waveform generators."""

    @ignore_pulse_deprecation_warnings
    def setUp(self) -> None:
        super().setUp()
        style = stylesheet.QiskitPulseStyle()
        self.formatter = style.formatter
        self.device = device_info.OpenPulseBackendInfo(
            name="test",
            dt=1,
            channel_frequency_map={
                pulse.DriveChannel(0): 5.0e9,
                pulse.DriveChannel(1): 5.1e9,
                pulse.MeasureChannel(0): 7.0e9,
                pulse.MeasureChannel(1): 7.1e9,
                pulse.ControlChannel(0): 5.0e9,
                pulse.ControlChannel(1): 5.1e9,
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
            },
        )

    def test_consecutive_index_all_equal(self):
        """Test for helper function to find consecutive index with identical numbers."""
        vec = np.array([1, 1, 1, 1, 1, 1])
        ref_inds = np.array([True, False, False, False, False, True], dtype=bool)

        inds = waveform._find_consecutive_index(vec, resolution=1e-6)

        np.testing.assert_array_equal(inds, ref_inds)

    def test_consecutive_index_tiny_diff(self):
        """Test for helper function to find consecutive index with vector with tiny change."""
        eps = 1e-10
        vec = np.array([0.5, 0.5 + eps, 0.5 - eps, 0.5 + eps, 0.5 - eps, 0.5])
        ref_inds = np.array([True, False, False, False, False, True], dtype=bool)

        inds = waveform._find_consecutive_index(vec, resolution=1e-6)

        np.testing.assert_array_equal(inds, ref_inds)

    def test_parse_waveform(self):
        """Test helper function that parse waveform with Waveform instance."""
        test_pulse = pulse.library.Gaussian(10, 0.1, 3).get_waveform()

        inst = pulse.Play(test_pulse, pulse.DriveChannel(0))
        inst_data = create_instruction(inst, 0, 0, 10, 0.1)

        x, y, _ = waveform._parse_waveform(inst_data)

        x_ref = np.arange(10, 20)
        y_ref = test_pulse.samples

        np.testing.assert_array_equal(x, x_ref)
        np.testing.assert_array_equal(y, y_ref)

    def test_parse_waveform_parametric(self):
        """Test helper function that parse waveform with ParametricPulse instance."""
        test_pulse = pulse.library.Gaussian(10, 0.1, 3)

        inst = pulse.Play(test_pulse, pulse.DriveChannel(0))
        inst_data = create_instruction(inst, 0, 0, 10, 0.1)

        x, y, _ = waveform._parse_waveform(inst_data)

        x_ref = np.arange(10, 20)
        y_ref = test_pulse.get_waveform().samples

        np.testing.assert_array_equal(x, x_ref)
        np.testing.assert_array_equal(y, y_ref)

    def test_gen_filled_waveform_stepwise_play(self):
        """Test gen_filled_waveform_stepwise with play instruction."""
        my_pulse = pulse.Waveform(samples=[0, 0.5 + 0.5j, 0.5 + 0.5j, 0], name="my_pulse")
        play = pulse.Play(my_pulse, pulse.DriveChannel(0))
        inst_data = create_instruction(play, np.pi / 2, 5e9, 5, 0.1)

        objs = waveform.gen_filled_waveform_stepwise(
            inst_data, formatter=self.formatter, device=self.device
        )

        self.assertEqual(len(objs), 2)

        # type check
        self.assertEqual(type(objs[0]), drawings.LineData)
        self.assertEqual(type(objs[1]), drawings.LineData)

        y_ref = np.array([0, 0, -0.5, -0.5, 0, 0])

        # data check
        self.assertListEqual(objs[0].channels, [pulse.DriveChannel(0)])
        self.assertListEqual(list(objs[0].xvals), [5, 6, 6, 8, 8, 9])
        np.testing.assert_array_almost_equal(objs[0].yvals, y_ref)

        # meta data check
        ref_meta = {
            "duration (cycle time)": 4,
            "duration (sec)": 0.4,
            "t0 (cycle time)": 5,
            "t0 (sec)": 0.5,
            "phase": np.pi / 2,
            "frequency": 5e9,
            "qubit": 0,
            "name": "my_pulse",
            "data": "real",
        }
        self.assertDictEqual(objs[0].meta, ref_meta)

        # style check
        ref_style = {
            "alpha": self.formatter["alpha.fill_waveform"],
            "zorder": self.formatter["layer.fill_waveform"],
            "linewidth": self.formatter["line_width.fill_waveform"],
            "linestyle": self.formatter["line_style.fill_waveform"],
            "color": self.formatter["color.waveforms"]["D"][0],
        }
        self.assertDictEqual(objs[0].styles, ref_style)

    def test_gen_filled_waveform_stepwise_acquire(self):
        """Test gen_filled_waveform_stepwise with acquire instruction."""
        acquire = pulse.Acquire(
            duration=4,
            channel=pulse.AcquireChannel(0),
            mem_slot=pulse.MemorySlot(0),
            discriminator=pulse.Discriminator(name="test_discr"),
            name="acquire",
        )
        inst_data = create_instruction(acquire, 0, 7e9, 5, 0.1)

        objs = waveform.gen_filled_waveform_stepwise(
            inst_data, formatter=self.formatter, device=self.device
        )

        # imaginary part is empty and not returned
        self.assertEqual(len(objs), 1)

        # type check
        self.assertEqual(type(objs[0]), drawings.LineData)

        y_ref = np.array([1, 1])

        # data check - data is compressed
        self.assertListEqual(objs[0].channels, [pulse.AcquireChannel(0)])
        self.assertListEqual(list(objs[0].xvals), [5, 9])
        np.testing.assert_array_almost_equal(objs[0].yvals, y_ref)

        # meta data check
        ref_meta = {
            "memory slot": "m0",
            "register slot": "N/A",
            "discriminator": "test_discr",
            "kernel": "N/A",
            "duration (cycle time)": 4,
            "duration (sec)": 0.4,
            "t0 (cycle time)": 5,
            "t0 (sec)": 0.5,
            "phase": 0,
            "frequency": 7e9,
            "qubit": 0,
            "name": "acquire",
            "data": "real",
        }

        self.assertDictEqual(objs[0].meta, ref_meta)

        # style check
        ref_style = {
            "alpha": self.formatter["alpha.fill_waveform"],
            "zorder": self.formatter["layer.fill_waveform"],
            "linewidth": self.formatter["line_width.fill_waveform"],
            "linestyle": self.formatter["line_style.fill_waveform"],
            "color": self.formatter["color.waveforms"]["A"][0],
        }
        self.assertDictEqual(objs[0].styles, ref_style)

    def test_gen_iqx_latex_waveform_name_x90(self):
        """Test gen_iqx_latex_waveform_name with x90 waveform."""
        iqx_pulse = pulse.Waveform(samples=[0, 0, 0, 0], name="X90p_d0_1234567")
        play = pulse.Play(iqx_pulse, pulse.DriveChannel(0))
        inst_data = create_instruction(play, 0, 0, 0, 0.1)

        obj = waveform.gen_ibmq_latex_waveform_name(
            inst_data, formatter=self.formatter, device=self.device
        )[0]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.DriveChannel(0)])
        self.assertEqual(obj.text, "X90p_d0_1234567")
        self.assertEqual(obj.latex, r"{\rm X}(\pi/2)")

        # style check
        ref_style = {
            "zorder": self.formatter["layer.annotate"],
            "color": self.formatter["color.annotate"],
            "size": self.formatter["text_size.annotate"],
            "va": "center",
            "ha": "center",
        }
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_iqx_latex_waveform_name_x180(self):
        """Test gen_iqx_latex_waveform_name with x180 waveform."""
        iqx_pulse = pulse.Waveform(samples=[0, 0, 0, 0], name="Xp_d0_1234567")
        play = pulse.Play(iqx_pulse, pulse.DriveChannel(0))
        inst_data = create_instruction(play, 0, 0, 0, 0.1)

        obj = waveform.gen_ibmq_latex_waveform_name(
            inst_data, formatter=self.formatter, device=self.device
        )[0]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.DriveChannel(0)])
        self.assertEqual(obj.text, "Xp_d0_1234567")
        self.assertEqual(obj.latex, r"{\rm X}(\pi)")

    def test_gen_iqx_latex_waveform_name_cr(self):
        """Test gen_iqx_latex_waveform_name with CR waveform."""
        iqx_pulse = pulse.Waveform(samples=[0, 0, 0, 0], name="CR90p_u0_1234567")
        play = pulse.Play(iqx_pulse, pulse.ControlChannel(0))
        inst_data = create_instruction(play, 0, 0, 0, 0.1)

        obj = waveform.gen_ibmq_latex_waveform_name(
            inst_data, formatter=self.formatter, device=self.device
        )[0]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.ControlChannel(0)])
        self.assertEqual(obj.text, "CR90p_u0_1234567")
        self.assertEqual(obj.latex, r"{\rm CR}(\pi/4)")

    def test_gen_iqx_latex_waveform_name_compensation_tone(self):
        """Test gen_iqx_latex_waveform_name with CR compensation waveform."""
        iqx_pulse = pulse.Waveform(samples=[0, 0, 0, 0], name="CR90p_d0_u0_1234567")
        play = pulse.Play(iqx_pulse, pulse.DriveChannel(0))
        inst_data = create_instruction(play, 0, 0, 0, 0.1)

        obj = waveform.gen_ibmq_latex_waveform_name(
            inst_data, formatter=self.formatter, device=self.device
        )[0]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.DriveChannel(0)])
        self.assertEqual(obj.text, "CR90p_d0_u0_1234567")
        self.assertEqual(obj.latex, r"\overline{\rm CR}(\pi/4)")

    def test_gen_waveform_max_value(self):
        """Test gen_waveform_max_value."""
        iqx_pulse = pulse.Waveform(samples=[0, 0.1, 0.3, -0.2j], name="test")
        play = pulse.Play(iqx_pulse, pulse.DriveChannel(0))
        inst_data = create_instruction(play, 0, 0, 0, 0.1)

        objs = waveform.gen_waveform_max_value(
            inst_data, formatter=self.formatter, device=self.device
        )

        # type check
        self.assertEqual(type(objs[0]), drawings.TextData)
        self.assertEqual(type(objs[1]), drawings.TextData)

        # data check, real part, positive max
        self.assertListEqual(objs[0].channels, [pulse.DriveChannel(0)])
        self.assertEqual(objs[0].text, "0.30\n\u25BE")

        # style check
        ref_style = {
            "zorder": self.formatter["layer.annotate"],
            "color": self.formatter["color.annotate"],
            "size": self.formatter["text_size.annotate"],
            "va": "bottom",
            "ha": "center",
        }
        self.assertDictEqual(objs[0].styles, ref_style)

        # data check, imaginary part, negative max
        self.assertListEqual(objs[1].channels, [pulse.DriveChannel(0)])
        self.assertEqual(objs[1].text, "\u25B4\n-0.20")

        # style check
        ref_style = {
            "zorder": self.formatter["layer.annotate"],
            "color": self.formatter["color.annotate"],
            "size": self.formatter["text_size.annotate"],
            "va": "top",
            "ha": "center",
        }
        self.assertDictEqual(objs[1].styles, ref_style)

    def test_gen_filled_waveform_stepwise_opaque(self):
        """Test generating waveform with unbound parameter."""
        amp = circuit.Parameter("amp")
        my_pulse = pulse.Gaussian(10, amp, 3, name="my_pulse")
        play = pulse.Play(my_pulse, pulse.DriveChannel(0))
        inst_data = create_instruction(play, np.pi / 2, 5e9, 5, 0.1, True)

        objs = waveform.gen_filled_waveform_stepwise(
            inst_data, formatter=self.formatter, device=self.device
        )

        self.assertEqual(len(objs), 2)

        # type check
        self.assertEqual(type(objs[0]), drawings.BoxData)
        self.assertEqual(type(objs[1]), drawings.TextData)

        x_ref = np.array([5, 15])
        y_ref = np.array(
            [
                -0.5 * self.formatter["box_height.opaque_shape"],
                0.5 * self.formatter["box_height.opaque_shape"],
            ]
        )

        # data check
        np.testing.assert_array_equal(objs[0].xvals, x_ref)
        np.testing.assert_array_equal(objs[0].yvals, y_ref)

        # meta data check
        ref_meta = {
            "duration (cycle time)": 10,
            "duration (sec)": 1.0,
            "t0 (cycle time)": 5,
            "t0 (sec)": 0.5,
            "waveform shape": "Gaussian",
            "amp": "amp",
            "angle": 0,
            "sigma": 3,
            "phase": np.pi / 2,
            "frequency": 5e9,
            "qubit": 0,
            "name": "my_pulse",
        }
        self.assertDictEqual(objs[0].meta, ref_meta)

        # style check
        ref_style = {
            "alpha": self.formatter["alpha.opaque_shape"],
            "zorder": self.formatter["layer.fill_waveform"],
            "linewidth": self.formatter["line_width.opaque_shape"],
            "linestyle": self.formatter["line_style.opaque_shape"],
            "facecolor": self.formatter["color.opaque_shape"][0],
            "edgecolor": self.formatter["color.opaque_shape"][1],
        }
        self.assertDictEqual(objs[0].styles, ref_style)

        # test label
        self.assertEqual(objs[1].text, "Gaussian(amp)")


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestChartGenerators(QiskitTestCase):
    """Tests for chart info generators."""

    @ignore_pulse_deprecation_warnings
    def setUp(self) -> None:
        super().setUp()
        style = stylesheet.QiskitPulseStyle()
        self.formatter = style.formatter
        self.device = device_info.OpenPulseBackendInfo(
            name="test",
            dt=1,
            channel_frequency_map={
                pulse.DriveChannel(0): 5.0e9,
                pulse.DriveChannel(1): 5.1e9,
                pulse.MeasureChannel(0): 7.0e9,
                pulse.MeasureChannel(1): 7.1e9,
                pulse.ControlChannel(0): 5.0e9,
                pulse.ControlChannel(1): 5.1e9,
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
            },
        )

    def test_gen_baseline(self):
        """Test gen_baseline."""
        channel_info = types.ChartAxis(name="D0", channels=[pulse.DriveChannel(0)])

        obj = chart.gen_baseline(channel_info, formatter=self.formatter, device=self.device)[0]

        # type check
        self.assertEqual(type(obj), drawings.LineData)

        # data check
        self.assertListEqual(obj.channels, [pulse.DriveChannel(0)])

        ref_x = [types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT]
        ref_y = [0, 0]

        self.assertListEqual(list(obj.xvals), ref_x)
        self.assertListEqual(list(obj.yvals), ref_y)

        # style check
        ref_style = {
            "alpha": self.formatter["alpha.baseline"],
            "zorder": self.formatter["layer.baseline"],
            "linewidth": self.formatter["line_width.baseline"],
            "linestyle": self.formatter["line_style.baseline"],
            "color": self.formatter["color.baseline"],
        }
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_chart_name(self):
        """Test gen_chart_name."""
        channel_info = types.ChartAxis(name="D0", channels=[pulse.DriveChannel(0)])

        obj = chart.gen_chart_name(channel_info, formatter=self.formatter, device=self.device)[0]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.DriveChannel(0)])
        self.assertEqual(obj.text, "D0")

        # style check
        ref_style = {
            "zorder": self.formatter["layer.axis_label"],
            "color": self.formatter["color.axis_label"],
            "size": self.formatter["text_size.axis_label"],
            "va": "center",
            "ha": "right",
        }
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_scaling_info(self):
        """Test gen_scaling_info."""
        channel_info = types.ChartAxis(name="D0", channels=[pulse.DriveChannel(0)])

        obj = chart.gen_chart_scale(channel_info, formatter=self.formatter, device=self.device)[0]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.DriveChannel(0)])
        self.assertEqual(obj.text, f"x{types.DynamicString.SCALE}")

        # style check
        ref_style = {
            "zorder": self.formatter["layer.axis_label"],
            "color": self.formatter["color.axis_label"],
            "size": self.formatter["text_size.annotate"],
            "va": "center",
            "ha": "right",
        }
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_frequency_info(self):
        """Test gen_scaling_info."""
        channel_info = types.ChartAxis(name="D0", channels=[pulse.DriveChannel(0)])

        obj = chart.gen_channel_freqs(channel_info, formatter=self.formatter, device=self.device)[0]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.DriveChannel(0)])
        self.assertEqual(obj.text, "5.00 GHz")

        # style check
        ref_style = {
            "zorder": self.formatter["layer.axis_label"],
            "color": self.formatter["color.axis_label"],
            "size": self.formatter["text_size.annotate"],
            "va": "center",
            "ha": "right",
        }
        self.assertDictEqual(obj.styles, ref_style)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestFrameGenerators(QiskitTestCase):
    """Tests for frame info generators."""

    @ignore_pulse_deprecation_warnings
    def setUp(self) -> None:
        super().setUp()
        style = stylesheet.QiskitPulseStyle()
        self.formatter = style.formatter
        self.device = device_info.OpenPulseBackendInfo(
            name="test",
            dt=1,
            channel_frequency_map={
                pulse.DriveChannel(0): 5.0e9,
                pulse.DriveChannel(1): 5.1e9,
                pulse.MeasureChannel(0): 7.0e9,
                pulse.MeasureChannel(1): 7.1e9,
                pulse.ControlChannel(0): 5.0e9,
                pulse.ControlChannel(1): 5.1e9,
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
            },
        )

    def test_phase_to_text(self):
        """Test helper function to convert phase to text."""
        plain, latex = frame._phase_to_text(self.formatter, np.pi, max_denom=10, flip=True)
        self.assertEqual(plain, "-pi")
        self.assertEqual(latex, r"-\pi")

        plain, latex = frame._phase_to_text(self.formatter, np.pi / 2, max_denom=10, flip=True)
        self.assertEqual(plain, "-pi/2")
        self.assertEqual(latex, r"-\pi/2")

        plain, latex = frame._phase_to_text(self.formatter, np.pi * 3 / 4, max_denom=10, flip=True)
        self.assertEqual(plain, "-3/4 pi")
        self.assertEqual(latex, r"-3/4 \pi")

    def test_frequency_to_text(self):
        """Test helper function to convert frequency to text."""
        plain, latex = frame._freq_to_text(self.formatter, 1e6, unit="MHz")
        self.assertEqual(plain, "1.00 MHz")
        self.assertEqual(latex, r"1.00~{\rm MHz}")

    def test_gen_formatted_phase(self):
        """Test gen_formatted_phase."""
        fcs = [
            pulse.ShiftPhase(np.pi / 2, pulse.DriveChannel(0)),
            pulse.ShiftFrequency(1e6, pulse.DriveChannel(0)),
        ]
        inst_data = create_instruction(fcs, np.pi / 2, 1e6, 5, 0.1)

        obj = frame.gen_formatted_phase(inst_data, formatter=self.formatter, device=self.device)[0]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.DriveChannel(0)])
        self.assertEqual(obj.latex, r"{\rm VZ}(-\pi/2)")
        self.assertEqual(obj.text, "VZ(-pi/2)")

        # style check
        ref_style = {
            "zorder": self.formatter["layer.frame_change"],
            "color": self.formatter["color.frame_change"],
            "size": self.formatter["text_size.annotate"],
            "va": "center",
            "ha": "center",
        }
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_formatted_freq_mhz(self):
        """Test gen_formatted_freq_mhz."""
        fcs = [
            pulse.ShiftPhase(np.pi / 2, pulse.DriveChannel(0)),
            pulse.ShiftFrequency(1e6, pulse.DriveChannel(0)),
        ]
        inst_data = create_instruction(fcs, np.pi / 2, 1e6, 5, 0.1)

        obj = frame.gen_formatted_freq_mhz(inst_data, formatter=self.formatter, device=self.device)[
            0
        ]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.DriveChannel(0)])
        self.assertEqual(obj.latex, r"\Delta f = 1.00~{\rm MHz}")
        self.assertEqual(obj.text, "\u0394f = 1.00 MHz")

        # style check
        ref_style = {
            "zorder": self.formatter["layer.frame_change"],
            "color": self.formatter["color.frame_change"],
            "size": self.formatter["text_size.annotate"],
            "va": "center",
            "ha": "center",
        }
        self.assertDictEqual(obj.styles, ref_style)

    def test_gen_formatted_frame_values(self):
        """Test gen_formatted_frame_values."""
        fcs = [
            pulse.ShiftPhase(np.pi / 2, pulse.DriveChannel(0)),
            pulse.ShiftFrequency(1e6, pulse.DriveChannel(0)),
        ]
        inst_data = create_instruction(fcs, np.pi / 2, 1e6, 5, 0.1)

        objs = frame.gen_formatted_frame_values(
            inst_data, formatter=self.formatter, device=self.device
        )

        # type check
        self.assertEqual(type(objs[0]), drawings.TextData)
        self.assertEqual(type(objs[1]), drawings.TextData)

    def test_gen_raw_operand_values_compact(self):
        """Test gen_raw_operand_values_compact."""
        fcs = [
            pulse.ShiftPhase(np.pi / 2, pulse.DriveChannel(0)),
            pulse.ShiftFrequency(1e6, pulse.DriveChannel(0)),
        ]
        inst_data = create_instruction(fcs, np.pi / 2, 1e6, 5, 0.1)

        obj = frame.gen_raw_operand_values_compact(
            inst_data, formatter=self.formatter, device=self.device
        )[0]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.DriveChannel(0)])
        self.assertEqual(obj.text, "1.57\n1.0e6")

    def gen_frame_symbol(self):
        """Test gen_frame_symbol."""
        fcs = [
            pulse.ShiftPhase(np.pi / 2, pulse.DriveChannel(0)),
            pulse.ShiftFrequency(1e6, pulse.DriveChannel(0)),
        ]
        inst_data = create_instruction(fcs, np.pi / 2, 1e6, 5, 0.1)

        obj = frame.gen_frame_symbol(inst_data, formatter=self.formatter, device=self.device)[0]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.DriveChannel(0)])
        self.assertEqual(obj.latex, self.formatter["latex_symbol.frame_change"])
        self.assertEqual(obj.text, self.formatter["unicode_symbol.frame_change"])

        # metadata check
        ref_meta = {
            "total phase change": np.pi / 2,
            "total frequency change": 1e6,
            "program": ["ShiftPhase(1.57 rad.)", "ShiftFrequency(1.00e+06 Hz)"],
            "t0 (cycle time)": 5,
            "t0 (sec)": 0.5,
        }
        self.assertDictEqual(obj.meta, ref_meta)

        # style check
        ref_style = {
            "zorder": self.formatter["layer.frame_change"],
            "color": self.formatter["color.frame_change"],
            "size": self.formatter["text_size.frame_change"],
            "va": "center",
            "ha": "center",
        }
        self.assertDictEqual(obj.styles, ref_style)

    def gen_frame_symbol_with_parameters(self):
        """Test gen_frame_symbol with parameterized frame."""
        theta = -1.0 * circuit.Parameter("P0")
        fcs = [pulse.ShiftPhase(theta, pulse.DriveChannel(0))]
        inst_data = create_instruction(fcs, np.pi / 2, 1e6, 5, 0.1)

        obj = frame.gen_frame_symbol(inst_data, formatter=self.formatter, device=self.device)[0]

        # metadata check
        ref_meta = {
            "total phase change": np.pi / 2,
            "total frequency change": 1e6,
            "program": ["ShiftPhase(-1.0*P0)"],
            "t0 (cycle time)": 5,
            "t0 (sec)": 0.5,
        }
        self.assertDictEqual(obj.meta, ref_meta)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestSnapshotGenerators(QiskitTestCase):
    """Tests for snapshot generators."""

    @ignore_pulse_deprecation_warnings
    def setUp(self) -> None:
        super().setUp()
        style = stylesheet.QiskitPulseStyle()
        self.formatter = style.formatter
        self.device = device_info.OpenPulseBackendInfo(
            name="test",
            dt=1,
            channel_frequency_map={
                pulse.DriveChannel(0): 5.0e9,
                pulse.DriveChannel(1): 5.1e9,
                pulse.MeasureChannel(0): 7.0e9,
                pulse.MeasureChannel(1): 7.1e9,
                pulse.ControlChannel(0): 5.0e9,
                pulse.ControlChannel(1): 5.1e9,
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
            },
        )

    def test_gen_snapshot_name(self):
        """Test gen_snapshot_name."""
        snap_inst = pulse.instructions.Snapshot(label="test_snapshot", snapshot_type="statevector")
        inst_data = types.SnapshotInstruction(5, 0.1, snap_inst)

        obj = snapshot.gen_snapshot_name(inst_data, formatter=self.formatter, device=self.device)[0]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.channels.SnapshotChannel()])
        self.assertEqual(obj.text, "test_snapshot")

        # style check
        ref_style = {
            "zorder": self.formatter["layer.snapshot"],
            "color": self.formatter["color.snapshot"],
            "size": self.formatter["text_size.annotate"],
            "va": "center",
            "ha": "center",
        }
        self.assertDictEqual(obj.styles, ref_style)

    def gen_snapshot_symbol(self):
        """Test gen_snapshot_symbol."""
        snap_inst = pulse.instructions.Snapshot(label="test_snapshot", snapshot_type="statevector")
        inst_data = types.SnapshotInstruction(5, 0.1, snap_inst)

        obj = snapshot.gen_snapshot_name(inst_data, formatter=self.formatter, device=self.device)[0]

        # type check
        self.assertEqual(type(obj), drawings.TextData)

        # data check
        self.assertListEqual(obj.channels, [pulse.channels.SnapshotChannel()])
        self.assertEqual(obj.text, self.formatter["unicode_symbol.snapshot"])
        self.assertEqual(obj.latex, self.formatter["latex_symbol.snapshot"])

        # metadata check
        ref_meta = {
            "snapshot type": "statevector",
            "t0 (cycle time)": 5,
            "t0 (sec)": 0.5,
            "name": "test_snapshot",
            "label": "test_snapshot",
        }
        self.assertDictEqual(obj.meta, ref_meta)

        # style check
        ref_style = {
            "zorder": self.formatter["layer.snapshot"],
            "color": self.formatter["color.snapshot"],
            "size": self.formatter["text_size.snapshot"],
            "va": "bottom",
            "ha": "center",
        }
        self.assertDictEqual(obj.styles, ref_style)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestBarrierGenerators(QiskitTestCase):
    """Tests for barrier generators."""

    @ignore_pulse_deprecation_warnings
    def setUp(self) -> None:
        super().setUp()
        style = stylesheet.QiskitPulseStyle()
        self.formatter = style.formatter
        self.device = device_info.OpenPulseBackendInfo(
            name="test",
            dt=1,
            channel_frequency_map={
                pulse.DriveChannel(0): 5.0e9,
                pulse.DriveChannel(1): 5.1e9,
                pulse.MeasureChannel(0): 7.0e9,
                pulse.MeasureChannel(1): 7.1e9,
                pulse.ControlChannel(0): 5.0e9,
                pulse.ControlChannel(1): 5.1e9,
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
            },
        )

    def test_gen_barrier(self):
        """Test gen_barrier."""
        inst_data = types.BarrierInstruction(
            5, 0.1, [pulse.DriveChannel(0), pulse.ControlChannel(0)]
        )
        obj = barrier.gen_barrier(inst_data, formatter=self.formatter, device=self.device)[0]

        # type check
        self.assertEqual(type(obj), drawings.LineData)

        # data check
        self.assertListEqual(obj.channels, [pulse.DriveChannel(0), pulse.ControlChannel(0)])

        ref_x = [5, 5]
        ref_y = [types.AbstractCoordinate.BOTTOM, types.AbstractCoordinate.TOP]

        self.assertListEqual(list(obj.xvals), ref_x)
        self.assertListEqual(list(obj.yvals), ref_y)

        # style check
        ref_style = {
            "alpha": self.formatter["alpha.barrier"],
            "zorder": self.formatter["layer.barrier"],
            "linewidth": self.formatter["line_width.barrier"],
            "linestyle": self.formatter["line_style.barrier"],
            "color": self.formatter["color.barrier"],
        }
        self.assertDictEqual(obj.styles, ref_style)

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

from qiskit import pulse, circuit
from qiskit.visualization.pulse_v2 import events
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestChannelEvents(QiskitTestCase):
    """Tests for ChannelEvents."""

    def test_parse_program(self):
        """Test typical pulse program."""
        test_pulse = pulse.Gaussian(10, 0.1, 3)

        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.SetPhase(3.14, pulse.DriveChannel(0)))
        sched = sched.insert(0, pulse.Play(test_pulse, pulse.DriveChannel(0)))
        sched = sched.insert(10, pulse.ShiftPhase(-1.57, pulse.DriveChannel(0)))
        sched = sched.insert(10, pulse.Play(test_pulse, pulse.DriveChannel(0)))

        ch_events = events.ChannelEvents.load_program(sched, pulse.DriveChannel(0))

        # check waveform data
        waveforms = list(ch_events.get_waveforms())
        inst_data0 = waveforms[0]
        self.assertEqual(inst_data0.t0, 0)
        self.assertEqual(inst_data0.frame.phase, 3.14)
        self.assertEqual(inst_data0.frame.freq, 0)
        self.assertEqual(inst_data0.inst, pulse.Play(test_pulse, pulse.DriveChannel(0)))

        inst_data1 = waveforms[1]
        self.assertEqual(inst_data1.t0, 10)
        self.assertEqual(inst_data1.frame.phase, 1.57)
        self.assertEqual(inst_data1.frame.freq, 0)
        self.assertEqual(inst_data1.inst, pulse.Play(test_pulse, pulse.DriveChannel(0)))

        # check frame data
        frames = list(ch_events.get_frame_changes())
        inst_data0 = frames[0]
        self.assertEqual(inst_data0.t0, 0)
        self.assertEqual(inst_data0.frame.phase, 3.14)
        self.assertEqual(inst_data0.frame.freq, 0)
        self.assertListEqual(inst_data0.inst, [pulse.SetPhase(3.14, pulse.DriveChannel(0))])

        inst_data1 = frames[1]
        self.assertEqual(inst_data1.t0, 10)
        self.assertEqual(inst_data1.frame.phase, -1.57)
        self.assertEqual(inst_data1.frame.freq, 0)
        self.assertListEqual(inst_data1.inst, [pulse.ShiftPhase(-1.57, pulse.DriveChannel(0))])

    def test_multiple_frames_at_the_same_time(self):
        """Test multiple frame instruction at the same time."""
        # shift phase followed by set phase
        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.ShiftPhase(-1.57, pulse.DriveChannel(0)))
        sched = sched.insert(0, pulse.SetPhase(3.14, pulse.DriveChannel(0)))

        ch_events = events.ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        frames = list(ch_events.get_frame_changes())
        inst_data0 = frames[0]
        self.assertAlmostEqual(inst_data0.frame.phase, 3.14)

        # set phase followed by shift phase
        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.SetPhase(3.14, pulse.DriveChannel(0)))
        sched = sched.insert(0, pulse.ShiftPhase(-1.57, pulse.DriveChannel(0)))

        ch_events = events.ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        frames = list(ch_events.get_frame_changes())
        inst_data0 = frames[0]
        self.assertAlmostEqual(inst_data0.frame.phase, 1.57)

    def test_frequency(self):
        """Test parse frequency."""
        sched = pulse.Schedule()
        sched = sched.insert(0, pulse.ShiftFrequency(1.0, pulse.DriveChannel(0)))
        sched = sched.insert(5, pulse.SetFrequency(5.0, pulse.DriveChannel(0)))

        ch_events = events.ChannelEvents.load_program(sched, pulse.DriveChannel(0))
        ch_events.set_config(dt=0.1, init_frequency=3.0, init_phase=0)
        frames = list(ch_events.get_frame_changes())

        inst_data0 = frames[0]
        self.assertAlmostEqual(inst_data0.frame.freq, 1.0)

        inst_data1 = frames[1]
        self.assertAlmostEqual(inst_data1.frame.freq, 1.0)

    def test_parameterized_parametric_pulse(self):
        """Test generating waveforms that are parameterized."""
        param = circuit.Parameter("amp")

        test_waveform = pulse.Play(pulse.Constant(10, param), pulse.DriveChannel(0))

        ch_events = events.ChannelEvents(
            waveforms={0: test_waveform}, frames={}, channel=pulse.DriveChannel(0)
        )

        pulse_inst = list(ch_events.get_waveforms())[0]

        self.assertTrue(pulse_inst.is_opaque)
        self.assertEqual(pulse_inst.inst, test_waveform)

    def test_parameterized_frame_change(self):
        """Test generating waveforms that are parameterized.

        Parameterized phase should be ignored when calculating waveform frame.
        This is due to phase modulated representation of waveforms,
        i.e. we cannot calculate the phase factor of waveform if the phase is unbound.
        """
        param = circuit.Parameter("phase")

        test_fc1 = pulse.ShiftPhase(param, pulse.DriveChannel(0))
        test_fc2 = pulse.ShiftPhase(1.57, pulse.DriveChannel(0))
        test_waveform = pulse.Play(pulse.Constant(10, 0.1), pulse.DriveChannel(0))

        ch_events = events.ChannelEvents(
            waveforms={0: test_waveform},
            frames={0: [test_fc1, test_fc2]},
            channel=pulse.DriveChannel(0),
        )

        # waveform frame
        pulse_inst = list(ch_events.get_waveforms())[0]

        self.assertFalse(pulse_inst.is_opaque)
        self.assertEqual(pulse_inst.frame.phase, 1.57)

        # framechange
        pulse_inst = list(ch_events.get_frame_changes())[0]

        self.assertTrue(pulse_inst.is_opaque)
        self.assertEqual(pulse_inst.frame.phase, param + 1.57)

    def test_zero_duration_delay(self):
        """Test generating waveforms that contains zero duration delay.

        Zero duration delay should be ignored.
        """
        ch = pulse.DriveChannel(0)

        test_sched = pulse.Schedule()
        test_sched += pulse.Play(pulse.Gaussian(160, 0.1, 40), ch)
        test_sched += pulse.Delay(0, ch)
        test_sched += pulse.Play(pulse.Gaussian(160, 0.1, 40), ch)
        test_sched += pulse.Delay(1, ch)
        test_sched += pulse.Play(pulse.Gaussian(160, 0.1, 40), ch)

        ch_events = events.ChannelEvents.load_program(test_sched, ch)

        self.assertEqual(len(list(ch_events.get_waveforms())), 4)

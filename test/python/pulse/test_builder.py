# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test pulse builder context utilities."""

from math import pi
import numpy as np

from qiskit import circuit, compiler, pulse
from qiskit.pulse import builder, exceptions, macros
from qiskit.pulse.instructions import directives
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.providers.fake_provider import FakeOpenPulse2Q, Fake127QPulseV1
from qiskit.pulse import library, instructions
from qiskit.pulse.exceptions import PulseError
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestBuilder(QiskitTestCase):
    """Test the pulse builder context."""

    @ignore_pulse_deprecation_warnings
    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            self.backend = FakeOpenPulse2Q()
        self.configuration = self.backend.configuration()
        self.defaults = self.backend.defaults()
        self.inst_map = self.defaults.instruction_schedule_map

    def assertScheduleEqual(self, program, target):
        """Assert an error when two pulse programs are not equal.

        .. note:: Two programs are converted into standard execution format then compared.
        """
        self.assertEqual(target_qobj_transform(program), target_qobj_transform(target))


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestBuilderBase(TestBuilder):
    """Test builder base."""

    def test_schedule_supplied(self):
        """Test that schedule is used if it is supplied to the builder."""
        d0 = pulse.DriveChannel(0)
        with pulse.build(name="reference") as reference:
            with pulse.align_sequential():
                pulse.delay(10, d0)

        with pulse.build(schedule=reference) as schedule:
            pass

        self.assertScheduleEqual(schedule, reference)
        self.assertEqual(schedule.name, "reference")

    def test_default_alignment_left(self):
        """Test default left alignment setting."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        with pulse.build(default_alignment="left") as schedule:
            pulse.delay(10, d0)
            pulse.delay(20, d1)

        with pulse.build(self.backend) as reference:
            with pulse.align_left():
                pulse.delay(10, d0)
                pulse.delay(20, d1)

        self.assertScheduleEqual(schedule, reference)

    def test_default_alignment_right(self):
        """Test default right alignment setting."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        with pulse.build(default_alignment="right") as schedule:
            pulse.delay(10, d0)
            pulse.delay(20, d1)

        with pulse.build() as reference:
            with pulse.align_right():
                pulse.delay(10, d0)
                pulse.delay(20, d1)

        self.assertScheduleEqual(schedule, reference)

    def test_default_alignment_sequential(self):
        """Test default sequential alignment setting."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        with pulse.build(default_alignment="sequential") as schedule:
            pulse.delay(10, d0)
            pulse.delay(20, d1)

        with pulse.build() as reference:
            with pulse.align_sequential():
                pulse.delay(10, d0)
                pulse.delay(20, d1)

        self.assertScheduleEqual(schedule, reference)

    def test_default_alignment_alignmentkind_instance(self):
        """Test default AlignmentKind instance"""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(0)

        with pulse.build(default_alignment=pulse.transforms.AlignEquispaced(100)) as schedule:
            pulse.delay(10, d0)
            pulse.delay(20, d1)

        with pulse.build() as reference:
            with pulse.align_equispaced(100):
                pulse.delay(10, d0)
                pulse.delay(20, d1)

        self.assertScheduleEqual(schedule, reference)

    def test_unknown_string_identifier(self):
        """Test that unknown string identifier raises an error"""

        with self.assertRaises(PulseError):
            with pulse.build(default_alignment="unknown") as _:
                pass


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestContexts(TestBuilder):
    """Test builder contexts."""

    def test_align_sequential(self):
        """Test the sequential alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as schedule:
            with pulse.align_sequential():
                pulse.delay(3, d0)
                pulse.delay(5, d1)
                pulse.delay(7, d0)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(8, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(3, instructions.Delay(5, d1), inplace=True)

        self.assertScheduleEqual(schedule, reference)

    def test_align_left(self):
        """Test the left alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as schedule:
            with pulse.align_left():
                pulse.delay(11, d2)
                pulse.delay(3, d0)
                with pulse.align_left():
                    pulse.delay(5, d1)
                    pulse.delay(7, d0)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(3, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(3, instructions.Delay(5, d1), inplace=True)
        # d2
        reference.insert(0, instructions.Delay(11, d2), inplace=True)

        self.assertScheduleEqual(schedule, reference)

    def test_align_right(self):
        """Test the right alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as schedule:
            with pulse.align_right():
                with pulse.align_right():
                    pulse.delay(11, d2)
                    pulse.delay(3, d0)
                pulse.delay(13, d0)
                pulse.delay(5, d1)

        reference = pulse.Schedule()
        # d0
        reference.insert(8, instructions.Delay(3, d0), inplace=True)
        reference.insert(11, instructions.Delay(13, d0), inplace=True)
        # d1
        reference.insert(19, instructions.Delay(5, d1), inplace=True)
        # d2
        reference.insert(0, instructions.Delay(11, d2), inplace=True)

        self.assertScheduleEqual(schedule, reference)

    def test_phase_offset(self):
        """Test the phase offset context."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            with pulse.phase_offset(3.14, d0):
                pulse.delay(10, d0)

        reference = pulse.Schedule()
        reference += instructions.ShiftPhase(3.14, d0)
        reference += instructions.Delay(10, d0)
        reference += instructions.ShiftPhase(-3.14, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_frequency_offset(self):
        """Test the frequency offset context."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            with pulse.frequency_offset(1e9, d0):
                pulse.delay(10, d0)

        reference = pulse.Schedule()
        reference += instructions.ShiftFrequency(1e9, d0)
        reference += instructions.Delay(10, d0)
        reference += instructions.ShiftFrequency(-1e9, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_phase_compensated_frequency_offset(self):
        """Test that the phase offset context properly compensates for phase
        accumulation."""
        d0 = pulse.DriveChannel(0)

        with pulse.build(self.backend) as schedule:
            with pulse.frequency_offset(1e9, d0, compensate_phase=True):
                pulse.delay(10, d0)

        reference = pulse.Schedule()
        reference += instructions.ShiftFrequency(1e9, d0)
        reference += instructions.Delay(10, d0)
        reference += instructions.ShiftPhase(
            -2 * np.pi * ((1e9 * 10 * self.configuration.dt) % 1), d0
        )
        reference += instructions.ShiftFrequency(-1e9, d0)

        self.assertScheduleEqual(schedule, reference)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestChannels(TestBuilder):
    """Test builder channels."""

    def test_drive_channel(self):
        """Text context builder drive channel."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.drive_channel(0), pulse.DriveChannel(0))

    def test_measure_channel(self):
        """Text context builder measure channel."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.measure_channel(0), pulse.MeasureChannel(0))

    def test_acquire_channel(self):
        """Text context builder acquire channel."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.acquire_channel(0), pulse.AcquireChannel(0))

    def test_control_channel(self):
        """Text context builder control channel."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.control_channels(0, 1)[0], pulse.ControlChannel(0))


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestInstructions(TestBuilder):
    """Test builder instructions."""

    def test_delay(self):
        """Test delay instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.delay(10, d0)

        reference = pulse.Schedule()
        reference += instructions.Delay(10, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_play_parametric_pulse(self):
        """Test play instruction with parametric pulse."""
        d0 = pulse.DriveChannel(0)
        test_pulse = library.Constant(10, 1.0)

        with pulse.build() as schedule:
            pulse.play(test_pulse, d0)

        reference = pulse.Schedule()
        reference += instructions.Play(test_pulse, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_play_sample_pulse(self):
        """Test play instruction with sample pulse."""
        d0 = pulse.DriveChannel(0)
        test_pulse = library.Waveform([0.0, 0.0])

        with pulse.build() as schedule:
            pulse.play(test_pulse, d0)

        reference = pulse.Schedule()
        reference += instructions.Play(test_pulse, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_play_array_pulse(self):
        """Test play instruction on an array directly."""
        d0 = pulse.DriveChannel(0)
        test_array = np.array([0.0, 0.0], dtype=np.complex128)

        with pulse.build() as schedule:
            pulse.play(test_array, d0)

        reference = pulse.Schedule()
        test_pulse = pulse.Waveform(test_array)
        reference += instructions.Play(test_pulse, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_play_name_argument(self):
        """Test name argument for play instruction."""
        d0 = pulse.DriveChannel(0)
        test_pulse = library.Constant(10, 1.0)

        with pulse.build() as schedule:
            pulse.play(test_pulse, channel=d0, name="new_name")

        self.assertEqual(schedule.instructions[0][1].name, "new_name")

    def test_acquire_memory_slot(self):
        """Test acquire instruction into memory slot."""
        acquire0 = pulse.AcquireChannel(0)
        mem0 = pulse.MemorySlot(0)

        with pulse.build() as schedule:
            pulse.acquire(10, acquire0, mem0)

        reference = pulse.Schedule()
        reference += pulse.Acquire(10, acquire0, mem_slot=mem0)

        self.assertScheduleEqual(schedule, reference)

    def test_acquire_register_slot(self):
        """Test acquire instruction into register slot."""
        acquire0 = pulse.AcquireChannel(0)
        reg0 = pulse.RegisterSlot(0)

        with pulse.build() as schedule:
            pulse.acquire(10, acquire0, reg0)

        reference = pulse.Schedule()
        reference += pulse.Acquire(10, acquire0, reg_slot=reg0)

        self.assertScheduleEqual(schedule, reference)

    def test_acquire_qubit(self):
        """Test acquire instruction on qubit."""
        acquire0 = pulse.AcquireChannel(0)
        mem0 = pulse.MemorySlot(0)

        with pulse.build() as schedule:
            pulse.acquire(10, 0, mem0)

        reference = pulse.Schedule()
        reference += pulse.Acquire(10, acquire0, mem_slot=mem0)

        self.assertScheduleEqual(schedule, reference)

    def test_instruction_name_argument(self):
        """Test setting the name of an instruction."""
        d0 = pulse.DriveChannel(0)

        for instruction_method in [
            pulse.delay,
            pulse.set_frequency,
            pulse.set_phase,
            pulse.shift_frequency,
            pulse.shift_phase,
        ]:
            with pulse.build() as schedule:
                instruction_method(0, d0, name="instruction_name")
            self.assertEqual(schedule.instructions[0][1].name, "instruction_name")

    def test_set_frequency(self):
        """Test set frequency instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.set_frequency(1e9, d0)

        reference = pulse.Schedule()
        reference += instructions.SetFrequency(1e9, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_shift_frequency(self):
        """Test shift frequency instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.shift_frequency(0.1e9, d0)

        reference = pulse.Schedule()
        reference += instructions.ShiftFrequency(0.1e9, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_set_phase(self):
        """Test set phase instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.set_phase(3.14, d0)

        reference = pulse.Schedule()
        reference += instructions.SetPhase(3.14, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_shift_phase(self):
        """Test shift phase instruction."""
        d0 = pulse.DriveChannel(0)

        with pulse.build() as schedule:
            pulse.shift_phase(3.14, d0)

        reference = pulse.Schedule()
        reference += instructions.ShiftPhase(3.14, d0)

        self.assertScheduleEqual(schedule, reference)

    def test_snapshot(self):
        """Test snapshot instruction."""
        with pulse.build() as schedule:
            pulse.snapshot("test", "state")

        reference = pulse.Schedule()
        reference += instructions.Snapshot("test", "state")

        self.assertScheduleEqual(schedule, reference)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestDirectives(TestBuilder):
    """Test builder directives."""

    def test_barrier_with_align_right(self):
        """Test barrier directive with right alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.delay(3, d0)
                pulse.barrier(d0, d1, d2)
                pulse.delay(11, d2)
                with pulse.align_right():
                    pulse.delay(5, d1)
                    pulse.delay(7, d0)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(7, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(9, instructions.Delay(5, d1), inplace=True)
        # d2
        reference.insert(3, instructions.Delay(11, d2), inplace=True)

        self.assertScheduleEqual(schedule, reference)

    def test_barrier_with_align_left(self):
        """Test barrier directive with left alignment context."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)

        with pulse.build() as schedule:
            with pulse.align_left():
                pulse.delay(3, d0)
                pulse.barrier(d0, d1, d2)
                pulse.delay(11, d2)
                with pulse.align_left():
                    pulse.delay(5, d1)
                    pulse.delay(7, d0)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Delay(3, d0), inplace=True)
        reference.insert(3, instructions.Delay(7, d0), inplace=True)
        # d1
        reference.insert(3, instructions.Delay(5, d1), inplace=True)
        # d2
        reference.insert(3, instructions.Delay(11, d2), inplace=True)

        self.assertScheduleEqual(schedule, reference)

    def test_barrier_on_qubits(self):
        """Test barrier directive on qubits."""
        with pulse.build(self.backend) as schedule:
            pulse.barrier(0, 1)

        reference = pulse.ScheduleBlock()
        reference += directives.RelativeBarrier(
            pulse.DriveChannel(0),
            pulse.DriveChannel(1),
            pulse.MeasureChannel(0),
            pulse.MeasureChannel(1),
            pulse.ControlChannel(0),
            pulse.ControlChannel(1),
            pulse.AcquireChannel(0),
            pulse.AcquireChannel(1),
        )
        self.assertEqual(schedule, reference)

    def test_trivial_barrier(self):
        """Test that trivial barrier is not added."""
        with pulse.build() as schedule:
            pulse.barrier(pulse.DriveChannel(0))

        self.assertEqual(schedule, pulse.ScheduleBlock())


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestUtilities(TestBuilder):
    """Test builder utilities."""

    def test_active_backend(self):
        """Test getting active builder backend."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.active_backend(), self.backend)

    def test_append_schedule(self):
        """Test appending a schedule to the active builder."""
        d0 = pulse.DriveChannel(0)
        reference = pulse.Schedule()
        reference += instructions.Delay(10, d0)

        with pulse.build() as schedule:
            builder.call(reference)

        self.assertScheduleEqual(schedule, reference)

    def test_append_instruction(self):
        """Test appending an instruction to the active builder."""
        d0 = pulse.DriveChannel(0)
        instruction = instructions.Delay(10, d0)

        with pulse.build() as schedule:
            builder.append_instruction(instruction)

        self.assertScheduleEqual(schedule, (0, instruction))

    def test_qubit_channels(self):
        """Test getting the qubit channels of the active builder's backend."""
        with pulse.build(self.backend):
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
        """Test builder utility to get number of qubits."""
        with pulse.build(self.backend):
            self.assertEqual(pulse.num_qubits(), 2)

    def test_samples_to_seconds(self):
        """Test samples to time"""
        config = self.backend.configuration()
        config.dt = 0.1
        with pulse.build(self.backend):
            time = pulse.samples_to_seconds(100)
            self.assertTrue(isinstance(time, float))
            self.assertEqual(pulse.samples_to_seconds(100), 10)

    def test_samples_to_seconds_array(self):
        """Test samples to time (array format)."""
        config = self.backend.configuration()
        config.dt = 0.1
        with pulse.build(self.backend):
            samples = np.array([100, 200, 300])
            times = pulse.samples_to_seconds(samples)
            self.assertTrue(np.issubdtype(times.dtype, np.floating))
            np.testing.assert_allclose(times, np.array([10, 20, 30]))

    def test_seconds_to_samples(self):
        """Test time to samples"""
        config = self.backend.configuration()
        config.dt = 0.1
        with pulse.build(self.backend):
            samples = pulse.seconds_to_samples(10)
            self.assertTrue(isinstance(samples, int))
            self.assertEqual(pulse.seconds_to_samples(10), 100)

    def test_seconds_to_samples_array(self):
        """Test time to samples (array format)."""
        config = self.backend.configuration()
        config.dt = 0.1
        with pulse.build(self.backend):
            times = np.array([10, 20, 30])
            samples = pulse.seconds_to_samples(times)
            self.assertTrue(np.issubdtype(samples.dtype, np.integer))
            np.testing.assert_allclose(pulse.seconds_to_samples(times), np.array([100, 200, 300]))


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestMacros(TestBuilder):
    """Test builder macros."""

    def test_macro(self):
        """Test builder macro decorator."""

        @pulse.macro
        def nested(a):
            pulse.play(pulse.Gaussian(100, a, 20), pulse.drive_channel(0))
            return a * 2

        @pulse.macro
        def test():
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
        """Test utility function - measure."""
        with pulse.build(self.backend) as schedule:
            reg = pulse.measure(0)

        self.assertEqual(reg, pulse.MemorySlot(0))

        reference = macros.measure(
            qubits=[0], inst_map=self.inst_map, meas_map=self.configuration.meas_map
        )

        self.assertScheduleEqual(schedule, reference)

    def test_measure_multi_qubits(self):
        """Test utility function - measure with multi qubits."""
        with pulse.build(self.backend) as schedule:
            regs = pulse.measure([0, 1])

        self.assertListEqual(regs, [pulse.MemorySlot(0), pulse.MemorySlot(1)])

        reference = macros.measure(
            qubits=[0, 1], inst_map=self.inst_map, meas_map=self.configuration.meas_map
        )

        self.assertScheduleEqual(schedule, reference)

    def test_measure_all(self):
        """Test utility function - measure."""
        with pulse.build(self.backend) as schedule:
            regs = pulse.measure_all()

        self.assertEqual(regs, [pulse.MemorySlot(0), pulse.MemorySlot(1)])
        reference = macros.measure_all(self.backend)

        self.assertScheduleEqual(schedule, reference)

        with self.assertWarns(DeprecationWarning):
            backend = Fake127QPulseV1()
        num_qubits = backend.configuration().num_qubits
        with pulse.build(backend) as schedule:
            with self.assertWarns(DeprecationWarning):
                regs = pulse.measure_all()

        reference = backend.defaults().instruction_schedule_map.get(
            "measure", list(range(num_qubits))
        )

        self.assertScheduleEqual(schedule, reference)

    def test_delay_qubit(self):
        """Test delaying on a qubit macro."""
        with pulse.build(self.backend) as schedule:
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
        """Test delaying on multiple qubits to make sure we don't insert delays twice."""
        with pulse.build(self.backend) as schedule:
            pulse.delay_qubits(10, 0, 1)

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        m0 = pulse.MeasureChannel(0)
        m1 = pulse.MeasureChannel(1)
        a0 = pulse.AcquireChannel(0)
        a1 = pulse.AcquireChannel(1)
        u0 = pulse.ControlChannel(0)
        u1 = pulse.ControlChannel(1)

        reference = pulse.Schedule()
        reference += instructions.Delay(10, d0)
        reference += instructions.Delay(10, d1)
        reference += instructions.Delay(10, m0)
        reference += instructions.Delay(10, m1)
        reference += instructions.Delay(10, a0)
        reference += instructions.Delay(10, a1)
        reference += instructions.Delay(10, u0)
        reference += instructions.Delay(10, u1)

        self.assertScheduleEqual(schedule, reference)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestBuilderComposition(TestBuilder):
    """Test more sophisticated composite builder examples."""

    def test_complex_build(self):
        """Test a general program build with nested contexts,
        circuits and macros."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        d2 = pulse.DriveChannel(2)
        delay_dur = 30
        short_dur = 20
        long_dur = 49

        def get_sched(qubit_idx: [int], backend):
            qc = circuit.QuantumCircuit(2)
            for idx in qubit_idx:
                qc.append(circuit.library.U2Gate(0, pi / 2), [idx])
            with self.assertWarnsRegex(
                DeprecationWarning,
                expected_regex="The `transpile` function will "
                "stop supporting inputs of type `BackendV1`",
            ):
                transpiled = compiler.transpile(qc, backend=backend, optimization_level=1)
            with self.assertWarns(DeprecationWarning):
                return compiler.schedule(transpiled, backend)

        with pulse.build(self.backend) as schedule:
            with pulse.align_sequential():
                pulse.delay(delay_dur, d0)
                pulse.call(get_sched([1], self.backend))

            with pulse.align_right():
                pulse.play(library.Constant(short_dur, 0.1), d1)
                pulse.play(library.Constant(long_dur, 0.1), d2)
                pulse.call(get_sched([1], self.backend))

            with pulse.align_left():
                pulse.call(get_sched([0, 1, 0], self.backend))

            pulse.measure(0)

        # prepare and schedule circuits that will be used.
        single_u2_qc = circuit.QuantumCircuit(2)
        single_u2_qc.append(circuit.library.U2Gate(0, pi / 2), [1])
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            single_u2_qc = compiler.transpile(single_u2_qc, self.backend, optimization_level=1)
        with self.assertWarns(DeprecationWarning):
            single_u2_sched = compiler.schedule(single_u2_qc, self.backend)

        # sequential context
        sequential_reference = pulse.Schedule()
        sequential_reference += instructions.Delay(delay_dur, d0)
        sequential_reference.insert(delay_dur, single_u2_sched, inplace=True)

        # align right
        align_right_reference = pulse.Schedule()
        align_right_reference += pulse.Play(library.Constant(long_dur, 0.1), d2)
        align_right_reference.insert(
            long_dur - single_u2_sched.duration, single_u2_sched, inplace=True
        )
        align_right_reference.insert(
            long_dur - single_u2_sched.duration - short_dur,
            pulse.Play(library.Constant(short_dur, 0.1), d1),
            inplace=True,
        )

        # align left
        triple_u2_qc = circuit.QuantumCircuit(2)
        triple_u2_qc.append(circuit.library.U2Gate(0, pi / 2), [0])
        triple_u2_qc.append(circuit.library.U2Gate(0, pi / 2), [1])
        triple_u2_qc.append(circuit.library.U2Gate(0, pi / 2), [0])
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            triple_u2_qc = compiler.transpile(triple_u2_qc, self.backend, optimization_level=1)
        with self.assertWarns(DeprecationWarning):
            align_left_reference = compiler.schedule(triple_u2_qc, self.backend, method="alap")

        # measurement
        measure_reference = macros.measure(
            qubits=[0], inst_map=self.inst_map, meas_map=self.configuration.meas_map
        )
        reference = pulse.Schedule()
        reference += sequential_reference
        # Insert so that the long pulse on d2 occurs as early as possible
        # without an overval on d1.
        insert_time = reference.ch_stop_time(d1) - align_right_reference.ch_start_time(d1)
        reference.insert(insert_time, align_right_reference, inplace=True)
        reference.insert(reference.ch_stop_time(d0, d1), align_left_reference, inplace=True)
        reference += measure_reference

        self.assertScheduleEqual(schedule, reference)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestSubroutineCall(TestBuilder):
    """Test for calling subroutine."""

    def test_call(self):
        """Test calling schedule instruction."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        reference = pulse.Schedule()
        reference = reference.insert(10, instructions.Delay(10, d0))
        reference += instructions.Delay(20, d1)

        ref_sched = pulse.Schedule()
        ref_sched += reference

        with pulse.build() as schedule:
            with pulse.align_right():
                builder.call(reference)

        self.assertScheduleEqual(schedule, ref_sched)

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.call(reference)

        self.assertScheduleEqual(schedule, ref_sched)

    def test_subroutine_not_transformed(self):
        """Test called schedule is not transformed."""
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        subprogram = pulse.Schedule()
        subprogram.insert(0, pulse.Delay(30, d0), inplace=True)
        subprogram.insert(10, pulse.Delay(10, d1), inplace=True)

        with pulse.build() as target:
            with pulse.align_right():
                pulse.delay(10, d1)
                pulse.call(subprogram)

        reference = pulse.Schedule()
        reference.insert(0, pulse.Delay(10, d1), inplace=True)
        reference.insert(10, pulse.Delay(30, d0), inplace=True)
        reference.insert(20, pulse.Delay(10, d1), inplace=True)

        self.assertScheduleEqual(target, reference)

    def test_deepcopying_subroutine(self):
        """Test if deepcopying the schedule can copy inline subroutine."""
        from copy import deepcopy

        with pulse.build() as subroutine:
            pulse.delay(10, pulse.DriveChannel(0))

        with pulse.build() as main_prog:
            pulse.call(subroutine)

        copied_prog = deepcopy(main_prog)

        main_call = main_prog.instructions[0]
        copy_call = copied_prog.instructions[0]

        self.assertNotEqual(id(main_call), id(copy_call))

    def test_call_with_parameters(self):
        """Test call subroutine with parameters."""
        amp = circuit.Parameter("amp")

        with pulse.build() as subroutine:
            pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(0))

        with pulse.build() as main_prog:
            pulse.call(subroutine, amp=0.1)
            pulse.call(subroutine, amp=0.3)

        self.assertEqual(main_prog.is_parameterized(), False)

        assigned_sched = target_qobj_transform(main_prog)

        play_0 = assigned_sched.instructions[0][1]
        play_1 = assigned_sched.instructions[1][1]

        self.assertEqual(play_0.pulse.amp, 0.1)
        self.assertEqual(play_1.pulse.amp, 0.3)

    def test_call_partly_with_parameters(self):
        """Test multiple calls partly with parameters then assign."""
        amp = circuit.Parameter("amp")

        with pulse.build() as subroutine:
            pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(0))

        with pulse.build() as main_prog:
            pulse.call(subroutine, amp=0.1)
            pulse.call(subroutine)

        self.assertEqual(main_prog.is_parameterized(), True)

        main_prog.assign_parameters({amp: 0.5})
        self.assertEqual(main_prog.is_parameterized(), False)

        assigned_sched = target_qobj_transform(main_prog)

        play_0 = assigned_sched.instructions[0][1]
        play_1 = assigned_sched.instructions[1][1]

        self.assertEqual(play_0.pulse.amp, 0.1)
        self.assertEqual(play_1.pulse.amp, 0.5)

    def test_call_with_not_existing_parameter(self):
        """Test call subroutine with parameter not defined."""
        amp = circuit.Parameter("amp1")

        with pulse.build() as subroutine:
            pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(0))

        with self.assertRaises(exceptions.PulseError):
            with pulse.build():
                pulse.call(subroutine, amp=0.1)

    def test_call_with_common_parameter(self):
        """Test call subroutine with parameter that is defined multiple times."""
        amp = circuit.Parameter("amp")

        with pulse.build() as subroutine:
            pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(320, amp, 80), pulse.DriveChannel(0))

        with pulse.build() as main_prog:
            pulse.call(subroutine, amp=0.1)

        assigned_sched = target_qobj_transform(main_prog)

        play_0 = assigned_sched.instructions[0][1]
        play_1 = assigned_sched.instructions[1][1]

        self.assertEqual(play_0.pulse.amp, 0.1)
        self.assertEqual(play_1.pulse.amp, 0.1)

    def test_call_with_parameter_name_collision(self):
        """Test call subroutine with duplicated parameter names."""
        amp1 = circuit.Parameter("amp")
        amp2 = circuit.Parameter("amp")
        sigma = circuit.Parameter("sigma")

        with pulse.build() as subroutine:
            pulse.play(pulse.Gaussian(160, amp1, sigma), pulse.DriveChannel(0))
            pulse.play(pulse.Gaussian(160, amp2, sigma), pulse.DriveChannel(0))

        with pulse.build() as main_prog:
            pulse.call(subroutine, value_dict={amp1: 0.1, amp2: 0.2}, sigma=40)

        assigned_sched = target_qobj_transform(main_prog)

        play_0 = assigned_sched.instructions[0][1]
        play_1 = assigned_sched.instructions[1][1]

        self.assertEqual(play_0.pulse.amp, 0.1)
        self.assertEqual(play_0.pulse.sigma, 40)
        self.assertEqual(play_1.pulse.amp, 0.2)
        self.assertEqual(play_1.pulse.sigma, 40)

    def test_call_subroutine_with_parametrized_duration(self):
        """Test call subroutine containing a parametrized duration."""
        dur = circuit.Parameter("dur")

        with pulse.build() as subroutine:
            pulse.play(pulse.Constant(dur, 0.1), pulse.DriveChannel(0))
            pulse.play(pulse.Constant(dur, 0.2), pulse.DriveChannel(0))

        with pulse.build() as main:
            pulse.call(subroutine)

        self.assertEqual(len(main.blocks), 1)

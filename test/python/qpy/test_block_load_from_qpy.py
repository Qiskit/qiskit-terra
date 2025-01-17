# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for the schedule block qpy loading and saving."""

import io
import unittest
import warnings
from ddt import ddt, data, unpack
import numpy as np
import symengine as sym

from qiskit.pulse import builder, Schedule
from qiskit.pulse.library import (
    SymbolicPulse,
    Gaussian,
    GaussianSquare,
    Drag,
    Constant,
    Waveform,
)
from qiskit.pulse.channels import (
    DriveChannel,
    ControlChannel,
    MeasureChannel,
    AcquireChannel,
    MemorySlot,
    RegisterSlot,
)
from qiskit.pulse.instructions import Play, TimeBlockade
from qiskit.circuit import Parameter, QuantumCircuit, Gate
from qiskit.qpy import dump, load
from qiskit.qpy.exceptions import QPYLoadingDeprecatedFeatureWarning
from qiskit.utils import optionals as _optional
from qiskit.pulse.configuration import Kernel, Discriminator
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class QpyScheduleTestCase(QiskitTestCase):
    """QPY schedule testing platform."""

    def assert_roundtrip_equal(self, block, use_symengine=False):
        """QPY roundtrip equal test."""
        qpy_file = io.BytesIO()
        with self.assertWarns(DeprecationWarning):
            dump(block, qpy_file, use_symengine=use_symengine)
        qpy_file.seek(0)
        new_block = load(qpy_file)[0]

        self.assertEqual(block, new_block)


@ddt
class TestLoadFromQPY(QpyScheduleTestCase):
    """Test loading and saving schedule block to qpy file."""

    @data(
        (Gaussian, DriveChannel, 160, 0.1, 40),
        (GaussianSquare, DriveChannel, 800, 0.1, 64, 544),
        (Drag, DriveChannel, 160, 0.1, 40, 0.5),
        (Constant, DriveChannel, 800, 0.1),
        (Constant, ControlChannel, 800, 0.1),
        (Constant, MeasureChannel, 800, 0.1),
    )
    @unpack
    def test_library_pulse_play(self, envelope, channel, *params):
        """Test playing standard pulses."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.play(
                    envelope(*params),
                    channel(0),
                )
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_playing_custom_symbolic_pulse(self):
        """Test playing a custom user pulse."""
        # pylint: disable=invalid-name
        t, amp, freq = sym.symbols("t, amp, freq")
        sym_envelope = 2 * amp * (freq * t - sym.floor(1 / 2 + freq * t))

        with self.assertWarns(DeprecationWarning):
            my_pulse = SymbolicPulse(
                pulse_type="Sawtooth",
                duration=100,
                parameters={"amp": 0.1, "freq": 0.05},
                envelope=sym_envelope,
                name="pulse1",
            )
            with builder.build() as test_sched:
                builder.play(my_pulse, DriveChannel(0))
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_symbolic_amplitude_limit(self):
        """Test applying amplitude limit to symbolic pulse."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.play(
                    Gaussian(160, 20, 40, limit_amplitude=False),
                    DriveChannel(0),
                )
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_waveform_amplitude_limit(self):
        """Test applying amplitude limit to waveform."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.play(
                    Waveform([1, 2, 3, 4, 5], limit_amplitude=False),
                    DriveChannel(0),
                )
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_playing_waveform(self):
        """Test playing waveform."""
        # pylint: disable=invalid-name
        t = np.linspace(0, 1, 100)
        waveform = 0.1 * np.sin(2 * np.pi * t)
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.play(waveform, DriveChannel(0))
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_phases(self):
        """Test phase."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.shift_phase(0.1, DriveChannel(0))
                builder.set_phase(0.4, DriveChannel(1))
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_frequencies(self):
        """Test frequency."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.shift_frequency(10e6, DriveChannel(0))
                builder.set_frequency(5e9, DriveChannel(1))
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_delay(self):
        """Test delay."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.delay(100, DriveChannel(0))
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_barrier(self):
        """Test barrier."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.barrier(DriveChannel(0), DriveChannel(1), ControlChannel(2))
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_time_blockade(self):
        """Test time blockade."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.append_instruction(TimeBlockade(10, DriveChannel(0)))
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_measure(self):
        """Test measurement."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.acquire(100, AcquireChannel(0), MemorySlot(0))
                builder.acquire(100, AcquireChannel(1), RegisterSlot(1))
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    @data(
        (0, Parameter("dur"), 0.1, 40),
        (Parameter("ch1"), 160, 0.1, 40),
        (Parameter("ch1"), Parameter("dur"), Parameter("amp"), Parameter("sigma")),
        (0, 160, Parameter("amp") * np.exp(1j * Parameter("phase")), 40),
    )
    @unpack
    def test_parameterized(self, channel, *params):
        """Test playing parameterized pulse."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.play(Gaussian(*params), DriveChannel(channel))
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_nested_blocks(self):
        """Test nested blocks with different alignment contexts."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                with builder.align_equispaced(duration=1200):
                    with builder.align_left():
                        builder.delay(100, DriveChannel(0))
                        builder.delay(200, DriveChannel(1))
                    with builder.align_right():
                        builder.delay(100, DriveChannel(0))
                        builder.delay(200, DriveChannel(1))
                    with builder.align_sequential():
                        builder.delay(100, DriveChannel(0))
                        builder.delay(200, DriveChannel(1))
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_called_schedule(self):
        """Test referenced pulse Schedule object.

        Referenced object is naively converted into ScheduleBlock with TimeBlockade instructions.
        Thus referenced Schedule is still QPY compatible.
        """
        with self.assertWarns(DeprecationWarning):
            refsched = Schedule()
            refsched.insert(20, Play(Constant(100, 0.1), DriveChannel(0)))
            refsched.insert(50, Play(Constant(100, 0.1), DriveChannel(1)))

            with builder.build() as test_sched:
                builder.call(refsched, name="test_ref")
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_unassigned_reference(self):
        """Test schedule with unassigned reference."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.reference("custom1", "q0")
                builder.reference("custom1", "q1")

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=DeprecationWarning)
            with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
                self.assert_roundtrip_equal(test_sched)

    def test_partly_assigned_reference(self):
        """Test schedule with partly assigned reference."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.reference("custom1", "q0")
                builder.reference("custom1", "q1")

            with builder.build() as sub_q0:
                builder.delay(Parameter("duration"), DriveChannel(0))

        test_sched.assign_references(
            {("custom1", "q0"): sub_q0},
            inplace=True,
        )

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=DeprecationWarning)
            with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
                self.assert_roundtrip_equal(test_sched)

    def test_nested_assigned_reference(self):
        """Test schedule with assigned reference for nested schedule."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                with builder.align_left():
                    builder.reference("custom1", "q0")
                builder.reference("custom1", "q1")

            with builder.build() as sub_q0:
                builder.delay(Parameter("duration"), DriveChannel(0))

            with builder.build() as sub_q1:
                builder.delay(Parameter("duration"), DriveChannel(1))

        test_sched.assign_references(
            {("custom1", "q0"): sub_q0, ("custom1", "q1"): sub_q1},
            inplace=True,
        )

        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_bell_schedule(self):
        """Test complex schedule to create a Bell state."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                with builder.align_sequential():
                    # H
                    builder.shift_phase(-1.57, DriveChannel(0))
                    builder.play(Drag(160, 0.05, 40, 1.3), DriveChannel(0))
                    builder.shift_phase(-1.57, DriveChannel(0))
                    # ECR
                    with builder.align_left():
                        builder.play(GaussianSquare(800, 0.05, 64, 544), DriveChannel(1))
                        builder.play(GaussianSquare(800, 0.22, 64, 544, 2), ControlChannel(0))
                    builder.play(Drag(160, 0.1, 40, 1.5), DriveChannel(0))
                    with builder.align_left():
                        builder.play(GaussianSquare(800, -0.05, 64, 544), DriveChannel(1))
                        builder.play(GaussianSquare(800, -0.22, 64, 544, 2), ControlChannel(0))
                    builder.play(Drag(160, 0.1, 40, 1.5), DriveChannel(0))
                    # Measure
                    with builder.align_left():
                        builder.play(GaussianSquare(8000, 0.2, 64, 7744), MeasureChannel(0))
                        builder.acquire(8000, AcquireChannel(0), MemorySlot(0))

        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    @unittest.skipUnless(_optional.HAS_SYMENGINE, "Symengine required for this test")
    def test_bell_schedule_use_symengine(self):
        """Test complex schedule to create a Bell state."""
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                with builder.align_sequential():
                    # H
                    builder.shift_phase(-1.57, DriveChannel(0))
                    builder.play(Drag(160, 0.05, 40, 1.3), DriveChannel(0))
                    builder.shift_phase(-1.57, DriveChannel(0))
                    # ECR
                    with builder.align_left():
                        builder.play(GaussianSquare(800, 0.05, 64, 544), DriveChannel(1))
                        builder.play(GaussianSquare(800, 0.22, 64, 544, 2), ControlChannel(0))
                    builder.play(Drag(160, 0.1, 40, 1.5), DriveChannel(0))
                    with builder.align_left():
                        builder.play(GaussianSquare(800, -0.05, 64, 544), DriveChannel(1))
                        builder.play(GaussianSquare(800, -0.22, 64, 544, 2), ControlChannel(0))
                    builder.play(Drag(160, 0.1, 40, 1.5), DriveChannel(0))
                    # Measure
                    with builder.align_left():
                        builder.play(GaussianSquare(8000, 0.2, 64, 7744), MeasureChannel(0))
                        builder.acquire(8000, AcquireChannel(0), MemorySlot(0))

        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched, True)

    def test_with_acquire_instruction_with_kernel(self):
        """Test a schedblk with acquire instruction with kernel."""
        kernel = Kernel(
            name="my_kernel", kernel={"real": np.ones(10), "imag": np.zeros(10)}, bias=[0, 0]
        )
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.acquire(100, AcquireChannel(0), MemorySlot(0), kernel=kernel)

        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)

    def test_with_acquire_instruction_with_discriminator(self):
        """Test a schedblk with acquire instruction with a discriminator."""
        discriminator = Discriminator(
            name="my_discriminator", discriminator_type="linear", params=[1, 0]
        )
        with self.assertWarns(DeprecationWarning):
            with builder.build() as test_sched:
                builder.acquire(100, AcquireChannel(0), MemorySlot(0), discriminator=discriminator)

        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            self.assert_roundtrip_equal(test_sched)


class TestPulseGate(QpyScheduleTestCase):
    """Test loading and saving pulse gate attached circuit to qpy file."""

    def test_1q_gate(self):
        """Test for single qubit pulse gate."""
        mygate = Gate("mygate", 1, [])

        with self.assertWarns(DeprecationWarning):
            with builder.build() as caldef:
                builder.play(Constant(100, 0.1), DriveChannel(0))

        qc = QuantumCircuit(2)
        qc.append(mygate, [0])
        with self.assertWarns(DeprecationWarning):
            qc.add_calibration(mygate, (0,), caldef)

        self.assert_roundtrip_equal(qc)

    def test_2q_gate(self):
        """Test for two qubit pulse gate."""
        mygate = Gate("mygate", 2, [])

        with self.assertWarns(DeprecationWarning):
            with builder.build() as caldef:
                builder.play(Constant(100, 0.1), ControlChannel(0))

        qc = QuantumCircuit(2)
        qc.append(mygate, [0, 1])
        with self.assertWarns(DeprecationWarning):
            qc.add_calibration(mygate, (0, 1), caldef)

        self.assert_roundtrip_equal(qc)

    def test_parameterized_gate(self):
        """Test for parameterized pulse gate."""
        amp = Parameter("amp")
        angle = Parameter("angle")
        mygate = Gate("mygate", 2, [amp, angle])

        with self.assertWarns(DeprecationWarning):
            with builder.build() as caldef:
                builder.play(Constant(100, amp * np.exp(1j * angle)), ControlChannel(0))

        qc = QuantumCircuit(2)
        qc.append(mygate, [0, 1])
        with self.assertWarns(DeprecationWarning):
            qc.add_calibration(mygate, (0, 1), caldef)

        self.assert_roundtrip_equal(qc)

    def test_override(self):
        """Test for overriding standard gate with pulse gate."""
        amp = Parameter("amp")

        with self.assertWarns(DeprecationWarning):
            with builder.build() as caldef:
                builder.play(Constant(100, amp), ControlChannel(0))

        qc = QuantumCircuit(2)
        qc.rx(amp, 0)
        with self.assertWarns(DeprecationWarning):
            qc.add_calibration("rx", (0,), caldef, [amp])

        self.assert_roundtrip_equal(qc)

    def test_multiple_calibrations(self):
        """Test for circuit with multiple pulse gates."""
        amp1 = Parameter("amp1")
        amp2 = Parameter("amp2")
        mygate = Gate("mygate", 1, [amp2])

        with self.assertWarns(DeprecationWarning):
            with builder.build() as caldef1:
                builder.play(Constant(100, amp1), DriveChannel(0))

            with builder.build() as caldef2:
                builder.play(Constant(100, amp2), DriveChannel(1))

        qc = QuantumCircuit(2)
        qc.rx(amp1, 0)
        qc.append(mygate, [1])
        with self.assertWarns(DeprecationWarning):
            qc.add_calibration("rx", (0,), caldef1, [amp1])
            qc.add_calibration(mygate, (1,), caldef2)

        self.assert_roundtrip_equal(qc)

    def test_with_acquire_instruction_with_kernel(self):
        """Test a pulse gate with acquire instruction with kernel."""
        kernel = Kernel(
            name="my_kernel", kernel={"real": np.zeros(10), "imag": np.zeros(10)}, bias=[0, 0]
        )

        with self.assertWarns(DeprecationWarning):
            with builder.build() as sched:
                builder.acquire(10, AcquireChannel(0), MemorySlot(0), kernel=kernel)

        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        with self.assertWarns(DeprecationWarning):
            qc.add_calibration("measure", (0,), sched)

        self.assert_roundtrip_equal(qc)

    def test_with_acquire_instruction_with_discriminator(self):
        """Test a pulse gate with acquire instruction with discriminator."""
        discriminator = Discriminator("my_discriminator")

        with self.assertWarns(DeprecationWarning):
            with builder.build() as sched:
                builder.acquire(10, AcquireChannel(0), MemorySlot(0), discriminator=discriminator)

        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        with self.assertWarns(DeprecationWarning):
            qc.add_calibration("measure", (0,), sched)

        self.assert_roundtrip_equal(qc)


class TestSymengineLoadFromQPY(QiskitTestCase):
    """Test use of symengine in qpy set of methods."""

    def setUp(self):
        super().setUp()

        # pylint: disable=invalid-name
        t, amp, freq = sym.symbols("t, amp, freq")
        sym_envelope = 2 * amp * (freq * t - sym.floor(1 / 2 + freq * t))

        with self.assertWarns(DeprecationWarning):
            my_pulse = SymbolicPulse(
                pulse_type="Sawtooth",
                duration=100,
                parameters={"amp": 0.1, "freq": 0.05},
                envelope=sym_envelope,
                name="pulse1",
            )
            with builder.build() as test_sched:
                builder.play(my_pulse, DriveChannel(0))

        self.test_sched = test_sched

    @unittest.skipIf(not _optional.HAS_SYMENGINE, "Install symengine to run this test.")
    def test_symengine_full_path(self):
        """Test use_symengine option for circuit with parameter expressions."""
        qpy_file = io.BytesIO()
        with self.assertWarns(DeprecationWarning):
            dump(self.test_sched, qpy_file, use_symengine=True)
        qpy_file.seek(0)
        with self.assertWarns(QPYLoadingDeprecatedFeatureWarning):
            new_sched = load(qpy_file)[0]
        self.assertEqual(self.test_sched, new_sched)

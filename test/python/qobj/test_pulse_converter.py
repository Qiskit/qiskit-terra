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

"""Converter Test."""

import hashlib
import numpy as np

from qiskit.pulse import LoConfig, Kernel, Discriminator
from qiskit.pulse.channels import (
    DriveChannel,
    ControlChannel,
    MeasureChannel,
    AcquireChannel,
    MemorySlot,
    RegisterSlot,
)
from qiskit.pulse.instructions import (
    SetPhase,
    ShiftPhase,
    SetFrequency,
    ShiftFrequency,
    Play,
    Delay,
    Acquire,
    Snapshot,
)
from qiskit.pulse.library import Waveform, Gaussian, GaussianSquare, Constant, Drag
from qiskit.pulse.schedule import Schedule
from qiskit.qobj import (
    PulseQobjInstruction,
    PulseQobjExperimentConfig,
    PulseLibraryItem,
    QobjMeasurementOption,
)
from qiskit.qobj.converters import (
    InstructionToQobjConverter,
    QobjToInstructionConverter,
    LoConfigConverter,
)
from qiskit.exceptions import QiskitError
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestInstructionToQobjConverter(QiskitTestCase):
    """Pulse converter tests."""

    def test_drive_instruction(self):
        """Test converted qobj from Play."""
        with self.assertWarns(DeprecationWarning):
            converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
            instruction = Play(Waveform(np.arange(0, 0.01), name="linear"), DriveChannel(0))
            valid_qobj = PulseQobjInstruction(name="linear", ch="d0", t0=0)
            self.assertEqual(converter(0, instruction), valid_qobj)

    def test_gaussian_pulse_instruction(self):
        """Test that parametric pulses are correctly converted to PulseQobjInstructions."""
        amp = 0.3
        angle = -0.7
        with self.assertWarns(DeprecationWarning):
            converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
            instruction = Play(
                Gaussian(duration=25, sigma=15, amp=amp, angle=angle), DriveChannel(0)
            )
        with self.assertWarns(DeprecationWarning):
            valid_qobj = PulseQobjInstruction(
                name="parametric_pulse",
                pulse_shape="gaussian",
                ch="d0",
                t0=0,
                parameters={"duration": 25, "sigma": 15, "amp": amp * np.exp(1j * angle)},
            )
            self.assertEqual(converter(0, instruction), valid_qobj)

    def test_gaussian_square_pulse_instruction(self):
        """Test that parametric pulses are correctly converted to PulseQobjInstructions."""
        with self.assertWarns(DeprecationWarning):
            converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
        amp = 0.7
        angle = -0.6
        with self.assertWarns(DeprecationWarning):
            instruction = Play(
                GaussianSquare(duration=1500, sigma=15, amp=amp, width=1300, angle=angle),
                MeasureChannel(1),
            )

        with self.assertWarns(DeprecationWarning):
            valid_qobj = PulseQobjInstruction(
                name="parametric_pulse",
                pulse_shape="gaussian_square",
                ch="m1",
                t0=10,
                parameters={
                    "duration": 1500,
                    "sigma": 15,
                    "amp": amp * np.exp(1j * angle),
                    "width": 1300,
                },
            )
            self.assertEqual(converter(10, instruction), valid_qobj)

    def test_constant_pulse_instruction(self):
        """Test that parametric pulses are correctly converted to PulseQobjInstructions."""
        with self.assertWarns(DeprecationWarning):
            converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
            instruction = Play(Constant(duration=25, amp=1, angle=np.pi), ControlChannel(2))

        with self.assertWarns(DeprecationWarning):
            valid_qobj = PulseQobjInstruction(
                name="parametric_pulse",
                pulse_shape="constant",
                ch="u2",
                t0=20,
                parameters={"duration": 25, "amp": 1 * np.exp(1j * np.pi)},
            )
            self.assertEqual(converter(20, instruction), valid_qobj)

    def test_drag_pulse_instruction(self):
        """Test that parametric pulses are correctly converted to PulseQobjInstructions."""
        amp = 0.7
        angle = -0.6
        with self.assertWarns(DeprecationWarning):
            converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
            instruction = Play(
                Drag(duration=25, sigma=15, amp=amp, angle=angle, beta=0.5), DriveChannel(0)
            )

        with self.assertWarns(DeprecationWarning):
            valid_qobj = PulseQobjInstruction(
                name="parametric_pulse",
                pulse_shape="drag",
                ch="d0",
                t0=30,
                parameters={
                    "duration": 25,
                    "sigma": 15,
                    "amp": amp * np.exp(1j * angle),
                    "beta": 0.5,
                },
            )
            self.assertEqual(converter(30, instruction), valid_qobj)

    def test_frame_change(self):
        """Test converted qobj from ShiftPhase."""
        with self.assertWarns(DeprecationWarning):
            converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
            valid_qobj = PulseQobjInstruction(name="fc", ch="d0", t0=0, phase=0.1)
            instruction = ShiftPhase(0.1, DriveChannel(0))
            self.assertEqual(converter(0, instruction), valid_qobj)

    def test_set_phase(self):
        """Test converted qobj from ShiftPhase."""
        with self.assertWarns(DeprecationWarning):
            converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
            instruction = SetPhase(3.14, DriveChannel(0))

            valid_qobj = PulseQobjInstruction(name="setp", ch="d0", t0=0, phase=3.14)

            self.assertEqual(converter(0, instruction), valid_qobj)

    def test_set_frequency(self):
        """Test converted qobj from SetFrequency."""
        with self.assertWarns(DeprecationWarning):
            converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
            instruction = SetFrequency(8.0e9, DriveChannel(0))

            valid_qobj = PulseQobjInstruction(name="setf", ch="d0", t0=0, frequency=8.0)

            self.assertEqual(converter(0, instruction), valid_qobj)

    def test_shift_frequency(self):
        """Test converted qobj from ShiftFrequency."""
        with self.assertWarns(DeprecationWarning):
            converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
            instruction = ShiftFrequency(8.0e9, DriveChannel(0))

            valid_qobj = PulseQobjInstruction(name="shiftf", ch="d0", t0=0, frequency=8.0)

            self.assertEqual(converter(0, instruction), valid_qobj)

    def test_acquire(self):
        """Test converted qobj from AcquireInstruction."""
        with self.assertWarns(DeprecationWarning):
            converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
            instruction = Acquire(10, AcquireChannel(0), MemorySlot(0), RegisterSlot(0))

            valid_qobj = PulseQobjInstruction(
                name="acquire", t0=0, duration=10, qubits=[0], memory_slot=[0], register_slot=[0]
            )
            self.assertEqual(converter(0, instruction), valid_qobj)

            # without register
            instruction = Acquire(10, AcquireChannel(0), MemorySlot(0))
            valid_qobj = PulseQobjInstruction(
                name="acquire", t0=0, duration=10, qubits=[0], memory_slot=[0]
            )
            self.assertEqual(converter(0, instruction), valid_qobj)

    def test_snapshot(self):
        """Test converted qobj from Snapshot."""
        with self.assertWarns(DeprecationWarning):
            converter = InstructionToQobjConverter(PulseQobjInstruction, meas_level=2)
            instruction = Snapshot(label="label", snapshot_type="type")

        with self.assertWarns(DeprecationWarning):
            valid_qobj = PulseQobjInstruction(name="snapshot", t0=0, label="label", type="type")

            self.assertEqual(converter(0, instruction), valid_qobj)


class TestQobjToInstructionConverter(QiskitTestCase):
    """Pulse converter tests."""

    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            self.linear = Waveform(np.arange(0, 0.01), name="linear")
            self.pulse_library = [
                PulseLibraryItem(name=self.linear.name, samples=self.linear.samples.tolist())
            ]

        with self.assertWarns(DeprecationWarning):
            self.converter = QobjToInstructionConverter(self.pulse_library, buffer=0)
        self.num_qubits = 2

    def test_drive_instruction(self):
        """Test converted qobj from PulseInstruction."""
        with self.assertWarns(DeprecationWarning):
            instruction = Play(self.linear, DriveChannel(0))
            qobj = PulseQobjInstruction(name="linear", ch="d0", t0=10)
            converted_instruction = self.converter(qobj)
            self.assertEqual(converted_instruction.instructions[0][-1], instruction)

    def test_parametric_pulses(self):
        """Test converted qobj from ParametricInstruction."""
        with self.assertWarns(DeprecationWarning):
            instruction = Play(
                Gaussian(duration=25, sigma=15, amp=0.5, angle=np.pi / 2, name="pulse1"),
                DriveChannel(0),
            )
            qobj = PulseQobjInstruction(
                name="parametric_pulse",
                label="pulse1",
                pulse_shape="gaussian",
                ch="d0",
                t0=0,
                parameters={"duration": 25, "sigma": 15, "amp": 0.5j},
            )
            converted_instruction = self.converter(qobj)
        self.assertEqual(converted_instruction.start_time, 0)
        self.assertEqual(converted_instruction.duration, 25)
        self.assertAlmostEqual(converted_instruction.instructions[0][-1], instruction)
        self.assertEqual(converted_instruction.instructions[0][-1].pulse.name, "pulse1")

    def test_parametric_pulses_no_label(self):
        """Test converted qobj from ParametricInstruction without label."""
        base_str = "gaussian_[('amp', (-0.5+0.2j)), ('duration', 25), ('sigma', 15)]"
        short_pulse_id = hashlib.md5(base_str.encode("utf-8")).hexdigest()[:4]
        pulse_name = f"gaussian_{short_pulse_id}"

        with self.assertWarns(DeprecationWarning):
            qobj = PulseQobjInstruction(
                name="parametric_pulse",
                pulse_shape="gaussian",
                ch="d0",
                t0=0,
                parameters={"duration": 25, "sigma": 15, "amp": -0.5 + 0.2j},
            )
            converted_instruction = self.converter(qobj)
        self.assertEqual(converted_instruction.instructions[0][-1].pulse.name, pulse_name)

    def test_frame_change(self):
        """Test converted qobj from ShiftPhase."""
        with self.assertWarns(DeprecationWarning):
            qobj = PulseQobjInstruction(name="fc", ch="m0", t0=0, phase=0.1)
            converted_instruction = self.converter(qobj)

            instruction = ShiftPhase(0.1, MeasureChannel(0))
        self.assertEqual(converted_instruction.start_time, 0)
        self.assertEqual(converted_instruction.duration, 0)
        self.assertEqual(converted_instruction.instructions[0][-1], instruction)

    def test_parameterized_frame_change(self):
        """Test converted qobj from ShiftPhase."""
        with self.assertWarns(DeprecationWarning):
            instruction = ShiftPhase(4.0, MeasureChannel(0))
            shifted = instruction << 10

            qobj = PulseQobjInstruction(name="fc", ch="m0", t0=10, phase="P1*2")
            converted_instruction = self.converter(qobj)

        self.assertIsInstance(converted_instruction, Schedule)

        bind_dict = {converted_instruction.get_parameters("P1")[0]: 2.0}
        evaluated_instruction = converted_instruction.assign_parameters(bind_dict)

        self.assertEqual(evaluated_instruction.start_time, shifted.start_time)
        self.assertEqual(evaluated_instruction.duration, shifted.duration)
        self.assertEqual(evaluated_instruction.instructions[0][-1], instruction)

    def test_set_phase(self):
        """Test converted qobj from SetPhase."""
        with self.assertWarns(DeprecationWarning):
            qobj = PulseQobjInstruction(name="setp", ch="m0", t0=0, phase=3.14)
            converted_instruction = self.converter(qobj)

            instruction = SetPhase(3.14, MeasureChannel(0))
        self.assertEqual(converted_instruction.start_time, 0)
        self.assertEqual(converted_instruction.duration, 0)
        self.assertEqual(converted_instruction.instructions[0][-1], instruction)

    def test_parameterized_set_phase(self):
        """Test converted qobj from SetPhase, with parameterized phase."""
        with self.assertWarns(DeprecationWarning):
            qobj = PulseQobjInstruction(name="setp", ch="m0", t0=0, phase="p/2")
            converted_instruction = self.converter(qobj)
        self.assertIsInstance(converted_instruction, Schedule)

        bind_dict = {converted_instruction.get_parameters("p")[0]: 3.14}
        evaluated_instruction = converted_instruction.assign_parameters(bind_dict)

        with self.assertWarns(DeprecationWarning):
            instruction = SetPhase(3.14 / 2, MeasureChannel(0))
        self.assertEqual(evaluated_instruction.start_time, 0)
        self.assertEqual(evaluated_instruction.duration, 0)
        self.assertEqual(evaluated_instruction.instructions[0][-1], instruction)

    def test_set_frequency(self):
        """Test converted qobj from SetFrequency."""
        with self.assertWarns(DeprecationWarning):
            instruction = SetFrequency(8.0e9, DriveChannel(0))

            qobj = PulseQobjInstruction(name="setf", ch="d0", t0=0, frequency=8.0)
            converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.start_time, 0)
        self.assertEqual(converted_instruction.duration, 0)
        self.assertEqual(converted_instruction.instructions[0][-1], instruction)
        self.assertTrue("frequency" in qobj.to_dict())

    def test_parameterized_set_frequency(self):
        """Test converted qobj from SetFrequency, when passing a parameterized frequency."""
        with self.assertWarns(DeprecationWarning):
            qobj = PulseQobjInstruction(name="setf", ch="d0", t0=2, frequency="f")
        self.assertTrue("frequency" in qobj.to_dict())

        with self.assertWarns(DeprecationWarning):
            converted_instruction = self.converter(qobj)
        self.assertIsInstance(converted_instruction, Schedule)

        bind_dict = {converted_instruction.get_parameters("f")[0]: 2.0}
        evaluated_instruction = converted_instruction.assign_parameters(bind_dict)

        with self.assertWarns(DeprecationWarning):
            instruction = SetFrequency(2.0e9, DriveChannel(0))

        self.assertEqual(evaluated_instruction.start_time, 2)
        self.assertEqual(evaluated_instruction.duration, 2)
        self.assertEqual(evaluated_instruction.instructions[0][-1], instruction)

    def test_shift_frequency(self):
        """Test converted qobj from ShiftFrequency."""
        with self.assertWarns(DeprecationWarning):
            instruction = ShiftFrequency(8.0e9, DriveChannel(0))

            qobj = PulseQobjInstruction(name="shiftf", ch="d0", t0=0, frequency=8.0)
            converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.start_time, 0)
        self.assertEqual(converted_instruction.duration, 0)
        self.assertEqual(converted_instruction.instructions[0][-1], instruction)
        self.assertTrue("frequency" in qobj.to_dict())

    def test_parameterized_shift_frequency(self):
        """Test converted qobj from ShiftFrequency, with a parameterized frequency."""
        with self.assertWarns(DeprecationWarning):
            qobj = PulseQobjInstruction(name="shiftf", ch="d0", t0=1, frequency="f / 1000")
        self.assertTrue("frequency" in qobj.to_dict())

        with self.assertWarns(DeprecationWarning):
            converted_instruction = self.converter(qobj)
        self.assertIsInstance(converted_instruction, Schedule)

        bind_dict = {converted_instruction.get_parameters("f")[0]: 3.14}
        evaluated_instruction = converted_instruction.assign_parameters(bind_dict)

        with self.assertWarns(DeprecationWarning):
            instruction = ShiftFrequency(3.14e6, DriveChannel(0))

        self.assertEqual(evaluated_instruction.start_time, 1)
        self.assertEqual(evaluated_instruction.duration, 1)
        self.assertEqual(evaluated_instruction.instructions[0][-1], instruction)

    def test_delay(self):
        """Test converted qobj from Delay."""
        with self.assertWarns(DeprecationWarning):
            instruction = Delay(10, DriveChannel(0))

            qobj = PulseQobjInstruction(name="delay", ch="d0", t0=0, duration=10)
            converted_instruction = self.converter(qobj)

        self.assertTrue("delay" in qobj.to_dict().values())
        self.assertEqual(converted_instruction.duration, instruction.duration)
        self.assertEqual(converted_instruction.instructions[0][-1], instruction)

    def test_acquire(self):
        """Test converted qobj from Acquire."""
        with self.assertWarns(DeprecationWarning):
            schedule = Schedule()
            for i in range(self.num_qubits):
                schedule |= Acquire(
                    10,
                    AcquireChannel(i),
                    MemorySlot(i),
                    RegisterSlot(i),
                    kernel=Kernel(name="test_kern", test_params="test"),
                    discriminator=Discriminator(name="test_disc", test_params=1.0),
                )
            qobj = PulseQobjInstruction(
                name="acquire",
                t0=0,
                duration=10,
                qubits=[0, 1],
                memory_slot=[0, 1],
                register_slot=[0, 1],
                kernels=[QobjMeasurementOption(name="test_kern", params={"test_params": "test"})],
                discriminators=[
                    QobjMeasurementOption(name="test_disc", params={"test_params": 1.0})
                ],
            )
        with self.assertWarns(DeprecationWarning):
            converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.start_time, 0)
        self.assertEqual(converted_instruction.duration, 10)
        self.assertEqual(converted_instruction.instructions[0][-1].duration, 10)
        self.assertEqual(
            converted_instruction.instructions[0][-1].kernel.params, {"test_params": "test"}
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(converted_instruction.instructions[1][-1].channel, AcquireChannel(1))

    def test_snapshot(self):
        """Test converted qobj from SnapShot."""
        with self.assertWarns(DeprecationWarning):
            instruction = Snapshot(label="label", snapshot_type="type")
            shifted = instruction << 10

            qobj = PulseQobjInstruction(name="snapshot", t0=10, label="label", type="type")
            converted_instruction = self.converter(qobj)

        self.assertEqual(converted_instruction.start_time, shifted.start_time)
        self.assertEqual(converted_instruction.duration, shifted.duration)
        self.assertEqual(converted_instruction.instructions[0][-1], instruction)

    def test_instruction_name_collision(self):
        """Avoid command name collision of pulse library items."""
        with self.assertWarns(DeprecationWarning):
            pulse_library_from_backend_x = [
                PulseLibraryItem(name="pulse123", samples=[0.1, 0.1, 0.1]),
                PulseLibraryItem(name="pulse456", samples=[0.3, 0.3, 0.3]),
            ]
            converter_of_backend_x = QobjToInstructionConverter(
                pulse_library_from_backend_x, buffer=0
            )

            pulse_library_from_backend_y = [
                PulseLibraryItem(name="pulse123", samples=[0.2, 0.2, 0.2])
            ]
            converter_of_backend_y = QobjToInstructionConverter(
                pulse_library_from_backend_y, buffer=0
            )

            qobj1 = PulseQobjInstruction(name="pulse123", qubits=[0], t0=0, ch="d0")
            qobj2 = PulseQobjInstruction(name="pulse456", qubits=[0], t0=0, ch="d0")

            sched_out_x = converter_of_backend_x(qobj1)
            sched_out_y = converter_of_backend_y(qobj1)

        # pulse123 have different definition on backend-x and backend-y
        self.assertNotEqual(sched_out_x, sched_out_y)

        with self.assertRaises(QiskitError):
            with self.assertWarns(DeprecationWarning):
                # This should not exist in backend-y command namespace.
                converter_of_backend_y(qobj2)


class TestLoConverter(QiskitTestCase):
    """LO converter tests."""

    def test_qubit_los(self):
        """Test qubit channel configuration."""
        with self.assertWarns(DeprecationWarning):
            user_lo_config = LoConfig({DriveChannel(0): 1.3e9})
            converter = LoConfigConverter(
                PulseQobjExperimentConfig, [1.2e9], [3.4e9], [(0.0, 5e9)], [(0.0, 5e9)]
            )
            valid_qobj = PulseQobjExperimentConfig(qubit_lo_freq=[1.3])

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(converter(user_lo_config), valid_qobj)

    def test_meas_los(self):
        """Test measurement channel configuration."""
        with self.assertWarns(DeprecationWarning):
            user_lo_config = LoConfig({MeasureChannel(0): 3.5e9})
            converter = LoConfigConverter(
                PulseQobjExperimentConfig, [1.2e9], [3.4e9], [(0.0, 5e9)], [(0.0, 5e9)]
            )

            valid_qobj = PulseQobjExperimentConfig(meas_lo_freq=[3.5])

            self.assertEqual(converter(user_lo_config), valid_qobj)

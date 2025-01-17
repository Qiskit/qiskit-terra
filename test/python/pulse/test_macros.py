# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for Pulse Macro functions."""

from qiskit.pulse import (
    Schedule,
    AcquireChannel,
    Acquire,
    InstructionScheduleMap,
    MeasureChannel,
    MemorySlot,
    GaussianSquare,
    Play,
)
from qiskit.pulse import macros
from qiskit.pulse.exceptions import PulseError
from qiskit.providers.fake_provider import FakeOpenPulse2Q, Fake27QPulseV1, GenericBackendV2
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestMeasure(QiskitTestCase):
    """Pulse measure macro."""

    @ignore_pulse_deprecation_warnings
    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            self.backend = FakeOpenPulse2Q()
            self.backend_v1 = Fake27QPulseV1()

        self.inst_map = self.backend.defaults().instruction_schedule_map
        with self.assertWarns(DeprecationWarning):
            self.backend_v2 = GenericBackendV2(
                num_qubits=27,
                calibrate_instructions=self.backend_v1.defaults().instruction_schedule_map,
                seed=42,
            )

    def test_measure(self):
        """Test macro - measure."""
        sched = macros.measure(qubits=[0], backend=self.backend)
        expected = Schedule(
            self.inst_map.get("measure", [0, 1]).filter(channels=[MeasureChannel(0)]),
            Acquire(10, AcquireChannel(0), MemorySlot(0)),
        )
        self.assertEqual(sched.instructions, expected.instructions)

    def test_measure_sched_with_qubit_mem_slots(self):
        """Test measure with custom qubit_mem_slots."""
        sched = macros.measure(qubits=[0], backend=self.backend, qubit_mem_slots={0: 1})
        expected = Schedule(
            self.inst_map.get("measure", [0, 1]).filter(channels=[MeasureChannel(0)]),
            Acquire(10, AcquireChannel(0), MemorySlot(1)),
        )
        self.assertEqual(sched.instructions, expected.instructions)

    def test_measure_sched_with_meas_map(self):
        """Test measure with custom meas_map as list and dict."""
        sched_with_meas_map_list = macros.measure(
            qubits=[0], backend=self.backend, meas_map=[[0, 1]]
        )
        sched_with_meas_map_dict = macros.measure(
            qubits=[0], backend=self.backend, meas_map={0: [0, 1], 1: [0, 1]}
        )
        expected = Schedule(
            self.inst_map.get("measure", [0, 1]).filter(channels=[MeasureChannel(0)]),
            Acquire(10, AcquireChannel(0), MemorySlot(0)),
        )
        self.assertEqual(sched_with_meas_map_list.instructions, expected.instructions)
        self.assertEqual(sched_with_meas_map_dict.instructions, expected.instructions)

    def test_measure_with_custom_inst_map(self):
        """Test measure with custom inst_map, meas_map with measure_name."""
        q0_sched = Play(GaussianSquare(1200, 1, 0.4, 1150), MeasureChannel(0))
        q0_sched += Acquire(1200, AcquireChannel(0), MemorySlot(0))
        inst_map = InstructionScheduleMap()
        inst_map.add("my_sched", 0, q0_sched)
        sched = macros.measure(
            qubits=[0], measure_name="my_sched", inst_map=inst_map, meas_map=[[0]]
        )
        self.assertEqual(sched.instructions, q0_sched.instructions)

        with self.assertRaises(PulseError):
            macros.measure(qubits=[0], measure_name="name", inst_map=inst_map, meas_map=[[0]])

    def test_fail_measure(self):
        """Test failing measure."""
        with self.assertRaises(PulseError):
            macros.measure(qubits=[0], meas_map=self.backend.configuration().meas_map)
        with self.assertRaises(PulseError):
            macros.measure(qubits=[0], inst_map=self.inst_map)

    def test_measure_v2(self):
        """Test macro - measure with backendV2."""
        sched = macros.measure(qubits=[0], backend=self.backend_v2)
        with self.assertWarns(DeprecationWarning):
            expected = self.backend_v2.target.get_calibration("measure", (0,)).filter(
                channels=[MeasureChannel(0), AcquireChannel(0)]
            )
        self.assertEqual(sched.instructions, expected.instructions)

    def test_measure_v2_sched_with_qubit_mem_slots(self):
        """Test measure with backendV2 and custom qubit_mem_slots."""
        sched = macros.measure(qubits=[0], backend=self.backend_v2, qubit_mem_slots={0: 2})
        with self.assertWarns(DeprecationWarning):
            expected = self.backend_v2.target.get_calibration("measure", (0,)).filter(
                channels=[
                    MeasureChannel(0),
                ]
            )
        measure_duration = expected.filter(instruction_types=[Play]).duration
        expected += Acquire(measure_duration, AcquireChannel(0), MemorySlot(2))
        self.assertEqual(sched.instructions, expected.instructions)

    def test_measure_v2_sched_with_meas_map(self):
        """Test measure with backendV2 custom meas_map as list and dict."""
        sched_with_meas_map_list = macros.measure(
            qubits=[0], backend=self.backend_v2, meas_map=[[0, 1]]
        )
        sched_with_meas_map_dict = macros.measure(
            qubits=[0], backend=self.backend_v2, meas_map={0: [0, 1], 1: [0, 1]}
        )
        with self.assertWarns(DeprecationWarning):
            expected = self.backend_v2.target.get_calibration("measure", (0,)).filter(
                channels=[
                    MeasureChannel(0),
                ]
            )
        measure_duration = expected.filter(instruction_types=[Play]).duration
        expected += Acquire(measure_duration, AcquireChannel(0), MemorySlot(0))
        self.assertEqual(sched_with_meas_map_list.instructions, expected.instructions)
        self.assertEqual(sched_with_meas_map_dict.instructions, expected.instructions)

    def test_multiple_measure_v2(self):
        """Test macro - multiple qubit measure with backendV2."""
        sched = macros.measure(qubits=[0, 1], backend=self.backend_v2)
        with self.assertWarns(DeprecationWarning):
            expected = self.backend_v2.target.get_calibration("measure", (0,)).filter(
                channels=[
                    MeasureChannel(0),
                ]
            )
            expected += self.backend_v2.target.get_calibration("measure", (1,)).filter(
                channels=[
                    MeasureChannel(1),
                ]
            )
        measure_duration = expected.filter(instruction_types=[Play]).duration
        expected += Acquire(measure_duration, AcquireChannel(0), MemorySlot(0))
        expected += Acquire(measure_duration, AcquireChannel(1), MemorySlot(1))
        self.assertEqual(sched.instructions, expected.instructions)

    def test_output_with_measure_v1_and_measure_v2(self):
        """Test make outputs of measure_v1 and measure_v2 consistent."""
        sched_measure_v1 = macros.measure(qubits=[0, 1], backend=self.backend_v1)
        sched_measure_v2 = macros.measure(qubits=[0, 1], backend=self.backend_v2)

        self.assertEqual(sched_measure_v1.instructions, sched_measure_v2.instructions)

    def test_output_with_measure_v1_and_measure_v2_sched_with_qubit_mem_slots(self):
        """Test make outputs of measure_v1 and measure_v2 with custom qubit_mem_slots consistent."""
        sched_measure_v1 = macros.measure(
            qubits=[0], backend=self.backend_v1, qubit_mem_slots={0: 2}
        )
        sched_measure_v2 = macros.measure(
            qubits=[0], backend=self.backend_v2, qubit_mem_slots={0: 2}
        )
        self.assertEqual(sched_measure_v1.instructions, sched_measure_v2.instructions)

    def test_output_with_measure_v1_and_measure_v2_sched_with_meas_map(self):
        """Test make outputs of measure_v1 and measure_v2
        with custom meas_map as list and dict consistent."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake27QPulseV1()
        num_qubits_list_measure_v1 = list(range(backend.configuration().num_qubits))
        num_qubits_list_measure_v2 = list(range(self.backend_v2.num_qubits))
        sched_with_meas_map_list_v1 = macros.measure(
            qubits=[0], backend=self.backend_v1, meas_map=[num_qubits_list_measure_v1]
        )
        sched_with_meas_map_dict_v1 = macros.measure(
            qubits=[0],
            backend=self.backend_v1,
            meas_map={0: num_qubits_list_measure_v1, 1: num_qubits_list_measure_v1},
        )
        sched_with_meas_map_list_v2 = macros.measure(
            qubits=[0], backend=self.backend_v2, meas_map=[num_qubits_list_measure_v2]
        )
        sched_with_meas_map_dict_v2 = macros.measure(
            qubits=[0],
            backend=self.backend_v2,
            meas_map={0: num_qubits_list_measure_v2, 1: num_qubits_list_measure_v2},
        )
        self.assertEqual(
            sched_with_meas_map_list_v1.instructions,
            sched_with_meas_map_list_v2.instructions,
        )
        self.assertEqual(
            sched_with_meas_map_dict_v1.instructions,
            sched_with_meas_map_dict_v2.instructions,
        )

    def test_output_with_multiple_measure_v1_and_measure_v2(self):
        """Test macro - consistent output of multiple qubit measure with backendV1 and backendV2."""
        sched_measure_v1 = macros.measure(qubits=[0, 1], backend=self.backend_v1)
        sched_measure_v2 = macros.measure(qubits=[0, 1], backend=self.backend_v2)
        self.assertEqual(sched_measure_v1.instructions, sched_measure_v2.instructions)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestMeasureAll(QiskitTestCase):
    """Pulse measure all macro."""

    @ignore_pulse_deprecation_warnings
    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            self.backend_v1 = FakeOpenPulse2Q()
        self.inst_map = self.backend_v1.defaults().instruction_schedule_map
        with self.assertWarns(DeprecationWarning):
            self.backend_v2 = GenericBackendV2(
                num_qubits=2,
                calibrate_instructions=self.backend_v1.defaults().instruction_schedule_map,
                seed=42,
            )

    def test_measure_all(self):
        """Test measure_all function."""
        sched = macros.measure_all(self.backend_v1)
        expected = Schedule(self.inst_map.get("measure", [0, 1]))
        self.assertEqual(sched.instructions, expected.instructions)

    def test_measure_all_v2(self):
        """Test measure_all function with backendV2."""
        sched = macros.measure_all(self.backend_v1)
        expected = Schedule(
            self.inst_map.get("measure", list(range(self.backend_v1.configuration().num_qubits)))
        )
        self.assertEqual(sched.instructions, expected.instructions)

    def test_output_of_measure_all_with_backend_v1_and_v2(self):
        """Test make outputs of measure_all with backendV1 and backendV2 consistent."""
        sched_measure_v1 = macros.measure_all(backend=self.backend_v1)
        sched_measure_v2 = macros.measure_all(backend=self.backend_v2)
        self.assertEqual(sched_measure_v1.instructions, sched_measure_v2.instructions)

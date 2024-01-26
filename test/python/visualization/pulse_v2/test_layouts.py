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
from qiskit.test import QiskitTestCase
from qiskit.visualization.pulse_v2 import layouts, device_info
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers import BackendPropertyError, BackendConfigurationError


class TestChannelArrangement(QiskitTestCase):
    """Tests for channel mapping functions."""

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


class TestFigureTitle(QiskitTestCase):
    """Tests for figure title generation."""

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


class TestPulseCreateFromBackendV1(QiskitTestCase):
    """Tests for OpenPulse create_from_backend using BackendV1 as input"""

    def setUp(self) -> None:
        super().setUp()

        class BaseConfiguration:
            """A class for store the backendV1 configuration."""

            def __init__(self):
                pass

        class BaseBackendV1(BackendV1):
            """A backed based on BackendV1."""

            def __init__(self, configuration):
                super().__init__(configuration)

            @classmethod
            def _default_options(cls):
                pass

            def run(self, run_input, **options):
                pass

        class BackendV1WithDefaults(BaseBackendV1):
            """A backend based on BackendV1 with defaults."""

            def __init__(self, configuration, defaults):
                super().__init__(configuration)
                self._defaults = defaults

            def defaults(self):
                """Get backend defaults."""
                return self._defaults

        class Defaults:
            """A class to store the defaults data for BackendV1."""

            def __init__(self):
                pass

        self.configuration = BaseConfiguration()
        self.defaults = Defaults()

        self.backend_v1 = BaseBackendV1(self.configuration)
        self.backend_v1_with_defaults = BackendV1WithDefaults(self.configuration, self.defaults)

        self.backend_info = device_info.OpenPulseBackendInfo()

    def test_backend_v1_adapter_invalid_defaults(self):
        """Test if an error is raised when a BackendV1 without defaults is passed."""
        with self.assertRaises(BackendPropertyError):
            self.backend_info.backend_v1_adapter(self.backend_v1)

    def test_backend_v1_adapter_invalid_configuration_name(self):
        """Test if an error is raised when a BackendV1 without name is passed."""
        with self.assertRaises(BackendConfigurationError):
            self.backend_info.backend_v1_adapter(self.backend_v1_with_defaults)

    def test_backend_v1_adapter_invalid_configuration_n_qubits(self):
        """Test if an error is raised when a BackendV1 without n_qubits is passed."""
        self.configuration.backend_name = "Dummy"  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendConfigurationError):
            self.backend_info.backend_v1_adapter(self.backend_v1_with_defaults)

    def test_backend_v1_adapter_invalid_configuration_u_channel_lo(self):
        """Test if an error is raised when a BackendV1 without u_channel_lo is passed."""
        self.configuration.backend_name = "Dummy"  # pylint: disable=attribute-defined-outside-init
        self.configuration.n_qubits = 1  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendConfigurationError):
            self.backend_info.backend_v1_adapter(self.backend_v1_with_defaults)

    def test_backend_v1_adapter_invalid_configuration_drive_channel(self):
        """Test if an error is raised when a BackendV1 without drive is passed."""
        self.configuration.backend_name = "Dummy"  # pylint: disable=attribute-defined-outside-init
        self.configuration.n_qubits = 1  # pylint: disable=attribute-defined-outside-init
        self.configuration.u_channel_lo = None  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendConfigurationError):
            self.backend_info.backend_v1_adapter(self.backend_v1_with_defaults)

    def test_backend_v1_adapter_invalid_configuration_measure_channel(self):
        """Test if an error is raised when a BackendV1 without a measure is passed."""
        self.configuration.backend_name = "Dummy"  # pylint: disable=attribute-defined-outside-init
        self.configuration.n_qubits = 1  # pylint: disable=attribute-defined-outside-init
        self.configuration.u_channel_lo = None  # pylint: disable=attribute-defined-outside-init
        self.configuration.drive = lambda: None  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendConfigurationError):
            self.backend_info.backend_v1_adapter(self.backend_v1_with_defaults)

    def test_backend_v1_adapter_invalid_configuration_control_channel(self):
        """Test if an error is raised when a BackendV1 without a control is passed."""
        self.configuration.backend_name = "Dummy"  # pylint: disable=attribute-defined-outside-init
        self.configuration.n_qubits = 1  # pylint: disable=attribute-defined-outside-init
        self.configuration.u_channel_lo = None  # pylint: disable=attribute-defined-outside-init
        self.configuration.drive = lambda: None  # pylint: disable=attribute-defined-outside-init
        self.configuration.measure = lambda: None  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendConfigurationError):
            self.backend_info.backend_v1_adapter(self.backend_v1_with_defaults)

    def test_backend_v1_adapter_invalid_configuration_dt(self):
        """Test if an error is raised when a BackendV1 without dt is passed."""
        self.configuration.backend_name = "Dummy"  # pylint: disable=attribute-defined-outside-init
        self.configuration.n_qubits = 1  # pylint: disable=attribute-defined-outside-init
        self.configuration.u_channel_lo = None  # pylint: disable=attribute-defined-outside-init
        self.configuration.drive = lambda: None  # pylint: disable=attribute-defined-outside-init
        self.configuration.measure = lambda: None  # pylint: disable=attribute-defined-outside-init
        self.configuration.control = lambda: None  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendConfigurationError):
            self.backend_info.backend_v1_adapter(self.backend_v1_with_defaults)

    def test_backend_v1_adapter_invalid_defaults_qubit_freq_est(self):
        """Test if an error is raised when a BackendV1 without qubit_freq_est is passed."""
        self.configuration.backend_name = "Dummy"  # pylint: disable=attribute-defined-outside-init
        self.configuration.n_qubits = 1  # pylint: disable=attribute-defined-outside-init
        self.configuration.u_channel_lo = None  # pylint: disable=attribute-defined-outside-init
        self.configuration.drive = lambda: None  # pylint: disable=attribute-defined-outside-init
        self.configuration.measure = lambda: None  # pylint: disable=attribute-defined-outside-init
        self.configuration.control = lambda: None  # pylint: disable=attribute-defined-outside-init
        self.configuration.dt = None  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendConfigurationError):
            self.backend_info.backend_v1_adapter(self.backend_v1_with_defaults)

    def test_backend_v1_adapter_invalid_defaults_meas_freq_est(self):
        """Test if an error is raised when a BackendV1 without meas_freq_est is passed."""
        self.configuration.backend_name = "Dummy"  # pylint: disable=attribute-defined-outside-init
        self.configuration.n_qubits = 1  # pylint: disable=attribute-defined-outside-init
        self.configuration.u_channel_lo = None  # pylint: disable=attribute-defined-outside-init
        self.configuration.drive = lambda: None  # pylint: disable=attribute-defined-outside-init
        self.configuration.measure = lambda: None  # pylint: disable=attribute-defined-outside-init
        self.configuration.control = lambda: None  # pylint: disable=attribute-defined-outside-init
        self.configuration.dt = None  # pylint: disable=attribute-defined-outside-init
        self.defaults.qubit_freq_est = []  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendConfigurationError):
            self.backend_info.backend_v1_adapter(self.backend_v1_with_defaults)

    def test_valid_backend_v1(self):
        """Test backendV1 adapter passing all valid values."""
        self.configuration.backend_name = "Dummy"  # pylint: disable=attribute-defined-outside-init
        self.configuration.n_qubits = 1  # pylint: disable=attribute-defined-outside-init
        self.configuration.u_channel_lo = None  # pylint: disable=attribute-defined-outside-init
        self.configuration.drive = lambda: None  # pylint: disable=attribute-defined-outside-init
        self.configuration.measure = lambda: None  # pylint: disable=attribute-defined-outside-init
        self.configuration.control = lambda: None  # pylint: disable=attribute-defined-outside-init
        self.configuration.dt = None  # pylint: disable=attribute-defined-outside-init
        self.defaults.qubit_freq_est = []  # pylint: disable=attribute-defined-outside-init
        self.defaults.meas_freq_est = []  # pylint: disable=attribute-defined-outside-init

        name, configuration, dt, defaults = self.backend_info.backend_v1_adapter(
            self.backend_v1_with_defaults
        )
        self.assertEqual(name, "Dummy")
        self.assertEqual(configuration.n_qubits, 1)
        self.assertIsNone(configuration.u_channel_lo)
        self.assertIsNone(dt)
        self.assertTrue(configuration.drive is not None)
        self.assertTrue(configuration.measure is not None)
        self.assertTrue(configuration.control is not None)
        self.assertEqual(defaults.qubit_freq_est, [])
        self.assertEqual(defaults.meas_freq_est, [])


class TestPulseCreateFromBackendV2(QiskitTestCase):
    """Tests for OpenPulse create_from_backend using BackendV2 as input"""

    def setUp(self) -> None:
        super().setUp()

        class BaseBackendV2(BackendV2):
            """A backend Based on BackendV2."""

            def __init__(self):
                super().__init__()
                self._target = None

            @classmethod
            def _default_options(cls):
                pass

            @property
            def max_circuits(self):
                pass

            def run(self, run_input, **options):
                pass

            @property
            def target(self):
                return self._target

        class BackendV2WithDefaults(BaseBackendV2):
            """A backend based on BackendV2 with defaults."""

            def __init__(self, defaults, target):
                super().__init__()
                self._defaults = defaults
                self._target = target

            def defaults(self):
                """Get backend defaults."""
                return self._defaults

        class Defaults:
            """A class to store the defaults data for BackendV2."""

            def __init__(self):
                pass

        class Target:
            """A class to store the Target data for BackendV2."""

            def __init__(self):
                pass

        self.defaults = Defaults()
        self.target = Target()

        self.backend_v2 = BaseBackendV2()
        self.backend_v2_with_defaults = BackendV2WithDefaults(self.defaults, self.target)

        self.backend_info = device_info.OpenPulseBackendInfo()

    def test_backend_v2_adapter_invalid_name(self):
        """Test if an error is raised when a BackendV2 without name is passed."""
        with self.assertRaises(BackendPropertyError):
            self.backend_info.backend_v2_adapter(self.backend_v2)

    def test_backend_v2_adapter_invalid_defaults(self):
        """Test if an error is raised when a BackendV2 without defaults is passed."""
        self.backend_v2.name = "Dummy"  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendPropertyError):
            self.backend_info.backend_v2_adapter(self.backend_v2)

    def test_backend_v2_adapter_invalid_dt(self):
        """Test if an error is raised when a BackendV2 without dt is passed."""
        self.backend_v2_with_defaults.name = (
            "Dummy"  # pylint: disable=attribute-defined-outside-init
        )
        with self.assertRaises(BackendPropertyError):
            self.backend_info.backend_v2_adapter(self.backend_v2_with_defaults)

    def test_backend_v2_adapter_invalid_num_qubits(self):
        """Test if an error is raised when a BackendV2 without num_qubits is passed."""
        self.backend_v2_with_defaults.name = (
            "Dummy"  # pylint: disable=attribute-defined-outside-init
        )
        self.target.dt = None  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendPropertyError):
            self.backend_info.backend_v2_adapter(self.backend_v2_with_defaults)

    def test_backend_v2_adapter_invalid_u_channel_lo(self):
        """Test if an error is raised when a BackendV2 without u_channel_lo is passed."""
        self.backend_v2_with_defaults.name = (
            "Dummy"  # pylint: disable=attribute-defined-outside-init
        )
        self.target.dt = None  # pylint: disable=attribute-defined-outside-init
        self.target.num_qubits = 1  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendPropertyError):
            self.backend_info.backend_v2_adapter(self.backend_v2_with_defaults)

    def test_backend_v2_adapter_invalid_defaults_no_qubit_freq_est(self):
        """Test if an error is raised when a BackendV2 without qubit_freq_est is passed."""
        self.backend_v2_with_defaults.name = (
            "Dummy"  # pylint: disable=attribute-defined-outside-init
        )
        # pylint: disable=attribute-defined-outside-init
        self.backend_v2_with_defaults.u_channel_lo = None
        self.target.dt = None  # pylint: disable=attribute-defined-outside-init
        self.target.num_qubits = 1  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendConfigurationError):
            self.backend_info.backend_v2_adapter(self.backend_v2_with_defaults)

    def test_backend_v2_adapter_invalid_defaults_no_qubit_freq_est_u_channel_lo_target(self):
        """Test if an error is raised when a BackendV2 without qubit_freq_est is passed."""
        self.backend_v2_with_defaults.name = (
            "Dummy"  # pylint: disable=attribute-defined-outside-init
        )
        self.target.u_channel_lo = None  # pylint: disable=attribute-defined-outside-init
        self.target.dt = None  # pylint: disable=attribute-defined-outside-init
        self.target.num_qubits = 1  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendConfigurationError):
            self.backend_info.backend_v2_adapter(self.backend_v2_with_defaults)

    def test_backend_v2_adapter_invalid_defaults_no_meas_freq_est(self):
        """Test if an error is raised when a BackendV2 without meas_freq_est is passed."""
        self.backend_v2_with_defaults.name = (
            "Dummy"  # pylint: disable=attribute-defined-outside-init
        )
        self.target.u_channel_lo = None  # pylint: disable=attribute-defined-outside-init
        self.target.dt = None  # pylint: disable=attribute-defined-outside-init
        self.target.num_qubits = 1  # pylint: disable=attribute-defined-outside-init
        self.defaults.qubit_freq_est = []  # pylint: disable=attribute-defined-outside-init
        with self.assertRaises(BackendConfigurationError):
            self.backend_info.backend_v2_adapter(self.backend_v2_with_defaults)

    def test_valid_backend_v2(self):
        """Test backendV2 adapter passing all valid values."""
        self.backend_v2_with_defaults.name = (
            "Dummy"  # pylint: disable=attribute-defined-outside-init
        )
        self.target.u_channel_lo = None  # pylint: disable=attribute-defined-outside-init
        self.target.dt = None  # pylint: disable=attribute-defined-outside-init
        self.target.num_qubits = 1  # pylint: disable=attribute-defined-outside-init
        self.defaults.qubit_freq_est = []  # pylint: disable=attribute-defined-outside-init
        self.defaults.meas_freq_est = []  # pylint: disable=attribute-defined-outside-init

        name, configuration, dt, defaults = self.backend_info.backend_v2_adapter(
            self.backend_v2_with_defaults
        )
        self.assertEqual(name, "Dummy")
        self.assertEqual(configuration.n_qubits, 1)
        self.assertTrue(configuration.measure is not None)
        self.assertTrue(configuration.drive is not None)
        self.assertTrue(configuration.control is not None)
        self.assertIsNone(configuration.u_channel_lo)
        self.assertIsNone(dt)
        self.assertEqual(defaults.qubit_freq_est, [])
        self.assertEqual(defaults.meas_freq_est, [])


class TestPulseCreateFrom(QiskitTestCase):
    """Tests for OpenPulse create_from_backend"""

    def setUp(self) -> None:
        super().setUp()

        class Defaults:
            """A class to store the defaults data for Backend (v1 and V2)."""

            qubit_freq_est = []
            meas_freq_est = []

        self.defaults = Defaults()
        self.backend_info = device_info.OpenPulseBackendInfo()

    def test_raise_attribute_doesnt_exist_fail(self):
        """Test if it raises the Expection."""
        with self.assertRaises(BackendPropertyError):
            self.backend_info.raise_attribute_doesnt_exist(
                ["a", "b", "d"], ["c", "d"], BackendPropertyError
            )

    def test_raise_attribute_doesnt_exist_success(self):
        """Test if it checks the attributes correctly."""
        try:
            self.backend_info.raise_attribute_doesnt_exist(
                ["a", "b", "c", "d"], ["a", "b"], BackendPropertyError
            )
        except BackendPropertyError:
            self.fail(
                "raise_attribute_doesnt_exist should raise no expections with equal sets of attributes!"
            )

    def test_get_backend_adapter_backend_v1(self):
        """Test if it gets the correct adapter for BackendV1."""

        class DummyBackendV1:
            """A class to simulate a BackendV1."""

            version = 1

        adapter = self.backend_info.get_backend_adapter(DummyBackendV1)
        self.assertEqual(adapter.__name__, "backend_v1_adapter")

    def test_get_backend_adapter_backend_v2(self):
        """Test if it gets the correct adapter for BackendV2."""

        class DummyBackendV2:
            """A class to simulate a BackendV2."""

            version = 2

        adapter = self.backend_info.get_backend_adapter(DummyBackendV2)
        self.assertEqual(adapter.__name__, "backend_v2_adapter")

    def test_get_backend_adapter_backend_for_a_smaller_version(self):
        """Test if it raises an Exception for versions smaller than 1."""

        class DummyBackendVm:
            """A class to simulate a backend version smaller than 1."""

            version = -10

        with self.assertRaises(BackendPropertyError):
            self.backend_info.get_backend_adapter(DummyBackendVm)

    def test_get_backend_adapter_backend_for_a_greater_version(self):
        """Test if it raises an Exception for versions greater than 2."""

        class DummyBackendVg:
            """A class to simulate a backend version greater than 2."""

            version = 5

        with self.assertRaises(BackendPropertyError):
            self.backend_info.get_backend_adapter(DummyBackendVg)

    def test_create_from_backend_using_backend_v1(self):
        """Test if create_from_backend raises no error for BackendV1."""

        class Configuration:
            """All the configurations for BackendV1."""

            backend_name = "V1"
            u_channel_lo = []
            drive = lambda: None
            measure = lambda: None
            control = lambda: None
            dt = None
            n_qubits = 0

        class Backend(BackendV1):
            """A backend based on BackendV1 with all the required data."""

            def __init__(self, configuration=Configuration(), defaults=self.defaults):
                super().__init__(configuration)
                self._defaults = defaults

            @classmethod
            def _default_options(cls):
                pass

            def run(self, run_input, **options):
                """Required run method."""
                pass

            def defaults(self):
                """Get backend Defaults"""
                return self._defaults

        try:
            self.backend_info.create_from_backend(Backend())
        except Exception:  # pylint: disable=broad-exception-caught
            self.fail("The create_from_backend method should accept a V1 backend")

    def test_create_from_backend_using_backend_v2(self):
        """Test if create_from_backend raises no error for BackendV2."""

        class Backend(BackendV2):
            """A backend based on BackendV2 with all the required data."""

            def __init__(self, defaults=self.defaults):
                super().__init__(name="V2")
                self._defaults = defaults

            @classmethod
            def _default_options(cls):
                pass

            @property
            def max_circuits(self):
                """Required method for BackendV2."""
                pass

            def run(self, run_input, **options):
                pass

            def defaults(self):
                """Get Backend defaults."""
                return self._defaults

            @property
            def target(self):
                pass

            def measure_channel(self, qubit):
                pass

            def drive_channel(self, qubit):
                pass

            @property
            def dt(self):
                return None

            @property
            def num_qubits(self):
                return 0

            @property
            def u_channel_lo(self):
                """Get u_channel_lo attribute"""
                return []

        try:
            self.backend_info.create_from_backend(Backend())
        except Exception:  # pylint: disable=broad-exception-caught
            self.fail("The create_from_backend method should accept a V2 backend")

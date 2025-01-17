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
"""
Test that the PulseBackendConfiguration methods work as expected with a mocked Pulse backend.
"""
# TODO the full file can be removed once BackendV1 is removed, since it is the
#  only one with backend.configuration()

import collections
import copy

from qiskit.providers.fake_provider import FakeOpenPulse2Q, FakeOpenPulse3Q, Fake27QPulseV1
from qiskit.pulse.channels import DriveChannel, MeasureChannel, ControlChannel, AcquireChannel
from qiskit.providers import BackendConfigurationError
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestBackendConfiguration(QiskitTestCase):
    """Test the methods on the BackendConfiguration class."""

    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            backend = FakeOpenPulse2Q()
        self.config = backend.configuration()

    def test_simple_config(self):
        """Test the most basic getters."""
        self.assertEqual(self.config.dt, 1.3333 * 1.0e-9)
        self.assertEqual(self.config.dtm, 10.5 * 1.0e-9)
        self.assertEqual(self.config.basis_gates, ["u1", "u2", "u3", "cx", "id"])

    def test_sample_rate(self):
        """Test that sample rate is 1/dt."""
        self.assertEqual(self.config.sample_rate, 1.0 / self.config.dt)

    def test_hamiltonian(self):
        """Test the hamiltonian method."""
        self.assertEqual(
            self.config.hamiltonian["description"],
            "A hamiltonian for a mocked 2Q device, with 1Q and 2Q terms.",
        )
        ref_vars = {
            "v0": 5.0 * 1e9,
            "v1": 5.1 * 1e9,
            "j": 0.01 * 1e9,
            "r": 0.02 * 1e9,
            "alpha0": -0.33 * 1e9,
            "alpha1": -0.33 * 1e9,
        }
        self.assertEqual(self.config.hamiltonian["vars"], ref_vars)
        # Test that on serialization inverse conversion is done.
        self.assertEqual(
            self.config.to_dict()["hamiltonian"]["vars"],
            {k: var * 1e-9 for k, var in ref_vars.items()},
        )
        # 3Q doesn't offer a hamiltonian -- test that we get a reasonable response
        with self.assertWarns(DeprecationWarning):
            backend_3q = FakeOpenPulse3Q()
        self.assertEqual(backend_3q.configuration().hamiltonian, None)

    def test_get_channels(self):
        """Test requesting channels from the system."""

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(self.config.drive(0), DriveChannel(0))
            self.assertEqual(self.config.measure(1), MeasureChannel(1))
            self.assertEqual(self.config.acquire(0), AcquireChannel(0))
        with self.assertRaises(BackendConfigurationError):
            # Check that an error is raised if the system doesn't have that many qubits
            self.assertEqual(self.config.acquire(10), AcquireChannel(10))
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(self.config.control(qubits=[0, 1]), [ControlChannel(0)])
        with self.assertRaises(BackendConfigurationError):
            # Check that an error is raised if key not found in self._qubit_channel_map
            self.config.control(qubits=(10, 1))

    def test_get_channel_qubits(self):
        """Test to get all qubits operated on a given channel."""
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(self.config.get_channel_qubits(channel=DriveChannel(0)), [0])
            self.assertEqual(self.config.get_channel_qubits(channel=ControlChannel(0)), [0, 1])
        with self.assertWarns(DeprecationWarning):
            backend_3q = FakeOpenPulse3Q()
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(
                backend_3q.configuration().get_channel_qubits(ControlChannel(2)), [2, 1]
            )
            self.assertEqual(
                backend_3q.configuration().get_channel_qubits(ControlChannel(1)), [1, 0]
            )
        with self.assertRaises(BackendConfigurationError):
            with self.assertWarns(DeprecationWarning):
                # Check that an error is raised if key not found in self._channel_qubit_map
                self.config.get_channel_qubits(MeasureChannel(10))

    def test_get_qubit_channels(self):
        """Test to get all channels operated on a given qubit."""
        with self.assertWarns(DeprecationWarning):
            self.assertTrue(
                self._test_lists_equal(
                    actual=self.config.get_qubit_channels(qubit=(1,)),
                    expected=[DriveChannel(1), MeasureChannel(1), AcquireChannel(1)],
                )
            )
        with self.assertWarns(DeprecationWarning):
            self.assertTrue(
                self._test_lists_equal(
                    actual=self.config.get_qubit_channels(qubit=1),
                    expected=[
                        ControlChannel(0),
                        ControlChannel(1),
                        AcquireChannel(1),
                        DriveChannel(1),
                        MeasureChannel(1),
                    ],
                )
            )
        with self.assertWarns(DeprecationWarning):
            backend_3q = FakeOpenPulse3Q()
            self.assertTrue(
                self._test_lists_equal(
                    actual=backend_3q.configuration().get_qubit_channels(1),
                    expected=[
                        MeasureChannel(1),
                        ControlChannel(0),
                        ControlChannel(2),
                        AcquireChannel(1),
                        DriveChannel(1),
                        ControlChannel(1),
                    ],
                )
            )
        with self.assertRaises(BackendConfigurationError):
            # Check that an error is raised if key not found in self._channel_qubit_map
            self.config.get_qubit_channels(10)

    def test_supported_instructions(self):
        """Test that supported instructions get entered into config dict properly."""
        # verify the supported instructions is not in the config dict when the flag is not set
        self.assertNotIn("supported_instructions", self.config.to_dict())
        # verify that supported instructions get added to config dict when set
        supp_instrs = ["u1", "u2", "play", "acquire"]
        setattr(self.config, "supported_instructions", supp_instrs)
        self.assertEqual(supp_instrs, self.config.to_dict()["supported_instructions"])

    def test_get_rep_times(self):
        """Test whether rep time property is the right size"""
        _rep_times_us = [100, 250, 500, 1000]
        _rep_times_s = [_rt * 1.0e-6 for _rt in _rep_times_us]

        for i, time in enumerate(_rep_times_s):
            self.assertAlmostEqual(self.config.rep_times[i], time)
        for i, time in enumerate(_rep_times_us):
            self.assertEqual(round(self.config.rep_times[i] * 1e6), time)
        for rep_time in self.config.to_dict()["rep_times"]:
            self.assertGreater(rep_time, 0)

    def test_get_default_rep_delay_and_range(self):
        """Test whether rep delay property is the right size."""
        _rep_delay_range_us = [100, 1000]
        _rep_delay_range_s = [_rd * 1.0e-6 for _rd in _rep_delay_range_us]
        _default_rep_delay_us = 500
        _default_rep_delay_s = 500 * 1.0e-6

        setattr(self.config, "rep_delay_range", _rep_delay_range_s)
        setattr(self.config, "default_rep_delay", _default_rep_delay_s)

        config_dict = self.config.to_dict()
        for i, rd in enumerate(config_dict["rep_delay_range"]):
            self.assertAlmostEqual(rd, _rep_delay_range_us[i], delta=1e-8)
        self.assertEqual(config_dict["default_rep_delay"], _default_rep_delay_us)

    def test_get_channel_prefix_index(self):
        """Test private method to get channel and index."""
        self.assertEqual(self.config._get_channel_prefix_index("acquire0"), ("acquire", 0))
        with self.assertRaises(BackendConfigurationError):
            self.config._get_channel_prefix_index("acquire")

    def _test_lists_equal(self, actual, expected):
        """Test if 2 lists are equal. It returns ``True`` is lists are equal."""
        return collections.Counter(actual) == collections.Counter(expected)

    def test_deepcopy(self):
        """Ensure that a deepcopy succeeds and results in an identical object."""
        copy_config = copy.deepcopy(self.config)
        self.assertEqual(copy_config, self.config)

    def test_u_channel_lo_scale(self):
        """Ensure that u_channel_lo scale is a complex number"""
        with self.assertWarns(DeprecationWarning):
            valencia_conf = Fake27QPulseV1().configuration()
        self.assertTrue(isinstance(valencia_conf.u_channel_lo[0][0].scale, complex))

    def test_processor_type(self):
        """Test the "processor_type" field in the backend configuration."""
        reference_processor_type = {
            "family": "Canary",
            "revision": "1.0",
            "segment": "A",
        }
        self.assertEqual(self.config.processor_type, reference_processor_type)
        self.assertEqual(self.config.to_dict()["processor_type"], reference_processor_type)

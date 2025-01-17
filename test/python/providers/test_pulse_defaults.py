# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test the PulseDefaults part of the backend."""
import copy
import warnings

import numpy as np

from qiskit.providers.fake_provider import FakeOpenPulse2Q, GenericBackendV2
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestPulseDefaults(QiskitTestCase):
    """Test the PulseDefaults creation and method usage."""

    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            # BackendV2 does not have defaults
            self.defs = FakeOpenPulse2Q().defaults()
            backend = GenericBackendV2(
                2, calibrate_instructions=True, basis_gates=["cx", "u1", "u2", "u3"], seed=42
            )
            self.inst_map = backend.instruction_schedule_map

    def test_buffer(self):
        """Test getting the buffer value."""
        self.assertEqual(self.defs.buffer, 10)

    def test_freq_est(self):
        """Test extracting qubit frequencies."""
        warnings.simplefilter("ignore")
        self.assertEqual(self.defs.qubit_freq_est[1], 5.0 * 1e9)
        self.assertEqual(self.defs.meas_freq_est[0], 6.5 * 1e9)
        warnings.simplefilter("default")

    def test_default_building(self):
        """Test building of ops definition is properly built from backend."""
        self.assertTrue(self.inst_map.has("u1", (0,)))
        self.assertTrue(self.inst_map.has("u3", (0,)))
        self.assertTrue(self.inst_map.has("u3", 1))
        self.assertTrue(self.inst_map.has("cx", (0, 1)))
        self.assertEqual(self.inst_map.get_parameters("u1", 0), ("P0",))
        u1_minus_pi = self.inst_map.get("u1", 0, P0=np.pi)
        fc_cmd = u1_minus_pi.instructions[0][-1]
        self.assertAlmostEqual(fc_cmd.phase, -np.pi)

    def test_str(self):
        """Test that __str__ method works."""
        self.assertEqual(
            "<PulseDefaults(<InstructionScheduleMap(1Q instructions:\n  q0:", str(self.defs)[:61]
        )
        self.assertTrue("Multi qubit instructions:\n  (0, 1): " in str(self.defs)[70:])
        self.assertTrue(
            "Qubit Frequencies [GHz]\n[4.9, 5.0]\nMeasurement Frequencies [GHz]\n[6.5, "
            "6.6] )>" in str(self.defs)[100:]
        )

    def test_deepcopy(self):
        """Test that deepcopy creates an identical object."""
        copy_defs = copy.deepcopy(self.defs)
        self.assertEqual(list(copy_defs.to_dict().keys()), list(self.defs.to_dict().keys()))

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qiskit/tools/parallel"""
import os
import time
import warnings

from unittest.mock import patch

from qiskit.utils.parallel import get_platform_parallel_default, parallel_map
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.pulse import Schedule
from test import QiskitTestCase  # pylint: disable=wrong-import-order


def _parfunc(x):
    """Function for testing parallel_map"""
    time.sleep(1)
    return x


def _build_simple_circuit(_):
    qreg = QuantumRegister(2)
    creg = ClassicalRegister(2)
    qc = QuantumCircuit(qreg, creg)
    return qc


def _build_simple_schedule(_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        # `Schedule` is deprecated in Qiskit 1.3
        return Schedule()


class TestGetPlatformParallelDefault(QiskitTestCase):
    """Tests get_parallel_default_for_platform."""

    def test_windows_parallel_default(self):
        """Verifies the parallel default for Windows."""
        with patch("sys.platform", "win32"):
            parallel_default = get_platform_parallel_default()
            self.assertEqual(parallel_default, False)

    def test_mac_os_unsupported_version_parallel_default(self):
        """Verifies the parallel default for macOS."""
        with patch("sys.platform", "darwin"):
            with patch("sys.version_info", (3, 8, 0, "final", 0)):
                parallel_default = get_platform_parallel_default()
                self.assertEqual(parallel_default, False)

    def test_other_os_parallel_default(self):
        """Verifies the parallel default for Linux and other OSes."""
        with patch("sys.platform", "linux"):
            parallel_default = get_platform_parallel_default()
            self.assertEqual(parallel_default, True)


class TestParallel(QiskitTestCase):
    """A class for testing parallel_map functionality."""

    def test_parallel_env_flag(self):
        """Verify parallel env flag is set"""
        self.assertEqual(os.getenv("QISKIT_IN_PARALLEL", None), "FALSE")

    def test_parallel(self):
        """Test parallel_map"""
        ans = parallel_map(_parfunc, list(range(10)))
        self.assertEqual(ans, list(range(10)))

    def test_parallel_circuit_names(self):
        """Verify unique circuit names in parallel"""
        out_circs = parallel_map(_build_simple_circuit, list(range(10)))
        names = [circ.name for circ in out_circs]
        self.assertEqual(len(names), len(set(names)))

    def test_parallel_schedule_names(self):
        """Verify unique schedule names in parallel"""
        out_schedules = parallel_map(_build_simple_schedule, list(range(10)))
        names = [schedule.name for schedule in out_schedules]
        self.assertEqual(len(names), len(set(names)))

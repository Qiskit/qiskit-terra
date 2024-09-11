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

"""
Fake backend abstract class for mock backends supporting OpenPulse.
"""

from qiskit.exceptions import QiskitError
from qiskit.providers.models.backendconfiguration import PulseBackendConfiguration
from qiskit.providers.models.pulsedefaults import PulseDefaults

from .fake_qasm_backend import FakeQasmBackend
from .utils.json_decoder import decode_pulse_defaults


class FakePulseBackend(FakeQasmBackend):
    """A fake pulse backend."""

    defs_filename = None

    def defaults(self):
        """Returns a snapshot of device defaults"""
        if not self._defaults:
            self._set_defaults_from_json()
        return self._defaults

    def _set_defaults_from_json(self):
        if not self.props_filename:
            raise QiskitError("No properties file has been defined")
        defs = self._load_json(self.defs_filename)
        decode_pulse_defaults(defs)
        self._defaults = PulseDefaults.from_dict(defs)

    def _get_config_from_dict(self, conf):
        return PulseBackendConfiguration.from_dict(conf)

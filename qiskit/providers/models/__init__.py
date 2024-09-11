# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
================================================
Backend Objects (:mod:`qiskit.providers.models`)
================================================

.. currentmodule:: qiskit.providers.models

Qiskit schema-conformant objects used by the backends and providers.

Classes
=======

.. autosummary::
   :toctree: ../stubs/

   BackendConfiguration
   BackendProperties
   BackendStatus
   QasmBackendConfiguration
   PulseBackendConfiguration
   UchannelLO
   GateConfig
   PulseDefaults
   Command
   JobStatus
   GateProperties
   Nduv
"""
# pylint: disable=undefined-all-variable
__all__ = [
    "BackendConfiguration",
    "PulseBackendConfiguration",
    "QasmBackendConfiguration",
    "UchannelLO",
    "GateConfig",
    "BackendProperties",
    "GateProperties",
    "Nduv",
    "BackendStatus",
    "JobStatus",
    "PulseDefaults",
    "Command",
]

import importlib
import warnings


_NAME_MAP = {
    # public object name mapped to containing module
    "BackendConfiguration": "qiskit.providers.models.backendconfiguration",
    "PulseBackendConfiguration": "qiskit.providers.models.backendconfiguration",
    "QasmBackendConfiguration": "qiskit.providers.models.backendconfiguration",
    "UchannelLO": "qiskit.providers.models.backendconfiguration",
    "GateConfig": "qiskit.providers.models.backendconfiguration",
    "BackendProperties": "qiskit.providers.models.backendproperties",
    "GateProperties": "qiskit.providers.models.backendproperties",
    "Nduv": "qiskit.providers.models.backendproperties",
    "BackendStatus": "qiskit.providers.models.backendstatus",
    "JobStatus": "qiskit.providers.models.jobstatus",
    "PulseDefaults": "qiskit.providers.models.pulsedefaults",
    "Command": "qiskit.providers.models.pulsedefaults",
}


def __getattr__(name):
    if (module_name := _NAME_MAP.get(name)) is not None:
        warnings.warn(
            "qiskit.providers.models is deprecated since Qiskit 1.2 and will be "
            "removed in Qiskit 2.0. With the removal of Qobj, there is no need for these "
            "schema-conformant objects. If you still need to use them, it could be because "
            "you are using a BackendV1, which is also deprecated in favor of BackendV2.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(module_name), name)
    raise AttributeError(f"module 'qiskit.providers.models' has no attribute '{name}'")

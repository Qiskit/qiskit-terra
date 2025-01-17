# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
======================================================
Fake Provider (:mod:`qiskit.providers.fake_provider`)
======================================================

.. currentmodule:: qiskit.providers.fake_provider

Overview
========

The fake provider module in Qiskit contains fake (simulated) backend classes
useful for testing the transpiler and other backend-facing functionality.

Example Usage
-------------

Here is an example of using a simulated backend for transpilation and running.

.. plot::
   :alt: Output from the previous code.
   :include-source:

   from qiskit import QuantumCircuit, transpile
   from qiskit.providers.fake_provider import GenericBackendV2
   from qiskit.visualization import plot_histogram

   # Generate a 5-qubit simulated backend
   backend = GenericBackendV2(num_qubits=5)

   # Create a simple circuit
   circuit = QuantumCircuit(3)
   circuit.h(0)
   circuit.cx(0,1)
   circuit.cx(0,2)
   circuit.measure_all()
   circuit.draw('mpl')

   # Transpile the ideal circuit to a circuit that can be directly executed by the backend
   transpiled_circuit = transpile(circuit, backend)
   transpiled_circuit.draw('mpl')

   # Run the transpiled circuit using the simulated backend
   job = backend.run(transpiled_circuit)
   counts = job.result().get_counts()
   plot_histogram(counts)


V2 Simulated Backends
=====================

.. autosummary::
    :toctree: ../stubs/

    GenericBackendV2

V1 Fake Backends (Legacy interface)
===================================

.. autosummary::
    :toctree: ../stubs/

    FakeOpenPulse2Q
    FakeOpenPulse3Q
    Fake1Q
    Fake5QV1
    Fake20QV1
    Fake7QPulseV1
    Fake27QPulseV1
    Fake127QPulseV1

Fake Backend Base Classes
=========================

The V1 fake backends are based on a set of base classes:

.. currentmodule:: qiskit.providers.fake_provider
.. autoclass:: FakeBackend
.. autoclass:: FakeQasmBackend
.. autoclass:: FakePulseBackend
"""

# Base classes for fake backends
from .fake_backend import FakeBackend
from .fake_qasm_backend import FakeQasmBackend
from .fake_pulse_backend import FakePulseBackend

# Special fake backends for special testing purposes
from .fake_openpulse_2q import FakeOpenPulse2Q
from .fake_openpulse_3q import FakeOpenPulse3Q
from .fake_1q import Fake1Q

# Generic fake backends
from .backends_v1 import Fake5QV1, Fake20QV1, Fake7QPulseV1, Fake27QPulseV1, Fake127QPulseV1
from .generic_backend_v2 import GenericBackendV2

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

"""Sampler implementation for an artibtrary Backend object."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.options import Options
from qiskit.result import QuasiDistribution, Result
from qiskit.transpiler.passmanager import PassManager

from .backend_estimator import _prepare_counts, _run_circuits
from .base import BaseSampler, SamplerResult
from .primitive_job import PrimitiveJob
from .utils import _circuit_key


class BackendSampler(BaseSampler[PrimitiveJob[SamplerResult]]):
    """A :class:`~.BaseSampler` implementation that provides an interface for
    leveraging the sampler interface from any backend.

    This class provides a sampler interface from any backend and doesn't do
    any measurement mitigation, it just computes the probability distribution
    from the counts. It facilitates using backends that do not provide a
    native :class:`~.BaseSampler` implementation in places that work with
    :class:`~.BaseSampler`, such as algorithms in :mod:`qiskit.algorithms`
    including :class:`~.qiskit.algorithms.minimum_eigensolvers.SamplingVQE`.
    However, if you're using a provider that has a native implementation of
    :class:`~.BaseSampler`, it is a better choice to leverage that native
    implementation as it will likely include additional optimizations and be
    a more efficient implementation. The generic nature of this class
    precludes doing any provider- or backend-specific optimizations.
    """

    def __init__(
        self,
        backend: BackendV1 | BackendV2,
        options: dict | None = None,
        bound_pass_manager: PassManager | None = None,
        skip_transpilation: bool = False,
    ):
        """Initialize a new BackendSampler

        Args:
            backend: Required: the backend to run the sampler primitive on
            options: Default options.
            bound_pass_manager: An optional pass manager to run after
                parameter binding.
            skip_transpilation: If this is set to True the internal compilation
                of the input circuits is skipped and the circuit objects
                will be directly executed when this objected is called.
        Raises:
            ValueError: If backend is not provided
        """

        super().__init__(options=options)
        self._backend = backend
        self._transpile_options = Options()
        self._bound_pass_manager = bound_pass_manager
        self._preprocessed_circuits: list[QuantumCircuit] | None = None
        self._transpiled_circuits: list[QuantumCircuit] = []
        self._skip_transpilation = skip_transpilation
        self._circuit_ids = {}

    @property
    def preprocessed_circuits(self) -> list[QuantumCircuit]:
        """
        Preprocessed quantum circuits produced by preprocessing
        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        return list(self._circuits)

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits.
        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        if self._skip_transpilation:
            self._transpiled_circuits = list(self._circuits)
        elif len(self._transpiled_circuits) < len(self._circuits):
            # transpile only circuits that are not transpiled yet
            self._transpile()
        return self._transpiled_circuits

    @property
    def backend(self) -> BackendV1 | BackendV2:
        """
        Returns:
            The backend which this sampler object based on
        """
        return self._backend

    @property
    def transpile_options(self) -> Options:
        """Return the transpiler options for transpiling the circuits."""
        return self._transpile_options

    def set_transpile_options(self, **fields):
        """Set the transpiler options for transpiler.
        Args:
            **fields: The fields to update the options.
        Returns:
            self.
        Raises:
            QiskitError: if the instance has been closed.
        """
        self._transpile_options.update_options(**fields)

    def _call(
        self,
        circuits: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:

        # This line does the actual transpilation
        transpiled_circuits = self.transpiled_circuits
        bound_circuits = [
            transpiled_circuits[i]
            if len(value) == 0
            else transpiled_circuits[i].bind_parameters(dict(zip(self._parameters[i], value)))
            for i, value in zip(circuits, parameter_values)
        ]
        bound_circuits = self._bound_pass_manager_run(bound_circuits)
        # Run
        result, _metadata = _run_circuits(bound_circuits, self._backend, **run_options)
        return self._postprocessing(result, bound_circuits)

    def _postprocessing(
        self, result: list[Result], circuits: list[QuantumCircuit]
    ) -> SamplerResult:
        counts = _prepare_counts(result)
        shots = sum(counts[0].values())

        probabilities = []
        metadata: list[dict[str, Any]] = [{} for _ in range(len(circuits))]
        for count in counts:
            prob_dist = {k: v / shots for k, v in count.items()}
            probabilities.append(
                QuasiDistribution(prob_dist, shots=shots, stddev_upper_bound=math.sqrt(1 / shots))
            )
            for metadatum in metadata:
                metadatum["shots"] = shots

        return SamplerResult(probabilities, metadata)

    def _transpile(self):
        from qiskit.compiler import transpile

        start = len(self._transpiled_circuits)
        self._transpiled_circuits.extend(
            transpile(
                self.preprocessed_circuits[start:],
                self.backend,
                **self.transpile_options.__dict__,
            ),
        )

    def _bound_pass_manager_run(self, circuits):
        if self._bound_pass_manager is None:
            return circuits
        else:
            output = self._bound_pass_manager.run(circuits)
            if not isinstance(output, list):
                output = [output]
            return output

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ):
        circuit_indices = []
        for circuit in circuits:
            index = self._circuit_ids.get(_circuit_key(circuit))
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[_circuit_key(circuit)] = len(self._circuits)
                self._circuits.append(circuit)
                self._parameters.append(circuit.parameters)
        job = PrimitiveJob(self._call, circuit_indices, parameter_values, **run_options)
        job.submit()
        return job

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
"""
Estimator V1 reference implementation
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.utils.deprecation import deprecate_func

from .base import BaseEstimator, EstimatorResult
from .primitive_job import PrimitiveJob
from .utils import (
    _circuit_key,
    _observable_key,
    _statevector_from_circuit,
    init_observable,
)


class Estimator(BaseEstimator[PrimitiveJob[EstimatorResult]]):
    """
    Reference implementation of :class:`BaseEstimator` (V1).

    :Run Options:

        - **shots** (None or int) --
          The number of shots. If None, it calculates the expectation values
          with full state vector simulation.
          Otherwise, it samples from normal distributions with standard errors as standard
          deviations using normal distribution approximation.

        - **seed** (np.random.Generator or int) --
          Set a fixed seed or generator for the normal distribution. If shots is None,
          this option is ignored.

    .. note::
        The result of this class is exact if the circuit contains only unitary operations.
        On the other hand, the result could be stochastic if the circuit contains a non-unitary
        operation such as a reset for a some subsystems.
        The stochastic result can be made reproducible by setting ``seed``, e.g.,
        ``Estimator(options={"seed":123})``.
    """

    @deprecate_func(
        since="1.2",
        additional_msg="All implementations of the `BaseEstimatorV1` interface "
        "have been deprecated in favor of their V2 counterparts. "
        "The V2 alternative for the `Estimator` class is `StatevectorEstimator`.",
    )
    def __init__(self, *, options: dict | None = None):
        """
        Args:
            options: Default options.

        Raises:
            QiskitError: if some classical bits are not used for measurements.
        """
        super().__init__(options=options)
        self._circuits = []
        self._parameters = []
        self._observables = []
        self._circuit_ids = {}
        self._observable_ids = {}

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        shots = run_options.pop("shots", None)
        seed = run_options.pop("seed", None)
        if seed is None:
            rng = np.random.default_rng()
        elif isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        # Initialize metadata
        metadata: list[dict[str, Any]] = [{} for _ in range(len(circuits))]

        bound_circuits = []
        for i, value in zip(circuits, parameter_values):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])})."
                )
            bound_circuits.append(
                self._circuits[i]
                if len(value) == 0
                else self._circuits[i].assign_parameters(dict(zip(self._parameters[i], value)))
            )
        sorted_observables = [self._observables[i] for i in observables]
        expectation_values = []
        for circ, obs, metadatum in zip(bound_circuits, sorted_observables, metadata):
            if circ.num_qubits != obs.num_qubits:
                raise QiskitError(
                    f"The number of qubits of a circuit ({circ.num_qubits}) does not match "
                    f"the number of qubits of a observable ({obs.num_qubits})."
                )
            final_state = _statevector_from_circuit(circ, rng)
            expectation_value = final_state.expectation_value(obs)
            if shots is None:
                expectation_values.append(expectation_value)
            else:
                expectation_value = np.real_if_close(expectation_value)
                sq_obs = (obs @ obs).simplify(atol=0)
                sq_exp_val = np.real_if_close(final_state.expectation_value(sq_obs))
                variance = sq_exp_val - expectation_value**2
                variance = max(variance, 0)
                standard_error = np.sqrt(variance / shots)
                expectation_value_with_error = rng.normal(expectation_value, standard_error)
                expectation_values.append(expectation_value_with_error)
                metadatum["variance"] = variance
                metadatum["shots"] = shots

        return EstimatorResult(np.real_if_close(expectation_values), metadata)

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[BaseOperator, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ):
        circuit_indices = []
        for circuit in circuits:
            key = _circuit_key(circuit)
            index = self._circuit_ids.get(key)
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[key] = len(self._circuits)
                self._circuits.append(circuit)
                self._parameters.append(circuit.parameters)
        observable_indices = []
        for observable in observables:
            observable = init_observable(observable)
            index = self._observable_ids.get(_observable_key(observable))
            if index is not None:
                observable_indices.append(index)
            else:
                observable_indices.append(len(self._observables))
                self._observable_ids[_observable_key(observable)] = len(self._observables)
                self._observables.append(observable)
        job = PrimitiveJob(
            self._call, circuit_indices, observable_indices, parameter_values, **run_options
        )
        job._submit()
        return job

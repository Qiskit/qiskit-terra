# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""The Quantum Phase Estimation Algorithm."""

from __future__ import annotations

import numpy as np

from qiskit.circuit import QuantumCircuit
import qiskit
from qiskit import circuit
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.utils.deprecation import deprecate_arg
from qiskit.result import Result
from qiskit.algorithms.exceptions import AlgorithmError
from .phase_estimation_result import PhaseEstimationResult, _sort_phases
from .phase_estimator import PhaseEstimator
from ...primitives import BaseSampler


class PhaseEstimation(PhaseEstimator):
    r"""Run the Quantum Phase Estimation (QPE) algorithm.

    This runs QPE with a multi-qubit register for reading the phases [1]
    of input states.

    The algorithm takes as input a unitary :math:`U` and a state :math:`|\psi\rangle`,
    which may be written

    .. math::

        |\psi\rangle = \sum_j c_j |\phi_j\rangle,

    where :math:`|\phi_j\rangle` are eigenstates of :math:`U`. We prepare the quantum register
    in the state :math:`|\psi\rangle` then apply :math:`U` leaving the register in the state

    .. math::

        U|\psi\rangle = \sum_j \exp(i \phi_j) c_j |\phi_j\rangle.

    In the ideal case, one then measures the phase :math:`\phi_j` with probability
    :math:`|c_j|^2`.  In practice, many (or all) of the bit strings may be measured due to
    noise and the possibility that :math:`\phi_j` may not be representable exactly by the
    output register. In the latter case the probability for each eigenphase will be spread
    across bitstrings, with amplitudes that decrease with distance from the bitstring most
    closely approximating the eigenphase.

    The main input to the constructor is the number of qubits in the phase-reading register.
    For phase estimation, there are two methods:

    first. `estimate`, which takes a state preparation circuit to prepare an input state, and
      a unitary that will act on the input state. In this case, an instance of
      :class:`qiskit.circuit.PhaseEstimation`, a QPE circuit, containing
      the state preparation and input unitary will be constructed.
    second. `estimate_from_pe_circuit`, which takes a quantum-phase-estimation circuit in which
      the unitary and state preparation are already embedded.

    In both estimation methods, the QPE circuit is run on a backend
    and the frequencies or counts of the phases represented by bitstrings
    are recorded. The results are returned as an instance of
    :class:`~qiskit.algorithms.phase_estimator_result.PhaseEstimationResult`.

    **Reference:**

    [1]: Michael A. Nielsen and Isaac L. Chuang. 2011.
         Quantum Computation and Quantum Information: 10th Anniversary Edition (10th ed.).
         Cambridge University Press, New York, NY, USA.

    """

    @deprecate_arg(
        "quantum_instance",
        additional_msg=(
            "Instead, use the ``sampler`` argument. See https://qisk.it/algo_migration for a "
            "migration guide."
        ),
        since="0.24.0",
    )
    def __init__(
        self,
        num_evaluation_qubits: int,
        quantum_instance: QuantumInstance | Backend | None = None,
        sampler: BaseSampler | None = None,
    ) -> None:
        r"""
        Args:
            num_evaluation_qubits: The number of qubits used in estimating the phase. The phase will
                be estimated as a binary string with this many bits.
            quantum_instance: Deprecated: The quantum instance on which the
                circuit will be run.
            sampler: The sampler primitive on which the circuit will be sampled.

        Raises:
            AlgorithmError: If neither sampler nor quantum instance is provided.
        """
        if sampler is None and quantum_instance is None:
            raise AlgorithmError(
                "Neither a sampler nor a quantum instance was provided. Please provide one of them."
            )
        self._measurements_added = False
        if num_evaluation_qubits is not None:
            self._num_evaluation_qubits = num_evaluation_qubits

        if isinstance(quantum_instance, Backend):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance
        self._sampler = sampler

    def construct_circuit(
        self, unitary: QuantumCircuit, state_preparation: QuantumCircuit | None = None
    ) -> QuantumCircuit:
        """Return the circuit to be executed to estimate phases.

        This circuit includes as sub-circuits the core phase estimation circuit,
        with the addition of the state-preparation circuit and possibly measurement instructions.
        """
        num_evaluation_qubits = self._num_evaluation_qubits
        num_unitary_qubits = unitary.num_qubits

        pe_circuit = circuit.library.PhaseEstimation(num_evaluation_qubits, unitary)

        if state_preparation is not None:
            pe_circuit.compose(
                state_preparation,
                qubits=range(num_evaluation_qubits, num_evaluation_qubits + num_unitary_qubits),
                inplace=True,
                front=True,
            )

        return pe_circuit

    def _add_measurement_if_required(self, pe_circuit):
        if self._sampler is not None or not self._quantum_instance.is_statevector:
            # Measure only the evaluation qubits.
            regname = "meas"
            creg = ClassicalRegister(self._num_evaluation_qubits, regname)
            pe_circuit.add_register(creg)
            pe_circuit.barrier()
            pe_circuit.measure(
                range(self._num_evaluation_qubits), range(self._num_evaluation_qubits)
            )

        return circuit

    def _compute_phases(
        self, num_unitary_qubits: int, circuit_result: Result
    ) -> np.ndarray | qiskit.result.Counts:
        """Compute frequencies/counts of phases from the result of running the QPE circuit.

        How the frequencies are computed depends on whether the backend computes amplitude or
        samples outcomes.

        1) If the backend is a statevector simulator, then the reduced density matrix of the
        phase-reading register is computed from the combined phase-reading- and input-state
        registers. The elements of the diagonal :math:`(i, i)` give the probability to measure the
        each of the states `i`. The index `i` expressed as a binary integer with the LSB rightmost
        gives the state of the phase-reading register with the LSB leftmost when interpreted as a
        phase. In order to maintain the compact representation, the phases are maintained as decimal
        integers.  They may be converted to other forms via the results object,
        `PhaseEstimationResult` or `HamiltonianPhaseEstimationResult`.

         2) If the backend samples bitstrings, then the counts are first retrieved as a dict.  The
        binary strings (the keys) are then reversed so that the LSB is rightmost and the counts are
        converted to frequencies. Then the keys are sorted according to increasing phase, so that
        they can be easily understood when displaying or plotting a histogram.

        Args:
            num_unitary_qubits: The number of qubits in the unitary.
            circuit_result: the result object returned by the backend that ran the QPE circuit.

        Returns:
            Either a dict or np.ndarray representing the frequencies of the phases.

        """
        if self._quantum_instance.is_statevector:
            state_vec = circuit_result.get_statevector()
            evaluation_density_matrix = qiskit.quantum_info.partial_trace(
                state_vec,
                range(
                    self._num_evaluation_qubits, self._num_evaluation_qubits + num_unitary_qubits
                ),
            )
            phases = evaluation_density_matrix.probabilities()
        else:
            # return counts with keys sorted numerically
            num_shots = circuit_result.results[0].shots
            counts = circuit_result.get_counts()
            phases = {k[::-1]: counts[k] / num_shots for k in counts.keys()}
            phases = _sort_phases(phases)
            phases = qiskit.result.Counts(
                phases, memory_slots=counts.memory_slots, creg_sizes=counts.creg_sizes
            )

        return phases

    def estimate_from_pe_circuit(
        self, pe_circuit: QuantumCircuit, num_unitary_qubits: int
    ) -> PhaseEstimationResult:
        """Run the phase estimation algorithm on a phase estimation circuit

        Args:
            pe_circuit: The phase estimation circuit.
            num_unitary_qubits: Must agree with the number of qubits in the unitary in `pe_circuit`.

        Returns:
            An instance of qiskit.algorithms.phase_estimator_result.PhaseEstimationResult.

        Raises:
            AlgorithmError: Primitive job failed.
        """

        self._add_measurement_if_required(pe_circuit)

        if self._sampler is not None:
            try:
                circuit_job = self._sampler.run([pe_circuit])
                circuit_result = circuit_job.result()
            except Exception as exc:
                raise AlgorithmError("The primitive job failed!") from exc
            phases = circuit_result.quasi_dists[0]
            phases_bitstrings = {}
            for key, phase in phases.items():
                bitstring_key = self._get_reversed_bitstring(self._num_evaluation_qubits, key)
                phases_bitstrings[bitstring_key] = phase
            phases = phases_bitstrings

        else:
            circuit_result = self._quantum_instance.execute(pe_circuit)
            phases = self._compute_phases(num_unitary_qubits, circuit_result)
        return PhaseEstimationResult(
            self._num_evaluation_qubits, circuit_result=circuit_result, phases=phases
        )

    def estimate(
        self,
        unitary: QuantumCircuit,
        state_preparation: QuantumCircuit | None = None,
    ) -> PhaseEstimationResult:
        """Build a phase estimation circuit and run the corresponding algorithm.

        Args:
            unitary: The circuit representing the unitary operator whose eigenvalues (via phase)
                will be measured.
            state_preparation: The circuit that prepares the state whose eigenphase will be
                measured.  If this parameter is omitted, no preparation circuit
                will be run and input state will be the all-zero state in the
                computational basis.

        Returns:
            An instance of qiskit.algorithms.phase_estimator_result.PhaseEstimationResult.
        """
        pe_circuit = self.construct_circuit(unitary, state_preparation)
        num_unitary_qubits = unitary.num_qubits

        return self.estimate_from_pe_circuit(pe_circuit, num_unitary_qubits)

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

"""Result of running PhaseEstimation"""
from __future__ import annotations
import numpy as np

from qiskit.utils.deprecation import deprecate_func
from qiskit.result import Result
from .phase_estimator import PhaseEstimatorResult


class PhaseEstimationResult(PhaseEstimatorResult):
    """Store and manipulate results from running `PhaseEstimation`.

    This class is instantiated by the ``PhaseEstimation`` class, not via user code.
    The ``PhaseEstimation`` class generates a list of phases and corresponding weights. Upon
    completion it returns the results as an instance of this class. The main method for
    accessing the results is `filter_phases`.

    The canonical phase satisfying the ``PhaseEstimator`` interface, returned by the
    attribute `phase`, is the most likely phase.
    """

    def __init__(
        self,
        num_evaluation_qubits: int,
        circuit_result: Result,
        phases: np.ndarray | dict[str, float],
    ) -> None:
        """
        Args:
            num_evaluation_qubits: number of qubits in phase-readout register.
            circuit_result: result object returned by method running circuit.
            phases: ndarray or dict of phases and frequencies determined by QPE.
        """
        super().__init__()

        self._phases = phases
        # int: number of qubits in phase-readout register
        self._num_evaluation_qubits = num_evaluation_qubits
        self._circuit_result = circuit_result

    @property
    def phases(self) -> np.ndarray | dict:
        """Return all phases and their frequencies computed by QPE.

        This is an array or dict whose values correspond to weights on bit strings.
        """
        return self._phases

    @property
    def circuit_result(self) -> Result:
        """Return the result object returned by running the QPE circuit (on hardware or simulator).

        This is useful for inspecting and troubleshooting the QPE algorithm.
        """
        return self._circuit_result

    @property
    @deprecate_func(
        additional_msg="Instead, use the property ``phase``, which behaves the same.",
        since="0.18.0",
        is_property=True,
    )
    def most_likely_phase(self) -> float:
        r"""DEPRECATED - Return the most likely phase as a number in :math:`[0.0, 1.0)`.

        1.0 corresponds to a phase of :math:`2\pi`. This selects the phase corresponding
        to the bit string with the highesest probability. This is the most likely phase.
        """
        return self.phase

    @property
    def phase(self) -> float:
        r"""Return the most likely phase as a number in :math:`[0.0, 1.0)`.

        1.0 corresponds to a phase of :math:`2\pi`. This selects the phase corresponding
        to the bit string with the highesest probability. This is the most likely phase.
        """
        if isinstance(self.phases, dict):
            binary_phase_string = max(self.phases, key=self.phases.get)
        else:
            # np.argmax ignores complex part of number. But, we take abs anyway
            idx = np.argmax(abs(self.phases))
            binary_phase_string = np.binary_repr(idx, self._num_evaluation_qubits)[::-1]
        phase = _bit_string_to_phase(binary_phase_string)
        return phase

    def filter_phases(self, cutoff: float = 0.0, as_float: bool = True) -> dict:
        """Return a filtered dict of phases (keys) and frequencies (values).

        Only phases with frequencies (counts) larger than `cutoff` are included.
        It is assumed that the `run` method has been called so that the phases have been computed.
        When using a noiseless, shot-based simulator to read a single phase that can
        be represented exactly by `num_evaluation_qubits`, all the weight will
        be concentrated on a single phase. In all other cases, many, or all, bit
        strings will have non-zero weight. This method is useful for filtering
        out these uninteresting bit strings.

        Args:
            cutoff: Minimum weight of number of counts required to keep a bit string.
                The default value is `0.0`.
            as_float: If `True`, returned keys are floats in :math:`[0.0, 1.0)`. If `False`
                returned keys are bit strings.

        Returns:
            A filtered dict of phases (keys) and frequencies (values).
        """
        if isinstance(self.phases, dict):
            counts = self.phases
            if as_float:
                phases = {
                    _bit_string_to_phase(k): counts[k] for k in counts.keys() if counts[k] > cutoff
                }
            else:
                phases = {k: counts[k] for k in counts.keys() if counts[k] > cutoff}

        else:
            phases = {}
            for idx, amplitude in enumerate(self.phases):
                if amplitude > cutoff:
                    # Each index corresponds to a computational basis state with the LSB rightmost.
                    # But, we chose to apply the unitaries such that the phase is recorded
                    # in reverse order. So, we reverse the bitstrings here.
                    binary_phase_string = np.binary_repr(idx, self._num_evaluation_qubits)[::-1]
                    if as_float:
                        _key: str | float = _bit_string_to_phase(binary_phase_string)
                    else:
                        _key = binary_phase_string
                    phases[_key] = amplitude

            phases = _sort_phases(phases)

        return phases


def _bit_string_to_phase(binary_string: str) -> float:
    """Convert bit string to a normalized phase in :math:`[0,1)`.

    It is assumed that the bit string is correctly padded and that the order of
    the bits has been reversed relative to their order when the counts
    were recorded. The LSB is the right most when interpreting the bitstring as
    a phase.

    Args:
        binary_string: A string of characters '0' and '1'.

    Returns:
        A phase scaled to :math:`[0,1)`.
    """
    n_qubits = len(binary_string)
    return int(binary_string, 2) / (2**n_qubits)


def _sort_phases(phases: dict) -> dict:
    """Sort a dict of bit strings representing phases (keys) and frequencies (values) by bit string.

    The bit strings are sorted according to increasing phase. This relies on Python
    preserving insertion order when building dicts.
    """
    pkeys = list(phases.keys())
    pkeys.sort(reverse=False)  # Sorts in order of the integer encoded by binary string
    phases = {k: phases[k] for k in pkeys}
    return phases

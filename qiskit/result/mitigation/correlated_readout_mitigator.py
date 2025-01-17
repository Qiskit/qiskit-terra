# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Readout mitigator class based on the A-matrix inversion method
"""

import math
from typing import Optional, List, Tuple, Iterable, Callable, Union, Dict
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.utils.deprecation import deprecate_func
from ..distributions.quasi import QuasiDistribution
from ..counts import Counts
from .base_readout_mitigator import BaseReadoutMitigator
from .utils import counts_probability_vector, z_diagonal, str2diag


class CorrelatedReadoutMitigator(BaseReadoutMitigator):
    """This class is DEPRECATED. N-qubit readout error mitigator.

    Mitigates :meth:`expectation_value` and :meth:`quasi_probabilities`.
    The mitigation_matrix should be calibrated using qiskit experiments.
    This mitigation method should be used in case the readout errors of the qubits
    are assumed to be correlated. The mitigation_matrix of *N* qubits is of size
    :math:`2^N x 2^N` so the mitigation complexity is :math:`O(4^N)`.
    """

    @deprecate_func(
        since="1.3",
        package_name="Qiskit",
        removal_timeline="in Qiskit 2.0",
        additional_msg="The `qiskit.result.mitigation` module is deprecated in favor of "
        "the https://github.com/Qiskit/qiskit-addon-mthree package.",
    )
    def __init__(self, assignment_matrix: np.ndarray, qubits: Optional[Iterable[int]] = None):
        """Initialize a CorrelatedReadoutMitigator

        Args:
            assignment_matrix: readout error assignment matrix.
            qubits: Optional, the measured physical qubits for mitigation.

        Raises:
            QiskitError: matrix size does not agree with number of qubits
        """
        if np.any(assignment_matrix < 0) or not np.allclose(np.sum(assignment_matrix, axis=0), 1):
            raise QiskitError("Assignment matrix columns must be valid probability distributions")
        assignment_matrix = np.asarray(assignment_matrix, dtype=float)
        matrix_qubits_num = int(math.log2(assignment_matrix.shape[0]))
        if qubits is None:
            self._num_qubits = matrix_qubits_num
            self._qubits = range(self._num_qubits)
        else:
            if len(qubits) != matrix_qubits_num:
                raise QiskitError(
                    f"The number of given qubits ({len(qubits)}) is different than the number of "
                    f"qubits inferred from the matrices ({matrix_qubits_num})"
                )
            self._qubits = qubits
            self._num_qubits = len(self._qubits)
        self._qubit_index = dict(zip(self._qubits, range(self._num_qubits)))
        self._assignment_mat = assignment_matrix
        self._mitigation_mats = {}

    @property
    def settings(self) -> Dict:
        """Return settings."""
        return {"assignment_matrix": self._assignment_mat, "qubits": self._qubits}

    def expectation_value(
        self,
        data: Counts,
        diagonal: Union[Callable, dict, str, np.ndarray] = None,
        qubits: Iterable[int] = None,
        clbits: Optional[List[int]] = None,
        shots: Optional[int] = None,
    ) -> Tuple[float, float]:
        r"""Compute the mitigated expectation value of a diagonal observable.

        This computes the mitigated estimator of
        :math:`\langle O \rangle = \mbox{Tr}[\rho. O]` of a diagonal observable
        :math:`O = \sum_{x\in\{0, 1\}^n} O(x)|x\rangle\!\langle x|`.

        Args:
            data: Counts object
            diagonal: Optional, the vector of diagonal values for summing the
                      expectation value. If ``None`` the default value is
                      :math:`[1, -1]^\otimes n`.
            qubits: Optional, the measured physical qubits the count
                    bitstrings correspond to. If None qubits are assumed to be
                    :math:`[0, ..., n-1]`.
            clbits: Optional, if not None marginalize counts to the specified bits.
            shots: the number of shots.

        Returns:
            (float, float): the expectation value and an upper bound of the standard deviation.

        Additional Information:
            The diagonal observable :math:`O` is input using the ``diagonal`` kwarg as
            a list or Numpy array :math:`[O(0), ..., O(2^n -1)]`. If no diagonal is specified
            the diagonal of the Pauli operator
            :math`O = \mbox{diag}(Z^{\otimes n}) = [1, -1]^{\otimes n}` is used.
            The ``clbits`` kwarg is used to marginalize the input counts dictionary
            over the specified bit-values, and the ``qubits`` kwarg is used to specify
            which physical qubits these bit-values correspond to as
            ``circuit.measure(qubits, clbits)``.
        """

        if qubits is None:
            qubits = self._qubits
        probs_vec, shots = counts_probability_vector(
            data, qubit_index=self._qubit_index, clbits=clbits, qubits=qubits
        )

        # Get qubit mitigation matrix and mitigate probs
        mit_mat = self.mitigation_matrix(qubits)

        # Get operator coeffs
        if diagonal is None:
            diagonal = z_diagonal(2**self._num_qubits)
        elif isinstance(diagonal, str):
            diagonal = str2diag(diagonal)

        # Apply transpose of mitigation matrix
        coeffs = mit_mat.T.dot(diagonal)
        expval = coeffs.dot(probs_vec)
        stddev_upper_bound = self.stddev_upper_bound(shots)

        return (expval, stddev_upper_bound)

    def quasi_probabilities(
        self,
        data: Counts,
        qubits: Optional[List[int]] = None,
        clbits: Optional[List[int]] = None,
        shots: Optional[int] = None,
    ) -> QuasiDistribution:
        """Compute mitigated quasi probabilities value.

        Args:
            data: counts object
            qubits: qubits the count bitstrings correspond to.
            clbits: Optional, marginalize counts to just these bits.
            shots: Optional, the total number of shots, if None shots will
                be calculated as the sum of all counts.

        Returns:
            QuasiDistribution: A dictionary containing pairs of [output, mean] where "output"
                is the key in the dictionaries,
                which is the length-N bitstring of a measured standard basis state,
                and "mean" is the mean of non-zero quasi-probability estimates.
        """
        if qubits is None:
            qubits = self._qubits
        probs_vec, calculated_shots = counts_probability_vector(
            data, qubit_index=self._qubit_index, clbits=clbits, qubits=qubits
        )
        if shots is None:
            shots = calculated_shots

        # Get qubit mitigation matrix and mitigate probs
        mit_mat = self.mitigation_matrix(qubits)

        # Apply transpose of mitigation matrix
        probs_vec = mit_mat.dot(probs_vec)
        probs_dict = {}
        for index, _ in enumerate(probs_vec):
            probs_dict[index] = probs_vec[index]

        quasi_dist = QuasiDistribution(
            probs_dict, stddev_upper_bound=self.stddev_upper_bound(shots)
        )

        return quasi_dist

    def mitigation_matrix(self, qubits: List[int] = None) -> np.ndarray:
        r"""Return the readout mitigation matrix for the specified qubits.

        The mitigation matrix :math:`A^{-1}` is defined as the inverse of the
        :meth:`assignment_matrix` :math:`A`.

        Args:
            qubits: Optional, qubits being measured.

        Returns:
            np.ndarray: the measurement error mitigation matrix :math:`A^{-1}`.
        """
        if qubits is None:
            qubits = self._qubits
        qubits = tuple(sorted(qubits))

        # Check for cached mitigation matrix
        # if not present compute
        if qubits not in self._mitigation_mats:
            marginal_matrix = self.assignment_matrix(qubits)
            try:
                mit_mat = np.linalg.inv(marginal_matrix)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                mit_mat = np.linalg.pinv(marginal_matrix)
            self._mitigation_mats[qubits] = mit_mat

        return self._mitigation_mats[qubits]

    def assignment_matrix(self, qubits: List[int] = None) -> np.ndarray:
        r"""Return the readout assignment matrix for specified qubits.

        The assignment matrix is the stochastic matrix :math:`A` which assigns
        a noisy readout probability distribution to an ideal input
        readout distribution: :math:`P(i|j) = \langle i | A | j \rangle`.

        Args:
            qubits: Optional, qubits being measured.

        Returns:
            np.ndarray: the assignment matrix A.
        """
        if qubits is None:
            qubits = self._qubits
        if qubits == self._num_qubits:
            return self._assignment_mat

        if isinstance(qubits, int):
            qubits = [qubits]

        qubit_indices = [self._qubit_index[qubit] for qubit in qubits]
        # Compute marginal matrix
        axis = tuple(
            self._num_qubits - 1 - i for i in set(range(self._num_qubits)).difference(qubit_indices)
        )
        num_qubits = len(qubits)

        new_amat = np.zeros(2 * [2**num_qubits], dtype=float)
        for i, col in enumerate(self._assignment_mat.T[self._keep_indexes(qubit_indices)]):
            new_amat[i] = (
                np.reshape(col, self._num_qubits * [2]).sum(axis=axis).reshape([2**num_qubits])
            )
        new_amat = new_amat.T
        return new_amat

    @staticmethod
    def _keep_indexes(qubits):
        indexes = [0]
        for i in sorted(qubits):
            indexes += [idx + (1 << i) for idx in indexes]
        return indexes

    def _compute_gamma(self):
        """Compute gamma for N-qubit mitigation"""
        mitmat = self.mitigation_matrix(qubits=self._qubits)
        return np.max(np.sum(np.abs(mitmat), axis=0))

    def stddev_upper_bound(self, shots: int):
        """Return an upper bound on standard deviation of expval estimator.

        Args:
            shots: Number of shots used for expectation value measurement.

        Returns:
            float: the standard deviation upper bound.
        """
        gamma = self._compute_gamma()
        return gamma / math.sqrt(shots)

    @property
    def qubits(self) -> Tuple[int]:
        """The device qubits for this mitigator"""
        return self._qubits

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
Readout mitigation data handling utils
"""

import logging
import math
from typing import Optional, List, Tuple, Dict
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.utils.deprecation import deprecate_func
from ..utils import marginal_counts
from ..counts import Counts

logger = logging.getLogger(__name__)


@deprecate_func(
    since="1.3",
    package_name="Qiskit",
    removal_timeline="in Qiskit 2.0",
    additional_msg="The `qiskit.result.mitigation` module is deprecated in favor of "
    "the https://github.com/Qiskit/qiskit-addon-mthree package.",
)
def z_diagonal(dim, dtype=float):
    r"""Return the diagonal for the operator :math:`Z^\otimes n`"""
    parity = np.zeros(dim, dtype=dtype)
    for i in range(dim):
        parity[i] = bin(i)[2:].count("1")
    return (-1) ** np.mod(parity, 2)


@deprecate_func(
    since="1.3",
    package_name="Qiskit",
    removal_timeline="in Qiskit 2.0",
    additional_msg="The `qiskit.result.mitigation` module is deprecated in favor of "
    "the https://github.com/Qiskit/qiskit-addon-mthree package.",
)
def expval_with_stddev(coeffs: np.ndarray, probs: np.ndarray, shots: int) -> Tuple[float, float]:
    """Compute expectation value and standard deviation.
    Args:
        coeffs: array of diagonal operator coefficients.
        probs: array of measurement probabilities.
        shots: total number of shots to obtain probabilities.
    Returns:
        tuple: (expval, stddev) expectation value and standard deviation.
    """
    # Compute expval
    expval = coeffs.dot(probs)

    # Compute variance
    sq_expval = (coeffs**2).dot(probs)
    variance = (sq_expval - expval**2) / shots

    # Compute standard deviation
    if variance < 0 and not np.isclose(variance, 0):
        logger.warning(
            "Encountered a negative variance in expectation value calculation."
            "(%f). Setting standard deviation of result to 0.",
            variance,
        )
    calc_stddev = math.sqrt(variance) if variance > 0 else 0.0
    return [expval, calc_stddev]


@deprecate_func(
    since="1.3",
    package_name="Qiskit",
    removal_timeline="in Qiskit 2.0",
    additional_msg="The `qiskit.result.mitigation` module is deprecated in favor of "
    "the https://github.com/Qiskit/qiskit-addon-mthree package.",
)
def stddev(probs, shots):
    """Calculate stddev dict"""
    ret = {}
    for key, prob in probs.items():
        std_err = math.sqrt(prob * (1 - prob) / shots)
        ret[key] = std_err
    return ret


@deprecate_func(
    since="1.3",
    package_name="Qiskit",
    removal_timeline="in Qiskit 2.0",
    additional_msg="The `qiskit.result.mitigation` module is deprecated in favor of "
    "the https://github.com/Qiskit/qiskit-addon-mthree package.",
)
def str2diag(string):
    """Transform diagonal from a string to a numpy array"""
    chars = {
        "I": np.array([1, 1], dtype=float),
        "Z": np.array([1, -1], dtype=float),
        "0": np.array([1, 0], dtype=float),
        "1": np.array([0, 1], dtype=float),
    }
    ret = np.array([1], dtype=float)
    for i in reversed(string):
        if i not in chars:
            raise QiskitError(f"Invalid diagonal string character {i}")
        ret = np.kron(chars[i], ret)
    return ret


@deprecate_func(
    since="1.3",
    package_name="Qiskit",
    removal_timeline="in Qiskit 2.0",
    additional_msg="The `qiskit.result.mitigation` module is deprecated in favor of "
    "the https://github.com/Qiskit/qiskit-addon-mthree package.",
)
def counts_to_vector(counts: Counts, num_qubits: int) -> Tuple[np.ndarray, int]:
    """Transforms Counts to a probability vector"""
    vec = np.zeros(2**num_qubits, dtype=float)
    shots = 0
    for key, val in counts.items():
        shots += val
        vec[int(key, 2)] = val
    vec /= shots
    return vec, shots


@deprecate_func(
    since="1.3",
    package_name="Qiskit",
    removal_timeline="in Qiskit 2.0",
    additional_msg="The `qiskit.result.mitigation` module is deprecated in favor of "
    "the https://github.com/Qiskit/qiskit-addon-mthree package.",
)
def remap_qubits(
    vec: np.ndarray, num_qubits: int, qubits: Optional[List[int]] = None
) -> np.ndarray:
    """Remapping the qubits"""
    if qubits is not None:
        if len(qubits) != num_qubits:
            raise QiskitError("Num qubits does not match vector length.")
        axes = [num_qubits - 1 - i for i in reversed(np.argsort(qubits))]
        vec = np.reshape(vec, num_qubits * [2]).transpose(axes).reshape(vec.shape)
    return vec


@deprecate_func(
    since="1.3",
    package_name="Qiskit",
    removal_timeline="in Qiskit 2.0",
    additional_msg="The `qiskit.result.mitigation` module is deprecated in favor of "
    "the https://github.com/Qiskit/qiskit-addon-mthree package.",
)
def marganalize_counts(
    counts: Counts,
    qubit_index: Dict[int, int],
    qubits: Optional[List[int]] = None,
    clbits: Optional[List[int]] = None,
) -> np.ndarray:
    """Marginalization of the Counts. Verify that number of clbits equals to the number of qubits."""
    if clbits is not None:
        qubits_len = len(qubits) if not qubits is None else 0
        clbits_len = len(clbits) if not clbits is None else 0
        if clbits_len not in (0, qubits_len):
            raise QiskitError(
                f"Num qubits ({qubits_len}) does not match number of clbits ({clbits_len})."
            )
        counts = marginal_counts(counts, clbits)
    if clbits is None and qubits is not None:
        clbits = [qubit_index[qubit] for qubit in qubits]
        counts = marginal_counts(counts, clbits)
    return counts


@deprecate_func(
    since="1.3",
    package_name="Qiskit",
    removal_timeline="in Qiskit 2.0",
    additional_msg="The `qiskit.result.mitigation` module is deprecated in favor of "
    "the https://github.com/Qiskit/qiskit-addon-mthree package.",
)
def counts_probability_vector(
    counts: Counts,
    qubit_index: Dict[int, int],
    qubits: Optional[List[int]] = None,
    clbits: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int]:
    """Compute a probability vector for all count outcomes.

    Args:
        counts: counts object
        qubit_index: For each qubit, its index in the mitigator qubits list
        qubits: qubits the count bitstrings correspond to.
        clbits: Optional, marginalize counts to just these bits.

    Raises:
        QiskitError: if qubits and clbits kwargs are not valid.

    Returns:
        np.ndarray: a probability vector for all count outcomes.
        int: Number of shots in the counts
    """
    counts = marganalize_counts(counts, qubit_index, qubits, clbits)
    if qubits is not None:
        num_qubits = len(qubits)
    else:
        num_qubits = len(qubit_index.keys())
    vec, shots = counts_to_vector(counts, num_qubits)
    vec = remap_qubits(vec, num_qubits, qubits)
    return vec, shots

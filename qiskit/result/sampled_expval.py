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
# pylint: disable=cyclic-import

"""Routines for computing expectation values from sampled distributions"""
import numpy as np

from qiskit._accelerate.sampled_exp_val import sampled_expval_float, sampled_expval_complex
from qiskit.exceptions import QiskitError
from .distributions import QuasiDistribution, ProbDistribution


# A list of valid diagonal operators
OPERS = {"Z", "I", "0", "1"}


def sampled_expectation_value(dist, oper):
    """Computes expectation value from a sampled distribution

    Note that passing a raw dict requires bit-string keys.

    Parameters:
        dist (Counts or QuasiDistribution or ProbDistribution or dict): Input sampled distribution
        oper (str or Pauli or PauliOp or PauliSumOp or SparsePauliOp): The operator for
                                                                       the observable

    Returns:
        float or complex: The expectation value
    Raises:
        QiskitError: if the input distribution or operator is an invalid type
    """
    from .counts import Counts
    from qiskit.quantum_info import Pauli, SparsePauliOp

    # This should be removed when these return bit-string keys
    if isinstance(dist, (QuasiDistribution, ProbDistribution)):
        dist = dist.binary_probabilities()

    if not isinstance(dist, (Counts, dict)):
        raise QiskitError("Invalid input distribution type")
    if isinstance(oper, str):
        oper_strs = [oper.upper()]
        coeffs = np.asarray([1.0])
    elif isinstance(oper, Pauli):
        oper_strs = [oper.to_label()]
        coeffs = np.asarray([1.0])
    elif isinstance(oper, SparsePauliOp):
        oper_strs = oper.paulis.to_labels()
        coeffs = np.asarray(oper.coeffs)
    else:
        raise QiskitError("Invalid operator type")

    # Do some validation here
    bitstring_len = len(next(iter(dist)))
    if any(len(op) != bitstring_len for op in oper_strs):
        raise QiskitError(
            f"One or more operators not same length ({bitstring_len}) as input bitstrings"
        )
    for op in oper_strs:
        if set(op).difference(OPERS):
            raise QiskitError(f"Input operator {op} is not diagonal")
    # Dispatch to Rust routines
    if coeffs.dtype == np.dtype(complex).type:
        return sampled_expval_complex(oper_strs, coeffs, dist)
    else:
        return sampled_expval_float(oper_strs, coeffs, dist)

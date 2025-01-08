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

"""Module for common pulse programming utilities."""
from typing import List, Dict, Union, Sequence
import warnings

import numpy as np

from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import UnassignedDurationError, QiskitError, PulseError


def format_meas_map(meas_map: List[List[int]]) -> Dict[int, List[int]]:
    """
    Return a mapping from qubit label to measurement group given the nested list meas_map returned
    by a backend configuration. (Qubits can not always be measured independently.) Sorts the
    measurement group for consistency.

    Args:
        meas_map: Groups of qubits that get measured together, for example: [[0, 1], [2, 3, 4]]
    Returns:
        Measure map in map format
    """
    qubit_mapping = {}
    for sublist in meas_map:
        sublist.sort()
        for q in sublist:
            qubit_mapping[q] = sublist
    return qubit_mapping


def format_parameter_value(
    operand: ParameterExpression,
    decimal: int = 10,
) -> Union[ParameterExpression, complex]:
    """Convert ParameterExpression into the most suitable data type.

    Args:
        operand: Operand value in arbitrary data type including ParameterExpression.
        decimal: Number of digit to round returned value.

    Returns:
        Value cast to non-parameter data type, when possible.
    """
    if isinstance(operand, ParameterExpression):
        try:
            operand = operand.numeric()
        except TypeError:
            # Unassigned expression
            return operand

    # Return integer before calling the numpy round function.
    # The input value is multiplied by 10**decimals, rounds to an integer
    # and divided by 10**decimals. For a large enough integer,
    # this operation may introduce a rounding error in the float operations
    # and accidentally returns a float number.
    if isinstance(operand, int):
        return operand

    # Remove truncation error and convert the result into Python builtin type.
    # Value could originally contain a rounding error, e.g. 1.00000000001
    # which may occur during the parameter expression evaluation.
    evaluated = np.round(operand, decimals=decimal).item()

    if isinstance(evaluated, complex):
        if np.isclose(evaluated.imag, 0.0):
            evaluated = evaluated.real
        else:
            warnings.warn(
                "Assignment of complex values to ParameterExpression in Qiskit Pulse objects is "
                "now pending deprecation. This will align the Pulse module with other modules "
                "where such assignment wasn't possible to begin with. The typical use case for complex "
                "parameters in the module was the SymbolicPulse library. As of Qiskit-Terra "
                "0.23.0 all library pulses were converted from complex amplitude representation"
                " to real representation using two floats (amp,angle), as used in the "
                "ScalableSymbolicPulse class. This eliminated the need for complex parameters. "
                "Any use of complex parameters (and particularly custom-built pulses) should be "
                "converted in a similar fashion to avoid the use of complex parameters.",
                PendingDeprecationWarning,
            )
            return evaluated
    # Type cast integer-like float into Python builtin integer, after rounding.
    if evaluated.is_integer():
        return int(evaluated)
    return evaluated


def instruction_duration_validation(duration: int):
    """Validate instruction duration.

    Args:
        duration: Instruction duration value to validate.

    Raises:
        UnassignedDurationError: When duration is unassigned.
        QiskitError: When invalid duration is assigned.
    """
    if isinstance(duration, ParameterExpression):
        raise UnassignedDurationError(
            f"Instruction duration {repr(duration)} is not assigned. "
            "Please bind all durations to an integer value before playing in the Schedule, "
            "or use ScheduleBlock to align instructions with unassigned duration."
        )

    if not isinstance(duration, (int, np.integer)) or duration < 0:
        raise QiskitError(
            f"Instruction duration must be a non-negative integer, got {duration} instead."
        )


def _validate_parameter_vector(parameter: ParameterVector, value):
    """Validate parameter vector and its value."""
    if not isinstance(value, Sequence):
        raise PulseError(
            f"Parameter vector '{parameter.name}' has length {len(parameter)},"
            f" but was assigned to {value}."
        )
    if len(parameter) != len(value):
        raise PulseError(
            f"Parameter vector '{parameter.name}' has length {len(parameter)},"
            f" but was assigned to {len(value)} values."
        )


def _validate_single_parameter(parameter: Parameter, value):
    """Validate single parameter and its value."""
    if not isinstance(value, (int, float, complex, ParameterExpression)):
        raise PulseError(f"Parameter '{parameter.name}' is not assignable to {value}.")


def _validate_parameter_value(parameter, value):
    """Validate parameter and its value."""
    if isinstance(parameter, ParameterVector):
        _validate_parameter_vector(parameter, value)
        return True
    else:
        _validate_single_parameter(parameter, value)
        return False

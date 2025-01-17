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

"""
Delay instruction (for circuit module).
"""
import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit import _utils
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.utils import deprecate_func


@_utils.with_gate_array(np.eye(2, dtype=complex))
class Delay(Instruction):
    """Do nothing and just delay/wait/idle for a specified duration."""

    def __init__(self, duration, unit="dt"):
        """
        Args:
            duration: the length of time of the duration.  Given in units of ``unit``.
            unit: the unit of the duration.  Must be ``"dt"`` or an SI-prefixed seconds unit.
        """
        if unit not in {"s", "ms", "us", "ns", "ps", "dt"}:
            raise CircuitError(f"Unknown unit {unit} is specified.")
        # Double underscore to differentiate from the private attribute in
        # `Instruction`. This can be changed to `_unit` in 2.0 after we
        # remove `unit` and `duration` from the standard instruction model
        # as it only will exist in `Delay` after that point.
        self.__unit = unit
        super().__init__("delay", 1, 0, params=[duration])

    broadcast_arguments = Gate.broadcast_arguments

    def inverse(self, annotated: bool = False):
        """Special case. Return self."""
        return self

    @deprecate_func(since="1.3.0", removal_timeline="in 2.0.0")
    def c_if(self, classical, val):
        raise CircuitError("Conditional Delay is not yet implemented.")

    @property
    def unit(self):

        return self.__unit

    @unit.setter
    def unit(self, value):
        if value not in {"s", "ms", "us", "ns", "ps", "dt"}:
            raise CircuitError(f"Unknown unit {value} is specified.")
        self.__unit = value

    @property
    def duration(self):
        """Get the duration of this delay."""
        return self.params[0]

    @duration.setter
    def duration(self, duration):
        """Set the duration of this delay."""
        self.params = [duration]

    def to_matrix(self) -> np.ndarray:
        """Return a Numpy.array for the unitary matrix. This has been
        added to enable simulation without making delay a full Gate type.

        Returns:
            np.ndarray: matrix representation.
        """
        return self.__array__(dtype=complex)

    def __repr__(self):
        """Return the official string representing the delay."""
        return f"{self.__class__.__name__}(duration={self.params[0]}[unit={self.unit}])"

    def validate_parameter(self, parameter):
        """Delay parameter (i.e. duration) must be int, float or ParameterExpression."""
        if isinstance(parameter, int):
            if parameter < 0:
                raise CircuitError(
                    f"Duration for Delay instruction must be positive. Found {parameter}"
                )
            return parameter
        elif isinstance(parameter, float):
            if parameter < 0:
                raise CircuitError(
                    f"Duration for Delay instruction must be positive. Found {parameter}"
                )
            if self.__unit == "dt":
                parameter_int = int(parameter)
                if parameter != parameter_int:
                    raise CircuitError("Integer duration is expected for 'dt' unit.")
                return parameter_int
            return parameter
        elif isinstance(parameter, ParameterExpression):
            if len(parameter.parameters) > 0:
                return parameter  # expression has free parameters, we cannot validate it
            if not parameter.is_real():
                raise CircuitError(f"Bound parameter expression is complex in delay {self.name}")
            fval = float(parameter)
            if fval < 0:
                raise CircuitError(f"Duration for Delay instruction must be positive. Found {fval}")
            if self.unit == "dt":
                ival = int(parameter)
                rounding_error = abs(fval - ival)
                if rounding_error > 1e-15:
                    raise CircuitError("Integer parameter is required for duration in 'dt' unit.")
                return ival
            return fval  # per default assume parameters must be real when bound
        else:
            raise CircuitError(f"Invalid param type {type(parameter)} for delay {self.name}.")

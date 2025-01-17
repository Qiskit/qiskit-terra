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

"""Reference instruction that is a placeholder for subroutine."""
from __future__ import annotations

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError, UnassignedReferenceError
from qiskit.pulse.instructions import instruction
from qiskit.utils.deprecate_pulse import deprecate_pulse_func


class Reference(instruction.Instruction):
    """Pulse compiler directive that refers to a subroutine.

    If a pulse program uses the same subset of instructions multiple times, then
    using the :class:`~.Reference` class may significantly reduce the memory footprint of
    the program. This instruction only stores the set of strings to identify the subroutine.

    The actual pulse program can be stored in the :attr:`ScheduleBlock.references` of the
    :class:`.ScheduleBlock` that this reference instruction belongs to.

    You can later assign schedules with the :meth:`ScheduleBlock.assign_references` method.
    This allows you to build the main program without knowing the actual subroutine,
    that is supplied at a later time.
    """

    # Delimiter for representing nested scope.
    scope_delimiter = "::"

    # Delimiter for tuple keys.
    key_delimiter = ","

    @deprecate_pulse_func
    def __init__(self, name: str, *extra_keys: str):
        """Create new reference.

        Args:
            name: Name of subroutine.
            extra_keys: Optional. A set of string keys that may be necessary to
                refer to a particular subroutine. For example, when we use
                "sx" as a name to refer to the subroutine of an sx pulse,
                this name might be used among schedules for different qubits.
                In this example, you may specify "q0" in the extra keys
                to distinguish the sx schedule for qubit 0 from others.
                The user can use an arbitrary number of extra string keys to
                uniquely determine the subroutine.
        """
        # Run validation
        ref_keys = (name,) + tuple(extra_keys)
        super().__init__(operands=ref_keys, name=name)

    def _validate(self):
        """Called after initialization to validate instruction data.

        Raises:
            PulseError: When a key is not a string.
            PulseError: When a key in ``ref_keys`` contains the scope delimiter.
        """
        for key in self.ref_keys:
            if not isinstance(key, str):
                raise PulseError(f"Keys must be strings. '{repr(key)}' is not a valid object.")
            if self.scope_delimiter in key or self.key_delimiter in key:
                raise PulseError(
                    f"'{self.scope_delimiter}' and '{self.key_delimiter}' are reserved. "
                    f"'{key}' is not a valid key string."
                )

    @property
    def ref_keys(self) -> tuple[str, ...]:
        """Returns unique key of the subroutine."""
        return self.operands

    @property
    def duration(self) -> int | ParameterExpression:
        """Duration of this instruction."""
        raise UnassignedReferenceError(f"Subroutine is not assigned to {self.ref_keys}.")

    @property
    def channels(self) -> tuple[Channel, ...]:
        """Returns the channels that this schedule uses."""
        raise UnassignedReferenceError(f"Subroutine is not assigned to {self.ref_keys}.")

    @property
    def parameters(self) -> set:
        """Parameters which determine the instruction behavior."""
        return set()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.key_delimiter.join(self.ref_keys)})"

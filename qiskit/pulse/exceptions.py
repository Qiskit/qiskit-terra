# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exception for errors raised by the pulse module."""
from qiskit.exceptions import QiskitError
from qiskit.utils.deprecate_pulse import deprecate_pulse_func


class PulseError(QiskitError):
    """Errors raised by the pulse module."""

    @deprecate_pulse_func
    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class BackendNotSet(PulseError):
    """Raised if the builder context does not have a backend."""


class NoActiveBuilder(PulseError):
    """Raised if no builder context is active."""


class UnassignedDurationError(PulseError):
    """Raised if instruction duration is unassigned."""


class UnassignedReferenceError(PulseError):
    """Raised if subroutine is unassigned."""

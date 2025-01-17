# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A simulator instruction to capture output within a simulation. The types of snapshot
instructions available are determined by the simulator being used.
"""
from typing import Optional, Tuple

from qiskit.pulse.channels import SnapshotChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.utils.deprecate_pulse import deprecate_pulse_func


class Snapshot(Instruction):
    """An instruction targeted for simulators, to capture a moment in the simulation."""

    @deprecate_pulse_func
    def __init__(self, label: str, snapshot_type: str = "statevector", name: Optional[str] = None):
        """Create new snapshot.

        Args:
            label: Snapshot label which is used to identify the snapshot in the output.
            snapshot_type: Type of snapshot, e.g., “state” (take a snapshot of the quantum state).
                           The types of snapshots offered are defined by the simulator used.
            name: Snapshot name which defaults to ``label``. This parameter is only for display
                  purposes and is not taken into account during comparison.
        """
        self._channel = SnapshotChannel()

        if name is None:
            name = label
        super().__init__(operands=(label, snapshot_type), name=name)

    def _validate(self):
        """Called after initialization to validate instruction data.

        Raises:
            PulseError: If snapshot label is invalid.
        """
        if not isinstance(self.label, str):
            raise PulseError("Snapshot label must be a string.")

    @property
    def label(self) -> str:
        """Label of snapshot."""
        return self.operands[0]

    @property
    def type(self) -> str:
        """Type of snapshot."""
        return self.operands[1]

    @property
    def channel(self) -> SnapshotChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on; trivially, a ``SnapshotChannel``.
        """
        return self._channel

    @property
    def channels(self) -> Tuple[SnapshotChannel]:
        """Returns the channels that this schedule uses."""
        return (self.channel,)

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return 0

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return False

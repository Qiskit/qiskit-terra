# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A qubit state tracker for synthesizing operations with auxiliary qubits."""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class QubitTracker:
    """Track qubits (by global index) and their state.

    The states are distinguished into clean (meaning in state :math:`|0\rangle`) or dirty (an
    unknown state).
    """

    # This could in future be extended to track different state types, if necessary.
    # However, using sets of integers here is much faster than e.g. storing a dictionary with
    # {index: state} entries.
    qubits: tuple[int]
    clean: set[int]
    dirty: set[int]

    def num_clean(self, active_qubits: Iterable[int] | None = None):
        """Return the number of clean qubits, not considering the active qubits."""
        # this could be cached if getting the set length becomes a performance bottleneck
        return len(self.clean.difference(active_qubits or set()))

    def num_dirty(self, active_qubits: Iterable[int] | None = None):
        """Return the number of dirty qubits, not considering the active qubits."""
        return len(self.dirty.difference(active_qubits or set()))

    def borrow(self, num_qubits: int, active_qubits: Iterable[int] | None = None) -> list[int]:
        """Get ``num_qubits`` qubits, excluding ``active_qubits``."""
        active_qubits = set(active_qubits or [])
        available_qubits = [qubit for qubit in self.qubits if qubit not in active_qubits]

        if num_qubits > (available := len(available_qubits)):
            raise RuntimeError(f"Cannot borrow {num_qubits} qubits, only {available} available.")

        # for now, prioritize returning clean qubits
        available_clean = [qubit for qubit in available_qubits if qubit in self.clean]
        available_dirty = [qubit for qubit in available_qubits if qubit in self.dirty]

        borrowed = available_clean[:num_qubits]
        return borrowed + available_dirty[: (num_qubits - len(borrowed))]

    def used(self, qubits: Iterable[int], check: bool = True) -> None:
        """Set the state of ``qubits`` to used (i.e. False)."""
        qubits = set(qubits)

        if check:
            if len(untracked := qubits.difference(self.qubits)) > 0:
                raise ValueError(f"Setting state of untracked qubits: {untracked}. Tracker: {self}")

        self.clean -= qubits
        self.dirty |= qubits

    def reset(self, qubits: Iterable[int], check: bool = True) -> None:
        """Set the state of ``qubits`` to 0 (i.e. True)."""
        qubits = set(qubits)

        if check:
            if len(untracked := qubits.difference(self.qubits)) > 0:
                raise ValueError(f"Setting state of untracked qubits: {untracked}. Tracker: {self}")

        self.clean |= qubits
        self.dirty -= qubits

    def drop(self, qubits: Iterable[int], check: bool = True) -> None:
        """Drop qubits from the tracker, meaning that they are no longer available."""
        qubits = set(qubits)

        if check:
            if len(untracked := qubits.difference(self.qubits)) > 0:
                raise ValueError(f"Dropping untracked qubits: {untracked}. Tracker: {self}")

        self.qubits = tuple(qubit for qubit in self.qubits if qubit not in qubits)
        self.clean -= qubits
        self.dirty -= qubits

    def copy(
        self, qubit_map: dict[int, int] | None = None, drop: Iterable[int] | None = None
    ) -> "QubitTracker":
        """Copy self.

        Args:
            qubit_map: If provided, apply the mapping ``{old_qubit: new_qubit}`` to
                the qubits in the tracker. Only those old qubits in the mapping will be
                part of the new one.
            drop: If provided, drop these qubits in the copied tracker. This argument is ignored
                if ``qubit_map`` is given, since the qubits can then just be dropped in the map.
        """
        if qubit_map is None and drop is not None:
            remaining_qubits = [qubit for qubit in self.qubits if qubit not in drop]
            qubit_map = dict(zip(remaining_qubits, remaining_qubits))

        if qubit_map is None:
            clean = self.clean.copy()
            dirty = self.dirty.copy()
            qubits = self.qubits  # tuple is immutable, no need to copy
        else:
            clean, dirty = set(), set()
            for old_index, new_index in qubit_map.items():
                if old_index in self.clean:
                    clean.add(new_index)
                elif old_index in self.dirty:
                    dirty.add(new_index)
                else:
                    raise ValueError(f"Unknown old qubit index: {old_index}. Tracker: {self}")

            qubits = tuple(qubit_map.values())

        return QubitTracker(qubits, clean=clean, dirty=dirty)

    def __str__(self) -> str:
        return (
            f"QubitTracker({len(self.qubits)}, clean: {self.num_clean()}, dirty: {self.num_dirty()})"
            + f"\n\tclean: {self.clean}"
            + f"\n\tdirty: {self.dirty}"
        )

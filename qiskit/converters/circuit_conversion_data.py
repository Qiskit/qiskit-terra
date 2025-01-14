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

"""Stores optional conversion data when switching between different circuit formats."""


class CircuitConversionData:
    """Conversion data when switching between different circuit formats."""

    def __init__(self):
        self._fwd_map = {}
        self._bwd_map = {}

    def store_mapping(self, from_node, to_node):
        """Stores that ``from_node`` is mapped to ``to_node``."""
        self._fwd_map[from_node] = to_node
        self._bwd_map[to_node] = from_node

    def forward_map(self, from_node):
        """Returns the node ``from_node`` was mapped to, or ``None`` if not present."""
        return self._fwd_map.get(from_node, None)

    def backward_map(self, to_node):
        """Returns the node that was mapped to ``to_node``, or ``None`` if not present."""
        return self._bwd_map.get(to_node, None)

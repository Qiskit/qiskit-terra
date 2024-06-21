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

"""Utility functions for handling permutations."""

# pylint: disable=unused-import
from qiskit._accelerate.permutation import (
    _inverse_pattern,
    _get_ordered_swap,
    _validate_permutation,
)


def _pattern_to_cycles(pattern):
    """Given a permutation pattern, creates its disjoint cycle decomposition."""
    nq = len(pattern)
    explored = [False] * nq
    cycles = []
    for i in pattern:
        cycle = []
        while not explored[i]:
            cycle.append(i)
            explored[i] = True
            i = pattern[i]
        if len(cycle) >= 2:
            cycles.append(cycle)
    return cycles


def _decompose_cycles(cycles):
    """Given a disjoint cycle decomposition, decomposes every cycle into a SWAP
    circuit of depth 2."""
    swap_list = []
    for cycle in cycles:
        m = len(cycle)
        for i in range((m - 1) // 2):
            swap_list.append((cycle[i - 1], cycle[m - 3 - i]))
        for i in range(m // 2):
            swap_list.append((cycle[i - 1], cycle[m - 2 - i]))
    return swap_list


def _compose_permutations(*perms):
    """Compose multiple permutations, with the permutations applied in the
    order they appear in the list.
    """
    out = range(len(perms[0]))
    for perm in perms:
        out = [perm[i] for i in out]
    return out

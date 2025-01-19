# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions to generate the basic approximations of single qubit gates for Solovay-Kitaev."""

from __future__ import annotations

import warnings
import collections
import numpy as np

import qiskit.circuit.library.standard_gates as gates
from qiskit.circuit import Gate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.utils import optionals

from .gate_sequence import GateSequence

Node = collections.namedtuple("Node", ("labels", "sequence", "children"))

_1q_inverses = {
    "i": "i",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "t": "tdg",
    "tdg": "t",
    "s": "sdg",
    "sdg": "s",
    "sx": "sxdg",
    "sxdg": "sx",
}

_1q_gates = {
    "i": gates.IGate(),
    "x": gates.XGate(),
    "y": gates.YGate(),
    "z": gates.ZGate(),
    "h": gates.HGate(),
    "t": gates.TGate(),
    "tdg": gates.TdgGate(),
    "s": gates.SGate(),
    "sdg": gates.SdgGate(),
    "sx": gates.SXGate(),
    "sxdg": gates.SXdgGate(),
}


def _check_candidate(candidate, existing_sequences, tol=1e-10):
    if optionals.HAS_SKLEARN:
        return _check_candidate_kdtree(candidate, existing_sequences, tol)

    warnings.warn(
        "The SolovayKitaev algorithm relies on scikit-learn's KDTree for a "
        "fast search over the basis approximations. Without this, we fallback onto a "
        "greedy search with is significantly slower. We highly suggest to install "
        "scikit-learn to use this feature.",
        category=RuntimeWarning,
    )
    return _check_candidate_greedy(candidate, existing_sequences, tol)


def _check_candidate_greedy(candidate, existing_sequences, tol=1e-10):
    # do a quick, string-based check if the same sequence already exists
    if any(candidate.name == existing.name for existing in existing_sequences):
        return False

    for existing in existing_sequences:
        if matrix_equal(existing.product_su2, candidate.product_su2, ignore_phase=True, atol=tol):
            # is the new sequence less or more efficient?
            return len(candidate.gates) < len(existing.gates)
    return True


@optionals.HAS_SKLEARN.require_in_call
def _check_candidate_kdtree(candidate, existing_sequences, tol=1e-10):
    """Check if there's a candidate implementing the same matrix up to ``tol``.

    This uses a k-d tree search and is much faster than the greedy, list-based search.
    """
    from sklearn.neighbors import KDTree

    # do a quick, string-based check if the same sequence already exists
    if any(candidate.name == existing.name for existing in existing_sequences):
        return False

    points = np.array([sequence.product.flatten() for sequence in existing_sequences])
    candidate = np.array([candidate.product.flatten()])

    kdtree = KDTree(points)
    dist, _ = kdtree.query(candidate)

    return dist[0][0] > tol


def _process_node(node: Node, basis: list[str], sequences: list[GateSequence]):
    inverse_last = _1q_inverses[node.labels[-1]] if node.labels else None

    for label in basis:
        if label == inverse_last:
            continue

        sequence = node.sequence.copy()
        sequence.append(_1q_gates[label])

        if _check_candidate(sequence, sequences):
            sequences.append(sequence)
            node.children.append(Node(node.labels + (label,), sequence, []))

    return node.children


def generate_basic_approximations(
    basis_gates: list[str | Gate], depth: int, filename: str | None = None
) -> list[GateSequence]:
    """Generates a list of :class:`GateSequence`\\ s with the gates in ``basis_gates``.

    Args:
        basis_gates: The gates from which to create the sequences of gates.
        depth: The maximum depth of the approximations.
        filename: If provided, the basic approximations are stored in this file.

    Returns:
        List of :class:`GateSequence`\\ s using the gates in ``basis_gates``.

    Raises:
        ValueError: If ``basis_gates`` contains an invalid gate identifier.
    """
    basis = []
    for gate in basis_gates:
        if isinstance(gate, str):
            if gate not in _1q_gates:
                raise ValueError(f"Invalid gate identifier: {gate}")
            basis.append(gate)
        else:  # gate is a qiskit.circuit.Gate
            basis.append(gate.name)

    tree = Node((), GateSequence(), [])
    cur_level = [tree]
    sequences = [tree.sequence]
    for _ in [None] * depth:
        next_level = []
        for node in cur_level:
            next_level.extend(_process_node(node, basis, sequences))
        cur_level = next_level

    if filename is not None:
        data = {}
        for sequence in sequences:
            gatestring = sequence.name
            data[gatestring] = (sequence.product, sequence.global_phase)

        np.save(filename, data)

    return sequences

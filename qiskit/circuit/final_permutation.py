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

"""Reasoning about the implicit permutation of output qubits."""

from qiskit.circuit.exceptions import CircuitError


class FinalPermutation:
    r"""
    Reasons about the implicit permutation of output qubits.

    The notation use here is the same as for the class :class:`~PermutationGate`:
    the permutation is stored as a list describing which qubits occupy the
    positions 0, 1, 2, etc. after applying the permutation. As an example,
    the permutation ``[2, 4, 3, 0, 1]`` means that the qubit ``2`` goes to
    position ``0``, qubit ``4`` goes to the position ``1``, and so on.
    In particular, a circuit with an implicit permutation :math:`\sigma`
    can be replaced by a :class:`~PermutationGate` with the same permutation
    pattern :math:`\sigma`.
    """

    def __init__(self, permutation=None):
        """Initializer."""
        if permutation is None:
            permutation = []
        self.permutation = permutation

    def add_qubit(self):
        """Extends the permutation when a new qubit is added to a DAGCircuit or to
        a QuantumCircuit."""
        self.permutation.append(len(self.permutation))

    def num_qubits(self) -> int:
        """Returns the length of the permutation (i.e. the total number of qubits)."""
        return len(self.permutation)

    def is_identity(self) -> bool:
        """Returns whether the permutation is the identity permutation."""
        return all(from_index == to_index for from_index, to_index in enumerate(self.permutation))

    def compose_with_permutation(self, permutation, front) -> "FinalPermutation":
        """Composes FinalPermution with a permutation."""
        # pylint: disable=cyclic-import
        from qiskit.synthesis.permutation.permutation_utils import _compose_permutations

        if front:
            composed_permutation = _compose_permutations(self.permutation, permutation)
        else:
            composed_permutation = _compose_permutations(permutation, self.permutation)
        return FinalPermutation(composed_permutation)

    def __repr__(self):
        return str(self.permutation)

    def copy(self) -> "FinalPermutation":
        """Creates a copy of the FinalPermutation object."""
        return FinalPermutation(self.permutation.copy())

    def push_using_mapping(self, forward_map, num_target_qubits=None) -> "FinalPermutation":
        r"""
        Applies a layout mapping (or more generally any mapping) to a
        permutation.

        More precisely, given a permutation :math:`\sigma: V \rightarrow V`,
        and a map :math:`\tau: V \rightarrow P`, returns a permutation
        :math:`\tilde{\sigma}: P\rightarrow P`, where
        :math:`\tilde{\sigma}` maps  :math:`\tau(a)` to :math:`\tau(b)`
        whenever :math:`\sigma` maps :math:`a` to :math:`b`, and
        :math:`\tilde{\sigma}` is identity on the remaining elements.
        """

        if num_target_qubits is None:
            num_target_qubits = len(forward_map)

        target_permutation = list(range(num_target_qubits))

        if isinstance(forward_map, list):
            for inp, out in enumerate(forward_map):
                target_permutation[out] = forward_map[self.permutation[inp]]
        elif isinstance(forward_map, dict):
            for inp, out in forward_map.items():
                target_permutation[out] = forward_map[self.permutation[inp]]
        else:
            raise CircuitError("The map should be given either as a list or as a dict.")

        return FinalPermutation(target_permutation)

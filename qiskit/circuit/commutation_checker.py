# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Code from commutative_analysis pass that checks commutation relations between DAG nodes."""

from typing import List, Union, Set, Optional

from qiskit.circuit.operation import Operation
from qiskit._accelerate.commutation_checker import CommutationChecker as RustChecker


class CommutationChecker:
    """This code is essentially copy-pasted from commutative_analysis.py.
    This code cleverly hashes commutativity and non-commutativity results between DAG nodes and seems
    quite efficient for large Clifford circuits.
    They may be other possible efficiency improvements: using rule-based commutativity analysis,
    evicting from the cache less useful entries, etc.
    """

    def __init__(
        self,
        standard_gate_commutations: dict = None,
        cache_max_entries: int = 10**6,
        *,
        gates: Optional[Set[str]] = None,
    ):
        self.cc = RustChecker(standard_gate_commutations, cache_max_entries, gates)

    def commute_nodes(
        self,
        op1,
        op2,
        max_num_qubits: int = 3,
    ) -> bool:
        """Checks if two DAGOpNodes commute."""
        return self.cc.commute_nodes(op1, op2, max_num_qubits)

    def commute(
        self,
        op1: Operation,
        qargs1: List,
        cargs1: List,
        op2: Operation,
        qargs2: List,
        cargs2: List,
        max_num_qubits: int = 3,
    ) -> bool:
        """
        Checks if two Operations commute. The return value of `True` means that the operations
        truly commute, and the return value of `False` means that either the operations do not
        commute or that the commutation check was skipped (for example, when the operations
        have conditions or have too many qubits).

        Args:
            op1: first operation.
            qargs1: first operation's qubits.
            cargs1: first operation's clbits.
            op2: second operation.
            qargs2: second operation's qubits.
            cargs2: second operation's clbits.
            max_num_qubits: the maximum number of qubits to consider, the check may be skipped if
                the number of qubits for either operation exceeds this amount.

        Returns:
            bool: whether two operations commute.
        """
        return self.cc.commute(op1, qargs1, cargs1, op2, qargs2, cargs2, max_num_qubits)

    def num_cached_entries(self):
        """Returns number of cached entries"""
        return self.cc.num_cached_entries()

    def clear_cached_commutations(self):
        """Clears the dictionary holding cached commutations"""
        self.cc.clear_cached_commutations()

    def check_commutation_entries(
        self,
        first_op: Operation,
        first_qargs: List,
        second_op: Operation,
        second_qargs: List,
    ) -> Union[bool, None]:
        """Returns stored commutation relation if any

        Args:
            first_op: first operation.
            first_qargs: first operation's qubits.
            second_op: second operation.
            second_qargs: second operation's qubits.

        Return:
            bool: True if the gates commute and false if it is not the case.
        """
        return self.cc.library.check_commutation_entries(
            first_op, first_qargs, second_op, second_qargs
        )

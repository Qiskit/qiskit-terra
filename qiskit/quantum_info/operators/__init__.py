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

"""Quantum Operators."""

from __future__ import annotations
from .channel import PTM, Chi, Choi, Kraus, Stinespring, SuperOp
from .dihedral import CNOTDihedral
from .measures import (
    average_gate_fidelity,
    diamond_norm,
    diamond_distance,
    gate_error,
    process_fidelity,
)
from .operator import Operator
from .scalar_op import ScalarOp
from .symplectic import (
    Clifford,
    Pauli,
    PauliList,
    SparsePauliOp,
    pauli_basis,
)
from .utils import anti_commutator, commutator, double_commutator

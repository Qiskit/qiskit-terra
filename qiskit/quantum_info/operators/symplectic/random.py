# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Random symplectic operator functions
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng

from qiskit._accelerate.synthesis.clifford import random_clifford_tableau

from .clifford import Clifford
from .pauli import Pauli
from .pauli_list import PauliList


def random_pauli(
    num_qubits: int, group_phase: bool = False, seed: int | np.random.Generator | None = None
):
    """Return a random Pauli.

    Args:
        num_qubits (int): the number of qubits.
        group_phase (bool): Optional. If True generate random phase.
                            Otherwise the phase will be set so that the
                            Pauli coefficient is +1 (default: False).
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Pauli: a random Pauli
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)
    z = rng.integers(2, size=num_qubits, dtype=bool)
    x = rng.integers(2, size=num_qubits, dtype=bool)
    phase = rng.integers(4) if group_phase else 0
    pauli = Pauli((z, x, phase))
    return pauli


def random_pauli_list(
    num_qubits: int,
    size: int = 1,
    seed: int | np.random.Generator | None = None,
    phase: bool = True,
):
    """Return a random PauliList.

    Args:
        num_qubits (int): the number of qubits.
        size (int): Optional. The length of the Pauli list (Default: 1).
        seed (int or np.random.Generator): Optional. Set a fixed seed or generator for RNG.
        phase (bool): If True the Pauli phases are randomized, otherwise the phases are fixed to 0.
                     [Default: True]

    Returns:
        PauliList: a random PauliList.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    z = rng.integers(2, size=(size, num_qubits)).astype(bool)
    x = rng.integers(2, size=(size, num_qubits)).astype(bool)
    if phase:
        _phase = rng.integers(4, size=size)
        return PauliList.from_symplectic(z, x, _phase)
    return PauliList.from_symplectic(z, x)


def random_clifford(num_qubits: int, seed: int | np.random.Generator | None = None):
    """Return a random Clifford operator.

    The Clifford is sampled using the method of Reference [1].

    Args:
        num_qubits (int): the number of qubits for the Clifford
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Clifford: a random Clifford operator.

    Reference:
        1. S. Bravyi and D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*.
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    seed = rng.integers(100000, size=1, dtype=np.uint64)[0]
    tableau = random_clifford_tableau(num_qubits, seed=seed)
    return Clifford(tableau, validate=False)

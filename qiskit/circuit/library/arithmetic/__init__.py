# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The arithmetic circuit library."""

from .functional_pauli_rotations import FunctionalPauliRotations
from .integer_comparator import IntegerComparator
from .linear_pauli_rotations import LinearPauliRotations
from .piecewise_linear_pauli_rotations import PiecewiseLinearPauliRotations
from .piecewise_polynomial_pauli_rotations import PiecewisePolynomialPauliRotations
from .polynomial_pauli_rotations import PolynomialPauliRotations
from .weighted_adder import WeightedAdder
from .quadratic_form import QuadraticForm
from .linear_amplitude_function import LinearAmplitudeFunction
from .adders import (
    VBERippleCarryAdder,
    CDKMRippleCarryAdder,
    DraperQFTAdder,
    ModularAdderGate,
    HalfAdderGate,
    FullAdderGate,
)
from .piecewise_chebyshev import PiecewiseChebyshev
from .multipliers import HRSCumulativeMultiplier, RGQFTMultiplier, MultiplierGate
from .exact_reciprocal import ExactReciprocal

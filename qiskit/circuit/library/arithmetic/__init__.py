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
from .integer_comparator import IntegerComparator, IntegerComparatorGate
from .linear_pauli_rotations import LinearPauliRotations, LinearPauliRotationsGate
from .piecewise_linear_pauli_rotations import (
    PiecewiseLinearPauliRotations,
    PiecewiseLinearPauliRotationsGate,
)
from .piecewise_polynomial_pauli_rotations import (
    PiecewisePolynomialPauliRotations,
    PiecewisePolynomialPauliRotationsGate,
)
from .polynomial_pauli_rotations import PolynomialPauliRotations, PolynomialPauliRotationsGate
from .weighted_adder import WeightedAdder
from .quadratic_form import QuadraticForm, QuadraticFormGate
from .linear_amplitude_function import LinearAmplitudeFunction, LinearAmplitudeFunctionGate
from .adders import VBERippleCarryAdder, CDKMRippleCarryAdder, DraperQFTAdder
from .piecewise_chebyshev import PiecewiseChebyshev, PiecewiseChebyshevGate
from .multipliers import HRSCumulativeMultiplier, RGQFTMultiplier
from .exact_reciprocal import ExactReciprocal, ExactReciprocalGate

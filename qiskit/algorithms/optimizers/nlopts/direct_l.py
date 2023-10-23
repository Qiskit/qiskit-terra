# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""DIviding RECTangles Locally-biased optimizer."""

from .nloptimizer import NLoptOptimizer, NLoptOptimizerType


class DIRECT_L(NLoptOptimizer):  # pylint: disable=invalid-name
    """
    DIviding RECTangles Locally-biased optimizer.

    DIviding RECTangles (DIRECT) is a deterministic-search algorithms based on
    systematic division of the search domain into increasingly smaller
    hyper-rectangles.
    The DIRECT-L version is a "locally biased" variant of DIRECT that makes the
    algorithm more biased towards local search, so that it is more efficient
    for functions with few local minima.

    NLopt global optimizer, derivative-free.
    For further detail, please refer to
    http://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#direct-and-direct-l
    """

    def get_nlopt_optimizer(self) -> NLoptOptimizerType:
        """
        Return NLopt optimizer type
        """
        return NLoptOptimizerType.GN_DIRECT_L

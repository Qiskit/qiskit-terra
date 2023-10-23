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

"""Controlled Random Search (CRS) with local mutation optimizer."""

from .nloptimizer import NLoptOptimizer, NLoptOptimizerType


class CRS(NLoptOptimizer):
    """
    Controlled Random Search (CRS) with local mutation optimizer.

    Controlled Random Search (CRS) with local mutation is part of the family of the
    CRS optimizers.
    The CRS optimizers start with a random population of points, and randomly
    evolve these points by heuristic rules. In the case of CRS with local
    mutation, the evolution is a randomized version of the
    :class:`NELDER_MEAD` local optimizer.


    NLopt global optimizer, derivative-free.
    For further detail, please refer to
    https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#controlled-random-search-crs-with-local-mutation
    """

    def get_nlopt_optimizer(self) -> NLoptOptimizerType:
        """Return NLopt optimizer type"""
        return NLoptOptimizerType.GN_CRS2_LM

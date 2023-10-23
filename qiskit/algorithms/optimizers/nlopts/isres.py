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

"""Improved Stochastic Ranking Evolution Strategy optimizer."""

from .nloptimizer import NLoptOptimizer, NLoptOptimizerType


class ISRES(NLoptOptimizer):
    """
    Improved Stochastic Ranking Evolution Strategy optimizer.

    Improved Stochastic Ranking Evolution Strategy (ISRES) is an algorithm for
    non-linearly constrained global optimization. It has heuristics to escape
    local optima, even though convergence to a global optima is not
    guaranteed. The evolution strategy is based on a combination of a mutation
    rule and differential variation. The fitness ranking is simply via the
    objective function for problems without nonlinear constraints. When
    nonlinear constraints are included, the `stochastic ranking proposed by
    Runarsson and Yao
    <https://notendur.hi.is/tpr/software/sres/Tec311r.pdf>`__
    is employed. This method supports arbitrary nonlinear inequality and
    equality constraints, in addition to the bound constraints.

    NLopt global optimizer, derivative-free.
    For further detail, please refer to
    http://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#isres-improved-stochastic-ranking-evolution-strategy
    """

    def get_nlopt_optimizer(self) -> NLoptOptimizerType:
        """Return NLopt optimizer type"""
        return NLoptOptimizerType.GN_ISRES

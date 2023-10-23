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

"""Minimize using objective function"""
from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from abc import abstractmethod
import logging
import numpy as np

from qiskit.utils import optionals as _optionals
from ..optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT

logger = logging.getLogger(__name__)


class NLoptOptimizerType(Enum):
    """NLopt Valid Optimizer"""

    GN_CRS2_LM = 1
    GN_DIRECT_L_RAND = 2
    GN_DIRECT_L = 3
    GN_ESCH = 4
    GN_ISRES = 5


@_optionals.HAS_NLOPT.require_in_instance
class NLoptOptimizer(Optimizer):
    """
    NLopt global optimizer base class
    """

    _OPTIONS = ["max_evals"]

    def __init__(self, max_evals: int = 1000) -> None:  # pylint: disable=unused-argument
        """
        Initialize the optimizer with the maximum number of function evaluations.

        Args:
            max_evals: Maximum allowed number of function evaluations.

        Raises:
            MissingOptionalLibraryError: NLopt library not installed.
        """
        import nlopt

        super().__init__()
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                self._options[k] = v

        self._optimizer_names = {
            NLoptOptimizerType.GN_CRS2_LM: nlopt.GN_CRS2_LM,
            NLoptOptimizerType.GN_DIRECT_L_RAND: nlopt.GN_DIRECT_L_RAND,
            NLoptOptimizerType.GN_DIRECT_L: nlopt.GN_DIRECT_L,
            NLoptOptimizerType.GN_ESCH: nlopt.GN_ESCH,
            NLoptOptimizerType.GN_ISRES: nlopt.GN_ISRES,
        }

    @abstractmethod
    def get_nlopt_optimizer(self) -> NLoptOptimizerType:
        """
        Return the NLopt optimizer enum type.
        """
        raise NotImplementedError

    def get_support_level(self):
        """
        Return the support level dictionary.
        """
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.supported,
            "initial_point": OptimizerSupportLevel.required,
        }

    @property
    def settings(self):
        return {"max_evals": self._options.get("max_evals", 1000)}

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        import nlopt

        x0 = np.asarray(x0)

        if bounds is None:
            bounds = [(None, None)] * x0.size

        threshold = 3 * np.pi
        low = [(l if l is not None else -threshold) for (l, u) in bounds]
        high = [(u if u is not None else threshold) for (l, u) in bounds]

        name = self._optimizer_names[self.get_nlopt_optimizer()]
        opt = nlopt.opt(name, len(low))
        logger.debug(opt.get_algorithm_name())

        opt.set_lower_bounds(low)
        opt.set_upper_bounds(high)

        eval_count = 0

        def wrap_objfunc_global(x, _grad):
            nonlocal eval_count
            eval_count += 1
            return fun(x)

        opt.set_min_objective(wrap_objfunc_global)
        opt.set_maxeval(self._options.get("max_evals", 1000))

        xopt = opt.optimize(x0)
        minf = opt.last_optimum_value()

        logger.debug("Global minimize found %s eval count %s", minf, eval_count)

        result = OptimizerResult()
        result.x = xopt
        result.fun = minf
        result.nfev = eval_count

        return result

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Bound Optimization BY Quadratic Approximation (BOBYQA) optimizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from qiskit.utils import optionals as _optionals
from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT


@_optionals.HAS_SKQUANT.require_in_instance
class BOBYQA(Optimizer):
    """Bound Optimization BY Quadratic Approximation algorithm.

    BOBYQA finds local solutions to nonlinear, non-convex minimization problems
    with optional bound constraints, without requirement of derivatives of the
    objective function.

    Uses skquant.opt installed with pip install scikit-quant.
    For further detail, please refer to
    https://github.com/scikit-quant/scikit-quant and https://qat4chem.lbl.gov/software.
    """

    def __init__(
        self,
        maxiter: int = 1000,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of function evaluations.

        Raises:
            MissingOptionalLibraryError: scikit-quant not installed
        """
        super().__init__()
        self._maxiter = maxiter

    def get_support_level(self):
        """Returns support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.required,
            "initial_point": OptimizerSupportLevel.required,
        }

    @property
    def settings(self) -> dict[str, Any]:
        """
        Returns a dictionary with the maximum number of iterations.

        Returns:
            dict[str, Any]: A dictionary containing the maximum number of iterations.
        """
        return {"maxiter": self._maxiter}

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """
        Minimize a function using the 'bobyqa' method.

        This function takes in a function 'fun', an initial point 'x0', a Jacobian
        function 'jac', and a list of bounds. It uses the 'skquant' library to
        perform minimization using the 'bobyqa' method. The result of the
        optimization is then extracted and stored in an 'OptimizerResult' object,
        which is returned.

        Args:
            fun: The objective function to be minimized.
            x0: The initial point for the optimization.
            jac: The Jacobian function to be used for optimization (optional).
            bounds: A list of bounds for the optimization variables (optional).

        Returns:
            OptimizerResult: An object containing the result of the optimization.
        """
        from skquant import opt as skq

        res, history = skq.minimize(
            func=fun,
            x0=np.asarray(x0),
            bounds=np.array(bounds),
            budget=self._maxiter,
            method="bobyqa",
        )

        optimizer_result = OptimizerResult()
        optimizer_result.x = res.optpar
        optimizer_result.fun = res.optval
        optimizer_result.nfev = len(history)
        return optimizer_result

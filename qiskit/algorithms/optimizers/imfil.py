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

"""IMplicit FILtering (IMFIL) optimizer."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qiskit.utils import optionals as _optionals
from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT


@_optionals.HAS_SKQUANT.require_in_instance
class IMFIL(Optimizer):
    """IMplicit FILtering algorithm.

    Implicit filtering is a way to solve bound-constrained optimization problems for
    which derivatives are not available. In comparison to methods that use
    interpolation to reconstruct the function and its higher derivatives,
    implicit filtering builds upon coordinate search followed by interpolation
    to get an approximate gradient.

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
        """Returns a dictionary of algorithm settings."""
        return {
            "maxiter": self._maxiter,
        }

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """
        Perform optimization using the 'imfil' method.

        Args:
            fun (Callable[[POINT], float]): The objective function to be minimized.
            x0 (POINT): The initial point for the optimization.
            jac (Callable[[POINT], POINT] | None, optional): The gradient function.
            Defaults to None.
            bounds (list[tuple[float, float]] | None, optional): The variable bounds.
            Defaults to None.

        Returns:
            OptimizerResult: The result of the optimization.
        """
        from skquant import opt as skq

        res, history = skq.minimize(
            func=fun,
            x0=x0,
            bounds=bounds,
            budget=self._maxiter,
            method="imfil",
        )

        optimizer_result = OptimizerResult()
        optimizer_result.x = res.optpar
        optimizer_result.fun = res.optval
        optimizer_result.nfev = len(history)
        return optimizer_result

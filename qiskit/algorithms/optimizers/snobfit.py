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

"""Stable Noisy Optimization by Branch and FIT algorithm (SNOBFIT) optimizer."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.utils import optionals as _optionals
from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT


@_optionals.HAS_SKQUANT.require_in_instance
@_optionals.HAS_SQSNOBFIT.require_in_instance
class SNOBFIT(Optimizer):
    """Stable Noisy Optimization by Branch and FIT algorithm.

    SnobFit is used for the optimization of derivative-free, noisy objective
    functions providing robust and fast solutions of problems with continuous
    variables varying within bound.

    Uses skquant.opt installed with pip install scikit-quant.
    For further detail, please refer to
    https://github.com/scikit-quant/scikit-quant and https://qat4chem.lbl.gov/software.
    """

    def __init__(
        self,
        maxiter: int = 1000,
        maxfail: int = 10,
        maxmp: int = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the class instance.

        Args:
            maxiter (int): Maximum number of function evaluations.
            maxfail (int): Maximum number of failures to improve the solution.
                Stops the algorithm after maxfail is reached.
            maxmp (int): Maximum number of model points requested for the local fit. Default is
                2 * number of parameters + 6 if None is provided.
            verbose (bool): Provide verbose (debugging) output.

        Raises:
            MissingOptionalLibraryError: Raised if the scikit-quant or SQSnobFit libraries
                are not installed.
            QiskitError: Raised if NumPy version 1.24.0 or above is installed.
                See https://github.com/scikit-quant/scikit-quant/issues/24 for more details.
        """
        # check version
        version = tuple(map(int, np.__version__.split(".")))
        if version >= (1, 24, 0):
            raise QiskitError(
                "SnobFit is incompatible with NumPy 1.24.0 or above, please "
                "install a previous version. See also scikit-quant/scikit-quant#24."
            )

        super().__init__()
        self._maxiter = maxiter
        self._maxfail = maxfail
        self._maxmp = maxmp
        self._verbose = verbose

    def get_support_level(self):
        """Returns support level dictionary.

        Returns:
            dict: A dictionary containing support levels for different optimizer features.
        """
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.required,
            "initial_point": OptimizerSupportLevel.required,
        }

    @property
    def settings(self) -> dict[str, Any]:
        """Returns a dictionary of settings for the optimizer.

        Returns:
            dict[str, Any]: A dictionary of settings for the optimizer.
        """
        return {
            "maxiter": self._maxiter,
            "maxfail": self._maxfail,
            "maxmp": self._maxmp,
            "verbose": self._verbose,
        }

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """
        Perform optimization using the SNOBFIT algorithm.

        This method minimizes the given function 'fun' with respect to the parameters
        using the SNOBFIT algorithm.

        Args:
            fun (Callable[[POINT], float]): The objective function to minimize.
            x0 (POINT): The initial point for optimization.
            jac (Callable[[POINT], POINT] | None, optional): The gradient function
                of 'fun'. Defaults to None.
            bounds (list[tuple[float, float]] | None, optional): The bounds for
                the parameters. Defaults to None.

        Returns:
            OptimizerResult: An object containing the optimization result.
        """
        import skquant.opt as skq
        from SQSnobFit import optset

        if bounds is None or any(None in bound_tuple for bound_tuple in bounds):
            raise ValueError("Optimizer SNOBFIT requires bounds for all parameters.")

        snobfit_settings = {
            "maxmp": self._maxmp,
            "maxfail": self._maxfail,
            "verbose": self._verbose,
        }
        options = optset(optin=snobfit_settings)
        # counters the error when initial point is outside the acceptable bounds
        x0 = np.asarray(x0)
        for idx, theta in enumerate(x0):
            if abs(theta) > bounds[idx][0]:
                x0[idx] = x0[idx] % bounds[idx][0]
            elif abs(theta) > bounds[idx][1]:
                x0[idx] = x0[idx] % bounds[idx][1]

        res, history = skq.minimize(
            fun,
            x0,
            bounds=bounds,
            budget=self._maxiter,
            method="snobfit",
            options=options,
        )

        optimizer_result = OptimizerResult()
        optimizer_result.x = res.optpar
        optimizer_result.fun = res.optval
        optimizer_result.nfev = len(history)
        return optimizer_result

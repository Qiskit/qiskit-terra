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

"""Wrapper class of scipy.optimize.minimize."""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize

from qiskit.utils.validation import validate_min

from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT


class SciPyOptimizer(Optimizer):
    """A general Qiskit Optimizer wrapping scipy.optimize.minimize.

    For further detail, please refer to
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _bounds_support_methods = {"l-bfgs-b", "tnc", "slsqp", "powell", "trust-constr"}
    _gradient_support_methods = {
        "cg",
        "bfgs",
        "newton-cg",
        "l-bfgs-b",
        "tnc",
        "slsqp",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "trust-constr",
    }

    def __init__(
        self,
        method: str | Callable,
        options: dict[str, Any] | None = None,
        max_evals_grouped: int = 1,
        **kwargs,
    ):
        """
        Initialize an instance of the class.

        Args:
            method (str | Callable): Type of solver.
            options (dict[str, Any] | None, optional): A dictionary of solver options.
            Defaults to None.
            max_evals_grouped (int, optional): Max number of default gradient
                evaluations performed simultaneously. Defaults to 1.
            **kwargs: Additional kwargs for scipy.optimize.minimize.
        """
        self._method = method.lower() if isinstance(method, str) else method
        # Set support level
        if self._method in self._bounds_support_methods:
            self._bounds_support_level = OptimizerSupportLevel.supported
        else:
            self._bounds_support_level = OptimizerSupportLevel.ignored
        if self._method in self._gradient_support_methods:
            self._gradient_support_level = OptimizerSupportLevel.supported
        else:
            self._gradient_support_level = OptimizerSupportLevel.ignored
        self._initial_point_support_level = OptimizerSupportLevel.required

        self._options = options if options is not None else {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            validate_min("max_evals_grouped", max_evals_grouped, 1)

        self._max_evals_grouped = max_evals_grouped
        self._kwargs = kwargs

    def get_support_level(self):
        """Return support level dictionary"""
        return {
            "gradient": self._gradient_support_level,
            "bounds": self._bounds_support_level,
            "initial_point": self._initial_point_support_level,
        }

    @property
    def settings(self) -> dict[str, Any]:
        options = self._options.copy()
        if hasattr(self, "_OPTIONS"):
            # all _OPTIONS should be keys in self._options, but add a failsafe here
            attributes = [
                option
                for option in self._OPTIONS  # pylint: disable=no-member
                if option in options.keys()
            ]

            settings = {attr: options.pop(attr) for attr in attributes}
        else:
            settings = {}

        settings["max_evals_grouped"] = self._max_evals_grouped
        settings["options"] = options
        settings.update(self._kwargs)

        # the subclasses don't need the "method" key as the class type specifies the method
        if self.__class__ == SciPyOptimizer:
            settings["method"] = self._method

        return settings

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """
        Perform a minimization optimization.

        This function uses the 'minimize' function from the 'scipy.optimize' module to
        perform a minimization optimization. It first checks if certain parameters
        should be ignored based on the state of the optimizer. It then sets the
        'jac' parameter based on whether gradient information is supported and the
        value of 'self._max_evals_grouped'. It also applies a workaround for the
        'l-bfgs-b' method.
        Finally, it calls the 'minimize' function and creates an 'OptimizerResult' object
        based on the result.

        Args:
            fun (Callable[[POINT], float]): The objective function to minimize.
            x0 (POINT): The initial guess for the optimization parameters.
            jac (Callable[[POINT], POINT] | None, optional): The function that
                computes the gradient of the objective function. Defaults to None.
            bounds (list[tuple[float, float]] | None, optional): The bounds on the
                optimization parameters. Defaults to None.

        Returns:
            OptimizerResult: The result of the optimization.
        """
        # Remove ignored parameters to supress the warning of scipy.optimize.minimize
        if self.is_bounds_ignored:
            bounds = None
        if self.is_gradient_ignored:
            jac = None

        if self.is_gradient_supported and jac is None and self._max_evals_grouped > 1:
            if "eps" in self._options:
                epsilon = self._options["eps"]
            else:
                epsilon = (
                    1e-8 if self._method in {"l-bfgs-b", "tnc"} else np.sqrt(np.finfo(float).eps)
                )
            jac = Optimizer.wrap_function(
                Optimizer.gradient_num_diff, (fun, epsilon, self._max_evals_grouped)
            )

        # Workaround for L_BFGS_B because it does not accept np.ndarray.
        # See https://github.com/Qiskit/qiskit-terra/pull/6373.
        if jac is not None and self._method == "l-bfgs-b":
            jac = self._wrap_gradient(jac)

        # Starting in scipy 1.9.0 maxiter is deprecated and maxfun (added in 1.5.0)
        # should be used instead
        swapped_deprecated_args = False
        if self._method == "tnc" and "maxiter" in self._options:
            swapped_deprecated_args = True
            self._options["maxfun"] = self._options.pop("maxiter")

        raw_result = minimize(
            fun=fun,
            x0=x0,
            method=self._method,
            jac=jac,
            bounds=bounds,
            options=self._options,
            **self._kwargs,
        )
        if swapped_deprecated_args:
            self._options["maxiter"] = self._options.pop("maxfun")

        result = OptimizerResult()
        result.x = raw_result.x
        result.fun = raw_result.fun
        result.nfev = raw_result.nfev
        result.njev = raw_result.get("njev", None)
        result.nit = raw_result.get("nit", None)

        return result

    @staticmethod
    def _wrap_gradient(gradient_function):

        def wrapped_gradient(x):
            gradient = gradient_function(x)
            if isinstance(gradient, np.ndarray):
                return gradient.tolist()
            return gradient

        return wrapped_gradient

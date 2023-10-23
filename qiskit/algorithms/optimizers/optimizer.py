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

"""Optimizer interface"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import IntEnum
import logging
from typing import Any, Union, Protocol

import numpy as np
import scipy

from qiskit.algorithms.algorithm_result import AlgorithmResult

logger = logging.getLogger(__name__)

POINT = Union[float, np.ndarray]


class OptimizerResult(AlgorithmResult):
    """The result of an optimization routine."""

    def __init__(self) -> None:
        super().__init__()
        self._x: POINT | None = None
        self._fun: float | None = None
        self._jac: POINT | None = None
        self._nfev: int | None = None
        self._njev: int | None = None
        self._nit: int | None = None

    @property
    def x(self) -> POINT | None:
        """The final point of the minimization."""
        return self._x

    @x.setter
    def x(self, x: POINT | None) -> None:
        """Set the final point of the minimization."""
        self._x = x

    @property
    def fun(self) -> float | None:
        """The final value of the minimization."""
        return self._fun

    @fun.setter
    def fun(self, fun: float | None) -> None:
        """Set the final value of the minimization."""
        self._fun = fun

    @property
    def jac(self) -> POINT | None:
        """The final gradient of the minimization."""
        return self._jac

    @jac.setter
    def jac(self, jac: POINT | None) -> None:
        """Set the final gradient of the minimization."""
        self._jac = jac

    @property
    def nfev(self) -> int | None:
        """The total number of function evaluations."""
        return self._nfev

    @nfev.setter
    def nfev(self, nfev: int | None) -> None:
        """Set the total number of function evaluations."""
        self._nfev = nfev

    @property
    def njev(self) -> int | None:
        """The total number of gradient evaluations."""
        return self._njev

    @njev.setter
    def njev(self, njev: int | None) -> None:
        """Set the total number of gradient evaluations."""
        self._njev = njev

    @property
    def nit(self) -> int | None:
        """The total number of iterations."""
        return self._nit

    @nit.setter
    def nit(self, nit: int | None) -> None:
        """Set the total number of iterations."""
        self._nit = nit


class Minimizer(Protocol):
    """Callable Protocol for minimizer.

    This interface is based on `SciPy's optimize module
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__.

     This protocol defines a callable taking the following parameters:

         fun
             The objective function to minimize (for example the energy in the case of the VQE).
         x0
             The initial point for the optimization.
         jac
             The gradient of the objective function.
         bounds
             Parameters bounds for the optimization. Note that these might not be supported
             by all optimizers.

     and which returns a minimization result object (either SciPy's or Qiskit's).
    """

    # pylint: disable=invalid-name
    def __call__(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        jac: Callable[[np.ndarray], np.ndarray] | None,
        bounds: list[tuple[float, float]] | None,
    ) -> scipy.optimize.OptimizeResult | OptimizerResult:
        """Minimize the objective function.

        This interface is based on `SciPy's optimize module <https://docs.scipy.org/doc
        /scipy/reference/generated/scipy.optimize.minimize.html>`__.

        Args:
            fun: The objective function to minimize (for example the energy in the
                case of the VQE).
            x0: The initial point for the optimization.
            jac: The gradient of the objective function.
            bounds: Parameters bounds for the optimization. Note that these might
                not be supported by all optimizers.

        Returns:
             The minimization result object (either SciPy's or Qiskit's).
        """
        ...


class OptimizerSupportLevel(IntEnum):
    """Support Level enum for features such as bounds, gradient and initial point"""

    # pylint: disable=invalid-name
    not_supported = 0  # Does not support the corresponding parameter in optimize()
    ignored = 1  # Feature can be passed as non None but will be ignored
    supported = 2  # Feature is supported
    required = 3  # Feature is required and must be given, None is invalid


class Optimizer(ABC):
    """Base class for optimization algorithm."""

    @abstractmethod
    def __init__(self):
        """
        Initialize the optimization algorithm, setting the support
        level for _gradient_support_level, _bound_support_level,
        _initial_point_support_level, and empty options.
        """
        self._gradient_support_level = self.get_support_level()["gradient"]
        self._bounds_support_level = self.get_support_level()["bounds"]
        self._initial_point_support_level = self.get_support_level()["initial_point"]
        self._options = {}
        self._max_evals_grouped = None

    @abstractmethod
    def get_support_level(self):
        """Return support level dictionary"""
        raise NotImplementedError

    def set_options(self, **kwargs):
        """
        Sets or updates values in the options dictionary.

        The options dictionary may be used internally by a given optimizer to
        pass additional optional values for the underlying
        optimizer/optimization function used. The options dictionary may be
        initially populated with a set of key/values when the given optimizer
        is constructed.

        Args:
            kwargs (dict): options, given as name=value.
        """
        for name, value in kwargs.items():
            self._options[name] = value
        logger.debug("options: %s", self._options)

    # pylint: disable=invalid-name
    @staticmethod
    def gradient_num_diff(x_center, f, epsilon, max_evals_grouped=None):
        """
        We compute the gradient with the numeric differentiation in the parallel way,
        around the point x_center.

        Args:
            x_center (ndarray): point around which we compute the gradient
            f (func): the function of which the gradient is to be computed.
            epsilon (float): the epsilon used in the numeric differentiation.
            max_evals_grouped (int): max evals grouped, defaults to 1 (i.e. no batching).
        Returns:
            grad: the gradient computed

        """
        if max_evals_grouped is None:  # no batching by default
            max_evals_grouped = 1

        forig = f(*((x_center,)))
        grad = []
        ei = np.zeros((len(x_center),), float)
        todos = []
        for k in range(len(x_center)):
            ei[k] = 1.0
            d = epsilon * ei
            todos.append(x_center + d)
            ei[k] = 0.0

        counter = 0
        chunk = []
        chunks = []
        length = len(todos)
        # split all points to chunks, where each chunk has batch_size points
        for i in range(length):
            x = todos[i]
            chunk.append(x)
            counter += 1
            # the last one does not have to reach batch_size
            if counter == max_evals_grouped or i == length - 1:
                chunks.append(chunk)
                chunk = []
                counter = 0

        for chunk in chunks:  # eval the chunks in order
            parallel_parameters = np.concatenate(chunk)
            todos_results = f(parallel_parameters)  # eval the points in a chunk (order preserved)
            if isinstance(todos_results, float):
                grad.append((todos_results - forig) / epsilon)
            else:
                for todor in todos_results:
                    grad.append((todor - forig) / epsilon)

        return np.array(grad)

    @staticmethod
    def wrap_function(function, args):
        """
        Wrap the function to implicitly inject the args at the call of the function.

        Args:
            function (func): the target function
            args (tuple): the args to be injected
        Returns:
            function_wrapper: wrapper
        """

        def function_wrapper(*wrapper_args):
            return function(*(wrapper_args + args))

        return function_wrapper

    @property
    def setting(self):
        """Return setting"""
        ret = f"Optimizer: {self.__class__.__name__}\n"
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                params += f"-- {key[1:]}: {value}\n"
        ret += f"{params}"
        return ret

    @property
    def settings(self) -> dict[str, Any]:
        """The optimizer settings in a dictionary format.

        The settings can for instance be used for JSON-serialization (if all settings are
        serializable, which e.g. doesn't hold per default for callables), such that the
        optimizer object can be reconstructed as

        .. code-block::

            settings = optimizer.settings
            # JSON serialize and send to another server
            optimizer = OptimizerClass(**settings)

        """
        raise NotImplementedError("The settings method is not implemented per default.")

    @abstractmethod
    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """Minimize the scalar function.

        Args:
            fun: The scalar function to minimize.
            x0: The initial point for the minimization.
            jac: The gradient of the scalar function ``fun``.
            bounds: Bounds for the variables of ``fun``. This argument might be ignored if the
                optimizer does not support bounds.

        Returns:
            The result of the optimization, containing e.g. the result as attribute ``x``.
        """
        raise NotImplementedError()

    @property
    def gradient_support_level(self):
        """Returns gradient support level"""
        return self._gradient_support_level

    @property
    def is_gradient_ignored(self):
        """Returns is gradient ignored"""
        return self._gradient_support_level == OptimizerSupportLevel.ignored

    @property
    def is_gradient_supported(self):
        """Returns is gradient supported"""
        return self._gradient_support_level != OptimizerSupportLevel.not_supported

    @property
    def is_gradient_required(self):
        """Returns is gradient required"""
        return self._gradient_support_level == OptimizerSupportLevel.required

    @property
    def bounds_support_level(self):
        """Returns bounds support level"""
        return self._bounds_support_level

    @property
    def is_bounds_ignored(self):
        """Returns is bounds ignored"""
        return self._bounds_support_level == OptimizerSupportLevel.ignored

    @property
    def is_bounds_supported(self):
        """Returns is bounds supported"""
        return self._bounds_support_level != OptimizerSupportLevel.not_supported

    @property
    def is_bounds_required(self):
        """Returns is bounds required"""
        return self._bounds_support_level == OptimizerSupportLevel.required

    @property
    def initial_point_support_level(self):
        """Returns initial point support level"""
        return self._initial_point_support_level

    @property
    def is_initial_point_ignored(self):
        """Returns is initial point ignored"""
        return self._initial_point_support_level == OptimizerSupportLevel.ignored

    @property
    def is_initial_point_supported(self):
        """Returns is initial point supported"""
        return self._initial_point_support_level != OptimizerSupportLevel.not_supported

    @property
    def is_initial_point_required(self):
        """Returns is initial point required"""
        return self._initial_point_support_level == OptimizerSupportLevel.required

    def print_options(self):
        """Print algorithm-specific options."""
        for name in sorted(self._options):
            logger.debug("%s = %s", name, str(self._options[name]))

    def set_max_evals_grouped(self, limit):
        """Set max evals grouped."""
        self._max_evals_grouped = limit

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Analytical Quantum Gradient Descent (AQGD) optimizer."""

from __future__ import annotations
import logging
from collections.abc import Callable
from typing import Any
import warnings

import numpy as np
from qiskit.utils.validation import validate_range_exclusive_max
from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT
from ..exceptions import AlgorithmError

logger = logging.getLogger(__name__)


class AQGD(Optimizer):
    """Analytic Quantum Gradient Descent (AQGD) with Epochs optimizer.
    Performs gradient descent optimization with a momentum term, analytic gradients,
    and customized step length schedule for parameterized quantum gates, i.e.
    Pauli Rotations. See, for example:

    * K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii. (2018).
      Quantum circuit learning. Phys. Rev. A 98, 032309.
      https://arxiv.org/abs/1803.00745

    * Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, Nathan Killoran. (2019).
      Evaluating analytic gradients on quantum hardware. Phys. Rev. A 99, 032331.
      https://arxiv.org/abs/1811.11184

    for further details on analytic gradients of parameterized quantum gates.

    Gradients are computed "analytically" using the quantum circuit when evaluating
    the objective function.

    """

    _OPTIONS = ["maxiter", "eta", "tol", "disp", "momentum", "param_tol", "averaging"]

    def __init__(
        self,
        maxiter: int | list[int] = 1000,
        eta: float | list[float] = 1.0,
        tol: float = 1e-6,  # this is tol
        momentum: float | list[float] = 0.25,
        param_tol: float = 1e-6,
        averaging: int = 10,
    ) -> None:
        """
        Performs Analytical Quantum Gradient Descent (AQGD) with Epochs.

        Args:
            maxiter: Maximum number of iterations (full gradient steps).
            eta: The coefficient of the gradient update. Increasing this value results
                in larger step sizes: param = previous_param - eta * deriv
            tol: Tolerance for change in windowed average of objective values.
                Convergence occurs when either objective tolerance is met OR parameter
                tolerance is met.
            momentum: Bias towards the previous gradient momentum in current
                update. Must be within the bounds: [0,1).
            param_tol: Tolerance for change in norm of parameters.
            averaging: Length of window over which to average objective values for
                objective convergence criterion.

        Raises:
            AlgorithmError: If the length of ``maxiter``, ``momentum``, and ``eta`` is not the same.
        """
        super().__init__()
        if isinstance(maxiter, int):
            maxiter = [maxiter]
        if isinstance(eta, (int, float)):
            eta = [eta]
        if isinstance(momentum, (int, float)):
            momentum = [momentum]
        if len(maxiter) != len(eta) or len(maxiter) != len(momentum):
            raise AlgorithmError(
                "AQGD input parameter length mismatch. Parameters `maxiter`, "
                "`eta`, and `momentum` must have the same length."
            )
        for m in momentum:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                validate_range_exclusive_max("momentum", m, 0, 1)

        self._eta = eta
        self._maxiter = maxiter
        self._momenta_coeff = momentum
        self._param_tol = param_tol
        self._tol = tol
        self._averaging = averaging

        # state
        self._avg_objval: float | None = None
        self._prev_param: np.ndarray | None = None
        self._eval_count = 0  # function evaluations
        self._prev_loss: list[float] = []
        self._prev_grad: list[list[float]] = []

    def get_support_level(self) -> dict[str, OptimizerSupportLevel]:
        """Support level dictionary

        Returns:
            Dict[str, int]: gradient, bounds and initial point
                            support information that is ignored/required.
        """
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

    @property
    def settings(self) -> dict[str, Any]:
        """Settings dictionary

        Returns:
            Dict[str, Any]: A dictionary with various optimizer settings.
        """
        return {
            "maxiter": self._maxiter,
            "eta": self._eta,
            "momentum": self._momenta_coeff,
            "param_tol": self._param_tol,
            "tol": self._tol,
            "averaging": self._averaging,
        }

    def _compute_objective_fn_and_gradient(
        self, params: np.ndarray | list[float], obj: Callable
    ) -> tuple[float, np.ndarray]:
        """
        Obtains the objective function value for params and the analytical quantum
        derivatives of the objective function with respect to each parameter.
        Requires
        2*(number parameters) + 1 objective evaluations

        Args:
            params: Current value of the parameters to evaluate the objective
            function obj: Objective function of interest

        Returns:
            Tuple containing the objective value and an array of gradients for the given
            parameter set.
        """
        num_params = len(params)
        param_sets_to_eval = params + np.concatenate(
            (
                np.zeros((1, num_params)),  # copy of the parameters as is
                np.eye(num_params) * np.pi / 2,  # copy of the parameters with the positive shift
                -np.eye(num_params) * np.pi / 2,
            ),  # copy of the parameters with the negative shift
            axis=0,
        )
        # Evaluate,
        # reshaping to flatten, as expected by objective function
        values = np.array(obj(param_sets_to_eval.reshape(-1)))

        # Update number of objective function evaluations
        self._eval_count += 2 * num_params + 1

        # return the objective function value
        obj_value = values[0]

        # return the gradient values
        gradient = 0.5 * (values[1 : num_params + 1] - values[1 + num_params :])
        return obj_value, gradient

    def _update(
        self,
        params: np.ndarray,
        gradient: np.ndarray,
        mprev: np.ndarray,
        step_size: float,
        momentum_coeff: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Updates full parameter array based on a step that is a convex
        combination of the gradient and previous momentum

        Args:
            params: Current value of the parameters to evaluate the objective function at
            gradient: Gradient of objective wrt parameters
            mprev: Momentum vector for each parameter
            step_size: The scaling of step to take
            momentum_coeff: Bias towards previous momentum vector when updating current
                momentum/step vector

        Returns:
            Tuple of the updated parameter and momentum vectors respectively.
        """
        # Momentum update:
        # Convex combination of previous momentum and current gradient estimate
        mnew = (1 - momentum_coeff) * gradient + momentum_coeff * mprev
        params -= step_size * mnew
        return params, mnew

    def _converged_objective(self, objval: float, tol: float, window_size: int) -> bool:
        """
        Tests convergence based on the change in a moving windowed average of past
        objective values

        Args:
            objval: Current value of the objective function
            tol: tolerance below which (average) objective function change must be
            window_size: size of averaging window

        Returns:
            Bool indicating whether or not the optimization has converged.
        """
        # If we haven't reached the required window length,
        # append the current value, but we haven't converged
        if len(self._prev_loss) < window_size:
            self._prev_loss.append(objval)
            return False

        # Update last value in list with current value
        self._prev_loss.append(objval)
        # (length now = n+1)

        # Calculate previous windowed average
        # and current windowed average of objective values
        prev_avg = np.mean(self._prev_loss[:window_size])
        curr_avg = np.mean(self._prev_loss[1 : window_size + 1])
        self._avg_objval = curr_avg

        # Update window of objective values
        # (Remove earliest value)
        self._prev_loss.pop(0)

        if np.absolute(prev_avg - curr_avg) < tol:
            # converged
            logger.info("Previous obj avg: %f\nCurr obj avg: %f", prev_avg, curr_avg)
            return True
        return False

    def _converged_parameter(self, parameter: np.ndarray, tol: float) -> bool:
        """
        Tests convergence based on change in parameter

        Args:
            parameter: current parameter values
            tol: tolerance for change in norm of parameters

        Returns:
            Bool indicating whether or not the optimization has converged
        """
        if self._prev_param is None:
            self._prev_param = np.copy(parameter)
            return False

        order = np.inf
        p_change = np.linalg.norm(self._prev_param - parameter, ord=order)
        if p_change < tol:
            # converged
            logger.info("Change in parameters (%f norm): %f", order, p_change)
            return True
        return False

    def _converged_alt(self, gradient: list[float], tol: float, window_size: int) -> bool:
        """
        Tests convergence from norm of windowed average of gradients

        Args:
            gradient: current gradient
            tol: tolerance for average gradient norm
            window_size: size of averaging window

        Returns:
            Bool indicating whether or not the optimization has converged
        """
        # If we haven't reached the required window length,
        # append the current value, but we haven't converged
        if len(self._prev_grad) < window_size - 1:
            self._prev_grad.append(gradient)
            return False

        # Update last value in list with current value
        self._prev_grad.append(gradient)
        # (length now = n)

        # Calculate previous windowed average
        # and current windowed average of objective values
        avg_grad = np.mean(self._prev_grad, axis=0)

        # Update window of values
        # (Remove earliest value)
        self._prev_grad.pop(0)

        if np.linalg.norm(avg_grad, ord=np.inf) < tol:
            # converged
            logger.info("Avg. grad. norm: %f", np.linalg.norm(avg_grad, ord=np.inf))
            return True
        return False

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """
        Perform optimization using a variant of stochastic gradient descent.

        This function takes in several arguments including a callable function 'fun'
        that represents the objective function to be minimized, an initial point
        'x0', an optional function 'jac' representing the Jacobian of the
        objective function, and optional bounds for the variables.
        The function returns an 'OptimizerResult' object containing the optimized parameters,
        the objective function value, the number of function evaluations, and the
        number of iterations.

        The function iterates over multiple epochs, updating the parameters using the
        specified learning rate 'eta' and momentum coefficient 'mom_coeff'. It
        checks for convergence of both the parameters and the objective function
        at each iteration.
        """
        params = np.asarray(x0)
        momentum = np.zeros(shape=(params.size,))
        # empty out history of previous objectives/gradients/parameters
        # (in case this object is re-used)
        self._prev_loss = []
        self._prev_grad = []
        self._prev_param = None
        self._eval_count = 0  # function evaluations

        iter_count = 0
        logger.info("Initial Params: %s", params)

        epoch = 0
        converged = False
        for (eta, mom_coeff) in zip(self._eta, self._momenta_coeff):
            logger.info("Epoch: %4d | Stepsize: %6.4f | Momentum: %6.4f", epoch, eta, mom_coeff)

            sum_max_iters = sum(self._maxiter[0 : epoch + 1])
            while iter_count < sum_max_iters:
                # update the iteration count
                iter_count += 1

                # Check for parameter convergence before potentially costly function evaluation
                converged = self._converged_parameter(params, self._param_tol)
                if converged:
                    break

                # Calculate objective function and estimate of analytical gradient
                if jac is None:
                    objval, gradient = self._compute_objective_fn_and_gradient(params, fun)
                else:
                    objval = fun(params)
                    gradient = jac(params)

                logger.info(
                    " Iter: %4d | Obj: %11.6f | Grad Norm: %f",
                    iter_count,
                    objval,
                    np.linalg.norm(gradient, ord=np.inf),
                )

                # Check for objective convergence
                converged = self._converged_objective(objval, self._tol, self._averaging)
                if converged:
                    break

                # Update parameters and momentum
                params, momentum = self._update(params, gradient, momentum, eta, mom_coeff)
            # end inner iteration
            # if converged, end iterating over epochs
            if converged:
                break
            epoch += 1
        # end epoch iteration

        result = OptimizerResult()
        result.x = params
        result.fun = objval
        result.nfev = self._eval_count
        result.nit = iter_count

        return result

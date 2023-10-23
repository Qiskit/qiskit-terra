# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

This implementation allows both, standard first-order as well as second-order SPSA.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Callable, Any, SupportsFloat
import logging
import warnings
from time import time

import scipy
import numpy as np

from qiskit.utils import algorithm_globals
from qiskit.utils.deprecation import deprecate_func

from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT

# number of function evaluations, parameters, loss, stepsize, accepted
CALLBACK = Callable[[int, np.ndarray, float, SupportsFloat, bool], None]
TERMINATIONCHECKER = Callable[[int, np.ndarray, float, SupportsFloat, bool], bool]

logger = logging.getLogger(__name__)


class SPSA(Optimizer):
    """Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    SPSA [1] is an gradient descent method for optimizing systems with multiple unknown parameters.
    As an optimization method, it is appropriately suited to large-scale population models,
    adaptive modeling, and simulation optimization.

    .. seealso::

        Many examples are presented at the `SPSA Web site <http://www.jhuapl.edu/SPSA>`__.

    The main feature of SPSA is the stochastic gradient approximation, which requires only two
    measurements of the objective function, regardless of the dimension of the optimization
    problem.

    Additionally to standard, first-order SPSA, where only gradient information is used, this
    implementation also allows second-order SPSA (2-SPSA) [2]. In 2-SPSA we also estimate the
    Hessian of the loss with a stochastic approximation and multiply the gradient with the
    inverse Hessian to take local curvature into account and improve convergence.
    Notably this Hessian estimate requires only a constant number of function evaluations
    unlike an exact evaluation of the Hessian, which scales quadratically in the number of
    function evaluations.

    .. note::

        SPSA can be used in the presence of noise, and it is therefore indicated in situations
        involving measurement uncertainty on a quantum computation when finding a minimum.
        If you are executing a variational algorithm using an OpenQASM
        simulator or a real device, SPSA would be the most recommended choice among the optimizers
        provided here.

    The optimization process can includes a calibration phase if neither the ``learning_rate`` nor
    ``perturbation`` is provided, which requires additional functional evaluations.
    (Note that either both or none must be set.) For further details on the automatic calibration,
    please refer to the supplementary information section IV. of [3].

    .. note::

        This component has some function that is normally random. If you want to reproduce behavior
        then you should set the random number generator seed in the algorithm_globals
        (``qiskit.utils.algorithm_globals.random_seed = seed``).


    Examples:

        This short example runs SPSA for the ground state calculation of the ``Z ^ Z``
        observable where the ansatz is a ``PauliTwoDesign`` circuit.

        .. code-block:: python

            import numpy as np
            from qiskit.algorithms.optimizers import SPSA
            from qiskit.circuit.library import PauliTwoDesign
            from qiskit.opflow import Z, StateFn

            ansatz = PauliTwoDesign(2, reps=1, seed=2)
            observable = Z ^ Z
            initial_point = np.random.random(ansatz.num_parameters)

            def loss(x):
                bound = ansatz.assign_parameters(x)
                return np.real((StateFn(observable, is_measurement=True) @ StateFn(bound)).eval())

            spsa = SPSA(maxiter=300)
            result = spsa.optimize(ansatz.num_parameters, loss, initial_point=initial_point)

        To use the Hessian information, i.e. 2-SPSA, you can add `second_order=True` to the
        initializer of the `SPSA` class, the rest of the code remains the same.

        .. code-block:: python

            two_spsa = SPSA(maxiter=300, second_order=True)
            result = two_spsa.optimize(ansatz.num_parameters, loss, initial_point=initial_point)

        The `termination_checker` can be used to implement a custom termination criterion.

        .. code-block:: python

            import numpy as np
            from qiskit.algorithms.optimizers import SPSA

            def objective(x):
                return np.linalg.norm(x) + .04*np.random.rand(1)

            class TerminationChecker:

                def __init__(self, N : int):
                    self.N = N
                    self.values = []

                def __call__(self, nfev, parameters, value, stepsize, accepted) -> bool:
                    self.values.append(value)

                    if len(self.values) > self.N:
                        last_values = self.values[-self.N:]
                        pp = np.polyfit(range(self.N), last_values, 1)
                        slope = pp[0] / self.N

                        if slope > 0:
                            return True
                    return False

            spsa = SPSA(maxiter=200, termination_checker=TerminationChecker(10))
            parameters, value, niter = spsa.optimize(2, objective, initial_point=[0.5, 0.5])
            print(f'SPSA completed after {niter} iterations')


    References:

        [1]: J. C. Spall (1998). An Overview of the Simultaneous Perturbation Method for Efficient
        Optimization, Johns Hopkins APL Technical Digest, 19(4), 482–492.
        `Online at jhuapl.edu. <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>`_

        [2]: J. C. Spall (1997). Accelerated second-order stochastic optimization using only
        function measurements, Proceedings of the 36th IEEE Conference on Decision and Control,
        1417-1424 vol.2. `Online at IEEE.org. <https://ieeexplore.ieee.org/document/657661>`_

        [3]: A. Kandala et al. (2017). Hardware-efficient Variational Quantum Eigensolver for
        Small Molecules and Quantum Magnets. Nature 549, pages242–246(2017).
        `arXiv:1704.05018v2 <https://arxiv.org/pdf/1704.05018v2.pdf#section*.11>`_

    """

    def __init__(
        self,
        maxiter: int = 100,
        blocking: bool = False,
        allowed_increase: float | None = None,
        trust_region: bool = False,
        learning_rate: float | np.ndarray | Callable[[], Iterator] | None = None,
        perturbation: float | np.ndarray | Callable[[], Iterator] | None = None,
        last_avg: int = 1,
        resamplings: int | dict[int, int] = 1,
        perturbation_dims: int | None = None,
        second_order: bool = False,
        regularization: float | None = None,
        hessian_delay: int = 0,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        initial_hessian: np.ndarray | None = None,
        callback: CALLBACK | None = None,
        termination_checker: TERMINATIONCHECKER | None = None,
    ) -> None:
        r"""
        Args:
            maxiter: The maximum number of iterations. Note that this is not the
                maximal number of function evaluations.
            blocking: If True, only accepts updates that improve the loss (up to
                some allowed increase, see next argument).
            allowed_increase: If ``blocking`` is ``True``, this argument
                determines by how much the loss can increase with the proposed
                parameters and still be accepted.
                If ``None``, the allowed increases is calibrated automatically to be twice the
                approximated standard deviation of the loss function.
            trust_region: If ``True``, restricts the norm of the update step to be :math:`\leq 1`.
            learning_rate: The update step is the learning rate is multiplied with the gradient.
                If the learning rate is a float, it remains constant over the course of the
                optimization. If a NumPy array, the :math:`i`-th element is the
                learning rate for the :math:`i`-th iteration. It can also be a
                callable returning an iterator which yields the learning rates for
                each optimization step.
                If ``learning_rate`` is set ``perturbation`` must also be provided.
            perturbation: Specifies the magnitude of the perturbation for the
                finite difference approximation of the gradients. See
                ``learning_rate`` for the supported types.
                If ``perturbation`` is set ``learning_rate`` must also be provided.
            last_avg: Return the average of the ``last_avg`` parameters instead of
                just the last parameter values.
            resamplings: The number of times the gradient (and Hessian) is sampled
                using a random direction to construct a gradient estimate. Per default
                the gradient is estimated using only one random direction. If an
                integer, all iterations use the same number of resamplings. If a
                dictionary, this is interpreted as
                ``{iteration: number of resamplings per iteration}``.
            perturbation_dims: The number of perturbed dimensions. Per default,
                all dimensions are perturbed, but a smaller, fixed number can be
                perturbed. If set, the perturbed dimensions are chosen uniformly at
                random.
            second_order: If True, use 2-SPSA instead of SPSA. In 2-SPSA, the
                Hessian is estimated additionally to the gradient, and the gradient is
                preconditioned with the inverse of the Hessian to improve convergence.
            regularization: To ensure the preconditioner is symmetric and positive
                definite, the identity times a small coefficient is added to it. This
                generator yields that coefficient.
            hessian_delay: Start multiplying the gradient with the inverse Hessian
                only after a certain number of iterations. The Hessian is still
                evaluated and therefore this argument can be useful to first get a
                stable average over the last iterations before using it as
                preconditioner.
            lse_solver: The method to solve for the inverse of the Hessian. Per
                default an exact LSE solver is used, but can e.g. be overwritten by a
                minimization routine.
            initial_hessian: The initial guess for the Hessian. By default the
                identity matrix is used.
            callback: A callback function passed information in each iteration
            step. The information is, in this order: the number of function
                evaluations, the parameters, the function value, the stepsize, whether
                the step was accepted.
            termination_checker: A callback function executed at the end of each
            iteration step. The arguments are, in this order: the parameters, the
                function value, the number of function evaluations, the stepsize,
                whether the step was accepted. If the callback returns True, the
                optimization is terminated.
            To prevent additional evaluations of the objective method, if the objective has not yet
                been evaluated, the objective is estimated by taking the mean of
                the objective evaluations used in the estimate of the gradient.

        Raises:
            ValueError: If ``learning_rate`` or ``perturbation`` is an array with less elements
                than the number of iterations.


        """
        super().__init__()

        # general optimizer arguments
        self.maxiter = maxiter
        self.trust_region = trust_region
        self.callback = callback
        self.termination_checker = termination_checker

        # if learning rate and perturbation are arrays, check they are sufficiently long
        for attr, name in zip([learning_rate, perturbation], ["learning_rate", "perturbation"]):
            if isinstance(attr, (list, np.ndarray)):
                if len(attr) < maxiter:
                    raise ValueError(f"Length of {name} is smaller than maxiter ({maxiter}).")

        self.learning_rate = learning_rate
        self.perturbation = perturbation

        # SPSA specific arguments
        self.blocking = blocking
        self.allowed_increase = allowed_increase
        self.last_avg = last_avg
        self.resamplings = resamplings
        self.perturbation_dims = perturbation_dims

        # 2-SPSA specific arguments
        if regularization is None:
            regularization = 0.01

        self.second_order = second_order
        self.hessian_delay = hessian_delay
        self.lse_solver = lse_solver
        self.regularization = regularization
        self.initial_hessian = initial_hessian

        # runtime arguments
        self._nfev: int | None = None  # the number of function evaluations
        self._smoothed_hessian: np.ndarray | None = None  # smoothed average of the Hessians

    @staticmethod
    def calibrate(
        loss: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        c: float = 0.2,
        stability_constant: float = 0,
        target_magnitude: float | None = None,  # 2 pi / 10
        alpha: float = 0.602,
        gamma: float = 0.101,
        modelspace: bool = False,
        max_evals_grouped: int = 1,
    ) -> tuple[Callable, Callable]:
        r"""Calibrate SPSA parameters with a powerseries as learning rate and perturbation coeffs.

        The powerseries are:

        .. math::

            a_k = \frac{a}{(A + k + 1)^\alpha}, c_k = \frac{c}{(k + 1)^\gamma}

        Args:
            loss: The loss function.
            initial_point: The initial guess of the iteration.
            c: The initial perturbation magnitude.
            stability_constant: The value of `A`.
            target_magnitude: The target magnitude for the first update step, defaults to
                :math:`2\pi / 10`.
            alpha: The exponent of the learning rate powerseries.
            gamma: The exponent of the perturbation powerseries.
            modelspace: Whether the target magnitude is the difference of parameter values
                or function values (= model space).
            max_evals_grouped: The number of grouped evaluations supported by the loss function.
                Defaults to 1, i.e. no grouping.

        Returns:
            tuple(generator, generator): A tuple of powerseries generators, the first one for the
                learning rate and the second one for the perturbation.
        """
        logger.info("SPSA: Starting calibration of learning rate and perturbation.")
        if target_magnitude is None:
            target_magnitude = 2 * np.pi / 10

        dim = len(initial_point)

        # compute the average magnitude of the first step
        steps = 25
        points = []
        for _ in range(steps):
            # compute the random directon
            pert = bernoulli_perturbation(dim)
            points += [initial_point + c * pert, initial_point - c * pert]

        losses = _batch_evaluate(loss, points, max_evals_grouped)

        avg_magnitudes = 0.0
        for i in range(steps):
            delta = losses[2 * i] - losses[2 * i + 1]
            avg_magnitudes += np.abs(delta / (2 * c))

        avg_magnitudes /= steps

        if modelspace:
            a = target_magnitude / (avg_magnitudes**2)
        else:
            a = target_magnitude / avg_magnitudes

        # compute the rescaling factor for correct first learning rate
        if a < 1e-10:
            warnings.warn(f"Calibration failed, using {target_magnitude} for `a`")
            a = target_magnitude

        logger.info("Finished calibration:")
        logger.info(
            " -- Learning rate: a / ((A + n) ^ alpha) with a = %s, A = %s, alpha = %s",
            a,
            stability_constant,
            alpha,
        )
        logger.info(" -- Perturbation: c / (n ^ gamma) with c = %s, gamma = %s", c, gamma)

        # set up the powerseries
        def learning_rate():
            return powerseries(a, alpha, stability_constant)

        def perturbation():
            return powerseries(c, gamma)

        return learning_rate, perturbation

    @staticmethod
    def estimate_stddev(
        loss: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        avg: int = 25,
        max_evals_grouped: int = 1,
    ) -> float:
        """Estimate the standard deviation of the loss function."""
        losses = _batch_evaluate(loss, avg * [initial_point], max_evals_grouped)
        return np.std(losses)

    @property
    def settings(self) -> dict[str, Any]:
        """
        Return a dictionary containing the settings of the object.

        The dictionary contains various attributes and their corresponding values.

        Returns:
            dict[str, Any]: A dictionary with the settings of the object.
        """
        # if learning rate or perturbation are custom iterators expand them
        if callable(self.learning_rate):
            iterator = self.learning_rate()
            learning_rate = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            learning_rate = self.learning_rate

        if callable(self.perturbation):
            iterator = self.perturbation()
            perturbation = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            perturbation = self.perturbation

        return {
            "maxiter": self.maxiter,
            "learning_rate": learning_rate,
            "perturbation": perturbation,
            "trust_region": self.trust_region,
            "blocking": self.blocking,
            "allowed_increase": self.allowed_increase,
            "resamplings": self.resamplings,
            "perturbation_dims": self.perturbation_dims,
            "second_order": self.second_order,
            "hessian_delay": self.hessian_delay,
            "regularization": self.regularization,
            "lse_solver": self.lse_solver,
            "initial_hessian": self.initial_hessian,
            "callback": self.callback,
            "termination_checker": self.termination_checker,
        }

    def _point_sample(self, loss, x, eps, delta1, delta2):
        """A single sample of the gradient at position `x` in direction `delta`.

        Args:
            loss: The loss function to be evaluated.
            x: The position at which the gradient is evaluated.
            eps (float): The small value used to calculate the points to be evaluated.
            delta1: The first direction vector.
            delta2: The second direction vector (only used if second_order is True).

        Returns:
            tuple: A tuple containing the mean of the evaluated points, the
            gradient sample, and the hessian sample (if applicable).
        """
        # points to evaluate
        points = [x + eps * delta1, x - eps * delta1]
        self._nfev += 2

        if self.second_order:
            points += [x + eps * (delta1 + delta2), x + eps * (-delta1 + delta2)]
            self._nfev += 2

        # batch evaluate the points (if possible)
        values = _batch_evaluate(loss, points, self._max_evals_grouped)

        plus = values[0]
        minus = values[1]
        gradient_sample = (plus - minus) / (2 * eps) * delta1

        hessian_sample = None
        if self.second_order:
            diff = (values[2] - plus) - (values[3] - minus)
            diff /= 2 * eps**2

            rank_one = np.outer(delta1, delta2)
            hessian_sample = diff * (rank_one + rank_one.T) / 2

        return np.mean(values), gradient_sample, hessian_sample

    def _point_estimate(self, loss, x, eps, num_samples):
        """The gradient estimate at point x.

        Args:
            loss: The loss function.
            x: The point at which to calculate the gradient estimate.
            eps: The perturbation parameter.
            num_samples: The number of samples to use for the estimate.

        Returns:
            Tuple[float, ndarray, ndarray]: The value estimate, gradient estimate, and
            hessian estimate.
        """
        # set up variables to store averages
        value_estimate = 0
        gradient_estimate = np.zeros(x.size)
        hessian_estimate = np.zeros((x.size, x.size))

        # iterate over the directions
        deltas1 = [
            bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(num_samples)
        ]

        if self.second_order:
            deltas2 = [
                bernoulli_perturbation(x.size, self.perturbation_dims) for _ in range(num_samples)
            ]
        else:
            deltas2 = None

        for i in range(num_samples):
            delta1 = deltas1[i]
            delta2 = deltas2[i] if self.second_order else None

            value_sample, gradient_sample, hessian_sample = self._point_sample(
                loss, x, eps, delta1, delta2
            )
            value_estimate += value_sample
            gradient_estimate += gradient_sample

            if self.second_order:
                hessian_estimate += hessian_sample

        return (
            value_estimate / num_samples,
            gradient_estimate / num_samples,
            hessian_estimate / num_samples,
        )

    def _compute_update(self, loss, x, k, eps, lse_solver):
        """
        Compute the perturbations and perform various calculations based on the input
        parameters.

        Args:
            loss: The loss function.
            x: The input data.
            k: The number of iterations.
            eps: The perturbation value.
            lse_solver: The solver for solving the linear system of equations.

        Returns:
            The computed value and gradient.
        """
        # compute the perturbations
        if isinstance(self.resamplings, dict):
            num_samples = self.resamplings.get(k, 1)
        else:
            num_samples = self.resamplings

        # accumulate the number of samples
        value, gradient, hessian = self._point_estimate(loss, x, eps, num_samples)

        # precondition gradient with inverse Hessian, if specified
        if self.second_order:
            smoothed = k / (k + 1) * self._smoothed_hessian + 1 / (k + 1) * hessian
            self._smoothed_hessian = smoothed

            if k > self.hessian_delay:
                spd_hessian = _make_spd(smoothed, self.regularization)

                # solve for the gradient update
                gradient = np.real(lse_solver(spd_hessian, gradient))

        return value, gradient

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """
        Perform optimization using the SPSA (Simultaneous Perturbation Stochastic
        Approximation) algorithm.

        This method minimizes a given function 'fun' by iteratively updating the
        parameters starting from an initial point 'x0'. Optionally, a Jacobian
        function 'jac' and bounds for the variables can be provided. The method
        returns an 'OptimizerResult' object containing the optimized parameters,
        the optimized function value, the number of function evaluations, and the
        number of iterations.

        Args:
            fun (Callable[[POINT], float]): The function to be minimized.
            x0 (POINT): The initial point for optimization.
            jac (Callable[[POINT], POINT], optional): The Jacobian of the function.
                Defaults to None.
            bounds (list[tuple[float, float]], optional): Bounds for the variables.
                Defaults to None.

        Returns:
            OptimizerResult: An object containing the optimized parameters, the optimized
            function value, the number of function evaluations, and the number of
            iterations.
        """
        # ensure learning rate and perturbation are correctly set: either none or both
        # this happens only here because for the calibration the loss function is required
        if self.learning_rate is None and self.perturbation is None:
            get_eta, get_eps = self.calibrate(fun, x0, max_evals_grouped=self._max_evals_grouped)
        else:
            get_eta, get_eps = _validate_pert_and_learningrate(
                self.perturbation, self.learning_rate
            )
        eta, eps = get_eta(), get_eps()

        if self.lse_solver is None:
            lse_solver = np.linalg.solve
        else:
            lse_solver = self.lse_solver

        # prepare some initials
        x = np.asarray(x0)
        if self.initial_hessian is None:
            self._smoothed_hessian = np.identity(x.size)
        else:
            self._smoothed_hessian = self.initial_hessian

        self._nfev = 0

        # if blocking is enabled we need to keep track of the function values
        if self.blocking:
            fx = fun(x)

            self._nfev += 1
            if self.allowed_increase is None:
                self.allowed_increase = 2 * self.estimate_stddev(
                    fun, x, max_evals_grouped=self._max_evals_grouped
                )

        logger.info("SPSA: Starting optimization.")
        start = time()

        # keep track of the last few steps to return their average
        last_steps = deque([x])

        # use a local variable and while loop to keep track of the number of iterations
        # if the termination checker terminates early
        k = 0
        while k < self.maxiter:
            k += 1
            iteration_start = time()
            # compute update
            fx_estimate, update = self._compute_update(fun, x, k, next(eps), lse_solver)

            # trust region
            if self.trust_region:
                norm = np.linalg.norm(update)
                if norm > 1:  # stop from dividing by 0
                    update = update / norm

            # compute next parameter value
            update = update * next(eta)
            x_next = x - update
            fx_next = None

            # blocking
            if self.blocking:
                self._nfev += 1
                fx_next = fun(x_next)

                if fx + self.allowed_increase <= fx_next:  # accept only if loss improved
                    if self.callback is not None:
                        self.callback(
                            self._nfev,  # number of function evals
                            x_next,  # next parameters
                            fx_next,  # loss at next parameters
                            np.linalg.norm(update),  # size of the update step
                            False,
                        )  # not accepted

                    logger.info(
                        "Iteration %s/%s rejected in %s.",
                        k,
                        self.maxiter + 1,
                        time() - iteration_start,
                    )
                    continue
                fx = fx_next

            logger.info(
                "Iteration %s/%s done in %s.", k, self.maxiter + 1, time() - iteration_start
            )

            if self.callback is not None:
                # if we didn't evaluate the function yet, do it now
                if not self.blocking:
                    self._nfev += 1
                    fx_next = fun(x_next)

                self.callback(
                    self._nfev,  # number of function evals
                    x_next,  # next parameters
                    fx_next,  # loss at next parameters
                    np.linalg.norm(update),  # size of the update step
                    True,
                )  # accepted

            # update parameters
            x = x_next

            # update the list of the last ``last_avg`` parameters
            if self.last_avg > 1:
                last_steps.append(x_next)
                if len(last_steps) > self.last_avg:
                    last_steps.popleft()

            if self.termination_checker is not None:
                fx_check = fx_estimate if fx_next is None else fx_next
                if self.termination_checker(
                    self._nfev, x_next, fx_check, np.linalg.norm(update), True
                ):
                    logger.info("terminated optimization at {k}/{self.maxiter} iterations")
                    break

        logger.info("SPSA: Finished in %s", time() - start)

        if self.last_avg > 1:
            x = np.mean(last_steps, axis=0)

        result = OptimizerResult()
        result.x = x
        result.fun = fun(x)
        result.nfev = self._nfev
        result.nit = k

        return result

    def get_support_level(self):
        """
        Get the support level dictionary.

        Returns:
            dict: A dictionary containing the support level of different optimizer features.

        """
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

    # pylint: disable=bad-docstring-quotes

    @deprecate_func(
        additional_msg=(
            "Instead, use ``SPSA.minimize`` as a replacement, which supports the same arguments "
            "but follows the interface of scipy.optimize and returns a complete result object "
            "containing additional information."
        ),
        since="0.21.0",
    )
    def optimize(
        self,
        num_vars,  # pylint: disable=unused-argument
        objective_function,
        gradient_function=None,  # pylint: disable=unused-argument
        variable_bounds=None,  # pylint: disable=unused-argument
        initial_point=None,
    ):
        """Perform optimization.

        Args:
            num_vars (int): Number of parameters to be optimized.
            objective_function (callable): A function that computes the objective function.
            gradient_function (callable): Not supported for SPSA.
            variable_bounds (list[(float, float)]): Not supported for SPSA.
            initial_point (numpy.ndarray[float]): Initial point.

        Returns:
            tuple: point, value, nfev
               point: is a 1D numpy.ndarray[float] containing the solution
               value: is a float with the objective function value
               nfev: number of objective function calls made if available or None
        """
        result = self.minimize(objective_function, initial_point)
        return result.x, result.fun, result.nfev


def bernoulli_perturbation(dim, perturbation_dims=None):
    """Get a Bernoulli random perturbation.

    Args:
        dim (int): The dimension of the perturbation.
        perturbation_dims (int or None, optional): The size of the perturbation. If
            None, it generates a perturbation of the same size as the dimension. Defaults
            to None.

    Returns:
        numpy.ndarray: The generated perturbation.
    """
    if perturbation_dims is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=dim)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        pert = 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=perturbation_dims)
        indices = algorithm_globals.random.choice(
            list(range(dim)), size=perturbation_dims, replace=False
        )
    result = np.zeros(dim)
    result[indices] = pert

    return result


def powerseries(eta=0.01, power=2, offset=0):
    """Yield a series decreasing by a powerlaw.

    Args:
        eta (float, optional): The initial value of the series. Defaults to 0.01.
        power (float, optional): The power for the power law. Defaults to 2.
        offset (float, optional): The offset for the power law. Defaults to 0.

    Yields:
        float: The next value in the series.
    """

    n = 1
    while True:
        yield eta / ((n + offset) ** power)
        n += 1


def constant(eta=0.01):
    """Yield a constant series.

    Args:
        eta (float, optional): The value of the series. Defaults to 0.01.

    Yields:
        float: The constant value.
    """

    while True:
        yield eta


def _batch_evaluate(function, points, max_evals_grouped, unpack_points=False):
    """
    Evaluate a function on all points with batches of max_evals_grouped.

    The points are a list of inputs, as ``[in1, in2, in3, ...]``. If the individual
    inputs are tuples (because the function takes multiple inputs), set
    ``unpack_points`` to ``True``.
    """

    # if the function cannot handle lists of points as input, cover this case immediately
    if max_evals_grouped is None or max_evals_grouped == 1:
        # support functions with multiple arguments where the points are given in a tuple
        return [
            function(*point) if isinstance(point, tuple) else function(point) for point in points
        ]

    num_points = len(points)

    # get the number of batches
    num_batches = num_points // max_evals_grouped
    if num_points % max_evals_grouped != 0:
        num_batches += 1

    # split the points
    batched_points = np.array_split(np.asarray(points), num_batches)

    results = []
    for batch in batched_points:
        if unpack_points:
            batch = _repack_points(batch)
            results += _as_list(function(*batch))
        else:
            results += _as_list(function(batch))

    return results


def _as_list(obj):
    """
    Convert a list or numpy array into a list.
    """
    return obj.tolist() if isinstance(obj, np.ndarray) else obj


def _repack_points(points):
    """Turn a list of tuples of points into a tuple of lists of points.
    E.g. turns
        [(a1, a2, a3), (b1, b2, b3)]
    into
        ([a1, b1], [a2, b2], [a3, b3])
    where all elements are np.ndarray.
    """
    num_sets = len(points[0])  # length of (a1, a2, a3)
    return ([x[i] for x in points] for i in range(num_sets))


def _make_spd(matrix, bias=0.01):
    identity = np.identity(matrix.shape[0])
    psd = scipy.linalg.sqrtm(matrix.dot(matrix))
    return psd + bias * identity


def _validate_pert_and_learningrate(perturbation, learning_rate):
    if learning_rate is None or perturbation is None:
        raise ValueError("If one of learning rate or perturbation is set, both must be set.")

    if isinstance(perturbation, float):

        def get_eps():
            return constant(perturbation)

    elif isinstance(perturbation, (list, np.ndarray)):

        def get_eps():
            return iter(perturbation)

    else:
        get_eps = perturbation

    if isinstance(learning_rate, float):

        def get_eta():
            return constant(learning_rate)

    elif isinstance(learning_rate, (list, np.ndarray)):

        def get_eta():
            return iter(learning_rate)

    else:
        get_eta = learning_rate

    return get_eta, get_eps

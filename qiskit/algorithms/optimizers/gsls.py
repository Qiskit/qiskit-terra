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

"""Line search with Gaussian-smoothed samples on a sphere."""

from __future__ import annotations

import warnings

from collections.abc import Callable
from typing import Any, SupportsFloat
import numpy as np

from qiskit.utils import algorithm_globals
from .optimizer import Optimizer, OptimizerSupportLevel, OptimizerResult, POINT


class GSLS(Optimizer):
    """Gaussian-smoothed Line Search.

    An implementation of the line search algorithm described in
    https://arxiv.org/pdf/1905.01332.pdf, using gradient approximation
    based on Gaussian-smoothed samples on a sphere.

    .. note::

        This component has some function that is normally random. If you want to reproduce behavior
        then you should set the random number generator seed in the algorithm_globals
        (``qiskit.utils.algorithm_globals.random_seed = seed``).
    """

    _OPTIONS = [
        "maxiter",
        "max_eval",
        "disp",
        "sampling_radius",
        "sample_size_factor",
        "initial_step_size",
        "min_step_size",
        "step_size_multiplier",
        "armijo_parameter",
        "min_gradient_norm",
        "max_failed_rejection_sampling",
    ]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: int = 10000,
        max_eval: int = 10000,
        disp: bool = False,
        sampling_radius: float = 1.0e-6,
        sample_size_factor: int = 1,
        initial_step_size: float = 1.0e-2,
        min_step_size: float = 1.0e-10,
        step_size_multiplier: float = 0.4,
        armijo_parameter: float = 1.0e-1,
        min_gradient_norm: float = 1e-8,
        max_failed_rejection_sampling: int = 50,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of iterations.
            max_eval: Maximum number of evaluations.
            disp: Set to True to display convergence messages.
            sampling_radius: Sampling radius to determine gradient estimate.
            sample_size_factor: The size of the sample set at each iteration is
                this number multiplied by the dimension of the problem, rounded to the
                nearest integer.
            initial_step_size: Initial step size for the descent algorithm.
            min_step_size: Minimum step size for the descent algorithm.
            step_size_multiplier: Step size reduction after unsuccessful steps, in
                the interval (0, 1).
            armijo_parameter: Armijo parameter for sufficient decrease criterion,
                in the interval (0, 1).
            min_gradient_norm: If the gradient norm is below this threshold, the algorithm stops.
            max_failed_rejection_sampling: Maximum number of attempts to sample
                points within bounds.
        """
        super().__init__()
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                self._options[k] = v

    def get_support_level(self) -> dict[str, int]:
        """Return support level dictionary.

        Returns:
            A dictionary containing the support levels for different options.
        """
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.supported,
            "initial_point": OptimizerSupportLevel.required,
        }

    @property
    def settings(self) -> dict[str, Any]:
        """Return the current settings.

        Returns:
            A dictionary with the current options.
        """
        return {key: self._options.get(key, None) for key in self._OPTIONS}

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """Perform the optimization.

        Args:
            fun: The objective function to be minimized.
            x0: The initial point for the optimization.
            jac: The gradient function, optional.
            bounds: The bounds for the variables, optional.

        Returns:
            An OptimizerResult instance with the optimization results.
        """
        if not isinstance(x0, np.ndarray):
            x0 = np.asarray(x0)

        if bounds is None:
            var_lb = np.array([-np.inf] * x0.size)
            var_ub = np.array([np.inf] * x0.size)
        else:
            var_lb = np.array([l for (l, _) in bounds])
            var_ub = np.array([u for (_, u) in bounds])

        x, fun, nfev, _ = self.ls_optimize(x0.size, fun, x0, var_lb, var_ub)

        result = OptimizerResult()
        result.x = x
        result.fun = fun
        result.nfev = nfev

        return result

    def ls_optimize(
        self,
        n: int,
        obj_fun: Callable[[POINT], float],
        initial_point: np.ndarray,
        var_lb: np.ndarray,
        var_ub: np.ndarray,
    ) -> tuple[np.ndarray, float, int, float]:
        """Run the line search optimization.

        Args:
            n: Dimension of the problem.
            obj_fun: Objective function.
            initial_point: Initial point.
            var_lb: Vector of lower bounds on the decision variables. Vector
                elements can be -np.inf if the corresponding variable is unbounded
                from below.
            var_ub: Vector of upper bounds on the decision variables. Vector
                elements can be np.inf if the corresponding variable is unbounded from
                below.

        Returns:
            Final iterate as a vector, corresponding objective function value,
            number of evaluations, and norm of the gradient estimate.

        Raises:
            ValueError: If the number of dimensions mismatches the size of the initial point or
                the length of the lower or upper bound.
        """
        if len(initial_point) != n:
            raise ValueError("Size of the initial point mismatches the number of dimensions.")
        if len(var_lb) != n:
            raise ValueError("Length of the lower bound mismatches the number of dimensions.")
        if len(var_ub) != n:
            raise ValueError("Length of the upper bound mismatches the number of dimensions.")

        # Initialize counters and data
        iter_count = 0
        n_evals = 0
        prev_iter_successful = True
        prev_directions, prev_sample_set_x, prev_sample_set_y = None, None, None
        consecutive_fail_iter = 0
        alpha = self._options["initial_step_size"]
        grad_norm: SupportsFloat = np.inf
        sample_set_size = int(round(self._options["sample_size_factor"] * n))

        # Initial point
        x = initial_point
        x_value = obj_fun(x)
        n_evals += 1
        while iter_count < self._options["maxiter"] and n_evals < self._options["max_eval"]:

            # Determine set of sample points
            directions, sample_set_x = self.sample_set(n, x, var_lb, var_ub, sample_set_size)

            if n_evals + len(sample_set_x) + 1 >= self._options["max_eval"]:
                # The evaluation budget is too small to allow for
                # another full iteration; we therefore exit now
                break

            sample_set_y = np.array([obj_fun(point) for point in sample_set_x])
            n_evals += len(sample_set_x)

            # Expand sample set if we could not improve
            if not prev_iter_successful:
                directions = np.vstack((prev_directions, directions))
                sample_set_x = np.vstack((prev_sample_set_x, sample_set_x))
                sample_set_y = np.hstack((prev_sample_set_y, sample_set_y))

            # Find gradient approximation and candidate point
            grad = self.gradient_approximation(
                n, x, x_value, directions, sample_set_x, sample_set_y
            )
            grad_norm = np.linalg.norm(grad)
            new_x = np.clip(x - alpha * grad, var_lb, var_ub)
            new_x_value = obj_fun(new_x)
            n_evals += 1

            # Print information
            if self._options["disp"]:
                print(f"Iter {iter_count:d}")
                print(f"Point {x} obj {x_value}")
                print(f"Gradient {grad}")
                print(f"Grad norm {grad_norm} new_x_value {new_x_value} step_size {alpha}")
                print(f"Direction {directions}")

            # Test Armijo condition for sufficient decrease
            if new_x_value <= x_value - self._options["armijo_parameter"] * alpha * grad_norm:
                # Accept point
                x, x_value = new_x, new_x_value
                alpha /= 2 * self._options["step_size_multiplier"]
                prev_iter_successful = True
                consecutive_fail_iter = 0

                # Reset sample set
                prev_directions = None
                prev_sample_set_x = None
                prev_sample_set_y = None
            else:
                # Do not accept point
                alpha *= self._options["step_size_multiplier"]
                prev_iter_successful = False
                consecutive_fail_iter += 1

                # Store sample set to enlarge it
                prev_directions = directions
                prev_sample_set_x, prev_sample_set_y = sample_set_x, sample_set_y

            iter_count += 1

            # Check termination criterion
            if (
                grad_norm <= self._options["min_gradient_norm"]
                or alpha <= self._options["min_step_size"]
            ):
                break

        return x, x_value, n_evals, grad_norm

    def sample_points(
        self, n: int, x: np.ndarray, num_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``num_points`` points around ``x`` on the ``n``-sphere of specified radius.

        The radius of the sphere is ``self._options['sampling_radius']``.

        Args:
            n: Dimension of the problem.
            x: Point around which the sample set is constructed.
            num_points: Number of points in the sample set.

        Returns:
            A tuple containing the sampling points and the directions.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            normal_samples = algorithm_globals.random.normal(size=(num_points, n))
        row_norms = np.linalg.norm(normal_samples, axis=1, keepdims=True)
        directions = normal_samples / row_norms
        points = x + self._options["sampling_radius"] * directions

        return points, directions

    def sample_set(
        self, n: int, x: np.ndarray, var_lb: np.ndarray, var_ub: np.ndarray, num_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct sample set of given size.

        Args:
            n: Dimension of the problem.
            x: Point around which the sample set is constructed.
            var_lb: Vector of lower bounds on the decision variables. Vector
                elements can be -np.inf if the corresponding variable is unbounded
                from below.
            var_ub: Vector of lower bounds on the decision variables. Vector
                elements can be np.inf if the corresponding variable is unbounded from
                above.
            num_points: Number of points in the sample set.

        Returns:
            Matrices of (unit-norm) sample directions and sample points, one per row.
            Both matrices are 2D arrays of floats.

        Raises:
            RuntimeError: If not enough samples could be generated within the bounds.
        """
        # Generate points uniformly on the sphere
        points, directions = self.sample_points(n, x, num_points)

        # Check bounds
        if (points >= var_lb).all() and (points <= var_ub).all():
            # If all points are within bounds, return them
            return directions, (x + self._options["sampling_radius"] * directions)
        else:
            # Otherwise we perform rejection sampling until we have
            # enough points that satisfy the bounds
            indices = np.where((points >= var_lb).all(axis=1) & (points <= var_ub).all(axis=1))[0]
            accepted = directions[indices]
            num_trials = 0

            while (
                len(accepted) < num_points
                and num_trials < self._options["max_failed_rejection_sampling"]
            ):
                # Generate points uniformly on the sphere
                points, directions = self.sample_points(n, x, num_points)
                indices = np.where((points >= var_lb).all(axis=1) & (points <= var_ub).all(axis=1))[
                    0
                ]
                accepted = np.vstack((accepted, directions[indices]))
                num_trials += 1

            # When we are at a corner point, the expected fraction of acceptable points may be
            # exponential small in the dimension of the problem. Thus, if we keep failing and
            # do not have enough points by now, we switch to a different method that guarantees
            # finding enough points, but they may not be uniformly distributed.
            if len(accepted) < num_points:
                points, directions = self.sample_points(n, x, num_points)
                to_be_flipped = (points < var_lb) | (points > var_ub)
                directions *= np.where(to_be_flipped, -1, 1)
                points = x + self._options["sampling_radius"] * directions
                indices = np.where((points >= var_lb).all(axis=1) & (points <= var_ub).all(axis=1))[
                    0
                ]
                accepted = np.vstack((accepted, directions[indices]))

            # If we still do not have enough sampling points, we have failed.
            if len(accepted) < num_points:
                raise RuntimeError(
                    "Could not generate enough samples within bounds; try smaller radius."
                )

            return (
                accepted[:num_points],
                x + self._options["sampling_radius"] * accepted[:num_points],
            )

    def gradient_approximation(
        self,
        n: int,
        x: np.ndarray,
        x_value: float,
        directions: np.ndarray,
        sample_set_x: np.ndarray,
        sample_set_y: np.ndarray,
    ) -> np.ndarray:
        """Construct gradient approximation from given sample.

        Args:
            n: Dimension of the problem.
            x: Point around which the sample set was constructed.
            x_value: Objective function value at x.
            directions: Directions of the sample points wrt the central point x, as a 2D array.
            sample_set_x: x-coordinates of the sample set, one point per row, as a 2D array.
            sample_set_y: Objective function values of the points in sample_set_x, as a 1D array.

        Returns:
            Gradient approximation at x, as a 1D array.
        """
        ffd = sample_set_y - x_value
        gradient = (
            float(n)
            / len(sample_set_y)
            * np.sum(
                ffd.reshape(len(sample_set_y), 1) / self._options["sampling_radius"] * directions, 0
            )
        )
        return gradient

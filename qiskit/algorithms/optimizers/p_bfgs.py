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

"""Parallelized Limited-memory BFGS optimizer"""
from __future__ import annotations

import logging
import multiprocessing
import platform
import warnings
from collections.abc import Callable
from typing import SupportsFloat

import numpy as np

from qiskit.utils import algorithm_globals
from qiskit.utils.validation import validate_min

from .optimizer import OptimizerResult, POINT
from .scipy_optimizer import SciPyOptimizer

logger = logging.getLogger(__name__)


class P_BFGS(SciPyOptimizer):  # pylint: disable=invalid-name
    """
    Parallelized Limited-memory BFGS optimizer.

    P-BFGS is a parallelized version of :class:`L_BFGS_B` with which it shares the
    same parameters.
    P-BFGS can be useful when the target hardware is a quantum simulator running on
    a classical machine. This allows the multiple processes to use simulation
    to potentially reach a minimum faster. The parallelization may also help
    the optimizer avoid getting stuck at local optima.

    Uses scipy.optimize.fmin_l_bfgs_b.
    For further detail, please refer to
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
    """

    _OPTIONS = ["maxfun", "ftol", "iprint"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxfun: int = 1000,
        ftol: SupportsFloat = 10 * np.finfo(float).eps,
        iprint: int = -1,
        max_processes: int | None = None,
        options: dict | None = None,
        max_evals_grouped: int = 1,
        **kwargs,
    ) -> None:
        r"""
        Initialize the class instance.

        Args:
            maxfun (int): Maximum number of function evaluations.
            ftol (SupportsFloat): The iteration stops when (f^k -
                f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
            iprint (int): Controls the frequency of output. iprint < 0 means no
                output; iprint = 0 print only one line at the last iteration; 0 <
                iprint < 99 print also f and |proj g| every iprint iterations; iprint
                = 99 print details of every iteration except n-vectors; iprint = 100
                print also the changes of active set and final x; iprint > 100 print
                details of every iteration including x and g.
            max_processes (int | None): maximum number of processes allowed, has a min.
                value of 1 if not None.
            options (dict | None): A dictionary of solver options.
            max_evals_grouped (int): Max number of default gradient evaluations
                performed simultaneously.
            kwargs: additional kwargs for scipy.optimize.minimize.
        """
        if max_processes:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                validate_min("max_processes", max_processes, 1)

        if options is None:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__(
            method="L-BFGS-B",
            options=options,
            max_evals_grouped=max_evals_grouped,
            **kwargs,
        )
        self._max_processes = max_processes

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        x0 = np.asarray(x0)

        num_procs = multiprocessing.cpu_count() - 1
        num_procs = (
            num_procs if self._max_processes is None else min(num_procs, self._max_processes)
        )
        num_procs = num_procs if num_procs >= 0 else 0

        if platform.system() == "Darwin":
            # Changed in version 3.8: On macOS, the spawn start method is now the
            # default. The fork start method should be considered unsafe as it can
            # lead to crashes.
            # However P_BFGS doesn't support spawn, so we revert to single process.
            num_procs = 0
            logger.warning(
                "For MacOS, python >= 3.8, using only current process. "
                "Multiple core use not supported."
            )
        elif platform.system() == "Windows":
            num_procs = 0
            logger.warning(
                "For Windows, using only current process. Multiple core use not supported."
            )

        queue: multiprocessing.queues.Queue[tuple[POINT, float, int]] = multiprocessing.Queue()

        # TODO: are automatic bounds a good idea? What if the circuit parameters are not
        # just from plain Pauli rotations but have a coefficient?

        # bounds for additional initial points in case bounds has any None values
        threshold = 2 * np.pi
        if bounds is None:
            bounds = [(-threshold, threshold)] * x0.size
        low = [(l if l is not None else -threshold) for (l, u) in bounds]
        high = [(u if u is not None else threshold) for (l, u) in bounds]

        def optimize_runner(_queue, _i_pt):  # Multi-process sampling
            _sol, _opt, _nfev = self._optimize(fun, _i_pt, jac, bounds)
            _queue.put((_sol, _opt, _nfev))

        # Start off as many other processes running the optimize (can be 0)
        processes = []
        for _ in range(num_procs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                i_pt = algorithm_globals.random.uniform(low, high)  # Another random point in bounds
            proc = multiprocessing.Process(target=optimize_runner, args=(queue, i_pt))
            processes.append(proc)
            proc.start()

        # While the one optimize in this process below runs the other processes will
        # be running too. This one runs
        # with the supplied initial point. The process ones have their own random one
        sol, opt, nfev = self._optimize(fun, x0, jac, bounds)

        for proc in processes:
            # For each other process we wait now for it to finish and see if it has
            # a better result than above
            proc.join()
            p_sol, p_opt, p_nfev = queue.get()
            if p_opt < opt:
                sol, opt = p_sol, p_opt
            nfev += p_nfev

        result = OptimizerResult()
        result.x = sol
        result.fun = opt
        result.nfev = nfev

        return result

    def _optimize(
        self,
        objective_function,
        initial_point,
        gradient_function=None,
        variable_bounds=None,
    ) -> tuple[POINT, float, int]:
        """
        Optimize the objective function using the minimize method.

        This method calls the minimize method from the superclass to perform the
        optimization of the objective function. It takes the objective function,
        initial point, optional gradient function, and optional variable bounds as
        input parameters. It returns a tuple containing the optimized point, the
        value of the objective function at the optimized point, and the number of
        function evaluations performed.

        Args:
            objective_function: The objective function to be optimized.
            initial_point: The initial point for the optimization.
            gradient_function: The gradient function for the optimization.
            (default: None) variable_bounds: The variable bounds for the
            optimization. (default: None)

        Returns:
            tuple[POINT, float, int]: A tuple containing the optimized point, the
            value of the objective function at the optimized point, and the number
            of function evaluations performed.
        """
        result = super().minimize(
            objective_function, initial_point, gradient_function, variable_bounds
        )
        return result.x, result.fun, result.nfev

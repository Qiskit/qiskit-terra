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

"""Powell optimizer."""
from __future__ import annotations

from .scipy_optimizer import SciPyOptimizer


class POWELL(SciPyOptimizer):
    """
    Powell optimizer.

    The Powell algorithm performs unconstrained optimization; it ignores bounds or
    constraints. Powell is a *conjugate direction method*: it performs sequential one-dimensional
    minimization along each directional vector, which is updated at
    each iteration of the main minimization loop. The function being minimized need not be
    differentiable, and no derivatives are taken.

    Uses scipy.optimize.minimize Powell.
    For further detail, please refer to
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _OPTIONS = ["maxiter", "maxfev", "disp", "xtol"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: int | None = None,
        maxfev: int = 1000,
        disp: bool = False,
        xtol: float = 0.0001,
        tol: float | None = None,
        options: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the class instance.

        Args:
            maxiter: Maximum allowed number of iterations. If both maxiter and
                maxfev are set, minimization will stop at the first reached.
            maxfev: Maximum allowed number of function evaluations. If both
                maxiter and maxfev are set, minimization will stop at the first
                reached.
            disp: Set to True to print convergence messages.
            xtol: Relative error in solution xopt acceptable for convergence.
            tol: Tolerance for termination.
            options: A dictionary of solver options.
            kwargs: additional kwargs for scipy.optimize.minimize.
        """
        if options is None:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__("Powell", options=options, tol=tol, **kwargs)

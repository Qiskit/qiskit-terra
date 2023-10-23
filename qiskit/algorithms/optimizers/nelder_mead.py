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

"""Nelder-Mead optimizer."""
from __future__ import annotations


from .scipy_optimizer import SciPyOptimizer


class NELDER_MEAD(SciPyOptimizer):  # pylint: disable=invalid-name
    """
    Nelder-Mead optimizer.

    The Nelder-Mead algorithm performs unconstrained optimization; it ignores bounds
    or constraints.  It is used to find the minimum or maximum of an objective
    function in a multidimensional space.  It is based on the Simplex
    algorithm. Nelder-Mead is robust in many applications, especially when the
    first and second derivatives of the objective function are not known.

    However, if the numerical computation of the derivatives can be trusted to be accurate,
    other algorithms using the first and/or second derivatives information might be preferred
    to Nelder-Mead for their better performance in the general case, especially in
    consideration of the fact that the Nelder–Mead technique is a heuristic
    search method that can converge to non-stationary points.

    Uses scipy.optimize.minimize Nelder-Mead.
    For further detail, please refer to
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _OPTIONS = ["maxiter", "maxfev", "disp", "xatol", "adaptive"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: int | None = None,
        maxfev: int = 1000,
        disp: bool = False,
        xatol: float = 0.0001,
        tol: float | None = None,
        adaptive: bool = False,
        options: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Initializes the object with various parameters and sets up options for optimization.

        Args:
            maxiter: Maximum allowed number of iterations. If both maxiter and
                maxfev are set, minimization will stop at the first reached.
            maxfev: Maximum allowed number of function evaluations. If both
                maxiter and maxfev are set, minimization will stop at the first
                reached.
            disp: Set to True to print convergence messages.
            xatol: Absolute error in xopt between iterations that is acceptable for convergence.
            tol: Tolerance for termination.
            adaptive: Adapt algorithm parameters to dimensionality of problem.
            options: A dictionary of solver options.
            kwargs: additional kwargs for scipy.optimize.minimize.
        """
        if options is None:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__(method="Nelder-Mead", options=options, tol=tol, **kwargs)

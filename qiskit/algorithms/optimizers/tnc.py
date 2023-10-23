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

"""Truncated Newton (TNC) optimizer."""
from __future__ import annotations


from .scipy_optimizer import SciPyOptimizer


class TNC(SciPyOptimizer):
    """
    Truncated Newton (TNC) optimizer.

    TNC uses a truncated Newton algorithm to minimize a function with variables
    subject to bounds.
    This algorithm uses gradient information; it is also called Newton Conjugate-Gradient.
    It differs from the :class:`CG` method as it wraps a C implementation and
    allows each variable to be given upper and lower bounds.

    Uses scipy.optimize.minimize TNC
    For further detail, please refer to
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _OPTIONS = ["maxiter", "disp", "accuracy", "ftol", "xtol", "gtol", "eps"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: int = 100,
        disp: bool = False,
        accuracy: float = 0,
        ftol: float = -1,
        xtol: float = -1,
        gtol: float = -1,
        tol: float | None = None,
        eps: float = 1e-08,
        options: dict | None = None,
        max_evals_grouped: int = 1,
        **kwargs,
    ) -> None:
        """
        Initialize the object.

        Args:
            maxiter (int): Maximum number of function evaluations.
            disp (bool): Set to True to print convergence messages.
            accuracy (float): Relative precision for finite difference calculations.
                If <= machine_precision, set to sqrt(machine_precision). Defaults to 0.
            ftol (float): Precision goal for the value of f in the stopping criterion.
                If ftol < 0.0, ftol is set to 0.0 defaults to -1.
            xtol (float): Precision goal for the value of x in the stopping criterion
                    (after applying x scaling factors).
                If xtol < 0.0, xtol is set to sqrt(machine_precision). Defaults to -1.
            gtol (float): Precision goal for the value of the projected gradient
                in the stopping criterion (after applying x scaling factors).
                If gtol < 0.0, gtol is set to 1e-2 * sqrt(accuracy).
                Setting it to 0.0 is not recommended. Defaults to -1.
            tol (float | None): Tolerance for termination.
            eps (float): Step size used for numerical approximation of the Jacobian.
            options (dict | None): A dictionary of solver options.
            max_evals_grouped (int): Max number of default gradient evaluations
                performed simultaneously.
            kwargs: additional kwargs for scipy.optimize.minimize.
        """
        if options is None:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__(
            "TNC",
            options=options,
            tol=tol,
            max_evals_grouped=max_evals_grouped,
            **kwargs,
        )

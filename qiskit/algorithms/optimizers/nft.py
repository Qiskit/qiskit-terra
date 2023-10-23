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

"""Nakanishi-Fujii-Todo algorithm."""
from __future__ import annotations


import numpy as np
from scipy.optimize import OptimizeResult

from .scipy_optimizer import SciPyOptimizer


class NFT(SciPyOptimizer):
    """
    Nakanishi-Fujii-Todo algorithm.

    See https://arxiv.org/abs/1903.12166
    """

    _OPTIONS = ["maxiter", "maxfev", "disp", "reset_interval"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: int | None = None,
        maxfev: int = 1024,
        disp: bool = False,
        reset_interval: int = 32,
        options: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Built out using scipy framework, for details, please refer to
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

        Args:
            maxiter: Maximum number of iterations to perform.
            maxfev: Maximum number of function evaluations to perform.
            disp: disp
            reset_interval: The minimum estimates directly once
                            in ``reset_interval`` times.
            options: A dictionary of solver options.
            kwargs: additional kwargs for scipy.optimize.minimize.

        Notes:
            In this optimization method, the optimization function have to satisfy
            three conditions written in [1]_.

        References:
            .. [1] K. M. Nakanishi, K. Fujii, and S. Todo. 2019.
                Sequential minimal optimization for quantum-classical hybrid algorithms.
                arXiv preprint arXiv:1903.12166.
        """
        if options is None:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__(method=nakanishi_fujii_todo, options=options, **kwargs)


# pylint: disable=invalid-name
def nakanishi_fujii_todo(
    fun, x0, args=(), maxiter=None, maxfev=1024, reset_interval=32, eps=1e-32, callback=None, **_
):
    """
    Find the global minimum of a function using the nakanishi_fujii_todo
    algorithm [1].

    Args:
        fun (callable): ``f(x, *args)``
            Function to be optimized.  ``args`` can be passed as an optional item
            in the dict ``minimizer_kwargs``.
            This function must satisfy the three condition written in Ref. [1].
        x0 (ndarray): shape (n,)
            Initial guess. Array of real elements of size (n,),
            where 'n' is the number of independent variables.
        args (tuple, optional):
            Extra arguments passed to the objective function.
        maxiter (int): Maximum number of iterations to perform.
            Default: None.
        maxfev (int): Maximum number of function evaluations to perform.
            Default: 1024.
        reset_interval (int): The minimum estimates directly once in ``reset_interval`` times.
            Default: 32.
        eps (float): eps
        callback (callable, optional): Called after each iteration.
        **_ : additional options

    Returns:
        OptimizeResult:
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array. See
            `OptimizeResult` for a description of other attributes.

    Notes:
        In this optimization method, the optimization function have to satisfy
        three conditions written in [2]_.

    References:
        .. [2] K. M. Nakanishi, K. Fujii, and S. Todo. 2019.
        Sequential minimal optimization for quantum-classical hybrid algorithms.
        arXiv preprint arXiv:1903.12166.
    """

    x0 = np.asarray(x0)
    recycle_z0 = None
    niter = 0
    funcalls = 0

    while True:

        idx = niter % x0.size

        if reset_interval > 0:
            if niter % reset_interval == 0:
                recycle_z0 = None

        if recycle_z0 is None:
            z0 = fun(np.copy(x0), *args)
            funcalls += 1
        else:
            z0 = recycle_z0

        p = np.copy(x0)
        p[idx] = x0[idx] + np.pi / 2
        z1 = fun(p, *args)
        funcalls += 1

        p = np.copy(x0)
        p[idx] = x0[idx] - np.pi / 2
        z3 = fun(p, *args)
        funcalls += 1

        z2 = z1 + z3 - z0
        c = (z1 + z3) / 2
        a = np.sqrt((z0 - z2) ** 2 + (z1 - z3) ** 2) / 2
        b = np.arctan((z1 - z3) / ((z0 - z2) + eps * (z0 == z2))) + x0[idx]
        b += 0.5 * np.pi + 0.5 * np.pi * np.sign((z0 - z2) + eps * (z0 == z2))

        x0[idx] = b
        recycle_z0 = c - a

        niter += 1

        if callback is not None:
            callback(np.copy(x0))

        if maxfev is not None:
            if funcalls >= maxfev:
                break

        if maxiter is not None:
            if niter >= maxiter:
                break

    return OptimizeResult(
        fun=fun(np.copy(x0), *args), x=x0, nit=niter, nfev=funcalls, success=(niter > 1)
    )

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The QN-SPSA optimizer."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Callable

import numpy as np
from qiskit.providers import Backend
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.opflow import StateFn, CircuitSampler, ExpectationBase
from qiskit.utils import QuantumInstance
from qiskit.utils.deprecation import deprecate_arg

from qiskit.primitives import BaseSampler, Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute

from .spsa import SPSA, CALLBACK, TERMINATIONCHECKER, _batch_evaluate

# the function to compute the fidelity
FIDELITY = Callable[[np.ndarray, np.ndarray], float]


class QNSPSA(SPSA):
    r"""The Quantum Natural SPSA (QN-SPSA) optimizer.

    The QN-SPSA optimizer [1] is a stochastic optimizer that belongs to the family of gradient
    descent methods. This optimizer is based on SPSA but attempts to improve the convergence by
    sampling the **natural gradient** instead of the vanilla, first-order gradient. It achieves
    this by approximating Hessian of the ``fidelity`` of the ansatz circuit.

    Compared to natural gradients, which require :math:`\mathcal{O}(d^2)` expectation value
    evaluations for a circuit with :math:`d` parameters, QN-SPSA only requires
    :math:`\mathcal{O}(1)` and can therefore significantly speed up the natural gradient calculation
    by sacrificing some accuracy. Compared to SPSA, QN-SPSA requires 4 additional function
    evaluations of the fidelity.

    The stochastic approximation of the natural gradient can be systematically improved by
    increasing the number of ``resamplings``. This leads to a Monte Carlo-style convergence to
    the exact, analytic value.

    .. note::

        This component has some function that is normally random. If you want to reproduce behavior
        then you should set the random number generator seed in the algorithm_globals
        (``qiskit.utils.algorithm_globals.random_seed = seed``).

    Examples:

        This short example runs QN-SPSA for the ground state calculation of the ``Z ^ Z``
        observable where the ansatz is a ``PauliTwoDesign`` circuit.

        .. code-block:: python

            import numpy as np
            from qiskit.algorithms.optimizers import QNSPSA
            from qiskit.circuit.library import PauliTwoDesign
            from qiskit.primitives import Estimator, Sampler
            from qiskit.quantum_info import Pauli

            # problem setup
            ansatz = PauliTwoDesign(2, reps=1, seed=2)
            observable = Pauli("ZZ")
            initial_point = np.random.random(ansatz.num_parameters)

            # loss function
            estimator = Estimator()

            def loss(x):
                result = estimator.run([ansatz], [observable], [x]).result()
                return np.real(result.values[0])

            # fidelity for estimation of the geometric tensor
            sampler = Sampler()
            fidelity = QNSPSA.get_fidelity(ansatz, sampler)

            # run QN-SPSA
            qnspsa = QNSPSA(fidelity, maxiter=300)
            result = qnspsa.optimize(ansatz.num_parameters, loss, initial_point=initial_point)

        This is a legacy version solving the same problem but using Qiskit Opflow instead
        of the Qiskit Primitives. Note however, that this usage is deprecated.

        .. code-block:: python

            import numpy as np
            from qiskit.algorithms.optimizers import QNSPSA
            from qiskit.circuit.library import PauliTwoDesign
            from qiskit.opflow import Z, StateFn

            ansatz = PauliTwoDesign(2, reps=1, seed=2)
            observable = Z ^ Z
            initial_point = np.random.random(ansatz.num_parameters)

            def loss(x):
                bound = ansatz.assign_parameters(x)
                return np.real((StateFn(observable, is_measurement=True) @ StateFn(bound)).eval())

            fidelity = QNSPSA.get_fidelity(ansatz)
            qnspsa = QNSPSA(fidelity, maxiter=300)
            result = qnspsa.optimize(ansatz.num_parameters, loss, initial_point=initial_point)


    References:

        [1] J. Gacon et al, "Simultaneous Perturbation Stochastic Approximation of the Quantum
        Fisher Information", `arXiv:2103.09232 <https://arxiv.org/abs/2103.09232>`_

    """

    def __init__(
        self,
        fidelity: FIDELITY,
        maxiter: int = 100,
        blocking: bool = True,
        allowed_increase: float | None = None,
        learning_rate: float | Callable[[], Iterator] | None = None,
        perturbation: float | Callable[[], Iterator] | None = None,
        resamplings: int | dict[int, int] = 1,
        perturbation_dims: int | None = None,
        regularization: float | None = None,
        hessian_delay: int = 0,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        initial_hessian: np.ndarray | None = None,
        callback: CALLBACK | None = None,
        termination_checker: TERMINATIONCHECKER | None = None,
    ) -> None:
        """
        Initialize an instance of the class with various parameters.

        Args:
            fidelity: A function to compute the fidelity of the ansatz state with
                itself for two different sets of parameters.
            maxiter: The maximum number of iterations. Note that this is not the
                maximal number of function evaluations.
            blocking: If True, only accepts updates that improve the loss (up to
                some allowed increase, see next argument).
            allowed_increase: If ``blocking`` is ``True``, this argument
                determines by how much the loss can increase with the proposed
                parameters and still be accepted.
                If ``None``, the allowed increases is calibrated automatically to be twice the
                approximated standard deviation of the loss function.
            learning_rate: The update step is the learning rate is multiplied with the gradient.
                If the learning rate is a float, it remains constant over the course of the
                optimization. It can also be a callable returning an iterator
                which yields the learning rates for each optimization step.
                If ``learning_rate`` is set ``perturbation`` must also be provided.
            perturbation: Specifies the magnitude of the perturbation for the
                finite difference approximation of the gradients. Can be either a
                float or a generator yielding the perturbation magnitudes per step.
                If ``perturbation`` is set ``learning_rate`` must also be provided.
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
            step. The information is, in this order: the parameters, the function
                value, the number of function evaluations, the stepsize, whether the
                step was accepted.
            termination_checker: A callback function executed at the end of each
            iteration step. The arguments are, in this order: the parameters, the
                function value, the number of function evaluations, the stepsize,
                whether the step was accepted. If the callback returns True, the
                optimization is terminated.
            To prevent additional evaluations of the objective method, if the objective has not yet
                been evaluated, the objective is estimated by taking the mean of
                the objective evaluations used in the estimate of the gradient.

        """
        super().__init__(
            maxiter,
            blocking,
            allowed_increase,
            # trust region *must* be false for natural gradients to work
            trust_region=False,
            learning_rate=learning_rate,
            perturbation=perturbation,
            resamplings=resamplings,
            callback=callback,
            second_order=True,
            hessian_delay=hessian_delay,
            lse_solver=lse_solver,
            regularization=regularization,
            perturbation_dims=perturbation_dims,
            initial_hessian=initial_hessian,
            termination_checker=termination_checker,
        )

        self.fidelity = fidelity

    def _point_sample(self, loss, x, eps, delta1, delta2):
        """
        Compute the point sample for a given loss function and input parameters.

        This method calculates the loss points and fidelity points based on the input
        parameters.
        It then evaluates the loss and fidelity functions for these points.
        The gradient approximation and the preconditioner point estimate are computed
        based on the evaluated values.
        Finally, the method returns the mean of the loss values, the gradient estimate,
        and the hessian estimate.

        Args:
            loss: The loss function.
            x: The input parameters.
            eps: The epsilon value.
            delta1: The first delta value.
            delta2: The second delta value.

        Returns:
            tuple: A tuple containing the mean of the loss values, the gradient
            estimate, and the hessian estimate.
        """
        loss_points = [x + eps * delta1, x - eps * delta1]
        fidelity_points = [
            (x, x + eps * delta1),
            (x, x - eps * delta1),
            (x, x + eps * (delta1 + delta2)),
            (x, x + eps * (-delta1 + delta2)),
        ]
        self._nfev += 6

        loss_values = _batch_evaluate(loss, loss_points, self._max_evals_grouped)
        fidelity_values = _batch_evaluate(
            self.fidelity, fidelity_points, self._max_evals_grouped, unpack_points=True
        )

        # compute the gradient approximation and additionally return the loss function evaluations
        gradient_estimate = (loss_values[0] - loss_values[1]) / (2 * eps) * delta1

        # compute the preconditioner point estimate
        fidelity_values = np.asarray(fidelity_values, dtype=float)
        diff = fidelity_values[2] - fidelity_values[0]
        diff = diff - (fidelity_values[3] - fidelity_values[1])
        diff = diff / (2 * eps**2)

        rank_one = np.outer(delta1, delta2)
        # -0.5 factor comes from the fact that we need -0.5 * fidelity
        hessian_estimate = -0.5 * diff * (rank_one + rank_one.T) / 2

        return np.mean(loss_values), gradient_estimate, hessian_estimate

    @property
    def settings(self) -> dict[str, Any]:
        """The optimizer settings in a dictionary format."""
        # re-use serialization from SPSA
        settings = super().settings
        settings.update({"fidelity": self.fidelity})

        # remove SPSA-specific arguments not in QNSPSA
        settings.pop("trust_region")
        settings.pop("second_order")

        return settings

    @staticmethod
    @deprecate_arg(
        "backend",
        since="0.24.0",
        additional_msg="See https://qisk.it/algo_migration for a migration guide.",
        # We allow passing a sampler as the second argument because that will become a positional
        # argument for `sampler` after removing `backend` and `expectation`.
        predicate=lambda backend: not isinstance(backend, BaseSampler),
    )
    @deprecate_arg(
        "expectation",
        since="0.24.0",
        additional_msg="See https://qisk.it/algo_migration for a migration guide.",
    )
    def get_fidelity(
        circuit: QuantumCircuit,
        backend: Backend | QuantumInstance | None = None,
        expectation: ExpectationBase | None = None,
        *,
        sampler: BaseSampler | None = None,
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        r"""Get a function to compute the fidelity of ``circuit`` with itself.

        .. note::

            Using this function with a backend and expectation converter is pending deprecation,
            instead pass a Qiskit Primitive sampler, such as :class:`~.Sampler`.
            The sampler can be passed as keyword argument or, positionally, as second argument.

        Let ``circuit`` be a parameterized quantum circuit performing the operation
        :math:`U(\theta)` given a set of parameters :math:`\theta`. Then this method returns
        a function to evaluate

        .. math::

            F(\theta, \phi) = \big|\langle 0 | U^\dagger(\theta) U(\phi) |0\rangle  \big|^2.

        The output of this function can be used as input for the ``fidelity`` to the
        :class:`~.QNSPSA` optimizer.

        Args:
            circuit: The circuit preparing the parameterized ansatz.
            backend: Deprecated. A backend of quantum instance to evaluate the circuits.
                If None, plain matrix multiplication will be used.
            expectation: Deprecated. An expectation converter to specify how the expected
                value is computed. If a shot-based readout is used this should be set to
                ``PauliExpectation``.
            sampler: A sampler primitive to sample from a quantum state.

        Returns:
            A handle to the function :math:`F`.

        """
        # allow passing sampler by position
        if isinstance(backend, BaseSampler):
            sampler = backend
            backend = None

        if expectation is None and backend is None and sampler is None:
            sampler = Sampler()

        if expectation is not None or backend is not None:
            return QNSPSA._legacy_get_fidelity(circuit, backend, expectation)

        fid = ComputeUncompute(sampler)

        num_parameters = circuit.num_parameters

        def fidelity(values_x, values_y):
            values_x = np.reshape(values_x, (-1, num_parameters)).tolist()
            batch_size_x = len(values_x)

            values_y = np.reshape(values_y, (-1, num_parameters)).tolist()
            batch_size_y = len(values_y)

            result = fid.run(
                batch_size_x * [circuit], batch_size_y * [circuit], values_x, values_y
            ).result()
            return np.asarray(result.fidelities)

        return fidelity

    @staticmethod
    def _legacy_get_fidelity(
        circuit: QuantumCircuit,
        backend: Backend | QuantumInstance | None = None,
        expectation: ExpectationBase | None = None,
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        r"""Deprecated. Get a function to compute the fidelity of ``circuit`` with itself.

        .. note::

            This method is deprecated. Instead use the :class:`~.ComputeUncompute`
            class which implements the fidelity calculation in the same fashion as this method.

        Let ``circuit`` be a parameterized quantum circuit performing the operation
        :math:`U(\theta)` given a set of parameters :math:`\theta`. Then this method returns
        a function to evaluate

        .. math::

            F(\theta, \phi) = \big|\langle 0 | U^\dagger(\theta) U(\phi) |0\rangle  \big|^2.

        The output of this function can be used as input for the ``fidelity`` to the
        :class:~`qiskit.algorithms.optimizers.QNSPSA` optimizer.

        Args:
            circuit: The circuit preparing the parameterized ansatz.
            backend: A backend of quantum instance to evaluate the circuits. If None, plain
                matrix multiplication will be used.
            expectation: An expectation converter to specify how the expected value is computed.
                If a shot-based readout is used this should be set to ``PauliExpectation``.

        Returns:
            A handle to the function :math:`F`.

        """
        params_x = ParameterVector("x", circuit.num_parameters)
        params_y = ParameterVector("y", circuit.num_parameters)

        expression = ~StateFn(circuit.assign_parameters(params_x)) @ StateFn(
            circuit.assign_parameters(params_y)
        )

        if expectation is not None:
            expression = expectation.convert(expression)

        if backend is None:

            def fidelity(values_x, values_y):
                value_dict = dict(
                    zip(params_x[:] + params_y[:], values_x.tolist() + values_y.tolist())
                )
                return np.abs(expression.assign_parameters(value_dict).eval()) ** 2

        else:
            sampler = CircuitSampler(backend)

            def fidelity(values_x, values_y=None):
                # no batches
                if isinstance(values_x, np.ndarray) and isinstance(values_y, np.ndarray):
                    value_dict = dict(
                        zip(params_x[:] + params_y[:], values_x.tolist() + values_y.tolist())
                    )
                # legacy batching -- remove once QNSPSA.get_fidelity is only supported with sampler
                elif values_y is None:
                    value_dict = {p: [] for p in params_x[:] + params_y[:]}
                    for values_xy in values_x:
                        for value_x, param_x in zip(values_xy[0, :], params_x):
                            value_dict[param_x].append(value_x)

                        for value_y, param_y in zip(values_xy[1, :], params_y):
                            value_dict[param_y].append(value_y)
                else:
                    value_dict = {p: [] for p in params_x[:] + params_y[:]}
                    for values_i_x, values_i_y in zip(values_x, values_y):
                        for value_x, param_x in zip(values_i_x, params_x):
                            value_dict[param_x].append(value_x)

                        for value_y, param_y in zip(values_i_y, params_y):
                            value_dict[param_y].append(value_y)

                return np.abs(sampler.convert(expression, params=value_dict).eval()) ** 2

        return fidelity

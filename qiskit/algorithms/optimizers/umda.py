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

"""Univariate Marginal Distribution Algorithm (Estimation-of-Distribution-Algorithm)."""

from __future__ import annotations

import warnings

from collections.abc import Callable
from typing import Any
import numpy as np
from scipy.stats import norm
from qiskit.utils import algorithm_globals

from .optimizer import OptimizerResult, POINT
from .scipy_optimizer import Optimizer, OptimizerSupportLevel


class UMDA(Optimizer):
    """Continuous Univariate Marginal Distribution Algorithm (UMDA).

    UMDA [1] is a specific type of Estimation of Distribution Algorithm (EDA) where new individuals
    are sampled from univariate normal distributions and are updated in each iteration of the
    algorithm by the best individuals found in the previous iteration.

    .. seealso::

        This original implementation of the UDMA optimizer for Qiskit was inspired by my
        (Vicente P. Soloviev) work on the EDAspy Python package [2].

    EDAs are stochastic search algorithms and belong to the family of the evolutionary algorithms.
    The main difference is that EDAs have a probabilistic model which is updated in each iteration
    from the best individuals of previous generations (elite selection). Depending on the complexity
    of the probabilistic model, EDAs can be classified in different ways. In this case, UMDA is a
    univariate EDA as the embedded probabilistic model is univariate.

    UMDA has been compared to some of the already implemented algorithms in Qiskit library to
    optimize the parameters of variational algorithms such as QAOA or VQE and competitive results
    have been obtained [1]. UMDA seems to provide very good solutions for those circuits in which
    the number of layers is not big.

    The optimization process can be personalized depending on the paremeters chosen in the
    initialization. The main parameter is the population size. The bigger it is, the final result
    will be better. However, this increases the complexity of the algorithm and the runtime will
    be much heavier. In the work [1] different experiments have been performed where population
    size has been set to 20 - 30.

    .. note::

        The UMDA implementation has more parameters but these have default values for the
        initialization for better understanding of the user. For example, ``\alpha`` parameter has
        been set to 0.5 and is the percentage of the population which is selected in each iteration
        to update the probabilistic model.


    Example:

        This short example runs UMDA to optimize the parameters of a variational algorithm. Here we
        will use the same operator as used in the algorithms introduction, which was originally
        computed by Qiskit Nature for an H2 molecule. The minimum energy of the H2 Hamiltonian can
        be found quite easily so we are able to set maxiters to a small value.

        .. code-block:: python

            from qiskit.opflow import X, Z, I
            from qiskit import Aer
            from qiskit.algorithms.optimizers import UMDA
            from qiskit.algorithms import QAOA
            from qiskit.utils import QuantumInstance


            H2_op = (-1.052373245772859 * I ^ I) + \
            (0.39793742484318045 * I ^ Z) + \
            (-0.39793742484318045 * Z ^ I) + \
            (-0.01128010425623538 * Z ^ Z) + \
            (0.18093119978423156 * X ^ X)

            p = 2  # Toy example: 2 layers with 2 parameters in each layer: 4 variables

            opt = UMDA(maxiter=100, size_gen=20)

            backend = Aer.get_backend('statevector_simulator')
            vqe = QAOA(opt,
                       quantum_instance=QuantumInstance(backend=backend),
                       reps=p)

            result = vqe.compute_minimum_eigenvalue(operator=H2_op)

        If it is desired to modify the percentage of individuals considered to update the
        probabilistic model, then this code can be used. Here for example we set the 60% instead
        of the 50% predefined.

        .. code-block:: python

            opt = UMDA(maxiter=100, size_gen=20, alpha = 0.6)

            backend = Aer.get_backend('statevector_simulator')
            vqe = QAOA(opt,
                       quantum_instance=QuantumInstance(backend=backend),
                       reps=p)

            result = vqe.compute_minimum_eigenvalue(operator=qubit_op)


    References:

        [1]: Vicente P. Soloviev, Pedro Larrañaga and Concha Bielza (2022, July). Quantum Parametric
        Circuit Optimization with Estimation of Distribution Algorithms. In 2022 The Genetic and
        Evolutionary Computation Conference (GECCO). DOI: https://doi.org/10.1145/3520304.3533963

        [2]: Vicente P. Soloviev. Python package EDAspy.
        https://github.com/VicentePerezSoloviev/EDAspy.
    """

    ELITE_FACTOR = 0.4
    STD_BOUND = 0.3

    def __init__(
        self,
        maxiter: int = 100,
        size_gen: int = 20,
        alpha: float = 0.5,
        callback: Callable[[int, np.array, float], None] | None = None,
    ) -> None:
        """
        Initialize the class with the specified parameters.

        Args:
            maxiter (int): Maximum number of iterations.
            size_gen (int): Population size of each generation.
            alpha (float): Percentage (0, 1] of the population to be selected as elite selection.
            callback (Callable[[int, np.array, float], None] | None): A callback
                function passed information in each iteration step. The information
            is, in this order: the number of function evaluations, the parameters,
                the best function value in this iteration.
        """

        self.size_gen = size_gen
        self.maxiter = maxiter
        self.alpha = alpha
        self._vector: np.ndarray | None = None
        # initialization of generation
        self._generation: np.ndarray | None = None
        self._dead_iter = int(self._maxiter / 5)

        self._truncation_length = int(size_gen * alpha)

        super().__init__()

        self._best_cost_global: float | None = None
        self._best_ind_global: int | None = None
        self._evaluations: np.ndarray | None = None

        self._n_variables: int | None = None

        self.callback = callback

    def _initialization(self) -> np.ndarray:
        """Initialize the vector for a new generation."""
        vector = np.zeros((4, self._n_variables))

        vector[0, :] = np.pi  # mu
        vector[1, :] = 0.5  # std

        return vector

    # build a generation of size SIZE_GEN from prob vector
    def _new_generation(self):
        """Build a new generation sampled from the vector of probabilities.
        Updates the generation pandas dataframe
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            gen = algorithm_globals.random.normal(
                self._vector[0, :], self._vector[1, :], [self._size_gen, self._n_variables]
            )

        self._generation = self._generation[: int(self.ELITE_FACTOR * len(self._generation))]
        self._generation = np.vstack((self._generation, gen))

    # truncate the generation at alpha percent
    def _truncation(self):
        """Selection of the best individuals of the actual generation.
        Updates the generation by selecting the best individuals.
        """
        best_indices = self._evaluations.argsort()[: self._truncation_length]
        self._generation = self._generation[best_indices, :]
        self._evaluations = np.take(self._evaluations, best_indices)

    # check each individual of the generation
    def _check_generation(self, objective_function):
        """Check the cost of each individual in the cost function implemented by the user."""
        self._evaluations = np.apply_along_axis(objective_function, 1, self._generation)

    # update the probability vector
    def _update_vector(self):
        """From the best individuals update the vector of normal distributions in order to the next
        generation can sample from it. Update the vector of normal distributions
        """

        for i in range(self._n_variables):
            self._vector[0, i], self._vector[1, i] = norm.fit(self._generation[:, i])
            if self._vector[1, i] < self.STD_BOUND:
                self._vector[1, i] = self.STD_BOUND

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """
        Perform optimization using the Truncated Random Search algorithm.

        This function takes a fitness function, an initial point, an optional Jacobian
        function, and optional bounds for the variables. It returns an
        OptimizerResult object containing the optimized values.

        Args:
            fun (Callable[[POINT], float]): The fitness function to be minimized.
            x0 (POINT): The initial point for the optimization.
            jac (Callable[[POINT], POINT] | None, optional): The Jacobian function
                of the fitness function. Defaults to None.
            bounds (list[tuple[float, float]] | None, optional): The bounds for
                the variables. Defaults to None.

        Returns:
            OptimizerResult: An object containing the optimized values.
        """

        not_better_count = 0
        result = OptimizerResult()

        if isinstance(x0, float):
            x0 = [x0]
        self._n_variables = len(x0)

        self._best_cost_global = 999999999999
        self._best_ind_global = 9999999
        history = []
        self._evaluations = np.array(0)

        self._vector = self._initialization()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            # initialization of generation
            self._generation = algorithm_globals.random.normal(
                self._vector[0, :], self._vector[1, :], [self._size_gen, self._n_variables]
            )

        for _ in range(self._maxiter):
            self._check_generation(fun)
            self._truncation()
            self._update_vector()

            best_mae_local: float = min(self._evaluations)

            history.append(best_mae_local)
            best_ind_local = np.where(self._evaluations == best_mae_local)[0][0]
            best_ind_local = self._generation[best_ind_local]

            # update the best values ever
            if best_mae_local < self._best_cost_global:
                self._best_cost_global = best_mae_local
                self._best_ind_global = best_ind_local
                not_better_count = 0

            else:
                not_better_count += 1
                if not_better_count >= self._dead_iter:
                    break

            if self.callback is not None:
                self.callback(
                    len(history) * self._size_gen, self._best_ind_global, self._best_cost_global
                )

            self._new_generation()

        result.x = self._best_ind_global
        result.fun = self._best_cost_global
        result.nfev = len(history) * self._size_gen

        return result

    @property
    def size_gen(self) -> int:
        """
        Sets the size of the generations of the algorithm.

        Args:
            value: Size of the generations (number of individuals per generation).

        Raises:
            ValueError: If `value` is lower than 1.
        """
        return self._size_gen

    @size_gen.setter
    def size_gen(self, value: int):
        """
        Sets the size of the generations of the algorithm.

        Args:
            value: Size of the generations (number of individuals per generation).

        Raises:
            ValueError: If `value` is lower than 1.
        """
        if value <= 0:
            raise ValueError("The size of the generation should be greater than 0.")
        self._size_gen = value

    @property
    def maxiter(self) -> int:
        """
        Sets the maximum number of iterations of the algorithm.

        Args:
            value: Maximum number of iterations of the algorithm.

        Raises:
            ValueError: If `value` is lower than 1.
        """
        return self._maxiter

    @maxiter.setter
    def maxiter(self, value: int):
        """
        Sets the maximum number of iterations of the algorithm.

        Args:
            value: Maximum number of iterations of the algorithm.

        Raises:
            ValueError: If `value` is lower than 1.
        """
        if value <= 0:
            raise ValueError("The maximum number of iterations should be greater than 0.")

        self._maxiter = value

    @property
    def alpha(self) -> float:
        """Returns the alpha parameter value (percentage of population selected to update
        probabilistic model)"""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        """
        Sets the alpha parameter (percentage of individuals selected to update the probabilistic
        model)

        Args:
            value: Percentage (0,1] of generation selected to update the probabilistic model.

        Raises:
            ValueError: If `value` is lower than 0 or greater than 1.
        """
        if (value <= 0) or (value > 1):
            raise ValueError(f"alpha must be in the range (0, 1], value given was {value}")

        self._alpha = value

    @property
    def settings(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the values of the 'maxiter', 'alpha',
        'size_gen', and 'callback' properties.
        """
        return {
            "maxiter": self.maxiter,
            "alpha": self.alpha,
            "size_gen": self.size_gen,
            "callback": self.callback,
        }

    def get_support_level(self):
        """
        Get the support level dictionary.
        """
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

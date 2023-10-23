# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""SteppableOptimizer interface"""
from __future__ import annotations

from abc import abstractmethod, ABC
from collections.abc import Callable
from dataclasses import dataclass
from .optimizer import Optimizer, POINT, OptimizerResult


@dataclass
class AskData(ABC):
    """Base class for return type of :meth:`~.SteppableOptimizer.ask`.

    Args:
        x_fun: Point or list of points where the function needs to be evaluated to compute the next
        state of the optimizer.
        x_jac: Point or list of points where the gradient/jacobian needs to be evaluated to compute
        the next state of the optimizer.

    """

    x_fun: POINT | list[POINT] | None = None
    x_jac: POINT | list[POINT] | None = None


@dataclass
class TellData(ABC):
    """Base class for argument type of :meth:`~.SteppableOptimizer.tell`.

    Args:
        eval_fun: Image of the function at :attr:`~.ask_data.x_fun`.
        eval_jac: Image of the gradient-jacobian at :attr:`~.ask_data.x_jac`.

    """

    eval_fun: float | list[float] | None = None
    eval_jac: POINT | list[POINT] | None = None


@dataclass
class OptimizerState:
    """Base class representing the state of the optimizer.

    This class stores the current state of the optimizer, given by the current point and
    (optionally) information like the function value, the gradient or the number of
    function evaluations. This dataclass can also store any other individual variables that
    change during the optimization.

    """

    x: POINT
    """Current optimization parameters."""
    fun: Callable[[POINT], float] | None
    """Function being  optimized."""
    jac: Callable[[POINT], POINT] | None
    """Jacobian of the function being optimized."""
    nfev: int | None
    """Number of function evaluations so far in the optimization."""
    njev: int | None
    """Number of jacobian evaluations so far in the opimization."""
    nit: int | None
    """Number of optmization steps performed so far in the optimization."""


class SteppableOptimizer(Optimizer):
    """
    Base class for a steppable optimizer.

    This family of optimizers uses the `ask and tell interface
    <https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html>`_.
    When using this interface the user has to call :meth:`~.ask` to get information about
    how to evaluate the fucntion (we are asking the optimizer about how to do the evaluation).
    This information is typically the next points at which the function is evaluated, but depending
    on the optimizer it can also determine whether to evaluate the function or its gradient.
    Once the function has been evaluated, the user calls the method :meth:`~..tell`
    to tell the optimizer what the result of the function evaluation(s) is. The optimizer then
    updates its state accordingly and the user can decide whether to stop the optimization process
    or to repeat a step.

    This interface is more customizable, and allows the user to have full control over the evaluation
    of the function.

    Examples:

        An example where the evaluation of the function has a chance of failing. The user, with
        specific knowledge about his function can catch this errors and handle them before passing
        the result to the optimizer.

        .. code-block:: python

            import random
            import numpy as np
            from qiskit.algorithms.optimizers import GradientDescent

            def objective(x):
                if random.choice([True, False]):
                    return None
                else:
                    return (np.linalg.norm(x) - 1) ** 2

            def grad(x):
                if random.choice([True, False]):
                    return None
                else:
                    return 2 * (np.linalg.norm(x) - 1) * x / np.linalg.norm(x)


            initial_point = np.random.normal(0, 1, size=(100,))

            optimizer = GradientDescent(maxiter=20)
            optimizer.start(x0=initial_point, fun=objective, jac=grad)

            while optimizer.continue_condition():
                ask_data = optimizer.ask()
                evaluated_gradient = None

                while evaluated_gradient is None:
                    evaluated_gradient = grad(ask_data.x_center)
                    optimizer.state.njev += 1

                optmizer.state.nit += 1

                 cf  = TellData(eval_jac=evaluated_gradient)
                optimizer.tell(ask_data=ask_data, tell_data=tell_data)

            result = optimizer.create_result()


    Users that aren't dealing with complicated functions and who are more familiar with step by step
    optimization algorithms can use the :meth:`~.step` method which wraps the :meth:`~.ask`
    and :meth:`~.tell` methods. In the same spirit the method :meth:`~.minimize` will optimize the
    function and return the result.

    To see other libraries that use this interface one can visit:
    https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html


    """

    def __init__(
        self,
        maxiter: int = 100,
    ):
        """
        Initialize the optimizer.

        Args:
            maxiter: Number of steps in the optimization process before ending the loop.
        """
        super().__init__()
        self._state: OptimizerState | None = None
        self.maxiter = maxiter

    @property
    def state(self) -> OptimizerState:
        """Return the current state of the optimizer."""
        return self._state

    @state.setter
    def state(self, state: OptimizerState) -> None:
        """Set the current state of the optimizer."""
        self._state = state

    def ask(self) -> AskData:
        """Ask the optimizer for a set of points to evaluate.

        This method asks the optimizer which are the next points to evaluate.
        These points can, e.g., correspond to function values and/or its derivative.
        It may also correspond to variables that let the user infer which points to evaluate.
        It is the first method inside of a :meth:`~.step` in the optimization process.

        Returns:
            An object containing the data needed to make the funciton evaluation to advance the
            optimization process.

        """
        raise NotImplementedError

    def tell(self, ask_data: AskData, tell_data: TellData) -> None:
        """Updates the optimization state using the results of the function evaluation.

        A canonical optimization example using :meth:`~.ask` and :meth:`~.tell` can be seen
        in :meth:`~.step`.

        Args:
            ask_data: Contains the information on how the evaluation was done.
            tell_data: Contains all relevant information about the evaluation of the objective
                function.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, ask_data: AskData) -> TellData:
        """Evaluates the function according to the instructions contained in :attr:`~.ask_data`.

        If the user decides to use :meth:`~.step` instead of :meth:`~.ask` and :meth:`~.tell`
        this function will contain the logic on how to evaluate the function.

        Args:
            ask_data: Contains the information on how to do the evaluation.

        Returns:
            Data of all relevant information about the function evaluation.

        """
        raise NotImplementedError

    def _callback_wrapper(self) -> None:
        """
        Wraps the callback function to accommodate each optimizer.
        """
        pass

    def step(self) -> None:
        """Performs one step in the optimization process.

        This method composes :meth:`~.ask`, :meth:`~.evaluate`, and :meth:`~.tell` to make a "step"
        in the optimization process.
        """
        ask_data = self.ask()
        tell_data = self.evaluate(ask_data=ask_data)
        self.tell(ask_data=ask_data, tell_data=tell_data)

    # pylint: disable=invalid-name
    @abstractmethod
    def start(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        """Populates the state of the optimizer with the data provided and sets all the counters to 0.

        Args:
            fun: Function to minimize.
            x0: Initial point.
            jac: Function to compute the gradient.
            bounds: Bounds of the search space.

        """
        raise NotImplementedError

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """Minimizes the function.

        For well behaved functions the user can call this method to minimize a function.
        If the user wants more control on how to evaluate the function a custom loop can be
        created using :meth:`~.ask` and :meth:`~.tell` and evaluating the function manually.

        Args:
            fun: Function to minimize.
            x0: Initial point.
            jac: Function to compute the gradient.
            bounds: Bounds of the search space.

        Returns:
            Object containing the result of the optimization.

        """
        self.start(x0=x0, fun=fun, jac=jac, bounds=bounds)
        while self.continue_condition():
            self.step()
            self._callback_wrapper()
        return self.create_result()

    @abstractmethod
    def create_result(self) -> OptimizerResult:
        """Returns the result of the optimization.

        All the information needed to create such a result should be stored in the optimizer state
        and will typically contain the best point found, the function value and gradient at that point,
        the number of function and gradient evaluation and the number of iterations in the optimization.

        Returns:
            The result of the optimization process.

        """
        raise NotImplementedError

    def continue_condition(self) -> bool:
        """Condition that indicates the optimization process should continue.

        Returns:
            ``True`` if the optimization process should continue, ``False`` otherwise.
        """
        return self.state.nit < self.maxiter

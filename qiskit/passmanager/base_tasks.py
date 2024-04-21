# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Baseclasses for the Qiskit passmanager optimization tasks."""
from __future__ import annotations

import logging
import time
import warnings
from abc import abstractmethod, ABC
from collections.abc import Iterable, Callable, Generator
from typing import Any

from .compilation_status import RunState, PassManagerState, PropertySet, AnyTarget

logger = logging.getLogger(__name__)

# Type alias
PassManagerIR = Any


class Task(ABC):
    """An interface of the pass manager task.

    The task takes a Qiskit IR, and outputs new Qiskit IR after some operation on it.
    A task can rely on the :class:`.PropertySet` to communicate intermediate data among tasks.
    """

    @abstractmethod
    def execute(
        self,
        passmanager_ir: PassManagerIR,
        state: PassManagerState,
        callback: Callable = None,
    ) -> tuple[PassManagerIR, PassManagerState]:
        """Execute optimization task for input Qiskit IR.

        Args:
            passmanager_ir: Qiskit IR to optimize.
            state: State associated with workflow execution by the pass manager itself.
            callback: A callback function which is caller per execution of optimization task.

        Returns:
            Optimized Qiskit IR and state of the workflow.
        """
        pass


class GenericPass(Task, ABC):
    """Base class of a single pass manager task.

    A pass instance can read and write to the provided :class:`.PropertySet`,
    and may modify the input pass manager IR.
    """

    def __init__(self):
        self.property_set: PropertySet = PropertySet()
        """Information about IR being optimized."""

        self.requires: Iterable[Task] = []
        """Dependency of this pass."""

        self.__usr_target: AnyTarget | None = None
        self.__target: AnyTarget | None = None

    @property
    def target(self) -> AnyTarget:
        """Constraints of target computing system."""
        # TODO remove after self.target is dropped from builtin passes.
        return self.__usr_target or self.__target

    @target.setter
    def target(self, usr_target):
        # TODO remove after self.target is dropped from builtin passes.
        warnings.warn(
            "Manually providing target is no longer necessary. "
            "Target is automatically set to the instance at run time. "
            "The target attribute is now reserved by the base GenericPass class.",
            PendingDeprecationWarning,
        )
        self.__usr_target = usr_target

    @property
    def _target(self) -> AnyTarget:
        # TODO remove after self.target is dropped from builtin passes.
        return self.target

    @target.setter
    def _target(self, usr_target):
        # TODO remove after self.target is dropped from builtin passes.
        self.target = usr_target

    def name(self) -> str:
        """Name of the pass."""
        return self.__class__.__name__

    def execute(
        self,
        passmanager_ir: PassManagerIR,
        state: PassManagerState,
        callback: Callable = None,
    ) -> tuple[PassManagerIR, PassManagerState]:
        # Overriding this method is not safe.
        # Pass subclass must keep current implementation.
        # Especially, task execution may break when method signature is modified.

        self.property_set = state.property_set
        self.__target = state.target

        if self.requires:
            # pylint: disable=cyclic-import
            from .flow_controllers import FlowControllerLinear

            passmanager_ir, state = FlowControllerLinear(self.requires).execute(
                passmanager_ir=passmanager_ir,
                state=state,
                callback=callback,
            )

        run_state = None
        ret = None
        start_time = time.time()
        try:
            if self not in state.workflow_status.completed_passes:
                ret = self.run(passmanager_ir)
                run_state = RunState.SUCCESS
            else:
                run_state = RunState.SKIP
        except Exception:
            run_state = RunState.FAIL
            raise
        finally:
            ret = ret or passmanager_ir
            if run_state != RunState.SKIP:
                running_time = time.time() - start_time
                logger.info("Pass: %s - %.5f (ms)", self.name(), running_time * 1000)
                if callback is not None:
                    callback(
                        task=self,
                        passmanager_ir=ret,
                        property_set=state.property_set,
                        running_time=running_time,
                        count=state.workflow_status.count,
                    )
        return ret, self.update_status(state, run_state)

    def update_status(
        self,
        state: PassManagerState,
        run_state: RunState,
    ) -> PassManagerState:
        """Update workflow status.

        Args:
            state: Pass manager state to update.
            run_state: Completion status of current task.

        Returns:
            Updated pass manager state.
        """
        state.workflow_status.previous_run = run_state
        if run_state == RunState.SUCCESS:
            state.workflow_status.count += 1
            state.workflow_status.completed_passes.add(self)
        return state

    @abstractmethod
    def run(
        self,
        passmanager_ir: PassManagerIR,
    ) -> PassManagerIR:
        """Run optimization task.

        Args:
            passmanager_ir: Qiskit IR to optimize.

        Returns:
            Optimized Qiskit IR.
        """
        pass


class BaseController(Task, ABC):
    """Base class of controller.

    A controller is built with a collection of pass manager tasks,
    and a subclass provides a custom logic to choose next task to run.
    Note a controller can be nested into another controller,
    and a controller itself doesn't provide any subroutine to modify the input IR.
    """

    def __init__(
        self,
        options: dict[str, Any] | None = None,
    ):
        """Create new flow controller.

        Args:
            options: Option for this flow controller.
        """
        self._options = options or {}

    @abstractmethod
    def iter_tasks(
        self,
        state: PassManagerState,
    ) -> Generator[Task, PassManagerState, None]:
        """A custom logic to choose a next task to run.

        Controller subclass can consume the state to build a proper task pipeline.  The updated
        state after a task execution will be fed back in as the "return" value of any ``yield``
        statements.  This indicates the order of task execution is only determined at running time.
        This method is not allowed to mutate the given state object.

        Args:
            state: The state of the passmanager workflow at the beginning of this flow controller's
                execution.

        Receives:
            state: the state of pass manager after the execution of the last task that was yielded.
                The generator does not need to inspect this if it is irrelevant to its logic, nor
                update it.

        Yields:
            Task: Next task to run.
        """
        pass

    def execute(
        self,
        passmanager_ir: PassManagerIR,
        state: PassManagerState,
        callback: Callable = None,
    ) -> tuple[PassManagerIR, PassManagerState]:
        # Overriding this method is not safe.
        # Pass subclass must keep current implementation.
        # Especially, task execution may break when method signature is modified.

        task_generator = self.iter_tasks(state)
        try:
            next_task = task_generator.send(None)
        except StopIteration:
            return passmanager_ir, state
        while True:
            passmanager_ir, state = next_task.execute(
                passmanager_ir=passmanager_ir,
                state=state,
                callback=callback,
            )
            try:
                # Sending the object through the generator implies the custom controllers
                # can always rely on the latest data to choose the next task to run.
                next_task = task_generator.send(state)
            except StopIteration:
                break

        return passmanager_ir, state

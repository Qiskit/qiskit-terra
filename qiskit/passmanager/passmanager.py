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

"""Manager for a set of Passes and their scheduling during transpilation."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from itertools import chain
from typing import Any


from qiskit.utils.parallel import parallel_map, should_run_in_parallel
from .base_tasks import Task, PassManagerIR
from .exceptions import PassManagerError
from .flow_controllers import FlowControllerLinear
from .compilation_status import PropertySet, WorkflowStatus, PassManagerState
from .qiskit_dill import loads, dumps


logger = logging.getLogger(__name__)


class BasePassManager(ABC):
    """Pass manager base class."""

    def __init__(
        self,
        tasks: Task | list[Task] = (),
        max_iteration: int = 1000,
    ):
        """Initialize an empty pass manager object.

        Args:
            tasks: A pass set to be added to the pass manager schedule.
            max_iteration: The maximum number of iterations the schedule will be looped if the
                condition is not met.
        """
        self._tasks = []
        self.max_iteration = max_iteration
        # This empty property set never gets used; it gets overridden at the completion of a
        # workflow run.
        self.property_set = PropertySet()

        if tasks:
            self.append(tasks)

    def append(
        self,
        tasks: Task | list[Task],
    ) -> None:
        """Append tasks to the schedule of passes.

        Args:
            tasks: A set of pass manager tasks to be added to schedule.

        Raises:
            TypeError: When any element of tasks is not a subclass of passmanager Task.
        """
        if isinstance(tasks, Task):
            tasks = [tasks]
        if any(not isinstance(t, Task) for t in tasks):
            raise TypeError("Added tasks are not all valid pass manager task types.")

        self._tasks.append(tasks)

    def replace(
        self,
        index: int,
        tasks: Task | list[Task],
    ) -> None:
        """Replace a particular pass in the scheduler.

        Args:
            index: Task index to replace, based on the position in :meth:`tasks`
            tasks: A set of pass manager tasks to be added to schedule.

        Raises:
            TypeError: When any element of tasks is not a subclass of passmanager Task.
            PassManagerError: If the index is not found.
        """
        try:
            self._tasks[index] = tasks
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exists") from ex

    def remove(self, index: int) -> None:
        """Removes a particular pass in the scheduler.

        Args:
            index: Pass index to remove, based on the position in :meth:`passes`.

        Raises:
            PassManagerError: If the index is not found.
        """
        try:
            del self._tasks[index]
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exists") from ex

    def __setitem__(self, index, item):
        self.replace(index, item)

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, index):
        new_passmanager = self.__class__(max_iteration=self.max_iteration)
        new_passmanager._tasks = self._tasks[index]
        return new_passmanager

    def __add__(self, other):
        new_passmanager = self.__class__(max_iteration=self.max_iteration)
        new_passmanager._tasks = self._tasks
        if isinstance(other, self.__class__):
            new_passmanager._tasks += other._tasks
            return new_passmanager
        else:
            try:
                new_passmanager.append(other)
                return new_passmanager
            except PassManagerError as ex:
                raise TypeError(
                    f"unsupported operand type + for {self.__class__} and {other.__class__}"
                ) from ex

    @abstractmethod
    def _passmanager_frontend(
        self,
        input_program: Any,
        **kwargs,
    ) -> PassManagerIR:
        """Convert input program into pass manager IR.

        Args:
            in_program: Input program.

        Returns:
            Pass manager IR.
        """
        pass

    @abstractmethod
    def _passmanager_backend(
        self,
        passmanager_ir: PassManagerIR,
        in_program: Any,
        **kwargs,
    ) -> Any:
        """Convert pass manager IR into output program.

        Args:
            passmanager_ir: Pass manager IR after optimization.
            in_program: The input program, this can be used if you need
                any metadata about the original input for the output.
                It should not be mutated.

        Returns:
            Output program.
        """
        pass

    def run(
        self,
        in_programs: Any | list[Any],
        callback: Callable = None,
        num_processes: int = None,
        **kwargs,
    ) -> Any:
        """Run all the passes on the specified ``in_programs``.

        Args:
            in_programs: Input programs to transform via all the registered passes.
                A single input object cannot be a Python builtin list object.
                A list object is considered as multiple input objects to optimize.
            callback: A callback function that will be called after each pass execution. The
                function will be called with 4 keyword arguments::

                    task (GenericPass): the pass being run
                    passmanager_ir (Any): depending on pass manager subclass
                    property_set (PropertySet): the property set
                    running_time (float): the time to execute the pass
                    count (int): the index for the pass execution

                The exact arguments pass expose the internals of the pass
                manager and are subject to change as the pass manager internals
                change. If you intend to reuse a callback function over
                multiple releases be sure to check that the arguments being
                passed are the same.

                To use the callback feature you define a function that will
                take in kwargs dict and access the variables. For example::

                    def callback_func(**kwargs):
                        task = kwargs['task']
                        passmanager_ir = kwargs['passmanager_ir']
                        property_set = kwargs['property_set']
                        running_time = kwargs['running_time']
                        count = kwargs['count']
                        ...
            num_processes: The maximum number of parallel processes to launch if parallel
                execution is enabled. This argument overrides ``num_processes`` in the user
                configuration file, and the ``QISKIT_NUM_PROCS`` environment variable. If set
                to ``None`` the system default or local user configuration will be used.

            kwargs: Arbitrary arguments passed to the compiler frontend and backend.

        Returns:
            The transformed program(s).
        """
        if not self._tasks and not kwargs and callback is None:
            return in_programs

        is_list = True
        if not isinstance(in_programs, list):
            in_programs = [in_programs]
            is_list = False

        # If we're not going to run in parallel, we want to avoid spending time `dill` serializing
        # ourselves, since that can be quite expensive.
        if len(in_programs) == 1 or not should_run_in_parallel(num_processes):
            out = [
                _run_workflow(program=program, pass_manager=self, callback=callback, **kwargs)
                for program in in_programs
            ]
            if len(in_programs) == 1 and not is_list:
                return out[0]
            return out

        del callback
        del kwargs

        # Pass manager may contain callable and we need to serialize through dill rather than pickle.
        # See https://github.com/Qiskit/qiskit-terra/pull/3290
        # Note that serialized object is deserialized as a different object.
        # Thus, we can resue the same manager without state collision, without building it per thread.
        qubit_cache = {}
        return parallel_map(
            _run_workflow_in_new_process,
            values=[dumps(circuit, qubit_cache) for circuit in in_programs],
            task_kwargs={"pass_manager_bin": dumps(self, qubit_cache), "qubit_cache": qubit_cache},
        )

    def to_flow_controller(self) -> FlowControllerLinear:
        """Linearize this manager into a single :class:`.FlowControllerLinear`,
        so that it can be nested inside another pass manager.

        Returns:
            A linearized pass manager.
        """
        flatten_tasks = list(self._flatten_tasks(self._tasks))
        return FlowControllerLinear(flatten_tasks)

    def _flatten_tasks(self, elements: Iterable | Task) -> Iterable:
        """A helper method to recursively flatten a nested task chain."""
        if not isinstance(elements, Iterable):
            return [elements]
        return chain(*map(self._flatten_tasks, elements))


def _run_workflow(
    program: Any,
    pass_manager: BasePassManager,
    **kwargs,
) -> Any:
    """Run single program optimization with a pass manager.

    Args:
        program: Arbitrary program to optimize.
        pass_manager: Pass manager with scheduled passes.
        **kwargs: Keyword arguments for IR conversion.

    Returns:
        Optimized program.
    """
    flow_controller = pass_manager.to_flow_controller()
    initial_status = WorkflowStatus()

    passmanager_ir = pass_manager._passmanager_frontend(
        input_program=program,
        **kwargs,
    )
    passmanager_ir, final_state = flow_controller.execute(
        passmanager_ir=passmanager_ir,
        state=PassManagerState(
            workflow_status=initial_status,
            property_set=PropertySet(),
        ),
        callback=kwargs.get("callback", None),
    )
    # The `property_set` has historically been returned as a mutable attribute on `PassManager`
    # This makes us non-reentrant (though `PassManager` would be dependent on its internal tasks to
    # be re-entrant if that was required), but is consistent with previous interfaces.  We're still
    # safe to be called in a serial loop, again assuming internal tasks are re-runnable.  The
    # conversion to the backend language is also allowed to use the property set, so it must be set
    # before calling it.
    pass_manager.property_set = final_state.property_set

    out_program = pass_manager._passmanager_backend(
        passmanager_ir=passmanager_ir,
        in_program=program,
        **kwargs,
    )

    return out_program


def _run_workflow_in_new_process(program: Any, pass_manager_bin: bytes, qubit_cache: bytes) -> Any:
    """Run single program optimization in new process.

    Args:
        program: Arbitrary program to optimize.
        pass_manager_bin: Binary of the pass manager with scheduled passes.

    Returns:
          Optimized program.
    """
    return _run_workflow(
        program=loads(program, qubit_cache),
        pass_manager=loads(pass_manager_bin, qubit_cache),
    )

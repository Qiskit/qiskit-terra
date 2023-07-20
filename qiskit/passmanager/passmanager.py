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
from abc import ABC
from collections.abc import Callable, Sequence
from typing import Any
import logging
import dill
from qiskit.tools.parallel import parallel_map

from .base_pass import GenericPass
from .passrunner import BasePassRunner
from .exceptions import PassManagerError
from .flow_controllers import FlowController, PassSequence

logger = logging.getLogger(__name__)


class BasePassManager(ABC):
    """Pass manager base class."""

    PASS_RUNNER = BasePassRunner

    def __init__(
        self,
        passes: PassSequence | None = None,
        max_iteration: int = 1000,
    ):
        """Initialize an empty pass manager object.

        Args:
            passes: A pass set to be added to the pass manager schedule.
            max_iteration: The maximum number of iterations the schedule will be looped if the
                condition is not met.
        """
        # the pass manager's schedule of passes, including any control-flow.
        # Populated via PassManager.append().
        self._pass_sets: list[dict[str, Any]] = []
        self.max_iteration = max_iteration

        if passes is not None:
            self.append(passes)

    def append(
        self,
        passes: PassSequence,
        **flow_controller_conditions: Callable,
    ) -> None:
        """Append a Pass Set to the schedule of passes.

        Args:
            passes: A set of passes (a pass set) to be added to schedule. A pass set is a list of
                    passes that are controlled by the same flow controller. If a single pass is
                    provided, the pass set will only have that pass a single element.
                    It is also possible to append a
                    :class:`~qiskit.transpiler.runningpassmanager.FlowController` instance and the
                    rest of the parameter will be ignored.
            flow_controller_conditions: Dictionary of control flow plugins. Default:

                * do_while (callable property_set -> boolean): The passes repeat until the
                  callable returns False.
                  Default: `lambda x: False # i.e. passes run once`

                * condition (callable property_set -> boolean): The passes run only if the
                  callable returns True.
                  Default: `lambda x: True # i.e. passes run`
        """
        passes = self._normalize_passes(passes)
        self._pass_sets.append({"passes": passes, "flow_controllers": flow_controller_conditions})

    def replace(
        self,
        index: int,
        passes: PassSequence,
        **flow_controller_conditions: Any,
    ) -> None:
        """Replace a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().
            passes: A pass set (as defined in :py:func:`qiskit.transpiler.PassManager.append`)
                to be added to the pass manager schedule.
            flow_controller_conditions: control flow plugins.

        Raises:
            PassManagerError: if a pass in passes is not a proper pass or index not found.
        """
        passes = self._normalize_passes(passes)

        try:
            self._pass_sets[index] = {
                "passes": passes,
                "flow_controllers": flow_controller_conditions,
            }
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exists") from ex

    def remove(self, index: int) -> None:
        """Removes a particular pass in the scheduler.

        Args:
            index: Pass index to replace, based on the position in passes().

        Raises:
            PassManagerError: if the index is not found.
        """
        try:
            del self._pass_sets[index]
        except IndexError as ex:
            raise PassManagerError(f"Index to replace {index} does not exists") from ex

    def __setitem__(self, index, item):
        self.replace(index, item)

    def __len__(self):
        return len(self._pass_sets)

    def __getitem__(self, index):
        new_passmanager = self.__class__(max_iteration=self.max_iteration)
        _pass_sets = self._pass_sets[index]
        if isinstance(_pass_sets, dict):
            _pass_sets = [_pass_sets]
        new_passmanager._pass_sets = _pass_sets
        return new_passmanager

    def __add__(self, other):
        if isinstance(other, self.__class__):
            new_passmanager = self.__class__(max_iteration=self.max_iteration)
            new_passmanager._pass_sets = self._pass_sets + other._pass_sets
            return new_passmanager
        else:
            try:
                new_passmanager = self.__class__(max_iteration=self.max_iteration)
                new_passmanager._pass_sets += self._pass_sets
                new_passmanager.append(other)
                return new_passmanager
            except PassManagerError as ex:
                raise TypeError(
                    f"unsupported operand type + for {self.__class__} and {other.__class__}"
                ) from ex

    def _normalize_passes(
        self,
        passes: PassSequence,
    ) -> Sequence[GenericPass | FlowController] | FlowController:
        if isinstance(passes, FlowController):
            return passes
        if isinstance(passes, GenericPass):
            passes = [passes]
        for pass_ in passes:
            if isinstance(pass_, FlowController):
                # Normalize passes in nested FlowController.
                # TODO: Internal renormalisation should be the responsibility of the
                # `FlowController`, but the separation between `FlowController`,
                # `RunningPassManager` and `PassManager` is so muddled right now, it would be better
                # to do this as part of more top-down refactoring.  ---Jake, 2022-10-03.
                pass_.passes = self._normalize_passes(pass_.passes)
            elif not isinstance(pass_, GenericPass):
                raise PassManagerError(
                    "%s is not a pass or FlowController instance " % pass_.__class__
                )
        return passes

    def run(
        self,
        in_programs: Any,
        callback: Callable | None = None,
        **metadata,
    ) -> Any:
        """Run all the passes on the specified ``circuits``.

        Args:
            in_programs: Input programs to transform via all the registered passes.
            callback: A callback function that will be called after each pass execution. The
                function will be called with 5 keyword arguments::

                    pass_ (Pass): the pass being run
                    passmanager_ir (Any): depending on pass manager subclass
                    time (float): the time to execute the pass
                    property_set (PropertySet): the property set
                    count (int): the index for the pass execution

                The exact arguments pass expose the internals of the pass
                manager and are subject to change as the pass manager internals
                change. If you intend to reuse a callback function over
                multiple releases be sure to check that the arguments being
                passed are the same.

                To use the callback feature you define a function that will
                take in kwargs dict and access the variables. For example::

                    def callback_func(**kwargs):
                        pass_ = kwargs['pass_']
                        dag = kwargs['dag']
                        time = kwargs['time']
                        property_set = kwargs['property_set']
                        count = kwargs['count']
                        ...

            metadata: Metadata which might be attached to output program.

        Returns:
            The transformed program(s).
        """
        if not self._pass_sets and not metadata and callback is None:
            return in_programs

        is_list = True
        if isinstance(in_programs, self.PASS_RUNNER.IN_PROGRAM_TYPE):
            in_programs = [in_programs]
            is_list = False

        if len(in_programs) == 1:
            out_program = self._run_single_circuit(in_programs[0], callback, **metadata)
            if is_list:
                return [out_program]
            return out_program

        # TODO support for List(output_name) and List(callback)
        del metadata
        del callback

        return self._run_several_circuits(in_programs)

    def _create_running_passmanager(self) -> BasePassRunner:
        # Must be implemented by followup PR.
        # BasePassRunner.append assumes normalized pass input, which is not pass_sets.
        raise NotImplementedError

    def _run_single_circuit(
        self,
        input_program: Any,
        callback: Callable | None = None,
        **metadata,
    ) -> Any:
        pass_runner = self._create_running_passmanager()
        return pass_runner.run(input_program, callback=callback, **metadata)

    def _run_several_circuits(
        self,
        input_programs: Sequence[Any],
    ) -> Any:
        # Pass runner may contain callable and we need to serialize through dill rather than pickle.
        # See https://github.com/Qiskit/qiskit-terra/pull/3290
        # Note that serialized object is deserialized as a different object.
        # Thus, we can resue the same runner without state collision, without building it per thread.
        return parallel_map(
            self._in_parallel, input_programs, task_kwargs={"pm_dill": dill.dumps(self)}
        )

    @staticmethod
    def _in_parallel(
        in_program: Any,
        pm_dill: bytes = None,
    ) -> Any:
        pass_runner = dill.loads(pm_dill)._create_running_passmanager()
        return pass_runner.run(in_program)

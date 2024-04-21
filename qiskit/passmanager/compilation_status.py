# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A property set dictionary that shared among optimization passes."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Type alias
AnyTarget = Any


class PropertySet(dict):
    """A default dictionary-like object."""

    def __missing__(self, key):
        return None


class RunState(Enum):
    """Allowed values for the result of a pass execution."""

    SUCCESS = 0
    FAIL = 1
    SKIP = 2


@dataclass
class WorkflowStatus:
    """Collection of compilation status of workflow, i.e. pass manager run.

    This data structure is initialized when the pass manager is run,
    and recursively handed over to underlying tasks.
    Each pass will update this status once after being executed, and the lifetime of the
    workflow status object is the time during which the pass manager is running.
    """

    count: int = 0
    """Current number of pass execution."""

    completed_passes: set = field(default_factory=set)
    """Passes already run that have not been invalidated."""

    previous_run: RunState = RunState.FAIL
    """Status of the latest pass run."""


@dataclass
class PassManagerState:
    """A portable container object that pass manager tasks communicate through generator.

    This object can contain every information about the running pass manager workflow,
    except for the IR object being optimized.
    This container aims at providing a robust interface for the the :meth:`.Task.execute`
    method, and no logic that modifies the container elements must be implemented.
    The data structure consists of three elements.

    * :attr:`.workflow_status`: Status of current workflow, which is consumed by
      the given pass manager callback and system logger.
      Execution of passes may be condition on the status.
    * :attr:`.property_set`: Information of the IR gained through pass executions.
      Typically, analysis-type pass mutates and updates the property set, and
      transform-type pass consumes the data for optimization task.
    * :attr:`.target`: Read-only information about a computing system that
      IR is optimized for. Any type of pass can consume this information
      to perform hardware-aware task. Altough this object may not be protected from
      falsification, modified target infomation may result in the failure in execution.

    This object is mutable, and might be mutated by pass executions except for :attr:`.target`.
    """

    workflow_status: WorkflowStatus
    """Status of the current compilation workflow."""

    property_set: PropertySet
    """Information about IR being optimized."""

    target: AnyTarget | None = None
    """Information about target system that IR is optimized for."""

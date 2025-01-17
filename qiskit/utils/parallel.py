# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

"""
Routines for running Python functions in parallel using process pools
from the multiprocessing library.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
import sys

from qiskit.exceptions import QiskitError
from qiskit.utils.multiprocessing import local_hardware_info
from qiskit import user_config


def get_platform_parallel_default():
    """
    Returns the default parallelism flag value for the current platform.

    Returns:
        parallel_default: The default parallelism flag value for the
        current platform.

    """
    # Default False on Windows
    if sys.platform == "win32":
        parallel_default = False
    # On macOS default false on Python >=3.8
    elif sys.platform == "darwin":
        parallel_default = False
    # On linux (and other OSes) default to True
    else:
        parallel_default = True

    return parallel_default


CONFIG = user_config.get_config()

if os.getenv("QISKIT_PARALLEL", None) is not None:
    PARALLEL_DEFAULT = os.getenv("QISKIT_PARALLEL", None).lower() == "true"
else:
    PARALLEL_DEFAULT = get_platform_parallel_default()

# Set parallel flag
if os.getenv("QISKIT_IN_PARALLEL") is None:
    os.environ["QISKIT_IN_PARALLEL"] = "FALSE"

if os.getenv("QISKIT_NUM_PROCS") is not None:
    CPU_COUNT = int(os.getenv("QISKIT_NUM_PROCS"))
else:
    CPU_COUNT = CONFIG.get("num_process", local_hardware_info()["cpus"])


def _task_wrapper(param):
    (task, value, task_args, task_kwargs) = param
    return task(value, *task_args, **task_kwargs)


def should_run_in_parallel(num_processes: int | None = None) -> bool:
    """Return whether the current parallelisation configuration suggests that we should run things
    like :func:`parallel_map` in parallel (``True``) or degrade to serial (``False``).

    Args:
        num_processes: the number of processes requested for use (if given).
    """
    num_processes = CPU_COUNT if num_processes is None else num_processes
    return (
        num_processes > 1
        and os.getenv("QISKIT_IN_PARALLEL", "FALSE") == "FALSE"
        and CONFIG.get("parallel_enabled", PARALLEL_DEFAULT)
    )


def parallel_map(  # pylint: disable=dangerous-default-value
    task, values, task_args=(), task_kwargs={}, num_processes=CPU_COUNT
):
    """
    Parallel execution of a mapping of `values` to the function `task`. This
    is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    This will parallelise the results if the number of ``values`` is greater than one, and the
    current system configuration permits parallelization.

    Args:
        task (func): Function that is to be called for each value in ``values``.
        values (array_like): List or array of values for which the ``task`` function is to be
            evaluated.
        task_args (list): Optional additional arguments to the ``task`` function.
        task_kwargs (dict): Optional additional keyword argument to the ``task`` function.
        num_processes (int): Number of processes to spawn.

    Returns:
        result: The result list contains the value of ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``.

    Raises:
        QiskitError: If user interrupts via keyboard.

    Examples:

        .. plot::
           :include-source:
           :nofigs:

            import time
            from qiskit.utils import parallel_map
            def func(_):
                    time.sleep(0.1)
                    return 0
            parallel_map(func, list(range(10)));
    """
    if num_processes is None:
        num_processes = CPU_COUNT
    if len(values) == 0:
        return []
    if len(values) == 1:
        return [task(values[0], *task_args, **task_kwargs)]

    if should_run_in_parallel(num_processes):
        os.environ["QISKIT_IN_PARALLEL"] = "TRUE"
        try:
            results = []
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                param = ((task, value, task_args, task_kwargs) for value in values)
                future = executor.map(_task_wrapper, param)

            results = list(future)

        except (KeyboardInterrupt, Exception) as error:
            if isinstance(error, KeyboardInterrupt):
                os.environ["QISKIT_IN_PARALLEL"] = "FALSE"
                raise QiskitError("Keyboard interrupt in parallel_map.") from error
            # Otherwise just reset parallel flag and error
            os.environ["QISKIT_IN_PARALLEL"] = "FALSE"
            raise error

        os.environ["QISKIT_IN_PARALLEL"] = "FALSE"
        return results

    results = []
    for _, value in enumerate(values):
        result = task(value, *task_args, **task_kwargs)
        results.append(result)
    return results

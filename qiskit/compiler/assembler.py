# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Assemble function for converting a list of circuits into a qobj"""
import copy
import logging
import uuid
import warnings
from time import time
from typing import Dict, List, Optional, Union

import numpy as np

from qiskit.assembler import assemble_schedules
from qiskit.assembler.run_config import RunConfig
from qiskit.circuit import Parameter, QuantumCircuit, Qubit
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.pulse import Instruction, LoConfig, Schedule, ScheduleBlock
from qiskit.pulse.channels import PulseChannel
from qiskit.qobj import QasmQobj, PulseQobj, QobjHeader
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.utils import deprecate_func
from qiskit.assembler.assemble_circuits import _assemble_circuits

logger = logging.getLogger(__name__)


def _log_assembly_time(start_time, end_time):
    log_msg = f"Total Assembly Time - {((end_time - start_time) * 1000):.5f} (ms)"
    logger.info(log_msg)


# TODO: parallelize over the experiments (serialize each separately, then add global header/config)
@deprecate_func(
    since="1.2",
    removal_timeline="in the 2.0 release",
    additional_msg="The `Qobj` class and related functionality are part of the deprecated "
    "`BackendV1` workflow,  and no longer necessary for `BackendV2`. If a user "
    "workflow requires `Qobj` it likely relies on deprecated functionality and "
    "should be updated to use `BackendV2`.",
)
def assemble(
    experiments: Union[
        QuantumCircuit,
        List[QuantumCircuit],
        Schedule,
        List[Schedule],
        ScheduleBlock,
        List[ScheduleBlock],
    ],
    backend: Optional[Backend] = None,
    qobj_id: Optional[str] = None,
    qobj_header: Optional[Union[QobjHeader, Dict]] = None,
    shots: Optional[int] = None,
    memory: Optional[bool] = False,
    seed_simulator: Optional[int] = None,
    qubit_lo_freq: Optional[List[float]] = None,
    meas_lo_freq: Optional[List[float]] = None,
    qubit_lo_range: Optional[List[float]] = None,
    meas_lo_range: Optional[List[float]] = None,
    schedule_los: Optional[
        Union[
            List[Union[Dict[PulseChannel, float], LoConfig]],
            Union[Dict[PulseChannel, float], LoConfig],
        ]
    ] = None,
    meas_level: Union[int, MeasLevel] = MeasLevel.CLASSIFIED,
    meas_return: Union[str, MeasReturnType] = MeasReturnType.AVERAGE,
    meas_map: Optional[List[List[Qubit]]] = None,
    memory_slot_size: int = 100,
    rep_time: Optional[int] = None,
    rep_delay: Optional[float] = None,
    parameter_binds: Optional[List[Dict[Parameter, float]]] = None,
    parametric_pulses: Optional[List[str]] = None,
    init_qubits: bool = True,
    **run_config: Dict,
) -> Union[QasmQobj, PulseQobj]:
    """Assemble a list of circuits or pulse schedules into a ``Qobj``.

    This function serializes the payloads, which could be either circuits or schedules,
    to create ``Qobj`` "experiments". It further annotates the experiment payload with
    header and configurations.

    NOTE: ``Backend.options`` is not used within assemble. The required values
    (previously given by backend.set_options) should be manually extracted
    from options and supplied directly when calling.

    Args:
        experiments: Circuit(s) or pulse schedule(s) to execute
        backend: If set, some runtime options are automatically grabbed from
            ``backend.configuration()`` and ``backend.defaults()``.
            If any other option is explicitly set (e.g., ``rep_time``), it
            will override the backend's.
            If any other options is set in the run_config, it will
            also override the backend's.
        qobj_id: String identifier to annotate the ``Qobj``
        qobj_header: User input that will be inserted in ``Qobj`` header, and will also be
            copied to the corresponding Result header. Headers do not affect the run.
        shots: Number of repetitions of each circuit, for sampling. Default: 1024
            or ``max_shots`` from the backend configuration, whichever is smaller
        memory: If ``True``, per-shot measurement bitstrings are returned as well
            (provided the backend supports it). For OpenPulse jobs, only
            measurement level 2 supports this option.
        seed_simulator: Random seed to control sampling, for when backend is a simulator
        qubit_lo_freq: List of job level qubit drive LO frequencies in Hz. Overridden by
            ``schedule_los`` if specified. Must have length ``n_qubits.``
        meas_lo_freq: List of measurement LO frequencies in Hz. Overridden by ``schedule_los`` if
            specified. Must have length ``n_qubits.``
        qubit_lo_range: List of job level drive LO ranges each of form ``[range_min, range_max]``
            in Hz. Used to validate ``qubit_lo_freq``. Must have length ``n_qubits.``
        meas_lo_range: List of job level measurement LO ranges each of form
            ``[range_min, range_max]`` in Hz. Used to validate ``meas_lo_freq``. Must have length
            ``n_qubits.``
        schedule_los: Experiment level (ie circuit or schedule) LO frequency configurations for
            qubit drive and measurement channels. These values override the job level values from
            ``default_qubit_los`` and ``default_meas_los``. Frequencies are in Hz. Settable for
            OpenQASM 2 and pulse jobs.
        meas_level: Set the appropriate level of the measurement output for pulse experiments.
        meas_return: Level of measurement data for the backend to return.

            For ``meas_level`` 0 and 1:
                * ``single`` returns information from every shot.
                * ``avg`` returns average measurement output (averaged over number of shots).
        meas_map: List of lists, containing qubits that must be measured together.
        memory_slot_size: Size of each memory slot if the output is Level 0.
        rep_time (int): Time per program execution in seconds. Must be from the list provided
            by the backend (``backend.configuration().rep_times``). Defaults to the first entry.
        rep_delay (float): Delay between programs in seconds. Only supported on certain
            backends (if ``backend.configuration().dynamic_reprate_enabled=True``). If supported,
            ``rep_delay`` will be used instead of ``rep_time`` and must be from the range supplied
            by the backend (``backend.configuration().rep_delay_range``). Default is given by
            ``backend.configuration().default_rep_delay``.
        parameter_binds: List of Parameter bindings over which the set of experiments will be
            executed. Each list element (bind) should be of the form
            {Parameter1: value1, Parameter2: value2, ...}. All binds will be
            executed across all experiments; e.g., if parameter_binds is a
            length-n list, and there are m experiments, a total of m x n
            experiments will be run (one for each experiment/bind pair).
        parametric_pulses: A list of pulse shapes which are supported internally on the backend.
            Example::

            ['gaussian', 'constant']
        init_qubits: Whether to reset the qubits to the ground state for each shot.
                     Default: ``True``.
        **run_config: Extra arguments used to configure the run (e.g., for Aer configurable
            backends). Refer to the backend documentation for details on these
            arguments.

    Returns:
            A ``Qobj`` that can be run on a backend. Depending on the type of input,
            this will be either a ``QasmQobj`` or a ``PulseQobj``.

    Raises:
        QiskitError: if the input cannot be interpreted as either circuits or schedules
    """
    return _assemble(
        experiments,
        backend,
        qobj_id,
        qobj_header,
        shots,
        memory,
        seed_simulator,
        qubit_lo_freq,
        meas_lo_freq,
        qubit_lo_range,
        meas_lo_range,
        schedule_los,
        meas_level,
        meas_return,
        meas_map,
        memory_slot_size,
        rep_time,
        rep_delay,
        parameter_binds,
        parametric_pulses,
        init_qubits,
        **run_config,
    )


# Note for future: this method is used in `BasicSimulator` and may need to be kept past the
# `assemble` removal deadline (2.0). If it is kept (potentially in a different location),
# we will need an alternative for the backend.configuration() access that currently takes
# place in L566 (`parse_circuit_args`) and L351 (`parse_common_args`)
# because backend.configuration() is also set for removal in 2.0.
# The ultimate goal will be to move away from relying on any kind of `assemble` implementation
# because of how tightly coupled it is to these legacy data structures. But as a transition step,
# given that we would only have to support the subcase of `BasicSimulator`, we could probably just
# inline the relevant config values that are already hardcoded in the basic simulator configuration
# generator.
def _assemble(
    experiments: Union[
        QuantumCircuit,
        List[QuantumCircuit],
        Schedule,
        List[Schedule],
        ScheduleBlock,
        List[ScheduleBlock],
    ],
    backend: Optional[Backend] = None,
    qobj_id: Optional[str] = None,
    qobj_header: Optional[Union[QobjHeader, Dict]] = None,
    shots: Optional[int] = None,
    memory: Optional[bool] = False,
    seed_simulator: Optional[int] = None,
    qubit_lo_freq: Optional[List[float]] = None,
    meas_lo_freq: Optional[List[float]] = None,
    qubit_lo_range: Optional[List[float]] = None,
    meas_lo_range: Optional[List[float]] = None,
    schedule_los: Optional[
        Union[
            List[Union[Dict[PulseChannel, float], LoConfig]],
            Union[Dict[PulseChannel, float], LoConfig],
        ]
    ] = None,
    meas_level: Union[int, MeasLevel] = MeasLevel.CLASSIFIED,
    meas_return: Union[str, MeasReturnType] = MeasReturnType.AVERAGE,
    meas_map: Optional[List[List[Qubit]]] = None,
    memory_slot_size: int = 100,
    rep_time: Optional[int] = None,
    rep_delay: Optional[float] = None,
    parameter_binds: Optional[List[Dict[Parameter, float]]] = None,
    parametric_pulses: Optional[List[str]] = None,
    init_qubits: bool = True,
    **run_config: Dict,
) -> Union[QasmQobj, PulseQobj]:
    start_time = time()
    experiments = experiments if isinstance(experiments, list) else [experiments]
    pulse_qobj = any(isinstance(exp, (ScheduleBlock, Schedule, Instruction)) for exp in experiments)
    with warnings.catch_warnings():
        # The Qobj class is deprecated, the backend.configuration() method is too
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
        qobj_id, qobj_header, run_config_common_dict = _parse_common_args(
            backend,
            qobj_id,
            qobj_header,
            shots,
            memory,
            seed_simulator,
            init_qubits,
            rep_delay,
            qubit_lo_freq,
            meas_lo_freq,
            qubit_lo_range,
            meas_lo_range,
            schedule_los,
            pulse_qobj=pulse_qobj,
            **run_config,
        )

    # assemble either circuits or schedules
    if all(isinstance(exp, QuantumCircuit) for exp in experiments):
        run_config = _parse_circuit_args(
            parameter_binds,
            backend,
            meas_level,
            meas_return,
            parametric_pulses,
            **run_config_common_dict,
        )

        # If circuits are parameterized, bind parameters and remove from run_config
        bound_experiments, run_config = _expand_parameters(
            circuits=experiments, run_config=run_config
        )
        end_time = time()
        _log_assembly_time(start_time, end_time)
        return _assemble_circuits(
            circuits=bound_experiments,
            qobj_id=qobj_id,
            qobj_header=qobj_header,
            run_config=run_config,
        )

    elif all(isinstance(exp, (ScheduleBlock, Schedule, Instruction)) for exp in experiments):
        run_config = _parse_pulse_args(
            backend,
            meas_level,
            meas_return,
            meas_map,
            memory_slot_size,
            rep_time,
            parametric_pulses,
            **run_config_common_dict,
        )

        end_time = time()
        _log_assembly_time(start_time, end_time)
        return assemble_schedules(
            schedules=experiments, qobj_id=qobj_id, qobj_header=qobj_header, run_config=run_config
        )

    else:
        raise QiskitError("bad input to assemble() function; must be either circuits or schedules")


# TODO: rework to return a list of RunConfigs (one for each experiments), and a global one
def _parse_common_args(
    backend,
    qobj_id,
    qobj_header,
    shots,
    memory,
    seed_simulator,
    init_qubits,
    rep_delay,
    qubit_lo_freq,
    meas_lo_freq,
    qubit_lo_range,
    meas_lo_range,
    schedule_los,
    pulse_qobj=False,
    **run_config,
):
    """Resolve the various types of args allowed to the assemble() function through
    duck typing, overriding args, etc. Refer to the assemble() docstring for details on
    what types of inputs are allowed.

    Here the args are resolved by converting them to standard instances, and prioritizing
    them in case a run option is passed through multiple args (explicitly setting an arg
    has more priority than the arg set by backend)

    Returns:
        RunConfig: a run config, which is a standardized object that configures the qobj
            and determines the runtime environment.

    Raises:
        QiskitError:
            - If the memory arg is True and the backend does not support memory.
            - If ``shots`` exceeds ``max_shots`` for the configured backend.
            - If ``shots`` are not int type.
            - If any of qubit or meas lo's, or associated ranges do not have length equal to
            ``n_qubits``.
            - If qubit or meas lo's do not fit into prescribed ranges.
    """
    # grab relevant info from backend if it exists
    backend_config = None
    backend_defaults = None
    n_qubits = None
    if backend:
        backend_config = backend.configuration()
        n_qubits = backend_config.n_qubits
        # check for memory flag applied to backend that does not support memory
        if memory and not backend_config.memory:
            raise QiskitError(f"memory not supported by backend {backend_config.backend_name}")

        # try to set defaults for pulse, other leave as None
        pulse_param_set = (
            qubit_lo_freq is not None
            or meas_lo_freq is not None
            or qubit_lo_range is not None
            or meas_lo_range is not None
            or schedule_los is not None
        )
        if pulse_qobj or (backend_config.open_pulse and pulse_param_set):
            try:
                backend_defaults = backend.defaults()
            except AttributeError:
                pass

    # an identifier for the Qobj
    qobj_id = qobj_id or str(uuid.uuid4())

    # The header that goes at the top of the Qobj (and later Result)
    # we process it as dict, then write entries that are not None to a QobjHeader object
    qobj_header = qobj_header or {}
    if isinstance(qobj_header, QobjHeader):
        qobj_header = qobj_header.to_dict()
    backend_name = getattr(backend_config, "backend_name", None)
    backend_version = getattr(backend_config, "backend_version", None)
    qobj_header = {
        "backend_name": backend_name,
        "backend_version": backend_version,
        **qobj_header,
    }
    qobj_header = QobjHeader(**{k: v for k, v in qobj_header.items() if v is not None})

    max_shots = getattr(backend_config, "max_shots", None)
    if shots is None:
        if max_shots:
            shots = min(1024, max_shots)
        else:
            shots = 1024
    elif not isinstance(shots, (int, np.integer)):
        raise QiskitError("Argument 'shots' should be of type 'int'")
    elif max_shots and max_shots < shots:
        raise QiskitError(
            f"Number of shots specified: {max_shots} exceeds max_shots property of the "
            f"backend: {max_shots}."
        )

    dynamic_reprate_enabled = getattr(backend_config, "dynamic_reprate_enabled", False)
    if dynamic_reprate_enabled:
        default_rep_delay = getattr(backend_config, "default_rep_delay", None)
        rep_delay_range = getattr(backend_config, "rep_delay_range", None)
        rep_delay = _parse_rep_delay(rep_delay, default_rep_delay, rep_delay_range)
    else:
        if rep_delay is not None:
            rep_delay = None
            warnings.warn(
                "Dynamic rep rates not supported on this backend, cannot use rep_delay.",
                RuntimeWarning,
            )

    qubit_lo_freq = qubit_lo_freq or getattr(backend_defaults, "qubit_freq_est", None)
    meas_lo_freq = meas_lo_freq or getattr(backend_defaults, "meas_freq_est", None)

    qubit_lo_range = qubit_lo_range or getattr(backend_config, "qubit_lo_range", None)
    meas_lo_range = meas_lo_range or getattr(backend_config, "meas_lo_range", None)

    # check that LO frequencies are in the perscribed range
    _check_lo_freqs(qubit_lo_freq, qubit_lo_range, "qubit")
    _check_lo_freqs(meas_lo_freq, meas_lo_range, "meas")

    # configure experiment level LO frequencies
    schedule_los = schedule_los or []
    if isinstance(schedule_los, (LoConfig, dict)):
        schedule_los = [schedule_los]

    # Convert to LoConfig if LO configuration supplied as dictionary
    schedule_los = [
        lo_config if isinstance(lo_config, LoConfig) else LoConfig(lo_config)
        for lo_config in schedule_los
    ]

    # create run configuration and populate
    run_config_dict = {
        "shots": shots,
        "memory": memory,
        "seed_simulator": seed_simulator,
        "init_qubits": init_qubits,
        "rep_delay": rep_delay,
        "qubit_lo_freq": qubit_lo_freq,
        "meas_lo_freq": meas_lo_freq,
        "qubit_lo_range": qubit_lo_range,
        "meas_lo_range": meas_lo_range,
        "schedule_los": schedule_los,
        "n_qubits": n_qubits,
        **run_config,
    }

    return qobj_id, qobj_header, run_config_dict


def _check_lo_freqs(
    lo_freq: Union[List[float], None],
    lo_range: Union[List[float], None],
    lo_type: str,
):
    """Check that LO frequencies are within the perscribed LO range.

    NOTE: Only checks if frequency/range lists have equal length. And does not check that the lists
    have length ``n_qubits``. This is because some backends, like simulator backends, do not
    require these constraints. For real hardware, these parameters will be validated on the backend.

    Args:
        lo_freq: List of LO frequencies.
        lo_range: Nested list of LO frequency ranges. Inner list is of the form
            ``[lo_min, lo_max]``.
        lo_type: The type of LO value--"qubit" or "meas".

    Raises:
        QiskitError:
            - If each element of the LO range is not a 2d list.
            - If the LO frequency is not in the LO range for a given qubit.
    """
    if lo_freq and lo_range and len(lo_freq) == len(lo_range):
        for i, freq in enumerate(lo_freq):
            freq_range = lo_range[i]
            if not (isinstance(freq_range, list) and len(freq_range) == 2):
                raise QiskitError(f"Each element of {lo_type} LO range must be a 2d list.")
            if freq < freq_range[0] or freq > freq_range[1]:
                raise QiskitError(
                    f"Qubit {i} {lo_type} LO frequency is {freq}. "
                    f"The range is [{freq_range[0]}, {freq_range[1]}]."
                )


def _parse_pulse_args(
    backend,
    meas_level,
    meas_return,
    meas_map,
    memory_slot_size,
    rep_time,
    parametric_pulses,
    **run_config,
):
    """Build a pulse RunConfig replacing unset arguments with defaults derived from the `backend`.
    See `assemble` for more information on the required arguments.

    Returns:
        RunConfig: a run config, which is a standardized object that configures the qobj
            and determines the runtime environment.
    Raises:
        QiskitError: If the given meas_level is not allowed for the given `backend`.
    """
    # grab relevant info from backend if it exists
    backend_config = None
    if backend:
        backend_config = backend.configuration()

        if meas_level not in getattr(backend_config, "meas_levels", [MeasLevel.CLASSIFIED]):
            raise QiskitError(
                f"meas_level = {meas_level} not supported for backend "
                f"{backend_config.backend_name}, only {backend_config.meas_levels} is supported"
            )

    meas_map = meas_map or getattr(backend_config, "meas_map", None)
    dynamic_reprate_enabled = getattr(backend_config, "dynamic_reprate_enabled", False)

    rep_time = rep_time or getattr(backend_config, "rep_times", None)
    if rep_time:
        if dynamic_reprate_enabled:
            warnings.warn(
                "Dynamic rep rates are supported on this backend. 'rep_delay' will be "
                "used instead of 'rep_time'.",
                RuntimeWarning,
            )
        if isinstance(rep_time, list):
            rep_time = rep_time[0]
        rep_time = int(rep_time * 1e6)  # convert sec to μs
    if parametric_pulses is None:
        parametric_pulses = getattr(backend_config, "parametric_pulses", [])

    # create run configuration and populate
    run_config_dict = {
        "meas_level": meas_level,
        "meas_return": meas_return,
        "meas_map": meas_map,
        "memory_slot_size": memory_slot_size,
        "rep_time": rep_time,
        "parametric_pulses": parametric_pulses,
        **run_config,
    }
    run_config = RunConfig(**{k: v for k, v in run_config_dict.items() if v is not None})

    return run_config


def _parse_circuit_args(
    parameter_binds, backend, meas_level, meas_return, parametric_pulses, **run_config
):
    """Build a circuit RunConfig replacing unset arguments with defaults derived from the `backend`.
    See `assemble` for more information on the required arguments.

    Returns:
        RunConfig: a run config, which is a standardized object that configures the qobj
            and determines the runtime environment.
    """
    parameter_binds = parameter_binds or []
    # create run configuration and populate
    run_config_dict = {"parameter_binds": parameter_binds, **run_config}
    if parametric_pulses is None:
        if backend:
            with warnings.catch_warnings():
                # TODO (2.0): See comment on L192 regarding backend.configuration removal
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
                run_config_dict["parametric_pulses"] = getattr(
                    backend.configuration(), "parametric_pulses", []
                )
        else:
            run_config_dict["parametric_pulses"] = []
    else:
        run_config_dict["parametric_pulses"] = parametric_pulses
    if meas_level:
        run_config_dict["meas_level"] = meas_level
        # only enable `meas_return` if `meas_level` isn't classified
        if meas_level != MeasLevel.CLASSIFIED:
            run_config_dict["meas_return"] = meas_return

    run_config = RunConfig(**{k: v for k, v in run_config_dict.items() if v is not None})

    return run_config


def _parse_rep_delay(
    rep_delay: float, default_rep_delay: float, rep_delay_range: List[float]
) -> float:
    """Parse and set ``rep_delay`` parameter in runtime config.

    Args:
        rep_delay: Initial rep delay.
        default_rep_delay: Backend default rep delay.
        rep_delay_range: Backend list defining allowable range of rep delays.

    Raises:
        QiskitError: If rep_delay is not in the backend rep_delay_range.
    Returns:
        float: Modified rep delay after parsing.
    """
    if rep_delay is None:
        rep_delay = default_rep_delay

    if rep_delay is not None:
        # check that rep_delay is in rep_delay_range
        if rep_delay_range is not None and isinstance(rep_delay_range, list):
            if len(rep_delay_range) != 2:
                raise QiskitError(
                    f"Backend rep_delay_range {rep_delay_range} must be a list with two entries."
                )
            if not rep_delay_range[0] <= rep_delay <= rep_delay_range[1]:
                raise QiskitError(
                    f"Supplied rep delay {rep_delay} not in the supported "
                    f"backend range {rep_delay_range}"
                )
        rep_delay = rep_delay * 1e6  # convert sec to μs

    return rep_delay


def _expand_parameters(circuits, run_config):
    """Verifies that there is a single common set of parameters shared between
    all circuits and all parameter binds in the run_config. Returns an expanded
    list of circuits (if parameterized) with all parameters bound, and a copy of
    the run_config with parameter_binds cleared.

    If neither the circuits nor the run_config specify parameters, the two are
    returned unmodified.

    Raises:
        QiskitError: if run_config parameters are not compatible with circuit parameters

    Returns:
        Tuple(List[QuantumCircuit], RunConfig):
          - List of input circuits expanded and with parameters bound
          - RunConfig with parameter_binds removed
    """
    parameter_binds = run_config.parameter_binds

    if parameter_binds and any(parameter_binds) or any(circuit.parameters for circuit in circuits):

        # Unroll params here in order to handle ParamVects
        all_bind_parameters = [
            QuantumCircuit()._unroll_param_dict(bind).keys() for bind in parameter_binds
        ]

        all_circuit_parameters = [circuit.parameters for circuit in circuits]

        # Collect set of all unique parameters across all circuits and binds
        unique_parameters = {
            param
            for param_list in all_bind_parameters + all_circuit_parameters
            for param in param_list
        }

        # Check that all parameters are common to all circuits and binds
        if (
            not all_bind_parameters
            or not all_circuit_parameters
            or any(unique_parameters != bind_params for bind_params in all_bind_parameters)
            or any(unique_parameters != parameters for parameters in all_circuit_parameters)
        ):
            raise QiskitError(
                (
                    "Mismatch between run_config.parameter_binds and all circuit parameters. "
                    + "Parameter binds: {} "
                    + "Circuit parameters: {}"
                ).format(all_bind_parameters, all_circuit_parameters)
            )

        circuits = [
            circuit.assign_parameters(binds) for circuit in circuits for binds in parameter_binds
        ]

        # All parameters have been expanded and bound, so remove from run_config
        run_config = copy.deepcopy(run_config)
        run_config.parameter_binds = []

    return circuits, run_config

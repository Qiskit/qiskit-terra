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

# pylint: disable=invalid-sequence-index

"""Circuit transpile function"""
import logging
from time import time
from typing import List, Union, Dict, Callable, Any, Optional, TypeVar
import warnings

from qiskit import user_config
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.backend import Backend
from qiskit.providers.backend_compat import BackendV2Converter
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.pulse import Schedule, InstructionScheduleMap
from qiskit.transpiler import Layout, CouplingMap, PropertySet
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.exceptions import TranspilerError, CircuitTooWideForTarget
from qiskit.transpiler.instruction_durations import InstructionDurationsType
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.target import Target
from qiskit.utils import deprecate_arg
from qiskit.utils.deprecate_pulse import deprecate_pulse_arg

logger = logging.getLogger(__name__)

_CircuitT = TypeVar("_CircuitT", bound=Union[QuantumCircuit, List[QuantumCircuit]])


@deprecate_arg(
    name="instruction_durations",
    since="1.3",
    package_name="Qiskit",
    removal_timeline="in Qiskit 2.0",
    additional_msg="The `target` parameter should be used instead. You can build a `Target` instance "
    "with defined instruction durations with "
    "`Target.from_configuration(..., instruction_durations=...)`",
)
@deprecate_arg(
    name="timing_constraints",
    since="1.3",
    package_name="Qiskit",
    removal_timeline="in Qiskit 2.0",
    additional_msg="The `target` parameter should be used instead. You can build a `Target` instance "
    "with defined timing constraints with "
    "`Target.from_configuration(..., timing_constraints=...)`",
)
@deprecate_arg(
    name="backend_properties",
    since="1.3",
    package_name="Qiskit",
    removal_timeline="in Qiskit 2.0",
    additional_msg="The `target` parameter should be used instead. You can build a `Target` instance "
    "with defined properties with Target.from_configuration(..., backend_properties=...)",
)
@deprecate_pulse_arg("inst_map", predicate=lambda inst_map: inst_map is not None)
def transpile(  # pylint: disable=too-many-return-statements
    circuits: _CircuitT,
    backend: Optional[Backend] = None,
    basis_gates: Optional[List[str]] = None,
    inst_map: Optional[List[InstructionScheduleMap]] = None,
    coupling_map: Optional[Union[CouplingMap, List[List[int]]]] = None,
    backend_properties: Optional[BackendProperties] = None,
    initial_layout: Optional[Union[Layout, Dict, List]] = None,
    layout_method: Optional[str] = None,
    routing_method: Optional[str] = None,
    translation_method: Optional[str] = None,
    scheduling_method: Optional[str] = None,
    instruction_durations: Optional[InstructionDurationsType] = None,
    dt: Optional[float] = None,
    approximation_degree: Optional[float] = 1.0,
    timing_constraints: Optional[Dict[str, int]] = None,
    seed_transpiler: Optional[int] = None,
    optimization_level: Optional[int] = None,
    callback: Optional[Callable[[BasePass, DAGCircuit, float, PropertySet, int], Any]] = None,
    output_name: Optional[Union[str, List[str]]] = None,
    unitary_synthesis_method: str = "default",
    unitary_synthesis_plugin_config: Optional[dict] = None,
    target: Optional[Target] = None,
    hls_config: Optional[HLSConfig] = None,
    init_method: Optional[str] = None,
    optimization_method: Optional[str] = None,
    ignore_backend_supplied_default_methods: bool = False,
    num_processes: Optional[int] = None,
    qubits_initially_zero: bool = True,
) -> _CircuitT:
    """Transpile one or more circuits, according to some desired transpilation targets.

    Transpilation is potentially done in parallel using multiprocessing when ``circuits``
    is a list with > 1 :class:`~.QuantumCircuit` object, depending on the local environment
    and configuration.

    The prioritization of transpilation target constraints works as follows: if a ``target``
    input is provided, it will take priority over any ``backend`` input or loose constraints
    (``basis_gates``, ``inst_map``, ``coupling_map``, ``backend_properties``, ``instruction_durations``,
    ``dt`` or ``timing_constraints``). If a ``backend`` is provided together with any loose constraint
    from the list above, the loose constraint will take priority over the corresponding backend
    constraint. This behavior is independent of whether the ``backend`` instance is of type
    :class:`.BackendV1` or :class:`.BackendV2`, as summarized in the table below. The first column
    in the table summarizes the potential user-provided constraints, and each cell shows whether
    the priority is assigned to that specific constraint input or another input
    (`target`/`backend(V1)`/`backend(V2)`).

    ============================ ========= ======================== =======================
    User Provided                target    backend(V1)              backend(V2)
    ============================ ========= ======================== =======================
    **basis_gates**              target    basis_gates              basis_gates
    **coupling_map**             target    coupling_map             coupling_map
    **instruction_durations**    target    instruction_durations    instruction_durations
    **inst_map**                 target    inst_map                 inst_map
    **dt**                       target    dt                       dt
    **timing_constraints**       target    timing_constraints       timing_constraints
    **backend_properties**       target    backend_properties       backend_properties
    ============================ ========= ======================== =======================

    Args:
        circuits: Circuit(s) to transpile
        backend: If set, the transpiler will compile the input circuit to this target
            device. If any other option is explicitly set (e.g., ``coupling_map``), it
            will override the backend's.
        basis_gates: List of basis gate names to unroll to
            (e.g: ``['u1', 'u2', 'u3', 'cx']``). If ``None``, do not unroll.
        inst_map: DEPRECATED. Mapping of unrolled gates to pulse schedules. If this is not provided,
            transpiler tries to get from the backend. If any user defined calibration
            is found in the map and this is used in a circuit, transpiler attaches
            the custom gate definition to the circuit. This enables one to flexibly
            override the low-level instruction implementation. This feature is available
            iff the backend supports the pulse gate experiment.
        coupling_map: Directed coupling map (perhaps custom) to target in mapping. If
            the coupling map is symmetric, both directions need to be specified.

            Multiple formats are supported:

            #. ``CouplingMap`` instance
            #. List, must be given as an adjacency matrix, where each entry
               specifies all directed two-qubit interactions supported by backend,
               e.g: ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``

        backend_properties: properties returned by a backend, including information on gate
            errors, readout errors, qubit coherence times, etc. Find a backend
            that provides this information with: ``backend.properties()``
        initial_layout: Initial position of virtual qubits on physical qubits.
            If this layout makes the circuit compatible with the coupling_map
            constraints, it will be used. The final layout is not guaranteed to be the same,
            as the transpiler may permute qubits through swaps or other means.
            Multiple formats are supported:

            #. ``Layout`` instance
            #. Dict
               * virtual to physical::

                    {qr[0]: 0,
                     qr[1]: 3,
                     qr[2]: 5}

               * physical to virtual::

                    {0: qr[0],
                     3: qr[1],
                     5: qr[2]}

            #. List

               * virtual to physical::

                    [0, 3, 5]  # virtual qubits are ordered (in addition to named)

               * physical to virtual::

                    [qr[0], None, None, qr[1], None, qr[2]]

        layout_method: Name of layout selection pass ('trivial', 'dense', 'sabre').
            This can also be the external plugin name to use for the ``layout`` stage.
            You can see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"layout"`` for the ``stage_name`` argument.
        routing_method: Name of routing pass
            ('basic', 'lookahead', 'stochastic', 'sabre', 'none'). Note
            This can also be the external plugin name to use for the ``routing`` stage.
            You can see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"routing"`` for the ``stage_name`` argument.
        translation_method: Name of translation pass ('unroller', 'translator', 'synthesis')
            This can also be the external plugin name to use for the ``translation`` stage.
            You can see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"translation"`` for the ``stage_name`` argument.
        scheduling_method: Name of scheduling pass.
            * ``'as_soon_as_possible'``: Schedule instructions greedily, as early as possible
            on a qubit resource. (alias: ``'asap'``)
            * ``'as_late_as_possible'``: Schedule instructions late, i.e. keeping qubits
            in the ground state when possible. (alias: ``'alap'``)
            If ``None``, no scheduling will be done. This can also be the external plugin name
            to use for the ``scheduling`` stage. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"scheduling"`` for the ``stage_name``
            argument.
        instruction_durations: Durations of instructions.
            Applicable only if scheduling_method is specified.
            The gate lengths defined in ``backend.properties`` are used as default.
            They are overwritten if this ``instruction_durations`` is specified.
            The format of ``instruction_durations`` must be as follows.
            The `instruction_durations` must be given as a list of tuples
            [(instruction_name, qubits, duration, unit), ...].
            | [('cx', [0, 1], 12.3, 'ns'), ('u3', [0], 4.56, 'ns')]
            | [('cx', [0, 1], 1000), ('u3', [0], 300)]
            If unit is omitted, the default is 'dt', which is a sample time depending on backend.
            If the time unit is 'dt', the duration must be an integer.
        dt: Backend sample time (resolution) in seconds.
            If ``None`` (default), ``backend.configuration().dt`` is used.
        approximation_degree (float): heuristic dial used for circuit approximation
            (1.0=no approximation, 0.0=maximal approximation)
        timing_constraints: An optional control hardware restriction on instruction time resolution.
            A quantum computer backend may report a set of restrictions, namely:

            - granularity: An integer value representing minimum pulse gate
              resolution in units of ``dt``. A user-defined pulse gate should have
              duration of a multiple of this granularity value.
            - min_length: An integer value representing minimum pulse gate
              length in units of ``dt``. A user-defined pulse gate should be longer
              than this length.
            - pulse_alignment: An integer value representing a time resolution of gate
              instruction starting time. Gate instruction should start at time which
              is a multiple of the alignment value.
            - acquire_alignment: An integer value representing a time resolution of measure
              instruction starting time. Measure instruction should start at time which
              is a multiple of the alignment value.

            This information will be provided by the backend configuration.
            If the backend doesn't have any restriction on the instruction time allocation,
            then ``timing_constraints`` is None and no adjustment will be performed.
        seed_transpiler: Sets random seed for the stochastic parts of the transpiler
        optimization_level: How much optimization to perform on the circuits.
            Higher levels generate more optimized circuits,
            at the expense of longer transpilation time.

            * 0: no optimization
            * 1: light optimization
            * 2: heavy optimization
            * 3: even heavier optimization

            If ``None``, level 2 will be chosen as default.
        callback: A callback function that will be called after each
            pass execution. The function will be called with 5 keyword
            arguments,
            | ``pass_``: the pass being run.
            | ``dag``: the dag output of the pass.
            | ``time``: the time to execute the pass.
            | ``property_set``: the property set.
            | ``count``: the index for the pass execution.
            The exact arguments passed expose the internals of the pass manager,
            and are subject to change as the pass manager internals change. If
            you intend to reuse a callback function over multiple releases, be
            sure to check that the arguments being passed are the same.
            To use the callback feature, define a function that will
            take in kwargs dict and access the variables. For example::

                def callback_func(**kwargs):
                    pass_ = kwargs['pass_']
                    dag = kwargs['dag']
                    time = kwargs['time']
                    property_set = kwargs['property_set']
                    count = kwargs['count']
                    ...
                transpile(circ, callback=callback_func)

        output_name: A list with strings to identify the output circuits. The length of
            the list should be exactly the length of the ``circuits`` parameter.
        unitary_synthesis_method (str): The name of the unitary synthesis
            method to use. By default ``'default'`` is used. You can see a list of installed
            plugins with :func:`.unitary_synthesis_plugin_names`.
        unitary_synthesis_plugin_config: An optional configuration dictionary
            that will be passed directly to the unitary synthesis plugin. By
            default this setting will have no effect as the default unitary
            synthesis method does not take custom configuration. This should
            only be necessary when a unitary synthesis plugin is specified with
            the ``unitary_synthesis_method`` argument. As this is custom for each
            unitary synthesis plugin refer to the plugin documentation for how
            to use this option.
        target: A backend transpiler target. Normally this is specified as part of
            the ``backend`` argument, but if you have manually constructed a
            :class:`~qiskit.transpiler.Target` object you can specify it manually here.
            This will override the target from ``backend``.
        hls_config: An optional configuration class
            :class:`~qiskit.transpiler.passes.synthesis.HLSConfig` that will be passed directly
            to :class:`~qiskit.transpiler.passes.synthesis.HighLevelSynthesis` transformation pass.
            This configuration class allows to specify for various high-level objects the lists of
            synthesis algorithms and their parameters.
        init_method: The plugin name to use for the ``init`` stage. By default an external
            plugin is not used. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"init"`` for the stage
            name argument.
        optimization_method: The plugin name to use for the
            ``optimization`` stage. By default an external
            plugin is not used. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"optimization"`` for the
            ``stage_name`` argument.
        ignore_backend_supplied_default_methods: If set to ``True`` any default methods specified by
            a backend will be ignored. Some backends specify alternative default methods
            to support custom compilation target-specific passes/plugins which support
            backend-specific compilation techniques. If you'd prefer that these defaults were
            not used this option is used to disable those backend-specific defaults.
        num_processes: The maximum number of parallel processes to launch for this call to
            transpile if parallel execution is enabled. This argument overrides
            ``num_processes`` in the user configuration file, and the ``QISKIT_NUM_PROCS``
            environment variable. If set to ``None`` the system default or local user configuration
            will be used.
        qubits_initially_zero: Indicates whether the input circuit is zero-initialized.

    Returns:
        The transpiled circuit(s).

    Raises:
        TranspilerError: in case of bad inputs to transpiler (like conflicting parameters)
            or errors in passes
    """
    arg_circuits_list = isinstance(circuits, list)
    circuits = circuits if arg_circuits_list else [circuits]

    if not circuits:
        return []

    # transpiling schedules is not supported yet.
    start_time = time()
    if all(isinstance(c, Schedule) for c in circuits):
        warnings.warn("Transpiling schedules is not supported yet.", UserWarning)
        end_time = time()
        _log_transpile_time(start_time, end_time)
        if arg_circuits_list:
            return circuits
        else:
            return circuits[0]

    if optimization_level is None:
        # Take optimization level from the configuration or 1 as default.
        config = user_config.get_config()
        optimization_level = config.get("transpile_optimization_level", 2)

    if backend is not None and getattr(backend, "version", 0) <= 1:
        warnings.warn(
            "The `transpile` function will stop supporting inputs of "
            f"type `BackendV1` ( {backend} ) in the `backend` parameter in a future "
            "release no earlier than 2.0. `BackendV1` is deprecated and implementations "
            "should move to `BackendV2`.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        with warnings.catch_warnings():
            # This is a temporary conversion step to allow for a smoother transition
            # to a fully target-based transpiler pipeline while maintaining the behavior
            # of `transpile` with BackendV1 inputs.
            # TODO BackendV1 is deprecated and this path can be
            #   removed once it gets removed:
            #   https://github.com/Qiskit/qiskit/pull/12850
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r".+qiskit\.providers\.backend_compat\.BackendV2Converter.+",
                module="qiskit",
            )
            backend = BackendV2Converter(backend)

    if (
        scheduling_method is not None
        and backend is None
        and target is None
        and not instruction_durations
    ):
        warnings.warn(
            "When scheduling circuits without backend,"
            " 'instruction_durations' should be usually provided.",
            UserWarning,
        )

    if not ignore_backend_supplied_default_methods:
        if scheduling_method is None and hasattr(backend, "get_scheduling_stage_plugin"):
            scheduling_method = backend.get_scheduling_stage_plugin()
        if translation_method is None and hasattr(backend, "get_translation_stage_plugin"):
            translation_method = backend.get_translation_stage_plugin()

    output_name = _parse_output_name(output_name, circuits)
    coupling_map = _parse_coupling_map(coupling_map)
    _check_circuits_coupling_map(circuits, coupling_map, backend)

    # Edge cases require using the old model (loose constraints) instead of building a target,
    # but we don't populate the passmanager config with loose constraints unless it's one of
    # the known edge cases to control the execution path.
    # Filter instruction_durations, timing_constraints, backend_properties and inst_map deprecation
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*``inst_map`` is deprecated as of Qiskit 1.3.*",
            module="qiskit",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*``timing_constraints`` is deprecated as of Qiskit 1.3.*",
            module="qiskit",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*``instruction_durations`` is deprecated as of Qiskit 1.3.*",
            module="qiskit",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*``backend_properties`` is deprecated as of Qiskit 1.3.*",
            module="qiskit",
        )
        pm = generate_preset_pass_manager(
            optimization_level,
            target=target,
            backend=backend,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            instruction_durations=instruction_durations,
            backend_properties=backend_properties,
            timing_constraints=timing_constraints,
            inst_map=inst_map,
            initial_layout=initial_layout,
            layout_method=layout_method,
            routing_method=routing_method,
            translation_method=translation_method,
            scheduling_method=scheduling_method,
            approximation_degree=approximation_degree,
            seed_transpiler=seed_transpiler,
            unitary_synthesis_method=unitary_synthesis_method,
            unitary_synthesis_plugin_config=unitary_synthesis_plugin_config,
            hls_config=hls_config,
            init_method=init_method,
            optimization_method=optimization_method,
            dt=dt,
            qubits_initially_zero=qubits_initially_zero,
        )

    out_circuits = pm.run(circuits, callback=callback, num_processes=num_processes)

    for name, circ in zip(output_name, out_circuits):
        circ.name = name
    end_time = time()
    _log_transpile_time(start_time, end_time)
    if arg_circuits_list:
        return out_circuits
    else:
        return out_circuits[0]


def _check_circuits_coupling_map(circuits, cmap, backend):
    # Check circuit width against number of qubits in coupling_map(s)
    max_qubits = None
    if cmap is not None:
        max_qubits = cmap.size()
    elif backend is not None:
        max_qubits = backend.num_qubits

    for circuit in circuits:
        # If coupling_map is not None or num_qubits == 1
        num_qubits = len(circuit.qubits)
        if max_qubits is not None and (num_qubits > max_qubits):
            raise CircuitTooWideForTarget(
                f"Number of qubits ({num_qubits}) in {circuit.name} "
                f"is greater than maximum ({max_qubits}) in the coupling_map"
            )


def _log_transpile_time(start_time, end_time):
    log_msg = f"Total Transpile Time - {((end_time - start_time) * 1000):.5f} (ms)"
    logger.info(log_msg)


def _parse_coupling_map(coupling_map):
    # coupling_map could be None, or a list of lists, e.g. [[0, 1], [2, 1]]
    if isinstance(coupling_map, list) and all(
        isinstance(i, list) and len(i) == 2 for i in coupling_map
    ):
        return CouplingMap(coupling_map)
    elif isinstance(coupling_map, list):
        raise TranspilerError(
            "Only a single input coupling map can be used with transpile() if you need to "
            "target different coupling maps for different circuits you must call transpile() "
            "multiple times"
        )
    else:
        return coupling_map


def _parse_output_name(output_name, circuits):
    # naming and returning circuits
    # output_name could be either a string or a list
    if output_name is not None:
        if isinstance(output_name, str):
            # single circuit
            if len(circuits) == 1:
                return [output_name]
            # multiple circuits
            else:
                raise TranspilerError(
                    "Expected a list object of length equal "
                    + "to that of the number of circuits "
                    + "being transpiled"
                )
        elif isinstance(output_name, list):
            if len(circuits) == len(output_name) and all(
                isinstance(name, str) for name in output_name
            ):
                return output_name
            else:
                raise TranspilerError(
                    "The length of output_name list "
                    "must be equal to the number of "
                    "transpiled circuits and the output_name "
                    "list should be strings."
                )
        else:
            raise TranspilerError(
                "The parameter output_name should be a string or a"
                f"list of strings: {type(output_name)} was used."
            )
    else:
        return [circuit.name for circuit in circuits]

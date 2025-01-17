# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Preset pass manager generation function
"""

import copy
import warnings

from qiskit.circuit.controlflow import (
    CONTROL_FLOW_OP_NAMES,
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    SwitchCaseOp,
)
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.circuit.quantumregister import Qubit
from qiskit.providers.backend import Backend
from qiskit.providers.backend_compat import BackendV2Converter
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.target import Target, target_to_backend_properties
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.utils import deprecate_arg
from qiskit.utils.deprecate_pulse import deprecate_pulse_arg

from .level0 import level_0_pass_manager
from .level1 import level_1_pass_manager
from .level2 import level_2_pass_manager
from .level3 import level_3_pass_manager


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
def generate_preset_pass_manager(
    optimization_level=2,
    backend=None,
    target=None,
    basis_gates=None,
    inst_map=None,
    coupling_map=None,
    instruction_durations=None,
    backend_properties=None,
    timing_constraints=None,
    initial_layout=None,
    layout_method=None,
    routing_method=None,
    translation_method=None,
    scheduling_method=None,
    approximation_degree=1.0,
    seed_transpiler=None,
    unitary_synthesis_method="default",
    unitary_synthesis_plugin_config=None,
    hls_config=None,
    init_method=None,
    optimization_method=None,
    dt=None,
    qubits_initially_zero=True,
    *,
    _skip_target=False,
):
    """Generate a preset :class:`~.PassManager`

    This function is used to quickly generate a preset pass manager. Preset pass
    managers are the default pass managers used by the :func:`~.transpile`
    function. This function provides a convenient and simple method to construct
    a standalone :class:`~.PassManager` object that mirrors what the :func:`~.transpile`
    function internally builds and uses.

    The target constraints for the pass manager construction can be specified through a :class:`.Target`
    instance, a :class:`.BackendV1` or :class:`.BackendV2` instance, or via loose constraints
    (``basis_gates``, ``inst_map``, ``coupling_map``, ``backend_properties``, ``instruction_durations``,
    ``dt`` or ``timing_constraints``).
    The order of priorities for target constraints works as follows: if a ``target``
    input is provided, it will take priority over any ``backend`` input or loose constraints.
    If a ``backend`` is provided together with any loose constraint
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
        optimization_level (int): The optimization level to generate a
            :class:`~.StagedPassManager` for. By default optimization level 2
            is used if this is not specified. This can be 0, 1, 2, or 3. Higher
            levels generate potentially more optimized circuits, at the expense
            of longer transpilation time:

                * 0: no optimization
                * 1: light optimization
                * 2: heavy optimization
                * 3: even heavier optimization

        backend (Backend): An optional backend object which can be used as the
            source of the default values for the ``basis_gates``, ``inst_map``,
            ``coupling_map``, ``backend_properties``, ``instruction_durations``,
            ``timing_constraints``, and ``target``. If any of those other arguments
            are specified in addition to ``backend`` they will take precedence
            over the value contained in the backend.
        target (Target): The :class:`~.Target` representing a backend compilation
            target. The following attributes will be inferred from this
            argument if they are not set: ``coupling_map``, ``basis_gates``,
            ``instruction_durations``, ``inst_map``, ``timing_constraints``
            and ``backend_properties``.
        basis_gates (list): List of basis gate names to unroll to
            (e.g: ``['u1', 'u2', 'u3', 'cx']``).
        inst_map (InstructionScheduleMap): DEPRECATED. Mapping object that maps gates to schedules.
            If any user defined calibration is found in the map and this is used in a
            circuit, transpiler attaches the custom gate definition to the circuit.
            This enables one to flexibly override the low-level instruction
            implementation.
        coupling_map (CouplingMap or list): Directed graph represented a coupling
            map. Multiple formats are supported:

            #. ``CouplingMap`` instance
            #. List, must be given as an adjacency matrix, where each entry
               specifies all directed two-qubit interactions supported by backend,
               e.g: ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``

        instruction_durations (InstructionDurations or list): Dictionary of duration
            (in dt) for each instruction. If specified, these durations overwrite the
            gate lengths in ``backend.properties``. Applicable only if ``scheduling_method``
            is specified.
            The format of ``instruction_durations`` must be as follows:
            They must be given as an :class:`.InstructionDurations` instance or a list of tuples

            ```
            [(instruction_name, qubits, duration, unit), ...].
            | [('cx', [0, 1], 12.3, 'ns'), ('u3', [0], 4.56, 'ns')]
            | [('cx', [0, 1], 1000), ('u3', [0], 300)]
            ```

            If ``unit`` is omitted, the default is ``'dt'``, which is a sample time depending on backend.
            If the time unit is ``'dt'``, the duration must be an integer.
        dt (float): Backend sample time (resolution) in seconds.
            If provided, this value will overwrite the ``dt`` value in ``instruction_durations``.
            If ``None`` (default) and a backend is provided, ``backend.dt`` is used.
        timing_constraints (TimingConstraints): Hardware time alignment restrictions.
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

        initial_layout (Layout | List[int]): Initial position of virtual qubits on
            physical qubits.
        layout_method (str): The :class:`~.Pass` to use for choosing initial qubit
            placement. Valid choices are ``'trivial'``, ``'dense'``,
            and ``'sabre'``, representing :class:`~.TrivialLayout`, :class:`~.DenseLayout` and
            :class:`~.SabreLayout` respectively. This can also
            be the external plugin name to use for the ``layout`` stage of the output
            :class:`~.StagedPassManager`. You can see a list of installed plugins by using
            :func:`~.list_stage_plugins` with ``"layout"`` for the ``stage_name`` argument.
        routing_method (str): The pass to use for routing qubits on the
            architecture. Valid choices are ``'basic'``, ``'lookahead'``, ``'stochastic'``,
            ``'sabre'``, and ``'none'`` representing :class:`~.BasicSwap`,
            :class:`~.LookaheadSwap`, :class:`~.StochasticSwap`, :class:`~.SabreSwap`, and
            erroring if routing is required respectively. This can also be the external plugin
            name to use for the ``routing`` stage of the output :class:`~.StagedPassManager`.
            You can see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"routing"`` for the ``stage_name`` argument.
        translation_method (str): The method to use for translating gates to
            basis gates. Valid choices ``'translator'``, ``'synthesis'`` representing
            :class:`~.BasisTranslator`, and :class:`~.UnitarySynthesis` respectively. This can
            also be the external plugin name to use for the ``translation`` stage of the output
            :class:`~.StagedPassManager`. You can see a list of installed plugins by using
            :func:`~.list_stage_plugins` with ``"translation"`` for the ``stage_name`` argument.
        scheduling_method (str): The pass to use for scheduling instructions. Valid choices
            are ``'alap'`` and ``'asap'``. This can also be the external plugin name to use
            for the ``scheduling`` stage of the output :class:`~.StagedPassManager`. You can
            see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"scheduling"`` for the ``stage_name`` argument.
        backend_properties (BackendProperties): Properties returned by a
            backend, including information on gate errors, readout errors,
            qubit coherence times, etc.
        approximation_degree (float): Heuristic dial used for circuit approximation
            (1.0=no approximation, 0.0=maximal approximation).
        seed_transpiler (int): Sets random seed for the stochastic parts of
            the transpiler.
        unitary_synthesis_method (str): The name of the unitary synthesis
            method to use. By default ``'default'`` is used. You can see a list of
            installed plugins with :func:`.unitary_synthesis_plugin_names`.
        unitary_synthesis_plugin_config (dict): An optional configuration dictionary
            that will be passed directly to the unitary synthesis plugin. By
            default this setting will have no effect as the default unitary
            synthesis method does not take custom configuration. This should
            only be necessary when a unitary synthesis plugin is specified with
            the ``unitary_synthesis_method`` argument. As this is custom for each
            unitary synthesis plugin refer to the plugin documentation for how
            to use this option.
        hls_config (HLSConfig): An optional configuration class :class:`~.HLSConfig`
            that will be passed directly to :class:`~.HighLevelSynthesis` transformation pass.
            This configuration class allows to specify for various high-level objects
            the lists of synthesis algorithms and their parameters.
        init_method (str): The plugin name to use for the ``init`` stage of
            the output :class:`~.StagedPassManager`. By default an external
            plugin is not used. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"init"`` for the stage
            name argument.
        optimization_method (str): The plugin name to use for the
            ``optimization`` stage of the output
            :class:`~.StagedPassManager`. By default an external
            plugin is not used. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"optimization"`` for the
            ``stage_name`` argument.
        qubits_initially_zero (bool): Indicates whether the input circuit is
                zero-initialized.

    Returns:
        StagedPassManager: The preset pass manager for the given options

    Raises:
        ValueError: if an invalid value for ``optimization_level`` is passed in.
    """

    # Handle positional arguments for target and backend. This enables the usage
    # pattern `generate_preset_pass_manager(backend.target)` to generate a default
    # pass manager for a given target.
    if isinstance(optimization_level, Target):
        target = optimization_level
        optimization_level = 2
    elif isinstance(optimization_level, Backend):
        backend = optimization_level
        optimization_level = 2

    if backend is not None and getattr(backend, "version", 0) <= 1:
        # This is a temporary conversion step to allow for a smoother transition
        # to a fully target-based transpiler pipeline while maintaining the behavior
        # of `transpile` with BackendV1 inputs.
        warnings.warn(
            "The `generate_preset_pass_manager` function will stop supporting inputs of "
            f"type `BackendV1` ( {backend} ) in the `backend` parameter in a future "
            "release no earlier than 2.0. `BackendV1` is deprecated and implementations "
            "should move to `BackendV2`.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        backend = BackendV2Converter(backend)

    # Check if a custom inst_map was specified before overwriting inst_map
    _given_inst_map = bool(inst_map)
    # If there are no loose constraints => use backend target if available
    _no_loose_constraints = (
        basis_gates is None
        and coupling_map is None
        and dt is None
        and instruction_durations is None
        and backend_properties is None
        and timing_constraints is None
    )
    # If it's an edge case => do not build target
    _skip_target = (
        target is None
        and backend is None
        # Note: instruction_durations is deprecated and will be removed in 2.0 (no need for alternative)
        and (basis_gates is None or coupling_map is None or instruction_durations is not None)
    )

    # Resolve loose constraints case-by-case against backend constraints.
    # The order of priority is loose constraints > backend.
    dt = _parse_dt(dt, backend)
    instruction_durations = _parse_instruction_durations(backend, instruction_durations, dt)
    timing_constraints = _parse_timing_constraints(backend, timing_constraints)
    inst_map = _parse_inst_map(inst_map, backend)
    # The basis gates parser will set _skip_target to True if a custom basis gate is found
    # (known edge case).
    basis_gates, name_mapping, _skip_target = _parse_basis_gates(
        basis_gates, backend, inst_map, _skip_target
    )
    coupling_map = _parse_coupling_map(coupling_map, backend)

    if target is None:
        if backend is not None and _no_loose_constraints:
            # If a backend is specified without loose constraints, use its target directly.
            target = backend.target
        elif not _skip_target:
            # Only parse backend properties when the target isn't skipped to
            # preserve the former behavior of transpile.
            backend_properties = _parse_backend_properties(backend_properties, backend)
            with warnings.catch_warnings():
                # TODO: inst_map will be removed in 2.0
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    message=".*``inst_map`` is deprecated as of Qiskit 1.3.*",
                    module="qiskit",
                )
                # Build target from constraints.
                target = Target.from_configuration(
                    basis_gates=basis_gates,
                    num_qubits=backend.num_qubits if backend is not None else None,
                    coupling_map=coupling_map,
                    # If the instruction map has custom gates, do not give as config, the information
                    # will be added to the target with update_from_instruction_schedule_map
                    inst_map=inst_map if inst_map and not inst_map.has_custom_gate() else None,
                    backend_properties=backend_properties,
                    instruction_durations=instruction_durations,
                    concurrent_measurements=(
                        backend.target.concurrent_measurements if backend is not None else None
                    ),
                    dt=dt,
                    timing_constraints=timing_constraints,
                    custom_name_mapping=name_mapping,
                )

    # Update target with custom gate information. Note that this is an exception to the priority
    # order (target > loose constraints), added to handle custom gates for scheduling passes.
    if target is not None and _given_inst_map and inst_map.has_custom_gate():
        target = copy.deepcopy(target)
        target.update_from_instruction_schedule_map(inst_map)

    if target is not None:
        if coupling_map is None:
            coupling_map = target.build_coupling_map()
        if basis_gates is None:
            basis_gates = target.operation_names
        if instruction_durations is None:
            instruction_durations = target.durations()
        if inst_map is None:
            inst_map = target._get_instruction_schedule_map()
        if timing_constraints is None:
            timing_constraints = target.timing_constraints()
        if backend_properties is None:
            with warnings.catch_warnings():
                # TODO this approach (target-to-properties) is going to be removed soon (1.3) in favor
                #   of backend-to-target approach
                #   https://github.com/Qiskit/qiskit/pull/12850
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    message=r".+qiskit\.transpiler\.target\.target_to_backend_properties.+",
                    module="qiskit",
                )
                backend_properties = target_to_backend_properties(target)

    # Parse non-target dependent pm options
    initial_layout = _parse_initial_layout(initial_layout)
    approximation_degree = _parse_approximation_degree(approximation_degree)
    seed_transpiler = _parse_seed_transpiler(seed_transpiler)

    pm_options = {
        "target": target,
        "basis_gates": basis_gates,
        "inst_map": inst_map,
        "coupling_map": coupling_map,
        "instruction_durations": instruction_durations,
        "backend_properties": backend_properties,
        "timing_constraints": timing_constraints,
        "layout_method": layout_method,
        "routing_method": routing_method,
        "translation_method": translation_method,
        "scheduling_method": scheduling_method,
        "approximation_degree": approximation_degree,
        "seed_transpiler": seed_transpiler,
        "unitary_synthesis_method": unitary_synthesis_method,
        "unitary_synthesis_plugin_config": unitary_synthesis_plugin_config,
        "initial_layout": initial_layout,
        "hls_config": hls_config,
        "init_method": init_method,
        "optimization_method": optimization_method,
        "qubits_initially_zero": qubits_initially_zero,
    }

    with warnings.catch_warnings():
        # inst_map is deprecated in the PassManagerConfig initializer
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*argument ``inst_map`` is deprecated as of Qiskit 1.3",
        )
        if backend is not None:
            pm_options["_skip_target"] = _skip_target
            pm_config = PassManagerConfig.from_backend(backend, **pm_options)
        else:
            pm_config = PassManagerConfig(**pm_options)
        if optimization_level == 0:
            pm = level_0_pass_manager(pm_config)
        elif optimization_level == 1:
            pm = level_1_pass_manager(pm_config)
        elif optimization_level == 2:
            pm = level_2_pass_manager(pm_config)
        elif optimization_level == 3:
            pm = level_3_pass_manager(pm_config)
        else:
            raise ValueError(f"Invalid optimization level {optimization_level}")
    return pm


def _parse_basis_gates(basis_gates, backend, inst_map, skip_target):
    standard_gates = get_standard_gate_name_mapping()
    # Add control flow gates by default to basis set and name mapping
    default_gates = {"measure", "delay", "reset"}.union(CONTROL_FLOW_OP_NAMES)
    name_mapping = {
        "if_else": IfElseOp,
        "while_loop": WhileLoopOp,
        "for_loop": ForLoopOp,
        "switch_case": SwitchCaseOp,
    }
    try:
        instructions = set(basis_gates)
        for name in default_gates:
            if name not in instructions:
                instructions.add(name)
    except TypeError:
        instructions = None

    if backend is None:
        # Check for custom instructions
        if instructions is None:
            return None, name_mapping, skip_target

        for inst in instructions:
            if inst not in standard_gates and inst not in default_gates:
                warnings.warn(
                    category=DeprecationWarning,
                    message=f"Providing non-standard gates ({inst}) through the ``basis_gates`` "
                    "argument is deprecated for both ``transpile`` and ``generate_preset_pass_manager`` "
                    "as of Qiskit 1.3.0. "
                    "It will be removed in Qiskit 2.0. The ``target`` parameter should be used instead. "
                    "You can build a target instance using ``Target.from_configuration()`` and provide "
                    "custom gate definitions with the ``custom_name_mapping`` argument.",
                )
                skip_target = True
                break

        return list(instructions), name_mapping, skip_target

    instructions = instructions or backend.operation_names
    name_mapping.update(
        {name: backend.target.operation_from_name(name) for name in backend.operation_names}
    )

    # Check for custom instructions before removing calibrations
    for inst in instructions:
        if inst not in standard_gates and inst not in default_gates:
            if inst not in backend.operation_names:
                # do not raise warning when the custom instruction comes from the backend
                # (common case with BasicSimulator)
                warnings.warn(
                    category=DeprecationWarning,
                    message="Providing custom gates through the ``basis_gates`` argument is deprecated "
                    "for both ``transpile`` and ``generate_preset_pass_manager`` as of Qiskit 1.3.0. "
                    "It will be removed in Qiskit 2.0. The ``target`` parameter should be used instead. "
                    "You can build a target instance using ``Target.from_configuration()`` and provide"
                    "custom gate definitions with the ``custom_name_mapping`` argument.",
                )
            skip_target = True
            break

    # Remove calibrated instructions, as they will be added later from the instruction schedule map
    if inst_map is not None and not skip_target:
        for inst in inst_map.instructions:
            for qubit in inst_map.qubits_with_instruction(inst):
                entry = inst_map._get_calibration_entry(inst, qubit)
                if entry.user_provided and inst in instructions:
                    instructions.remove(inst)

    return list(instructions) if instructions else None, name_mapping, skip_target


def _parse_inst_map(inst_map, backend):
    # try getting inst_map from user, else backend
    if inst_map is None and backend is not None:
        inst_map = backend.target._get_instruction_schedule_map()
    return inst_map


def _parse_backend_properties(backend_properties, backend):
    # try getting backend_props from user, else backend
    if backend_properties is None and backend is not None:
        with warnings.catch_warnings():
            # filter target_to_backend_properties warning
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=".*``qiskit.transpiler.target.target_to_backend_properties\\(\\)``.*",
                module="qiskit",
            )
            backend_properties = target_to_backend_properties(backend.target)
    return backend_properties


def _parse_dt(dt, backend):
    # try getting dt from user, else backend
    if dt is None and backend is not None:
        dt = backend.target.dt
    return dt


def _parse_coupling_map(coupling_map, backend):
    # try getting coupling_map from user, else backend
    if coupling_map is None and backend is not None:
        coupling_map = backend.coupling_map

    # coupling_map could be None, or a list of lists, e.g. [[0, 1], [2, 1]]
    if coupling_map is None or isinstance(coupling_map, CouplingMap):
        return coupling_map
    if isinstance(coupling_map, list) and all(
        isinstance(i, list) and len(i) == 2 for i in coupling_map
    ):
        return CouplingMap(coupling_map)
    else:
        raise TranspilerError(
            "Only a single input coupling map can be used with generate_preset_pass_manager()."
        )


def _parse_instruction_durations(backend, inst_durations, dt):
    """Create a list of ``InstructionDuration``s. If ``inst_durations`` is provided,
    the backend will be ignored, otherwise, the durations will be populated from the
    backend.
    """
    final_durations = InstructionDurations()
    if not inst_durations:
        backend_durations = InstructionDurations()
        if backend is not None:
            backend_durations = backend.instruction_durations
        final_durations.update(backend_durations, dt or backend_durations.dt)
    else:
        final_durations.update(inst_durations, dt or getattr(inst_durations, "dt", None))
    return final_durations


def _parse_timing_constraints(backend, timing_constraints):
    if isinstance(timing_constraints, TimingConstraints):
        return timing_constraints
    if backend is None and timing_constraints is None:
        timing_constraints = TimingConstraints()
    elif backend is not None:
        timing_constraints = backend.target.timing_constraints()
    return timing_constraints


def _parse_initial_layout(initial_layout):
    # initial_layout could be None, or a list of ints, e.g. [0, 5, 14]
    # or a list of tuples/None e.g. [qr[0], None, qr[1]] or a dict e.g. {qr[0]: 0}
    if initial_layout is None or isinstance(initial_layout, Layout):
        return initial_layout
    if isinstance(initial_layout, dict):
        return Layout(initial_layout)
    initial_layout = list(initial_layout)
    if all(phys is None or isinstance(phys, Qubit) for phys in initial_layout):
        return Layout.from_qubit_list(initial_layout)
    return initial_layout


def _parse_approximation_degree(approximation_degree):
    if approximation_degree is None:
        return None
    if approximation_degree < 0.0 or approximation_degree > 1.0:
        raise TranspilerError("Approximation degree must be in [0.0, 1.0]")
    return approximation_degree


def _parse_seed_transpiler(seed_transpiler):
    if seed_transpiler is None:
        return None
    if not isinstance(seed_transpiler, int) or seed_transpiler < 0:
        raise ValueError("Expected non-negative integer as seed for transpiler.")
    return seed_transpiler

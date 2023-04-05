# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pass manager for optimization level 0, providing no explicit optimization.

Level 0 pass manager: no explicit optimization other than mapping to backend.
"""

from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager import StagedPassManager

from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePluginManager,
)


def level_0_pass_manager(pass_manager_config: PassManagerConfig) -> StagedPassManager:
    """Level 0 pass manager: no explicit optimization other than mapping to backend.

    This pass manager applies the user-given initial layout. If none is given, a trivial
    layout consisting of mapping the i-th virtual qubit to the i-th physical qubit is used.
    Any unused physical qubit is allocated as ancilla space.

    The pass manager then unrolls the circuit to the desired basis, and transforms the
    circuit to match the coupling map.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 0 pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    plugin_manager = PassManagerStagePluginManager()
    basis_gates = pass_manager_config.basis_gates
    inst_map = pass_manager_config.inst_map
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    init_method = pass_manager_config.init_method
    layout_method = pass_manager_config.layout_method or "trivial"
    routing_method = pass_manager_config.routing_method or "stochastic"
    translation_method = pass_manager_config.translation_method or "translator"
    optimization_method = pass_manager_config.optimization_method
    scheduling_method = pass_manager_config.scheduling_method
    instruction_durations = pass_manager_config.instruction_durations
    seed_transpiler = pass_manager_config.seed_transpiler
    backend_properties = pass_manager_config.backend_properties
    approximation_degree = pass_manager_config.approximation_degree
    timing_constraints = pass_manager_config.timing_constraints or TimingConstraints()
    unitary_synthesis_method = pass_manager_config.unitary_synthesis_method
    unitary_synthesis_plugin_config = pass_manager_config.unitary_synthesis_plugin_config
    target = pass_manager_config.target
    hls_config = pass_manager_config.hls_config

    # Choose an initial layout if not set by user (default: trivial layout)
    _given_layout = SetLayout(initial_layout)

    def _choose_layout_condition(property_set):
        return not property_set["layout"]

    if target is None:
        coupling_map_layout = coupling_map
    else:
        coupling_map_layout = target

    if layout_method == "trivial":
        _choose_layout = TrivialLayout(coupling_map_layout)
    elif layout_method == "dense":
        _choose_layout = DenseLayout(coupling_map, backend_properties, target=target)
    elif layout_method == "noise_adaptive":
        if backend_properties is None:
            _choose_layout = NoiseAdaptiveLayout(target)
        else:
            _choose_layout = NoiseAdaptiveLayout(backend_properties)
    elif layout_method == "sabre":
        skip_routing = pass_manager_config.routing_method is not None and routing_method != "sabre"
        _choose_layout = SabreLayout(
            coupling_map_layout,
            max_iterations=1,
            seed=seed_transpiler,
            swap_trials=5,
            layout_trials=5,
            skip_routing=skip_routing,
        )

    # Choose routing pass
    routing_pm = plugin_manager.get_passmanager_stage(
        "routing", routing_method, pass_manager_config, optimization_level=0
    )

    unroll_3q = None
    # Build pass manager
    if coupling_map or initial_layout:
        unroll_3q = common.generate_unroll_3q(
            target,
            basis_gates,
            approximation_degree,
            unitary_synthesis_method,
            unitary_synthesis_plugin_config,
            hls_config,
        )
        if layout_method not in {"trivial", "dense", "noise_adaptive", "sabre"}:
            layout = plugin_manager.get_passmanager_stage(
                "layout", layout_method, pass_manager_config, optimization_level=0
            )
        else:

            def _swap_mapped(property_set):
                return property_set["final_layout"] is None

            layout = PassManager()
            layout.append(_given_layout)
            layout.append(_choose_layout, condition=_choose_layout_condition)
            embed = common.generate_embed_passmanager(coupling_map, target)
            layout.append(
                [pass_ for x in embed.passes() for pass_ in x["passes"]], condition=_swap_mapped
            )
        routing = routing_pm
    else:
        layout = None
        routing = None
    if translation_method not in {"translator", "synthesis", "unroller"}:
        translation = plugin_manager.get_passmanager_stage(
            "translation", translation_method, pass_manager_config, optimization_level=0
        )
    else:
        translation = common.generate_translation_passmanager(
            target,
            basis_gates,
            translation_method,
            approximation_degree,
            coupling_map,
            backend_properties,
            unitary_synthesis_method,
            unitary_synthesis_plugin_config,
            hls_config,
        )

    if (coupling_map and not coupling_map.is_symmetric) or (
        target is not None and target.get_non_global_operation_names(strict_direction=True)
    ):
        pre_opt = common.generate_pre_op_passmanager(target, coupling_map)
        pre_opt += translation
    else:
        pre_opt = None
    if scheduling_method is None or scheduling_method in {"alap", "asap"}:
        sched = common.generate_scheduling(
            instruction_durations, scheduling_method, timing_constraints, inst_map, target=target
        )
    else:
        sched = plugin_manager.get_passmanager_stage(
            "scheduling", scheduling_method, pass_manager_config, optimization_level=0
        )
    init = common.generate_control_flow_options_check(
        layout_method=layout_method,
        routing_method=routing_method,
        translation_method=translation_method,
        optimization_method=optimization_method,
        scheduling_method=scheduling_method,
    )
    if init_method is not None:
        init += plugin_manager.get_passmanager_stage(
            "init", init_method, pass_manager_config, optimization_level=0
        )
    elif unroll_3q is not None:
        init += unroll_3q
    optimization = None
    if optimization_method is not None:
        optimization = plugin_manager.get_passmanager_stage(
            "optimization", optimization_method, pass_manager_config, optimization_level=0
        )

    return StagedPassManager(
        init=init,
        layout=layout,
        routing=routing,
        translation=translation,
        pre_optimization=pre_opt,
        optimization=optimization,
        scheduling=sched,
    )

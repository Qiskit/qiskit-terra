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

"""Built-in transpiler stage plugins for preset pass managers."""

import os
import warnings

from qiskit.transpiler.passes.optimization.split_2q_unitaries import Split2QUnitaries
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import Error
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import VF2Layout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import ElidePermutations
from qiskit.transpiler.passes import RemoveDiagonalGatesBeforeMeasure
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePlugin,
    PassManagerStagePluginManager,
)
from qiskit.transpiler.passes.optimization import (
    Optimize1qGatesDecomposition,
    CommutativeCancellation,
    ConsolidateBlocks,
    InverseCancellation,
    RemoveIdentityEquivalent,
)
from qiskit.transpiler.passes import Depth, Size, FixedPoint, MinimumPoint
from qiskit.transpiler.passes.utils.gates_basis import GatesInBasis
from qiskit.transpiler.passes.synthesis.unitary_synthesis import UnitarySynthesis
from qiskit.passmanager.flow_controllers import ConditionalController, DoWhileController
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.circuit.library.standard_gates import (
    CXGate,
    ECRGate,
    CZGate,
    XGate,
    YGate,
    ZGate,
    TGate,
    TdgGate,
    SwapGate,
    SGate,
    SdgGate,
    HGate,
    CYGate,
    SXGate,
    SXdgGate,
)
from qiskit.utils.parallel import CPU_COUNT
from qiskit import user_config

CONFIG = user_config.get_config()

_discrete_skipped_ops = {
    "delay",
    "reset",
    "measure",
    "switch_case",
    "if_else",
    "for_loop",
    "while_loop",
}


class DefaultInitPassManager(PassManagerStagePlugin):
    """Plugin class for default init stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        if optimization_level == 0:
            init = None
            if (
                pass_manager_config.initial_layout
                or pass_manager_config.coupling_map
                or (
                    pass_manager_config.target is not None
                    and pass_manager_config.target.build_coupling_map() is not None
                )
            ):
                init = common.generate_unroll_3q(
                    pass_manager_config.target,
                    pass_manager_config.basis_gates,
                    pass_manager_config.approximation_degree,
                    pass_manager_config.unitary_synthesis_method,
                    pass_manager_config.unitary_synthesis_plugin_config,
                    pass_manager_config.hls_config,
                    pass_manager_config.qubits_initially_zero,
                )
        elif optimization_level == 1:
            init = PassManager()
            if (
                pass_manager_config.initial_layout
                or pass_manager_config.coupling_map
                or (
                    pass_manager_config.target is not None
                    and pass_manager_config.target.build_coupling_map() is not None
                )
            ):
                init += common.generate_unroll_3q(
                    pass_manager_config.target,
                    pass_manager_config.basis_gates,
                    pass_manager_config.approximation_degree,
                    pass_manager_config.unitary_synthesis_method,
                    pass_manager_config.unitary_synthesis_plugin_config,
                    pass_manager_config.hls_config,
                    pass_manager_config.qubits_initially_zero,
                )
            init.append(
                InverseCancellation(
                    [
                        CXGate(),
                        ECRGate(),
                        CZGate(),
                        CYGate(),
                        XGate(),
                        YGate(),
                        ZGate(),
                        HGate(),
                        SwapGate(),
                        (TGate(), TdgGate()),
                        (SGate(), SdgGate()),
                        (SXGate(), SXdgGate()),
                    ]
                )
            )

        elif optimization_level in {2, 3}:
            init = common.generate_unroll_3q(
                pass_manager_config.target,
                pass_manager_config.basis_gates,
                pass_manager_config.approximation_degree,
                pass_manager_config.unitary_synthesis_method,
                pass_manager_config.unitary_synthesis_plugin_config,
                pass_manager_config.hls_config,
                pass_manager_config.qubits_initially_zero,
            )
            if pass_manager_config.routing_method != "none":
                init.append(ElidePermutations())
            init.append(RemoveDiagonalGatesBeforeMeasure())
            # Target not set on RemoveIdentityEquivalent because we haven't applied a Layout
            # yet so doing anything relative to an error rate in the target is not valid.
            init.append(
                RemoveIdentityEquivalent(
                    approximation_degree=pass_manager_config.approximation_degree
                )
            )
            init.append(
                InverseCancellation(
                    [
                        CXGate(),
                        ECRGate(),
                        CZGate(),
                        CYGate(),
                        XGate(),
                        YGate(),
                        ZGate(),
                        HGate(),
                        SwapGate(),
                        (TGate(), TdgGate()),
                        (SGate(), SdgGate()),
                        (SXGate(), SXdgGate()),
                    ]
                )
            )
            init.append(CommutativeCancellation())
            init.append(ConsolidateBlocks())
            # If approximation degree is None that indicates a request to approximate up to the
            # error rates in the target. However, in the init stage we don't yet know the target
            # qubits being used to figure out the fidelity so just use the default fidelity parameter
            # in this case.
            if pass_manager_config.approximation_degree is not None:
                init.append(Split2QUnitaries(pass_manager_config.approximation_degree))
            else:
                init.append(Split2QUnitaries())
        else:
            raise TranspilerError(f"Invalid optimization level {optimization_level}")
        return init


class BasisTranslatorPassManager(PassManagerStagePlugin):
    """Plugin class for translation stage with :class:`~.BasisTranslator`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        return common.generate_translation_passmanager(
            pass_manager_config.target,
            basis_gates=pass_manager_config.basis_gates,
            method="translator",
            approximation_degree=pass_manager_config.approximation_degree,
            coupling_map=pass_manager_config.coupling_map,
            backend_props=pass_manager_config.backend_properties,
            unitary_synthesis_method=pass_manager_config.unitary_synthesis_method,
            unitary_synthesis_plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
            hls_config=pass_manager_config.hls_config,
            qubits_initially_zero=pass_manager_config.qubits_initially_zero,
        )


class UnitarySynthesisPassManager(PassManagerStagePlugin):
    """Plugin class for translation stage with :class:`~.BasisTranslator`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        return common.generate_translation_passmanager(
            pass_manager_config.target,
            basis_gates=pass_manager_config.basis_gates,
            method="synthesis",
            approximation_degree=pass_manager_config.approximation_degree,
            coupling_map=pass_manager_config.coupling_map,
            backend_props=pass_manager_config.backend_properties,
            unitary_synthesis_method=pass_manager_config.unitary_synthesis_method,
            unitary_synthesis_plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
            hls_config=pass_manager_config.hls_config,
            qubits_initially_zero=pass_manager_config.qubits_initially_zero,
        )


class BasicSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.BasicSwap`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        backend_properties = pass_manager_config.backend_properties
        if target is None:
            routing_pass = BasicSwap(coupling_map)
        else:
            routing_pass = BasicSwap(target)

        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
            optimization_level,
            pass_manager_config.layout_method,
            pass_manager_config.initial_layout,
        )
        if optimization_level == 0:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 1:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=-1,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 2:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 3:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        raise TranspilerError(f"Invalid optimization level specified: {optimization_level}")


class StochasticSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.StochasticSwap`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        coupling_map_routing = target
        if coupling_map_routing is None:
            coupling_map_routing = coupling_map
        backend_properties = pass_manager_config.backend_properties
        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
            optimization_level,
            pass_manager_config.layout_method,
            pass_manager_config.initial_layout,
        )
        if optimization_level == 3:
            routing_pass = StochasticSwap(coupling_map_routing, trials=200, seed=seed_transpiler)
        else:
            routing_pass = StochasticSwap(coupling_map_routing, trials=20, seed=seed_transpiler)

        if optimization_level == 0:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 1:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=-1,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        if optimization_level in {2, 3}:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        raise TranspilerError(f"Invalid optimization level specified: {optimization_level}")


class LookaheadSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.LookaheadSwap`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        coupling_map_routing = target
        if coupling_map_routing is None:
            coupling_map_routing = coupling_map
        backend_properties = pass_manager_config.backend_properties
        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
            optimization_level,
            pass_manager_config.layout_method,
            pass_manager_config.initial_layout,
        )
        if optimization_level == 0:
            routing_pass = LookaheadSwap(coupling_map_routing, search_depth=2, search_width=2)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 1:
            routing_pass = LookaheadSwap(coupling_map_routing, search_depth=4, search_width=4)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=-1,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 2:
            routing_pass = LookaheadSwap(coupling_map_routing, search_depth=5, search_width=6)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 3:
            routing_pass = LookaheadSwap(coupling_map_routing, search_depth=5, search_width=6)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        raise TranspilerError(f"Invalid optimization level specified: {optimization_level}")


class SabreSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.SabreSwap`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        coupling_map_routing = target
        if coupling_map_routing is None:
            coupling_map_routing = coupling_map
        backend_properties = pass_manager_config.backend_properties
        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
            optimization_level,
            pass_manager_config.layout_method,
            pass_manager_config.initial_layout,
        )
        if optimization_level == 0:
            trial_count = _get_trial_count(5)
            routing_pass = SabreSwap(
                coupling_map_routing,
                heuristic="basic",
                seed=seed_transpiler,
                trials=trial_count,
            )
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 1:
            trial_count = _get_trial_count(5)
            routing_pass = SabreSwap(
                coupling_map_routing,
                heuristic="decay",
                seed=seed_transpiler,
                trials=trial_count,
            )
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=-1,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 2:
            trial_count = _get_trial_count(20)

            routing_pass = SabreSwap(
                coupling_map_routing,
                heuristic="decay",
                seed=seed_transpiler,
                trials=trial_count,
            )
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 3:
            trial_count = _get_trial_count(20)
            routing_pass = SabreSwap(
                coupling_map_routing,
                heuristic="decay",
                seed=seed_transpiler,
                trials=trial_count,
            )
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=-1,
                use_barrier_before_measurement=True,
            )
        raise TranspilerError(f"Invalid optimization level specified: {optimization_level}")


class NoneRoutingPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with error on routing."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        routing_pass = Error(
            msg="No routing method selected, but circuit is not routed to device. "
            "CheckMap Error: {check_map_msg}",
            action="raise",
        )
        return common.generate_routing_passmanager(
            routing_pass,
            target,
            coupling_map=coupling_map,
            seed_transpiler=-1,
            use_barrier_before_measurement=True,
        )


class OptimizationPassManager(PassManagerStagePlugin):
    """Plugin class for optimization stage"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build pass manager for optimization stage."""
        # Obtain the translation method required for this pass to work
        translation_method = pass_manager_config.translation_method or "translator"
        optimization = PassManager()
        if optimization_level != 0:
            plugin_manager = PassManagerStagePluginManager()
            _depth_check = [Depth(recurse=True), FixedPoint("depth")]
            _size_check = [Size(recurse=True), FixedPoint("size")]
            # Minimum point check for optimization level 3.
            _minimum_point_check = [
                Depth(recurse=True),
                Size(recurse=True),
                MinimumPoint(["depth", "size"], "optimization_loop"),
            ]

            def _opt_control(property_set):
                return (not property_set["depth_fixed_point"]) or (
                    not property_set["size_fixed_point"]
                )

            translation = plugin_manager.get_passmanager_stage(
                "translation",
                translation_method,
                pass_manager_config,
                optimization_level=optimization_level,
            )
            if optimization_level == 1:
                # Steps for optimization level 1
                _opt = [
                    Optimize1qGatesDecomposition(
                        basis=pass_manager_config.basis_gates, target=pass_manager_config.target
                    ),
                    InverseCancellation(
                        [
                            CXGate(),
                            ECRGate(),
                            CZGate(),
                            CYGate(),
                            XGate(),
                            YGate(),
                            ZGate(),
                            HGate(),
                            SwapGate(),
                            (TGate(), TdgGate()),
                            (SGate(), SdgGate()),
                            (SXGate(), SXdgGate()),
                        ]
                    ),
                ]

            elif optimization_level == 2:
                _opt = [
                    RemoveIdentityEquivalent(
                        approximation_degree=pass_manager_config.approximation_degree,
                        target=pass_manager_config.target,
                    ),
                    Optimize1qGatesDecomposition(
                        basis=pass_manager_config.basis_gates, target=pass_manager_config.target
                    ),
                    CommutativeCancellation(target=pass_manager_config.target),
                ]
            elif optimization_level == 3:
                # Steps for optimization level 3
                _opt = [
                    ConsolidateBlocks(
                        basis_gates=pass_manager_config.basis_gates,
                        target=pass_manager_config.target,
                        approximation_degree=pass_manager_config.approximation_degree,
                    ),
                    UnitarySynthesis(
                        pass_manager_config.basis_gates,
                        approximation_degree=pass_manager_config.approximation_degree,
                        coupling_map=pass_manager_config.coupling_map,
                        backend_props=pass_manager_config.backend_properties,
                        method=pass_manager_config.unitary_synthesis_method,
                        plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
                        target=pass_manager_config.target,
                    ),
                    RemoveIdentityEquivalent(
                        approximation_degree=pass_manager_config.approximation_degree,
                        target=pass_manager_config.target,
                    ),
                    Optimize1qGatesDecomposition(
                        basis=pass_manager_config.basis_gates, target=pass_manager_config.target
                    ),
                    CommutativeCancellation(target=pass_manager_config.target),
                ]

                def _opt_control(property_set):
                    return not property_set["optimization_loop_minimum_point"]

            else:
                raise TranspilerError(f"Invalid optimization_level: {optimization_level}")

            unroll = translation.to_flow_controller()

            # Build nested Flow controllers
            def _unroll_condition(property_set):
                return not property_set["all_gates_in_basis"]

            # Check if any gate is not in the basis, and if so, run unroll passes
            _unroll_if_out_of_basis = [
                GatesInBasis(pass_manager_config.basis_gates, target=pass_manager_config.target),
                ConditionalController(unroll, condition=_unroll_condition),
            ]

            if optimization_level == 3:
                optimization.append(_minimum_point_check)
            elif optimization_level == 2:
                optimization.append(
                    [
                        ConsolidateBlocks(
                            basis_gates=pass_manager_config.basis_gates,
                            target=pass_manager_config.target,
                            approximation_degree=pass_manager_config.approximation_degree,
                        ),
                        UnitarySynthesis(
                            pass_manager_config.basis_gates,
                            approximation_degree=pass_manager_config.approximation_degree,
                            coupling_map=pass_manager_config.coupling_map,
                            backend_props=pass_manager_config.backend_properties,
                            method=pass_manager_config.unitary_synthesis_method,
                            plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
                            target=pass_manager_config.target,
                        ),
                    ]
                )
                optimization.append(_depth_check + _size_check)
            else:
                optimization.append(_depth_check + _size_check)
            opt_loop = (
                _opt + _unroll_if_out_of_basis + _minimum_point_check
                if optimization_level == 3
                else _opt + _unroll_if_out_of_basis + _depth_check + _size_check
            )
            optimization.append(DoWhileController(opt_loop, do_while=_opt_control))
            return optimization
        else:
            return None


class AlapSchedulingPassManager(PassManagerStagePlugin):
    """Plugin class for alap scheduling stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build scheduling stage PassManager"""

        instruction_durations = pass_manager_config.instruction_durations
        scheduling_method = pass_manager_config.scheduling_method
        timing_constraints = pass_manager_config.timing_constraints
        inst_map = pass_manager_config.inst_map
        target = pass_manager_config.target

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=DeprecationWarning)
            # Passing `inst_map` to `generate_scheduling` is deprecated in Qiskit 1.3
            # so filtering these warning when building pass managers
            return common.generate_scheduling(
                instruction_durations, scheduling_method, timing_constraints, inst_map, target
            )


class AsapSchedulingPassManager(PassManagerStagePlugin):
    """Plugin class for alap scheduling stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build scheduling stage PassManager"""

        instruction_durations = pass_manager_config.instruction_durations
        scheduling_method = pass_manager_config.scheduling_method
        timing_constraints = pass_manager_config.timing_constraints
        inst_map = pass_manager_config.inst_map
        target = pass_manager_config.target

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=DeprecationWarning)
            # Passing `inst_map` to `generate_scheduling` is deprecated in Qiskit 1.3
            # so filtering these warning when building pass managers
            return common.generate_scheduling(
                instruction_durations, scheduling_method, timing_constraints, inst_map, target
            )


class DefaultSchedulingPassManager(PassManagerStagePlugin):
    """Plugin class for alap scheduling stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build scheduling stage PassManager"""

        instruction_durations = pass_manager_config.instruction_durations
        scheduling_method = None
        timing_constraints = pass_manager_config.timing_constraints or TimingConstraints()
        inst_map = pass_manager_config.inst_map
        target = pass_manager_config.target

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=DeprecationWarning)
            # Passing `inst_map` to `generate_scheduling` is deprecated in Qiskit 1.3
            # so filtering these warning when building pass managers
            return common.generate_scheduling(
                instruction_durations, scheduling_method, timing_constraints, inst_map, target
            )


class DefaultLayoutPassManager(PassManagerStagePlugin):
    """Plugin class for default layout stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        _given_layout = SetLayout(pass_manager_config.initial_layout)

        def _choose_layout_condition(property_set):
            return not property_set["layout"]

        def _layout_not_perfect(property_set):
            """Return ``True`` if the first attempt at layout has been checked and found to be
            imperfect.  In this case, perfection means "does not require any swap routing"."""
            return property_set["is_swap_mapped"] is not None and not property_set["is_swap_mapped"]

        def _vf2_match_not_found(property_set):
            # If a layout hasn't been set by the time we run vf2 layout we need to
            # run layout
            if property_set["layout"] is None:
                return True
            # if VF2 layout stopped for any reason other than solution found we need
            # to run layout since VF2 didn't converge.
            return (
                property_set["VF2Layout_stop_reason"] is not None
                and property_set["VF2Layout_stop_reason"] is not VF2LayoutStopReason.SOLUTION_FOUND
            )

        def _swap_mapped(property_set):
            return property_set["final_layout"] is None

        if pass_manager_config.target is None:
            coupling_map = pass_manager_config.coupling_map
        else:
            coupling_map = pass_manager_config.target

        layout = PassManager()
        layout.append(_given_layout)
        if optimization_level == 0:
            layout.append(
                ConditionalController(
                    TrivialLayout(coupling_map), condition=_choose_layout_condition
                )
            )
            layout += common.generate_embed_passmanager(coupling_map)
            return layout
        elif optimization_level == 1:
            layout.append(
                ConditionalController(
                    [TrivialLayout(coupling_map), CheckMap(coupling_map)],
                    condition=_choose_layout_condition,
                )
            )
            choose_layout_1 = VF2Layout(
                coupling_map=pass_manager_config.coupling_map,
                seed=-1,
                call_limit=int(5e4),  # Set call limit to ~100ms with rustworkx 0.10.2
                properties=pass_manager_config.backend_properties,
                target=pass_manager_config.target,
                max_trials=2500,  # Limits layout scoring to < 600ms on ~400 qubit devices
            )
            layout.append(ConditionalController(choose_layout_1, condition=_layout_not_perfect))

            trial_count = _get_trial_count(5)

            choose_layout_2 = SabreLayout(
                coupling_map,
                max_iterations=2,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method is not None
                and pass_manager_config.routing_method != "sabre",
            )
            layout.append(
                ConditionalController(
                    [
                        BarrierBeforeFinalMeasurements(
                            "qiskit.transpiler.internal.routing.protection.barrier"
                        ),
                        choose_layout_2,
                    ],
                    condition=_vf2_match_not_found,
                )
            )
        elif optimization_level == 2:
            choose_layout_0 = VF2Layout(
                coupling_map=pass_manager_config.coupling_map,
                seed=-1,
                call_limit=int(5e6),  # Set call limit to ~10s with rustworkx 0.10.2
                properties=pass_manager_config.backend_properties,
                target=pass_manager_config.target,
                max_trials=2500,  # Limits layout scoring to < 600ms on ~400 qubit devices
            )
            layout.append(
                ConditionalController(choose_layout_0, condition=_choose_layout_condition)
            )

            trial_count = _get_trial_count(20)

            choose_layout_1 = SabreLayout(
                coupling_map,
                max_iterations=2,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method is not None
                and pass_manager_config.routing_method != "sabre",
            )
            layout.append(
                ConditionalController(
                    [
                        BarrierBeforeFinalMeasurements(
                            "qiskit.transpiler.internal.routing.protection.barrier"
                        ),
                        choose_layout_1,
                    ],
                    condition=_vf2_match_not_found,
                )
            )
        elif optimization_level == 3:
            choose_layout_0 = VF2Layout(
                coupling_map=pass_manager_config.coupling_map,
                seed=-1,
                call_limit=int(3e7),  # Set call limit to ~60s with rustworkx 0.10.2
                properties=pass_manager_config.backend_properties,
                target=pass_manager_config.target,
                max_trials=250000,  # Limits layout scoring to < 60s on ~400 qubit devices
            )
            layout.append(
                ConditionalController(choose_layout_0, condition=_choose_layout_condition)
            )

            trial_count = _get_trial_count(20)

            choose_layout_1 = SabreLayout(
                coupling_map,
                max_iterations=4,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method is not None
                and pass_manager_config.routing_method != "sabre",
            )
            layout.append(
                ConditionalController(
                    [
                        BarrierBeforeFinalMeasurements(
                            "qiskit.transpiler.internal.routing.protection.barrier"
                        ),
                        choose_layout_1,
                    ],
                    condition=_vf2_match_not_found,
                )
            )
        else:
            raise TranspilerError(f"Invalid optimization level: {optimization_level}")

        embed = common.generate_embed_passmanager(coupling_map)
        layout.append(ConditionalController(embed.to_flow_controller(), condition=_swap_mapped))
        return layout


class TrivialLayoutPassManager(PassManagerStagePlugin):
    """Plugin class for trivial layout stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        _given_layout = SetLayout(pass_manager_config.initial_layout)

        def _choose_layout_condition(property_set):
            return not property_set["layout"]

        if pass_manager_config.target is None:
            coupling_map = pass_manager_config.coupling_map
        else:
            coupling_map = pass_manager_config.target

        layout = PassManager()
        layout.append(_given_layout)
        layout.append(
            ConditionalController(TrivialLayout(coupling_map), condition=_choose_layout_condition)
        )
        layout += common.generate_embed_passmanager(coupling_map)
        return layout


class DenseLayoutPassManager(PassManagerStagePlugin):
    """Plugin class for dense layout stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        _given_layout = SetLayout(pass_manager_config.initial_layout)

        def _choose_layout_condition(property_set):
            return not property_set["layout"]

        if pass_manager_config.target is None:
            coupling_map = pass_manager_config.coupling_map
        else:
            coupling_map = pass_manager_config.target

        layout = PassManager()
        layout.append(_given_layout)
        layout.append(
            ConditionalController(
                DenseLayout(
                    coupling_map=pass_manager_config.coupling_map,
                    backend_prop=pass_manager_config.backend_properties,
                    target=pass_manager_config.target,
                ),
                condition=_choose_layout_condition,
            )
        )
        layout += common.generate_embed_passmanager(coupling_map)
        return layout


class SabreLayoutPassManager(PassManagerStagePlugin):
    """Plugin class for sabre layout stage."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        _given_layout = SetLayout(pass_manager_config.initial_layout)

        def _choose_layout_condition(property_set):
            return not property_set["layout"]

        def _swap_mapped(property_set):
            return property_set["final_layout"] is None

        if pass_manager_config.target is None:
            coupling_map = pass_manager_config.coupling_map
        else:
            coupling_map = pass_manager_config.target

        layout = PassManager()
        layout.append(_given_layout)
        if optimization_level == 0:
            trial_count = _get_trial_count(5)

            layout_pass = SabreLayout(
                coupling_map,
                max_iterations=1,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method is not None
                and pass_manager_config.routing_method != "sabre",
            )
        elif optimization_level == 1:
            trial_count = _get_trial_count(5)

            layout_pass = SabreLayout(
                coupling_map,
                max_iterations=2,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method is not None
                and pass_manager_config.routing_method != "sabre",
            )
        elif optimization_level == 2:
            trial_count = _get_trial_count(20)

            layout_pass = SabreLayout(
                coupling_map,
                max_iterations=2,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method is not None
                and pass_manager_config.routing_method != "sabre",
            )
        elif optimization_level == 3:
            trial_count = _get_trial_count(20)

            layout_pass = SabreLayout(
                coupling_map,
                max_iterations=4,
                seed=pass_manager_config.seed_transpiler,
                swap_trials=trial_count,
                layout_trials=trial_count,
                skip_routing=pass_manager_config.routing_method is not None
                and pass_manager_config.routing_method != "sabre",
            )
        else:
            raise TranspilerError(f"Invalid optimization level: {optimization_level}")
        layout.append(
            ConditionalController(
                [
                    BarrierBeforeFinalMeasurements(
                        "qiskit.transpiler.internal.routing.protection.barrier"
                    ),
                    layout_pass,
                ],
                condition=_choose_layout_condition,
            )
        )
        embed = common.generate_embed_passmanager(coupling_map)
        layout.append(ConditionalController(embed.to_flow_controller(), condition=_swap_mapped))
        return layout


def _get_trial_count(default_trials=5):
    if CONFIG.get("sabre_all_threads", None) or os.getenv("QISKIT_SABRE_ALL_THREADS"):
        return max(CPU_COUNT, default_trials)
    return default_trials

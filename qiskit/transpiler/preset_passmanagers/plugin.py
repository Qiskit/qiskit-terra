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

"""
.. _transpiler-preset-stage-plugins:

=======================================================================================
Transpiler Stage Plugin Interface (:mod:`qiskit.transpiler.preset_passmanagers.plugin`)
=======================================================================================

.. currentmodule:: qiskit.transpiler.preset_passmanagers.plugin

This module defines the plugin interface for providing custom stage
implementations for the preset pass managers and the :func:`~.transpile`
function. This enables external Python packages to provide
:class:`~.PassManager` objects that can be used for each named stage.

The plugin interfaces are built using setuptools
`entry points <https://setuptools.readthedocs.io/en/latest/userguide/entry_point.html>`__
which enable packages external to Qiskit to advertise they include a transpiler stage(s).

For details on how to instead write plugins for transpiler synthesis methods,
see :mod:`qiskit.transpiler.passes.synthesis.plugin`.


.. _stage_table:

Plugin Stages
=============

Currently, there are 6 stages in the preset pass managers, all of which actively
load external plugins via corresponding entry points.  The table below gives a quick summary of each
stage.  For more details on the description and expectations of each stage, follow the link in the
stages' names to the full documentation.

.. list-table:: Stages
   :header-rows: 1

   * - Stage Name
     - Entry Point
     - Reserved Names
     - Summary

   * - :ref:`init <transpiler-preset-stage-init>`
     - ``qiskit.transpiler.init``
     - ``default``
     - High-level, logical optimizations on abstract circuits, and reduction of multi-qubit
       operations to 1- and 2-qubit operations.

   * - :ref:`layout <transpiler-preset-stage-layout>`
     - ``qiskit.transpiler.layout``
     - ``trivial``, ``dense``, ``sabre``, ``default``
     - Choose an initial mapping of virtual qubits to physical qubits, including expansion of the
       circuit to include explicit ancillas.  This stage is sometimes combined with ``routing``.

   * - :ref:`routing <transpiler-preset-stage-routing>`
     - ``qiskit.transpiler.routing``
     - ``basic``, ``stochastic``, ``lookahead``, ``sabre``
     - Insert gates into the circuit to ensure it matches the connectivity constraints of the
       :class:`.Target`.  The inserted gates do not need to be in the target ISA yet, so are often
       just output as ``swap`` instructions.  This stage is sometimes subsumed by ``layout``.

   * - :ref:`translation <transpiler-preset-stage-translation>`
     - ``qiskit.transpiler.translation``
     - ``translator``, ``synthesis``, ``unroller``
     - Rewrite all gates outside the target ISA to use only gates within the ISA.

   * - :ref:`optimization <transpiler-preset-stage-optimization>`
     - ``qiskit.transpiler.optimization``
     - ``default``
     - Low-level, physical-circuit-aware optimizations.  Unlike ``init``, the ``optimization`` stage
       acts at the level of a physical circuit.

   * - :ref:`scheduling <transpiler-preset-stage-scheduling>`
     - ``qiskit.transpiler.scheduling``
     - ``alap``, ``asap``, ``default``
     - Insert :class:`~.circuit.Delay` instructions to make the wall-clock timing of the circuit
       fully explicit.


Writing Plugins
===============

To write a pass manager stage plugin there are 2 main steps. The first step is
to create a subclass of the abstract plugin class
:class:`~.PassManagerStagePlugin` which is used to define how the :class:`~.PassManager`
for the stage will be constructed. For example, to create a ``layout`` stage plugin that just
runs :class:`~.VF2Layout` (with increasing amount of trials, depending on the optimization level)
and falls back to using :class:`~.TrivialLayout` if
:class:`~VF2Layout` is unable to find a perfect layout::

    from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
    from qiskit.transpiler.preset_passmanagers import common
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import VF2Layout, TrivialLayout
    from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason


    def _vf2_match_not_found(property_set):
        return property_set["layout"] is None or (
            property_set["VF2Layout_stop_reason"] is not None
            and property_set["VF2Layout_stop_reason"] is not VF2LayoutStopReason.SOLUTION_FOUND


    class VF2LayoutPlugin(PassManagerStagePlugin):

        def pass_manager(self, pass_manager_config, optimization_level):
            layout_pm = PassManager(
                [
                    VF2Layout(
                        coupling_map=pass_manager_config.coupling_map,
                        properties=pass_manager_config.backend_properties,
                        max_trials=optimization_level * 10 + 1
                        target=pass_manager_config.target
                    )
                ]
            )
            layout_pm.append(
                TrivialLayout(pass_manager_config.coupling_map),
                condition=_vf2_match_not_found,
            )
            layout_pm += common.generate_embed_passmanager(pass_manager_config.coupling_map)
            return layout_pm

The second step is to expose the :class:`~.PassManagerStagePlugin`
subclass as a setuptools entry point in the package metadata. This can be done
an ``entry-points`` table in ``pyproject.toml`` for the plugin package with the necessary entry
points under the appropriate namespace for the stage your plugin is for. You can see the list of
stages, entry points, and expectations from the stage in :ref:`stage_table`.  For example,
continuing from the example plugin above:

.. code-block:: toml

    [project.entry-points."qiskit.transpiler.layout"]
    "vf2" = "qiskit_plugin_pkg.module.plugin:VF2LayoutPlugin"

There isn't a limit to the number of plugins a single package can include as long as each plugin has
a unique name. So a single package can expose multiple plugins if necessary. Refer to
:ref:`stage_table` for a list of reserved names for each stage.

Plugin API
==========

.. autosummary::
   :toctree: ../stubs/

   PassManagerStagePlugin
   PassManagerStagePluginManager

.. autofunction:: list_stage_plugins
.. autofunction:: passmanager_stage_plugins
"""

import abc
from typing import List, Optional, Dict

import stevedore

from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager_config import PassManagerConfig


class PassManagerStagePlugin(abc.ABC):
    """A ``PassManagerStagePlugin`` is a plugin interface object for using custom
    stages in :func:`~.transpile`.

    A ``PassManagerStagePlugin`` object can be added to an external package and
    integrated into the :func:`~.transpile` function with an entry point. This
    will enable users to use the output of :meth:`.pass_manager` to implement
    a stage in the compilation process.
    """

    @abc.abstractmethod
    def pass_manager(
        self, pass_manager_config: PassManagerConfig, optimization_level: Optional[int] = None
    ) -> PassManager:
        """This method is designed to return a :class:`~.PassManager` for the stage this implements

        Args:
            pass_manager_config: A configuration object that defines all the target device
                specifications and any user specified options to :func:`~.transpile` or
                :func:`~.generate_preset_pass_manager`
            optimization_level: The optimization level of the transpilation, if set this
                should be used to set values for any tunable parameters to trade off runtime
                for potential optimization. Valid values should be ``0``, ``1``, ``2``, or ``3``
                and the higher the number the more optimization is expected.
        """
        pass


class PassManagerStagePluginManager:
    """Manager class for preset pass manager stage plugins."""

    def __init__(self):
        super().__init__()
        self.init_plugins = stevedore.ExtensionManager(
            "qiskit.transpiler.init", invoke_on_load=True, propagate_map_exceptions=True
        )
        self.layout_plugins = stevedore.ExtensionManager(
            "qiskit.transpiler.layout", invoke_on_load=True, propagate_map_exceptions=True
        )
        self.routing_plugins = stevedore.ExtensionManager(
            "qiskit.transpiler.routing", invoke_on_load=True, propagate_map_exceptions=True
        )
        self.translation_plugins = stevedore.ExtensionManager(
            "qiskit.transpiler.translation", invoke_on_load=True, propagate_map_exceptions=True
        )
        self.optimization_plugins = stevedore.ExtensionManager(
            "qiskit.transpiler.optimization", invoke_on_load=True, propagate_map_exceptions=True
        )
        self.scheduling_plugins = stevedore.ExtensionManager(
            "qiskit.transpiler.scheduling", invoke_on_load=True, propagate_map_exceptions=True
        )

    def get_passmanager_stage(
        self,
        stage_name: str,
        plugin_name: str,
        pm_config: PassManagerConfig,
        optimization_level=None,
    ) -> PassManager:
        """Get a stage"""
        if stage_name == "init":
            return self._build_pm(
                self.init_plugins, stage_name, plugin_name, pm_config, optimization_level
            )
        elif stage_name == "layout":
            return self._build_pm(
                self.layout_plugins, stage_name, plugin_name, pm_config, optimization_level
            )
        elif stage_name == "routing":
            return self._build_pm(
                self.routing_plugins, stage_name, plugin_name, pm_config, optimization_level
            )
        elif stage_name == "translation":
            return self._build_pm(
                self.translation_plugins, stage_name, plugin_name, pm_config, optimization_level
            )
        elif stage_name == "optimization":
            return self._build_pm(
                self.optimization_plugins, stage_name, plugin_name, pm_config, optimization_level
            )
        elif stage_name == "scheduling":
            return self._build_pm(
                self.scheduling_plugins, stage_name, plugin_name, pm_config, optimization_level
            )
        else:
            raise TranspilerError(f"Invalid stage name: {stage_name}")

    def _build_pm(
        self,
        stage_obj: stevedore.ExtensionManager,
        stage_name: str,
        plugin_name: str,
        pm_config: PassManagerConfig,
        optimization_level: Optional[int] = None,
    ):
        if plugin_name not in stage_obj:
            raise TranspilerError(f"Invalid plugin name {plugin_name} for stage {stage_name}")
        plugin_obj = stage_obj[plugin_name]
        return plugin_obj.obj.pass_manager(pm_config, optimization_level)


def list_stage_plugins(stage_name: str) -> List[str]:
    """Get a list of installed plugins for a stage.

    Args:
        stage_name: The stage name to get the plugin names for

    Returns:
        plugins: The list of installed plugin names for the specified stages

    Raises:
       TranspilerError: If an invalid stage name is specified.
    """
    plugin_mgr = PassManagerStagePluginManager()
    if stage_name == "init":
        return plugin_mgr.init_plugins.names()
    elif stage_name == "layout":
        return plugin_mgr.layout_plugins.names()
    elif stage_name == "routing":
        return plugin_mgr.routing_plugins.names()
    elif stage_name == "translation":
        return plugin_mgr.translation_plugins.names()
    elif stage_name == "optimization":
        return plugin_mgr.optimization_plugins.names()
    elif stage_name == "scheduling":
        return plugin_mgr.scheduling_plugins.names()
    else:
        raise TranspilerError(f"Invalid stage name: {stage_name}")


def passmanager_stage_plugins(stage: str) -> Dict[str, PassManagerStagePlugin]:
    """Return a dict with, for each stage name, the class type of the plugin.

    This function is useful for getting more information about a plugin:

    .. plot::
       :include-source:
       :nofigs:

        from qiskit.transpiler.preset_passmanagers.plugin import passmanager_stage_plugins
        routing_plugins = passmanager_stage_plugins('routing')
        basic_plugin = routing_plugins['basic']
        help(basic_plugin)

    .. code-block:: text

        Help on BasicSwapPassManager in module ...preset_passmanagers.builtin_plugins object:

        class BasicSwapPassManager(...preset_passmanagers.plugin.PassManagerStagePlugin)
         |  Plugin class for routing stage with :class:`~.BasicSwap`
         |
         |  Method resolution order:
         |      BasicSwapPassManager
         |      ...preset_passmanagers.plugin.PassManagerStagePlugin
         |      abc.ABC
         |      builtins.object
         ...

    Args:
        stage: The stage name to get

    Returns:
        dict: the key is the name of the plugin and the value is the class type for each.

    Raises:
       TranspilerError: If an invalid stage name is specified.
    """
    plugin_mgr = PassManagerStagePluginManager()
    try:
        manager = getattr(plugin_mgr, f"{stage}_plugins")
    except AttributeError as exc:
        raise TranspilerError(f"Passmanager stage {stage} not found") from exc

    return {name: manager[name].obj for name in manager.names()}

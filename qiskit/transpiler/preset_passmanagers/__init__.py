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

"""
==================================================================
Preset Passmanagers (:mod:`qiskit.transpiler.preset_passmanagers`)
==================================================================

.. currentmodule:: qiskit.transpiler.preset_passmanagers

This module contains functions for generating the preset pass managers
for the transpiler. The preset pass managers are instances of
:class:`~.StagedPassManager` which are used to execute the circuit
transformations as part of Qiskit's compiler inside the
:func:`~.transpile` function at the different optimization levels, but
can also be used in a standalone manner.
The functionality here is divided into two parts. The first includes the
functions used to generate the entire pass manager, which is used by
:func:`~.transpile` (:ref:`preset_pass_manager_generators`), and the
second includes functions that are used to build (either entirely or in
part) the stages that comprise the preset pass managers
(:ref:`stage_generators`).

.. _preset_pass_manager_generators:

Preset Pass Manager Generation
------------------------------

.. autofunction:: generate_preset_pass_manager
.. autofunction:: level_0_pass_manager
.. autofunction:: level_1_pass_manager
.. autofunction:: level_2_pass_manager
.. autofunction:: level_3_pass_manager

.. _stage_generators:

Stage Generator Functions
-------------------------

.. currentmodule:: qiskit.transpiler.preset_passmanagers.common
.. autofunction:: generate_control_flow_options_check
.. autofunction:: generate_error_on_control_flow
.. autofunction:: generate_unroll_3q
.. autofunction:: generate_embed_passmanager
.. autofunction:: generate_routing_passmanager
.. autofunction:: generate_pre_op_passmanager
.. autofunction:: generate_translation_passmanager
.. autofunction:: generate_scheduling
.. currentmodule:: qiskit.transpiler.preset_passmanagers
"""
from .generate_preset_pass_manager import generate_preset_pass_manager
from .level0 import level_0_pass_manager
from .level1 import level_1_pass_manager
from .level2 import level_2_pass_manager
from .level3 import level_3_pass_manager


__all__ = [
    "level_0_pass_manager",
    "level_1_pass_manager",
    "level_2_pass_manager",
    "level_3_pass_manager",
    "generate_preset_pass_manager",
]

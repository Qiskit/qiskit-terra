# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Instruction sub-classes for dynamic circuits."""


from ._builder_utils import condition_resources, node_resources, LegacyResources
from .control_flow import ControlFlowOp
from .continue_loop import ContinueLoopOp
from .break_loop import BreakLoopOp

from .if_else import IfElseOp
from .while_loop import WhileLoopOp
from .for_loop import ForLoopOp
from .switch_case import SwitchCaseOp, CASE_DEFAULT


CONTROL_FLOW_OP_NAMES = frozenset(("for_loop", "while_loop", "if_else", "switch_case"))
"""Set of the instruction names of Qiskit's known control-flow operations."""

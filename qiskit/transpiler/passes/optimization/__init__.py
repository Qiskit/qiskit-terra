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

"""Module containing transpiler optimization passes."""

from .optimize_1q_gates import Optimize1qGates
from .optimize_1q_decomposition import Optimize1qGatesDecomposition
from .collect_2q_blocks import Collect2qBlocks
from .collect_multiqubit_blocks import CollectMultiQBlocks
from .consolidate_blocks import ConsolidateBlocks
from .commutation_analysis import CommutationAnalysis
from .commutative_cancellation import CommutativeCancellation
from .commutative_inverse_cancellation import CommutativeInverseCancellation
from .cx_cancellation import CXCancellation
from .optimize_1q_commutation import Optimize1qGatesSimpleCommutation
from .optimize_swap_before_measure import OptimizeSwapBeforeMeasure
from .remove_reset_in_zero_state import RemoveResetInZeroState
from .remove_final_reset import RemoveFinalReset
from .remove_diagonal_gates_before_measure import RemoveDiagonalGatesBeforeMeasure
from .hoare_opt import HoareOptimizer
from .template_optimization import TemplateOptimization
from .inverse_cancellation import InverseCancellation
from .collect_1q_runs import Collect1qRuns
from .echo_rzx_weyl_decomposition import EchoRZXWeylDecomposition
from .collect_linear_functions import CollectLinearFunctions
from .reset_after_measure_simplification import ResetAfterMeasureSimplification
from .optimize_cliffords import OptimizeCliffords
from .collect_cliffords import CollectCliffords
from .elide_permutations import ElidePermutations
from .normalize_rx_angle import NormalizeRXAngle
from .optimize_annotated import OptimizeAnnotated
from .split_2q_unitaries import Split2QUnitaries
from .collect_and_collapse import CollectAndCollapse

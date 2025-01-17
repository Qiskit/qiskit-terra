# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=wrong-import-position,wrong-import-order

"""Main Qiskit public functionality."""

import importlib.metadata
import importlib.util
import os
import sys
import warnings

try:
    importlib.metadata.version("qiskit-terra")
except importlib.metadata.PackageNotFoundError:
    # All good!
    pass
else:
    # 'qiskit.tools' is present in all 0.x series of Qiskit and not in Qiskit 1.0+.  If a dev has an
    # editable install and switches from 0.x branches to 1.0+ branches, they might have an empty
    # `qiskit/tools` folder in their path, which will appear as a "namespace package" with no valid
    # location.  We catch that case as "not actually having Qiskit 0.x" as a convenience to devs.
    _has_tools = getattr(importlib.util.find_spec("qiskit.tools"), "has_location", False)
    _suppress_error = os.environ.get("QISKIT_SUPPRESS_1_0_IMPORT_ERROR", False) == "1"
    if not _suppress_error and _has_tools:
        raise ImportError(
            "Qiskit is installed in an invalid environment that has both Qiskit >=1.0"
            " and an earlier version."
            " You should create a new virtual environment, and ensure that you do not mix"
            " dependencies between Qiskit <1.0 and >=1.0."
            " Any packages that depend on 'qiskit-terra' are not compatible with Qiskit 1.0 and"
            " will need to be updated."
            " Qiskit unfortunately cannot enforce this requirement during environment resolution."
            " See https://qisk.it/packaging-1-0 for more detail."
        )

from . import _accelerate
import qiskit._numpy_compat

# Globally define compiled submodules. The normal import mechanism will not find compiled submodules
# in _accelerate because it relies on file paths, but PyO3 generates only one shared library file.
# We manually define them on import so people can directly import qiskit._accelerate.* submodules
# and not have to rely on attribute access.  No action needed for top-level extension packages.
sys.modules["qiskit._accelerate.circuit"] = _accelerate.circuit
sys.modules["qiskit._accelerate.circuit_library"] = _accelerate.circuit_library
sys.modules["qiskit._accelerate.basis"] = _accelerate.basis
sys.modules["qiskit._accelerate.basis.basis_translator"] = _accelerate.basis.basis_translator
sys.modules["qiskit._accelerate.converters"] = _accelerate.converters
sys.modules["qiskit._accelerate.basis"] = _accelerate.basis
sys.modules["qiskit._accelerate.basis.basis_translator"] = _accelerate.basis.basis_translator
sys.modules["qiskit._accelerate.dense_layout"] = _accelerate.dense_layout
sys.modules["qiskit._accelerate.equivalence"] = _accelerate.equivalence
sys.modules["qiskit._accelerate.error_map"] = _accelerate.error_map
sys.modules["qiskit._accelerate.gates_in_basis"] = _accelerate.gates_in_basis
sys.modules["qiskit._accelerate.isometry"] = _accelerate.isometry
sys.modules["qiskit._accelerate.uc_gate"] = _accelerate.uc_gate
sys.modules["qiskit._accelerate.euler_one_qubit_decomposer"] = (
    _accelerate.euler_one_qubit_decomposer
)
sys.modules["qiskit._accelerate.nlayout"] = _accelerate.nlayout
sys.modules["qiskit._accelerate.optimize_1q_gates"] = _accelerate.optimize_1q_gates
sys.modules["qiskit._accelerate.pauli_expval"] = _accelerate.pauli_expval
sys.modules["qiskit._accelerate.qasm2"] = _accelerate.qasm2
sys.modules["qiskit._accelerate.qasm3"] = _accelerate.qasm3
sys.modules["qiskit._accelerate.remove_diagonal_gates_before_measure"] = (
    _accelerate.remove_diagonal_gates_before_measure
)
sys.modules["qiskit._accelerate.results"] = _accelerate.results
sys.modules["qiskit._accelerate.sabre"] = _accelerate.sabre
sys.modules["qiskit._accelerate.sampled_exp_val"] = _accelerate.sampled_exp_val
sys.modules["qiskit._accelerate.sparse_observable"] = _accelerate.sparse_observable
sys.modules["qiskit._accelerate.sparse_pauli_op"] = _accelerate.sparse_pauli_op
sys.modules["qiskit._accelerate.star_prerouting"] = _accelerate.star_prerouting
sys.modules["qiskit._accelerate.stochastic_swap"] = _accelerate.stochastic_swap
sys.modules["qiskit._accelerate.elide_permutations"] = _accelerate.elide_permutations
sys.modules["qiskit._accelerate.target"] = _accelerate.target
sys.modules["qiskit._accelerate.two_qubit_decompose"] = _accelerate.two_qubit_decompose
sys.modules["qiskit._accelerate.unitary_synthesis"] = _accelerate.unitary_synthesis
sys.modules["qiskit._accelerate.vf2_layout"] = _accelerate.vf2_layout
sys.modules["qiskit._accelerate.synthesis.permutation"] = _accelerate.synthesis.permutation
sys.modules["qiskit._accelerate.synthesis.linear"] = _accelerate.synthesis.linear
sys.modules["qiskit._accelerate.synthesis.clifford"] = _accelerate.synthesis.clifford
sys.modules["qiskit._accelerate.barrier_before_final_measurement"] = (
    _accelerate.barrier_before_final_measurement
)
sys.modules["qiskit._accelerate.commutation_checker"] = _accelerate.commutation_checker
sys.modules["qiskit._accelerate.commutation_analysis"] = _accelerate.commutation_analysis
sys.modules["qiskit._accelerate.commutation_cancellation"] = _accelerate.commutation_cancellation
sys.modules["qiskit._accelerate.consolidate_blocks"] = _accelerate.consolidate_blocks
sys.modules["qiskit._accelerate.synthesis.linear_phase"] = _accelerate.synthesis.linear_phase
sys.modules["qiskit._accelerate.synthesis.evolution"] = _accelerate.synthesis.evolution
sys.modules["qiskit._accelerate.synthesis.multi_controlled"] = (
    _accelerate.synthesis.multi_controlled
)
sys.modules["qiskit._accelerate.split_2q_unitaries"] = _accelerate.split_2q_unitaries
sys.modules["qiskit._accelerate.gate_direction"] = _accelerate.gate_direction
sys.modules["qiskit._accelerate.inverse_cancellation"] = _accelerate.inverse_cancellation
sys.modules["qiskit._accelerate.check_map"] = _accelerate.check_map
sys.modules["qiskit._accelerate.filter_op_nodes"] = _accelerate.filter_op_nodes
sys.modules["qiskit._accelerate.twirling"] = _accelerate.twirling
sys.modules["qiskit._accelerate.high_level_synthesis"] = _accelerate.high_level_synthesis
sys.modules["qiskit._accelerate.remove_identity_equiv"] = _accelerate.remove_identity_equiv

from qiskit.exceptions import QiskitError, MissingOptionalLibraryError

# The main qiskit operators
from qiskit.circuit import ClassicalRegister
from qiskit.circuit import QuantumRegister
from qiskit.circuit import AncillaRegister
from qiskit.circuit import QuantumCircuit

# user config
from qiskit import user_config as _user_config

import qiskit.circuit.measure
import qiskit.circuit.reset

_config = _user_config.get_config()

from qiskit.compiler import transpile, assemble, schedule, sequence
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from .version import __version__

__all__ = [
    "AncillaRegister",
    "ClassicalRegister",
    "MissingOptionalLibraryError",
    "QiskitError",
    "QuantumCircuit",
    "QuantumRegister",
    "assemble",
    "schedule",
    "sequence",
    "transpile",
    "generate_preset_pass_manager",
]

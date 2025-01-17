# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""

High Level Synthesis Plugins
-----------------------------

Clifford Synthesis
''''''''''''''''''

.. list-table:: Plugins for :class:`qiskit.quantum_info.Clifford` (key = ``"clifford"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Targeted connectivity
      - Description
    * - ``"ag"``
      - :class:`~.AGSynthesisClifford`
      - all-to-all
      - greedily optimizes CX-count
    * - ``"bm"``
      - :class:`~.BMSynthesisClifford`
      - all-to-all
      - optimal count for `n=2,3`; used in ``"default"`` for `n=2,3`
    * - ``"greedy"``
      - :class:`~.GreedySynthesisClifford`
      - all-to-all
      - greedily optimizes CX-count; used in ``"default"`` for `n>=4`
    * - ``"layers"``
      - :class:`~.LayerSynthesisClifford`
      - all-to-all
      -
    * - ``"lnn"``
      - :class:`~.LayerLnnSynthesisClifford`
      - linear
      - many CX-gates but guarantees CX-depth of at most `7*n+2`
    * - ``"default"``
      - :class:`~.DefaultSynthesisClifford`
      - all-to-all
      - usually best for optimizing CX-count (and optimal CX-count for `n=2,3`)

.. autosummary::
   :toctree: ../stubs/

   AGSynthesisClifford
   BMSynthesisClifford
   GreedySynthesisClifford
   LayerSynthesisClifford
   LayerLnnSynthesisClifford
   DefaultSynthesisClifford


Linear Function Synthesis
'''''''''''''''''''''''''

.. list-table:: Plugins for :class:`.LinearFunction` (key = ``"linear"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Targeted connectivity
      - Description
    * - ``"kms"``
      - :class:`~.KMSSynthesisLinearFunction`
      - linear
      - many CX-gates but guarantees CX-depth of at most `5*n`
    * - ``"pmh"``
      - :class:`~.PMHSynthesisLinearFunction`
      - all-to-all
      - greedily optimizes CX-count; used in ``"default"``
    * - ``"default"``
      - :class:`~.DefaultSynthesisLinearFunction`
      - all-to-all
      - best for optimizing CX-count

.. autosummary::
   :toctree: ../stubs/

   KMSSynthesisLinearFunction
   PMHSynthesisLinearFunction
   DefaultSynthesisLinearFunction


Permutation Synthesis
'''''''''''''''''''''

.. list-table:: Plugins for :class:`.PermutationGate` (key = ``"permutation"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Targeted connectivity
      - Description
    * - ``"basic"``
      - :class:`~.BasicSynthesisPermutation`
      - all-to-all
      - optimal SWAP-count; used in ``"default"``
    * - ``"acg"``
      - :class:`~.ACGSynthesisPermutation`
      - all-to-all
      - guarantees SWAP-depth of at most `2`
    * - ``"kms"``
      - :class:`~.KMSSynthesisPermutation`
      - linear
      - many SWAP-gates, but guarantees SWAP-depth of at most `n`
    * - ``"token_swapper"``
      - :class:`~.TokenSwapperSynthesisPermutation`
      - any
      - greedily optimizes SWAP-count for arbitrary connectivity
    * - ``"default"``
      - :class:`~.BasicSynthesisPermutation`
      - all-to-all
      - best for optimizing SWAP-count

.. autosummary::
   :toctree: ../stubs/

   BasicSynthesisPermutation
   ACGSynthesisPermutation
   KMSSynthesisPermutation
   TokenSwapperSynthesisPermutation


QFT Synthesis
'''''''''''''

.. list-table:: Plugins for :class:`.QFTGate` (key = ``"qft"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Targeted connectivity
    * - ``"full"``
      - :class:`~.QFTSynthesisFull`
      - all-to-all
    * - ``"line"``
      - :class:`~.QFTSynthesisLine`
      - linear
    * - ``"default"``
      - :class:`~.QFTSynthesisFull`
      - all-to-all

.. autosummary::
   :toctree: ../stubs/

   QFTSynthesisFull
   QFTSynthesisLine


MCX Synthesis
'''''''''''''

The following table lists synthesis plugins available for an :class:`.MCXGate` gate
with `k` control qubits. If the available number of clean/dirty auxiliary qubits is
not sufficient, the corresponding synthesis method will return `None`.

.. list-table:: Plugins for :class:`.MCXGate` (key = ``"mcx"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Number of clean ancillas
      - Number of dirty ancillas
      - Description
    * - ``"gray_code"``
      - :class:`~.MCXSynthesisGrayCode`
      - `0`
      - `0`
      - exponentially many CX gates; use only for small values of `k`
    * - ``"noaux_v24"``
      - :class:`~.MCXSynthesisNoAuxV24`
      - `0`
      - `0`
      - quadratic number of CX gates; use instead of ``"gray_code"`` for large values of `k`
    * - ``"n_clean_m15"``
      - :class:`~.MCXSynthesisNCleanM15`
      - `k-2`
      - `0`
      - at most `6*k-6` CX gates
    * - ``"n_dirty_i15"``
      - :class:`~.MCXSynthesisNDirtyI15`
      - `0`
      - `k-2`
      - at most `8*k-6` CX gates
    * - ``"1_clean_b95"``
      - :class:`~.MCXSynthesis1CleanB95`
      - `1`
      - `0`
      - at most `16*k-8` CX gates
    * - ``"default"``
      - :class:`~.MCXSynthesisDefault`
      - any
      - any
      - chooses the best algorithm based on the ancillas available

.. autosummary::
   :toctree: ../stubs/

   MCXSynthesisGrayCode
   MCXSynthesisNoAuxV24
   MCXSynthesisNCleanM15
   MCXSynthesisNDirtyI15
   MCXSynthesis1CleanB95
   MCXSynthesisDefault


MCMT Synthesis
''''''''''''''

.. list-table:: Plugins for :class:`.MCMTGate` (key = ``"mcmt"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Number of clean ancillas
      - Number of dirty ancillas
      - Description
    * - ``"vchain"``
      - :class:`.MCMTSynthesisVChain`
      - `k-1`
      - `0`
      - uses a linear number of Toffoli gates
    * - ``"noaux"``
      - :class:`~.MCMTSynthesisNoAux`
      - `0`
      - `0`
      - uses Qiskit's standard control mechanism
    * - ``"default"``
      - :class:`~.MCMTSynthesisDefault`
      - any
      - any
      - chooses the best algorithm based on the ancillas available

.. autosummary::
   :toctree: ../stubs/

   MCMTSynthesisVChain
   MCMTSynthesisNoAux
   MCMTSynthesisDefault


Pauli Evolution Synthesis
'''''''''''''''''''''''''

.. list-table:: Plugins for :class:`.PauliEvolutionGate` (key = ``"PauliEvolution"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Description
      - Targeted connectivity
    * - ``"rustiq"``
      - :class:`~.PauliEvolutionSynthesisRustiq`
      - use the synthesis method from `Rustiq circuit synthesis library 
        <https://github.com/smartiel/rustiq-core>`_
      - all-to-all
    * - ``"default"``
      - :class:`~.PauliEvolutionSynthesisDefault`
      - use a diagonalizing Clifford per Pauli term
      - all-to-all

.. autosummary::
   :toctree: ../stubs/

   PauliEvolutionSynthesisDefault
   PauliEvolutionSynthesisRustiq


Modular Adder Synthesis
'''''''''''''''''''''''

.. list-table:: Plugins for :class:`.ModularAdderGate` (key = ``"ModularAdder"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Number of clean ancillas
      - Description
    * - ``"ripple_cdkm"``
      - :class:`.ModularAdderSynthesisC04`
      - 1
      - a ripple-carry adder
    * - ``"ripple_vbe"``
      - :class:`.ModularAdderSynthesisV95`
      - :math:`n-1`, for :math:`n`-bit numbers
      - a ripple-carry adder
    * - ``"qft"``
      - :class:`.ModularAdderSynthesisD00`
      - 0
      - a QFT-based adder
    * - ``"default"``
      - :class:`~.ModularAdderSynthesisDefault`
      - any
      - chooses the best algorithm based on the ancillas available

.. autosummary::
   :toctree: ../stubs/

   ModularAdderSynthesisC04
   ModularAdderSynthesisD00
   ModularAdderSynthesisV95
   ModularAdderSynthesisDefault

Half Adder Synthesis
''''''''''''''''''''

.. list-table:: Plugins for :class:`.HalfAdderGate` (key = ``"HalfAdder"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Number of clean ancillas
      - Description
    * - ``"ripple_cdkm"``
      - :class:`.HalfAdderSynthesisC04`
      - 1
      - a ripple-carry adder
    * - ``"ripple_vbe"``
      - :class:`.HalfAdderSynthesisV95`
      - :math:`n-1`, for :math:`n`-bit numbers
      - a ripple-carry adder
    * - ``"qft"``
      - :class:`.HalfAdderSynthesisD00`
      - 0
      - a QFT-based adder
    * - ``"default"``
      - :class:`~.HalfAdderSynthesisDefault`
      - any
      - chooses the best algorithm based on the ancillas available

.. autosummary::
   :toctree: ../stubs/

   HalfAdderSynthesisC04
   HalfAdderSynthesisD00
   HalfAdderSynthesisV95
   HalfAdderSynthesisDefault

Full Adder Synthesis
''''''''''''''''''''

.. list-table:: Plugins for :class:`.FullAdderGate` (key = ``"FullAdder"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Number of clean ancillas
      - Description
    * - ``"ripple_cdkm"``
      - :class:`.FullAdderSynthesisC04`
      - 0
      - a ripple-carry adder
    * - ``"ripple_vbe"``
      - :class:`.FullAdderSynthesisV95`
      - :math:`n-1`, for :math:`n`-bit numbers
      - a ripple-carry adder
    * - ``"default"``
      - :class:`~.FullAdderSynthesisDefault`
      - any
      - chooses the best algorithm based on the ancillas available

.. autosummary::
   :toctree: ../stubs/

   FullAdderSynthesisC04
   FullAdderSynthesisV95
   FullAdderSynthesisDefault


Multiplier Synthesis
''''''''''''''''''''

.. list-table:: Plugins for :class:`.MultiplierGate` (key = ``"Multiplier"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Number of clean ancillas
      - Description
    * - ``"cumulative"``
      - :class:`.MultiplierSynthesisH18`
      - depending on the :class:`.AdderGate` used
      - a cumulative adder based on controlled adders
    * - ``"qft"``
      - :class:`.MultiplierSynthesisR17`
      - 0
      - a QFT-based multiplier

.. autosummary::
   :toctree: ../stubs/

   MultiplierSynthesisH18
   MultiplierSynthesisR17

"""

from __future__ import annotations

import warnings
import numpy as np
import rustworkx as rx

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library import (
    LinearFunction,
    QFTGate,
    MCXGate,
    C3XGate,
    C4XGate,
    PauliEvolutionGate,
    PermutationGate,
    MCMTGate,
    ModularAdderGate,
    HalfAdderGate,
    FullAdderGate,
    MultiplierGate,
)
from qiskit.transpiler.coupling import CouplingMap

from qiskit.synthesis.clifford import (
    synth_clifford_full,
    synth_clifford_layers,
    synth_clifford_depth_lnn,
    synth_clifford_greedy,
    synth_clifford_ag,
    synth_clifford_bm,
)
from qiskit.synthesis.linear import (
    synth_cnot_count_full_pmh,
    synth_cnot_depth_line_kms,
    calc_inverse_matrix,
)
from qiskit.synthesis.linear.linear_circuits_utils import transpose_cx_circ
from qiskit.synthesis.permutation import (
    synth_permutation_basic,
    synth_permutation_acg,
    synth_permutation_depth_lnn_kms,
)
from qiskit.synthesis.qft import (
    synth_qft_full,
    synth_qft_line,
)
from qiskit.synthesis.multi_controlled import (
    synth_mcx_n_dirty_i15,
    synth_mcx_n_clean_m15,
    synth_mcx_1_clean_b95,
    synth_mcx_gray_code,
    synth_mcx_noaux_v24,
    synth_mcmt_vchain,
)
from qiskit.synthesis.evolution import ProductFormula, synth_pauli_network_rustiq
from qiskit.synthesis.arithmetic import (
    adder_ripple_c04,
    adder_qft_d00,
    adder_ripple_v95,
    multiplier_qft_r17,
    multiplier_cumulative_h18,
)
from qiskit.quantum_info.operators import Clifford
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper
from .plugin import HighLevelSynthesisPlugin


class DefaultSynthesisClifford(HighLevelSynthesisPlugin):
    """The default clifford synthesis plugin.

    For N <= 3 qubits this is the optimal CX cost decomposition by Bravyi, Maslov.
    For N > 3 qubits this is done using the general non-optimal greedy compilation
    routine from reference by Bravyi, Hu, Maslov, Shaydulin.

    This plugin name is :``clifford.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if not isinstance(high_level_object, Clifford):
            return None

        decomposition = synth_clifford_full(high_level_object)
        return decomposition


class AGSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Aaronson-Gottesman method.

    This plugin name is :``clifford.ag`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if not isinstance(high_level_object, Clifford):
            return None

        decomposition = synth_clifford_ag(high_level_object)
        return decomposition


class BMSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method.

    The method only works on Cliffords with at most 3 qubits, for which it
    constructs the optimal CX cost decomposition.

    This plugin name is :``clifford.bm`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if not isinstance(high_level_object, Clifford):
            return None

        if high_level_object.num_qubits <= 3:
            decomposition = synth_clifford_bm(high_level_object)
        else:
            decomposition = None

        return decomposition


class GreedySynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the greedy synthesis
    Bravyi-Hu-Maslov-Shaydulin method.

    This plugin name is :``clifford.greedy`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if not isinstance(high_level_object, Clifford):
            return None

        decomposition = synth_clifford_greedy(high_level_object)
        return decomposition


class LayerSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method
    to synthesize Cliffords into layers.

    This plugin name is :``clifford.layers`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if not isinstance(high_level_object, Clifford):
            return None

        decomposition = synth_clifford_layers(high_level_object)
        return decomposition


class LayerLnnSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method
    to synthesize Cliffords into layers, with each layer synthesized
    adhering to LNN connectivity.

    This plugin name is :``clifford.lnn`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if not isinstance(high_level_object, Clifford):
            return None

        decomposition = synth_clifford_depth_lnn(high_level_object)
        return decomposition


class DefaultSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """The default linear function synthesis plugin.

    This plugin name is :``linear_function.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""
        if not isinstance(high_level_object, LinearFunction):
            return None

        decomposition = synth_cnot_count_full_pmh(high_level_object.linear)
        return decomposition


class KMSSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Kutin-Moulton-Smithline method.

    This plugin name is :``linear_function.kms`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The plugin supports the following plugin-specific options:

    * use_inverted: Indicates whether to run the algorithm on the inverse matrix
        and to invert the synthesized circuit.
        In certain cases this provides a better decomposition than the direct approach.
    * use_transposed: Indicates whether to run the algorithm on the transposed matrix
        and to invert the order of CX gates in the synthesized circuit.
        In certain cases this provides a better decomposition than the direct approach.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""
        if not isinstance(high_level_object, LinearFunction):
            return None

        use_inverted = options.get("use_inverted", False)
        use_transposed = options.get("use_transposed", False)

        mat = high_level_object.linear.astype(bool, copy=False)

        if use_transposed:
            mat = np.transpose(mat)
        if use_inverted:
            mat = calc_inverse_matrix(mat)

        decomposition = synth_cnot_depth_line_kms(mat)

        if use_transposed:
            decomposition = transpose_cx_circ(decomposition)
        if use_inverted:
            decomposition = decomposition.inverse()

        return decomposition


class PMHSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Patel-Markov-Hayes method.

    This plugin name is :``linear_function.pmh`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The plugin supports the following plugin-specific options:

    * section size: The size of each section used in the Patel–Markov–Hayes algorithm [1].
    * use_inverted: Indicates whether to run the algorithm on the inverse matrix
        and to invert the synthesized circuit.
        In certain cases this provides a better decomposition than the direct approach.
    * use_transposed: Indicates whether to run the algorithm on the transposed matrix
        and to invert the order of CX gates in the synthesized circuit.
        In certain cases this provides a better decomposition than the direct approach.

    References:
        1. Patel, Ketan N., Igor L. Markov, and John P. Hayes,
           *Optimal synthesis of linear reversible circuits*,
           Quantum Information & Computation 8.3 (2008): 282-294.
           `arXiv:quant-ph/0302002 [quant-ph] <https://arxiv.org/abs/quant-ph/0302002>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""
        if not isinstance(high_level_object, LinearFunction):
            return None

        section_size = options.get("section_size", 2)
        use_inverted = options.get("use_inverted", False)
        use_transposed = options.get("use_transposed", False)

        mat = high_level_object.linear.astype(bool, copy=False)

        if use_transposed:
            mat = np.transpose(mat)
        if use_inverted:
            mat = calc_inverse_matrix(mat)

        decomposition = synth_cnot_count_full_pmh(mat, section_size=section_size)

        if use_transposed:
            decomposition = transpose_cx_circ(decomposition)
        if use_inverted:
            decomposition = decomposition.inverse()

        return decomposition


class KMSSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Kutin, Moulton, Smithline method.

    This plugin name is :``permutation.kms`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        if not isinstance(high_level_object, PermutationGate):
            return None

        decomposition = synth_permutation_depth_lnn_kms(high_level_object.pattern)
        return decomposition


class BasicSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on sorting.

    This plugin name is :``permutation.basic`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        if not isinstance(high_level_object, PermutationGate):
            return None

        decomposition = synth_permutation_basic(high_level_object.pattern)
        return decomposition


class ACGSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Alon, Chung, Graham method.

    This plugin name is :``permutation.acg`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        if not isinstance(high_level_object, PermutationGate):
            return None

        decomposition = synth_permutation_acg(high_level_object.pattern)
        return decomposition


class QFTSynthesisFull(HighLevelSynthesisPlugin):
    """Synthesis plugin for QFT gates using all-to-all connectivity.

    This plugin name is :``qft.full`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    Note that the plugin mechanism is not applied if the gate is called ``qft`` but
    is not an instance of ``QFTGate``. This allows users to create custom gates with
    name ``qft``.

    The plugin supports the following additional options:

    * reverse_qubits (bool): Whether to synthesize the "QFT" operation (if ``False``,
        which is the default) or the "QFT-with-reversal" operation (if ``True``).
        Some implementation of the ``QFTGate`` include a layer of swap gates at the end
        of the synthesized circuit, which can in principle be dropped if the ``QFTGate``
        itself is the last gate in the circuit.
    * approximation_degree (int): The degree of approximation (0 for no approximation).
        It is possible to implement the QFT approximately by ignoring
        controlled-phase rotations with the angle beneath a threshold. This is discussed
        in more detail in [1] or [2].
    * insert_barriers (bool): If True, barriers are inserted as visualization improvement.
    * inverse (bool): If True, the inverse Fourier transform is constructed.
    * name (str): The name of the circuit.

    References:
        1. Adriano Barenco, Artur Ekert, Kalle-Antti Suominen, and Päivi Törmä,
           *Approximate Quantum Fourier Transform and Decoherence*,
           Physical Review A (1996).
           `arXiv:quant-ph/9601018 [quant-ph] <https://arxiv.org/abs/quant-ph/9601018>`_
        2. Donny Cheung,
           *Improved Bounds for the Approximate QFT* (2004),
           `arXiv:quant-ph/0403071 [quant-ph] <https://https://arxiv.org/abs/quant-ph/0403071>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given QFTGate."""

        # Even though the gate is called "qft", it's not a QFTGate,
        # and we should not synthesize it using the plugin.
        if not isinstance(high_level_object, QFTGate):
            return None

        reverse_qubits = options.get("reverse_qubits", False)
        approximation_degree = options.get("approximation_degree", 0)
        insert_barriers = options.get("insert_barriers", False)
        inverse = options.get("inverse", False)
        name = options.get("name", None)

        decomposition = synth_qft_full(
            num_qubits=high_level_object.num_qubits,
            do_swaps=not reverse_qubits,
            approximation_degree=approximation_degree,
            insert_barriers=insert_barriers,
            inverse=inverse,
            name=name,
        )
        return decomposition


class QFTSynthesisLine(HighLevelSynthesisPlugin):
    """Synthesis plugin for QFT gates using linear connectivity.

    This plugin name is :``qft.line`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    Note that the plugin mechanism is not applied if the gate is called ``qft`` but
    is not an instance of ``QFTGate``. This allows users to create custom gates with
    name ``qft``.

    The plugin supports the following additional options:

    * reverse_qubits (bool): Whether to synthesize the "QFT" operation (if ``False``,
        which is the default) or the "QFT-with-reversal" operation (if ``True``).
        Some implementation of the ``QFTGate`` include a layer of swap gates at the end
        of the synthesized circuit, which can in principle be dropped if the ``QFTGate``
        itself is the last gate in the circuit.
    * approximation_degree (int): the degree of approximation (0 for no approximation).
        It is possible to implement the QFT approximately by ignoring
        controlled-phase rotations with the angle beneath a threshold. This is discussed
        in more detail in [1] or [2].

    References:
        1. Adriano Barenco, Artur Ekert, Kalle-Antti Suominen, and Päivi Törmä,
           *Approximate Quantum Fourier Transform and Decoherence*,
           Physical Review A (1996).
           `arXiv:quant-ph/9601018 [quant-ph] <https://arxiv.org/abs/quant-ph/9601018>`_
        2. Donny Cheung,
           *Improved Bounds for the Approximate QFT* (2004),
           `arXiv:quant-ph/0403071 [quant-ph] <https://https://arxiv.org/abs/quant-ph/0403071>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given QFTGate."""

        # Even though the gate is called "qft", it's not a QFTGate,
        # and we should not synthesize it using the plugin.
        if not isinstance(high_level_object, QFTGate):
            return None

        reverse_qubits = options.get("reverse_qubits", False)
        approximation_degree = options.get("approximation_degree", 0)

        decomposition = synth_qft_line(
            num_qubits=high_level_object.num_qubits,
            do_swaps=not reverse_qubits,
            approximation_degree=approximation_degree,
        )
        return decomposition


class TokenSwapperSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the token swapper algorithm.

    This plugin name is :``permutation.token_swapper`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    In more detail, this plugin is used to synthesize objects of type `PermutationGate`.
    When synthesis succeeds, the plugin outputs a quantum circuit consisting only of swap
    gates. When synthesis does not succeed, the plugin outputs `None`.

    If either `coupling_map` or `qubits` is None, then the synthesized circuit
    is not required to adhere to connectivity constraints, as is the case
    when the synthesis is done before layout/routing.

    On the other hand, if both `coupling_map` and `qubits` are specified, the synthesized
    circuit is supposed to adhere to connectivity constraints. At the moment, the
    plugin only creates swap gates between qubits in `qubits`, i.e. it does not use
    any other qubits in the coupling map (if such synthesis is not possible, the
    plugin  outputs `None`).

    The plugin supports the following plugin-specific options:

    * trials: The number of trials for the token swapper to perform the mapping. The
      circuit with the smallest number of SWAPs is returned.
    * seed: The argument to the token swapper specifying the seed for random trials.
    * parallel_threshold: The argument to the token swapper specifying the number of nodes
      in the graph beyond which the algorithm will use parallel processing.

    For more details on the token swapper algorithm, see to the paper:
    `arXiv:1902.09102 <https://arxiv.org/abs/1902.09102>`__.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""

        if not isinstance(high_level_object, PermutationGate):
            return None

        trials = options.get("trials", 5)
        seed = options.get("seed", 0)
        parallel_threshold = options.get("parallel_threshold", 50)

        pattern = high_level_object.pattern
        pattern_as_dict = {j: i for i, j in enumerate(pattern)}

        # When the plugin is called from the HighLevelSynthesis transpiler pass,
        # the coupling map already takes target into account.
        if coupling_map is None or qubits is None:
            # The abstract synthesis uses a fully connected coupling map, allowing
            # arbitrary connections between qubits.
            used_coupling_map = CouplingMap.from_full(len(pattern))
        else:
            # The concrete synthesis uses the coupling map restricted to the set of
            # qubits over which the permutation gate is defined. If we allow using other
            # qubits in the coupling map, replacing the node in the DAGCircuit that
            # defines this PermutationGate by the DAG corresponding to the constructed
            # decomposition becomes problematic. Note that we allow the reduced
            # coupling map to be disconnected.
            used_coupling_map = coupling_map.reduce(qubits, check_if_connected=False)

        graph = used_coupling_map.graph.to_undirected()
        swapper = ApproximateTokenSwapper(graph, seed=seed)

        try:
            swapper_result = swapper.map(
                pattern_as_dict, trials, parallel_threshold=parallel_threshold
            )
        except rx.InvalidMapping:
            swapper_result = None

        if swapper_result is not None:
            decomposition = QuantumCircuit(len(graph.node_indices()))
            for swap in swapper_result:
                decomposition.swap(*swap)
            return decomposition

        return None


class MCXSynthesisNDirtyI15(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the paper
    by Iten et al. (2016).

    See [1] for details.

    This plugin name is :``mcx.n_dirty_i15`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k\ge 3` control qubits this synthesis
    method requires :math:`k - 2` additional dirty auxiliary qubits. The synthesized
    circuit consists of :math:`2 * k - 1` qubits and at most :math:`8 * k - 6` CX gates.

    The plugin supports the following plugin-specific options:

    * num_clean_ancillas: The number of clean auxiliary qubits available.
    * num_dirty_ancillas: The number of dirty auxiliary qubits available.
    * relative_phase: When set to ``True``, the method applies the optimized multi-controlled
      X gate up to a relative phase, in a way that, by lemma 8 of [1], the relative
      phases of the ``action part`` cancel out with the phases of the ``reset part``.
    * action_only: when set to ``True``, the method applies only the ``action part``
      of lemma 8 of [1].

    References:
        1. Iten et. al., *Quantum Circuits for Isometries*, Phys. Rev. A 93, 032318 (2016),
           `arXiv:1501.06911 <http://arxiv.org/abs/1501.06911>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        num_clean_ancillas = options.get("num_clean_ancillas", 0)
        num_dirty_ancillas = options.get("num_dirty_ancillas", 0)
        relative_phase = options.get("relative_phase", False)
        action_only = options.get("actions_only", False)

        if num_ctrl_qubits >= 3 and num_dirty_ancillas + num_clean_ancillas < num_ctrl_qubits - 2:
            # This synthesis method is not applicable as there are not enough ancilla qubits
            return None

        decomposition = synth_mcx_n_dirty_i15(num_ctrl_qubits, relative_phase, action_only)
        return decomposition


class MCXSynthesisNCleanM15(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the paper by
    Maslov (2016).

    See [1] for details.

    This plugin name is :``mcx.n_clean_m15`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k\ge 3` control qubits this synthesis
    method requires :math:`k - 2` additional clean auxiliary qubits. The synthesized
    circuit consists of :math:`2 * k - 1` qubits and at most :math:`6 * k - 6` CX gates.

    The plugin supports the following plugin-specific options:

    * num_clean_ancillas: The number of clean auxiliary qubits available.

    References:
        1. Maslov., Phys. Rev. A 93, 022311 (2016),
           `arXiv:1508.03273 <https://arxiv.org/pdf/1508.03273>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        num_clean_ancillas = options.get("num_clean_ancillas", 0)

        if num_ctrl_qubits >= 3 and num_clean_ancillas < num_ctrl_qubits - 2:
            # This synthesis method is not applicable as there are not enough ancilla qubits
            return None

        decomposition = synth_mcx_n_clean_m15(num_ctrl_qubits)
        return decomposition


class MCXSynthesis1CleanB95(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the paper by
    Barenco et al. (1995).

    See [1] for details.

    This plugin name is :``mcx.1_clean_b95`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k\ge 5` control qubits this synthesis
    method requires a single additional clean auxiliary qubit. The synthesized
    circuit consists of :math:`k + 2` qubits and at most :math:`16 * k - 8` CX gates.

    The plugin supports the following plugin-specific options:

    * num_clean_ancillas: The number of clean auxiliary qubits available.

    References:
        1. Barenco et. al., Phys.Rev. A52 3457 (1995),
           `arXiv:quant-ph/9503016 <https://arxiv.org/abs/quant-ph/9503016>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits

        if num_ctrl_qubits <= 2:
            # The method requires at least 3 control qubits
            return None

        num_clean_ancillas = options.get("num_clean_ancillas", 0)

        if num_ctrl_qubits >= 5 and num_clean_ancillas == 0:
            # This synthesis method is not applicable as there are not enough ancilla qubits
            return None

        decomposition = synth_mcx_1_clean_b95(num_ctrl_qubits)
        return decomposition


class MCXSynthesisGrayCode(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the Gray code.

    This plugin name is :``mcx.gray_code`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k` control qubits this synthesis
    method requires no additional clean auxiliary qubits. The synthesized
    circuit consists of :math:`k + 1` qubits.

    It is not recommended to use this method for large values of :math:`k + 1`
    as it produces exponentially many gates.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        decomposition = synth_mcx_gray_code(num_ctrl_qubits)
        return decomposition


class MCXSynthesisNoAuxV24(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the
    implementation for MCPhaseGate, which is in turn based on the
    paper by Vale et al. (2024).

    See [1] for details.

    This plugin name is :``mcx.noaux_v24`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k` control qubits this synthesis
    method requires no additional clean auxiliary qubits. The synthesized
    circuit consists of :math:`k + 1` qubits.

    References:
        1. Vale et. al., *Circuit Decomposition of Multicontrolled Special Unitary
           Single-Qubit Gates*, IEEE TCAD 43(3) (2024),
           `arXiv:2302.06377 <https://arxiv.org/abs/2302.06377>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        decomposition = synth_mcx_noaux_v24(num_ctrl_qubits)
        return decomposition


class MCXSynthesisDefault(HighLevelSynthesisPlugin):
    r"""The default synthesis plugin for a multi-controlled X gate.

    This plugin name is :``mcx.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        # Iteratively run other synthesis methods available

        if (
            decomposition := MCXSynthesisNCleanM15().run(
                high_level_object, coupling_map, target, qubits, **options
            )
        ) is not None:
            return decomposition

        if (
            decomposition := MCXSynthesisNDirtyI15().run(
                high_level_object, coupling_map, target, qubits, **options
            )
        ) is not None:
            return decomposition

        if (
            decomposition := MCXSynthesis1CleanB95().run(
                high_level_object, coupling_map, target, qubits, **options
            )
        ) is not None:
            return decomposition

        return MCXSynthesisNoAuxV24().run(
            high_level_object, coupling_map, target, qubits, **options
        )


class MCMTSynthesisDefault(HighLevelSynthesisPlugin):
    """A default decomposition for MCMT gates."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        # first try to use the V-chain synthesis if enough auxiliary qubits are available
        if not isinstance(high_level_object, MCMTGate):
            return None

        if (
            decomposition := MCMTSynthesisVChain().run(
                high_level_object, coupling_map, target, qubits, **options
            )
        ) is not None:
            return decomposition

        return MCMTSynthesisNoAux().run(high_level_object, coupling_map, target, qubits, **options)


class MCMTSynthesisNoAux(HighLevelSynthesisPlugin):
    """A V-chain based synthesis for ``MCMTGate``."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, MCMTGate):
            return None

        base_gate = high_level_object.base_gate
        ctrl_state = options.get("ctrl_state", None)

        if high_level_object.num_target_qubits == 1:
            # no broadcasting needed (makes for better circuit diagrams)
            circuit = QuantumCircuit(high_level_object.num_qubits)
            circuit.append(
                base_gate.control(high_level_object.num_ctrl_qubits, ctrl_state=ctrl_state),
                circuit.qubits,
            )

        else:
            base = QuantumCircuit(high_level_object.num_target_qubits, name=high_level_object.label)
            for i in range(high_level_object.num_target_qubits):
                base.append(base_gate, [i], [])

            circuit = base.control(high_level_object.num_ctrl_qubits, ctrl_state=ctrl_state)

        return circuit.decompose()


class MCMTSynthesisVChain(HighLevelSynthesisPlugin):
    """A V-chain based synthesis for ``MCMTGate``."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, MCMTGate):
            return None

        if options.get("num_clean_ancillas", 0) < high_level_object.num_ctrl_qubits - 1:
            return None  # insufficient number of auxiliary qubits

        ctrl_state = options.get("ctrl_state", None)

        return synth_mcmt_vchain(
            high_level_object.base_gate,
            high_level_object.num_ctrl_qubits,
            high_level_object.num_target_qubits,
            ctrl_state,
        )


class ModularAdderSynthesisDefault(HighLevelSynthesisPlugin):
    """The default modular adder (no carry in, no carry out qubit) synthesis.

    This plugin name is:``ModularAdder.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    If at least one clean auxiliary qubit is available, the :class:`ModularAdderSynthesisC04`
    is used, otherwise :class:`ModularAdderSynthesisD00`.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, ModularAdderGate):
            return None

        # For up to 5 qubits, the QFT-based adder is best
        if high_level_object.num_state_qubits <= 5:
            decomposition = ModularAdderSynthesisD00().run(
                high_level_object, coupling_map, target, qubits, **options
            )
            if decomposition is not None:
                return decomposition

        # Otherwise, the following decomposition is best (if there are enough ancillas)
        if (
            decomposition := ModularAdderSynthesisC04().run(
                high_level_object, coupling_map, target, qubits, **options
            )
        ) is not None:
            return decomposition

        # Otherwise, use the QFT-adder again
        return ModularAdderSynthesisD00().run(
            high_level_object, coupling_map, target, qubits, **options
        )


class ModularAdderSynthesisC04(HighLevelSynthesisPlugin):
    r"""A ripple-carry adder, modulo :math:`2^n`.

    This plugin name is:``ModularAdder.ripple_c04`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    This plugin requires at least one clean auxiliary qubit.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, ModularAdderGate):
            return None

        # unless we implement the full adder, this implementation needs an ancilla qubit
        if options.get("num_clean_ancillas", 0) < 1:
            return None

        return adder_ripple_c04(high_level_object.num_state_qubits, kind="fixed")


class ModularAdderSynthesisV95(HighLevelSynthesisPlugin):
    r"""A ripple-carry adder, modulo :math:`2^n`.

    This plugin name is:``ModularAdder.ripple_v95`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For an adder on 2 registers with :math:`n` qubits each, this plugin requires at
    least :math:`n-1` clean auxiliary qubit.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, ModularAdderGate):
            return None

        num_state_qubits = high_level_object.num_state_qubits

        # The synthesis method needs n-1 clean ancilla qubits
        if num_state_qubits - 1 > options.get("num_clean_ancillas", 0):
            return None

        return adder_ripple_v95(num_state_qubits, kind="fixed")


class ModularAdderSynthesisD00(HighLevelSynthesisPlugin):
    r"""A QFT-based adder, modulo :math:`2^n`.

    This plugin name is:``ModularAdder.qft_d00`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, ModularAdderGate):
            return None

        return adder_qft_d00(high_level_object.num_state_qubits, kind="fixed")


class HalfAdderSynthesisDefault(HighLevelSynthesisPlugin):
    r"""The default half-adder (no carry in, but a carry out qubit) synthesis.

    If we have an auxiliary qubit available, the Cuccaro ripple-carry adder uses
    :math:`O(n)` CX gates and 1 auxiliary qubit, whereas the Vedral ripple-carry uses more CX
    and :math:`n-1` auxiliary qubits. The QFT-based adder uses no auxiliary qubits, but
    :math:`O(n^2)`, hence it is only used if no auxiliary qubits are available.

    This plugin name is:``HalfAdder.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    If at least one clean auxiliary qubit is available, the :class:`HalfAdderSynthesisC04`
    is used, otherwise :class:`HalfAdderSynthesisD00`.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, HalfAdderGate):
            return None

        # For up to 3 qubits, ripple_v95 is better (if there are enough ancilla qubits)
        if high_level_object.num_state_qubits <= 3:
            decomposition = HalfAdderSynthesisV95().run(
                high_level_object, coupling_map, target, qubits, **options
            )
            if decomposition is not None:
                return decomposition

        # The next best option is to use ripple_c04 (if there are enough ancilla qubits)
        if (
            decomposition := HalfAdderSynthesisC04().run(
                high_level_object, coupling_map, target, qubits, **options
            )
        ) is not None:
            return decomposition

        # The QFT-based adder does not require ancilla qubits and should always succeed
        return HalfAdderSynthesisD00().run(
            high_level_object, coupling_map, target, qubits, **options
        )


class HalfAdderSynthesisC04(HighLevelSynthesisPlugin):
    """A ripple-carry adder with a carry-out bit.

    This plugin name is:``HalfAdder.ripple_c04`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    This plugin requires at least one clean auxiliary qubit.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, HalfAdderGate):
            return None

        # unless we implement the full adder, this implementation needs an ancilla qubit
        if options.get("num_clean_ancillas", 0) < 1:
            return None

        return adder_ripple_c04(high_level_object.num_state_qubits, kind="half")


class HalfAdderSynthesisV95(HighLevelSynthesisPlugin):
    """A ripple-carry adder with a carry-out bit.

    This plugin name is:``HalfAdder.ripple_v95`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For an adder on 2 registers with :math:`n` qubits each, this plugin requires at
    least :math:`n-1` clean auxiliary qubit.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, HalfAdderGate):
            return None

        num_state_qubits = high_level_object.num_state_qubits

        # The synthesis method needs n-1 clean ancilla qubits
        if num_state_qubits - 1 > options.get("num_clean_ancillas", 0):
            return None

        return adder_ripple_v95(num_state_qubits, kind="half")


class HalfAdderSynthesisD00(HighLevelSynthesisPlugin):
    """A QFT-based adder with a carry-in and a carry-out bit.

    This plugin name is:``HalfAdder.qft_d00`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, HalfAdderGate):
            return None

        return adder_qft_d00(high_level_object.num_state_qubits, kind="half")


class FullAdderSynthesisDefault(HighLevelSynthesisPlugin):
    """A ripple-carry adder with a carry-in and a carry-out bit.

    This plugin name is:``FullAdder.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, FullAdderGate):
            return None

        # FullAdderSynthesisC04 requires no ancilla qubits and returns better results
        # than FullAdderSynthesisV95 in all cases except for n=1.
        if high_level_object.num_state_qubits == 1:
            decomposition = FullAdderSynthesisV95().run(
                high_level_object, coupling_map, target, qubits, **options
            )
            if decomposition is not None:
                return decomposition

        return FullAdderSynthesisC04().run(
            high_level_object, coupling_map, target, qubits, **options
        )


class FullAdderSynthesisC04(HighLevelSynthesisPlugin):
    """A ripple-carry adder with a carry-in and a carry-out bit.

    This plugin name is:``FullAdder.ripple_c04`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    This plugin requires no auxiliary qubits.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, FullAdderGate):
            return None

        return adder_ripple_c04(high_level_object.num_state_qubits, kind="full")


class FullAdderSynthesisV95(HighLevelSynthesisPlugin):
    """A ripple-carry adder with a carry-in and a carry-out bit.

    This plugin name is:``FullAdder.ripple_v95`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For an adder on 2 registers with :math:`n` qubits each, this plugin requires at
    least :math:`n-1` clean auxiliary qubits.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, FullAdderGate):
            return None

        num_state_qubits = high_level_object.num_state_qubits

        # The synthesis method needs n-1 clean ancilla qubits
        if num_state_qubits - 1 > options.get("num_clean_ancillas", 0):
            return None

        return adder_ripple_v95(num_state_qubits, kind="full")


class MultiplierSynthesisH18(HighLevelSynthesisPlugin):
    """A cumulative multiplier based on controlled adders.

    This plugin name is:``Multiplier.cumulative_h18`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, MultiplierGate):
            return None

        return multiplier_cumulative_h18(
            high_level_object.num_state_qubits, high_level_object.num_result_qubits
        )


class MultiplierSynthesisR17(HighLevelSynthesisPlugin):
    """A QFT-based multiplier.

    This plugin name is:``Multiplier.qft_r17`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, MultiplierGate):
            return None

        return multiplier_qft_r17(
            high_level_object.num_state_qubits, high_level_object.num_result_qubits
        )


class PauliEvolutionSynthesisDefault(HighLevelSynthesisPlugin):
    """Synthesize a :class:`.PauliEvolutionGate` using the default synthesis algorithm.

    This plugin name is:``PauliEvolution.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The following plugin option can be set:

    * preserve_order: If ``False``, allow re-ordering the Pauli terms in the Hamiltonian to
        reduce the circuit depth of the decomposition.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, PauliEvolutionGate):
            # Don't do anything if a gate is called "evolution" but is not an
            # actual PauliEvolutionGate
            return None

        algo = high_level_object.synthesis

        if "preserve_order" in options and isinstance(algo, ProductFormula):
            algo.preserve_order = options["preserve_order"]

        return algo.synthesize(high_level_object)


class PauliEvolutionSynthesisRustiq(HighLevelSynthesisPlugin):
    """Synthesize a :class:`.PauliEvolutionGate` using Rustiq.

    This plugin name is :``PauliEvolution.rustiq`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The Rustiq synthesis algorithm is described in [1], and is implemented in
    Rust-based quantum circuit synthesis library available at
    https://github.com/smartiel/rustiq-core.

    On large circuits the plugin may take a significant runtime.

    The plugin supports the following additional options:

    * optimize_count (bool): if `True` the synthesis algorithm will try to optimize
        the 2-qubit gate count; and if `False` then the 2-qubit depth.
    * preserve_order (bool): whether the order of paulis should be preserved, up to
        commutativity.
    * upto_clifford (bool): if `True`, the final Clifford operator is not synthesized.
    * upto_phase (bool): if `True`, the global phase of the returned circuit may
        differ from the global phase of the given pauli network.
    * resynth_clifford_method (int): describes the strategy to synthesize the final
        Clifford operator. Allowed values are `0` (naive approach), `1` (qiskit
        greedy synthesis), `2` (rustiq isometry synthesis).

    References:
        1. Timothée Goubault de Brugière and Simon Martiel,
           *Faster and shorter synthesis of Hamiltonian simulation circuits*,
           `arXiv:2404.03280 [quant-ph] <https://arxiv.org/abs/2404.03280>`_

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, PauliEvolutionGate):
            # Don't do anything if a gate is called "evolution" but is not an
            # actual PauliEvolutionGate
            return None

        algo = high_level_object.synthesis

        if not isinstance(algo, ProductFormula):
            warnings.warn(
                "Cannot apply Rustiq if the evolution synthesis does not implement ``expand``. ",
                stacklevel=2,
                category=RuntimeWarning,
            )
            return None

        if "preserve_order" in options:
            algo.preserve_order = options["preserve_order"]

        num_qubits = high_level_object.num_qubits
        pauli_network = algo.expand(high_level_object)

        optimize_count = options.get("optimize_count", True)
        preserve_order = options.get("preserve_order", True)
        upto_clifford = options.get("upto_clifford", False)
        upto_phase = options.get("upto_phase", False)
        resynth_clifford_method = options.get("resynth_clifford_method", 1)

        return synth_pauli_network_rustiq(
            num_qubits=num_qubits,
            pauli_network=pauli_network,
            optimize_count=optimize_count,
            preserve_order=preserve_order,
            upto_clifford=upto_clifford,
            upto_phase=upto_phase,
            resynth_clifford_method=resynth_clifford_method,
        )

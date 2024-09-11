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
"""

import numpy as np
import rustworkx as rx

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library import LinearFunction, QFTGate, MCXGate, C3XGate, C4XGate
from qiskit.transpiler.exceptions import TranspilerError
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
)
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
        decomposition = synth_clifford_full(high_level_object)
        return decomposition


class AGSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Aaronson-Gottesman method.

    This plugin name is :``clifford.ag`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
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
        decomposition = synth_clifford_depth_lnn(high_level_object)
        return decomposition


class DefaultSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """The default linear function synthesis plugin.

    This plugin name is :``linear_function.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""
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
            raise TranspilerError(
                "PMHSynthesisLinearFunction only accepts objects of type LinearFunction"
            )

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
            raise TranspilerError(
                "PMHSynthesisLinearFunction only accepts objects of type LinearFunction"
            )

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
        decomposition = synth_permutation_depth_lnn_kms(high_level_object.pattern)
        return decomposition


class BasicSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on sorting.

    This plugin name is :``permutation.basic`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_basic(high_level_object.pattern)
        return decomposition


class ACGSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Alon, Chung, Graham method.

    This plugin name is :``permutation.acg`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        decomposition = synth_permutation_acg(high_level_object.pattern)
        return decomposition


class QFTSynthesisFull(HighLevelSynthesisPlugin):
    """Synthesis plugin for QFT gates using all-to-all connectivity.

    This plugin name is :``qft.full`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

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
        if not isinstance(high_level_object, QFTGate):
            raise TranspilerError(
                "The synthesis plugin 'qft.full` only applies to objects of type QFTGate."
            )

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
        if not isinstance(high_level_object, QFTGate):
            raise TranspilerError(
                "The synthesis plugin 'qft.line` only applies to objects of type QFTGate."
            )

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

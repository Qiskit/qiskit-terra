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
High-level-synthesis transpiler pass.
"""

from __future__ import annotations

import typing
from collections.abc import Callable

import numpy as np

from qiskit.circuit.controlflow.control_flow import ControlFlowOp
from qiskit.circuit.operation import Operation
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import ControlledGate, EquivalenceLibrary, equivalence, Qubit
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError

from qiskit._accelerate.high_level_synthesis import QubitTracker
from .plugin import HighLevelSynthesisPluginManager

if typing.TYPE_CHECKING:
    from qiskit.dagcircuit import DAGOpNode


class HLSConfig:
    """The high-level-synthesis config allows to specify a list of "methods" used by
    :class:`~.HighLevelSynthesis` transformation pass to synthesize different types
    of higher-level objects.

    A higher-level object is an object of type :class:`~.Operation` (e.g., :class:`.Clifford` or
    :class:`.LinearFunction`).  Each object is referred to by its :attr:`~.Operation.name` field
    (e.g., ``"clifford"`` for :class:`.Clifford` objects), and the applicable synthesis methods are
    tied to this name.

    In the config, each method is specified in one of several ways:

    1. a tuple consisting of the name of a known synthesis plugin and a dictionary providing
       additional arguments for the algorithm.
    2. a tuple consisting of an instance of :class:`.HighLevelSynthesisPlugin` and additional
       arguments for the algorithm.
    3. a single string of a known synthesis plugin
    4. a single instance of :class:`.HighLevelSynthesisPlugin`.

    The following example illustrates different ways how a config file can be created::

        from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
        from qiskit.transpiler.passes.synthesis.high_level_synthesis import ACGSynthesisPermutation

        # All the ways to specify hls_config are equivalent
        hls_config = HLSConfig(permutation=[("acg", {})])
        hls_config = HLSConfig(permutation=["acg"])
        hls_config = HLSConfig(permutation=[(ACGSynthesisPermutation(), {})])
        hls_config = HLSConfig(permutation=[ACGSynthesisPermutation()])

    The names of the synthesis plugins should be declared in ``entry-points`` table for
    ``qiskit.synthesis`` in ``pyproject.toml``, in the form
    <higher-level-object-name>.<synthesis-method-name>.

    The standard higher-level-objects are recommended to have a synthesis method
    called "default", which would be called automatically when synthesizing these objects,
    without having to explicitly set these methods in the config.

    To avoid synthesizing a given higher-level-object, one can give it an empty list of methods.

    For an explicit example of using such config files, refer to the documentation for
    :class:`~.HighLevelSynthesis`.

    For an overview of the complete process of using high-level synthesis, see
    :ref:`using-high-level-synthesis-plugins`.
    """

    def __init__(
        self,
        use_default_on_unspecified: bool = True,
        plugin_selection: str = "sequential",
        plugin_evaluation_fn: Callable[[QuantumCircuit], int] | None = None,
        **kwargs,
    ):
        """Creates a high-level-synthesis config.

        Args:
            use_default_on_unspecified: if True, every higher-level-object without an
                explicitly specified list of methods will be synthesized using the "default"
                algorithm if it exists.
            plugin_selection: if set to ``"sequential"`` (default), for every higher-level-object
                the synthesis pass will consider the specified methods sequentially, stopping
                at the first method that is able to synthesize the object. If set to ``"all"``,
                all the specified methods will be considered, and the best synthesized circuit,
                according to ``plugin_evaluation_fn`` will be chosen.
            plugin_evaluation_fn: a callable that evaluates the quality of the synthesized
                quantum circuit; a smaller value means a better circuit. If ``None``, the
                quality of the circuit its size (i.e. the number of gates that it contains).
            kwargs: a dictionary mapping higher-level-objects to lists of synthesis methods.
        """
        self.use_default_on_unspecified = use_default_on_unspecified
        self.plugin_selection = plugin_selection
        self.plugin_evaluation_fn = (
            plugin_evaluation_fn if plugin_evaluation_fn is not None else lambda qc: qc.size()
        )
        self.methods = {}

        for key, value in kwargs.items():
            self.set_methods(key, value)

    def set_methods(self, hls_name, hls_methods):
        """Sets the list of synthesis methods for a given higher-level-object. This overwrites
        the lists of methods if also set previously."""
        self.methods[hls_name] = hls_methods


class HLSData:
    """Internal class for keeping immutable data required by HighLevelSynthesis."""

    def __init__(
        self,
        hls_config,
        hls_plugin_manager,
        coupling_map,
        target,
        use_qubit_indices,
        qubits_initially_zero,
        equivalence_library,
        min_qubits,
        top_level_only,
        device_insts,
    ):
        self.hls_config = hls_config
        self.hls_plugin_manager = hls_plugin_manager
        self.coupling_map = coupling_map
        self.target = target
        self.use_qubit_indices = use_qubit_indices
        self.qubits_initially_zero = qubits_initially_zero
        self.equivalence_library = equivalence_library
        self.min_qubits = min_qubits
        self.top_level_only = top_level_only
        self.device_insts = device_insts


class HighLevelSynthesis(TransformationPass):
    r"""Synthesize higher-level objects and unroll custom definitions.

    The input to this pass is a DAG that may contain higher-level objects,
    including abstract mathematical objects (e.g., objects of type :class:`.LinearFunction`),
    annotated operations (objects of type :class:`.AnnotatedOperation`), and
    custom gates.

    In the most common use-case when either ``basis_gates`` or ``target`` is specified,
    all higher-level objects are synthesized, so the output is a :class:`.DAGCircuit`
    without such objects.
    More precisely, every gate in the output DAG is either directly supported by the target,
    or is in ``equivalence_library``.

    The abstract mathematical objects are synthesized using synthesis plugins, applying
    synthesis methods specified in the high-level-synthesis config (refer to the documentation
    for :class:`~.HLSConfig`).

    As an example, let us assume that ``op_a`` and ``op_b`` are names of two higher-level objects,
    that ``op_a``-objects have two synthesis methods ``default`` which does require any additional
    parameters and ``other`` with two optional integer parameters ``option_1`` and ``option_2``,
    that ``op_b``-objects have a single synthesis method ``default``, and ``qc`` is a quantum
    circuit containing ``op_a`` and ``op_b`` objects. The following code snippet::

        hls_config = HLSConfig(op_b=[("other", {"option_1": 7, "option_2": 4})])
        pm = PassManager([HighLevelSynthesis(hls_config=hls_config)])
        transpiled_qc = pm.run(qc)

    shows how to run the alternative synthesis method ``other`` for ``op_b``-objects, while using the
    ``default`` methods for all other high-level objects, including ``op_a``-objects.

    The annotated operations (consisting of a base operation and a list of inverse, control and power
    modifiers) are synthesizing recursively, first synthesizing the base operation, and then applying
    synthesis methods for creating inverted, controlled, or powered versions of that).

    The custom gates are synthesized by recursively unrolling their definitions, until every gate
    is either supported by the target or is in the equivalence library.

    When neither ``basis_gates`` nor ``target`` is specified, the pass synthesizes only the top-level
    abstract mathematical objects and annotated operations, without descending into the gate
    ``definitions``. This is consistent with the older behavior of the pass, allowing to synthesize
    some higher-level objects using plugins and leaving the other gates untouched.

    The high-level-synthesis passes information about available auxiliary qubits, and whether their
    state is clean (defined as :math:`|0\rangle`) or dirty (unknown state) to the synthesis routine
    via the respective arguments ``"num_clean_ancillas"`` and ``"num_dirty_ancillas"``.
    If ``qubits_initially_zero`` is ``True`` (default), the qubits are assumed to be in the
    :math:`|0\rangle` state. When appending a synthesized block using auxiliary qubits onto the
    circuit, we first use the clean auxiliary qubits.

    .. note::

        Synthesis methods are assumed to maintain the state of the auxiliary qubits.
        Concretely this means that clean auxiliary qubits must still be in the :math:`|0\rangle`
        state after the synthesized block, while dirty auxiliary qubits are re-used only
        as dirty qubits.

    """

    def __init__(
        self,
        hls_config: HLSConfig | None = None,
        coupling_map: CouplingMap | None = None,
        target: Target | None = None,
        use_qubit_indices: bool = False,
        equivalence_library: EquivalenceLibrary | None = None,
        basis_gates: list[str] | None = None,
        min_qubits: int = 0,
        qubits_initially_zero: bool = True,
    ):
        r"""
        HighLevelSynthesis initializer.

        Args:
            hls_config: Optional, the high-level-synthesis config that specifies synthesis methods
                and parameters for various high-level-objects in the circuit. If it is not specified,
                the default synthesis methods and parameters will be used.
            coupling_map: Optional, directed graph represented as a coupling map.
            target: Optional, the backend target to use for this pass. If it is specified,
                it will be used instead of the coupling map.
            use_qubit_indices: a flag indicating whether this synthesis pass is running before or after
                the layout is set, that is, whether the qubit indices of higher-level-objects correspond
                to qubit indices on the target backend.
            equivalence_library: The equivalence library used (instructions in this library will not
                be unrolled by this pass).
            basis_gates: Optional, target basis names to unroll to, e.g. `['u3', 'cx']`.
                Ignored if ``target`` is also specified.
            min_qubits: The minimum number of qubits for operations in the input
                dag to translate.
            qubits_initially_zero: Indicates whether the qubits are initially in the state
                :math:`|0\rangle`. This allows the high-level-synthesis to use clean auxiliary qubits
                (i.e. in the zero state) to synthesize an operation.
        """
        super().__init__()

        # When the config file is not provided, we will use the "default" method
        # to synthesize Operations (when available).
        hls_config = hls_config or HLSConfig(True)
        hls_plugin_manager = HighLevelSynthesisPluginManager()

        if target is not None:
            coupling_map = target.build_coupling_map()

        top_level_only = basis_gates is None and target is None

        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        if not top_level_only and (target is None or target.num_qubits is None):
            basic_insts = {"measure", "reset", "barrier", "snapshot", "delay", "store"}
            device_insts = basic_insts | set(basis_gates)
        else:
            device_insts = set()

        self.data = HLSData(
            hls_config=hls_config,
            hls_plugin_manager=hls_plugin_manager,
            coupling_map=coupling_map,
            target=target,
            use_qubit_indices=use_qubit_indices,
            qubits_initially_zero=qubits_initially_zero,
            equivalence_library=equivalence_library,
            min_qubits=min_qubits,
            top_level_only=top_level_only,
            device_insts=device_insts,
        )

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the HighLevelSynthesis pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with higher-level operations synthesized.

        Raises:
            TranspilerError: when the transpiler is unable to synthesize the given DAG
            (for instance, when the specified synthesis method is not available).
        """

        # Fast-path: check if HighLevelSynthesis can be skipped altogether. This is only
        # done at the top-level since this does not track the qubit states.
        for node in dag.op_nodes():
            qubits = tuple(dag.find_bit(q).index for q in node.qargs)
            if not _definitely_skip_node(self.data, node, qubits, dag):
                break
        else:
            # The for-loop terminates without reaching the break statement
            return dag

        # Regular-path: we synthesize the circuit recursively. Except for
        # this conversion from DAGCircuit to QuantumCircuit and back, all
        # the recursive functions work with QuantumCircuit objects only.
        circuit = dag_to_circuit(dag)
        input_qubits = list(range(circuit.num_qubits))
        tracker = QubitTracker(num_qubits=dag.num_qubits())
        if self.data.qubits_initially_zero:
            tracker.set_clean(input_qubits)
        output_circuit, _ = _run(circuit, input_qubits, self.data, tracker)
        return circuit_to_dag(output_circuit)


def _run(
    input_circuit: QuantumCircuit,
    input_qubits: tuple[int],
    data: HLSData,
    tracker: QubitTracker,
) -> tuple[QuantumCircuit, tuple[int]]:
    """
    Recursively synthesizes a subcircuit. This subcircuit may be either the original
    circuit, the definition circuit for one of the gates, or a circuit returned by
    by a plugin.

    Input:
        input_circuit: the subcircuit to be synthesized.
        input_qubits: a list of global qubits (qubits in the original circuit) over
            which the input circuit is defined.
        data: high-level-synthesis data and options.
        tracker: the global tracker, tracking the state of global qubits.

    The function returns the synthesized circuit and the global qubits over which this
    output circuit is defined. Note that by using auxiliary qubits, the output circuit
    may be defined over more qubits than the input circuit.

    The function also updates in-place the qubit tracker which keeps track of the status of
    each global qubit (whether it's clean, dirty, or cannot be used).
    """

    if not isinstance(input_circuit, QuantumCircuit) or (
        input_circuit.num_qubits != len(input_qubits)
    ):
        raise TranspilerError("HighLevelSynthesis: the input to 'run' is incorrect.")

    # We iteratively process circuit instructions in the order they appear in the input circuit,
    # and add the synthesized instructions to the output circuit. Note that in the process the
    # output circuit may need to be extended with additional qubits. In addition, we keep track
    # of the status of the original qubits using the qubits tracker.
    #
    # Note: This is a first version of a potentially more elaborate approach to find
    # good operation/ancilla allocations. The current approach is greedy and just gives
    # all available ancilla qubits to the current operation ("the-first-takes-all" approach).
    # It does not distribute ancilla qubits between different operations present in the circuit.

    output_circuit = input_circuit.copy_empty_like()
    output_qubits = input_qubits

    # The "inverse" map from the global qubits to the output circuit's qubits.
    # This map may be extended if additional auxiliary qubits get used.
    global_to_local = {q: i for i, q in enumerate(output_qubits)}

    for inst in input_circuit:
        op = inst.operation

        # op's qubits as viewed globally
        op_qubits = [input_qubits[input_circuit.find_bit(q).index] for q in inst.qubits]

        # Start by handling special operations.
        # In the future, we can also consider other possible optimizations, e.g.:
        #   - improved qubit tracking after a SWAP gate
        #   - automatically simplify control gates with control at 0.
        if op.name in ["id", "delay", "barrier"]:
            output_circuit.append(op, inst.qubits, inst.clbits)
            # tracker is not updated, these are no-ops
            continue

        if op.name == "reset":
            output_circuit.append(op, inst.qubits, inst.clbits)
            tracker.set_clean(op_qubits)
            continue

        # Check if synthesis for this operation can be skipped
        if _definitely_skip_op(op, op_qubits, data):
            output_circuit.append(op, inst.qubits, inst.clbits)
            tracker.set_dirty(op_qubits)
            continue

        # Recursively handle control-flow.
        # Currently we do not allow subcircuits within the control flow to use auxiliary qubits
        # and mark all the usable qubits as dirty. This is done in order to avoid complications
        # that different subcircuits may choose to use different auxiliary global qubits, and to
        # avoid complications related to tracking qubit status for while- loops.
        # In the future, this handling can potentially be improved.
        if isinstance(op, ControlFlowOp):
            new_blocks = []
            block_tracker = tracker.copy()
            block_tracker.disable([q for q in range(tracker.num_qubits()) if q not in op_qubits])
            block_tracker.set_dirty(op_qubits)
            for block in op.blocks:
                new_block = _run(block, input_qubits=op_qubits, data=data, tracker=block_tracker)[0]
                new_blocks.append(new_block)
            synthesized_op = op.replace_blocks(new_blocks)
            # The block circuits are defined over the same qubits and clbits as the original
            # instruction.
            output_circuit.append(synthesized_op, inst.qubits, inst.clbits)
            tracker.set_dirty(op_qubits)
            continue

        # Now we synthesize the operation.
        # The function synthesize_operation returns None if the operation does not need to be
        # synthesized, or a quantum circuit together with the global qubits on which this
        # circuit is defined. Also note that the synthesized circuit may involve auxiliary
        # global qubits not used by the input circuit.
        synthesized_circuit, synthesized_circuit_qubits = _synthesize_operation(
            op, op_qubits, data, tracker
        )

        # If the synthesis did not change anything, we add the operation to the output circuit
        # and update the qubit tracker.
        if synthesized_circuit is None:
            output_circuit.append(op, inst.qubits, inst.clbits)
            tracker.set_dirty(op_qubits)
            continue

        # This pedantic check can possibly be removed.
        if not isinstance(synthesized_circuit, QuantumCircuit) or (
            synthesized_circuit.num_qubits != len(synthesized_circuit_qubits)
        ):
            raise TranspilerError(
                "HighLevelSynthesis: the output from 'synthesize_operation' is incorrect."
            )

        # If the synthesized circuit uses (auxiliary) global qubits that are not in the output circuit,
        # we add these qubits to the output circuit.
        for q in synthesized_circuit_qubits:
            if q not in global_to_local:
                global_to_local[q] = len(output_qubits)
                output_qubits.append(q)
                output_circuit.add_bits([Qubit()])

        # Add the operations from the synthesized circuit to the output circuit, using the correspondence
        # syntesized circuit's qubits -> global qubits -> output circuit's qubits
        qubit_map = {
            synthesized_circuit.qubits[i]: output_circuit.qubits[global_to_local[q]]
            for (i, q) in enumerate(synthesized_circuit_qubits)
        }
        clbit_map = dict(zip(synthesized_circuit.clbits, output_circuit.clbits))

        for inst_inner in synthesized_circuit:
            output_circuit.append(
                inst_inner.operation,
                tuple(qubit_map[q] for q in inst_inner.qubits),
                tuple(clbit_map[c] for c in inst_inner.clbits),
            )

        output_circuit.global_phase += synthesized_circuit.global_phase

    # Another pedantic check that can possibly be removed.
    if output_circuit.num_qubits != len(output_qubits):
        raise TranspilerError("HighLevelSynthesis: the input from 'run' is incorrect.")

    return (output_circuit, output_qubits)


def _synthesize_operation(
    operation: Operation,
    input_qubits: tuple[int],
    data: HLSData,
    tracker: QubitTracker,
) -> tuple[QuantumCircuit | None, tuple[int]]:
    """
    Recursively synthesizes a single operation.

    Input:
        operation: the operation to be synthesized.
        input_qubits: a list of global qubits (qubits in the original circuit) over
            which the operation is defined.
        data: high-level-synthesis data and options.
        tracker: the global tracker, tracking the state of global qubits.

    The function returns the synthesized circuit and the global qubits over which this
    output circuit is defined. Note that by using auxiliary qubits, the output circuit
    may be defined over more qubits than the input operation. In addition, the output
    circuit may be ``None``, which means that the operation should remain as it is.

    The function also updates in-place the qubit tracker which keeps track of the status of
    each global qubit (whether it's clean, dirty, or cannot be used).
    """

    if operation.num_qubits != len(input_qubits):
        raise TranspilerError(
            "HighLevelSynthesis: the input to 'synthesize_operation' is incorrect."
        )

    # Synthesize the operation:
    #
    #  (1) Synthesis plugins: try running the battery of high-level synthesis plugins (e.g.
    #      if the operation is a Clifford). If succeeds, this returns a circuit. The plugin
    #      mechanism also includes handling of AnnonatedOperations.
    #  (2) Unrolling custom definitions: try defining the operation if it is not yet
    #       in the set of supported instructions. If succeeds, this returns a circuit.
    #
    # If any of the above is triggered, the returned circuit is recursively synthesized,
    # so that the final circuit only consists of supported operations. If there was no change,
    # we just return ``None``.
    num_original_qubits = len(input_qubits)

    output_circuit = None
    output_qubits = input_qubits

    # Try synthesis via HLS -- which will return ``None`` if unsuccessful.
    if len(hls_methods := _methods_to_try(data, operation.name)) > 0:
        output_circuit, output_qubits = _synthesize_op_using_plugins(
            operation,
            input_qubits,
            data,
            tracker,
            hls_methods,
        )

    # If HLS did not apply, or was unsuccessful, try unrolling custom definitions.
    if output_circuit is None and not data.top_level_only:
        output_circuit, output_qubits = _get_custom_definition(operation, input_qubits, data)

    if output_circuit is not None:
        if not isinstance(output_circuit, QuantumCircuit) or (
            output_circuit.num_qubits != len(output_qubits)
        ):
            raise TranspilerError("HighLevelSynthesis: the intermediate circuit is incorrect.")

    if output_circuit is None:
        # if we didn't synthesize, there is nothing to do.
        # Updating the tracker will be handled upstream.
        pass
    else:
        # Output circuit is a quantum circuit which we want to process recursively.
        # Currently, neither 'synthesize_op_using_plugins' nor 'get_custom_definition'
        # update the tracker (we might want to change this in the future), which makes
        # sense because we have not synthesized the output circuit yet.
        # So we pass the tracker to '_run' but make sure to restore the status of
        # clean ancilla qubits after the circuit is synthesized. In order to do that,
        # we save the current state of the tracker.
        saved_tracker = tracker.copy()
        output_circuit, output_qubits = _run(output_circuit, output_qubits, data, tracker)

        if len(output_qubits) > num_original_qubits:
            tracker.replace_state(saved_tracker, output_qubits[num_original_qubits:])

    if (output_circuit is not None) and (output_circuit.num_qubits != len(output_qubits)):
        raise TranspilerError(
            "HighLevelSynthesis: the output of 'synthesize_operation' is incorrect."
        )

    return output_circuit, output_qubits


def _get_custom_definition(
    operation: Operation, input_qubits: tuple[int], data: HLSData
) -> tuple[QuantumCircuit | None, tuple[int]]:
    """Returns the definition for the given operation.

    Returns None if the operation is already supported or does not have
    the definition.
    """

    # check if the operation is already supported natively
    if not (isinstance(operation, ControlledGate) and operation._open_ctrl):
        # include path for when target exists but target.num_qubits is None (BasicSimulator)
        qubits = input_qubits if data.use_qubit_indices else None
        inst_supported = _instruction_supported(data, operation.name, qubits)
        if inst_supported or (
            data.equivalence_library is not None and data.equivalence_library.has_entry(operation)
        ):
            return (None, input_qubits)  # we support this operation already

    # if not, try to get the definition
    try:
        definition = operation.definition
    except (TypeError, AttributeError) as err:
        raise TranspilerError(f"HighLevelSynthesis is unable to define {operation.name}.") from err

    if definition is None:
        raise TranspilerError(f"HighLevelSynthesis is unable to synthesize {operation}.")

    return (definition, input_qubits)


def _methods_to_try(data: HLSData, name: str):
    """Get a sequence of methods to try for a given op name."""
    if (methods := data.hls_config.methods.get(name)) is not None:
        # the operation's name appears in the user-provided config,
        # we use the list of methods provided by the user
        return methods
    if (
        data.hls_config.use_default_on_unspecified
        and "default" in data.hls_plugin_manager.method_names(name)
    ):
        # the operation's name does not appear in the user-specified config,
        # we use the "default" method when instructed to do so and the "default"
        # method is available
        return ["default"]
    return []


def _synthesize_op_using_plugins(
    operation: Operation,
    input_qubits: tuple[int],
    data: HLSData,
    tracker: QubitTracker,
    hls_methods: list,
) -> tuple[QuantumCircuit | None, tuple[int]]:
    """
    Attempts to synthesize an operation using plugin mechanism.

    Input:
        operation: the operation to be synthesized.
        input_qubits: a list of global qubits (qubits in the original circuit) over
            which the operation is defined.
        data: high-level-synthesis data and options.
        tracker: the global tracker, tracking the state of global qubits.
        hls_methods: the list of synthesis methods to try.

    Returns either the synthesized circuit or ``None`` (which may occur
    when no synthesis methods is available or specified, or when there is
    an insufficient number of auxiliary qubits).
    """

    hls_plugin_manager = data.hls_plugin_manager
    num_clean_ancillas = tracker.num_clean(input_qubits)
    num_dirty_ancillas = tracker.num_dirty(input_qubits)

    best_decomposition = None
    best_score = np.inf

    for method in hls_methods:
        # There are two ways to specify a synthesis method. The more explicit
        # way is to specify it as a tuple consisting of a synthesis algorithm and a
        # list of additional arguments, e.g.,
        #   ("kms", {"all_mats": 1, "max_paths": 100, "orig_circuit": 0}), or
        #   ("pmh", {}).
        # When the list of additional arguments is empty, one can also specify
        # just the synthesis algorithm, e.g.,
        #   "pmh".
        if isinstance(method, tuple):
            plugin_specifier, plugin_args = method
        else:
            plugin_specifier = method
            plugin_args = {}

        # There are two ways to specify a synthesis algorithm being run,
        # either by name, e.g. "kms" (which then should be specified in entry_points),
        # or directly as a class inherited from HighLevelSynthesisPlugin (which then
        # does not need to be specified in entry_points).
        if isinstance(plugin_specifier, str):
            if plugin_specifier not in hls_plugin_manager.method_names(operation.name):
                raise TranspilerError(
                    f"Specified method: {plugin_specifier} not found in available "
                    f"plugins for {operation.name}"
                )
            plugin_method = hls_plugin_manager.method(operation.name, plugin_specifier)
        else:
            plugin_method = plugin_specifier

        # The additional arguments we pass to every plugin include the list of global
        # qubits over which the operation is defined, high-level-synthesis data and options,
        # and the tracker that tracks the state for global qubits.
        #
        # Note: the difference between the argument "qubits" passed explicitely to "run"
        # and "input_qubits" passed via "plugin_args" is that for backwards compatibility
        # the former should be None if the synthesis is done before layout/routing.
        # However, plugins may need access to the global qubits over which the operation
        # is defined, as well as their state, in particular the plugin for AnnotatedOperations
        # requires these arguments to be able to process the base operation recursively.
        #
        # We may want to refactor the inputs and the outputs for the plugins' "run" method,
        # however this needs to be backwards-compatible.
        plugin_args["input_qubits"] = input_qubits
        plugin_args["hls_data"] = data
        plugin_args["qubit_tracker"] = tracker
        plugin_args["num_clean_ancillas"] = num_clean_ancillas
        plugin_args["num_dirty_ancillas"] = num_dirty_ancillas

        qubits = input_qubits if data.use_qubit_indices else None

        decomposition = plugin_method.run(
            operation,
            coupling_map=data.coupling_map,
            target=data.target,
            qubits=qubits,
            **plugin_args,
        )

        # The synthesis methods that are not suited for the given higher-level-object
        # will return None.
        if decomposition is not None:
            if data.hls_config.plugin_selection == "sequential":
                # In the "sequential" mode the first successful decomposition is
                # returned.
                best_decomposition = decomposition
                break

            # In the "run everything" mode we update the best decomposition
            # discovered
            current_score = data.hls_config.plugin_evaluation_fn(decomposition)
            if current_score < best_score:
                best_decomposition = decomposition
                best_score = current_score

    # A synthesis method may have potentially used available ancilla qubits.
    # The following greedily grabs global qubits available. In the additional
    # refactoring mentioned previously, we want each plugin to actually return
    # the global qubits used, especially when the synthesis is done on the physical
    # circuit, and the choice of which ancilla qubits to use really matters.
    output_qubits = input_qubits
    if best_decomposition is not None:
        if best_decomposition.num_qubits > len(input_qubits):
            global_aux_qubits = tracker.borrow(
                best_decomposition.num_qubits - len(input_qubits), input_qubits
            )
            output_qubits = output_qubits + global_aux_qubits

        # This checks (in particular) that there is indeed a sufficient number
        # of ancilla qubits to borrow from the tracker.
        if best_decomposition.num_qubits != len(output_qubits):
            raise TranspilerError(
                "HighLevelSynthesis: the result from 'synthesize_op_using_plugin' is incorrect."
            )

    return (best_decomposition, output_qubits)


def _definitely_skip_node(
    data: HLSData, node: DAGOpNode, qubits: tuple[int] | None, dag: DAGCircuit
) -> bool:
    """Fast-path determination of whether a DAG node can certainly be skipped
    (i.e. nothing will attempt to synthesise it) without accessing its Python-space
    `Operation`.

    This exists as a temporary measure to avoid Python-space `Operation` creation from a
    `DAGOpNode` if we wouldn't do anything to the node (which is _most_ nodes).
    """

    if (
        dag._has_calibration_for(node)
        or len(node.qargs) < data.min_qubits
        or node.is_directive()
        or (_instruction_supported(data, node.name, qubits) and not node.is_control_flow())
    ):
        return True

    return (
        # The fast path is just for Rust-space standard gates (which excludes
        # `AnnotatedOperation`).
        node.is_standard_gate()
        # We don't have the fast-path for controlled gates over 3 or more qubits.
        # However, we most probably want the fast-path for controlled 2-qubit gates
        # (such as CX, CZ, CY, CH, CRX, and so on), so "_definitely_skip_node" should
        # not immediately return False when encountering a controlled gate over 2 qubits.
        and not (node.is_controlled_gate() and node.num_qubits >= 3)
        # If there are plugins to try, they need to be tried.
        and not _methods_to_try(data, node.name)
        # If all the above constraints hold, and it's already supported or the basis translator
        # can handle it, we'll leave it be.
        and (
            # This uses unfortunately private details of `EquivalenceLibrary`, but so does the
            # `BasisTranslator`, and this is supposed to just be temporary til this is moved
            # into Rust space.
            data.equivalence_library is not None
            and equivalence.Key(name=node.name, num_qubits=node.num_qubits)
            in data.equivalence_library.keys()
        )
    )


def _definitely_skip_op(op: Operation, qubits: tuple[int], data: HLSData) -> bool:
    """Check if an operation does not need to be synthesized."""

    # It would be nice to avoid code duplication with the previous function, the difference is
    # that this function is called on "Operation"s rather than "DAGOpNode"s.

    if (
        len(qubits) < data.min_qubits
        or getattr(op, "_directive", False)
        or (_instruction_supported(data, op.name, qubits) and not isinstance(op, ControlFlowOp))
    ):
        return True

    return (
        # The fast path is just for Rust-space standard gates (which excludes
        # `AnnotatedOperation`).
        getattr(op, "_standard_gate", False)
        # We don't have the fast-path for controlled gates over 3 or more qubits.
        # However, we most probably want the fast-path for controlled 2-qubit gates
        # (such as CX, CZ, CY, CH, CRX, and so on), so "_definitely_skip_node" should
        # not immediately return False when encountering a controlled gate over 2 qubits.
        and not (isinstance(op, ControlFlowOp) and op.num_qubits >= 3)
        # If there are plugins to try, they need to be tried.
        and not _methods_to_try(data, op.name)
        # If all the above constraints hold, and it's already supported or the basis translator
        # can handle it, we'll leave it be.
        and (
            # This uses unfortunately private details of `EquivalenceLibrary`, but so does the
            # `BasisTranslator`, and this is supposed to just be temporary til this is moved
            # into Rust space.
            data.equivalence_library is not None
            and equivalence.Key(name=op.name, num_qubits=len(qubits))
            in data.equivalence_library.keys()
        )
    )


def _instruction_supported(data: HLSData, name: str, qubits: tuple[int] | None) -> bool:
    """Check whether operation is natively supported."""
    # include path for when target exists but target.num_qubits is None (BasicSimulator)
    if data.target is None or data.target.num_qubits is None:
        return name in data.device_insts
    return data.target.instruction_supported(operation_name=name, qargs=qubits)

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

"""
=====================================
Transpiler (:mod:`qiskit.transpiler`)
=====================================

.. currentmodule:: qiskit.transpiler

Overview
========

Transpilation is the process of rewriting a given input circuit to match
the topology of a specific quantum device, and/or to optimize the circuit
for execution on present day noisy quantum systems.

Most circuits must undergo a series of transformations that make them compatible with
a given target device, and optimize them to reduce the effects of noise on the
resulting outcomes.  Rewriting quantum circuits to match hardware constraints and
optimizing for performance can be far from trivial.  The flow of logic in the rewriting
tool chain need not be linear, and can often have iterative sub-loops, conditional
branches, and other complex behaviors. That being said, the standard
compilation flow follows the structure given below:

.. image:: /source_images/transpiling_core_steps.png

Qiskit uses the graph-based :class:`.DAGCircuit` intermediate representation (IR) of a circuit
throughout the transpiler stack, rather than the tree-based :class:`.QuantumCircuit`.  A transpiler
pipeline is a :class:`.PassManager` object, whose :meth:`.PassManager.run` method takes in a
:class:`.QuantumCircuit`, converts it to a :class:`.DAGCircuit`, then subjects the IR to a sequence
of *passes*, finally returning a :class:`.QuantumCircuit` back.  A pass is either an
:class:`.AnalysisPass`, whose purpose is to calculate and store properties about the circuit in the
stateful :class:`.PropertySet`, or a :class:`.TransformationPass`, whose purpose is to modify the IR
to achieve a particular singular goal.  We typically think of a pipeline as being split into
"stages", where each stage is responsible for one high-level transformation.

Qiskit exposes a default transpilation pipeline builder via the function
:func:`.generate_preset_pass_manager`.  This will return a properly configured pipeline for complete
transpilation, at a chosen ``optimization_level`` (between 0 and 3, inclusive).  Unless you are
looking for something highly specialized, this is almost certainly the entry point you want.  A
sample transpilation looks like::

    from qiskit.circuit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService

    # Any abstract circuit you want:
    abstract = QuantumCircuit(2)
    abstract.h(0)
    abstract.cx(0, 1)

    # Any method you like to retrieve the backend you want to run on:
    backend = QiskitRuntimeService().backend("some-backend")

    # Create the pass manager for the transpilation ...
    pm = generate_preset_pass_manager(backend=backend)
    # ... and use it (as many times as you like).
    physical = pm.run(abstract)

For most use cases, this is all you will need.
All of Qiskit's transpiler infrastructure is highly extensible and configurable, however.
The rest of this page details how to harness the low-level capabilities of the transpiler stack.

.. _transpiler-preset:

Preset Pass Managers
====================

The function :func:`.generate_preset_pass_manager` creates the "preset pass managers".
These are all instances of :class:`.PassManager`, so are used by pass a :class:`.QuantumCircuit` to
the :meth:`.PassManager.run` method.  More specifically, the preset pass managers are instances of
:class:`.StagedPassManager`, which allows greater configuration of the individual stages of a
tranpsilation.

A preset pass manager has up to six named stages.  These are summarized, in order of execution,
below, with more in-depth information in the following subsections.

``init``
    Abstract-circuit optimizations, and reduction of multi-qubit operations to 1- and 2-qubit
    operations.  See :ref:`transpiler-preset-stage-init` for more detail.

``layout``
    Choose an initial mapping of virtual qubits to physical qubits, including expansion of the
    circuit to contain explicit ancillas.  This stage sometimes subsumes ``routing``.  See
    :ref:`transpiler-preset-stage-layout` for more detail.

``routing``
    Insert gates into the circuit to ensure it matches connectivity constraints of the
    :class:`.Target`.  The inserted gates need not match the target ISA yet, so are often just
    ``swap`` instructions.  This stage is sometimes omitted, when the ``layout`` stage handles its
    job.  See :ref:`transpiler-preset-stage-routing` for more detail.

``translation``
    Convert all gates in the circuit to ones matching the :class:`Target`\\ s ISA.  See
    :ref:`transpiler-preset-stage-translation` for more detail.

``optimization``
    Low-level, hardware-aware optimizations.  Unlike the abstract optimizations of the ``init``
    stage, this stage acts on a physical circuit.  See :ref:`transpiler-preset-stage-optimization`
    for more detail.

``scheduling``
    Insert :class:`~.circuit.Delay` instructions to make the wall-clock timing of a circuit
    explicit.  This may also include hardware-aware online error reduction techniques such as
    dynamical decoupling, which are dependent on knowing wall-clock timings.  See
    :ref:`transpiler-preset-stage-scheduling` for more details.

The preset transpiler pipelines can also be configured at a high level by setting an
``optimization_level``.  This is in integer from 0 to 3 inclusive, indicating the relative effort to
put into attempting to optimize the circuit for the hardware.  Level 0 disables all unnecessary
optimizations; only transformations needed to make the circuit runnable at all will be present.  On
the other end, level 3 enables a full barrage of optimization techniques, some of which can be very
expensive in compilation time.  Similar to classical compilers, optimization level 3 is not always
guaranteed to produce the best results.  Qiskit defaults to optimization level 2.

The optimization level affects which implementations are used for a given stage by default, though
this can be overridden by passing explicit ``<stage>_method="<choice>"`` arguments to
:func:`.generate_preset_pass_manager`.

.. note::

    The preset pass managers almost always include stochastic, heuristic-based passes.  If you need
    to ensure reproducibility of a compilation, pass a known integer to the ``seed_transpiler``
    argument to the generator functions.

    This stochasticity is because many of the problems the transpiler must solve are known to be
    non-polynomial in complexity, and we need to ensure we finish the job in a workable amount of
    time.

Choosing Preset Stage Implementations
-------------------------------------

Qiskit includes several implementations of several of the above stages, and more can be installed as
separate "plugins".  To control which implementation of a stage is used, pass its name to the
``<stage>_method`` keyword argument of the two functions, such as
``translation_method="translator"``.  To read more about implementing such external plugins for a
stage, see :mod:`qiskit.transpiler.preset_passmanagers.plugin`.

For example, to generate a preset pass manager at optimization level 1 that explicitly uses the
``trivial`` method for layout with the ``sabre`` method for routing, we would do:

.. plot::
    :include-source:
    :nofigs:

    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit.providers.fake_provider import GenericBackendV2

    # Whatever backend you like:
    backend = GenericBackendV2(num_qubits=5)

    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend,
        layout_method="trivial",
        routing_method="sabre",
    )

.. note::

    The built-in set of available plugins for each stage is part of Qiskit's public API, and subject
    to all the stability guarantees.  This includes the high-level logical effects of that method
    (for example, ``routing_method="sabre"`` will always use a Sabre-derived algorithm).  The exact
    internal construction of the :class:`.PassManager` representing the stage is not, however; the
    order of passes might change between minor versions, or new passes might be introduced.

    For any stage that has one, the method named ``"default"`` is most subject to change.  We will
    typically only make complete algorithmic changes in the default method across a major-version
    boundary, but we may well rebalance heuristics and add new passes to default methods between
    minor versions.

Since the output of :func:`.generate_preset_pass_manager` is a :class:`.StagedPassManager`, you can
also modify the pass manager after its creation to provide an entirely custom stage implementation.
For example, if you wanted to run a custom scheduling stage using dynamical decoupling (via the
:class:`~.PadDynamicalDecoupling` pass) and also add initial logical optimization prior to routing,
you would do something like (building off the previous example):

.. plot::
   :include-source:
   :nofigs:

    import numpy as np
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.circuit import library as lib
    from qiskit.transpiler import PassManager, generate_preset_pass_manager
    from qiskit.transpiler.passes import (
        ALAPScheduleAnalysis,
        InverseCancellation,
        PadDynamicalDecoupling,
    )

    backend = GenericBackendV2(num_qubits=5)
    dd_sequence = [lib.XGate(), lib.XGate()]
    scheduling_pm = PassManager(
        [
            ALAPScheduleAnalysis(target=backend.target),
            PadDynamicalDecoupling(target=backend.target, dd_sequence=dd_sequence),
        ]
    )
    inverse_gate_list = [
        lib.CXGate(),
        lib.HGate(),
        (lib.RXGate(np.pi / 4), lib.RXGate(-np.pi / 4)),
        (lib.PhaseGate(np.pi / 4), lib.PhaseGate(-np.pi / 4)),
        (lib.TGate(), lib.TdgGate()),
    ]
    logical_opt = PassManager([InverseCancellation(inverse_gate_list)])

    pass_manager = generate_preset_pass_manager(optimization_level=0)
    # Add pre-layout stage to run extra logical optimization
    pass_manager.pre_layout = logical_opt
    # Set scheduling stage to custom pass manager
    pass_manager.scheduling = scheduling_pm

Now, when the staged pass manager is run via the :meth:`~.StagedPassManager.run` method,
the ``logical_opt`` pass manager will be called before the ``layout`` stage, and the
``scheduling_pm`` pass manager will be used for the ``scheduling`` stage instead of the default.

If you are constructing custom stages for the preset pass managers, you may find some of the
low-level helper functions in :mod:`qiskit.transpiler.preset_passmanagers` useful.

.. _transpiler-preset-stage-init:

Initialization Stage
--------------------

.. seealso::
    `Init stage explanation <https://docs.quantum.ibm.com/guides/transpiler-stages#init-stage>`__
        Higher-level user-facing explanation of the init stage in the IBM Quantum guide.

The ``init`` stage is responsible for high-level, logical optimizations on abstract circuits, and
for lowering multi-qubit (3+) operations down to a series of 1- and 2-qubit operations.  As this is
the first stage run, its input is a fully abstract circuit.  The ``init`` stage must be able to
handle custom user-defined gates, and all the high-level abstract circuit-description objects, such
as :class:`.AnnotatedOperation`.

The output of the ``init`` stage is an abstract circuit that contains only 1- and 2-qubit
operations.

When writing :ref:`stage plugins <transpiler-preset-stage-plugins>`, the entry point for ``init`` is
``qiskit.transpiler.init``.  The built-in plugins are:

.. list-table::
    :header-rows: 1

    * - Method
      - Summary

    * - :ref:`default <transpiler-preset-stage-init-default>`
      - Built-in unrolling of multi-qubit operations and abstract optimizations.


.. _transpiler-preset-stage-init-default:

Built-in ``default`` plugin
...........................

At optimization level 0, no abstract optimization is done.  The default plugin simply "unrolls"
operations with more than three qubits by accessing their hierarchical
:class:`~.circuit.Instruction.definition` fields.

At optimization levels 1 and above, the default plugin also does simple cancellation of adjacent
inverse gates, such as two back-to-back ``cx`` gates.

At optimization levels 2 and 3, the default plugin enables a much wider range of abstract
optimizations.  This includes:

* "virtual permutation elision" (see :class:`.ElidePermutations`), where explicit
  permutation-inducing operations are removed and instead effected as remapping of virtual qubits.
* analysis of the commutation structure of the IR to find pairs of gates that can be cancelled out.
* numerical splitting of two-qubit operations that can be expressed as series of separable one-qubit
  operations.
* removal of imperceivable operations, such as tiny-angle Pauli rotations and diagonal operations
  immediately preceding measurements.

.. _transpiler-preset-stage-layout:

Layout Stage
------------

.. seealso::
    `Layout stage explanation <https://docs.quantum.ibm.com/guides/transpiler-stages#layout-stage>`__
        Higher-level user-facing explanation of the layout stage in the IBM Quantum guide.

The layout stage is responsible for making an initial mapping between the virtual qubits of the
input circuit, and the hardware qubits of the target.  This includes expanding the input circuit
with explicit ancillas so it has as many qubits as the target has, and rewriting all operations in
terms of hardware qubits.  You may also see this problem called the "placement" problem in other
toolkits or literature.

The layout stage must set the properties ``layout`` and ``original_qubit_indices`` in the pipeline's
:class:`.PropertySet`.

.. note::

    All built-in plugins for the layout stage will defer to an explicit layout selected using the
    ``initial_layout`` argument to :func:`.generate_preset_pass_manager` or :func:`.transpile`.

At any given point in a circuit, we can identify a mapping between currently active "virtual" qubits
of the input circuit to hardware qubits of the backend.  A hardware qubit can only ever represent a
single virtual qubit at a given point, but the mapping might vary over the course of the circuit.
In principle, some virtual qubits may not necessarily be mapped at all points in the circuit
execution, if the lifetime of a virtual qubit state can be shortened, though Qiskit's built-in
pipelines do not use this currently.

.. image:: /source_images/mapping.png
    :alt: Illustration of how virtual qubits from an input circuit could be mapped to hardware
        qubits on a backend device's connectivity map.

The layout stage is not responsible for ensuring that the connectivity of the target of the target
is respected all the way through the circuit, nor that all operations are valid for direct execution
on the target; these are the responsibilities of the :ref:`routing
<transpiler-preset-stage-routing>` and :ref:`translation <transpiler-preset-stage-translation>`
stages, respectively.

The choice of initial layout is one of the most important factors that affects the quality of the
output circuit. The layout stage is often the most computationally expensive stage in the default
pipelines; the default plugin for layout even tries several different algorithms (described in more
detail in :ref:`transpiler-preset-stage-layout-default`).

The ideal situation for the layout stage is to find a "perfect" layout, which causes all operations
to already respect the connectivity constraints of the :class:`.Target` such that the routing stage
is not required.  This is typically not possible for arbitrary input circuits, but when it is, the
:class:`.VF2Layout` pass can be used to find an valid initial layout.  If multiple perfect layouts
are found, a scoring heuristic based on estimated error rates is used to decide.

In all built-in plugins, passing the :func:`.generate_preset_pass_manager` argument
``initial_layout`` causes the given layout to be used verbatim, skipping the individual "choosing"
logic.  All built-in plugins also handle embedding the circuit into the full width of the device,
including assigning ancillas.

If writing your own layout plugin, you might find :func:`.generate_embed_passmanager` useful for
automating the "embedding" stage of the layout application.

When writing :ref:`stage plugins <transpiler-preset-stage-plugins>`, the entry point for ``layout``
is ``qiskit.transpiler.layout``.  The built-in plugins are:

.. list-table::
    :header-rows: 1

    * - Method
      - Summary

    * - :ref:`default <transpiler-preset-stage-layout-default>`
      - At the highest optimization levels, attempts to find a perfect layout, then tries a
        Sabre-based layout-and-routing combined pass.

    * - :ref:`dense <transpiler-preset-stage-layout-dense>`
      - Finds the densest subgraph (in terms of qubit link degrees) of the backend to use as the
        initial qubits.

    * - :ref:`trivial <transpiler-preset-stage-layout-trivial>`
      - Maps virtual qubit 0 to physical qubit 0, etc.

    * - :ref:`sabre <transpiler-preset-stage-layout-sabre>`
      - Uses `Qiskit's enhanced Sabre layout algorithm <sabre-lightsabre-paper_>`_.

At all optimization levels, the default layout method is ``default``, though the structure of this
stage itself changes dramatically based on the level.

.. _transpiler-preset-stage-layout-default:

Built-in ``default`` plugin
...........................

An amalgamation of several different layout techniques.

At optimization level 0, the trivial layout is chosen.

At optimization levels above 0, there is a two-step process:

#. First, use :class:`.VF2Layout` to attempt to find a "perfect" layout.  The maximum number of
   calls to the isomorphism evaluator increases with optimization level.  For huge, complex targets,
   we are not guaranteed to find perfect layouts even if they exist, but the chance increases with
   the optimization level.

#. If no perfect layout can be found, use :class:`.SabreLayout` to choose an initial layout, with
   the numbers of initial layout trials, swap-map trials, and forwards–backwards iterations
   increasing with the optimization level.

In addition, optimization level also attempts the trivial layout before the VF2-based version, for
historical backwards compatibility.


.. _transpiler-preset-stage-layout-dense:

Built-in ``dense`` plugin
.........................

Uses the class:`.DenseLayout` pass to choose the layout.  This pass finds the densest connected
subgraph of the complete target connectivity graph, where "densest" means that hardware qubits with
the greatest number of available connections are preferred.  The virtual-to-hardware mapping is
completed by assigning the highest-degree virtual qubits to the highest-degree hardware qubits.

This is a relatively cheap heuristic for choosing an initial layout, but typically has far worse
output quality than Sabre-based methods.  The :ref:`default layout plugin
<transpiler-preset-stage-layout-default>` uses the initial mapping selected by :class:`.DenseLayout`
as one of its initial layouts to seed the Sabre algorithm.

.. _transpiler-preset-stage-layout-trivial:

Built-in ``trivial`` plugin
...........................

Uses the :class:`.TrivialLayout` pass to choose the layout.  This is the simplest assignment, where
each virtual qubit is assigned to the hardware qubit with the same index, so virtual qubit 0 is
mapped to hardware qubit 0, and so on.

This method is most useful for hardware-characterization experiments, where the incoming "abstract"
circuit is already full-width on the device, its operations correspond to physical operations, and
the transpiler is just being invoked to formalize the creation of a physical
:class:`.QuantumCircuit`.


.. _transpiler-preset-stage-layout-sabre:

Built-in ``sabre`` plugin
.........................

Uses the :class:`.SabreLayout` to choose an initial layout, using Qiskit's modified :ref:`Sabre
routing algorithm <transpiler-preset-stage-routing-sabre>` as the subroutine to swap-map the
candidate circuit both forwards and backwards.

Summarily, the layout component of `the original Sabre algorithm <sabre-original-paper_>`_ is to
choose an initial layout arbitrarily, then to "improve" it by running routing on the circuit,
reversing the circuit, and running routing on the reversed circuit with the previous "final"
virtual-to-hardware assignment as the initial state.  The configured optimization level decides how
many iterations of this to-and-fro we do, and how many different random initial layouts we try.

The principal difference to the :ref:`default stage <transpiler-preset-stage-layout-default>` at
optimization levels other than zero is that this plugin *only* runs the Sabre-based algorithm.  It
does not attempt to find a perfect layout, nor attempt the trivial layout.



.. _transpiler-preset-stage-routing:

Routing Stage
-------------

.. seealso::
    `Routing stage explanation <https://docs.quantum.ibm.com/guides/transpiler-stages#routing-stage>`__
        Higher-level user-facing explanation of the routing stage in the IBM Quantum guide.

The routing stage ensures that the virtual connectivity graph of the circuit is compatible with the
hardware connectivity graph of the target.  In simpler terms, the routing stage makes sure that all
two-qubit gates in the circuit take place on hardware qubits that have a defined two-qubit operation
in the target ISA.  You may also see this problem referred to as the "mapping" or "swap-mapping"
problem in other toolkits or literature.

Routing algorithms typically do this by inserting ``swap`` gates into the circuit, and modifying the
virtual-to-hardware mapping of qubits over the course of the circuit execution.

The routing stage does not need to ensure that all the gates in the circuit are valid for the target
ISA.  For example, a routing plugin can leave literal ``swap`` gates in the circuit, even if the
:class:`.Target` does not contain :class:`.SwapGate`.  However, there must be at least one two-qubit
gate defined in the :class:`.Target` for any pair of hardware qubits that has a gate applied in the
circuit.

The routing stage must set the properties ``final_layout`` and ``virtual_permutation_layout`` in
the :class:`.PropertySet` if routing has taken place.

All of Qiskit's built-in routing stages will additionally run the :class:`.VF2PostLayout` pass after
routing.  This may choose to reassign the initial layout, if lower-error qubits can be found.  This
pass is very similar to the :class:`.VF2Layout` class that :ref:`the default layout plugin
<transpiler-preset-stage-layout-default>` uses, except in :class:`.VF2PostLayout` we can guarantee
that there is at least one isomorphic induced subgraph of the target topology that matches the
circuit topology.

.. note::

    Qiskit's built-in routing plugins will all generally assume that all pairs of qubits with a
    defined two-qubit link have a *universal* set of gates defined for those two qubits.  Hardware
    does not necessarily need to respect this (for example, if the only defined two-qubit gate is
    ``swap``, then entangling operations like ``cx`` cannot be realised), but Qiskit does not yet
    consider this possibility.

.. note::

    Finding the minimal number of swaps to insert is known to be a non-polynomial problem.  This
    means it is prohibitively expensive to attempt, so many of Qiskit's built-in algorithms are
    stochastic, and you may see large variation between different compilations.  If you need
    reproducibility, be sure to set the ``seed_transpiler`` argument of
    :func:`.generate_preset_pass_manager` or :func:`.transpile`.

When writing :ref:`stage plugins <transpiler-preset-stage-plugins>`, the entry point for ``routing``
is ``qiskit.transpiler.routing``.  The built-in plugins are:

.. list-table::
    :header-rows: 1

    * - Method
      - Summary

    * - :ref:`sabre <transpiler-preset-stage-routing-sabre>`
      - Default.  Uses `Qiskit's modified Sabre routing algorithm <sabre-lightsabre-paper_>`_ to swap
        map.

    * - :ref:`none <transpiler-preset-stage-routing-none>`
      - Disable routing.  Raises an error if routing is required.

    * - :ref:`basic <transpiler-preset-stage-routing-basic>`
      - Greedy swap insertion to route a single operation at a time.

    * - :ref:`stochastic <transpiler-preset-stage-routing-stochastic>`
      - Consider operations layer-by-layer, using a stochastic algorithm to find swap networks that
        implement a suitable permutation to make the layer executable.

    * - :ref:`lookahead <transpiler-preset-stage-routing-lookahead>`
      - Breadth-first search with heuristic pruning to find swaps that make gates executable.

.. _transpiler-preset-stage-routing-none:

Built-in ``none`` plugin
........................

A dummy plugin used to disable routing entirely.  This can occasionally be useful for
hardware-configuration experiments, or in certain special cases of partial compilation.

.. _transpiler-preset-stage-routing-basic:

Built-in ``basic`` plugin
.........................

Uses the :class:`.BasisSwap` greedy swap-insertion algorithm.  This is conceptually very simple; for
each operation in topological order, insert the shortest-path swaps needed to make the connection
executable on the device.

The optimization level only affects the amount of work the :class:`.VF2PostLayout` step does to
attempt to improve the initial layout after routing.

This method typically has incredibly poor output quality.

.. _transpiler-preset-stage-routing-stochastic:

Built-in ``stochastic`` plugin
..............................

Uses the :class:`.StochasticSwap` algorithm to route.  In short, this stratifies the circuit into
layers, then uses a stochastic algorithm to find a permutation that will allow the layer to execute,
and a series of swaps that will implement that permutation in a hardware-valid way.

The optimization level affects the number of stochastic trials used for each layer, and the amount
of work spent in :class:`.VF2PostLayout` to optimize the initial layout.

This was Qiskit's primary routing algorithm for several years, until approximately 2021.  Now, it
is reliably beaten in runtime and output quality by :ref:`Qiskit's custom Sabre-based routing
algorithm <transpiler-preset-stage-routing-sabre>`.

.. _transpiler-preset-stage-routing-lookahead:

Built-in ``lookahead`` plugin
.............................

Uses the :class:`.LookaheadSwap` algorithm to route.  Approximately, this is a breadth-first search
at producing a swap network, where the tree being explored is pruned down to a small number of
candidate swaps at each depth.

This algorithm is somewhat akin to the ``basic`` heuristic of :ref:`the "sabre" plugin
<transpiler-preset-stage-routing-sabre>`, except it considers the following effects of each swap to
a small depth as well.

The optimization level affects the search depth, the amount of per-depth pruning, and amount of work
done by :class:`.VF2PostLayout` to post-optimize the initial layout.

In practice, :ref:`the "sabre" plugin <transpiler-preset-stage-routing-sabre>` runs several orders
of magnitude faster, and produces better output.

.. _transpiler-preset-stage-routing-sabre:

Built-in ``sabre`` plugin
.........................

Uses the :class:`.SabreSwap` algorithm to route.  This uses `Qiskit's enhanced version
<sabre-lightsabre-paper_>`_ of `the original Sabre routing algorithm <sabre-original-paper_>`_.

This routing algorithm runs with threaded parallelism to consider several different possibilities
for routing, choosing the one that minimizes the number of inserted swaps.

The optimization level affects how many different stochastic seeds are attempted for the full
routing, and the amount of work done by :class:`.VF2PostLayout` to post-optimize the initial layout.

This is almost invariably the best-performing built-in plugin, and the one Qiskit uses by default in
all cases where routing is necessary.

.. _transpiler-preset-stage-translation:

Translation Stage
-----------------

.. seealso::
    `Translation stage explanation`__
        Higher-level user-facing explanation of the translation stage in the IBM Quantum guide.

.. __: https://docs.quantum.ibm.com/guides/transpiler-stages#translation-stage

The translation stage is responsible for rewriting all gates in the circuit into ones that are
supported by the target ISA.  For example, if a ``cx`` is requested on hardware qubits 0 and 1, but
the ISA only contains a ``cz`` operation on those qubits, the translation stage must find a way of
representing the ``cx`` gate using the ``cz`` and available one-qubit gates.

.. note::

    In the Qiskit 1.x series, translation plugins need not output gates with the correct
    directionality, provided the gate exists with opposite directionality on the given qubit pair.
    For example, if ``cx(0, 1)`` is ISA-supported, the translation stage is permitted to output
    ``cx(1, 0)``.

    This is likely to change in later versions of Qiskit.

The translation stage is called before entering the optimization stage. Optimization plugins
(including Qiskit's built-in plugins) may also use the translation stage as a "fixup" stage after
the optimization loop, if the optimization loop returns a circuit that includes non-ISA gates.  This
latter situation is fairly common; the optimization loop may only be concerned with minimizing
properties like "number of two qubit gates", and will leave its output in terms of locally
equivalent gates, which the translation stage can easily rewrite without affecting the target
optimization properties.  This allows easier separation of concerns between the two stages.  Some
optimization plugins may be stricter in their output, and so this follow-up to the translation stage
may no longer be necessary.

When writing :ref:`stage plugins <transpiler-preset-stage-plugins>`, the entry point for
``translation`` is ``qiskit.transpiler.translation``.  The built-in plugins are:

.. list-table::
    :header-rows: 1

    * - Method
      - Summary

    * - :ref:`translator <transpiler-preset-stage-translation-translator>`
      - Symbolic translation of gates to the target basis using known equivalences.

    * - :ref:`synthesis <transpiler-preset-stage-translation-synthesis>`
      - Collect each run of one- and two-qubit gates into a matrix representation, and resynthesis
        from there.

.. _transpiler-preset-stage-translation-synthesis:

Built-in ``synthesis`` plugin
.............................

Collect runs of gates on the same qubits into matrix form, and then resynthesise using the
:class:`.UnitarySynthesis` pass (with the configured ``unitary_synthesis_method``).  This is, in
large part, similar to the optimization loop itself at high optimization levels.

The collection to matrices is typically more expensive than matrix-free translations, but in
principle the quality of the translations can be better.  In practice, this requires a synthesis
algorithm tailored to the target ISA, which makes this method less general than other methods. It
can produce higher quality results when targeting simple ISAs that match the synthesis routines
already in Qiskit.

If this method is used, one often does not need the optimization loop at all.

The optimization level has no effect on this plugin.


.. _transpiler-preset-stage-translation-translator:

Built-in ``translator`` plugin
..............................

Uses the :class:`.BasisTranslator` algorithm to symbolically translate gates into the target basis.
At a high level, this starts from the set of gates requested by the circuit, and uses rules from a
given :class:`.EquivalenceLibrary` (typically the :data:`.SessionEquivalenceLibrary`) to move
towards the ISA.

This is the default translation method.

The optimization level has no effect on this plugin.


.. _transpiler-preset-stage-optimization:

Optimization Stage
------------------

.. seealso::
    `Optimization stage explanation`__
        Higher-level user-facing explanation of the optimization stage in the IBM Quantum guide.

.. __: https://docs.quantum.ibm.com/guides/transpiler-stages#optimization-stage

The optimization stage is for low level hardware-aware optimizations.  Unlike :ref:`the init stage
<transpiler-preset-stage-init>`, the input to this stage is a circuit that is already
ISA-compatible, so a low-level optimization plugin can be tailored for a particular ISA.

There are very few requirements on an optimization plugin, other than it takes in ISA-supported
circuits, and returns ISA-supported circuits.  An optimization plugin will often contain a loop,
such as the :class:`.DoWhileController`, and may choose to include the configured translation stage
as a fix-up pipeline.

Qiskit's built-in optimization plugins are general, and apply well to most real-word ISAs for
non-error-corrected devices.  The built-in plugins are less well suited to ISAs that have no
continuously parametrized single-qubit gate, such as a Clifford+T basis set.

When writing :ref:`stage plugins <transpiler-preset-stage-plugins>`, the entry point for
``optimization`` is ``qiskit.transpiler.optimization``.  The built-in plugins are:

.. list-table::
    :header-rows: 1

    * - Method
      - Summary

    * - :ref:`default <transpiler-preset-stage-optimization-default>`
      - A default set of optimization passes.  This varies significantly between optimization
        levels.

.. _transpiler-preset-stage-optimization-default:

Built-in ``default`` plugin
...........................

This varies significantly between optimization levels.

The specifics of this pipeline are subject to change between Qiskit versions. The broad principles
are described below.

At optimization level 0, the stage is empty.

At optimization level 1, the stage does matrix-based resynthesis of runs of 1q gates, and very
simply symbolic inverse cancellation of two-qubit gates, if they appear consecutively.  This runs
in a loop until the size and depth of the circuit are fixed.

At optimization level 2, in addition the optimizations of level 1, the loop contains commutation
analysis of sets of gates to widen the range of gates that can be considered for cancellation.
Before the loop, runs of both one- and two-qubit gates undergo a single matrix-based resynthesis.

At optimization level 3, the two-qubit matrix-based resynthesis runs inside the optimization loop.
The optimization loop condition also tries multiple runs and chooses the minimum point in the case
of fluctuating output; this is necessary because matrix-based resynthesis is relatively unstable in
terms of concrete gates.

Optimization level 3 is typically very expensive for large circuits.


.. _transpiler-preset-stage-scheduling:

Scheduling Stage
----------------

.. seealso::
    :ref:`transpiler-scheduling-description`
        A guide-level explanation of scheduling concepts.

The scheduling stage, if requested, is responsible for inserting explicit :class:`~.circuit.Delay`
instructions to make idle periods of qubits explicit.  Plugins may optionally choose to do
walltime-sensitive transformations, such as inserting dynamical decoupling sequences.

The input to the scheduling stage is an ISA-compatible circuit.  The output of the scheduling stage
must also be an ISA-compatible circuit, with explicit :class:`~.circuit.Delay` instructions that
satisfy the hardware's timing information, if appropriate.

The scheduling stage should set the ``node_start_time`` property in the pipeline's
:class:`.PropertySet`.

When writing :ref:`stage plugins <transpiler-preset-stage-plugins>`, the entry point for
``scheduling`` is ``qiskit.transpiler.scheduling``.  The built-in plugins are:

.. list-table::
    :header-rows: 1

    * - Method
      - Summary

    * - :ref:`default <transpiler-preset-stage-scheduling-default>`
      - Attempt to satisfy timing alignment constraints without otherwise scheduling

    * - :ref:`alap <transpiler-preset-stage-scheduling-alap>`
      - Schedule the circuit preferring operations to be as late as possible.

    * - :ref:`asap <transpiler-preset-stage-scheduling-asap>`
      - Schedule the circuit preferring operations to be as soon as possible.

.. _transpiler-preset-stage-scheduling-default:

Built-in ``default`` plugin
...........................

Do nothing, unless the circuit already contains instructions with explicit timings.  If there are
explicitly timed operations in the circuit, insert additional padding to ensure that these timings
satisfy the alignment and other constraints of the hardware.

.. _transpiler-preset-stage-scheduling-alap:

Builtin ``alap`` plugin
.......................

Explicitly schedule all operations using an "as late as possible" strategy.  This uses the
:class:`.ALAPScheduleAnalysis` algorithm to decide where to place gates.

.. _transpiler-preset-stage-scheduling-asap:

Builtin ``asap`` plugin
.......................

Explicitly schedule all operations using an "as soon as possible" strategy.  This uses the
:class:`.ASAPScheduleAnalysis` algorithm to decide where to place gates.


Custom Pass Managers
====================

In addition to modifying preset pass managers, it is also possible to construct a pass
manager to build an entirely custom pipeline for transforming input
circuits. You can use the :class:`~.StagedPassManager` class directly to do
this. You can define arbitrary stage names and populate them with a :class:`~.PassManager`
instance. For example, the following code creates a new :class:`~.StagedPassManager`
that has 2 stages, ``init`` and ``translation``.

.. code-block::

    from qiskit.transpiler.passes import (
        UnitarySynthesis,
        Collect2qBlocks,
        ConsolidateBlocks,
        UnitarySynthesis,
        Unroll3qOrMore,
    )
    from qiskit.transpiler import PassManager, StagedPassManager

    basis_gates = ["rx", "ry", "rxx"]
    init = PassManager([UnitarySynthesis(basis_gates, min_qubits=3), Unroll3qOrMore()])
    translate = PassManager(
        [
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=basis_gates),
            UnitarySynthesis(basis_gates),
        ]
    )

    staged_pm = StagedPassManager(
        stages=["init", "translation"], init=init, translation=translate
    )

There is no limit on the number of stages you can put in a :class:`~.StagedPassManager`.  The stages
do not need to correspond to the stages used by Qiskit's preset pipelines.

The :ref:`stage_generators` may be useful for the construction of custom :class:`~.StagedPassManager`s.
They generate pass managers which provide common functionality used in many stages.
For example, :func:`~.generate_embed_passmanager` generates a :class:`~.PassManager`
to "embed" a selected initial :class:`~.Layout` from a layout pass to the specified target device.

Representing Quantum Computers
==============================

To be able to compile a :class:`~.QuantumCircuit` for a specific backend, the transpiler needs a
specialized representation of that backend, including its constraints, instruction set, qubit
properties, and more, to be able to compile and optimize effectively. While the
:class:`~.BackendV2` class defines an interface for querying and interacting
with backends, its scope is larger than just the transpiler's needs including
managing job submission and potentially interfacing with remote services.
The specific information needed by the transpiler is described by the
:class:`~.Target` class

For example, to construct a simple :class:`~.Target` object, one can iteratively add
descriptions of the instructions it supports:

.. plot::
   :include-source:
   :nofigs:

    from qiskit.circuit import Parameter, Measure
    from qiskit.transpiler import Target, InstructionProperties
    from qiskit.circuit.library import UGate, RZGate, RXGate, RYGate, CXGate, CZGate

    target = Target(num_qubits=3)
    target.add_instruction(CXGate(), {(0, 1): InstructionProperties(error=.0001, duration=5e-7)})
    target.add_instruction(
        UGate(Parameter('theta'), Parameter('phi'), Parameter('lam')),
        {
            (0,): InstructionProperties(error=.00001, duration=5e-8),
            (1,): InstructionProperties(error=.00002, duration=6e-8)
        }
    )
    target.add_instruction(
        RZGate(Parameter('theta')),
        {
            (1,): InstructionProperties(error=.00001, duration=5e-8),
            (2,): InstructionProperties(error=.00002, duration=6e-8)
        }
    )
    target.add_instruction(
        RYGate(Parameter('theta')),
        {
            (1,): InstructionProperties(error=.00001, duration=5e-8),
            (2,): InstructionProperties(error=.00002, duration=6e-8)
        }
    )
    target.add_instruction(
        RXGate(Parameter('theta')),
        {
            (1,): InstructionProperties(error=.00001, duration=5e-8),
            (2,): InstructionProperties(error=.00002, duration=6e-8)
        }
    )
    target.add_instruction(
        CZGate(),
        {
            (1, 2): InstructionProperties(error=.0001, duration=5e-7),
            (2, 0): InstructionProperties(error=.0001, duration=5e-7)
        }
    )
    target.add_instruction(
        Measure(),
        {
            (0,): InstructionProperties(error=.001, duration=5e-5),
            (1,): InstructionProperties(error=.002, duration=6e-5),
            (2,): InstructionProperties(error=.2, duration=5e-7)
        }
    )
    print(target)

.. code-block:: text

    Target
    Number of qubits: 3
    Instructions:
        cx
            (0, 1):
                Duration: 5e-07 sec.
                Error Rate: 0.0001
        u
            (0,):
                Duration: 5e-08 sec.
                Error Rate: 1e-05
            (1,):
                Duration: 6e-08 sec.
                Error Rate: 2e-05
        rz
            (1,):
                Duration: 5e-08 sec.
                Error Rate: 1e-05
            (2,):
                Duration: 6e-08 sec.
                Error Rate: 2e-05
        ry
            (1,):
                Duration: 5e-08 sec.
                Error Rate: 1e-05
            (2,):
                Duration: 6e-08 sec.
                Error Rate: 2e-05
        rx
            (1,):
                Duration: 5e-08 sec.
                Error Rate: 1e-05
            (2,):
                Duration: 6e-08 sec.
                Error Rate: 2e-05
        cz
            (1, 2):
                Duration: 5e-07 sec.
                Error Rate: 0.0001
            (2, 0):
                Duration: 5e-07 sec.
                Error Rate: 0.0001
        measure
            (0,):
                Duration: 5e-05 sec.
                Error Rate: 0.001
            (1,):
                Duration: 6e-05 sec.
                Error Rate: 0.002
            (2,):
                Duration: 5e-07 sec.
                Error Rate: 0.2

This :class:`~.Target` represents a 3 qubit backend that supports :class:`~.CXGate` between qubits
0 and 1, :class:`~.UGate` on qubits 0 and 1, :class:`~.RZGate`, :class:`~.RXGate`,
and :class:`~.RYGate` on qubits 1 and 2, :class:`~.CZGate` between qubits 1 and 2, and qubits
2 and 0, and :class:`~.Measure` on all qubits.

There are also specific data structures to represent a specific subset of information from the
:class:`~.Target`. For example, the :class:`~.CouplingMap` class is used to solely represent the
connectivity constraints of a backend as a directed graph. A coupling map can be generated from
a :class:`~.Target` using the :meth:`.Target.build_coupling_map` method. These data structures
typically pre-date the :class:`~.Target` class but are still used by some transpiler passes that do
not work natively with a :class:`~.Target` instance yet or when dealing with backends that aren't
using the latest :class:`~.BackendV2` interface.

For example, if we wanted to visualize the :class:`~.CouplingMap` for the
example 3 qubit :class:`~.Target` above:

.. plot::
   :include-source:

   from qiskit.circuit import Parameter, Measure
   from qiskit.transpiler import Target, InstructionProperties
   from qiskit.circuit.library import UGate, RZGate, RXGate, RYGate, CXGate, CZGate

   target = Target(num_qubits=3)
   target.add_instruction(CXGate(), {(0, 1): InstructionProperties(error=.0001, duration=5e-7)})
   target.add_instruction(
       UGate(Parameter('theta'), Parameter('phi'), Parameter('lam')),
       {
           (0,): InstructionProperties(error=.00001, duration=5e-8),
           (1,): InstructionProperties(error=.00002, duration=6e-8)
       }
   )
   target.add_instruction(
       RZGate(Parameter('theta')),
       {
           (1,): InstructionProperties(error=.00001, duration=5e-8),
           (2,): InstructionProperties(error=.00002, duration=6e-8)
       }
   )
   target.add_instruction(
       RYGate(Parameter('theta')),
       {
           (1,): InstructionProperties(error=.00001, duration=5e-8),
           (2,): InstructionProperties(error=.00002, duration=6e-8)
       }
   )
   target.add_instruction(
       RXGate(Parameter('theta')),
       {
           (1,): InstructionProperties(error=.00001, duration=5e-8),
           (2,): InstructionProperties(error=.00002, duration=6e-8)
       }
   )
   target.add_instruction(
       CZGate(),
       {
           (1, 2): InstructionProperties(error=.0001, duration=5e-7),
           (2, 0): InstructionProperties(error=.0001, duration=5e-7)
       }
   )
   target.add_instruction(
       Measure(),
       {
           (0,): InstructionProperties(error=.001, duration=5e-5),
           (1,): InstructionProperties(error=.002, duration=6e-5),
           (2,): InstructionProperties(error=.2, duration=5e-7)
       }
   )

   target.build_coupling_map().draw()

This shows the global connectivity of the :class:`~.Target` which is the
combination of the supported qubits for :class:`~.CXGate` and :class:`~.CZGate`. To
see the individual connectivity, you can pass the operation name to
:meth:`.CouplingMap.build_coupling_map`:

.. plot::
   :include-source:

   from qiskit.circuit import Parameter, Measure
   from qiskit.transpiler import Target, InstructionProperties
   from qiskit.circuit.library import UGate, RZGate, RXGate, RYGate, CXGate, CZGate

   target = Target(num_qubits=3)
   target.add_instruction(CXGate(), {(0, 1): InstructionProperties(error=.0001, duration=5e-7)})
   target.add_instruction(
       UGate(Parameter('theta'), Parameter('phi'), Parameter('lam')),
       {
           (0,): InstructionProperties(error=.00001, duration=5e-8),
           (1,): InstructionProperties(error=.00002, duration=6e-8)
       }
   )
   target.add_instruction(
       RZGate(Parameter('theta')),
       {
           (1,): InstructionProperties(error=.00001, duration=5e-8),
           (2,): InstructionProperties(error=.00002, duration=6e-8)
       }
   )
   target.add_instruction(
       RYGate(Parameter('theta')),
       {
           (1,): InstructionProperties(error=.00001, duration=5e-8),
           (2,): InstructionProperties(error=.00002, duration=6e-8)
       }
   )
   target.add_instruction(
       RXGate(Parameter('theta')),
       {
           (1,): InstructionProperties(error=.00001, duration=5e-8),
           (2,): InstructionProperties(error=.00002, duration=6e-8)
       }
   )
   target.add_instruction(
       CZGate(),
       {
           (1, 2): InstructionProperties(error=.0001, duration=5e-7),
           (2, 0): InstructionProperties(error=.0001, duration=5e-7)
       }
   )
   target.add_instruction(
       Measure(),
       {
           (0,): InstructionProperties(error=.001, duration=5e-5),
           (1,): InstructionProperties(error=.002, duration=6e-5),
           (2,): InstructionProperties(error=.2, duration=5e-7)
       }
   )

   target.build_coupling_map('cx').draw()

.. plot::
   :include-source:

   from qiskit.circuit import Parameter, Measure
   from qiskit.transpiler import Target, InstructionProperties
   from qiskit.circuit.library import UGate, RZGate, RXGate, RYGate, CXGate, CZGate

   target = Target(num_qubits=3)
   target.add_instruction(CXGate(), {(0, 1): InstructionProperties(error=.0001, duration=5e-7)})
   target.add_instruction(
       UGate(Parameter('theta'), Parameter('phi'), Parameter('lam')),
       {
           (0,): InstructionProperties(error=.00001, duration=5e-8),
           (1,): InstructionProperties(error=.00002, duration=6e-8)
       }
   )
   target.add_instruction(
       RZGate(Parameter('theta')),
       {
           (1,): InstructionProperties(error=.00001, duration=5e-8),
           (2,): InstructionProperties(error=.00002, duration=6e-8)
       }
   )
   target.add_instruction(
       RYGate(Parameter('theta')),
       {
           (1,): InstructionProperties(error=.00001, duration=5e-8),
           (2,): InstructionProperties(error=.00002, duration=6e-8)
       }
   )
   target.add_instruction(
       RXGate(Parameter('theta')),
       {
           (1,): InstructionProperties(error=.00001, duration=5e-8),
           (2,): InstructionProperties(error=.00002, duration=6e-8)
       }
   )
   target.add_instruction(
       CZGate(),
       {
           (1, 2): InstructionProperties(error=.0001, duration=5e-7),
           (2, 0): InstructionProperties(error=.0001, duration=5e-7)
       }
   )
   target.add_instruction(
       Measure(),
       {
           (0,): InstructionProperties(error=.001, duration=5e-5),
           (1,): InstructionProperties(error=.002, duration=6e-5),
           (2,): InstructionProperties(error=.2, duration=5e-7)
       }
   )

   target.build_coupling_map('cz').draw()


.. _transpiler-scheduling-description:

Scheduling of Circuits
======================

..
    This section is still here because the content hasn't fully migrated to other places yet, unlike
    other discussions of the components of quantum compilation.

.. seealso::
    :ref:`transpiler-preset-stage-scheduling`
        How to configure the scheduling stages of the preset pass managers.

After the circuit has been translated to the target basis, mapped to the device, and optimized,
a scheduling phase can be applied to optionally account for all the idle time in the circuit.
At a high level, the scheduling can be thought of as inserting delays into the circuit to account
for idle time on the qubits between the execution of instructions. For example, if we start with a
circuit such as:

.. plot::

   from qiskit import QuantumCircuit

   ghz = QuantumCircuit(5)
   ghz.h(0)
   ghz.cx(0,range(1,5))
   ghz.draw(output='mpl')

we can then call :func:`~.transpile` on it with ``scheduling_method`` set:

.. plot::
   :include-source:

   from qiskit import QuantumCircuit, transpile
   from qiskit.providers.fake_provider import GenericBackendV2

   backend = GenericBackendV2(5)

   ghz = QuantumCircuit(5)
   ghz.h(0)
   ghz.cx(0,range(1,5))

   circ = transpile(ghz, backend, scheduling_method="asap")
   circ.draw(output='mpl')

You can see here that the transpiler inserted :class:`~qiskit.circuit.Delay` instructions to
account for idle time on each qubit. To get a better idea of the timing of the circuit we can
also look at it with the :func:`.timeline.draw` function:

.. plot::

   from qiskit.visualization.timeline import draw as timeline_draw

   from qiskit import QuantumCircuit, transpile
   from qiskit.providers.fake_provider import GenericBackendV2

   backend = GenericBackendV2(5)

   ghz = QuantumCircuit(5)
   ghz.h(0)
   ghz.cx(0,range(1,5))

   circ = transpile(ghz, backend, scheduling_method="asap")

   timeline_draw(circ)

The scheduling of a circuit involves two parts: analysis and constraint mapping, followed by a
padding pass. The first part requires running a scheduling analysis pass such as
:class:`~.ALAPSchedulingAnalysis` or :class:`~.ASAPSchedulingAnalysis` which analyzes the circuit
and records the start time of each instruction in the circuit using a scheduling algorithm ("as late
as possible" for  :class:`~.ALAPSchedulingAnalysis` and "as soon as possible" for
:class:`~.ASAPSchedulingAnalysis`) in the property set. Once the circuit has an initial scheduling,
additional passes can be run to account for any timing constraints on the target backend, such
as alignment constraints. This is typically done with the
:class:`~.ConstrainedReschedule` pass which will adjust the scheduling
set in the property set to the constraints of the target backend. Once all
the scheduling and adjustments/rescheduling are finished, a padding pass,
such as :class:`~.PadDelay` or :class:`~.PadDynamicalDecoupling` is run
to insert the instructions into the circuit, which completes the scheduling.

Scheduling Analysis with control flow instructions
--------------------------------------------------

When running scheduling analysis passes on a circuit, you must keep in mind that there
are additional constraints on classical conditions and control flow instructions. This section
covers the details of these additional
constraints that any scheduling pass will need to account for.

Topological node ordering in scheduling
.......................................

The DAG representation of ``QuantumCircuit`` respects the node ordering in the
classical register wires, though theoretically two conditional instructions
conditioned on the same register could commute, i.e. read-access to the
classical register doesn't change its state.

.. code-block:: text

    qc = QuantumCircuit(2, 1)
    qc.delay(100, 0)
    qc.x(0).c_if(0, True)
    qc.x(1).c_if(0, True)

The scheduler SHOULD comply with the above topological ordering policy of the
DAG circuit.
Accordingly, the `asap`-scheduled circuit will become

.. code-block:: text

         ┌────────────────┐   ┌───┐
    q_0: ┤ Delay(100[dt]) ├───┤ X ├──────────────
         ├────────────────┤   └─╥─┘      ┌───┐
    q_1: ┤ Delay(100[dt]) ├─────╫────────┤ X ├───
         └────────────────┘     ║        └─╥─┘
                           ┌────╨────┐┌────╨────┐
    c: 1/══════════════════╡ c_0=0x1 ╞╡ c_0=0x1 ╞
                           └─────────┘└─────────┘

Note that this scheduling might be inefficient in some cases,
because the second conditional operation could start without waiting
for the 100 dt delay.
However, any additional optimization should be done in a different pass,
not to break the topological ordering of the original circuit.

Realistic control flow scheduling (respecting microarchitecture)
................................................................

In the dispersive QND readout scheme, the qubit (Q) is measured by sending
a microwave stimulus, followed by a resonator ring-down (depopulation). This
microwave signal is recorded in the buffer memory (B) with the hardware kernel,
then a discriminated (D) binary value is moved to the classical register (C).
A sequence from t0 to t1 of the measure instruction interval could be
modeled as follows:

.. code-block:: text

    Q ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░
    B ░░▒▒▒▒▒▒▒▒░░░░░░░░░
    D ░░░░░░░░░░▒▒▒▒▒▒░░░
    C ░░░░░░░░░░░░░░░░▒▒░

However, the :class:`.QuantumCircuit` representation is not accurate enough to represent
this model. In the circuit representation, the corresponding :class:`.circuit.Qubit` is occupied
by the stimulus microwave signal during the first half of the interval,
and the :class:`.Clbit` is only occupied at the very end of the interval.

The lack of precision representing the physical model may induce
edge cases in the scheduling:

.. code-block:: text

            ┌───┐
    q_0: ───┤ X ├──────
            └─╥─┘   ┌─┐
    q_1: ─────╫─────┤M├
         ┌────╨────┐└╥┘
    c: 1/╡ c_0=0x1 ╞═╩═
         └─────────┘ 0

In this example, a user may intend to measure the state of ``q_1`` after the
:class:`.XGate` is applied to ``q_0``. This is the correct interpretation from
the viewpoint of topological node ordering, i.e. The :class:`.XGate` node comes in
front of the :class:`.Measure` node.
However, according to the measurement model above, the data in the register
is unchanged during the application of the stimulus, so two nodes are
simultaneously operated.
If one tries to `alap`-schedule this circuit, it may return following circuit:

.. code-block:: text

         ┌────────────────┐   ┌───┐
    q_0: ┤ Delay(500[dt]) ├───┤ X ├──────
         └────────────────┘   └─╥─┘   ┌─┐
    q_1: ───────────────────────╫─────┤M├
                           ┌────╨────┐└╥┘
    c: 1/══════════════════╡ c_0=0x1 ╞═╩═
                           └─────────┘ 0

Note that there is no delay on the ``q_1`` wire, and the measure instruction
immediately starts after t=0, while the conditional gate starts after the delay.
It looks like the topological ordering between the nodes is flipped in the
scheduled view.
This behavior can be understood by considering the control flow model described above,

.. code-block:: text

    : Quantum Circuit, first-measure
    0 ░░░░░░░░░░░░▒▒▒▒▒▒░
    1 ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░

    : In wire q0
    Q ░░░░░░░░░░░░░░░▒▒▒░
    C ░░░░░░░░░░░░▒▒░░░░░

    : In wire q1
    Q ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░
    B ░░▒▒▒▒▒▒▒▒░░░░░░░░░
    D ░░░░░░░░░░▒▒▒▒▒▒░░░
    C ░░░░░░░░░░░░░░░░▒▒░

Since there is no qubit register overlap between Q0 and Q1, the node ordering is
determined by the shared classical register C. As you can see, the execution order is still
preserved on C, i.e. read C then apply ``XGate``, finally store the measured outcome in C.
But because ``DAGOpNode`` cannot define different durations for the associated registers,
the time ordering of the two nodes is inverted.

This behavior can be controlled by ``clbit_write_latency`` and ``conditional_latency``.
``clbit_write_latency`` determines the delay of the register write-access from
the beginning of the measure instruction (t0), while ``conditional_latency`` determines
the delay of conditional gate operations with respect to t0, which is determined
by the register read-access.
This information is accessible in the backend configuration and should
be copied to the pass manager property set before the pass is called.

Due to default latencies, the `alap`-scheduled circuit of above example may become

.. code-block:: text

            ┌───┐
    q_0: ───┤ X ├──────
            └─╥─┘   ┌─┐
    q_1: ─────╫─────┤M├
         ┌────╨────┐└╥┘
    c: 1/╡ c_0=0x1 ╞═╩═
         └─────────┘ 0

If the backend microarchitecture supports smart scheduling of the control flow
instructions, such as separately scheduling qubits and classical registers,
the insertion of the delay yields an unnecessarily longer total execution time.

.. code-block:: text

    : Quantum Circuit, first-XGate
    0 ░▒▒▒░░░░░░░░░░░░░░░
    1 ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░

    : In wire q0
    Q ░▒▒▒░░░░░░░░░░░░░░░
    C ░░░░░░░░░░░░░░░░░░░ (zero latency)

    : In wire q1
    Q ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░
    C ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░ (zero latency, scheduled after C0 read-access)

However, this result is much more intuitive in the topological ordering view.
If a finite conditional latency value is provided, for example, 30 dt, the circuit
is scheduled as follows:

.. code-block:: text

         ┌───────────────┐   ┌───┐
    q_0: ┤ Delay(30[dt]) ├───┤ X ├──────
         ├───────────────┤   └─╥─┘   ┌─┐
    q_1: ┤ Delay(30[dt]) ├─────╫─────┤M├
         └───────────────┘┌────╨────┐└╥┘
    c: 1/═════════════════╡ c_0=0x1 ╞═╩═
                          └─────────┘ 0

with the timing model:

.. code-block:: text

    : Quantum Circuit, first-xgate
    0 ░░▒▒▒░░░░░░░░░░░░░░░
    1 ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░

    : In wire q0
    Q ░░▒▒▒░░░░░░░░░░░░░░░
    C ░▒░░░░░░░░░░░░░░░░░░ (30dt latency)

    : In wire q1
    Q ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░
    C ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░

See https://arxiv.org/abs/2102.01682 for more details.

Transpiler API
==============

Hardware Description
--------------------

.. autosummary::
   :toctree: ../stubs/

   Target
   InstructionProperties

Pass Manager Definition
-----------------------

.. autosummary::
   :toctree: ../stubs/

   StagedPassManager
   PassManager
   PassManagerConfig
   generate_preset_pass_manager

Layout and Topology
-------------------

.. autosummary::
   :toctree: ../stubs/

   Layout
   CouplingMap
   TranspileLayout

Scheduling
----------

.. autosummary::
   :toctree: ../stubs/

   InstructionDurations

Abstract Passes
---------------

.. autosummary::
   :toctree: ../stubs/

   TransformationPass
   AnalysisPass

Exceptions
----------

.. autoexception:: TranspilerError
.. autoexception:: TranspilerAccessError
.. autoexception:: CouplingError
.. autoexception:: LayoutError
.. autoexception:: CircuitTooWideForTarget
.. autoexception:: InvalidLayoutError

.. _sabre-original-paper: https://arxiv.org/abs/1809.02573
.. _sabre-lightsabre-paper: https://arxiv.org/abs/2409.08368
"""

# For backward compatibility
from qiskit.passmanager import (
    ConditionalController,
    DoWhileController,
)
from qiskit.passmanager.compilation_status import PropertySet

from .passmanager import PassManager, StagedPassManager
from .passmanager_config import PassManagerConfig
from .exceptions import (
    TranspilerError,
    TranspilerAccessError,
    CouplingError,
    LayoutError,
    CircuitTooWideForTarget,
    InvalidLayoutError,
)
from .basepasses import AnalysisPass, TransformationPass
from .coupling import CouplingMap
from .layout import Layout, TranspileLayout
from .instruction_durations import InstructionDurations
from .preset_passmanagers import generate_preset_pass_manager
from .target import Target
from .target import InstructionProperties
from .target import QubitProperties

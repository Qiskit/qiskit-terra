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

.. raw:: html

   <br>

Qiskit has four pre-built transpilation pipelines available here:
:mod:`qiskit.transpiler.preset_passmanagers`.  Unless the reader is familiar with
quantum circuit optimization methods and their usage, it is best to use one of
these ready-made routines. By default the preset pass managers are composed
of six stages:

#. ``init`` - This stage runs any initial passes that are required before we start embedding the
   circuit to the backend. This typically involves unrolling custom instructions and converting
   the circuit to all 1 and 2 qubit gates.
#. ``layout`` - This stage applies a layout, mapping the virtual qubits in the circuit to the
   physical qubits on a backend. See :ref:`layout_stage` for more details.
#. ``routing`` - This stage runs after a layout has been applied and will inject
   gates (i.e. swaps) into the original circuit to make it compatible
   with the backend's connectivity. See :ref:`routing_stage` for more details.
#. ``translation`` - This stage translates the gates in the circuit to the target backend's basis set.
   See :ref:`translation_stage` for more details.
#. ``optimization`` - This stage runs the main optimization loop repeatedly
   until a condition (such as fixed depth) is reached. See :ref:`optimization_stage` for more details.
#. ``scheduling`` - This stage is for any hardware-aware scheduling passes. See
   :ref:`scheduling_stage` for more details.

When using :func:`~.transpile`, the implementation of each stage can be modified with the ``*_method``
arguments (e.g. ``layout_method``). These can be set to one of the built-in methods and
can also refer to available external plugins. See
:mod:`qiskit.transpiler.preset_passmanagers.plugin` for details on this plugin interface.

.. _working_with_preset_pass_managers:

Working with Preset Pass Managers
=================================

Qiskit includes functions to build preset :class:`~.PassManager` objects.
These preset passmanagers are used by the :func:`~.transpile` function
for each optimization level. There are 4 optimization levels ranging from 0 to 3, where higher
optimization levels take more time and computational effort but may yield a
more optimal circuit.
Optimization level 0 is intended for device characterization experiments and, as such, only
maps the input circuit to the constraints of the target backend, without
performing any optimizations. Optimization level 3 spends the most effort to optimize the circuit.
However, as many of the optimization techniques in the transpiler are heuristic based, spending more
computational effort does not always result in an improvement in the quality of the output
circuit.

If you'd like to work directly with a
preset pass manager you can use the :func:`~.generate_preset_pass_manager`
function to easily generate one. For example:

.. code-block:: python

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.providers.fake_provider import FakeLagosV2

    backend = FakeLagosV2()
    pass_manager = generate_preset_pass_manager(3, backend)

which will generate a :class:`~.StagedPassManager` object for optimization level 3
targeting the :class:`~.FakeLagosV2` backend (equivalent to what is used internally
by :func:`~.transpile` with ``backend=FakeLagosV2()`` and ``optimization_level=3``).
You can use this just like you would any other :class:`~.PassManager`. However,
because it is a :class:`~.StagedPassManager` it also makes it easy to compose and/or
replace stages of the pipeline. For example, if you wanted to run a custom scheduling
stage using dynamical decoupling (via the :class:`~.PadDynamicalDecoupling` pass) and
also add initial logical optimization prior to routing, you would do something like
(building off the previous example):

.. code-block:: python

    from qiskit.circuit.library import XGate, HGate, RXGate, PhaseGate, TGate, TdgGate
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
    from qiskit.transpiler.passes import CXCancellation, InverseCancellation

    backend_durations = backend.target.durations()
    dd_sequence = [XGate(), XGate()]
    scheduling_pm = PassManager([
        ALAPScheduleAnalysis(backend_durations),
        PadDynamicalDecoupling(backend_durations, dd_sequence),
    ])
    inverse_gate_list = [
        HGate(),
        (RXGate(np.pi / 4), RXGate(-np.pi / 4)),
        (PhaseGate(np.pi / 4), PhaseGate(-np.pi / 4)),
        (TGate(), TdgGate()),

    ])
    logical_opt = PassManager([
        CXCancellation(),
        InverseCancellation([HGate(), (RXGate(np.pi / 4), RXGate(-np.pi / 4))
    ])


    # Add pre-layout stage to run extra logical optimization
    pass_manager.pre_layout = logical_opt
    # Set scheduling stage to custom pass manager
    pass_manager.scheduling = scheduling_pm


Then when :meth:`~.StagedPassManager.run` is called on ``pass_manager`` the
``logical_opt`` :class:`~.PassManager` will be called prior to the ``layout`` stage
and for the ``scheduling`` stage our custom :class:`~.PassManager`
``scheduling_pm`` will be used.

Custom Pass Managers
====================

In addition to modifying preset pass managers, it is also possible to construct a pass
manager to build an entirely custom pipeline for transforming input
circuits. You can leverage the :class:`~.StagedPassManager` class directly to do
this. You can define arbitrary stage names and populate them with a :class:`~.PassManager`
instance. For example::

    from qiskit.transpiler.passes import (
        UnitarySynthesis,
        Collect2qBlocks,
        ConsolidateBlocks,
        UnitarySynthesis,
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

will create a new :class:`~.StagedPassManager` that has 2 stages ``init`` and ``translation``.
There is no limit on the number of stages you can put in a custom :class:`~.StagedPassManager`
instance.

The :ref:`stage_generators` functions may be useful for the construction of custom pass managers.
They generate stages which provide common functionality used in many pass managers.
For example, :func:`~.generate_embed_passmanager` can be used to generate a stage
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

.. code-block::

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

.. parsed-literal::

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

.. _transpiler_stage_descriptions:

Transpiler Stage Details
========================

Below are a description of the default transpiler stages and the problems
they solve. The default passes used for each stage are described, but
the specifics are configurable via the ``*_method`` keyword arguments for
the :func:`~.transpile` and :func:`~.generate_preset_pass_manager` functions
which can be used to override the methods described in this section.

.. _translation_stage:

Translation Stage
-----------------

When writing a quantum circuit you are free to use any quantum gate (unitary operator) that
you like, along with a collection of non-gate operations such as qubit measurements and
reset operations.  However, most quantum devices only natively support a handful of quantum gates
and non-gate operations. The allowed instructions for a given backend can be found by querying the
:class:`~.Target` for the devices:

.. code-block:

   from qiskit.providers.fake_provider import FakeVigoV2
   backend = FakeVigoV2()

   print(backend.target)

Every quantum circuit run on the target device must be expressed using only these instructions.
For example, to run a simple phase estimation circuit:

.. plot::
   :include-source:

   import numpy as np
   from qiskit import QuantumCircuit
   from qiskit.providers.fake_provider import FakeVigoV2

   backend = FakeVigoV2()

   qc = QuantumCircuit(2, 1)

   qc.h(0)
   qc.x(1)
   qc.cp(np.pi/4, 0, 1)
   qc.h(0)
   qc.measure([0], [0])
   qc.draw(output='mpl')

We have :math:`H`, :math:`X`, and controlled-:math:`P` gates, none of which are
in our device's basis gate set, and thus must be translated.  This translation is taken
care of for us in the :func:`qiskit.execute` function. However, we can
transpile the circuit to show what it will look like in the native gate set of
the target IBM Quantum device (the :class:`~.FakeVigoV2` backend is a fake backend that
models the historical IBM Vigo 5 qubit device for test purposes):

.. plot::
   :include-source:

   from qiskit import transpile
   from qiskit import QuantumCircuit
   from qiskit.providers.fake_provider import FakeVigoV2

   backend = FakeVigoV2()

   qc = QuantumCircuit(2, 1)

   qc.h(0)
   qc.x(1)
   qc.cp(np.pi/4, 0, 1)
   qc.h(0)
   qc.measure([0], [0])

   qc_basis = transpile(qc, backend)
   qc_basis.draw(output='mpl')

A few things to highlight. First, the circuit has gotten longer with respect to the
original.  This can be verified by checking the depth of both circuits:

.. code-block::

   print('Original depth:', qc.depth(), 'Transpiled Circuit Depth:', qc_basis.depth())

.. parsed-literal::

    Original depth: 4 Transpiled Circuit Depth: 10

Second, although we had a single controlled gate, the fact that it was not in the basis
set means that, when expanded, it requires more than a single :class:`~.CXGate` to implement.
All said, unrolling to the basis set of gates leads to an increase in the depth of a
quantum circuit and the number of gates.

It is important to highlight two special cases:

1. If A swap gate is not a native gate and must be decomposed this requires three CNOT gates:

.. code-block::

    from qiskit.providers.fake_provider import FakeVigoV2
    backend = FakeVigoV2()

    print(backend.operation_names)
.. parsed-literal::

    ['id', 'rz', 'sx', 'x', 'cx', 'measure', 'delay']

.. plot:
    :include-source:

    from qiskit.circuit import QuantumCircuit

    swap_circ = QuantumCircuit(2)
    swap_circ.swap(0, 1)
    swap_circ.decompose().draw(output='mpl')

   As a product of three CNOT gates, swap gates are expensive operations to perform on
   noisy quantum devices.  However, such operations are usually necessary for embedding a
   circuit into the limited gate connectivities of many devices. Thus,
   minimizing the number of swap gates in a circuit is a primary goal in the
   transpilation process.


2. A Toffoli, or controlled-controlled-not gate (``ccx``), is a three-qubit gate.  Given
   that our basis gate set includes only single- and two-qubit gates, it is obvious that
   this gate must be decomposed.  This decomposition is quite costly:

   .. plot::
      :include-source:

      from qiskit.circuit import QuantumCircuit

      ccx_circ = QuantumCircuit(3)
      ccx_circ.ccx(0, 1, 2)
      ccx_circ.decompose().draw(output='mpl')

   For every Toffoli gate in a quantum circuit, the hardware may execute up to six CNOT
   gates, and a handful of single-qubit gates.  From this example, it should be
   clear that any algorithm that makes use of multiple Toffoli gates will end up as a
   circuit with large depth and will therefore be appreciably affected by noise and gate
   errors.


.. _layout_stage:

Layout Stage
------------

Quantum circuits are abstract entities whose qubits are "virtual" representations of actual
qubits used in computations.  We need to be able to map these virtual qubits in a one-to-one
manner to the "physical" qubits in an actual quantum device.

.. image:: /source_images/mapping.png




By default, qiskit will do this mapping for you. The choice of mapping depends on the
properties of the circuit, the particular device you are targeting, and the optimization
level that is chosen. The choice of initial layout is extremely important for minimizing the
number of swap operations needed to map the input circuit onto the device topology and
for minimizing the loss due to non-uniform noise properties across a device. Due to the
importance of this stage, the preset pass managers
try a few different methods to find the best layout. Typically this involves 2 steps: first,
trying to find a "perfect" layout (a layout which does not require any swap operations) and then,
a heuristic pass that tries to find the best layout to use if a perfect layout cannot be found.
For the first stage there are 2 passes typically used for this:

- :class:`~.VF2Layout`: Models layout selection as a subgraph isomorphism problem and tries
  to find a subgraph of the connectivity graph that is isomorphic to the
  graph of 2 qubit interactions in the circuit. If more than one isomorphic mapping is found a
  scoring heuristic is run to select the mapping which would result in the lowest average error
  when executing the circuit.

- :class:`~.TrivialLayout`: Map each virtual qubit to the same numbered physical qubit on the device,
  i.e. ``[0,1,2,3,4]`` -> ``[0,1,2,3,4]``. This is historical behavior used only in
  ``optimization_level=1`` to try to find a perfect layout. If it fails to do so, :class:`~.VF2Layout`
  is tried next).

Next, for the heuristic stage, 2 passes are used by default:

- :class:`~.SabreLayout`: Selects a layout by starting from an initial random layout and then
  repeatedly running a routing algorithm (by default :class:`~.SabreSwap`) both forwards and
  backwards over the circuit using the permutation caused by swap insertions to adjust that
  initial random layout. For more details you can refer to the paper describing the algorithm:
  `arXiv:1809.02573 <https://arxiv.org/abs/1809.02573>`__
  :class:`~.SabreLayout` is use to select a layout if a perfect layout isn't found for
  optimization levels 1, 2, and 3.
- :class:`~.TrivialLayout`: Always used for the layout at optimization level 0
- :class:`~.DenseLayout`: Find the sub-graph of the device with greatest connectivity
  that has the same number of qubits as the circuit. Used for
  optimization level 1 if there are control flow operations (such as
  :class:`~.IfElseOp`) present in the circuit.

Let's see what layouts are automatically picked at various optimization levels.  The circuits
returned by :func:`qiskit.compiler.transpile` are annotated with this initial layout information,
and we can view this layout selection graphically using
:func:`qiskit.visualization.plot_circuit_layout`:

.. plot::
   :include-source:

   from qiskit import QuantumCircuit, transpile
   from qiskit.visualization import plot_circuit_layout
   from qiskit.providers.fake_provider import FakeVigo
   backend = FakeVigo()

   ghz = QuantumCircuit(3, 3)
   ghz.h(0)
   ghz.cx(0,range(1,3))
   ghz.barrier()
   ghz.measure(range(3), range(3))
   ghz.draw(output='mpl')


- **Layout Using Optimization Level 0**

   .. plot::
      :include-source:

      from qiskit import QuantumCircuit, transpile
      from qiskit.visualization import plot_circuit_layout
      from qiskit.providers.fake_provider import FakeVigo
      backend = FakeVigo()

      ghz = QuantumCircuit(3, 3)
      ghz.h(0)
      ghz.cx(0,range(1,3))
      ghz.barrier()
      ghz.measure(range(3), range(3))

      new_circ_lv0 = transpile(ghz, backend=backend, optimization_level=0)
      plot_circuit_layout(new_circ_lv0, backend)

- **Layout Using Optimization Level 3**

   .. plot::
      :include-source:

      from qiskit import QuantumCircuit, transpile
      from qiskit.visualization import plot_circuit_layout
      from qiskit.providers.fake_provider import FakeVigo
      backend = FakeVigo()

      ghz = QuantumCircuit(3, 3)
      ghz.h(0)
      ghz.cx(0,range(1,3))
      ghz.barrier()
      ghz.measure(range(3), range(3))

      new_circ_lv3 = transpile(ghz, backend=backend, optimization_level=3)
      plot_circuit_layout(new_circ_lv3, backend)


It is possible to override automatic layout selection by specifying an initial layout.  To do so we can
pass a list of integers to :func:`qiskit.compiler.transpile` via the `initial_layout`
keyword argument, where the index labels the virtual qubit in the circuit and the
corresponding value is the label for the physical qubit to map onto:

.. plot::
   :include-source:

   from qiskit import QuantumCircuit, transpile
   from qiskit.visualization import plot_circuit_layout
   from qiskit.providers.fake_provider import FakeVigo
   backend = FakeVigo()

   ghz = QuantumCircuit(3, 3)
   ghz.h(0)
   ghz.cx(0,range(1,3))
   ghz.barrier()
   ghz.measure(range(3), range(3))

   # Virtual -> physical
   #    0    ->    3
   #    1    ->    4
   #    2    ->    2

   my_ghz = transpile(ghz, backend, initial_layout=[3, 4, 2])
   plot_circuit_layout(my_ghz, backend)

.. _routing_stage:

Routing Stage
-------------

In order to implement a 2-qubit gate between qubits in a quantum circuit that are not directly
connected on a quantum device, one or more swap gates must be inserted into the circuit to
move the qubit states around until they are adjacent on the device gate map. Each swap
gate typically represents an expensive and noisy operation to perform. Thus, finding the
minimum number of swap gates needed to map a circuit onto a given device, is an important
step (if not the most important) in the whole execution process.

However,as with many important things in life,finding the optimal swap mapping is hard.
In fact it is in a class of problems called NP-hard, and is thus prohibitively expensive 
to compute for all but the smallest quantum devices and input circuits. 
To get around this, by default Qiskit uses a stochastic heuristic 
algorithm called :class:`~.SabreSwap` to compute a good, but not necessarily
optimal swap mapping. The use of a stochastic method means the
circuits generated by :func:`~.transpile` (or :func:`.execute` that calls :func:`~.transpile`
internally) are not guaranteed to be the same over repeated runs. Indeed, running the same
circuit repeatedly will in general result in a distribution of circuit depths and gate counts
at the output.

In order to highlight this, we run a GHZ circuit 100 times, using a "bad" (disconnected)
`initial_layout`:

.. plot::

   from qiskit import QuantumCircuit, transpile

   ghz = QuantumCircuit(15)
   ghz.h(0)
   ghz.cx(0, range(1, 15))
   ghz.draw(output='mpl')

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from qiskit import QuantumCircuit, transpile
   from qiskit.providers.fake_provider import FakeAuckland
   backend = FakeAuckland()

   ghz = QuantumCircuit(15)
   ghz.h(0)
   ghz.cx(0, range(1, 15))

   depths = []
   for _ in range(100):
       depths.append(
           transpile(
               ghz,
               backend,
               layout_method='trivial'  # Fixed layout mapped in circuit order
           ).depth()
       )

   plt.figure(figsize=(8, 6))
   plt.hist(depths, align='left', color='#AC557C')
   plt.xlabel('Depth', fontsize=14)
   plt.ylabel('Counts', fontsize=14);


This distribution is quite wide, signaling the difficulty the swap mapper is having
in computing the best mapping.  Most circuits will have a distribution of depths,
perhaps not as wide as this one, due to the stochastic nature of the default swap
mapper. Of course, we want the best circuit we can get, especially in cases where
the depth is critical to success or failure. The :class:`~.SabreSwap` pass will by default by run its
algorithm in parallel with multiple seed values and select the output which
uses the fewest swaps. If you would like to increase the number of trials
:class:`~.SabreSwap` runs you can refer to :ref:`working_with_preset_pass_managers`
and modify the ``routing`` stage with a custom instance of :class:`~.SabreSwap`
with a larger value for the ``trials`` argument.

Typically, following the swap mapper, the routing stage in the preset pass managers
also includes running the :class:`~.VF2PostLayout` pass. As its name implies,
:class:`~.VF2PostLayout` uses the same basic algorithm as :class:`~.VF2Layout`,
but instead of using it to find a perfect initial layout, it is designed to run after
mapping and try to find a layout on qubits with lower error rates which will
result in better output fidelity when running the circuit. The details of this
algorithm are described in `arXiv:2209.15512 <https://arxiv.org/abs/2209.15512>`__.

.. _optimization_stage:

Optimization Stage
------------------

Decomposing quantum circuits into the basis gate set of the target device,
and the addition of swap gates needed to match hardware topology, conspire to
increase the depth and gate count of quantum circuits.  Fortunately many routines
for optimizing circuits by combining or eliminating gates exist.  In some cases
these methods are so effective the output circuits have lower depth than the inputs.
In other cases, not much can be done, and the computation may be difficult to
perform on noisy devices.  Different gate optimizations are turned on with
different ``optimization_level`` values.  Below we show the benefits gained from
setting the optimization level higher:

.. important::

   The output from :func:`.transpile` varies due to the stochastic swap mapper.
   So the numbers below will likely change each time you run the code.


.. plot::

   import matplotlib.pyplot as plt
   from qiskit import QuantumCircuit, transpile
   from qiskit.providers.fake_provider import FakeAuckland
   backend = FakeAuckland()

   ghz = QuantumCircuit(15)
   ghz.h(0)
   ghz.cx(0, range(1, 15))
   ghz.draw(output='mpl')

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from qiskit import QuantumCircuit, transpile
   from qiskit.providers.fake_provider import FakeAuckland
   backend = FakeAuckland()

   ghz = QuantumCircuit(15)
   ghz.h(0)
   ghz.cx(0, range(1, 15))

   depths = []
   gate_counts = []
   non_local_gate_counts = []
   levels = [str(x) for x in range(4)]
   for level in range(4):
        circ = transpile(ghz, backend, optimization_level=level)
        depths.append(circ.depth())
        gate_counts.append(sum(circ.count_ops().values()))
        non_local_gate_counts.append(circ.num_nonlocal_gates())
   fig, (ax1, ax2) = plt.subplots(2, 1)
   ax1.bar(levels, depths, label='Depth')
   ax1.set_xlabel("Optimization Level")
   ax1.set_ylabel("Depth")
   ax1.set_title("Output Circuit Depth")
   ax2.bar(levels, gate_counts, label='Number of Circuit Operations')
   ax2.bar(levels, non_local_gate_counts, label='Number of non-local gates')
   ax2.set_xlabel("Optimization Level")
   ax2.set_ylabel("Number of gates")
   ax2.legend()
   ax2.set_title("Number of output circuit gates")
   fig.tight_layout()
   plt.show()

.. _scheduling_stage:

Scheduling Stage
----------------

After the circuit has been translated to the target basis, mapped to the device, and optimized,
a scheduling phase can be applied to optionally account for all the idle time in the circuit.
At a high level the scheduling can be thought of as inserting delays into the circuit to account
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
   from qiskit.providers.fake_provider import FakeBoeblingen

   backend = FakeBoeblingen()

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
   from qiskit.providers.fake_provider import FakeBoeblingen

   backend = FakeBoeblingen()

   ghz = QuantumCircuit(5)
   ghz.h(0)
   ghz.cx(0,range(1,5))

   circ = transpile(ghz, backend, scheduling_method="asap")

   timeline_draw(circ)

The scheduling of a circuit involves two parts, analysis and constraint mapping followed by a
padding pass. The first part requires running a scheduling analysis pass such as
:class:`~.ALAPSchedulingAnalysis` or :class:`~.ASAPSchedulingAnalysis` which analyzes the circuit
and records the start time of each instruction in the circuit using a scheduling algorithm ("as late
as possible" for  :class:`~.ALAPSchedulingAnalysis` and "as soon as possible" for
:class:`~.ASAPSchedulingAnalysis`) in the property set. Once the circuit has an initial scheduling
additional passes can be run to account for any timing constraints on the target backend, such
as alignment constraints. This is typically done with the
:class:`~.ConstrainedReschedule` pass which will adjust the scheduling
set in the property set to the constraints of the target backend. Once all
the scheduling and adjustments/rescheduling are finished a padding pass,
such as :class:`~.PadDelay` or :class:`~.PadDynamicalDecoupling` is run
to insert the instructions into the circuit, which completes the scheduling.

Scheduling Analysis with control flow instructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When scheduling analysis passes run there are additional constraints on classical conditions
and control flow instructions in a circuit. This section covers the details of these additional
constraints that any scheduling pass will need to account for.

Policy of topological node ordering in scheduling
'''''''''''''''''''''''''''''''''''''''''''''''''

The DAG representation of ``QuantumCircuit`` respects the node ordering also in the
classical register wires, though theoretically two conditional instructions
conditioned on the same register could commute, i.e. read-access to the
classical register doesn't change its state.

.. parsed-literal::

    qc = QuantumCircuit(2, 1)
    qc.delay(100, 0)
    qc.x(0).c_if(0, True)
    qc.x(1).c_if(0, True)

The scheduler SHOULD comply with above topological ordering policy of the DAG circuit.
Accordingly, the `asap`-scheduled circuit will become

.. parsed-literal::

         ┌────────────────┐   ┌───┐
    q_0: ┤ Delay(100[dt]) ├───┤ X ├──────────────
         ├────────────────┤   └─╥─┘      ┌───┐
    q_1: ┤ Delay(100[dt]) ├─────╫────────┤ X ├───
         └────────────────┘     ║        └─╥─┘
                           ┌────╨────┐┌────╨────┐
    c: 1/══════════════════╡ c_0=0x1 ╞╡ c_0=0x1 ╞
                           └─────────┘└─────────┘

Note that this scheduling might be inefficient in some cases,
because the second conditional operation can start without waiting the delay of 100 dt.
However, such optimization should be done by another pass,
otherwise scheduling may break topological ordering of the original circuit.

Realistic control flow scheduling respecting for microarchitecture
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

In the dispersive QND readout scheme, qubit is measured with microwave stimulus to qubit (Q)
followed by resonator ring-down (depopulation). This microwave signal is recorded
in the buffer memory (B) with hardware kernel, then a discriminated (D) binary value
is moved to the classical register (C).
The sequence from t0 to t1 of the measure instruction interval might be modeled as follows:

.. parsed-literal::

    Q ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░
    B ░░▒▒▒▒▒▒▒▒░░░░░░░░░
    D ░░░░░░░░░░▒▒▒▒▒▒░░░
    C ░░░░░░░░░░░░░░░░▒▒░

However, ``QuantumCircuit`` representation is not enough accurate to represent
this model. In the circuit representation, thus ``Qubit`` is occupied by the
stimulus microwave signal during the first half of the interval,
and ``Clbit`` is only occupied at the very end of the interval.

This precise model may induce weird edge case.

.. parsed-literal::

            ┌───┐
    q_0: ───┤ X ├──────
            └─╥─┘   ┌─┐
    q_1: ─────╫─────┤M├
         ┌────╨────┐└╥┘
    c: 1/╡ c_0=0x1 ╞═╩═
         └─────────┘ 0

In this example, user may intend to measure the state of ``q_1``, after ``XGate`` is
applied to the ``q_0``. This is correct interpretation from viewpoint of
the topological node ordering, i.e. x gate node come in front of the measure node.
However, according to the measurement model above, the data in the register
is unchanged during the stimulus, thus two nodes are simultaneously operated.
If one `alap`-schedule this circuit, it may return following circuit.

.. parsed-literal::

         ┌────────────────┐   ┌───┐
    q_0: ┤ Delay(500[dt]) ├───┤ X ├──────
         └────────────────┘   └─╥─┘   ┌─┐
    q_1: ───────────────────────╫─────┤M├
                           ┌────╨────┐└╥┘
    c: 1/══════════════════╡ c_0=0x1 ╞═╩═
                           └─────────┘ 0

Note that there is no delay on ``q_1`` wire, and the measure instruction immediately
start after t=0, while the conditional gate starts after the delay.
It looks like the topological ordering between the nodes are flipped in the scheduled view.
This behavior can be understood by considering the control flow model described above,

.. parsed-literal::

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

Since there is no qubit register (Q0, Q1) overlap, the node ordering is determined by the
shared classical register C. As you can see, the execution order is still
preserved on C, i.e. read C then apply ``XGate``, finally store the measured outcome in C.
Because ``DAGOpNode`` cannot define different durations for associated registers,
the time ordering of two nodes is inverted anyways.

This behavior can be controlled by ``clbit_write_latency`` and ``conditional_latency``.
The former parameter determines the delay of the register write-access from
the beginning of the measure instruction t0, and another parameter determines
the delay of conditional gate operation from t0 which comes from the register read-access.
These information might be found in the backend configuration and then should
be copied to the pass manager property set before the pass is called.

By default latencies, the `alap`-scheduled circuit of above example may become

.. parsed-literal::

            ┌───┐
    q_0: ───┤ X ├──────
            └─╥─┘   ┌─┐
    q_1: ─────╫─────┤M├
         ┌────╨────┐└╥┘
    c: 1/╡ c_0=0x1 ╞═╩═
         └─────────┘ 0

If the backend microarchitecture supports smart scheduling of the control flow, i.e.
it may separately schedule qubit and classical register,
insertion of the delay yields unnecessary longer total execution time.

.. parsed-literal::
    : Quantum Circuit, first-xgate
    0 ░▒▒▒░░░░░░░░░░░░░░░
    1 ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░

    : In wire q0
    Q ░▒▒▒░░░░░░░░░░░░░░░
    C ░░░░░░░░░░░░░░░░░░░ (zero latency)

    : In wire q1
    Q ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░
    C ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░ (zero latency, scheduled after C0 read-access)

However this result is much more intuitive in the topological ordering view.
If finite conditional latency is provided, for example, 30 dt, the circuit
is scheduled as follows.

.. parsed-literal::

         ┌───────────────┐   ┌───┐
    q_0: ┤ Delay(30[dt]) ├───┤ X ├──────
         ├───────────────┤   └─╥─┘   ┌─┐
    q_1: ┤ Delay(30[dt]) ├─────╫─────┤M├
         └───────────────┘┌────╨────┐└╥┘
    c: 1/═════════════════╡ c_0=0x1 ╞═╩═
                          └─────────┘ 0

with the timing model:

.. parsed-literal::
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

Transpiler Target
-----------------

.. autosummary::
   :toctree: ../stubs/

   Target
   InstructionProperties

Pass Manager Construction
-------------------------

.. autosummary::
   :toctree: ../stubs/

   StagedPassManager
   PassManager
   PassManagerConfig

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

Fenced Objects
--------------

.. autosummary::
   :toctree: ../stubs/

   FencedPropertySet
   FencedDAGCircuit

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
"""

# For backward compatibility
from qiskit.passmanager import (
    FlowController,
    ConditionalController,
    DoWhileController,
)

from .passmanager import PassManager, StagedPassManager
from .passmanager_config import PassManagerConfig
from .propertyset import PropertySet  # pylint: disable=no-name-in-module
from .exceptions import TranspilerError, TranspilerAccessError, CouplingError, LayoutError
from .fencedobjs import FencedDAGCircuit, FencedPropertySet
from .basepasses import AnalysisPass, TransformationPass
from .coupling import CouplingMap
from .layout import Layout, TranspileLayout
from .instruction_durations import InstructionDurations
from .target import Target
from .target import InstructionProperties
from .target import QubitProperties

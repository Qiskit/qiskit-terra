# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""

.. _pulse_builder:

=============
Pulse Builder
=============

..
    We actually want people to think of these functions as being defined within the ``qiskit.pulse``
    namespace, not the submodule ``qiskit.pulse.builder``.

.. currentmodule: qiskit.pulse

Use the pulse builder DSL to write pulse programs with an imperative syntax.

.. warning::
    The pulse builder interface is still in active development. It may have
    breaking API changes without deprecation warnings in future releases until
    otherwise indicated.


The pulse builder provides an imperative API for writing pulse programs
with less difficulty than the :class:`~qiskit.pulse.Schedule` API.
It contextually constructs a pulse schedule and then emits the schedule for
execution. For example, to play a series of pulses on channels is as simple as:


.. plot::
   :include-source:

   from qiskit import pulse

   dc = pulse.DriveChannel
   d0, d1, d2, d3, d4 = dc(0), dc(1), dc(2), dc(3), dc(4)

   with pulse.build(name='pulse_programming_in') as pulse_prog:
       pulse.play([1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1], d0)
       pulse.play([1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], d1)
       pulse.play([1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], d2)
       pulse.play([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], d3)
       pulse.play([1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0], d4)

   pulse_prog.draw()

To begin pulse programming we must first initialize our program builder
context with :func:`build`, after which we can begin adding program
statements. For example, below we write a simple program that :func:`play`\s
a pulse:

.. plot::
   :include-source:

   from qiskit import execute, pulse

   d0 = pulse.DriveChannel(0)

   with pulse.build() as pulse_prog:
       pulse.play(pulse.Constant(100, 1.0), d0)

   pulse_prog.draw()

The builder initializes a :class:`.pulse.Schedule`, ``pulse_prog``
and then begins to construct the program within the context. The output pulse
schedule will survive after the context is exited and can be executed like a
normal Qiskit schedule using ``qiskit.execute(pulse_prog, backend)``.

Pulse programming has a simple imperative style. This leaves the programmer
to worry about the raw experimental physics of pulse programming and not
constructing cumbersome data structures.

We can optionally pass a :class:`~qiskit.providers.Backend` to
:func:`build` to enable enhanced functionality. Below, we prepare a Bell state
by automatically compiling the required pulses from their gate-level
representations, while simultaneously applying a long decoupling pulse to a
neighboring qubit. We terminate the experiment with a measurement to observe the
state we prepared. This program which mixes circuits and pulses will be
automatically lowered to be run as a pulse program:

.. plot::
   :include-source:

   import math

   from qiskit import pulse
   from qiskit.providers.fake_provider import FakeOpenPulse3Q

   # TODO: This example should use a real mock backend.
   backend = FakeOpenPulse3Q()

   d2 = pulse.DriveChannel(2)

   with pulse.build(backend) as bell_prep:
       pulse.u2(0, math.pi, 0)
       pulse.cx(0, 1)

   with pulse.build(backend) as decoupled_bell_prep_and_measure:
       # We call our bell state preparation schedule constructed above.
       with pulse.align_right():
           pulse.call(bell_prep)
           pulse.play(pulse.Constant(bell_prep.duration, 0.02), d2)
           pulse.barrier(0, 1, 2)
           registers = pulse.measure_all()

   decoupled_bell_prep_and_measure.draw()

With the pulse builder we are able to blend programming on qubits and channels.
While the pulse schedule is based on instructions that operate on
channels, the pulse builder automatically handles the mapping from qubits to
channels for you.

In the example below we demonstrate some more features of the pulse builder:

.. code-block::

   import math

   from qiskit import pulse, QuantumCircuit
   from qiskit.pulse import library
   from qiskit.providers.fake_provider import FakeOpenPulse2Q

   backend = FakeOpenPulse2Q()

   with pulse.build(backend) as pulse_prog:
       # Create a pulse.
       gaussian_pulse = library.gaussian(10, 1.0, 2)
       # Get the qubit's corresponding drive channel from the backend.
       d0 = pulse.drive_channel(0)
       d1 = pulse.drive_channel(1)
       # Play a pulse at t=0.
       pulse.play(gaussian_pulse, d0)
       # Play another pulse directly after the previous pulse at t=10.
       pulse.play(gaussian_pulse, d0)
       # The default scheduling behavior is to schedule pulses in parallel
       # across channels. For example, the statement below
       # plays the same pulse on a different channel at t=0.
       pulse.play(gaussian_pulse, d1)

       # We also provide pulse scheduling alignment contexts.
       # The default alignment context is align_left.

       # The sequential context schedules pulse instructions sequentially in time.
       # This context starts at t=10 due to earlier pulses above.
       with pulse.align_sequential():
           pulse.play(gaussian_pulse, d0)
           # Play another pulse after at t=20.
           pulse.play(gaussian_pulse, d1)

           # We can also nest contexts as each instruction is
           # contained in its local scheduling context.
           # The output of a child context is a context-schedule
           # with the internal instructions timing fixed relative to
           # one another. This is schedule is then called in the parent context.

           # Context starts at t=30.
           with pulse.align_left():
               # Start at t=30.
               pulse.play(gaussian_pulse, d0)
               # Start at t=30.
               pulse.play(gaussian_pulse, d1)
           # Context ends at t=40.

           # Alignment context where all pulse instructions are
           # aligned to the right, ie., as late as possible.
           with pulse.align_right():
               # Shift the phase of a pulse channel.
               pulse.shift_phase(math.pi, d1)
               # Starts at t=40.
               pulse.delay(100, d0)
               # Ends at t=140.

               # Starts at t=130.
               pulse.play(gaussian_pulse, d1)
               # Ends at t=140.

           # Acquire data for a qubit and store in a memory slot.
           pulse.acquire(100, 0, pulse.MemorySlot(0))

           # We also support a variety of macros for common operations.

           # Measure all qubits.
           pulse.measure_all()

           # Delay on some qubits.
           # This requires knowledge of which channels belong to which qubits.
           # delay for 100 cycles on qubits 0 and 1.
           pulse.delay_qubits(100, 0, 1)

           # Call a quantum circuit. The pulse builder lazily constructs a quantum
           # circuit which is then transpiled and scheduled before inserting into
           # a pulse schedule.
           # NOTE: Quantum register indices correspond to physical qubit indices.
           qc = QuantumCircuit(2, 2)
           qc.cx(0, 1)
           pulse.call(qc)
           # Calling a small set of standard gates and decomposing to pulses is
           # also supported with more natural syntax.
           pulse.u3(0, math.pi, 0, 0)
           pulse.cx(0, 1)


           # It is also be possible to call a preexisting schedule
           tmp_sched = pulse.Schedule()
           tmp_sched += pulse.Play(gaussian_pulse, d0)
           pulse.call(tmp_sched)

           # We also support:

           # frequency instructions
           pulse.set_frequency(5.0e9, d0)

           # phase instructions
           pulse.shift_phase(0.1, d0)

           # offset contexts
           with pulse.phase_offset(math.pi, d0):
               pulse.play(gaussian_pulse, d0)

The above is just a small taste of what is possible with the builder. See the rest of the module
documentation for more information on its capabilities.

.. autosummary::
    :toctree: ../stubs/

    build


Channels
========

Methods to return the correct channels for the respective qubit indices.

.. code-block::

    from qiskit import pulse
    from qiskit.providers.fake_provider import FakeArmonk

    backend = FakeArmonk()

    with pulse.build(backend) as drive_sched:
        d0 = pulse.drive_channel(0)
        print(d0)

.. parsed-literal::

    DriveChannel(0)

.. autosummary::
    :toctree: ../stubs/

    acquire_channel
    control_channels
    drive_channel
    measure_channel


Instructions
============

Pulse instructions are available within the builder interface. Here's an example:

.. plot::
   :include-source:

    from qiskit import pulse
    from qiskit.providers.fake_provider import FakeArmonk

    backend = FakeArmonk()

    with pulse.build(backend) as drive_sched:
        d0 = pulse.drive_channel(0)
        a0 = pulse.acquire_channel(0)

        pulse.play(pulse.library.Constant(10, 1.0), d0)
        pulse.delay(20, d0)
        pulse.shift_phase(3.14/2, d0)
        pulse.set_phase(3.14, d0)
        pulse.shift_frequency(1e7, d0)
        pulse.set_frequency(5e9, d0)

        with pulse.build() as temp_sched:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), d0)
            pulse.play(pulse.library.Gaussian(20, -1.0, 3.0), d0)

        pulse.call(temp_sched)
        pulse.acquire(30, a0, pulse.MemorySlot(0))

    drive_sched.draw()


.. autosummary::
    :toctree: ../stubs/

    acquire
    barrier
    call
    delay
    play
    reference
    set_frequency
    set_phase
    shift_frequency
    shift_phase
    snapshot


Contexts
========

Builder aware contexts that modify the construction of a pulse program. For
example an alignment context like :func:`align_right` may
be used to align all pulses as late as possible in a pulse program.

.. plot::
   :include-source:

   from qiskit import pulse

   d0 = pulse.DriveChannel(0)
   d1 = pulse.DriveChannel(1)

   with pulse.build() as pulse_prog:
       with pulse.align_right():
           # this pulse will start at t=0
           pulse.play(pulse.Constant(100, 1.0), d0)
           # this pulse will start at t=80
           pulse.play(pulse.Constant(20, 1.0), d1)

   pulse_prog.draw()

.. autosummary::
    :toctree: ../stubs/

    align_equispaced
    align_func
    align_left
    align_right
    align_sequential
    circuit_scheduler_settings
    frequency_offset
    phase_offset
    transpiler_settings


Macros
======

Macros help you add more complex functionality to your pulse program.

.. code-block::

    from qiskit import pulse
    from qiskit.providers.fake_provider import FakeArmonk

    backend = FakeArmonk()

    with pulse.build(backend) as measure_sched:
        mem_slot = pulse.measure(0)
        print(mem_slot)

.. parsed-literal::

    MemorySlot(0)

.. autosummary::
    :toctree: ../stubs/

    measure
    measure_all
    delay_qubits


Circuit Gates
=============

To use circuit level gates within your pulse program call a circuit
with :func:`call`.

.. warning::
    These will be removed in future versions with the release of a circuit
    builder interface in which it will be possible to calibrate a gate in
    terms of pulses and use that gate in a circuit.

.. code-block::

    import math

    from qiskit import pulse
    from qiskit.providers.fake_provider import FakeArmonk

    backend = FakeArmonk()

    with pulse.build(backend) as u3_sched:
        pulse.u3(math.pi, 0, math.pi, 0)

.. autosummary::
    :toctree: ../stubs/

    cx
    u1
    u2
    u3
    x


Utilities
=========

The utility functions can be used to gather attributes about the backend and modify
how the program is built.

.. code-block::

    from qiskit import pulse

    from qiskit.providers.fake_provider import FakeArmonk

    backend = FakeArmonk()

    with pulse.build(backend) as u3_sched:
        print('Number of qubits in backend: {}'.format(pulse.num_qubits()))

        samples = 160
        print('There are {} samples in {} seconds'.format(
            samples, pulse.samples_to_seconds(160)))

        seconds = 1e-6
        print('There are {} seconds in {} samples.'.format(
            seconds, pulse.seconds_to_samples(1e-6)))

.. parsed-literal::

    Number of qubits in backend: 1
    There are 160 samples in 3.5555555555555554e-08 seconds
    There are 1e-06 seconds in 4500 samples.

.. autosummary::
    :toctree: ../stubs/

    active_backend
    active_transpiler_settings
    active_circuit_scheduler_settings
    num_qubits
    qubit_channels
    samples_to_seconds
    seconds_to_samples
"""
import collections
import contextvars
import functools
import itertools
import uuid
import warnings
from contextlib import contextmanager
from functools import singledispatchmethod
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    NewType,
)

import numpy as np

from qiskit import circuit
from qiskit.circuit.library import standard_gates as gates
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import (
    channels as chans,
    configuration,
    exceptions,
    instructions,
    macros,
    library,
    transforms,
)
from qiskit.providers.backend import BackendV2
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import AlignmentKind


#: contextvars.ContextVar[BuilderContext]: active builder
BUILDER_CONTEXTVAR = contextvars.ContextVar("backend")

T = TypeVar("T")

StorageLocation = NewType("StorageLocation", Union[chans.MemorySlot, chans.RegisterSlot])


def _compile_lazy_circuit_before(function: Callable[..., T]) -> Callable[..., T]:
    """Decorator thats schedules and calls the lazily compiled circuit before
    executing the decorated builder method."""

    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        self._compile_lazy_circuit()
        return function(self, *args, **kwargs)

    return wrapper


def _requires_backend(function: Callable[..., T]) -> Callable[..., T]:
    """Decorator a function to raise if it is called without a builder with a
    set backend.
    """

    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        if self.backend is None:
            raise exceptions.BackendNotSet(
                'This function requires the builder to have a "backend" set.'
            )
        return function(self, *args, **kwargs)

    return wrapper


class _PulseBuilder:
    """Builder context class."""

    __alignment_kinds__ = {
        "left": transforms.AlignLeft(),
        "right": transforms.AlignRight(),
        "sequential": transforms.AlignSequential(),
    }

    def __init__(
        self,
        backend=None,
        block: Optional[ScheduleBlock] = None,
        name: Optional[str] = None,
        default_alignment: Union[str, AlignmentKind] = "left",
        default_transpiler_settings: Mapping = None,
        default_circuit_scheduler_settings: Mapping = None,
    ):
        """Initialize the builder context.

        .. note::
            At some point we may consider incorporating the builder into
            the :class:`~qiskit.pulse.Schedule` class. However, the risk of
            this is tying the user interface to the intermediate
            representation. For now we avoid this at the cost of some code
            duplication.

        Args:
            backend (Backend): Input backend to use in
                builder. If not set certain functionality will be unavailable.
            block: Initital ``ScheduleBlock`` to build on.
            name: Name of pulse program to be built.
            default_alignment: Default scheduling alignment for builder.
                One of ``left``, ``right``, ``sequential`` or an instance of
                :class:`~qiskit.pulse.transforms.alignments.AlignmentKind` subclass.
            default_transpiler_settings: Default settings for the transpiler.
            default_circuit_scheduler_settings: Default settings for the
                circuit to pulse scheduler.

        Raises:
            PulseError: When invalid ``default_alignment`` or `block` is specified.
        """
        #: Backend: Backend instance for context builder.
        self._backend = backend

        #: Union[None, ContextVar]: Token for this ``_PulseBuilder``'s ``ContextVar``.
        self._backend_ctx_token = None

        #: QuantumCircuit: Lazily constructed quantum circuit
        self._lazy_circuit = None

        #: Dict[str, Any]: Transpiler setting dictionary.
        self._transpiler_settings = default_transpiler_settings or {}

        #: Dict[str, Any]: Scheduler setting dictionary.
        self._circuit_scheduler_settings = default_circuit_scheduler_settings or {}

        #: List[ScheduleBlock]: Stack of context.
        self._context_stack = []

        #: str: Name of the output program
        self._name = name

        # Add root block if provided. Schedule will be built on top of this.
        if block is not None:
            if isinstance(block, ScheduleBlock):
                root_block = block
            elif isinstance(block, Schedule):
                root_block = self._naive_typecast_schedule(block)
            else:
                raise exceptions.PulseError(
                    f"Input `block` type {block.__class__.__name__} is "
                    "not a valid format. Specify a pulse program."
                )
            self._context_stack.append(root_block)

        # Set default alignment context
        alignment = _PulseBuilder.__alignment_kinds__.get(default_alignment, default_alignment)
        if not isinstance(alignment, AlignmentKind):
            raise exceptions.PulseError(
                f"Given `default_alignment` {repr(default_alignment)} is "
                "not a valid transformation. Set one of "
                f'{", ".join(_PulseBuilder.__alignment_kinds__.keys())}, '
                "or set an instance of `AlignmentKind` subclass."
            )
        self.push_context(alignment)

    def __enter__(self) -> ScheduleBlock:
        """Enter this builder context and yield either the supplied schedule
        or the schedule created for the user.

        Returns:
            The schedule that the builder will build on.
        """
        self._backend_ctx_token = BUILDER_CONTEXTVAR.set(self)
        output = self._context_stack[0]
        output._name = self._name or output.name

        return output

    @_compile_lazy_circuit_before
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the builder context and compile the built pulse program."""
        self.compile()
        BUILDER_CONTEXTVAR.reset(self._backend_ctx_token)

    @property
    def backend(self):
        """Returns the builder backend if set.

        Returns:
            Optional[Backend]: The builder's backend.
        """
        return self._backend

    @_compile_lazy_circuit_before
    def push_context(self, alignment: AlignmentKind):
        """Push new context to the stack."""
        self._context_stack.append(ScheduleBlock(alignment_context=alignment))

    @_compile_lazy_circuit_before
    def pop_context(self) -> ScheduleBlock:
        """Pop the last context from the stack."""
        if len(self._context_stack) == 1:
            raise exceptions.PulseError("The root context cannot be popped out.")

        return self._context_stack.pop()

    def get_context(self) -> ScheduleBlock:
        """Get current context.

        Notes:
            New instruction can be added by `.append_subroutine` or `.append_instruction` method.
            Use above methods rather than directly accessing to the current context.
        """
        return self._context_stack[-1]

    @property
    @_requires_backend
    def num_qubits(self):
        """Get the number of qubits in the backend."""
        # backendV2
        if isinstance(self.backend, BackendV2):
            return self.backend.num_qubits
        return self.backend.configuration().n_qubits

    @property
    def transpiler_settings(self) -> Mapping:
        """The builder's transpiler settings."""
        return self._transpiler_settings

    @transpiler_settings.setter
    @_compile_lazy_circuit_before
    def transpiler_settings(self, settings: Mapping):
        self._compile_lazy_circuit()
        self._transpiler_settings = settings

    @property
    def circuit_scheduler_settings(self) -> Mapping:
        """The builder's circuit to pulse scheduler settings."""
        return self._circuit_scheduler_settings

    @circuit_scheduler_settings.setter
    @_compile_lazy_circuit_before
    def circuit_scheduler_settings(self, settings: Mapping):
        self._compile_lazy_circuit()
        self._circuit_scheduler_settings = settings

    @_compile_lazy_circuit_before
    def compile(self) -> ScheduleBlock:
        """Compile and output the built pulse program."""
        # Not much happens because we currently compile as we build.
        # This should be offloaded to a true compilation module
        # once we define a more sophisticated IR.

        while len(self._context_stack) > 1:
            current = self.pop_context()
            self.append_subroutine(current)

        return self._context_stack[0]

    def _compile_lazy_circuit(self):
        """Call a context QuantumCircuit (lazy circuit) and append the output pulse schedule
        to the builder's context schedule.

        Note that the lazy circuit is not stored as a call instruction.
        """
        if self._lazy_circuit:
            lazy_circuit = self._lazy_circuit
            # reset lazy circuit
            self._lazy_circuit = self._new_circuit()
            self.call_subroutine(self._compile_circuit(lazy_circuit))

    def _compile_circuit(self, circ) -> Schedule:
        """Take a QuantumCircuit and output the pulse schedule associated with the circuit."""
        from qiskit import compiler  # pylint: disable=cyclic-import

        transpiled_circuit = compiler.transpile(circ, self.backend, **self.transpiler_settings)
        sched = compiler.schedule(
            transpiled_circuit, self.backend, **self.circuit_scheduler_settings
        )
        return sched

    def _new_circuit(self):
        """Create a new circuit for lazy circuit scheduling."""
        return circuit.QuantumCircuit(self.num_qubits)

    @_compile_lazy_circuit_before
    def append_instruction(self, instruction: instructions.Instruction):
        """Add an instruction to the builder's context schedule.

        Args:
            instruction: Instruction to append.
        """
        self._context_stack[-1].append(instruction)

    def append_reference(self, name: str, *extra_keys: str):
        """Add external program as a :class:`~qiskit.pulse.instructions.Reference` instruction.

        Args:
            name: Name of subroutine.
            extra_keys: Assistance keys to uniquely specify the subroutine.
        """
        inst = instructions.Reference(name, *extra_keys)
        self.append_instruction(inst)

    @_compile_lazy_circuit_before
    def append_subroutine(self, subroutine: Union[Schedule, ScheduleBlock]):
        """Append a :class:`ScheduleBlock` to the builder's context schedule.

        This operation doesn't create a reference. Subroutine is directly
        appended to current context schedule.

        Args:
            subroutine: ScheduleBlock to append to the current context block.

        Raises:
            PulseError: When subroutine is not Schedule nor ScheduleBlock.
        """
        if not isinstance(subroutine, (ScheduleBlock, Schedule)):
            raise exceptions.PulseError(
                f"'{subroutine.__class__.__name__}' is not valid data format in the builder. "
                "'Schedule' and 'ScheduleBlock' can be appended to the builder context."
            )

        if len(subroutine) == 0:
            return
        if isinstance(subroutine, Schedule):
            subroutine = self._naive_typecast_schedule(subroutine)
        self._context_stack[-1].append(subroutine)

    @singledispatchmethod
    def call_subroutine(
        self,
        subroutine: Union[circuit.QuantumCircuit, Schedule, ScheduleBlock],
        name: Optional[str] = None,
        value_dict: Optional[Dict[ParameterExpression, ParameterValueType]] = None,
        **kw_params: ParameterValueType,
    ):
        """Call a schedule or circuit defined outside of the current scope.

        The ``subroutine`` is appended to the context schedule as a call instruction.
        This logic just generates a convenient program representation in the compiler.
        Thus, this doesn't affect execution of inline subroutines.
        See :class:`~pulse.instructions.Call` for more details.

        Args:
            subroutine: Target schedule or circuit to append to the current context.
            name: Name of subroutine if defined.
            value_dict: Parameter object and assigned value mapping. This is more precise way to
                identify a parameter since mapping is managed with unique object id rather than
                name. Especially there is any name collision in a parameter table.
            kw_params: Parameter values to bind to the target subroutine
                with string parameter names. If there are parameter name overlapping,
                these parameters are updated with the same assigned value.

        Raises:
            PulseError:
                - When input subroutine is not valid data format.
        """
        raise exceptions.PulseError(
            f"Subroutine type {subroutine.__class__.__name__} is "
            "not valid data format. Call QuantumCircuit, "
            "Schedule, or ScheduleBlock."
        )

    @call_subroutine.register
    def _(
        self,
        target_block: ScheduleBlock,
        name: Optional[str] = None,
        value_dict: Optional[Dict[ParameterExpression, ParameterValueType]] = None,
        **kw_params: ParameterValueType,
    ):
        if len(target_block) == 0:
            return

        # Create local parameter assignment
        local_assignment = {}
        for param_name, value in kw_params.items():
            params = target_block.get_parameters(param_name)
            if not params:
                raise exceptions.PulseError(
                    f"Parameter {param_name} is not defined in the target subroutine. "
                    f'{", ".join(map(str, target_block.parameters))} can be specified.'
                )
            for param in params:
                local_assignment[param] = value

        if value_dict:
            if local_assignment.keys() & value_dict.keys():
                warnings.warn(
                    "Some parameters provided by 'value_dict' conflict with one through "
                    "keyword arguments. Parameter values in the keyword arguments "
                    "are overridden by the dictionary values.",
                    UserWarning,
                )
            local_assignment.update(value_dict)

        if local_assignment:
            target_block = target_block.assign_parameters(local_assignment, inplace=False)

        if name is None:
            # Add unique string, not to accidentally override existing reference entry.
            keys = (target_block.name, uuid.uuid4().hex)
        else:
            keys = (name,)

        self.append_reference(*keys)
        self.get_context().assign_references({keys: target_block}, inplace=True)

    @call_subroutine.register
    def _(
        self,
        target_schedule: Schedule,
        name: Optional[str] = None,
        value_dict: Optional[Dict[ParameterExpression, ParameterValueType]] = None,
        **kw_params: ParameterValueType,
    ):
        if len(target_schedule) == 0:
            return

        self.call_subroutine(
            self._naive_typecast_schedule(target_schedule),
            name=name,
            value_dict=value_dict,
            **kw_params,
        )

    @call_subroutine.register
    def _(
        self,
        target_circuit: circuit.QuantumCircuit,
        name: Optional[str] = None,
        value_dict: Optional[Dict[ParameterExpression, ParameterValueType]] = None,
        **kw_params: ParameterValueType,
    ):
        if len(target_circuit) == 0:
            return

        self._compile_lazy_circuit()
        self.call_subroutine(
            self._compile_circuit(target_circuit),
            name=name,
            value_dict=value_dict,
            **kw_params,
        )

    @_requires_backend
    def call_gate(self, gate: circuit.Gate, qubits: Tuple[int, ...], lazy: bool = True):
        """Call the circuit ``gate`` in the pulse program.

        The qubits are assumed to be defined on physical qubits.

        If ``lazy == True`` this circuit will extend a lazily constructed
        quantum circuit. When an operation occurs that breaks the underlying
        circuit scheduling assumptions such as adding a pulse instruction or
        changing the alignment context the circuit will be
        transpiled and scheduled into pulses with the current active settings.

        Args:
            gate: Gate to call.
            qubits: Qubits to call gate on.
            lazy: If false the circuit will be transpiled and pulse scheduled
                immediately. Otherwise, it will extend the active lazy circuit
                as defined above.
        """
        try:
            iter(qubits)
        except TypeError:
            qubits = (qubits,)

        if lazy:
            self._call_gate(gate, qubits)
        else:
            self._compile_lazy_circuit()
            self._call_gate(gate, qubits)
            self._compile_lazy_circuit()

    def _call_gate(self, gate, qargs):
        if self._lazy_circuit is None:
            self._lazy_circuit = self._new_circuit()

        self._lazy_circuit.append(gate, qargs=qargs)

    @staticmethod
    def _naive_typecast_schedule(schedule: Schedule):
        # Naively convert into ScheduleBlock
        from qiskit.pulse.transforms import inline_subroutines, flatten, pad

        preprocessed_schedule = inline_subroutines(flatten(schedule))
        pad(preprocessed_schedule, inplace=True, pad_with=instructions.TimeBlockade)

        # default to left alignment, namely ASAP scheduling
        target_block = ScheduleBlock(name=schedule.name)
        for _, inst in preprocessed_schedule.instructions:
            target_block.append(inst, inplace=True)

        return target_block

    def get_dt(self):
        """Retrieve dt differently based on the type of Backend"""
        if isinstance(self.backend, BackendV2):
            return self.backend.dt
        return self.backend.configuration().dt


def build(
    backend=None,
    schedule: Optional[ScheduleBlock] = None,
    name: Optional[str] = None,
    default_alignment: Optional[Union[str, AlignmentKind]] = "left",
    default_transpiler_settings: Optional[Dict[str, Any]] = None,
    default_circuit_scheduler_settings: Optional[Dict[str, Any]] = None,
) -> ContextManager[ScheduleBlock]:
    """Create a context manager for launching the imperative pulse builder DSL.

    To enter a building context and starting building a pulse program:

    .. code-block::

        from qiskit import execute, pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.play(pulse.Constant(100, 0.5), d0)


    While the output program ``pulse_prog`` cannot be executed as we are using
    a mock backend. If a real backend is being used, executing the program is
    done with:

    .. code-block:: python

        qiskit.execute(pulse_prog, backend)

    Args:
        backend (Backend): A Qiskit backend. If not supplied certain
            builder functionality will be unavailable.
        schedule: A pulse ``ScheduleBlock`` in which your pulse program will be built.
        name: Name of pulse program to be built.
        default_alignment: Default scheduling alignment for builder.
            One of ``left``, ``right``, ``sequential`` or an alignment context.
        default_transpiler_settings: Default settings for the transpiler.
        default_circuit_scheduler_settings: Default settings for the
            circuit to pulse scheduler.

    Returns:
        A new builder context which has the active builder initialized.
    """
    return _PulseBuilder(
        backend=backend,
        block=schedule,
        name=name,
        default_alignment=default_alignment,
        default_transpiler_settings=default_transpiler_settings,
        default_circuit_scheduler_settings=default_circuit_scheduler_settings,
    )


# Builder Utilities


def _active_builder() -> _PulseBuilder:
    """Get the active builder in the active context.

    Returns:
        The active active builder in this context.

    Raises:
        exceptions.NoActiveBuilder: If a pulse builder function is called
        outside of a builder context.
    """
    try:
        return BUILDER_CONTEXTVAR.get()
    except LookupError as ex:
        raise exceptions.NoActiveBuilder(
            "A Pulse builder function was called outside of "
            "a builder context. Try calling within a builder "
            'context, eg., "with pulse.build() as schedule: ...".'
        ) from ex


def active_backend():
    """Get the backend of the currently active builder context.

    Returns:
        Backend: The active backend in the currently active
            builder context.

    Raises:
        exceptions.BackendNotSet: If the builder does not have a backend set.
    """
    builder = _active_builder().backend
    if builder is None:
        raise exceptions.BackendNotSet(
            'This function requires the active builder to have a "backend" set.'
        )
    return builder


def append_schedule(schedule: Union[Schedule, ScheduleBlock]):
    """Call a schedule by appending to the active builder's context block.

    Args:
        schedule: Schedule or ScheduleBlock to append.
    """
    _active_builder().append_subroutine(schedule)


def append_instruction(instruction: instructions.Instruction):
    """Append an instruction to the active builder's context schedule.

    Examples:

    .. code-block::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.builder.append_instruction(pulse.Delay(10, d0))

        print(pulse_prog.instructions)

    .. parsed-literal::

        ((0, Delay(10, DriveChannel(0))),)
    """
    _active_builder().append_instruction(instruction)


def num_qubits() -> int:
    """Return number of qubits in the currently active backend.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            print(pulse.num_qubits())

    .. parsed-literal::

       2

    .. note:: Requires the active builder context to have a backend set.
    """
    if isinstance(active_backend(), BackendV2):
        return active_backend().num_qubits
    return active_backend().configuration().n_qubits


def seconds_to_samples(seconds: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """Obtain the number of samples that will elapse in ``seconds`` on the
    active backend.

    Rounds down.

    Args:
        seconds: Time in seconds to convert to samples.

    Returns:
        The number of samples for the time to elapse
    """
    dt = _active_builder().get_dt()
    if isinstance(seconds, np.ndarray):
        return (seconds / dt).astype(int)
    return int(seconds / dt)


def samples_to_seconds(samples: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """Obtain the time in seconds that will elapse for the input number of
    samples on the active backend.

    Args:
        samples: Number of samples to convert to time in seconds.

    Returns:
        The time that elapses in ``samples``.
    """
    return samples * _active_builder().get_dt()


def qubit_channels(qubit: int) -> Set[chans.Channel]:
    """Returns the set of channels associated with a qubit.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            print(pulse.qubit_channels(0))

    .. parsed-literal::

       {MeasureChannel(0), ControlChannel(0), DriveChannel(0), AcquireChannel(0), ControlChannel(1)}

    .. note:: Requires the active builder context to have a backend set.

    .. note:: A channel may still be associated with another qubit in this list
        such as in the case where significant crosstalk exists.

    """

    # implement as the inner function to avoid API change for a patch release in 0.24.2.
    def get_qubit_channels_v2(backend: BackendV2, qubit: int):
        r"""Return a list of channels which operate on the given ``qubit``.
        Returns:
            List of ``Channel``\s operated on my the given ``qubit``.
        """
        channels = []

        # add multi-qubit channels
        for node_qubits in backend.coupling_map:
            if qubit in node_qubits:
                control_channel = backend.control_channel(node_qubits)
                if control_channel:
                    channels.extend(control_channel)

        # add single qubit channels
        channels.append(backend.drive_channel(qubit))
        channels.append(backend.measure_channel(qubit))
        channels.append(backend.acquire_channel(qubit))
        return channels

    # backendV2
    if isinstance(active_backend(), BackendV2):
        return set(get_qubit_channels_v2(active_backend(), qubit))
    return set(active_backend().configuration().get_qubit_channels(qubit))


def _qubits_to_channels(*channels_or_qubits: Union[int, chans.Channel]) -> Set[chans.Channel]:
    """Returns the unique channels of the input qubits."""
    channels = set()
    for channel_or_qubit in channels_or_qubits:
        if isinstance(channel_or_qubit, int):
            channels |= qubit_channels(channel_or_qubit)
        elif isinstance(channel_or_qubit, chans.Channel):
            channels.add(channel_or_qubit)
        else:
            raise exceptions.PulseError(
                f'{channel_or_qubit} is not a "Channel" or qubit (integer).'
            )
    return channels


def active_transpiler_settings() -> Dict[str, Any]:
    """Return the current active builder context's transpiler settings.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        transpiler_settings = {'optimization_level': 3}

        with pulse.build(backend,
                         default_transpiler_settings=transpiler_settings):
            print(pulse.active_transpiler_settings())

    .. parsed-literal::

        {'optimization_level': 3}

    """
    return dict(_active_builder().transpiler_settings)


def active_circuit_scheduler_settings() -> Dict[str, Any]:
    """Return the current active builder context's circuit scheduler settings.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        circuit_scheduler_settings = {'method': 'alap'}

        with pulse.build(
                backend,
                default_circuit_scheduler_settings=circuit_scheduler_settings):
            print(pulse.active_circuit_scheduler_settings())

    .. parsed-literal::

       {'method': 'alap'}

    """
    return dict(_active_builder().circuit_scheduler_settings)


# Contexts


@contextmanager
def align_left() -> ContextManager[None]:
    """Left alignment pulse scheduling context.

    Pulse instructions within this context are scheduled as early as possible
    by shifting them left to the earliest available time.

    Examples:

    .. code-block::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as pulse_prog:
            with pulse.align_left():
                # this pulse will start at t=0
                pulse.play(pulse.Constant(100, 1.0), d0)
                # this pulse will start at t=0
                pulse.play(pulse.Constant(20, 1.0), d1)
        pulse_prog = pulse.transforms.block_to_schedule(pulse_prog)

        assert pulse_prog.ch_start_time(d0) == pulse_prog.ch_start_time(d1)

    Yields:
        None
    """
    builder = _active_builder()
    builder.push_context(transforms.AlignLeft())
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)


@contextmanager
def align_right() -> AlignmentKind:
    """Right alignment pulse scheduling context.

    Pulse instructions within this context are scheduled as late as possible
    by shifting them right to the latest available time.

    Examples:

    .. code-block::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as pulse_prog:
            with pulse.align_right():
                # this pulse will start at t=0
                pulse.play(pulse.Constant(100, 1.0), d0)
                # this pulse will start at t=80
                pulse.play(pulse.Constant(20, 1.0), d1)
        pulse_prog = pulse.transforms.block_to_schedule(pulse_prog)

        assert pulse_prog.ch_stop_time(d0) == pulse_prog.ch_stop_time(d1)

    Yields:
        None
    """
    builder = _active_builder()
    builder.push_context(transforms.AlignRight())
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)


@contextmanager
def align_sequential() -> AlignmentKind:
    """Sequential alignment pulse scheduling context.

    Pulse instructions within this context are scheduled sequentially in time
    such that no two instructions will be played at the same time.

    Examples:

    .. code-block::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build() as pulse_prog:
            with pulse.align_sequential():
                # this pulse will start at t=0
                pulse.play(pulse.Constant(100, 1.0), d0)
                # this pulse will also start at t=100
                pulse.play(pulse.Constant(20, 1.0), d1)
        pulse_prog = pulse.transforms.block_to_schedule(pulse_prog)

        assert pulse_prog.ch_stop_time(d0) == pulse_prog.ch_start_time(d1)

    Yields:
        None
    """
    builder = _active_builder()
    builder.push_context(transforms.AlignSequential())
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)


@contextmanager
def align_equispaced(duration: Union[int, ParameterExpression]) -> AlignmentKind:
    """Equispaced alignment pulse scheduling context.

    Pulse instructions within this context are scheduled with the same interval spacing such that
    the total length of the context block is ``duration``.
    If the total free ``duration`` cannot be evenly divided by the number of instructions
    within the context, the modulo is split and then prepended and appended to
    the returned schedule. Delay instructions are automatically inserted in between pulses.

    This context is convenient to write a schedule for periodical dynamic decoupling or
    the Hahn echo sequence.

    Examples:

    .. plot::
       :include-source:

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        x90 = pulse.Gaussian(10, 0.1, 3)
        x180 = pulse.Gaussian(10, 0.2, 3)

        with pulse.build() as hahn_echo:
            with pulse.align_equispaced(duration=100):
                pulse.play(x90, d0)
                pulse.play(x180, d0)
                pulse.play(x90, d0)

        hahn_echo.draw()

    Args:
        duration: Duration of this context. This should be larger than the schedule duration.

    Yields:
        None

    Notes:
        The scheduling is performed for sub-schedules within the context rather than
        channel-wise. If you want to apply the equispaced context for each channel,
        you should use the context independently for channels.
    """
    builder = _active_builder()
    builder.push_context(transforms.AlignEquispaced(duration=duration))
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)


@contextmanager
def align_func(
    duration: Union[int, ParameterExpression], func: Callable[[int], float]
) -> AlignmentKind:
    """Callback defined alignment pulse scheduling context.

    Pulse instructions within this context are scheduled at the location specified by
    arbitrary callback function `position` that takes integer index and returns
    the associated fractional location within [0, 1].
    Delay instruction is automatically inserted in between pulses.

    This context may be convenient to write a schedule of arbitrary dynamical decoupling
    sequences such as Uhrig dynamical decoupling.

    Examples:

    .. plot::
       :include-source:

        import numpy as np
        from qiskit import pulse

        d0 = pulse.DriveChannel(0)
        x90 = pulse.Gaussian(10, 0.1, 3)
        x180 = pulse.Gaussian(10, 0.2, 3)

        def udd10_pos(j):
            return np.sin(np.pi*j/(2*10 + 2))**2

        with pulse.build() as udd_sched:
            pulse.play(x90, d0)
            with pulse.align_func(duration=300, func=udd10_pos):
                for _ in range(10):
                    pulse.play(x180, d0)
            pulse.play(x90, d0)

        udd_sched.draw()

    Args:
        duration: Duration of context. This should be larger than the schedule duration.
        func: A function that takes an index of sub-schedule and returns the
            fractional coordinate of of that sub-schedule.
            The returned value should be defined within [0, 1].
            The pulse index starts from 1.

    Yields:
        None

    Notes:
        The scheduling is performed for sub-schedules within the context rather than
        channel-wise. If you want to apply the numerical context for each channel,
        you need to apply the context independently to channels.
    """
    builder = _active_builder()
    builder.push_context(transforms.AlignFunc(duration=duration, func=func))
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)


@contextmanager
def general_transforms(alignment_context: AlignmentKind) -> ContextManager[None]:
    """Arbitrary alignment transformation defined by a subclass instance of
    :class:`~qiskit.pulse.transforms.alignments.AlignmentKind`.

    Args:
        alignment_context: Alignment context instance that defines schedule transformation.

    Yields:
        None

    Raises:
        PulseError: When input ``alignment_context`` is not ``AlignmentKind`` subclasses.
    """
    if not isinstance(alignment_context, AlignmentKind):
        raise exceptions.PulseError("Input alignment context is not `AlignmentKind` subclass.")

    builder = _active_builder()
    builder.push_context(alignment_context)
    try:
        yield
    finally:
        current = builder.pop_context()
        builder.append_subroutine(current)


@contextmanager
def transpiler_settings(**settings) -> ContextManager[None]:
    """Set the currently active transpiler settings for this context.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            print(pulse.active_transpiler_settings())
            with pulse.transpiler_settings(optimization_level=3):
                print(pulse.active_transpiler_settings())

    .. parsed-literal::

        {}
        {'optimization_level': 3}
    """
    builder = _active_builder()
    curr_transpiler_settings = builder.transpiler_settings
    builder.transpiler_settings = collections.ChainMap(settings, curr_transpiler_settings)
    try:
        yield
    finally:
        builder.transpiler_settings = curr_transpiler_settings


@contextmanager
def circuit_scheduler_settings(**settings) -> ContextManager[None]:
    """Set the currently active circuit scheduler settings for this context.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            print(pulse.active_circuit_scheduler_settings())
            with pulse.circuit_scheduler_settings(method='alap'):
                print(pulse.active_circuit_scheduler_settings())

    .. parsed-literal::

       {}
       {'method': 'alap'}

    """
    builder = _active_builder()
    curr_circuit_scheduler_settings = builder.circuit_scheduler_settings
    builder.circuit_scheduler_settings = collections.ChainMap(
        settings, curr_circuit_scheduler_settings
    )
    try:
        yield
    finally:
        builder.circuit_scheduler_settings = curr_circuit_scheduler_settings


@contextmanager
def phase_offset(phase: float, *channels: chans.PulseChannel) -> ContextManager[None]:
    """Shift the phase of input channels on entry into context and undo on exit.

    Examples:

    .. code-block::

        import math

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            with pulse.phase_offset(math.pi, d0):
                pulse.play(pulse.Constant(10, 1.0), d0)

        assert len(pulse_prog.instructions) == 3

    Args:
        phase: Amount of phase offset in radians.
        channels: Channels to offset phase of.

    Yields:
        None
    """
    for channel in channels:
        shift_phase(phase, channel)
    try:
        yield
    finally:
        for channel in channels:
            shift_phase(-phase, channel)


@contextmanager
def frequency_offset(
    frequency: float, *channels: chans.PulseChannel, compensate_phase: bool = False
) -> ContextManager[None]:
    """Shift the frequency of inputs channels on entry into context and undo on exit.

    Examples:

    .. code-block:: python
        :emphasize-lines: 7, 16

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build(backend) as pulse_prog:
            # shift frequency by 1GHz
            with pulse.frequency_offset(1e9, d0):
                pulse.play(pulse.Constant(10, 1.0), d0)

        assert len(pulse_prog.instructions) == 3

        with pulse.build(backend) as pulse_prog:
            # Shift frequency by 1GHz.
            # Undo accumulated phase in the shifted frequency frame
            # when exiting the context.
            with pulse.frequency_offset(1e9, d0, compensate_phase=True):
                pulse.play(pulse.Constant(10, 1.0), d0)

        assert len(pulse_prog.instructions) == 4

    Args:
        frequency: Amount of frequency offset in Hz.
        channels: Channels to offset frequency of.
        compensate_phase: Compensate for accumulated phase accumulated with
            respect to the channels' frame at its initial frequency.

    Yields:
        None
    """
    builder = _active_builder()
    # TODO: Need proper implementation of compensation. t0 may depend on the parent context.
    #  For example, the instruction position within the equispaced context depends on
    #  the current total number of instructions, thus adding more instruction after
    #  offset context may change the t0 when the parent context is transformed.
    t0 = builder.get_context().duration

    for channel in channels:
        shift_frequency(frequency, channel)
    try:
        yield
    finally:
        if compensate_phase:
            duration = builder.get_context().duration - t0

            accumulated_phase = 2 * np.pi * ((duration * builder.get_dt() * frequency) % 1)
            for channel in channels:
                shift_phase(-accumulated_phase, channel)

        for channel in channels:
            shift_frequency(-frequency, channel)


# Channels
def drive_channel(qubit: int) -> chans.DriveChannel:
    """Return ``DriveChannel`` for ``qubit`` on the active builder backend.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            assert pulse.drive_channel(0) == pulse.DriveChannel(0)

    .. note:: Requires the active builder context to have a backend set.
    """
    # backendV2
    if isinstance(active_backend(), BackendV2):
        return active_backend().drive_channel(qubit)
    return active_backend().configuration().drive(qubit)


def measure_channel(qubit: int) -> chans.MeasureChannel:
    """Return ``MeasureChannel`` for ``qubit`` on the active builder backend.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            assert pulse.measure_channel(0) == pulse.MeasureChannel(0)

    .. note:: Requires the active builder context to have a backend set.
    """
    # backendV2
    if isinstance(active_backend(), BackendV2):
        return active_backend().measure_channel(qubit)
    return active_backend().configuration().measure(qubit)


def acquire_channel(qubit: int) -> chans.AcquireChannel:
    """Return ``AcquireChannel`` for ``qubit`` on the active builder backend.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend):
            assert pulse.acquire_channel(0) == pulse.AcquireChannel(0)

    .. note:: Requires the active builder context to have a backend set.
    """
    # backendV2
    if isinstance(active_backend(), BackendV2):
        return active_backend().acquire_channel(qubit)
    return active_backend().configuration().acquire(qubit)


def control_channels(*qubits: Iterable[int]) -> List[chans.ControlChannel]:
    """Return ``ControlChannel`` for ``qubit`` on the active builder backend.

    Return the secondary drive channel for the given qubit -- typically
    utilized for controlling multi-qubit interactions.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()
        with pulse.build(backend):
            assert pulse.control_channels(0, 1) == [pulse.ControlChannel(0)]

    .. note:: Requires the active builder context to have a backend set.

    Args:
      qubits: Tuple or list of ordered qubits of the form
        `(control_qubit, target_qubit)`.

    Returns:
        List of control channels associated with the supplied ordered list
        of qubits.
    """
    # backendV2
    if isinstance(active_backend(), BackendV2):
        return active_backend().control_channel(qubits)
    return active_backend().configuration().control(qubits=qubits)


# Base Instructions
def delay(duration: int, channel: chans.Channel, name: Optional[str] = None):
    """Delay on a ``channel`` for a ``duration``.

    Examples:

    .. code-block::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.delay(10, d0)

    Args:
        duration: Number of cycles to delay for on ``channel``.
        channel: Channel to delay on.
        name: Name of the instruction.
    """
    append_instruction(instructions.Delay(duration, channel, name=name))


def play(
    pulse: Union[library.Pulse, np.ndarray], channel: chans.PulseChannel, name: Optional[str] = None
):
    """Play a ``pulse`` on a ``channel``.

    Examples:

    .. code-block::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.play(pulse.Constant(10, 1.0), d0)

    Args:
        pulse: Pulse to play.
        channel: Channel to play pulse on.
        name: Name of the pulse.
    """
    if not isinstance(pulse, library.Pulse):
        pulse = library.Waveform(pulse)

    append_instruction(instructions.Play(pulse, channel, name=name))


def acquire(
    duration: int,
    qubit_or_channel: Union[int, chans.AcquireChannel],
    register: StorageLocation,
    **metadata: Union[configuration.Kernel, configuration.Discriminator],
):
    """Acquire for a ``duration`` on a ``channel`` and store the result
    in a ``register``.

    Examples:

    .. code-block::

        from qiskit import pulse

        acq0 = pulse.AcquireChannel(0)
        mem0 = pulse.MemorySlot(0)

        with pulse.build() as pulse_prog:
            pulse.acquire(100, acq0, mem0)

            # measurement metadata
            kernel = pulse.configuration.Kernel('linear_discriminator')
            pulse.acquire(100, acq0, mem0, kernel=kernel)

    .. note:: The type of data acquire will depend on the execution ``meas_level``.

    Args:
        duration: Duration to acquire data for
        qubit_or_channel: Either the qubit to acquire data for or the specific
            :class:`~qiskit.pulse.channels.AcquireChannel` to acquire on.
        register: Location to store measured result.
        metadata: Additional metadata for measurement. See
            :class:`~qiskit.pulse.instructions.Acquire` for more information.

    Raises:
        exceptions.PulseError: If the register type is not supported.
    """
    if isinstance(qubit_or_channel, int):
        qubit_or_channel = chans.AcquireChannel(qubit_or_channel)

    if isinstance(register, chans.MemorySlot):
        append_instruction(
            instructions.Acquire(duration, qubit_or_channel, mem_slot=register, **metadata)
        )
    elif isinstance(register, chans.RegisterSlot):
        append_instruction(
            instructions.Acquire(duration, qubit_or_channel, reg_slot=register, **metadata)
        )
    else:
        raise exceptions.PulseError(f'Register of type: "{type(register)}" is not supported')


def set_frequency(frequency: float, channel: chans.PulseChannel, name: Optional[str] = None):
    """Set the ``frequency`` of a pulse ``channel``.

    Examples:

    .. code-block::

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.set_frequency(1e9, d0)

    Args:
        frequency: Frequency in Hz to set channel to.
        channel: Channel to set frequency of.
        name: Name of the instruction.
    """
    append_instruction(instructions.SetFrequency(frequency, channel, name=name))


def shift_frequency(frequency: float, channel: chans.PulseChannel, name: Optional[str] = None):
    """Shift the ``frequency`` of a pulse ``channel``.

    Examples:

    .. code-block:: python
        :emphasize-lines: 6

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.shift_frequency(1e9, d0)

    Args:
        frequency: Frequency in Hz to shift channel frequency by.
        channel: Channel to shift frequency of.
        name: Name of the instruction.
    """
    append_instruction(instructions.ShiftFrequency(frequency, channel, name=name))


def set_phase(phase: float, channel: chans.PulseChannel, name: Optional[str] = None):
    """Set the ``phase`` of a pulse ``channel``.

    Examples:

    .. code-block:: python
        :emphasize-lines: 8

        import math

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.set_phase(math.pi, d0)

    Args:
        phase: Phase in radians to set channel carrier signal to.
        channel: Channel to set phase of.
        name: Name of the instruction.
    """
    append_instruction(instructions.SetPhase(phase, channel, name=name))


def shift_phase(phase: float, channel: chans.PulseChannel, name: Optional[str] = None):
    """Shift the ``phase`` of a pulse ``channel``.

    Examples:

    .. code-block::

        import math

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            pulse.shift_phase(math.pi, d0)

    Args:
        phase: Phase in radians to shift channel carrier signal by.
        channel: Channel to shift phase of.
        name: Name of the instruction.
    """
    append_instruction(instructions.ShiftPhase(phase, channel, name))


def snapshot(label: str, snapshot_type: str = "statevector"):
    """Simulator snapshot.

    Examples:

    .. code-block::

        from qiskit import pulse

        with pulse.build() as pulse_prog:
            pulse.snapshot('first', 'statevector')

    Args:
        label: Label for snapshot.
        snapshot_type: Type of snapshot.
    """
    append_instruction(instructions.Snapshot(label, snapshot_type=snapshot_type))


def call(
    target: Optional[Union[circuit.QuantumCircuit, Schedule, ScheduleBlock]],
    name: Optional[str] = None,
    value_dict: Optional[Dict[ParameterValueType, ParameterValueType]] = None,
    **kw_params: ParameterValueType,
):
    """Call the subroutine within the currently active builder context with arbitrary
    parameters which will be assigned to the target program.

    .. note::

        If the ``target`` program is a :class:`.ScheduleBlock`, then a :class:`.Reference`
        instruction will be created and appended to the current context.
        The ``target`` program will be immediately assigned to the current scope as a subroutine.
        If the ``target`` program is :class:`.Schedule`, it will be wrapped by the
        :class:`.Call` instruction and appended to the current context to avoid
        a mixed representation of :class:`.ScheduleBlock` and :class:`.Schedule`.
        If the ``target`` program is a :class:`.QuantumCircuit` it will be scheduled
        and the new :class:`.Schedule` will be added as a :class:`.Call` instruction.

    Examples:

        1. Calling a schedule block (recommended)

        .. code-block::

            from qiskit import circuit, pulse
            from qiskit.providers.fake_provider import FakeBogotaV2

            backend = FakeBogotaV2()

            with pulse.build() as x_sched:
                pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))

            with pulse.build() as pulse_prog:
                pulse.call(x_sched)

            print(pulse_prog)

        .. parsed-literal::

            ScheduleBlock(
                ScheduleBlock(
                    Play(
                        Gaussian(duration=160, amp=(0.1+0j), sigma=40),
                        DriveChannel(0)
                    ),
                    name="block0",
                    transform=AlignLeft()
                ),
                name="block1",
                transform=AlignLeft()
            )

        The actual program is stored in the reference table attached to the schedule.

        .. code-block::

            print(pulse_prog.references)

        .. parsed-literal::

            ReferenceManager:
              - ('block0', '634b3b50bd684e26a673af1fbd2d6c81'): ScheduleBlock(Play(Gaussian(...

        In addition, you can call a parameterized target program with parameter assignment.

        .. code-block::

            amp = circuit.Parameter("amp")

            with pulse.build() as subroutine:
                pulse.play(pulse.Gaussian(160, amp, 40), pulse.DriveChannel(0))

            with pulse.build() as pulse_prog:
                pulse.call(subroutine, amp=0.1)
                pulse.call(subroutine, amp=0.3)

            print(pulse_prog)

        .. parsed-literal::

            ScheduleBlock(
                ScheduleBlock(
                    Play(
                        Gaussian(duration=160, amp=(0.1+0j), sigma=40),
                        DriveChannel(0)
                    ),
                    name="block2",
                    transform=AlignLeft()
                ),
                ScheduleBlock(
                    Play(
                        Gaussian(duration=160, amp=(0.3+0j), sigma=40),
                        DriveChannel(0)
                    ),
                    name="block2",
                    transform=AlignLeft()
                ),
                name="block3",
                transform=AlignLeft()
            )

        If there is a name collision between parameters, you can distinguish them by specifying
        each parameter object in a python dictionary. For example,

        .. code-block::

            amp1 = circuit.Parameter('amp')
            amp2 = circuit.Parameter('amp')

            with pulse.build() as subroutine:
                pulse.play(pulse.Gaussian(160, amp1, 40), pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(160, amp2, 40), pulse.DriveChannel(1))

            with pulse.build() as pulse_prog:
                pulse.call(subroutine, value_dict={amp1: 0.1, amp2: 0.3})

            print(pulse_prog)

        .. parsed-literal::

            ScheduleBlock(
                ScheduleBlock(
                    Play(Gaussian(duration=160, amp=(0.1+0j), sigma=40), DriveChannel(0)),
                    Play(Gaussian(duration=160, amp=(0.3+0j), sigma=40), DriveChannel(1)),
                    name="block4",
                    transform=AlignLeft()
                ),
                name="block5",
                transform=AlignLeft()
            )

        2. Calling a schedule

        .. code-block::

            x_sched = backend.instruction_schedule_map.get("x", (0,))

            with pulse.build(backend) as pulse_prog:
                pulse.call(x_sched)

            print(pulse_prog)

        .. parsed-literal::

            ScheduleBlock(
                Call(
                    Schedule(
                        (
                            0,
                            Play(
                                Drag(
                                    duration=160,
                                    amp=(0.18989731546729305+0j),
                                    sigma=40,
                                    beta=-1.201258305015517,
                                    name='drag_86a8'
                                ),
                                DriveChannel(0),
                                name='drag_86a8'
                            )
                        ),
                        name="x"
                    ),
                    name='x'
                ),
                name="block6",
                transform=AlignLeft()
            )

        Currently, the backend calibrated gates are provided in the form of :class:`~.Schedule`.
        The parameter assignment mechanism is available also for schedules.
        However, the called schedule is not treated as a reference.

        3. Calling a quantum circuit

        .. code-block::

            backend = FakeBogotaV2()

            qc = circuit.QuantumCircuit(1)
            qc.x(0)

            with pulse.build(backend) as pulse_prog:
                pulse.call(qc)

            print(pulse_prog)

        .. parsed-literal::

            ScheduleBlock(
                Call(
                    Schedule(
                        (
                            0,
                            Play(
                                Drag(
                                    duration=160,
                                    amp=(0.18989731546729305+0j),
                                    sigma=40,
                                    beta=-1.201258305015517,
                                    name='drag_86a8'
                                ),
                                DriveChannel(0),
                                name='drag_86a8'
                            )
                        ),
                        name="circuit-87"
                    ),
                    name='circuit-87'
                ),
                name="block7",
                transform=AlignLeft()
            )

        .. warning::

            Calling a circuit from a schedule is not encouraged. Currently, the Qiskit execution model
            is migrating toward the pulse gate model, where schedules are attached to
            circuits through the :meth:`.QuantumCircuit.add_calibration` method.

    Args:
        target: Target circuit or pulse schedule to call.
        name: Optional. A unique name of subroutine if defined. When the name is explicitly
            provided, one cannot call different schedule blocks with the same name.
        value_dict: Optional. Parameters assigned to the ``target`` program.
            If this dictionary is provided, the ``target`` program is copied and
            then stored in the main built schedule and its parameters are assigned to the given values.
            This dictionary is keyed on :class:`~.Parameter` objects,
            allowing parameter name collision to be avoided.
        kw_params: Alternative way to provide parameters.
            Since this is keyed on the string parameter name,
            the parameters having the same name are all updated together.
            If you want to avoid name collision, use ``value_dict`` with :class:`~.Parameter`
            objects instead.
    """
    _active_builder().call_subroutine(target, name, value_dict, **kw_params)


def reference(name: str, *extra_keys: str):
    """Refer to undefined subroutine by string keys.

    A :class:`~qiskit.pulse.instructions.Reference` instruction is implicitly created
    and a schedule can be separately registered to the reference at a later stage.

    .. code-block:: python

        from qiskit import pulse

        with pulse.build() as main_prog:
            pulse.reference("x_gate", "q0")

        with pulse.build() as subroutine:
            pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))

        main_prog.assign_references(subroutine_dict={("x_gate", "q0"): subroutine})

    Args:
        name: Name of subroutine.
        extra_keys: Helper keys to uniquely specify the subroutine.
    """
    _active_builder().append_reference(name, *extra_keys)


# Directives
def barrier(*channels_or_qubits: Union[chans.Channel, int], name: Optional[str] = None):
    """Barrier directive for a set of channels and qubits.

    This directive prevents the compiler from moving instructions across
    the barrier. Consider the case where we want to enforce that one pulse
    happens after another on separate channels, this can be done with:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)

        with pulse.build(backend) as barrier_pulse_prog:
            pulse.play(pulse.Constant(10, 1.0), d0)
            pulse.barrier(d0, d1)
            pulse.play(pulse.Constant(10, 1.0), d1)

    Of course this could have been accomplished with:

    .. code-block::

        from qiskit.pulse import transforms

        with pulse.build(backend) as aligned_pulse_prog:
            with pulse.align_sequential():
                pulse.play(pulse.Constant(10, 1.0), d0)
                pulse.play(pulse.Constant(10, 1.0), d1)

        barrier_pulse_prog = transforms.target_qobj_transform(barrier_pulse_prog)
        aligned_pulse_prog = transforms.target_qobj_transform(aligned_pulse_prog)

        assert barrier_pulse_prog == aligned_pulse_prog

    The barrier allows the pulse compiler to take care of more advanced
    scheduling alignment operations across channels. For example
    in the case where we are calling an outside circuit or schedule and
    want to align a pulse at the end of one call:

    .. code-block::

        import math

        d0 = pulse.DriveChannel(0)

        with pulse.build(backend) as pulse_prog:
            with pulse.align_right():
                pulse.x(1)
                # Barrier qubit 1 and d0.
                pulse.barrier(1, d0)
                # Due to barrier this will play before the gate on qubit 1.
                pulse.play(pulse.Constant(10, 1.0), d0)
                # This will end at the same time as the pulse above due to
                # the barrier.
                pulse.x(1)

    .. note:: Requires the active builder context to have a backend set if
        qubits are barriered on.

    Args:
        channels_or_qubits: Channels or qubits to barrier.
        name: Name for the barrier
    """
    channels = _qubits_to_channels(*channels_or_qubits)
    if len(channels) > 1:
        append_instruction(directives.RelativeBarrier(*channels, name=name))


# Macros
def macro(func: Callable):
    """Wrap a Python function and activate the parent builder context at calling time.

    This enables embedding Python functions as builder macros. This generates a new
    :class:`pulse.Schedule` that is embedded in the parent builder context with
    every call of the decorated macro function. The decorated macro function will
    behave as if the function code was embedded inline in the parent builder context
    after parameter substitution.


    Examples:

    .. plot::
       :include-source:

        from qiskit import pulse

        @pulse.macro
        def measure(qubit: int):
            pulse.play(pulse.GaussianSquare(16384, 256, 15872), pulse.measure_channel(qubit))
            mem_slot = pulse.MemorySlot(qubit)
            pulse.acquire(16384, pulse.acquire_channel(qubit), mem_slot)

            return mem_slot

        with pulse.build(backend=backend) as sched:
            mem_slot = measure(0)
            print(f"Qubit measured into {mem_slot}")

        sched.draw()


    Args:
        func: The Python function to enable as a builder macro. There are no
            requirements on the signature of the function, any calls to pulse
            builder methods will be added to builder context the wrapped function
            is called from.

    Returns:
        Callable: The wrapped ``func``.
    """
    func_name = getattr(func, "__name__", repr(func))

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _builder = _active_builder()
        # activate the pulse builder before calling the function
        with build(backend=_builder.backend, name=func_name) as built:
            output = func(*args, **kwargs)

        _builder.call_subroutine(built)
        return output

    return wrapper


def measure(
    qubits: Union[List[int], int],
    registers: Union[List[StorageLocation], StorageLocation] = None,
) -> Union[List[StorageLocation], StorageLocation]:
    """Measure a qubit within the currently active builder context.

    At the pulse level a measurement is composed of both a stimulus pulse and
    an acquisition instruction which tells the systems measurement unit to
    acquire data and process it. We provide this measurement macro to automate
    the process for you, but if desired full control is still available with
    :func:`acquire` and :func:`play`.

    To use the measurement it is as simple as specifying the qubit you wish to
    measure:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        qubit = 0

        with pulse.build(backend) as pulse_prog:
            # Do something to the qubit.
            qubit_drive_chan = pulse.drive_channel(0)
            pulse.play(pulse.Constant(100, 1.0), qubit_drive_chan)
            # Measure the qubit.
            reg = pulse.measure(qubit)

    For now it is not possible to do much with the handle to ``reg`` but in the
    future we will support using this handle to a result register to build
    up ones program. It is also possible to supply this register:

    .. code-block::

        with pulse.build(backend) as pulse_prog:
            pulse.play(pulse.Constant(100, 1.0), qubit_drive_chan)
            # Measure the qubit.
            mem0 = pulse.MemorySlot(0)
            reg = pulse.measure(qubit, mem0)

        assert reg == mem0

    .. note:: Requires the active builder context to have a backend set.

    Args:
        qubits: Physical qubit to measure.
        registers: Register to store result in. If not selected the current
            behavior is to return the :class:`MemorySlot` with the same
            index as ``qubit``. This register will be returned.
    Returns:
        The ``register`` the qubit measurement result will be stored in.
    """
    backend = active_backend()

    try:
        qubits = list(qubits)
    except TypeError:
        qubits = [qubits]

    if registers is None:
        registers = [chans.MemorySlot(qubit) for qubit in qubits]
    else:
        try:
            registers = list(registers)
        except TypeError:
            registers = [registers]
    measure_sched = macros.measure(
        qubits=qubits,
        backend=backend,
        qubit_mem_slots={qubit: register.index for qubit, register in zip(qubits, registers)},
    )

    # note this is not a subroutine.
    # just a macro to automate combination of stimulus and acquisition.
    # prepare unique reference name based on qubit and memory slot index.
    qubits_repr = "&".join(map(str, qubits))
    mslots_repr = "&".join(str(r.index) for r in registers)
    _active_builder().call_subroutine(measure_sched, name=f"measure_{qubits_repr}..{mslots_repr}")

    if len(qubits) == 1:
        return registers[0]
    else:
        return registers


def measure_all() -> List[chans.MemorySlot]:
    r"""Measure all qubits within the currently active builder context.

    A simple macro function to measure all of the qubits in the device at the
    same time. This is useful for handling device ``meas_map`` and single
    measurement constraints.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            # Measure all qubits and return associated registers.
            regs = pulse.measure_all()

    .. note::
        Requires the active builder context to have a backend set.

    Returns:
        The ``register``\s the qubit measurement results will be stored in.
    """
    backend = active_backend()
    qubits = range(num_qubits())
    registers = [chans.MemorySlot(qubit) for qubit in qubits]

    measure_sched = macros.measure(
        qubits=qubits,
        backend=backend,
        qubit_mem_slots={qubit: qubit for qubit in qubits},
    )

    # note this is not a subroutine.
    # just a macro to automate combination of stimulus and acquisition.
    _active_builder().call_subroutine(measure_sched, name="measure_all")

    return registers


def delay_qubits(duration: int, *qubits: Union[int, Iterable[int]]):
    r"""Insert delays on all of the :class:`channels.Channel`\s that correspond
    to the input ``qubits`` at the same time.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse3Q

        backend = FakeOpenPulse3Q()

        with pulse.build(backend) as pulse_prog:
            # Delay for 100 cycles on qubits 0, 1 and 2.
            regs = pulse.delay_qubits(100, 0, 1, 2)

    .. note:: Requires the active builder context to have a backend set.

    Args:
        duration: Duration to delay for.
        qubits: Physical qubits to delay on. Delays will be inserted based on
            the channels returned by :func:`pulse.qubit_channels`.
    """
    qubit_chans = set(itertools.chain.from_iterable(qubit_channels(qubit) for qubit in qubits))
    with align_left():
        for chan in qubit_chans:
            delay(duration, chan)


# Gate instructions
def call_gate(gate: circuit.Gate, qubits: Tuple[int, ...], lazy: bool = True):
    """Call a gate and lazily schedule it to its corresponding
    pulse instruction.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favor of tight integration with a circuit
        builder interface which is under development.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.pulse import builder
        from qiskit.circuit.library import standard_gates as gates
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            builder.call_gate(gates.CXGate(), (0, 1))

    We can see the role of the transpiler in scheduling gates by optimizing
    away two consecutive CNOT gates:

    .. code-block::

        with pulse.build(backend) as pulse_prog:
            with pulse.transpiler_settings(optimization_level=3):
                builder.call_gate(gates.CXGate(), (0, 1))
                builder.call_gate(gates.CXGate(), (0, 1))

        assert pulse_prog == pulse.Schedule()

    .. note:: If multiple gates are called in a row they may be optimized by
        the transpiler, depending on the
        :func:`pulse.active_transpiler_settings``.

    .. note:: Requires the active builder context to have a backend set.

    Args:
        gate: Circuit gate instance to call.
        qubits: Qubits to call gate on.
        lazy: If ``false`` the gate will be compiled immediately, otherwise
            it will be added onto a lazily evaluated quantum circuit to be
            compiled when the builder is forced to by a circuit assumption
            being broken, such as the inclusion of a pulse instruction or
            new alignment context.
    """
    _active_builder().call_gate(gate, qubits, lazy=lazy)


def cx(control: int, target: int):  # pylint: disable=invalid-name
    """Call a :class:`~qiskit.circuit.library.standard_gates.CXGate` on the
    input physical qubits.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favor of tight integration with a circuit
        builder interface which is under development.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            pulse.cx(0, 1)

    """
    call_gate(gates.CXGate(), (control, target))


def u1(theta: float, qubit: int):  # pylint: disable=invalid-name
    """Call a :class:`~qiskit.circuit.library.standard_gates.U1Gate` on the
    input physical qubit.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favor of tight integration with a circuit
        builder interface which is under development.

    Examples:

    .. code-block::

        import math

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            pulse.u1(math.pi, 1)

    """
    call_gate(gates.U1Gate(theta), qubit)


def u2(phi: float, lam: float, qubit: int):  # pylint: disable=invalid-name
    """Call a :class:`~qiskit.circuit.library.standard_gates.U2Gate` on the
    input physical qubit.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favor of tight integration with a circuit
        builder interface which is under development.

    Examples:

    .. code-block::

        import math

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            pulse.u2(0, math.pi, 1)

    """
    call_gate(gates.U2Gate(phi, lam), qubit)


def u3(theta: float, phi: float, lam: float, qubit: int):  # pylint: disable=invalid-name
    """Call a :class:`~qiskit.circuit.library.standard_gates.U3Gate` on the
    input physical qubit.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favor of tight integration with a circuit
        builder interface which is under development.

    Examples:

    .. code-block::

        import math

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            pulse.u3(math.pi, 0, math.pi, 1)

    """
    call_gate(gates.U3Gate(theta, phi, lam), qubit)


def x(qubit: int):
    """Call a :class:`~qiskit.circuit.library.standard_gates.XGate` on the
    input physical qubit.

    .. note::
        Calling gates directly within the pulse builder namespace will be
        deprecated in the future in favor of tight integration with a circuit
        builder interface which is under development.

    Examples:

    .. code-block::

        from qiskit import pulse
        from qiskit.providers.fake_provider import FakeOpenPulse2Q

        backend = FakeOpenPulse2Q()

        with pulse.build(backend) as pulse_prog:
            pulse.x(0)

    """
    call_gate(gates.XGate(), qubit)

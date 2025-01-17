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

"""Builder types for the basic control-flow constructs."""

# This file is in circuit.controlflow rather than the root of circuit because the constructs here
# are only intended to be localized to constructing the control flow instructions.  We anticipate
# having a far more complete builder of all circuits, with more classical control and creation, in
# the future.

from __future__ import annotations

import abc
import itertools
import typing
from typing import Collection, Iterable, FrozenSet, Tuple, Union, Optional, Sequence

from qiskit._accelerate.circuit import CircuitData
from qiskit.circuit.classical import expr
from qiskit.circuit.classicalregister import Clbit, ClassicalRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuitdata import CircuitInstruction
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.circuit.register import Register

from ._builder_utils import condition_resources, node_resources

if typing.TYPE_CHECKING:
    import qiskit


class CircuitScopeInterface(abc.ABC):
    """An interface that circuits and builder blocks explicitly fulfill, which contains the primitive
    methods of circuit construction and object validation.

    This allows core circuit methods to be applied to the currently open builder scope, and allows
    the builders to hook into all places where circuit resources might be used.  This allows the
    builders to track the resources being used, without getting in the way of
    :class:`.QuantumCircuit` doing its own thing.
    """

    __slots__ = ()

    @property
    @abc.abstractmethod
    def instructions(self) -> Sequence[CircuitInstruction]:
        """Indexable view onto the :class:`.CircuitInstruction`s backing this scope."""

    @abc.abstractmethod
    def append(
        self, instruction: CircuitInstruction, *, _standard_gate=False
    ) -> CircuitInstruction:
        """Low-level 'append' primitive; this may assume that the qubits, clbits and operation are
        all valid for the circuit.

        Abstraction of :meth:`.QuantumCircuit._append` (the low-level one, not the high-level).

        Args:
            instruction: the resource-validated instruction context object.

        Returns:
            the instruction context object actually appended.  This is not required to be the same
            as the object given (but typically will be).
        """

    @abc.abstractmethod
    def extend(self, data: CircuitData):
        """Appends all instructions from ``data`` to the scope.

        Args:
            data: The instruction listing.
        """

    @abc.abstractmethod
    def resolve_classical_resource(
        self, specifier: Clbit | ClassicalRegister | int
    ) -> Clbit | ClassicalRegister:
        """Resolve a single bit-like classical-resource specifier.

        A resource refers to either a classical bit or a register, where integers index into the
        classical bits of the greater circuit.

        This is called whenever a classical bit or register is being used outside the standard
        :class:`.Clbit` usage of instructions in :meth:`append`, such as in a legacy two-tuple
        condition.

        Args:
            specifier: the classical resource specifier.

        Returns:
            the resolved resource.  This cannot be an integer any more; an integer input is resolved
            into a classical bit.

        Raises:
            CircuitError: if the resource cannot be used by the scope, such as an out-of-range index
                or a :class:`.Clbit` that isn't actually in the circuit.
        """

    @abc.abstractmethod
    def add_uninitialized_var(self, var: expr.Var):
        """Add an uninitialized variable to the circuit scope.

        The general circuit context is responsible for ensuring the variable is initialized.  These
        uninitialized variables are guaranteed to be standalone.

        Args:
            var: the variable to add, if valid.

        Raises:
            CircuitError: if the variable cannot be added, such as because it invalidly shadows or
                redefines an existing name.
        """

    @abc.abstractmethod
    def remove_var(self, var: expr.Var):
        """Remove a variable from the locals of this scope.

        This is only called in the case that an exception occurred while initializing the variable,
        and is not exposed to users.

        Args:
            var: the variable to remove.  It can be assumed that this was already the subject of an
                :meth:`add_uninitialized_var` call.
        """

    @abc.abstractmethod
    def use_var(self, var: expr.Var):
        """Called for every standalone classical real-time variable being used by some circuit
        instruction.

        The given variable is guaranteed to be a stand-alone variable; bit-like resource-wrapping
        variables will have been filtered out and their resources given to
        :meth:`resolve_classical_resource`.

        Args:
            var: the variable to validate.

        Returns:
            the same variable.

        Raises:
            CircuitError: if the variable is not valid for this scope.
        """

    @abc.abstractmethod
    def get_var(self, name: str) -> Optional[expr.Var]:
        """Get the variable (if any) in scope with the given name.

        This should call up to the parent scope if in a control-flow builder scope, in case the
        variable exists in an outer scope.

        Args:
            name: the name of the symbol to lookup.

        Returns:
            the variable if it is found, otherwise ``None``.
        """


class InstructionResources(typing.NamedTuple):
    """The quantum and classical resources used within a particular instruction.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.

    Attributes:
        qubits: A collection of qubits that will be used by the instruction.
        clbits: A collection of clbits that will be used by the instruction.
        qregs: A collection of quantum registers that are used by the instruction.
        cregs: A collection of classical registers that are used by the instruction.
    """

    qubits: Collection[Qubit] = ()
    clbits: Collection[Clbit] = ()
    qregs: Collection[QuantumRegister] = ()
    cregs: Collection[ClassicalRegister] = ()


class InstructionPlaceholder(Instruction, abc.ABC):
    """A fake instruction that lies about its number of qubits and clbits.

    These instances are used to temporarily represent control-flow instructions during the builder
    process, when their lengths cannot be known until the end of the block.  This is necessary to
    allow constructs like::

        with qc.for_loop(range(5)):
            qc.h(0)
            qc.measure(0, 0)
            qc.break_loop().c_if(0, 0)

    since ``qc.break_loop()`` needs to return a (mostly) functional
    :obj:`~qiskit.circuit.Instruction` in order for :meth:`.InstructionSet.c_if` to work correctly.

    When appending a placeholder instruction into a circuit scope, you should create the
    placeholder, and then ask it what resources it should be considered as using from the start by
    calling :meth:`.InstructionPlaceholder.placeholder_instructions`.  This set will be a subset of
    the final resources it asks for, but it is used for initializing resources that *must* be
    supplied, such as the bits used in the conditions of placeholder ``if`` statements.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    _directive = True

    @abc.abstractmethod
    def concrete_instruction(
        self, qubits: FrozenSet[Qubit], clbits: FrozenSet[Clbit]
    ) -> Tuple[Instruction, InstructionResources]:
        """Get a concrete, complete instruction that is valid to act over all the given resources.

        The returned resources may not be the full width of the given resources, but will certainly
        be a subset of them; this can occur if (for example) a placeholder ``if`` statement is
        present, but does not itself contain any placeholder instructions.  For resource efficiency,
        the returned :class:`.ControlFlowOp` will not unnecessarily span all resources, but only the
        ones that it needs.

        .. note::

            The caller of this function is responsible for ensuring that the inputs to this function
            are non-strict supersets of the bits returned by :meth:`placeholder_resources`.

        Any condition added in by a call to :obj:`.Instruction.c_if` will be propagated through, but
        set properties like ``duration`` will not; it doesn't make sense for control-flow operations
        to have pulse scheduling on them.

        Args:
            qubits: The qubits the created instruction should be defined across.
            clbits: The clbits the created instruction should be defined across.

        Returns:
            A full version of the relevant control-flow instruction, and the resources that it uses.
            This is a "proper" instruction instance, as if it had been defined with the correct
            number of qubits and clbits from the beginning.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def placeholder_resources(self) -> InstructionResources:
        """Get the qubit and clbit resources that this placeholder instruction should be considered
        as using before construction.

        This will likely not include *all* resources after the block has been built, but using the
        output of this method ensures that all resources will pass through a
        :meth:`.QuantumCircuit.append` call, even if they come from a placeholder, and consequently
        will be tracked by the scope managers.

        Returns:
            A collection of the quantum and classical resources this placeholder instruction will
            certainly use.
        """
        raise NotImplementedError

    def _copy_mutable_properties(self, instruction: Instruction) -> Instruction:
        """Copy mutable properties from ourselves onto a non-placeholder instruction.

        The mutable properties are expected to be things like ``condition``, added onto a
        placeholder by the :meth:`c_if` method.  This mutates ``instruction``, and returns the same
        instance that was passed.  This is mostly intended to make writing concrete versions of
        :meth:`.concrete_instruction` easy.

        The complete list of mutations is:

        * ``condition``, added by :meth:`c_if`.

        Args:
            instruction: the concrete instruction instance to be mutated.

        Returns:
            The same instruction instance that was passed, but mutated to propagate the tracked
            changes to this class.
        """
        instruction._condition = self._condition
        return instruction

    # Provide some better error messages, just in case something goes wrong during development and
    # the placeholder type leaks out to somewhere visible.

    def assemble(self):
        raise CircuitError("Cannot assemble a placeholder instruction.")

    def repeat(self, n):
        raise CircuitError("Cannot repeat a placeholder instruction.")


class ControlFlowBuilderBlock(CircuitScopeInterface):
    """A lightweight scoped block for holding instructions within a control-flow builder context.

    This class is designed only to be used by :obj:`.QuantumCircuit` as an internal context for
    control-flow builder instructions, and in general should never be instantiated by any code other
    than that.

    Note that the instructions that are added to this scope may not be valid yet, so this elides
    some of the type-checking of :obj:`.QuantumCircuit` until those things are known.

    The general principle of the resource tracking through these builder blocks is that every
    necessary resource should pass through an :meth:`.append` call, so that at the point that
    :meth:`.build` is called, the scope knows all the concrete resources that it requires.  However,
    the scope can also contain "placeholder" instructions, which may need extra resources filling in
    from outer scopes (such as a ``break`` needing to know the width of its containing ``for``
    loop).  This means that :meth:`.build` takes all the *containing* scope's resources as well.
    This does not break the "all resources pass through an append" rule, because the containing
    scope will only begin to build its instructions once it has received them all.

    In short, :meth:`.append` adds resources, and :meth:`.build` may use only a subset of the extra
    ones passed.  This ensures that all instructions know about all the resources they need, even in
    the case of ``break``, but do not block any resources that they do *not* need.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    __slots__ = (
        "_instructions",
        "registers",
        "global_phase",
        "_allow_jumps",
        "_parent",
        "_built",
        "_forbidden_message",
        "_vars_local",
        "_vars_capture",
    )

    def __init__(
        self,
        qubits: Iterable[Qubit],
        clbits: Iterable[Clbit],
        *,
        parent: CircuitScopeInterface,
        registers: Iterable[Register] = (),
        allow_jumps: bool = True,
        forbidden_message: Optional[str] = None,
    ):
        """
        Args:
            qubits: Any qubits this scope should consider itself as using from the beginning.
            clbits: Any clbits this scope should consider itself as using from the beginning.  Along
                with ``qubits``, this is useful for things such as ``if`` and ``while`` loop
                builders, where the classical condition has associated resources, and is known when
                this scope is created.
            registers: Any registers this scope should consider itself as using from the
                beginning.  This is useful for :obj:`.IfElseOp` and :obj:`.WhileLoopOp` instances
                which use a classical register as their condition.
            allow_jumps: Whether this builder scope should allow ``break`` and ``continue``
                statements within it.  This is intended to help give sensible error messages when
                dangerous behavior is encountered, such as using ``break`` inside an ``if`` context
                manager that is not within a ``for`` manager.  This can only be safe if the user is
                going to place the resulting :obj:`.QuantumCircuit` inside a :obj:`.ForLoopOp` that
                uses *exactly* the same set of resources.  We cannot verify this from within the
                builder interface (and it is too expensive to do when the ``for`` op is made), so we
                fail safe, and require the user to use the more verbose, internal form.
            parent: The scope interface of the containing scope.
            forbidden_message: If a string is given here, a :exc:`.CircuitError` will be raised on
                any attempts to append instructions to the scope with this message.  This is used by
                pseudo scopes where the state machine of the builder scopes has changed into a
                position where no instructions should be accepted, such as when inside a ``switch``
                but outside any cases.
        """
        self._instructions = CircuitData(qubits, clbits)
        self.registers = set(registers)
        self.global_phase = 0.0
        self._vars_local = {}
        self._vars_capture = {}
        self._allow_jumps = allow_jumps
        self._parent = parent
        self._built = False
        self._forbidden_message = forbidden_message

    def qubits(self):
        """The set of qubits associated with this scope."""
        return set(self.instructions.qubits)

    def clbits(self):
        """The set of clbits associated with this scope."""
        return set(self.instructions.clbits)

    @property
    def allow_jumps(self):
        """Whether this builder scope should allow ``break`` and ``continue`` statements within it.

        This is intended to help give sensible error messages when dangerous behavior is
        encountered, such as using ``break`` inside an ``if`` context manager that is not within a
        ``for`` manager.  This can only be safe if the user is going to place the resulting
        :obj:`.QuantumCircuit` inside a :obj:`.ForLoopOp` that uses *exactly* the same set of
        resources.  We cannot verify this from within the builder interface (and it is too expensive
        to do when the ``for`` op is made), so we fail safe, and require the user to use the more
        verbose, internal form.
        """
        return self._allow_jumps

    @property
    def instructions(self):
        return self._instructions

    @staticmethod
    def _raise_on_jump(operation):
        # pylint: disable=cyclic-import
        from .break_loop import BreakLoopOp, BreakLoopPlaceholder
        from .continue_loop import ContinueLoopOp, ContinueLoopPlaceholder

        forbidden = (BreakLoopOp, BreakLoopPlaceholder, ContinueLoopOp, ContinueLoopPlaceholder)
        if isinstance(operation, forbidden):
            raise CircuitError(
                f"The current builder scope cannot take a '{operation.name}'"
                " because it is not in a loop."
            )

    def append(
        self, instruction: CircuitInstruction, *, _standard_gate: bool = False
    ) -> CircuitInstruction:
        if self._forbidden_message is not None:
            raise CircuitError(self._forbidden_message)
        if not self._allow_jumps:
            self._raise_on_jump(instruction.operation)
        for b in instruction.qubits:
            self.instructions.add_qubit(b, strict=False)
        for b in instruction.clbits:
            self.instructions.add_clbit(b, strict=False)
        self._instructions.append(instruction)
        return instruction

    def extend(self, data: CircuitData):
        if self._forbidden_message is not None:
            raise CircuitError(self._forbidden_message)
        if not self._allow_jumps:
            data.foreach_op(self._raise_on_jump)
        active_qubits, active_clbits = data.active_bits()
        # Add bits in deterministic order.
        for b in data.qubits:
            if b in active_qubits:
                self.instructions.add_qubit(b, strict=False)
        for b in data.clbits:
            if b in active_clbits:
                self.instructions.add_clbit(b, strict=False)
        self.instructions.extend(data)

    def resolve_classical_resource(self, specifier):
        if self._built:
            raise CircuitError("Cannot add resources after the scope has been built.")
        # Allow the inner resolve to propagate exceptions.
        resource = self._parent.resolve_classical_resource(specifier)
        if isinstance(resource, Clbit):
            self.add_bits((resource,))
        else:
            self.add_register(resource)
        return resource

    def add_uninitialized_var(self, var: expr.Var):
        if self._built:
            raise CircuitError("Cannot add resources after the scope has been built.")
        # We can shadow a name if it was declared in an outer scope, but only if we haven't already
        # captured it ourselves yet.
        if (previous := self._vars_local.get(var.name)) is not None:
            if previous == var:
                raise CircuitError(f"'{var}' is already present in the scope")
            raise CircuitError(f"cannot add '{var}' as its name shadows the existing '{previous}'")
        if var.name in self._vars_capture:
            raise CircuitError(f"cannot add '{var}' as its name shadows the existing '{previous}'")
        self._vars_local[var.name] = var

    def remove_var(self, var: expr.Var):
        if self._built:
            raise RuntimeError("exception handler 'remove_var' called after scope built")
        self._vars_local.pop(var.name)

    def get_var(self, name: str):
        if (out := self._vars_local.get(name)) is not None:
            return out
        return self._parent.get_var(name)

    def use_var(self, var: expr.Var):
        if (local := self._vars_local.get(var.name)) is not None:
            if local == var:
                return
            raise CircuitError(f"cannot use '{var}' which is shadowed by the local '{local}'")
        if self._vars_capture.get(var.name) == var:
            return
        if self._parent.get_var(var.name) != var:
            raise CircuitError(f"cannot close over '{var}', which is not in scope")
        self._parent.use_var(var)
        self._vars_capture[var.name] = var

    def iter_local_vars(self):
        """Iterator over the variables currently declared in this scope."""
        return self._vars_local.values()

    def iter_captured_vars(self):
        """Iterator over the variables currently captured in this scope."""
        return self._vars_capture.values()

    def peek(self) -> CircuitInstruction:
        """Get the value of the most recent instruction tuple in this scope."""
        if not self._instructions:
            raise CircuitError("This scope contains no instructions.")
        return self._instructions[-1]

    def pop(self) -> CircuitInstruction:
        """Get the value of the most recent instruction in this scope, and remove it from this
        object."""
        if not self._instructions:
            raise CircuitError("This scope contains no instructions.")
        return self._instructions.pop()

    def add_bits(self, bits: Iterable[Union[Qubit, Clbit]]):
        """Add extra bits to this scope that are not associated with any concrete instruction yet.

        This is useful for expanding a scope's resource width when it may contain ``break`` or
        ``continue`` statements, or when its width needs to be expanded to match another scope's
        width (as in the case of :obj:`.IfElseOp`).

        Args:
            bits: The qubits and clbits that should be added to a scope.  It is not an error if
                there are duplicates, either within the iterable or with the bits currently in
                scope.

        Raises:
            TypeError: if the provided bit is of an incorrect type.
        """
        for bit in bits:
            if isinstance(bit, Qubit):
                self.instructions.add_qubit(bit, strict=False)
            elif isinstance(bit, Clbit):
                self.instructions.add_clbit(bit, strict=False)
            else:
                raise TypeError(f"Can only add qubits or classical bits, but received '{bit}'.")

    def add_register(self, register: Register):
        """Add a :obj:`.Register` to the set of resources used by this block, ensuring that
        all bits contained within are also accounted for.

        Args:
            register: the register to add to the block.
        """
        if register in self.registers:
            # Fast return to avoid iterating through the bits.
            return
        self.registers.add(register)
        self.add_bits(register)

    def build(
        self, all_qubits: FrozenSet[Qubit], all_clbits: FrozenSet[Clbit]
    ) -> "qiskit.circuit.QuantumCircuit":
        """Build this scoped block into a complete :obj:`.QuantumCircuit` instance.

        This will build a circuit which contains all of the necessary qubits and clbits and no
        others.

        The ``qubits`` and ``clbits`` arguments should be sets that contains all the resources in
        the outer scope; these will be passed down to inner placeholder instructions, so they can
        apply themselves across the whole scope should they need to.  The resulting
        :obj:`.QuantumCircuit` will be defined over a (nonstrict) subset of these resources.  This
        is used to let ``break`` and ``continue`` span all resources, even if they are nested within
        several :obj:`.IfElsePlaceholder` objects, without requiring :obj:`.IfElsePlaceholder`
        objects *without* any ``break`` or ``continue`` statements to be full-width.

        Args:
            all_qubits: all the qubits in the containing scope of this block.  The block may expand
                to use some or all of these qubits, but will never gain qubits that are not in this
                set.
            all_clbits: all the clbits in the containing scope of this block.  The block may expand
                to use some or all of these clbits, but will never gain clbits that are not in this
                set.

        Returns:
            A circuit containing concrete versions of all the instructions that were in the scope,
            and using the minimal set of resources necessary to support them, within the enclosing
            scope.
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit, SwitchCaseOp

        # There's actually no real problem with building a scope more than once.  This flag is more
        # so _other_ operations, which aren't safe can be forbidden, such as mutating instructions
        # that may have been built into other objects.
        self._built = True

        if self._forbidden_message is not None:
            # Reaching this implies a logic error in the builder interface.
            raise RuntimeError("Cannot build a forbidden scope. Please report this as a bug.")

        potential_qubits = set(all_qubits) - self.qubits()
        potential_clbits = set(all_clbits) - self.clbits()

        # We start off by only giving the QuantumCircuit the qubits we _know_ it will need, and add
        # more later as needed.
        out = QuantumCircuit(
            self._instructions.qubits,
            self._instructions.clbits,
            *self.registers,
            global_phase=self.global_phase,
            captures=self._vars_capture.values(),
        )
        for var in self._vars_local.values():
            # The requisite `Store` instruction to initialise the variable will have been appended
            # into the instructions.
            out.add_uninitialized_var(var)

        # Maps placeholder index to the newly concrete instruction.
        placeholder_to_concrete = {}

        def update_registers(index, op):
            if isinstance(op, InstructionPlaceholder):
                op, resources = op.concrete_instruction(all_qubits, all_clbits)
                qubits = tuple(resources.qubits)
                clbits = tuple(resources.clbits)
                placeholder_to_concrete[index] = CircuitInstruction(op, qubits, clbits)
                # We want to avoid iterating over the tuples unnecessarily if there's no chance
                # we'll need to add bits to the circuit.
                if potential_qubits and qubits:
                    add_qubits = potential_qubits.intersection(qubits)
                    if add_qubits:
                        potential_qubits.difference_update(add_qubits)
                        out.add_bits(add_qubits)
                if potential_clbits and clbits:
                    add_clbits = potential_clbits.intersection(clbits)
                    if add_clbits:
                        potential_clbits.difference_update(add_clbits)
                        out.add_bits(add_clbits)
                for register in itertools.chain(resources.qregs, resources.cregs):
                    if register not in self.registers:
                        # As of 2021-12-09, QuantumCircuit doesn't have an efficient way to check if
                        # a register is already present, so we use our own tracking.
                        self.add_register(register)
                        out.add_register(register)
            if getattr(op, "_condition", None) is not None:
                for register in condition_resources(op._condition).cregs:
                    if register not in self.registers:
                        self.add_register(register)
                        out.add_register(register)
            elif isinstance(op, SwitchCaseOp):
                target = op.target
                if isinstance(target, Clbit):
                    target_registers = ()
                elif isinstance(target, ClassicalRegister):
                    target_registers = (target,)
                else:
                    target_registers = node_resources(target).cregs
                for register in target_registers:
                    if register not in self.registers:
                        self.add_register(register)
                        out.add_register(register)

        # Update registers and bits of 'out'.
        self._instructions.foreach_op_indexed(update_registers)

        # Create the concrete instruction listing.
        out_data = self._instructions.copy()
        out_data.replace_bits(out.qubits, out.clbits)
        for i, instruction in placeholder_to_concrete.items():
            out_data[i] = instruction

        # Add listing to 'out'.
        out._current_scope().extend(out_data)
        return out

    def copy(self) -> "ControlFlowBuilderBlock":
        """Return a semi-shallow copy of this builder block.

        The instruction lists and sets of qubits and clbits will be new instances (so mutations will
        not propagate), but any :obj:`.Instruction` instances within them will not be copied.

        Returns:
            a semi-shallow copy of this object.
        """
        out = type(self).__new__(type(self))
        out._instructions = self._instructions.copy()
        out.registers = self.registers.copy()
        out.global_phase = self.global_phase
        out._vars_local = self._vars_local.copy()
        out._vars_capture = self._vars_capture.copy()
        out._parent = self._parent
        out._allow_jumps = self._allow_jumps
        out._forbidden_message = self._forbidden_message
        return out

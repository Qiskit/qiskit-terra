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

"""Circuit operation representing an ``if/else`` statement."""

from __future__ import annotations

from typing import Optional, Union, Iterable, TYPE_CHECKING
import itertools

from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.classical import expr
from qiskit.circuit.instructionset import InstructionSet
from qiskit.circuit.exceptions import CircuitError

from .builder import ControlFlowBuilderBlock, InstructionPlaceholder, InstructionResources
from .control_flow import ControlFlowOp
from ._builder_utils import (
    partition_registers,
    unify_circuit_resources,
    validate_condition,
    condition_resources,
)

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


# This is just an indication of what's actually meant to be the public API.
__all__ = ("IfElseOp",)


class IfElseOp(ControlFlowOp):
    """A circuit operation which executes a program (``true_body``) if a
    provided condition (``condition``) evaluates to true, and
    optionally evaluates another program (``false_body``) otherwise.

    If provided, ``false_body`` must be of the same ``num_qubits`` and
    ``num_clbits`` as ``true_body``.

    The classical bits used in ``condition`` must be a subset of those attached
    to the circuit on which this ``IfElseOp`` will be appended.
    """

    def __init__(
        self,
        condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr,
        true_body: QuantumCircuit,
        false_body: QuantumCircuit | None = None,
        label: str | None = None,
    ):
        """
        Args:
            condition: A condition to be evaluated in real time during circuit execution which,
                if true, will trigger the evaluation of ``true_body``. Can be
                specified as either a tuple of a ``ClassicalRegister`` to be
                tested for equality with a given ``int``, or as a tuple of a
                ``Clbit`` to be compared to either a ``bool`` or an ``int``.
            true_body: A program to be executed if ``condition`` evaluates
                to true.
            false_body: A optional program to be executed if ``condition``
                evaluates to false.
            label: An optional label for identifying the instruction.
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit

        # Type checking generally left to @params.setter, but required here for
        # finding num_qubits and num_clbits.
        if not isinstance(true_body, QuantumCircuit):
            raise CircuitError(
                "IfElseOp expects a true_body parameter "
                f"of type QuantumCircuit, but received {type(true_body)}."
            )

        num_qubits = true_body.num_qubits
        num_clbits = true_body.num_clbits

        super().__init__("if_else", num_qubits, num_clbits, [true_body, false_body], label=label)

        self._condition = validate_condition(condition)

    @property
    def params(self):
        return self._params

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, value):
        self._condition = value

    @params.setter
    def params(self, parameters):
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit

        true_body, false_body = parameters

        if not isinstance(true_body, QuantumCircuit):
            raise CircuitError(
                "IfElseOp expects a true_body parameter of type "
                f"QuantumCircuit, but received {type(true_body)}."
            )

        if true_body.num_qubits != self.num_qubits or true_body.num_clbits != self.num_clbits:
            raise CircuitError(
                "Attempted to assign a true_body parameter with a num_qubits or "
                "num_clbits different than that of the IfElseOp. "
                f"IfElseOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} "
                f"Supplied body num_qubits/clbits: {true_body.num_qubits}/{true_body.num_clbits}."
            )

        if false_body is not None:
            if not isinstance(false_body, QuantumCircuit):
                raise CircuitError(
                    "IfElseOp expects a false_body parameter of type "
                    f"QuantumCircuit, but received {type(false_body)}."
                )

            if false_body.num_qubits != self.num_qubits or false_body.num_clbits != self.num_clbits:
                raise CircuitError(
                    "Attempted to assign a false_body parameter with a num_qubits or "
                    "num_clbits different than that of the IfElseOp. "
                    f"IfElseOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} "
                    f"Supplied body num_qubits/clbits: {false_body.num_qubits}/{false_body.num_clbits}."
                )

        self._params = [true_body, false_body]

    @property
    def blocks(self):
        if self.params[1] is None:
            return (self.params[0],)
        else:
            return (self.params[0], self.params[1])

    def replace_blocks(self, blocks: Iterable[QuantumCircuit]) -> "IfElseOp":
        """Replace blocks and return new instruction.

        Args:
            blocks: Iterable of circuits for "if" and "else" condition. If there is no "else"
                circuit it may be set to None or omitted.

        Returns:
            New IfElseOp with replaced blocks.
        """

        true_body, false_body = (
            ablock for ablock, _ in itertools.zip_longest(blocks, range(2), fillvalue=None)
        )
        return IfElseOp(self._condition, true_body, false_body=false_body, label=self.label)

    def c_if(self, classical, val):
        raise NotImplementedError(
            "IfElseOp cannot be classically controlled through Instruction.c_if. "
            "Please nest it in an IfElseOp instead."
        )


class IfElsePlaceholder(InstructionPlaceholder):
    """A placeholder instruction to use in control-flow context managers, when calculating the
    number of resources this instruction should block is deferred until the construction of the
    outer loop.

    This generally should not be instantiated manually; only :obj:`.IfContext` and
    :obj:`.ElseContext` should do it when they need to defer creation of the concrete instruction.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    def __init__(
        self,
        condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr,
        true_block: ControlFlowBuilderBlock,
        false_block: ControlFlowBuilderBlock | None = None,
        *,
        label: Optional[str] = None,
    ):
        """
        Args:
            condition: the condition to execute the true block on.  This has the same semantics as
                the ``condition`` argument to :obj:`.IfElseOp`.
            true_block: the unbuilt scope block that will become the "true" branch at creation time.
            false_block: if given, the unbuilt scope block that will become the "false" branch at
                creation time.
            label: the label to give the operator when it is created.
        """
        # These are protected names because we're not trying to clash with parent attributes.
        self.__true_block = true_block
        self.__false_block: Optional[ControlFlowBuilderBlock] = false_block
        self.__resources = self._calculate_placeholder_resources()
        super().__init__(
            "if_else", len(self.__resources.qubits), len(self.__resources.clbits), [], label=label
        )
        # Set the condition after super().__init__() has initialized it to None.
        self._condition = validate_condition(condition)

    def with_false_block(self, false_block: ControlFlowBuilderBlock) -> "IfElsePlaceholder":
        """Return a new placeholder instruction, with the false block set to the given value,
        updating the bits used by both it and the true body, if necessary.

        It is an error to try and set the false block on a placeholder that already has one.

        Args:
            false_block: The (unbuilt) instruction scope to set the false body to.

        Returns:
            A new placeholder, with ``false_block`` set to the given input, and both true and false
            blocks expanded to account for all resources.

        Raises:
            CircuitError: if the false block of this placeholder instruction is already set.
        """
        if self.__false_block is not None:
            raise CircuitError(f"false block is already set to {self.__false_block}")
        true_block = self.__true_block.copy()
        true_bits = true_block.qubits() | true_block.clbits()
        false_bits = false_block.qubits() | false_block.clbits()
        true_block.add_bits(false_bits - true_bits)
        false_block.add_bits(true_bits - false_bits)
        return type(self)(self._condition, true_block, false_block, label=self.label)

    def registers(self):
        """Get the registers used by the interior blocks."""
        if self.__false_block is None:
            return self.__true_block.registers.copy()
        return self.__true_block.registers | self.__false_block.registers

    def _calculate_placeholder_resources(self) -> InstructionResources:
        """Get the placeholder resources (see :meth:`.placeholder_resources`).

        This is a separate function because we use the resources during the initialization to
        determine how we should set our ``num_qubits`` and ``num_clbits``, so we implement the
        public version as a cache access for efficiency.
        """
        if self.__false_block is None:
            qregs, cregs = partition_registers(self.__true_block.registers)
            return InstructionResources(
                qubits=tuple(self.__true_block.qubits()),
                clbits=tuple(self.__true_block.clbits()),
                qregs=tuple(qregs),
                cregs=tuple(cregs),
            )
        true_qregs, true_cregs = partition_registers(self.__true_block.registers)
        false_qregs, false_cregs = partition_registers(self.__false_block.registers)
        return InstructionResources(
            qubits=tuple(self.__true_block.qubits() | self.__false_block.qubits()),
            clbits=tuple(self.__true_block.clbits() | self.__false_block.clbits()),
            qregs=tuple(true_qregs) + tuple(false_qregs),
            cregs=tuple(true_cregs) + tuple(false_cregs),
        )

    def placeholder_resources(self):
        # All the elements of our InstructionResources are immutable (tuple, Bit and Register).
        return self.__resources

    def concrete_instruction(self, qubits, clbits):
        current_qubits = self.__true_block.qubits()
        current_clbits = self.__true_block.clbits()
        if self.__false_block is not None:
            current_qubits = current_qubits | self.__false_block.qubits()
            current_clbits = current_clbits | self.__false_block.clbits()
        all_bits = qubits | clbits
        current_bits = current_qubits | current_clbits
        if current_bits - all_bits:
            # This _shouldn't_ trigger if the context managers are being used correctly, but is here
            # to make any potential logic errors noisy.
            raise CircuitError(
                "This block contains bits that are not in the operands sets:"
                f" {current_bits - all_bits!r}"
            )
        true_body = self.__true_block.build(qubits, clbits)
        if self.__false_block is None:
            false_body = None
        else:
            # The bodies are not compelled to use all the resources that the
            # ControlFlowBuilderBlock.build calls get passed, but they do need to be as wide as each
            # other.  Now we ensure that they are.
            true_body, false_body = unify_circuit_resources(
                (true_body, self.__false_block.build(qubits, clbits))
            )
        return (
            self._copy_mutable_properties(
                IfElseOp(self._condition, true_body, false_body, label=self.label)
            ),
            InstructionResources(
                qubits=tuple(true_body.qubits),
                clbits=tuple(true_body.clbits),
                qregs=tuple(true_body.qregs),
                cregs=tuple(true_body.cregs),
            ),
        )

    def c_if(self, classical, val):
        raise NotImplementedError(
            "IfElseOp cannot be classically controlled through Instruction.c_if. "
            "Please nest it in another IfElseOp instead."
        )


class IfContext:
    """A context manager for building up ``if`` statements onto circuits in a natural order, without
    having to construct the statement body first.

    The return value of this context manager can be used immediately following the block to create
    an attached ``else`` statement.

    This context should almost invariably be created by a :meth:`.QuantumCircuit.if_test` call, and
    the resulting instance is a "friend" of the calling circuit.  The context will manipulate the
    circuit's defined scopes when it is entered (by pushing a new scope onto the stack) and exited
    (by popping its scope, building it, and appending the resulting :obj:`.IfElseOp`).

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    __slots__ = ("_appended_instructions", "_circuit", "_condition", "_in_loop", "_label")

    def __init__(
        self,
        circuit: QuantumCircuit,
        condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr,
        *,
        in_loop: bool,
        label: str | None = None,
    ):
        self._circuit = circuit
        self._condition = validate_condition(condition)
        self._label = label
        self._appended_instructions = None
        self._in_loop = in_loop

    # Only expose the necessary public interface, and make it read-only.  If Python had friend
    # classes, or a "protected" access modifier, that's what we'd use (since these are only
    # necessary for ElseContext), but alas.

    @property
    def circuit(self) -> QuantumCircuit:
        """Get the circuit that this context manager is attached to."""
        return self._circuit

    @property
    def condition(self) -> tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr:
        """Get the expression that this statement is conditioned on."""
        return self._condition

    @property
    def appended_instructions(self) -> Union[InstructionSet, None]:
        """Get the instruction set that was created when this block finished.  If the block has not
        yet finished, then this will be ``None``."""
        return self._appended_instructions

    @property
    def in_loop(self) -> bool:
        """Whether this context manager is enclosed within a loop."""
        return self._in_loop

    def __enter__(self):
        resources = condition_resources(self._condition)
        self._circuit._push_scope(
            clbits=resources.clbits,
            registers=resources.cregs,
            allow_jumps=self._in_loop,
        )
        return ElseContext(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # If we're leaving the context manager because an exception was raised, there's nothing
            # to do except restore the circuit state.
            self._circuit._pop_scope()
            return False
        true_block = self._circuit._pop_scope()
        if self._in_loop:
            # It's possible that we don't actually have any placeholder instructions in our scope,
            # but we still need to emit a placeholder instruction here in case we get an ``else``
            # attached which _does_ gain them.  We emit a placeholder to defer defining the
            # resources we use until the containing loop concludes, to support ``break``.
            operation = IfElsePlaceholder(self._condition, true_block, label=self._label)
            resources = operation.placeholder_resources()
            self._appended_instructions = self._circuit.append(
                operation, resources.qubits, resources.clbits
            )
        else:
            # If we're not in a loop, we don't need to be worried about passing in any outer-scope
            # resources because there can't be anything that will consume them.
            true_body = true_block.build(true_block.qubits(), true_block.clbits())
            self._appended_instructions = self._circuit.append(
                IfElseOp(self._condition, true_body=true_body, false_body=None, label=self._label),
                tuple(true_body.qubits),
                tuple(true_body.clbits),
            )
        return False


class ElseContext:
    """A context manager for building up an ``else`` statements onto circuits in a natural order,
    without having to construct the statement body first.

    Instances of this context manager should only ever be gained as the output of the
    :obj:`.IfContext` manager, so they know what they refer to.  Instances of this context are
    "friends" of the circuit that created the :obj:`.IfContext` that in turn created this object.
    The context will manipulate the circuit's defined scopes when it is entered (by popping the old
    :obj:`.IfElseOp` if it exists and pushing a new scope onto the stack) and exited (by popping its
    scope, building it, and appending the resulting :obj:`.IfElseOp`).

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    __slots__ = ("_if_instruction", "_if_registers", "_if_context", "_used")

    def __init__(self, if_context: IfContext):
        # We want to avoid doing any processing until we're actually used, because the `if` block
        # likely isn't finished yet, and we want to have as small a penalty a possible if you don't
        # use an `else` branch.
        self._if_instruction = None
        self._if_registers = None
        self._if_context = if_context
        self._used = False

    def __enter__(self):
        if self._used:
            raise CircuitError("Cannot re-use an 'else' context.")
        self._used = True
        appended_instructions = self._if_context.appended_instructions
        circuit = self._if_context.circuit
        if appended_instructions is None:
            raise CircuitError("Cannot attach an 'else' branch to an incomplete 'if' block.")
        if len(appended_instructions) != 1:
            # I'm not even sure how you'd get this to trigger, but just in case...
            raise CircuitError("Cannot attach an 'else' to a broadcasted 'if' block.")
        appended = appended_instructions[0]
        instruction = circuit._peek_previous_instruction_in_scope()
        if appended.operation is not instruction.operation:
            raise CircuitError(
                "The 'if' block is not the most recent instruction in the circuit."
                f" Expected to find: {appended!r}, but instead found: {instruction!r}."
            )
        self._if_instruction = circuit._pop_previous_instruction_in_scope()
        if isinstance(self._if_instruction.operation, IfElseOp):
            self._if_registers = set(self._if_instruction.operation.blocks[0].cregs).union(
                self._if_instruction.operation.blocks[0].qregs
            )
        else:
            self._if_registers = self._if_instruction.operation.registers()
        circuit._push_scope(
            self._if_instruction.qubits,
            self._if_instruction.clbits,
            registers=self._if_registers,
            allow_jumps=self._if_context.in_loop,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        circuit = self._if_context.circuit
        if exc_type is not None:
            # If we're leaving the context manager because an exception was raised, we need to
            # restore the "if" block we popped off.  At that point, it's safe to re-use this context
            # manager, assuming nothing else untoward happened to the circuit, but that's checked by
            # the __enter__ method.
            circuit._pop_scope()
            circuit._append(self._if_instruction)
            self._used = False
            return False

        false_block = circuit._pop_scope()
        # `if_block` is a placeholder if this context is in a loop, and a concrete instruction if it
        # is not.
        if isinstance(self._if_instruction.operation, IfElsePlaceholder):
            if_operation = self._if_instruction.operation.with_false_block(false_block)
            resources = if_operation.placeholder_resources()
            circuit.append(if_operation, resources.qubits, resources.clbits)
        else:
            # In this case, we need to update both true_body and false_body to have exactly the same
            # widths.  Passing extra resources to `ControlFlowBuilderBlock.build` doesn't _compel_
            # the resulting object to use them (because it tries to be minimal), so it's best to
            # pass it nothing extra (allows some fast path constructions), and add all necessary
            # bits onto the circuits at the end.
            true_body = self._if_instruction.operation.blocks[0]
            false_body = false_block.build(false_block.qubits(), false_block.clbits())
            true_body, false_body = unify_circuit_resources((true_body, false_body))
            circuit.append(
                IfElseOp(
                    self._if_context.condition,
                    true_body,
                    false_body,
                    label=self._if_instruction.operation.label,
                ),
                tuple(true_body.qubits),
                tuple(true_body.clbits),
            )
        return False

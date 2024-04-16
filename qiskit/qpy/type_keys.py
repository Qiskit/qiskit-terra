# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-return-statements

"""
QPY Type keys for several namespace.
"""
from __future__ import annotations

from abc import abstractmethod
from enum import Enum, IntEnum

import numpy as np

from qiskit.circuit import (
    Gate,
    Instruction,
    QuantumCircuit,
    ControlledGate,
    CASE_DEFAULT,
    Clbit,
    ClassicalRegister,
)
from qiskit.circuit.annotated_operation import AnnotatedOperation, Modifier
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.pulse.channels import (
    Channel,
    DriveChannel,
    MeasureChannel,
    ControlChannel,
    AcquireChannel,
    MemorySlot,
    RegisterSlot,
)
from qiskit.pulse.configuration import Discriminator, Kernel
from qiskit.pulse.instructions import (
    Acquire,
    Play,
    Delay,
    SetFrequency,
    ShiftFrequency,
    SetPhase,
    ShiftPhase,
    RelativeBarrier,
    TimeBlockade,
    Reference,
)
from qiskit.pulse.library import Waveform, SymbolicPulse
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.pulse.transforms.alignments import (
    AlignLeft,
    AlignRight,
    AlignSequential,
    AlignEquispaced,
)
from qiskit.qpy import exceptions


class TypeKeyBase(bytes, Enum):
    """Abstract baseclass for type key Enums."""

    @classmethod
    @abstractmethod
    def assign(cls, obj):
        """Assign type key to given object.

        Args:
            obj (any): Arbitrary object to evaluate.

        Returns:
            TypeKey: Corresponding key object.
        """
        pass

    @classmethod
    @abstractmethod
    def retrieve(cls, type_key: bytes):
        """Get a class from given type key.

        Args:
            type_key (bytes): Object type key.

        Returns:
            any: Corresponding class.
        """
        pass


class Value(TypeKeyBase):
    """Type key enum for value object."""

    INTEGER = b"i"
    FLOAT = b"f"
    COMPLEX = b"c"
    CASE_DEFAULT = b"d"
    REGISTER = b"R"
    NUMPY_OBJ = b"n"
    PARAMETER = b"p"
    PARAMETER_VECTOR = b"v"
    PARAMETER_EXPRESSION = b"e"
    STRING = b"s"
    NULL = b"z"
    EXPRESSION = b"x"
    MODIFIER = b"m"

    @classmethod
    def assign(cls, obj: object) -> "Value":
        if isinstance(obj, int):
            return cls.INTEGER
        if isinstance(obj, float):
            return cls.FLOAT
        if isinstance(obj, complex):
            return cls.COMPLEX
        if isinstance(obj, (np.integer, np.floating, np.complexfloating, np.ndarray)):
            return cls.NUMPY_OBJ
        if isinstance(obj, ParameterVectorElement):
            return cls.PARAMETER_VECTOR
        if isinstance(obj, Parameter):
            return cls.PARAMETER
        if isinstance(obj, ParameterExpression):
            return cls.PARAMETER_EXPRESSION
        if isinstance(obj, str):
            return cls.STRING
        if isinstance(obj, (Clbit, ClassicalRegister)):
            return cls.REGISTER
        if obj is None:
            return cls.NULL
        if obj is CASE_DEFAULT:
            return cls.CASE_DEFAULT
        if isinstance(obj, expr.Expr):
            return cls.EXPRESSION
        if isinstance(obj, Modifier):
            return cls.MODIFIER

        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class Condition(IntEnum):
    """Type keys for the ``conditional_key`` field of the INSTRUCTION struct."""

    # This class is deliberately raw integers and not in terms of ASCII characters for backwards
    # compatiblity in the form as an old Boolean value was expanded; `NONE` and `TWO_TUPLE` must
    # have the enumeration values 0 and 1.

    NONE = 0
    TWO_TUPLE = 1
    EXPRESSION = 2


class Container(TypeKeyBase):
    """Typle key enum for container-like object."""

    RANGE = b"r"
    TUPLE = b"t"

    @classmethod
    def assign(cls, obj) -> bytes:
        if isinstance(obj, range):
            return cls.RANGE
        if isinstance(obj, tuple):
            return cls.TUPLE

        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class CircuitInstruction(TypeKeyBase):
    """Type key enum for circuit instruction object."""

    INSTRUCTION = b"i"
    GATE = b"g"
    PAULI_EVOL_GATE = b"p"
    CONTROLLED_GATE = b"c"
    ANNOTATED_OPERATION = b"a"

    @classmethod
    def assign(
        cls, obj: PauliEvolutionGate | ControlledGate | AnnotatedOperation | Gate | Instruction
    ) -> bytes:
        if isinstance(obj, PauliEvolutionGate):
            return cls.PAULI_EVOL_GATE
        if isinstance(obj, ControlledGate):
            return cls.CONTROLLED_GATE
        if isinstance(obj, AnnotatedOperation):
            return cls.ANNOTATED_OPERATION
        if isinstance(obj, Gate):
            return cls.GATE
        if isinstance(obj, Instruction):
            return cls.INSTRUCTION

        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class ScheduleAlignment(TypeKeyBase):
    """Type key enum for schedule block alignment context object."""

    LEFT = b"l"
    RIGHT = b"r"
    SEQUENTIAL = b"s"
    EQUISPACED = b"e"

    # AlignFunc is not serializable due to the callable in context parameter

    @classmethod
    def assign(cls, obj: AlignLeft | AlignRight | AlignSequential | AlignEquispaced) -> bytes:
        if isinstance(obj, AlignLeft):
            return cls.LEFT
        if isinstance(obj, AlignRight):
            return cls.RIGHT
        if isinstance(obj, AlignSequential):
            return cls.SEQUENTIAL
        if isinstance(obj, AlignEquispaced):
            return cls.EQUISPACED

        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key: bytes):
        if type_key == cls.LEFT:
            return AlignLeft
        if type_key == cls.RIGHT:
            return AlignRight
        if type_key == cls.SEQUENTIAL:
            return AlignSequential
        if type_key == cls.EQUISPACED:
            return AlignEquispaced

        raise exceptions.QpyError(
            f"A class corresponding to type key '{type_key}' is not found in {cls.__name__} namespace."
        )


class ScheduleInstruction(TypeKeyBase):
    """Type key enum for schedule instruction object."""

    ACQUIRE = b"a"
    PLAY = b"p"
    DELAY = b"d"
    SET_FREQUENCY = b"f"
    SHIFT_FREQUENCY = b"g"
    SET_PHASE = b"q"
    SHIFT_PHASE = b"r"
    BARRIER = b"b"
    TIME_BLOCKADE = b"t"
    REFERENCE = b"y"

    # 's' is reserved by ScheduleBlock, i.e. block can be nested as an element.
    # Call instructon is not supported by QPY.
    # This instruction has been excluded from ScheduleBlock instructions with
    # qiskit-terra/#8005 and new instruction Reference will be added instead.
    # Call is only applied to Schedule which is not supported by QPY.
    # Also snapshot is not suppored because of its limited usecase.

    @classmethod
    def assign(
        cls,
        obj: (
            Acquire
            | Play
            | Delay
            | SetFrequency
            | ShiftFrequency
            | SetPhase
            | ShiftPhase
            | RelativeBarrier
            | TimeBlockade
            | Reference
        ),
    ) -> bytes:
        if isinstance(obj, Acquire):
            return cls.ACQUIRE
        if isinstance(obj, Play):
            return cls.PLAY
        if isinstance(obj, Delay):
            return cls.DELAY
        if isinstance(obj, SetFrequency):
            return cls.SET_FREQUENCY
        if isinstance(obj, ShiftFrequency):
            return cls.SHIFT_FREQUENCY
        if isinstance(obj, SetPhase):
            return cls.SET_PHASE
        if isinstance(obj, ShiftPhase):
            return cls.SHIFT_PHASE
        if isinstance(obj, RelativeBarrier):
            return cls.BARRIER
        if isinstance(obj, TimeBlockade):
            return cls.TIME_BLOCKADE
        if isinstance(obj, Reference):
            return cls.REFERENCE

        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        if type_key == cls.ACQUIRE:
            return Acquire
        if type_key == cls.PLAY:
            return Play
        if type_key == cls.DELAY:
            return Delay
        if type_key == cls.SET_FREQUENCY:
            return SetFrequency
        if type_key == cls.SHIFT_FREQUENCY:
            return ShiftFrequency
        if type_key == cls.SET_PHASE:
            return SetPhase
        if type_key == cls.SHIFT_PHASE:
            return ShiftPhase
        if type_key == cls.BARRIER:
            return RelativeBarrier
        if type_key == cls.TIME_BLOCKADE:
            return TimeBlockade
        if type_key == cls.REFERENCE:
            return Reference

        raise exceptions.QpyError(
            f"A class corresponding to type key '{type_key}' is not found in {cls.__name__} namespace."
        )


class ScheduleOperand(TypeKeyBase):
    """Type key enum for schedule instruction operand object."""

    WAVEFORM = b"w"
    SYMBOLIC_PULSE = b"s"
    CHANNEL = b"c"
    KERNEL = b"k"
    DISCRIMINATOR = b"d"

    # We need to have own string type definition for operands of schedule instruction.
    # Note that string type is already defined in the Value namespace,
    # but its key "s" conflicts with the SYMBOLIC_PULSE in the ScheduleOperand namespace.
    # New in QPY version 7.
    OPERAND_STR = b"o"

    @classmethod
    def assign(cls, obj) -> bytes:
        if isinstance(obj, Waveform):
            return cls.WAVEFORM
        if isinstance(obj, SymbolicPulse):
            return cls.SYMBOLIC_PULSE
        if isinstance(obj, Channel):
            return cls.CHANNEL
        if isinstance(obj, str):
            return cls.OPERAND_STR
        if isinstance(obj, Kernel):
            return cls.KERNEL
        if isinstance(obj, Discriminator):
            return cls.DISCRIMINATOR

        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class ScheduleChannel(TypeKeyBase):
    """Type key enum for schedule channel object."""

    DRIVE = b"d"
    CONTROL = b"c"
    MEASURE = b"m"
    ACQURE = b"a"
    MEM_SLOT = b"e"
    REG_SLOT = b"r"

    # SnapShot channel is not defined because of its limited usecase.

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, DriveChannel):
            return cls.DRIVE
        if isinstance(obj, ControlChannel):
            return cls.CONTROL
        if isinstance(obj, MeasureChannel):
            return cls.MEASURE
        if isinstance(obj, AcquireChannel):
            return cls.ACQURE
        if isinstance(obj, MemorySlot):
            return cls.MEM_SLOT
        if isinstance(obj, RegisterSlot):
            return cls.REG_SLOT

        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        if type_key == cls.DRIVE:
            return DriveChannel
        if type_key == cls.CONTROL:
            return ControlChannel
        if type_key == cls.MEASURE:
            return MeasureChannel
        if type_key == cls.ACQURE:
            return AcquireChannel
        if type_key == cls.MEM_SLOT:
            return MemorySlot
        if type_key == cls.REG_SLOT:
            return RegisterSlot

        raise exceptions.QpyError(
            f"A class corresponding to type key '{type_key}' is not found in {cls.__name__} namespace."
        )


class Program(TypeKeyBase):
    """Typle key enum for program that QPY supports."""

    CIRCUIT = b"q"
    SCHEDULE_BLOCK = b"s"

    @classmethod
    def assign(cls, obj: QuantumCircuit | ScheduleBlock) -> bytes:
        if isinstance(obj, QuantumCircuit):
            return cls.CIRCUIT
        if isinstance(obj, ScheduleBlock):
            return cls.SCHEDULE_BLOCK

        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class Expression(TypeKeyBase):
    """Type keys for the ``EXPRESSION`` QPY item."""

    VAR = b"x"
    VALUE = b"v"
    CAST = b"c"
    UNARY = b"u"
    BINARY = b"b"

    @classmethod
    def assign(cls, obj):
        if (
            isinstance(obj, expr.Expr)
            and (key := getattr(cls, obj.__class__.__name__.upper(), None)) is not None
        ):
            return key
        raise exceptions.QpyError(f"Object '{obj}' is not supported in {cls.__name__} namespace.")

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class ExprType(TypeKeyBase):
    """Type keys for the ``EXPR_TYPE`` QPY item."""

    BOOL = b"b"
    UINT = b"u"

    @classmethod
    def assign(cls, obj):
        if (
            isinstance(obj, types.Type)
            and (key := getattr(cls, obj.__class__.__name__.upper(), None)) is not None
        ):
            return key
        raise exceptions.QpyError(f"Object '{obj}' is not supported in {cls.__name__} namespace.")

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class ExprVar(TypeKeyBase):
    """Type keys for the ``EXPR_VAR`` QPY item."""

    CLBIT = b"C"
    REGISTER = b"R"

    @classmethod
    def assign(cls, obj: Clbit | ClassicalRegister) -> "ExprVar":
        if isinstance(obj, Clbit):
            return cls.CLBIT
        if isinstance(obj, ClassicalRegister):
            return cls.REGISTER
        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class ExprValue(TypeKeyBase):
    """Type keys for the ``EXPR_VALUE`` QPY item."""

    BOOL = b"b"
    INT = b"i"

    @classmethod
    def assign(cls, obj: bool | int) -> "ExprValue":
        if isinstance(obj, bool):
            return cls.BOOL
        if isinstance(obj, int):
            return cls.INT
        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class SymExprEncoding(TypeKeyBase):
    """Type keys for the symbolic encoding field in the file header."""

    SYMPY = b"p"
    SYMENGINE = b"e"

    @classmethod
    def assign(cls, obj):
        if obj:
            return cls.SYMENGINE
        else:
            return cls.SYMPY

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError

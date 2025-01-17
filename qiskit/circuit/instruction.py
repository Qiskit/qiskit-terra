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

"""
A generic quantum instruction.

Instructions can be implementable on hardware (u, cx, etc.) or in simulation
(snapshot, noise, etc.).

Instructions can be unitary (a.k.a Gate) or non-unitary.

Instructions are identified by the following:

    name: A string to identify the type of instruction.
          Used to request a specific instruction on the backend, or in visualizing circuits.

    num_qubits, num_clbits: dimensions of the instruction.

    params: List of parameters to specialize a specific instruction instance.

Instructions do not have any context about where they are in a circuit (which qubits/clbits).
The circuit itself keeps this context.
"""

from __future__ import annotations

import copy
import warnings
from itertools import zip_longest
import math
from typing import List, Type

import numpy

from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.qobj.qasm_qobj import QasmQobjInstruction
from qiskit.circuit.parameter import ParameterExpression
from qiskit.circuit.operation import Operation

from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier
from qiskit.utils import deprecate_func

_CUTOFF_PRECISION = 1e-10


class Instruction(Operation):
    """Generic quantum instruction."""

    # Class attribute to treat like barrier for transpiler, unroller, drawer
    # NOTE: Using this attribute may change in the future (See issue # 5811)
    _directive = False
    _standard_gate = None

    def __init__(self, name, num_qubits, num_clbits, params, duration=None, unit="dt", label=None):
        """Create a new instruction.

        Args:
            name (str): instruction name
            num_qubits (int): instruction's qubit width
            num_clbits (int): instruction's clbit width
            params (list[int|float|complex|str|ndarray|list|ParameterExpression]):
                list of parameters
            duration (int or float): instruction's duration. it must be integer if ``unit`` is 'dt'
            unit (str): time unit of duration
            label (str or None): An optional label for identifying the instruction.

        Raises:
            CircuitError: when the register is not in the correct format.
            TypeError: when the optional label is provided, but it is not a string.
        """
        if not isinstance(num_qubits, int) or not isinstance(num_clbits, int):
            raise CircuitError("num_qubits and num_clbits must be integer.")
        if num_qubits < 0 or num_clbits < 0:
            raise CircuitError(
                f"bad instruction dimensions: {num_qubits} qubits, {num_clbits} clbits."
            )
        self._name = name
        self._num_qubits = num_qubits
        self._num_clbits = num_clbits

        self._params = []  # a list of gate params stored
        # Custom instruction label
        # NOTE: The conditional statement checking if the `_label` attribute is
        #       already set is a temporary work around that can be removed after
        #       the next stable qiskit-aer release
        if not hasattr(self, "_label"):
            if label is not None and not isinstance(label, str):
                raise TypeError("label expects a string or None")
            self._label = label
        # tuple (ClassicalRegister, int), tuple (Clbit, bool) or tuple (Clbit, int)
        # when the instruction has a conditional ("if")
        self._condition = None
        # list of instructions (and their contexts) that this instruction is composed of
        # empty definition means opaque or fundamental instruction
        self._definition = None
        if duration is not None:
            warnings.warn(
                "Setting a custom duration per instruction is deprecated as of Qiskit "
                "1.3.0. It will be removed in Qiskit 2.0.0. An instruction's duration "
                "is defined in a backend's Target object.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._duration = duration
        if unit is not None and unit != "dt":
            warnings.warn(
                "Setting a custom unit for duration per instruction is deprecated as of Qiskit "
                "1.3.0. It will be removed in Qiskit 2.0.0. An instruction's duration "
                "is defined in a backend's Target object which has a fixed unit in seconds.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._unit = unit

        self.params = params  # must be at last (other properties may be required for validation)

    @property
    def base_class(self) -> Type[Instruction]:
        """Get the base class of this instruction.  This is guaranteed to be in the inheritance tree
        of ``self``.

        The "base class" of an instruction is the lowest class in its inheritance tree that the
        object should be considered entirely compatible with for _all_ circuit applications.  This
        typically means that the subclass is defined purely to offer some sort of programmer
        convenience over the base class, and the base class is the "true" class for a behavioral
        perspective.  In particular, you should *not* override :attr:`base_class` if you are
        defining a custom version of an instruction that will be implemented differently by
        hardware, such as an alternative measurement strategy, or a version of a parametrized gate
        with a particular set of parameters for the purposes of distinguishing it in a
        :class:`.Target` from the full parametrized gate.

        This is often exactly equivalent to ``type(obj)``, except in the case of singleton instances
        of standard-library instructions.  These singleton instances are special subclasses of their
        base class, and this property will return that base.  For example::

            >>> isinstance(XGate(), XGate)
            True
            >>> type(XGate()) is XGate
            False
            >>> XGate().base_class is XGate
            True

        In general, you should not rely on the precise class of an instruction; within a given
        circuit, it is expected that :attr:`Instruction.name` should be a more suitable
        discriminator in most situations.
        """
        return type(self)

    @property
    def mutable(self) -> bool:
        """Is this instance is a mutable unique instance or not.

        If this attribute is ``False`` the gate instance is a shared singleton
        and is not mutable.
        """
        return True

    def to_mutable(self):
        """Return a mutable copy of this gate.

        This method will return a new mutable copy of this gate instance.
        If a singleton instance is being used this will be a new unique
        instance that can be mutated. If the instance is already mutable it
        will be a deepcopy of that instance.
        """
        return self.copy()

    @property
    @deprecate_func(since="1.3.0", removal_timeline="in 2.0.0", is_property=True)
    def condition(self):
        """The classical condition on the instruction."""
        return self._condition

    @condition.setter
    def condition(self, condition):
        self._condition = condition

    def __eq__(self, other):
        """Two instructions are the same if they have the same name,
        same dimensions, and same params.

        Args:
            other (instruction): other instruction

        Returns:
            bool: are self and other equal.
        """
        if (  # pylint: disable=too-many-boolean-expressions
            not isinstance(other, Instruction)
            or self.base_class is not other.base_class
            or self.name != other.name
            or self.num_qubits != other.num_qubits
            or self.num_clbits != other.num_clbits
            or self.definition != other.definition
        ):
            return False

        for self_param, other_param in zip_longest(self.params, other.params):
            if isinstance(self_param, numpy.ndarray):
                if numpy.array_equal(self_param, other_param):
                    continue
            else:
                if self_param == other_param:
                    continue

            try:
                self_asarray = numpy.asarray(self_param)
                other_asarray = numpy.asarray(other_param)
                if numpy.shape(self_asarray) == numpy.shape(other_asarray) and numpy.allclose(
                    self_param, other_param, atol=_CUTOFF_PRECISION, rtol=0
                ):
                    continue
            except (ValueError, TypeError):
                pass

            try:
                if numpy.isclose(
                    float(self_param), float(other_param), atol=_CUTOFF_PRECISION, rtol=0
                ):
                    continue
            except TypeError:
                pass

            return False

        return True

    def __repr__(self) -> str:
        """Generates a representation of the Instruction object instance
        Returns:
            str: A representation of the Instruction instance with the name,
                 number of qubits, classical bits and params( if any )
        """
        return (
            f"Instruction(name='{self.name}', num_qubits={self.num_qubits}, "
            f"num_clbits={self.num_clbits}, params={self.params})"
        )

    def soft_compare(self, other: "Instruction") -> bool:
        """
        Soft comparison between gates. Their names, number of qubits, and classical
        bit numbers must match. The number of parameters must match. Each parameter
        is compared. If one is a ParameterExpression then it is not taken into
        account.

        Args:
            other (instruction): other instruction.

        Returns:
            bool: are self and other equal up to parameter expressions.
        """
        if (
            self.name != other.name
            or self.num_qubits != other.num_qubits
            or self.num_clbits != other.num_clbits
            or len(self.params) != len(other.params)
        ):
            return False

        for self_param, other_param in zip_longest(self.params, other.params):
            if isinstance(self_param, ParameterExpression) or isinstance(
                other_param, ParameterExpression
            ):
                continue
            if isinstance(self_param, numpy.ndarray) and isinstance(other_param, numpy.ndarray):
                if numpy.shape(self_param) == numpy.shape(other_param) and numpy.allclose(
                    self_param, other_param, atol=_CUTOFF_PRECISION
                ):
                    continue
            else:
                try:
                    if numpy.isclose(self_param, other_param, atol=_CUTOFF_PRECISION):
                        continue
                except TypeError:
                    pass

            return False

        return True

    def _define(self):
        """Populate the cached :attr:`_definition` field of this :class:`Instruction`.

        Subclasses should implement this method to provide lazy construction of their public
        :attr:`definition` attribute.  A subclass can use its :attr:`params` at the time of the
        call.  The method should populate :attr:`_definition` with a :class:`.QuantumCircuit` and
        not return a value."""
        pass

    @property
    def params(self):
        """The parameters of this :class:`Instruction`.  Ideally these will be gate angles."""
        return self._params

    @params.setter
    def params(self, parameters):
        self._params = []
        for single_param in parameters:
            if isinstance(single_param, ParameterExpression):
                self._params.append(single_param)
            else:
                self._params.append(self.validate_parameter(single_param))

    def validate_parameter(self, parameter):
        """Instruction parameters has no validation or normalization."""
        return parameter

    def is_parameterized(self):
        """Return whether the :class:`Instruction` contains :ref:`compile-time parameters
        <circuit-compile-time-parameters>`."""
        return any(
            isinstance(param, ParameterExpression) and param.parameters for param in self._params
        )

    @property
    def definition(self):
        """Return definition in terms of other basic gates."""
        if self._definition is None:
            self._define()
        return self._definition

    @definition.setter
    def definition(self, array):
        """Set gate representation"""
        self._definition = array

    @property
    def decompositions(self):
        """Get the decompositions of the instruction from the SessionEquivalenceLibrary."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

        return sel.get_entry(self)

    @decompositions.setter
    def decompositions(self, decompositions):
        """Set the decompositions of the instruction from the SessionEquivalenceLibrary."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

        sel.set_entry(self, decompositions)

    def add_decomposition(self, decomposition):
        """Add a decomposition of the instruction to the SessionEquivalenceLibrary."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

        sel.add_equivalence(self, decomposition)

    @property
    @deprecate_func(since="1.3.0", removal_timeline="in Qiskit 2.0.0", is_property=True)
    def duration(self):
        """Get the duration."""
        return self._duration

    @duration.setter
    def duration(self, value):
        """Set the duration."""
        self._duration = value

    @property
    @deprecate_func(since="1.3.0", removal_timeline="in Qiskit 2.0.0", is_property=True)
    def unit(self):
        """Get the time unit of duration."""
        return self._unit

    @unit.setter
    def unit(self, value):
        """Set the time unit of duration."""
        self._unit = value

    @deprecate_func(
        since="1.2",
        removal_timeline="in the 2.0 release",
        additional_msg="The `Qobj` class and related functionality are part of the deprecated "
        "`BackendV1` workflow,  and no longer necessary for `BackendV2`. If a user "
        "workflow requires `Qobj` it likely relies on deprecated functionality and "
        "should be updated to use `BackendV2`.",
    )
    def assemble(self):
        """Assemble a QasmQobjInstruction"""
        return self._assemble()

    def _assemble(self):
        with warnings.catch_warnings():
            # The class QasmQobjInstruction is deprecated
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
            instruction = QasmQobjInstruction(name=self.name)
        # Evaluate parameters
        if self.params:
            params = [x.evalf(x) if hasattr(x, "evalf") else x for x in self.params]
            instruction.params = params
        # Add placeholder for qarg and carg params
        if self.num_qubits:
            instruction.qubits = list(range(self.num_qubits))
        if self.num_clbits:
            instruction.memory = list(range(self.num_clbits))
        # Add label if defined
        if self.label:
            instruction.label = self.label
        # Add condition parameters for assembler. This is needed to convert
        # to a qobj conditional instruction at assemble time and after
        # conversion will be deleted by the assembler.
        if self._condition:
            instruction._condition = self._condition
        return instruction

    @property
    def label(self) -> str:
        """Return instruction label"""
        return self._label

    @label.setter
    def label(self, name: str):
        """Set instruction label to name

        Args:
            name (str or None): label to assign instruction

        Raises:
            TypeError: name is not string or None.
        """
        if isinstance(name, (str, type(None))):
            self._label = name
        else:
            raise TypeError("label expects a string or None")

    def reverse_ops(self):
        """For a composite instruction, reverse the order of sub-instructions.

        This is done by recursively reversing all sub-instructions.
        It does not invert any gate.

        Returns:
            qiskit.circuit.Instruction: a new instruction with
                sub-instructions reversed.
        """
        # A single `Instruction` cannot really determine whether it is a "composite" instruction or
        # not; it depends on greater context whether it needs to be decomposed.  The `_definition`
        # not existing is flaky; all that means is that nobody has _yet_ asked for its definition;
        # for efficiency, most gates define this on-the-fly.  The checks here are a very very
        # approximate check for an "atomic" instruction, that are mostly just this way for
        # historical consistency.
        if not self._definition or not self.mutable:
            return self.copy()

        reverse_inst = self.copy(name=self.name + "_reverse")
        reversed_definition = self._definition.copy_empty_like()
        for inst in reversed(self._definition):
            reversed_definition.append(inst.operation.reverse_ops(), inst.qubits, inst.clbits)
        reverse_inst.definition = reversed_definition
        return reverse_inst

    def inverse(self, annotated: bool = False):
        """Invert this instruction.

        If `annotated` is `False`, the inverse instruction is implemented as
        a fresh instruction with the recursively inverted definition.

        If `annotated` is `True`, the inverse instruction is implemented as
        :class:`.AnnotatedOperation`, and corresponds to the given instruction
        annotated with the "inverse modifier".

        Special instructions inheriting from Instruction can
        implement their own inverse (e.g. T and Tdg, Barrier, etc.)
        In particular, they can choose how to handle the argument ``annotated``
        which may include ignoring it and always returning a concrete gate class
        if the inverse is defined as a standard gate.

        Args:
            annotated: if set to `True` the output inverse gate will be returned
                as :class:`.AnnotatedOperation`.

        Returns:
            The inverse operation.

        Raises:
            CircuitError: if the instruction is not composite
                and an inverse has not been implemented for it.
        """
        if annotated:
            return AnnotatedOperation(self, InverseModifier())

        if self.definition is None:
            raise CircuitError(f"inverse() not implemented for {self.name}.")

        from qiskit.circuit import Gate  # pylint: disable=cyclic-import

        if self.name.endswith("_dg"):
            name = self.name[:-3]
        else:
            name = self.name + "_dg"
        if self.num_clbits:
            inverse_gate = Instruction(
                name=name,
                num_qubits=self.num_qubits,
                num_clbits=self.num_clbits,
                params=self.params.copy(),
            )

        else:
            inverse_gate = Gate(name=name, num_qubits=self.num_qubits, params=self.params.copy())

        inverse_definition = self._definition.copy_empty_like()
        inverse_definition.global_phase = -inverse_definition.global_phase
        for inst in reversed(self._definition):
            inverse_definition._append(inst.operation.inverse(), inst.qubits, inst.clbits)
        inverse_gate.definition = inverse_definition
        return inverse_gate

    @deprecate_func(since="1.3.0", removal_timeline="in 2.0.0")
    def c_if(self, classical, val):
        """Set a classical equality condition on this instruction between the register or cbit
        ``classical`` and value ``val``.

        .. note::

            This is a setter method, not an additive one.  Calling this multiple times will silently
            override any previously set condition; it does not stack.
        """
        if not isinstance(classical, (ClassicalRegister, Clbit)):
            raise CircuitError("c_if must be used with a classical register or classical bit")
        if val < 0:
            raise CircuitError("condition value should be non-negative")
        if isinstance(classical, Clbit):
            # Casting the conditional value as Boolean when
            # the classical condition is on a classical bit.
            val = bool(val)
        self._condition = (classical, val)
        return self

    def copy(self, name=None):
        """
        Copy of the instruction.

        Args:
            name (str): name to be given to the copied circuit, if ``None`` then the name stays the same.

        Returns:
            qiskit.circuit.Instruction: a copy of the current instruction, with the name updated if it
            was provided
        """
        cpy = self.__deepcopy__()

        if name:
            cpy.name = name
        return cpy

    def __deepcopy__(self, memo=None):
        cpy = copy.copy(self)
        cpy._params = copy.copy(self._params)
        if self._definition:
            cpy._definition = copy.deepcopy(self._definition, memo)
        return cpy

    def broadcast_arguments(self, qargs, cargs):
        """
        Validation of the arguments.

        Args:
            qargs (List): List of quantum bit arguments.
            cargs (List): List of classical bit arguments.

        Yields:
            Tuple(List, List): A tuple with single arguments.

        Raises:
            CircuitError: If the input is not valid. For example, the number of
                arguments does not match the gate expectation.
        """
        if len(qargs) != self.num_qubits:
            raise CircuitError(
                f"The amount of qubit arguments {len(qargs)} does not match"
                f" the instruction expectation ({self.num_qubits})."
            )
        if len(cargs) != self.num_clbits:
            raise CircuitError(
                f"The amount of clbit arguments {len(cargs)} does not match"
                f" the instruction expectation ({self.num_clbits})."
            )

        #  [[q[0], q[1]], [c[0], c[1]]] -> [q[0], c[0]], [q[1], c[1]]
        flat_qargs = [qarg for sublist in qargs for qarg in sublist]
        flat_cargs = [carg for sublist in cargs for carg in sublist]
        yield flat_qargs, flat_cargs

    def _return_repeat(self, exponent):
        return Instruction(
            name=f"{self.name}*{exponent}",
            num_qubits=self.num_qubits,
            num_clbits=self.num_clbits,
            params=self.params,
        )

    def repeat(self, n):
        """Creates an instruction with ``self`` repeated :math`n` times.

        If this operation has a conditional, the output instruction will have the same conditional
        and the inner repeated operations will be unconditional; instructions within a compound
        definition cannot be conditioned on registers within Qiskit's data model.  This means that
        it is not valid to apply a repeated instruction to a clbit that it both writes to and reads
        from in its condition.

        Args:
            n (int): Number of times to repeat the instruction

        Returns:
            qiskit.circuit.Instruction: Containing the definition.

        Raises:
            CircuitError: If n < 1.
        """
        if int(n) != n or n < 1:
            raise CircuitError("Repeat can only be called with strictly positive integer.")

        n = int(n)

        instruction = self._return_repeat(n)
        if instruction.definition is None:
            # pylint: disable=cyclic-import
            from qiskit.circuit import QuantumCircuit, CircuitInstruction

            qc = QuantumCircuit(self.num_qubits, self.num_clbits)
            qargs = tuple(qc.qubits)
            cargs = tuple(qc.clbits)
            base = self.copy()
            if self._condition:
                # Condition is handled on the outer instruction.
                base = base.to_mutable()
                base.condition = None
            for _ in [None] * n:
                qc._append(CircuitInstruction(base, qargs, cargs))

            instruction.definition = qc
        if self._condition:
            instruction = instruction.c_if(*self._condition)
        return instruction

    @property
    @deprecate_func(since="1.3.0", removal_timeline="in 2.0.0", is_property=True)
    def condition_bits(self) -> List[Clbit]:
        """Get Clbits in condition."""
        from qiskit.circuit.controlflow import condition_resources  # pylint: disable=cyclic-import

        if self._condition is None:
            return []
        return list(condition_resources(self._condition).clbits)

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        self._name = name

    @property
    def num_qubits(self):
        """Return the number of qubits."""
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits):
        """Set num_qubits."""
        self._num_qubits = num_qubits

    @property
    def num_clbits(self):
        """Return the number of clbits."""
        return self._num_clbits

    @num_clbits.setter
    def num_clbits(self, num_clbits):
        """Set num_clbits."""
        self._num_clbits = num_clbits

    def _compare_parameters(self, other):
        for x, y in zip(self.params, other.params):
            try:
                if not math.isclose(x, y, rel_tol=0, abs_tol=1e-10):
                    return False
            except TypeError:
                if x != y:
                    return False
        return True

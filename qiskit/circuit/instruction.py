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

import copy
from itertools import zip_longest
from typing import List

import numpy as np

from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.qobj.qasm_qobj import QasmQobjInstruction
from qiskit.circuit.parameter import ParameterExpression
from qiskit.circuit.operation import Operation
from .tools import pi_check

_CUTOFF_PRECISION = 1e-10


class Instruction(Operation):
    """Generic quantum instruction."""

    # Class attribute to treat like barrier for transpiler, unroller, drawer
    # NOTE: Using this attribute may change in the future (See issue # 5811)
    _directive = False

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
                "bad instruction dimensions: %d qubits, %d clbits." % num_qubits, num_clbits
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
        self.condition = None
        # list of instructions (and their contexts) that this instruction is composed of
        # empty definition means opaque or fundamental instruction
        self._definition = None

        self._duration = duration
        self._unit = unit

        self.params = params  # must be at last (other properties may be required for validation)

    def __eq__(self, other):
        """Two instructions are the same if they have the same name,
        same dimensions, and same params.

        Args:
            other (instruction): other instruction

        Returns:
            bool: are self and other equal.
        """
        if (
            type(self) is not type(other)
            or self.name != other.name
            or self.num_qubits != other.num_qubits
            or self.num_clbits != other.num_clbits
            or self.definition != other.definition
        ):
            return False

        for self_param, other_param in zip_longest(self.params, other.params):
            try:
                if self_param == other_param:
                    continue
            except ValueError:
                pass

            try:
                self_asarray = np.asarray(self_param)
                other_asarray = np.asarray(other_param)
                if np.shape(self_asarray) == np.shape(other_asarray) and np.allclose(
                    self_param, other_param, atol=_CUTOFF_PRECISION, rtol=0
                ):
                    continue
            except (ValueError, TypeError):
                pass

            try:
                if np.isclose(
                    float(self_param), float(other_param), atol=_CUTOFF_PRECISION, rtol=0
                ):
                    continue
            except TypeError:
                pass

            return False

        return True

    def __repr__(self) -> str:
        """Generates a representation of the Intruction object instance
        Returns:
            str: A representation of the Instruction instance with the name,
                 number of qubits, classical bits and params( if any )
        """
        return "Instruction(name='{}', num_qubits={}, num_clbits={}, params={})".format(
            self.name, self.num_qubits, self.num_clbits, self.params
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
            or other.num_qubits != other.num_qubits
            or other.num_clbits != other.num_clbits
            or len(self.params) != len(other.params)
        ):
            return False

        for self_param, other_param in zip_longest(self.params, other.params):
            if isinstance(self_param, ParameterExpression) or isinstance(
                other_param, ParameterExpression
            ):
                continue
            if isinstance(self_param, np.ndarray) and isinstance(other_param, np.ndarray):
                if np.shape(self_param) == np.shape(other_param) and np.allclose(
                    self_param, other_param, atol=_CUTOFF_PRECISION
                ):
                    continue
            else:
                try:
                    if np.isclose(self_param, other_param, atol=_CUTOFF_PRECISION):
                        continue
                except TypeError:
                    pass

            return False

        return True

    def _define(self):
        """Populates self.definition with a decomposition of this gate."""
        pass

    @property
    def params(self):
        """return instruction params."""
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
        """Return True .IFF. instruction is parameterized else False"""
        return any(
            isinstance(param, ParameterExpression) and param.parameters for param in self.params
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
    def duration(self):
        """Get the duration."""
        return self._duration

    @duration.setter
    def duration(self, duration):
        """Set the duration."""
        self._duration = duration

    @property
    def unit(self):
        """Get the time unit of duration."""
        return self._unit

    @unit.setter
    def unit(self, unit):
        """Set the time unit of duration."""
        self._unit = unit

    def assemble(self):
        """Assemble a QasmQobjInstruction"""
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
        if self.condition:
            instruction._condition = self.condition
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
        if not self._definition:
            return self.copy()

        reverse_inst = self.copy(name=self.name + "_reverse")
        reversed_definition = self._definition.copy_empty_like()
        for inst in reversed(self._definition):
            reversed_definition.append(inst.operation.reverse_ops(), inst.qubits, inst.clbits)
        reverse_inst.definition = reversed_definition
        return reverse_inst

    def inverse(self):
        """Invert this instruction.

        If the instruction is composite (i.e. has a definition),
        then its definition will be recursively inverted.

        Special instructions inheriting from Instruction can
        implement their own inverse (e.g. T and Tdg, Barrier, etc.)

        Returns:
            qiskit.circuit.Instruction: a fresh instruction for the inverse

        Raises:
            CircuitError: if the instruction is not composite
                and an inverse has not been implemented for it.
        """
        if self.definition is None:
            raise CircuitError("inverse() not implemented for %s." % self.name)

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
        self.condition = (classical, val)
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

    def __deepcopy__(self, _memo=None):
        cpy = copy.copy(self)
        cpy._params = copy.copy(self._params)
        if self._definition:
            cpy._definition = copy.deepcopy(self._definition, _memo)
        return cpy

    def _qasmif(self, string):
        """Print an if statement if needed."""
        from qiskit.qasm2 import QASM2ExportError  # pylint: disable=cyclic-import

        if self.condition is None:
            return string
        if not isinstance(self.condition[0], ClassicalRegister):
            raise QASM2ExportError(
                "OpenQASM 2 can only condition on registers, but got '{self.condition[0]}'"
            )
        return "if(%s==%d) " % (self.condition[0].name, self.condition[1]) + string

    def qasm(self):
        """Return a default OpenQASM string for the instruction.

        Derived instructions may override this to print in a
        different format (e.g. measure q[0] -> c[0];).
        """
        name_param = self.name
        if self.params:
            name_param = "{}({})".format(
                name_param,
                ",".join([pi_check(i, output="qasm", eps=1e-12) for i in self.params]),
            )

        return self._qasmif(name_param)

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
        """Creates an instruction with `gate` repeated `n` amount of times.

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
        qargs = [] if self.num_qubits == 0 else QuantumRegister(self.num_qubits, "q")
        cargs = [] if self.num_clbits == 0 else ClassicalRegister(self.num_clbits, "c")

        if instruction.definition is None:
            # pylint: disable=cyclic-import
            from qiskit.circuit import QuantumCircuit, CircuitInstruction

            qc = QuantumCircuit()
            if qargs:
                qc.add_register(qargs)
            if cargs:
                qc.add_register(cargs)
            circuit_instruction = CircuitInstruction(self, qargs, cargs)
            for _ in [None] * n:
                qc._append(circuit_instruction)
        instruction.definition = qc
        return instruction

    @property
    def condition_bits(self) -> List[Clbit]:
        """Get Clbits in condition."""
        if self.condition is None:
            return []
        if isinstance(self.condition[0], Clbit):
            return [self.condition[0]]
        else:  # ClassicalRegister
            return list(self.condition[0])

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

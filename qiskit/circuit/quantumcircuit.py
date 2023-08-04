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

# pylint: disable=bad-docstring-quotes,invalid-name

"""Quantum circuit object."""

from __future__ import annotations
import collections.abc
import copy
import itertools
import multiprocessing as mp
import string
import re
import warnings
import typing
from collections import OrderedDict, defaultdict, namedtuple
from typing import (
    Union,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Sequence,
    Callable,
    Mapping,
    Iterable,
    Any,
    DefaultDict,
    Literal,
    overload,
)
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.utils.multiprocessing import is_main_process
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils import optionals as _optionals
from . import _classical_resource_map
from ._utils import sort_parameters
from .classical import expr
from .parameterexpression import ParameterExpression, ParameterValueType
from .quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from .classicalregister import ClassicalRegister, Clbit
from .parametertable import ParameterReferences, ParameterTable, ParameterView
from .parametervector import ParameterVector
from .instructionset import InstructionSet
from .operation import Operation
from .register import Register
from .bit import Bit
from .quantumcircuitdata import QuantumCircuitData, CircuitInstruction
from .delay import Delay
from .measure import Measure
from .reset import Reset
from .tools import pi_check

if typing.TYPE_CHECKING:
    import qiskit  # pylint: disable=cyclic-import
    from qiskit.transpiler.layout import TranspileLayout  # pylint: disable=cyclic-import

BitLocations = namedtuple("BitLocations", ("index", "registers"))


# The following types are not marked private to avoid leaking this "private/public" abstraction out
# into the documentation.  They are not imported by circuit.__init__, nor are they meant to be.

# Arbitrary type variables for marking up generics.
S = TypeVar("S")
T = TypeVar("T")

# Types that can be coerced to a valid Qubit specifier in a circuit.
QubitSpecifier = Union[
    Qubit,
    QuantumRegister,
    int,
    slice,
    Sequence[Union[Qubit, int]],
]

# Types that can be coerced to a valid Clbit specifier in a circuit.
ClbitSpecifier = Union[
    Clbit,
    ClassicalRegister,
    int,
    slice,
    Sequence[Union[Clbit, int]],
]

# Generic type which is either :obj:`~Qubit` or :obj:`~Clbit`, used to specify types of functions
# which operate on either type of bit, but not both at the same time.
BitType = TypeVar("BitType", Qubit, Clbit)

# Regex pattern to match valid OpenQASM identifiers
VALID_QASM2_IDENTIFIER = re.compile("[a-z][a-zA-Z_0-9]*")
QASM2_RESERVED = {
    "OPENQASM",
    "qreg",
    "creg",
    "include",
    "gate",
    "opaque",
    "U",
    "CX",
    "measure",
    "reset",
    "if",
    "barrier",
}


class QuantumCircuit:
    """Create a new circuit.

    A circuit is a list of instructions bound to some registers.

    Args:
        regs (list(:class:`~.Register`) or list(``int``) or list(list(:class:`~.Bit`))): The
            registers to be included in the circuit.

            * If a list of :class:`~.Register` objects, represents the :class:`.QuantumRegister`
              and/or :class:`.ClassicalRegister` objects to include in the circuit.

              For example:

                * ``QuantumCircuit(QuantumRegister(4))``
                * ``QuantumCircuit(QuantumRegister(4), ClassicalRegister(3))``
                * ``QuantumCircuit(QuantumRegister(4, 'qr0'), QuantumRegister(2, 'qr1'))``

            * If a list of ``int``, the amount of qubits and/or classical bits to include in
              the circuit. It can either be a single int for just the number of quantum bits,
              or 2 ints for the number of quantum bits and classical bits, respectively.

              For example:

                * ``QuantumCircuit(4) # A QuantumCircuit with 4 qubits``
                * ``QuantumCircuit(4, 3) # A QuantumCircuit with 4 qubits and 3 classical bits``

            * If a list of python lists containing :class:`.Bit` objects, a collection of
              :class:`.Bit` s to be added to the circuit.


        name (str): the name of the quantum circuit. If not set, an
            automatically generated string will be assigned.
        global_phase (float or ParameterExpression): The global phase of the circuit in radians.
        metadata (dict): Arbitrary key value metadata to associate with the
            circuit. This gets stored as free-form data in a dict in the
            :attr:`~qiskit.circuit.QuantumCircuit.metadata` attribute. It will
            not be directly used in the circuit.

    Raises:
        CircuitError: if the circuit name, if given, is not valid.

    Examples:

        Construct a simple Bell state circuit.

        .. plot::
           :include-source:

           from qiskit import QuantumCircuit

           qc = QuantumCircuit(2, 2)
           qc.h(0)
           qc.cx(0, 1)
           qc.measure([0, 1], [0, 1])
           qc.draw('mpl')

        Construct a 5-qubit GHZ circuit.

        .. code-block::

           from qiskit import QuantumCircuit

           qc = QuantumCircuit(5)
           qc.h(0)
           qc.cx(0, range(1, 5))
           qc.measure_all()

        Construct a 4-qubit Bernstein-Vazirani circuit using registers.

        .. plot::
           :include-source:

           from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

           qr = QuantumRegister(3, 'q')
           anc = QuantumRegister(1, 'ancilla')
           cr = ClassicalRegister(3, 'c')
           qc = QuantumCircuit(qr, anc, cr)

           qc.x(anc[0])
           qc.h(anc[0])
           qc.h(qr[0:3])
           qc.cx(qr[0:3], anc[0])
           qc.h(qr[0:3])
           qc.barrier(qr)
           qc.measure(qr, cr)

           qc.draw('mpl')
    """

    instances = 0
    prefix = "circuit"

    # Class variable OPENQASM header
    header = "OPENQASM 2.0;"
    extension_lib = 'include "qelib1.inc";'

    def __init__(
        self,
        *regs: Register | int | Sequence[Bit],
        name: str | None = None,
        global_phase: ParameterValueType = 0,
        metadata: dict | None = None,
    ):
        if any(not isinstance(reg, (list, QuantumRegister, ClassicalRegister)) for reg in regs):
            # check if inputs are integers, but also allow e.g. 2.0

            try:
                valid_reg_size = all(reg == int(reg) for reg in regs)
            except (ValueError, TypeError):
                valid_reg_size = False

            if not valid_reg_size:
                raise CircuitError(
                    "Circuit args must be Registers or integers. (%s '%s' was "
                    "provided)" % ([type(reg).__name__ for reg in regs], regs)
                )

            regs = tuple(int(reg) for reg in regs)  # cast to int
        self._base_name = None
        if name is None:
            self._base_name = self.cls_prefix()
            self._name_update()
        elif not isinstance(name, str):
            raise CircuitError(
                "The circuit name should be a string (or None to auto-generate a name)."
            )
        else:
            self._base_name = name
            self.name = name
        self._increment_instances()

        # Data contains a list of instructions and their contexts,
        # in the order they were applied.
        self._data: list[CircuitInstruction] = []
        self._op_start_times = None

        # A stack to hold the instruction sets that are being built up during for-, if- and
        # while-block construction.  These are stored as a stripped down sequence of instructions,
        # and sets of qubits and clbits, rather than a full QuantumCircuit instance because the
        # builder interfaces need to wait until they are completed before they can fill in things
        # like `break` and `continue`.  This is because these instructions need to "operate" on the
        # full width of bits, but the builder interface won't know what bits are used until the end.
        self._control_flow_scopes: list[
            "qiskit.circuit.controlflow.builder.ControlFlowBuilderBlock"
        ] = []

        self.qregs: list[QuantumRegister] = []
        self.cregs: list[ClassicalRegister] = []
        self._qubits: list[Qubit] = []
        self._clbits: list[Clbit] = []

        # Dict mapping Qubit or Clbit instances to tuple comprised of 0) the
        # corresponding index in circuit.{qubits,clbits} and 1) a list of
        # Register-int pairs for each Register containing the Bit and its index
        # within that register.
        self._qubit_indices: dict[Qubit, BitLocations] = {}
        self._clbit_indices: dict[Clbit, BitLocations] = {}

        self._ancillas: list[AncillaQubit] = []
        self._calibrations: DefaultDict[str, dict[tuple, Any]] = defaultdict(dict)
        self.add_register(*regs)

        # Parameter table tracks instructions with variable parameters.
        self._parameter_table = ParameterTable()

        # Cache to avoid re-sorting parameters
        self._parameters = None

        self._layout = None
        self._global_phase: ParameterValueType = 0
        self.global_phase = global_phase

        self.duration = None
        self.unit = "dt"
        self.metadata = {} if metadata is None else metadata

    @staticmethod
    def from_instructions(
        instructions: Iterable[
            CircuitInstruction
            | tuple[qiskit.circuit.Instruction]
            | tuple[qiskit.circuit.Instruction, Iterable[Qubit]]
            | tuple[qiskit.circuit.Instruction, Iterable[Qubit], Iterable[Clbit]]
        ],
        *,
        qubits: Iterable[Qubit] = (),
        clbits: Iterable[Clbit] = (),
        name: str | None = None,
        global_phase: ParameterValueType = 0,
        metadata: dict | None = None,
    ) -> "QuantumCircuit":
        """Construct a circuit from an iterable of CircuitInstructions.

        Args:
            instructions: The instructions to add to the circuit.
            qubits: Any qubits to add to the circuit. This argument can be used,
                for example, to enforce a particular ordering of qubits.
            clbits: Any classical bits to add to the circuit. This argument can be used,
                for example, to enforce a particular ordering of classical bits.
            name: The name of the circuit.
            global_phase: The global phase of the circuit in radians.
            metadata: Arbitrary key value metadata to associate with the circuit.

        Returns:
            The quantum circuit.
        """
        circuit = QuantumCircuit(name=name, global_phase=global_phase, metadata=metadata)
        added_qubits = set()
        added_clbits = set()
        if qubits:
            qubits = list(qubits)
            circuit.add_bits(qubits)
            added_qubits.update(qubits)
        if clbits:
            clbits = list(clbits)
            circuit.add_bits(clbits)
            added_clbits.update(clbits)
        for instruction in instructions:
            if not isinstance(instruction, CircuitInstruction):
                instruction = CircuitInstruction(*instruction)
            qubits = [qubit for qubit in instruction.qubits if qubit not in added_qubits]
            clbits = [clbit for clbit in instruction.clbits if clbit not in added_clbits]
            circuit.add_bits(qubits)
            circuit.add_bits(clbits)
            added_qubits.update(qubits)
            added_clbits.update(clbits)
            circuit._append(instruction)
        return circuit

    @property
    def layout(self) -> Optional[TranspileLayout]:
        r"""Return any associated layout information anout the circuit

        This attribute contains an optional :class:`~.TranspileLayout`
        object. This is typically set on the output from :func:`~.transpile`
        or :meth:`.PassManager.run` to retain information about the
        permutations caused on the input circuit by transpilation.

        There are two types of permutations caused by the :func:`~.transpile`
        function, an initial layout which permutes the qubits based on the
        selected physical qubits on the :class:`~.Target`, and a final layout
        which is an output permutation caused by :class:`~.SwapGate`\s
        inserted during routing.
        """
        return self._layout

    @property
    def data(self) -> QuantumCircuitData:
        """Return the circuit data (instructions and context).

        Returns:
            QuantumCircuitData: a list-like object containing the :class:`.CircuitInstruction`\\ s
            for each instruction.
        """
        return QuantumCircuitData(self)

    @data.setter
    def data(self, data_input: Iterable):
        """Sets the circuit data from a list of instructions and context.

        Args:
            data_input (Iterable): A sequence of instructions with their execution contexts.  The
                elements must either be instances of :class:`.CircuitInstruction` (preferred), or a
                3-tuple of ``(instruction, qargs, cargs)`` (legacy).  In the legacy format,
                ``instruction`` must be an :class:`~.circuit.Instruction`, while ``qargs`` and
                ``cargs`` must be iterables of :class:`.Qubit` or :class:`.Clbit` specifiers
                (similar to the allowed forms in calls to :meth:`append`).
        """
        # If data_input is QuantumCircuitData(self), clearing self._data
        # below will also empty data_input, so make a shallow copy first.
        data_input = list(data_input)
        self._data = []
        self._parameter_table = ParameterTable()
        if not data_input:
            return
        if isinstance(data_input[0], CircuitInstruction):
            for instruction in data_input:
                self.append(instruction)
        else:
            for instruction, qargs, cargs in data_input:
                self.append(instruction, qargs, cargs)

    @property
    def op_start_times(self) -> list[int]:
        """Return a list of operation start times.

        This attribute is enabled once one of scheduling analysis passes
        runs on the quantum circuit.

        Returns:
            List of integers representing instruction start times.
            The index corresponds to the index of instruction in :attr:`QuantumCircuit.data`.

        Raises:
            AttributeError: When circuit is not scheduled.
        """
        if self._op_start_times is None:
            raise AttributeError(
                "This circuit is not scheduled. "
                "To schedule it run the circuit through one of the transpiler scheduling passes."
            )
        return self._op_start_times

    @property
    def calibrations(self) -> dict:
        """Return calibration dictionary.

        The custom pulse definition of a given gate is of the form
        ``{'gate_name': {(qubits, params): schedule}}``
        """
        return dict(self._calibrations)

    @calibrations.setter
    def calibrations(self, calibrations: dict):
        """Set the circuit calibration data from a dictionary of calibration definition.

        Args:
            calibrations (dict): A dictionary of input in the format
               ``{'gate_name': {(qubits, gate_params): schedule}}``
        """
        self._calibrations = defaultdict(dict, calibrations)

    def has_calibration_for(self, instruction: CircuitInstruction | tuple):
        """Return True if the circuit has a calibration defined for the instruction context. In this
        case, the operation does not need to be translated to the device basis.
        """
        if isinstance(instruction, CircuitInstruction):
            operation = instruction.operation
            qubits = instruction.qubits
        else:
            operation, qubits, _ = instruction
        if not self.calibrations or operation.name not in self.calibrations:
            return False
        qubits = tuple(self.qubits.index(qubit) for qubit in qubits)
        params = []
        for p in operation.params:
            if isinstance(p, ParameterExpression) and not p.parameters:
                params.append(float(p))
            else:
                params.append(p)
        params = tuple(params)
        return (qubits, params) in self.calibrations[operation.name]

    @property
    def metadata(self) -> dict:
        """The user provided metadata associated with the circuit.

        The metadata for the circuit is a user provided ``dict`` of metadata
        for the circuit. It will not be used to influence the execution or
        operation of the circuit, but it is expected to be passed between
        all transforms of the circuit (ie transpilation) and that providers will
        associate any circuit metadata with the results it returns from
        execution of that circuit.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: dict | None):
        """Update the circuit metadata"""
        if metadata is None:
            metadata = {}
            warnings.warn(
                "Setting metadata to None was deprecated in Terra 0.24.0 and this ability will be "
                "removed in a future release. Instead, set metadata to an empty dictionary.",
                DeprecationWarning,
                stacklevel=2,
            )
        elif not isinstance(metadata, dict):
            raise TypeError("Only a dictionary is accepted for circuit metadata")
        self._metadata = metadata

    def __str__(self) -> str:
        return str(self.draw(output="text"))

    def __eq__(self, other) -> bool:
        if not isinstance(other, QuantumCircuit):
            return False

        # TODO: remove the DAG from this function
        from qiskit.converters import circuit_to_dag

        return circuit_to_dag(self, copy_operations=False) == circuit_to_dag(
            other, copy_operations=False
        )

    @classmethod
    def _increment_instances(cls):
        cls.instances += 1

    @classmethod
    def cls_instances(cls) -> int:
        """Return the current number of instances of this class,
        useful for auto naming."""
        return cls.instances

    @classmethod
    def cls_prefix(cls) -> str:
        """Return the prefix to use for auto naming."""
        return cls.prefix

    def _name_update(self) -> None:
        """update name of instance using instance number"""
        if not is_main_process():
            pid_name = f"-{mp.current_process().pid}"
        else:
            pid_name = ""

        self.name = f"{self._base_name}-{self.cls_instances()}{pid_name}"

    def has_register(self, register: Register) -> bool:
        """
        Test if this circuit has the register r.

        Args:
            register (Register): a quantum or classical register.

        Returns:
            bool: True if the register is contained in this circuit.
        """
        has_reg = False
        if isinstance(register, QuantumRegister) and register in self.qregs:
            has_reg = True
        elif isinstance(register, ClassicalRegister) and register in self.cregs:
            has_reg = True
        return has_reg

    def reverse_ops(self) -> "QuantumCircuit":
        """Reverse the circuit by reversing the order of instructions.

        This is done by recursively reversing all instructions.
        It does not invert (adjoint) any gate.

        Returns:
            QuantumCircuit: the reversed circuit.

        Examples:

            input:

            .. parsed-literal::

                     ┌───┐
                q_0: ┤ H ├─────■──────
                     └───┘┌────┴─────┐
                q_1: ─────┤ RX(1.57) ├
                          └──────────┘

            output:

            .. parsed-literal::

                                 ┌───┐
                q_0: ─────■──────┤ H ├
                     ┌────┴─────┐└───┘
                q_1: ┤ RX(1.57) ├─────
                     └──────────┘
        """
        reverse_circ = QuantumCircuit(
            self.qubits, self.clbits, *self.qregs, *self.cregs, name=self.name + "_reverse"
        )

        for instruction in reversed(self.data):
            reverse_circ._append(instruction.replace(operation=instruction.operation.reverse_ops()))

        reverse_circ.duration = self.duration
        reverse_circ.unit = self.unit
        return reverse_circ

    def reverse_bits(self) -> "QuantumCircuit":
        """Return a circuit with the opposite order of wires.

        The circuit is "vertically" flipped. If a circuit is
        defined over multiple registers, the resulting circuit will have
        the same registers but with their order flipped.

        This method is useful for converting a circuit written in little-endian
        convention to the big-endian equivalent, and vice versa.

        Returns:
            QuantumCircuit: the circuit with reversed bit order.

        Examples:

            input:

            .. parsed-literal::

                     ┌───┐
                a_0: ┤ H ├──■─────────────────
                     └───┘┌─┴─┐
                a_1: ─────┤ X ├──■────────────
                          └───┘┌─┴─┐
                a_2: ──────────┤ X ├──■───────
                               └───┘┌─┴─┐
                b_0: ───────────────┤ X ├──■──
                                    └───┘┌─┴─┐
                b_1: ────────────────────┤ X ├
                                         └───┘

            output:

            .. parsed-literal::

                                         ┌───┐
                b_0: ────────────────────┤ X ├
                                    ┌───┐└─┬─┘
                b_1: ───────────────┤ X ├──■──
                               ┌───┐└─┬─┘
                a_0: ──────────┤ X ├──■───────
                          ┌───┐└─┬─┘
                a_1: ─────┤ X ├──■────────────
                     ┌───┐└─┬─┘
                a_2: ┤ H ├──■─────────────────
                     └───┘
        """
        circ = QuantumCircuit(
            list(reversed(self.qubits)),
            list(reversed(self.clbits)),
            name=self.name,
            global_phase=self.global_phase,
        )
        new_qubit_map = circ.qubits[::-1]
        new_clbit_map = circ.clbits[::-1]
        for reg in reversed(self.qregs):
            bits = [new_qubit_map[self.find_bit(qubit).index] for qubit in reversed(reg)]
            circ.add_register(QuantumRegister(bits=bits, name=reg.name))
        for reg in reversed(self.cregs):
            bits = [new_clbit_map[self.find_bit(clbit).index] for clbit in reversed(reg)]
            circ.add_register(ClassicalRegister(bits=bits, name=reg.name))

        for instruction in self.data:
            qubits = [new_qubit_map[self.find_bit(qubit).index] for qubit in instruction.qubits]
            clbits = [new_clbit_map[self.find_bit(clbit).index] for clbit in instruction.clbits]
            circ._append(instruction.replace(qubits=qubits, clbits=clbits))
        return circ

    def inverse(self) -> "QuantumCircuit":
        """Invert (take adjoint of) this circuit.

        This is done by recursively inverting all gates.

        Returns:
            QuantumCircuit: the inverted circuit

        Raises:
            CircuitError: if the circuit cannot be inverted.

        Examples:

            input:

            .. parsed-literal::

                     ┌───┐
                q_0: ┤ H ├─────■──────
                     └───┘┌────┴─────┐
                q_1: ─────┤ RX(1.57) ├
                          └──────────┘

            output:

            .. parsed-literal::

                                  ┌───┐
                q_0: ──────■──────┤ H ├
                     ┌─────┴─────┐└───┘
                q_1: ┤ RX(-1.57) ├─────
                     └───────────┘
        """
        inverse_circ = QuantumCircuit(
            self.qubits,
            self.clbits,
            *self.qregs,
            *self.cregs,
            name=self.name + "_dg",
            global_phase=-self.global_phase,
        )

        for instruction in reversed(self._data):
            inverse_circ._append(instruction.replace(operation=instruction.operation.inverse()))
        return inverse_circ

    def repeat(self, reps: int) -> "QuantumCircuit":
        """Repeat this circuit ``reps`` times.

        Args:
            reps (int): How often this circuit should be repeated.

        Returns:
            QuantumCircuit: A circuit containing ``reps`` repetitions of this circuit.
        """
        repeated_circ = QuantumCircuit(
            self.qubits, self.clbits, *self.qregs, *self.cregs, name=self.name + f"**{reps}"
        )

        # benefit of appending instructions: decomposing shows the subparts, i.e. the power
        # is actually `reps` times this circuit, and it is currently much faster than `compose`.
        if reps > 0:
            try:  # try to append as gate if possible to not disallow to_gate
                inst: Instruction = self.to_gate()
            except QiskitError:
                inst = self.to_instruction()
            for _ in range(reps):
                repeated_circ._append(inst, self.qubits, self.clbits)

        return repeated_circ

    def power(self, power: float, matrix_power: bool = False) -> "QuantumCircuit":
        """Raise this circuit to the power of ``power``.

        If ``power`` is a positive integer and ``matrix_power`` is ``False``, this implementation
        defaults to calling ``repeat``. Otherwise, if the circuit is unitary, the matrix is
        computed to calculate the matrix power.

        Args:
            power (float): The power to raise this circuit to.
            matrix_power (bool): If True, the circuit is converted to a matrix and then the
                matrix power is computed. If False, and ``power`` is a positive integer,
                the implementation defaults to ``repeat``.

        Raises:
            CircuitError: If the circuit needs to be converted to a gate but it is not unitary.

        Returns:
            QuantumCircuit: A circuit implementing this circuit raised to the power of ``power``.
        """
        if power >= 0 and isinstance(power, (int, np.integer)) and not matrix_power:
            return self.repeat(power)

        # attempt conversion to gate
        if self.num_parameters > 0:
            raise CircuitError(
                "Cannot raise a parameterized circuit to a non-positive power "
                "or matrix-power, please bind the free parameters: "
                "{}".format(self.parameters)
            )

        try:
            gate = self.to_gate()
        except QiskitError as ex:
            raise CircuitError(
                "The circuit contains non-unitary operations and cannot be "
                "controlled. Note that no qiskit.circuit.Instruction objects may "
                "be in the circuit for this operation."
            ) from ex

        power_circuit = QuantumCircuit(self.qubits, self.clbits, *self.qregs, *self.cregs)
        power_circuit.append(gate.power(power), list(range(gate.num_qubits)))
        return power_circuit

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> "QuantumCircuit":
        """Control this circuit on ``num_ctrl_qubits`` qubits.

        Args:
            num_ctrl_qubits (int): The number of control qubits.
            label (str): An optional label to give the controlled operation for visualization.
            ctrl_state (str or int): The control state in decimal or as a bitstring
                (e.g. '111'). If None, use ``2**num_ctrl_qubits - 1``.

        Returns:
            QuantumCircuit: The controlled version of this circuit.

        Raises:
            CircuitError: If the circuit contains a non-unitary operation and cannot be controlled.
        """
        try:
            gate = self.to_gate()
        except QiskitError as ex:
            raise CircuitError(
                "The circuit contains non-unitary operations and cannot be "
                "controlled. Note that no qiskit.circuit.Instruction objects may "
                "be in the circuit for this operation."
            ) from ex

        controlled_gate = gate.control(num_ctrl_qubits, label, ctrl_state)
        control_qreg = QuantumRegister(num_ctrl_qubits)
        controlled_circ = QuantumCircuit(
            control_qreg, self.qubits, *self.qregs, name=f"c_{self.name}"
        )
        controlled_circ.append(controlled_gate, controlled_circ.qubits)

        return controlled_circ

    def compose(
        self,
        other: Union["QuantumCircuit", Instruction],
        qubits: QubitSpecifier | Sequence[QubitSpecifier] | None = None,
        clbits: ClbitSpecifier | Sequence[ClbitSpecifier] | None = None,
        front: bool = False,
        inplace: bool = False,
        wrap: bool = False,
    ) -> Optional["QuantumCircuit"]:
        """Compose circuit with ``other`` circuit or instruction, optionally permuting wires.

        ``other`` can be narrower or of equal width to ``self``.

        Args:
            other (qiskit.circuit.Instruction or QuantumCircuit):
                (sub)circuit or instruction to compose onto self.  If not a :obj:`.QuantumCircuit`,
                this can be anything that :obj:`.append` will accept.
            qubits (list[Qubit|int]): qubits of self to compose onto.
            clbits (list[Clbit|int]): clbits of self to compose onto.
            front (bool): If True, front composition will be performed.  This is not possible within
                control-flow builder context managers.
            inplace (bool): If True, modify the object. Otherwise return composed circuit.
            wrap (bool): If True, wraps the other circuit into a gate (or instruction, depending on
                whether it contains only unitary instructions) before composing it onto self.

        Returns:
            QuantumCircuit: the composed circuit (returns None if inplace==True).

        Raises:
            CircuitError: if no correct wire mapping can be made between the two circuits, such as
                if ``other`` is wider than ``self``.
            CircuitError: if trying to emit a new circuit while ``self`` has a partially built
                control-flow context active, such as the context-manager forms of :meth:`if_test`,
                :meth:`for_loop` and :meth:`while_loop`.
            CircuitError: if trying to compose to the front of a circuit when a control-flow builder
                block is active; there is no clear meaning to this action.

        Examples:
            .. code-block:: python

                >>> lhs.compose(rhs, qubits=[3, 2], inplace=True)

            .. parsed-literal::

                            ┌───┐                   ┌─────┐                ┌───┐
                lqr_1_0: ───┤ H ├───    rqr_0: ──■──┤ Tdg ├    lqr_1_0: ───┤ H ├───────────────
                            ├───┤              ┌─┴─┐└─────┘                ├───┤
                lqr_1_1: ───┤ X ├───    rqr_1: ┤ X ├───────    lqr_1_1: ───┤ X ├───────────────
                         ┌──┴───┴──┐           └───┘                    ┌──┴───┴──┐┌───┐
                lqr_1_2: ┤ U1(0.1) ├  +                     =  lqr_1_2: ┤ U1(0.1) ├┤ X ├───────
                         └─────────┘                                    └─────────┘└─┬─┘┌─────┐
                lqr_2_0: ─────■─────                           lqr_2_0: ─────■───────■──┤ Tdg ├
                            ┌─┴─┐                                          ┌─┴─┐        └─────┘
                lqr_2_1: ───┤ X ├───                           lqr_2_1: ───┤ X ├───────────────
                            └───┘                                          └───┘
                lcr_0: 0 ═══════════                           lcr_0: 0 ═══════════════════════

                lcr_1: 0 ═══════════                           lcr_1: 0 ═══════════════════════

        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.controlflow.switch_case import SwitchCaseOp

        if inplace and front and self._control_flow_scopes:
            # If we're composing onto ourselves while in a stateful control-flow builder context,
            # there's no clear meaning to composition to the "front" of the circuit.
            raise CircuitError(
                "Cannot compose to the front of a circuit while a control-flow context is active."
            )
        if not inplace and self._control_flow_scopes:
            # If we're inside a stateful control-flow builder scope, even if we successfully cloned
            # the partial builder scope (not simple), the scope wouldn't be controlled by an active
            # `with` statement, so the output circuit would be permanently broken.
            raise CircuitError(
                "Cannot emit a new composed circuit while a control-flow context is active."
            )

        dest = self if inplace else self.copy()

        # As a special case, allow composing some clbits onto no clbits - normally the destination
        # has to be strictly larger. This allows composing final measurements onto unitary circuits.
        if isinstance(other, QuantumCircuit):
            if not self.clbits and other.clbits:
                dest.add_bits(other.clbits)
                for reg in other.cregs:
                    dest.add_register(reg)

        if wrap and isinstance(other, QuantumCircuit):
            other = (
                other.to_gate()
                if all(isinstance(ins.operation, Gate) for ins in other.data)
                else other.to_instruction()
            )

        if not isinstance(other, QuantumCircuit):
            if qubits is None:
                qubits = self.qubits[: other.num_qubits]
            if clbits is None:
                clbits = self.clbits[: other.num_clbits]
            if front:
                # Need to keep a reference to the data for use after we've emptied it.
                old_data = list(dest.data)
                dest.clear()
                dest.append(other, qubits, clbits)
                for instruction in old_data:
                    dest._append(instruction)
            else:
                dest.append(other, qargs=qubits, cargs=clbits)
            if inplace:
                return None
            return dest

        if other.num_qubits > dest.num_qubits or other.num_clbits > dest.num_clbits:
            raise CircuitError(
                "Trying to compose with another QuantumCircuit which has more 'in' edges."
            )

        # number of qubits and clbits must match number in circuit or None
        edge_map: dict[Qubit | Clbit, Qubit | Clbit] = {}
        if qubits is None:
            edge_map.update(zip(other.qubits, dest.qubits))
        else:
            mapped_qubits = dest.qbit_argument_conversion(qubits)
            if len(mapped_qubits) != len(other.qubits):
                raise CircuitError(
                    f"Number of items in qubits parameter ({len(mapped_qubits)}) does not"
                    f" match number of qubits in the circuit ({len(other.qubits)})."
                )
            edge_map.update(zip(other.qubits, mapped_qubits))

        if clbits is None:
            edge_map.update(zip(other.clbits, dest.clbits))
        else:
            mapped_clbits = dest.cbit_argument_conversion(clbits)
            if len(mapped_clbits) != len(other.clbits):
                raise CircuitError(
                    f"Number of items in clbits parameter ({len(mapped_clbits)}) does not"
                    f" match number of clbits in the circuit ({len(other.clbits)})."
                )
            edge_map.update(zip(other.clbits, dest.cbit_argument_conversion(clbits)))

        variable_mapper = _classical_resource_map.VariableMapper(
            dest.cregs, edge_map, dest.add_register
        )
        mapped_instrs: list[CircuitInstruction] = []
        for instr in other.data:
            n_qargs: list[Qubit] = [edge_map[qarg] for qarg in instr.qubits]
            n_cargs: list[Clbit] = [edge_map[carg] for carg in instr.clbits]
            n_op = instr.operation.copy()
            if (condition := getattr(n_op, "condition", None)) is not None:
                n_op.condition = variable_mapper.map_condition(condition)
            if isinstance(n_op, SwitchCaseOp):
                n_op.target = variable_mapper.map_target(n_op.target)
            mapped_instrs.append(CircuitInstruction(n_op, n_qargs, n_cargs))

        if front:
            # adjust new instrs before original ones and update all parameters
            mapped_instrs += dest.data
            dest.clear()
        append = dest._control_flow_scopes[-1].append if dest._control_flow_scopes else dest._append
        for instr in mapped_instrs:
            append(instr)

        for gate, cals in other.calibrations.items():
            dest._calibrations[gate].update(cals)

        dest.global_phase += other.global_phase

        if inplace:
            return None

        return dest

    def tensor(self, other: "QuantumCircuit", inplace: bool = False) -> Optional["QuantumCircuit"]:
        """Tensor ``self`` with ``other``.

        Remember that in the little-endian convention the leftmost operation will be at the bottom
        of the circuit. See also
        `the docs <qiskit.org/documentation/tutorials/circuits/3_summary_of_quantum_operations.html>`__
        for more information.

        .. parsed-literal::

                 ┌────────┐        ┌─────┐          ┌─────┐
            q_0: ┤ bottom ├ ⊗ q_0: ┤ top ├  = q_0: ─┤ top ├──
                 └────────┘        └─────┘         ┌┴─────┴─┐
                                              q_1: ┤ bottom ├
                                                   └────────┘

        Args:
            other (QuantumCircuit): The other circuit to tensor this circuit with.
            inplace (bool): If True, modify the object. Otherwise return composed circuit.

        Examples:

            .. plot::
               :include-source:

               from qiskit import QuantumCircuit
               top = QuantumCircuit(1)
               top.x(0);
               bottom = QuantumCircuit(2)
               bottom.cry(0.2, 0, 1);
               tensored = bottom.tensor(top)
               tensored.draw('mpl')

        Returns:
            QuantumCircuit: The tensored circuit (returns None if inplace==True).
        """
        num_qubits = self.num_qubits + other.num_qubits
        num_clbits = self.num_clbits + other.num_clbits

        # If a user defined both circuits with via register sizes and not with named registers
        # (e.g. QuantumCircuit(2, 2)) then we have a naming collision, as the registers are by
        # default called "q" resp. "c". To still allow tensoring we define new registers of the
        # correct sizes.
        if (
            len(self.qregs) == len(other.qregs) == 1
            and self.qregs[0].name == other.qregs[0].name == "q"
        ):
            # check if classical registers are in the circuit
            if num_clbits > 0:
                dest = QuantumCircuit(num_qubits, num_clbits)
            else:
                dest = QuantumCircuit(num_qubits)

        # handle case if ``measure_all`` was called on both circuits, in which case the
        # registers are both named "meas"
        elif (
            len(self.cregs) == len(other.cregs) == 1
            and self.cregs[0].name == other.cregs[0].name == "meas"
        ):
            cr = ClassicalRegister(self.num_clbits + other.num_clbits, "meas")
            dest = QuantumCircuit(*other.qregs, *self.qregs, cr)

        # Now we don't have to handle any more cases arising from special implicit naming
        else:
            dest = QuantumCircuit(
                other.qubits,
                self.qubits,
                other.clbits,
                self.clbits,
                *other.qregs,
                *self.qregs,
                *other.cregs,
                *self.cregs,
            )

        # compose self onto the output, and then other
        dest.compose(other, range(other.num_qubits), range(other.num_clbits), inplace=True)
        dest.compose(
            self,
            range(other.num_qubits, num_qubits),
            range(other.num_clbits, num_clbits),
            inplace=True,
        )

        # Replace information from tensored circuit into self when inplace = True
        if inplace:
            self.__dict__.update(dest.__dict__)
            return None
        return dest

    @property
    def qubits(self) -> list[Qubit]:
        """
        Returns a list of quantum bits in the order that the registers were added.
        """
        return self._qubits

    @property
    def clbits(self) -> list[Clbit]:
        """
        Returns a list of classical bits in the order that the registers were added.
        """
        return self._clbits

    @property
    def ancillas(self) -> list[AncillaQubit]:
        """
        Returns a list of ancilla bits in the order that the registers were added.
        """
        return self._ancillas

    def __and__(self, rhs: "QuantumCircuit") -> "QuantumCircuit":
        """Overload & to implement self.compose."""
        return self.compose(rhs)

    def __iand__(self, rhs: "QuantumCircuit") -> "QuantumCircuit":
        """Overload &= to implement self.compose in place."""
        self.compose(rhs, inplace=True)
        return self

    def __xor__(self, top: "QuantumCircuit") -> "QuantumCircuit":
        """Overload ^ to implement self.tensor."""
        return self.tensor(top)

    def __ixor__(self, top: "QuantumCircuit") -> "QuantumCircuit":
        """Overload ^= to implement self.tensor in place."""
        self.tensor(top, inplace=True)
        return self

    def __len__(self) -> int:
        """Return number of operations in circuit."""
        return len(self._data)

    @typing.overload
    def __getitem__(self, item: int) -> CircuitInstruction:
        ...

    @typing.overload
    def __getitem__(self, item: slice) -> list[CircuitInstruction]:
        ...

    def __getitem__(self, item):
        """Return indexed operation."""
        return self._data[item]

    @staticmethod
    def cast(value: S, type_: Callable[..., T]) -> Union[S, T]:
        """Best effort to cast value to type. Otherwise, returns the value."""
        try:
            return type_(value)
        except (ValueError, TypeError):
            return value

    def qbit_argument_conversion(self, qubit_representation: QubitSpecifier) -> list[Qubit]:
        """
        Converts several qubit representations (such as indexes, range, etc.)
        into a list of qubits.

        Args:
            qubit_representation (Object): representation to expand

        Returns:
            List(Qubit): the resolved instances of the qubits.
        """
        return _bit_argument_conversion(
            qubit_representation, self.qubits, self._qubit_indices, Qubit
        )

    def cbit_argument_conversion(self, clbit_representation: ClbitSpecifier) -> list[Clbit]:
        """
        Converts several classical bit representations (such as indexes, range, etc.)
        into a list of classical bits.

        Args:
            clbit_representation (Object): representation to expand

        Returns:
            List(tuple): Where each tuple is a classical bit.
        """
        return _bit_argument_conversion(
            clbit_representation, self.clbits, self._clbit_indices, Clbit
        )

    def _resolve_classical_resource(self, specifier):
        """Resolve a single classical resource specifier into a concrete resource, raising an error
        if the specifier is invalid.

        This is slightly different to :meth:`.cbit_argument_conversion`, because it should not
        unwrap :obj:`.ClassicalRegister` instances into lists, and in general it should not allow
        iterables or broadcasting.  It is expected to be used as a callback for things like
        :meth:`.InstructionSet.c_if` to check the validity of their arguments.

        Args:
            specifier (Union[Clbit, ClassicalRegister, int]): a specifier of a classical resource
                present in this circuit.  An ``int`` will be resolved into a :obj:`.Clbit` using the
                same conventions as measurement operations on this circuit use.

        Returns:
            Union[Clbit, ClassicalRegister]: the resolved resource.

        Raises:
            CircuitError: if the resource is not present in this circuit, or if the integer index
                passed is out-of-bounds.
        """
        if isinstance(specifier, Clbit):
            if specifier not in self._clbit_indices:
                raise CircuitError(f"Clbit {specifier} is not present in this circuit.")
            return specifier
        if isinstance(specifier, ClassicalRegister):
            # This is linear complexity for something that should be constant, but QuantumCircuit
            # does not currently keep a hashmap of registers, and requires non-trivial changes to
            # how it exposes its registers publically before such a map can be safely stored so it
            # doesn't miss updates. (Jake, 2021-11-10).
            if specifier not in self.cregs:
                raise CircuitError(f"Register {specifier} is not present in this circuit.")
            return specifier
        if isinstance(specifier, int):
            try:
                return self._clbits[specifier]
            except IndexError:
                raise CircuitError(f"Classical bit index {specifier} is out-of-range.") from None
        raise CircuitError(f"Unknown classical resource specifier: '{specifier}'.")

    def _validate_expr(self, node: expr.Expr) -> expr.Expr:
        for var in expr.iter_vars(node):
            if isinstance(var.var, Clbit):
                if var.var not in self._clbit_indices:
                    raise CircuitError(f"Clbit {var.var} is not present in this circuit.")
            elif isinstance(var.var, ClassicalRegister):
                if var.var not in self.cregs:
                    raise CircuitError(f"Register {var.var} is not present in this circuit.")
        return node

    def append(
        self,
        instruction: Operation | CircuitInstruction,
        qargs: Sequence[QubitSpecifier] | None = None,
        cargs: Sequence[ClbitSpecifier] | None = None,
    ) -> InstructionSet:
        """Append one or more instructions to the end of the circuit, modifying the circuit in
        place.

        The ``qargs`` and ``cargs`` will be expanded and broadcast according to the rules of the
        given :class:`~.circuit.Instruction`, and any non-:class:`.Bit` specifiers (such as
        integer indices) will be resolved into the relevant instances.

        If a :class:`.CircuitInstruction` is given, it will be unwrapped, verified in the context of
        this circuit, and a new object will be appended to the circuit.  In this case, you may not
        pass ``qargs`` or ``cargs`` separately.

        Args:
            instruction: :class:`~.circuit.Instruction` instance to append, or a
                :class:`.CircuitInstruction` with all its context.
            qargs: specifiers of the :class:`.Qubit`\\ s to attach instruction to.
            cargs: specifiers of the :class:`.Clbit`\\ s to attach instruction to.

        Returns:
            qiskit.circuit.InstructionSet: a handle to the :class:`.CircuitInstruction`\\ s that
            were actually added to the circuit.

        Raises:
            CircuitError: if the operation passed is not an instance of :class:`~.circuit.Instruction` .
        """
        if isinstance(instruction, CircuitInstruction):
            operation = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits
        else:
            operation = instruction

        # Convert input to instruction
        if not isinstance(operation, Operation):
            if hasattr(operation, "to_instruction"):
                operation = operation.to_instruction()
                if not isinstance(operation, Operation):
                    raise CircuitError("operation.to_instruction() is not an Operation.")
            else:
                if issubclass(operation, Operation):
                    raise CircuitError(
                        "Object is a subclass of Operation, please add () to "
                        "pass an instance of this object."
                    )

                raise CircuitError(
                    "Object to append must be an Operation or have a to_instruction() method."
                )

        # Make copy of parameterized gate instances
        if hasattr(operation, "params"):
            is_parameter = any(isinstance(param, Parameter) for param in operation.params)
            if is_parameter:
                operation = copy.deepcopy(operation)

        expanded_qargs = [self.qbit_argument_conversion(qarg) for qarg in qargs or []]
        expanded_cargs = [self.cbit_argument_conversion(carg) for carg in cargs or []]

        if self._control_flow_scopes:
            appender = self._control_flow_scopes[-1].append
            requester = self._control_flow_scopes[-1].request_classical_resource
        else:
            appender = self._append
            requester = self._resolve_classical_resource
        instructions = InstructionSet(resource_requester=requester)
        if isinstance(operation, Instruction):
            for qarg, carg in operation.broadcast_arguments(expanded_qargs, expanded_cargs):
                self._check_dups(qarg)
                instruction = CircuitInstruction(operation, qarg, carg)
                appender(instruction)
                instructions.add(instruction)
        else:
            # For Operations that are non-Instructions, we use the Instruction's default method
            for qarg, carg in Instruction.broadcast_arguments(
                operation, expanded_qargs, expanded_cargs
            ):
                self._check_dups(qarg)
                instruction = CircuitInstruction(operation, qarg, carg)
                appender(instruction)
                instructions.add(instruction)
        return instructions

    # Preferred new style.
    @typing.overload
    def _append(
        self, instruction: CircuitInstruction, _qargs: None = None, _cargs: None = None
    ) -> CircuitInstruction:
        ...

    # To-be-deprecated old style.
    @typing.overload
    def _append(
        self,
        operation: Operation,
        qargs: Sequence[Qubit],
        cargs: Sequence[Clbit],
    ) -> Operation:
        ...

    def _append(
        self,
        instruction: CircuitInstruction | Instruction,
        qargs: Sequence[Qubit] | None = None,
        cargs: Sequence[Clbit] | None = None,
    ):
        """Append an instruction to the end of the circuit, modifying the circuit in place.

        .. warning::

            This is an internal fast-path function, and it is the responsibility of the caller to
            ensure that all the arguments are valid; there is no error checking here.  In
            particular, all the qubits and clbits must already exist in the circuit and there can be
            no duplicates in the list.

        .. note::

            This function may be used by callers other than :obj:`.QuantumCircuit` when the caller
            is sure that all error-checking, broadcasting and scoping has already been performed,
            and the only reference to the circuit the instructions are being appended to is within
            that same function.  In particular, it is not safe to call
            :meth:`QuantumCircuit._append` on a circuit that is received by a function argument.
            This is because :meth:`.QuantumCircuit._append` will not recognise the scoping
            constructs of the control-flow builder interface.

        Args:
            instruction: Operation instance to append
            qargs: Qubits to attach the instruction to.
            cargs: Clbits to attach the instruction to.

        Returns:
            Operation: a handle to the instruction that was just added

        :meta public:
        """
        old_style = not isinstance(instruction, CircuitInstruction)
        if old_style:
            instruction = CircuitInstruction(instruction, qargs, cargs)
        self._data.append(instruction)
        if isinstance(instruction.operation, Instruction):
            self._update_parameter_table(instruction)

        # mark as normal circuit if a new instruction is added
        self.duration = None
        self.unit = "dt"
        return instruction.operation if old_style else instruction

    def _update_parameter_table(self, instruction: CircuitInstruction):
        for param_index, param in enumerate(instruction.operation.params):
            if isinstance(param, (ParameterExpression, QuantumCircuit)):
                # Scoped constructs like the control-flow ops use QuantumCircuit as a parameter.
                atomic_parameters = set(param.parameters)
            else:
                atomic_parameters = set()

            for parameter in atomic_parameters:
                if parameter in self._parameter_table:
                    self._parameter_table[parameter].add((instruction.operation, param_index))
                else:
                    if parameter.name in self._parameter_table.get_names():
                        raise CircuitError(f"Name conflict on adding parameter: {parameter.name}")
                    self._parameter_table[parameter] = ParameterReferences(
                        ((instruction.operation, param_index),)
                    )

                    # clear cache if new parameter is added
                    self._parameters = None

    def add_register(self, *regs: Register | int | Sequence[Bit]) -> None:
        """Add registers."""
        if not regs:
            return

        if any(isinstance(reg, int) for reg in regs):
            # QuantumCircuit defined without registers
            if len(regs) == 1 and isinstance(regs[0], int):
                # QuantumCircuit with anonymous quantum wires e.g. QuantumCircuit(2)
                if regs[0] == 0:
                    regs = ()
                else:
                    regs = (QuantumRegister(regs[0], "q"),)
            elif len(regs) == 2 and all(isinstance(reg, int) for reg in regs):
                # QuantumCircuit with anonymous wires e.g. QuantumCircuit(2, 3)
                if regs[0] == 0:
                    qregs: tuple[QuantumRegister, ...] = ()
                else:
                    qregs = (QuantumRegister(regs[0], "q"),)
                if regs[1] == 0:
                    cregs: tuple[ClassicalRegister, ...] = ()
                else:
                    cregs = (ClassicalRegister(regs[1], "c"),)
                regs = qregs + cregs
            else:
                raise CircuitError(
                    "QuantumCircuit parameters can be Registers or Integers."
                    " If Integers, up to 2 arguments. QuantumCircuit was called"
                    " with %s." % (regs,)
                )

        for register in regs:
            if isinstance(register, Register) and any(
                register.name == reg.name for reg in self.qregs + self.cregs
            ):
                raise CircuitError('register name "%s" already exists' % register.name)

            if isinstance(register, AncillaRegister):
                for bit in register:
                    if bit not in self._qubit_indices:
                        self._ancillas.append(bit)

            if isinstance(register, QuantumRegister):
                self.qregs.append(register)

                for idx, bit in enumerate(register):
                    if bit in self._qubit_indices:
                        self._qubit_indices[bit].registers.append((register, idx))
                    else:
                        self._qubits.append(bit)
                        self._qubit_indices[bit] = BitLocations(
                            len(self._qubits) - 1, [(register, idx)]
                        )

            elif isinstance(register, ClassicalRegister):
                self.cregs.append(register)

                for idx, bit in enumerate(register):
                    if bit in self._clbit_indices:
                        self._clbit_indices[bit].registers.append((register, idx))
                    else:
                        self._clbits.append(bit)
                        self._clbit_indices[bit] = BitLocations(
                            len(self._clbits) - 1, [(register, idx)]
                        )

            elif isinstance(register, list):
                self.add_bits(register)
            else:
                raise CircuitError("expected a register")

    def add_bits(self, bits: Iterable[Bit]) -> None:
        """Add Bits to the circuit."""
        duplicate_bits = set(self._qubit_indices).union(self._clbit_indices).intersection(bits)
        if duplicate_bits:
            raise CircuitError(f"Attempted to add bits found already in circuit: {duplicate_bits}")

        for bit in bits:
            if isinstance(bit, AncillaQubit):
                self._ancillas.append(bit)
            if isinstance(bit, Qubit):
                self._qubits.append(bit)
                self._qubit_indices[bit] = BitLocations(len(self._qubits) - 1, [])
            elif isinstance(bit, Clbit):
                self._clbits.append(bit)
                self._clbit_indices[bit] = BitLocations(len(self._clbits) - 1, [])
            else:
                raise CircuitError(
                    "Expected an instance of Qubit, Clbit, or "
                    "AncillaQubit, but was passed {}".format(bit)
                )

    def find_bit(self, bit: Bit) -> BitLocations:
        """Find locations in the circuit which can be used to reference a given :obj:`~Bit`.

        Args:
            bit (Bit): The bit to locate.

        Returns:
            namedtuple(int, List[Tuple(Register, int)]): A 2-tuple. The first element (``index``)
                contains the index at which the ``Bit`` can be found (in either
                :obj:`~QuantumCircuit.qubits`, :obj:`~QuantumCircuit.clbits`, depending on its
                type). The second element (``registers``) is a list of ``(register, index)``
                pairs with an entry for each :obj:`~Register` in the circuit which contains the
                :obj:`~Bit` (and the index in the :obj:`~Register` at which it can be found).

        Notes:
            The circuit index of an :obj:`~AncillaQubit` will be its index in
            :obj:`~QuantumCircuit.qubits`, not :obj:`~QuantumCircuit.ancillas`.

        Raises:
            CircuitError: If the supplied :obj:`~Bit` was of an unknown type.
            CircuitError: If the supplied :obj:`~Bit` could not be found on the circuit.
        """

        try:
            if isinstance(bit, Qubit):
                return self._qubit_indices[bit]
            elif isinstance(bit, Clbit):
                return self._clbit_indices[bit]
            else:
                raise CircuitError(f"Could not locate bit of unknown type: {type(bit)}")
        except KeyError as err:
            raise CircuitError(
                f"Could not locate provided bit: {bit}. Has it been added to the QuantumCircuit?"
            ) from err

    def _check_dups(self, qubits: Sequence[Qubit]) -> None:
        """Raise exception if list of qubits contains duplicates."""
        squbits = set(qubits)
        if len(squbits) != len(qubits):
            raise CircuitError("duplicate qubit arguments")

    def to_instruction(
        self,
        parameter_map: dict[Parameter, ParameterValueType] | None = None,
        label: str | None = None,
    ) -> Instruction:
        """Create an Instruction out of this circuit.

        Args:
            parameter_map(dict): For parameterized circuits, a mapping from
               parameters in the circuit to parameters to be used in the
               instruction. If None, existing circuit parameters will also
               parameterize the instruction.
            label (str): Optional gate label.

        Returns:
            qiskit.circuit.Instruction: a composite instruction encapsulating this circuit
            (can be decomposed back)
        """
        from qiskit.converters.circuit_to_instruction import circuit_to_instruction

        return circuit_to_instruction(self, parameter_map, label=label)

    def to_gate(
        self,
        parameter_map: dict[Parameter, ParameterValueType] | None = None,
        label: str | None = None,
    ) -> Gate:
        """Create a Gate out of this circuit.

        Args:
            parameter_map(dict): For parameterized circuits, a mapping from
               parameters in the circuit to parameters to be used in the
               gate. If None, existing circuit parameters will also
               parameterize the gate.
            label (str): Optional gate label.

        Returns:
            Gate: a composite gate encapsulating this circuit
            (can be decomposed back)
        """
        from qiskit.converters.circuit_to_gate import circuit_to_gate

        return circuit_to_gate(self, parameter_map, label=label)

    def decompose(
        self,
        gates_to_decompose: Type[Gate] | Sequence[Type[Gate]] | Sequence[str] | str | None = None,
        reps: int = 1,
    ) -> "QuantumCircuit":
        """Call a decomposition pass on this circuit,
        to decompose one level (shallow decompose).

        Args:
            gates_to_decompose (type or str or list(type, str)): Optional subset of gates
                to decompose. Can be a gate type, such as ``HGate``, or a gate name, such
                as 'h', or a gate label, such as 'My H Gate', or a list of any combination
                of these. If a gate name is entered, it will decompose all gates with that
                name, whether the gates have labels or not. Defaults to all gates in circuit.
            reps (int): Optional number of times the circuit should be decomposed.
                For instance, ``reps=2`` equals calling ``circuit.decompose().decompose()``.
                can decompose specific gates specific time

        Returns:
            QuantumCircuit: a circuit one level decomposed
        """
        # pylint: disable=cyclic-import
        from qiskit.transpiler.passes.basis.decompose import Decompose
        from qiskit.transpiler.passes.synthesis import HighLevelSynthesis
        from qiskit.converters.circuit_to_dag import circuit_to_dag
        from qiskit.converters.dag_to_circuit import dag_to_circuit

        dag = circuit_to_dag(self)
        dag = HighLevelSynthesis().run(dag)
        pass_ = Decompose(gates_to_decompose)
        for _ in range(reps):
            dag = pass_.run(dag)
        return dag_to_circuit(dag)

    def qasm(
        self,
        formatted: bool = False,
        filename: str | None = None,
        encoding: str | None = None,
    ) -> str | None:
        """Return OpenQASM string.

        Args:
            formatted (bool): Return formatted Qasm string.
            filename (str): Save Qasm to file with name 'filename'.
            encoding (str): Optionally specify the encoding to use for the
                output file if ``filename`` is specified. By default this is
                set to the system's default encoding (ie whatever
                ``locale.getpreferredencoding()`` returns) and can be set to
                any valid codec or alias from stdlib's
                `codec module <https://docs.python.org/3/library/codecs.html#standard-encodings>`__

        Returns:
            str: If formatted=False.

        Raises:
            MissingOptionalLibraryError: If pygments is not installed and ``formatted`` is
                ``True``.
            QASM2ExportError: If circuit has free parameters.
            QASM2ExportError: If an operation that has no OpenQASM 2 representation is encountered.
        """
        from qiskit.qasm2 import QASM2ExportError  # pylint: disable=cyclic-import

        if self.num_parameters > 0:
            raise QASM2ExportError(
                "Cannot represent circuits with unbound parameters in OpenQASM 2."
            )

        existing_gate_names = {
            "barrier",
            "measure",
            "reset",
            "u3",
            "u2",
            "u1",
            "cx",
            "id",
            "u0",
            "u",
            "p",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "t",
            "tdg",
            "rx",
            "ry",
            "rz",
            "sx",
            "sxdg",
            "cz",
            "cy",
            "swap",
            "ch",
            "ccx",
            "cswap",
            "crx",
            "cry",
            "crz",
            "cu1",
            "cp",
            "cu3",
            "csx",
            "cu",
            "rxx",
            "rzz",
            "rccx",
            "rc3x",
            "c3x",
            "c3sx",  # This is the Qiskit gate name, but the qelib1.inc name is 'c3sqrtx'.
            "c4x",
        }

        # Mapping of instruction name to a pair of the source for a definition, and an OQ2 string
        # that includes the `gate` or `opaque` statement that defines the gate.
        gates_to_define: OrderedDict[str, tuple[Instruction, str]] = OrderedDict()

        regless_qubits = [bit for bit in self.qubits if not self.find_bit(bit).registers]
        regless_clbits = [bit for bit in self.clbits if not self.find_bit(bit).registers]
        dummy_registers: list[QuantumRegister | ClassicalRegister] = []
        if regless_qubits:
            dummy_registers.append(QuantumRegister(name="qregless", bits=regless_qubits))
        if regless_clbits:
            dummy_registers.append(ClassicalRegister(name="cregless", bits=regless_clbits))
        register_escaped_names: dict[str, QuantumRegister | ClassicalRegister] = {}
        for regs in (self.qregs, self.cregs, dummy_registers):
            for reg in regs:
                register_escaped_names[
                    _make_unique(_qasm_escape_name(reg.name, "reg_"), register_escaped_names)
                ] = reg
        bit_labels: dict[Qubit | Clbit, str] = {
            bit: "%s[%d]" % (name, idx)
            for name, register in register_escaped_names.items()
            for (idx, bit) in enumerate(register)
        }
        register_definitions_qasm = "".join(
            f"{'qreg' if isinstance(reg, QuantumRegister) else 'creg'} {name}[{reg.size}];\n"
            for name, reg in register_escaped_names.items()
        )
        instruction_calls = []
        for instruction in self._data:
            operation = instruction.operation
            if operation.name == "measure":
                qubit = instruction.qubits[0]
                clbit = instruction.clbits[0]
                instruction_qasm = f"measure {bit_labels[qubit]} -> {bit_labels[clbit]};"
            elif operation.name == "reset":
                instruction_qasm = f"reset {bit_labels[instruction.qubits[0]]};"
            elif operation.name == "barrier":
                if not instruction.qubits:
                    # Barriers with no operands are invalid in (strict) OQ2, and the statement
                    # would have no meaning anyway.
                    continue
                qargs = ",".join(bit_labels[q] for q in instruction.qubits)
                instruction_qasm = "barrier;" if not qargs else f"barrier {qargs};"
            else:
                instruction_qasm = _qasm2_custom_operation_statement(
                    instruction, existing_gate_names, gates_to_define, bit_labels
                )
            instruction_calls.append(instruction_qasm)
        instructions_qasm = "".join(f"{call}\n" for call in instruction_calls)
        gate_definitions_qasm = "".join(f"{qasm}\n" for _, qasm in gates_to_define.values())

        out = "".join(
            (
                self.header,
                "\n",
                self.extension_lib,
                "\n",
                gate_definitions_qasm,
                register_definitions_qasm,
                instructions_qasm,
            )
        )

        if filename:
            with open(filename, "w+", encoding=encoding) as file:
                file.write(out)

        if formatted:
            _optionals.HAS_PYGMENTS.require_now("formatted OpenQASM 2 output")

            import pygments
            from pygments.formatters import (  # pylint: disable=no-name-in-module
                Terminal256Formatter,
            )
            from qiskit.qasm.pygments import OpenQASMLexer
            from qiskit.qasm.pygments import QasmTerminalStyle

            code = pygments.highlight(
                out, OpenQASMLexer(), Terminal256Formatter(style=QasmTerminalStyle)
            )
            print(code)
            return None
        return out

    def draw(
        self,
        output: str | None = None,
        scale: float | None = None,
        filename: str | None = None,
        style: dict | str | None = None,
        interactive: bool = False,
        plot_barriers: bool = True,
        reverse_bits: bool = None,
        justify: str | None = None,
        vertical_compression: str | None = "medium",
        idle_wires: bool = True,
        with_layout: bool = True,
        fold: int | None = None,
        # The type of ax is matplotlib.axes.Axes, but this is not a fixed dependency, so cannot be
        # safely forward-referenced.
        ax: Any | None = None,
        initial_state: bool = False,
        cregbundle: bool = None,
        wire_order: list = None,
    ):
        """Draw the quantum circuit. Use the output parameter to choose the drawing format:

        **text**: ASCII art TextDrawing that can be printed in the console.

        **mpl**: images with color rendered purely in Python using matplotlib.

        **latex**: high-quality images compiled via latex.

        **latex_source**: raw uncompiled latex output.

        .. warning::

            Support for :class:`~.expr.Expr` nodes in conditions and :attr:`.SwitchCaseOp.target`
            fields is preliminary and incomplete.  The ``text`` and ``mpl`` drawers will make a
            best-effort attempt to show data dependencies, but the LaTeX-based drawers will skip
            these completely.

        Args:
            output (str): select the output method to use for drawing the circuit.
                Valid choices are ``text``, ``mpl``, ``latex``, ``latex_source``.
                By default the `text` drawer is used unless the user config file
                (usually ``~/.qiskit/settings.conf``) has an alternative backend set
                as the default. For example, ``circuit_drawer = latex``. If the output
                kwarg is set, that backend will always be used over the default in
                the user config file.
            scale (float): scale of image to draw (shrink if < 1.0). Only used by
                the `mpl`, `latex` and `latex_source` outputs. Defaults to 1.0.
            filename (str): file path to save image to. Defaults to None.
            style (dict or str): dictionary of style or file name of style json file.
                This option is only used by the `mpl` or `latex` output type.
                If `style` is a str, it is used as the path to a json file
                which contains a style dict. The file will be opened, parsed, and
                then any style elements in the dict will replace the default values
                in the input dict. A file to be loaded must end in ``.json``, but
                the name entered here can omit ``.json``. For example,
                ``style='iqx.json'`` or ``style='iqx'``.
                If `style` is a dict and the ``'name'`` key is set, that name
                will be used to load a json file, followed by loading the other
                items in the style dict. For example, ``style={'name': 'iqx'}``.
                If `style` is not a str and `name` is not a key in the style dict,
                then the default value from the user config file (usually
                ``~/.qiskit/settings.conf``) will be used, for example,
                ``circuit_mpl_style = iqx``.
                If none of these are set, the `default` style will be used.
                The search path for style json files can be specified in the user
                config, for example,
                ``circuit_mpl_style_path = /home/user/styles:/home/user``.
                See: :class:`~qiskit.visualization.qcstyle.DefaultStyle` for more
                information on the contents.
            interactive (bool): when set to true, show the circuit in a new window
                (for `mpl` this depends on the matplotlib backend being used
                supporting this). Note when used with either the `text` or the
                `latex_source` output type this has no effect and will be silently
                ignored. Defaults to False.
            reverse_bits (bool): when set to True, reverse the bit order inside
                registers for the output visualization. Defaults to False unless the
                user config file (usually ``~/.qiskit/settings.conf``) has an
                alternative value set. For example, ``circuit_reverse_bits = True``.
            plot_barriers (bool): enable/disable drawing barriers in the output
                circuit. Defaults to True.
            justify (string): options are ``left``, ``right`` or ``none``. If
                anything else is supplied, it defaults to left justified. It refers
                to where gates should be placed in the output circuit if there is
                an option. ``none`` results in each gate being placed in its own
                column.
            vertical_compression (string): ``high``, ``medium`` or ``low``. It
                merges the lines generated by the `text` output so the drawing
                will take less vertical room.  Default is ``medium``. Only used by
                the `text` output, will be silently ignored otherwise.
            idle_wires (bool): include idle wires (wires with no circuit elements)
                in output visualization. Default is True.
            with_layout (bool): include layout information, with labels on the
                physical layout. Default is True.
            fold (int): sets pagination. It can be disabled using -1. In `text`,
                sets the length of the lines. This is useful when the drawing does
                not fit in the console. If None (default), it will try to guess the
                console width using ``shutil.get_terminal_size()``. However, if
                running in jupyter, the default line length is set to 80 characters.
                In `mpl`, it is the number of (visual) layers before folding.
                Default is 25.
            ax (matplotlib.axes.Axes): Only used by the `mpl` backend. An optional
                Axes object to be used for the visualization output. If none is
                specified, a new matplotlib Figure will be created and used.
                Additionally, if specified there will be no returned Figure since
                it is redundant.
            initial_state (bool): Optional. Adds ``|0>`` in the beginning of the wire.
                Default is False.
            cregbundle (bool): Optional. If set True, bundle classical registers.
                Default is True, except for when ``output`` is set to  ``"text"``.
            wire_order (list): Optional. A list of integers used to reorder the display
                of the bits. The list must have an entry for every bit with the bits
                in the range 0 to (``num_qubits`` + ``num_clbits``).

        Returns:
            :class:`.TextDrawing` or :class:`matplotlib.figure` or :class:`PIL.Image` or
            :class:`str`:

            * `TextDrawing` (output='text')
                A drawing that can be printed as ascii art.
            * `matplotlib.figure.Figure` (output='mpl')
                A matplotlib figure object for the circuit diagram.
            * `PIL.Image` (output='latex')
                An in-memory representation of the image of the circuit diagram.
            * `str` (output='latex_source')
                The LaTeX source code for visualizing the circuit diagram.

        Raises:
            VisualizationError: when an invalid output method is selected
            ImportError: when the output methods requires non-installed libraries.

        Example:
            .. plot::
               :include-source:

               from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
               q = QuantumRegister(1)
               c = ClassicalRegister(1)
               qc = QuantumCircuit(q, c)
               qc.h(q)
               qc.measure(q, c)
               qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
        """

        # pylint: disable=cyclic-import
        from qiskit.visualization import circuit_drawer

        return circuit_drawer(
            self,
            scale=scale,
            filename=filename,
            style=style,
            output=output,
            interactive=interactive,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            vertical_compression=vertical_compression,
            idle_wires=idle_wires,
            with_layout=with_layout,
            fold=fold,
            ax=ax,
            initial_state=initial_state,
            cregbundle=cregbundle,
            wire_order=wire_order,
        )

    def size(
        self,
        filter_function: Callable[..., int] = lambda x: not getattr(
            x.operation, "_directive", False
        ),
    ) -> int:
        """Returns total number of instructions in circuit.

        Args:
            filter_function (callable): a function to filter out some instructions.
                Should take as input a tuple of (Instruction, list(Qubit), list(Clbit)).
                By default filters out "directives", such as barrier or snapshot.

        Returns:
            int: Total number of gate operations.
        """
        return sum(map(filter_function, self._data))

    def depth(
        self,
        filter_function: Callable[..., int] = lambda x: not getattr(
            x.operation, "_directive", False
        ),
    ) -> int:
        """Return circuit depth (i.e., length of critical path).

        Args:
            filter_function (callable): A function to filter instructions.
                Should take as input a tuple of (Instruction, list(Qubit), list(Clbit)).
                Instructions for which the function returns False are ignored in the
                computation of the circuit depth.
                By default filters out "directives", such as barrier or snapshot.

        Returns:
            int: Depth of circuit.

        Notes:
            The circuit depth and the DAG depth need not be the
            same.
        """
        # Assign each bit in the circuit a unique integer
        # to index into op_stack.
        bit_indices: dict[Qubit | Clbit, int] = {
            bit: idx for idx, bit in enumerate(self.qubits + self.clbits)
        }

        # If no bits, return 0
        if not bit_indices:
            return 0

        # A list that holds the height of each qubit
        # and classical bit.
        op_stack = [0] * len(bit_indices)

        # Here we are playing a modified version of
        # Tetris where we stack gates, but multi-qubit
        # gates, or measurements have a block for each
        # qubit or cbit that are connected by a virtual
        # line so that they all stacked at the same depth.
        # Conditional gates act on all cbits in the register
        # they are conditioned on.
        # The max stack height is the circuit depth.
        for instruction in self._data:
            levels = []
            reg_ints = []
            for ind, reg in enumerate(instruction.qubits + instruction.clbits):
                # Add to the stacks of the qubits and
                # cbits used in the gate.
                reg_ints.append(bit_indices[reg])
                if filter_function(instruction):
                    levels.append(op_stack[reg_ints[ind]] + 1)
                else:
                    levels.append(op_stack[reg_ints[ind]])
            # Assuming here that there is no conditional
            # snapshots or barriers ever.
            if getattr(instruction.operation, "condition", None):
                # Controls operate over all bits of a classical register
                # or over a single bit
                if isinstance(instruction.operation.condition[0], Clbit):
                    condition_bits = [instruction.operation.condition[0]]
                else:
                    condition_bits = instruction.operation.condition[0]
                for cbit in condition_bits:
                    idx = bit_indices[cbit]
                    if idx not in reg_ints:
                        reg_ints.append(idx)
                        levels.append(op_stack[idx] + 1)

            max_level = max(levels)
            for ind in reg_ints:
                op_stack[ind] = max_level

        return max(op_stack)

    def width(self) -> int:
        """Return number of qubits plus clbits in circuit.

        Returns:
            int: Width of circuit.

        """
        return len(self.qubits) + len(self.clbits)

    @property
    def num_qubits(self) -> int:
        """Return number of qubits."""
        return len(self.qubits)

    @property
    def num_ancillas(self) -> int:
        """Return the number of ancilla qubits."""
        return len(self.ancillas)

    @property
    def num_clbits(self) -> int:
        """Return number of classical bits."""
        return len(self.clbits)

    def count_ops(self) -> OrderedDict[Instruction, int]:
        """Count each operation kind in the circuit.

        Returns:
            OrderedDict: a breakdown of how many operations of each kind, sorted by amount.
        """
        count_ops: dict[Instruction, int] = {}
        for instruction in self._data:
            count_ops[instruction.operation.name] = count_ops.get(instruction.operation.name, 0) + 1
        return OrderedDict(sorted(count_ops.items(), key=lambda kv: kv[1], reverse=True))

    def num_nonlocal_gates(self) -> int:
        """Return number of non-local gates (i.e. involving 2+ qubits).

        Conditional nonlocal gates are also included.
        """
        multi_qubit_gates = 0
        for instruction in self._data:
            if instruction.operation.num_qubits > 1 and not getattr(
                instruction.operation, "_directive", False
            ):
                multi_qubit_gates += 1
        return multi_qubit_gates

    def get_instructions(self, name: str) -> list[CircuitInstruction]:
        """Get instructions matching name.

        Args:
            name (str): The name of instruction to.

        Returns:
            list(tuple): list of (instruction, qargs, cargs).
        """
        return [match for match in self._data if match.operation.name == name]

    def num_connected_components(self, unitary_only: bool = False) -> int:
        """How many non-entangled subcircuits can the circuit be factored to.

        Args:
            unitary_only (bool): Compute only unitary part of graph.

        Returns:
            int: Number of connected components in circuit.
        """
        # Convert registers to ints (as done in depth).
        bits = self.qubits if unitary_only else (self.qubits + self.clbits)
        bit_indices: dict[Qubit | Clbit, int] = {bit: idx for idx, bit in enumerate(bits)}

        # Start with each qubit or cbit being its own subgraph.
        sub_graphs = [[bit] for bit in range(len(bit_indices))]

        num_sub_graphs = len(sub_graphs)

        # Here we are traversing the gates and looking to see
        # which of the sub_graphs the gate joins together.
        for instruction in self._data:
            if unitary_only:
                args = instruction.qubits
                num_qargs = len(args)
            else:
                args = instruction.qubits + instruction.clbits
                num_qargs = len(args) + (
                    1 if getattr(instruction.operation, "condition", None) else 0
                )

            if num_qargs >= 2 and not getattr(instruction.operation, "_directive", False):
                graphs_touched = []
                num_touched = 0
                # Controls necessarily join all the cbits in the
                # register that they use.
                if not unitary_only:
                    for bit in instruction.operation.condition_bits:
                        idx = bit_indices[bit]
                        for k in range(num_sub_graphs):
                            if idx in sub_graphs[k]:
                                graphs_touched.append(k)
                                break

                for item in args:
                    reg_int = bit_indices[item]
                    for k in range(num_sub_graphs):
                        if reg_int in sub_graphs[k]:
                            if k not in graphs_touched:
                                graphs_touched.append(k)
                                break

                graphs_touched = list(set(graphs_touched))
                num_touched = len(graphs_touched)

                # If the gate touches more than one subgraph
                # join those graphs together and return
                # reduced number of subgraphs
                if num_touched > 1:
                    connections = []
                    for idx in graphs_touched:
                        connections.extend(sub_graphs[idx])
                    _sub_graphs = []
                    for idx in range(num_sub_graphs):
                        if idx not in graphs_touched:
                            _sub_graphs.append(sub_graphs[idx])
                    _sub_graphs.append(connections)
                    sub_graphs = _sub_graphs
                    num_sub_graphs -= num_touched - 1
            # Cannot go lower than one so break
            if num_sub_graphs == 1:
                break
        return num_sub_graphs

    def num_unitary_factors(self) -> int:
        """Computes the number of tensor factors in the unitary
        (quantum) part of the circuit only.
        """
        return self.num_connected_components(unitary_only=True)

    def num_tensor_factors(self) -> int:
        """Computes the number of tensor factors in the unitary
        (quantum) part of the circuit only.

        Notes:
            This is here for backwards compatibility, and will be
            removed in a future release of Qiskit. You should call
            `num_unitary_factors` instead.
        """
        return self.num_unitary_factors()

    def copy(self, name: str | None = None) -> "QuantumCircuit":
        """Copy the circuit.

        Args:
          name (str): name to be given to the copied circuit. If None, then the name stays the same.

        Returns:
          QuantumCircuit: a deepcopy of the current circuit, with the specified name
        """
        cpy = self.copy_empty_like(name)

        operation_copies = {
            id(instruction.operation): instruction.operation.copy() for instruction in self._data
        }

        cpy._parameter_table = ParameterTable(
            {
                param: ParameterReferences(
                    (operation_copies[id(operation)], param_index)
                    for operation, param_index in self._parameter_table[param]
                )
                for param in self._parameter_table
            }
        )

        cpy._data = [
            instruction.replace(operation=operation_copies[id(instruction.operation)])
            for instruction in self._data
        ]

        return cpy

    def copy_empty_like(self, name: str | None = None) -> "QuantumCircuit":
        """Return a copy of self with the same structure but empty.

        That structure includes:
            * name, calibrations and other metadata
            * global phase
            * all the qubits and clbits, including the registers

        Args:
            name (str): Name for the copied circuit. If None, then the name stays the same.

        Returns:
            QuantumCircuit: An empty copy of self.
        """
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                f"invalid name for a circuit: '{name}'. The name must be a string or 'None'."
            )
        cpy = copy.copy(self)
        # copy registers correctly, in copy.copy they are only copied via reference
        cpy.qregs = self.qregs.copy()
        cpy.cregs = self.cregs.copy()
        cpy._qubits = self._qubits.copy()
        cpy._ancillas = self._ancillas.copy()
        cpy._clbits = self._clbits.copy()
        cpy._qubit_indices = self._qubit_indices.copy()
        cpy._clbit_indices = self._clbit_indices.copy()

        cpy._parameter_table = ParameterTable()
        cpy._data = []

        cpy._calibrations = copy.deepcopy(self._calibrations)
        cpy._metadata = copy.deepcopy(self._metadata)

        if name:
            cpy.name = name
        return cpy

    def clear(self) -> None:
        """Clear all instructions in self.

        Clearing the circuits will keep the metadata and calibrations.
        """
        self._data.clear()
        self._parameter_table.clear()

    def _create_creg(self, length: int, name: str) -> ClassicalRegister:
        """Creates a creg, checking if ClassicalRegister with same name exists"""
        if name in [creg.name for creg in self.cregs]:
            save_prefix = ClassicalRegister.prefix
            ClassicalRegister.prefix = name
            new_creg = ClassicalRegister(length)
            ClassicalRegister.prefix = save_prefix
        else:
            new_creg = ClassicalRegister(length, name)
        return new_creg

    def _create_qreg(self, length: int, name: str) -> QuantumRegister:
        """Creates a qreg, checking if QuantumRegister with same name exists"""
        if name in [qreg.name for qreg in self.qregs]:
            save_prefix = QuantumRegister.prefix
            QuantumRegister.prefix = name
            new_qreg = QuantumRegister(length)
            QuantumRegister.prefix = save_prefix
        else:
            new_qreg = QuantumRegister(length, name)
        return new_qreg

    def reset(self, qubit: QubitSpecifier) -> InstructionSet:
        """Reset the quantum bit(s) to their default state.

        Args:
            qubit: qubit(s) to reset.

        Returns:
            qiskit.circuit.InstructionSet: handle to the added instruction.
        """
        return self.append(Reset(), [qubit], [])

    def measure(self, qubit: QubitSpecifier, cbit: ClbitSpecifier) -> InstructionSet:
        r"""Measure a quantum bit (``qubit``) in the Z basis into a classical bit (``cbit``).

        When a quantum state is measured, a qubit is projected in the computational (Pauli Z) basis
        to either :math:`\lvert 0 \rangle` or :math:`\lvert 1 \rangle`. The classical bit ``cbit``
        indicates the result
        of that projection as a ``0`` or a ``1`` respectively. This operation is non-reversible.

        Args:
            qubit: qubit(s) to measure.
            cbit: classical bit(s) to place the measurement result(s) in.

        Returns:
            qiskit.circuit.InstructionSet: handle to the added instructions.

        Raises:
            CircuitError: if arguments have bad format.

        Examples:
            In this example, a qubit is measured and the result of that measurement is stored in the
            classical bit (usually expressed in diagrams as a double line):

            .. code-block::

               from qiskit import QuantumCircuit
               circuit = QuantumCircuit(1, 1)
               circuit.h(0)
               circuit.measure(0, 0)
               circuit.draw()


            .. parsed-literal::

                      ┌───┐┌─┐
                   q: ┤ H ├┤M├
                      └───┘└╥┘
                 c: 1/══════╩═
                            0

            It is possible to call ``measure`` with lists of ``qubits`` and ``cbits`` as a shortcut
            for one-to-one measurement. These two forms produce identical results:

            .. code-block::

               circuit = QuantumCircuit(2, 2)
               circuit.measure([0,1], [0,1])

            .. code-block::

               circuit = QuantumCircuit(2, 2)
               circuit.measure(0, 0)
               circuit.measure(1, 1)

            Instead of lists, you can use :class:`~qiskit.circuit.QuantumRegister` and
            :class:`~qiskit.circuit.ClassicalRegister` under the same logic.

            .. code-block::

                from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
                qreg = QuantumRegister(2, "qreg")
                creg = ClassicalRegister(2, "creg")
                circuit = QuantumCircuit(qreg, creg)
                circuit.measure(qreg, creg)

            This is equivalent to:

            .. code-block::

                circuit = QuantumCircuit(qreg, creg)
                circuit.measure(qreg[0], creg[0])
                circuit.measure(qreg[1], creg[1])

        """
        return self.append(Measure(), [qubit], [cbit])

    def measure_active(self, inplace: bool = True) -> Optional["QuantumCircuit"]:
        """Adds measurement to all non-idle qubits. Creates a new ClassicalRegister with
        a size equal to the number of non-idle qubits being measured.

        Returns a new circuit with measurements if `inplace=False`.

        Args:
            inplace (bool): All measurements inplace or return new circuit.

        Returns:
            QuantumCircuit: Returns circuit with measurements when `inplace = False`.
        """
        from qiskit.converters.circuit_to_dag import circuit_to_dag

        if inplace:
            circ = self
        else:
            circ = self.copy()
        dag = circuit_to_dag(circ)
        qubits_to_measure = [qubit for qubit in circ.qubits if qubit not in dag.idle_wires()]
        new_creg = circ._create_creg(len(qubits_to_measure), "measure")
        circ.add_register(new_creg)
        circ.barrier()
        circ.measure(qubits_to_measure, new_creg)

        if not inplace:
            return circ
        else:
            return None

    def measure_all(
        self, inplace: bool = True, add_bits: bool = True
    ) -> Optional["QuantumCircuit"]:
        """Adds measurement to all qubits.

        By default, adds new classical bits in a :obj:`.ClassicalRegister` to store these
        measurements.  If ``add_bits=False``, the results of the measurements will instead be stored
        in the already existing classical bits, with qubit ``n`` being measured into classical bit
        ``n``.

        Returns a new circuit with measurements if ``inplace=False``.

        Args:
            inplace (bool): All measurements inplace or return new circuit.
            add_bits (bool): Whether to add new bits to store the results.

        Returns:
            QuantumCircuit: Returns circuit with measurements when ``inplace=False``.

        Raises:
            CircuitError: if ``add_bits=False`` but there are not enough classical bits.
        """
        if inplace:
            circ = self
        else:
            circ = self.copy()
        if add_bits:
            new_creg = circ._create_creg(len(circ.qubits), "meas")
            circ.add_register(new_creg)
            circ.barrier()
            circ.measure(circ.qubits, new_creg)
        else:
            if len(circ.clbits) < len(circ.qubits):
                raise CircuitError(
                    "The number of classical bits must be equal or greater than "
                    "the number of qubits."
                )
            circ.barrier()
            circ.measure(circ.qubits, circ.clbits[0 : len(circ.qubits)])

        if not inplace:
            return circ
        else:
            return None

    def remove_final_measurements(self, inplace: bool = True) -> Optional["QuantumCircuit"]:
        """Removes final measurements and barriers on all qubits if they are present.
        Deletes the classical registers that were used to store the values from these measurements
        that become idle as a result of this operation, and deletes classical bits that are
        referenced only by removed registers, or that aren't referenced at all but have
        become idle as a result of this operation.

        Measurements and barriers are considered final if they are
        followed by no other operations (aside from other measurements or barriers.)

        Args:
            inplace (bool): All measurements removed inplace or return new circuit.

        Returns:
            QuantumCircuit: Returns the resulting circuit when ``inplace=False``, else None.
        """
        # pylint: disable=cyclic-import
        from qiskit.transpiler.passes import RemoveFinalMeasurements
        from qiskit.converters import circuit_to_dag

        if inplace:
            circ = self
        else:
            circ = self.copy()

        dag = circuit_to_dag(circ)
        remove_final_meas = RemoveFinalMeasurements()
        new_dag = remove_final_meas.run(dag)
        kept_cregs = set(new_dag.cregs.values())
        kept_clbits = set(new_dag.clbits)

        # Filter only cregs/clbits still in new DAG, preserving original circuit order
        cregs_to_add = [creg for creg in circ.cregs if creg in kept_cregs]
        clbits_to_add = [clbit for clbit in circ._clbits if clbit in kept_clbits]

        # Clear cregs and clbits
        circ.cregs = []
        circ._clbits = []
        circ._clbit_indices = {}

        # We must add the clbits first to preserve the original circuit
        # order. This way, add_register never adds clbits and just
        # creates registers that point to them.
        circ.add_bits(clbits_to_add)
        for creg in cregs_to_add:
            circ.add_register(creg)

        # Clear instruction info
        circ.data.clear()
        circ._parameter_table.clear()

        # Set circ instructions to match the new DAG
        for node in new_dag.topological_op_nodes():
            # Get arguments for classical condition (if any)
            inst = node.op.copy()
            circ.append(inst, node.qargs, node.cargs)

        if not inplace:
            return circ
        else:
            return None

    @staticmethod
    def from_qasm_file(path: str) -> "QuantumCircuit":
        """Take in a QASM file and generate a QuantumCircuit object.

        Args:
          path (str): Path to the file for a QASM program

        Return:
          QuantumCircuit: The QuantumCircuit object for the input QASM

        See also:
            :func:`.qasm2.load`: the complete interface to the OpenQASM 2 importer.
        """
        # pylint: disable=cyclic-import
        from qiskit import qasm2

        return qasm2.load(
            path,
            include_path=qasm2.LEGACY_INCLUDE_PATH,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
            strict=False,
        )

    @staticmethod
    def from_qasm_str(qasm_str: str) -> "QuantumCircuit":
        """Take in a QASM string and generate a QuantumCircuit object.

        Args:
          qasm_str (str): A QASM program string
        Return:
          QuantumCircuit: The QuantumCircuit object for the input QASM

        See also:
            :func:`.qasm2.loads`: the complete interface to the OpenQASM 2 importer.
        """
        # pylint: disable=cyclic-import
        from qiskit import qasm2

        return qasm2.loads(
            qasm_str,
            include_path=qasm2.LEGACY_INCLUDE_PATH,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
            strict=False,
        )

    @property
    def global_phase(self) -> ParameterValueType:
        """Return the global phase of the circuit in radians."""
        return self._global_phase

    @global_phase.setter
    def global_phase(self, angle: ParameterValueType):
        """Set the phase of the circuit.

        Args:
            angle (float, ParameterExpression): radians
        """
        if isinstance(angle, ParameterExpression) and angle.parameters:
            self._global_phase = angle
        else:
            # Set the phase to the [0, 2π) interval
            angle = float(angle)
            if not angle:
                self._global_phase = 0
            else:
                self._global_phase = angle % (2 * np.pi)

    @property
    def parameters(self) -> ParameterView:
        """The parameters defined in the circuit.

        This attribute returns the :class:`.Parameter` objects in the circuit sorted
        alphabetically. Note that parameters instantiated with a :class:`.ParameterVector`
        are still sorted numerically.

        Examples:

            The snippet below shows that insertion order of parameters does not matter.

            .. code-block:: python

                >>> from qiskit.circuit import QuantumCircuit, Parameter
                >>> a, b, elephant = Parameter("a"), Parameter("b"), Parameter("elephant")
                >>> circuit = QuantumCircuit(1)
                >>> circuit.rx(b, 0)
                >>> circuit.rz(elephant, 0)
                >>> circuit.ry(a, 0)
                >>> circuit.parameters  # sorted alphabetically!
                ParameterView([Parameter(a), Parameter(b), Parameter(elephant)])

            Bear in mind that alphabetical sorting might be unituitive when it comes to numbers.
            The literal "10" comes before "2" in strict alphabetical sorting.

            .. code-block:: python

                >>> from qiskit.circuit import QuantumCircuit, Parameter
                >>> angles = [Parameter("angle_1"), Parameter("angle_2"), Parameter("angle_10")]
                >>> circuit = QuantumCircuit(1)
                >>> circuit.u(*angles, 0)
                >>> circuit.draw()
                   ┌─────────────────────────────┐
                q: ┤ U(angle_1,angle_2,angle_10) ├
                   └─────────────────────────────┘
                >>> circuit.parameters
                ParameterView([Parameter(angle_1), Parameter(angle_10), Parameter(angle_2)])

            To respect numerical sorting, a :class:`.ParameterVector` can be used.

            .. code-block:: python

            >>> from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
            >>> x = ParameterVector("x", 12)
            >>> circuit = QuantumCircuit(1)
            >>> for x_i in x:
            ...     circuit.rx(x_i, 0)
            >>> circuit.parameters
            ParameterView([
                ParameterVectorElement(x[0]), ParameterVectorElement(x[1]),
                ParameterVectorElement(x[2]), ParameterVectorElement(x[3]),
                ..., ParameterVectorElement(x[11])
            ])


        Returns:
            The sorted :class:`.Parameter` objects in the circuit.
        """
        # parameters from gates
        if self._parameters is None:
            self._parameters = sort_parameters(self._unsorted_parameters())
        # return as parameter view, which implements the set and list interface
        return ParameterView(self._parameters)

    @property
    def num_parameters(self) -> int:
        """The number of parameter objects in the circuit."""
        return len(self._unsorted_parameters())

    def _unsorted_parameters(self) -> set[Parameter]:
        """Efficiently get all parameters in the circuit, without any sorting overhead."""
        parameters = set(self._parameter_table)
        if isinstance(self.global_phase, ParameterExpression):
            parameters.update(self.global_phase.parameters)

        return parameters

    @overload
    def assign_parameters(
        self,
        parameters: Union[Mapping[Parameter, ParameterValueType], Sequence[ParameterValueType]],
        inplace: Literal[False] = ...,
        *,
        flat_input: bool = ...,
        strict: bool = ...,
    ) -> "QuantumCircuit":
        ...

    @overload
    def assign_parameters(
        self,
        parameters: Union[Mapping[Parameter, ParameterValueType], Sequence[ParameterValueType]],
        inplace: Literal[True] = ...,
        *,
        flat_input: bool = ...,
        strict: bool = ...,
    ) -> None:
        ...

    def assign_parameters(  # pylint: disable=missing-raises-doc
        self,
        parameters: Union[Mapping[Parameter, ParameterValueType], Sequence[ParameterValueType]],
        inplace: bool = False,
        *,
        flat_input: bool = False,
        strict: bool = True,
    ) -> Optional["QuantumCircuit"]:
        """Assign parameters to new parameters or values.

        If ``parameters`` is passed as a dictionary, the keys must be :class:`.Parameter`
        instances in the current circuit. The values of the dictionary can either be numeric values
        or new parameter objects.

        If ``parameters`` is passed as a list or array, the elements are assigned to the
        current parameters in the order of :attr:`parameters` which is sorted
        alphabetically (while respecting the ordering in :class:`.ParameterVector` objects).

        The values can be assigned to the current circuit object or to a copy of it.

        Args:
            parameters: Either a dictionary or iterable specifying the new parameter values.
            inplace: If False, a copy of the circuit with the bound parameters is returned.
                If True the circuit instance itself is modified.
            flat_input: If ``True`` and ``parameters`` is a mapping type, it is assumed to be
                exactly a mapping of ``{parameter: value}``.  By default (``False``), the mapping
                may also contain :class:`.ParameterVector` keys that point to a corresponding
                sequence of values, and these will be unrolled during the mapping.
            strict: If ``False``, any parameters given in the mapping that are not used in the
                circuit will be ignored.  If ``True`` (the default), an error will be raised
                indicating a logic error.

        Raises:
            CircuitError: If parameters is a dict and contains parameters not present in the
                circuit.
            ValueError: If parameters is a list/array and the length mismatches the number of free
                parameters in the circuit.

        Returns:
            A copy of the circuit with bound parameters if ``inplace`` is False, otherwise None.

        Examples:

            Create a parameterized circuit and assign the parameters in-place.

            .. plot::
               :include-source:

               from qiskit.circuit import QuantumCircuit, Parameter

               circuit = QuantumCircuit(2)
               params = [Parameter('A'), Parameter('B'), Parameter('C')]
               circuit.ry(params[0], 0)
               circuit.crx(params[1], 0, 1)
               circuit.draw('mpl')
               circuit.assign_parameters({params[0]: params[2]}, inplace=True)
               circuit.draw('mpl')

            Bind the values out-of-place by list and get a copy of the original circuit.

            .. plot::
               :include-source:

               from qiskit.circuit import QuantumCircuit, ParameterVector

               circuit = QuantumCircuit(2)
               params = ParameterVector('P', 2)
               circuit.ry(params[0], 0)
               circuit.crx(params[1], 0, 1)

               bound_circuit = circuit.assign_parameters([1, 2])
               bound_circuit.draw('mpl')

               circuit.draw('mpl')

        """
        if inplace:
            target = self
        else:
            target = self.copy()
            target._increment_instances()
            target._name_update()

        # Normalise the inputs into simple abstract interfaces, so we've dispatched the "iteration"
        # logic in one place at the start of the function.  This lets us do things like calculate
        # and cache expensive properties for (e.g.) the sequence format only if they're used; for
        # many large, close-to-hardware circuits, we won't need the extra handling for
        # `global_phase` or recursive definition binding.
        #
        # During normalisation, be sure to reference 'parameters' and related things from 'self' not
        # 'target' so we can take advantage of any caching we might be doing.
        if isinstance(parameters, dict):
            raw_mapping = parameters if flat_input else self._unroll_param_dict(parameters)
            our_parameters = self._unsorted_parameters()
            if strict and (extras := raw_mapping.keys() - our_parameters):
                raise CircuitError(
                    f"Cannot bind parameters ({', '.join(str(x) for x in extras)}) not present in"
                    " the circuit."
                )
            parameter_binds = _ParameterBindsDict(raw_mapping, our_parameters)
        else:
            our_parameters = self.parameters
            if len(parameters) != len(our_parameters):
                raise ValueError(
                    "Mismatching number of values and parameters. For partial binding "
                    "please pass a dictionary of {parameter: value} pairs."
                )
            parameter_binds = _ParameterBindsSequence(our_parameters, parameters)

        # Clear out the parameter table for the relevant entries, since we'll be binding those.
        # Any new references to parameters are reinserted as part of the bind.
        target._parameters = None
        # This is deliberately eager, because we want the side effect of clearing the table.
        all_references = [
            (parameter, value, target._parameter_table.pop(parameter, ()))
            for parameter, value in parameter_binds.items()
        ]
        seen_operations = {}
        # The meat of the actual binding for regular operations.
        for to_bind, bound_value, references in all_references:
            update_parameters = (
                tuple(bound_value.parameters)
                if isinstance(bound_value, ParameterExpression)
                else ()
            )
            for operation, index in references:
                seen_operations[id(operation)] = operation
                assignee = operation.params[index]
                if isinstance(assignee, ParameterExpression):
                    new_parameter = assignee.assign(to_bind, bound_value)
                    for parameter in update_parameters:
                        if parameter not in target._parameter_table:
                            target._parameter_table[parameter] = ParameterReferences(())
                        target._parameter_table[parameter].add((operation, index))
                    if not new_parameter.parameters:
                        if new_parameter.is_real():
                            new_parameter = (
                                int(new_parameter)
                                if new_parameter._symbol_expr.is_integer
                                else float(new_parameter)
                            )
                        else:
                            new_parameter = complex(new_parameter)
                        new_parameter = operation.validate_parameter(new_parameter)
                elif isinstance(assignee, QuantumCircuit):
                    new_parameter = assignee.assign_parameters(
                        {to_bind: bound_value}, inplace=False, flat_input=True
                    )
                else:
                    raise RuntimeError(  # pragma: no cover
                        f"Saw an unknown type during symbolic binding: {assignee}."
                        " This may indicate an internal logic error in symbol tracking."
                    )
                operation.params[index] = new_parameter

        # After we've been through everything at the top level, make a single visit to each
        # operation we've seen, rebinding its definition if necessary.
        for operation in seen_operations.values():
            if (
                definition := getattr(operation, "_definition", None)
            ) is not None and definition.num_parameters:
                definition.assign_parameters(
                    parameter_binds.mapping, inplace=True, flat_input=True, strict=False
                )

        if isinstance(target.global_phase, ParameterExpression):
            new_phase = target.global_phase
            for parameter in new_phase.parameters & parameter_binds.mapping.keys():
                new_phase = new_phase.assign(parameter, parameter_binds.mapping[parameter])
            target.global_phase = new_phase

        # Finally, assign the parameters inside any of the calibrations.  We don't track these in
        # the `ParameterTable`, so we manually reconstruct things.
        def map_calibration(qubits, parameters, schedule):
            modified = False
            new_parameters = list(parameters)
            for i, parameter in enumerate(new_parameters):
                if not isinstance(parameter, ParameterExpression):
                    continue
                if not (contained := parameter.parameters & parameter_binds.mapping.keys()):
                    continue
                for to_bind in contained:
                    parameter = parameter.assign(to_bind, parameter_binds.mapping[to_bind])
                if not parameter.parameters:
                    parameter = (
                        int(parameter) if parameter._symbol_expr.is_integer else float(parameter)
                    )
                new_parameters[i] = parameter
                modified = True
            if modified:
                schedule.assign_parameters(parameter_binds.mapping)
            return (qubits, tuple(new_parameters)), schedule

        target._calibrations = defaultdict(
            dict,
            (
                (
                    gate,
                    dict(
                        map_calibration(qubits, parameters, schedule)
                        for (qubits, parameters), schedule in calibrations.items()
                    ),
                )
                for gate, calibrations in target._calibrations.items()
            ),
        )
        return None if inplace else target

    @staticmethod
    def _unroll_param_dict(
        parameter_binds: Mapping[Parameter, ParameterValueType]
    ) -> Mapping[Parameter, ParameterValueType]:
        out = {}
        for parameter, value in parameter_binds.items():
            if isinstance(parameter, ParameterVector):
                if len(parameter) != len(value):
                    raise CircuitError(
                        f"Parameter vector '{parameter.name}' has length {len(parameter)},"
                        f" but was assigned to {len(value)} values."
                    )
                out.update(zip(parameter, value))
            else:
                out[parameter] = value
        return out

    def bind_parameters(
        self, values: Union[Mapping[Parameter, float], Sequence[float]]
    ) -> "QuantumCircuit":
        """Assign numeric parameters to values yielding a new circuit.

        If the values are given as list or array they are bound to the circuit in the order
        of :attr:`parameters` (see the docstring for more details).

        To assign new Parameter objects or bind the values in-place, without yielding a new
        circuit, use the :meth:`assign_parameters` method.

        Args:
            values: ``{parameter: value, ...}`` or ``[value1, value2, ...]``

        Raises:
            CircuitError: If values is a dict and contains parameters not present in the circuit.
            TypeError: If values contains a ParameterExpression.

        Returns:
            Copy of self with assignment substitution.
        """
        if isinstance(values, dict):
            if any(isinstance(value, ParameterExpression) for value in values.values()):
                raise TypeError(
                    "Found ParameterExpression in values; use assign_parameters() instead."
                )
            return self.assign_parameters(values)
        else:
            if any(isinstance(value, ParameterExpression) for value in values):
                raise TypeError(
                    "Found ParameterExpression in values; use assign_parameters() instead."
                )
            return self.assign_parameters(values)

    def barrier(self, *qargs: QubitSpecifier, label=None) -> InstructionSet:
        """Apply :class:`~.library.Barrier`. If ``qargs`` is empty, applies to all qubits
        in the circuit.

        Args:
            qargs (QubitSpecifier): Specification for one or more qubit arguments.
            label (str): The string label of the barrier.

        Returns:
            qiskit.circuit.InstructionSet: handle to the added instructions.
        """
        from .barrier import Barrier

        qubits: list[QubitSpecifier] = []

        if not qargs:  # None
            qubits.extend(self.qubits)

        for qarg in qargs:
            if isinstance(qarg, QuantumRegister):
                qubits.extend([qarg[j] for j in range(qarg.size)])
            elif isinstance(qarg, list):
                qubits.extend(qarg)
            elif isinstance(qarg, range):
                qubits.extend(list(qarg))
            elif isinstance(qarg, slice):
                qubits.extend(self.qubits[qarg])
            else:
                qubits.append(qarg)

        return self.append(Barrier(len(qubits), label=label), qubits, [])

    def delay(
        self,
        duration: ParameterValueType,
        qarg: QubitSpecifier | None = None,
        unit: str = "dt",
    ) -> InstructionSet:
        """Apply :class:`~.circuit.Delay`. If qarg is ``None``, applies to all qubits.
        When applying to multiple qubits, delays with the same duration will be created.

        Args:
            duration (int or float or ParameterExpression): duration of the delay.
            qarg (Object): qubit argument to apply this delay.
            unit (str): unit of the duration. Supported units: ``'s'``, ``'ms'``, ``'us'``,
                ``'ns'``, ``'ps'``, and ``'dt'``. Default is ``'dt'``, i.e. integer time unit
                depending on the target backend.

        Returns:
            qiskit.circuit.InstructionSet: handle to the added instructions.

        Raises:
            CircuitError: if arguments have bad format.
        """
        qubits: list[QubitSpecifier] = []
        if qarg is None:  # -> apply delays to all qubits
            for q in self.qubits:
                qubits.append(q)
        else:
            if isinstance(qarg, QuantumRegister):
                qubits.extend([qarg[j] for j in range(qarg.size)])
            elif isinstance(qarg, list):
                qubits.extend(qarg)
            elif isinstance(qarg, (range, tuple)):
                qubits.extend(list(qarg))
            elif isinstance(qarg, slice):
                qubits.extend(self.qubits[qarg])
            else:
                qubits.append(qarg)

        instructions = InstructionSet(resource_requester=self._resolve_classical_resource)
        for q in qubits:
            inst: tuple[
                Instruction, Sequence[QubitSpecifier] | None, Sequence[ClbitSpecifier] | None
            ] = (Delay(duration, unit), [q], [])
            self.append(*inst)
            instructions.add(*inst)
        return instructions

    def h(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.HGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.h import HGate

        return self.append(HGate(), [qubit], [])

    def ch(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CHGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.h import CHGate

        return self.append(
            CHGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def i(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.IGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.i import IGate

        return self.append(IGate(), [qubit], [])

    def id(self, qubit: QubitSpecifier) -> InstructionSet:  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.IGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.

        See Also:
            QuantumCircuit.i: the same function.
        """
        return self.i(qubit)

    def ms(self, theta: ParameterValueType, qubits: Sequence[QubitSpecifier]) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.MSGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            qubits: The qubits to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        # pylint: disable=cyclic-import
        from .library.generalized_gates.gms import MSGate

        return self.append(MSGate(len(qubits), theta), qubits)

    def p(self, theta: ParameterValueType, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.PhaseGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: THe angle of the rotation.
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.p import PhaseGate

        return self.append(PhaseGate(theta), [qubit], [])

    def cp(
        self,
        theta: ParameterValueType,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CPhaseGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.p import CPhaseGate

        return self.append(
            CPhaseGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def mcp(
        self,
        lam: ParameterValueType,
        control_qubits: Sequence[QubitSpecifier],
        target_qubit: QubitSpecifier,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.MCPhaseGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            lam: The angle of the rotation.
            control_qubits: The qubits used as the controls.
            target_qubit: The qubit(s) targeted by the gate.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.p import MCPhaseGate

        num_ctrl_qubits = len(control_qubits)
        return self.append(
            MCPhaseGate(lam, num_ctrl_qubits), control_qubits[:] + [target_qubit], []
        )

    def r(
        self, theta: ParameterValueType, phi: ParameterValueType, qubit: QubitSpecifier
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            phi: The angle of the axis of rotation in the x-y plane.
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.r import RGate

        return self.append(RGate(theta, phi), [qubit], [])

    def rv(
        self,
        vx: ParameterValueType,
        vy: ParameterValueType,
        vz: ParameterValueType,
        qubit: QubitSpecifier,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RVGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Rotation around an arbitrary rotation axis :math:`v`, where :math:`|v|` is the angle of
        rotation in radians.

        Args:
            vx: x-component of the rotation axis.
            vy: y-component of the rotation axis.
            vz: z-component of the rotation axis.
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.generalized_gates.rv import RVGate

        return self.append(RVGate(vx, vy, vz), [qubit], [])

    def rccx(
        self,
        control_qubit1: QubitSpecifier,
        control_qubit2: QubitSpecifier,
        target_qubit: QubitSpecifier,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RCCXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit1: The qubit(s) used as the first control.
            control_qubit2: The qubit(s) used as the second control.
            target_qubit: The qubit(s) targeted by the gate.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.x import RCCXGate

        return self.append(RCCXGate(), [control_qubit1, control_qubit2, target_qubit], [])

    def rcccx(
        self,
        control_qubit1: QubitSpecifier,
        control_qubit2: QubitSpecifier,
        control_qubit3: QubitSpecifier,
        target_qubit: QubitSpecifier,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RC3XGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit1: The qubit(s) used as the first control.
            control_qubit2: The qubit(s) used as the second control.
            control_qubit3: The qubit(s) used as the third control.
            target_qubit: The qubit(s) targeted by the gate.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.x import RC3XGate

        return self.append(
            RC3XGate(), [control_qubit1, control_qubit2, control_qubit3, target_qubit], []
        )

    def rx(
        self, theta: ParameterValueType, qubit: QubitSpecifier, label: str | None = None
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The rotation angle of the gate.
            qubit: The qubit(s) to apply the gate to.
            label: The string label of the gate in the circuit.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.rx import RXGate

        return self.append(RXGate(theta, label=label), [qubit], [])

    def crx(
        self,
        theta: ParameterValueType,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CRXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.rx import CRXGate

        return self.append(
            CRXGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def rxx(
        self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RXXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            qubit1: The qubit(s) to apply the gate to.
            qubit2: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.rxx import RXXGate

        return self.append(RXXGate(theta), [qubit1, qubit2], [])

    def ry(
        self, theta: ParameterValueType, qubit: QubitSpecifier, label: str | None = None
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RYGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The rotation angle of the gate.
            qubit: The qubit(s) to apply the gate to.
            label: The string label of the gate in the circuit.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.ry import RYGate

        return self.append(RYGate(theta, label=label), [qubit], [])

    def cry(
        self,
        theta: ParameterValueType,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CRYGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.ry import CRYGate

        return self.append(
            CRYGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def ryy(
        self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RYYGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The rotation angle of the gate.
            qubit1: The qubit(s) to apply the gate to.
            qubit2: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.ryy import RYYGate

        return self.append(RYYGate(theta), [qubit1, qubit2], [])

    def rz(self, phi: ParameterValueType, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RZGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            phi: The rotation angle of the gate.
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.rz import RZGate

        return self.append(RZGate(phi), [qubit], [])

    def crz(
        self,
        theta: ParameterValueType,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CRZGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.rz import CRZGate

        return self.append(
            CRZGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def rzx(
        self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RZXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The rotation angle of the gate.
            qubit1: The qubit(s) to apply the gate to.
            qubit2: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.rzx import RZXGate

        return self.append(RZXGate(theta), [qubit1, qubit2], [])

    def rzz(
        self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RZZGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The rotation angle of the gate.
            qubit1: The qubit(s) to apply the gate to.
            qubit2: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.rzz import RZZGate

        return self.append(RZZGate(theta), [qubit1, qubit2], [])

    def ecr(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.ECRGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit1, qubit2: The qubits to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.ecr import ECRGate

        return self.append(ECRGate(), [qubit1, qubit2], [])

    def s(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.SGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.s import SGate

        return self.append(SGate(), [qubit], [])

    def sdg(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.SdgGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.s import SdgGate

        return self.append(SdgGate(), [qubit], [])

    def cs(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CSGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.s import CSGate

        return self.append(
            CSGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
        )

    def csdg(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CSdgGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.s import CSdgGate

        return self.append(
            CSdgGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
        )

    def swap(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.SwapGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit1, qubit2: The qubits to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.swap import SwapGate

        return self.append(SwapGate(), [qubit1, qubit2], [])

    def iswap(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.iSwapGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit1, qubit2: The qubits to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.iswap import iSwapGate

        return self.append(iSwapGate(), [qubit1, qubit2], [])

    def cswap(
        self,
        control_qubit: QubitSpecifier,
        target_qubit1: QubitSpecifier,
        target_qubit2: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CSwapGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit1: The qubit(s) targeted by the gate.
            target_qubit2: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. ``'1'``).  Defaults to controlling
                on the ``'1'`` state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.swap import CSwapGate

        return self.append(
            CSwapGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit1, target_qubit2],
            [],
        )

    def fredkin(
        self,
        control_qubit: QubitSpecifier,
        target_qubit1: QubitSpecifier,
        target_qubit2: QubitSpecifier,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CSwapGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit1: The qubit(s) targeted by the gate.
            target_qubit2: The qubit(s) targeted by the gate.

        Returns:
            A handle to the instructions created.

        See Also:
            QuantumCircuit.cswap: the same function with a different name.
        """
        return self.cswap(control_qubit, target_qubit1, target_qubit2)

    def sx(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.SXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.sx import SXGate

        return self.append(SXGate(), [qubit], [])

    def sxdg(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.SXdgGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.sx import SXdgGate

        return self.append(SXdgGate(), [qubit], [])

    def csx(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CSXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.sx import CSXGate

        return self.append(
            CSXGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
        )

    def t(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.TGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.t import TGate

        return self.append(TGate(), [qubit], [])

    def tdg(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.TdgGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.t import TdgGate

        return self.append(TdgGate(), [qubit], [])

    def u(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        qubit: QubitSpecifier,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.UGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The :math:`\theta` rotation angle of the gate.
            phi: The :math:`\phi` rotation angle of the gate.
            lam: The :math:`\lambda` rotation angle of the gate.
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.u import UGate

        return self.append(UGate(theta, phi, lam), [qubit], [])

    def cu(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        gamma: ParameterValueType,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CUGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The :math:`\theta` rotation angle of the gate.
            phi: The :math:`\phi` rotation angle of the gate.
            lam: The :math:`\lambda` rotation angle of the gate.
            gamma: The global phase applied of the U gate, if applied.
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.u import CUGate

        return self.append(
            CUGate(theta, phi, lam, gamma, label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
        )

    def x(self, qubit: QubitSpecifier, label: str | None = None) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.XGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.
            label: The string label of the gate in the circuit.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.x import XGate

        return self.append(XGate(label=label), [qubit], [])

    def cx(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """

        from .library.standard_gates.x import CXGate

        return self.append(
            CXGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def cnot(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.

        See Also:
            QuantumCircuit.cx: the same function with a different name.
        """
        return self.cx(control_qubit, target_qubit, label, ctrl_state)

    def dcx(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.DCXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit1: The qubit(s) to apply the gate to.
            qubit2: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.dcx import DCXGate

        return self.append(DCXGate(), [qubit1, qubit2], [])

    def ccx(
        self,
        control_qubit1: QubitSpecifier,
        control_qubit2: QubitSpecifier,
        target_qubit: QubitSpecifier,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CCXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit1: The qubit(s) used as the first control.
            control_qubit2: The qubit(s) used as the second control.
            target_qubit: The qubit(s) targeted by the gate.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.x import CCXGate

        return self.append(
            CCXGate(ctrl_state=ctrl_state),
            [control_qubit1, control_qubit2, target_qubit],
            [],
        )

    def toffoli(
        self,
        control_qubit1: QubitSpecifier,
        control_qubit2: QubitSpecifier,
        target_qubit: QubitSpecifier,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CCXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit1: The qubit(s) used as the first control.
            control_qubit2: The qubit(s) used as the second control.
            target_qubit: The qubit(s) targeted by the gate.

        Returns:
            A handle to the instructions created.

        See Also:
            QuantumCircuit.ccx: the same gate with a different name.
        """
        return self.ccx(control_qubit1, control_qubit2, target_qubit)

    def mcx(
        self,
        control_qubits: Sequence[QubitSpecifier],
        target_qubit: QubitSpecifier,
        ancilla_qubits: QubitSpecifier | Sequence[QubitSpecifier] | None = None,
        mode: str = "noancilla",
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.MCXGate`.

        The multi-cX gate can be implemented using different techniques, which use different numbers
        of ancilla qubits and have varying circuit depth. These modes are:

        - ``'noancilla'``: Requires 0 ancilla qubits.
        - ``'recursion'``: Requires 1 ancilla qubit if more than 4 controls are used, otherwise 0.
        - ``'v-chain'``: Requires 2 less ancillas than the number of control qubits.
        - ``'v-chain-dirty'``: Same as for the clean ancillas (but the circuit will be longer).

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubits: The qubits used as the controls.
            target_qubit: The qubit(s) targeted by the gate.
            ancilla_qubits: The qubits used as the ancillae, if the mode requires them.
            mode: The choice of mode, explained further above.

        Returns:
            A handle to the instructions created.

        Raises:
            ValueError: if the given mode is not known, or if too few ancilla qubits are passed.
            AttributeError: if no ancilla qubits are passed, but some are needed.
        """
        from .library.standard_gates.x import MCXGrayCode, MCXRecursive, MCXVChain

        num_ctrl_qubits = len(control_qubits)

        available_implementations = {
            "noancilla": MCXGrayCode(num_ctrl_qubits),
            "recursion": MCXRecursive(num_ctrl_qubits),
            "v-chain": MCXVChain(num_ctrl_qubits, False),
            "v-chain-dirty": MCXVChain(num_ctrl_qubits, dirty_ancillas=True),
            # outdated, previous names
            "advanced": MCXRecursive(num_ctrl_qubits),
            "basic": MCXVChain(num_ctrl_qubits, dirty_ancillas=False),
            "basic-dirty-ancilla": MCXVChain(num_ctrl_qubits, dirty_ancillas=True),
        }

        # check ancilla input
        if ancilla_qubits:
            _ = self.qbit_argument_conversion(ancilla_qubits)

        try:
            gate = available_implementations[mode]
        except KeyError as ex:
            all_modes = list(available_implementations.keys())
            raise ValueError(
                f"Unsupported mode ({mode}) selected, choose one of {all_modes}"
            ) from ex

        if hasattr(gate, "num_ancilla_qubits") and gate.num_ancilla_qubits > 0:
            required = gate.num_ancilla_qubits
            if ancilla_qubits is None:
                raise AttributeError(f"No ancillas provided, but {required} are needed!")

            # convert ancilla qubits to a list if they were passed as int or qubit
            if not hasattr(ancilla_qubits, "__len__"):
                ancilla_qubits = [ancilla_qubits]

            if len(ancilla_qubits) < required:
                actually = len(ancilla_qubits)
                raise ValueError(f"At least {required} ancillas required, but {actually} given.")
            # size down if too many ancillas were provided
            ancilla_qubits = ancilla_qubits[:required]
        else:
            ancilla_qubits = []

        return self.append(gate, control_qubits[:] + [target_qubit] + ancilla_qubits[:], [])

    def mct(
        self,
        control_qubits: Sequence[QubitSpecifier],
        target_qubit: QubitSpecifier,
        ancilla_qubits: QubitSpecifier | Sequence[QubitSpecifier] | None = None,
        mode: str = "noancilla",
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.MCXGate`.

        The multi-cX gate can be implemented using different techniques, which use different numbers
        of ancilla qubits and have varying circuit depth. These modes are:

        - ``'noancilla'``: Requires 0 ancilla qubits.
        - ``'recursion'``: Requires 1 ancilla qubit if more than 4 controls are used, otherwise 0.
        - ``'v-chain'``: Requires 2 less ancillas than the number of control qubits.
        - ``'v-chain-dirty'``: Same as for the clean ancillas (but the circuit will be longer).

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubits: The qubits used as the controls.
            target_qubit: The qubit(s) targeted by the gate.
            ancilla_qubits: The qubits used as the ancillae, if the mode requires them.
            mode: The choice of mode, explained further above.

        Returns:
            A handle to the instructions created.

        Raises:
            ValueError: if the given mode is not known, or if too few ancilla qubits are passed.
            AttributeError: if no ancilla qubits are passed, but some are needed.

        See Also:
            QuantumCircuit.mcx: the same gate with a different name.
        """
        return self.mcx(control_qubits, target_qubit, ancilla_qubits, mode)

    def y(self, qubit: QubitSpecifier) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.YGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.y import YGate

        return self.append(YGate(), [qubit], [])

    def cy(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CYGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the controls.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.y import CYGate

        return self.append(
            CYGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def z(self, qubit: QubitSpecifier) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.ZGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.z import ZGate

        return self.append(ZGate(), [qubit], [])

    def cz(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CZGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the controls.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.z import CZGate

        return self.append(
            CZGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def ccz(
        self,
        control_qubit1: QubitSpecifier,
        control_qubit2: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CCZGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit1: The qubit(s) used as the first control.
            control_qubit2: The qubit(s) used as the second control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '10').  Defaults to controlling
                on the '11' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.z import CCZGate

        return self.append(
            CCZGate(label=label, ctrl_state=ctrl_state),
            [control_qubit1, control_qubit2, target_qubit],
            [],
        )

    def pauli(
        self,
        pauli_string: str,
        qubits: Sequence[QubitSpecifier],
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.PauliGate`.

        Args:
            pauli_string: A string representing the Pauli operator to apply, e.g. 'XX'.
            qubits: The qubits to apply this gate to.

        Returns:
            A handle to the instructions created.
        """
        from qiskit.circuit.library.generalized_gates.pauli import PauliGate

        return self.append(PauliGate(pauli_string), qubits, [])

    def _push_scope(
        self,
        qubits: Iterable[Qubit] = (),
        clbits: Iterable[Clbit] = (),
        registers: Iterable[Register] = (),
        allow_jumps: bool = True,
        forbidden_message: Optional[str] = None,
    ):
        """Add a scope for collecting instructions into this circuit.

        This should only be done by the control-flow context managers, which will handle cleaning up
        after themselves at the end as well.

        Args:
            qubits: Any qubits that this scope should automatically use.
            clbits: Any clbits that this scope should automatically use.
            allow_jumps: Whether this scope allows jumps to be used within it.
            forbidden_message: If given, all attempts to add instructions to this scope will raise a
                :exc:`.CircuitError` with this message.
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.controlflow.builder import ControlFlowBuilderBlock

        # Chain resource requests so things like registers added to inner scopes via conditions are
        # requested in the outer scope as well.
        if self._control_flow_scopes:
            resource_requester = self._control_flow_scopes[-1].request_classical_resource
        else:
            resource_requester = self._resolve_classical_resource

        self._control_flow_scopes.append(
            ControlFlowBuilderBlock(
                qubits,
                clbits,
                resource_requester=resource_requester,
                registers=registers,
                allow_jumps=allow_jumps,
                forbidden_message=forbidden_message,
            )
        )

    def _pop_scope(self) -> "qiskit.circuit.controlflow.builder.ControlFlowBuilderBlock":
        """Finish a scope used in the control-flow builder interface, and return it to the caller.

        This should only be done by the control-flow context managers, since they naturally
        synchronise the creation and deletion of stack elements."""
        return self._control_flow_scopes.pop()

    def _peek_previous_instruction_in_scope(self) -> CircuitInstruction:
        """Return the instruction 3-tuple of the most recent instruction in the current scope, even
        if that scope is currently under construction.

        This function is only intended for use by the control-flow ``if``-statement builders, which
        may need to modify a previous instruction."""
        if self._control_flow_scopes:
            return self._control_flow_scopes[-1].peek()
        if not self._data:
            raise CircuitError("This circuit contains no instructions.")
        return self._data[-1]

    def _pop_previous_instruction_in_scope(self) -> CircuitInstruction:
        """Return the instruction 3-tuple of the most recent instruction in the current scope, even
        if that scope is currently under construction, and remove it from that scope.

        This function is only intended for use by the control-flow ``if``-statement builders, which
        may need to replace a previous instruction with another.
        """
        if self._control_flow_scopes:
            return self._control_flow_scopes[-1].pop()
        if not self._data:
            raise CircuitError("This circuit contains no instructions.")
        instruction = self._data.pop()
        if isinstance(instruction.operation, Instruction):
            self._update_parameter_table_on_instruction_removal(instruction)
        return instruction

    def _update_parameter_table_on_instruction_removal(self, instruction: CircuitInstruction):
        """Update the :obj:`.ParameterTable` of this circuit given that an instance of the given
        ``instruction`` has just been removed from the circuit.

        .. note::

            This does not account for the possibility for the same instruction instance being added
            more than once to the circuit.  At the time of writing (2021-11-17, main commit 271a82f)
            there is a defensive ``deepcopy`` of parameterised instructions inside
            :meth:`.QuantumCircuit.append`, so this should be safe.  Trying to account for it would
            involve adding a potentially quadratic-scaling loop to check each entry in ``data``.
        """
        atomic_parameters: list[tuple[Parameter, int]] = []
        for index, parameter in enumerate(instruction.operation.params):
            if isinstance(parameter, (ParameterExpression, QuantumCircuit)):
                atomic_parameters.extend((p, index) for p in parameter.parameters)
        for atomic_parameter, index in atomic_parameters:
            new_entries = self._parameter_table[atomic_parameter].copy()
            new_entries.discard((instruction.operation, index))
            if not new_entries:
                del self._parameter_table[atomic_parameter]
                # Invalidate cache.
                self._parameters = None
            else:
                self._parameter_table[atomic_parameter] = new_entries

    @typing.overload
    def while_loop(
        self,
        condition: tuple[ClassicalRegister | Clbit, int] | expr.Expr,
        body: None,
        qubits: None,
        clbits: None,
        *,
        label: str | None,
    ) -> "qiskit.circuit.controlflow.while_loop.WhileLoopContext":
        ...

    @typing.overload
    def while_loop(
        self,
        condition: tuple[ClassicalRegister | Clbit, int] | expr.Expr,
        body: "QuantumCircuit",
        qubits: Sequence[QubitSpecifier],
        clbits: Sequence[ClbitSpecifier],
        *,
        label: str | None,
    ) -> InstructionSet:
        ...

    def while_loop(self, condition, body=None, qubits=None, clbits=None, *, label=None):
        """Create a ``while`` loop on this circuit.

        There are two forms for calling this function.  If called with all its arguments (with the
        possible exception of ``label``), it will create a
        :obj:`~qiskit.circuit.controlflow.WhileLoopOp` with the given ``body``.  If ``body`` (and
        ``qubits`` and ``clbits``) are *not* passed, then this acts as a context manager, which
        will automatically build a :obj:`~qiskit.circuit.controlflow.WhileLoopOp` when the scope
        finishes.  In this form, you do not need to keep track of the qubits or clbits you are
        using, because the scope will handle it for you.

        Example usage::

            from qiskit.circuit import QuantumCircuit, Clbit, Qubit
            bits = [Qubit(), Qubit(), Clbit()]
            qc = QuantumCircuit(bits)

            with qc.while_loop((bits[2], 0)):
                qc.h(0)
                qc.cx(0, 1)
                qc.measure(0, 0)

        Args:
            condition (Tuple[Union[ClassicalRegister, Clbit], int]): An equality condition to be
                checked prior to executing ``body``. The left-hand side of the condition must be a
                :obj:`~ClassicalRegister` or a :obj:`~Clbit`, and the right-hand side must be an
                integer or boolean.
            body (Optional[QuantumCircuit]): The loop body to be repeatedly executed.  Omit this to
                use the context-manager mode.
            qubits (Optional[Sequence[Qubit]]): The circuit qubits over which the loop body should
                be run.  Omit this to use the context-manager mode.
            clbits (Optional[Sequence[Clbit]]): The circuit clbits over which the loop body should
                be run.  Omit this to use the context-manager mode.
            label (Optional[str]): The string label of the instruction in the circuit.

        Returns:
            InstructionSet or WhileLoopContext: If used in context-manager mode, then this should be
            used as a ``with`` resource, which will infer the block content and operands on exit.
            If the full form is used, then this returns a handle to the instructions created.

        Raises:
            CircuitError: if an incorrect calling convention is used.
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.controlflow.while_loop import WhileLoopOp, WhileLoopContext

        if isinstance(condition, expr.Expr):
            condition = self._validate_expr(condition)
        else:
            condition = (self._resolve_classical_resource(condition[0]), condition[1])

        if body is None:
            if qubits is not None or clbits is not None:
                raise CircuitError(
                    "When using 'while_loop' as a context manager,"
                    " you cannot pass qubits or clbits."
                )
            return WhileLoopContext(self, condition, label=label)
        elif qubits is None or clbits is None:
            raise CircuitError(
                "When using 'while_loop' with a body, you must pass qubits and clbits."
            )

        return self.append(WhileLoopOp(condition, body, label), qubits, clbits)

    @typing.overload
    def for_loop(
        self,
        indexset: Iterable[int],
        loop_parameter: Parameter | None,
        body: None,
        qubits: None,
        clbits: None,
        *,
        label: str | None,
    ) -> "qiskit.circuit.controlflow.for_loop.ForLoopContext":
        ...

    @typing.overload
    def for_loop(
        self,
        indexset: Iterable[int],
        loop_parameter: Union[Parameter, None],
        body: "QuantumCircuit",
        qubits: Sequence[QubitSpecifier],
        clbits: Sequence[ClbitSpecifier],
        *,
        label: str | None,
    ) -> InstructionSet:
        ...

    def for_loop(
        self, indexset, loop_parameter=None, body=None, qubits=None, clbits=None, *, label=None
    ):
        """Create a ``for`` loop on this circuit.

        There are two forms for calling this function.  If called with all its arguments (with the
        possible exception of ``label``), it will create a
        :class:`~qiskit.circuit.ForLoopOp` with the given ``body``.  If ``body`` (and
        ``qubits`` and ``clbits``) are *not* passed, then this acts as a context manager, which,
        when entered, provides a loop variable (unless one is given, in which case it will be
        reused) and will automatically build a :class:`~qiskit.circuit.ForLoopOp` when the
        scope finishes.  In this form, you do not need to keep track of the qubits or clbits you are
        using, because the scope will handle it for you.

        For example::

            from qiskit import QuantumCircuit
            qc = QuantumCircuit(2, 1)

            with qc.for_loop(range(5)) as i:
                qc.h(0)
                qc.cx(0, 1)
                qc.measure(0, 0)
                qc.break_loop().c_if(0, True)

        Args:
            indexset (Iterable[int]): A collection of integers to loop over.  Always necessary.
            loop_parameter (Optional[Parameter]): The parameter used within ``body`` to which
                the values from ``indexset`` will be assigned.  In the context-manager form, if this
                argument is not supplied, then a loop parameter will be allocated for you and
                returned as the value of the ``with`` statement.  This will only be bound into the
                circuit if it is used within the body.

                If this argument is ``None`` in the manual form of this method, ``body`` will be
                repeated once for each of the items in ``indexset`` but their values will be
                ignored.
            body (Optional[QuantumCircuit]): The loop body to be repeatedly executed.  Omit this to
                use the context-manager mode.
            qubits (Optional[Sequence[QubitSpecifier]]): The circuit qubits over which the loop body
                should be run.  Omit this to use the context-manager mode.
            clbits (Optional[Sequence[ClbitSpecifier]]): The circuit clbits over which the loop body
                should be run.  Omit this to use the context-manager mode.
            label (Optional[str]): The string label of the instruction in the circuit.

        Returns:
            InstructionSet or ForLoopContext: depending on the call signature, either a context
            manager for creating the for loop (it will automatically be added to the circuit at the
            end of the block), or an :obj:`~InstructionSet` handle to the appended loop operation.

        Raises:
            CircuitError: if an incorrect calling convention is used.
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.controlflow.for_loop import ForLoopOp, ForLoopContext

        if body is None:
            if qubits is not None or clbits is not None:
                raise CircuitError(
                    "When using 'for_loop' as a context manager, you cannot pass qubits or clbits."
                )
            return ForLoopContext(self, indexset, loop_parameter, label=label)
        elif qubits is None or clbits is None:
            raise CircuitError(
                "When using 'for_loop' with a body, you must pass qubits and clbits."
            )

        return self.append(ForLoopOp(indexset, loop_parameter, body, label), qubits, clbits)

    @typing.overload
    def if_test(
        self,
        condition: tuple[ClassicalRegister | Clbit, int],
        true_body: None,
        qubits: None,
        clbits: None,
        *,
        label: str | None,
    ) -> "qiskit.circuit.controlflow.if_else.IfContext":
        ...

    @typing.overload
    def if_test(
        self,
        condition: tuple[ClassicalRegister | Clbit, int],
        true_body: "QuantumCircuit",
        qubits: Sequence[QubitSpecifier],
        clbits: Sequence[ClbitSpecifier],
        *,
        label: str | None = None,
    ) -> InstructionSet:
        ...

    def if_test(
        self,
        condition,
        true_body=None,
        qubits=None,
        clbits=None,
        *,
        label=None,
    ):
        """Create an ``if`` statement on this circuit.

        There are two forms for calling this function.  If called with all its arguments (with the
        possible exception of ``label``), it will create a
        :obj:`~qiskit.circuit.IfElseOp` with the given ``true_body``, and there will be
        no branch for the ``false`` condition (see also the :meth:`.if_else` method).  However, if
        ``true_body`` (and ``qubits`` and ``clbits``) are *not* passed, then this acts as a context
        manager, which can be used to build ``if`` statements.  The return value of the ``with``
        statement is a chainable context manager, which can be used to create subsequent ``else``
        blocks.  In this form, you do not need to keep track of the qubits or clbits you are using,
        because the scope will handle it for you.

        For example::

            from qiskit.circuit import QuantumCircuit, Qubit, Clbit
            bits = [Qubit(), Qubit(), Qubit(), Clbit(), Clbit()]
            qc = QuantumCircuit(bits)

            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 1)

            with qc.if_test((bits[3], 0)) as else_:
                qc.x(2)
            with else_:
                qc.h(2)
                qc.z(2)

        Args:
            condition (Tuple[Union[ClassicalRegister, Clbit], int]): A condition to be evaluated at
                circuit runtime which, if true, will trigger the evaluation of ``true_body``. Can be
                specified as either a tuple of a ``ClassicalRegister`` to be tested for equality
                with a given ``int``, or as a tuple of a ``Clbit`` to be compared to either a
                ``bool`` or an ``int``.
            true_body (Optional[QuantumCircuit]): The circuit body to be run if ``condition`` is
                true.
            qubits (Optional[Sequence[QubitSpecifier]]): The circuit qubits over which the if/else
                should be run.
            clbits (Optional[Sequence[ClbitSpecifier]]): The circuit clbits over which the if/else
                should be run.
            label (Optional[str]): The string label of the instruction in the circuit.

        Returns:
            InstructionSet or IfContext: depending on the call signature, either a context
            manager for creating the ``if`` block (it will automatically be added to the circuit at
            the end of the block), or an :obj:`~InstructionSet` handle to the appended conditional
            operation.

        Raises:
            CircuitError: If the provided condition references Clbits outside the
                enclosing circuit.
            CircuitError: if an incorrect calling convention is used.

        Returns:
            A handle to the instruction created.
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.controlflow.if_else import IfElseOp, IfContext

        if isinstance(condition, expr.Expr):
            condition = self._validate_expr(condition)
        else:
            condition = (self._resolve_classical_resource(condition[0]), condition[1])

        if true_body is None:
            if qubits is not None or clbits is not None:
                raise CircuitError(
                    "When using 'if_test' as a context manager, you cannot pass qubits or clbits."
                )
            # We can only allow jumps if we're in a loop block, but the default path (no scopes)
            # also allows adding jumps to support the more verbose internal mode.
            in_loop = bool(self._control_flow_scopes and self._control_flow_scopes[-1].allow_jumps)
            return IfContext(self, condition, in_loop=in_loop, label=label)
        elif qubits is None or clbits is None:
            raise CircuitError("When using 'if_test' with a body, you must pass qubits and clbits.")

        return self.append(IfElseOp(condition, true_body, None, label), qubits, clbits)

    def if_else(
        self,
        condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | tuple[Clbit, bool],
        true_body: "QuantumCircuit",
        false_body: "QuantumCircuit",
        qubits: Sequence[QubitSpecifier],
        clbits: Sequence[ClbitSpecifier],
        label: str | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.IfElseOp`.

        .. note::

            This method does not have an associated context-manager form, because it is already
            handled by the :meth:`.if_test` method.  You can use the ``else`` part of that with
            something such as::

                from qiskit.circuit import QuantumCircuit, Qubit, Clbit
                bits = [Qubit(), Qubit(), Clbit()]
                qc = QuantumCircuit(bits)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure(0, 0)
                with qc.if_test((bits[2], 0)) as else_:
                    qc.h(0)
                with else_:
                    qc.x(0)

        Args:
            condition: A condition to be evaluated at circuit runtime which,
                if true, will trigger the evaluation of ``true_body``. Can be
                specified as either a tuple of a ``ClassicalRegister`` to be
                tested for equality with a given ``int``, or as a tuple of a
                ``Clbit`` to be compared to either a ``bool`` or an ``int``.
            true_body: The circuit body to be run if ``condition`` is true.
            false_body: The circuit to be run if ``condition`` is false.
            qubits: The circuit qubits over which the if/else should be run.
            clbits: The circuit clbits over which the if/else should be run.
            label: The string label of the instruction in the circuit.

        Raises:
            CircuitError: If the provided condition references Clbits outside the
                enclosing circuit.

        Returns:
            A handle to the instruction created.
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.controlflow.if_else import IfElseOp

        if isinstance(condition, expr.Expr):
            condition = self._validate_expr(condition)
        else:
            condition = (self._resolve_classical_resource(condition[0]), condition[1])

        return self.append(IfElseOp(condition, true_body, false_body, label), qubits, clbits)

    @typing.overload
    def switch(
        self,
        target: Union[ClbitSpecifier, ClassicalRegister],
        cases: None,
        qubits: None,
        clbits: None,
        *,
        label: Optional[str],
    ) -> "qiskit.circuit.controlflow.switch_case.SwitchContext":
        ...

    @typing.overload
    def switch(
        self,
        target: Union[ClbitSpecifier, ClassicalRegister],
        cases: Iterable[Tuple[typing.Any, QuantumCircuit]],
        qubits: Sequence[QubitSpecifier],
        clbits: Sequence[ClbitSpecifier],
        *,
        label: Optional[str],
    ) -> InstructionSet:
        ...

    def switch(self, target, cases=None, qubits=None, clbits=None, *, label=None):
        """Create a ``switch``/``case`` structure on this circuit.

        There are two forms for calling this function.  If called with all its arguments (with the
        possible exception of ``label``), it will create a :class:`.SwitchCaseOp` with the given
        case structure.  If ``cases`` (and ``qubits`` and ``clbits``) are *not* passed, then this
        acts as a context manager, which will automatically build a :class:`.SwitchCaseOp` when the
        scope finishes.  In this form, you do not need to keep track of the qubits or clbits you are
        using, because the scope will handle it for you.

        Example usage::

            from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister
            qreg = QuantumRegister(3)
            creg = ClassicalRegister(3)
            qc = QuantumCircuit(qreg, creg)
            qc.h([0, 1, 2])
            qc.measure([0, 1, 2], [0, 1, 2])

            with qc.switch(creg) as case:
                with case(0):
                    qc.x(0)
                with case(1, 2):
                    qc.z(1)
                with case(case.DEFAULT):
                    qc.cx(0, 1)

        Args:
            target (Union[ClassicalRegister, Clbit]): The classical value to switch one.  This must
                be integer-like.
            cases (Iterable[Tuple[typing.Any, QuantumCircuit]]): A sequence of case specifiers.
                Each tuple defines one case body (the second item).  The first item of the tuple can
                be either a single integer value, the special value :data:`.CASE_DEFAULT`, or a
                tuple of several integer values.  Each of the integer values will be tried in turn;
                control will then pass to the body corresponding to the first match.
                :data:`.CASE_DEFAULT` matches all possible values.  Omit in context-manager form.
            qubits (Sequence[Qubit]): The circuit qubits over which all case bodies execute. Omit in
                context-manager form.
            clbits (Sequence[Clbit]): The circuit clbits over which all case bodies execute. Omit in
                context-manager form.
            label (Optional[str]): The string label of the instruction in the circuit.

        Returns:
            InstructionSet or SwitchCaseContext: If used in context-manager mode, then this should
            be used as a ``with`` resource, which will return an object that can be repeatedly
            entered to produce cases for the switch statement.  If the full form is used, then this
            returns a handle to the instructions created.

        Raises:
            CircuitError: if an incorrect calling convention is used.
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.controlflow.switch_case import SwitchCaseOp, SwitchContext

        if isinstance(target, expr.Expr):
            target = self._validate_expr(target)
        else:
            target = self._resolve_classical_resource(target)
        if cases is None:
            if qubits is not None or clbits is not None:
                raise CircuitError(
                    "When using 'switch' as a context manager, you cannot pass qubits or clbits."
                )
            in_loop = bool(self._control_flow_scopes and self._control_flow_scopes[-1].allow_jumps)
            return SwitchContext(self, target, in_loop=in_loop, label=label)

        if qubits is None or clbits is None:
            raise CircuitError("When using 'switch' with cases, you must pass qubits and clbits.")
        return self.append(SwitchCaseOp(target, cases, label=label), qubits, clbits)

    def break_loop(self) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.BreakLoopOp`.

        .. warning::

            If you are using the context-manager "builder" forms of :meth:`.if_test`,
            :meth:`.for_loop` or :meth:`.while_loop`, you can only call this method if you are
            within a loop context, because otherwise the "resource width" of the operation cannot be
            determined.  This would quickly lead to invalid circuits, and so if you are trying to
            construct a reusable loop body (without the context managers), you must also use the
            non-context-manager form of :meth:`.if_test` and :meth:`.if_else`.  Take care that the
            :obj:`.BreakLoopOp` instruction must span all the resources of its containing loop, not
            just the immediate scope.

        Returns:
            A handle to the instruction created.

        Raises:
            CircuitError: if this method was called within a builder context, but not contained
                within a loop.
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.controlflow.break_loop import BreakLoopOp, BreakLoopPlaceholder

        if self._control_flow_scopes:
            operation = BreakLoopPlaceholder()
            resources = operation.placeholder_resources()
            return self.append(operation, resources.qubits, resources.clbits)
        return self.append(BreakLoopOp(self.num_qubits, self.num_clbits), self.qubits, self.clbits)

    def continue_loop(self) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.ContinueLoopOp`.

        .. warning::

            If you are using the context-manager "builder" forms of :meth:`.if_test`,
            :meth:`.for_loop` or :meth:`.while_loop`, you can only call this method if you are
            within a loop context, because otherwise the "resource width" of the operation cannot be
            determined.  This would quickly lead to invalid circuits, and so if you are trying to
            construct a reusable loop body (without the context managers), you must also use the
            non-context-manager form of :meth:`.if_test` and :meth:`.if_else`.  Take care that the
            :class:`~qiskit.circuit.ContinueLoopOp` instruction must span all the resources of its
            containing loop, not just the immediate scope.

        Returns:
            A handle to the instruction created.

        Raises:
            CircuitError: if this method was called within a builder context, but not contained
                within a loop.
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.controlflow.continue_loop import ContinueLoopOp, ContinueLoopPlaceholder

        if self._control_flow_scopes:
            operation = ContinueLoopPlaceholder()
            resources = operation.placeholder_resources()
            return self.append(operation, resources.qubits, resources.clbits)
        return self.append(
            ContinueLoopOp(self.num_qubits, self.num_clbits), self.qubits, self.clbits
        )

    def add_calibration(
        self,
        gate: Union[Gate, str],
        qubits: Sequence[int],
        # Schedule has the type `qiskit.pulse.Schedule`, but `qiskit.pulse` cannot be imported
        # while this module is, and so Sphinx will not accept a forward reference to it.  Sphinx
        # needs the types available at runtime, whereas mypy will accept it, because it handles the
        # type checking by static analysis.
        schedule,
        params: Sequence[ParameterValueType] | None = None,
    ) -> None:
        """Register a low-level, custom pulse definition for the given gate.

        Args:
            gate (Union[Gate, str]): Gate information.
            qubits (Union[int, Tuple[int]]): List of qubits to be measured.
            schedule (Schedule): Schedule information.
            params (Optional[List[Union[float, Parameter]]]): A list of parameters.

        Raises:
            Exception: if the gate is of type string and params is None.
        """

        def _format(operand):
            try:
                # Using float/complex value as a dict key is not good idea.
                # This makes the mapping quite sensitive to the rounding error.
                # However, the mechanism is already tied to the execution model (i.e. pulse gate)
                # and we cannot easily update this rule.
                # The same logic exists in DAGCircuit.add_calibration.
                evaluated = complex(operand)
                if np.isreal(evaluated):
                    evaluated = float(evaluated.real)
                    if evaluated.is_integer():
                        evaluated = int(evaluated)
                return evaluated
            except TypeError:
                # Unassigned parameter
                return operand

        if isinstance(gate, Gate):
            params = gate.params
            gate = gate.name
        if params is not None:
            params = tuple(map(_format, params))
        else:
            params = ()

        self._calibrations[gate][(tuple(qubits), params)] = schedule

    # Functions only for scheduled circuits
    def qubit_duration(self, *qubits: Union[Qubit, int]) -> float:
        """Return the duration between the start and stop time of the first and last instructions,
        excluding delays, over the supplied qubits. Its time unit is ``self.unit``.

        Args:
            *qubits: Qubits within ``self`` to include.

        Returns:
            Return the duration between the first start and last stop time of non-delay instructions
        """
        return self.qubit_stop_time(*qubits) - self.qubit_start_time(*qubits)

    def qubit_start_time(self, *qubits: Union[Qubit, int]) -> float:
        """Return the start time of the first instruction, excluding delays,
        over the supplied qubits. Its time unit is ``self.unit``.

        Return 0 if there are no instructions over qubits

        Args:
            *qubits: Qubits within ``self`` to include. Integers are allowed for qubits, indicating
            indices of ``self.qubits``.

        Returns:
            Return the start time of the first instruction, excluding delays, over the qubits

        Raises:
            CircuitError: if ``self`` is a not-yet scheduled circuit.
        """
        if self.duration is None:
            # circuit has only delays, this is kind of scheduled
            for instruction in self._data:
                if not isinstance(instruction.operation, Delay):
                    raise CircuitError(
                        "qubit_start_time undefined. Circuit must be scheduled first."
                    )
            return 0

        qubits = [self.qubits[q] if isinstance(q, int) else q for q in qubits]

        starts = {q: 0 for q in qubits}
        dones = {q: False for q in qubits}
        for instruction in self._data:
            for q in qubits:
                if q in instruction.qubits:
                    if isinstance(instruction.operation, Delay):
                        if not dones[q]:
                            starts[q] += instruction.operation.duration
                    else:
                        dones[q] = True
            if len(qubits) == len([done for done in dones.values() if done]):  # all done
                return min(start for start in starts.values())

        return 0  # If there are no instructions over bits

    def qubit_stop_time(self, *qubits: Union[Qubit, int]) -> float:
        """Return the stop time of the last instruction, excluding delays, over the supplied qubits.
        Its time unit is ``self.unit``.

        Return 0 if there are no instructions over qubits

        Args:
            *qubits: Qubits within ``self`` to include. Integers are allowed for qubits, indicating
            indices of ``self.qubits``.

        Returns:
            Return the stop time of the last instruction, excluding delays, over the qubits

        Raises:
            CircuitError: if ``self`` is a not-yet scheduled circuit.
        """
        if self.duration is None:
            # circuit has only delays, this is kind of scheduled
            for instruction in self._data:
                if not isinstance(instruction.operation, Delay):
                    raise CircuitError(
                        "qubit_stop_time undefined. Circuit must be scheduled first."
                    )
            return 0

        qubits = [self.qubits[q] if isinstance(q, int) else q for q in qubits]

        stops = {q: self.duration for q in qubits}
        dones = {q: False for q in qubits}
        for instruction in reversed(self._data):
            for q in qubits:
                if q in instruction.qubits:
                    if isinstance(instruction.operation, Delay):
                        if not dones[q]:
                            stops[q] -= instruction.operation.duration
                    else:
                        dones[q] = True
            if len(qubits) == len([done for done in dones.values() if done]):  # all done
                return max(stop for stop in stops.values())

        return 0  # If there are no instructions over bits


class _ParameterBindsDict:
    __slots__ = ("mapping", "allowed_keys")

    def __init__(self, mapping, allowed_keys):
        self.mapping = mapping
        self.allowed_keys = allowed_keys

    def items(self):
        """Iterator through all the keys in the mapping that we care about.  Wrapping the main
        mapping allows us to avoid reconstructing a new 'dict', but just use the given 'mapping'
        without any copy / reconstruction."""
        for parameter, value in self.mapping.items():
            if parameter in self.allowed_keys:
                yield parameter, value


class _ParameterBindsSequence:
    __slots__ = ("parameters", "values", "mapping_cache")

    def __init__(self, parameters, values):
        self.parameters = parameters
        self.values = values
        self.mapping_cache = None

    def items(self):
        """Iterator through all the keys in the mapping that we care about."""
        return zip(self.parameters, self.values)

    @property
    def mapping(self):
        """Cached version of a mapping.  This is only generated on demand."""
        if self.mapping_cache is None:
            self.mapping_cache = dict(zip(self.parameters, self.values))
        return self.mapping_cache


# Used by the OQ2 exporter.  Just needs to have enough parameters to support the largest standard
# (non-controlled) gate in our standard library.  We have to use the same `Parameter` instances each
# time so the equality comparisons will work.
_QASM2_FIXED_PARAMETERS = [Parameter("param0"), Parameter("param1"), Parameter("param2")]


def _qasm2_custom_operation_statement(
    instruction, existing_gate_names, gates_to_define, bit_labels
):
    operation = _qasm2_define_custom_operation(
        instruction.operation, existing_gate_names, gates_to_define
    )
    # Insert qasm representation of the original instruction
    if instruction.clbits:
        bits = itertools.chain(instruction.qubits, instruction.clbits)
    else:
        bits = instruction.qubits
    bits_qasm = ",".join(bit_labels[j] for j in bits)
    instruction_qasm = f"{_instruction_qasm2(operation)} {bits_qasm};"
    return instruction_qasm


def _qasm2_define_custom_operation(operation, existing_gate_names, gates_to_define):
    """Extract a custom definition from the given operation, and append any necessary additional
    subcomponents' definitions to the ``gates_to_define`` ordered dictionary.

    Returns a potentially new :class:`.Instruction`, which should be used for the
    :meth:`~.Instruction.qasm` call (it may have been renamed)."""
    # pylint: disable=cyclic-import
    from qiskit.circuit import library as lib
    from qiskit.qasm2 import QASM2ExportError

    if operation.name in existing_gate_names:
        return operation

    # Check instructions names or label are valid
    escaped = _qasm_escape_name(operation.name, "gate_")
    if escaped != operation.name:
        operation = operation.copy(name=escaped)

    # These are built-in gates that are known to be safe to construct by passing the correct number
    # of `Parameter` instances positionally, and have no other information.  We can't guarantee that
    # if they've been subclassed, though.  This is a total hack; ideally we'd be able to inspect the
    # "calling" signatures of Qiskit `Gate` objects to know whether they're safe to re-parameterise.
    known_good_parameterized = {
        lib.PhaseGate,
        lib.RGate,
        lib.RXGate,
        lib.RXXGate,
        lib.RYGate,
        lib.RYYGate,
        lib.RZGate,
        lib.RZXGate,
        lib.RZZGate,
        lib.XXMinusYYGate,
        lib.XXPlusYYGate,
        lib.UGate,
        lib.U1Gate,
        lib.U2Gate,
        lib.U3Gate,
    }

    # In known-good situations we want to use a manually parametrised object as the source of the
    # definition, but still continue to return the given object as the call-site object.
    if type(operation) in known_good_parameterized:
        parameterized_operation = type(operation)(*_QASM2_FIXED_PARAMETERS[: len(operation.params)])
    elif hasattr(operation, "_qasm2_decomposition"):
        new_op = operation._qasm2_decomposition()
        parameterized_operation = operation = new_op.copy(
            name=_qasm_escape_name(new_op.name, "gate_")
        )
    else:
        parameterized_operation = operation

    # If there's an _equal_ operation in the existing circuits to be defined, then our job is done.
    previous_definition_source, _ = gates_to_define.get(operation.name, (None, None))
    if parameterized_operation == previous_definition_source:
        return operation

    # Otherwise, if there's a naming clash, we need a unique name.
    if operation.name in gates_to_define:
        operation = _rename_operation(operation)

    new_name = operation.name

    if parameterized_operation.params:
        parameters_qasm = (
            "(" + ",".join(f"param{i}" for i in range(len(parameterized_operation.params))) + ")"
        )
    else:
        parameters_qasm = ""

    if operation.num_qubits == 0:
        raise QASM2ExportError(
            f"OpenQASM 2 cannot represent '{operation.name}, which acts on zero qubits."
        )
    if operation.num_clbits != 0:
        raise QASM2ExportError(
            f"OpenQASM 2 cannot represent '{operation.name}', which acts on {operation.num_clbits}"
            " classical bits."
        )

    qubits_qasm = ",".join(f"q{i}" for i in range(parameterized_operation.num_qubits))
    parameterized_definition = getattr(parameterized_operation, "definition", None)
    if parameterized_definition is None:
        gates_to_define[new_name] = (
            parameterized_operation,
            f"opaque {new_name}{parameters_qasm} {qubits_qasm};",
        )
    else:
        qubit_labels = {bit: f"q{i}" for i, bit in enumerate(parameterized_definition.qubits)}
        body_qasm = " ".join(
            _qasm2_custom_operation_statement(
                instruction, existing_gate_names, gates_to_define, qubit_labels
            )
            for instruction in parameterized_definition.data
        )

        # if an inner operation has the same name as the actual operation, it needs to be renamed
        if operation.name in gates_to_define:
            operation = _rename_operation(operation)
            new_name = operation.name

        definition_qasm = f"gate {new_name}{parameters_qasm} {qubits_qasm} {{ {body_qasm} }}"
        gates_to_define[new_name] = (parameterized_operation, definition_qasm)
    return operation


def _rename_operation(operation):
    """Returns the operation with a new name following this pattern: {operation name}_{operation id}"""
    new_name = f"{operation.name}_{id(operation)}"
    updated_operation = operation.copy(name=new_name)
    return updated_operation


def _qasm_escape_name(name: str, prefix: str) -> str:
    """Returns a valid OpenQASM identifier, using `prefix` as a prefix if necessary.  `prefix` must
    itself be a valid identifier."""
    # Replace all non-ASCII-word characters (letters, digits, underscore) with the underscore.
    escaped_name = re.sub(r"\W", "_", name, flags=re.ASCII)
    if (
        not escaped_name
        or escaped_name[0] not in string.ascii_lowercase
        or escaped_name in QASM2_RESERVED
    ):
        escaped_name = prefix + escaped_name
    return escaped_name


def _instruction_qasm2(operation):
    """Return an OpenQASM 2 string for the instruction."""
    from qiskit.qasm2 import QASM2ExportError  # pylint: disable=cyclic-import

    if operation.name == "c3sx":
        qasm2_call = "c3sqrtx"
    else:
        qasm2_call = operation.name
    if operation.params:
        qasm2_call = "{}({})".format(
            qasm2_call,
            ",".join([pi_check(i, output="qasm", eps=1e-12) for i in operation.params]),
        )
    if operation.condition is not None:
        if not isinstance(operation.condition[0], ClassicalRegister):
            raise QASM2ExportError(
                "OpenQASM 2 can only condition on registers, but got '{operation.condition[0]}'"
            )
        qasm2_call = (
            "if(%s==%d) " % (operation.condition[0].name, operation.condition[1]) + qasm2_call
        )
    return qasm2_call


def _make_unique(name: str, already_defined: collections.abc.Set[str]) -> str:
    """Generate a name by suffixing the given stem that is unique within the defined set."""
    if name not in already_defined:
        return name
    used = {in_use[len(name) :] for in_use in already_defined if in_use.startswith(name)}
    characters = (string.digits + string.ascii_letters) if name else string.ascii_letters
    for parts in itertools.chain.from_iterable(
        itertools.product(characters, repeat=n) for n in itertools.count(1)
    ):
        suffix = "".join(parts)
        if suffix not in used:
            return name + suffix
    # This isn't actually reachable because the above loop is infinite.
    return name


def _bit_argument_conversion(specifier, bit_sequence, bit_set, type_) -> list[Bit]:
    """Get the list of bits referred to by the specifier ``specifier``.

    Valid types for ``specifier`` are integers, bits of the correct type (as given in ``type_``), or
    iterables of one of those two scalar types.  Integers are interpreted as indices into the
    sequence ``bit_sequence``.  All allowed bits must be in ``bit_set`` (which should implement
    fast lookup), which is assumed to contain the same bits as ``bit_sequence``.

    Returns:
        List[Bit]: a list of the specified bits from ``bits``.

    Raises:
        CircuitError: if an incorrect type or index is encountered, if the same bit is specified
            more than once, or if the specifier is to a bit not in the ``bit_set``.
    """
    # The duplication between this function and `_bit_argument_conversion_scalar` is so that fast
    # paths return as quickly as possible, and all valid specifiers will resolve without needing to
    # try/catch exceptions (which is too slow for inner-loop code).
    if isinstance(specifier, type_):
        if specifier in bit_set:
            return [specifier]
        raise CircuitError(f"Bit '{specifier}' is not in the circuit.")
    if isinstance(specifier, (int, np.integer)):
        try:
            return [bit_sequence[specifier]]
        except IndexError as ex:
            raise CircuitError(
                f"Index {specifier} out of range for size {len(bit_sequence)}."
            ) from ex
    # Slices can't raise IndexError - they just return an empty list.
    if isinstance(specifier, slice):
        return bit_sequence[specifier]
    try:
        return [
            _bit_argument_conversion_scalar(index, bit_sequence, bit_set, type_)
            for index in specifier
        ]
    except TypeError as ex:
        message = (
            f"Incorrect bit type: expected '{type_.__name__}' but got '{type(specifier).__name__}'"
            if isinstance(specifier, Bit)
            else f"Invalid bit index: '{specifier}' of type '{type(specifier)}'"
        )
        raise CircuitError(message) from ex


def _bit_argument_conversion_scalar(specifier, bit_sequence, bit_set, type_):
    if isinstance(specifier, type_):
        if specifier in bit_set:
            return specifier
        raise CircuitError(f"Bit '{specifier}' is not in the circuit.")
    if isinstance(specifier, (int, np.integer)):
        try:
            return bit_sequence[specifier]
        except IndexError as ex:
            raise CircuitError(
                f"Index {specifier} out of range for size {len(bit_sequence)}."
            ) from ex
    message = (
        f"Incorrect bit type: expected '{type_.__name__}' but got '{type(specifier).__name__}'"
        if isinstance(specifier, Bit)
        else f"Invalid bit index: '{specifier}' of type '{type(specifier)}'"
    )
    raise CircuitError(message)

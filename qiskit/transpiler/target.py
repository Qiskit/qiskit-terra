# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
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
A target object represents the minimum set of information the transpiler needs
from a backend
"""

from __future__ import annotations

import itertools

from typing import Optional, List, Any
from collections.abc import Mapping
from collections import defaultdict
import datetime
import io
import logging
import inspect

import rustworkx as rx

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.calibration_entries import CalibrationEntry, ScheduleDef
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.pulse.exceptions import PulseError, UnassignedDurationError
from qiskit.exceptions import QiskitError

# import QubitProperties here to provide convenience alias for building a
# full target
from qiskit.providers.backend import QubitProperties  # pylint: disable=unused-import
from qiskit.providers.models.backendproperties import BackendProperties


logger = logging.getLogger(__name__)


class InstructionProperties:
    """A representation of the properties of a gate implementation.

    This class provides the optional properties that a backend can provide
    about an instruction. These represent the set that the transpiler can
    currently work with if present. However, if your backend provides additional
    properties for instructions you should subclass this to add additional
    custom attributes for those custom/additional properties by the backend.
    """

    __slots__ = ("duration", "error", "_calibration")

    def __init__(
        self,
        duration: float | None = None,
        error: float | None = None,
        calibration: Schedule | ScheduleBlock | CalibrationEntry | None = None,
    ):
        """Create a new ``InstructionProperties`` object

        Args:
            duration: The duration, in seconds, of the instruction on the
                specified set of qubits
            error: The average error rate for the instruction on the specified
                set of qubits.
            calibration: The pulse representation of the instruction.
        """
        self._calibration: CalibrationEntry | None = None

        self.duration = duration
        self.error = error
        self.calibration = calibration

    @property
    def calibration(self):
        """The pulse representation of the instruction.

        .. note::

            This attribute always returns a Qiskit pulse program, but it is internally
            wrapped by the :class:`.CalibrationEntry` to manage unbound parameters
            and to uniformly handle different data representation,
            for example, un-parsed Pulse Qobj JSON that a backend provider may provide.

            This value can be overridden through the property setter in following manner.
            When you set either :class:`.Schedule` or :class:`.ScheduleBlock` this is
            always treated as a user-defined (custom) calibration and
            the transpiler may automatically attach the calibration data to the output circuit.
            This calibration data may appear in the wire format as an inline calibration,
            which may further update the backend standard instruction set architecture.

            If you are a backend provider who provides a default calibration data
            that is not needed to be attached to the transpiled quantum circuit,
            you can directly set :class:`.CalibrationEntry` instance to this attribute,
            in which you should set :code:`user_provided=False` when you define
            calibration data for the entry. End users can still intentionally utilize
            the calibration data, for example, to run pulse-level simulation of the circuit.
            However, such entry doesn't appear in the wire format, and backend must
            use own definition to compile the circuit down to the execution format.

        """
        if self._calibration is None:
            return None
        return self._calibration.get_schedule()

    @calibration.setter
    def calibration(self, calibration: Schedule | ScheduleBlock | CalibrationEntry):
        if isinstance(calibration, (Schedule, ScheduleBlock)):
            new_entry = ScheduleDef()
            new_entry.define(calibration, user_provided=True)
        else:
            new_entry = calibration
        self._calibration = new_entry

    def __repr__(self):
        return (
            f"InstructionProperties(duration={self.duration}, error={self.error}"
            f", calibration={self._calibration})"
        )


class Target(Mapping):
    """
    The intent of the ``Target`` object is to inform Qiskit's compiler about
    the constraints of a particular backend so the compiler can compile an
    input circuit to something that works and is optimized for a device. It
    currently contains a description of instructions on a backend and their
    properties as well as some timing information. However, this exact
    interface may evolve over time as the needs of the compiler change. These
    changes will be done in a backwards compatible and controlled manner when
    they are made (either through versioning, subclassing, or mixins) to add
    on to the set of information exposed by a target.

    As a basic example, let's assume backend has two qubits, supports
    :class:`~qiskit.circuit.library.UGate` on both qubits and
    :class:`~qiskit.circuit.library.CXGate` in both directions. To model this
    you would create the target like::

        from qiskit.transpiler import Target, InstructionProperties
        from qiskit.circuit.library import UGate, CXGate
        from qiskit.circuit import Parameter

        gmap = Target()
        theta = Parameter('theta')
        phi = Parameter('phi')
        lam = Parameter('lambda')
        u_props = {
            (0,): InstructionProperties(duration=5.23e-8, error=0.00038115),
            (1,): InstructionProperties(duration=4.52e-8, error=0.00032115),
        }
        gmap.add_instruction(UGate(theta, phi, lam), u_props)
        cx_props = {
            (0,1): InstructionProperties(duration=5.23e-7, error=0.00098115),
            (1,0): InstructionProperties(duration=4.52e-7, error=0.00132115),
        }
        gmap.add_instruction(CXGate(), cx_props)

    Each instruction in the ``Target`` is indexed by a unique string name that uniquely
    identifies that instance of an :class:`~qiskit.circuit.Instruction` object in
    the Target. There is a 1:1 mapping between a name and an
    :class:`~qiskit.circuit.Instruction` instance in the target and each name must
    be unique. By default, the name is the :attr:`~qiskit.circuit.Instruction.name`
    attribute of the instruction, but can be set to anything. This lets a single
    target have multiple instances of the same instruction class with different
    parameters. For example, if a backend target has two instances of an
    :class:`~qiskit.circuit.library.RXGate` one is parameterized over any theta
    while the other is tuned up for a theta of pi/6 you can add these by doing something
    like::

        import math

        from qiskit.transpiler import Target, InstructionProperties
        from qiskit.circuit.library import RXGate
        from qiskit.circuit import Parameter

        target = Target()
        theta = Parameter('theta')
        rx_props = {
            (0,): InstructionProperties(duration=5.23e-8, error=0.00038115),
        }
        target.add_instruction(RXGate(theta), rx_props)
        rx_30_props = {
            (0,): InstructionProperties(duration=1.74e-6, error=.00012)
        }
        target.add_instruction(RXGate(math.pi / 6), rx_30_props, name='rx_30')

    Then in the ``target`` object accessing by ``rx_30`` will get the fixed
    angle :class:`~qiskit.circuit.library.RXGate` while ``rx`` will get the
    parameterized :class:`~qiskit.circuit.library.RXGate`.

    .. note::

        This class assumes that qubit indices start at 0 and are a contiguous
        set if you want a submapping the bits will need to be reindexed in
        a new``Target`` object.

    .. note::

        This class only supports additions of gates, qargs, and qubits.
        If you need to remove one of these the best option is to iterate over
        an existing object and create a new subset (or use one of the methods
        to do this). The object internally caches different views and these
        would potentially be invalidated by removals.
    """

    __slots__ = (
        "num_qubits",
        "_gate_map",
        "_gate_name_map",
        "_qarg_gate_map",
        "description",
        "_coupling_graph",
        "_instruction_durations",
        "_instruction_schedule_map",
        "dt",
        "granularity",
        "min_length",
        "pulse_alignment",
        "acquire_alignment",
        "_non_global_basis",
        "_non_global_strict_basis",
        "qubit_properties",
        "_global_operations",
        "concurrent_measurements",
    )

    def __init__(
        self,
        description=None,
        num_qubits=0,
        dt=None,
        granularity=1,
        min_length=1,
        pulse_alignment=1,
        acquire_alignment=1,
        qubit_properties=None,
        concurrent_measurements=None,
    ):
        """
        Create a new ``Target`` object

        Args:
            description (str): An optional string to describe the Target.
            num_qubits (int): An optional int to specify the number of qubits
                the backend target has. If not set it will be implicitly set
                based on the qargs when :meth:`~qiskit.Target.add_instruction`
                is called. Note this must be set if the backend target is for a
                noiseless simulator that doesn't have constraints on the
                instructions so the transpiler knows how many qubits are
                available.
            dt (float): The system time resolution of input signals in seconds
            granularity (int): An integer value representing minimum pulse gate
                resolution in units of ``dt``. A user-defined pulse gate should
                have duration of a multiple of this granularity value.
            min_length (int): An integer value representing minimum pulse gate
                length in units of ``dt``. A user-defined pulse gate should be
                longer than this length.
            pulse_alignment (int): An integer value representing a time
                resolution of gate instruction starting time. Gate instruction
                should start at time which is a multiple of the alignment
                value.
            acquire_alignment (int): An integer value representing a time
                resolution of measure instruction starting time. Measure
                instruction should start at time which is a multiple of the
                alignment value.
            qubit_properties (list): A list of :class:`~.QubitProperties`
                objects defining the characteristics of each qubit on the
                target device. If specified the length of this list must match
                the number of qubits in the target, where the index in the list
                matches the qubit number the properties are defined for. If some
                qubits don't have properties available you can set that entry to
                ``None``
            concurrent_measurements(list): A list of sets of qubits that must be
                measured together. This must be provided
                as a nested list like ``[[0, 1], [2, 3, 4]]``.
        Raises:
            ValueError: If both ``num_qubits`` and ``qubit_properties`` are both
                defined and the value of ``num_qubits`` differs from the length of
                ``qubit_properties``.
        """
        self.num_qubits = num_qubits
        # A mapping of gate name -> gate instance
        self._gate_name_map = {}
        # A nested mapping of gate name -> qargs -> properties
        self._gate_map = {}
        # A mapping of number of qubits to set of op names which are global
        self._global_operations = defaultdict(set)
        # A mapping of qarg -> set(gate name)
        self._qarg_gate_map = defaultdict(set)
        self.dt = dt
        self.description = description
        self._coupling_graph = None
        self._instruction_durations = None
        self._instruction_schedule_map = None
        self.granularity = granularity
        self.min_length = min_length
        self.pulse_alignment = pulse_alignment
        self.acquire_alignment = acquire_alignment
        self._non_global_basis = None
        self._non_global_strict_basis = None
        if qubit_properties is not None:
            if not self.num_qubits:
                self.num_qubits = len(qubit_properties)
            else:
                if self.num_qubits != len(qubit_properties):
                    raise ValueError(
                        "The value of num_qubits specified does not match the "
                        "length of the input qubit_properties list"
                    )
        self.qubit_properties = qubit_properties
        self.concurrent_measurements = concurrent_measurements

    def add_instruction(self, instruction, properties=None, name=None):
        """Add a new instruction to the :class:`~qiskit.transpiler.Target`

        As ``Target`` objects are strictly additive this is the primary method
        for modifying a ``Target``. Typically, you will use this to fully populate
        a ``Target`` before using it in :class:`~qiskit.providers.BackendV2`. For
        example::

            from qiskit.circuit.library import CXGate
            from qiskit.transpiler import Target, InstructionProperties

            target = Target()
            cx_properties = {
                (0, 1): None,
                (1, 0): None,
                (0, 2): None,
                (2, 0): None,
                (0, 3): None,
                (2, 3): None,
                (3, 0): None,
                (3, 2): None
            }
            target.add_instruction(CXGate(), cx_properties)

        Will add a :class:`~qiskit.circuit.library.CXGate` to the target with no
        properties (duration, error, etc) with the coupling edge list:
        ``(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (2, 3), (3, 0), (3, 2)``. If
        there are properties available for the instruction you can replace the
        ``None`` value in the properties dictionary with an
        :class:`~qiskit.transpiler.InstructionProperties` object. This pattern
        is repeated for each :class:`~qiskit.circuit.Instruction` the target
        supports.

        Args:
            instruction (Union[qiskit.circuit.Instruction, Type[qiskit.circuit.Instruction]]):
                The operation object to add to the map. If it's parameterized any value
                of the parameter can be set. Optionally for variable width
                instructions (such as control flow operations such as :class:`~.ForLoop` or
                :class:`~MCXGate`) you can specify the class. If the class is specified than the
                ``name`` argument must be specified. When a class is used the gate is treated as global
                and not having any properties set.
            properties (dict): A dictionary of qarg entries to an
                :class:`~qiskit.transpiler.InstructionProperties` object for that
                instruction implementation on the backend. Properties are optional
                for any instruction implementation, if there are no
                :class:`~qiskit.transpiler.InstructionProperties` available for the
                backend the value can be None. If there are no constraints on the
                instruction (as in a noiseless/ideal simulation) this can be set to
                ``{None, None}`` which will indicate it runs on all qubits (or all
                available permutations of qubits for multi-qubit gates). The first
                ``None`` indicates it applies to all qubits and the second ``None``
                indicates there are no
                :class:`~qiskit.transpiler.InstructionProperties` for the
                instruction. By default, if properties is not set it is equivalent to
                passing ``{None: None}``.
            name (str): An optional name to use for identifying the instruction. If not
                specified the :attr:`~qiskit.circuit.Instruction.name` attribute
                of ``gate`` will be used. All gates in the ``Target`` need unique
                names. Backends can differentiate between different
                parameterization of a single gate by providing a unique name for
                each (e.g. `"rx30"`, `"rx60", ``"rx90"`` similar to the example in the
                documentation for the :class:`~qiskit.transpiler.Target` class).
        Raises:
            AttributeError: If gate is already in map
            TranspilerError: If an operation class is passed in for ``instruction`` and no name
                is specified or ``properties`` is set.
        """
        is_class = inspect.isclass(instruction)
        if not is_class:
            instruction_name = name or instruction.name
        else:
            # Invalid to have class input without a name with characters set "" is not a valid name
            if not name:
                raise TranspilerError(
                    "A name must be specified when defining a supported global operation by class"
                )
            if properties is not None:
                raise TranspilerError(
                    "An instruction added globally by class can't have properties set."
                )
            instruction_name = name
        if properties is None:
            properties = {None: None}
        if instruction_name in self._gate_map:
            raise AttributeError(f"Instruction {instruction_name} is already in the target")
        self._gate_name_map[instruction_name] = instruction
        if is_class:
            qargs_val = {None: None}
        else:
            if None in properties:
                self._global_operations[instruction.num_qubits].add(instruction_name)
            qargs_val = {}
            for qarg in properties:
                if qarg is not None and len(qarg) != instruction.num_qubits:
                    raise TranspilerError(
                        f"The number of qubits for {instruction} does not match the number "
                        f"of qubits in the properties dictionary: {qarg}"
                    )
                if qarg is not None:
                    self.num_qubits = max(
                        self.num_qubits if self.num_qubits is not None else 0, max(qarg) + 1
                    )
                qargs_val[qarg] = properties[qarg]
                self._qarg_gate_map[qarg].add(instruction_name)
        self._gate_map[instruction_name] = qargs_val
        self._coupling_graph = None
        self._instruction_durations = None
        self._instruction_schedule_map = None
        self._non_global_basis = None
        self._non_global_strict_basis = None

    def update_instruction_properties(self, instruction, qargs: tuple, properties):
        """Update the property object for an instruction qarg pair already in the Target

        Args:
            instruction (str): The instruction name to update
            qargs (tuple): The qargs to update the properties of
            properties (InstructionProperties): The properties to set for this instruction
        Raises:
            KeyError: If ``instruction`` or ``qarg`` are not in the target
        """
        if instruction not in self._gate_map:
            raise KeyError(f"Provided instruction: '{instruction}' not in this Target")
        if qargs not in self._gate_map[instruction]:
            raise KeyError(f"Provided qarg: '{qargs}' not in this Target for {instruction}")
        self._gate_map[instruction][qargs] = properties
        self._instruction_durations = None
        self._instruction_schedule_map = None

    def update_from_instruction_schedule_map(self, inst_map, inst_name_map=None, error_dict=None):
        """Update the target from an instruction schedule map.

        If the input instruction schedule map contains new instructions not in
        the target they will be added. However, if it contains additional qargs
        for an existing instruction in the target it will error.

        Args:
            inst_map (InstructionScheduleMap): The instruction
            inst_name_map (dict): An optional dictionary that maps any
                instruction name in ``inst_map`` to an instruction object.
                If not provided, instruction is pulled from the standard Qiskit gates,
                and finally custom gate instance is created with schedule name.
            error_dict (dict): A dictionary of errors of the form::

                {gate_name: {qarg: error}}

            for example::

                {'rx': {(0, ): 1.4e-4, (1, ): 1.2e-4}}

            For each entry in the ``inst_map`` if ``error_dict`` is defined
            a when updating the ``Target`` the error value will be pulled from
            this dictionary. If one is not found in ``error_dict`` then
            ``None`` will be used.
        """
        get_calibration = getattr(inst_map, "_get_calibration_entry")

        # Expand name mapping with custom gate name provided by user.
        qiskit_inst_name_map = get_standard_gate_name_mapping()
        if inst_name_map is not None:
            qiskit_inst_name_map.update(inst_name_map)

        for inst_name in inst_map.instructions:
            # Prepare dictionary of instruction properties
            out_props = {}
            for qargs in inst_map.qubits_with_instruction(inst_name):
                try:
                    qargs = tuple(qargs)
                except TypeError:
                    qargs = (qargs,)
                try:
                    props = self._gate_map[inst_name][qargs]
                except (KeyError, TypeError):
                    props = None

                entry = get_calibration(inst_name, qargs)
                if entry.user_provided and getattr(props, "_calibration", None) != entry:
                    # It only copies user-provided calibration from the inst map.
                    # Backend defined entry must already exist in Target.
                    if self.dt is not None:
                        try:
                            duration = entry.get_schedule().duration * self.dt
                        except UnassignedDurationError:
                            # duration of schedule is parameterized
                            duration = None
                    else:
                        duration = None
                    props = InstructionProperties(
                        duration=duration,
                        calibration=entry,
                    )
                else:
                    if props is None:
                        # Edge case. Calibration is backend defined, but this is not
                        # registered in the backend target. Ignore this entry.
                        continue
                try:
                    # Update gate error if provided.
                    props.error = error_dict[inst_name][qargs]
                except (KeyError, TypeError):
                    pass
                out_props[qargs] = props
            if not out_props:
                continue
            # Prepare Qiskit Gate object assigned to the entries
            if inst_name not in self._gate_map:
                # Entry not found: Add new instruction
                if inst_name in qiskit_inst_name_map:
                    # Remove qargs with length that doesn't match with instruction qubit number
                    inst_obj = qiskit_inst_name_map[inst_name]
                    normalized_props = {}
                    for qargs, prop in out_props.items():
                        if len(qargs) != inst_obj.num_qubits:
                            continue
                        normalized_props[qargs] = prop
                    self.add_instruction(inst_obj, normalized_props, name=inst_name)
                else:
                    # Check qubit length parameter name uniformity.
                    qlen = set()
                    param_names = set()
                    for qargs in inst_map.qubits_with_instruction(inst_name):
                        if isinstance(qargs, int):
                            qargs = (qargs,)
                        qlen.add(len(qargs))
                        cal = getattr(out_props[tuple(qargs)], "_calibration")
                        param_names.add(tuple(cal.get_signature().parameters.keys()))
                    if len(qlen) > 1 or len(param_names) > 1:
                        raise QiskitError(
                            f"Schedules for {inst_name} are defined non-uniformly for "
                            f"multiple qubit lengths {qlen}, "
                            f"or different parameter names {param_names}. "
                            "Provide these schedules with inst_name_map or define them with "
                            "different names for different gate parameters."
                        )
                    inst_obj = Gate(
                        name=inst_name,
                        num_qubits=next(iter(qlen)),
                        params=list(map(Parameter, next(iter(param_names)))),
                    )
                    self.add_instruction(inst_obj, out_props, name=inst_name)
            else:
                # Entry found: Update "existing" instructions.
                for qargs, prop in out_props.items():
                    if qargs not in self._gate_map[inst_name]:
                        continue
                    self.update_instruction_properties(inst_name, qargs, prop)

    @property
    def qargs(self):
        """The set of qargs in the target."""
        qargs = set(self._qarg_gate_map)
        if len(qargs) == 1 and next(iter(qargs)) is None:
            return None
        return qargs

    def qargs_for_operation_name(self, operation):
        """Get the qargs for a given operation name

        Args:
           operation (str): The operation name to get qargs for
        Returns:
            set: The set of qargs the gate instance applies to.
        """
        if None in self._gate_map[operation]:
            return None
        return self._gate_map[operation].keys()

    def durations(self):
        """Get an InstructionDurations object from the target

        Returns:
            InstructionDurations: The instruction duration represented in the
                target
        """
        if self._instruction_durations is not None:
            return self._instruction_durations
        out_durations = []
        for instruction, props_map in self._gate_map.items():
            for qarg, properties in props_map.items():
                if properties is not None and properties.duration is not None:
                    out_durations.append((instruction, list(qarg), properties.duration, "s"))
        self._instruction_durations = InstructionDurations(out_durations, dt=self.dt)
        return self._instruction_durations

    def timing_constraints(self):
        """Get an :class:`~qiskit.transpiler.TimingConstraints` object from the target

        Returns:
            TimingConstraints: The timing constraints represented in the ``Target``
        """
        return TimingConstraints(
            self.granularity, self.min_length, self.pulse_alignment, self.acquire_alignment
        )

    def instruction_schedule_map(self):
        """Return an :class:`~qiskit.pulse.InstructionScheduleMap` for the
        instructions in the target with a pulse schedule defined.

        Returns:
            InstructionScheduleMap: The instruction schedule map for the
            instructions in this target with a pulse schedule defined.
        """
        if self._instruction_schedule_map is not None:
            return self._instruction_schedule_map
        out_inst_schedule_map = InstructionScheduleMap()
        for instruction, qargs in self._gate_map.items():
            for qarg, properties in qargs.items():
                # Directly getting CalibrationEntry not to invoke .get_schedule().
                # This keeps PulseQobjDef un-parsed.
                cal_entry = getattr(properties, "_calibration", None)
                if cal_entry is not None:
                    # Use fast-path to add entries to the inst map.
                    out_inst_schedule_map._add(instruction, qarg, cal_entry)
        self._instruction_schedule_map = out_inst_schedule_map
        return out_inst_schedule_map

    def operation_from_name(self, instruction):
        """Get the operation class object for a given name

        Args:
            instruction (str): The instruction name to get the
                :class:`~qiskit.circuit.Instruction` instance for
        Returns:
            qiskit.circuit.Instruction: The Instruction instance corresponding to the
            name. This also can also be the class for globally defined variable with
            operations.
        """
        return self._gate_name_map[instruction]

    def operations_for_qargs(self, qargs):
        """Get the operation class object for a specified qargs tuple

        Args:
            qargs (tuple): A qargs tuple of the qubits to get the gates that apply
                to it. For example, ``(0,)`` will return the set of all
                instructions that apply to qubit 0. If set to ``None`` this will
                return any globally defined operations in the target.
        Returns:
            list: The list of :class:`~qiskit.circuit.Instruction` instances
            that apply to the specified qarg. This may also be a class if
            a variable width operation is globally defined.

        Raises:
            KeyError: If qargs is not in target
        """
        if qargs is not None and any(x not in range(0, self.num_qubits) for x in qargs):
            raise KeyError(f"{qargs} not in target.")
        res = [self._gate_name_map[x] for x in self._qarg_gate_map[qargs]]
        if qargs is not None:
            res += self._global_operations.get(len(qargs), [])
        for op in self._gate_name_map.values():
            if inspect.isclass(op):
                res.append(op)
        if not res:
            raise KeyError(f"{qargs} not in target.")
        return list(res)

    def operation_names_for_qargs(self, qargs):
        """Get the operation names for a specified qargs tuple

        Args:
            qargs (tuple): A ``qargs`` tuple of the qubits to get the gates that apply
                to it. For example, ``(0,)`` will return the set of all
                instructions that apply to qubit 0. If set to ``None`` this will
                return the names for any globally defined operations in the target.
        Returns:
            set: The set of operation names that apply to the specified ``qargs``.

        Raises:
            KeyError: If ``qargs`` is not in target
        """
        # if num_qubits == 0, we will return globally defined operations
        if self.num_qubits == 0 or self.num_qubits is None:
            qargs = None
        if qargs is not None and any(x not in range(0, self.num_qubits) for x in qargs):
            raise KeyError(f"{qargs} not in target.")
        res = self._qarg_gate_map.get(qargs, set())
        if qargs is not None:
            res.update(self._global_operations.get(len(qargs), set()))
        for name, op in self._gate_name_map.items():
            if inspect.isclass(op):
                res.add(name)
        if not res:
            raise KeyError(f"{qargs} not in target.")
        return res

    def instruction_supported(
        self, operation_name=None, qargs=None, operation_class=None, parameters=None
    ):
        """Return whether the instruction (operation + qubits) is supported by the target

        Args:
            operation_name (str): The name of the operation for the instruction. Either
                this or ``operation_class`` must be specified, if both are specified
                ``operation_class`` will take priority and this argument will be ignored.
            qargs (tuple): The tuple of qubit indices for the instruction. If this is
                not specified then this method will return ``True`` if the specified
                operation is supported on any qubits. The typical application will
                always have this set (otherwise it's the same as just checking if the
                target contains the operation). Normally you would not set this argument
                if you wanted to check more generally that the target supports an operation
                with the ``parameters`` on any qubits.
            operation_class (Type[qiskit.circuit.Instruction]): The operation class to check whether
                the target supports a particular operation by class rather
                than by name. This lookup is more expensive as it needs to
                iterate over all operations in the target instead of just a
                single lookup. If this is specified it will supersede the
                ``operation_name`` argument. The typical use case for this
                operation is to check whether a specific variant of an operation
                is supported on the backend. For example, if you wanted to
                check whether a :class:`~.RXGate` was supported on a specific
                qubit with a fixed angle. That fixed angle variant will
                typically have a name different from the object's
                :attr:`~.Instruction.name` attribute (``"rx"``) in the target.
                This can be used to check if any instances of the class are
                available in such a case.
            parameters (list): A list of parameters to check if the target
                supports them on the specified qubits. If the instruction
                supports the parameter values specified in the list on the
                operation and qargs specified this will return ``True`` but
                if the parameters are not supported on the specified
                instruction it will return ``False``. If this argument is not
                specified this method will return ``True`` if the instruction
                is supported independent of the instruction parameters. If
                specified with any :class:`~.Parameter` objects in the list,
                that entry will be treated as supporting any value, however parameter names
                will not be checked (for example if an operation in the target
                is listed as parameterized with ``"theta"`` and ``"phi"`` is
                passed into this function that will return ``True``). For
                example, if called with::

                    parameters = [Parameter("theta")]
                    target.instruction_supported("rx", (0,), parameters=parameters)

                will return ``True`` if an :class:`~.RXGate` is supported on qubit 0
                that will accept any parameter. If you need to check for a fixed numeric
                value parameter this argument is typically paired with the ``operation_class``
                argument. For example::

                    target.instruction_supported("rx", (0,), RXGate, parameters=[pi / 4])

                will return ``True`` if an RXGate(pi/4) exists on qubit 0.

        Returns:
            bool: Returns ``True`` if the instruction is supported and ``False`` if it isn't.

        """

        def check_obj_params(parameters, obj):
            for index, param in enumerate(parameters):
                if isinstance(param, Parameter) and not isinstance(obj.params[index], Parameter):
                    return False
                if param != obj.params[index] and not isinstance(obj.params[index], Parameter):
                    return False
            return True

        # Handle case where num_qubits is None by always checking globally supported operations
        if self.num_qubits is None:
            qargs = None
        # Case a list if passed in by mistake
        if qargs is not None:
            qargs = tuple(qargs)
        if operation_class is not None:
            for op_name, obj in self._gate_name_map.items():
                if inspect.isclass(obj):
                    if obj != operation_class:
                        continue
                    # If no qargs operation class is supported
                    if qargs is None:
                        return True
                    # If qargs set then validate no duplicates and all indices are valid on device
                    elif all(qarg <= self.num_qubits for qarg in qargs) and len(set(qargs)) == len(
                        qargs
                    ):
                        return True
                    else:
                        return False

                if isinstance(obj, operation_class):
                    if parameters is not None:
                        if len(parameters) != len(obj.params):
                            continue
                        if not check_obj_params(parameters, obj):
                            continue
                    if qargs is None:
                        return True
                    if qargs in self._gate_map[op_name]:
                        return True
                    if self._gate_map[op_name] is None or None in self._gate_map[op_name]:
                        return obj.num_qubits == len(qargs) and all(
                            x < self.num_qubits for x in qargs
                        )
            return False
        if operation_name in self._gate_map:
            if parameters is not None:
                obj = self._gate_name_map[operation_name]
                if inspect.isclass(obj):
                    # The parameters argument was set and the operation_name specified is
                    # defined as a globally supported class in the target. This means
                    # there is no available validation (including whether the specified
                    # operation supports parameters), the returned value will not factor
                    # in the argument `parameters`,

                    # If no qargs a operation class is supported
                    if qargs is None:
                        return True
                    # If qargs set then validate no duplicates and all indices are valid on device
                    elif all(qarg <= self.num_qubits for qarg in qargs) and len(set(qargs)) == len(
                        qargs
                    ):
                        return True
                    else:
                        return False
                if len(parameters) != len(obj.params):
                    return False
                for index, param in enumerate(parameters):
                    matching_param = False
                    if isinstance(obj.params[index], Parameter):
                        matching_param = True
                    elif param == obj.params[index]:
                        matching_param = True
                    if not matching_param:
                        return False
                return True
            if qargs is None:
                return True
            if qargs in self._gate_map[operation_name]:
                return True
            if self._gate_map[operation_name] is None or None in self._gate_map[operation_name]:
                obj = self._gate_name_map[operation_name]
                if inspect.isclass(obj):
                    if qargs is None:
                        return True
                    # If qargs set then validate no duplicates and all indices are valid on device
                    elif all(qarg <= self.num_qubits for qarg in qargs) and len(set(qargs)) == len(
                        qargs
                    ):
                        return True
                    else:
                        return False
                else:
                    return self._gate_name_map[operation_name].num_qubits == len(qargs) and all(
                        x < self.num_qubits for x in qargs
                    )
        return False

    def has_calibration(
        self,
        operation_name: str,
        qargs: tuple[int, ...],
    ) -> bool:
        """Return whether the instruction (operation + qubits) defines a calibration.

        Args:
            operation_name: The name of the operation for the instruction.
            qargs: The tuple of qubit indices for the instruction.

        Returns:
            Returns ``True`` if the calibration is supported and ``False`` if it isn't.
        """
        qargs = tuple(qargs)
        if operation_name not in self._gate_map:
            return False
        if qargs not in self._gate_map[operation_name]:
            return False
        return getattr(self._gate_map[operation_name][qargs], "_calibration", None) is not None

    def get_calibration(
        self,
        operation_name: str,
        qargs: tuple[int, ...],
        *args: ParameterValueType,
        **kwargs: ParameterValueType,
    ) -> Schedule | ScheduleBlock:
        """Get calibrated pulse schedule for the instruction.

        If calibration is templated with parameters, one can also provide those values
        to build a schedule with assigned parameters.

        Args:
            operation_name: The name of the operation for the instruction.
            qargs: The tuple of qubit indices for the instruction.
            args: Parameter values to build schedule if any.
            kwargs: Parameter values with name to build schedule if any.

        Returns:
            Calibrated pulse schedule of corresponding instruction.
        """
        if not self.has_calibration(operation_name, qargs):
            raise KeyError(
                f"Calibration of instruction {operation_name} for qubit {qargs} is not defined."
            )
        cal_entry = getattr(self._gate_map[operation_name][qargs], "_calibration")
        return cal_entry.get_schedule(*args, **kwargs)

    @property
    def operation_names(self):
        """Get the operation names in the target."""
        return self._gate_map.keys()

    @property
    def operations(self):
        """Get the operation class objects in the target."""
        return list(self._gate_name_map.values())

    @property
    def instructions(self):
        """Get the list of tuples ``(:class:`~qiskit.circuit.Instruction`, (qargs))``
        for the target

        For globally defined variable width operations the tuple will be of the form
        ``(class, None)`` where class is the actual operation class that
        is globally defined.
        """
        return [
            (self._gate_name_map[op], qarg)
            for op, qargs in self._gate_map.items()
            for qarg in qargs
        ]

    def instruction_properties(self, index):
        """Get the instruction properties for a specific instruction tuple

        This method is to be used in conjunction with the
        :attr:`~qiskit.transpiler.Target.instructions` attribute of a
        :class:`~qiskit.transpiler.Target` object. You can use this method to quickly
        get the instruction properties for an element of
        :attr:`~qiskit.transpiler.Target.instructions` by using the index in that list.
        However, if you're not working with :attr:`~qiskit.transpiler.Target.instructions`
        directly it is likely more efficient to access the target directly via the name
        and qubits to get the instruction properties. For example, if
        :attr:`~qiskit.transpiler.Target.instructions` returned::

            [(XGate(), (0,)), (XGate(), (1,))]

        you could get the properties of the ``XGate`` on qubit 1 with::

            props = target.instruction_properties(1)

        but just accessing it directly via the name would be more efficient::

            props = target['x'][(1,)]

        (assuming the ``XGate``'s canonical name in the target is ``'x'``)
        This is especially true for larger targets as this will scale worse with the number
        of instruction tuples in a target.

        Args:
            index (int): The index of the instruction tuple from the
                :attr:`~qiskit.transpiler.Target.instructions` attribute. For, example
                if you want the properties from the third element in
                :attr:`~qiskit.transpiler.Target.instructions` you would set this to be ``2``.
        Returns:
            InstructionProperties: The instruction properties for the specified instruction tuple
        """
        instruction_properties = [
            inst_props for qargs in self._gate_map.values() for inst_props in qargs.values()
        ]
        return instruction_properties[index]

    def _build_coupling_graph(self):
        self._coupling_graph = rx.PyDiGraph(multigraph=False)
        if self.num_qubits is not None:
            self._coupling_graph.add_nodes_from([{} for _ in range(self.num_qubits)])
        for gate, qarg_map in self._gate_map.items():
            if qarg_map is None:
                if self._gate_name_map[gate].num_qubits == 2:
                    self._coupling_graph = None
                    return
                continue
            for qarg, properties in qarg_map.items():
                if qarg is None:
                    if self._gate_name_map[gate].num_qubits == 2:
                        self._coupling_graph = None
                        return
                    continue
                if len(qarg) == 1:
                    self._coupling_graph[qarg[0]] = properties
                elif len(qarg) == 2:
                    try:
                        edge_data = self._coupling_graph.get_edge_data(*qarg)
                        edge_data[gate] = properties
                    except rx.NoEdgeBetweenNodes:
                        self._coupling_graph.add_edge(*qarg, {gate: properties})
        if self._coupling_graph.num_edges() == 0 and any(x is None for x in self._qarg_gate_map):
            self._coupling_graph = None

    def build_coupling_map(self, two_q_gate=None, filter_idle_qubits=False):
        """Get a :class:`~qiskit.transpiler.CouplingMap` from this target.

        If there is a mix of two qubit operations that have a connectivity
        constraint and those that are globally defined this will also return
        ``None`` because the globally connectivity means there is no constraint
        on the target. If you wish to see the constraints of the two qubit
        operations that have constraints you should use the ``two_q_gate``
        argument to limit the output to the gates which have a constraint.

        Args:
            two_q_gate (str): An optional gate name for a two qubit gate in
                the ``Target`` to generate the coupling map for. If specified the
                output coupling map will only have edges between qubits where
                this gate is present.
            filter_idle_qubits (bool): If set to ``True`` the output :class:`~.CouplingMap`
                will remove any qubits that don't have any operations defined in the
                target. Note that using this argument will result in an output
                :class:`~.CouplingMap` object which has holes in its indices
                which might differ from the assumptions of the class. The typical use
                case of this argument is to be paired with
                :meth:`.CouplingMap.connected_components` which will handle the holes
                as expected.
        Returns:
            CouplingMap: The :class:`~qiskit.transpiler.CouplingMap` object
                for this target. If there are no connectivity constraints in
                the target this will return ``None``.

        Raises:
            ValueError: If a non-two qubit gate is passed in for ``two_q_gate``.
            IndexError: If an Instruction not in the ``Target`` is passed in for
                ``two_q_gate``.
        """
        if self.qargs is None:
            return None
        if None not in self.qargs and any(len(x) > 2 for x in self.qargs):
            logger.warning(
                "This Target object contains multiqubit gates that "
                "operate on > 2 qubits. This will not be reflected in "
                "the output coupling map."
            )

        if two_q_gate is not None:
            coupling_graph = rx.PyDiGraph(multigraph=False)
            coupling_graph.add_nodes_from([None] * self.num_qubits)
            for qargs, properties in self._gate_map[two_q_gate].items():
                if len(qargs) != 2:
                    raise ValueError(
                        f"Specified two_q_gate: {two_q_gate} is not a 2 qubit instruction"
                    )
                coupling_graph.add_edge(*qargs, {two_q_gate: properties})
            cmap = CouplingMap()
            cmap.graph = coupling_graph
            return cmap
        if self._coupling_graph is None:
            self._build_coupling_graph()
        # if there is no connectivity constraints in the coupling graph treat it as not
        # existing and return
        if self._coupling_graph is not None:
            cmap = CouplingMap()
            if filter_idle_qubits:
                cmap.graph = self._filter_coupling_graph()
            else:
                cmap.graph = self._coupling_graph.copy()
            return cmap
        else:
            return None

    def _filter_coupling_graph(self):
        has_operations = set(itertools.chain.from_iterable(x for x in self.qargs if x is not None))
        graph = self._coupling_graph.copy()
        to_remove = set(graph.node_indices()).difference(has_operations)
        if to_remove:
            graph.remove_nodes_from(list(to_remove))
        return graph

    @property
    def physical_qubits(self):
        """Returns a sorted list of physical_qubits"""
        return list(range(self.num_qubits))

    def get_non_global_operation_names(self, strict_direction: bool = False) -> list[str]:
        """Return the non-global operation names for the target

        The non-global operations are those in the target which don't apply
        on all qubits (for single qubit operations) or all multi-qubit qargs
        (for multi-qubit operations).

        Args:
            strict_direction (bool): If set to ``True`` the multi-qubit
                operations considered as non-global respect the strict
                direction (or order of qubits in the qargs is significant). For
                example, if ``cx`` is defined on ``(0, 1)`` and ``ecr`` is
                defined over ``(1, 0)`` by default neither would be considered
                non-global, but if ``strict_direction`` is set ``True`` both
                ``cx`` and ``ecr`` would be returned.

        Returns:
            List[str]: A list of operation names for operations that aren't global in this target
        """
        if strict_direction:
            if self._non_global_strict_basis is not None:
                return self._non_global_strict_basis
            search_set = self._qarg_gate_map.keys()
        else:
            if self._non_global_basis is not None:
                return self._non_global_basis

            search_set = {
                frozenset(qarg)
                for qarg in self._qarg_gate_map
                if qarg is not None and len(qarg) != 1
            }
        incomplete_basis_gates = []
        size_dict = defaultdict(int)
        size_dict[1] = self.num_qubits
        for qarg in search_set:
            if qarg is None or len(qarg) == 1:
                continue
            size_dict[len(qarg)] += 1
        for inst, qargs in self._gate_map.items():
            qarg_sample = next(iter(qargs))
            if qarg_sample is None:
                continue
            if not strict_direction:
                qargs = {frozenset(qarg) for qarg in qargs}
            if len(qargs) != size_dict[len(qarg_sample)]:
                incomplete_basis_gates.append(inst)
        if strict_direction:
            self._non_global_strict_basis = incomplete_basis_gates
        else:
            self._non_global_basis = incomplete_basis_gates
        return incomplete_basis_gates

    def __iter__(self):
        return iter(self._gate_map)

    def __getitem__(self, key):
        return self._gate_map[key]

    def __len__(self):
        return len(self._gate_map)

    def __contains__(self, item):
        return item in self._gate_map

    def keys(self):
        return self._gate_map.keys()

    def values(self):
        return self._gate_map.values()

    def items(self):
        return self._gate_map.items()

    def __str__(self):
        output = io.StringIO()
        if self.description is not None:
            output.write(f"Target: {self.description}\n")
        else:
            output.write("Target\n")
        output.write(f"Number of qubits: {self.num_qubits}\n")
        output.write("Instructions:\n")
        for inst, qarg_props in self._gate_map.items():
            output.write(f"\t{inst}\n")
            for qarg, props in qarg_props.items():
                if qarg is None:
                    continue
                if props is None:
                    output.write(f"\t\t{qarg}\n")
                    continue
                prop_str_pieces = [f"\t\t{qarg}:\n"]
                duration = getattr(props, "duration", None)
                if duration is not None:
                    prop_str_pieces.append(f"\t\t\tDuration: {duration} sec.\n")
                error = getattr(props, "error", None)
                if error is not None:
                    prop_str_pieces.append(f"\t\t\tError Rate: {error}\n")
                schedule = getattr(props, "_calibration", None)
                if schedule is not None:
                    prop_str_pieces.append("\t\t\tWith pulse schedule calibration\n")
                extra_props = getattr(props, "properties", None)
                if extra_props is not None:
                    extra_props_pieces = [
                        f"\t\t\t\t{key}: {value}\n" for key, value in extra_props.items()
                    ]
                    extra_props_str = "".join(extra_props_pieces)
                    prop_str_pieces.append(f"\t\t\tExtra properties:\n{extra_props_str}\n")
                output.write("".join(prop_str_pieces))
        return output.getvalue()

    @classmethod
    def from_configuration(
        cls,
        basis_gates: list[str],
        num_qubits: int | None = None,
        coupling_map: CouplingMap | None = None,
        inst_map: InstructionScheduleMap | None = None,
        backend_properties: BackendProperties | None = None,
        instruction_durations: InstructionDurations | None = None,
        concurrent_measurements: Optional[List[List[int]]] = None,
        dt: float | None = None,
        timing_constraints: TimingConstraints | None = None,
        custom_name_mapping: dict[str, Any] | None = None,
    ) -> Target:
        """Create a target object from the individual global configuration

        Prior to the creation of the :class:`~.Target` class, the constraints
        of a backend were represented by a collection of different objects
        which combined represent a subset of the information contained in
        the :class:`~.Target`. This function provides a simple interface
        to convert those separate objects to a :class:`~.Target`.

        This constructor will use the input from ``basis_gates``, ``num_qubits``,
        and ``coupling_map`` to build a base model of the backend and the
        ``instruction_durations``, ``backend_properties``, and ``inst_map`` inputs
        are then queried (in that order) based on that model to look up the properties
        of each instruction and qubit. If there is an inconsistency between the inputs
        any extra or conflicting information present in ``instruction_durations``,
        ``backend_properties``, or ``inst_map`` will be ignored.

        Args:
            basis_gates: The list of basis gate names for the backend. For the
                target to be created these names must either be in the output
                from :func:`~.get_standard_gate_name_mapping` or present in the
                specified ``custom_name_mapping`` argument.
            num_qubits: The number of qubits supported on the backend.
            coupling_map: The coupling map representing connectivity constraints
                on the backend. If specified all gates from ``basis_gates`` will
                be supported on all qubits (or pairs of qubits).
            inst_map: The instruction schedule map representing the pulse
               :class:`~.Schedule` definitions for each instruction. If this
               is specified ``coupling_map`` must be specified. The
               ``coupling_map`` is used as the source of truth for connectivity
               and if ``inst_map`` is used the schedule is looked up based
               on the instructions from the pair of ``basis_gates`` and
               ``coupling_map``. If you want to define a custom gate for
               a particular qubit or qubit pair, you can manually build :class:`.Target`.
            backend_properties: The :class:`~.BackendProperties` object which is
                used for instruction properties and qubit properties.
                If specified and instruction properties are intended to be used
                then the ``coupling_map`` argument must be specified. This is
                only used to lookup error rates and durations (unless
                ``instruction_durations`` is specified which would take
                precedence) for instructions specified via ``coupling_map`` and
                ``basis_gates``.
            instruction_durations: Optional instruction durations for instructions. If specified
                it will take priority for setting the ``duration`` field in the
                :class:`~InstructionProperties` objects for the instructions in the target.
            concurrent_measurements(list): A list of sets of qubits that must be
                measured together. This must be provided
                as a nested list like ``[[0, 1], [2, 3, 4]]``.
            dt: The system time resolution of input signals in seconds
            timing_constraints: Optional timing constraints to include in the
                :class:`~.Target`
            custom_name_mapping: An optional dictionary that maps custom gate/operation names in
                ``basis_gates`` to an :class:`~.Operation` object representing that
                gate/operation. By default, most standard gates names are mapped to the
                standard gate object from :mod:`qiskit.circuit.library` this only needs
                to be specified if the input ``basis_gates`` defines gates in names outside
                that set.

        Returns:
            Target: the target built from the input configuration

        Raises:
            TranspilerError: If the input basis gates contain > 2 qubits and ``coupling_map`` is
            specified.
            KeyError: If no mapping is available for a specified ``basis_gate``.
        """
        granularity = 1
        min_length = 1
        pulse_alignment = 1
        acquire_alignment = 1
        if timing_constraints is not None:
            granularity = timing_constraints.granularity
            min_length = timing_constraints.min_length
            pulse_alignment = timing_constraints.pulse_alignment
            acquire_alignment = timing_constraints.acquire_alignment

        qubit_properties = None
        if backend_properties is not None:
            # pylint: disable=cyclic-import
            from qiskit.providers.backend_compat import qubit_props_list_from_props

            qubit_properties = qubit_props_list_from_props(properties=backend_properties)

        target = cls(
            num_qubits=num_qubits,
            dt=dt,
            granularity=granularity,
            min_length=min_length,
            pulse_alignment=pulse_alignment,
            acquire_alignment=acquire_alignment,
            qubit_properties=qubit_properties,
            concurrent_measurements=concurrent_measurements,
        )
        name_mapping = get_standard_gate_name_mapping()
        if custom_name_mapping is not None:
            name_mapping.update(custom_name_mapping)

        # While BackendProperties can also contain coupling information we
        # rely solely on CouplingMap to determine connectivity. This is because
        # in legacy transpiler usage (and implicitly in the BackendV1 data model)
        # the coupling map is used to define connectivity constraints and
        # the properties is only used for error rate and duration population.
        # If coupling map is not specified we ignore the backend_properties
        if coupling_map is None:
            for gate in basis_gates:
                if gate not in name_mapping:
                    raise KeyError(
                        f"The specified basis gate: {gate} is not present in the standard gate "
                        "names or a provided custom_name_mapping"
                    )
                target.add_instruction(name_mapping[gate], name=gate)
        else:
            one_qubit_gates = []
            two_qubit_gates = []
            global_ideal_variable_width_gates = []  # pylint: disable=invalid-name
            if num_qubits is None:
                num_qubits = len(coupling_map.graph)
            for gate in basis_gates:
                if gate not in name_mapping:
                    raise KeyError(
                        f"The specified basis gate: {gate} is not present in the standard gate "
                        "names or a provided custom_name_mapping"
                    )
                gate_obj = name_mapping[gate]
                if gate_obj.num_qubits == 1:
                    one_qubit_gates.append(gate)
                elif gate_obj.num_qubits == 2:
                    two_qubit_gates.append(gate)
                elif inspect.isclass(gate_obj):
                    global_ideal_variable_width_gates.append(gate)
                else:
                    raise TranspilerError(
                        f"The specified basis gate: {gate} has {gate_obj.num_qubits} "
                        "qubits. This constructor method only supports fixed width operations "
                        "with <= 2 qubits (because connectivity is defined on a CouplingMap)."
                    )
            for gate in one_qubit_gates:
                gate_properties: dict[tuple, InstructionProperties] = {}
                for qubit in range(num_qubits):
                    error = None
                    duration = None
                    calibration = None
                    if backend_properties is not None:
                        if duration is None:
                            try:
                                duration = backend_properties.gate_length(gate, qubit)
                            except BackendPropertyError:
                                duration = None
                        try:
                            error = backend_properties.gate_error(gate, qubit)
                        except BackendPropertyError:
                            error = None
                    if inst_map is not None:
                        try:
                            calibration = inst_map._get_calibration_entry(gate, qubit)
                            # If we have dt defined and there is a custom calibration which is user
                            # generate use that custom pulse schedule for the duration. If it is
                            # not user generated than we assume it's the same duration as what is
                            # defined in the backend properties
                            if dt and calibration.user_provided:
                                duration = calibration.get_schedule().duration * dt
                        except PulseError:
                            calibration = None
                    # Durations if specified manually should override model objects
                    if instruction_durations is not None:
                        try:
                            duration = instruction_durations.get(gate, qubit, unit="s")
                        except TranspilerError:
                            duration = None

                    if error is None and duration is None and calibration is None:
                        gate_properties[(qubit,)] = None
                    else:
                        gate_properties[(qubit,)] = InstructionProperties(
                            duration=duration, error=error, calibration=calibration
                        )
                target.add_instruction(name_mapping[gate], properties=gate_properties, name=gate)
            edges = list(coupling_map.get_edges())
            for gate in two_qubit_gates:
                gate_properties = {}
                for edge in edges:
                    error = None
                    duration = None
                    calibration = None
                    if backend_properties is not None:
                        if duration is None:
                            try:
                                duration = backend_properties.gate_length(gate, edge)
                            except BackendPropertyError:
                                duration = None
                        try:
                            error = backend_properties.gate_error(gate, edge)
                        except BackendPropertyError:
                            error = None
                    if inst_map is not None:
                        try:
                            calibration = inst_map._get_calibration_entry(gate, edge)
                            # If we have dt defined and there is a custom calibration which is user
                            # generate use that custom pulse schedule for the duration. If it is
                            # not user generated than we assume it's the same duration as what is
                            # defined in the backend properties
                            if dt and calibration.user_provided:
                                duration = calibration.get_schedule().duration * dt
                        except PulseError:
                            calibration = None
                    # Durations if specified manually should override model objects
                    if instruction_durations is not None:
                        try:
                            duration = instruction_durations.get(gate, edge, unit="s")
                        except TranspilerError:
                            duration = None

                    if error is None and duration is None and calibration is None:
                        gate_properties[edge] = None
                    else:
                        gate_properties[edge] = InstructionProperties(
                            duration=duration, error=error, calibration=calibration
                        )
                target.add_instruction(name_mapping[gate], properties=gate_properties, name=gate)
            for gate in global_ideal_variable_width_gates:
                target.add_instruction(name_mapping[gate], name=gate)
        return target


def target_to_backend_properties(target: Target):
    """Convert a :class:`~.Target` object into a legacy :class:`~.BackendProperties`"""

    properties_dict: dict[str, Any] = {
        "backend_name": "",
        "backend_version": "",
        "last_update_date": None,
        "general": [],
    }
    gates = []
    qubits = []
    for gate, qargs_list in target.items():
        if gate != "measure":
            for qargs, props in qargs_list.items():
                property_list = []
                if getattr(props, "duration", None) is not None:
                    property_list.append(
                        {
                            "date": datetime.datetime.now(datetime.timezone.utc),
                            "name": "gate_length",
                            "unit": "s",
                            "value": props.duration,
                        }
                    )
                if getattr(props, "error", None) is not None:
                    property_list.append(
                        {
                            "date": datetime.datetime.now(datetime.timezone.utc),
                            "name": "gate_error",
                            "unit": "",
                            "value": props.error,
                        }
                    )
                if property_list:
                    gates.append(
                        {
                            "gate": gate,
                            "qubits": list(qargs),
                            "parameters": property_list,
                            "name": gate + "_".join([str(x) for x in qargs]),
                        }
                    )
        else:
            qubit_props: dict[int, Any] = {}
            if target.num_qubits is not None:
                qubit_props = {x: None for x in range(target.num_qubits)}
            for qargs, props in qargs_list.items():
                if qargs is None:
                    continue
                qubit = qargs[0]
                props_list = []
                if getattr(props, "error", None) is not None:
                    props_list.append(
                        {
                            "date": datetime.datetime.now(datetime.timezone.utc),
                            "name": "readout_error",
                            "unit": "",
                            "value": props.error,
                        }
                    )
                if getattr(props, "duration", None) is not None:
                    props_list.append(
                        {
                            "date": datetime.datetime.now(datetime.timezone.utc),
                            "name": "readout_length",
                            "unit": "s",
                            "value": props.duration,
                        }
                    )
                if not props_list:
                    qubit_props = {}
                    break
                qubit_props[qubit] = props_list
            if qubit_props and all(x is not None for x in qubit_props.values()):
                qubits = [qubit_props[i] for i in range(target.num_qubits)]
    if gates or qubits:
        properties_dict["gates"] = gates
        properties_dict["qubits"] = qubits
        return BackendProperties.from_dict(properties_dict)
    else:
        return None

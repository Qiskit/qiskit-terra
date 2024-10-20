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

# pylint: disable=invalid-name

""""Management of pulse program parameters.

Background
==========

In contrast to ``QuantumCircuit``, in pulse programs, parameter objects can be stored in
multiple places at different layers, for example

- program variables: ``ScheduleBlock.alignment_context._context_params``

- instruction operands: ``ShiftPhase.phase``, ...

- operand parameters: ``pulse.parameters``, ``channel.index`` ...

This complexity is due to the tight coupling of the program to an underlying device Hamiltonian,
i.e. the variance of physical parameters between qubits and their couplings.
If we want to define a program that can be used with arbitrary qubits,
we should be able to parametrize every control parameter in the program.

Implementation
==============

Managing parameters in each object within a program, i.e. the ``ParameterTable`` model,
makes the framework quite complicated. With the ``ParameterManager`` class within this module,
the parameter assignment operation is performed by a visitor instance.

The visitor pattern is a way of separating data processing from the object on which it operates.
This removes the overhead of parameter management from each piece of the program.
The computational complexity of the parameter assignment operation may be increased
from the parameter table model of ~O(1), however, usually, this calculation occurs
only once before the program is executed. Thus this doesn't hurt user experience during
pulse programming. On the contrary, it removes parameter table object and associated logic
from each object, yielding smaller object creation cost and higher performance
as the data amount scales.

Note that we don't need to write any parameter management logic for each object,
and thus this parameter framework gives greater scalability to the pulse module.
"""
from __future__ import annotations
from copy import copy
from typing import Any, Mapping, Sequence

from qiskit.circuit.parametervector import ParameterVector, ParameterVectorElement
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import instructions, channels
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library import SymbolicPulse, Waveform
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import AlignmentKind
from qiskit.pulse.utils import (
    format_parameter_value,
    _validate_parameter_vector,
    _validate_parameter_value,
)


class NodeVisitor:
    """A node visitor base class that walks instruction data in a pulse program and calls
    visitor functions for every node.

    Though this class implementation is based on Python AST, each node doesn't have
    a dedicated node class due to the lack of an abstract syntax tree for pulse programs in
    Qiskit. Instead of parsing pulse programs, this visitor class finds the associated visitor
    function based on class name of the instruction node, i.e. ``Play``, ``Call``, etc...
    The `.visit` method recursively checks superclass of given node since some parametrized
    components such as ``DriveChannel`` may share a common superclass with other subclasses.
    In this example, we can just define ``visit_Channel`` method instead of defining
    the same visitor function for every subclasses.

    Some instructions may have special logic or data structure to store parameter objects,
    and visitor functions for these nodes should be individually defined.

    Because pulse programs can be nested into another pulse program,
    the visitor function should be able to recursively call proper visitor functions.
    If visitor function is not defined for a given node, ``generic_visit``
    method is called. Usually, this method is provided for operating on object defined
    outside of the Qiskit Pulse module.
    """

    def visit(self, node: Any):
        """Visit a node."""
        visitor = self._get_visitor(type(node))
        return visitor(node)

    def _get_visitor(self, node_class):
        """A helper function to recursively investigate superclass visitor method."""
        if node_class == object:
            return self.generic_visit

        try:
            return getattr(self, f"visit_{node_class.__name__}")
        except AttributeError:
            # check super class
            return self._get_visitor(node_class.__base__)

    def visit_ScheduleBlock(self, node: ScheduleBlock):
        """Visit ``ScheduleBlock``. Recursively visit context blocks and overwrite.

        .. note:: ``ScheduleBlock`` can have parameters in blocks and its alignment.
        """
        raise NotImplementedError

    def visit_Schedule(self, node: Schedule):
        """Visit ``Schedule``. Recursively visit schedule children and overwrite."""
        raise NotImplementedError

    def generic_visit(self, node: Any):
        """Called if no explicit visitor function exists for a node."""
        raise NotImplementedError


class ParameterSetter(NodeVisitor):
    """Node visitor for parameter binding.

    This visitor is initialized with a dictionary of parameters to be assigned,
    and assign values to operands of nodes found.
    """

    def __init__(self, param_map: dict[ParameterExpression, ParameterValueType]):
        self._param_map = param_map

    # Top layer: Assign parameters to programs

    def visit_ScheduleBlock(self, node: ScheduleBlock):
        """Visit ``ScheduleBlock``. Recursively visit context blocks and overwrite.

        .. note:: ``ScheduleBlock`` can have parameters in blocks and its alignment.
        """
        node._alignment_context = self.visit_AlignmentKind(node.alignment_context)
        for elm in node._blocks:
            self.visit(elm)

        self._update_parameter_manager(node)
        return node

    def visit_Schedule(self, node: Schedule):
        """Visit ``Schedule``. Recursively visit schedule children and overwrite."""
        # accessing to private member
        # TODO: consider updating Schedule to handle this more gracefully
        node._Schedule__children = [(t0, self.visit(sched)) for t0, sched in node.instructions]
        node._renew_timeslots()

        self._update_parameter_manager(node)
        return node

    def visit_AlignmentKind(self, node: AlignmentKind):
        """Assign parameters to block's ``AlignmentKind`` specification."""
        new_parameters = tuple(self.visit(param) for param in node._context_params)
        node._context_params = new_parameters

        return node

    # Mid layer: Assign parameters to instructions

    def visit_Instruction(self, node: instructions.Instruction):
        """Assign parameters to general pulse instruction.

        .. note:: All parametrized object should be stored in the operands.
            Otherwise parameter cannot be detected.
        """
        if node.is_parameterized():
            node._operands = tuple(self.visit(op) for op in node.operands)

        return node

    # Lower layer: Assign parameters to operands

    def visit_Channel(self, node: channels.Channel):
        """Assign parameters to ``Channel`` object."""
        if node.is_parameterized():
            new_index = self._assign_parameter_expression(node.index)

            # validate
            if not isinstance(new_index, ParameterExpression):
                if not isinstance(new_index, int) or new_index < 0:
                    raise PulseError("Channel index must be a nonnegative integer")

            # return new instance to prevent accidentally override timeslots without evaluation
            return node.__class__(index=new_index)

        return node

    def visit_SymbolicPulse(self, node: SymbolicPulse):
        """Assign parameters to ``SymbolicPulse`` object."""
        if node.is_parameterized():
            # Assign duration
            if isinstance(node.duration, ParameterExpression):
                node.duration = self._assign_parameter_expression(node.duration)
            # Assign other parameters
            for name in node._params:
                pval = node._params[name]
                if isinstance(pval, ParameterExpression):
                    new_val = self._assign_parameter_expression(pval)
                    node._params[name] = new_val
            if not node.disable_validation:
                node.validate_parameters()

        return node

    def visit_Waveform(self, node: Waveform):
        """Assign parameters to ``Waveform`` object.

        .. node:: No parameter can be assigned to ``Waveform`` object.
        """
        return node

    def generic_visit(self, node: Any):
        """Assign parameters to object that doesn't belong to Qiskit Pulse module."""
        if isinstance(node, ParameterExpression):
            return self._assign_parameter_expression(node)
        else:
            return node

    def _assign_parameter_expression(self, param_expr: ParameterExpression):
        """A helper function to assign parameter value to parameter expression."""
        new_value = copy(param_expr)
        updated = param_expr.parameters & self._param_map.keys()
        for param in updated:
            new_value = new_value.assign(param, self._param_map[param])
        new_value = format_parameter_value(new_value)
        return new_value

    def _update_parameter_manager(self, node: Schedule | ScheduleBlock):
        """A helper function to update parameter manager of pulse program."""
        if not hasattr(node, "_parameter_manager"):
            raise PulseError(f"Node type {node.__class__.__name__} has no parameter manager.")

        param_manager = node._parameter_manager
        updated = param_manager.parameters & self._param_map.keys()

        new_parameters = set()
        for param in param_manager.parameters:
            if param not in updated:
                new_parameters.add(param)
                continue
            new_value = self._param_map[param]
            if isinstance(new_value, ParameterExpression):
                new_parameters |= new_value.parameters
        param_manager._parameters = new_parameters


class ParameterGetter(NodeVisitor):
    """Node visitor for parameter finding.

    This visitor initializes empty parameter array, and recursively visits nodes
    and add parameters found to the array.
    """

    def __init__(self):
        self.parameters = set()

    # Top layer: Get parameters from programs

    def visit_ScheduleBlock(self, node: ScheduleBlock):
        """Visit ``ScheduleBlock``. Recursively visit context blocks and search parameters.

        .. note:: ``ScheduleBlock`` can have parameters in blocks and its alignment.
        """
        # Note that node.parameters returns parameters of main program with subroutines.
        # The manager of main program is not aware of parameters in subroutines.
        self.parameters |= node._parameter_manager.parameters

    def visit_Schedule(self, node: Schedule):
        """Visit ``Schedule``. Recursively visit schedule children and search parameters."""
        self.parameters |= node.parameters

    def visit_AlignmentKind(self, node: AlignmentKind):
        """Get parameters from block's ``AlignmentKind`` specification."""
        for param in node._context_params:
            if isinstance(param, ParameterExpression):
                self.parameters |= param.parameters

    # Mid layer: Get parameters from instructions

    def visit_Instruction(self, node: instructions.Instruction):
        """Get parameters from general pulse instruction.

        .. note:: All parametrized object should be stored in the operands.
            Otherwise, parameter cannot be detected.
        """
        for op in node.operands:
            self.visit(op)

    # Lower layer: Get parameters from operands

    def visit_Channel(self, node: channels.Channel):
        """Get parameters from ``Channel`` object."""
        self.parameters |= node.parameters

    def visit_SymbolicPulse(self, node: SymbolicPulse):
        """Get parameters from ``SymbolicPulse`` object."""
        for op_value in node.parameters.values():
            if isinstance(op_value, ParameterExpression):
                self.parameters |= op_value.parameters

    def visit_Waveform(self, node: Waveform):
        """Get parameters from ``Waveform`` object.

        .. node:: No parameter can be assigned to ``Waveform`` object.
        """
        pass

    def generic_visit(self, node: Any):
        """Get parameters from object that doesn't belong to Qiskit Pulse module."""
        if isinstance(node, ParameterExpression):
            self.parameters |= node.parameters


class ParameterManager:
    """Helper class to manage parameter objects associated with arbitrary pulse programs.

    This object is implicitly initialized with the parameter object storage
    that stores parameter objects added to the parent pulse program.

    Parameter assignment logic is implemented based on the visitor pattern.
    Instruction data and its location are not directly associated with this object.
    """

    disable_parameter_validation = False

    def __init__(self):
        """Create new parameter table for pulse programs."""
        self._parameters = set()

    @property
    def parameters(self) -> set[Parameter]:
        """Parameters which determine the schedule behavior."""
        return self._parameters

    def clear(self):
        """Remove the parameters linked to this manager."""
        self._parameters.clear()

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return bool(self.parameters)

    def get_parameters(self, parameter_name: str) -> list[Parameter]:
        """Get parameter object bound to this schedule by string name.

        Because different ``Parameter`` objects can have the same name,
        this method returns a list of ``Parameter`` s for the provided name.

        Args:
            parameter_name: Name of parameter.

        Returns:
            Parameter objects that have corresponding name.
        """
        return [param for param in self.parameters if param.name == parameter_name]

    def assign_parameters(
        self,
        pulse_program: Any,
        value_dict: dict[
            ParameterExpression | ParameterVector | str,
            ParameterValueType | Sequence[ParameterValueType],
        ],
    ) -> Any:
        """Modify and return program data with parameters assigned according to the input.

        Args:
            pulse_program: Arbitrary pulse program associated with this manager instance.
            value_dict: A mapping from Parameters to either numeric values or another
                Parameter expression.

        Returns:
            Updated program data.
        """
        unrolled_value_dict = self._unroll_param_dict(value_dict)
        valid_map = {
            k: unrolled_value_dict[k] for k in unrolled_value_dict.keys() & self._parameters
        }
        if valid_map:
            visitor = ParameterSetter(param_map=valid_map)
            return visitor.visit(pulse_program)
        return pulse_program

    def update_parameter_table(self, new_node: Any):
        """A helper function to update parameter table with given data node.

        Args:
            new_node: A new data node to be added.
        """
        visitor = ParameterGetter()
        visitor.visit(new_node)
        self._parameters |= visitor.parameters

    def _unroll_param_dict(
        self,
        parameter_binds: Mapping[
            Parameter | ParameterVector | str, ParameterValueType | Sequence[ParameterValueType]
        ],
    ) -> Mapping[Parameter, ParameterValueType]:
        """
        Unroll parameter dictionary to a map from parameter to value.

        Args:
            parameter_binds: A dictionary from parameter to value or a list of values.

        Returns:
            A dictionary from parameter to value.
        """
        out = {}
        param_name_dict = {param.name: [] for param in self.parameters}
        for param in self.parameters:
            param_name_dict[param.name].append(param)
        param_vec_dict = {
            param.vector.name: param.vector
            for param in self.parameters
            if isinstance(param, ParameterVectorElement)
        }
        for name in param_vec_dict.keys():
            if name in param_name_dict:
                param_name_dict[name].append(param_vec_dict[name])
            else:
                param_name_dict[name] = [param_vec_dict[name]]

        for parameter, value in parameter_binds.items():
            if isinstance(parameter, ParameterVector):
                if not self.disable_parameter_validation:
                    _validate_parameter_vector(parameter, value)
                out.update(zip(parameter, value))
            elif isinstance(parameter, str):
                for param in param_name_dict[parameter]:
                    if not self.disable_parameter_validation:
                        is_vec = _validate_parameter_value(param, value)
                    else:
                        is_vec = isinstance(param, ParameterVector)
                    if is_vec:
                        out.update(zip(param, value))
                    else:
                        out[param] = value
            elif isinstance(parameter, Sequence):
                for param, val in zip(parameter, value):
                    if isinstance(param, str):
                        # BindingsArray case
                        for p in param_name_dict[param]:
                            if not self.disable_parameter_validation:
                                is_vec = _validate_parameter_value(p, val)
                            else:
                                is_vec = isinstance(p, ParameterVector)
                            if is_vec:
                                out.update(zip(p, val))
                            else:
                                out[p] = val
                    elif isinstance(param, ParameterVector):
                        if not self.disable_parameter_validation:
                            _validate_parameter_vector(param, val)
                        out.update(zip(param, val))
                    elif isinstance(param, Parameter):
                        if not self.disable_parameter_validation:
                            _validate_parameter_value(param, val)
                        out[param] = val

            else:
                out[parameter] = value
        return out

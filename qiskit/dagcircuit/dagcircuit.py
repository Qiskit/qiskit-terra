# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Object to represent a quantum circuit as a directed acyclic graph (DAG).

The nodes in the graph are either input/output nodes or operation nodes.
The edges correspond to qubits or bits in the circuit. A directed edge
from node A to node B means that the (qu)bit passes from the output of A
to the input of B. The object's methods allow circuits to be constructed,
composed, and modified. Some natural properties like depth can be computed
directly from the graph.
"""
from collections import OrderedDict, defaultdict
import copy
import itertools
import math
from typing import Generator, Any, List

import numpy as np
import rustworkx as rx

from qiskit.circuit import ControlFlowOp, ForLoopOp, IfElseOp, WhileLoopOp, SwitchCaseOp
from qiskit.circuit.controlflow import is_control_flow_name
from qiskit.circuit.controlflow.condition import condition_bits
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.utils.deprecation import deprecate_func


class DAGCircuit:
    """
    Quantum circuit as a directed acyclic graph.

    There are 3 types of nodes in the graph: inputs, outputs, and operations.
    The nodes are connected by directed edges that correspond to qubits and
    bits.
    """

    # pylint: disable=invalid-name

    def __init__(self):
        """Create an empty circuit."""

        # Circuit name.  Generally, this corresponds to the name
        # of the QuantumCircuit from which the DAG was generated.
        self.name = None

        # Circuit metadata
        self.metadata = {}

        # Set of wires (Register,idx) in the dag
        self._wires = set()

        # Map from wire (Register,idx) to input nodes of the graph
        self.input_map = OrderedDict()

        # Map from wire (Register,idx) to output nodes of the graph
        self.output_map = OrderedDict()

        # Directed multigraph whose nodes are inputs, outputs, or operations.
        # Operation nodes have equal in- and out-degrees and carry
        # additional data about the operation, including the argument order
        # and parameter values.
        # Input nodes have out-degree 1 and output nodes have in-degree 1.
        # Edges carry wire labels (reg,idx) and each operation has
        # corresponding in- and out-edges with the same wire labels.
        self._multi_graph = rx.PyDAG()

        # Map of qreg/creg name to Register object.
        self.qregs = OrderedDict()
        self.cregs = OrderedDict()

        # List of Qubit/Clbit wires that the DAG acts on.
        self.qubits: List[Qubit] = []
        self.clbits: List[Clbit] = []

        self._global_phase = 0
        self._calibrations = defaultdict(dict)

        self._op_names = {}

        self.duration = None
        self.unit = "dt"

    @property
    def wires(self):
        """Return a list of the wires in order."""
        return self.qubits + self.clbits

    @property
    def node_counter(self):
        """
        Returns the number of nodes in the dag.
        """
        return len(self._multi_graph)

    @property
    def global_phase(self):
        """Return the global phase of the circuit."""
        return self._global_phase

    @global_phase.setter
    def global_phase(self, angle):
        """Set the global phase of the circuit.

        Args:
            angle (float, ParameterExpression)
        """
        if isinstance(angle, ParameterExpression):
            self._global_phase = angle
        else:
            # Set the phase to the [0, 2π) interval
            angle = float(angle)
            if not angle:
                self._global_phase = 0
            else:
                self._global_phase = angle % (2 * math.pi)

    @property
    def calibrations(self):
        """Return calibration dictionary.

        The custom pulse definition of a given gate is of the form
            {'gate_name': {(qubits, params): schedule}}
        """
        return dict(self._calibrations)

    @calibrations.setter
    def calibrations(self, calibrations):
        """Set the circuit calibration data from a dictionary of calibration definition.

        Args:
            calibrations (dict): A dictionary of input in the format
                {'gate_name': {(qubits, gate_params): schedule}}
        """
        self._calibrations = defaultdict(dict, calibrations)

    def add_calibration(self, gate, qubits, schedule, params=None):
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
                # The same logic exists in QuantumCircuit.add_calibration.
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

    def has_calibration_for(self, node):
        """Return True if the dag has a calibration defined for the node operation. In this
        case, the operation does not need to be translated to the device basis.
        """
        if not self.calibrations or node.op.name not in self.calibrations:
            return False
        qubits = tuple(self.qubits.index(qubit) for qubit in node.qargs)
        params = []
        for p in node.op.params:
            if isinstance(p, ParameterExpression) and not p.parameters:
                params.append(float(p))
            else:
                params.append(p)
        params = tuple(params)
        return (qubits, params) in self.calibrations[node.op.name]

    def remove_all_ops_named(self, opname):
        """Remove all operation nodes with the given name."""
        for n in self.named_nodes(opname):
            self.remove_op_node(n)

    def add_qubits(self, qubits):
        """Add individual qubit wires."""
        if any(not isinstance(qubit, Qubit) for qubit in qubits):
            raise DAGCircuitError("not a Qubit instance.")

        duplicate_qubits = set(self.qubits).intersection(qubits)
        if duplicate_qubits:
            raise DAGCircuitError("duplicate qubits %s" % duplicate_qubits)

        self.qubits.extend(qubits)
        for qubit in qubits:
            self._add_wire(qubit)

    def add_clbits(self, clbits):
        """Add individual clbit wires."""
        if any(not isinstance(clbit, Clbit) for clbit in clbits):
            raise DAGCircuitError("not a Clbit instance.")

        duplicate_clbits = set(self.clbits).intersection(clbits)
        if duplicate_clbits:
            raise DAGCircuitError("duplicate clbits %s" % duplicate_clbits)

        self.clbits.extend(clbits)
        for clbit in clbits:
            self._add_wire(clbit)

    def add_qreg(self, qreg):
        """Add all wires in a quantum register."""
        if not isinstance(qreg, QuantumRegister):
            raise DAGCircuitError("not a QuantumRegister instance.")
        if qreg.name in self.qregs:
            raise DAGCircuitError("duplicate register %s" % qreg.name)
        self.qregs[qreg.name] = qreg
        existing_qubits = set(self.qubits)
        for j in range(qreg.size):
            if qreg[j] not in existing_qubits:
                self.qubits.append(qreg[j])
                self._add_wire(qreg[j])

    def add_creg(self, creg):
        """Add all wires in a classical register."""
        if not isinstance(creg, ClassicalRegister):
            raise DAGCircuitError("not a ClassicalRegister instance.")
        if creg.name in self.cregs:
            raise DAGCircuitError("duplicate register %s" % creg.name)
        self.cregs[creg.name] = creg
        existing_clbits = set(self.clbits)
        for j in range(creg.size):
            if creg[j] not in existing_clbits:
                self.clbits.append(creg[j])
                self._add_wire(creg[j])

    def _add_wire(self, wire):
        """Add a qubit or bit to the circuit.

        Args:
            wire (Bit): the wire to be added

            This adds a pair of in and out nodes connected by an edge.

        Raises:
            DAGCircuitError: if trying to add duplicate wire
        """
        if wire not in self._wires:
            self._wires.add(wire)

            inp_node = DAGInNode(wire=wire)
            outp_node = DAGOutNode(wire=wire)
            input_map_id, output_map_id = self._multi_graph.add_nodes_from([inp_node, outp_node])
            inp_node._node_id = input_map_id
            outp_node._node_id = output_map_id
            self.input_map[wire] = inp_node
            self.output_map[wire] = outp_node
            self._multi_graph.add_edge(inp_node._node_id, outp_node._node_id, wire)
        else:
            raise DAGCircuitError(f"duplicate wire {wire}")

    def remove_clbits(self, *clbits):
        """
        Remove classical bits from the circuit. All bits MUST be idle.
        Any registers with references to at least one of the specified bits will
        also be removed.

        Args:
            clbits (List[Clbit]): The bits to remove.

        Raises:
            DAGCircuitError: a clbit is not a :obj:`.Clbit`, is not in the circuit,
                or is not idle.
        """
        if any(not isinstance(clbit, Clbit) for clbit in clbits):
            raise DAGCircuitError(
                "clbits not of type Clbit: %s" % [b for b in clbits if not isinstance(b, Clbit)]
            )

        clbits = set(clbits)
        unknown_clbits = clbits.difference(self.clbits)
        if unknown_clbits:
            raise DAGCircuitError("clbits not in circuit: %s" % unknown_clbits)

        busy_clbits = {bit for bit in clbits if not self._is_wire_idle(bit)}
        if busy_clbits:
            raise DAGCircuitError("clbits not idle: %s" % busy_clbits)

        # remove any references to bits
        cregs_to_remove = {creg for creg in self.cregs.values() if not clbits.isdisjoint(creg)}
        self.remove_cregs(*cregs_to_remove)

        for clbit in clbits:
            self._remove_idle_wire(clbit)
            self.clbits.remove(clbit)

    def remove_cregs(self, *cregs):
        """
        Remove classical registers from the circuit, leaving underlying bits
        in place.

        Raises:
            DAGCircuitError: a creg is not a ClassicalRegister, or is not in
            the circuit.
        """
        if any(not isinstance(creg, ClassicalRegister) for creg in cregs):
            raise DAGCircuitError(
                "cregs not of type ClassicalRegister: %s"
                % [r for r in cregs if not isinstance(r, ClassicalRegister)]
            )

        unknown_cregs = set(cregs).difference(self.cregs.values())
        if unknown_cregs:
            raise DAGCircuitError("cregs not in circuit: %s" % unknown_cregs)

        for creg in cregs:
            del self.cregs[creg.name]

    def remove_qubits(self, *qubits):
        """
        Remove quantum bits from the circuit. All bits MUST be idle.
        Any registers with references to at least one of the specified bits will
        also be removed.

        Args:
            qubits (List[Qubit]): The bits to remove.

        Raises:
            DAGCircuitError: a qubit is not a :obj:`.Qubit`, is not in the circuit,
                or is not idle.
        """
        if any(not isinstance(qubit, Qubit) for qubit in qubits):
            raise DAGCircuitError(
                "qubits not of type Qubit: %s" % [b for b in qubits if not isinstance(b, Qubit)]
            )

        qubits = set(qubits)
        unknown_qubits = qubits.difference(self.qubits)
        if unknown_qubits:
            raise DAGCircuitError("qubits not in circuit: %s" % unknown_qubits)

        busy_qubits = {bit for bit in qubits if not self._is_wire_idle(bit)}
        if busy_qubits:
            raise DAGCircuitError("qubits not idle: %s" % busy_qubits)

        # remove any references to bits
        qregs_to_remove = {qreg for qreg in self.qregs.values() if not qubits.isdisjoint(qreg)}
        self.remove_qregs(*qregs_to_remove)

        for qubit in qubits:
            self._remove_idle_wire(qubit)
            self.qubits.remove(qubit)

    def remove_qregs(self, *qregs):
        """
        Remove classical registers from the circuit, leaving underlying bits
        in place.

        Raises:
            DAGCircuitError: a qreg is not a QuantumRegister, or is not in
            the circuit.
        """
        if any(not isinstance(qreg, QuantumRegister) for qreg in qregs):
            raise DAGCircuitError(
                "qregs not of type QuantumRegister: %s"
                % [r for r in qregs if not isinstance(r, QuantumRegister)]
            )

        unknown_qregs = set(qregs).difference(self.qregs.values())
        if unknown_qregs:
            raise DAGCircuitError("qregs not in circuit: %s" % unknown_qregs)

        for qreg in qregs:
            del self.qregs[qreg.name]

    def _is_wire_idle(self, wire):
        """Check if a wire is idle.

        Args:
            wire (Bit): a wire in the circuit.

        Returns:
            bool: true if the wire is idle, false otherwise.

        Raises:
            DAGCircuitError: the wire is not in the circuit.
        """
        if wire not in self._wires:
            raise DAGCircuitError("wire %s not in circuit" % wire)

        try:
            child = next(self.successors(self.input_map[wire]))
        except StopIteration as e:
            raise DAGCircuitError(
                "Invalid dagcircuit input node %s has no output" % self.input_map[wire]
            ) from e
        return child is self.output_map[wire]

    def _remove_idle_wire(self, wire):
        """Remove an idle qubit or bit from the circuit.

        Args:
            wire (Bit): the wire to be removed, which MUST be idle.
        """
        inp_node = self.input_map[wire]
        oup_node = self.output_map[wire]

        self._multi_graph.remove_node(inp_node._node_id)
        self._multi_graph.remove_node(oup_node._node_id)
        self._wires.remove(wire)
        del self.input_map[wire]
        del self.output_map[wire]

    def _check_condition(self, name, condition):
        """Verify that the condition is valid.

        Args:
            name (string): used for error reporting
            condition (tuple or None): a condition tuple (ClassicalRegister, int) or (Clbit, bool)

        Raises:
            DAGCircuitError: if conditioning on an invalid register
        """
        if (
            condition is not None
            and condition[0] not in self.clbits
            and condition[0].name not in self.cregs
        ):
            raise DAGCircuitError("invalid creg in condition for %s" % name)

    def _check_bits(self, args, amap):
        """Check the values of a list of (qu)bit arguments.

        For each element of args, check that amap contains it.

        Args:
            args (list[Bit]): the elements to be checked
            amap (dict): a dictionary keyed on Qubits/Clbits

        Raises:
            DAGCircuitError: if a qubit is not contained in amap
        """
        # Check for each wire
        for wire in args:
            if wire not in amap:
                raise DAGCircuitError(f"(qu)bit {wire} not found in {amap}")

    @staticmethod
    def _bits_in_condition(cond):
        """Return a list of bits in the given condition.

        Args:
            cond (tuple or None): optional condition (ClassicalRegister, int) or (Clbit, bool)

        Returns:
            list[Clbit]: list of classical bits

        Raises:
            CircuitError: if cond[0] is not ClassicalRegister or Clbit
        """
        if cond is None:
            return []
        elif isinstance(cond[0], ClassicalRegister):
            # Returns a list of all the cbits in the given creg cond[0].
            return cond[0][:]
        elif isinstance(cond[0], Clbit):
            # Returns a singleton list of the conditional cbit.
            return [cond[0]]
        else:
            raise CircuitError("Condition must be used with ClassicalRegister or Clbit.")

    def _increment_op(self, op):
        if op.name in self._op_names:
            self._op_names[op.name] += 1
        else:
            self._op_names[op.name] = 1

    def _decrement_op(self, op):
        if self._op_names[op.name] == 1:
            del self._op_names[op.name]
        else:
            self._op_names[op.name] -= 1

    def _add_op_node(self, op, qargs, cargs):
        """Add a new operation node to the graph and assign properties.

        Args:
            op (qiskit.circuit.Operation): the operation associated with the DAG node
            qargs (list[Qubit]): list of quantum wires to attach to.
            cargs (list[Clbit]): list of classical wires to attach to.
        Returns:
            int: The integer node index for the new op node on the DAG
        """
        # Add a new operation node to the graph
        new_node = DAGOpNode(op=op, qargs=qargs, cargs=cargs)
        node_index = self._multi_graph.add_node(new_node)
        new_node._node_id = node_index
        self._increment_op(op)
        return node_index

    @deprecate_func(
        additional_msg="Instead, use :meth:`~copy_empty_like()`, which acts identically.",
        since="0.20.0",
    )
    def _copy_circuit_metadata(self):
        """DEPRECATED"""
        return self.copy_empty_like()

    def copy_empty_like(self):
        """Return a copy of self with the same structure but empty.

        That structure includes:
            * name and other metadata
            * global phase
            * duration
            * all the qubits and clbits, including the registers.

        Returns:
            DAGCircuit: An empty copy of self.
        """
        target_dag = DAGCircuit()
        target_dag.name = self.name
        target_dag._global_phase = self._global_phase
        target_dag.duration = self.duration
        target_dag.unit = self.unit
        target_dag.metadata = self.metadata

        target_dag.add_qubits(self.qubits)
        target_dag.add_clbits(self.clbits)

        for qreg in self.qregs.values():
            target_dag.add_qreg(qreg)
        for creg in self.cregs.values():
            target_dag.add_creg(creg)

        return target_dag

    def apply_operation_back(self, op, qargs=(), cargs=()):
        """Apply an operation to the output of the circuit.

        Args:
            op (qiskit.circuit.Operation): the operation associated with the DAG node
            qargs (tuple[Qubit]): qubits that op will be applied to
            cargs (tuple[Clbit]): cbits that op will be applied to
        Returns:
            DAGOpNode: the node for the op that was added to the dag

        Raises:
            DAGCircuitError: if a leaf node is connected to multiple outputs

        """
        qargs = tuple(qargs) if qargs is not None else ()
        cargs = tuple(cargs) if cargs is not None else ()

        all_cbits = self._bits_in_condition(getattr(op, "condition", None))
        all_cbits = set(all_cbits).union(cargs)

        self._check_condition(op.name, getattr(op, "condition", None))
        self._check_bits(qargs, self.output_map)
        self._check_bits(all_cbits, self.output_map)

        node_index = self._add_op_node(op, qargs, cargs)

        # Add new in-edges from predecessors of the output nodes to the
        # operation node while deleting the old in-edges of the output nodes
        # and adding new edges from the operation node to each output node

        al = [qargs, all_cbits]
        self._multi_graph.insert_node_on_in_edges_multiple(
            node_index, [self.output_map[q]._node_id for q in itertools.chain(*al)]
        )
        return self._multi_graph[node_index]

    def apply_operation_front(self, op, qargs=(), cargs=()):
        """Apply an operation to the input of the circuit.

        Args:
            op (qiskit.circuit.Operation): the operation associated with the DAG node
            qargs (tuple[Qubit]): qubits that op will be applied to
            cargs (tuple[Clbit]): cbits that op will be applied to
        Returns:
            DAGOpNode: the node for the op that was added to the dag

        Raises:
            DAGCircuitError: if initial nodes connected to multiple out edges
        """
        all_cbits = self._bits_in_condition(getattr(op, "condition", None))
        all_cbits.extend(cargs)

        self._check_condition(op.name, getattr(op, "condition", None))
        self._check_bits(qargs, self.input_map)
        self._check_bits(all_cbits, self.input_map)
        node_index = self._add_op_node(op, qargs, cargs)

        # Add new out-edges to successors of the input nodes from the
        # operation node while deleting the old out-edges of the input nodes
        # and adding new edges to the operation node from each input node
        al = [qargs, all_cbits]
        self._multi_graph.insert_node_on_out_edges_multiple(
            node_index, [self.input_map[q]._node_id for q in itertools.chain(*al)]
        )
        return self._multi_graph[node_index]

    @staticmethod
    def _map_condition(wire_map, condition, target_cregs):
        """Use the wire_map dict to change the condition tuple's creg name.

        Args:
            wire_map (dict): a map from source wires to destination wires
            condition (tuple or None): (ClassicalRegister,int)
            target_cregs (list[ClassicalRegister]): List of all cregs in the
              target circuit onto which the condition might possibly be mapped.
        Returns:
            tuple(ClassicalRegister,int): new condition
        Raises:
            DAGCircuitError: if condition register not in wire_map, or if
                wire_map maps condition onto more than one creg, or if the
                specified condition is not present in a classical register.
        """

        if condition is None:
            new_condition = None
        else:
            # if there is a condition, map the condition bits to the
            # composed cregs based on the wire_map
            is_reg = False
            if isinstance(condition[0], Clbit):
                cond_creg = [condition[0]]
            else:
                cond_creg = condition[0]
                is_reg = True
            cond_val = condition[1]
            new_cond_val = 0
            new_creg = None
            bits_in_condcreg = [bit for bit in wire_map if bit in cond_creg]
            for bit in bits_in_condcreg:
                if is_reg:
                    try:
                        candidate_creg = next(
                            creg for creg in target_cregs if wire_map[bit] in creg
                        )
                    except StopIteration as ex:
                        raise DAGCircuitError(
                            "Did not find creg containing mapped clbit in conditional."
                        ) from ex
                else:
                    # If cond is on a single Clbit then the candidate_creg is
                    # the target Clbit to which 'bit' is mapped to.
                    candidate_creg = wire_map[bit]
                if new_creg is None:
                    new_creg = candidate_creg
                elif new_creg != candidate_creg:
                    # Raise if wire_map maps condition creg on to more than one
                    # creg in target DAG.
                    raise DAGCircuitError(
                        "wire_map maps conditional register onto more than one creg."
                    )

                if not is_reg:
                    # If the cond is on a single Clbit then the new_cond_val is the
                    # same as the cond_val since the new_creg is also a single Clbit.
                    new_cond_val = cond_val
                elif 2 ** (cond_creg[:].index(bit)) & cond_val:
                    # If the conditional values of the Clbit 'bit' is 1 then the new_cond_val
                    # is updated such that the conditional value of the Clbit to which 'bit'
                    # is mapped to in new_creg is 1.
                    new_cond_val += 2 ** (new_creg[:].index(wire_map[bit]))
            if new_creg is None:
                raise DAGCircuitError("Condition registers not found in wire_map.")
            new_condition = (new_creg, new_cond_val)
        return new_condition

    def _map_classical_resource_with_import(self, resource, wire_map, creg_map):
        """Map the classical ``resource`` (a bit or register) in its counterpart in ``self`` using
        ``wire_map`` and ``creg_map`` as lookup caches.  All single-bit conditions should have a
        cache hit in the ``wire_map``, but registers may involve a full linear search the first time
        they are encountered.  ``creg_map`` is mutated by this function.  ``wire_map`` is not; it is
        an error if a wire is not in the map.

        This is different to the logic in ``_map_condition`` because it always succeeds; since the
        mapping for all wires in the condition is assumed to exist, there can be no fragmented
        registers.  If there is no matching register (has the same bits in the same order) in
        ``self``, a new register alias is added to represent the condition.  This does not change
        the bits available to ``self``, it just adds a new aliased grouping of them."""
        if isinstance(resource, Clbit):
            return wire_map[resource]
        if resource.name not in creg_map:
            mapped_bits = [wire_map[bit] for bit in resource]
            for our_creg in self.cregs.values():
                if mapped_bits == list(our_creg):
                    new_resource = our_creg
                    break
            else:
                new_resource = ClassicalRegister(bits=[wire_map[bit] for bit in resource])
                self.add_creg(new_resource)
            creg_map[resource.name] = new_resource
        return creg_map[resource.name]

    def compose(self, other, qubits=None, clbits=None, front=False, inplace=True):
        """Compose the ``other`` circuit onto the output of this circuit.

        A subset of input wires of ``other`` are mapped
        to a subset of output wires of this circuit.

        ``other`` can be narrower or of equal width to ``self``.

        Args:
            other (DAGCircuit): circuit to compose with self
            qubits (list[Qubit|int]): qubits of self to compose onto.
            clbits (list[Clbit|int]): clbits of self to compose onto.
            front (bool): If True, front composition will be performed (not implemented yet)
            inplace (bool): If True, modify the object. Otherwise return composed circuit.

        Returns:
            DAGCircuit: the composed dag (returns None if inplace==True).

        Raises:
            DAGCircuitError: if ``other`` is wider or there are duplicate edge mappings.
        """
        if front:
            raise DAGCircuitError("Front composition not supported yet.")

        if len(other.qubits) > len(self.qubits) or len(other.clbits) > len(self.clbits):
            raise DAGCircuitError(
                "Trying to compose with another DAGCircuit which has more 'in' edges."
            )

        # number of qubits and clbits must match number in circuit or None
        identity_qubit_map = dict(zip(other.qubits, self.qubits))
        identity_clbit_map = dict(zip(other.clbits, self.clbits))
        if qubits is None:
            qubit_map = identity_qubit_map
        elif len(qubits) != len(other.qubits):
            raise DAGCircuitError(
                "Number of items in qubits parameter does not"
                " match number of qubits in the circuit."
            )
        else:
            qubit_map = {
                other.qubits[i]: (self.qubits[q] if isinstance(q, int) else q)
                for i, q in enumerate(qubits)
            }
        if clbits is None:
            clbit_map = identity_clbit_map
        elif len(clbits) != len(other.clbits):
            raise DAGCircuitError(
                "Number of items in clbits parameter does not"
                " match number of clbits in the circuit."
            )
        else:
            clbit_map = {
                other.clbits[i]: (self.clbits[c] if isinstance(c, int) else c)
                for i, c in enumerate(clbits)
            }
        edge_map = {**qubit_map, **clbit_map} or None

        # if no edge_map, try to do a 1-1 mapping in order
        if edge_map is None:
            edge_map = {**identity_qubit_map, **identity_clbit_map}

        # Check the edge_map for duplicate values
        if len(set(edge_map.values())) != len(edge_map):
            raise DAGCircuitError("duplicates in wire_map")

        # Compose
        if inplace:
            dag = self
        else:
            dag = copy.deepcopy(self)
        dag.global_phase += other.global_phase

        for gate, cals in other.calibrations.items():
            dag._calibrations[gate].update(cals)

        for nd in other.topological_nodes():
            if isinstance(nd, DAGInNode):
                # if in edge_map, get new name, else use existing name
                m_wire = edge_map.get(nd.wire, nd.wire)
                # the mapped wire should already exist
                if m_wire not in dag.output_map:
                    raise DAGCircuitError(
                        "wire %s[%d] not in self" % (m_wire.register.name, m_wire.index)
                    )
                if nd.wire not in other._wires:
                    raise DAGCircuitError(
                        "inconsistent wire type for %s[%d] in other"
                        % (nd.register.name, nd.wire.index)
                    )
            elif isinstance(nd, DAGOutNode):
                # ignore output nodes
                pass
            elif isinstance(nd, DAGOpNode):
                condition = dag._map_condition(
                    edge_map, getattr(nd.op, "condition", None), dag.cregs.values()
                )
                dag._check_condition(nd.op.name, condition)
                m_qargs = [edge_map.get(x, x) for x in nd.qargs]
                m_cargs = [edge_map.get(x, x) for x in nd.cargs]
                op = nd.op.copy()
                if condition and not isinstance(op, Instruction):
                    raise DAGCircuitError("Cannot add a condition on a generic Operation.")
                op.condition = condition
                dag.apply_operation_back(op, m_qargs, m_cargs)
            else:
                raise DAGCircuitError("bad node type %s" % type(nd))

        if not inplace:
            return dag
        else:
            return None

    def reverse_ops(self):
        """Reverse the operations in the ``self`` circuit.

        Returns:
            DAGCircuit: the reversed dag.
        """
        # TODO: speed up
        # pylint: disable=cyclic-import
        from qiskit.converters import dag_to_circuit, circuit_to_dag

        qc = dag_to_circuit(self)
        reversed_qc = qc.reverse_ops()
        reversed_dag = circuit_to_dag(reversed_qc)
        return reversed_dag

    def idle_wires(self, ignore=None):
        """Return idle wires.

        Args:
            ignore (list(str)): List of node names to ignore. Default: []

        Yields:
            Bit: Bit in idle wire.

        Raises:
            DAGCircuitError: If the DAG is invalid
        """
        if ignore is None:
            ignore = set()
        ignore_set = set(ignore)
        for wire in self._wires:
            if not ignore:
                if self._is_wire_idle(wire):
                    yield wire
            else:
                for node in self.nodes_on_wire(wire, only_ops=True):
                    if node.op.name not in ignore_set:
                        # If we found an op node outside of ignore we can stop iterating over the wire
                        break
                else:
                    yield wire

    def size(self, *, recurse: bool = False):
        """Return the number of operations.  If there is control flow present, this count may only
        be an estimate, as the complete control-flow path cannot be statically known.

        Args:
            recurse: if ``True``, then recurse into control-flow operations.  For loops with
                known-length iterators are counted unrolled.  If-else blocks sum both of the two
                branches.  While loops are counted as if the loop body runs once only.  Defaults to
                ``False`` and raises :class:`.DAGCircuitError` if any control flow is present, to
                avoid silently returning a mostly meaningless number.

        Returns:
            int: the circuit size

        Raises:
            DAGCircuitError: if an unknown :class:`.ControlFlowOp` is present in a call with
                ``recurse=True``, or any control flow is present in a non-recursive call.
        """
        length = len(self._multi_graph) - 2 * len(self._wires)
        if not recurse:
            if any(is_control_flow_name(op_name) for op_name in self._op_names):
                raise DAGCircuitError(
                    "Size with control flow is ambiguous."
                    " You may use `recurse=True` to get a result,"
                    " but see this method's documentation for the meaning of this."
                )
            return length
        # pylint: disable=cyclic-import
        from qiskit.converters import circuit_to_dag

        for node in self.op_nodes(ControlFlowOp):
            if isinstance(node.op, ForLoopOp):
                indexset = node.op.params[0]
                inner = len(indexset) * circuit_to_dag(node.op.blocks[0]).size(recurse=True)
            elif isinstance(node.op, WhileLoopOp):
                inner = circuit_to_dag(node.op.blocks[0]).size(recurse=True)
            elif isinstance(node.op, (IfElseOp, SwitchCaseOp)):
                inner = sum(circuit_to_dag(block).size(recurse=True) for block in node.op.blocks)
            else:
                raise DAGCircuitError(f"unknown control-flow type: '{node.op.name}'")
            # Replace the "1" for the node itself with the actual count.
            length += inner - 1
        return length

    def depth(self, *, recurse: bool = False):
        """Return the circuit depth.  If there is control flow present, this count may only be an
        estimate, as the complete control-flow path cannot be staticly known.

        Args:
            recurse: if ``True``, then recurse into control-flow operations.  For loops
                with known-length iterators are counted as if the loop had been manually unrolled
                (*i.e.* with each iteration of the loop body written out explicitly).
                If-else blocks take the longer case of the two branches.  While loops are counted as
                if the loop body runs once only.  Defaults to ``False`` and raises
                :class:`.DAGCircuitError` if any control flow is present, to avoid silently
                returning a nonsensical number.

        Returns:
            int: the circuit depth

        Raises:
            DAGCircuitError: if not a directed acyclic graph
            DAGCircuitError: if unknown control flow is present in a recursive call, or any control
                flow is present in a non-recursive call.
        """
        if recurse:
            from qiskit.converters import circuit_to_dag  # pylint: disable=cyclic-import

            node_lookup = {}
            for node in self.op_nodes(ControlFlowOp):
                weight = len(node.op.params[0]) if isinstance(node.op, ForLoopOp) else 1
                if weight == 0:
                    node_lookup[node._node_id] = 0
                else:
                    node_lookup[node._node_id] = weight * max(
                        circuit_to_dag(block).depth(recurse=True) for block in node.op.blocks
                    )

            def weight_fn(_source, target, _edge):
                return node_lookup.get(target, 1)

        else:
            if any(is_control_flow_name(op_name) for op_name in self._op_names):
                raise DAGCircuitError(
                    "Depth with control flow is ambiguous."
                    " You may use `recurse=True` to get a result,"
                    " but see this method's documentation for the meaning of this."
                )
            weight_fn = None

        try:
            depth = rx.dag_longest_path_length(self._multi_graph, weight_fn) - 1
        except rx.DAGHasCycle as ex:
            raise DAGCircuitError("not a DAG") from ex
        return depth if depth >= 0 else 0

    def width(self):
        """Return the total number of qubits + clbits used by the circuit.
        This function formerly returned the number of qubits by the calculation
        return len(self._wires) - self.num_clbits()
        but was changed by issue #2564 to return number of qubits + clbits
        with the new function DAGCircuit.num_qubits replacing the former
        semantic of DAGCircuit.width().
        """
        return len(self._wires)

    def num_qubits(self):
        """Return the total number of qubits used by the circuit.
        num_qubits() replaces former use of width().
        DAGCircuit.width() now returns qubits + clbits for
        consistency with Circuit.width() [qiskit-terra #2564].
        """
        return len(self.qubits)

    def num_clbits(self):
        """Return the total number of classical bits used by the circuit."""
        return len(self.clbits)

    def num_tensor_factors(self):
        """Compute how many components the circuit can decompose into."""
        return rx.number_weakly_connected_components(self._multi_graph)

    def __eq__(self, other):
        # Try to convert to float, but in case of unbound ParameterExpressions
        # a TypeError will be raise, fallback to normal equality in those
        # cases
        try:
            self_phase = float(self.global_phase)
            other_phase = float(other.global_phase)
            if (
                abs((self_phase - other_phase + np.pi) % (2 * np.pi) - np.pi) > 1.0e-10
            ):  # TODO: atol?
                return False
        except TypeError:
            if self.global_phase != other.global_phase:
                return False
        if self.calibrations != other.calibrations:
            return False

        self_bit_indices = {bit: idx for idx, bit in enumerate(self.qubits + self.clbits)}
        other_bit_indices = {bit: idx for idx, bit in enumerate(other.qubits + other.clbits)}

        self_qreg_indices = {
            regname: [self_bit_indices[bit] for bit in reg] for regname, reg in self.qregs.items()
        }
        self_creg_indices = {
            regname: [self_bit_indices[bit] for bit in reg] for regname, reg in self.cregs.items()
        }

        other_qreg_indices = {
            regname: [other_bit_indices[bit] for bit in reg] for regname, reg in other.qregs.items()
        }
        other_creg_indices = {
            regname: [other_bit_indices[bit] for bit in reg] for regname, reg in other.cregs.items()
        }
        if self_qreg_indices != other_qreg_indices or self_creg_indices != other_creg_indices:
            return False

        def node_eq(node_self, node_other):
            return DAGNode.semantic_eq(node_self, node_other, self_bit_indices, other_bit_indices)

        return rx.is_isomorphic_node_match(self._multi_graph, other._multi_graph, node_eq)

    def topological_nodes(self, key=None):
        """
        Yield nodes in topological order.

        Args:
            key (Callable): A callable which will take a DAGNode object and
                return a string sort key. If not specified the
                :attr:`~qiskit.dagcircuit.DAGNode.sort_key` attribute will be
                used as the sort key for each node.

        Returns:
            generator(DAGOpNode, DAGInNode, or DAGOutNode): node in topological order
        """

        def _key(x):
            return x.sort_key

        if key is None:
            key = _key

        return iter(rx.lexicographical_topological_sort(self._multi_graph, key=key))

    def topological_op_nodes(self, key=None) -> Generator[DAGOpNode, Any, Any]:
        """
        Yield op nodes in topological order.

        Allowed to pass in specific key to break ties in top order

        Args:
            key (Callable): A callable which will take a DAGNode object and
                return a string sort key. If not specified the
                :attr:`~qiskit.dagcircuit.DAGNode.sort_key` attribute will be
                used as the sort key for each node.

        Returns:
            generator(DAGOpNode): op node in topological order
        """
        return (nd for nd in self.topological_nodes(key) if isinstance(nd, DAGOpNode))

    def replace_block_with_op(self, node_block, op, wire_pos_map, cycle_check=True):
        """Replace a block of nodes with a single node.

        This is used to consolidate a block of DAGOpNodes into a single
        operation. A typical example is a block of gates being consolidated
        into a single ``UnitaryGate`` representing the unitary matrix of the
        block.

        Args:
            node_block (List[DAGNode]): A list of dag nodes that represents the
                node block to be replaced
            op (qiskit.circuit.Operation): The operation to replace the
                block with
            wire_pos_map (Dict[Qubit, int]): The dictionary mapping the qarg to
                the position. This is necessary to reconstruct the qarg order
                over multiple gates in the combined single op node.
            cycle_check (bool): When set to True this method will check that
                replacing the provided ``node_block`` with a single node
                would introduce a cycle (which would invalidate the
                ``DAGCircuit``) and will raise a ``DAGCircuitError`` if a cycle
                would be introduced. This checking comes with a run time
                penalty. If you can guarantee that your input ``node_block`` is
                a contiguous block and won't introduce a cycle when it's
                contracted to a single node, this can be set to ``False`` to
                improve the runtime performance of this method.

        Raises:
            DAGCircuitError: if ``cycle_check`` is set to ``True`` and replacing
                the specified block introduces a cycle or if ``node_block`` is
                empty.

        Returns:
            DAGOpNode: The op node that replaces the block.
        """
        block_qargs = set()
        block_cargs = set()
        block_ids = [x._node_id for x in node_block]

        # If node block is empty return early
        if not node_block:
            raise DAGCircuitError("Can't replace an empty node_block")

        for nd in node_block:
            block_qargs |= set(nd.qargs)
            block_cargs |= set(nd.cargs)
            cond = getattr(nd.op, "condition", None)
            if cond is not None:
                block_cargs.update(condition_bits(cond))

        # Create replacement node
        new_node = DAGOpNode(
            op,
            sorted(block_qargs, key=lambda x: wire_pos_map[x]),
            sorted(block_cargs, key=lambda x: wire_pos_map[x]),
        )

        try:
            new_node._node_id = self._multi_graph.contract_nodes(
                block_ids, new_node, check_cycle=cycle_check
            )
        except rx.DAGWouldCycle as ex:
            raise DAGCircuitError(
                "Replacing the specified node block would introduce a cycle"
            ) from ex

        self._increment_op(op)

        for nd in node_block:
            self._decrement_op(nd.op)

        return new_node

    def substitute_node_with_dag(self, node, input_dag, wires=None, propagate_condition=True):
        """Replace one node with dag.

        Args:
            node (DAGOpNode): node to substitute
            input_dag (DAGCircuit): circuit that will substitute the node
            wires (list[Bit] | Dict[Bit, Bit]): gives an order for (qu)bits
                in the input circuit. If a list, then the bits refer to those in the ``input_dag``,
                and the order gets matched to the node wires by qargs first, then cargs, then
                conditions.  If a dictionary, then a mapping of bits in the ``input_dag`` to those
                that the ``node`` acts on.
            propagate_condition (bool): If ``True`` (default), then any ``condition`` attribute on
                the operation within ``node`` is propagated to each node in the ``input_dag``.  If
                ``False``, then the ``input_dag`` is assumed to faithfully implement suitable
                conditional logic already.

        Returns:
            dict: maps node IDs from `input_dag` to their new node incarnations in `self`.

        Raises:
            DAGCircuitError: if met with unexpected predecessor/successors
        """
        if not isinstance(node, DAGOpNode):
            raise DAGCircuitError(f"expected node DAGOpNode, got {type(node)}")

        if isinstance(wires, dict):
            wire_map = wires
        else:
            wires = input_dag.wires if wires is None else wires
            node_cargs = set(node.cargs)
            node_wire_order = list(node.qargs) + list(node.cargs)
            # If we're not propagating it, the number of wires in the input DAG should include the
            # condition as well.
            if not propagate_condition:
                node_wire_order += [
                    bit
                    for bit in self._bits_in_condition(getattr(node.op, "condition", None))
                    if bit not in node_cargs
                ]
            if len(wires) != len(node_wire_order):
                raise DAGCircuitError(
                    f"bit mapping invalid: expected {len(node_wire_order)}, got {len(wires)}"
                )
            wire_map = dict(zip(wires, node_wire_order))
            if len(wire_map) != len(node_wire_order):
                raise DAGCircuitError("bit mapping invalid: some bits have duplicate entries")
        for input_dag_wire, our_wire in wire_map.items():
            if our_wire not in self.input_map:
                raise DAGCircuitError(f"bit mapping invalid: {our_wire} is not in this DAG")
            # Support mapping indiscriminately between Qubit and AncillaQubit, etc.
            check_type = Qubit if isinstance(our_wire, Qubit) else Clbit
            if not isinstance(input_dag_wire, check_type):
                raise DAGCircuitError(
                    f"bit mapping invalid: {input_dag_wire} and {our_wire} are different bit types"
                )

        reverse_wire_map = {b: a for a, b in wire_map.items()}
        creg_map = {}
        op_condition = getattr(node.op, "condition", None)
        if propagate_condition and op_condition is not None:
            in_dag = input_dag.copy_empty_like()
            target, value = op_condition
            if isinstance(target, Clbit):
                new_target = reverse_wire_map.get(target, Clbit())
                if new_target not in wire_map:
                    in_dag.add_clbits([new_target])
                    wire_map[new_target], reverse_wire_map[target] = target, new_target
                target_cargs = {new_target}
            else:  # ClassicalRegister
                mapped_bits = [reverse_wire_map.get(bit, Clbit()) for bit in target]
                for ours, theirs in zip(target, mapped_bits):
                    # Update to any new dummy bits we just created to the wire maps.
                    wire_map[theirs], reverse_wire_map[ours] = ours, theirs
                new_target = ClassicalRegister(bits=mapped_bits)
                creg_map[new_target.name] = target
                in_dag.add_creg(new_target)
                target_cargs = set(new_target)
            new_condition = (new_target, value)
            for in_node in input_dag.topological_op_nodes():
                if getattr(in_node.op, "condition", None) is not None:
                    raise DAGCircuitError(
                        "cannot propagate a condition to an element that already has one"
                    )
                if target_cargs.intersection(in_node.cargs):
                    # This is for backwards compatibility with early versions of the method, as it is
                    # a tested part of the API.  In the newer model of a condition being an integral
                    # part of the operation (not a separate property to be copied over), this error
                    # is overzealous, because it forbids a custom instruction from implementing the
                    # condition within its definition rather than at the top level.
                    raise DAGCircuitError(
                        "cannot propagate a condition to an element that acts on those bits"
                    )
                new_op = copy.copy(in_node.op)
                new_op.condition = new_condition
                in_dag.apply_operation_back(new_op, in_node.qargs, in_node.cargs)
        else:
            in_dag = input_dag

        if in_dag.global_phase:
            self.global_phase += in_dag.global_phase

        # Add wire from pred to succ if no ops on mapped wire on ``in_dag``
        # rustworkx's substitute_node_with_subgraph lacks the DAGCircuit
        # context to know what to do in this case (the method won't even see
        # these nodes because they're filtered) so we manually retain the
        # edges prior to calling substitute_node_with_subgraph and set the
        # edge_map_fn callback kwarg to skip these edges when they're
        # encountered.
        for in_dag_wire, self_wire in wire_map.items():
            input_node = in_dag.input_map[in_dag_wire]
            output_node = in_dag.output_map[in_dag_wire]
            if in_dag._multi_graph.has_edge(input_node._node_id, output_node._node_id):
                pred = self._multi_graph.find_predecessors_by_edge(
                    node._node_id, lambda edge, wire=self_wire: edge == wire
                )[0]
                succ = self._multi_graph.find_successors_by_edge(
                    node._node_id, lambda edge, wire=self_wire: edge == wire
                )[0]
                self._multi_graph.add_edge(pred._node_id, succ._node_id, self_wire)

        # Exlude any nodes from in_dag that are not a DAGOpNode or are on
        # bits outside the set specified by the wires kwarg
        def filter_fn(node):
            if not isinstance(node, DAGOpNode):
                return False
            for qarg in node.qargs:
                if qarg not in wire_map:
                    return False
            return True

        # Map edges into and out of node to the appropriate node from in_dag
        def edge_map_fn(source, _target, self_wire):
            wire = reverse_wire_map[self_wire]
            # successor edge
            if source == node._node_id:
                wire_output_id = in_dag.output_map[wire]._node_id
                out_index = in_dag._multi_graph.predecessor_indices(wire_output_id)[0]
                # Edge directly from from input nodes to output nodes in in_dag are
                # already handled prior to calling rustworkx. Don't map these edges
                # in rustworkx.
                if not isinstance(in_dag._multi_graph[out_index], DAGOpNode):
                    return None
            # predecessor edge
            else:
                wire_input_id = in_dag.input_map[wire]._node_id
                out_index = in_dag._multi_graph.successor_indices(wire_input_id)[0]
                # Edge directly from from input nodes to output nodes in in_dag are
                # already handled prior to calling rustworkx. Don't map these edges
                # in rustworkx.
                if not isinstance(in_dag._multi_graph[out_index], DAGOpNode):
                    return None
            return out_index

        # Adjust edge weights from in_dag
        def edge_weight_map(wire):
            return wire_map[wire]

        node_map = self._multi_graph.substitute_node_with_subgraph(
            node._node_id, in_dag._multi_graph, edge_map_fn, filter_fn, edge_weight_map
        )
        self._decrement_op(node.op)

        # Iterate over nodes of input_circuit and update wires in node objects migrated
        # from in_dag
        for old_node_index, new_node_index in node_map.items():
            # update node attributes
            old_node = in_dag._multi_graph[old_node_index]
            if isinstance(old_node.op, SwitchCaseOp):
                m_op = SwitchCaseOp(
                    self._map_classical_resource_with_import(
                        old_node.op.target, wire_map, creg_map
                    ),
                    old_node.op.cases_specifier(),
                    label=old_node.op.label,
                )
            elif getattr(old_node.op, "condition", None) is not None:
                cond_target, cond_value = old_node.op.condition
                m_op = copy.copy(old_node.op)
                m_op.condition = (
                    self._map_classical_resource_with_import(cond_target, wire_map, creg_map),
                    cond_value,
                )
            else:
                m_op = old_node.op
            m_qargs = [wire_map[x] for x in old_node.qargs]
            m_cargs = [wire_map[x] for x in old_node.cargs]
            new_node = DAGOpNode(m_op, qargs=m_qargs, cargs=m_cargs)
            new_node._node_id = new_node_index
            self._multi_graph[new_node_index] = new_node
            self._increment_op(new_node.op)

        return {k: self._multi_graph[v] for k, v in node_map.items()}

    def substitute_node(self, node, op, inplace=False):
        """Replace an DAGOpNode with a single operation. qargs, cargs and
        conditions for the new operation will be inferred from the node to be
        replaced. The new operation will be checked to match the shape of the
        replaced operation.

        Args:
            node (DAGOpNode): Node to be replaced
            op (qiskit.circuit.Operation): The :class:`qiskit.circuit.Operation`
                instance to be added to the DAG
            inplace (bool): Optional, default False. If True, existing DAG node
                will be modified to include op. Otherwise, a new DAG node will
                be used.

        Returns:
            DAGOpNode: the new node containing the added operation.

        Raises:
            DAGCircuitError: If replacement operation was incompatible with
            location of target node.
        """

        if not isinstance(node, DAGOpNode):
            raise DAGCircuitError("Only DAGOpNodes can be replaced.")

        if node.op.num_qubits != op.num_qubits or node.op.num_clbits != op.num_clbits:
            raise DAGCircuitError(
                "Cannot replace node of width ({} qubits, {} clbits) with "
                "operation of mismatched width ({} qubits, {} clbits).".format(
                    node.op.num_qubits, node.op.num_clbits, op.num_qubits, op.num_clbits
                )
            )

        if inplace:
            if op.name != node.op.name:
                self._increment_op(op)
                self._decrement_op(node.op)
            save_condition = getattr(node.op, "condition", None)
            node.op = op
            if save_condition and not isinstance(op, Instruction):
                raise DAGCircuitError("Cannot add a condition on a generic Operation.")
            node.op.condition = save_condition
            return node

        new_node = copy.copy(node)
        save_condition = getattr(new_node.op, "condition", None)
        new_node.op = op
        if save_condition and not isinstance(new_node.op, Instruction):
            raise DAGCircuitError("Cannot add a condition on a generic Operation.")
        new_node.op.condition = save_condition
        self._multi_graph[node._node_id] = new_node
        if op.name != node.op.name:
            self._increment_op(op)
            self._decrement_op(node.op)
        return new_node

    def swap_nodes(self, node1, node2):
        """Swap connected nodes e.g. due to commutation.

        Args:
            node1 (OpNode): predecessor node
            node2 (OpNode): successor node

        Raises:
            DAGCircuitError: if either node is not an OpNode or nodes are not connected
        """
        if not (isinstance(node1, DAGOpNode) and isinstance(node2, DAGOpNode)):
            raise DAGCircuitError("nodes to swap are not both DAGOpNodes")
        try:
            connected_edges = self._multi_graph.get_all_edge_data(node1._node_id, node2._node_id)
        except rx.NoEdgeBetweenNodes as no_common_edge:
            raise DAGCircuitError("attempt to swap unconnected nodes") from no_common_edge
        node1_id = node1._node_id
        node2_id = node2._node_id
        for edge in connected_edges[::-1]:
            edge_find = lambda x, y=edge: x == y
            edge_parent = self._multi_graph.find_predecessors_by_edge(node1_id, edge_find)[0]
            self._multi_graph.remove_edge(edge_parent._node_id, node1_id)
            self._multi_graph.add_edge(edge_parent._node_id, node2_id, edge)
            edge_child = self._multi_graph.find_successors_by_edge(node2_id, edge_find)[0]
            self._multi_graph.remove_edge(node1_id, node2_id)
            self._multi_graph.add_edge(node2_id, node1_id, edge)
            self._multi_graph.remove_edge(node2_id, edge_child._node_id)
            self._multi_graph.add_edge(node1_id, edge_child._node_id, edge)

    def node(self, node_id):
        """Get the node in the dag.

        Args:
            node_id(int): Node identifier.

        Returns:
            node: the node.
        """
        return self._multi_graph[node_id]

    def nodes(self):
        """Iterator for node values.

        Yield:
            node: the node.
        """
        yield from self._multi_graph.nodes()

    def edges(self, nodes=None):
        """Iterator for edge values and source and dest node

        This works by returning the output edges from the specified nodes. If
        no nodes are specified all edges from the graph are returned.

        Args:
            nodes(DAGOpNode, DAGInNode, or DAGOutNode|list(DAGOpNode, DAGInNode, or DAGOutNode):
                Either a list of nodes or a single input node. If none is specified,
                all edges are returned from the graph.

        Yield:
            edge: the edge in the same format as out_edges the tuple
                (source node, destination node, edge data)
        """
        if nodes is None:
            nodes = self._multi_graph.nodes()

        elif isinstance(nodes, (DAGOpNode, DAGInNode, DAGOutNode)):
            nodes = [nodes]
        for node in nodes:
            raw_nodes = self._multi_graph.out_edges(node._node_id)
            for source, dest, edge in raw_nodes:
                yield (self._multi_graph[source], self._multi_graph[dest], edge)

    def op_nodes(self, op=None, include_directives=True):
        """Get the list of "op" nodes in the dag.

        Args:
            op (Type): :class:`qiskit.circuit.Operation` subclass op nodes to
                return. If None, return all op nodes.
            include_directives (bool): include `barrier`, `snapshot` etc.

        Returns:
            list[DAGOpNode]: the list of node ids containing the given op.
        """
        nodes = []
        for node in self._multi_graph.nodes():
            if isinstance(node, DAGOpNode):
                if not include_directives and getattr(node.op, "_directive", False):
                    continue
                if op is None or isinstance(node.op, op):
                    nodes.append(node)
        return nodes

    def gate_nodes(self):
        """Get the list of gate nodes in the dag.

        Returns:
            list[DAGOpNode]: the list of DAGOpNodes that represent gates.
        """
        nodes = []
        for node in self.op_nodes():
            if isinstance(node.op, Gate):
                nodes.append(node)
        return nodes

    def named_nodes(self, *names):
        """Get the set of "op" nodes with the given name."""
        named_nodes = []
        for node in self._multi_graph.nodes():
            if isinstance(node, DAGOpNode) and node.op.name in names:
                named_nodes.append(node)
        return named_nodes

    def two_qubit_ops(self):
        """Get list of 2 qubit operations. Ignore directives like snapshot and barrier."""
        ops = []
        for node in self.op_nodes(include_directives=False):
            if len(node.qargs) == 2:
                ops.append(node)
        return ops

    def multi_qubit_ops(self):
        """Get list of 3+ qubit operations. Ignore directives like snapshot and barrier."""
        ops = []
        for node in self.op_nodes(include_directives=False):
            if len(node.qargs) >= 3:
                ops.append(node)
        return ops

    def longest_path(self):
        """Returns the longest path in the dag as a list of DAGOpNodes, DAGInNodes, and DAGOutNodes."""
        return [self._multi_graph[x] for x in rx.dag_longest_path(self._multi_graph)]

    def successors(self, node):
        """Returns iterator of the successors of a node as DAGOpNodes and DAGOutNodes."""
        return iter(self._multi_graph.successors(node._node_id))

    def predecessors(self, node):
        """Returns iterator of the predecessors of a node as DAGOpNodes and DAGInNodes."""
        return iter(self._multi_graph.predecessors(node._node_id))

    def is_successor(self, node, node_succ):
        """Checks if a second node is in the successors of node."""
        return self._multi_graph.has_edge(node._node_id, node_succ._node_id)

    def is_predecessor(self, node, node_pred):
        """Checks if a second node is in the predecessors of node."""
        return self._multi_graph.has_edge(node_pred._node_id, node._node_id)

    def quantum_predecessors(self, node):
        """Returns iterator of the predecessors of a node that are
        connected by a quantum edge as DAGOpNodes and DAGInNodes."""
        return iter(
            self._multi_graph.find_predecessors_by_edge(
                node._node_id, lambda edge_data: isinstance(edge_data, Qubit)
            )
        )

    def classical_predecessors(self, node):
        """Returns iterator of the predecessors of a node that are
        connected by a classical edge as DAGOpNodes and DAGInNodes."""
        return iter(
            self._multi_graph.find_predecessors_by_edge(
                node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
            )
        )

    def ancestors(self, node):
        """Returns set of the ancestors of a node as DAGOpNodes and DAGInNodes."""
        return {self._multi_graph[x] for x in rx.ancestors(self._multi_graph, node._node_id)}

    def descendants(self, node):
        """Returns set of the descendants of a node as DAGOpNodes and DAGOutNodes."""
        return {self._multi_graph[x] for x in rx.descendants(self._multi_graph, node._node_id)}

    def bfs_successors(self, node):
        """
        Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
        and [DAGNode] is its successors in  BFS order.
        """
        return iter(rx.bfs_successors(self._multi_graph, node._node_id))

    def quantum_successors(self, node):
        """Returns iterator of the successors of a node that are
        connected by a quantum edge as Opnodes and DAGOutNodes."""
        return iter(
            self._multi_graph.find_successors_by_edge(
                node._node_id, lambda edge_data: isinstance(edge_data, Qubit)
            )
        )

    def classical_successors(self, node):
        """Returns iterator of the successors of a node that are
        connected by a classical edge as DAGOpNodes and DAGInNodes."""
        return iter(
            self._multi_graph.find_successors_by_edge(
                node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
            )
        )

    def remove_op_node(self, node):
        """Remove an operation node n.

        Add edges from predecessors to successors.
        """
        if not isinstance(node, DAGOpNode):
            raise DAGCircuitError(
                'The method remove_op_node only works on DAGOpNodes. A "%s" '
                "node type was wrongly provided." % type(node)
            )

        self._multi_graph.remove_node_retain_edges(
            node._node_id, use_outgoing=False, condition=lambda edge1, edge2: edge1 == edge2
        )
        self._decrement_op(node.op)

    def remove_ancestors_of(self, node):
        """Remove all of the ancestor operation nodes of node."""
        anc = rx.ancestors(self._multi_graph, node)
        # TODO: probably better to do all at once using
        # multi_graph.remove_nodes_from; same for related functions ...

        for anc_node in anc:
            if isinstance(anc_node, DAGOpNode):
                self.remove_op_node(anc_node)

    def remove_descendants_of(self, node):
        """Remove all of the descendant operation nodes of node."""
        desc = rx.descendants(self._multi_graph, node)
        for desc_node in desc:
            if isinstance(desc_node, DAGOpNode):
                self.remove_op_node(desc_node)

    def remove_nonancestors_of(self, node):
        """Remove all of the non-ancestors operation nodes of node."""
        anc = rx.ancestors(self._multi_graph, node)
        comp = list(set(self._multi_graph.nodes()) - set(anc))
        for n in comp:
            if isinstance(n, DAGOpNode):
                self.remove_op_node(n)

    def remove_nondescendants_of(self, node):
        """Remove all of the non-descendants operation nodes of node."""
        dec = rx.descendants(self._multi_graph, node)
        comp = list(set(self._multi_graph.nodes()) - set(dec))
        for n in comp:
            if isinstance(n, DAGOpNode):
                self.remove_op_node(n)

    def front_layer(self):
        """Return a list of op nodes in the first layer of this dag."""
        graph_layers = self.multigraph_layers()
        try:
            next(graph_layers)  # Remove input nodes
        except StopIteration:
            return []

        op_nodes = [node for node in next(graph_layers) if isinstance(node, DAGOpNode)]

        return op_nodes

    def layers(self):
        """Yield a shallow view on a layer of this DAGCircuit for all d layers of this circuit.

        A layer is a circuit whose gates act on disjoint qubits, i.e.,
        a layer has depth 1. The total number of layers equals the
        circuit depth d. The layers are indexed from 0 to d-1 with the
        earliest layer at index 0. The layers are constructed using a
        greedy algorithm. Each returned layer is a dict containing
        {"graph": circuit graph, "partition": list of qubit lists}.

        The returned layer contains new (but semantically equivalent) DAGOpNodes, DAGInNodes,
        and DAGOutNodes. These are not the same as nodes of the original dag, but are equivalent
        via DAGNode.semantic_eq(node1, node2).

        TODO: Gates that use the same cbits will end up in different
        layers as this is currently implemented. This may not be
        the desired behavior.
        """
        graph_layers = self.multigraph_layers()
        try:
            next(graph_layers)  # Remove input nodes
        except StopIteration:
            return

        for graph_layer in graph_layers:

            # Get the op nodes from the layer, removing any input and output nodes.
            op_nodes = [node for node in graph_layer if isinstance(node, DAGOpNode)]

            # Sort to make sure they are in the order they were added to the original DAG
            # It has to be done by node_id as graph_layer is just a list of nodes
            # with no implied topology
            # Drawing tools rely on _node_id to infer order of node creation
            # so we need this to be preserved by layers()
            op_nodes.sort(key=lambda nd: nd._node_id)

            # Stop yielding once there are no more op_nodes in a layer.
            if not op_nodes:
                return

            # Construct a shallow copy of self
            new_layer = self.copy_empty_like()

            for node in op_nodes:
                # this creates new DAGOpNodes in the new_layer
                new_layer.apply_operation_back(node.op, node.qargs, node.cargs)

            # The quantum registers that have an operation in this layer.
            support_list = [
                op_node.qargs
                for op_node in new_layer.op_nodes()
                if not getattr(op_node.op, "_directive", False)
            ]

            yield {"graph": new_layer, "partition": support_list}

    def serial_layers(self):
        """Yield a layer for all gates of this circuit.

        A serial layer is a circuit with one gate. The layers have the
        same structure as in layers().
        """
        for next_node in self.topological_op_nodes():
            new_layer = self.copy_empty_like()

            # Save the support of the operation we add to the layer
            support_list = []
            # Operation data
            op = copy.copy(next_node.op)
            qargs = copy.copy(next_node.qargs)
            cargs = copy.copy(next_node.cargs)
            condition = copy.copy(getattr(next_node.op, "condition", None))
            _ = self._bits_in_condition(condition)

            # Add node to new_layer
            new_layer.apply_operation_back(op, qargs, cargs)
            # Add operation to partition
            if not getattr(next_node.op, "_directive", False):
                support_list.append(list(qargs))
            l_dict = {"graph": new_layer, "partition": support_list}
            yield l_dict

    def multigraph_layers(self):
        """Yield layers of the multigraph."""
        first_layer = [x._node_id for x in self.input_map.values()]
        return iter(rx.layers(self._multi_graph, first_layer))

    def collect_runs(self, namelist):
        """Return a set of non-conditional runs of "op" nodes with the given names.

        For example, "... h q[0]; cx q[0],q[1]; cx q[0],q[1]; h q[1]; .."
        would produce the tuple of cx nodes as an element of the set returned
        from a call to collect_runs(["cx"]). If instead the cx nodes were
        "cx q[0],q[1]; cx q[1],q[0];", the method would still return the
        pair in a tuple. The namelist can contain names that are not
        in the circuit's basis.

        Nodes must have only one successor to continue the run.
        """

        def filter_fn(node):
            return (
                isinstance(node, DAGOpNode)
                and node.op.name in namelist
                and getattr(node.op, "condition", None) is None
            )

        group_list = rx.collect_runs(self._multi_graph, filter_fn)
        return {tuple(x) for x in group_list}

    def collect_1q_runs(self):
        """Return a set of non-conditional runs of 1q "op" nodes."""

        def filter_fn(node):
            return (
                isinstance(node, DAGOpNode)
                and len(node.qargs) == 1
                and len(node.cargs) == 0
                and getattr(node.op, "condition", None) is None
                and not node.op.is_parameterized()
                and isinstance(node.op, Gate)
                and hasattr(node.op, "__array__")
            )

        return rx.collect_runs(self._multi_graph, filter_fn)

    def collect_2q_runs(self):
        """Return a set of non-conditional runs of 2q "op" nodes."""

        to_qid = {}
        for i, qubit in enumerate(self.qubits):
            to_qid[qubit] = i

        def filter_fn(node):
            if isinstance(node, DAGOpNode):
                return (
                    isinstance(node.op, Gate)
                    and len(node.qargs) <= 2
                    and not getattr(node.op, "condition", None)
                    and not node.op.is_parameterized()
                )
            else:
                return None

        def color_fn(edge):
            if isinstance(edge, Qubit):
                return to_qid[edge]
            else:
                return None

        return rx.collect_bicolor_runs(self._multi_graph, filter_fn, color_fn)

    def nodes_on_wire(self, wire, only_ops=False):
        """
        Iterator for nodes that affect a given wire.

        Args:
            wire (Bit): the wire to be looked at.
            only_ops (bool): True if only the ops nodes are wanted;
                        otherwise, all nodes are returned.
        Yield:
             Iterator: the successive nodes on the given wire

        Raises:
            DAGCircuitError: if the given wire doesn't exist in the DAG
        """
        current_node = self.input_map.get(wire, None)

        if not current_node:
            raise DAGCircuitError("The given wire %s is not present in the circuit" % str(wire))

        more_nodes = True
        while more_nodes:
            more_nodes = False
            # allow user to just get ops on the wire - not the input/output nodes
            if isinstance(current_node, DAGOpNode) or not only_ops:
                yield current_node

            try:
                current_node = self._multi_graph.find_adjacent_node_by_edge(
                    current_node._node_id, lambda x: wire == x
                )
                more_nodes = True
            except rx.NoSuitableNeighbors:
                pass

    def count_ops(self, *, recurse: bool = True):
        """Count the occurrences of operation names.

        Args:
            recurse: if ``True`` (default), then recurse into control-flow operations.  In all
                cases, this counts only the number of times the operation appears in any possible
                block; both branches of if-elses are counted, and for- and while-loop blocks are
                only counted once.

        Returns:
            Mapping[str, int]: a mapping of operation names to the number of times it appears.
        """
        if not recurse:
            return self._op_names.copy()

        # pylint: disable=cyclic-import
        from qiskit.converters import circuit_to_dag

        def inner(dag, counts):
            for name, count in dag._op_names.items():
                counts[name] += count
            for node in dag.op_nodes(ControlFlowOp):
                for block in node.op.blocks:
                    counts = inner(circuit_to_dag(block), counts)
            return counts

        return dict(inner(self, defaultdict(int)))

    def count_ops_longest_path(self):
        """Count the occurrences of operation names on the longest path.

        Returns a dictionary of counts keyed on the operation name.
        """
        op_dict = {}
        path = self.longest_path()
        path = path[1:-1]  # remove qubits at beginning and end of path
        for node in path:
            name = node.op.name
            if name not in op_dict:
                op_dict[name] = 1
            else:
                op_dict[name] += 1
        return op_dict

    def properties(self):
        """Return a dictionary of circuit properties."""
        summary = {
            "size": self.size(),
            "depth": self.depth(),
            "width": self.width(),
            "qubits": self.num_qubits(),
            "bits": self.num_clbits(),
            "factors": self.num_tensor_factors(),
            "operations": self.count_ops(),
        }
        return summary

    def draw(self, scale=0.7, filename=None, style="color"):
        """
        Draws the dag circuit.

        This function needs `pydot <https://github.com/erocarrera/pydot>`_, which in turn needs
        `Graphviz <https://www.graphviz.org/>`_ to be installed.

        Args:
            scale (float): scaling factor
            filename (str): file path to save image to (format inferred from name)
            style (str):
                'plain': B&W graph;
                'color' (default): color input/output/op nodes

        Returns:
            Ipython.display.Image: if in Jupyter notebook and not saving to file,
            otherwise None.
        """
        from qiskit.visualization.dag_visualization import dag_drawer

        return dag_drawer(dag=self, scale=scale, filename=filename, style=style)

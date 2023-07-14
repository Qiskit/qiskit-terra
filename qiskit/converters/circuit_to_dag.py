# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper function for converting a circuit to a dag"""
import copy

from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.controlflow import is_control_flow_name

def circuit_to_dag(circuit, copy_operations=True, recurse=True):
    """Build a ``DAGCircuit`` object from a ``QuantumCircuit``.

    Args:
        circuit (QuantumCircuit): the input circuit.
        copy_operations (bool): Deep copy the operation objects
            in the :class:`~.QuantumCircuit` for the output :class:`~.DAGCircuit`.
            This should only be set to ``False`` if the input :class:`~.QuantumCircuit`
            will not be used anymore as the operations in the output
            :class:`~.DAGCircuit` will be shared instances and modifications to
            operations in the :class:`~.DAGCircuit` will be reflected in the
            :class:`~.QuantumCircuit` (and vice versa).

    Return:
        DAGCircuit: the DAG representing the input circuit.

    Example:
        .. code-block::

            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.dagcircuit import DAGCircuit
            from qiskit.converters import circuit_to_dag

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            circ = QuantumCircuit(q, c)
            circ.h(q[0])
            circ.cx(q[0], q[1])
            circ.measure(q[0], c[0])
            circ.rz(0.5, q[1]).c_if(c, 2)
            dag = circuit_to_dag(circ)
    """
    if isinstance(circuit, DAGCircuit):
        return circuit

    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name
    dagcircuit.global_phase = circuit.global_phase
    dagcircuit.calibrations = circuit.calibrations
    dagcircuit.metadata = circuit.metadata

    dagcircuit.add_qubits(circuit.qubits)
    dagcircuit.add_clbits(circuit.clbits)

    for register in circuit.qregs:
        dagcircuit.add_qreg(register)

    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    for instruction in circuit.data:
        op = instruction.operation
        if copy_operations:
            op = copy.deepcopy(op)
        if is_control_flow_name(op.name) and recurse:
#        if is_control_flow_name(op.name) and recurse and op.name in ("if_else", "for_loop", "while_loop", "switch_case"):
            op = op.replace_blocks([circuit_to_dag(block, recurse=True) if block is not None else None for block in op.blocks])
        dagcircuit.apply_operation_back(op, instruction.qubits, instruction.clbits)

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    return dagcircuit

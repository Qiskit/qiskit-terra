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


def circuit_to_dag(circuit, copy_operations=True, *, qubit_order=None, clbit_order=None):
    """Build a :class:`.DAGCircuit` object from a :class:`.QuantumCircuit`.

    Args:
        circuit (QuantumCircuit): the input circuit.
        copy_operations (bool): Deep copy the operation objects
            in the :class:`~.QuantumCircuit` for the output :class:`~.DAGCircuit`.
            This should only be set to ``False`` if the input :class:`~.QuantumCircuit`
            will not be used anymore as the operations in the output
            :class:`~.DAGCircuit` will be shared instances and modifications to
            operations in the :class:`~.DAGCircuit` will be reflected in the
            :class:`~.QuantumCircuit` (and vice versa).
        qubit_order (Iterable[~qiskit.circuit.Qubit] or None): the order that the qubits should be
            indexed in the output DAG.  Defaults to the same order as in the circuit.
        clbit_order (Iterable[Clbit] or None): the order that the clbits should be indexed in the
            output DAG.  Defaults to the same order as in the circuit.

    Return:
        DAGCircuit: the DAG representing the input circuit.

    Raises:
        ValueError: if the ``qubit_order`` or ``clbit_order`` parameters do not match the bits in
            the circuit.

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
    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name
    dagcircuit.global_phase = circuit.global_phase
    dagcircuit.calibrations = circuit.calibrations
    dagcircuit.metadata = circuit.metadata

    if qubit_order is None:
        qubits = circuit.qubits
    elif len(qubit_order) != circuit.num_qubits or set(qubit_order) != set(circuit.qubits):
        raise ValueError("'qubit_order' does not contain exactly the same qubits as the circuit")
    else:
        qubits = qubit_order

    if clbit_order is None:
        clbits = circuit.clbits
    elif len(clbit_order) != circuit.num_clbits or set(clbit_order) != set(circuit.clbits):
        raise ValueError("'clbit_order' does not contain exactly the same clbits as the circuit")
    else:
        clbits = clbit_order

    dagcircuit.add_qubits(qubits)
    dagcircuit.add_clbits(clbits)

    for var in circuit.iter_input_vars():
        dagcircuit.add_input_var(var)
    for var in circuit.iter_captured_vars():
        dagcircuit.add_captured_var(var)
    for var in circuit.iter_declared_vars():
        dagcircuit.add_declared_var(var)

    for register in circuit.qregs:
        dagcircuit.add_qreg(register)

    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    for instruction in circuit.data:
        op = instruction.operation
        if copy_operations:
            op = copy.deepcopy(op)
        dagcircuit.apply_operation_back(op, instruction.qubits, instruction.clbits, check=False)

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    dagcircuit._final_permutation = circuit._final_permutation.copy()

    return dagcircuit

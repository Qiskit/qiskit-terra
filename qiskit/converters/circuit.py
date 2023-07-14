from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit

def is_circuit(obj):
    """Return `True` if `obj` implements methods that a circuit must."""
    return isinstance(obj, (QuantumCircuit, DAGCircuit))

def num_qubits(obj):
    if isinstance(obj, QuantumCircuit):
        return obj.num_qubits
    elif isinstance(obj, DAGCircuit):
        return obj.num_qubits()
    else:
        raise RuntimeError(f"num_qubits does not support object of type {type(obj)}")

def num_clbits(obj):
    if isinstance(obj, QuantumCircuit):
        return obj.num_clbits
    elif isinstance(obj, DAGCircuit):
        return obj.num_clbits()
    else:
        raise RuntimeError(f"num_clbits does not support object of type {type(obj)}")

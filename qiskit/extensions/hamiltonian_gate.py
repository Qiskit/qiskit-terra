# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Gate described by the time evolution of a Hermitian Hamiltonian operator.
"""

from numbers import Number
import numpy as np

from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister, ParameterExpression
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.extensions.exceptions import ExtensionError
from qiskit.circuit.exceptions import CircuitError

from .unitary import UnitaryGate


class HamiltonianGate(Gate):
    """Class for representing evolution by a Hamiltonian operator as a gate.

    This gate resolves to a :class:`.UnitaryGate` as :math:`U(t) = exp(-i t H)`,
    which can be decomposed into basis gates if it is 2 qubits or less, or
    simulated directly in Aer for more qubits. Note that you can also directly
    use :meth:`.QuantumCircuit.hamiltonian`.
    """

    def __init__(self, data, time, label=None):
        """Create a gate from a hamiltonian operator and evolution time parameter t

        Args:
            data (matrix or Operator): a hermitian operator.
            time (float or ParameterExpression): time evolution parameter.
            label (str): unitary name for backend [Default: None].

        Raises:
            ExtensionError: if input data is not an N-qubit unitary operator.
        """
        if hasattr(data, "to_matrix"):
            # If input is Gate subclass or some other class object that has
            # a to_matrix method this will call that method.
            data = data.to_matrix()
        elif hasattr(data, "to_operator"):
            # If input is a BaseOperator subclass this attempts to convert
            # the object to an Operator so that we can extract the underlying
            # numpy matrix from `Operator.data`.
            data = data.to_operator().data
        # Convert to numpy array in case not already an array
        data = np.array(data, dtype=complex)
        # Check input is unitary
        if not is_hermitian_matrix(data):
            raise ExtensionError("Input matrix is not Hermitian.")
        if isinstance(time, Number) and time != np.real(time):
            raise ExtensionError("Evolution time is not real.")
        # Check input is N-qubit matrix
        input_dim, output_dim = data.shape
        num_qubits = int(np.log2(input_dim))
        if input_dim != output_dim or 2**num_qubits != input_dim:
            raise ExtensionError("Input matrix is not an N-qubit operator.")

        # Store instruction params
        super().__init__("hamiltonian", num_qubits, [data, time], label=label)

    def __eq__(self, other):
        if not isinstance(other, HamiltonianGate):
            return False
        if self.label != other.label:
            return False
        operators_eq = matrix_equal(self.params[0], other.params[0], ignore_phase=False)
        times_eq = self.params[1] == other.params[1]
        return operators_eq and times_eq

    def __array__(self, dtype=None):
        """Return matrix for the unitary."""
        # pylint: disable=unused-argument
        import scipy.linalg

        try:
            return scipy.linalg.expm(-1j * self.params[0] * float(self.params[1]))
        except TypeError as ex:
            raise TypeError(
                "Unable to generate Unitary matrix for "
                "unbound t parameter {}".format(self.params[1])
            ) from ex

    def inverse(self):
        """Return the adjoint of the unitary."""
        return self.adjoint()

    def conjugate(self):
        """Return the conjugate of the Hamiltonian."""
        return HamiltonianGate(np.conj(self.params[0]), -self.params[1])

    def adjoint(self):
        """Return the adjoint of the unitary."""
        return HamiltonianGate(self.params[0], -self.params[1])

    def transpose(self):
        """Return the transpose of the Hamiltonian."""
        return HamiltonianGate(np.transpose(self.params[0]), self.params[1])

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        qc._append(UnitaryGate(self.to_matrix()), q[:], [])
        self.definition = qc

    def qasm(self):
        """Raise an error, as QASM is not defined for the HamiltonianGate."""
        raise ExtensionError("HamiltonianGate has no QASM definition.")

    def validate_parameter(self, parameter):
        """Hamiltonian parameter has to be an ndarray, operator or float."""
        if isinstance(parameter, (float, int, np.ndarray)):
            return parameter
        elif isinstance(parameter, ParameterExpression) and len(parameter.parameters) == 0:
            return float(parameter)
        else:
            raise CircuitError(f"invalid param type {type(parameter)} for gate {self.name}")


def hamiltonian(self, operator, time, qubits, label=None):
    """Apply hamiltonian evolution to qubits.

    This gate resolves to a :class:`.UnitaryGate` as :math:`U(t) = exp(-i t H)`,
    which can be decomposed into basis gates if it is 2 qubits or less, or
    simulated directly in Aer for more qubits.

    Args:
        operator (matrix or Operator): a hermitian operator.
        time (float or ParameterExpression): time evolution parameter.
        qubits (Union[int, Tuple[int]]): The circuit qubits to apply the
            transformation to.
        label (str): unitary name for backend [Default: None].

    Returns:
        QuantumCircuit: The quantum circuit.

    Raises:
        ExtensionError: if input data is not an N-qubit unitary operator.
    """
    if not isinstance(qubits, list):
        qubits = [qubits]

    return self.append(HamiltonianGate(data=operator, time=time, label=label), qubits, [])


QuantumCircuit.hamiltonian = hamiltonian

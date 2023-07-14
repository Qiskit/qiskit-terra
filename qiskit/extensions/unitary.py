# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Arbitrary unitary circuit instruction.
"""

import numpy as np

from qiskit.circuit import Gate, ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister, Qubit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit._utils import _compute_control_matrix
from qiskit.circuit.library.standard_gates import U3Gate
from qiskit.extensions.quantum_initializer import isometry
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import two_qubit_cnot_decompose
from qiskit.extensions.exceptions import ExtensionError

_DECOMPOSER1Q = OneQubitEulerDecomposer("U3")


class UnitaryGate(Gate):
    """Class quantum gates specified by a unitary matrix.

    Example:

        We can create a unitary gate from a unitary matrix then add it to a
        quantum circuit. The matrix can also be directly applied to the quantum
        circuit, see :meth:`.QuantumCircuit.unitary`.

        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.extensions import UnitaryGate

            matrix = [[0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0]]
            gate = UnitaryGate(matrix)

            circuit = QuantumCircuit(2)
            circuit.append(gate, [0, 1])
    """

    def __init__(self, data, label=None):
        """Create a gate from a numeric unitary matrix.

        Args:
            data (matrix or Operator): unitary operator.
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
        if not is_unitary_matrix(data):
            raise ExtensionError("Input matrix is not unitary.")
        # Check input is N-qubit matrix
        input_dim, output_dim = data.shape
        num_qubits = int(np.log2(input_dim))
        if input_dim != output_dim or 2**num_qubits != input_dim:
            raise ExtensionError("Input matrix is not an N-qubit operator.")

        # Store instruction params
        super().__init__("unitary", num_qubits, [data], label=label)

    def __eq__(self, other):
        if not isinstance(other, UnitaryGate):
            return False
        if self.label != other.label:
            return False
        # Should we match unitaries as equal if they are equal
        # up to global phase?
        return matrix_equal(self.params[0], other.params[0], ignore_phase=True)

    def __array__(self, dtype=None):
        """Return matrix for the unitary."""
        # pylint: disable=unused-argument
        return self.params[0]

    def inverse(self):
        """Return the adjoint of the unitary."""
        return self.adjoint()

    def conjugate(self):
        """Return the conjugate of the unitary."""
        return UnitaryGate(np.conj(self.to_matrix()))

    def adjoint(self):
        """Return the adjoint of the unitary."""
        return self.transpose().conjugate()

    def transpose(self):
        """Return the transpose of the unitary."""
        return UnitaryGate(np.transpose(self.to_matrix()))

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""
        if self.num_qubits == 1:
            q = QuantumRegister(1, "q")
            qc = QuantumCircuit(q, name=self.name)
            theta, phi, lam, global_phase = _DECOMPOSER1Q.angles_and_phase(self.to_matrix())
            qc._append(U3Gate(theta, phi, lam), [q[0]], [])
            qc.global_phase = global_phase
            self.definition = qc
        elif self.num_qubits == 2:
            self.definition = two_qubit_cnot_decompose(self.to_matrix())
        else:
            from qiskit.quantum_info.synthesis.qsd import (  # pylint: disable=cyclic-import
                qs_decomposition,
            )

            self.definition = qs_decomposition(self.to_matrix())

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return controlled version of gate

        Args:
            num_ctrl_qubits (int): number of controls to add to gate (default=1)
            label (str): optional gate label
            ctrl_state (int or str or None): The control state in decimal or as a
                bit string (e.g. '1011'). If None, use 2**num_ctrl_qubits-1.

        Returns:
            UnitaryGate: controlled version of gate.

        Raises:
            QiskitError: Invalid ctrl_state.
            ExtensionError: Non-unitary controlled unitary.
        """
        mat = self.to_matrix()
        cmat = _compute_control_matrix(mat, num_ctrl_qubits, ctrl_state=None)
        iso = isometry.Isometry(cmat, 0, 0)
        return ControlledGate(
            "c-unitary",
            num_qubits=self.num_qubits + num_ctrl_qubits,
            params=[mat],
            label=label,
            num_ctrl_qubits=num_ctrl_qubits,
            definition=iso.definition,
            ctrl_state=ctrl_state,
            base_gate=self.copy(),
        )

    def _qasm2_decomposition(self):
        """Return an unparameterized version of ourselves, so the OQ2 exporter doesn't choke on the
        non-standard things in our `params` field."""
        out = self.definition.to_gate()
        out.name = self.name
        return out

    def validate_parameter(self, parameter):
        """Unitary gate parameter has to be an ndarray."""
        if isinstance(parameter, np.ndarray):
            return parameter
        else:
            raise CircuitError(f"invalid param type {type(parameter)} in gate {self.name}")


def unitary(self, obj, qubits, label=None):
    """Apply unitary gate specified by ``obj`` to ``qubits``.

    Args:
        obj (matrix or Operator): unitary operator.
        qubits (Union[int, Tuple[int]]): The circuit qubits to apply the
            transformation to.
        label (str): unitary name for backend [Default: None].

    Returns:
        QuantumCircuit: The quantum circuit.

    Raises:
        ExtensionError: if input data is not an N-qubit unitary operator.

    Example:

        Apply a gate specified by a unitary matrix to a quantum circuit

        .. code-block:: python

            from qiskit import QuantumCircuit
            matrix = [[0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0]]
            circuit = QuantumCircuit(2)
            circuit.unitary(matrix, [0, 1])
    """
    gate = UnitaryGate(obj, label=label)
    if isinstance(qubits, QuantumRegister):
        qubits = qubits[:]
    # for single qubit unitary gate, allow an 'int' or a 'list of ints' as qubits.
    if gate.num_qubits == 1:
        if isinstance(qubits, (int, Qubit)) or len(qubits) > 1:
            qubits = [qubits]
    return self.append(gate, qubits, [])


QuantumCircuit.unitary = unitary

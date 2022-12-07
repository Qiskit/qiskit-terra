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
Initialize qubit registers to desired arbitrary state.
"""
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Instruction
from qiskit.circuit import Qubit
from qiskit.circuit.library.data_preparation import StatePreparation

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class Initialize(Instruction):
    """Complex amplitude initialization.

    Class that initializes some flexible collection of qubit registers, implemented by calling
    the :class:`qiskit.circuit.library.data_preparation.StatePreparation` Class.
    Note that Initialize is an Instruction and not a Gate since it contains a reset instruction,
    which is not unitary.
    """

    def __init__(self, params, num_qubits=None):
        r"""Create new initialize composite.

        Args:
            params (str, list, int or Statevector):
                * Statevector: Statevector to initialize to.
                * list: vector of complex amplitudes to initialize to.
                * string: labels of basis states of the Pauli eigenstates Z, X, Y. See
                  :meth:`.Statevector.from_label`.
                  Notice the order of the labels is reversed with respect to the qubit index to
                  be applied to. Example label '01' initializes the qubit zero to :math:`|1\rangle`
                  and the qubit one to :math:`|0\rangle`.
                * int: an integer that is used as a bitmap indicating which qubits to initialize
                  to :math:`|1\rangle`. Example: setting params to 5 would initialize qubit 0 and qubit 2
                  to :math:`|1\rangle` and qubit 1 to :math:`|0\rangle`.

            num_qubits (int): This parameter is only used if params is an int. Indicates the total
                number of qubits in the `initialize` call. Example: `initialize` covers 5 qubits
                and params is 3. This allows qubits 0 and 1 to be initialized to :math:`|1\rangle`
                and the remaining 3 qubits to be initialized to :math:`|0\rangle`.
        """
        self._stateprep = StatePreparation(params, num_qubits)

        super().__init__("initialize", self._stateprep.num_qubits, 0, self._stateprep.params)

    def _define(self):
        q = QuantumRegister(self.num_qubits, "q")
        initialize_circuit = QuantumCircuit(q, name="init_def")
        initialize_circuit.reset(q)
        initialize_circuit.append(self._stateprep, q)
        self.definition = initialize_circuit

    def gates_to_uncompute(self):
        """Call to create a circuit with gates that take the desired vector to zero.

        Returns:
            QuantumCircuit: circuit to take self.params vector to :math:`|{00\\ldots0}\\rangle`
        """
        return self._stateprep._gates_to_uncompute()

    @property
    def params(self):
        """Return initialize params."""
        return self._stateprep.params

    @params.setter
    def params(self, parameters):
        """Set initialize params."""
        self._stateprep.params = parameters

    def broadcast_arguments(self, qargs, cargs):
        return self._stateprep.broadcast_arguments(qargs, cargs)


def initialize(self, params, qubits=None):
    r"""Initialize qubits in a specific state.

    Qubit initialization is done by first resetting the qubits to :math:`|0\rangle`
    followed by calling :class:`qiskit.circuit.library.data_preparation.StatePreparation`
    class to prepare the qubits in a specified state.
    Both these steps are included in the
    :class:`qiskit.extensions.Initialize` instruction.

    Args:
        params (str or list or int):
            * str: labels of basis states of the Pauli eigenstates Z, X, Y. See
              :meth:`.Statevector.from_label`. Notice the order of the labels is reversed with respect
              to the qubit index to be applied to. Example label '01' initializes the qubit zero to
              :math:`|1\rangle` and the qubit one to :math:`|0\rangle`.
            * list: vector of complex amplitudes to initialize to.
            * int: an integer that is used as a bitmap indicating which qubits to initialize
              to :math:`|1\rangle`. Example: setting params to 5 would initialize qubit 0 and qubit 2
              to :math:`|1\rangle` and qubit 1 to :math:`|0\rangle`.

        qubits (QuantumRegister or Qubit or int):
            * QuantumRegister: A list of qubits to be initialized [Default: None].
            * Qubit: Single qubit to be initialized [Default: None].
            * int: Index of qubit to be initialized [Default: None].
            * list: Indexes of qubits to be initialized [Default: None].

    Returns:
        qiskit.circuit.Instruction: a handle to the instruction that was just initialized

    Examples:
        Prepare a qubit in the state :math:`(|0\rangle - |1\rangle) / \sqrt{2}`.

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.initialize([1/np.sqrt(2), -1/np.sqrt(2)], 0)
            circuit.draw()

        output:

        .. parsed-literal::

                 ┌──────────────────────────────┐
            q_0: ┤ Initialize(0.70711,-0.70711) ├
                 └──────────────────────────────┘


        Initialize from a string two qubits in the state :math:`|10\rangle`.
        The order of the labels is reversed with respect to qubit index.
        More information about labels for basis states are in
        :meth:`.Statevector.from_label`.

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.initialize('01', circuit.qubits)
            circuit.draw()

        output:

        .. parsed-literal::

                 ┌──────────────────┐
            q_0: ┤0                 ├
                 │  Initialize(0,1) │
            q_1: ┤1                 ├
                 └──────────────────┘

        Initialize two qubits from an array of complex amplitudes.

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.initialize([0, 1/np.sqrt(2), -1.j/np.sqrt(2), 0], circuit.qubits)
            circuit.draw()

        output:

        .. parsed-literal::

                 ┌────────────────────────────────────┐
            q_0: ┤0                                   ├
                 │  Initialize(0,0.70711,-0.70711j,0) │
            q_1: ┤1                                   ├
                 └────────────────────────────────────┘
    """
    if qubits is None:
        qubits = self.qubits
    elif isinstance(qubits, (int, np.integer, slice, Qubit)):
        qubits = [qubits]
    num_qubits = len(qubits) if isinstance(params, int) else None

    return self.append(Initialize(params, num_qubits), qubits)


QuantumCircuit.initialize = initialize

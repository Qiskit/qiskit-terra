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

"""The evolved operator ansatz."""

from __future__ import annotations
from collections.abc import Sequence

import numpy as np

from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp

from .n_local import NLocal


class EvolvedOperatorAnsatz(NLocal):
    """The evolved operator ansatz."""

    def __init__(
        self,
        operators=None,
        reps: int = 1,
        evolution=None,
        insert_barriers: bool = False,
        name: str = "EvolvedOps",
        parameter_prefix: str | Sequence[str] = "t",
        initial_state: QuantumCircuit | None = None,
        flatten: bool | None = None,
    ):
        """
        Args:
            operators (BaseOperator | QuantumCircuit | list | None): The operators
                to evolve. If a circuit is passed, we assume it implements an already evolved
                operator and thus the circuit is not evolved again. Can be a single operator
                (circuit) or a list of operators (and circuits).
            reps: The number of times to repeat the evolved operators.
            evolution (EvolutionBase | EvolutionSynthesis | None):
                A specification of which evolution synthesis to use for the
                :class:`.PauliEvolutionGate`.
                Defaults to first order Trotterization.
            insert_barriers: Whether to insert barriers in between each evolution.
            name: The name of the circuit.
            parameter_prefix: Set the names of the circuit parameters. If a string, the same prefix
                will be used for each parameters. Can also be a list to specify a prefix per
                operator.
            initial_state: A :class:`.QuantumCircuit` object to prepend to the circuit.
            flatten: Set this to ``True`` to output a flat circuit instead of nesting it inside multiple
                layers of gate objects. By default currently the contents of
                the output circuit will be wrapped in nested objects for
                cleaner visualization. However, if you're using this circuit
                for anything besides visualization its **strongly** recommended
                to set this flag to ``True`` to avoid a large performance
                overhead for parameter binding.
        """
        super().__init__(
            initial_state=initial_state,
            parameter_prefix=parameter_prefix,
            reps=reps,
            insert_barriers=insert_barriers,
            name=name,
            flatten=flatten,
        )
        self._operators = None

        if operators is not None:
            self.operators = operators

        self._evolution = evolution

        # a list of which operators are parameterized, used for internal settings
        self._ops_are_parameterized = None

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        if not super()._check_configuration(raise_on_failure):
            return False

        if self.operators is None:
            if raise_on_failure:
                raise ValueError("The operators are not set.")
            return False

        return True

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this circuit.

        Returns:
            The number of qubits.
        """
        if self.operators is None:
            return 0

        if isinstance(self.operators, list):
            if len(self.operators) == 0:
                return 0
            return self.operators[0].num_qubits

        return self.operators.num_qubits

    @property
    def evolution(self):
        """The evolution converter used to compute the evolution.

        Returns:
            EvolutionSynthesis: The evolution converter used to compute the evolution.
        """

        return self._evolution

    @evolution.setter
    def evolution(self, evol) -> None:
        """Sets the evolution converter used to compute the evolution.

        Args:
            evol (EvolutionSynthesis): An evolution synthesis object
        """
        self._invalidate()
        self._evolution = evol

    @property
    def operators(self):
        """The operators that are evolved in this circuit.

        Returns:
            list: The operators to be evolved (and circuits) contained in this ansatz.
        """
        return self._operators

    @operators.setter
    def operators(self, operators=None) -> None:
        """Set the operators to be evolved.

        operators (Optional[Union[QuantumCircuit, list]]): The operators to evolve.
            If a circuit is passed, we assume it implements an already evolved operator and thus
            the circuit is not evolved again. Can be a single operator (circuit) or a list of
            operators (and circuits).
        """
        operators = _validate_operators(operators)
        self._invalidate()
        self._operators = operators
        if self.num_qubits == 0:
            self.qregs = []
        else:
            self.qregs = [QuantumRegister(self.num_qubits, name="q")]

    # TODO: the `preferred_init_points`-implementation can (and should!) be improved!
    @property
    def preferred_init_points(self):
        """Getter of preferred initial points based on the given initial state."""
        if self._initial_state is None:
            return None
        else:
            # If an initial state was set by the user, then we want to make sure that the VQE does
            # not start from a random point. Thus, we return an all-zero initial point for the
            # optimizer which is used (unless it gets overwritten by a higher-priority setting at
            # runtime of the VQE).
            # However, in order to determine the correct length, we must build the QuantumCircuit
            # first, because otherwise the operators may not be set yet.
            self._build()
            return np.zeros(self.reps * len(self.operators), dtype=float)

    def _evolve_operator(self, operator, time):

        # pylint: disable=cyclic-import
        from qiskit.circuit.library.hamiltonian_gate import HamiltonianGate

        # if the operator is specified as matrix use exact matrix exponentiation
        if isinstance(operator, Operator):
            gate = HamiltonianGate(operator, time)
        # otherwise, use the PauliEvolutionGate
        else:
            from qiskit.synthesis.evolution import LieTrotter

            evolution = LieTrotter() if self._evolution is None else self._evolution
            gate = PauliEvolutionGate(operator, time, synthesis=evolution)

        evolved = QuantumCircuit(operator.num_qubits)
        if not self.flatten:
            evolved.append(gate, evolved.qubits)
        else:
            evolved.compose(gate.definition, evolved.qubits, inplace=True)
        return evolved

    def _build(self):
        if self._is_built:
            return

        # need to check configuration here to ensure the operators are not None
        self._check_configuration()

        coeff = Parameter("c")
        circuits = []

        for op in self.operators:
            # if the operator is already the evolved circuit just append it
            if isinstance(op, QuantumCircuit):
                circuits.append(op)
            else:
                # check if the operator is just the identity, if yes, skip it
                if _is_pauli_identity(op):
                    continue

                evolved = self._evolve_operator(op, coeff)
                circuits.append(evolved)

        self.rotation_blocks = []
        self.entanglement_blocks = circuits

        super()._build()


def _validate_operators(operators):
    if not isinstance(operators, list):
        operators = [operators]

    if len(operators) > 1:
        num_qubits = operators[0].num_qubits
        if any(operators[i].num_qubits != num_qubits for i in range(1, len(operators))):
            raise ValueError("All operators must act on the same number of qubits.")

    return operators


def _validate_prefix(parameter_prefix, operators):
    if isinstance(parameter_prefix, str):
        return len(operators) * [parameter_prefix]
    if len(parameter_prefix) != len(operators):
        raise ValueError("The number of parameter prefixes must match the operators.")

    return parameter_prefix


def _is_pauli_identity(operator):
    if isinstance(operator, SparsePauliOp):
        if len(operator.paulis) == 1:
            operator = operator.paulis[0]  # check if the single Pauli is identity below
        else:
            return False
    if isinstance(operator, Pauli):
        return not np.any(np.logical_or(operator.x, operator.z))
    return False

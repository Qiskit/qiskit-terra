# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A gate to implement time-evolution of operators."""

from __future__ import annotations

from typing import Union, Optional, TYPE_CHECKING
import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.quantum_info import Pauli, SparsePauliOp

if TYPE_CHECKING:
    from qiskit.synthesis.evolution import EvolutionSynthesis


class PauliEvolutionGate(Gate):
    r"""Time-evolution of an operator consisting of Paulis.

    For an operator :math:`H` consisting of Pauli terms and (real) evolution time :math:`t`
    this gate implements

    .. math::

        U(t) = e^{-itH}.

    This gate serves as a high-level definition of the evolution and can be synthesized into
    a circuit using different algorithms.

    The evolution gates are related to the Pauli rotation gates by a factor of 2. For example
    the time evolution of the Pauli :math:`X` operator is connected to the Pauli :math:`X` rotation
    :math:`R_X` by

    .. math::

        U(t) = e^{-itX} = R_X(2t).

    **Examples:**

    .. code-block:: python

        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp

        X = SparsePauliOp("X")
        Z = SparsePauliOp("Z")

        # build the evolution gate
        operator = (Z ^ Z) - 0.1 * (X ^ I)
        evo = PauliEvolutionGate(operator, time=0.2)

        # plug it into a circuit
        circuit = QuantumCircuit(2)
        circuit.append(evo, range(2))
        print(circuit.draw())

    The above will print (note that the ``-0.1`` coefficient is not printed!)::

             ┌──────────────────────────┐
        q_0: ┤0                         ├
             │  exp(-it (ZZ + XI))(0.2) │
        q_1: ┤1                         ├
             └──────────────────────────┘


    **References:**

    [1] G. Li et al. Paulihedral: A Generalized Block-Wise Compiler Optimization
    Framework For Quantum Simulation Kernels (2021).
    [`arXiv:2109.03371 <https://arxiv.org/abs/2109.03371>`_]
    """

    def __init__(
        self,
        operator,
        time: Union[int, float, ParameterExpression] = 1.0,
        label: Optional[str] = None,
        synthesis: Optional[EvolutionSynthesis] = None,
    ) -> None:
        """
        Args:
            operator (Pauli | SparsePauliOp | list):
                The operator to evolve. Can also be provided as list of non-commuting
                operators where the elements are sums of commuting operators.
                For example: ``[XY + YX, ZZ + ZI + IZ, YY]``.
            time: The evolution time.
            label: A label for the gate to display in visualizations. Per default, the label is
                set to ``exp(-it <operators>)`` where ``<operators>`` is the sum of the Paulis.
                Note that the label does not include any coefficients of the Paulis. See the
                class docstring for an example.
            synthesis: A synthesis strategy. If None, the default synthesis is the Lie-Trotter
                product formula with a single repetition.
        """
        if isinstance(operator, list):
            operator = [_to_sparse_pauli_op(op) for op in operator]
        else:
            operator = _to_sparse_pauli_op(operator)

        if synthesis is None:
            from qiskit.synthesis.evolution import LieTrotter

            synthesis = LieTrotter()

        if label is None:
            label = _get_default_label(operator)

        num_qubits = operator[0].num_qubits if isinstance(operator, list) else operator.num_qubits
        super().__init__(name="PauliEvolution", num_qubits=num_qubits, params=[time], label=label)
        self.operator = operator
        self.synthesis = synthesis

    @property
    def time(self) -> Union[float, ParameterExpression]:
        """Return the evolution time as stored in the gate parameters.

        Returns:
            The evolution time.
        """
        return self.params[0]

    @time.setter
    def time(self, time: Union[float, ParameterExpression]) -> None:
        """Set the evolution time.

        Args:
            time: The evolution time.
        """
        self.params = [time]

    def _define(self):
        """Unroll, where the default synthesis is matrix based."""
        self.definition = self.synthesis.synthesize(self)

    def validate_parameter(
        self, parameter: Union[int, float, ParameterExpression]
    ) -> Union[float, ParameterExpression]:
        """Gate parameters should be int, float, or ParameterExpression"""
        if isinstance(parameter, int):
            parameter = float(parameter)

        return super().validate_parameter(parameter)


def _to_sparse_pauli_op(operator):
    """Cast the operator to a SparsePauliOp."""

    if isinstance(operator, Pauli):
        sparse_pauli = SparsePauliOp(operator)
    elif isinstance(operator, SparsePauliOp):
        sparse_pauli = operator
    else:
        raise ValueError(f"Unsupported operator type for evolution: {type(operator)}.")

    if any(np.iscomplex(sparse_pauli.coeffs)):
        raise ValueError("Operator contains complex coefficients, which are not supported.")
    if any(isinstance(coeff, ParameterExpression) for coeff in sparse_pauli.coeffs):
        raise ValueError("Operator contains ParameterExpression, which are not supported.")

    return sparse_pauli


def _get_default_label(operator):
    if isinstance(operator, list):
        label = f"exp(-it ({[' + '.join(op.paulis.to_labels()) for op in operator]}))"
    else:
        if len(operator.paulis) == 1:
            label = f"exp(-it {operator.paulis.to_labels()[0]})"
        # for just a single Pauli don't add brackets around the sum
        else:
            label = f"exp(-it ({' + '.join(operator.paulis.to_labels())}))"

    return label

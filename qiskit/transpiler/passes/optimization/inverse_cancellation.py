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

"""
A generic InverseCancellation pass for any set of gate-inverse pairs.
"""
from typing import List, Tuple, Union

from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from qiskit._accelerate.inverse_cancellation import inverse_cancellation


class InverseCancellation(TransformationPass):
    """Cancel specific Gates which are inverses of each other when they occur back-to-
    back."""

    def __init__(self, gates_to_cancel: List[Union[Gate, Tuple[Gate, Gate]]]):
        """Initialize InverseCancellation pass.

        Args:
            gates_to_cancel: List describing the gates to cancel. Each element of the
                list is either a single gate or a pair of gates. If a single gate, then
                it should be self-inverse. If a pair of gates, then the gates in the
                pair should be inverses of each other.

        Raises:
            TranspilerError: Input is not a self-inverse gate or a pair of inverse gates.
        """

        for gates in gates_to_cancel:
            if isinstance(gates, Gate):
                if gates != gates.inverse():
                    raise TranspilerError(f"Gate {gates.name} is not self-inverse")
            elif isinstance(gates, tuple):
                if len(gates) != 2:
                    raise TranspilerError(
                        f"Too many or too few inputs: {gates}. Only two are allowed."
                    )
                if gates[0] != gates[1].inverse():
                    raise TranspilerError(
                        f"Gate {gates[0].name} and {gates[1].name} are not inverse."
                    )
            else:
                raise TranspilerError(
                    f"InverseCancellation pass does not take input type {type(gates)}. Input must be"
                    " a Gate."
                )

        self.self_inverse_gates = []
        self.inverse_gate_pairs = []
        self.self_inverse_gate_names = set()
        self.inverse_gate_pairs_names = set()

        for gates in gates_to_cancel:
            if isinstance(gates, Gate):
                self.self_inverse_gates.append(gates)
                self.self_inverse_gate_names.add(gates.name)
            else:
                self.inverse_gate_pairs.append(gates)
                self.inverse_gate_pairs_names.update(x.name for x in gates)

        super().__init__()

    def run(self, dag: DAGCircuit):
        """Run the InverseCancellation pass on `dag`.

        Args:
            dag: the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        inverse_cancellation(
            dag,
            self.inverse_gate_pairs,
            self.self_inverse_gates,
            self.inverse_gate_pairs_names,
            self.self_inverse_gate_names,
        )
        return dag

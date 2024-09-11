# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Splits each two-qubit gate in the `dag` into two single-qubit gates, if possible without error."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit._accelerate.split_2q_unitaries import split_2q_unitaries


class Split2QUnitaries(TransformationPass):
    """Attempt to splits two-qubit unitaries in a :class:`.DAGCircuit` into two single-qubit gates.

    This pass will analyze all :class:`.UnitaryGate` instances and determine whether the
    matrix is actually a product of 2 single qubit gates. In these cases the 2q gate can be
    simplified into two single qubit gates and this pass will perform this optimization and will
    replace the two qubit gate with two single qubit :class:`.UnitaryGate`.
    """

    def __init__(self, fidelity: float = 1.0 - 1e-16):
        """
        Args:
            fidelity: Allowed tolerance for splitting two-qubit unitaries and gate decompositions.
        """
        super().__init__()
        self.requested_fidelity = fidelity

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the Split2QUnitaries pass on `dag`."""
        split_2q_unitaries(dag, self.requested_fidelity)
        return dag

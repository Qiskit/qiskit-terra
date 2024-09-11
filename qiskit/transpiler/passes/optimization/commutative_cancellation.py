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

"""Cancel the redundant (self-adjoint) gates through commutation relations."""
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.commutation_library import StandardGateCommutations

from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit._accelerate import commutation_cancellation
from qiskit._accelerate.commutation_checker import CommutationChecker

from qiskit.transpiler.passes.utils.control_flow import trivial_recurse

_CUTOFF_PRECISION = 1e-5


class CommutativeCancellation(TransformationPass):
    """Cancel the redundant (self-adjoint) gates through commutation relations.

    Pass for cancelling self-inverse gates/rotations. The cancellation utilizes
    the commutation relations in the circuit. Gates considered include::

        H, X, Y, Z, CX, CY, CZ
    """

    def __init__(self, basis_gates=None, target=None):
        """
        CommutativeCancellation initializer.

        Args:
            basis_gates (list[str]): Basis gates to consider, e.g.
                ``['u3', 'cx']``. For the effects of this pass, the basis is
                the set intersection between the ``basis_gates`` parameter
                and the gates in the dag.
            target (Target): The :class:`~.Target` representing the target backend, if both
                ``basis_gates`` and ``target`` are specified then this argument will take
                precedence and ``basis_gates`` will be ignored.
        """
        super().__init__()
        if basis_gates:
            self.basis = set(basis_gates)
        else:
            self.basis = set()
        self.target = target
        if target is not None:
            self.basis = set(target.operation_names)

        self._var_z_map = {"rz": RZGate, "p": PhaseGate, "u1": U1Gate}

        self._z_rotations = {"p", "z", "u1", "rz", "t", "s"}
        self._x_rotations = {"x", "rx"}
        self._gates = {"cx", "cy", "cz", "h", "y"}  # Now the gates supported are hard-coded

        # build a commutation checker restricted to the gates we cancel -- the others we
        # do not have to investigate, which allows to save time
        self._commutation_checker = CommutationChecker(
            StandardGateCommutations, gates=self._gates | self._z_rotations | self._x_rotations
        )

    @trivial_recurse
    def run(self, dag):
        """Run the CommutativeCancellation pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        commutation_cancellation.cancel_commutations(dag, self._commutation_checker, self.basis)
        return dag

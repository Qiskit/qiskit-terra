# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Analysis pass to find commutation relations between DAG nodes."""

from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit._accelerate.commutation_analysis import analyze_commutations


class CommutationAnalysis(AnalysisPass):
    """Analysis pass to find commutation relations between DAG nodes.

    ``property_set['commutation_set']`` is a dictionary that describes
    the commutation relations on a given wire, all the gates on a wire
    are grouped into a set of gates that commute.
    """

    def __init__(self, *, _commutation_checker=None):
        super().__init__()
        # allow setting a private commutation checker, this allows better performance if we
        # do not care about commutations of all gates, but just a subset
        if _commutation_checker is None:
            _commutation_checker = scc

        self.comm_checker = _commutation_checker

    def run(self, dag):
        """Run the CommutationAnalysis pass on `dag`.

        Run the pass on the DAG, and write the discovered commutation relations
        into the ``property_set``.
        """
        # Initiate the commutation set
        self.property_set["commutation_set"] = analyze_commutations(dag, self.comm_checker.cc)

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

from __future__ import annotations

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes.optimization import Collect2qBlocks, ConsolidateBlocks
from qiskit.transpiler.passes.synthesis import UnitarySynthesis
from qiskit.transpiler.target import Target
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit._accelerate.two_qubit_peephole import two_qubit_unitary_peephole_optimize


class TwoQubitPeepholeOptimization(TransformationPass):
    """Unified two qubit unitary peephole optimization"""

    def __init__(
        self,
        target: Target,
        approximation_degree: float | None = 1.0,
        method: str = "default",
        plugin_config: dict = None,
    ):
        super().__init__()
        self._target = target
        self._approximation_degree = approximation_degree
        self._pm = None
        if method != "default":
            self._pm = PassManager(
                [
                    Collect2qBlocks(),
                    ConsolidateBlocks(
                        target=self._target, approximation_degree=self._approximation_degree
                    ),
                    UnitarySynthesis(
                        target=target,
                        approximation_degree=approximation_degree,
                        method=method,
                        plugin_config=plugin_config,
                    ),
                ]
            )

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if self._pm is not None:
            return self._pm.run(dag)
        return two_qubit_unitary_peephole_optimize(dag, self._target, self._approximation_degree)

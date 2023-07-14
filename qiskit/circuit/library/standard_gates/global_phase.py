# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Global Phase Gate"""

from typing import Optional
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType


class GlobalPhaseGate(Gate):
    r"""The global phase gate (:math:`e^{i\theta}`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`

    **Mathematical Representation:**

    .. math::
        \text{GlobalPhaseGate}\ =
            \begin{pmatrix}
                e^{i\theta}
            \end{pmatrix}
    """

    def __init__(self, phase: ParameterValueType, label: Optional[str] = None):
        """
        Args:
            phase: The value of phase it takes.
            label: An optional label for the gate.
        """
        super().__init__("global_phase", 0, [phase], label=label)

    def _define(self):
        q = QuantumRegister(0, "q")
        qc = QuantumCircuit(q, name=self.name, global_phase=self.params[0])

        self.definition = qc

    def inverse(self):
        r"""Return inverted GLobalPhaseGate gate.

        :math:`\text{GlobalPhaseGate}(\lambda)^{\dagger} = \text{GlobalPhaseGate}(-\lambda)`
        """
        return GlobalPhaseGate(-self.params[0])

    def __array__(self, dtype=complex):
        """Return a np.array for the global_phase gate."""
        theta = self.params[0]
        return np.array([[np.exp(1j * theta)]], dtype=dtype)

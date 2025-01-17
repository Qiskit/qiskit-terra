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

# pylint: disable=missing-module-docstring

from __future__ import annotations

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType


def rzx_xz(theta: ParameterValueType | None = None):
    """RZX-based template for CX - RXGate - CX.

    .. code-block:: text

             ┌───┐         ┌───┐┌─────────┐┌─────────┐┌─────────┐┌──────────┐»
        q_0: ┤ X ├─────────┤ X ├┤ RZ(π/2) ├┤ RX(π/2) ├┤ RZ(π/2) ├┤0         ├»
             └─┬─┘┌───────┐└─┬─┘└─────────┘└─────────┘└─────────┘│  RZX(-ϴ) │»
        q_1: ──■──┤ RX(ϴ) ├──■───────────────────────────────────┤1         ├»
                  └───────┘                                      └──────────┘»
        «     ┌─────────┐┌─────────┐┌─────────┐
        «q_0: ┤ RZ(π/2) ├┤ RX(π/2) ├┤ RZ(π/2) ├
        «     └─────────┘└─────────┘└─────────┘
        «q_1: ─────────────────────────────────
        «
    """
    if theta is None:
        theta = Parameter("ϴ")

    qc = QuantumCircuit(2)
    qc.cx(1, 0)
    qc.rx(theta, 1)
    qc.cx(1, 0)

    qc.rz(np.pi / 2, 0)
    qc.rx(np.pi / 2, 0)
    qc.rz(np.pi / 2, 0)
    qc.rzx(-1 * theta, 0, 1)
    qc.rz(np.pi / 2, 0)
    qc.rx(np.pi / 2, 0)
    qc.rz(np.pi / 2, 0)
    return qc

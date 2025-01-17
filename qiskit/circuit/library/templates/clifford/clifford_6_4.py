# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-module-docstring

from qiskit.circuit.quantumcircuit import QuantumCircuit


def clifford_6_4():
    """
    Clifford template 6_4:

    .. code-block:: text

             ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
        q_0: ┤ S ├┤ H ├┤ S ├┤ H ├┤ S ├┤ H ├
             └───┘└───┘└───┘└───┘└───┘└───┘

    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(1)
    qc.s(0)
    qc.h(0)
    qc.s(0)
    qc.h(0)
    qc.s(0)
    qc.h(0)
    return qc

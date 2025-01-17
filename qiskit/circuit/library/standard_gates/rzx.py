# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit ZX-rotation gate."""

from __future__ import annotations

import math
from typing import Optional
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType, ParameterExpression
from qiskit._accelerate.circuit import StandardGate


class RZXGate(Gate):
    r"""A parametric 2-qubit :math:`Z \otimes X` interaction (rotation about ZX).

    This gate is maximally entangling at :math:`\theta = \pi/2`.

    The cross-resonance gate (CR) for superconducting qubits implements
    a ZX interaction (however other terms are also present in an experiment).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rzx` method.

    **Circuit Symbol:**

    .. code-block:: text

             ┌─────────┐
        q_0: ┤0        ├
             │  Rzx(θ) │
        q_1: ┤1        ├
             └─────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        R_{ZX}(\theta)\ q_0, q_1 = \exp\left(-i \frac{\theta}{2} X{\otimes}Z\right) =
            \begin{pmatrix}
                \cos\left(\rotationangle\right) & 0 & -i\sin\left(\rotationangle\right) & 0 \\
                0 & \cos\left(\rotationangle\right) & 0 & i\sin\left(\rotationangle\right) \\
                -i\sin\left(\rotationangle\right) & 0 & \cos\left(\rotationangle\right) & 0 \\
                0 & i\sin\left(\rotationangle\right) & 0 & \cos\left(\rotationangle\right)
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In the above example we apply the gate
        on (q_0, q_1) which results in the :math:`X \otimes Z` tensor order.
        Instead, if we apply it on (q_1, q_0), the matrix will
        be :math:`Z \otimes X`:

        .. code-block:: text

                 ┌─────────┐
            q_0: ┤1        ├
                 │  Rzx(θ) │
            q_1: ┤0        ├
                 └─────────┘

        .. math::

            \newcommand{\rotationangle}{\frac{\theta}{2}}

            R_{ZX}(\theta)\ q_1, q_0 = exp(-i \frac{\theta}{2} Z{\otimes}X) =
                \begin{pmatrix}
                    \cos(\rotationangle)   & -i\sin(\rotationangle) & 0           & 0          \\
                    -i\sin(\rotationangle) & \cos(\rotationangle)   & 0           & 0          \\
                    0           & 0           & \cos(\rotationangle)   & i\sin(\rotationangle) \\
                    0           & 0           & i\sin(\rotationangle)  & \cos(\rotationangle)
                \end{pmatrix}

        This is a direct sum of RX rotations, so this gate is equivalent to a
        uniformly controlled (multiplexed) RX gate:

        .. math::

            R_{ZX}(\theta)\ q_1, q_0 =
                \begin{pmatrix}
                    RX(\theta) & 0 \\
                    0 & RX(-\theta)
                \end{pmatrix}

    **Examples:**

        .. math::

            R_{ZX}(\theta = 0)\ q_0, q_1 = I

        .. math::

            R_{ZX}(\theta = 2\pi)\ q_0, q_1 = -I

        .. math::

            R_{ZX}(\theta = \pi)\ q_0, q_1 = -i X \otimes Z

        .. math::

            R_{ZX}(\theta = \frac{\pi}{2})\ q_0, q_1 = \frac{1}{\sqrt{2}}
                                    \begin{pmatrix}
                                        1  & 0 & -i & 0 \\
                                        0  & 1 & 0  & i \\
                                        -i & 0 & 1  & 0 \\
                                        0  & i & 0  & 1
                                    \end{pmatrix}
    """

    _standard_gate = StandardGate.RZXGate

    def __init__(
        self, theta: ParameterValueType, label: Optional[str] = None, *, duration=None, unit="dt"
    ):
        """Create new RZX gate."""
        super().__init__("rzx", 2, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        """
        gate rzx(theta) a, b { h b; cx a, b; u1(theta) b; cx a, b; h b;}
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .x import CXGate
        from .rz import RZGate

        # q_0: ───────■─────────────■───────
        #      ┌───┐┌─┴─┐┌───────┐┌─┴─┐┌───┐
        # q_1: ┤ H ├┤ X ├┤ Rz(0) ├┤ X ├┤ H ├
        #      └───┘└───┘└───────┘└───┘└───┘
        theta = self.params[0]
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (HGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RZGate(theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a (multi-)controlled-RZX gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate. If ``None``, this is set to ``True`` if
                the gate contains free parameters, in which case it cannot
                yet be synthesized.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if annotated is None:
            annotated = any(isinstance(p, ParameterExpression) for p in self.params)

        gate = super().control(
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            annotated=annotated,
        )
        return gate

    def inverse(self, annotated: bool = False):
        """Return inverse RZX gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RZXGate` with an inverted parameter value.

         Returns:
            RZXGate: inverse gate.
        """
        return RZXGate(-self.params[0])

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the RZX gate."""
        import numpy

        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        half_theta = float(self.params[0]) / 2
        cos = math.cos(half_theta)
        isin = 1j * math.sin(half_theta)
        return numpy.array(
            [[cos, 0, -isin, 0], [0, cos, 0, isin], [-isin, 0, cos, 0], [0, isin, 0, cos]],
            dtype=dtype,
        )

    def power(self, exponent: float, annotated: bool = False):
        (theta,) = self.params
        return RZXGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, RZXGate):
            return self._compare_parameters(other)
        return False

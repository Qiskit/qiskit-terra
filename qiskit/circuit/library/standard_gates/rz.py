# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rotation around the Z axis."""

from __future__ import annotations

from cmath import exp
from typing import Optional, Union
from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit._accelerate.circuit import StandardGate


class RZGate(Gate):
    r"""Single-qubit rotation about the Z axis.

    This is a diagonal gate. It can be implemented virtually in hardware
    via framechanges (i.e. at zero error and duration).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rz` method.

    **Circuit symbol:**

    .. code-block:: text

             ┌───────┐
        q_0: ┤ Rz(λ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        RZ(\lambda) = \exp\left(-i\frac{\lambda}{2}Z\right) =
            \begin{pmatrix}
                e^{-i\frac{\lambda}{2}} & 0 \\
                0 & e^{i\frac{\lambda}{2}}
            \end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.U1Gate`
        This gate is equivalent to U1 up to a phase factor.

            .. math::

                U1(\lambda) = e^{i{\lambda}/2}RZ(\lambda)

        Reference for virtual Z gate implementation:
        `1612.00858 <https://arxiv.org/abs/1612.00858>`_
    """

    _standard_gate = StandardGate.RZGate

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new RZ gate."""
        super().__init__("rz", 1, [phi], label=label)

    def _define(self):
        """
        gate rz(phi) a { u1(phi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate

        q = QuantumRegister(1, "q")
        theta = self.params[0]
        qc = QuantumCircuit(q, name=self.name, global_phase=-theta / 2)
        rules = [(U1Gate(theta), [q[0]], [])]
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
        """Return a (multi-)controlled-RZ gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate. If ``None``, this is set to ``True`` if
                the gate contains free parameters and more than one control qubit, in which
                case it cannot yet be synthesized. Otherwise it is set to ``False``.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        # deliberately capture annotated in [None, False] here
        if not annotated and num_ctrl_qubits == 1:
            gate = CRZGate(self.params[0], label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted RZ gate

        :math:`RZ(\lambda)^{\dagger} = RZ(-\lambda)`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RZGate` with an inverted parameter value.

        Returns:
            RZGate: inverse gate.
        """
        return RZGate(-self.params[0])

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the RZ gate."""
        import numpy as np

        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        ilam2 = 0.5j * float(self.params[0])
        return np.array([[exp(-ilam2), 0], [0, exp(ilam2)]], dtype=dtype)

    def power(self, exponent: float, annotated: bool = False):
        (theta,) = self.params
        return RZGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, RZGate):
            return self._compare_parameters(other)
        return False


class CRZGate(ControlledGate):
    r"""Controlled-RZ gate.

    This is a diagonal but non-symmetric gate that induces a
    phase on the state of the target qubit, depending on the control state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.crz` method.

    **Circuit symbol:**

    .. code-block:: text

        q_0: ────■────
             ┌───┴───┐
        q_1: ┤ Rz(θ) ├
             └───────┘

    **Matrix representation:**

    .. math::

        CRZ(\theta)\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| + RZ(\theta) \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & e^{-i\frac{\lambda}{2}} & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\frac{\theta}{2}}
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌───────┐
            q_0: ┤ Rz(θ) ├
                 └───┬───┘
            q_1: ────■────

        .. math::

            CRZ(\theta)\ q_1, q_0 =
                |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes RZ(\theta) =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & e^{-i\frac{\theta}{2}} & 0 \\
                    0 & 0 & 0 & e^{i\frac{\theta}{2}}
                \end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CU1Gate`:
        Due to the global phase difference in the matrix definitions
        of U1 and RZ, CU1 and CRZ are different gates with a relative
        phase difference.
    """

    _standard_gate = StandardGate.CRZGate

    def __init__(
        self,
        theta: ParameterValueType,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        _base_label=None,
    ):
        """Create new CRZ gate."""
        super().__init__(
            "crz",
            2,
            [theta],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=RZGate(theta, label=_base_label),
        )

    def _define(self):
        """
        gate crz(lambda) a,b
        { rz(lambda/2) b; cx a,b;
          rz(-lambda/2) b; cx a,b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate

        # q_0: ─────────────■────────────────■──
        #      ┌─────────┐┌─┴─┐┌──────────┐┌─┴─┐
        # q_1: ┤ Rz(λ/2) ├┤ X ├┤ Rz(-λ/2) ├┤ X ├
        #      └─────────┘└───┘└──────────┘└───┘
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (RZGate(self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RZGate(-self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Return inverse CRZ gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CRZGate` with an inverted parameter value.

         Returns:
            CRZGate: inverse gate.
        """
        return CRZGate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the CRZ gate."""
        import numpy

        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        arg = 1j * float(self.params[0]) / 2
        if self.ctrl_state:
            return numpy.array(
                [[1, 0, 0, 0], [0, exp(-arg), 0, 0], [0, 0, 1, 0], [0, 0, 0, exp(arg)]],
                dtype=dtype,
            )
        else:
            return numpy.array(
                [[exp(-arg), 0, 0, 0], [0, 1, 0, 0], [0, 0, exp(arg), 0], [0, 0, 0, 1]],
                dtype=dtype,
            )

    def __eq__(self, other):
        if isinstance(other, CRZGate):
            return self._compare_parameters(other) and self.ctrl_state == other.ctrl_state
        return False

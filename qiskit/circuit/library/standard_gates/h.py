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

"""Hadamard gate."""

from __future__ import annotations

from math import sqrt, pi
from typing import Optional, Union
import numpy
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit._accelerate.circuit import StandardGate

_H_ARRAY = 1 / sqrt(2) * numpy.array([[1, 1], [1, -1]], dtype=numpy.complex128)


@with_gate_array(_H_ARRAY)
class HGate(SingletonGate):
    r"""Single-qubit Hadamard gate.

    This gate is a \pi rotation about the X+Z axis, and has the effect of
    changing computation basis from :math:`|0\rangle,|1\rangle` to
    :math:`|+\rangle,|-\rangle` and vice-versa.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.h` method.

    **Circuit symbol:**

    .. code-block:: text

             ┌───┐
        q_0: ┤ H ├
             └───┘

    **Matrix Representation:**

    .. math::

        H = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                1 & 1 \\
                1 & -1
            \end{pmatrix}
    """

    _standard_gate = StandardGate.HGate

    def __init__(self, label: Optional[str] = None, *, duration=None, unit="dt"):
        """Create new H gate."""
        super().__init__("h", 1, [], label=label, duration=duration, unit=unit)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate h a { u2(0,pi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u2 import U2Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U2Gate(0, pi), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        """Return a (multi-)controlled-H gate.

        One control qubit returns a CH gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate. If ``None``, this is handled as ``False``.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated and num_ctrl_qubits == 1:
            gate = CHGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted H gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            HGate: inverse gate (self-inverse).
        """
        return HGate()  # self-inverse

    def __eq__(self, other):
        return isinstance(other, HGate)


@with_controlled_gate_array(_H_ARRAY, num_ctrl_qubits=1)
class CHGate(SingletonControlledGate):
    r"""Controlled-Hadamard gate.

    Applies a Hadamard on the target qubit if the control is
    in the :math:`|1\rangle` state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ch` method.

    **Circuit symbol:**

    .. code-block:: text

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ H ├
             └───┘

    **Matrix Representation:**

    .. math::

        CH\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| + H \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & \frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\
                0 & 0 & 1 & 0 \\
                0 & \frac{1}{\sqrt{2}} & 0 & -\frac{1}{\sqrt{2}}
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌───┐
            q_0: ┤ H ├
                 └─┬─┘
            q_1: ──■──

        .. math::

            CH\ q_1, q_0 =
                |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes H =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
                    0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
                \end{pmatrix}
    """

    _standard_gate = StandardGate.CHGate

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[int, str]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create new CH gate."""
        super().__init__(
            "ch",
            2,
            [],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=HGate(label=_base_label),
            duration=duration,
            unit=unit,
            _base_label=_base_label,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """
        gate ch a,b {
            s b;
            h b;
            t b;
            cx a, b;
            tdg b;
            h b;
            sdg b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate  # pylint: disable=cyclic-import
        from .t import TGate, TdgGate
        from .s import SGate, SdgGate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (SGate(), [q[1]], []),
            (HGate(), [q[1]], []),
            (TGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (TdgGate(), [q[1]], []),
            (HGate(), [q[1]], []),
            (SdgGate(), [q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Return inverted CH gate (itself)."""
        return CHGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __eq__(self, other):
        return isinstance(other, CHGate) and self.ctrl_state == other.ctrl_state

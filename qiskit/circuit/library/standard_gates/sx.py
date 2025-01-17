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

"""Sqrt(X) and C-Sqrt(X) gates."""

from __future__ import annotations

from math import pi
from typing import Optional, Union
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit._accelerate.circuit import StandardGate


_SX_ARRAY = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]
_SXDG_ARRAY = [[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]]


@with_gate_array(_SX_ARRAY)
class SXGate(SingletonGate):
    r"""The single-qubit Sqrt(X) gate (:math:`\sqrt{X}`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.sx` method.

    **Matrix Representation:**

    .. math::

        \sqrt{X} = \frac{1}{2} \begin{pmatrix}
                1 + i & 1 - i \\
                1 - i & 1 + i
            \end{pmatrix}

    **Circuit symbol:**

    .. code-block:: text

             ┌────┐
        q_0: ┤ √X ├
             └────┘

    .. note::

        A global phase difference exists between the definitions of
        :math:`RX(\pi/2)` and :math:`\sqrt{X}`.

        .. math::

            RX(\pi/2) = \frac{1}{\sqrt{2}} \begin{pmatrix}
                        1 & -i \\
                        -i & 1
                      \end{pmatrix}
                    = e^{-i \pi/4} \sqrt{X}

    """

    _standard_gate = StandardGate.SXGate

    def __init__(self, label: Optional[str] = None, *, duration=None, unit="dt"):
        """Create new SX gate."""
        super().__init__("sx", 1, [], label=label, duration=duration, unit=unit)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate sx a { rz(-pi/2) a; h a; rz(-pi/2); }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .s import SdgGate
        from .h import HGate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name, global_phase=pi / 4)
        rules = [(SdgGate(), [q[0]], []), (HGate(), [q[0]], []), (SdgGate(), [q[0]], [])]
        for operation, qubits, clbits in rules:
            qc._append(operation, qubits, clbits)
        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Return inverse SX gate (i.e. SXdg).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.SXdgGate`.

        Returns:
            SXdgGate: inverse of :class:`.SXGate`.
        """
        return SXdgGate()

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a (multi-)controlled-SX gate.

        One control returns a CSX gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate. If ``None``, this is handled as ``False``.

        Returns:
            SingletonControlledGate: controlled version of this gate.
        """
        if not annotated and num_ctrl_qubits == 1:
            gate = CSXGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def __eq__(self, other):
        return isinstance(other, SXGate)


@with_gate_array(_SXDG_ARRAY)
class SXdgGate(SingletonGate):
    r"""The inverse single-qubit Sqrt(X) gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.sxdg` method.

    .. math::

        \sqrt{X}^{\dagger} = \frac{1}{2} \begin{pmatrix}
                1 - i & 1 + i \\
                1 + i & 1 - i
            \end{pmatrix}


    .. note::

        A global phase difference exists between the definitions of
        :math:`RX(-\pi/2)` and :math:`\sqrt{X}^{\dagger}`.

        .. math::

            RX(-\pi/2) = \frac{1}{\sqrt{2}} \begin{pmatrix}
                        1 & i \\
                        i & 1
                      \end{pmatrix}
                    = e^{-i \pi/4} \sqrt{X}^{\dagger}
    """

    _standard_gate = StandardGate.SXdgGate

    def __init__(self, label: Optional[str] = None, *, duration=None, unit="dt"):
        """Create new SXdg gate."""
        super().__init__("sxdg", 1, [], label=label, duration=duration, unit=unit)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate sxdg a { rz(pi/2) a; h a; rz(pi/2); }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .s import SGate
        from .h import HGate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name, global_phase=-pi / 4)
        rules = [(SGate(), [q[0]], []), (HGate(), [q[0]], []), (SGate(), [q[0]], [])]
        for operation, qubits, clbits in rules:
            qc._append(operation, qubits, clbits)
        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Return inverse SXdg gate (i.e. SX).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.SXGate`.

        Returns:
            SXGate: inverse of :class:`.SXdgGate`
        """
        return SXGate()

    def __eq__(self, other):
        return isinstance(other, SXdgGate)


@with_controlled_gate_array(_SX_ARRAY, num_ctrl_qubits=1)
class CSXGate(SingletonControlledGate):
    r"""Controlled-√X gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.csx` method.

    **Circuit symbol:**

    .. code-block:: text

        q_0: ──■──
             ┌─┴──┐
        q_1: ┤ √X ├
             └────┘

    **Matrix representation:**

    .. math::

        C\sqrt{X} \ q_0, q_1 =
        I \otimes |0 \rangle\langle 0| + \sqrt{X} \otimes |1 \rangle\langle 1|  =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & (1 + i) / 2 & 0 & (1 - i) / 2 \\
                0 & 0 & 1 & 0 \\
                0 & (1 - i) / 2 & 0 & (1 + i) / 2
            \end{pmatrix}


    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be `q_1`. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌────┐
            q_0: ┤ √X ├
                 └─┬──┘
            q_1: ──■──

        .. math::

            C\sqrt{X}\ q_1, q_0 =
                |0 \rangle\langle 0| \otimes I + |1 \rangle\langle 1| \otimes \sqrt{X} =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & (1 + i) / 2 & (1 - i) / 2 \\
                    0 & 0 & (1 - i) / 2 & (1 + i) / 2
                \end{pmatrix}

    """

    _standard_gate = StandardGate.CSXGate

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create new CSX gate."""
        super().__init__(
            "csx",
            2,
            [],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=SXGate(label=_base_label),
            duration=duration,
            unit=unit,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """
        gate csx a,b { h b; cu1(pi/2) a,b; h b; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .u1 import CU1Gate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(HGate(), [q[1]], []), (CU1Gate(pi / 2), [q[0], q[1]], []), (HGate(), [q[1]], [])]
        for operation, qubits, clbits in rules:
            qc._append(operation, qubits, clbits)
        self.definition = qc

    def __eq__(self, other):
        return isinstance(other, CSXGate) and self.ctrl_state == other.ctrl_state

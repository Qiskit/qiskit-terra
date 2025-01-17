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

"""The S, Sdg, CS and CSdg gates."""

from __future__ import annotations

from math import pi
from typing import Optional, Union

import numpy

from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit._accelerate.circuit import StandardGate


_S_ARRAY = numpy.array([[1, 0], [0, 1j]])
_SDG_ARRAY = numpy.array([[1, 0], [0, -1j]])


@with_gate_array(_S_ARRAY)
class SGate(SingletonGate):
    r"""Single qubit S gate (Z**0.5).

    It induces a :math:`\pi/2` phase, and is sometimes called the P gate (phase).

    This is a Clifford gate and a square-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.s` method.

    **Matrix Representation:**

    .. math::

        S = \begin{pmatrix}
                1 & 0 \\
                0 & i
            \end{pmatrix}

    **Circuit symbol:**

    .. code-block:: text

             ┌───┐
        q_0: ┤ S ├
             └───┘

    Equivalent to a :math:`\pi/2` radian rotation about the Z axis.
    """

    _standard_gate = StandardGate.SGate

    def __init__(self, label: Optional[str] = None, *, duration=None, unit="dt"):
        """Create new S gate."""
        super().__init__("s", 1, [], label=label, duration=duration, unit=unit)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate s a { u1(pi/2) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        from .u1 import U1Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(pi / 2), [q[0]], [])]
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
        """Return a (multi-)controlled-S gate.

        One control qubit returns a :class:`.CSGate`.

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
            gate = CSGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        """Return inverse of S (SdgGate).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.SdgGate`.

        Returns:
            SdgGate: inverse of :class:`.SGate`
        """
        return SdgGate()

    def power(self, exponent: float, annotated: bool = False):
        from .p import PhaseGate

        return PhaseGate(0.5 * numpy.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, SGate)


@with_gate_array(_SDG_ARRAY)
class SdgGate(SingletonGate):
    r"""Single qubit S-adjoint gate (~Z**0.5).

    It induces a :math:`-\pi/2` phase.

    This is a Clifford gate and a square-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.sdg` method.

    **Matrix Representation:**

    .. math::

        Sdg = \begin{pmatrix}
                1 & 0 \\
                0 & -i
            \end{pmatrix}

    **Circuit symbol:**

    .. code-block:: text

             ┌─────┐
        q_0: ┤ Sdg ├
             └─────┘

    Equivalent to a :math:`-\pi/2` radian rotation about the Z axis.
    """

    _standard_gate = StandardGate.SdgGate

    def __init__(self, label: Optional[str] = None, *, duration=None, unit="dt"):
        """Create new Sdg gate."""
        super().__init__("sdg", 1, [], label=label, duration=duration, unit=unit)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate sdg a { u1(-pi/2) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        from .u1 import U1Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(-pi / 2), [q[0]], [])]
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
        """Return a (multi-)controlled-Sdg gate.

        One control qubit returns a :class:`.CSdgGate`.

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
            gate = CSdgGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        """Return inverse of Sdg (SGate).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.SGate`.

        Returns:
            SGate: inverse of :class:`.SdgGate`
        """
        return SGate()

    def power(self, exponent: float, annotated: bool = False):
        from .p import PhaseGate

        return PhaseGate(-0.5 * numpy.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, SdgGate)


@with_controlled_gate_array(_S_ARRAY, num_ctrl_qubits=1)
class CSGate(SingletonControlledGate):
    r"""Controlled-S gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cs` method.

    **Circuit symbol:**

    .. code-block:: text

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ S ├
             └───┘

    **Matrix representation:**

    .. math::

        CS \ q_0, q_1 =
        I \otimes |0 \rangle\langle 0| + S \otimes |1 \rangle\langle 1|  =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & i
            \end{pmatrix}
    """

    _standard_gate = StandardGate.CSGate

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create new CS gate."""
        super().__init__(
            "cs",
            2,
            [],
            label=label,
            num_ctrl_qubits=1,
            ctrl_state=ctrl_state,
            base_gate=SGate(label=_base_label),
            duration=duration,
            _base_label=_base_label,
            unit=unit,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """
        gate cs a,b { h b; cp(pi/2) a,b; h b; }
        """
        from .p import CPhaseGate

        self.definition = CPhaseGate(theta=pi / 2).definition

    def inverse(self, annotated: bool = False):
        """Return inverse of CSGate (CSdgGate).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CSdgGate`.

        Returns:
            CSdgGate: inverse of :class:`.CSGate`
        """
        return CSdgGate(ctrl_state=self.ctrl_state)

    def power(self, exponent: float, annotated: bool = False):
        from .p import CPhaseGate

        return CPhaseGate(0.5 * numpy.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, CSGate) and self.ctrl_state == other.ctrl_state


@with_controlled_gate_array(_SDG_ARRAY, num_ctrl_qubits=1)
class CSdgGate(SingletonControlledGate):
    r"""Controlled-S^\dagger gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.csdg` method.

    **Circuit symbol:**

    .. code-block:: text

        q_0: ───■───
             ┌──┴──┐
        q_1: ┤ Sdg ├
             └─────┘

    **Matrix representation:**

    .. math::

        CS^\dagger \ q_0, q_1 =
        I \otimes |0 \rangle\langle 0| + S^\dagger \otimes |1 \rangle\langle 1|  =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & -i
            \end{pmatrix}
    """

    _standard_gate = StandardGate.CSdgGate

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create new CSdg gate."""
        super().__init__(
            "csdg",
            2,
            [],
            label=label,
            num_ctrl_qubits=1,
            ctrl_state=ctrl_state,
            base_gate=SdgGate(label=_base_label),
            duration=duration,
            unit=unit,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """
        gate csdg a,b { h b; cp(-pi/2) a,b; h b; }
        """
        from .p import CPhaseGate

        self.definition = CPhaseGate(theta=-pi / 2).definition

    def inverse(self, annotated: bool = False):
        """Return inverse of CSdgGate (CSGate).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CSGate`.

        Returns:
            CSGate: inverse of :class:`.CSdgGate`
        """
        return CSGate(ctrl_state=self.ctrl_state)

    def power(self, exponent: float, annotated: bool = False):
        from .p import CPhaseGate

        return CPhaseGate(-0.5 * numpy.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, CSdgGate) and self.ctrl_state == other.ctrl_state

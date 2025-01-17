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

"""Two-pulse single-qubit gate."""

from __future__ import annotations

import cmath
import copy as _copy
import math
from cmath import exp
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType, ParameterExpression
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit._accelerate.circuit import StandardGate


class UGate(Gate):
    r"""Generic single-qubit rotation gate with 3 Euler angles.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.u` method.

    **Circuit symbol:**

    .. code-block:: text

             ┌──────────┐
        q_0: ┤ U(ϴ,φ,λ) ├
             └──────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        U(\theta, \phi, \lambda) =
        \begin{pmatrix}
            \cos\left(\rotationangle\right) & -e^{i\lambda}\sin\left(\rotationangle\right) \\
            e^{i\phi}\sin\left(\rotationangle\right) & e^{i(\phi+\lambda)}\cos\left(\rotationangle\right)
        \end{pmatrix}

    .. note::

        The matrix representation shown here is the same as in the `OpenQASM 3.0 specification
        <https://openqasm.com/language/gates.html#built-in-gates>`_,
        which differs from the `OpenQASM 2.0 specification
        <https://doi.org/10.48550/arXiv.1707.03429>`_ by a global phase of
        :math:`e^{i(\phi+\lambda)/2}`.

    **Examples:**

    .. math::

        U\left(\theta, -\frac{\pi}{2}, \frac{\pi}{2}\right) = RX(\theta)

    .. math::

        U(\theta, 0, 0) = RY(\theta)
    """

    _standard_gate = StandardGate.UGate

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        label: Optional[str] = None,
        *,
        duration=None,
        unit="dt",
    ):
        """Create new U gate."""
        super().__init__("u", 1, [theta, phi, lam], label=label, duration=duration, unit=unit)

    def inverse(self, annotated: bool = False):
        r"""Return inverted U gate.

        :math:`U(\theta,\phi,\lambda)^{\dagger} =U(-\theta,-\lambda,-\phi))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the
                inverse of this gate is always a :class:`.UGate` with inverse parameter values.

        Returns:
            UGate: inverse gate.
        """
        return UGate(-self.params[0], -self.params[2], -self.params[1])

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a (multi-)controlled-U gate.

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
        if not annotated and num_ctrl_qubits == 1:
            gate = CUGate(
                self.params[0],
                self.params[1],
                self.params[2],
                0,
                label=label,
                ctrl_state=ctrl_state,
            )
            gate.base_gate.label = self.label
        else:
            # If the gate parameters contain free parameters, we cannot eagerly synthesize
            # the controlled gate decomposition. In this case, we annotate the gate per default.
            if annotated is None:
                annotated = any(isinstance(p, ParameterExpression) for p in self.params)

            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the U gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta, phi, lam = (float(param) for param in self.params)
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        return numpy.array(
            [
                [cos, -exp(1j * lam) * sin],
                [exp(1j * phi) * sin, exp(1j * (phi + lam)) * cos],
            ],
            dtype=dtype or complex,
        )

    def __eq__(self, other):
        if isinstance(other, UGate):
            return self._compare_parameters(other)
        return False


class _CUGateParams(list):
    # This awful class is to let `CUGate.params` have its keys settable (as
    # `QuantumCircuit.assign_parameters` requires), while accounting for the problem that `CUGate`
    # was defined to have a different number of parameters to its `base_gate`, which breaks
    # `ControlledGate`'s assumptions, and would make most parametric `CUGate`s invalid.
    #
    # It's constructed only as part of the `CUGate.params` getter, and given that the general
    # circuit model assumes that that's a directly mutable list that _must_ be kept in sync with the
    # gate's requirements, we don't need this to support arbitrary mutation, just enough for
    # `QuantumCircuit.assign_parameters` to work.

    __slots__ = ("_gate",)

    def __init__(self, gate):
        super().__init__(gate._params)
        self._gate = gate

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._gate._params[key] = value
        # Magic numbers: CUGate has 4 parameters, UGate has 3, with the last of CUGate's missing.
        if isinstance(key, slice):
            # We don't need to worry about the case of the slice being used to insert extra / remove
            # elements because that would be "undefined behavior" in a gate already, so we're
            # within our rights to do anything at all.
            for i, base_key in enumerate(range(*key.indices(4))):
                if base_key < 0:
                    base_key = 4 + base_key
                if base_key < 3:
                    self._gate.base_gate.params[base_key] = value[i]
        else:
            if key < 0:
                key = 4 + key
            if key < 3:
                self._gate.base_gate.params[key] = value


class CUGate(ControlledGate):
    r"""Controlled-U gate (4-parameter two-qubit gate).

    This is a controlled version of the U gate (generic single qubit rotation),
    including a possible global phase :math:`e^{i\gamma}` of the U gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cu` method.

    **Circuit symbol:**

    .. code-block:: text

        q_0: ──────■──────
             ┌─────┴──────┐
        q_1: ┤ U(ϴ,φ,λ,γ) ├
             └────────────┘

    **Matrix representation:**

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        CU(\theta, \phi, \lambda, \gamma)\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| +
            e^{i\gamma} U(\theta,\phi,\lambda) \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & e^{i\gamma}\cos(\rotationangle) &
                0 & -e^{i(\gamma + \lambda)}\sin(\rotationangle) \\
                0 & 0 & 1 & 0 \\
                0 & e^{i(\gamma+\phi)}\sin(\rotationangle) &
                0 & e^{i(\gamma+\phi+\lambda)}\cos(\rotationangle)
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌────────────┐
            q_0: ┤ U(ϴ,φ,λ,γ) ├
                 └─────┬──────┘
            q_1: ──────■───────

        .. math::

            \newcommand{\rotationangle}{\frac{\theta}{2}}
            CU(\theta, \phi, \lambda, \gamma)\ q_1, q_0 =
            |0\rangle\langle 0| \otimes I +
            e^{i\gamma}|1\rangle\langle 1| \otimes U(\theta,\phi,\lambda) =
            \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & e^{i\gamma} \cos(\rotationangle) & -e^{i(\gamma + \lambda)}\sin(\rotationangle) \\
            0 & 0 &
            e^{i(\gamma + \phi)}\sin(\rotationangle) & e^{i(\gamma + \phi+\lambda)}\cos(\rotationangle)
            \end{pmatrix}
    """

    _standard_gate = StandardGate.CUGate

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        gamma: ParameterValueType,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create new CU gate."""
        super().__init__(
            "cu",
            2,
            [theta, phi, lam, gamma],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=UGate(theta, phi, lam, label=_base_label),
            duration=duration,
            unit=unit,
        )

    def _define(self):
        """
        gate cu(theta,phi,lambda,gamma) c, t
        { phase(gamma) c;
          phase((lambda+phi)/2) c;
          phase((lambda-phi)/2) t;
          cx c,t;
          u(-theta/2,0,-(phi+lambda)/2) t;
          cx c,t;
          u(theta/2,phi,0) t;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        #          ┌──────┐    ┌──────────────┐
        # q_0: ────┤ P(γ) ├────┤ P(λ/2 + φ/2) ├──■────────────────────────────■────────────────
        #      ┌───┴──────┴───┐└──────────────┘┌─┴─┐┌──────────────────────┐┌─┴─┐┌────────────┐
        # q_1: ┤ P(λ/2 - φ/2) ├────────────────┤ X ├┤ U(-0/2,0,-λ/2 - φ/2) ├┤ X ├┤ U(0/2,φ,0) ├
        #      └──────────────┘                └───┘└──────────────────────┘└───┘└────────────┘
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        qc.p(self.params[3], 0)
        qc.p((self.params[2] + self.params[1]) / 2, 0)
        qc.p((self.params[2] - self.params[1]) / 2, 1)
        qc.cx(0, 1)
        qc.u(-self.params[0] / 2, 0, -(self.params[1] + self.params[2]) / 2, 1)
        qc.cx(0, 1)
        qc.u(self.params[0] / 2, self.params[1], 0, 1)
        self.definition = qc

    def inverse(self, annotated: bool = False):
        r"""Return inverted CU gate.

        :math:`CU(\theta,\phi,\lambda,\gamma)^{\dagger} = CU(-\theta,-\phi,-\lambda,-\gamma))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CUGate` with inverse parameter
                values.

        Returns:
            CUGate: inverse gate.
        """
        return CUGate(
            -self.params[0],
            -self.params[2],
            -self.params[1],
            -self.params[3],
            ctrl_state=self.ctrl_state,
        )

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the CU gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta, phi, lam, gamma = (float(param) for param in self.params)
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        a = cmath.exp(1j * gamma) * cos
        b = -cmath.exp(1j * (gamma + lam)) * sin
        c = cmath.exp(1j * (gamma + phi)) * sin
        d = cmath.exp(1j * (gamma + phi + lam)) * cos
        if self.ctrl_state:
            return numpy.array(
                [[1, 0, 0, 0], [0, a, 0, b], [0, 0, 1, 0], [0, c, 0, d]], dtype=dtype
            )
        else:
            return numpy.array(
                [[a, 0, b, 0], [0, 1, 0, 0], [c, 0, d, 0], [0, 0, 0, 1]], dtype=dtype
            )

    @property
    def params(self):
        return _CUGateParams(self)

    @params.setter
    def params(self, parameters):
        # We need to skip `ControlledGate` in the inheritance tree, since it defines
        # that all controlled gates are `(1-|c><c|).1 + |c><c|.base` for control-state `c`, which
        # this class does _not_ satisfy (so it shouldn't really be a `ControlledGate`).
        super(ControlledGate, type(self)).params.fset(self, parameters)
        self.base_gate.params = parameters[:-1]

    def __deepcopy__(self, memo=None):
        # We have to override this because `ControlledGate` doesn't copy the `_params` list,
        # assuming that `params` will be a view onto the base gate's `_params`.
        memo = memo if memo is not None else {}
        out = super().__deepcopy__(memo)
        out._params = _copy.deepcopy(out._params, memo)
        return out

    def __eq__(self, other):
        return (
            isinstance(other, CUGate)
            and self.ctrl_state == other.ctrl_state
            and self._compare_parameters(other)
        )

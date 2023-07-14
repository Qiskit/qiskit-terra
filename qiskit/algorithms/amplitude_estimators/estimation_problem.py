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

"""The Estimation problem class."""

from __future__ import annotations
import warnings
from collections.abc import Callable

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import GroverOperator


class EstimationProblem:
    """The estimation problem is the input to amplitude estimation algorithm.

    This class contains all problem-specific information required to run an amplitude estimation
    algorithm. That means, it minimally contains the state preparation and the specification
    of the good state. It can further hold some post processing on the estimation of the amplitude
    or a custom Grover operator.
    """

    def __init__(
        self,
        state_preparation: QuantumCircuit,
        objective_qubits: int | list[int],
        grover_operator: QuantumCircuit | None = None,
        post_processing: Callable[[float], float] | None = None,
        is_good_state: Callable[[str], bool] | None = None,
    ) -> None:
        r"""
        Args:
            state_preparation: A circuit preparing the input state, referred to as
                :math:`\mathcal{A}`.
            objective_qubits: A single qubit index or a list of qubit indices to specify which
                qubits to measure. The ``is_good_state`` function is applied on the bitstring of
                these objective qubits.
            grover_operator: The Grover operator :math:`\mathcal{Q}` used as unitary in the
                phase estimation circuit.
            post_processing: A mapping applied to the result of the algorithm
                :math:`0 \leq a \leq 1`, usually used to map the estimate to a target interval.
                Defaults to the identity.
            is_good_state: A function to check whether a string represents a good state. Defaults
                to all objective qubits being in state :math:`|1\rangle`.
        """
        self._state_preparation = state_preparation
        self._objective_qubits = objective_qubits
        self._grover_operator = grover_operator
        self._post_processing = post_processing
        self._is_good_state = is_good_state

    @property
    def state_preparation(self) -> QuantumCircuit | None:
        r"""Get the :math:`\mathcal{A}` operator encoding the amplitude :math:`a`.

        Returns:
            The :math:`\mathcal{A}` operator as `QuantumCircuit`.
        """
        return self._state_preparation

    @state_preparation.setter
    def state_preparation(self, state_preparation: QuantumCircuit) -> None:
        r"""Set the :math:`\mathcal{A}` operator, that encodes the amplitude to be estimated.

        Args:
            state_preparation: The new :math:`\mathcal{A}` operator.
        """
        self._state_preparation = state_preparation

    @property
    def objective_qubits(self) -> list[int]:
        """Get the criterion for a measurement outcome to be in a 'good' state.

        Returns:
            The criterion as list of qubit indices.
        """
        if isinstance(self._objective_qubits, int):
            return [self._objective_qubits]

        return self._objective_qubits

    @objective_qubits.setter
    def objective_qubits(self, objective_qubits: int | list[int]) -> None:
        """Set the criterion for a measurement outcome to be in a 'good' state.

        Args:
            objective_qubits: The criterion as callable of list of qubit indices.
        """
        self._objective_qubits = objective_qubits

    @property
    def post_processing(self) -> Callable[[float], float]:
        """Apply post processing to the input value.

        Returns:
            A handle to the post processing function. Acts as identity by default.
        """
        if self._post_processing is None:
            return lambda x: x

        return self._post_processing

    @post_processing.setter
    def post_processing(self, post_processing: Callable[[float], float] | None) -> None:
        """Set the post processing function.

        Args:
            post_processing: A handle to the post processing function. If set to ``None``, the
                identity will be used as post processing.
        """
        self._post_processing = post_processing

    @property
    def has_good_state(self) -> bool:
        """Check whether an :attr:`is_good_state` function is set.

        Some amplitude estimators, such as :class:`.AmplitudeEstimation` do not support
        a custom implementation of the :attr:`is_good_state` function, and can only handle
        the default.

        Returns:
            ``True``, if a custom :attr:`is_good_state` is set, otherwise returns ``False``.
        """
        return self._is_good_state is not None

    @property
    def is_good_state(self) -> Callable[[str], bool]:
        """Checks whether a bitstring represents a good state.

        Returns:
            Handle to the ``is_good_state`` callable.
        """
        if self._is_good_state is None:
            return lambda x: all(bit == "1" for bit in x)

        return self._is_good_state

    @is_good_state.setter
    def is_good_state(self, is_good_state: Callable[[str], bool] | None) -> None:
        """Set the ``is_good_state`` function.

        Args:
            is_good_state: A function to determine whether a bitstring represents a good state.
                If set to ``None``, the good state will be defined as all bits being one.
        """
        self._is_good_state = is_good_state

    @property
    def grover_operator(self) -> QuantumCircuit | None:
        r"""Get the :math:`\mathcal{Q}` operator, or Grover operator.

        If the Grover operator is not set, we try to build it from the :math:`\mathcal{A}` operator
        and `objective_qubits`. This only works if `objective_qubits` is a list of integers.

        Returns:
            The Grover operator, or None if neither the Grover operator nor the
            :math:`\mathcal{A}` operator is  set.
        """
        if self._grover_operator is not None:
            return self._grover_operator

        # build the reflection about the bad state: a MCZ with open controls (thus X gates
        # around the controls) and X gates around the target to change from a phaseflip on
        # |1> to a phaseflip on |0>
        num_state_qubits = self.state_preparation.num_qubits - self.state_preparation.num_ancillas

        oracle = QuantumCircuit(num_state_qubits)
        oracle.h(self.objective_qubits[-1])
        if len(self.objective_qubits) == 1:
            oracle.x(self.objective_qubits[0])
        else:
            oracle.mcx(self.objective_qubits[:-1], self.objective_qubits[-1])
        oracle.h(self.objective_qubits[-1])

        # construct the grover operator
        return GroverOperator(oracle, self.state_preparation)

    @grover_operator.setter
    def grover_operator(self, grover_operator: QuantumCircuit | None) -> None:
        r"""Set the :math:`\mathcal{Q}` operator.

        Args:
            grover_operator: The new :math:`\mathcal{Q}` operator. If set to ``None``,
                the default construction via ``qiskit.circuit.library.GroverOperator`` is used.
        """
        self._grover_operator = grover_operator

    def rescale(self, scaling_factor: float) -> "EstimationProblem":
        """Rescale the good state amplitude in the estimation problem.

        Args:
            scaling_factor: The scaling factor in [0, 1].

        Returns:
            A rescaled estimation problem.
        """
        if self._grover_operator is not None:
            warnings.warn("Rescaling discards the Grover operator.")

        # rescale the amplitude by a factor of 1/4 by adding an auxiliary qubit
        rescaled_stateprep = _rescale_amplitudes(self.state_preparation, scaling_factor)
        num_qubits = self.state_preparation.num_qubits
        objective_qubits = self.objective_qubits + [num_qubits]

        # add the scaling qubit to the good state qualifier
        def is_good_state(bitstr):
            return self.is_good_state(bitstr[1:]) and bitstr[0] == "1"

        # rescaled estimation problem
        problem = EstimationProblem(
            rescaled_stateprep,
            objective_qubits=objective_qubits,
            post_processing=self.post_processing,
            is_good_state=is_good_state,
        )

        return problem


def _rescale_amplitudes(circuit: QuantumCircuit, scaling_factor: float) -> QuantumCircuit:
    r"""Uses an auxiliary qubit to scale the amplitude of :math:`|1\rangle` by ``scaling_factor``.

    Explained in Section 2.1. of [1].

    For example, for a scaling factor of 0.25 this turns this circuit

    .. parsed-literal::

                      ┌───┐
        state_0: ─────┤ H ├─────────■────
                  ┌───┴───┴───┐ ┌───┴───┐
          obj_0: ─┤ RY(0.125) ├─┤ RY(1) ├
                  └───────────┘ └───────┘

    into

    .. parsed-literal::

                      ┌───┐
        state_0: ─────┤ H ├─────────■────
                  ┌───┴───┴───┐ ┌───┴───┐
          obj_0: ─┤ RY(0.125) ├─┤ RY(1) ├
                 ┌┴───────────┴┐└───────┘
      scaling_0: ┤ RY(0.50536) ├─────────
                 └─────────────┘

    References:

        [1]: K. Nakaji. Faster Amplitude Estimation, 2020;
            `arXiv:2002.02417 <https://arxiv.org/pdf/2003.02417.pdf>`_

    Args:
        circuit: The circuit whose amplitudes to rescale.
        scaling_factor: The rescaling factor.

    Returns:
        A copy of the circuit with an additional qubit and RY gate for the rescaling.
    """
    qr = QuantumRegister(1, "scaling")
    rescaled = QuantumCircuit(*circuit.qregs, qr)
    rescaled.compose(circuit, circuit.qubits, inplace=True)
    rescaled.ry(2 * np.arcsin(scaling_factor), qr)
    return rescaled

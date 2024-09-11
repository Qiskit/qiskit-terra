# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Multiple-Controlled U3 gate. Not using ancillary qubits.
"""

from math import pi
import math
from typing import Optional, Union, Tuple, List
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit, ParameterExpression
from qiskit.circuit.library.standard_gates.x import MCXGate
from qiskit.circuit.library.standard_gates.u3 import _generate_gray_code
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError


def _apply_cu(circuit, theta, phi, lam, control, target, use_basis_gates=True):
    if use_basis_gates:
        # pylint: disable=cyclic-import
        #          ┌──────────────┐
        # control: ┤ P(λ/2 + φ/2) ├──■──────────────────────────────────■────────────────
        #          ├──────────────┤┌─┴─┐┌────────────────────────────┐┌─┴─┐┌────────────┐
        #  target: ┤ P(λ/2 - φ/2) ├┤ X ├┤ U(-0.5*0,0,-0.5*λ - 0.5*φ) ├┤ X ├┤ U(0/2,φ,0) ├
        #          └──────────────┘└───┘└────────────────────────────┘└───┘└────────────┘
        circuit.p((lam + phi) / 2, [control])
        circuit.p((lam - phi) / 2, [target])
        circuit.cx(control, target)
        circuit.u(-theta / 2, 0, -(phi + lam) / 2, [target])
        circuit.cx(control, target)
        circuit.u(theta / 2, phi, 0, [target])
    else:
        circuit.cu(theta, phi, lam, 0, control, target)


def _apply_mcu_graycode(circuit, theta, phi, lam, ctls, tgt, use_basis_gates):
    """Apply multi-controlled u gate from ctls to tgt using graycode
    pattern with single-step angles theta, phi, lam."""

    n = len(ctls)

    gray_code = _generate_gray_code(n)
    last_pattern = None

    for pattern in gray_code:
        if "1" not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        # find left most set bit
        lm_pos = list(pattern).index("1")

        # find changed bit
        comp = [i != j for i, j in zip(pattern, last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                circuit.cx(ctls[pos], ctls[lm_pos])
            else:
                indices = [i for i, x in enumerate(pattern) if x == "1"]
                for idx in indices[1:]:
                    circuit.cx(ctls[idx], ctls[lm_pos])
        # check parity and undo rotation
        if pattern.count("1") % 2 == 0:
            # inverse CU: u(theta, phi, lamb)^dagger = u(-theta, -lam, -phi)
            _apply_cu(
                circuit, -theta, -lam, -phi, ctls[lm_pos], tgt, use_basis_gates=use_basis_gates
            )
        else:
            _apply_cu(circuit, theta, phi, lam, ctls[lm_pos], tgt, use_basis_gates=use_basis_gates)
        last_pattern = pattern


def _mcsu2_real_diagonal(
    unitary: np.ndarray,
    num_controls: int,
    ctrl_state: Optional[str] = None,
    use_basis_gates: bool = False,
) -> QuantumCircuit:
    """
    Return a multi-controlled SU(2) gate [1]_ with a real main diagonal or secondary diagonal.

    Args:
        unitary: SU(2) unitary matrix with one real diagonal.
        num_controls: The number of control qubits.
        ctrl_state: The state on which the SU(2) operation is controlled. Defaults to all
            control qubits being in state 1.
        use_basis_gates: If ``True``, use ``[p, u, cx]`` gates to implement the decomposition.

    Returns:
        A :class:`.QuantumCircuit` implementing the multi-controlled SU(2) gate.

    Raises:
        QiskitError: If the input matrix is invalid.

    References:

        .. [1]: R. Vale et al. Decomposition of Multi-controlled Special Unitary Single-Qubit Gates
            `arXiv:2302.06377 (2023) <https://arxiv.org/abs/2302.06377>`__

    """
    # pylint: disable=cyclic-import
    from .x import MCXVChain
    from qiskit.circuit.library.generalized_gates import UnitaryGate
    from qiskit.quantum_info.operators.predicates import is_unitary_matrix
    from qiskit.compiler import transpile

    if unitary.shape != (2, 2):
        raise QiskitError(f"The unitary must be a 2x2 matrix, but has shape {unitary.shape}.")

    if not is_unitary_matrix(unitary):
        raise QiskitError(f"The unitary in must be an unitary matrix, but is {unitary}.")

    if not np.isclose(1.0, np.linalg.det(unitary)):
        raise QiskitError("Invalid Value _mcsu2_real_diagonal requires det(unitary) equal to one.")

    is_main_diag_real = np.isclose(unitary[0, 0].imag, 0.0) and np.isclose(unitary[1, 1].imag, 0.0)
    is_secondary_diag_real = np.isclose(unitary[0, 1].imag, 0.0) and np.isclose(
        unitary[1, 0].imag, 0.0
    )

    if not is_main_diag_real and not is_secondary_diag_real:
        raise QiskitError("The unitary must have one real diagonal.")

    if is_secondary_diag_real:
        x = unitary[0, 1]
        z = unitary[1, 1]
    else:
        x = -unitary[0, 1].real
        z = unitary[1, 1] - unitary[0, 1].imag * 1.0j

    if np.isclose(z, -1):
        s_op = [[1.0, 0.0], [0.0, 1.0j]]
    else:
        alpha_r = math.sqrt((math.sqrt((z.real + 1.0) / 2.0) + 1.0) / 2.0)
        alpha_i = z.imag / (
            2.0 * math.sqrt((z.real + 1.0) * (math.sqrt((z.real + 1.0) / 2.0) + 1.0))
        )
        alpha = alpha_r + 1.0j * alpha_i
        beta = x / (2.0 * math.sqrt((z.real + 1.0) * (math.sqrt((z.real + 1.0) / 2.0) + 1.0)))

        # S gate definition
        s_op = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]])

    s_gate = UnitaryGate(s_op)

    k_1 = math.ceil(num_controls / 2.0)
    k_2 = math.floor(num_controls / 2.0)

    ctrl_state_k_1 = None
    ctrl_state_k_2 = None

    if ctrl_state is not None:
        str_ctrl_state = f"{ctrl_state:0{num_controls}b}"
        ctrl_state_k_1 = str_ctrl_state[::-1][:k_1][::-1]
        ctrl_state_k_2 = str_ctrl_state[::-1][k_1:][::-1]

    circuit = QuantumCircuit(num_controls + 1, name="MCSU2")
    controls = list(range(num_controls))  # control indices, defined for code legibility
    target = num_controls  # target index, defined for code legibility

    if not is_secondary_diag_real:
        circuit.h(target)

    mcx_1 = MCXVChain(num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1)
    circuit.append(mcx_1, controls[:k_1] + [target] + controls[k_1 : 2 * k_1 - 2])
    circuit.append(s_gate, [target])

    mcx_2 = MCXVChain(
        num_ctrl_qubits=k_2,
        dirty_ancillas=True,
        ctrl_state=ctrl_state_k_2,
        # action_only=general_su2_optimization # Requires PR #9687
    )
    circuit.append(mcx_2.inverse(), controls[k_1:] + [target] + controls[k_1 - k_2 + 2 : k_1])
    circuit.append(s_gate.inverse(), [target])

    mcx_3 = MCXVChain(num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1)
    circuit.append(mcx_3, controls[:k_1] + [target] + controls[k_1 : 2 * k_1 - 2])
    circuit.append(s_gate, [target])

    mcx_4 = MCXVChain(num_ctrl_qubits=k_2, dirty_ancillas=True, ctrl_state=ctrl_state_k_2)
    circuit.append(mcx_4, controls[k_1:] + [target] + controls[k_1 - k_2 + 2 : k_1])
    circuit.append(s_gate.inverse(), [target])

    if not is_secondary_diag_real:
        circuit.h(target)

    if use_basis_gates:
        circuit = transpile(circuit, basis_gates=["p", "u", "cx"], qubits_initially_zero=False)

    return circuit


def mcrx(
    self,
    theta: ParameterValueType,
    q_controls: Union[QuantumRegister, List[Qubit]],
    q_target: Qubit,
    use_basis_gates: bool = False,
):
    """
    Apply Multiple-Controlled X rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcrx gate on.
        theta (float): angle theta
        q_controls (QuantumRegister or list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """
    from .rx import RXGate

    control_qubits = self._qbit_argument_conversion(q_controls)
    target_qubit = self._qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError("The mcrz gate needs a single qubit as target.")
    all_qubits = control_qubits + target_qubit
    target_qubit = target_qubit[0]
    self._check_dups(all_qubits)

    n_c = len(control_qubits)
    if n_c == 1:  # cu
        _apply_cu(
            self,
            theta,
            -pi / 2,
            pi / 2,
            control_qubits[0],
            target_qubit,
            use_basis_gates=use_basis_gates,
        )
    elif n_c < 4:
        theta_step = theta * (1 / (2 ** (n_c - 1)))
        _apply_mcu_graycode(
            self,
            theta_step,
            -pi / 2,
            pi / 2,
            control_qubits,
            target_qubit,
            use_basis_gates=use_basis_gates,
        )
    else:
        if isinstance(theta, ParameterExpression):
            raise QiskitError(f"Cannot synthesize MCRX with unbound parameter: {theta}.")

        cgate = _mcsu2_real_diagonal(
            RXGate(theta).to_matrix(),
            num_controls=len(control_qubits),
            use_basis_gates=use_basis_gates,
        )
        self.compose(cgate, control_qubits + [target_qubit], inplace=True)


def mcry(
    self,
    theta: ParameterValueType,
    q_controls: Union[QuantumRegister, List[Qubit]],
    q_target: Qubit,
    q_ancillae: Optional[Union[QuantumRegister, Tuple[QuantumRegister, int]]] = None,
    mode: Optional[str] = None,
    use_basis_gates: bool = False,
):
    """
    Apply Multiple-Controlled Y rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcry gate on.
        theta (float): angle theta
        q_controls (list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        q_ancillae (QuantumRegister or tuple(QuantumRegister, int)): The list of ancillary qubits.
        mode (string): The implementation mode to use
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """
    from .ry import RYGate

    control_qubits = self._qbit_argument_conversion(q_controls)
    target_qubit = self._qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError("The mcrz gate needs a single qubit as target.")
    ancillary_qubits = [] if q_ancillae is None else self._qbit_argument_conversion(q_ancillae)
    all_qubits = control_qubits + target_qubit + ancillary_qubits
    target_qubit = target_qubit[0]
    self._check_dups(all_qubits)

    # auto-select the best mode
    if mode is None:
        # if enough ancillary qubits are provided, use the 'v-chain' method
        additional_vchain = MCXGate.get_num_ancilla_qubits(len(control_qubits), "v-chain")
        if len(ancillary_qubits) >= additional_vchain:
            mode = "basic"
        else:
            mode = "noancilla"

    if mode == "basic":
        self.ry(theta / 2, q_target)
        self.mcx(q_controls, q_target, q_ancillae, mode="v-chain")
        self.ry(-theta / 2, q_target)
        self.mcx(q_controls, q_target, q_ancillae, mode="v-chain")
    elif mode == "noancilla":
        n_c = len(control_qubits)
        if n_c == 1:  # cu
            _apply_cu(
                self, theta, 0, 0, control_qubits[0], target_qubit, use_basis_gates=use_basis_gates
            )
        elif n_c < 4:
            theta_step = theta * (1 / (2 ** (n_c - 1)))
            _apply_mcu_graycode(
                self,
                theta_step,
                0,
                0,
                control_qubits,
                target_qubit,
                use_basis_gates=use_basis_gates,
            )
        else:
            if isinstance(theta, ParameterExpression):
                raise QiskitError(f"Cannot synthesize MCRY with unbound parameter: {theta}.")

            cgate = _mcsu2_real_diagonal(
                RYGate(theta).to_matrix(),
                num_controls=len(control_qubits),
                use_basis_gates=use_basis_gates,
            )
            self.compose(cgate, control_qubits + [target_qubit], inplace=True)
    else:
        raise QiskitError(f"Unrecognized mode for building MCRY circuit: {mode}.")


def mcrz(
    self,
    lam: ParameterValueType,
    q_controls: Union[QuantumRegister, List[Qubit]],
    q_target: Qubit,
    use_basis_gates: bool = False,
):
    """
    Apply Multiple-Controlled Z rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcrz gate on.
        lam (float): angle lambda
        q_controls (list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """
    from .rz import CRZGate, RZGate

    control_qubits = self._qbit_argument_conversion(q_controls)
    target_qubit = self._qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError("The mcrz gate needs a single qubit as target.")
    all_qubits = control_qubits + target_qubit
    target_qubit = target_qubit[0]
    self._check_dups(all_qubits)

    n_c = len(control_qubits)
    if n_c == 1:
        if use_basis_gates:
            self.u(0, 0, lam / 2, target_qubit)
            self.cx(control_qubits[0], target_qubit)
            self.u(0, 0, -lam / 2, target_qubit)
            self.cx(control_qubits[0], target_qubit)
        else:
            self.append(CRZGate(lam), control_qubits + [target_qubit])
    else:
        if isinstance(lam, ParameterExpression):
            raise QiskitError(f"Cannot synthesize MCRZ with unbound parameter: {lam}.")

        cgate = _mcsu2_real_diagonal(
            RZGate(lam).to_matrix(),
            num_controls=len(control_qubits),
            use_basis_gates=use_basis_gates,
        )
        self.compose(cgate, control_qubits + [target_qubit], inplace=True)


QuantumCircuit.mcrx = mcrx
QuantumCircuit.mcry = mcry
QuantumCircuit.mcrz = mcrz

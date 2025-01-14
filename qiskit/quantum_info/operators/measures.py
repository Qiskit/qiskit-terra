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

"""
A collection of useful quantum information functions for operators.
"""

from __future__ import annotations
import logging
import numpy as np

from qiskit.exceptions import QiskitError, MissingOptionalLibraryError
from qiskit.circuit.gate import Gate
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel import Choi, SuperOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.states.measures import state_fidelity
from qiskit.utils import optionals as _optionals

logger = logging.getLogger(__name__)


def process_fidelity(
    channel: Operator | QuantumChannel,
    target: Operator | QuantumChannel | None = None,
    require_cp: bool = True,
    require_tp: bool = True,
) -> float:
    r"""Return the process fidelity of a noisy quantum channel.


    The process fidelity :math:`F_{\text{pro}}(\mathcal{E}, \mathcal{F})`
    between two quantum channels :math:`\mathcal{E}, \mathcal{F}` is given by

    .. math::
        F_{\text{pro}}(\mathcal{E}, \mathcal{F})
            = F(\rho_{\mathcal{E}}, \rho_{\mathcal{F}})

    where :math:`F` is the :func:`~qiskit.quantum_info.state_fidelity`,
    :math:`\rho_{\mathcal{E}} = \Lambda_{\mathcal{E}} / d` is the
    normalized :class:`~qiskit.quantum_info.Choi` matrix for the channel
    :math:`\mathcal{E}`, and :math:`d` is the input dimension of
    :math:`\mathcal{E}`.

    When the target channel is unitary this is equivalent to

    .. math::
        F_{\text{pro}}(\mathcal{E}, U)
            = \frac{Tr[S_U^\dagger S_{\mathcal{E}}]}{d^2}

    where :math:`S_{\mathcal{E}}, S_{U}` are the
    :class:`~qiskit.quantum_info.SuperOp` matrices for the *input* quantum
    channel :math:`\mathcal{E}` and *target* unitary :math:`U` respectively,
    and :math:`d` is the input dimension of the channel.

    Args:
        channel (Operator or QuantumChannel): input quantum channel.
        target (Operator or QuantumChannel or None): target quantum channel.
            If `None` target is the identity operator [Default: None].
        require_cp (bool): check if input and target channels are
                           completely-positive and if non-CP log warning
                           containing negative eigenvalues of Choi-matrix
                           [Default: True].
        require_tp (bool): check if input and target channels are
                           trace-preserving and if non-TP log warning
                           containing negative eigenvalues of partial
                           Choi-matrix :math:`Tr_{\text{out}}[\mathcal{E}] - I`
                           [Default: True].

    Returns:
        float: The process fidelity :math:`F_{\text{pro}}`.

    Raises:
        QiskitError: if the channel and target do not have the same dimensions.
    """
    # Format inputs
    channel = _input_formatter(channel, SuperOp, "process_fidelity", "channel")
    target = _input_formatter(target, Operator, "process_fidelity", "target")

    if target:
        # Validate dimensions
        if channel.dim != target.dim:
            raise QiskitError(
                "Input quantum channel and target unitary must have the same "
                f"dimensions ({channel.dim} != {target.dim})."
            )

    # Validate complete-positivity and trace-preserving
    for label, chan in [("Input", channel), ("Target", target)]:
        if chan is not None and require_cp:
            cp_cond = _cp_condition(chan)
            neg = cp_cond < -1 * chan.atol
            if np.any(neg):
                logger.warning(
                    "%s channel is not CP. Choi-matrix has negative eigenvalues: %s",
                    label,
                    cp_cond[neg],
                )
        if chan is not None and require_tp:
            tp_cond = _tp_condition(chan)
            non_zero = np.logical_not(np.isclose(tp_cond, 0, atol=chan.atol, rtol=chan.rtol))
            if np.any(non_zero):
                logger.warning(
                    "%s channel is not TP. Tr_2[Choi] - I has non-zero eigenvalues: %s",
                    label,
                    tp_cond[non_zero],
                )

    if isinstance(target, Operator):
        # Compute fidelity with unitary target by applying the inverse
        # to channel and computing fidelity with the identity
        channel = channel.compose(target.adjoint())
        target = None

    input_dim, _ = channel.dim
    if target is None:
        # Compute process fidelity with identity channel
        if isinstance(channel, Operator):
            # |Tr[U]/dim| ** 2
            fid = np.abs(np.trace(channel.data) / input_dim) ** 2
        else:
            # Tr[S] / (dim ** 2)
            fid = np.trace(SuperOp(channel).data) / (input_dim**2)
        return float(np.real(fid))

    # For comparing two non-unitary channels we compute the state fidelity of
    # the normalized Choi-matrices. This is equivalent to the previous definition
    # when the target is a unitary channel.
    state1 = DensityMatrix(Choi(channel).data / input_dim)
    state2 = DensityMatrix(Choi(target).data / input_dim)
    return state_fidelity(state1, state2, validate=False)


def average_gate_fidelity(
    channel: QuantumChannel | Operator,
    target: Operator | None = None,
    require_cp: bool = True,
    require_tp: bool = False,
) -> float:
    r"""Return the average gate fidelity of a noisy quantum channel.

    The average gate fidelity :math:`F_{\text{ave}}` is given by

    .. math::
        \begin{aligned}
        F_{\text{ave}}(\mathcal{E}, U)
            &= \int d\psi \langle\psi|U^\dagger
                \mathcal{E}(|\psi\rangle\!\langle\psi|)U|\psi\rangle \\
            &= \frac{d F_{\text{pro}}(\mathcal{E}, U) + 1}{d + 1}
        \end{aligned}

    where :math:`F_{\text{pro}}(\mathcal{E}, U)` is the
    :meth:`~qiskit.quantum_info.process_fidelity` of the input quantum
    *channel* :math:`\mathcal{E}` with a *target* unitary :math:`U`, and
    :math:`d` is the dimension of the *channel*.

    Args:
        channel (QuantumChannel or Operator): noisy quantum channel.
        target (Operator or None): target unitary operator.
            If `None` target is the identity operator [Default: None].
        require_cp (bool): check if input and target channels are
                           completely-positive and if non-CP log warning
                           containing negative eigenvalues of Choi-matrix
                           [Default: True].
        require_tp (bool): check if input and target channels are
                           trace-preserving and if non-TP log warning
                           containing negative eigenvalues of partial
                           Choi-matrix :math:`Tr_{\text{out}}[\mathcal{E}] - I`
                           [Default: True].

    Returns:
        float: The average gate fidelity :math:`F_{\text{ave}}`.

    Raises:
        QiskitError: if the channel and target do not have the same dimensions,
                     or have different input and output dimensions.
    """
    # Format inputs
    channel = _input_formatter(channel, SuperOp, "average_gate_fidelity", "channel")
    target = _input_formatter(target, Operator, "average_gate_fidelity", "target")

    if target is not None:
        try:
            target = Operator(target)
        except QiskitError as ex:
            raise QiskitError(
                "Target channel is not a unitary channel. To compare "
                "two non-unitary channels use the "
                "`qiskit.quantum_info.process_fidelity` function instead."
            ) from ex
    dim, _ = channel.dim
    f_pro = process_fidelity(channel, target=target, require_cp=require_cp, require_tp=require_tp)
    return (dim * f_pro + 1) / (dim + 1)


def gate_error(
    channel: QuantumChannel,
    target: Operator | None = None,
    require_cp: bool = True,
    require_tp: bool = False,
) -> float:
    r"""Return the gate error of a noisy quantum channel.

    The gate error :math:`E` is given by the average gate infidelity

    .. math::
        E(\mathcal{E}, U) = 1 - F_{\text{ave}}(\mathcal{E}, U)

    where :math:`F_{\text{ave}}(\mathcal{E}, U)` is the
    :meth:`~qiskit.quantum_info.average_gate_fidelity` of the input
    quantum *channel* :math:`\mathcal{E}` with a *target* unitary
    :math:`U`.

    Args:
        channel (QuantumChannel): noisy quantum channel.
        target (Operator or None): target unitary operator.
            If `None` target is the identity operator [Default: None].
        require_cp (bool): check if input and target channels are
                           completely-positive and if non-CP log warning
                           containing negative eigenvalues of Choi-matrix
                           [Default: True].
        require_tp (bool): check if input and target channels are
                           trace-preserving and if non-TP log warning
                           containing negative eigenvalues of partial
                           Choi-matrix :math:`Tr_{\text{out}}[\mathcal{E}] - I`
                           [Default: True].

    Returns:
        float: The average gate error :math:`E`.

    Raises:
        QiskitError: if the channel and target do not have the same dimensions,
                     or have different input and output dimensions.
    """
    # Format inputs
    channel = _input_formatter(channel, SuperOp, "gate_error", "channel")
    target = _input_formatter(target, Operator, "gate_error", "target")
    return 1 - average_gate_fidelity(
        channel, target=target, require_cp=require_cp, require_tp=require_tp
    )


def diamond_norm(choi: Choi | QuantumChannel, solver: str = "SCS", **kwargs) -> float:
    r"""Return the diamond norm of the input quantum channel object.

    This function computes the completely-bounded trace-norm (often
    referred to as the diamond-norm) of the input quantum channel object
    using the semidefinite-program from reference [1].

    Args:
        choi(Choi or QuantumChannel): a quantum channel object or
                                      Choi-matrix array.
        solver (str): The solver to use.
        kwargs: optional arguments to pass to CVXPY solver.

    Returns:
        float: The completely-bounded trace norm :math:`\|\mathcal{E}\|_{\diamond}`.

    Raises:
        QiskitError: if CVXPY package cannot be found.

    Additional Information:
        The input to this function is typically *not* a CPTP quantum
        channel, but rather the *difference* between two quantum channels
        :math:`\|\Delta\mathcal{E}\|_\diamond` where
        :math:`\Delta\mathcal{E} = \mathcal{E}_1 - \mathcal{E}_2`.

    Reference:
        J. Watrous. "Simpler semidefinite programs for completely bounded
        norms", arXiv:1207.5726 [quant-ph] (2012).

    .. note::

        This function requires the optional CVXPY package to be installed.
        Any additional kwargs will be passed to the ``cvxpy.solve``
        function. See the CVXPY documentation for information on available
        SDP solvers.
    """
    from scipy import sparse

    cvxpy = _cvxpy_check("`diamond_norm`")  # Check CVXPY is installed

    choi = Choi(_input_formatter(choi, Choi, "diamond_norm", "choi"))

    def cvx_bmat(mat_r, mat_i):
        """Block matrix for embedding complex matrix in reals"""
        return cvxpy.bmat([[mat_r, -mat_i], [mat_i, mat_r]])

    # Dimension of input and output spaces
    dim_in = choi._input_dim
    dim_out = choi._output_dim
    size = dim_in * dim_out

    # SDP Variables to convert to real valued problem
    r0_r = cvxpy.Variable((dim_in, dim_in))
    r0_i = cvxpy.Variable((dim_in, dim_in))
    r0 = cvx_bmat(r0_r, r0_i)

    r1_r = cvxpy.Variable((dim_in, dim_in))
    r1_i = cvxpy.Variable((dim_in, dim_in))
    r1 = cvx_bmat(r1_r, r1_i)

    x_r = cvxpy.Variable((size, size))
    x_i = cvxpy.Variable((size, size))
    iden = sparse.eye(dim_out)

    # Watrous uses row-vec convention for his Choi matrix while we use
    # col-vec. It turns out row-vec convention is required for CVXPY too
    # since the cvxpy.kron function must have a constant as its first argument.
    c_r = cvxpy.bmat([[cvxpy.kron(iden, r0_r), x_r], [x_r.T, cvxpy.kron(iden, r1_r)]])
    c_i = cvxpy.bmat([[cvxpy.kron(iden, r0_i), x_i], [-x_i.T, cvxpy.kron(iden, r1_i)]])
    c = cvx_bmat(c_r, c_i)

    # Convert col-vec convention Choi-matrix to row-vec convention and
    # then take Transpose: Choi_C -> Choi_R.T
    choi_rt = np.transpose(
        np.reshape(choi.data, (dim_in, dim_out, dim_in, dim_out)), (3, 2, 1, 0)
    ).reshape(choi.data.shape)
    choi_rt_r = choi_rt.real
    choi_rt_i = choi_rt.imag

    # Constraints
    cons = [
        r0 >> 0,
        r0_r == r0_r.T,
        r0_i == -r0_i.T,
        cvxpy.trace(r0_r) == 1,
        r1 >> 0,
        r1_r == r1_r.T,
        r1_i == -r1_i.T,
        cvxpy.trace(r1_r) == 1,
        c >> 0,
    ]

    # Objective function
    obj = cvxpy.Maximize(cvxpy.trace(choi_rt_r @ x_r) + cvxpy.trace(choi_rt_i @ x_i))
    prob = cvxpy.Problem(obj, cons)
    sol = prob.solve(solver=solver, **kwargs)
    return sol


def diamond_distance(op1: BaseOperator, op2: BaseOperator) -> float:
    r"""Return the diamond distance between two unitary operators.

    This function computes the completely-bounded trace-norm (often
    referred to as the diamond norm) of a difference of two quantum maps.
    It is often used as a distance metric in quantum information where

    :math:`d(E, F) = ||E-F||_diamond`.


    Args:
        op1 (Operator): a unitary operator.
        op2 (Operator): a unitary operator.

    Returns:
        float: The completely-bounded trace norm of `op1 - op2`.

    Raises:
        ValueError: if the input operators do not have the same dimensions.

    Additional Information:
        If both operators are unitary, the implementation uses a result in Aharonov
        et al. (1998) resulting in a significant optimisation. Geometrically, we
        compute the distance :math:`d` between the origin and the convex hull of
        the eigenvalues of :math:`U^\dag V` which is plugged into :math:`\sqrt{1 - d^2}`.

        If the operators are `Pauli` objects, we use an optimisation to find eigenvalues.

    Reference:
        D. Aharonov, A. Kitaev, and N. Nisan. “Quantum circuits with
        mixed states” in Proceedings of the thirtieth annual ACM symposium
        on Theory of computing, pp. 20-30, 1998.

    .. note::

        This function requires the optional CVXPY package to be installed.
        Any additional kwargs will be passed to the ``cvxpy.solve``
        function. See the CVXPY documentation for information on available
        SDP solvers.
    """
    op1 = _input_formatter(op1, BaseOperator, "diamond_distance", "op1")
    op2 = _input_formatter(op2, BaseOperator, "diamond_distance", "op2")

    # Check base operators have same dimension
    if op1.dim != op2.dim:
        raise ValueError(
            "Input quantum channel and target unitary must have the same "
            f"dimensions ({op1.dim} != {op2.dim})."
        )

    # Attempt to run unitary optimisation
    if (
        isinstance(op1, Operator)
        and isinstance(op2, Operator)
        and op1.is_unitary()
        and op2.is_unitary()
    ):
        # Compute the diamond norm
        mat1 = op1.data
        mat2 = op2.data
        pre_diag = np.conj(mat1).T @ mat2
        eigenvals = np.linalg.eigvals(pre_diag)
        d = _find_poly_distance(eigenvals)
        return 2 * np.sqrt(1 - d**2)
    elif isinstance(op1, Pauli) and isinstance(op2, Pauli):
        # Check Pauli equality up to phase
        if (op1.z == op2.z).all() and (op1.x == op2.x).all():
            return 0.0
        else:
            return 2.0
    else:
        # TODO: Implement special case for pauli channels (Benenti and Strini 2010)
        # as well as a potential optimisation for clifford circuits

        # Compute the diamond norm
        return diamond_norm(Choi(op1) - Choi(op2))


def _find_poly_distance(eigenvals: np.ndarray) -> float:
    """Function to find the distance between origin and the convex hull of eigenvalues."""
    phases = np.angle(eigenvals)
    phase_max = phases.max()
    phase_min = phases.min()

    if phase_min > 0:  # all eigenvals have pos phase: hull is above x axis
        return np.cos((phase_max - phase_min) / 2)

    if phase_max <= 0:  # all eigenvals have neg phase: hull is below x axis
        return np.cos((np.abs(phase_min) - np.abs(phase_max)) / 2)

    pos_phase_min = np.where(phases > 0, phases, np.inf).min()
    neg_phase_max = np.where(phases <= 0, phases, -np.inf).max()

    big_angle = phase_max - phase_min
    small_angle = pos_phase_min - neg_phase_max
    if big_angle >= np.pi:
        if small_angle <= np.pi:  # hull contains the origin
            return 0
        else:  # hull is left of y axis
            return np.cos((2 * np.pi - small_angle) / 2)
    else:  # hull is right of y axis
        return np.cos(big_angle / 2)


def _cvxpy_check(name):
    """Check that a supported CVXPY version is installed"""
    # Check if CVXPY package is installed
    _optionals.HAS_CVXPY.require_now(name)
    import cvxpy

    # Check CVXPY version
    version = cvxpy.__version__
    if version[0] != "1":
        raise MissingOptionalLibraryError(
            "CVXPY >= 1.0",
            "diamond_norm",
            msg=f"Incompatible CVXPY version {version} found.",
        )
    return cvxpy


# pylint: disable=too-many-return-statements
def _input_formatter(obj, fallback_class, func_name, arg_name):
    """Formatting function for input conversion"""
    # Empty input
    if obj is None:
        return obj

    # Channel-like input
    if isinstance(obj, QuantumChannel):
        return obj
    if hasattr(obj, "to_quantumchannel"):
        return obj.to_quantumchannel()
    if hasattr(obj, "to_channel"):
        return obj.to_channel()

    # Pauli-like input
    if isinstance(obj, Pauli):
        return obj

    # Unitary-like input
    if isinstance(obj, (Gate, BaseOperator)):
        return Operator(obj)
    if hasattr(obj, "to_operator"):
        return obj.to_operator()
    raise TypeError(
        f"invalid type supplied to {arg_name} of {func_name}."
        f" A {fallback_class.__name__} is best."
    )


def _cp_condition(channel):
    """Return Choi-matrix eigenvalues for checking if channel is CP"""
    if isinstance(channel, QuantumChannel):
        if not isinstance(channel, Choi):
            channel = Choi(channel)
        return np.linalg.eigvalsh(channel.data)
    unitary = Operator(channel).data
    return np.tensordot(unitary, unitary.conj(), axes=([0, 1], [0, 1])).real


def _tp_condition(channel):
    """Return partial tr Choi-matrix eigenvalues for checking if channel is TP"""
    if isinstance(channel, QuantumChannel):
        if not isinstance(channel, Choi):
            channel = Choi(channel)
        choi = channel.data
        dims = tuple(np.sqrt(choi.shape).astype(int))
        shape = dims + dims
        tr_choi = np.trace(np.reshape(choi, shape), axis1=1, axis2=3)
    else:
        unitary = Operator(channel).data
        tr_choi = np.tensordot(unitary, unitary.conj(), axes=(0, 0))
    return np.linalg.eigvalsh(tr_choi - np.eye(len(tr_choi)))

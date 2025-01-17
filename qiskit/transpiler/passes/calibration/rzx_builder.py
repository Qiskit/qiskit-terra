# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""RZX calibration builders."""
from __future__ import annotations

import enum
import math
import warnings
from collections.abc import Sequence
from math import pi, erf

import numpy as np
from qiskit.circuit import Instruction as CircuitInst
from qiskit.circuit.library.standard_gates import RZXGate
from qiskit.exceptions import QiskitError
from qiskit.pulse import (
    Play,
    Schedule,
    ScheduleBlock,
    ControlChannel,
    DriveChannel,
    GaussianSquare,
    Waveform,
)
from qiskit.pulse import builder
from qiskit.pulse.filters import filter_instructions
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.transpiler.target import Target
from qiskit.utils.deprecate_pulse import deprecate_pulse_dependency

from .base_builder import CalibrationBuilder
from .exceptions import CalibrationNotAvailable


class CRCalType(enum.Enum):
    """Estimated calibration type of backend cross resonance operations."""

    ECR_FORWARD = "Echoed Cross Resonance corresponding to native operation"
    ECR_REVERSE = "Echoed Cross Resonance reverse of native operation"
    ECR_CX_FORWARD = "Echoed Cross Resonance CX corresponding to native operation"
    ECR_CX_REVERSE = "Echoed Cross Resonance CX reverse of native operation"
    DIRECT_CX_FORWARD = "Direct CX corresponding to native operation"
    DIRECT_CX_REVERSE = "Direct CX reverse of native operation"


class RZXCalibrationBuilder(CalibrationBuilder):
    """
    Creates calibrations for RZXGate(theta) by stretching and compressing
    Gaussian square pulses in the CX gate. This is done by retrieving (for a given pair of
    qubits) the CX schedule in the instruction schedule map of the backend defaults.
    The CX schedule must be an echoed cross-resonance gate optionally with rotary tones.
    The cross-resonance drive tones and rotary pulses must be Gaussian square pulses.
    The width of the Gaussian square pulse is adjusted so as to match the desired rotation angle.
    If the rotation angle is small such that the width disappears then the amplitude of the
    zero width Gaussian square pulse (i.e. a Gaussian) is reduced to reach the target rotation
    angle. Additional details can be found in https://arxiv.org/abs/2012.11660.
    """

    @deprecate_pulse_dependency
    def __init__(
        self,
        instruction_schedule_map: InstructionScheduleMap = None,
        verbose: bool = True,
        target: Target = None,
    ):
        """
        Initializes a RZXGate calibration builder.

        Args:
            instruction_schedule_map: The :obj:`InstructionScheduleMap` object representing the
                default pulse calibrations for the target backend
            verbose: Set True to raise a user warning when RZX schedule cannot be built.
            target: The :class:`~.Target` representing the target backend, if both
                 ``instruction_schedule_map`` and this are specified then this argument will take
                 precedence and ``instruction_schedule_map`` will be ignored.

        Raises:
            QiskitError: Instruction schedule map is not provided.
        """
        super().__init__()
        self._inst_map = instruction_schedule_map
        self._verbose = verbose
        if target:
            self._inst_map = target._instruction_schedule_map()
        if self._inst_map is None:
            raise QiskitError("Calibrations can only be added to Pulse-enabled backends")

    def supported(self, node_op: CircuitInst, qubits: list) -> bool:
        """Determine if a given node supports the calibration.

        Args:
            node_op: Target instruction object.
            qubits: Integer qubit indices to check.

        Returns:
            Return ``True`` is calibration can be provided.
        """
        return isinstance(node_op, RZXGate) and (
            "cx" in self._inst_map.instructions or "ecr" in self._inst_map.instructions
        )

    @staticmethod
    @builder.macro
    def rescale_cr_inst(instruction: Play, theta: float, sample_mult: int = 16) -> int:
        """A builder macro to play stretched pulse.

        Args:
            instruction: The instruction from which to create a new shortened or lengthened pulse.
            theta: desired angle, pi/2 is assumed to be the angle that the pulse in the given
                play instruction implements.
            sample_mult: All pulses must be a multiple of sample_mult.

        Returns:
            Duration of stretched pulse.

        Raises:
            QiskitError: if rotation angle is not assigned.
        """
        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        # This method is called for instructions which are guaranteed to play GaussianSquare pulse
        params = instruction.pulse.parameters.copy()
        risefall_sigma_ratio = (params["duration"] - params["width"]) / params["sigma"]

        # The error function is used because the Gaussian may have chopped tails.
        # Area is normalized by amplitude.
        # This makes widths robust to the rounding error.
        risefall_area = params["sigma"] * math.sqrt(2 * pi) * erf(risefall_sigma_ratio)
        full_area = params["width"] + risefall_area

        # Get estimate of target area. Assume this is pi/2 controlled rotation.
        cal_angle = pi / 2
        target_area = abs(theta) / cal_angle * full_area
        new_width = target_area - risefall_area

        if new_width >= 0:
            width = new_width
            params["amp"] *= np.sign(theta)
        else:
            width = 0
            params["amp"] *= np.sign(theta) * target_area / risefall_area

        round_duration = (
            round((width + risefall_sigma_ratio * params["sigma"]) / sample_mult) * sample_mult
        )
        params["duration"] = round_duration
        params["width"] = width

        stretched_pulse = GaussianSquare(**params)
        builder.play(stretched_pulse, instruction.channel)

        return round_duration

    def get_calibration(self, node_op: CircuitInst, qubits: list) -> Schedule | ScheduleBlock:
        """Builds the calibration schedule for the RZXGate(theta) with echos.

        Args:
            node_op: Instruction of the RZXGate(theta). I.e. params[0] is theta.
            qubits: List of qubits for which to get the schedules. The first qubit is
                the control and the second is the target.

        Returns:
            schedule: The calibration schedule for the RZXGate(theta).

        Raises:
            QiskitError: if rotation angle is not assigned.
            QiskitError: If the control and target qubits cannot be identified.
            CalibrationNotAvailable: RZX schedule cannot be built for input node.
        """
        theta = node_op.params[0]

        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        if np.isclose(theta, 0.0):
            return ScheduleBlock(name="rzx(0.000)")

        cal_type, cr_tones, comp_tones = _check_calibration_type(self._inst_map, qubits)

        if cal_type in [CRCalType.DIRECT_CX_FORWARD, CRCalType.DIRECT_CX_REVERSE]:
            if self._verbose:
                warnings.warn(
                    f"CR instruction for qubits {qubits} is likely {cal_type.value} sequence. "
                    "Pulse stretch for this calibration is not currently implemented. "
                    "RZX schedule is not generated for this qubit pair.",
                    UserWarning,
                )
            raise CalibrationNotAvailable

        # The CR instruction is in the forward (native) direction
        if cal_type in [CRCalType.ECR_CX_FORWARD, CRCalType.ECR_FORWARD]:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=DeprecationWarning)
                # `InstructionScheduleMap.get` and the pulse builder emit deprecation warnings
                # as they use classes and methods which are deprecated in Qiskit 1.3 as part of the
                # Qiskit Pulse deprecation
                xgate = self._inst_map.get("x", qubits[0])
                with builder.build(
                    default_alignment="sequential", name=f"rzx({theta:.3f})"
                ) as rzx_theta_native:
                    for cr_tone, comp_tone in zip(cr_tones, comp_tones):
                        with builder.align_left():
                            self.rescale_cr_inst(cr_tone, theta)
                            self.rescale_cr_inst(comp_tone, theta)
                        builder.call(xgate)
            return rzx_theta_native

        # The direction is not native. Add Hadamard gates to flip the direction.
        xgate = self._inst_map.get("x", qubits[1])
        szc = self._inst_map.get("rz", qubits[1], pi / 2)
        sxc = self._inst_map.get("sx", qubits[1])
        szt = self._inst_map.get("rz", qubits[0], pi / 2)
        sxt = self._inst_map.get("sx", qubits[0])
        with builder.build(name="hadamard") as hadamard:
            # Control qubit
            builder.call(szc, name="szc")
            builder.call(sxc, name="sxc")
            builder.call(szc, name="szc")
            # Target qubit
            builder.call(szt, name="szt")
            builder.call(sxt, name="sxt")
            builder.call(szt, name="szt")

        with builder.build(
            default_alignment="sequential", name=f"rzx({theta:.3f})"
        ) as rzx_theta_flip:
            builder.call(hadamard, name="hadamard")
            for cr_tone, comp_tone in zip(cr_tones, comp_tones):
                with builder.align_left():
                    self.rescale_cr_inst(cr_tone, theta)
                    self.rescale_cr_inst(comp_tone, theta)
                builder.call(xgate)
            builder.call(hadamard, name="hadamard")
        return rzx_theta_flip


class RZXCalibrationBuilderNoEcho(RZXCalibrationBuilder):
    """
    Creates calibrations for RZXGate(theta) by stretching and compressing
    Gaussian square pulses in the CX gate.

    The ``RZXCalibrationBuilderNoEcho`` is a variation of the
    :class:`~qiskit.transpiler.passes.RZXCalibrationBuilder` pass
    that creates calibrations for the cross-resonance pulses without inserting
    the echo pulses in the pulse schedule. This enables exposing the echo in
    the cross-resonance sequence as gates so that the transpiler can simplify them.
    The ``RZXCalibrationBuilderNoEcho`` only supports the hardware-native direction
    of the CX gate.
    """

    def get_calibration(self, node_op: CircuitInst, qubits: list) -> Schedule | ScheduleBlock:
        """Builds the calibration schedule for the RZXGate(theta) without echos.

        Args:
            node_op: Instruction of the RZXGate(theta). I.e. params[0] is theta.
            qubits: List of qubits for which to get the schedules. The first qubit is
                the control and the second is the target.

        Returns:
            schedule: The calibration schedule for the RZXGate(theta).

        Raises:
            QiskitError: if rotation angle is not assigned.
            QiskitError: If the control and target qubits cannot be identified,
                or the backend does not natively support the specified direction of the cx.
            CalibrationNotAvailable: RZX schedule cannot be built for input node.
        """
        theta = node_op.params[0]

        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError("Target rotation angle is not assigned.") from ex

        if np.isclose(theta, 0.0):
            return ScheduleBlock(name="rzx(0.000)")

        cal_type, cr_tones, comp_tones = _check_calibration_type(self._inst_map, qubits)

        if cal_type in [CRCalType.DIRECT_CX_FORWARD, CRCalType.DIRECT_CX_REVERSE]:
            if self._verbose:
                warnings.warn(
                    f"CR instruction for qubits {qubits} is likely {cal_type.value} sequence. "
                    "Pulse stretch for this calibration is not currently implemented. "
                    "RZX schedule is not generated for this qubit pair.",
                    UserWarning,
                )
            raise CalibrationNotAvailable

        # RZXCalibrationNoEcho only good for forward CR direction
        if cal_type in [CRCalType.ECR_CX_FORWARD, CRCalType.ECR_FORWARD]:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=DeprecationWarning)
                # Pulse builder emits deprecation warnings as part of the
                # Qiskit Pulse deprecation
                with builder.build(default_alignment="left", name=f"rzx({theta:.3f})") as rzx_theta:
                    stretched_dur = self.rescale_cr_inst(cr_tones[0], 2 * theta)
                    self.rescale_cr_inst(comp_tones[0], 2 * theta)
                    # Placeholder to make pulse gate work
                    builder.delay(stretched_dur, DriveChannel(qubits[0]))
            return rzx_theta

        raise QiskitError("RZXCalibrationBuilderNoEcho only supports hardware-native RZX gates.")


def _filter_cr_tone(time_inst_tup):
    """A helper function to filter pulses on control channels."""
    valid_types = ["GaussianSquare"]

    _, inst = time_inst_tup
    if isinstance(inst, Play) and isinstance(inst.channel, ControlChannel):
        pulse = inst.pulse
        if isinstance(pulse, Waveform) or pulse.pulse_type in valid_types:
            return True
    return False


def _filter_comp_tone(time_inst_tup):
    """A helper function to filter pulses on drive channels."""
    valid_types = ["GaussianSquare"]

    _, inst = time_inst_tup
    if isinstance(inst, Play) and isinstance(inst.channel, DriveChannel):
        pulse = inst.pulse
        if isinstance(pulse, Waveform) or pulse.pulse_type in valid_types:
            return True
    return False


def _check_calibration_type(
    inst_sched_map: InstructionScheduleMap, qubits: Sequence[int]
) -> tuple[CRCalType, list[Play], list[Play]]:
    """A helper function to check type of CR calibration.

    Args:
        inst_sched_map: instruction schedule map of the backends
        qubits: ordered tuple of qubits for cross resonance (q_control, q_target)

    Returns:
        Filtered instructions and most-likely type of calibration.

    Raises:
        QiskitError: Unknown calibration type is detected.
    """
    cal_type = None
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=DeprecationWarning)
        # `InstructionScheduleMap.get` and `filter_instructions` emit deprecation warnings
        # as they use classes and methods which are deprecated in Qiskit 1.3 as part of the
        # Qiskit Pulse deprecation
        if inst_sched_map.has("cx", qubits):
            cr_sched = inst_sched_map.get("cx", qubits=qubits)
        elif inst_sched_map.has("ecr", qubits):
            cr_sched = inst_sched_map.get("ecr", qubits=qubits)
            cal_type = CRCalType.ECR_FORWARD
        elif inst_sched_map.has("ecr", tuple(reversed(qubits))):
            cr_sched = inst_sched_map.get("ecr", tuple(reversed(qubits)))
            cal_type = CRCalType.ECR_REVERSE
        else:
            raise QiskitError(
                f"Native direction cannot be determined: operation on qubits {qubits} "
                f"for the following instruction schedule map:\n{inst_sched_map}"
            )

        cr_tones = [t[1] for t in filter_instructions(cr_sched, [_filter_cr_tone]).instructions]
        comp_tones = [t[1] for t in filter_instructions(cr_sched, [_filter_comp_tone]).instructions]

    if cal_type is None:
        if len(comp_tones) == 0:
            raise QiskitError(
                f"{repr(cr_sched)} has no target compensation tones. "
                "Native ECR direction cannot be determined."
            )
        # Determine native direction, assuming only single drive channel per qubit.
        # This guarantees channel and qubit index equality.
        if comp_tones[0].channel.index == qubits[1]:
            cal_type = CRCalType.ECR_CX_FORWARD
        else:
            cal_type = CRCalType.ECR_CX_REVERSE

    if len(cr_tones) == 2 and len(comp_tones) in (0, 2):
        # ECR can be implemented without compensation tone at price of lower fidelity.
        # Remarkable noisy terms are usually eliminated by echo.
        return cal_type, cr_tones, comp_tones

    if len(cr_tones) == 1 and len(comp_tones) == 1:
        # Direct CX must have compensation tone on target qubit.
        # Otherwise, it cannot eliminate IX interaction.
        if comp_tones[0].channel.index == qubits[1]:
            return CRCalType.DIRECT_CX_FORWARD, cr_tones, comp_tones
        else:
            return CRCalType.DIRECT_CX_REVERSE, cr_tones, comp_tones
    raise QiskitError(
        f"{repr(cr_sched)} is undefined pulse sequence. "
        "Check if this is a calibration for cross resonance operation."
    )

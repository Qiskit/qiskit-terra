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

"""Module containing circuit scheduling passes."""

from .alap import ALAPSchedule
from .asap import ASAPSchedule
from .dynamical_decoupling import DynamicalDecoupling
from .scheduling import ALAPScheduleAnalysis, ASAPScheduleAnalysis, SetIOLatency
from .time_unit_conversion import TimeUnitConversion
from .padding import (
    PadDelay,
    PadDynamicalDecoupling,
    ContextAwareDynamicalDecoupling,
)
from .alignments import InstructionDurationCheck, ValidatePulseGates, ConstrainedReschedule

# For backward compatibility
from . import alignments as instruction_alignments

# TODO Deprecated pass. Will be removed after deprecation period.
from .alignments import AlignMeasures

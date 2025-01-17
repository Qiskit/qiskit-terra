# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""A pass to check if input circuit requires reschedule."""

from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.target import Target


class InstructionDurationCheck(AnalysisPass):
    """Duration validation pass for reschedule.

    This pass investigates the input quantum circuit and checks if the circuit requires
    rescheduling for execution. Note that this pass can be triggered without scheduling.
    This pass only checks the duration of delay instructions and user defined pulse gates,
    which report duration values without pre-scheduling.

    This pass assumes backend supported instructions, i.e. basis gates, have no violation
    of the hardware alignment constraints, which is true in general.
    """

    def __init__(self, acquire_alignment: int = 1, pulse_alignment: int = 1, target: Target = None):
        """Create new duration validation pass.

        The alignment values depend on the control electronics of your quantum processor.

        Args:
            acquire_alignment: Integer number representing the minimum time resolution to
                trigger acquisition instruction in units of ``dt``.
            pulse_alignment: Integer number representing the minimum time resolution to
                trigger gate instruction in units of ``dt``.
            target: The :class:`~.Target` representing the target backend, if
                ``target`` is specified then this argument will take
                precedence and ``acquire_alignment`` and ``pulse_alignment`` will be ignored.
        """
        super().__init__()
        self.acquire_align = acquire_alignment
        self.pulse_align = pulse_alignment
        if target is not None:
            self.acquire_align = target.acquire_alignment
            self.pulse_align = target.pulse_alignment

    def run(self, dag: DAGCircuit):
        """Run duration validation passes.

        Args:
            dag: DAG circuit to check instruction durations.
        """
        self.property_set["reschedule_required"] = False

        # Rescheduling is not necessary
        if self.acquire_align == 1 and self.pulse_align == 1:
            return

        # Check delay durations
        for delay_node in dag.op_nodes(Delay):
            dur = delay_node.op.duration
            if not (dur % self.acquire_align == 0 and dur % self.pulse_align == 0):
                self.property_set["reschedule_required"] = True
                return

        # Check custom gate durations
        for inst_defs in dag._calibrations_prop.values():
            for caldef in inst_defs.values():
                dur = caldef.duration
                if not (dur % self.acquire_align == 0 and dur % self.pulse_align == 0):
                    self.property_set["reschedule_required"] = True
                    return

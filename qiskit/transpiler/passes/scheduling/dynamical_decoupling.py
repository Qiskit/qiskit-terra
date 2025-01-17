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

"""Dynamical Decoupling insertion pass."""

import itertools
import warnings

import numpy as np
from qiskit.circuit import Gate, Delay, Reset
from qiskit.circuit.library.standard_gates import IGate, UGate, U3Gate
from qiskit.dagcircuit import DAGOpNode, DAGInNode
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.synthesis.one_qubit import OneQubitEulerDecomposer
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.utils.deprecation import deprecate_func


class DynamicalDecoupling(TransformationPass):
    """Dynamical decoupling insertion pass.

    This pass works on a scheduled, physical circuit. It scans the circuit for
    idle periods of time (i.e. those containing delay instructions) and inserts
    a DD sequence of gates in those spots. These gates amount to the identity,
    so do not alter the logical action of the circuit, but have the effect of
    mitigating decoherence in those idle periods.

    As a special case, the pass allows a length-1 sequence (e.g. [XGate()]).
    In this case the DD insertion happens only when the gate inverse can be
    absorbed into a neighboring gate in the circuit (so we would still be
    replacing Delay with something that is equivalent to the identity).
    This can be used, for instance, as a Hahn echo.

    This pass ensures that the inserted sequence preserves the circuit exactly
    (including global phase).

    .. plot::
       :alt: Output from the previous code.
       :include-source:

        import numpy as np
        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import XGate
        from qiskit.transpiler import PassManager, InstructionDurations
        from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling
        from qiskit.visualization import timeline_drawer

        # Because the legacy passes do not propagate the scheduling information correctly, it is
        # necessary to run a no-op "re-schedule" before the output circuits can be drawn.
        def draw(circuit):
            from qiskit import transpile

            scheduled = transpile(
                circuit,
                optimization_level=0,
                instruction_durations=InstructionDurations(),
                scheduling_method="alap",
            )
            return timeline_drawer(scheduled)

        circ = QuantumCircuit(4)
        circ.h(0)
        circ.cx(0, 1)
        circ.cx(1, 2)
        circ.cx(2, 3)
        circ.measure_all()
        durations = InstructionDurations(
            [("h", 0, 50), ("cx", [0, 1], 700), ("reset", None, 10),
             ("cx", [1, 2], 200), ("cx", [2, 3], 300),
             ("x", None, 50), ("measure", None, 1000)]
        )
        # balanced X-X sequence on all qubits
        dd_sequence = [XGate(), XGate()]
        pm = PassManager([ALAPSchedule(durations),
                          DynamicalDecoupling(durations, dd_sequence)])
        circ_dd = pm.run(circ)
        draw(circ_dd)

        # Uhrig sequence on qubit 0
        n = 8
        dd_sequence = [XGate()] * n
        def uhrig_pulse_location(k):
            return np.sin(np.pi * (k + 1) / (2 * n + 2)) ** 2
        spacing = []
        for k in range(n):
            spacing.append(uhrig_pulse_location(k) - sum(spacing))
        spacing.append(1 - sum(spacing))
        pm = PassManager(
            [
                ALAPSchedule(durations),
                DynamicalDecoupling(durations, dd_sequence, qubits=[0], spacing=spacing),
            ]
        )
        circ_dd = pm.run(circ)
        draw(circ_dd)
    """

    @deprecate_func(
        additional_msg=(
            "Instead, use :class:`~.PadDynamicalDecoupling`, which performs the same "
            "function but requires scheduling and alignment analysis passes to run prior to it."
        ),
        since="1.1.0",
    )
    def __init__(
        self, durations, dd_sequence, qubits=None, spacing=None, skip_reset_qubits=True, target=None
    ):
        """Dynamical decoupling initializer.

        Args:
            durations (InstructionDurations): Durations of instructions to be
                used in scheduling.
            dd_sequence (list[Gate]): sequence of gates to apply in idle spots.
            qubits (list[int]): physical qubits on which to apply DD.
                If None, all qubits will undergo DD (when possible).
            spacing (list[float]): a list of spacings between the DD gates.
                The available slack will be divided according to this.
                The list length must be one more than the length of dd_sequence,
                and the elements must sum to 1. If None, a balanced spacing
                will be used [d/2, d, d, ..., d, d, d/2].
            skip_reset_qubits (bool): if True, does not insert DD on idle
                periods that immediately follow initialized/reset qubits (as
                qubits in the ground state are less susceptible to decoherence).
            target (Target): The :class:`~.Target` representing the target backend, if both
                  ``durations`` and this are specified then this argument will take
                  precedence and ``durations`` will be ignored.
        """
        super().__init__()
        self._durations = durations
        self._dd_sequence = dd_sequence
        self._qubits = qubits
        self._spacing = spacing
        self._skip_reset_qubits = skip_reset_qubits
        self._target = target
        if target is not None:
            self._durations = target.durations()
            for gate in dd_sequence:
                if gate.name not in target.operation_names:
                    raise TranspilerError(
                        f"{gate.name} in dd_sequence is not supported in the target"
                    )

    def run(self, dag):
        """Run the DynamicalDecoupling pass on dag.

        Args:
            dag (DAGCircuit): a scheduled DAG.

        Returns:
            DAGCircuit: equivalent circuit with delays interrupted by DD,
                where possible.

        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("DD runs on physical circuits only.")

        if dag.duration is None:
            raise TranspilerError("DD runs after circuit is scheduled.")

        durations = self._update_inst_durations(dag)

        num_pulses = len(self._dd_sequence)
        sequence_gphase = 0
        if num_pulses != 1:
            if num_pulses % 2 != 0:
                raise TranspilerError("DD sequence must contain an even number of gates (or 1).")
            noop = np.eye(2)
            for gate in self._dd_sequence:
                noop = noop.dot(gate.to_matrix())
            if not matrix_equal(noop, IGate().to_matrix(), ignore_phase=True):
                raise TranspilerError("The DD sequence does not make an identity operation.")
            sequence_gphase = np.angle(noop[0][0])

        if self._qubits is None:
            self._qubits = set(range(dag.num_qubits()))
        else:
            self._qubits = set(self._qubits)

        if self._spacing:
            if sum(self._spacing) != 1 or any(a < 0 for a in self._spacing):
                raise TranspilerError(
                    "The spacings must be given in terms of fractions "
                    "of the slack period and sum to 1."
                )
        else:  # default to balanced spacing
            mid = 1 / num_pulses
            end = mid / 2
            self._spacing = [end] + [mid] * (num_pulses - 1) + [end]

        for qarg in list(self._qubits):
            for gate in self._dd_sequence:
                if not self.__gate_supported(gate, qarg):
                    self._qubits.discard(qarg)
                    break

        index_sequence_duration_map = {}
        for physical_qubit in self._qubits:
            dd_sequence_duration = 0
            for index, gate in enumerate(self._dd_sequence):
                gate = gate.to_mutable()
                self._dd_sequence[index] = gate
                gate.duration = durations.get(gate, physical_qubit)

                dd_sequence_duration += gate.duration
            index_sequence_duration_map[physical_qubit] = dd_sequence_duration

        new_dag = dag.copy_empty_like()

        for nd in dag.topological_op_nodes():
            if not isinstance(nd.op, Delay):
                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs, check=False)
                continue

            dag_qubit = nd.qargs[0]
            physical_qubit = dag.find_bit(dag_qubit).index
            if physical_qubit not in self._qubits:  # skip unwanted qubits
                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs, check=False)
                continue

            pred = next(dag.predecessors(nd))
            succ = next(dag.successors(nd))
            if self._skip_reset_qubits:  # discount initial delays
                if isinstance(pred, DAGInNode) or isinstance(pred.op, Reset):
                    new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs, check=False)
                    continue

            dd_sequence_duration = index_sequence_duration_map[physical_qubit]
            slack = nd.op.duration - dd_sequence_duration
            if slack <= 0:  # dd doesn't fit
                new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs, check=False)
                continue

            if num_pulses == 1:  # special case of using a single gate for DD
                u_inv = self._dd_sequence[0].inverse().to_matrix()
                theta, phi, lam, phase = OneQubitEulerDecomposer().angles_and_phase(u_inv)
                # absorb the inverse into the successor (from left in circuit)
                if isinstance(succ, DAGOpNode) and isinstance(succ.op, (UGate, U3Gate)):
                    theta_r, phi_r, lam_r = succ.op.params
                    succ.op.params = Optimize1qGates.compose_u3(
                        theta_r, phi_r, lam_r, theta, phi, lam
                    )
                    sequence_gphase += phase
                # absorb the inverse into the predecessor (from right in circuit)
                elif isinstance(pred, DAGOpNode) and isinstance(pred.op, (UGate, U3Gate)):
                    theta_l, phi_l, lam_l = pred.op.params
                    pred.op.params = Optimize1qGates.compose_u3(
                        theta, phi, lam, theta_l, phi_l, lam_l
                    )
                    sequence_gphase += phase
                # don't do anything if there's no single-qubit gate to absorb the inverse
                else:
                    new_dag.apply_operation_back(nd.op, nd.qargs, nd.cargs, check=False)
                    continue

            # insert the actual DD sequence
            taus = [int(slack * a) for a in self._spacing]
            unused_slack = slack - sum(taus)  # unused, due to rounding to int multiples of dt
            middle_index = int((len(taus) - 1) / 2)  # arbitrary: redistribute to middle
            taus[middle_index] += unused_slack  # now we add up to original delay duration

            for tau, gate in itertools.zip_longest(taus, self._dd_sequence):
                if tau > 0:
                    new_dag.apply_operation_back(Delay(tau), [dag_qubit], check=False)
                if gate is not None:
                    new_dag.apply_operation_back(gate, [dag_qubit], check=False)

            new_dag.global_phase = new_dag.global_phase + sequence_gphase

        return new_dag

    def _update_inst_durations(self, dag):
        """Update instruction durations with circuit information. If the dag contains gate
        calibrations and no instruction durations were provided through the target or as a
        standalone input, the circuit calibration durations will be used.
        The priority order for instruction durations is: target > standalone > circuit.
        """
        circ_durations = InstructionDurations()

        if dag._calibrations_prop:
            cal_durations = []
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=DeprecationWarning)
                # `schedule.duration` emits pulse deprecation warnings which we don't want
                # to see here
                for gate, gate_cals in dag._calibrations_prop.items():
                    for (qubits, parameters), schedule in gate_cals.items():
                        cal_durations.append((gate, qubits, parameters, schedule.duration))
            circ_durations.update(cal_durations, circ_durations.dt)

        if self._durations is not None:
            circ_durations.update(self._durations, getattr(self._durations, "dt", None))

        return circ_durations

    def __gate_supported(self, gate: Gate, qarg: int) -> bool:
        """A gate is supported on the qubit (qarg) or not."""
        if self._target is None or self._target.instruction_supported(gate.name, qargs=(qarg,)):
            return True
        return False

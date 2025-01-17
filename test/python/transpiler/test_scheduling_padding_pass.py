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

"""Test the Scheduling/PadDelay passes"""

import unittest

from ddt import ddt, data, unpack
from qiskit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit.library import CXGate, HGate
from qiskit.pulse import Schedule, Play, Constant, DriveChannel
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes import (
    ASAPScheduleAnalysis,
    ALAPScheduleAnalysis,
    PadDelay,
    SetIOLatency,
)
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target, InstructionProperties
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestSchedulingAndPaddingPass(QiskitTestCase):
    """Tests the Scheduling passes"""

    def test_alap_agree_with_reverse_asap_reverse(self):
        """Test if ALAP schedule agrees with doubly-reversed ASAP schedule."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        qc.measure_all()

        durations = InstructionDurations(
            [("h", 0, 200), ("cx", [0, 1], 700), ("measure", None, 1000)]
        )

        pm = PassManager([ALAPScheduleAnalysis(durations), PadDelay()])
        alap_qc = pm.run(qc)

        pm = PassManager([ASAPScheduleAnalysis(durations), PadDelay()])
        new_qc = pm.run(qc.reverse_ops())
        new_qc = new_qc.reverse_ops()
        new_qc.name = new_qc.name

        self.assertEqual(alap_qc, new_qc)

    def test_alap_agree_with_reverse_asap_with_target(self):
        """Test if ALAP schedule agrees with doubly-reversed ASAP schedule."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        qc.measure_all()

        target = Target(num_qubits=2, dt=3.5555555555555554)
        target.add_instruction(HGate(), {(0,): InstructionProperties(duration=200)})
        target.add_instruction(CXGate(), {(0, 1): InstructionProperties(duration=700)})
        target.add_instruction(
            Measure(),
            {
                (0,): InstructionProperties(duration=1000),
                (1,): InstructionProperties(duration=1000),
            },
        )

        pm = PassManager([ALAPScheduleAnalysis(target=target), PadDelay()])
        alap_qc = pm.run(qc)

        pm = PassManager([ASAPScheduleAnalysis(target=target), PadDelay()])
        new_qc = pm.run(qc.reverse_ops())
        new_qc = new_qc.reverse_ops()
        new_qc.name = new_qc.name

        self.assertEqual(alap_qc, new_qc)

    @data(ALAPScheduleAnalysis, ASAPScheduleAnalysis)
    def test_classically_controlled_gate_after_measure(self, schedule_pass):
        """Test if ALAP/ASAP schedules circuits with c_if after measure with a common clbit.
        See: https://github.com/Qiskit/qiskit-terra/issues/7654

        (input)
             ┌─┐
        q_0: ┤M├───────────
             └╥┘   ┌───┐
        q_1: ─╫────┤ X ├───
              ║    └─╥─┘
              ║ ┌────╨────┐
        c: 1/═╩═╡ c_0 = T ╞
              0 └─────────┘

        (scheduled)
                                ┌─┐┌────────────────┐
        q_0: ───────────────────┤M├┤ Delay(200[dt]) ├
             ┌─────────────────┐└╥┘└─────┬───┬──────┘
        q_1: ┤ Delay(1000[dt]) ├─╫───────┤ X ├───────
             └─────────────────┘ ║       └─╥─┘
                                 ║    ┌────╨────┐
        c: 1/════════════════════╩════╡ c_0=0x1 ╞════
                                 0    └─────────┘
        """
        qc = QuantumCircuit(2, 1)
        qc.measure(0, 0)
        with self.assertWarns(DeprecationWarning):
            qc.x(1).c_if(0, True)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)])
        pm = PassManager([schedule_pass(durations), PadDelay()])
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.measure(0, 0)
        expected.delay(1000, 1)  # x.c_if starts after measure
        with self.assertWarns(DeprecationWarning):
            expected.x(1).c_if(0, True)
        expected.delay(200, 0)

        self.assertEqual(expected, scheduled)

    @data(ALAPScheduleAnalysis, ASAPScheduleAnalysis)
    def test_measure_after_measure(self, schedule_pass):
        """Test if ALAP/ASAP schedules circuits with measure after measure with a common clbit.
        See: https://github.com/Qiskit/qiskit-terra/issues/7654

        (input)
             ┌───┐┌─┐
        q_0: ┤ X ├┤M├───
             └───┘└╥┘┌─┐
        q_1: ──────╫─┤M├
                   ║ └╥┘
        c: 1/══════╩══╩═
                   0  0

        (scheduled)
                    ┌───┐       ┌─┐┌─────────────────┐
        q_0: ───────┤ X ├───────┤M├┤ Delay(1000[dt]) ├
             ┌──────┴───┴──────┐└╥┘└───────┬─┬───────┘
        q_1: ┤ Delay(1200[dt]) ├─╫─────────┤M├────────
             └─────────────────┘ ║         └╥┘
        c: 1/════════════════════╩══════════╩═════════
                                 0          0
        """
        qc = QuantumCircuit(2, 1)
        qc.x(0)
        qc.measure(0, 0)
        qc.measure(1, 0)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)])
        pm = PassManager([schedule_pass(durations), PadDelay()])
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.x(0)
        expected.measure(0, 0)
        expected.delay(1200, 1)
        expected.measure(1, 0)
        expected.delay(1000, 0)

        self.assertEqual(expected, scheduled)

    @data(ALAPScheduleAnalysis, ASAPScheduleAnalysis)
    def test_c_if_on_different_qubits(self, schedule_pass):
        """Test if ALAP/ASAP schedules circuits with `c_if`s on different qubits.

        (input)
             ┌─┐
        q_0: ┤M├──────────────────────
             └╥┘   ┌───┐
        q_1: ─╫────┤ X ├──────────────
              ║    └─╥─┘      ┌───┐
        q_2: ─╫──────╫────────┤ X ├───
              ║      ║        └─╥─┘
              ║ ┌────╨────┐┌────╨────┐
        c: 1/═╩═╡ c_0 = T ╞╡ c_0 = T ╞
              0 └─────────┘└─────────┘

        (scheduled)

                                ┌─┐┌────────────────┐
        q_0: ───────────────────┤M├┤ Delay(200[dt]) ├───────────
             ┌─────────────────┐└╥┘└─────┬───┬──────┘
        q_1: ┤ Delay(1000[dt]) ├─╫───────┤ X ├──────────────────
             ├─────────────────┤ ║       └─╥─┘          ┌───┐
        q_2: ┤ Delay(1000[dt]) ├─╫─────────╫────────────┤ X ├───
             └─────────────────┘ ║         ║            └─╥─┘
                                 ║    ┌────╨────┐    ┌────╨────┐
        c: 1/════════════════════╩════╡ c_0=0x1 ╞════╡ c_0=0x1 ╞
                                 0    └─────────┘    └─────────┘
        """
        qc = QuantumCircuit(3, 1)
        qc.measure(0, 0)
        with self.assertWarns(DeprecationWarning):
            qc.x(1).c_if(0, True)
        with self.assertWarns(DeprecationWarning):
            qc.x(2).c_if(0, True)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)])
        pm = PassManager([schedule_pass(durations), PadDelay()])
        scheduled = pm.run(qc)

        expected = QuantumCircuit(3, 1)
        expected.measure(0, 0)
        expected.delay(1000, 1)
        expected.delay(1000, 2)
        with self.assertWarns(DeprecationWarning):
            expected.x(1).c_if(0, True)
        with self.assertWarns(DeprecationWarning):
            expected.x(2).c_if(0, True)
        expected.delay(200, 0)

        self.assertEqual(expected, scheduled)

    @data(ALAPScheduleAnalysis, ASAPScheduleAnalysis)
    def test_shorter_measure_after_measure(self, schedule_pass):
        """Test if ALAP/ASAP schedules circuits with shorter measure after measure with a common clbit.

        (input)
             ┌─┐
        q_0: ┤M├───
             └╥┘┌─┐
        q_1: ─╫─┤M├
              ║ └╥┘
        c: 1/═╩══╩═
              0  0

        (scheduled)
                                ┌─┐┌────────────────┐
        q_0: ───────────────────┤M├┤ Delay(700[dt]) ├
             ┌─────────────────┐└╥┘└──────┬─┬───────┘
        q_1: ┤ Delay(1000[dt]) ├─╫────────┤M├────────
             └─────────────────┘ ║        └╥┘
        c: 1/════════════════════╩═════════╩═════════
                                 0         0
        """
        qc = QuantumCircuit(2, 1)
        qc.measure(0, 0)
        qc.measure(1, 0)

        durations = InstructionDurations([("measure", [0], 1000), ("measure", [1], 700)])
        pm = PassManager([schedule_pass(durations), PadDelay()])
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.measure(0, 0)
        expected.delay(1000, 1)
        expected.measure(1, 0)
        expected.delay(700, 0)

        self.assertEqual(expected, scheduled)

    @data(ALAPScheduleAnalysis, ASAPScheduleAnalysis)
    def test_measure_after_c_if(self, schedule_pass):
        """Test if ALAP/ASAP schedules circuits with c_if after measure with a common clbit.

        (input)
             ┌─┐
        q_0: ┤M├──────────────
             └╥┘   ┌───┐
        q_1: ─╫────┤ X ├──────
              ║    └─╥─┘   ┌─┐
        q_2: ─╫──────╫─────┤M├
              ║ ┌────╨────┐└╥┘
        c: 1/═╩═╡ c_0 = T ╞═╩═
              0 └─────────┘ 0

        (scheduled)
                                ┌─┐┌─────────────────┐
        q_0: ───────────────────┤M├┤ Delay(1000[dt]) ├──────────────────
             ┌─────────────────┐└╥┘└──────┬───┬──────┘┌────────────────┐
        q_1: ┤ Delay(1000[dt]) ├─╫────────┤ X ├───────┤ Delay(800[dt]) ├
             ├─────────────────┤ ║        └─╥─┘       └──────┬─┬───────┘
        q_2: ┤ Delay(1000[dt]) ├─╫──────────╫────────────────┤M├────────
             └─────────────────┘ ║     ┌────╨────┐           └╥┘
        c: 1/════════════════════╩═════╡ c_0=0x1 ╞════════════╩═════════
                                 0     └─────────┘            0
        """
        qc = QuantumCircuit(3, 1)
        qc.measure(0, 0)
        with self.assertWarns(DeprecationWarning):
            qc.x(1).c_if(0, 1)
        qc.measure(2, 0)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)])
        pm = PassManager([schedule_pass(durations), PadDelay()])
        scheduled = pm.run(qc)

        expected = QuantumCircuit(3, 1)
        expected.delay(1000, 1)
        expected.delay(1000, 2)
        expected.measure(0, 0)
        with self.assertWarns(DeprecationWarning):
            expected.x(1).c_if(0, 1)
        expected.measure(2, 0)
        expected.delay(1000, 0)
        expected.delay(800, 1)

        self.assertEqual(expected, scheduled)

    def test_parallel_gate_different_length(self):
        """Test circuit having two parallel instruction with different length.

        (input)
             ┌───┐┌─┐
        q_0: ┤ X ├┤M├───
             ├───┤└╥┘┌─┐
        q_1: ┤ X ├─╫─┤M├
             └───┘ ║ └╥┘
        c: 2/══════╩══╩═
                   0  1

        (expected, ALAP)
             ┌────────────────┐┌───┐┌─┐
        q_0: ┤ Delay(200[dt]) ├┤ X ├┤M├
             └─────┬───┬──────┘└┬─┬┘└╥┘
        q_1: ──────┤ X ├────────┤M├──╫─
                   └───┘        └╥┘  ║
        c: 2/════════════════════╩═══╩═
                                 1   0

        (expected, ASAP)
             ┌───┐┌─┐┌────────────────┐
        q_0: ┤ X ├┤M├┤ Delay(200[dt]) ├
             ├───┤└╥┘└──────┬─┬───────┘
        q_1: ┤ X ├─╫────────┤M├────────
             └───┘ ║        └╥┘
        c: 2/══════╩═════════╩═════════
                   0         1

        """
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.x(1)
        qc.measure(0, 0)
        qc.measure(1, 1)

        durations = InstructionDurations(
            [("x", [0], 200), ("x", [1], 400), ("measure", None, 1000)]
        )
        pm = PassManager([ALAPScheduleAnalysis(durations), PadDelay()])
        qc_alap = pm.run(qc)

        alap_expected = QuantumCircuit(2, 2)
        alap_expected.delay(200, 0)
        alap_expected.x(0)
        alap_expected.x(1)
        alap_expected.measure(0, 0)
        alap_expected.measure(1, 1)

        self.assertEqual(qc_alap, alap_expected)

        pm = PassManager([ASAPScheduleAnalysis(durations), PadDelay()])
        qc_asap = pm.run(qc)

        asap_expected = QuantumCircuit(2, 2)
        asap_expected.x(0)
        asap_expected.x(1)
        asap_expected.measure(0, 0)  # immediately start after X gate
        asap_expected.measure(1, 1)
        asap_expected.delay(200, 0)

        self.assertEqual(qc_asap, asap_expected)

    def test_parallel_gate_different_length_with_barrier(self):
        """Test circuit having two parallel instruction with different length with barrier.

        (input)
             ┌───┐┌─┐
        q_0: ┤ X ├┤M├───
             ├───┤└╥┘┌─┐
        q_1: ┤ X ├─╫─┤M├
             └───┘ ║ └╥┘
        c: 2/══════╩══╩═
                   0  1

        (expected, ALAP)
             ┌────────────────┐┌───┐ ░ ┌─┐
        q_0: ┤ Delay(200[dt]) ├┤ X ├─░─┤M├───
             └─────┬───┬──────┘└───┘ ░ └╥┘┌─┐
        q_1: ──────┤ X ├─────────────░──╫─┤M├
                   └───┘             ░  ║ └╥┘
        c: 2/═══════════════════════════╩══╩═
                                        0  1

        (expected, ASAP)
             ┌───┐┌────────────────┐ ░ ┌─┐
        q_0: ┤ X ├┤ Delay(200[dt]) ├─░─┤M├───
             ├───┤└────────────────┘ ░ └╥┘┌─┐
        q_1: ┤ X ├───────────────────░──╫─┤M├
             └───┘                   ░  ║ └╥┘
        c: 2/═══════════════════════════╩══╩═
                                        0  1
        """
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.x(1)
        qc.barrier()
        qc.measure(0, 0)
        qc.measure(1, 1)

        durations = InstructionDurations(
            [("x", [0], 200), ("x", [1], 400), ("measure", None, 1000)]
        )
        pm = PassManager([ALAPScheduleAnalysis(durations), PadDelay()])
        qc_alap = pm.run(qc)

        alap_expected = QuantumCircuit(2, 2)
        alap_expected.delay(200, 0)
        alap_expected.x(0)
        alap_expected.x(1)
        alap_expected.barrier()
        alap_expected.measure(0, 0)
        alap_expected.measure(1, 1)

        self.assertEqual(qc_alap, alap_expected)

        pm = PassManager([ASAPScheduleAnalysis(durations), PadDelay()])
        qc_asap = pm.run(qc)

        asap_expected = QuantumCircuit(2, 2)
        asap_expected.x(0)
        asap_expected.delay(200, 0)
        asap_expected.x(1)
        asap_expected.barrier()
        asap_expected.measure(0, 0)
        asap_expected.measure(1, 1)

        self.assertEqual(qc_asap, asap_expected)

    def test_measure_after_c_if_on_edge_locking(self):
        """Test if ALAP/ASAP schedules circuits with c_if after measure with a common clbit.

        The scheduler is configured to reproduce behavior of the 0.20.0,
        in which clbit lock is applied to the end-edge of measure instruction.
        See https://github.com/Qiskit/qiskit-terra/pull/7655

        (input)
             ┌─┐
        q_0: ┤M├──────────────
             └╥┘   ┌───┐
        q_1: ─╫────┤ X ├──────
              ║    └─╥─┘   ┌─┐
        q_2: ─╫──────╫─────┤M├
              ║ ┌────╨────┐└╥┘
        c: 1/═╩═╡ c_0 = T ╞═╩═
              0 └─────────┘ 0

        (ASAP scheduled)
                                ┌─┐┌────────────────┐
        q_0: ───────────────────┤M├┤ Delay(200[dt]) ├─────────────────────
             ┌─────────────────┐└╥┘└─────┬───┬──────┘
        q_1: ┤ Delay(1000[dt]) ├─╫───────┤ X ├────────────────────────────
             └─────────────────┘ ║       └─╥─┘       ┌─┐┌────────────────┐
        q_2: ────────────────────╫─────────╫─────────┤M├┤ Delay(200[dt]) ├
                                 ║    ┌────╨────┐    └╥┘└────────────────┘
        c: 1/════════════════════╩════╡ c_0=0x1 ╞═════╩═══════════════════
                                 0    └─────────┘     0

        (ALAP scheduled)
                                ┌─┐┌────────────────┐
        q_0: ───────────────────┤M├┤ Delay(200[dt]) ├───
             ┌─────────────────┐└╥┘└─────┬───┬──────┘
        q_1: ┤ Delay(1000[dt]) ├─╫───────┤ X ├──────────
             └┬────────────────┤ ║       └─╥─┘       ┌─┐
        q_2: ─┤ Delay(200[dt]) ├─╫─────────╫─────────┤M├
              └────────────────┘ ║    ┌────╨────┐    └╥┘
        c: 1/════════════════════╩════╡ c_0=0x1 ╞═════╩═
                                 0    └─────────┘     0

        """
        qc = QuantumCircuit(3, 1)
        qc.measure(0, 0)
        with self.assertWarns(DeprecationWarning):
            qc.x(1).c_if(0, 1)
        qc.measure(2, 0)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)])

        # lock at the end edge
        actual_asap = PassManager(
            [
                SetIOLatency(clbit_write_latency=1000),
                ASAPScheduleAnalysis(durations),
                PadDelay(),
            ]
        ).run(qc)
        actual_alap = PassManager(
            [
                SetIOLatency(clbit_write_latency=1000),
                ALAPScheduleAnalysis(durations),
                PadDelay(),
            ]
        ).run(qc)

        # start times of 2nd measure depends on ASAP/ALAP
        expected_asap = QuantumCircuit(3, 1)
        expected_asap.measure(0, 0)
        expected_asap.delay(1000, 1)
        with self.assertWarns(DeprecationWarning):
            expected_asap.x(1).c_if(0, 1)
        expected_asap.measure(2, 0)
        expected_asap.delay(200, 0)
        expected_asap.delay(200, 2)
        self.assertEqual(expected_asap, actual_asap)

        expected_alap = QuantumCircuit(3, 1)
        expected_alap.measure(0, 0)
        expected_alap.delay(1000, 1)
        with self.assertWarns(DeprecationWarning):
            expected_alap.x(1).c_if(0, 1)
        expected_alap.delay(200, 2)
        expected_alap.measure(2, 0)
        expected_alap.delay(200, 0)
        self.assertEqual(expected_alap, actual_alap)

    @data([100, 200], [500, 0], [1000, 200])
    @unpack
    def test_active_reset_circuit(self, write_lat, cond_lat):
        """Test practical example of reset circuit.

        Because of the stimulus pulse overlap with the previous XGate on the q register,
        measure instruction is always triggered after XGate regardless of write latency.
        Thus only conditional latency matters in the scheduling.

        (input)
             ┌─┐   ┌───┐   ┌─┐   ┌───┐   ┌─┐   ┌───┐
          q: ┤M├───┤ X ├───┤M├───┤ X ├───┤M├───┤ X ├───
             └╥┘   └─╥─┘   └╥┘   └─╥─┘   └╥┘   └─╥─┘
              ║ ┌────╨────┐ ║ ┌────╨────┐ ║ ┌────╨────┐
        c: 1/═╩═╡ c_0=0x1 ╞═╩═╡ c_0=0x1 ╞═╩═╡ c_0=0x1 ╞
              0 └─────────┘ 0 └─────────┘ 0 └─────────┘

        """
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        with self.assertWarns(DeprecationWarning):
            qc.x(0).c_if(0, 1)
        qc.measure(0, 0)
        with self.assertWarns(DeprecationWarning):
            qc.x(0).c_if(0, 1)
        qc.measure(0, 0)
        with self.assertWarns(DeprecationWarning):
            qc.x(0).c_if(0, 1)

        durations = InstructionDurations([("x", None, 100), ("measure", None, 1000)])

        actual_asap = PassManager(
            [
                SetIOLatency(clbit_write_latency=write_lat, conditional_latency=cond_lat),
                ASAPScheduleAnalysis(durations),
                PadDelay(),
            ]
        ).run(qc)

        actual_alap = PassManager(
            [
                SetIOLatency(clbit_write_latency=write_lat, conditional_latency=cond_lat),
                ALAPScheduleAnalysis(durations),
                PadDelay(),
            ]
        ).run(qc)

        expected = QuantumCircuit(1, 1)
        expected.measure(0, 0)
        if cond_lat > 0:
            expected.delay(cond_lat, 0)
        with self.assertWarns(DeprecationWarning):
            expected.x(0).c_if(0, 1)
        expected.measure(0, 0)
        if cond_lat > 0:
            expected.delay(cond_lat, 0)
        with self.assertWarns(DeprecationWarning):
            expected.x(0).c_if(0, 1)
        expected.measure(0, 0)
        if cond_lat > 0:
            expected.delay(cond_lat, 0)
        with self.assertWarns(DeprecationWarning):
            expected.x(0).c_if(0, 1)

        self.assertEqual(expected, actual_asap)
        self.assertEqual(expected, actual_alap)

    def test_random_complicated_circuit(self):
        """Test scheduling complicated circuit with control flow.

        (input)
             ┌────────────────┐   ┌───┐    ░                  ┌───┐   »
        q_0: ┤ Delay(100[dt]) ├───┤ X ├────░──────────────────┤ X ├───»
             └────────────────┘   └─╥─┘    ░       ┌───┐      └─╥─┘   »
        q_1: ───────────────────────╫──────░───────┤ X ├────────╫─────»
                                    ║      ░ ┌─┐   └─╥─┘        ║     »
        q_2: ───────────────────────╫──────░─┤M├─────╫──────────╫─────»
                               ┌────╨────┐ ░ └╥┘┌────╨────┐┌────╨────┐»
        c: 1/══════════════════╡ c_0=0x1 ╞════╩═╡ c_0=0x0 ╞╡ c_0=0x0 ╞»
                               └─────────┘    0 └─────────┘└─────────┘»
        «     ┌────────────────┐┌───┐
        «q_0: ┤ Delay(300[dt]) ├┤ X ├─────■─────
        «     └────────────────┘└───┘   ┌─┴─┐
        «q_1: ────────■─────────────────┤ X ├───
        «           ┌─┴─┐        ┌─┐    └─╥─┘
        «q_2: ──────┤ X ├────────┤M├──────╫─────
        «           └───┘        └╥┘ ┌────╨────┐
        «c: 1/════════════════════╩══╡ c_0=0x0 ╞
        «                         0  └─────────┘

        (ASAP scheduled) duration = 2800 dt
             ┌────────────────┐   ┌───┐    ░ ┌─────────────────┐           »
        q_0: ┤ Delay(200[dt]) ├───┤ X ├────░─┤ Delay(1400[dt]) ├───────────»
             ├────────────────┤   └─╥─┘    ░ ├─────────────────┤   ┌───┐   »
        q_1: ┤ Delay(300[dt]) ├─────╫──────░─┤ Delay(1200[dt]) ├───┤ X ├───»
             ├────────────────┤     ║      ░ └───────┬─┬───────┘   └─╥─┘   »
        q_2: ┤ Delay(300[dt]) ├─────╫──────░─────────┤M├─────────────╫─────»
             └────────────────┘┌────╨────┐ ░         └╥┘        ┌────╨────┐»
        c: 1/══════════════════╡ c_0=0x1 ╞════════════╩═════════╡ c_0=0x0 ╞»
                               └─────────┘            0         └─────────┘»
        «                          ┌───┐   ┌────────────────┐      ┌───┐       »
        «q_0: ─────────────────────┤ X ├───┤ Delay(300[dt]) ├──────┤ X ├───────»
        «                          └─╥─┘   └────────────────┘┌─────┴───┴──────┐»
        «q_1: ───────────────────────╫─────────────■─────────┤ Delay(400[dt]) ├»
        «     ┌────────────────┐     ║           ┌─┴─┐       ├────────────────┤»
        «q_2: ┤ Delay(300[dt]) ├─────╫───────────┤ X ├───────┤ Delay(300[dt]) ├»
        «     └────────────────┘┌────╨────┐      └───┘       └────────────────┘»
        «c: 1/══════════════════╡ c_0=0x0 ╞════════════════════════════════════»
        «                       └─────────┘                                    »
        «                ┌────────────────┐
        «q_0: ─────■─────┤ Delay(700[dt]) ├
        «        ┌─┴─┐   ├────────────────┤
        «q_1: ───┤ X ├───┤ Delay(700[dt]) ├
        «        └─╥─┘   └──────┬─┬───────┘
        «q_2: ─────╫────────────┤M├────────
        «     ┌────╨────┐       └╥┘
        «c: 1/╡ c_0=0x0 ╞════════╩═════════
        «     └─────────┘        0

        (ALAP scheduled) duration = 3100
             ┌────────────────┐   ┌───┐    ░ ┌─────────────────┐           »
        q_0: ┤ Delay(200[dt]) ├───┤ X ├────░─┤ Delay(1400[dt]) ├───────────»
             ├────────────────┤   └─╥─┘    ░ ├─────────────────┤   ┌───┐   »
        q_1: ┤ Delay(300[dt]) ├─────╫──────░─┤ Delay(1200[dt]) ├───┤ X ├───»
             ├────────────────┤     ║      ░ └───────┬─┬───────┘   └─╥─┘   »
        q_2: ┤ Delay(300[dt]) ├─────╫──────░─────────┤M├─────────────╫─────»
             └────────────────┘┌────╨────┐ ░         └╥┘        ┌────╨────┐»
        c: 1/══════════════════╡ c_0=0x1 ╞════════════╩═════════╡ c_0=0x0 ╞»
                               └─────────┘            0         └─────────┘»
        «                          ┌───┐   ┌────────────────┐      ┌───┐       »
        «q_0: ─────────────────────┤ X ├───┤ Delay(300[dt]) ├──────┤ X ├───────»
        «     ┌────────────────┐   └─╥─┘   └────────────────┘┌─────┴───┴──────┐»
        «q_1: ┤ Delay(300[dt]) ├─────╫─────────────■─────────┤ Delay(100[dt]) ├»
        «     ├────────────────┤     ║           ┌─┴─┐       └──────┬─┬───────┘»
        «q_2: ┤ Delay(600[dt]) ├─────╫───────────┤ X ├──────────────┤M├────────»
        «     └────────────────┘┌────╨────┐      └───┘              └╥┘        »
        «c: 1/══════════════════╡ c_0=0x0 ╞══════════════════════════╩═════════»
        «                       └─────────┘                          0         »
        «                ┌────────────────┐
        «q_0: ─────■─────┤ Delay(700[dt]) ├
        «        ┌─┴─┐   ├────────────────┤
        «q_1: ───┤ X ├───┤ Delay(700[dt]) ├
        «        └─╥─┘   └────────────────┘
        «q_2: ─────╫───────────────────────
        «     ┌────╨────┐
        «c: 1/╡ c_0=0x0 ╞══════════════════
        «     └─────────┘

        """
        qc = QuantumCircuit(3, 1)
        qc.delay(100, 0)
        with self.assertWarns(DeprecationWarning):
            qc.x(0).c_if(0, 1)
        qc.barrier()
        qc.measure(2, 0)
        with self.assertWarns(DeprecationWarning):
            qc.x(1).c_if(0, 0)
        with self.assertWarns(DeprecationWarning):
            qc.x(0).c_if(0, 0)
        qc.delay(300, 0)
        qc.cx(1, 2)
        qc.x(0)
        with self.assertWarns(DeprecationWarning):
            qc.cx(0, 1).c_if(0, 0)
        qc.measure(2, 0)

        durations = InstructionDurations(
            [("x", None, 100), ("measure", None, 1000), ("cx", None, 200)]
        )

        actual_asap = PassManager(
            [
                SetIOLatency(clbit_write_latency=100, conditional_latency=200),
                ASAPScheduleAnalysis(durations),
                PadDelay(),
            ]
        ).run(qc)

        actual_alap = PassManager(
            [
                SetIOLatency(clbit_write_latency=100, conditional_latency=200),
                ALAPScheduleAnalysis(durations),
                PadDelay(),
            ]
        ).run(qc)

        expected_asap = QuantumCircuit(3, 1)
        expected_asap.delay(200, 0)  # due to conditional latency of 200dt
        expected_asap.delay(300, 1)
        expected_asap.delay(300, 2)
        with self.assertWarns(DeprecationWarning):
            expected_asap.x(0).c_if(0, 1)
        expected_asap.barrier()
        expected_asap.delay(1400, 0)
        expected_asap.delay(1200, 1)
        expected_asap.measure(2, 0)
        with self.assertWarns(DeprecationWarning):
            expected_asap.x(1).c_if(0, 0)
        with self.assertWarns(DeprecationWarning):
            expected_asap.x(0).c_if(0, 0)
        expected_asap.delay(300, 0)
        expected_asap.x(0)
        expected_asap.delay(300, 2)
        expected_asap.cx(1, 2)
        expected_asap.delay(400, 1)
        with self.assertWarns(DeprecationWarning):
            expected_asap.cx(0, 1).c_if(0, 0)
        expected_asap.delay(700, 0)  # creg is released at t0 of cx(0,1).c_if(0,0)
        expected_asap.delay(
            700, 1
        )  # no creg write until 100dt. thus measure can move left by 300dt.
        expected_asap.delay(300, 2)
        expected_asap.measure(2, 0)
        self.assertEqual(expected_asap, actual_asap)
        self.assertEqual(actual_asap.duration, 3100)

        expected_alap = QuantumCircuit(3, 1)
        expected_alap.delay(200, 0)  # due to conditional latency of 200dt
        expected_alap.delay(300, 1)
        expected_alap.delay(300, 2)
        with self.assertWarns(DeprecationWarning):
            expected_alap.x(0).c_if(0, 1)
        expected_alap.barrier()
        expected_alap.delay(1400, 0)
        expected_alap.delay(1200, 1)
        expected_alap.measure(2, 0)
        with self.assertWarns(DeprecationWarning):
            expected_alap.x(1).c_if(0, 0)
        with self.assertWarns(DeprecationWarning):
            expected_alap.x(0).c_if(0, 0)
        expected_alap.delay(300, 0)
        expected_alap.x(0)
        expected_alap.delay(300, 1)
        expected_alap.delay(600, 2)
        expected_alap.cx(1, 2)
        expected_alap.delay(100, 1)
        with self.assertWarns(DeprecationWarning):
            expected_alap.cx(0, 1).c_if(0, 0)
        expected_alap.measure(2, 0)
        expected_alap.delay(700, 0)
        expected_alap.delay(700, 1)
        self.assertEqual(expected_alap, actual_alap)
        self.assertEqual(actual_alap.duration, 3100)

    def test_dag_introduces_extra_dependency_between_conditionals(self):
        """Test dependency between conditional operations in the scheduling.

        In the below example circuit, the conditional x on q1 could start at time 0,
        however it must be scheduled after the conditional x on q0 in ASAP scheduling.
        That is because circuit model used in the transpiler passes (DAGCircuit)
        interprets instructions acting on common clbits must be run in the order
        given by the original circuit (QuantumCircuit).

        (input)
             ┌────────────────┐   ┌───┐
        q_0: ┤ Delay(100[dt]) ├───┤ X ├───
             └─────┬───┬──────┘   └─╥─┘
        q_1: ──────┤ X ├────────────╫─────
                   └─╥─┘            ║
                ┌────╨────┐    ┌────╨────┐
        c: 1/═══╡ c_0=0x1 ╞════╡ c_0=0x1 ╞
                └─────────┘    └─────────┘

        (ASAP scheduled)
             ┌────────────────┐   ┌───┐
        q_0: ┤ Delay(100[dt]) ├───┤ X ├──────────────
             ├────────────────┤   └─╥─┘      ┌───┐
        q_1: ┤ Delay(100[dt]) ├─────╫────────┤ X ├───
             └────────────────┘     ║        └─╥─┘
                               ┌────╨────┐┌────╨────┐
        c: 1/══════════════════╡ c_0=0x1 ╞╡ c_0=0x1 ╞
                               └─────────┘└─────────┘
        """
        qc = QuantumCircuit(2, 1)
        qc.delay(100, 0)
        with self.assertWarns(DeprecationWarning):
            qc.x(0).c_if(0, True)
        with self.assertWarns(DeprecationWarning):
            qc.x(1).c_if(0, True)

        durations = InstructionDurations([("x", None, 160)])
        pm = PassManager([ASAPScheduleAnalysis(durations), PadDelay()])
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.delay(100, 0)
        expected.delay(100, 1)  # due to extra dependency on clbits
        with self.assertWarns(DeprecationWarning):
            expected.x(0).c_if(0, True)
        with self.assertWarns(DeprecationWarning):
            expected.x(1).c_if(0, True)

        self.assertEqual(expected, scheduled)

    def test_scheduling_with_calibration(self):
        """Test if calibrated instruction can update node duration."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cx(0, 1)
        qc.x(1)
        qc.cx(0, 1)

        with self.assertWarns(DeprecationWarning):
            xsched = Schedule(Play(Constant(300, 0.1), DriveChannel(0)))
            qc.add_calibration("x", (0,), xsched)

        durations = InstructionDurations([("x", None, 160), ("cx", None, 600)])
        pm = PassManager([ASAPScheduleAnalysis(durations), PadDelay()])
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2)
        expected.x(0)
        expected.delay(300, 1)
        expected.cx(0, 1)
        expected.x(1)
        expected.delay(160, 0)
        expected.cx(0, 1)
        with self.assertWarns(DeprecationWarning):
            expected.add_calibration("x", (0,), xsched)

        self.assertEqual(expected, scheduled)

    def test_padding_not_working_without_scheduling(self):
        """Test padding fails when un-scheduled DAG is input."""
        qc = QuantumCircuit(1, 1)
        qc.delay(100, 0)
        qc.x(0)
        qc.measure(0, 0)

        with self.assertRaises(TranspilerError):
            PassManager(PadDelay()).run(qc)

    def test_no_pad_very_end_of_circuit(self):
        """Test padding option that inserts no delay at the very end of circuit.

        This circuit will be unchanged after ASAP-schedule/padding.

             ┌────────────────┐┌─┐
        q_0: ┤ Delay(100[dt]) ├┤M├
             └─────┬───┬──────┘└╥┘
        q_1: ──────┤ X ├────────╫─
                   └───┘        ║
        c: 1/═══════════════════╩═
                                0
        """
        qc = QuantumCircuit(2, 1)
        qc.delay(100, 0)
        qc.x(1)
        qc.measure(0, 0)

        durations = InstructionDurations([("x", None, 160), ("measure", None, 1000)])

        scheduled = PassManager(
            [
                ASAPScheduleAnalysis(durations),
                PadDelay(fill_very_end=False),
            ]
        ).run(qc)

        self.assertEqual(scheduled, qc)

    @data(ALAPScheduleAnalysis, ASAPScheduleAnalysis)
    def test_respect_target_instruction_constraints(self, schedule_pass):
        """Test if DD pass does not pad delays for qubits that do not support delay instructions.
        See: https://github.com/Qiskit/qiskit-terra/issues/9993
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 2)

        target = Target(dt=1)
        target.add_instruction(CXGate(), {(1, 2): InstructionProperties(duration=1000)})
        # delays are not supported

        pm = PassManager([schedule_pass(target=target), PadDelay(target=target)])
        scheduled = pm.run(qc)

        self.assertEqual(qc, scheduled)


if __name__ == "__main__":
    unittest.main()

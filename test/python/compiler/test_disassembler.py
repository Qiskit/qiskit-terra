# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Assembler Test.
FULLY REMOVE ONCE Qobj, assemble AND disassemble ARE REMOVED.
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from qiskit import pulse
from qiskit.compiler.assembler import assemble
from qiskit.assembler.disassemble import disassemble
from qiskit.assembler.run_config import RunConfig
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter

from qiskit.circuit.library import RXGate, Isometry
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.providers.fake_provider import FakeOpenPulse2Q
import qiskit.quantum_info as qi
from test import QiskitTestCase  # pylint: disable=wrong-import-order


def _parametric_to_waveforms(schedule):
    instructions = list(schedule.instructions)
    for i, time_instruction_tuple in enumerate(schedule.instructions):
        time, instruction = time_instruction_tuple
        if not isinstance(instruction.pulse, pulse.library.Waveform):
            new_inst = pulse.Play(instruction.pulse.get_waveform(), instruction.channel)
            instructions[i] = (time, new_inst)
    return tuple(instructions)


class TestQuantumCircuitDisassembler(QiskitTestCase):
    """Tests for disassembling circuits to qobj."""

    def test_disassemble_single_circuit(self):
        """Test disassembling a single circuit."""
        qr = QuantumRegister(2, name="q")
        cr = ClassicalRegister(2, name="c")
        circ = QuantumCircuit(qr, cr, name="circ")
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, cr)

        qubit_lo_freq = [5e9, 5e9]
        meas_lo_freq = [6.7e9, 6.7e9]
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                circ,
                shots=2000,
                memory=True,
                qubit_lo_freq=qubit_lo_freq,
                meas_lo_freq=meas_lo_freq,
            )
            circuits, run_config_out, headers = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(run_config_out.shots, 2000)
        self.assertEqual(run_config_out.memory, True)
        self.assertEqual(run_config_out.qubit_lo_freq, qubit_lo_freq)
        self.assertEqual(run_config_out.meas_lo_freq, meas_lo_freq)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, headers)

    def test_disassemble_multiple_circuits(self):
        """Test disassembling multiple circuits, all should have the same config."""
        qr0 = QuantumRegister(2, name="q0")
        qc0 = ClassicalRegister(2, name="c0")
        circ0 = QuantumCircuit(qr0, qc0, name="circ0")
        circ0.h(qr0[0])
        circ0.cx(qr0[0], qr0[1])
        circ0.measure(qr0, qc0)

        qr1 = QuantumRegister(3, name="q1")
        qc1 = ClassicalRegister(3, name="c1")
        circ1 = QuantumCircuit(qr1, qc1, name="circ0")
        circ1.h(qr1[0])
        circ1.cx(qr1[0], qr1[1])
        circ1.cx(qr1[0], qr1[2])
        circ1.measure(qr1, qc1)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble([circ0, circ1], shots=100, memory=False, seed=6)
            circuits, run_config_out, headers = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 3)
        self.assertEqual(run_config_out.memory_slots, 3)
        self.assertEqual(run_config_out.shots, 100)
        self.assertEqual(run_config_out.memory, False)
        self.assertEqual(run_config_out.seed, 6)
        self.assertEqual(len(circuits), 2)
        for circuit in circuits:
            self.assertIn(circuit, [circ0, circ1])
        self.assertEqual({}, headers)

    def test_disassemble_no_run_config(self):
        """Test disassembling with no run_config, relying on default."""
        qr = QuantumRegister(2, name="q")
        qc = ClassicalRegister(2, name="c")
        circ = QuantumCircuit(qr, qc, name="circ")
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, qc)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(circ)
            circuits, run_config_out, headers = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, headers)

    def test_disassemble_initialize(self):
        """Test disassembling a circuit with an initialize."""
        q = QuantumRegister(2, name="q")
        circ = QuantumCircuit(q, name="circ")
        circ.initialize([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], q[:])

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(circ)
            circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 0)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, header)

    def test_disassemble_isometry(self):
        """Test disassembling a circuit with an isometry."""
        q = QuantumRegister(2, name="q")
        circ = QuantumCircuit(q, name="circ")
        circ.append(Isometry(qi.random_unitary(4).data, 0, 0), circ.qubits)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(circ)
            circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 0)
        self.assertEqual(len(circuits), 1)
        # params array
        assert_allclose(circuits[0]._data[0].operation.params[0], circ._data[0].operation.params[0])
        # all other data
        self.assertEqual(
            circuits[0]._data[0].operation.params[1:], circ._data[0].operation.params[1:]
        )
        self.assertEqual(circuits[0]._data[0].qubits, circ._data[0].qubits)
        self.assertEqual(circuits[0]._data[0].clbits, circ._data[0].clbits)
        self.assertEqual(circuits[0]._data[1:], circ._data[1:])
        self.assertEqual({}, header)

    def test_opaque_instruction(self):
        """Test the disassembler handles opaque instructions correctly."""
        opaque_inst = Instruction(name="my_inst", num_qubits=4, num_clbits=2, params=[0.5, 0.4])
        q = QuantumRegister(6, name="q")
        c = ClassicalRegister(4, name="c")
        circ = QuantumCircuit(q, c, name="circ")
        circ.append(opaque_inst, [q[0], q[2], q[5], q[3]], [c[3], c[0]])
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(circ)
            circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 6)
        self.assertEqual(run_config_out.memory_slots, 4)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, header)

    def test_circuit_with_conditionals(self):
        """Verify disassemble sets conditionals correctly."""
        qr = QuantumRegister(2)
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr1, cr2)
        qc.measure(qr[0], cr1)  # Measure not required for a later conditional
        qc.measure(qr[1], cr2[1])  # Measure required for a later conditional
        with self.assertWarns(DeprecationWarning):
            qc.h(qr[1]).c_if(cr2, 3)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(qc)
            circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 3)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)

    def test_circuit_with_simple_conditional(self):
        """Verify disassemble handles a simple conditional on the only bits."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            qc.h(qr[0]).c_if(cr, 1)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(qc)
            circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 1)
        self.assertEqual(run_config_out.memory_slots, 1)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)

    def test_circuit_with_single_bit_conditions(self):
        """Verify disassemble handles a simple conditional on a single bit of a register."""
        # This circuit would fail to perfectly round-trip if 'cr' below had only one bit in it.
        # This is because the format of QasmQobj is insufficient to disambiguate single-bit
        # conditions from conditions on registers with only one bit. Since single-bit conditions are
        # mostly a hack for the QasmQobj format at all, `disassemble` always prefers to return the
        # register if it can.  It would also fail if registers overlap.
        qr = QuantumRegister(1)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            qc.h(qr[0]).c_if(cr[0], 1)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(qc)
            circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, len(qr))
        self.assertEqual(run_config_out.memory_slots, len(cr))
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)

    def test_circuit_with_mcx(self):
        """Verify disassemble handles mcx gate - #6271."""
        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        qc = QuantumCircuit(qr, cr)
        qc.mcx([0, 1, 2], 4)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(qc)
            circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 5)
        self.assertEqual(run_config_out.memory_slots, 5)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)

    def test_multiple_conditionals_multiple_registers(self):
        """Verify disassemble handles multiple conditionals and registers."""
        qr = QuantumRegister(3)
        cr1 = ClassicalRegister(3)
        cr2 = ClassicalRegister(5)
        cr3 = ClassicalRegister(6)
        cr4 = ClassicalRegister(1)

        qc = QuantumCircuit(qr, cr1, cr2, cr3, cr4)
        qc.x(qr[1])
        qc.h(qr)
        with self.assertWarns(DeprecationWarning):
            qc.cx(qr[1], qr[0]).c_if(cr3, 14)
        with self.assertWarns(DeprecationWarning):
            qc.ccx(qr[0], qr[2], qr[1]).c_if(cr4, 1)
        with self.assertWarns(DeprecationWarning):
            qc.h(qr).c_if(cr1, 3)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(qc)
            circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 3)
        self.assertEqual(run_config_out.memory_slots, 15)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)

    def test_circuit_with_bit_conditional_1(self):
        """Verify disassemble handles conditional on a single bit."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        with self.assertWarns(DeprecationWarning):
            qc.h(qr[0]).c_if(cr[1], True)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(qc)
            circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)

    def test_circuit_with_bit_conditional_2(self):
        """Verify disassemble handles multiple single bit conditionals."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        cr1 = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr, cr1)
        with self.assertWarns(DeprecationWarning):
            qc.h(qr[0]).c_if(cr1[1], False)
        with self.assertWarns(DeprecationWarning):
            qc.h(qr[1]).c_if(cr[0], True)
        with self.assertWarns(DeprecationWarning):
            qc.cx(qr[0], qr[1]).c_if(cr1[0], False)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(qc)
            circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 4)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], qc)
        self.assertEqual({}, header)

    def assertCircuitCalibrationsEqual(self, in_circuits, out_circuits):
        """Verify circuit calibrations are equivalent pre-assembly and post-disassembly"""
        self.assertEqual(len(in_circuits), len(out_circuits))
        for in_qc, out_qc in zip(in_circuits, out_circuits):
            with self.assertWarns(DeprecationWarning):
                in_cals = in_qc.calibrations
                out_cals = out_qc.calibrations
            self.assertEqual(in_cals.keys(), out_cals.keys())
            for gate_name in in_cals:
                self.assertEqual(in_cals[gate_name].keys(), out_cals[gate_name].keys())
                for gate_params, in_sched in in_cals[gate_name].items():
                    out_sched = out_cals[gate_name][gate_params]
                    self.assertEqual(*map(_parametric_to_waveforms, (in_sched, out_sched)))

    def test_single_circuit_calibrations(self):
        """Test that disassembler parses single circuit QOBJ calibrations (from QOBJ-level)."""
        theta = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(np.pi, 0)
        qc.rx(theta, 1)
        qc = qc.assign_parameters({theta: np.pi})

        with self.assertWarns(DeprecationWarning):
            with pulse.build() as h_sched:
                pulse.play(pulse.library.Drag(1, 0.15, 4, 2), pulse.DriveChannel(0))

            with pulse.build() as x180:
                pulse.play(pulse.library.Gaussian(1, 0.2, 5), pulse.DriveChannel(0))

                qc.add_calibration("h", [0], h_sched)
                qc.add_calibration(RXGate(np.pi), [0], x180)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(qc, FakeOpenPulse2Q())
            output_circuits, _, _ = disassemble(qobj)

            self.assertCircuitCalibrationsEqual([qc], output_circuits)

    def test_parametric_pulse_circuit_calibrations(self):
        """Test that disassembler parses parametric pulses back to pulse gates."""
        with self.assertWarns(DeprecationWarning):
            with pulse.build() as h_sched:
                pulse.play(pulse.library.Drag(50, 0.15, 4, 2), pulse.DriveChannel(0))

        qc = QuantumCircuit(2)
        qc.h(0)

        with self.assertWarns(DeprecationWarning):
            qc.add_calibration("h", [0], h_sched)
            backend = FakeOpenPulse2Q()
            backend.configuration().parametric_pulses = ["drag"]

            qobj = assemble(qc, backend)
            output_circuits, _, _ = disassemble(qobj)
        out_qc = output_circuits[0]

        with self.assertWarns(DeprecationWarning):
            self.assertCircuitCalibrationsEqual([qc], output_circuits)
            self.assertTrue(
                all(
                    qc_sched.instructions == out_qc_sched.instructions
                    for (_, qc_gate), (_, out_qc_gate) in zip(
                        qc.calibrations.items(), out_qc.calibrations.items()
                    )
                    for qc_sched, out_qc_sched in zip(qc_gate.values(), out_qc_gate.values())
                ),
            )

    def test_multi_circuit_uncommon_calibrations(self):
        """Test that disassembler parses uncommon calibrations (stored at QOBJ experiment-level)."""
        with self.assertWarns(DeprecationWarning):
            with pulse.build() as sched:
                pulse.play(pulse.library.Drag(50, 0.15, 4, 2), pulse.DriveChannel(0))

        qc_0 = QuantumCircuit(2)
        qc_0.h(0)
        qc_0.append(RXGate(np.pi), [1])
        with self.assertWarns(DeprecationWarning):
            qc_0.add_calibration("h", [0], sched)
            qc_0.add_calibration(RXGate(np.pi), [1], sched)

        qc_1 = QuantumCircuit(2)
        qc_1.h(0)

        circuits = [qc_0, qc_1]
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(circuits, FakeOpenPulse2Q())
            output_circuits, _, _ = disassemble(qobj)

            self.assertCircuitCalibrationsEqual(circuits, output_circuits)

    def test_multi_circuit_common_calibrations(self):
        """Test that disassembler parses common calibrations (stored at QOBJ-level)."""
        with self.assertWarns(DeprecationWarning):
            with pulse.build() as sched:
                pulse.play(pulse.library.Drag(1, 0.15, 4, 2), pulse.DriveChannel(0))

        qc_0 = QuantumCircuit(2)
        qc_0.h(0)
        qc_0.append(RXGate(np.pi), [1])
        with self.assertWarns(DeprecationWarning):
            qc_0.add_calibration("h", [0], sched)
            qc_0.add_calibration(RXGate(np.pi), [1], sched)

        qc_1 = QuantumCircuit(2)
        qc_1.h(0)
        with self.assertWarns(DeprecationWarning):
            qc_1.add_calibration(RXGate(np.pi), [1], sched)

        circuits = [qc_0, qc_1]
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(circuits, FakeOpenPulse2Q())
            output_circuits, _, _ = disassemble(qobj)

            self.assertCircuitCalibrationsEqual(circuits, output_circuits)

    def test_single_circuit_delay_calibrations(self):
        """Test that disassembler parses delay instruction back to delay gate."""
        qc = QuantumCircuit(2)
        qc.append(Gate("test", 1, []), [0])
        with self.assertWarns(DeprecationWarning):
            test_sched = pulse.Delay(64, pulse.DriveChannel(0)) + pulse.Delay(
                160, pulse.DriveChannel(0)
            )

            qc.add_calibration("test", [0], test_sched)

            qobj = assemble(qc, FakeOpenPulse2Q())
            output_circuits, _, _ = disassemble(qobj)

            self.assertEqual(len(qc.calibrations), len(output_circuits[0].calibrations))
            self.assertEqual(qc.calibrations.keys(), output_circuits[0].calibrations.keys())
            self.assertTrue(
                all(
                    qc_cal.keys() == out_qc_cal.keys()
                    for qc_cal, out_qc_cal in zip(
                        qc.calibrations.values(), output_circuits[0].calibrations.values()
                    )
                )
            )
            self.assertEqual(
                qc.calibrations["test"][((0,), ())],
                output_circuits[0].calibrations["test"][((0,), ())],
            )


class TestPulseScheduleDisassembler(QiskitTestCase):
    """Tests for disassembling pulse schedules to qobj."""

    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            self.backend = FakeOpenPulse2Q()
        self.backend_config = self.backend.configuration()
        self.backend_config.parametric_pulses = ["constant", "gaussian", "gaussian_square", "drag"]

    def test_disassemble_single_schedule(self):
        """Test disassembling a single schedule."""
        with self.assertWarns(DeprecationWarning):
            d0 = pulse.DriveChannel(0)
            d1 = pulse.DriveChannel(1)
            with pulse.build(self.backend) as sched:
                with pulse.align_right():
                    pulse.play(pulse.library.Constant(10, 1.0), d0)
                    pulse.set_phase(1.0, d0)
                    pulse.shift_phase(3.11, d0)
                    pulse.set_frequency(1e9, d0)
                    pulse.shift_frequency(1e7, d0)
                    pulse.delay(20, d0)
                    pulse.delay(10, d1)
                    pulse.play(pulse.library.Constant(8, 0.1), d1)
                    pulse.measure_all()

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(sched, backend=self.backend, shots=2000)
            scheds, run_config_out, _ = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(run_config_out.shots, 2000)
        self.assertEqual(run_config_out.memory, False)
        self.assertEqual(run_config_out.meas_level, 2)
        self.assertEqual(run_config_out.meas_lo_freq, self.backend.defaults().meas_freq_est)
        self.assertEqual(run_config_out.qubit_lo_freq, self.backend.defaults().qubit_freq_est)
        self.assertEqual(run_config_out.rep_time, 99)
        self.assertEqual(len(scheds), 1)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(scheds[0], target_qobj_transform(sched))

    def test_disassemble_multiple_schedules(self):
        """Test disassembling multiple schedules, all should have the same config."""
        with self.assertWarns(DeprecationWarning):
            d0 = pulse.DriveChannel(0)
            d1 = pulse.DriveChannel(1)
            with pulse.build(self.backend) as sched0:
                with pulse.align_right():
                    pulse.play(pulse.library.Constant(10, 1.0), d0)
                    pulse.set_phase(1.0, d0)
                    pulse.shift_phase(3.11, d0)
                    pulse.set_frequency(1e9, d0)
                    pulse.shift_frequency(1e7, d0)
                    pulse.delay(20, d0)
                    pulse.delay(10, d1)
                    pulse.play(pulse.library.Constant(8, 0.1), d1)
                    pulse.measure_all()

            with pulse.build(self.backend) as sched1:
                with pulse.align_right():
                    pulse.play(pulse.library.Constant(8, 0.1), d0)
                    pulse.play(pulse.library.Waveform([0.0, 1.0]), d1)
                    pulse.set_phase(1.1, d0)
                    pulse.shift_phase(3.5, d0)
                    pulse.set_frequency(2e9, d0)
                    pulse.shift_frequency(3e7, d1)
                    pulse.delay(20, d1)
                    pulse.delay(10, d0)
                    pulse.play(pulse.library.Constant(8, 0.4), d1)
                    pulse.measure_all()

        with self.assertWarns(DeprecationWarning):
            qobj = assemble([sched0, sched1], backend=self.backend, shots=2000)
            scheds, run_config_out, _ = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(run_config_out.shots, 2000)
        self.assertEqual(run_config_out.memory, False)
        self.assertEqual(len(scheds), 2)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(scheds[0], target_qobj_transform(sched0))
            self.assertEqual(scheds[1], target_qobj_transform(sched1))

    def test_disassemble_parametric_pulses(self):
        """Test disassembling multiple schedules all should have the same config."""
        with self.assertWarns(DeprecationWarning):
            d0 = pulse.DriveChannel(0)
            with pulse.build(self.backend) as sched:
                with pulse.align_right():
                    pulse.play(pulse.library.Constant(10, 1.0), d0)
                    pulse.play(pulse.library.Gaussian(10, 1.0, 2.0), d0)
                    pulse.play(pulse.library.GaussianSquare(10, 1.0, 2.0, 3), d0)
                    pulse.play(pulse.library.Drag(10, 1.0, 2.0, 0.1), d0)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(sched, backend=self.backend, shots=2000)
            scheds, _, _ = disassemble(qobj)
            self.assertEqual(scheds[0], target_qobj_transform(sched))

    def test_disassemble_schedule_los(self):
        """Test disassembling schedule los."""
        with self.assertWarns(DeprecationWarning):
            d0 = pulse.DriveChannel(0)
            m0 = pulse.MeasureChannel(0)
            d1 = pulse.DriveChannel(1)
            m1 = pulse.MeasureChannel(1)

            sched0 = pulse.Schedule()
            sched1 = pulse.Schedule()

        schedule_los = [
            {d0: 4.5e9, d1: 5e9, m0: 6e9, m1: 7e9},
            {d0: 5e9, d1: 4.5e9, m0: 7e9, m1: 6e9},
        ]
        with self.assertWarns(DeprecationWarning):
            qobj = assemble([sched0, sched1], backend=self.backend, schedule_los=schedule_los)
            _, run_config_out, _ = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)

        self.assertEqual(run_config_out.schedule_los, schedule_los)


if __name__ == "__main__":
    unittest.main(verbosity=2)

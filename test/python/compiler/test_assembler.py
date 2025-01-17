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

"""Assembler Test."""

import unittest
import io
from logging import StreamHandler, getLogger
import sys
import copy
import numpy as np

from qiskit import pulse
from qiskit.circuit import Instruction, Gate, Parameter, ParameterVector
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler.assembler import assemble
from qiskit.exceptions import QiskitError
from qiskit.pulse import Schedule, Acquire, Play
from qiskit.pulse.channels import MemorySlot, AcquireChannel, DriveChannel, MeasureChannel
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.qobj import QasmQobj, PulseQobj
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.pulse.macros import measure
from qiskit.providers.fake_provider import (
    FakeOpenPulse2Q,
    FakeOpenPulse3Q,
    Fake5QV1,
    Fake27QPulseV1,
)
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class RxGate(Gate):
    """Used to test custom gate assembly.

    Useful for testing pulse gates with parameters, as well.
    Note: Parallel maps (e.g., in assemble_circuits) pickle their input,
          so circuit features have to be defined top level.
    """

    def __init__(self, theta):
        super().__init__("rxtheta", 1, [theta])


class TestCircuitAssembler(QiskitTestCase):
    """Tests for assembling circuits to qobj."""

    def setUp(self):
        super().setUp()
        qr = QuantumRegister(2, name="q")
        cr = ClassicalRegister(2, name="c")
        self.circ = QuantumCircuit(qr, cr, name="circ")
        self.circ.h(qr[0])
        self.circ.cx(qr[0], qr[1])
        self.circ.measure(qr, cr)

        with self.assertWarns(DeprecationWarning):
            self.backend = Fake5QV1()
        self.backend_config = self.backend.configuration()
        self.num_qubits = self.backend_config.n_qubits

        # lo test values
        self.default_qubit_lo_freq = [5e9 for _ in range(self.num_qubits)]
        self.default_meas_lo_freq = [6.7e9 for _ in range(self.num_qubits)]
        with self.assertWarns(DeprecationWarning):
            self.user_lo_config_dict = {
                pulse.DriveChannel(0): 5.55e9,
                pulse.MeasureChannel(0): 6.64e9,
                pulse.DriveChannel(3): 4.91e9,
                pulse.MeasureChannel(4): 6.1e9,
            }
        self.user_lo_config = pulse.LoConfig(self.user_lo_config_dict)

    def test_assemble_single_circuit(self):
        """Test assembling a single circuit."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ, shots=2000, memory=True)

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.shots, 2000)
        self.assertEqual(qobj.config.memory, True)
        self.assertEqual(len(qobj.experiments), 1)
        self.assertEqual(qobj.experiments[0].instructions[1].name, "cx")

    def test_assemble_multiple_circuits(self):
        """Test assembling multiple circuits, all should have the same config."""
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
            qobj = assemble([circ0, circ1], shots=100, memory=False, seed_simulator=6)

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.seed_simulator, 6)
        self.assertEqual(len(qobj.experiments), 2)
        self.assertEqual(qobj.experiments[1].config.n_qubits, 3)
        self.assertEqual(len(qobj.experiments), 2)
        self.assertEqual(len(qobj.experiments[1].instructions), 6)

    def test_assemble_no_run_config(self):
        """Test assembling with no run_config, relying on default."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ)

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.shots, 1024)

    def test_shots_greater_than_max_shots(self):
        """Test assembling with shots greater than max shots"""
        with self.assertWarns(DeprecationWarning):
            self.assertRaises(QiskitError, assemble, self.backend, shots=1024000)

    def test_shots_not_of_type_int(self):
        """Test assembling with shots having type other than int"""
        with self.assertWarns(DeprecationWarning):
            self.assertRaises(QiskitError, assemble, self.backend, shots="1024")

    def test_shots_of_type_numpy_int64(self):
        """Test assembling with shots having type numpy.int64"""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ, shots=np.int64(2048))
        self.assertEqual(qobj.config.shots, 2048)

    def test_default_shots_greater_than_max_shots(self):
        """Test assembling with default shots greater than max shots"""
        self.backend_config.max_shots = 5

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ, self.backend)

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.shots, 5)

    def test_assemble_initialize(self):
        """Test assembling a circuit with an initialize."""
        q = QuantumRegister(2, name="q")
        circ = QuantumCircuit(q, name="circ")
        circ.initialize([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], q[:])

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(circ)

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.experiments[0].instructions[0].name, "initialize")
        np.testing.assert_almost_equal(
            qobj.experiments[0].instructions[0].params, [0.7071067811865, 0, 0, 0.707106781186]
        )

    def test_assemble_meas_level_meas_return(self):
        """Test assembling a circuit schedule with `meas_level`."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ, meas_level=1, meas_return="single")

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.meas_level, 1)
        self.assertEqual(qobj.config.meas_return, "single")

        # no meas_level set
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ)

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.meas_level, 2)
        self.assertEqual(hasattr(qobj.config, "meas_return"), False)

    def test_assemble_backend_rep_delays(self):
        """Check that rep_delay is properly set from backend values."""
        rep_delay_range = [2.5e-3, 4.5e-3]  # sec
        default_rep_delay = 3.0e-3
        setattr(self.backend_config, "rep_delay_range", rep_delay_range)
        setattr(self.backend_config, "default_rep_delay", default_rep_delay)

        # dynamic rep rates off
        setattr(self.backend_config, "dynamic_reprate_enabled", False)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ, self.backend)
        self.assertEqual(hasattr(qobj.config, "rep_delay"), False)

        # dynamic rep rates on
        setattr(self.backend_config, "dynamic_reprate_enabled", True)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ, self.backend)
        self.assertEqual(qobj.config.rep_delay, default_rep_delay * 1e6)

    def test_assemble_user_rep_time_delay(self):
        """Check that user runtime config rep_delay works."""
        # set custom rep_delay in runtime config
        rep_delay = 2.2e-6
        rep_delay_range = [0, 3e-6]  # sec
        setattr(self.backend_config, "rep_delay_range", rep_delay_range)

        # dynamic rep rates off (no default so shouldn't be in qobj config)
        setattr(self.backend_config, "dynamic_reprate_enabled", False)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ, self.backend, rep_delay=rep_delay)
        self.assertEqual(hasattr(qobj.config, "rep_delay"), False)

        # turn on dynamic rep rates, rep_delay should be set
        setattr(self.backend_config, "dynamic_reprate_enabled", True)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ, self.backend, rep_delay=rep_delay)
        self.assertEqual(qobj.config.rep_delay, 2.2)

        # test ``rep_delay=0``
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ, self.backend, rep_delay=0)
        self.assertEqual(qobj.config.rep_delay, 0)

        # use ``rep_delay`` outside of ``rep_delay_range```
        rep_delay_large = 5.0e-6
        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(QiskitError):
                assemble(self.circ, self.backend, rep_delay=rep_delay_large)

    def test_assemble_opaque_inst(self):
        """Test opaque instruction is assembled as-is"""
        opaque_inst = Instruction(name="my_inst", num_qubits=4, num_clbits=2, params=[0.5, 0.4])
        q = QuantumRegister(6, name="q")
        c = ClassicalRegister(4, name="c")
        circ = QuantumCircuit(q, c, name="circ")
        circ.append(opaque_inst, [q[0], q[2], q[5], q[3]], [c[3], c[0]])

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(circ)

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(len(qobj.experiments[0].instructions), 1)
        self.assertEqual(qobj.experiments[0].instructions[0].name, "my_inst")
        self.assertEqual(qobj.experiments[0].instructions[0].qubits, [0, 2, 5, 3])
        self.assertEqual(qobj.experiments[0].instructions[0].memory, [3, 0])
        self.assertEqual(qobj.experiments[0].instructions[0].params, [0.5, 0.4])

    def test_assemble_unroll_parametervector(self):
        """Verify that assemble unrolls parametervectors ref #5467"""
        pv1 = ParameterVector("pv1", 3)
        pv2 = ParameterVector("pv2", 3)
        qc = QuantumCircuit(2, 2)
        for i in range(3):
            qc.rx(pv1[i], 0)
            qc.ry(pv2[i], 1)
        qc.barrier()
        qc.measure([0, 1], [0, 1])

        qc.assign_parameters({pv1: [0.1, 0.2, 0.3], pv2: [0.4, 0.5, 0.6]})

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(qc, parameter_binds=[{pv1: [0.1, 0.2, 0.3], pv2: [0.4, 0.5, 0.6]}])

        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.experiments[0].instructions[0].params[0], 0.100000000000000)
        self.assertEqual(qobj.experiments[0].instructions[1].params[0], 0.400000000000000)
        self.assertEqual(qobj.experiments[0].instructions[2].params[0], 0.200000000000000)
        self.assertEqual(qobj.experiments[0].instructions[3].params[0], 0.500000000000000)
        self.assertEqual(qobj.experiments[0].instructions[4].params[0], 0.300000000000000)
        self.assertEqual(qobj.experiments[0].instructions[5].params[0], 0.600000000000000)

    def test_measure_to_registers_when_conditionals(self):
        """Verify assemble_circuits maps all measure ops on to a register slot
        for a circuit containing conditionals."""
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

        first_measure, second_measure = (
            op for op in qobj.experiments[0].instructions if op.name == "measure"
        )

        self.assertTrue(hasattr(first_measure, "register"))
        self.assertEqual(first_measure.register, first_measure.memory)
        self.assertTrue(hasattr(second_measure, "register"))
        self.assertEqual(second_measure.register, second_measure.memory)

    def test_convert_to_bfunc_plus_conditional(self):
        """Verify assemble_circuits converts conditionals from QASM to Qobj."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)

        with self.assertWarns(DeprecationWarning):
            qc.h(qr[0]).c_if(cr, 1)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(qc)

        bfunc_op, h_op = qobj.experiments[0].instructions

        self.assertEqual(bfunc_op.name, "bfunc")
        self.assertEqual(bfunc_op.mask, "0x1")
        self.assertEqual(bfunc_op.val, "0x1")
        self.assertEqual(bfunc_op.relation, "==")

        self.assertTrue(hasattr(h_op, "conditional"))
        self.assertEqual(bfunc_op.register, h_op.conditional)

    def test_convert_to_bfunc_plus_conditional_onebit(self):
        """Verify assemble_circuits converts single bit conditionals from QASM to Qobj."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(3)
        qc = QuantumCircuit(qr, cr)

        with self.assertWarns(DeprecationWarning):
            qc.h(qr[0]).c_if(cr[2], 1)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(qc)

        inst_set = qobj.experiments[0].instructions
        [bfunc_op, h_op] = inst_set

        self.assertEqual(len(inst_set), 2)
        self.assertEqual(bfunc_op.name, "bfunc")
        self.assertEqual(bfunc_op.mask, "0x4")
        self.assertEqual(bfunc_op.val, "0x4")
        self.assertEqual(bfunc_op.relation, "==")

        self.assertTrue(hasattr(h_op, "conditional"))
        self.assertEqual(bfunc_op.register, h_op.conditional)

    def test_resize_value_to_register(self):
        """Verify assemble_circuits converts the value provided on the classical
        creg to its mapped location on the device register."""
        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2)
        cr2 = ClassicalRegister(2)
        cr3 = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr1, cr2, cr3)

        with self.assertWarns(DeprecationWarning):
            qc.h(qr[0]).c_if(cr2, 2)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(qc)

        bfunc_op, h_op = qobj.experiments[0].instructions

        self.assertEqual(bfunc_op.name, "bfunc")
        self.assertEqual(bfunc_op.mask, "0xC")
        self.assertEqual(bfunc_op.val, "0x8")
        self.assertEqual(bfunc_op.relation, "==")

        self.assertTrue(hasattr(h_op, "conditional"))
        self.assertEqual(bfunc_op.register, h_op.conditional)

    def test_assemble_circuits_raises_for_bind_circuit_mismatch(self):
        """Verify assemble_circuits raises error for parameterized circuits without matching
        binds."""
        qr = QuantumRegister(2)
        x = Parameter("x")
        y = Parameter("y")

        full_bound_circ = QuantumCircuit(qr)
        full_param_circ = QuantumCircuit(qr)
        partial_param_circ = QuantumCircuit(qr)

        partial_param_circ.p(x, qr[0])

        full_param_circ.p(x, qr[0])
        full_param_circ.p(y, qr[1])

        partial_bind_args = {"parameter_binds": [{x: 1}, {x: 0}]}
        full_bind_args = {"parameter_binds": [{x: 1, y: 1}, {x: 0, y: 0}]}
        inconsistent_bind_args = {"parameter_binds": [{x: 1}, {x: 0, y: 0}]}

        with self.assertWarns(DeprecationWarning):
            # Raise when parameters passed for non-parametric circuit
            self.assertRaises(QiskitError, assemble, full_bound_circ, **partial_bind_args)

            # Raise when no parameters passed for parametric circuit
            self.assertRaises(QiskitError, assemble, partial_param_circ)
            self.assertRaises(QiskitError, assemble, full_param_circ)

            # Raise when circuit has more parameters than run_config
            self.assertRaises(QiskitError, assemble, full_param_circ, **partial_bind_args)

            # Raise when not all circuits have all parameters
            self.assertRaises(
                QiskitError, assemble, [full_param_circ, partial_param_circ], **full_bind_args
            )

            # Raise when not all binds have all circuit params
            self.assertRaises(QiskitError, assemble, full_param_circ, **inconsistent_bind_args)

    def test_assemble_circuits_rases_for_bind_mismatch_over_expressions(self):
        """Verify assemble_circuits raises for invalid binds for circuit including
        ParameterExpressions.
        """
        qr = QuantumRegister(1)
        x = Parameter("x")
        y = Parameter("y")

        expr_circ = QuantumCircuit(qr)

        expr_circ.p(x + y, qr[0])

        partial_bind_args = {"parameter_binds": [{x: 1}, {x: 0}]}

        with self.assertWarns(DeprecationWarning):
            # Raise when no parameters passed for parametric circuit
            self.assertRaises(QiskitError, assemble, expr_circ)

        with self.assertWarns(DeprecationWarning):
            # Raise when circuit has more parameters than run_config
            self.assertRaises(QiskitError, assemble, expr_circ, **partial_bind_args)

    def test_assemble_circuits_binds_parameters(self):
        """Verify assemble_circuits applies parameter bindings and output circuits are bound."""
        qr = QuantumRegister(1)
        qc1 = QuantumCircuit(qr)
        qc2 = QuantumCircuit(qr)
        qc3 = QuantumCircuit(qr)

        x = Parameter("x")
        y = Parameter("y")
        sum_ = x + y
        product_ = x * y

        qc1.u(x, y, 0, qr[0])

        qc2.rz(x, qr[0])
        qc2.rz(y, qr[0])

        qc3.u(sum_, product_, 0, qr[0])

        bind_args = {"parameter_binds": [{x: 0, y: 0}, {x: 1, y: 0}, {x: 1, y: 1}]}

        with self.assertWarns(DeprecationWarning):
            qobj = assemble([qc1, qc2, qc3], **bind_args)

        self.assertEqual(len(qobj.experiments), 9)
        self.assertEqual(
            [len(expt.instructions) for expt in qobj.experiments], [1, 1, 1, 2, 2, 2, 1, 1, 1]
        )

        def _qobj_inst_params(expt_no, inst_no):
            expt = qobj.experiments[expt_no]
            inst = expt.instructions[inst_no]
            return [float(p) for p in inst.params]

        self.assertEqual(_qobj_inst_params(0, 0), [0, 0, 0])
        self.assertEqual(_qobj_inst_params(1, 0), [1, 0, 0])
        self.assertEqual(_qobj_inst_params(2, 0), [1, 1, 0])

        self.assertEqual(_qobj_inst_params(3, 0), [0])
        self.assertEqual(_qobj_inst_params(3, 1), [0])
        self.assertEqual(_qobj_inst_params(4, 0), [1])
        self.assertEqual(_qobj_inst_params(4, 1), [0])
        self.assertEqual(_qobj_inst_params(5, 0), [1])
        self.assertEqual(_qobj_inst_params(5, 1), [1])

        self.assertEqual(_qobj_inst_params(6, 0), [0, 0, 0])
        self.assertEqual(_qobj_inst_params(7, 0), [1, 0, 0])
        self.assertEqual(_qobj_inst_params(8, 0), [2, 1, 0])

    def test_init_qubits_default(self):
        """Check that the init_qubits=None assemble option is passed on to the qobj."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ)
        self.assertEqual(qobj.config.init_qubits, True)

    def test_init_qubits_true(self):
        """Check that the init_qubits=True assemble option is passed on to the qobj."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ, init_qubits=True)
        self.assertEqual(qobj.config.init_qubits, True)

    def test_init_qubits_false(self):
        """Check that the init_qubits=False assemble option is passed on to the qobj."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ, init_qubits=False)
        self.assertEqual(qobj.config.init_qubits, False)

    def test_circuit_with_global_phase(self):
        """Test that global phase for a circuit is handled correctly."""
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        circ.measure_all()
        circ.global_phase = 0.3 * np.pi
        with self.assertWarns(DeprecationWarning):
            qobj = assemble([circ, self.circ])
        self.assertEqual(getattr(qobj.experiments[1].header, "global_phase"), 0)
        self.assertEqual(getattr(qobj.experiments[0].header, "global_phase"), 0.3 * np.pi)

    def test_circuit_global_phase_gate_definitions(self):
        """Test circuit with global phase on gate definitions."""

        class TestGate(Gate):
            """dummy gate"""

            def __init__(self):
                super().__init__("test_gate", 1, [])

            def _define(self):
                circ_def = QuantumCircuit(1)
                circ_def.x(0)
                circ_def.global_phase = np.pi
                self._definition = circ_def

        gate = TestGate()
        circ = QuantumCircuit(1)
        circ.append(gate, [0])
        with self.assertWarns(DeprecationWarning):
            qobj = assemble([circ])
        self.assertEqual(getattr(qobj.experiments[0].header, "global_phase"), 0)
        circ.global_phase = np.pi / 2
        with self.assertWarns(DeprecationWarning):
            qobj = assemble([circ])
        self.assertEqual(getattr(qobj.experiments[0].header, "global_phase"), np.pi / 2)

    def test_pulse_gates_single_circ(self):
        """Test that we can add calibrations to circuits."""
        theta = Parameter("theta")
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.append(RxGate(3.14), [0])
        circ.append(RxGate(theta), [1])
        circ = circ.assign_parameters({theta: 3.14})

        with self.assertWarns(DeprecationWarning):
            with pulse.build() as custom_h_schedule:
                pulse.play(pulse.library.Drag(50, 0.15, 4, 2), pulse.DriveChannel(0))

            with pulse.build() as x180:
                pulse.play(pulse.library.Gaussian(50, 0.2, 5), pulse.DriveChannel(1))

            circ.add_calibration("h", [0], custom_h_schedule)
            circ.add_calibration(RxGate(3.14), [0], x180)
            circ.add_calibration(RxGate(3.14), [1], x180)

            qobj = assemble(circ, FakeOpenPulse2Q())
        # Only one circuit, so everything is stored at the job level
        cals = qobj.config.calibrations
        lib = qobj.config.pulse_library
        self.assertFalse(hasattr(qobj.experiments[0].config, "calibrations"))
        self.assertEqual([gate.name == "rxtheta" for gate in cals.gates].count(True), 2)
        self.assertEqual([gate.name == "h" for gate in cals.gates].count(True), 1)
        self.assertEqual(len(lib), 2)
        self.assertTrue(all(len(item.samples) == 50 for item in lib))

    def test_custom_pulse_gates_single_circ(self):
        """Test that we can add calibrations to circuits of pulses which are not in
        qiskit.qobj.converters.pulse_instruction.ParametricPulseShapes"""
        circ = QuantumCircuit(2)
        circ.h(0)

        with self.assertWarns(DeprecationWarning):
            with pulse.build() as custom_h_schedule:
                pulse.play(pulse.library.Triangle(50, 0.1, 0.2), pulse.DriveChannel(0))

            circ.add_calibration("h", [0], custom_h_schedule)

            qobj = assemble(circ, FakeOpenPulse2Q())
        lib = qobj.config.pulse_library
        self.assertEqual(len(lib), 1)
        with self.assertWarns(DeprecationWarning):
            np.testing.assert_almost_equal(
                lib[0].samples, pulse.library.Triangle(50, 0.1, 0.2).get_waveform().samples
            )

    def test_pulse_gates_with_parameteric_pulses(self):
        """Test that pulse gates are assembled efficiently for backends that enable
        parametric pulses.
        """
        with self.assertWarns(DeprecationWarning):
            with pulse.build() as custom_h_schedule:
                pulse.play(pulse.library.Drag(50, 0.15, 4, 2), pulse.DriveChannel(0))

        circ = QuantumCircuit(2)
        circ.h(0)

        with self.assertWarns(DeprecationWarning):
            circ.add_calibration("h", [0], custom_h_schedule)
            backend = FakeOpenPulse2Q()
        backend.configuration().parametric_pulses = ["drag"]
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(circ, backend)
        self.assertFalse(hasattr(qobj.config, "pulse_library"))
        self.assertTrue(hasattr(qobj.config, "calibrations"))

    def test_pulse_gates_multiple_circuits(self):
        """Test one circuit with cals and another without."""
        with self.assertWarns(DeprecationWarning):
            with pulse.build() as dummy_sched:
                pulse.play(pulse.library.Drag(50, 0.15, 4, 2), pulse.DriveChannel(0))

        circ = QuantumCircuit(2)
        circ.h(0)
        circ.append(RxGate(3.14), [1])
        with self.assertWarns(DeprecationWarning):
            circ.add_calibration("h", [0], dummy_sched)
            circ.add_calibration(RxGate(3.14), [1], dummy_sched)

        circ2 = QuantumCircuit(2)
        circ2.h(0)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble([circ, circ2], FakeOpenPulse2Q())
        self.assertEqual(len(qobj.config.pulse_library), 1)
        self.assertEqual(len(qobj.experiments[0].config.calibrations.gates), 2)
        self.assertFalse(hasattr(qobj.config, "calibrations"))
        self.assertFalse(hasattr(qobj.experiments[1].config, "calibrations"))

    def test_pulse_gates_common_cals(self):
        """Test that common calibrations are added at the top level."""
        with self.assertWarns(DeprecationWarning):
            with pulse.build() as dummy_sched:
                pulse.play(pulse.library.Drag(50, 0.15, 4, 2), pulse.DriveChannel(0))

        circ = QuantumCircuit(2)
        circ.h(0)
        circ.append(RxGate(3.14), [1])
        with self.assertWarns(DeprecationWarning):
            circ.add_calibration("h", [0], dummy_sched)
            circ.add_calibration(RxGate(3.14), [1], dummy_sched)

        circ2 = QuantumCircuit(2)
        circ2.h(0)
        with self.assertWarns(DeprecationWarning):
            circ2.add_calibration(RxGate(3.14), [1], dummy_sched)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble([circ, circ2], FakeOpenPulse2Q())
        # Identical pulses are only added once
        self.assertEqual(len(qobj.config.pulse_library), 1)
        # Identical calibrations are only added once
        self.assertEqual(qobj.config.calibrations.gates[0].name, "rxtheta")
        self.assertEqual(qobj.config.calibrations.gates[0].params, [3.14])
        self.assertEqual(qobj.config.calibrations.gates[0].qubits, [1])
        self.assertEqual(len(qobj.experiments[0].config.calibrations.gates), 1)
        self.assertFalse(hasattr(qobj.experiments[1].config, "calibrations"))

    def test_assemble_adds_circuit_metadata_to_experiment_header(self):
        """Verify that any circuit metadata is added to the experiment header."""
        circ = QuantumCircuit(2, metadata={"experiment_type": "gst", "execution_number": "1234"})
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(circ, shots=100, memory=False, seed_simulator=6)
        self.assertEqual(
            qobj.experiments[0].header.metadata,
            {"experiment_type": "gst", "execution_number": "1234"},
        )

    def test_pulse_gates_delay_only(self):
        """Test that a single delay gate is translated to an instruction."""
        circ = QuantumCircuit(2)
        circ.append(Gate("test", 1, []), [0])
        with self.assertWarns(DeprecationWarning):
            test_sched = pulse.Delay(64, DriveChannel(0)) + pulse.Delay(160, DriveChannel(0))
            circ.add_calibration("test", [0], test_sched)
            qobj = assemble(circ, FakeOpenPulse2Q())
        self.assertEqual(len(qobj.config.calibrations.gates[0].instructions), 2)
        self.assertEqual(
            qobj.config.calibrations.gates[0].instructions[1].to_dict(),
            {"name": "delay", "t0": 64, "ch": "d0", "duration": 160},
        )

    def test_job_qubit_meas_los_no_range(self):
        """Test that adding job qubit/meas lo freq lists are assembled into the qobj.config, w/ out
        any lo range."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.circ,
                backend=self.backend,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
            )

        # convert to ghz
        qubit_lo_freq_ghz = [freq / 1e9 for freq in self.default_qubit_lo_freq]
        meas_lo_freq_ghz = [freq / 1e9 for freq in self.default_meas_lo_freq]
        self.assertEqual(qobj.config.qubit_lo_freq, qubit_lo_freq_ghz)
        self.assertEqual(qobj.config.meas_lo_freq, meas_lo_freq_ghz)

    def test_job_lo_errors(self):
        """Test that job lo's are checked against the lo ranges and that errors are thrown if either
        quantity has an incorrect length or type."""
        qubit_lo_range = [[freq - 5e6, freq + 5e6] for freq in self.default_qubit_lo_freq]
        meas_lo_range = [[freq - 5e6, freq + 5e6] for freq in self.default_meas_lo_freq]

        # lo range not a nested list
        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(QiskitError):
                assemble(
                    self.circ,
                    backend=self.backend,
                    qubit_lo_freq=self.default_qubit_lo_freq,
                    meas_lo_freq=self.default_meas_lo_freq,
                    qubit_lo_range=[4.995e9 for i in range(self.num_qubits)],
                    meas_lo_range=meas_lo_range,
                )

        # qubit lo range inner list not 2d
        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(QiskitError):
                assemble(
                    self.circ,
                    backend=self.backend,
                    qubit_lo_freq=self.default_qubit_lo_freq,
                    meas_lo_freq=self.default_meas_lo_freq,
                    qubit_lo_range=qubit_lo_range,
                    meas_lo_range=[[6.695e9] for i in range(self.num_qubits)],
                )

        # meas lo range inner list not 2d
        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(QiskitError):
                assemble(
                    self.circ,
                    backend=self.backend,
                    qubit_lo_freq=self.default_qubit_lo_freq,
                    meas_lo_freq=self.default_meas_lo_freq,
                    qubit_lo_range=qubit_lo_range,
                    meas_lo_range=[[6.695e9] for i in range(self.num_qubits)],
                )

        # qubit lo out of range
        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(QiskitError):
                assemble(
                    self.circ,
                    backend=self.backend,
                    qubit_lo_freq=self.default_qubit_lo_freq,
                    meas_lo_freq=self.default_meas_lo_freq,
                    qubit_lo_range=[[5.005e9, 5.010e9] for i in range(self.num_qubits)],
                    meas_lo_range=meas_lo_range,
                )

        # meas lo out of range
        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(QiskitError):
                assemble(
                    self.circ,
                    backend=self.backend,
                    qubit_lo_freq=self.default_qubit_lo_freq,
                    meas_lo_freq=self.default_meas_lo_freq,
                    qubit_lo_range=qubit_lo_range,
                    meas_lo_range=[[6.705e9, 6.710e9] for i in range(self.num_qubits)],
                )

    def test_job_qubit_meas_los_w_range(self):
        """Test that adding job qubit/meas lo freq lists are assembled into the qobj.config, w/ lo
        ranges input. Verify that lo ranges do not enter into the config."""
        qubit_lo_range = [[freq - 5e6, freq + 5e6] for freq in self.default_qubit_lo_freq]
        meas_lo_range = [[freq - 5e6, freq + 5e6] for freq in self.default_meas_lo_freq]
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.circ,
                backend=self.backend,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                qubit_lo_range=qubit_lo_range,
                meas_lo_range=meas_lo_range,
            )

        # convert to ghz
        qubit_lo_freq_ghz = [freq / 1e9 for freq in self.default_qubit_lo_freq]
        meas_lo_freq_ghz = [freq / 1e9 for freq in self.default_meas_lo_freq]
        self.assertEqual(qobj.config.qubit_lo_freq, qubit_lo_freq_ghz)
        self.assertEqual(qobj.config.meas_lo_freq, meas_lo_freq_ghz)
        self.assertNotIn("qubit_lo_range", qobj.config.to_dict())
        self.assertNotIn("meas_lo_range", qobj.config.to_dict())

    def test_assemble_single_circ_single_lo_config(self):
        """Test assembling a single circuit, with a single experiment level lo config."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.circ,
                self.backend,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=self.user_lo_config,
            )

        self.assertListEqual(qobj.config.qubit_lo_freq, [5.55, 5, 5, 4.91, 5])
        self.assertListEqual(qobj.config.meas_lo_freq, [6.64, 6.7, 6.7, 6.7, 6.1])
        self.assertEqual(len(qobj.experiments), 1)

    def test_assemble_single_circ_single_lo_config_dict(self):
        """Test assembling a single circuit, with a single experiment level lo config supplied as
        dictionary."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.circ,
                self.backend,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=self.user_lo_config_dict,
            )

        self.assertListEqual(qobj.config.qubit_lo_freq, [5.55, 5, 5, 4.91, 5])
        self.assertListEqual(qobj.config.meas_lo_freq, [6.64, 6.7, 6.7, 6.7, 6.1])
        self.assertEqual(len(qobj.experiments), 1)

    def test_assemble_single_circ_multi_lo_config(self):
        """Test assembling a single circuit, with multiple experiment level lo configs (frequency
        sweep).
        """
        with self.assertWarns(DeprecationWarning):
            user_lo_config_dict2 = {
                pulse.DriveChannel(1): 5.55e9,
                pulse.MeasureChannel(1): 6.64e9,
                pulse.DriveChannel(4): 4.91e9,
                pulse.MeasureChannel(3): 6.1e9,
            }
        user_lo_config2 = pulse.LoConfig(user_lo_config_dict2)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.circ,
                self.backend,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=[self.user_lo_config, user_lo_config2],
            )

        qubit_lo_freq_ghz = [freq / 1e9 for freq in self.default_qubit_lo_freq]
        meas_lo_freq_ghz = [freq / 1e9 for freq in self.default_meas_lo_freq]
        self.assertListEqual(qobj.config.qubit_lo_freq, qubit_lo_freq_ghz)
        self.assertListEqual(qobj.config.meas_lo_freq, meas_lo_freq_ghz)
        self.assertEqual(len(qobj.experiments), 2)

        # experiment 0 los
        self.assertEqual(qobj.experiments[0].config.qubit_lo_freq, [5.55, 5, 5, 4.91, 5])
        self.assertEqual(qobj.experiments[0].config.meas_lo_freq, [6.64, 6.7, 6.7, 6.7, 6.1])
        # experiment 1 los
        self.assertEqual(qobj.experiments[1].config.qubit_lo_freq, [5, 5.55, 5, 5, 4.91])
        self.assertEqual(qobj.experiments[1].config.meas_lo_freq, [6.7, 6.64, 6.7, 6.1, 6.7])

    def test_assemble_multi_circ_multi_lo_config(self):
        """Test assembling circuits, with the same number of experiment level lo configs (n:n
        setup)."""
        with self.assertWarns(DeprecationWarning):
            user_lo_config_dict2 = {
                pulse.DriveChannel(1): 5.55e9,
                pulse.MeasureChannel(1): 6.64e9,
                pulse.DriveChannel(4): 4.91e9,
                pulse.MeasureChannel(3): 6.1e9,
            }
        user_lo_config2 = pulse.LoConfig(user_lo_config_dict2)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                [self.circ, self.circ],
                self.backend,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=[self.user_lo_config, user_lo_config2],
            )

        qubit_lo_freq_ghz = [freq / 1e9 for freq in self.default_qubit_lo_freq]
        meas_lo_freq_ghz = [freq / 1e9 for freq in self.default_meas_lo_freq]
        self.assertListEqual(qobj.config.qubit_lo_freq, qubit_lo_freq_ghz)
        self.assertListEqual(qobj.config.meas_lo_freq, meas_lo_freq_ghz)
        self.assertEqual(len(qobj.experiments), 2)

        # experiment 0 los
        self.assertEqual(qobj.experiments[0].config.qubit_lo_freq, [5.55, 5, 5, 4.91, 5])
        self.assertEqual(qobj.experiments[0].config.meas_lo_freq, [6.64, 6.7, 6.7, 6.7, 6.1])
        # experiment 1 los
        self.assertEqual(qobj.experiments[1].config.qubit_lo_freq, [5, 5.55, 5, 5, 4.91])
        self.assertEqual(qobj.experiments[1].config.meas_lo_freq, [6.7, 6.64, 6.7, 6.1, 6.7])

    def test_assemble_multi_circ_single_lo_config(self):
        """Test assembling multiple circuits, with a single experiment level lo config (should
        override job level)."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                [self.circ, self.circ],
                self.backend,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=self.user_lo_config,
            )

        self.assertListEqual(qobj.config.qubit_lo_freq, [5.55, 5, 5, 4.91, 5])
        self.assertListEqual(qobj.config.meas_lo_freq, [6.64, 6.7, 6.7, 6.7, 6.1])
        self.assertEqual(len(qobj.experiments), 2)

    def test_assemble_multi_circ_wrong_number_of_multi_lo_configs(self):
        """Test assembling circuits, with a different number of experiment level lo configs (n:m
        setup).
        """
        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(QiskitError):
                assemble(
                    [self.circ, self.circ, self.circ],
                    self.backend,
                    qubit_lo_freq=self.default_qubit_lo_freq,
                    meas_lo_freq=self.default_meas_lo_freq,
                    schedule_los=[self.user_lo_config, self.user_lo_config],
                )

    def test_assemble_circ_lo_config_errors(self):
        """Test that lo config errors are raised properly if experiment level los are provided and
        some are missing or if default values are not provided. Also check that experiment level lo
        range is validated."""
        # no defaults, but have drive/meas experiment level los for each qubit (no error)
        with self.assertWarns(DeprecationWarning):
            full_lo_config_dict = {
                pulse.DriveChannel(0): 4.85e9,
                pulse.DriveChannel(1): 4.9e9,
                pulse.DriveChannel(2): 4.95e9,
                pulse.DriveChannel(3): 5e9,
                pulse.DriveChannel(4): 5.05e9,
                pulse.MeasureChannel(0): 6.8e9,
                pulse.MeasureChannel(1): 6.85e9,
                pulse.MeasureChannel(2): 6.9e9,
                pulse.MeasureChannel(3): 6.95e9,
                pulse.MeasureChannel(4): 7e9,
            }

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.circ, self.backend, schedule_los=full_lo_config_dict)

        self.assertListEqual(qobj.config.qubit_lo_freq, [4.85, 4.9, 4.95, 5, 5.05])
        self.assertListEqual(qobj.config.meas_lo_freq, [6.8, 6.85, 6.9, 6.95, 7])
        self.assertEqual(len(qobj.experiments), 1)

        # no defaults and missing experiment level drive lo raises
        missing_drive_lo_config_dict = copy.deepcopy(full_lo_config_dict)
        with self.assertWarns(DeprecationWarning):
            missing_drive_lo_config_dict.pop(pulse.DriveChannel(0))
            with self.assertRaises(QiskitError):
                qobj = assemble(self.circ, self.backend, schedule_los=missing_drive_lo_config_dict)

        # no defaults and missing experiment level meas lo raises
        missing_meas_lo_config_dict = copy.deepcopy(full_lo_config_dict)
        with self.assertWarns(DeprecationWarning):
            missing_meas_lo_config_dict.pop(pulse.MeasureChannel(0))
            with self.assertRaises(QiskitError):
                qobj = assemble(self.circ, self.backend, schedule_los=missing_meas_lo_config_dict)

        # verify lo ranges are checked at experiment level
        lo_values = list(full_lo_config_dict.values())
        qubit_lo_range = [[freq - 5e6, freq + 5e6] for freq in lo_values[:5]]
        meas_lo_range = [[freq - 5e6, freq + 5e6] for freq in lo_values[5:]]

        # out of range drive lo
        with self.assertWarns(DeprecationWarning):
            full_lo_config_dict[pulse.DriveChannel(0)] -= 5.5e6
            with self.assertRaises(QiskitError):
                qobj = assemble(
                    self.circ,
                    self.backend,
                    qubit_lo_range=qubit_lo_range,
                    schedule_los=full_lo_config_dict,
                )
            full_lo_config_dict[pulse.DriveChannel(0)] += 5.5e6  # reset drive value

        with self.assertWarns(DeprecationWarning):
            # out of range meas lo
            full_lo_config_dict[pulse.MeasureChannel(0)] += 5.5e6
            with self.assertRaises(QiskitError):
                qobj = assemble(
                    self.circ,
                    self.backend,
                    meas_lo_range=meas_lo_range,
                    schedule_los=full_lo_config_dict,
                )


class TestPulseAssembler(QiskitTestCase):
    """Tests for assembling schedules to qobj."""

    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            self.backend = FakeOpenPulse2Q()
        self.backend_config = self.backend.configuration()

        with self.assertWarns(DeprecationWarning):
            test_pulse = pulse.Waveform(
                samples=np.array([0.02739068, 0.05, 0.05, 0.05, 0.02739068], dtype=np.complex128),
                name="pulse0",
            )

            self.schedule = pulse.Schedule(name="fake_experiment")
            self.schedule = self.schedule.insert(0, Play(test_pulse, self.backend_config.drive(0)))
            for i in range(self.backend_config.n_qubits):
                self.schedule = self.schedule.insert(
                    5, Acquire(5, self.backend_config.acquire(i), MemorySlot(i))
                )

            self.user_lo_config_dict = {self.backend_config.drive(0): 4.91e9}
        self.user_lo_config = pulse.LoConfig(self.user_lo_config_dict)

        self.default_qubit_lo_freq = [4.9e9, 5.0e9]
        self.default_meas_lo_freq = [6.5e9, 6.6e9]

        self.config = {"meas_level": 1, "memory_slot_size": 100, "meas_return": "avg"}

        self.header = {"backend_name": "FakeOpenPulse2Q", "backend_version": "0.0.0"}

    def test_assemble_adds_schedule_metadata_to_experiment_header(self):
        """Verify that any circuit metadata is added to the experiment header."""
        self.schedule.metadata = {"experiment_type": "gst", "execution_number": "1234"}
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.schedule,
                shots=100,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=[],
            )
        self.assertEqual(
            qobj.experiments[0].header.metadata,
            {"experiment_type": "gst", "execution_number": "1234"},
        )

    def test_assemble_sample_pulse(self):
        """Test that the pulse lib and qobj instruction can be paired up."""
        with self.assertWarns(DeprecationWarning):
            schedule = pulse.Schedule()
            schedule += pulse.Play(
                pulse.Waveform([0.1] * 16, name="test0"), pulse.DriveChannel(0), name="test1"
            )
            schedule += pulse.Play(
                pulse.Waveform([0.1] * 16, name="test1"), pulse.DriveChannel(0), name="test2"
            )
            schedule += pulse.Play(
                pulse.Waveform([0.5] * 16, name="test0"), pulse.DriveChannel(0), name="test1"
            )

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedule,
                qobj_header=self.header,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=[],
                **self.config,
            )

        test_dict = qobj.to_dict()
        experiment = test_dict["experiments"][0]
        inst0_name = experiment["instructions"][0]["name"]
        inst1_name = experiment["instructions"][1]["name"]
        inst2_name = experiment["instructions"][2]["name"]
        pulses = {}
        for item in test_dict["config"]["pulse_library"]:
            pulses[item["name"]] = item["samples"]
        self.assertTrue(all(name in pulses for name in [inst0_name, inst1_name, inst2_name]))
        # Their pulses are the same
        self.assertEqual(inst0_name, inst1_name)
        self.assertTrue(np.allclose(pulses[inst0_name], [0.1] * 16))
        self.assertTrue(np.allclose(pulses[inst2_name], [0.5] * 16))

    def test_assemble_single_schedule_without_lo_config(self):
        """Test assembling a single schedule, no lo config."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.schedule,
                qobj_header=self.header,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=[],
                **self.config,
            )

        test_dict = qobj.to_dict()
        self.assertListEqual(test_dict["config"]["qubit_lo_freq"], [4.9, 5.0])
        self.assertEqual(len(test_dict["experiments"]), 1)
        self.assertEqual(len(test_dict["experiments"][0]["instructions"]), 2)

    def test_assemble_multi_schedules_without_lo_config(self):
        """Test assembling schedules, no lo config."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                [self.schedule, self.schedule],
                qobj_header=self.header,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                **self.config,
            )

        test_dict = qobj.to_dict()
        self.assertListEqual(test_dict["config"]["qubit_lo_freq"], [4.9, 5.0])
        self.assertEqual(len(test_dict["experiments"]), 2)
        self.assertEqual(len(test_dict["experiments"][0]["instructions"]), 2)

    def test_assemble_single_schedule_with_lo_config(self):
        """Test assembling a single schedule, with a single lo config."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.schedule,
                qobj_header=self.header,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=self.user_lo_config,
                **self.config,
            )

        test_dict = qobj.to_dict()
        self.assertListEqual(test_dict["config"]["qubit_lo_freq"], [4.91, 5.0])
        self.assertEqual(len(test_dict["experiments"]), 1)
        self.assertEqual(len(test_dict["experiments"][0]["instructions"]), 2)

    def test_assemble_single_schedule_with_lo_config_dict(self):
        """Test assembling a single schedule, with a single lo config supplied as dictionary."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.schedule,
                qobj_header=self.header,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=self.user_lo_config_dict,
                **self.config,
            )

        test_dict = qobj.to_dict()
        self.assertListEqual(test_dict["config"]["qubit_lo_freq"], [4.91, 5.0])
        self.assertEqual(len(test_dict["experiments"]), 1)
        self.assertEqual(len(test_dict["experiments"][0]["instructions"]), 2)

    def test_assemble_single_schedule_with_multi_lo_configs(self):
        """Test assembling a single schedule, with multiple lo configs (frequency sweep)."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.schedule,
                qobj_header=self.header,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=[self.user_lo_config, self.user_lo_config],
                **self.config,
            )
        test_dict = qobj.to_dict()

        self.assertListEqual(test_dict["config"]["qubit_lo_freq"], [4.9, 5.0])
        self.assertEqual(len(test_dict["experiments"]), 2)
        self.assertEqual(len(test_dict["experiments"][0]["instructions"]), 2)
        self.assertDictEqual(test_dict["experiments"][0]["config"], {"qubit_lo_freq": [4.91, 5.0]})

    def test_assemble_multi_schedules_with_multi_lo_configs(self):
        """Test assembling schedules, with the same number of lo configs (n:n setup)."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                [self.schedule, self.schedule],
                qobj_header=self.header,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=[self.user_lo_config, self.user_lo_config],
                **self.config,
            )

        test_dict = qobj.to_dict()
        self.assertListEqual(test_dict["config"]["qubit_lo_freq"], [4.9, 5.0])
        self.assertEqual(len(test_dict["experiments"]), 2)
        self.assertEqual(len(test_dict["experiments"][0]["instructions"]), 2)
        self.assertDictEqual(test_dict["experiments"][0]["config"], {"qubit_lo_freq": [4.91, 5.0]})

    def test_assemble_multi_schedules_with_wrong_number_of_multi_lo_configs(self):
        """Test assembling schedules, with a different number of lo configs (n:m setup)."""
        with self.assertRaises(QiskitError):
            with self.assertWarns(DeprecationWarning):
                assemble(
                    [self.schedule, self.schedule, self.schedule],
                    qobj_header=self.header,
                    qubit_lo_freq=self.default_qubit_lo_freq,
                    meas_lo_freq=self.default_meas_lo_freq,
                    schedule_los=[self.user_lo_config, self.user_lo_config],
                    **self.config,
                )

    def test_assemble_meas_map(self):
        """Test assembling a single schedule, no lo config."""
        with self.assertWarns(DeprecationWarning):
            schedule = Schedule(name="fake_experiment")
            schedule = schedule.insert(5, Acquire(5, AcquireChannel(0), MemorySlot(0)))
            schedule = schedule.insert(5, Acquire(5, AcquireChannel(1), MemorySlot(1)))

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0], [1]],
            )
        self.assertIsInstance(qobj, PulseQobj)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0, 1, 2]],
            )
        self.assertIsInstance(qobj, PulseQobj)

    def test_assemble_memory_slots(self):
        """Test assembling a schedule and inferring number of memoryslots."""
        n_memoryslots = 10

        # single acquisition
        with self.assertWarns(DeprecationWarning):
            schedule = Acquire(
                5, self.backend_config.acquire(0), mem_slot=pulse.MemorySlot(n_memoryslots - 1)
            )

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0], [1]],
            )

        self.assertEqual(qobj.config.memory_slots, n_memoryslots)
        # this should be in experimental header as well
        self.assertEqual(qobj.experiments[0].header.memory_slots, n_memoryslots)

        # multiple acquisition
        with self.assertWarns(DeprecationWarning):
            schedule = Acquire(
                5, self.backend_config.acquire(0), mem_slot=pulse.MemorySlot(n_memoryslots - 1)
            )
            schedule = schedule.insert(
                10,
                Acquire(
                    5, self.backend_config.acquire(0), mem_slot=pulse.MemorySlot(n_memoryslots - 1)
                ),
            )

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0], [1]],
            )

        self.assertEqual(qobj.config.memory_slots, n_memoryslots)
        # this should be in experimental header as well
        self.assertEqual(qobj.experiments[0].header.memory_slots, n_memoryslots)

    def test_assemble_memory_slots_for_schedules(self):
        """Test assembling schedules with different memory slots."""
        n_memoryslots = [10, 5, 7]

        schedules = []
        with self.assertWarns(DeprecationWarning):
            for n_memoryslot in n_memoryslots:
                schedule = Acquire(
                    5, self.backend_config.acquire(0), mem_slot=pulse.MemorySlot(n_memoryslot - 1)
                )
                schedules.append(schedule)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedules,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0], [1]],
            )

        self.assertEqual(qobj.config.memory_slots, max(n_memoryslots))
        self.assertEqual(qobj.experiments[0].header.memory_slots, n_memoryslots[0])
        self.assertEqual(qobj.experiments[1].header.memory_slots, n_memoryslots[1])
        self.assertEqual(qobj.experiments[2].header.memory_slots, n_memoryslots[2])

    def test_pulse_name_conflicts(self):
        """Test that pulse name conflicts can be resolved."""
        with self.assertWarns(DeprecationWarning):
            name_conflict_pulse = pulse.Waveform(
                samples=np.array([0.02, 0.05, 0.05, 0.05, 0.02], dtype=np.complex128), name="pulse0"
            )

            self.schedule = self.schedule.insert(
                1, Play(name_conflict_pulse, self.backend_config.drive(1))
            )

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.schedule,
                qobj_header=self.header,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=[],
                **self.config,
            )

        self.assertNotEqual(qobj.config.pulse_library[0].name, qobj.config.pulse_library[1].name)

    def test_pulse_name_conflicts_in_other_schedule(self):
        """Test two pulses with the same name in different schedule can be resolved."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake27QPulseV1()
        defaults = backend.defaults()

        schedules = []
        with self.assertWarns(DeprecationWarning):
            ch_d0 = pulse.DriveChannel(0)
            for amp in (0.1, 0.2):
                sched = Schedule()
                sched += Play(
                    pulse.Gaussian(duration=100, amp=amp, sigma=30, name="my_pulse"), ch_d0
                )
                sched += measure(qubits=[0], backend=backend) << 100
                schedules.append(sched)

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedules,
                qubit_lo_freq=defaults.qubit_freq_est,
                meas_lo_freq=defaults.meas_freq_est,
            )

        # two user pulses and one measurement pulse should be contained
        self.assertEqual(len(qobj.config.pulse_library), 3)

    def test_assemble_with_delay(self):
        """Test that delay instruction is not ignored in assembly."""
        with self.assertWarns(DeprecationWarning):
            delay_schedule = pulse.Delay(10, self.backend_config.drive(0))
            delay_schedule += self.schedule

        with self.assertWarns(DeprecationWarning):
            delay_qobj = assemble(delay_schedule, self.backend)

        self.assertEqual(delay_qobj.experiments[0].instructions[0].name, "delay")
        self.assertEqual(delay_qobj.experiments[0].instructions[0].duration, 10)
        self.assertEqual(delay_qobj.experiments[0].instructions[0].t0, 0)

    def test_delay_removed_on_acq_ch(self):
        """Test that delay instructions on acquire channels are skipped on assembly with times
        shifted properly.
        """
        with self.assertWarns(DeprecationWarning):
            delay0 = pulse.Delay(5, self.backend_config.acquire(0))
            delay1 = pulse.Delay(7, self.backend_config.acquire(1))

        sched0 = delay0
        with self.assertWarns(DeprecationWarning):
            sched0 += self.schedule  # includes ``Acquire`` instr
            sched0 += delay1

        with self.assertWarns(DeprecationWarning):
            sched1 = self.schedule  # includes ``Acquire`` instr
            sched1 += delay0
            sched1 += delay1

            sched2 = delay0
            sched2 += delay1
            sched2 += self.schedule  # includes ``Acquire`` instr

        with self.assertWarns(DeprecationWarning):
            delay_qobj = assemble([sched0, sched1, sched2], self.backend)

        # check that no delay instrs occur on acquire channels
        is_acq_delay = False
        for exp in delay_qobj.experiments:
            for instr in exp.instructions:
                if instr.name == "delay" and "a" in instr.ch:
                    is_acq_delay = True

        self.assertFalse(is_acq_delay)

        # check that acquire instr are shifted from ``t0=5`` as needed
        self.assertEqual(delay_qobj.experiments[0].instructions[1].t0, 10)
        self.assertEqual(delay_qobj.experiments[0].instructions[1].name, "acquire")
        self.assertEqual(delay_qobj.experiments[1].instructions[1].t0, 5)
        self.assertEqual(delay_qobj.experiments[1].instructions[1].name, "acquire")
        self.assertEqual(delay_qobj.experiments[2].instructions[1].t0, 12)
        self.assertEqual(delay_qobj.experiments[2].instructions[1].name, "acquire")

    def test_assemble_schedule_enum(self):
        """Test assembling a schedule with enum input values to assemble."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.schedule,
                qobj_header=self.header,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                schedule_los=[],
                meas_level=MeasLevel.CLASSIFIED,
                meas_return=MeasReturnType.AVERAGE,
            )

        test_dict = qobj.to_dict()
        self.assertEqual(test_dict["config"]["meas_return"], "avg")
        self.assertEqual(test_dict["config"]["meas_level"], 2)

    def test_assemble_parametric(self):
        """Test that parametric pulses can be assembled properly into a PulseQobj."""
        amp = [0.5, 0.6, 1, 0.2]
        angle = [np.pi / 2, 0.6, 0, 0]
        with self.assertWarns(DeprecationWarning):
            sched = pulse.Schedule(name="test_parametric")
            sched += Play(
                pulse.Gaussian(duration=25, sigma=4, amp=amp[0], angle=angle[0]), DriveChannel(0)
            )
            sched += Play(
                pulse.Drag(duration=25, amp=amp[1], angle=angle[1], sigma=7.8, beta=4),
                DriveChannel(1),
            )
            sched += Play(pulse.Constant(duration=25, amp=amp[2], angle=angle[2]), DriveChannel(2))
            sched += (
                Play(
                    pulse.GaussianSquare(
                        duration=150, amp=amp[3], angle=angle[3], sigma=8, width=140
                    ),
                    MeasureChannel(0),
                )
                << sched.duration
            )

        with self.assertWarns(DeprecationWarning):
            backend = FakeOpenPulse3Q()
        backend.configuration().parametric_pulses = [
            "gaussian",
            "drag",
            "gaussian_square",
            "constant",
        ]
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(sched, backend)

        self.assertEqual(qobj.config.pulse_library, [])
        qobj_insts = qobj.experiments[0].instructions
        self.assertTrue(all(inst.name == "parametric_pulse" for inst in qobj_insts))
        self.assertEqual(qobj_insts[0].pulse_shape, "gaussian")
        self.assertEqual(qobj_insts[1].pulse_shape, "drag")
        self.assertEqual(qobj_insts[2].pulse_shape, "constant")
        self.assertEqual(qobj_insts[3].pulse_shape, "gaussian_square")
        self.assertDictEqual(
            qobj_insts[0].parameters,
            {"duration": 25, "sigma": 4, "amp": amp[0] * np.exp(1j * angle[0])},
        )
        self.assertDictEqual(
            qobj_insts[1].parameters,
            {"duration": 25, "sigma": 7.8, "amp": amp[1] * np.exp(1j * angle[1]), "beta": 4},
        )
        self.assertDictEqual(
            qobj_insts[2].parameters, {"duration": 25, "amp": amp[2] * np.exp(1j * angle[2])}
        )
        self.assertDictEqual(
            qobj_insts[3].parameters,
            {"duration": 150, "sigma": 8, "amp": amp[3] * np.exp(1j * angle[3]), "width": 140},
        )
        self.assertEqual(
            qobj.to_dict()["experiments"][0]["instructions"][0]["parameters"]["amp"],
            amp[0] * np.exp(1j * angle[0]),
        )

    def test_assemble_parametric_unsupported(self):
        """Test that parametric pulses are translated to Waveform if they're not supported
        by the backend during assemble time.
        """
        with self.assertWarns(DeprecationWarning):
            sched = pulse.Schedule(name="test_parametric_to_sample_pulse")
            sched += Play(
                pulse.Drag(duration=25, amp=0.5, angle=-0.3, sigma=7.8, beta=4), DriveChannel(1)
            )
            sched += Play(pulse.Constant(duration=25, amp=1), DriveChannel(2))

        with self.assertWarns(DeprecationWarning):
            backend = FakeOpenPulse3Q()
        backend.configuration().parametric_pulses = ["something_extra"]

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(sched, backend)

        self.assertNotEqual(qobj.config.pulse_library, [])
        qobj_insts = qobj.experiments[0].instructions
        self.assertFalse(hasattr(qobj_insts[0], "pulse_shape"))

    def test_assemble_parametric_pulse_kwarg_with_backend_setting(self):
        """Test that parametric pulses respect the kwarg over backend"""
        with self.assertWarns(DeprecationWarning):
            backend = Fake27QPulseV1()

        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        with self.assertWarns(DeprecationWarning):
            with pulse.build(backend, name="x") as x_q0:
                pulse.play(pulse.Gaussian(duration=128, amp=0.1, sigma=16), pulse.drive_channel(0))

            qc.add_calibration("x", (0,), x_q0)

            qobj = assemble(qc, backend, parametric_pulses=["gaussian"])
        self.assertEqual(qobj.config.parametric_pulses, ["gaussian"])

    def test_assemble_parametric_pulse_kwarg_empty_list_with_backend_setting(self):
        """Test that parametric pulses respect the kwarg as empty list over backend"""
        with self.assertWarns(DeprecationWarning):
            backend = Fake27QPulseV1()

        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        with self.assertWarns(DeprecationWarning):
            with pulse.build(backend, name="x") as x_q0:
                pulse.play(pulse.Gaussian(duration=128, amp=0.1, sigma=16), pulse.drive_channel(0))

        with self.assertWarns(DeprecationWarning):
            qc.add_calibration("x", (0,), x_q0)
            qobj = assemble(qc, backend, parametric_pulses=[])
        self.assertEqual(qobj.config.parametric_pulses, [])

    def test_init_qubits_default(self):
        """Check that the init_qubits=None assemble option is passed on to the qobj."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.schedule, self.backend)
        self.assertEqual(qobj.config.init_qubits, True)

    def test_init_qubits_true(self):
        """Check that the init_qubits=True assemble option is passed on to the qobj."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.schedule, self.backend, init_qubits=True)
        self.assertEqual(qobj.config.init_qubits, True)

    def test_init_qubits_false(self):
        """Check that the init_qubits=False assemble option is passed on to the qobj."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.schedule, self.backend, init_qubits=False)
        self.assertEqual(qobj.config.init_qubits, False)

    def test_assemble_backend_rep_times_delays(self):
        """Check that rep_time and rep_delay are properly set from backend values."""
        # use first entry from allowed backend values
        rep_times = [2.0, 3.0, 4.0]  # sec
        rep_delay_range = [2.5e-3, 4.5e-3]
        default_rep_delay = 3.0e-3
        self.backend_config.rep_times = rep_times
        setattr(self.backend_config, "rep_delay_range", rep_delay_range)
        setattr(self.backend_config, "default_rep_delay", default_rep_delay)

        # dynamic rep rates off
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.schedule, self.backend)
        self.assertEqual(qobj.config.rep_time, int(rep_times[0] * 1e6))
        self.assertEqual(hasattr(qobj.config, "rep_delay"), False)

        # dynamic rep rates on
        setattr(self.backend_config, "dynamic_reprate_enabled", True)
        # RuntimeWarning bc ``rep_time`` is specified`` when dynamic rep rates not enabled
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.schedule, self.backend)
        self.assertEqual(qobj.config.rep_time, int(rep_times[0] * 1e6))
        self.assertEqual(qobj.config.rep_delay, default_rep_delay * 1e6)

    def test_assemble_user_rep_time_delay(self):
        """Check that user runtime config rep_time and rep_delay work."""
        # set custom rep_time and rep_delay in runtime config
        rep_time = 200.0e-6
        rep_delay = 2.5e-6
        self.config["rep_time"] = rep_time
        self.config["rep_delay"] = rep_delay

        # dynamic rep rates off
        # RuntimeWarning bc using ``rep_delay`` when dynamic rep rates off
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.schedule, self.backend, **self.config)
        self.assertEqual(qobj.config.rep_time, int(rep_time * 1e6))
        self.assertEqual(hasattr(qobj.config, "rep_delay"), False)

        # now remove rep_delay and enable dynamic rep rates
        # RuntimeWarning bc using ``rep_time`` when dynamic rep rates are enabled
        del self.config["rep_delay"]
        setattr(self.backend_config, "dynamic_reprate_enabled", True)
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.schedule, self.backend, **self.config)
        self.assertEqual(qobj.config.rep_time, int(rep_time * 1e6))
        self.assertEqual(hasattr(qobj.config, "rep_delay"), False)

        # use ``default_rep_delay``
        # ``rep_time`` comes from allowed backend rep_times
        rep_times = [0.5, 1.0, 1.5]  # sec
        self.backend_config.rep_times = rep_times
        setattr(self.backend_config, "rep_delay_range", [0, 3.0e-6])
        setattr(self.backend_config, "default_rep_delay", 2.2e-6)
        del self.config["rep_time"]
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.schedule, self.backend, **self.config)
        self.assertEqual(qobj.config.rep_time, int(rep_times[0] * 1e6))
        self.assertEqual(qobj.config.rep_delay, 2.2)

        # use qobj ``default_rep_delay``
        self.config["rep_delay"] = 1.5e-6
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(self.schedule, self.backend, **self.config)
        self.assertEqual(qobj.config.rep_time, int(rep_times[0] * 1e6))
        self.assertEqual(qobj.config.rep_delay, 1.5)

        # use ``rep_delay`` outside of ``rep_delay_range
        self.config["rep_delay"] = 5.0e-6
        with self.assertRaises(DeprecationWarning):
            assemble(self.schedule, self.backend, **self.config)

    def test_assemble_with_individual_discriminators(self):
        """Test that assembly works with individual discriminators."""
        disc_one = Discriminator("disc_one", test_params=True)
        disc_two = Discriminator("disc_two", test_params=False)

        with self.assertWarns(DeprecationWarning):
            schedule = Schedule()
            schedule = schedule.append(
                Acquire(5, AcquireChannel(0), MemorySlot(0), discriminator=disc_one),
            )
            schedule = schedule.append(
                Acquire(5, AcquireChannel(1), MemorySlot(1), discriminator=disc_two),
            )

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0, 1]],
            )

        qobj_discriminators = qobj.experiments[0].instructions[0].discriminators
        self.assertEqual(len(qobj_discriminators), 2)
        self.assertEqual(qobj_discriminators[0].name, "disc_one")
        self.assertEqual(qobj_discriminators[0].params["test_params"], True)
        self.assertEqual(qobj_discriminators[1].name, "disc_two")
        self.assertEqual(qobj_discriminators[1].params["test_params"], False)

    def test_assemble_with_single_discriminators(self):
        """Test that assembly works with both a single discriminator."""
        disc_one = Discriminator("disc_one", test_params=True)

        with self.assertWarns(DeprecationWarning):
            schedule = Schedule()
            schedule = schedule.append(
                Acquire(5, AcquireChannel(0), MemorySlot(0), discriminator=disc_one),
            )
            schedule = schedule.append(
                Acquire(5, AcquireChannel(1), MemorySlot(1)),
            )

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0, 1]],
            )

        qobj_discriminators = qobj.experiments[0].instructions[0].discriminators
        self.assertEqual(len(qobj_discriminators), 1)
        self.assertEqual(qobj_discriminators[0].name, "disc_one")
        self.assertEqual(qobj_discriminators[0].params["test_params"], True)

    def test_assemble_with_unequal_discriminators(self):
        """Test that assembly works with incorrect number of discriminators for
        number of qubits."""
        disc_one = Discriminator("disc_one", test_params=True)
        disc_two = Discriminator("disc_two", test_params=False)

        with self.assertWarns(DeprecationWarning):
            schedule = Schedule()
            schedule += Acquire(5, AcquireChannel(0), MemorySlot(0), discriminator=disc_one)
            schedule += Acquire(5, AcquireChannel(1), MemorySlot(1), discriminator=disc_two)
            schedule += Acquire(5, AcquireChannel(2), MemorySlot(2))

        with self.assertRaises(QiskitError), self.assertWarns(DeprecationWarning):
            assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0, 1, 2]],
            )

    def test_assemble_with_individual_kernels(self):
        """Test that assembly works with individual kernels."""
        disc_one = Kernel("disc_one", test_params=True)
        disc_two = Kernel("disc_two", test_params=False)

        with self.assertWarns(DeprecationWarning):
            schedule = Schedule()
            schedule = schedule.append(
                Acquire(5, AcquireChannel(0), MemorySlot(0), kernel=disc_one),
            )
            schedule = schedule.append(
                Acquire(5, AcquireChannel(1), MemorySlot(1), kernel=disc_two),
            )

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0, 1]],
            )

        qobj_kernels = qobj.experiments[0].instructions[0].kernels
        self.assertEqual(len(qobj_kernels), 2)
        self.assertEqual(qobj_kernels[0].name, "disc_one")
        self.assertEqual(qobj_kernels[0].params["test_params"], True)
        self.assertEqual(qobj_kernels[1].name, "disc_two")
        self.assertEqual(qobj_kernels[1].params["test_params"], False)

    def test_assemble_with_single_kernels(self):
        """Test that assembly works with both a single kernel."""
        disc_one = Kernel("disc_one", test_params=True)

        with self.assertWarns(DeprecationWarning):
            schedule = Schedule()
            schedule = schedule.append(
                Acquire(5, AcquireChannel(0), MemorySlot(0), kernel=disc_one),
            )
            schedule = schedule.append(
                Acquire(5, AcquireChannel(1), MemorySlot(1)),
            )

        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0, 1]],
            )

        qobj_kernels = qobj.experiments[0].instructions[0].kernels
        self.assertEqual(len(qobj_kernels), 1)
        self.assertEqual(qobj_kernels[0].name, "disc_one")
        self.assertEqual(qobj_kernels[0].params["test_params"], True)

    def test_assemble_with_unequal_kernels(self):
        """Test that assembly works with incorrect number of discriminators for
        number of qubits."""
        disc_one = Kernel("disc_one", test_params=True)
        disc_two = Kernel("disc_two", test_params=False)

        with self.assertWarns(DeprecationWarning):
            schedule = Schedule()
            schedule += Acquire(5, AcquireChannel(0), MemorySlot(0), kernel=disc_one)
            schedule += Acquire(5, AcquireChannel(1), MemorySlot(1), kernel=disc_two)
            schedule += Acquire(5, AcquireChannel(2), MemorySlot(2))

        with self.assertRaises(QiskitError), self.assertWarns(DeprecationWarning):
            assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0, 1, 2]],
            )

    def test_assemble_single_instruction(self):
        """Test assembling schedules, no lo config."""
        with self.assertWarns(DeprecationWarning):
            inst = pulse.Play(pulse.Constant(100, 1.0), pulse.DriveChannel(0))
            self.assertIsInstance(assemble(inst, self.backend), PulseQobj)

    def test_assemble_overlapping_time(self):
        """Test that assembly errors when qubits are measured in overlapping time."""
        with self.assertWarns(DeprecationWarning):
            schedule = Schedule()
            schedule = schedule.append(
                Acquire(5, AcquireChannel(0), MemorySlot(0)),
            )
            schedule = schedule.append(
                Acquire(5, AcquireChannel(1), MemorySlot(1)) << 1,
            )
        with self.assertRaises(QiskitError):
            with self.assertWarns(DeprecationWarning):
                assemble(
                    schedule,
                    qubit_lo_freq=self.default_qubit_lo_freq,
                    meas_lo_freq=self.default_meas_lo_freq,
                    meas_map=[[0, 1]],
                )

    def test_assemble_meas_map_vs_insts(self):
        """Test that assembly errors when the qubits are measured in overlapping time
        and qubits are not in the first meas_map list."""
        with self.assertWarns(DeprecationWarning):
            schedule = Schedule()
            schedule += Acquire(5, AcquireChannel(0), MemorySlot(0))
            schedule += Acquire(5, AcquireChannel(1), MemorySlot(1))
            schedule += Acquire(5, AcquireChannel(2), MemorySlot(2)) << 2
            schedule += Acquire(5, AcquireChannel(3), MemorySlot(3)) << 2

        with self.assertRaises(QiskitError):
            with self.assertWarns(DeprecationWarning):
                assemble(
                    schedule,
                    qubit_lo_freq=self.default_qubit_lo_freq,
                    meas_lo_freq=self.default_meas_lo_freq,
                    meas_map=[[0], [1, 2], [3]],
                )

    def test_assemble_non_overlapping_time_single_meas_map(self):
        """Test that assembly works when qubits are measured in non-overlapping
        time within the same measurement map list."""
        with self.assertWarns(DeprecationWarning):
            schedule = Schedule()
            schedule = schedule.append(
                Acquire(5, AcquireChannel(0), MemorySlot(0)),
            )
            schedule = schedule.append(
                Acquire(5, AcquireChannel(1), MemorySlot(1)) << 5,
            )
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0, 1]],
            )
        self.assertIsInstance(qobj, PulseQobj)

    def test_assemble_disjoint_time(self):
        """Test that assembly works when qubits are in disjoint meas map sets."""
        with self.assertWarns(DeprecationWarning):
            schedule = Schedule()
            schedule = schedule.append(
                Acquire(5, AcquireChannel(0), MemorySlot(0)),
            )
            schedule = schedule.append(
                Acquire(5, AcquireChannel(1), MemorySlot(1)) << 1,
            )
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0, 2], [1, 3]],
            )
        self.assertIsInstance(qobj, PulseQobj)

    def test_assemble_valid_qubits(self):
        """Test that assembly works when qubits that are in the measurement map
        is measured."""
        with self.assertWarns(DeprecationWarning):
            schedule = Schedule()
            schedule = schedule.append(
                Acquire(5, AcquireChannel(1), MemorySlot(1)),
            )
            schedule = schedule.append(
                Acquire(5, AcquireChannel(2), MemorySlot(2)),
            )
            schedule = schedule.append(
                Acquire(5, AcquireChannel(3), MemorySlot(3)),
            )
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                schedule,
                qubit_lo_freq=self.default_qubit_lo_freq,
                meas_lo_freq=self.default_meas_lo_freq,
                meas_map=[[0, 1, 2], [3]],
            )
        self.assertIsInstance(qobj, PulseQobj)


class TestPulseAssemblerMissingKwargs(QiskitTestCase):
    """Verify that errors are raised in case backend is not provided and kwargs are missing."""

    def setUp(self):
        super().setUp()
        with self.assertWarns(DeprecationWarning):
            self.schedule = pulse.Schedule(name="fake_experiment")
            self.backend = FakeOpenPulse2Q()

        self.config = self.backend.configuration()
        self.defaults = self.backend.defaults()
        self.qubit_lo_freq = list(self.defaults.qubit_freq_est)
        self.meas_lo_freq = list(self.defaults.meas_freq_est)
        self.qubit_lo_range = self.config.qubit_lo_range
        self.meas_lo_range = self.config.meas_lo_range
        with self.assertWarns(DeprecationWarning):
            self.schedule_los = {
                pulse.DriveChannel(0): self.qubit_lo_freq[0],
                pulse.DriveChannel(1): self.qubit_lo_freq[1],
                pulse.MeasureChannel(0): self.meas_lo_freq[0],
                pulse.MeasureChannel(1): self.meas_lo_freq[1],
            }
        self.meas_map = self.config.meas_map
        self.memory_slots = self.config.n_qubits

        # default rep_time and rep_delay
        self.rep_time = self.config.rep_times[0]
        self.rep_delay = None

    def test_defaults(self):
        """Test defaults work."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.schedule,
                qubit_lo_freq=self.qubit_lo_freq,
                meas_lo_freq=self.meas_lo_freq,
                qubit_lo_range=self.qubit_lo_range,
                meas_lo_range=self.meas_lo_range,
                schedule_los=self.schedule_los,
                meas_map=self.meas_map,
                memory_slots=self.memory_slots,
                rep_time=self.rep_time,
                rep_delay=self.rep_delay,
            )
        self.assertIsInstance(qobj, PulseQobj)

    def test_missing_qubit_lo_freq(self):
        """Test error raised if qubit_lo_freq missing."""

        with self.assertRaises(QiskitError), self.assertWarns(DeprecationWarning):

            assemble(
                self.schedule,
                qubit_lo_freq=None,
                meas_lo_freq=self.meas_lo_freq,
                qubit_lo_range=self.qubit_lo_range,
                meas_lo_range=self.meas_lo_range,
                meas_map=self.meas_map,
                memory_slots=self.memory_slots,
                rep_time=self.rep_time,
                rep_delay=self.rep_delay,
            )

    def test_missing_meas_lo_freq(self):
        """Test error raised if meas_lo_freq missing."""

        with self.assertRaises(QiskitError), self.assertWarns(DeprecationWarning):
            assemble(
                self.schedule,
                qubit_lo_freq=self.qubit_lo_freq,
                meas_lo_freq=None,
                qubit_lo_range=self.qubit_lo_range,
                meas_lo_range=self.meas_lo_range,
                meas_map=self.meas_map,
                memory_slots=self.memory_slots,
                rep_time=self.rep_time,
                rep_delay=self.rep_delay,
            )

    def test_missing_memory_slots(self):
        """Test error is not raised if memory_slots are missing."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.schedule,
                qubit_lo_freq=self.qubit_lo_freq,
                meas_lo_freq=self.meas_lo_freq,
                qubit_lo_range=self.qubit_lo_range,
                meas_lo_range=self.meas_lo_range,
                schedule_los=self.schedule_los,
                meas_map=self.meas_map,
                memory_slots=None,
                rep_time=self.rep_time,
                rep_delay=self.rep_delay,
            )
        self.assertIsInstance(qobj, PulseQobj)

    def test_missing_rep_time_and_delay(self):
        """Test qobj is valid if rep_time and rep_delay are missing."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.schedule,
                qubit_lo_freq=self.qubit_lo_freq,
                meas_lo_freq=self.meas_lo_freq,
                qubit_lo_range=self.qubit_lo_range,
                meas_lo_range=self.meas_lo_range,
                schedule_los=self.schedule_los,
                meas_map=self.meas_map,
                memory_slots=None,
                rep_time=None,
                rep_delay=None,
            )
        self.assertEqual(hasattr(qobj, "rep_time"), False)
        self.assertEqual(hasattr(qobj, "rep_delay"), False)

    def test_missing_meas_map(self):
        """Test that assembly still works if meas_map is missing."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.schedule,
                qubit_lo_freq=self.qubit_lo_freq,
                meas_lo_freq=self.meas_lo_freq,
                qubit_lo_range=self.qubit_lo_range,
                meas_lo_range=self.meas_lo_range,
                schedule_los=self.schedule_los,
                meas_map=None,
                memory_slots=self.memory_slots,
                rep_time=self.rep_time,
                rep_delay=self.rep_delay,
            )
        self.assertIsInstance(qobj, PulseQobj)

    def test_missing_lo_ranges(self):
        """Test that assembly still works if lo_ranges are missing."""
        with self.assertWarns(DeprecationWarning):
            qobj = assemble(
                self.schedule,
                qubit_lo_freq=self.qubit_lo_freq,
                meas_lo_freq=self.meas_lo_freq,
                qubit_lo_range=None,
                meas_lo_range=None,
                schedule_los=self.schedule_los,
                meas_map=self.meas_map,
                memory_slots=self.memory_slots,
                rep_time=self.rep_time,
                rep_delay=self.rep_delay,
            )
        self.assertIsInstance(qobj, PulseQobj)

    def test_unsupported_meas_level(self):
        """Test that assembly raises an error if meas_level is not supported"""
        with self.assertWarns(DeprecationWarning):
            backend = FakeOpenPulse2Q()
        backend.configuration().meas_levels = [1, 2]
        with self.assertRaises(QiskitError), self.assertWarns(DeprecationWarning):
            assemble(
                self.schedule,
                backend,
                qubit_lo_freq=self.qubit_lo_freq,
                meas_lo_freq=self.meas_lo_freq,
                qubit_lo_range=self.qubit_lo_range,
                meas_lo_range=self.meas_lo_range,
                schedule_los=self.schedule_los,
                meas_level=0,
                meas_map=self.meas_map,
                memory_slots=self.memory_slots,
                rep_time=self.rep_time,
                rep_delay=self.rep_delay,
            )

    def test_single_and_deprecated_acquire_styles(self):
        """Test that acquires are identically combined with Acquires that take a single channel."""
        with self.assertWarns(DeprecationWarning):
            backend = FakeOpenPulse2Q()
            new_style_schedule = Schedule()
        acq_dur = 1200
        with self.assertWarns(DeprecationWarning):
            for i in range(2):
                new_style_schedule += Acquire(acq_dur, AcquireChannel(i), MemorySlot(i))

            deprecated_style_schedule = Schedule()
            for i in range(2):
                deprecated_style_schedule += Acquire(1200, AcquireChannel(i), MemorySlot(i))

        # The Qobj IDs will be different
        with self.assertWarns(DeprecationWarning):
            n_qobj = assemble(new_style_schedule, backend)
        n_qobj.qobj_id = None
        n_qobj.experiments[0].header.name = None
        with self.assertWarns(DeprecationWarning):
            d_qobj = assemble(deprecated_style_schedule, backend)
        d_qobj.qobj_id = None
        d_qobj.experiments[0].header.name = None
        self.assertEqual(n_qobj, d_qobj)

        assembled_acquire = n_qobj.experiments[0].instructions[0]
        self.assertEqual(assembled_acquire.qubits, [0, 1])
        self.assertEqual(assembled_acquire.memory_slot, [0, 1])


class StreamHandlerRaiseException(StreamHandler):
    """Handler class that will raise an exception on formatting errors."""

    def handleError(self, record):
        raise sys.exc_info()


class TestLogAssembler(QiskitTestCase):
    """Testing the log_assembly option."""

    def setUp(self):
        super().setUp()
        logger = getLogger()
        self.addCleanup(logger.setLevel, logger.level)
        logger.setLevel("DEBUG")
        self.output = io.StringIO()
        logger.addHandler(StreamHandlerRaiseException(self.output))
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def assertAssembleLog(self, log_msg):
        """Runs assemble and checks for logs containing specified message"""
        with self.assertWarns(DeprecationWarning):
            assemble(self.circuit, shots=2000, memory=True)
        self.output.seek(0)
        # Filter unrelated log lines
        output_lines = self.output.readlines()
        assembly_log_lines = [x for x in output_lines if log_msg in x]
        self.assertTrue(len(assembly_log_lines) == 1)

    def test_assembly_log_time(self):
        """Check Total Assembly Time is logged"""
        self.assertAssembleLog("Total Assembly Time")


if __name__ == "__main__":
    unittest.main(verbosity=2)

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the InstructionScheduleMap."""
import copy
import pickle

import numpy as np

from qiskit.circuit.library.standard_gates import U1Gate, U3Gate, CXGate, XGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse import (
    InstructionScheduleMap,
    Play,
    PulseError,
    Schedule,
    ScheduleBlock,
    Waveform,
    ShiftPhase,
    Constant,
)
from qiskit.pulse.calibration_entries import CalibrationPublisher
from qiskit.pulse.channels import DriveChannel
from qiskit.qobj import PulseQobjInstruction
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.providers.fake_provider import FakeOpenPulse2Q, Fake7QPulseV1
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestInstructionScheduleMap(QiskitTestCase):
    """Test the InstructionScheduleMap."""

    def test_add(self):
        """Test add, and that errors are raised when expected."""
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()

        inst_map.add("u1", 1, sched)
        inst_map.add("u1", 0, sched)

        self.assertIn("u1", inst_map.instructions)
        self.assertEqual(inst_map.qubits_with_instruction("u1"), [0, 1])
        self.assertTrue("u1" in inst_map.qubit_instructions(0))

        with self.assertRaises(PulseError):
            inst_map.add("u1", (), sched)
        with self.assertRaises(PulseError):
            inst_map.add("u1", 1, "not a schedule")

    def test_add_block(self):
        """Test add block, and that errors are raised when expected."""
        sched = ScheduleBlock()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()

        inst_map.add("u1", 1, sched)
        inst_map.add("u1", 0, sched)

        self.assertIn("u1", inst_map.instructions)
        self.assertEqual(inst_map.qubits_with_instruction("u1"), [0, 1])
        self.assertTrue("u1" in inst_map.qubit_instructions(0))

    def test_instructions(self):
        """Test `instructions`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("u1", 1, sched)
        inst_map.add("u3", 0, sched)

        instructions = inst_map.instructions
        for inst in ["u1", "u3"]:
            self.assertTrue(inst in instructions)

    def test_has(self):
        """Test `has` and `assert_has`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("u1", (0,), sched)
        inst_map.add("cx", [0, 1], sched)

        self.assertTrue(inst_map.has("u1", [0]))
        self.assertTrue(inst_map.has("cx", (0, 1)))
        with self.assertRaises(PulseError):
            inst_map.assert_has("dne", [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has("cx", 100)

    def test_has_from_mock(self):
        """Test `has` and `assert_has` from mock data."""
        with self.assertWarns(DeprecationWarning):
            backend = FakeOpenPulse2Q()
        inst_map = backend.defaults().instruction_schedule_map
        self.assertTrue(inst_map.has("u1", [0]))
        self.assertTrue(inst_map.has("cx", (0, 1)))
        self.assertTrue(inst_map.has("u3", 0))
        self.assertTrue(inst_map.has("measure", [0, 1]))
        self.assertFalse(inst_map.has("u1", [0, 1]))
        with self.assertRaises(PulseError):
            inst_map.assert_has("dne", [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has("cx", 100)

    def test_qubits_with_instruction(self):
        """Test `qubits_with_instruction`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("u1", (0,), sched)
        inst_map.add("u1", (1,), sched)
        inst_map.add("cx", [0, 1], sched)

        self.assertEqual(inst_map.qubits_with_instruction("u1"), [0, 1])
        self.assertEqual(inst_map.qubits_with_instruction("cx"), [(0, 1)])
        self.assertEqual(inst_map.qubits_with_instruction("none"), [])

    def test_qubit_instructions(self):
        """Test `qubit_instructions`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("u1", (0,), sched)
        inst_map.add("u1", (1,), sched)
        inst_map.add("cx", [0, 1], sched)

        self.assertEqual(inst_map.qubit_instructions(0), ["u1"])
        self.assertEqual(inst_map.qubit_instructions(1), ["u1"])
        self.assertEqual(inst_map.qubit_instructions((0, 1)), ["cx"])
        self.assertEqual(inst_map.qubit_instructions(10), [])

    def test_get(self):
        """Test `get`."""
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add("x", 0, sched)

        self.assertEqual(sched, inst_map.get("x", (0,)))

    def test_get_block(self):
        """Test `get` block."""
        sched = ScheduleBlock()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)), inplace=True)
        inst_map = InstructionScheduleMap()
        inst_map.add("x", 0, sched)

        self.assertEqual(sched, inst_map.get("x", (0,)))

    def test_remove(self):
        """Test removing a defined operation and removing an undefined operation."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("tmp", 0, sched)
        inst_map.remove("tmp", 0)
        self.assertFalse(inst_map.has("tmp", 0))
        with self.assertRaises(PulseError):
            inst_map.remove("not_there", (0,))
        self.assertFalse("tmp" in inst_map.qubit_instructions(0))

    def test_pop(self):
        """Test pop with default."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("tmp", 100, sched)
        self.assertEqual(inst_map.pop("tmp", 100), sched)
        self.assertFalse(inst_map.has("tmp", 100))

        self.assertEqual(inst_map.qubit_instructions(100), [])
        self.assertEqual(inst_map.qubits_with_instruction("tmp"), [])
        with self.assertRaises(PulseError):
            inst_map.pop("not_there", (0,))

    def test_add_gate(self):
        """Test add, and that errors are raised when expected."""
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)))
        inst_map = InstructionScheduleMap()

        inst_map.add(U1Gate(0), 1, sched)
        inst_map.add(U1Gate(0), 0, sched)

        self.assertIn("u1", inst_map.instructions)
        self.assertEqual(inst_map.qubits_with_instruction(U1Gate(0)), [0, 1])
        self.assertTrue("u1" in inst_map.qubit_instructions(0))

        with self.assertRaises(PulseError):
            inst_map.add(U1Gate(0), (), sched)
        with self.assertRaises(PulseError):
            inst_map.add(U1Gate(0), 1, "not a schedule")

    def test_instructions_gate(self):
        """Test `instructions`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add(U1Gate(0), 1, sched)
        inst_map.add(U3Gate(0, 0, 0), 0, sched)

        instructions = inst_map.instructions
        for inst in ["u1", "u3"]:
            self.assertTrue(inst in instructions)

    def test_has_gate(self):
        """Test `has` and `assert_has`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add(U1Gate(0), (0,), sched)
        inst_map.add(CXGate(), [0, 1], sched)

        self.assertTrue(inst_map.has(U1Gate(0), [0]))
        self.assertTrue(inst_map.has(CXGate(), (0, 1)))
        with self.assertRaises(PulseError):
            inst_map.assert_has("dne", [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has(CXGate(), 100)

    def test_has_from_mock_gate(self):
        """Test `has` and `assert_has` from mock data."""
        with self.assertWarns(DeprecationWarning):
            backend = FakeOpenPulse2Q()
        inst_map = backend.defaults().instruction_schedule_map
        self.assertTrue(inst_map.has(U1Gate(0), [0]))
        self.assertTrue(inst_map.has(CXGate(), (0, 1)))
        self.assertTrue(inst_map.has(U3Gate(0, 0, 0), 0))
        self.assertTrue(inst_map.has("measure", [0, 1]))
        self.assertFalse(inst_map.has(U1Gate(0), [0, 1]))
        with self.assertRaises(PulseError):
            inst_map.assert_has("dne", [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has(CXGate(), 100)

    def test_qubits_with_instruction_gate(self):
        """Test `qubits_with_instruction`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add(U1Gate(0), (0,), sched)
        inst_map.add(U1Gate(0), (1,), sched)
        inst_map.add(CXGate(), [0, 1], sched)

        self.assertEqual(inst_map.qubits_with_instruction(U1Gate(0)), [0, 1])
        self.assertEqual(inst_map.qubits_with_instruction(CXGate()), [(0, 1)])
        self.assertEqual(inst_map.qubits_with_instruction("none"), [])

    def test_qubit_instructions_gate(self):
        """Test `qubit_instructions`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add(U1Gate(0), (0,), sched)
        inst_map.add(U1Gate(0), (1,), sched)
        inst_map.add(CXGate(), [0, 1], sched)

        self.assertEqual(inst_map.qubit_instructions(0), ["u1"])
        self.assertEqual(inst_map.qubit_instructions(1), ["u1"])
        self.assertEqual(inst_map.qubit_instructions((0, 1)), ["cx"])
        self.assertEqual(inst_map.qubit_instructions(10), [])

    def test_get_gate(self):
        """Test `get`."""
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)))
        inst_map = InstructionScheduleMap()
        inst_map.add(XGate(), 0, sched)

        self.assertEqual(sched, inst_map.get(XGate(), (0,)))

    def test_remove_gate(self):
        """Test removing a defined operation and removing an undefined operation."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add("tmp", 0, sched)
        inst_map.remove("tmp", 0)
        self.assertFalse(inst_map.has("tmp", 0))
        with self.assertRaises(PulseError):
            inst_map.remove("not_there", (0,))
        self.assertFalse("tmp" in inst_map.qubit_instructions(0))

    def test_pop_gate(self):
        """Test pop with default."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add(XGate(), 100, sched)
        self.assertEqual(inst_map.pop(XGate(), 100), sched)
        self.assertFalse(inst_map.has(XGate(), 100))

        self.assertEqual(inst_map.qubit_instructions(100), [])
        self.assertEqual(inst_map.qubits_with_instruction(XGate()), [])
        with self.assertRaises(PulseError):
            inst_map.pop("not_there", (0,))

    def test_sequenced_parameterized_schedule(self):
        """Test parameterized schedule consists of multiple instruction."""

        with self.assertWarns(DeprecationWarning):
            converter = QobjToInstructionConverter([], buffer=0)
            qobjs = [
                PulseQobjInstruction(name="fc", ch="d0", t0=10, phase="P1"),
                PulseQobjInstruction(name="fc", ch="d0", t0=20, phase="P2"),
                PulseQobjInstruction(name="fc", ch="d0", t0=30, phase="P3"),
            ]
        converted_instruction = [converter(qobj) for qobj in qobjs]

        inst_map = InstructionScheduleMap()

        inst_map.add("inst_seq", 0, Schedule(*converted_instruction, name="inst_seq"))

        with self.assertRaises(PulseError):
            inst_map.get("inst_seq", 0, P1=1, P2=2, P3=3, P4=4, P5=5)

        with self.assertRaises(PulseError):
            inst_map.get("inst_seq", 0, 1, 2, 3, 4, 5, 6, 7, 8)

        p3_expr = Parameter("p3")
        p3_expr = p3_expr.bind({p3_expr: 3})

        sched = inst_map.get("inst_seq", 0, 1, 2, p3_expr)
        self.assertEqual(sched.instructions[0][-1].phase, 1)
        self.assertEqual(sched.instructions[1][-1].phase, 2)
        self.assertEqual(sched.instructions[2][-1].phase, 3)

        sched = inst_map.get("inst_seq", 0, P1=1, P2=2, P3=p3_expr)
        self.assertEqual(sched.instructions[0][-1].phase, 1)
        self.assertEqual(sched.instructions[1][-1].phase, 2)
        self.assertEqual(sched.instructions[2][-1].phase, 3)

        sched = inst_map.get("inst_seq", 0, 1, 2, P3=p3_expr)
        self.assertEqual(sched.instructions[0][-1].phase, 1)
        self.assertEqual(sched.instructions[1][-1].phase, 2)
        self.assertEqual(sched.instructions[2][-1].phase, 3)

    def test_schedule_generator(self):
        """Test schedule generator functionality."""

        dur_val = 10
        amp = 1.0

        def test_func(dur: int):
            sched = Schedule()
            waveform = Constant(int(dur), amp).get_waveform()
            sched += Play(waveform, DriveChannel(0))
            return sched

        expected_sched = Schedule()
        cons_waveform = Constant(dur_val, amp).get_waveform()
        expected_sched += Play(cons_waveform, DriveChannel(0))

        inst_map = InstructionScheduleMap()
        inst_map.add("f", (0,), test_func)
        self.assertEqual(inst_map.get("f", (0,), dur_val), expected_sched)

        self.assertEqual(inst_map.get_parameters("f", (0,)), ("dur",))

    def test_schedule_generator_supports_parameter_expressions(self):
        """Test expression-based schedule generator functionality."""

        t_param = Parameter("t")
        amp = 1.0

        def test_func(dur: ParameterExpression, t_val: int):
            dur_bound = dur.bind({t_param: t_val})
            sched = Schedule()
            waveform = Constant(int(float(dur_bound)), amp).get_waveform()
            sched += Play(waveform, DriveChannel(0))
            return sched

        expected_sched = Schedule()
        cons_waveform = Constant(10, amp).get_waveform()
        expected_sched += Play(cons_waveform, DriveChannel(0))

        inst_map = InstructionScheduleMap()
        inst_map.add("f", (0,), test_func)
        self.assertEqual(inst_map.get("f", (0,), dur=2 * t_param, t_val=5), expected_sched)

        self.assertEqual(
            inst_map.get_parameters("f", (0,)),
            (
                "dur",
                "t_val",
            ),
        )

    def test_schedule_with_non_alphanumeric_ordering(self):
        """Test adding and getting schedule with non obvious parameter ordering."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        lamb = Parameter("lam")

        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(theta, DriveChannel(0)), inplace=True)
        target_sched.insert(10, ShiftPhase(phi, DriveChannel(0)), inplace=True)
        target_sched.insert(20, ShiftPhase(lamb, DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()
        inst_map.add("target_sched", (0,), target_sched, arguments=["theta", "phi", "lam"])

        ref_sched = Schedule()
        ref_sched.insert(0, ShiftPhase(0, DriveChannel(0)), inplace=True)
        ref_sched.insert(10, ShiftPhase(1, DriveChannel(0)), inplace=True)
        ref_sched.insert(20, ShiftPhase(2, DriveChannel(0)), inplace=True)

        # if parameter is alphanumerical ordering this maps to
        # theta -> 2
        # phi -> 1
        # lamb -> 0
        # however non alphanumerical ordering is specified in add method thus mapping should be
        # theta -> 0
        # phi -> 1
        # lamb -> 2
        test_sched = inst_map.get("target_sched", (0,), 0, 1, 2)

        for test_inst, ref_inst in zip(test_sched.instructions, ref_sched.instructions):
            self.assertEqual(test_inst[0], ref_inst[0])
            self.assertEqual(test_inst[1], ref_inst[1])

    def test_binding_too_many_parameters(self):
        """Test getting schedule with too many parameter binding."""
        param = Parameter("param")

        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(param, DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()
        inst_map.add("target_sched", (0,), target_sched)

        with self.assertRaises(PulseError):
            inst_map.get("target_sched", (0,), 0, 1, 2, 3)

    def test_binding_unassigned_parameters(self):
        """Test getting schedule with unassigned parameter binding."""
        param = Parameter("param")

        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(param, DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()
        inst_map.add("target_sched", (0,), target_sched)

        with self.assertRaises(PulseError):
            inst_map.get("target_sched", (0,), P0=0)

    def test_schedule_with_multiple_parameters_under_same_name(self):
        """Test getting schedule with parameters that have the same name."""
        param1 = Parameter("param")
        param2 = Parameter("param")
        param3 = Parameter("param")

        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(param1, DriveChannel(0)), inplace=True)
        target_sched.insert(10, ShiftPhase(param2, DriveChannel(0)), inplace=True)
        target_sched.insert(20, ShiftPhase(param3, DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()
        inst_map.add("target_sched", (0,), target_sched)

        ref_sched = Schedule()
        ref_sched.insert(0, ShiftPhase(1.23, DriveChannel(0)), inplace=True)
        ref_sched.insert(10, ShiftPhase(1.23, DriveChannel(0)), inplace=True)
        ref_sched.insert(20, ShiftPhase(1.23, DriveChannel(0)), inplace=True)

        test_sched = inst_map.get("target_sched", (0,), param=1.23)

        for test_inst, ref_inst in zip(test_sched.instructions, ref_sched.instructions):
            self.assertEqual(test_inst[0], ref_inst[0])
            self.assertAlmostEqual(test_inst[1], ref_inst[1])

    def test_get_schedule_with_unbound_parameter(self):
        """Test get schedule with partial binding."""
        param1 = Parameter("param1")
        param2 = Parameter("param2")

        target_sched = Schedule()
        target_sched.insert(0, ShiftPhase(param1, DriveChannel(0)), inplace=True)
        target_sched.insert(10, ShiftPhase(param2, DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()
        inst_map.add("target_sched", (0,), target_sched)

        ref_sched = Schedule()
        ref_sched.insert(0, ShiftPhase(param1, DriveChannel(0)), inplace=True)
        ref_sched.insert(10, ShiftPhase(1.23, DriveChannel(0)), inplace=True)

        test_sched = inst_map.get("target_sched", (0,), param2=1.23)

        for test_inst, ref_inst in zip(test_sched.instructions, ref_sched.instructions):
            self.assertEqual(test_inst[0], ref_inst[0])
            self.assertAlmostEqual(test_inst[1], ref_inst[1])

    def test_partially_bound_callable(self):
        """Test register partial function."""
        import functools

        def callable_schedule(par_b, par_a):
            sched = Schedule()
            sched.insert(10, Play(Constant(10, par_b), DriveChannel(0)), inplace=True)
            sched.insert(20, Play(Constant(10, par_a), DriveChannel(0)), inplace=True)
            return sched

        ref_sched = Schedule()
        ref_sched.insert(10, Play(Constant(10, 0.1), DriveChannel(0)), inplace=True)
        ref_sched.insert(20, Play(Constant(10, 0.2), DriveChannel(0)), inplace=True)

        inst_map = InstructionScheduleMap()

        def test_callable_sched1(par_b):
            return callable_schedule(par_b, 0.2)

        inst_map.add("my_gate1", (0,), test_callable_sched1, ["par_b"])
        ret_sched = inst_map.get("my_gate1", (0,), par_b=0.1)
        self.assertEqual(ret_sched, ref_sched)

        # bind partially
        test_callable_sched2 = functools.partial(callable_schedule, par_a=0.2)

        inst_map.add("my_gate2", (0,), test_callable_sched2, ["par_b"])
        ret_sched = inst_map.get("my_gate2", (0,), par_b=0.1)
        self.assertEqual(ret_sched, ref_sched)

    def test_two_instmaps_equal(self):
        """Test eq method when two instmaps are identical."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake7QPulseV1()
            instmap1 = backend.defaults().instruction_schedule_map
        instmap2 = copy.deepcopy(instmap1)

        self.assertEqual(instmap1, instmap2)

    def test_two_instmaps_different(self):
        """Test eq method when two instmaps are not identical."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake7QPulseV1()
            instmap1 = backend.defaults().instruction_schedule_map
        instmap2 = copy.deepcopy(instmap1)

        # override one of instruction
        instmap2.add("sx", (0,), Schedule())

        self.assertNotEqual(instmap1, instmap2)

    def test_instmap_picklable(self):
        """Test if instmap can be pickled."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake7QPulseV1()
            instmap = backend.defaults().instruction_schedule_map

        ser_obj = pickle.dumps(instmap)
        deser_instmap = pickle.loads(ser_obj)

        self.assertEqual(instmap, deser_instmap)

    def test_instmap_picklable_with_arguments(self):
        """Test instmap pickling with an edge case.

        This test attempts to pickle instmap with custom entry,
        in which arguments are provided by users in the form of
        python dict key object that is not picklable.
        """
        with self.assertWarns(DeprecationWarning):
            backend = Fake7QPulseV1()
            instmap = backend.defaults().instruction_schedule_map

        param1 = Parameter("P1")
        param2 = Parameter("P2")
        sched = Schedule()
        sched.insert(0, Play(Constant(100, param1), DriveChannel(0)), inplace=True)
        sched.insert(0, Play(Constant(100, param2), DriveChannel(1)), inplace=True)
        to_assign = {"P1": 0.1, "P2": 0.2}

        # Note that dict keys is not picklable
        # Instmap should typecast it into list to pickle itself.
        instmap.add("custom", (0, 1), sched, arguments=to_assign.keys())

        ser_obj = pickle.dumps(instmap)
        deser_instmap = pickle.loads(ser_obj)

        self.assertEqual(instmap, deser_instmap)

    def test_check_backend_provider_cals(self):
        """Test if schedules provided by backend provider is distinguishable."""
        with self.assertWarns(DeprecationWarning):
            backend = FakeOpenPulse2Q()
            instmap = backend.defaults().instruction_schedule_map
        publisher = instmap.get("u1", (0,), P0=0).metadata["publisher"]

        self.assertEqual(publisher, CalibrationPublisher.BACKEND_PROVIDER)

    def test_check_user_cals(self):
        """Test if schedules provided by user is distinguishable."""
        with self.assertWarns(DeprecationWarning):
            backend = FakeOpenPulse2Q()
            instmap = backend.defaults().instruction_schedule_map

        test_u1 = Schedule()
        test_u1 += ShiftPhase(Parameter("P0"), DriveChannel(0))

        instmap.add("u1", (0,), test_u1, arguments=["P0"])
        publisher = instmap.get("u1", (0,), P0=0).metadata["publisher"]

        self.assertEqual(publisher, CalibrationPublisher.QISKIT)

    def test_has_custom_gate(self):
        """Test method to check custom gate."""
        with self.assertWarns(DeprecationWarning):
            backend = FakeOpenPulse2Q()
            instmap = backend.defaults().instruction_schedule_map

        self.assertFalse(instmap.has_custom_gate())

        # add custom schedule
        some_sched = Schedule()
        instmap.add("u3", (0,), some_sched)

        self.assertTrue(instmap.has_custom_gate())

        # delete custom schedule
        instmap.remove("u3", (0,))
        self.assertFalse(instmap.has_custom_gate())

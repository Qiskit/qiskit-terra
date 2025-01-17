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

"""Test schedule block subroutine reference mechanism."""

import numpy as np

from qiskit import circuit, pulse
from qiskit.pulse import ScheduleBlock, builder
from qiskit.pulse.transforms import inline_subroutines
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestReference(QiskitTestCase):
    """Test for basic behavior of reference mechanism."""

    def test_append_schedule(self):
        """Test appending schedule without calling.

        Appended schedules are not subroutines.
        These are directly exposed to the outer block.
        """
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        with pulse.build() as sched_y1:
            builder.append_schedule(sched_x1)

        with pulse.build() as sched_z1:
            builder.append_schedule(sched_y1)

        self.assertEqual(len(sched_z1.references), 0)

    def test_refer_schedule(self):
        """Test refer to schedule by name.

        Outer block is only aware of its inner reference.
        Nested reference is not directly exposed to the most outer block.
        """
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        with pulse.build() as sched_y1:
            builder.reference("x1", "d0")

        with pulse.build() as sched_z1:
            builder.reference("y1", "d0")

        sched_y1.assign_references({("x1", "d0"): sched_x1})
        sched_z1.assign_references({("y1", "d0"): sched_y1})

        self.assertEqual(len(sched_z1.references), 1)
        self.assertEqual(sched_z1.references[("y1", "d0")], sched_y1)

        self.assertEqual(len(sched_y1.references), 1)
        self.assertEqual(sched_y1.references[("x1", "d0")], sched_x1)

    def test_refer_schedule_parameter_scope(self):
        """Test refer to schedule by name.

        Parameter in the called schedule has the scope of called schedule.
        """
        param = circuit.Parameter("name")

        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param, name="x1"), pulse.DriveChannel(0))

        with pulse.build() as sched_y1:
            builder.reference("x1", "d0")

        with pulse.build() as sched_z1:
            builder.reference("y1", "d0")

        sched_y1.assign_references({("x1", "d0"): sched_x1})
        sched_z1.assign_references({("y1", "d0"): sched_y1})

        self.assertEqual(sched_z1.parameters, sched_x1.parameters)
        self.assertEqual(sched_z1.parameters, sched_y1.parameters)

    def test_refer_schedule_parameter_assignment(self):
        """Test assigning to parameter in referenced schedule"""
        param = circuit.Parameter("name")

        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param, name="x1"), pulse.DriveChannel(0))

        with pulse.build() as sched_y1:
            builder.reference("x1", "d0")

        with pulse.build() as sched_z1:
            builder.reference("y1", "d0")

        sched_y1.assign_references({("x1", "d0"): sched_x1})
        sched_z1.assign_references({("y1", "d0"): sched_y1})

        assigned_z1 = sched_z1.assign_parameters({param: 0.5}, inplace=False)

        assigned_x1 = sched_x1.assign_parameters({param: 0.5}, inplace=False)
        ref_assigned_y1 = ScheduleBlock()
        ref_assigned_y1.append(assigned_x1)
        ref_assigned_z1 = ScheduleBlock()
        ref_assigned_z1.append(ref_assigned_y1)

        # Test that assignment was successful and resolved references
        self.assertEqual(assigned_z1, ref_assigned_z1)

        # Test that inplace=False for sched_z1 also did not modify sched_z1 or  subroutine sched_x1
        self.assertEqual(sched_z1.parameters, {param})
        self.assertEqual(sched_x1.parameters, {param})
        self.assertEqual(assigned_z1.parameters, set())

        # Now test inplace=True
        sched_z1.assign_parameters({param: 0.5}, inplace=True)
        self.assertEqual(sched_z1, assigned_z1)
        # assign_references copies the subroutine, so the original subschedule
        # is still not modified here:
        self.assertNotEqual(sched_x1, assigned_x1)

    def test_call_schedule(self):
        """Test call schedule.

        Outer block is only aware of its inner reference.
        Nested reference is not directly exposed to the most outer block.
        """
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        with pulse.build() as sched_y1:
            builder.call(sched_x1, name="x1")

        with pulse.build() as sched_z1:
            builder.call(sched_y1, name="y1")

        self.assertEqual(len(sched_z1.references), 1)
        self.assertEqual(sched_z1.references[("y1",)], sched_y1)

        self.assertEqual(len(sched_y1.references), 1)
        self.assertEqual(sched_y1.references[("x1",)], sched_x1)

    def test_call_schedule_parameter_scope(self):
        """Test call schedule.

        Parameter in the called schedule has the scope of called schedule.
        """
        param = circuit.Parameter("name")

        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param, name="x1"), pulse.DriveChannel(0))

        with pulse.build() as sched_y1:
            builder.call(sched_x1, name="x1")

        with pulse.build() as sched_z1:
            builder.call(sched_y1, name="y1")

        self.assertEqual(sched_z1.parameters, sched_x1.parameters)
        self.assertEqual(sched_z1.parameters, sched_y1.parameters)

    def test_append_and_call_schedule(self):
        """Test call and append schedule.

        Reference is copied to the outer schedule by appending.
        Original reference remains unchanged.
        """
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        with pulse.build() as sched_y1:
            builder.call(sched_x1, name="x1")

        with pulse.build() as sched_z1:
            builder.append_schedule(sched_y1)

        self.assertEqual(len(sched_z1.references), 1)
        self.assertEqual(sched_z1.references[("x1",)], sched_x1)

        # blocks[0] is sched_y1 and its reference is now point to outer block reference
        self.assertIs(sched_z1.blocks[0].references, sched_z1.references)

        # however the original program is protected to prevent unexpected mutation
        self.assertIsNot(sched_y1.references, sched_z1.references)

        # appended schedule is preserved
        self.assertEqual(len(sched_y1.references), 1)
        self.assertEqual(sched_y1.references[("x1",)], sched_x1)

    def test_calling_similar_schedule(self):
        """Test calling schedules with the same representation.

        sched_x1 and sched_y1 are the different subroutines, but same representation.
        Two references should be created.
        """
        param1 = circuit.Parameter("param")
        param2 = circuit.Parameter("param")

        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param1, name="p"), pulse.DriveChannel(0))

        with pulse.build() as sched_y1:
            pulse.play(pulse.Constant(100, param2, name="p"), pulse.DriveChannel(0))

        with pulse.build() as sched_z1:
            pulse.call(sched_x1)
            pulse.call(sched_y1)

        self.assertEqual(len(sched_z1.references), 2)

    def test_calling_same_schedule(self):
        """Test calling same schedule twice.

        Because it calls the same schedule, no duplication should occur in reference table.
        """
        param = circuit.Parameter("param")

        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param, name="x1"), pulse.DriveChannel(0))

        with pulse.build() as sched_z1:
            pulse.call(sched_x1, name="same_sched")
            pulse.call(sched_x1, name="same_sched")

        self.assertEqual(len(sched_z1.references), 1)

    def test_calling_same_schedule_with_different_assignment(self):
        """Test calling same schedule twice but with different parameters.

        Same schedule is called twice but with different assignment.
        Two references should be created.
        """
        param = circuit.Parameter("param")

        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param, name="x1"), pulse.DriveChannel(0))

        with pulse.build() as sched_z1:
            pulse.call(sched_x1, param=0.1)
            pulse.call(sched_x1, param=0.2)

        self.assertEqual(len(sched_z1.references), 2)

    def test_alignment_context(self):
        """Test nested alignment context.

        Inline alignment is identical to append_schedule operation.
        Thus scope is not newly generated.
        """
        with pulse.build(name="x1") as sched_x1:
            with pulse.align_right():
                with pulse.align_left():
                    pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        self.assertEqual(len(sched_x1.references), 0)

    def test_appending_child_block(self):
        """Test for edge case.

        User can append blocks which is an element of another schedule block.
        But this is not standard use case.

        In this case, references may contain subroutines which don't exist in the context.
        This is because all references within the program are centrally
        managed in the most outer block.
        """
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        with pulse.build() as sched_y1:
            pulse.play(pulse.Constant(100, 0.2, name="y1"), pulse.DriveChannel(0))

        with pulse.build() as sched_x2:
            builder.call(sched_x1, name="x1")
        self.assertEqual(list(sched_x2.references.keys()), [("x1",)])

        with pulse.build() as sched_y2:
            builder.call(sched_y1, name="y1")
        self.assertEqual(list(sched_y2.references.keys()), [("y1",)])

        with pulse.build() as sched_z1:
            builder.append_schedule(sched_x2)
            builder.append_schedule(sched_y2)
        self.assertEqual(list(sched_z1.references.keys()), [("x1",), ("y1",)])

        # child block references point to its parent, i.e. sched_z1
        self.assertIs(sched_z1.blocks[0].references, sched_z1._reference_manager)
        self.assertIs(sched_z1.blocks[1].references, sched_z1._reference_manager)

        with pulse.build() as sched_z2:
            # Append child block
            # The reference of this block is sched_z1.reference thus it contains both x1 and y1.
            # However, y1 doesn't exist in the context, so only x1 should be added.

            # Usually, user will append sched_x2 directly here, rather than sched_z1.blocks[0]
            # This is why this situation is an edge case.
            builder.append_schedule(sched_z1.blocks[0])

        self.assertEqual(len(sched_z2.references), 1)
        self.assertEqual(sched_z2.references[("x1",)], sched_x1)

    def test_replacement(self):
        """Test nested alignment context.

        Replacing schedule block with schedule block.
        Removed block contains own reference, that should be removed with replacement.
        New block also contains reference, that should be passed to the current reference.
        """
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1, name="x1"), pulse.DriveChannel(0))

        with pulse.build() as sched_y1:
            pulse.play(pulse.Constant(100, 0.2, name="y1"), pulse.DriveChannel(0))

        with pulse.build() as sched_x2:
            builder.call(sched_x1, name="x1")

        with pulse.build() as sched_y2:
            builder.call(sched_y1, name="y1")

        with pulse.build() as sched_z1:
            builder.append_schedule(sched_x2)
            builder.append_schedule(sched_y2)
        self.assertEqual(len(sched_z1.references), 2)
        self.assertEqual(sched_z1.references[("x1",)], sched_x1)
        self.assertEqual(sched_z1.references[("y1",)], sched_y1)

        # Define schedule to replace
        with pulse.build() as sched_r1:
            pulse.play(pulse.Constant(100, 0.1, name="r1"), pulse.DriveChannel(0))

        with pulse.build() as sched_r2:
            pulse.call(sched_r1, name="r1")

        sched_z2 = sched_z1.replace(sched_x2, sched_r2)
        self.assertEqual(len(sched_z2.references), 2)
        self.assertEqual(sched_z2.references[("r1",)], sched_r1)
        self.assertEqual(sched_z2.references[("y1",)], sched_y1)

    def test_parameter_in_multiple_scope(self):
        """Test that using parameter in multiple scopes causes no error"""
        param = circuit.Parameter("name")

        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, param), pulse.DriveChannel(0))

        with pulse.build() as sched_y1:
            pulse.play(pulse.Constant(100, param), pulse.DriveChannel(1))

        with pulse.build() as sched_z1:
            pulse.call(sched_x1, name="x1")
            pulse.call(sched_y1, name="y1")

        self.assertEqual(len(sched_z1.parameters), 1)
        self.assertEqual(sched_z1.parameters, {param})

    def test_parallel_alignment_equality(self):
        """Testcase for potential edge case.

        In parallel alignment context, reference instruction is broadcasted to
        all channels. When new channel is added after reference, this should be
        connected with reference node.
        """

        with pulse.build() as subroutine:
            pulse.reference("unassigned")

        with pulse.build() as sched1:
            with pulse.align_left():
                pulse.delay(10, pulse.DriveChannel(0))
                pulse.call(subroutine)  # This should be broadcasted to d1 as well
                pulse.delay(10, pulse.DriveChannel(1))

        with pulse.build() as sched2:
            with pulse.align_left():
                pulse.delay(10, pulse.DriveChannel(0))
                pulse.delay(10, pulse.DriveChannel(1))
                pulse.call(subroutine)

        self.assertNotEqual(sched1, sched2)

    def test_subroutine_conflict(self):
        """Test for edge case of appending two schedule blocks having the
        references with conflicting reference key.

        This operation should fail because one of references will be gone after assignment.
        """
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1), pulse.DriveChannel(0))

        with pulse.build() as sched_x2:
            pulse.call(sched_x1, name="conflict_name")

        self.assertEqual(sched_x2.references[("conflict_name",)], sched_x1)

        with pulse.build() as sched_y1:
            pulse.play(pulse.Constant(100, 0.2), pulse.DriveChannel(0))

        with pulse.build() as sched_y2:
            pulse.call(sched_y1, name="conflict_name")

        self.assertEqual(sched_y2.references[("conflict_name",)], sched_y1)

        with self.assertRaises(pulse.exceptions.PulseError):
            with pulse.build():
                builder.append_schedule(sched_x2)
                builder.append_schedule(sched_y2)

    def test_assign_existing_reference(self):
        """Test for explicitly assign existing reference.

        This operation should fail because overriding reference is not allowed.
        """
        with pulse.build() as sched_x1:
            pulse.play(pulse.Constant(100, 0.1), pulse.DriveChannel(0))

        with pulse.build() as sched_y1:
            pulse.play(pulse.Constant(100, 0.2), pulse.DriveChannel(0))

        with pulse.build() as sched_z1:
            pulse.call(sched_x1, name="conflict_name")

        with self.assertRaises(pulse.exceptions.PulseError):
            sched_z1.assign_references({("conflict_name",): sched_y1})


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestSubroutineWithCXGate(QiskitTestCase):
    """Test called program scope with practical example of building fully parametrized CX gate."""

    @ignore_pulse_deprecation_warnings
    def setUp(self):
        super().setUp()

        # parameters of X pulse
        self.xp_dur = circuit.Parameter("dur")
        self.xp_amp = circuit.Parameter("amp")
        self.xp_sigma = circuit.Parameter("sigma")
        self.xp_beta = circuit.Parameter("beta")

        # amplitude of SX pulse
        self.sxp_amp = circuit.Parameter("amp")

        # parameters of CR pulse
        self.cr_dur = circuit.Parameter("dur")
        self.cr_amp = circuit.Parameter("amp")
        self.cr_sigma = circuit.Parameter("sigma")
        self.cr_risefall = circuit.Parameter("risefall")

        # channels
        self.control_ch = circuit.Parameter("ctrl")
        self.target_ch = circuit.Parameter("tgt")
        self.cr_ch = circuit.Parameter("cr")

        # echo pulse on control qubit
        with pulse.build(name="xp") as xp_sched_q0:
            pulse.play(
                pulse.Drag(
                    duration=self.xp_dur,
                    amp=self.xp_amp,
                    sigma=self.xp_sigma,
                    beta=self.xp_beta,
                ),
                channel=pulse.DriveChannel(self.control_ch),
            )
        self.xp_sched = xp_sched_q0

        # local rotation on target qubit
        with pulse.build(name="sx") as sx_sched_q1:
            pulse.play(
                pulse.Drag(
                    duration=self.xp_dur,
                    amp=self.sxp_amp,
                    sigma=self.xp_sigma,
                    beta=self.xp_beta,
                ),
                channel=pulse.DriveChannel(self.target_ch),
            )
        self.sx_sched = sx_sched_q1

        # cross resonance
        with pulse.build(name="cr") as cr_sched:
            pulse.play(
                pulse.GaussianSquare(
                    duration=self.cr_dur,
                    amp=self.cr_amp,
                    sigma=self.cr_sigma,
                    risefall_sigma_ratio=self.cr_risefall,
                ),
                channel=pulse.ControlChannel(self.cr_ch),
            )
        self.cr_sched = cr_sched

    def test_lazy_ecr(self):
        """Test lazy subroutines through ECR schedule construction."""

        with pulse.build(name="lazy_ecr") as sched:
            with pulse.align_sequential():
                pulse.reference("cr", "q0", "q1")
                pulse.reference("xp", "q0")
                with pulse.phase_offset(np.pi, pulse.ControlChannel(self.cr_ch)):
                    pulse.reference("cr", "q0", "q1")
                pulse.reference("xp", "q0")

        # Schedule has references
        self.assertTrue(sched.is_referenced())

        # Schedule is not schedulable because of unassigned references
        self.assertFalse(sched.is_schedulable())

        # Two references cr and xp are called
        self.assertEqual(len(sched.references), 2)

        # Parameters in the current scope are Parameter("cr") which is used in phase_offset
        # References are not assigned yet.
        params = {p.name for p in sched.parameters}
        self.assertSetEqual(params, {"cr"})

        # Assign CR and XP schedule to the empty reference
        sched.assign_references({("cr", "q0", "q1"): self.cr_sched})
        sched.assign_references({("xp", "q0"): self.xp_sched})

        # Check updated references
        assigned_refs = sched.references
        self.assertEqual(assigned_refs[("cr", "q0", "q1")], self.cr_sched)
        self.assertEqual(assigned_refs[("xp", "q0")], self.xp_sched)

        # Parameter added from subroutines
        ref_params = {self.cr_ch} | self.cr_sched.parameters | self.xp_sched.parameters
        self.assertSetEqual(sched.parameters, ref_params)

        # Get parameter without scope, cr amp and xp amp are hit.
        params = sched.get_parameters(parameter_name="amp")
        self.assertEqual(len(params), 2)

    def test_cnot(self):
        """Integration test with CNOT schedule construction."""
        # echoed cross resonance
        with pulse.build(name="ecr", default_alignment="sequential") as ecr_sched:
            pulse.call(self.cr_sched, name="cr")
            pulse.call(self.xp_sched, name="xp")
            with pulse.phase_offset(np.pi, pulse.ControlChannel(self.cr_ch)):
                pulse.call(self.cr_sched, name="cr")
            pulse.call(self.xp_sched, name="xp")

        # cnot gate, locally equivalent to ecr
        with pulse.build(name="cx", default_alignment="sequential") as cx_sched:
            pulse.shift_phase(np.pi / 2, pulse.DriveChannel(self.control_ch))
            pulse.call(self.sx_sched, name="sx")
            pulse.call(ecr_sched, name="ecr")

        # assign parameters
        assigned_cx = cx_sched.assign_parameters(
            value_dict={
                self.cr_ch: 0,
                self.control_ch: 0,
                self.target_ch: 1,
                self.sxp_amp: 0.1,
                self.xp_amp: 0.2,
                self.xp_dur: 160,
                self.xp_sigma: 40,
                self.xp_beta: 3.0,
                self.cr_amp: 0.5,
                self.cr_dur: 800,
                self.cr_sigma: 64,
                self.cr_risefall: 2,
            },
            inplace=True,
        )
        flatten_cx = inline_subroutines(assigned_cx)

        with pulse.build(default_alignment="sequential") as ref_cx:
            # sz
            pulse.shift_phase(np.pi / 2, pulse.DriveChannel(0))
            with pulse.align_left():
                # sx
                pulse.play(
                    pulse.Drag(
                        duration=160,
                        amp=0.1,
                        sigma=40,
                        beta=3.0,
                    ),
                    channel=pulse.DriveChannel(1),
                )
            with pulse.align_sequential():
                # cr
                with pulse.align_left():
                    pulse.play(
                        pulse.GaussianSquare(
                            duration=800,
                            amp=0.5,
                            sigma=64,
                            risefall_sigma_ratio=2,
                        ),
                        channel=pulse.ControlChannel(0),
                    )
                # xp
                with pulse.align_left():
                    pulse.play(
                        pulse.Drag(
                            duration=160,
                            amp=0.2,
                            sigma=40,
                            beta=3.0,
                        ),
                        channel=pulse.DriveChannel(0),
                    )
                with pulse.phase_offset(np.pi, pulse.ControlChannel(0)):
                    # cr
                    with pulse.align_left():
                        pulse.play(
                            pulse.GaussianSquare(
                                duration=800,
                                amp=0.5,
                                sigma=64,
                                risefall_sigma_ratio=2,
                            ),
                            channel=pulse.ControlChannel(0),
                        )
                # xp
                with pulse.align_left():
                    pulse.play(
                        pulse.Drag(
                            duration=160,
                            amp=0.2,
                            sigma=40,
                            beta=3.0,
                        ),
                        channel=pulse.DriveChannel(0),
                    )

        self.assertEqual(flatten_cx, ref_cx)

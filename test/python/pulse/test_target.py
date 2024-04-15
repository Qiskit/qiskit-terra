# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for pulse target for pulse compiler."""

from ddt import ddt, named_data, unpack

from qiskit.pulse import model
from qiskit.pulse.compiler.target import QiskitPulseTarget, ControlPort, MeasurePort
from qiskit.pulse.exceptions import NotExistingComponent
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestPulseTarget(QiskitTestCase):
    """Test cases of hardware information acquitision from QiskitPulseTarget with mock data."""

    def setUp(self):
        super().setUp()
        self.target = QiskitPulseTarget(
            qubit_frames={
                0: "Q0",
                1: "Q1",
            },
            meas_frames={
                0: "M0",
                1: "M1",
            },
            tx_ports=[
                ControlPort(
                    identifier="Q_channel-0",
                    qubits=(0,),
                    num_frames=2,
                    reserved_frames=["Q0", "Q1"],
                ),
                ControlPort(
                    identifier="Q_channel-1",
                    qubits=(1,),
                    num_frames=2,
                    reserved_frames=["Q1"],
                ),
                ControlPort(
                    identifier="Coupler-A",
                    qubits=(0, 1),
                    num_frames=0,
                    reserved_frames=[],
                ),
                MeasurePort(
                    identifier="R_channel-0",
                    qubits=(0,),
                    num_frames=1,
                    reserved_frames=["M0"],
                ),
                MeasurePort(
                    identifier="R_channel-1",
                    qubits=(1,),
                    num_frames=1,
                    reserved_frames=["M1"],
                ),
            ],
        )

    @named_data(
        ["qubit0_query", model.Qubit(0), "Q_channel-0"],
        ["qubit1_query", model.Qubit(1), "Q_channel-1"],
        ["coupler_query", model.Coupler(0, 1), "Coupler-A"],
        ["generic_query1", model.Port("Q_channel-0"), "Q_channel-0"],
        ["generic_query2", model.Port("Coupler-A"), "Coupler-A"],
    )
    @unpack
    def test_get_control_port_uid(self, port, uid):
        """Test get control Qiskit port UID from pulse target."""
        self.assertEqual(
            self.target.get_port_identifier(pulse_endpoint=port, op_type="control"),
            uid,
        )

    @named_data(
        ["qubit0_query", model.Qubit(0), "R_channel-0"],
        ["qubit1_query", model.Qubit(1), "R_channel-1"],
    )
    @unpack
    def test_get_measure_port_uid(self, port, uid):
        """Test get measure Qiskit port UID from pulse target."""
        self.assertEqual(
            self.target.get_port_identifier(pulse_endpoint=port, op_type="measure"),
            uid,
        )

    @named_data(
        ["random_name_query", model.Port("NotDefinedPort")],
        ["invalid_index_query", model.Qubit(100)],
    )
    def test_port_not_found(self, port):
        """Test port is not available in pulse target."""
        with self.assertRaises(NotExistingComponent):
            self.target.get_port_identifier(pulse_endpoint=port, op_type="control")

    @named_data(
        ["frame_q0_query", model.QubitFrame(0), "Q0"],
        ["frame_q1_query", model.QubitFrame(1), "Q1"],
        ["frame_m0_query", model.MeasurementFrame(0), "M0"],
        ["frame_m1_query", model.MeasurementFrame(1), "M1"],
        ["generic_query", model.GenericFrame("my_frame"), "my_frame"],
    )
    def test_get_frame_uid(self, frame, uid):
        """Test get Qiskit frame UID from pulse target."""
        self.assertEqual(self.target.get_frame_identifier(frame), uid)

    def test_frame_not_found(self):
        """Test frame is not available in pulse target."""
        with self.assertRaises(NotExistingComponent):
            self.target.get_frame_identifier(model.QubitFrame(100))

    def test_filter_calibrated_mixed_frame_only_port(self):
        """Test get backend-reserved mixed frame list filtered by port."""
        self.assertListEqual(
            self.target.reserved_mixed_frames(pulse_endpoint=model.Qubit(0)),
            [
                # self-drive
                model.MixedFrame(model.Qubit(0), model.QubitFrame(0)),
                # cr-like
                model.MixedFrame(model.Qubit(0), model.QubitFrame(1)),
                # measure
                model.MixedFrame(model.Qubit(0), model.MeasurementFrame(0)),
            ],
        )

    def test_filter_calibrated_mixed_frame_only_frame(self):
        """Test get backend-reserved mixed frame list filtered by frame."""
        self.assertListEqual(
            self.target.reserved_mixed_frames(frame=model.QubitFrame(1)),
            [
                # cr-like
                model.MixedFrame(model.Qubit(0), model.QubitFrame(1)),
                # self-drive
                model.MixedFrame(model.Qubit(1), model.QubitFrame(1)),
            ],
        )

    def test_filter_calibrated_mixed_frame_both_port_frame(self):
        """Test get backend-reserved mixed frame list filtered by both port and frame."""
        self.assertListEqual(
            self.target.reserved_mixed_frames(
                frame=model.QubitFrame(1), pulse_endpoint=model.Qubit(0)
            ),
            [
                model.MixedFrame(model.Qubit(0), model.QubitFrame(1)),
            ],
        )

    def test_filter_calibrated_mixed_frame_non_existing(self):
        """Test no mixed frame mached with the condition."""
        self.assertListEqual(
            self.target.reserved_mixed_frames(pulse_endpoint=model.Qubit(100)),
            [],
        )

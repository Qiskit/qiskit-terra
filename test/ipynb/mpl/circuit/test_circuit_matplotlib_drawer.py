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

"""Tests for circuit MPL drawer"""

import unittest

import json
import os
from contextlib import contextmanager
import math
import numpy as np
from numpy import pi

from qiskit.test import QiskitTestCase
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.fake_provider import FakeTenerife
from qiskit.visualization.circuit.circuit_visualization import _matplotlib_circuit_drawer
from qiskit.circuit.library import (
    XGate,
    CXGate,
    MCXGate,
    HGate,
    RZZGate,
    iSwapGate,
    SwapGate,
    DCXGate,
    ZGate,
    SGate,
    U1Gate,
    CPhaseGate,
)
from qiskit.circuit.library import MCXVChain
from qiskit.extensions import HamiltonianGate
from qiskit.circuit import Parameter, Qubit, Clbit
from qiskit.circuit.library import IQP
from qiskit.quantum_info.random import random_unitary
from qiskit.utils import optionals

if optionals.HAS_MATPLOTLIB:
    from matplotlib.pyplot import close as mpl_close
else:
    raise ImportError('Must have Matplotlib installed. To install, run "pip install matplotlib".')


RESULTDIR = os.path.dirname(os.path.abspath(__file__))


@contextmanager
def cwd(path):
    """A context manager to run in a particular path"""
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


class TestMatplotlibDrawer(QiskitTestCase):
    """Circuit MPL visualization"""

    def setUp(self):
        super().setUp()
        self.circuit_drawer = TestMatplotlibDrawer.save_data_wrap(
            _matplotlib_circuit_drawer, str(self)
        )

    def tearDown(self):
        super().tearDown()
        mpl_close("all")

    @staticmethod
    def save_data_wrap(func, testname):
        """A wrapper to save the data from a test"""

        def wrapper(*args, **kwargs):
            image_filename = kwargs["filename"]
            with cwd(RESULTDIR):
                results = func(*args, **kwargs)
                TestMatplotlibDrawer.save_data(image_filename, testname)
            return results

        return wrapper

    @staticmethod
    def save_data(image_filename, testname):
        """Saves result data of a test"""
        datafilename = "result_test.json"
        if os.path.exists(datafilename):
            with open(datafilename) as datafile:
                data = json.load(datafile)
        else:
            data = {}
        data[image_filename] = {"testname": testname}
        with open(datafilename, "w") as datafile:
            json.dump(data, datafile)

    def test_empty_circuit(self):
        """Test empty circuit"""
        circuit = QuantumCircuit()

        self.circuit_drawer(circuit, filename="empty_circut.png")

    def test_calibrations(self):
        """Test calibrations annotations
        See https://github.com/Qiskit/qiskit-terra/issues/5920
        """

        circuit = QuantumCircuit(2, 2)
        circuit.h(0)

        from qiskit import pulse

        with pulse.build(name="hadamard") as h_q0:
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.DriveChannel(0)
            )

        circuit.add_calibration("h", [0], h_q0)

        self.circuit_drawer(circuit, filename="calibrations.png")

    def test_calibrations_with_control_gates(self):
        """Test calibrations annotations
        See https://github.com/Qiskit/qiskit-terra/issues/5920
        """

        circuit = QuantumCircuit(2, 2)
        circuit.cx(0, 1)
        circuit.ch(0, 1)

        from qiskit import pulse

        with pulse.build(name="cnot") as cx_q01:
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.DriveChannel(1)
            )

        circuit.add_calibration("cx", [0, 1], cx_q01)

        with pulse.build(name="ch") as ch_q01:
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.DriveChannel(1)
            )

        circuit.add_calibration("ch", [0, 1], ch_q01)

        self.circuit_drawer(circuit, filename="calibrations_with_control_gates.png")

    def test_calibrations_with_swap_and_reset(self):
        """Test calibrations annotations
        See https://github.com/Qiskit/qiskit-terra/issues/5920
        """

        circuit = QuantumCircuit(2, 2)
        circuit.swap(0, 1)
        circuit.reset(0)

        from qiskit import pulse

        with pulse.build(name="swap") as swap_q01:
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.DriveChannel(1)
            )

        circuit.add_calibration("swap", [0, 1], swap_q01)

        with pulse.build(name="reset") as reset_q0:
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.DriveChannel(1)
            )

        circuit.add_calibration("reset", [0], reset_q0)

        self.circuit_drawer(circuit, filename="calibrations_with_swap_and_reset.png")

    def test_calibrations_with_rzz_and_rxx(self):
        """Test calibrations annotations
        See https://github.com/Qiskit/qiskit-terra/issues/5920
        """
        circuit = QuantumCircuit(2, 2)
        circuit.rzz(pi, 0, 1)
        circuit.rxx(pi, 0, 1)

        from qiskit import pulse

        with pulse.build(name="rzz") as rzz_q01:
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.DriveChannel(1)
            )

        circuit.add_calibration("rzz", [0, 1], rzz_q01)

        with pulse.build(name="rxx") as rxx_q01:
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.DriveChannel(1)
            )

        circuit.add_calibration("rxx", [0, 1], rxx_q01)

        self.circuit_drawer(circuit, filename="calibrations_with_rzz_and_rxx.png")

    def test_no_ops(self):
        """Test circuit with no ops.
        See https://github.com/Qiskit/qiskit-terra/issues/5393"""
        circuit = QuantumCircuit(2, 3)
        self.circuit_drawer(circuit, filename="no_op_circut.png")

    def test_long_name(self):
        """Test to see that long register names can be seen completely
        As reported in #2605
        """

        # add a register with a very long name
        qr = QuantumRegister(4, "veryLongQuantumRegisterName")
        # add another to make sure adjustments are made based on longest
        qrr = QuantumRegister(1, "q0")
        circuit = QuantumCircuit(qr, qrr)

        # check gates are shifted over accordingly
        circuit.h(qr)
        circuit.h(qr)
        circuit.h(qr)

        self.circuit_drawer(circuit, filename="long_name.png")

    def test_multi_underscore_reg_names(self):
        """Test that multi-underscores in register names display properly"""
        q_reg1 = QuantumRegister(1, "q1_re__g__g")
        q_reg3 = QuantumRegister(3, "q3_re_g__g")
        c_reg1 = ClassicalRegister(1, "c1_re_g__g")
        c_reg3 = ClassicalRegister(3, "c3_re_g__g")
        circuit = QuantumCircuit(q_reg1, q_reg3, c_reg1, c_reg3)

        self.circuit_drawer(circuit, cregbundle=True, filename="multi_underscore_true.png")
        self.circuit_drawer(circuit, cregbundle=False, filename="multi_underscore_false.png")

    def test_conditional(self):
        """Test that circuits with conditionals draw correctly"""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)

        # check gates are shifted over accordingly
        circuit.h(qr)
        circuit.measure(qr, cr)
        circuit.h(qr[0]).c_if(cr, 2)

        self.circuit_drawer(circuit, filename="reg_conditional.png")

    def test_bit_conditional_with_cregbundle(self):
        """Test that circuits with single bit conditionals draw correctly
        with cregbundle=True."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)

        circuit.x(qr[0])
        circuit.measure(qr, cr)
        circuit.h(qr[0]).c_if(cr[0], 1)
        circuit.x(qr[1]).c_if(cr[1], 0)

        self.circuit_drawer(circuit, filename="bit_conditional_bundle.png")

    def test_bit_conditional_no_cregbundle(self):
        """Test that circuits with single bit conditionals draw correctly
        with cregbundle=False."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)

        circuit.x(qr[0])
        circuit.measure(qr, cr)
        circuit.h(qr[0]).c_if(cr[0], 1)
        circuit.x(qr[1]).c_if(cr[1], 0)

        self.circuit_drawer(circuit, filename="bit_conditional_no_bundle.png", cregbundle=False)

    def test_plot_partial_barrier(self):
        """Test plotting of partial barriers."""

        # generate a circuit with barrier and other barrier like instructions in
        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(q, c)

        # check for barriers
        circuit.h(q[0])
        circuit.barrier(0)
        circuit.h(q[0])

        self.circuit_drawer(circuit, filename="plot_partial_barrier.png", plot_barriers=True)

    def test_plot_barriers(self):
        """Test to see that plotting barriers works.
        If it is set to False, no blank columns are introduced"""

        # generate a circuit with barriers and other barrier like instructions in
        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(q, c)

        # check for barriers
        circuit.h(q[0])
        circuit.barrier()

        # check for other barrier like commands
        circuit.h(q[1])

        # this import appears to be unused, but is actually needed to get snapshot instruction
        import qiskit.extensions.simulator  # pylint: disable=unused-import

        circuit.snapshot("1")

        # check the barriers plot properly when plot_barriers= True
        self.circuit_drawer(circuit, filename="plot_barriers_true.png", plot_barriers=True)
        self.circuit_drawer(circuit, filename="plot_barriers_false.png", plot_barriers=False)

    def test_no_barriers_false(self):
        """Generate the same circuit as test_plot_barriers but without the barrier commands
        as this is what the circuit should look like when displayed with plot barriers false"""
        q1 = QuantumRegister(2, "q")
        c1 = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(q1, c1)
        circuit.h(q1[0])
        circuit.h(q1[1])

        self.circuit_drawer(circuit, filename="no_barriers.png", plot_barriers=False)

    def test_fold_minus1(self):
        """Test to see that fold=-1 is no folding"""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        for _ in range(3):
            circuit.h(0)
            circuit.x(0)

        self.circuit_drawer(circuit, fold=-1, filename="fold_minus1.png")

    def test_fold_4(self):
        """Test to see that fold=4 is folding"""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        for _ in range(3):
            circuit.h(0)
            circuit.x(0)

        self.circuit_drawer(circuit, fold=4, filename="fold_4.png")

    def test_big_gates(self):
        """Test large gates with params"""
        qr = QuantumRegister(6, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(IQP([[6, 5, 3], [5, 4, 5], [3, 5, 1]]), [0, 1, 2])

        desired_vector = [
            1 / math.sqrt(16) * complex(0, 1),
            1 / math.sqrt(8) * complex(1, 0),
            1 / math.sqrt(16) * complex(1, 1),
            0,
            0,
            1 / math.sqrt(8) * complex(1, 2),
            1 / math.sqrt(16) * complex(1, 0),
            0,
        ]

        circuit.initialize(desired_vector, [qr[3], qr[4], qr[5]])
        circuit.unitary([[1, 0], [0, 1]], [qr[0]])
        matrix = np.zeros((4, 4))
        theta = Parameter("theta")
        circuit.append(HamiltonianGate(matrix, theta), [qr[1], qr[2]])
        circuit = circuit.bind_parameters({theta: 1})
        circuit.isometry(np.eye(4, 4), list(range(3, 5)), [])

        self.circuit_drawer(circuit, filename="big_gates.png")

    def test_cnot(self):
        """Test different cnot gates (ccnot, mcx, etc)"""
        qr = QuantumRegister(6, "q")
        circuit = QuantumCircuit(qr)
        circuit.x(0)
        circuit.cx(0, 1)
        circuit.ccx(0, 1, 2)
        circuit.append(XGate().control(3, ctrl_state="010"), [qr[2], qr[3], qr[0], qr[1]])
        circuit.append(MCXGate(num_ctrl_qubits=3, ctrl_state="101"), [qr[0], qr[1], qr[2], qr[4]])
        circuit.append(MCXVChain(3, dirty_ancillas=True), [qr[0], qr[1], qr[2], qr[3], qr[5]])

        self.circuit_drawer(circuit, filename="cnot.png")

    def test_cz(self):
        """Test Z and Controlled-Z Gates"""
        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.z(0)
        circuit.cz(0, 1)
        circuit.append(ZGate().control(3, ctrl_state="101"), [0, 1, 2, 3])
        circuit.append(ZGate().control(2), [1, 2, 3])
        circuit.append(ZGate().control(1, ctrl_state="0", label="CZ Gate"), [2, 3])

        self.circuit_drawer(circuit, filename="cz.png")

    def test_cx_iswap_swap(self):
        """Test CX, iSWAP and Swap gates in a circuit."""
        circuit = QuantumCircuit(3)
        circuit = circuit.compose(CXGate(), [0, 1])
        circuit = circuit.compose(iSwapGate(), [0, 1])
        circuit = circuit.compose(SwapGate(), [0, 1])
        circuit = circuit.compose(CXGate().control(), [0, 1, 2])
        circuit = circuit.compose(iSwapGate().control(), [0, 1, 2])
        circuit = circuit.compose(SwapGate().control(), [0, 1, 2])
        self.circuit_drawer(circuit, filename="cx_iswap_swap.png")

    def test_pauli_clifford(self):
        """Test Pauli(green) and Clifford(blue) gates"""
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.x(0)
        circuit.y(0)
        circuit.z(0)
        circuit.id(0)
        circuit.h(1)
        circuit.cx(1, 2)
        circuit.cy(1, 2)
        circuit.cz(1, 2)
        circuit.swap(3, 4)
        circuit.s(3)
        circuit.sdg(3)
        circuit.iswap(3, 4)
        circuit.dcx(3, 4)

        self.circuit_drawer(circuit, filename="pauli_clifford.png")

    def test_creg_initial(self):
        """Test cregbundle and initial state options"""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.x(0)
        circuit.h(0)
        circuit.x(1)

        self.circuit_drawer(
            circuit, filename="creg_initial_true.png", cregbundle=True, initial_state=True
        )
        self.circuit_drawer(
            circuit, filename="creg_initial_false.png", cregbundle=False, initial_state=False
        )

    def test_r_gates(self):
        """Test all R gates"""
        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.r(3 * pi / 4, 3 * pi / 8, 0)
        circuit.rx(pi / 2, 1)
        circuit.ry(-pi / 2, 2)
        circuit.rz(3 * pi / 4, 3)
        circuit.rxx(pi / 2, 0, 1)
        circuit.ryy(3 * pi / 4, 2, 3)
        circuit.rzx(-pi / 2, 0, 1)
        circuit.rzz(pi / 2, 2, 3)

        self.circuit_drawer(circuit, filename="r_gates.png")

    def test_ctrl_labels(self):
        """Test control labels"""
        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cy(1, 0, label="Bottom Y label")
        circuit.cu(pi / 2, pi / 2, pi / 2, 0, 2, 3, label="Top U label")
        circuit.ch(0, 1, label="Top H label")
        circuit.append(
            HGate(label="H gate label").control(3, label="H control label", ctrl_state="010"),
            [qr[1], qr[2], qr[3], qr[0]],
        )

        self.circuit_drawer(circuit, filename="ctrl_labels.png")

    def test_cswap_rzz(self):
        """Test controlled swap and rzz gates"""
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cswap(0, 1, 2)
        circuit.append(RZZGate(3 * pi / 4).control(3, ctrl_state="010"), [2, 1, 4, 3, 0])

        self.circuit_drawer(circuit, filename="cswap_rzz.png")

    def test_ghz_to_gate(self):
        """Test controlled GHZ to_gate circuit"""
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        ghz_circuit = QuantumCircuit(3, name="this is a WWWWWWWWWWWide name Ctrl-GHZ Circuit")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        ccghz = ghz.control(2, ctrl_state="10")
        circuit.append(ccghz, [4, 0, 1, 3, 2])

        self.circuit_drawer(circuit, filename="ghz_to_gate.png")

    def test_scale(self):
        """Tests scale
        See: https://github.com/Qiskit/qiskit-terra/issues/4179"""
        circuit = QuantumCircuit(5)
        circuit.unitary(random_unitary(2**5), circuit.qubits)

        self.circuit_drawer(circuit, filename="scale_default.png")
        self.circuit_drawer(circuit, filename="scale_half.png", scale=0.5)
        self.circuit_drawer(circuit, filename="scale_double.png", scale=2)

    def test_pi_param_expr(self):
        """Test pi in circuit with parameter expression."""
        x, y = Parameter("x"), Parameter("y")
        circuit = QuantumCircuit(1)
        circuit.rx((pi - x) * (pi - y), 0)

        self.circuit_drawer(circuit, filename="pi_in_param_expr.png")

    def test_partial_layout(self):
        """Tests partial_layout
        See: https://github.com/Qiskit/qiskit-terra/issues/4757"""
        circuit = QuantumCircuit(3)
        circuit.h(1)
        transpiled = transpile(
            circuit,
            backend=FakeTenerife(),
            basis_gates=["id", "cx", "rz", "sx", "x"],
            optimization_level=0,
            initial_layout=[1, 2, 0],
            seed_transpiler=0,
        )

        self.circuit_drawer(transpiled, filename="partial_layout.png")

    def test_init_reset(self):
        """Test reset and initialize with 1 and 2 qubits"""
        circuit = QuantumCircuit(2)
        circuit.initialize([0, 1], 0)
        circuit.reset(1)
        circuit.initialize([0, 1, 0, 0], [0, 1])

        self.circuit_drawer(circuit, filename="init_reset.png")

    def test_with_global_phase(self):
        """Tests with global phase"""
        circuit = QuantumCircuit(3, global_phase=1.57079632679)
        circuit.h(range(3))

        self.circuit_drawer(circuit, filename="global_phase.png")

    def test_alternative_colors(self):
        """Tests alternative color schemes"""
        for style in ["iqx", "iqx-dark", "textbook"]:
            with self.subTest(style=style):
                circuit = QuantumCircuit(7)
                circuit.h(0)
                circuit.x(0)
                circuit.cx(0, 1)
                circuit.ccx(0, 1, 2)
                circuit.swap(0, 1)
                circuit.cswap(0, 1, 2)
                circuit.append(SwapGate().control(2), [0, 1, 2, 3])
                circuit.dcx(0, 1)
                circuit.append(DCXGate().control(1), [0, 1, 2])
                circuit.append(DCXGate().control(2), [0, 1, 2, 3])
                circuit.z(4)
                circuit.s(4)
                circuit.sdg(4)
                circuit.t(4)
                circuit.tdg(4)
                circuit.p(pi / 2, 4)
                circuit.cz(5, 6)
                circuit.cp(pi / 2, 5, 6)
                circuit.y(5)
                circuit.rx(pi / 3, 5)
                circuit.rzx(pi / 2, 5, 6)
                circuit.u(pi / 2, pi / 2, pi / 2, 5)
                circuit.barrier(5, 6)
                circuit.reset(5)

                self.circuit_drawer(circuit, style={"name": style}, filename=f"{style}_color.png")

    def test_reverse_bits(self):
        """Tests reverse_bits parameter"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ccx(2, 1, 0)

        self.circuit_drawer(circuit, reverse_bits=True, filename="reverse_bits.png")

    def test_bw(self):
        """Tests black and white style parameter"""
        circuit = QuantumCircuit(3, 3)
        circuit.h(0)
        circuit.x(1)
        circuit.sdg(2)
        circuit.cx(0, 1)
        circuit.ccx(2, 1, 0)
        circuit.swap(1, 2)
        circuit.measure_all()

        self.circuit_drawer(circuit, style={"name": "bw"}, filename="bw.png")

    def test_user_style(self):
        """Tests loading a user style"""
        circuit = QuantumCircuit(7)
        circuit.h(0)
        circuit.append(HGate(label="H2"), [1])
        circuit.x(0)
        circuit.cx(0, 1)
        circuit.ccx(0, 1, 2)
        circuit.swap(0, 1)
        circuit.cswap(0, 1, 2)
        circuit.append(SwapGate().control(2), [0, 1, 2, 3])
        circuit.dcx(0, 1)
        circuit.append(DCXGate().control(1), [0, 1, 2])
        circuit.append(DCXGate().control(2), [0, 1, 2, 3])
        circuit.z(4)
        circuit.append(SGate(label="S1"), [4])
        circuit.sdg(4)
        circuit.t(4)
        circuit.tdg(4)
        circuit.p(pi / 2, 4)
        circuit.cz(5, 6)
        circuit.cp(pi / 2, 5, 6)
        circuit.y(5)
        circuit.rx(pi / 3, 5)
        circuit.rzx(pi / 2, 5, 6)
        circuit.u(pi / 2, pi / 2, pi / 2, 5)
        circuit.barrier(5, 6)
        circuit.reset(5)

        self.circuit_drawer(
            circuit,
            style={
                "name": "user_style",
                "displaytext": {"H2": "H_2"},
                "displaycolor": {"H2": ("#EEDD00", "#FF0000")},
            },
            filename="user_style.png",
        )

    def test_subfont_change(self):
        """Tests changing the subfont size"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(0)
        circuit.u(pi / 2, pi / 2, pi / 2, 1)
        circuit.p(pi / 2, 2)
        style = {"name": "iqx", "subfontsize": 11}

        self.circuit_drawer(circuit, style=style, filename="subfont.png")
        self.assertEqual(style, {"name": "iqx", "subfontsize": 11})  # check does not change style

    def test_meas_condition(self):
        """Tests measure with a condition"""
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[0])
        circuit.h(qr[1]).c_if(cr, 1)
        self.circuit_drawer(circuit, filename="meas_condition.png")

    def test_reverse_bits_condition(self):
        """Tests reverse_bits with a condition and gate above"""
        cr = ClassicalRegister(2, "cr")
        cr2 = ClassicalRegister(1, "cr2")
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr, cr, cr2)
        circuit.h(0)
        circuit.h(1)
        circuit.h(2)
        circuit.x(0)
        circuit.x(0)
        circuit.measure(2, 1)
        circuit.x(2).c_if(cr, 2)
        self.circuit_drawer(
            circuit, cregbundle=False, reverse_bits=True, filename="reverse_bits_cond_true.png"
        )
        self.circuit_drawer(
            circuit, cregbundle=False, reverse_bits=False, filename="reverse_bits_cond_false.png"
        )

    def test_style_custom_gates(self):
        """Tests style for custom gates"""

        def cnotnot(gate_label):
            gate_circuit = QuantumCircuit(3, name=gate_label)
            gate_circuit.cnot(0, 1)
            gate_circuit.cnot(0, 2)
            gate = gate_circuit.to_gate()
            return gate

        q = QuantumRegister(3, name="q")

        circuit = QuantumCircuit(q)

        circuit.append(cnotnot("CNOTNOT"), [q[0], q[1], q[2]])
        circuit.append(cnotnot("CNOTNOT_PRIME"), [q[0], q[1], q[2]])
        circuit.h(q[0])

        self.circuit_drawer(
            circuit,
            style={
                "displaycolor": {"CNOTNOT": ("#000000", "#FFFFFF"), "h": ("#A1A1A1", "#043812")},
                "displaytext": {"CNOTNOT_PRIME": "$\\mathrm{CNOTNOT}'$"},
            },
            filename="style_custom_gates.png",
        )

    def test_6095(self):
        """Tests controlled-phase gate style
        See https://github.com/Qiskit/qiskit-terra/issues/6095"""
        circuit = QuantumCircuit(2)
        circuit.cp(1.0, 0, 1)
        circuit.h(1)

        self.circuit_drawer(
            circuit,
            style={"displaycolor": {"cp": ("#A27486", "#000000"), "h": ("#A27486", "#000000")}},
            filename="6095.png",
        )

    def test_instruction_1q_1c(self):
        """Tests q0-cr0 instruction on a circuit"""
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)
        inst = QuantumCircuit(1, 1, name="Inst").to_instruction()
        circuit.append(inst, [qr[0]], [cr[0]])
        self.circuit_drawer(circuit, filename="instruction_1q_1c.png")

    def test_instruction_3q_3c_circ1(self):
        """Tests q0-q1-q2-cr_20-cr0-cr1 instruction on a circuit"""
        qr = QuantumRegister(4, "qr")
        cr = ClassicalRegister(2, "cr")
        cr2 = ClassicalRegister(2, "cr2")
        circuit = QuantumCircuit(qr, cr, cr2)
        inst = QuantumCircuit(3, 3, name="Inst").to_instruction()
        circuit.append(inst, [qr[0], qr[1], qr[2]], [cr2[0], cr[0], cr[1]])
        self.circuit_drawer(circuit, filename="instruction_3q_3c_circ1.png")

    def test_instruction_3q_3c_circ2(self):
        """Tests q3-q0-q2-cr0-cr1-cr_20 instruction on a circuit"""
        qr = QuantumRegister(4, "qr")
        cr = ClassicalRegister(2, "cr")
        cr2 = ClassicalRegister(2, "cr2")
        circuit = QuantumCircuit(qr, cr, cr2)
        inst = QuantumCircuit(3, 3, name="Inst").to_instruction()
        circuit.append(inst, [qr[3], qr[0], qr[2]], [cr[0], cr[1], cr2[0]])
        self.circuit_drawer(circuit, filename="instruction_3q_3c_circ2.png")

    def test_instruction_3q_3c_circ3(self):
        """Tests q3-q1-q2-cr_31-cr1-cr_30 instruction on a circuit"""
        qr = QuantumRegister(4, "qr")
        cr = ClassicalRegister(2, "cr")
        cr2 = ClassicalRegister(1, "cr2")
        cr3 = ClassicalRegister(2, "cr3")
        circuit = QuantumCircuit(qr, cr, cr2, cr3)
        inst = QuantumCircuit(3, 3, name="Inst").to_instruction()
        circuit.append(inst, [qr[3], qr[1], qr[2]], [cr3[1], cr[1], cr3[0]])
        self.circuit_drawer(circuit, filename="instruction_3q_3c_circ3.png")

    def test_overwide_gates(self):
        """Test gates don't exceed width of default fold"""
        circuit = QuantumCircuit(5)
        initial_state = np.zeros(2**5)
        initial_state[5] = 1
        circuit.initialize(initial_state)
        self.circuit_drawer(circuit, filename="wide_params.png")

    def test_one_bit_regs(self):
        """Test registers with only one bit display without number"""
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        cr1 = ClassicalRegister(1, "cr1")
        cr2 = ClassicalRegister(2, "cr2")
        circuit = QuantumCircuit(qr1, qr2, cr1, cr2)
        circuit.h(0)
        circuit.measure(0, 0)
        self.circuit_drawer(circuit, cregbundle=False, filename="one_bit_regs.png")

    def test_user_ax_subplot(self):
        """Test for when user supplies ax for a subplot"""
        import matplotlib.pyplot as plt

        fig = plt.figure(1, figsize=(6, 4))
        fig.patch.set_facecolor("white")
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot([1, 2, 3])

        circuit = QuantumCircuit(4)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.h(1)
        circuit.cx(1, 2)
        plt.close(fig)
        self.circuit_drawer(circuit, ax=ax2, filename="user_ax.png")

    def test_figwidth(self):
        """Test style dict 'figwidth'"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.x(1)
        circuit.cx(1, 2)
        circuit.x(2)
        self.circuit_drawer(circuit, style={"figwidth": 5}, filename="figwidth.png")

    def test_registerless_one_bit(self):
        """Test circuit with one-bit registers and registerless bits."""
        qrx = QuantumRegister(2, "qrx")
        qry = QuantumRegister(1, "qry")
        crx = ClassicalRegister(2, "crx")
        circuit = QuantumCircuit(qrx, [Qubit(), Qubit()], qry, [Clbit(), Clbit()], crx)
        self.circuit_drawer(circuit, filename="registerless_one_bit.png")

    def test_measures_with_conditions(self):
        """Test that a measure containing a condition displays"""
        qr = QuantumRegister(2, "qr")
        cr1 = ClassicalRegister(2, "cr1")
        cr2 = ClassicalRegister(2, "cr2")
        circuit = QuantumCircuit(qr, cr1, cr2)
        circuit.h(0)
        circuit.h(1)
        circuit.measure(0, cr1[1])
        circuit.measure(1, cr2[0]).c_if(cr1, 1)
        circuit.h(0).c_if(cr2, 3)
        self.circuit_drawer(circuit, cregbundle=False, filename="measure_cond_false.png")
        self.circuit_drawer(circuit, cregbundle=True, filename="measure_cond_true.png")

    def test_conditions_measures_with_bits(self):
        """Test that gates with conditions and measures work with bits"""
        bits = [Qubit(), Qubit(), Clbit(), Clbit()]
        cr = ClassicalRegister(2, "cr")
        crx = ClassicalRegister(3, "cs")
        circuit = QuantumCircuit(bits, cr, [Clbit()], crx)
        circuit.x(0).c_if(crx[1], 0)
        circuit.measure(0, bits[3])
        self.circuit_drawer(circuit, cregbundle=False, filename="measure_cond_bits_false.png")
        self.circuit_drawer(circuit, cregbundle=True, filename="measure_cond_bits_true.png")

    def test_conditional_gates_right_of_measures_with_bits(self):
        """Test that gates with conditions draw to right of measures when same bit"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[1])
        circuit.h(qr[1]).c_if(cr[1], 0)
        circuit.h(qr[2]).c_if(cr[0], 0)
        self.circuit_drawer(circuit, cregbundle=False, filename="measure_cond_bits_right.png")

    def test_conditions_with_bits_reverse(self):
        """Test that gates with conditions work with bits reversed"""
        bits = [Qubit(), Qubit(), Clbit(), Clbit()]
        cr = ClassicalRegister(2, "cr")
        crx = ClassicalRegister(2, "cs")
        circuit = QuantumCircuit(bits, cr, [Clbit()], crx)
        circuit.x(0).c_if(bits[3], 0)
        self.circuit_drawer(
            circuit, cregbundle=False, reverse_bits=True, filename="cond_bits_reverse.png"
        )

    def test_sidetext_with_condition(self):
        """Test that sidetext gates align properly with conditions"""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(CPhaseGate(pi / 2), [qr[0], qr[1]]).c_if(cr[1], 1)
        self.circuit_drawer(circuit, cregbundle=False, filename="sidetext_condition.png")

    def test_fold_with_conditions(self):
        """Test that gates with conditions draw correctly when folding"""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(5)
        circuit = QuantumCircuit(qr, cr)

        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 1)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 3)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 5)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 7)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 9)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 11)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 13)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 15)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 17)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 19)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 21)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 23)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 25)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 27)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 29)
        circuit.append(U1Gate(0).control(1), [1, 0]).c_if(cr, 31)
        self.circuit_drawer(circuit, cregbundle=False, filename="fold_with_conditions.png")

    def test_idle_wires_barrier(self):
        """Test that idle_wires False works with barrier"""
        circuit = QuantumCircuit(4, 4)
        circuit.x(2)
        circuit.barrier()
        self.circuit_drawer(circuit, cregbundle=False, filename="idle_wires_barrier.png")

    def test_wire_order(self):
        """Test the wire_order option"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        cr2 = ClassicalRegister(2, "cx")
        circuit = QuantumCircuit(qr, cr, cr2)
        circuit.h(0)
        circuit.h(3)
        circuit.x(1)
        circuit.x(3).c_if(cr, 10)
        self.circuit_drawer(
            circuit,
            cregbundle=False,
            wire_order=[2, 1, 3, 0, 6, 8, 9, 5, 4, 7],
            filename="wire_order.png",
        )

    def test_barrier_label(self):
        """Test the barrier label"""
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.y(1)
        circuit.barrier()
        circuit.y(0)
        circuit.x(1)
        circuit.barrier(label="End Y/X")
        self.circuit_drawer(circuit, filename="barrier_label.png")


if __name__ == "__main__":
    unittest.main(verbosity=1)

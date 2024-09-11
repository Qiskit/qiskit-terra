# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the SabreLayout pass"""

import unittest

import math

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import EfficientSU2
from qiskit.transpiler import CouplingMap, AnalysisPass, PassManager
from qiskit.transpiler.passes import SabreLayout, DenseLayout, StochasticSwap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.converters import circuit_to_dag
from qiskit.compiler.transpiler import transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.passes.layout.sabre_pre_layout import SabrePreLayout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from test import QiskitTestCase  # pylint: disable=wrong-import-order

from ..legacy_cmaps import ALMADEN_CMAP, MUMBAI_CMAP


class TestSabreLayout(QiskitTestCase):
    """Tests the SabreLayout pass"""

    def setUp(self):
        super().setUp()
        self.cmap20 = ALMADEN_CMAP

    def test_5q_circuit_20q_coupling(self):
        """Test finds layout for 5q circuit on 20q device."""
        #                ┌───┐
        # q_0: ──■───────┤ X ├───────────────
        #        │       └─┬─┘┌───┐
        # q_1: ──┼────■────┼──┤ X ├───────■──
        #      ┌─┴─┐  │    │  ├───┤┌───┐┌─┴─┐
        # q_2: ┤ X ├──┼────┼──┤ X ├┤ X ├┤ X ├
        #      └───┘┌─┴─┐  │  └───┘└─┬─┘└───┘
        # q_3: ─────┤ X ├──■─────────┼───────
        #           └───┘            │
        # q_4: ──────────────────────■───────
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[1], qr[3])
        circuit.cx(qr[3], qr[0])
        circuit.x(qr[2])
        circuit.cx(qr[4], qr[2])
        circuit.x(qr[1])
        circuit.cx(qr[1], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = SabreLayout(CouplingMap(self.cmap20), seed=0, swap_trials=32, layout_trials=32)
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual([layout[q] for q in circuit.qubits], [11, 10, 16, 5, 17])

    def test_6q_circuit_20q_coupling(self):
        """Test finds layout for 6q circuit on 20q device."""
        #       ┌───┐┌───┐┌───┐┌───┐┌───┐
        # q0_0: ┤ X ├┤ X ├┤ X ├┤ X ├┤ X ├
        #       └─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘
        # q0_1: ──┼────■────┼────┼────┼──
        #         │  ┌───┐  │    │    │
        # q0_2: ──┼──┤ X ├──┼────■────┼──
        #         │  └───┘  │         │
        # q1_0: ──■─────────┼─────────┼──
        #            ┌───┐  │         │
        # q1_1: ─────┤ X ├──┼─────────■──
        #            └───┘  │
        # q1_2: ────────────■────────────
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])

        dag = circuit_to_dag(circuit)
        pass_ = SabreLayout(CouplingMap(self.cmap20), seed=0, swap_trials=32, layout_trials=32)
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual([layout[q] for q in circuit.qubits], [7, 8, 12, 6, 11, 13])

    def test_6q_circuit_20q_coupling_with_partial(self):
        """Test finds layout for 6q circuit on 20q device."""
        #       ┌───┐┌───┐┌───┐┌───┐┌───┐
        # q0_0: ┤ X ├┤ X ├┤ X ├┤ X ├┤ X ├
        #       └─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘
        # q0_1: ──┼────■────┼────┼────┼──
        #         │  ┌───┐  │    │    │
        # q0_2: ──┼──┤ X ├──┼────■────┼──
        #         │  └───┘  │         │
        # q1_0: ──■─────────┼─────────┼──
        #            ┌───┐  │         │
        # q1_1: ─────┤ X ├──┼─────────■──
        #            └───┘  │
        # q1_2: ────────────■────────────
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])

        pm = PassManager(
            [
                DensePartialSabreTrial(CouplingMap(self.cmap20)),
                SabreLayout(CouplingMap(self.cmap20), seed=0, swap_trials=32, layout_trials=0),
            ]
        )
        pm.run(circuit)
        layout = pm.property_set["layout"]
        self.assertEqual([layout[q] for q in circuit.qubits], [1, 3, 5, 2, 6, 0])

    def test_6q_circuit_20q_coupling_with_target(self):
        """Test finds layout for 6q circuit on 20q device."""
        #       ┌───┐┌───┐┌───┐┌───┐┌───┐
        # q0_0: ┤ X ├┤ X ├┤ X ├┤ X ├┤ X ├
        #       └─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘
        # q0_1: ──┼────■────┼────┼────┼──
        #         │  ┌───┐  │    │    │
        # q0_2: ──┼──┤ X ├──┼────■────┼──
        #         │  └───┘  │         │
        # q1_0: ──■─────────┼─────────┼──
        #            ┌───┐  │         │
        # q1_1: ─────┤ X ├──┼─────────■──
        #            └───┘  │
        # q1_2: ────────────■────────────
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])

        dag = circuit_to_dag(circuit)
        target = GenericBackendV2(num_qubits=20, coupling_map=self.cmap20).target
        pass_ = SabreLayout(target, seed=0, swap_trials=32, layout_trials=32)
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual([layout[q] for q in circuit.qubits], [7, 8, 12, 6, 11, 13])

    def test_layout_with_classical_bits(self):
        """Test sabre layout with classical bits recreate from issue #8635."""
        qc = QuantumCircuit.from_qasm_str(
            """
OPENQASM 2.0;
include "qelib1.inc";
qreg q4833[1];
qreg q4834[6];
qreg q4835[7];
creg c982[2];
creg c983[2];
creg c984[2];
rzz(0) q4833[0],q4834[4];
cu(0,-6.1035156e-05,0,1e-05) q4834[1],q4835[2];
swap q4834[0],q4834[2];
cu(-1.1920929e-07,0,-0.33333333,0) q4833[0],q4834[2];
ccx q4835[2],q4834[5],q4835[4];
measure q4835[4] -> c984[0];
ccx q4835[2],q4835[5],q4833[0];
measure q4835[5] -> c984[1];
measure q4834[0] -> c982[1];
u(10*pi,0,1.9) q4834[5];
measure q4834[3] -> c984[1];
measure q4835[0] -> c982[0];
rz(0) q4835[1];
"""
        )
        backend = GenericBackendV2(
            num_qubits=27,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=MUMBAI_CMAP,
            seed=42,
        )
        res = transpile(
            qc, backend, layout_method="sabre", seed_transpiler=1234, optimization_level=1
        )
        self.assertIsInstance(res, QuantumCircuit)
        layout = res._layout.initial_layout
        self.assertEqual(
            [layout[q] for q in qc.qubits], [11, 19, 18, 16, 26, 8, 21, 1, 5, 15, 3, 12, 14, 13]
        )

    # pylint: disable=line-too-long
    def test_layout_many_search_trials(self):
        """Test recreate failure from randomized testing that overflowed."""
        qc = QuantumCircuit.from_qasm_str(
            """
    OPENQASM 2.0;
include "qelib1.inc";
qreg q18585[14];
creg c1423[5];
creg c1424[4];
creg c1425[3];
barrier q18585[4],q18585[5],q18585[12],q18585[1];
cz q18585[11],q18585[3];
cswap q18585[8],q18585[10],q18585[6];
u(-2.00001,6.1035156e-05,-1.9) q18585[2];
barrier q18585[3],q18585[6],q18585[5],q18585[8],q18585[10],q18585[9],q18585[11],q18585[2],q18585[12],q18585[7],q18585[13],q18585[4],q18585[0],q18585[1];
cp(0) q18585[2],q18585[4];
cu(-0.99999,0,0,0) q18585[7],q18585[1];
cu(0,0,0,2.1507119) q18585[6],q18585[3];
barrier q18585[13],q18585[0],q18585[12],q18585[3],q18585[2],q18585[10];
ry(-1.1044662) q18585[13];
barrier q18585[13];
id q18585[12];
barrier q18585[12],q18585[6];
cu(-1.9,1.9,-1.5,0) q18585[10],q18585[0];
barrier q18585[13];
id q18585[8];
barrier q18585[12];
barrier q18585[12],q18585[1],q18585[9];
sdg q18585[2];
rz(-10*pi) q18585[6];
u(0,27.566433,1.9) q18585[1];
barrier q18585[12],q18585[11],q18585[9],q18585[4],q18585[7],q18585[0],q18585[13],q18585[3];
cu(-0.99999,-5.9604645e-08,-0.5,2.00001) q18585[3],q18585[13];
rx(-5.9604645e-08) q18585[7];
p(1.1) q18585[13];
barrier q18585[12],q18585[13],q18585[10],q18585[9],q18585[7],q18585[4];
z q18585[10];
measure q18585[7] -> c1423[2];
barrier q18585[0],q18585[3],q18585[7],q18585[4],q18585[1],q18585[8],q18585[6],q18585[11],q18585[5];
barrier q18585[5],q18585[2],q18585[8],q18585[3],q18585[6];
"""
        )
        backend = GenericBackendV2(
            num_qubits=27,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=MUMBAI_CMAP,
            seed=42,
        )
        with self.assertWarns(DeprecationWarning):
            res = transpile(
                qc,
                backend,
                layout_method="sabre",
                routing_method="stochastic",
                seed_transpiler=12345,
                optimization_level=1,
            )
        self.assertIsInstance(res, QuantumCircuit)
        layout = res._layout.initial_layout
        self.assertEqual(
            [layout[q] for q in qc.qubits], [22, 7, 2, 12, 1, 5, 14, 4, 11, 0, 16, 15, 3, 10]
        )

    def test_support_var_with_rust_fastpath(self):
        """Test that the joint layout/embed/routing logic for the Rust-space fast-path works in the
        presence of standalone `Var` nodes."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        qc = QuantumCircuit(5, inputs=[a])
        qc.add_var(b, 12)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(4, 0)

        out = SabreLayout(CouplingMap.from_line(8), seed=0, swap_trials=2, layout_trials=2)(qc)

        self.assertIsInstance(out, QuantumCircuit)
        self.assertEqual(out.layout.initial_index_layout(), [4, 5, 6, 3, 2, 0, 1, 7])

    def test_support_var_with_explicit_routing_pass(self):
        """Test that the logic works if an explicit routing pass is given."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        qc = QuantumCircuit(5, inputs=[a])
        qc.add_var(b, 12)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(4, 0)

        cm = CouplingMap.from_line(8)
        with self.assertWarns(DeprecationWarning):
            pass_ = SabreLayout(
                cm, seed=0, routing_pass=StochasticSwap(cm, trials=1, seed=0, fake_run=True)
            )
            _ = pass_(qc)
        layout = pass_.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [2, 3, 4, 1, 5])


class DensePartialSabreTrial(AnalysisPass):
    """Pass to run dense layout as a sabre trial."""

    def __init__(self, cmap):
        self.dense_pass = DenseLayout(cmap)
        super().__init__()

    def run(self, dag):
        self.dense_pass.run(dag)
        self.property_set["sabre_starting_layouts"] = [self.dense_pass.property_set["layout"]]


class TestDisjointDeviceSabreLayout(QiskitTestCase):
    """Test SabreLayout with a disjoint coupling map."""

    def setUp(self):
        super().setUp()
        self.dual_grid_cmap = CouplingMap(
            [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [5, 8]]
        )

    def test_dual_ghz(self):
        """Test a basic example with 2 circuit components and 2 cmap components."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        layout_routing_pass = SabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 1, 2, 5, 4, 6, 7, 8])

    def test_dual_ghz_with_wide_barrier(self):
        """Test a basic example with 2 circuit components and 2 cmap components."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        layout_routing_pass = SabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 1, 2, 5, 4, 6, 7, 8])

    def test_dual_ghz_with_intermediate_barriers(self):
        """Test dual ghz circuit with intermediate barriers local to each component."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.barrier(0, 1, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.barrier(4, 5, 6)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        layout_routing_pass = SabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 1, 2, 5, 4, 6, 7, 8])

    def test_dual_ghz_with_intermediate_spanning_barriers(self):
        """Test dual ghz circuit with barrier in the middle across components."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.barrier(0, 1, 2, 4, 5)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        layout_routing_pass = SabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 1, 2, 5, 4, 6, 7, 8])

    def test_too_large_components(self):
        """Assert trying to run a circuit with too large a connected component raises."""
        qc = QuantumCircuit(8)
        qc.h(0)
        for i in range(1, 6):
            qc.cx(0, i)
        qc.h(7)
        qc.cx(7, 6)
        layout_routing_pass = SabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        with self.assertRaises(TranspilerError):
            layout_routing_pass(qc)

    def test_with_partial_layout(self):
        """Test a partial layout with a disjoint connectivity graph."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        pm = PassManager(
            [
                DensePartialSabreTrial(self.dual_grid_cmap),
                SabreLayout(self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1),
            ]
        )
        pm.run(qc)
        layout = pm.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 1, 2, 5, 4, 6, 7, 8])


class TestSabrePreLayout(QiskitTestCase):
    """Tests the SabreLayout pass with starting layout created by SabrePreLayout."""

    def setUp(self):
        super().setUp()
        circuit = EfficientSU2(16, entanglement="circular", reps=6, flatten=True)
        circuit.assign_parameters([math.pi / 2] * len(circuit.parameters), inplace=True)
        circuit.measure_all()
        self.circuit = circuit
        self.coupling_map = CouplingMap.from_heavy_hex(7)

    def test_starting_layout(self):
        """Test that a starting layout is created and looks as expected."""
        pm = PassManager(
            [
                SabrePreLayout(coupling_map=self.coupling_map),
                SabreLayout(self.coupling_map, seed=123456, swap_trials=1, layout_trials=1),
            ]
        )
        pm.run(self.circuit)
        layout = pm.property_set["layout"]
        self.assertEqual(
            [layout[q] for q in self.circuit.qubits],
            [30, 98, 104, 36, 103, 35, 65, 28, 61, 91, 22, 92, 23, 93, 62, 99],
        )

    def test_integration_with_pass_manager(self):
        """Tests SabrePreLayoutIntegration with the rest of PassManager pipeline."""
        backend = GenericBackendV2(num_qubits=20, coupling_map=ALMADEN_CMAP, seed=42)
        pm = generate_preset_pass_manager(
            0, backend, layout_method="sabre", routing_method="sabre", seed_transpiler=0
        )
        pm.pre_layout = PassManager([SabrePreLayout(backend.target)])
        qct = pm.run(self.circuit)
        qct_initial_layout = qct.layout.initial_layout
        self.assertEqual(
            [qct_initial_layout[q] for q in self.circuit.qubits],
            [3, 8, 7, 12, 13, 14, 18, 17, 16, 11, 10, 5, 6, 1, 2, 4],
        )


if __name__ == "__main__":
    unittest.main()

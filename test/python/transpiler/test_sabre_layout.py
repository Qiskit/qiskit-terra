# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
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

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import SabreLayout
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.compiler.transpiler import transpile
from qiskit.providers.fake_provider import FakeAlmaden, FakeAlmadenV2
from qiskit.providers.fake_provider import FakeKolkata
from qiskit.providers.fake_provider import FakeMontreal


class TestSabreLayout(QiskitTestCase):
    """Tests the SabreLayout pass"""

    def setUp(self):
        super().setUp()
        self.cmap20 = FakeAlmaden().configuration().coupling_map

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
        self.assertEqual([layout[q] for q in circuit.qubits], [18, 11, 13, 12, 14])

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
        target = FakeAlmadenV2().target
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
        res = transpile(qc, FakeKolkata(), layout_method="sabre", seed_transpiler=1234)
        self.assertIsInstance(res, QuantumCircuit)
        layout = res._layout.initial_layout
        self.assertEqual(
            [layout[q] for q in qc.qubits], [13, 10, 11, 12, 17, 14, 22, 26, 5, 16, 25, 19, 7, 8]
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
        res = transpile(
            qc,
            FakeMontreal(),
            layout_method="sabre",
            routing_method="stochastic",
            seed_transpiler=12345,
        )
        self.assertIsInstance(res, QuantumCircuit)
        layout = res._layout.initial_layout
        self.assertEqual(
            [layout[q] for q in qc.qubits], [22, 21, 4, 12, 1, 23, 16, 18, 19, 25, 14, 13, 10, 7]
        )


if __name__ == "__main__":
    unittest.main()

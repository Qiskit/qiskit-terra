# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the CSPLayout pass"""

import unittest
from time import process_time

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import CSPLayout
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeTenerife, FakeRueschlikon, FakeTokyo, FakeYorktownV2


class TestCSPLayout(QiskitTestCase):
    """Tests the CSPLayout pass"""

    seed = 42

    def test_2q_circuit_2q_coupling(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap([[0, 1]]), strict_direction=False, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr[0]], 1)
        self.assertEqual(layout[qr[1]], 0)
        self.assertEqual(pass_.property_set["CSPLayout_stop_reason"], "solution found")

    def test_3q_circuit_5q_coupling(self):
        """3 qubits in Tenerife, without considering the direction
            qr1
           /  |
        qr0 - qr2 - 3
              |   /
               4
        """
        cmap5 = FakeTenerife().configuration().coupling_map

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap(cmap5), strict_direction=False, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr[0]], 3)
        self.assertEqual(layout[qr[1]], 2)
        self.assertEqual(layout[qr[2]], 4)
        self.assertEqual(pass_.property_set["CSPLayout_stop_reason"], "solution found")

    def test_3q_circuit_5q_coupling_with_target(self):
        """3 qubits in Tenerife, without considering the direction
            qr1
           /  |
        qr0 - qr2 - 3
              |   /
               4
        """
        target = FakeYorktownV2().target

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(target, strict_direction=False, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr[0]], 3)
        self.assertEqual(layout[qr[1]], 2)
        self.assertEqual(layout[qr[2]], 4)
        self.assertEqual(pass_.property_set["CSPLayout_stop_reason"], "solution found")

    def test_9q_circuit_16q_coupling(self):
        """9 qubits in Rueschlikon, without considering the direction
        q0[1] - q0[0] - q1[3] - q0[3] - q1[0] - q1[1] - q1[2] - 8
          |       |       |       |       |       |       |     |
        q0[2] - q1[4] -- 14 ---- 13 ---- 12 ---- 11 ---- 10 --- 9
        """
        cmap16 = FakeRueschlikon().configuration().coupling_map

        qr0 = QuantumRegister(4, "q0")
        qr1 = QuantumRegister(5, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[1], qr0[2])  # q0[1] -> q0[2]
        circuit.cx(qr0[0], qr1[3])  # q0[0] -> q1[3]
        circuit.cx(qr1[4], qr0[2])  # q1[4] -> q0[2]

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap(cmap16), strict_direction=False, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr0[0]], 9)
        self.assertEqual(layout[qr0[1]], 6)
        self.assertEqual(layout[qr0[2]], 7)
        self.assertEqual(layout[qr0[3]], 5)
        self.assertEqual(layout[qr1[0]], 14)
        self.assertEqual(layout[qr1[1]], 12)
        self.assertEqual(layout[qr1[2]], 1)
        self.assertEqual(layout[qr1[3]], 8)
        self.assertEqual(layout[qr1[4]], 10)
        self.assertEqual(pass_.property_set["CSPLayout_stop_reason"], "solution found")

    def test_2q_circuit_2q_coupling_sd(self):
        """A simple example, considering the direction
         0  -> 1
        qr1 -> qr0
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap([[0, 1]]), strict_direction=True, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr[0]], 1)
        self.assertEqual(layout[qr[1]], 0)
        self.assertEqual(pass_.property_set["CSPLayout_stop_reason"], "solution found")

    def test_3q_circuit_5q_coupling_sd(self):
        """3 qubits in Tenerife, considering the direction
              qr0
            ↙  ↑
        qr2 ← qr1 ← 3
               ↑  ↙
               4
        """
        cmap5 = FakeTenerife().configuration().coupling_map

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap(cmap5), strict_direction=True, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr[0]], 1)
        self.assertEqual(layout[qr[1]], 2)
        self.assertEqual(layout[qr[2]], 0)
        self.assertEqual(pass_.property_set["CSPLayout_stop_reason"], "solution found")

    def test_9q_circuit_16q_coupling_sd(self):
        """9 qubits in Rueschlikon, considering the direction
        q0[1] → q0[0] → q1[3] → q0[3] ← q1[0] ← q1[1] → q1[2] ← 8
          ↓       ↑      ↓      ↓       ↑       ↓        ↓      ↑
        q0[2] ← q1[4] → 14  ←  13   ←  12   →  11   →   10   ←  9
        """
        cmap16 = FakeRueschlikon().configuration().coupling_map

        qr0 = QuantumRegister(4, "q0")
        qr1 = QuantumRegister(5, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[1], qr0[2])  # q0[1] -> q0[2]
        circuit.cx(qr0[0], qr1[3])  # q0[0] -> q1[3]
        circuit.cx(qr1[4], qr0[2])  # q1[4] -> q0[2]

        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap(cmap16), strict_direction=True, seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]

        self.assertEqual(layout[qr0[0]], 9)
        self.assertEqual(layout[qr0[1]], 6)
        self.assertEqual(layout[qr0[2]], 7)
        self.assertEqual(layout[qr0[3]], 5)
        self.assertEqual(layout[qr1[0]], 14)
        self.assertEqual(layout[qr1[1]], 12)
        self.assertEqual(layout[qr1[2]], 1)
        self.assertEqual(layout[qr1[3]], 10)
        self.assertEqual(layout[qr1[4]], 8)
        self.assertEqual(pass_.property_set["CSPLayout_stop_reason"], "solution found")

    def test_5q_circuit_16q_coupling_no_solution(self):
        """5 qubits in Rueschlikon, no solution

        q0[1] ↖     ↗ q0[2]
               q0[0]
        q0[3] ↙     ↘ q0[4]
        """
        cmap16 = FakeRueschlikon().configuration().coupling_map

        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[0], qr[4])
        dag = circuit_to_dag(circuit)
        pass_ = CSPLayout(CouplingMap(cmap16), seed=self.seed)
        pass_.run(dag)
        layout = pass_.property_set["layout"]
        self.assertIsNone(layout)
        self.assertEqual(pass_.property_set["CSPLayout_stop_reason"], "nonexistent solution")

    @staticmethod
    def create_hard_dag():
        """Creates a particularly hard circuit (returns its dag) for Tokyo"""
        circuit = QuantumCircuit(20)
        circuit.cx(13, 12)
        circuit.cx(6, 0)
        circuit.cx(5, 10)
        circuit.cx(10, 7)
        circuit.cx(5, 12)
        circuit.cx(2, 15)
        circuit.cx(16, 18)
        circuit.cx(6, 4)
        circuit.cx(10, 3)
        circuit.cx(11, 10)
        circuit.cx(18, 16)
        circuit.cx(5, 12)
        circuit.cx(4, 0)
        circuit.cx(18, 16)
        circuit.cx(2, 15)
        circuit.cx(7, 8)
        circuit.cx(9, 6)
        circuit.cx(16, 17)
        circuit.cx(9, 3)
        circuit.cx(14, 12)
        circuit.cx(2, 15)
        circuit.cx(1, 16)
        circuit.cx(5, 3)
        circuit.cx(8, 12)
        circuit.cx(2, 1)
        circuit.cx(5, 3)
        circuit.cx(13, 5)
        circuit.cx(12, 14)
        circuit.cx(12, 13)
        circuit.cx(6, 4)
        circuit.cx(15, 18)
        circuit.cx(15, 18)
        return circuit_to_dag(circuit)

    def test_time_limit(self):
        """Hard to solve situations hit the time limit"""
        dag = TestCSPLayout.create_hard_dag()
        coupling_map = CouplingMap(FakeTokyo().configuration().coupling_map)
        pass_ = CSPLayout(coupling_map, call_limit=None, time_limit=1)

        start = process_time()
        pass_.run(dag)
        runtime = process_time() - start

        self.assertLess(runtime, 3)
        self.assertEqual(pass_.property_set["CSPLayout_stop_reason"], "time limit reached")

    def test_call_limit(self):
        """Hard to solve situations hit the call limit"""
        dag = TestCSPLayout.create_hard_dag()
        coupling_map = CouplingMap(FakeTokyo().configuration().coupling_map)
        pass_ = CSPLayout(coupling_map, call_limit=1, time_limit=None)

        start = process_time()
        pass_.run(dag)
        runtime = process_time() - start

        self.assertLess(runtime, 1)
        self.assertEqual(pass_.property_set["CSPLayout_stop_reason"], "call limit reached")

    def test_seed(self):
        """Different seeds yield different results"""
        seed_1 = 42
        seed_2 = 43

        cmap5 = FakeTenerife().configuration().coupling_map

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2
        dag = circuit_to_dag(circuit)

        pass_1 = CSPLayout(CouplingMap(cmap5), seed=seed_1)
        pass_1.run(dag)
        layout_1 = pass_1.property_set["layout"]

        pass_2 = CSPLayout(CouplingMap(cmap5), seed=seed_2)
        pass_2.run(dag)
        layout_2 = pass_2.property_set["layout"]

        self.assertNotEqual(layout_1, layout_2)


if __name__ == "__main__":
    unittest.main()

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the VF2Layout pass"""

import rustworkx

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import ControlFlowOp
from qiskit.circuit.library import CXGate, XGate
from qiskit.transpiler import CouplingMap, Layout, TranspilerError
from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayout, VF2PostLayoutStopReason
from qiskit.converters import circuit_to_dag
from qiskit.providers.fake_provider import Fake5QV1, GenericBackendV2
from qiskit.circuit import Qubit
from qiskit.compiler.transpiler import transpile
from qiskit.transpiler.target import Target, InstructionProperties
from test import QiskitTestCase  # pylint: disable=wrong-import-order

from ..legacy_cmaps import LIMA_CMAP, YORKTOWN_CMAP, BOGOTA_CMAP


class TestVF2PostLayout(QiskitTestCase):
    """Tests the VF2Layout pass"""

    seed = 42

    def assertLayout(self, dag, coupling_map, property_set):
        """Checks if the circuit in dag was a perfect layout in property_set for the given
        coupling_map"""
        self.assertEqual(
            property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.SOLUTION_FOUND
        )

        layout = property_set["post_layout"]
        edges = coupling_map.graph.edge_list()

        def run(dag, wire_map):
            for gate in dag.two_qubit_ops():
                with self.assertWarns(DeprecationWarning):
                    if dag.has_calibration_for(gate) or isinstance(gate.op, ControlFlowOp):
                        continue
                physical_q0 = wire_map[gate.qargs[0]]
                physical_q1 = wire_map[gate.qargs[1]]
                self.assertTrue((physical_q0, physical_q1) in edges)
            for node in dag.op_nodes(ControlFlowOp):
                for block in node.op.blocks:
                    inner_wire_map = {
                        inner: wire_map[outer] for outer, inner in zip(node.qargs, block.qubits)
                    }
                    run(circuit_to_dag(block), inner_wire_map)

        run(dag, {bit: layout[bit] for bit in dag.qubits if bit in layout})

    def assertLayoutV2(self, dag, target, property_set):
        """Checks if the circuit in dag was a perfect layout in property_set for the given
        coupling_map"""
        self.assertEqual(
            property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.SOLUTION_FOUND
        )

        layout = property_set["post_layout"]

        def run(dag, wire_map):
            for gate in dag.two_qubit_ops():
                with self.assertWarns(DeprecationWarning):
                    if dag.has_calibration_for(gate) or isinstance(gate.op, ControlFlowOp):
                        continue
                physical_q0 = wire_map[gate.qargs[0]]
                physical_q1 = wire_map[gate.qargs[1]]
                qargs = (physical_q0, physical_q1)
                self.assertTrue(target.instruction_supported(gate.name, qargs))
            for node in dag.op_nodes(ControlFlowOp):
                for block in node.op.blocks:
                    inner_wire_map = {
                        inner: wire_map[outer] for outer, inner in zip(node.qargs, block.qubits)
                    }
                    run(circuit_to_dag(block), inner_wire_map)

        run(dag, {bit: layout[bit] for bit in dag.qubits if bit in layout})

    def test_no_constraints(self):
        """Test we raise at runtime if no target or coupling graph specified."""
        qc = QuantumCircuit(2)
        empty_pass = VF2PostLayout()
        with self.assertRaises(TranspilerError):
            empty_pass.run(circuit_to_dag(qc))

    def test_no_backend_properties(self):
        """Test we raise at runtime if no properties are provided with a coupling graph."""
        qc = QuantumCircuit(2)
        empty_pass = VF2PostLayout(coupling_map=CouplingMap([(0, 1), (1, 2)]))
        with self.assertRaises(TranspilerError):
            empty_pass.run(circuit_to_dag(qc))

    def test_empty_circuit(self):
        """Test no solution found for empty circuit"""
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()
        qc = QuantumCircuit(2, 2)
        cmap = CouplingMap(backend.configuration().coupling_map)
        props = backend.properties()
        vf2_pass = VF2PostLayout(coupling_map=cmap, properties=props)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND,
        )

    def test_empty_circuit_v2(self):
        """Test no solution found for empty circuit with v2 backend"""
        qc = QuantumCircuit(2, 2)
        target = GenericBackendV2(
            num_qubits=5, basis_gates=["cx", "id", "rz", "sx", "x"], coupling_map=LIMA_CMAP, seed=42
        ).target
        vf2_pass = VF2PostLayout(target=target)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND,
        )

    def test_skip_3q_circuit(self):
        """Test that the pass is a no-op on circuits with >2q gates."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        cmap = CouplingMap(backend.configuration().coupling_map)
        props = backend.properties()
        vf2_pass = VF2PostLayout(coupling_map=cmap, properties=props)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.MORE_THAN_2Q
        )

    def test_skip_3q_circuit_control_flow(self):
        """Test that the pass is a no-op on circuits with >2q gates."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()
        qc = QuantumCircuit(3)
        with qc.for_loop((1,)):
            qc.ccx(0, 1, 2)
        cmap = CouplingMap(backend.configuration().coupling_map)
        props = backend.properties()
        vf2_pass = VF2PostLayout(coupling_map=cmap, properties=props)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.MORE_THAN_2Q
        )

    def test_skip_3q_circuit_v2(self):
        """Test that the pass is a no-op on circuits with >2q gates with a target."""
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        target = GenericBackendV2(
            num_qubits=5, basis_gates=["cx", "id", "rz", "sx", "x"], coupling_map=LIMA_CMAP, seed=42
        ).target
        vf2_pass = VF2PostLayout(target=target)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.MORE_THAN_2Q
        )

    def test_skip_3q_circuit_control_flow_v2(self):
        """Test that the pass is a no-op on circuits with >2q gates with a target."""
        qc = QuantumCircuit(3)
        with qc.for_loop((1,)):
            qc.ccx(0, 1, 2)
        target = GenericBackendV2(
            num_qubits=5, basis_gates=["cx", "id", "rz", "sx", "x"], coupling_map=LIMA_CMAP, seed=42
        ).target
        vf2_pass = VF2PostLayout(target=target)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.MORE_THAN_2Q
        )

    def test_2q_circuit_5q_backend(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            tqc = transpile(circuit, backend, layout_method="dense")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        cmap = CouplingMap(backend.configuration().coupling_map)
        props = backend.properties()
        pass_ = VF2PostLayout(coupling_map=cmap, properties=props, seed=self.seed)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_2q_circuit_5q_backend_controlflow(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()

        circuit = QuantumCircuit(2, 1)
        with circuit.for_loop((1,)):
            circuit.cx(1, 0)  # qr1 -> qr0
        with circuit.if_test((circuit.clbits[0], True)) as else_:
            pass
        with else_:
            with circuit.while_loop((circuit.clbits[0], True)):
                circuit.cx(1, 0)  # qr1 -> qr0
        initial_layout = Layout(dict(enumerate(circuit.qubits)))
        circuit._layout = initial_layout
        dag = circuit_to_dag(circuit)
        cmap = CouplingMap(backend.configuration().coupling_map)
        props = backend.properties()
        pass_ = VF2PostLayout(coupling_map=cmap, properties=props, seed=self.seed)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_2q_circuit_5q_backend_max_trials(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        max_trials = 11
        backend = GenericBackendV2(
            num_qubits=5,
            coupling_map=YORKTOWN_CMAP,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            seed=1,
        )

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        tqc = transpile(circuit, backend, layout_method="dense")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        cmap = CouplingMap(backend.coupling_map)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed, max_trials=max_trials)
        with self.assertLogs(
            "qiskit.transpiler.passes.layout.vf2_post_layout", level="DEBUG"
        ) as cm:
            pass_.run(dag)
        self.assertIn(
            f"DEBUG:qiskit.transpiler.passes.layout.vf2_post_layout:Trial {max_trials} "
            f"is >= configured max trials {max_trials}",
            cm.output,
        )
        print(pass_.property_set["VF2PostLayout_stop_reason"])
        self.assertLayout(dag, cmap, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_2q_circuit_5q_backend_max_trials_v1(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        max_trials = 11
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            tqc = transpile(circuit, backend, layout_method="dense")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        cmap = CouplingMap(backend.configuration().coupling_map)
        props = backend.properties()
        pass_ = VF2PostLayout(
            coupling_map=cmap, properties=props, seed=self.seed, max_trials=max_trials
        )

        with self.assertLogs(
            "qiskit.transpiler.passes.layout.vf2_post_layout", level="DEBUG"
        ) as cm:
            pass_.run(dag)
        self.assertIn(
            f"DEBUG:qiskit.transpiler.passes.layout.vf2_post_layout:Trial {max_trials} "
            f"is >= configured max trials {max_trials}",
            cm.output,
        )

        self.assertLayout(dag, cmap, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_best_mapping_ghz_state_full_device_multiple_qregs(self):
        """Test best mappings with multiple registers"""
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LIMA_CMAP,
            seed=123,
        )
        qr_a = QuantumRegister(2)
        qr_b = QuantumRegister(3)
        qc = QuantumCircuit(qr_a, qr_b)
        qc.h(qr_a[0])
        qc.cx(qr_a[0], qr_a[1])
        qc.cx(qr_a[0], qr_b[0])
        qc.cx(qr_a[0], qr_b[1])
        qc.cx(qr_a[0], qr_b[2])
        qc.measure_all()
        tqc = transpile(qc, backend, seed_transpiler=self.seed, layout_method="trivial")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed)
        pass_.run(dag)
        self.assertLayoutV2(dag, backend.target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_2q_circuit_5q_backend_v2(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=YORKTOWN_CMAP,
            seed=42,
        )

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        tqc = transpile(circuit, backend, layout_method="dense")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed)
        pass_.run(dag)
        self.assertLayoutV2(dag, backend.target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_2q_circuit_5q_backend_v2_control_flow(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        target = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=YORKTOWN_CMAP,
            seed=42,
        ).target

        circuit = QuantumCircuit(2, 1)
        with circuit.for_loop((1,)):
            circuit.cx(1, 0)  # qr1 -> qr0
        with circuit.if_test((circuit.clbits[0], True)) as else_:
            pass
        with else_:
            with circuit.while_loop((circuit.clbits[0], True)):
                circuit.cx(1, 0)  # qr1 -> qr0
        initial_layout = Layout(dict(enumerate(circuit.qubits)))
        circuit._layout = initial_layout
        dag = circuit_to_dag(circuit)
        pass_ = VF2PostLayout(target=target, seed=self.seed)
        pass_.run(dag)
        self.assertLayoutV2(dag, target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_target_invalid_2q_gate(self):
        """Test that we don't find a solution with a gate outside target."""
        target = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=YORKTOWN_CMAP,
            seed=42,
        ).target
        qc = QuantumCircuit(2)
        qc.ecr(0, 1)
        dag = circuit_to_dag(qc)
        pass_ = VF2PostLayout(target=target, seed=self.seed)
        pass_.run(dag)
        self.assertEqual(
            pass_.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.NO_SOLUTION_FOUND,
        )

    def test_target_invalid_2q_gate_control_flow(self):
        """Test that we don't find a solution with a gate outside target."""
        target = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=YORKTOWN_CMAP,
            seed=42,
        ).target
        qc = QuantumCircuit(2)
        with qc.for_loop((1,)):
            qc.ecr(0, 1)
        dag = circuit_to_dag(qc)
        pass_ = VF2PostLayout(target=target, seed=self.seed)
        pass_.run(dag)
        self.assertEqual(
            pass_.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.NO_SOLUTION_FOUND,
        )

    def test_target_no_error(self):
        """Test that running vf2layout on a pass against a target with no error rates works."""
        n_qubits = 15
        target = Target()
        target.add_instruction(CXGate(), {(i, i + 1): None for i in range(n_qubits - 1)})
        vf2_pass = VF2PostLayout(target=target)
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        dag = circuit_to_dag(circuit)
        vf2_pass.run(dag)
        self.assertNotIn("post_layout", vf2_pass.property_set)

    def test_target_some_error(self):
        """Test that running vf2layout on a pass against a target with some error rates works."""
        n_qubits = 15
        target = Target()
        target.add_instruction(
            XGate(), {(i,): InstructionProperties(error=0.00123) for i in range(n_qubits)}
        )
        target.add_instruction(CXGate(), {(i, i + 1): None for i in range(n_qubits - 1)})
        vf2_pass = VF2PostLayout(target=target, seed=1234, strict_direction=False)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        dag = circuit_to_dag(circuit)
        vf2_pass.run(dag)
        # No layout selected because nothing will beat initial layout
        self.assertNotIn("post_layout", vf2_pass.property_set)

    def test_trivial_layout_is_best(self):
        """Test that vf2postlayout reports no better solution if the trivial layout is the best layout"""
        n_qubits = 4
        trivial_target = Target()
        trivial_target.add_instruction(
            CXGate(), {(i, i + 1): InstructionProperties(error=0.001) for i in range(n_qubits - 1)}
        )

        circuit = QuantumCircuit(n_qubits)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        vf2_pass = VF2PostLayout(target=trivial_target, seed=self.seed, strict_direction=False)
        dag = circuit_to_dag(circuit)
        vf2_pass.run(dag)
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND,
        )

    def test_last_qubits_best(self):
        """Test that vf2postlayout determines the best layout when the last qubits have least error"""
        n_qubits = 4
        target_last_qubits_best = Target()
        target_last_qubits_best.add_instruction(
            CXGate(),
            {(i, i + 1): InstructionProperties(error=10**-i) for i in range(n_qubits - 1)},
        )

        circuit = QuantumCircuit(n_qubits)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        vf2_pass = VF2PostLayout(
            target=target_last_qubits_best, seed=self.seed, strict_direction=False
        )
        dag = circuit_to_dag(circuit)
        vf2_pass.run(dag)
        self.assertLayout(dag, target_last_qubits_best.build_coupling_map(), vf2_pass.property_set)


class TestVF2PostLayoutScoring(QiskitTestCase):
    """Test scoring heuristic function for VF2PostLayout."""

    def test_empty_score(self):
        """Test error rate is 0 for empty circuit."""
        bit_map = {}
        reverse_bit_map = {}
        im_graph = rustworkx.PyDiGraph()
        target = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=YORKTOWN_CMAP,
            seed=42,
        ).target
        vf2_pass = VF2PostLayout(target=target)
        layout = Layout()
        score = vf2_pass._score_layout(layout, bit_map, reverse_bit_map, im_graph)
        self.assertEqual(0, score)

    def test_all_1q_score(self):
        """Test error rate for all 1q input."""
        bit_map = {Qubit(): 0, Qubit(): 1}
        reverse_bit_map = {v: k for k, v in bit_map.items()}
        im_graph = rustworkx.PyDiGraph()
        im_graph.add_node({"sx": 1})
        im_graph.add_node({"sx": 1})

        target = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=YORKTOWN_CMAP,
            seed=42,
        ).target

        target.update_instruction_properties(
            "sx", (0,), InstructionProperties(duration=3.56e-08, error=0.0013043388897769352)
        )
        target.update_instruction_properties(
            "sx", (1,), InstructionProperties(duration=3.56e-08, error=0.0016225037300878712)
        )

        vf2_pass = VF2PostLayout(target=target)
        layout = Layout(bit_map)
        score = vf2_pass._score_layout(layout, bit_map, reverse_bit_map, im_graph)
        self.assertAlmostEqual(0.002925, score, places=5)


class TestVF2PostLayoutUndirected(QiskitTestCase):
    """Tests the VF2Layout pass"""

    seed = 42

    def assertLayout(self, dag, coupling_map, property_set):
        """Checks if the circuit in dag was a perfect layout in property_set for the given
        coupling_map"""
        self.assertEqual(
            property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.SOLUTION_FOUND
        )

        layout = property_set["post_layout"]
        for gate in dag.two_qubit_ops():
            with self.assertWarns(DeprecationWarning):
                if dag.has_calibration_for(gate):
                    continue
            physical_q0 = layout[gate.qargs[0]]
            physical_q1 = layout[gate.qargs[1]]
            self.assertTrue(coupling_map.graph.has_edge(physical_q0, physical_q1))

    def assertLayoutV2(self, dag, target, property_set):
        """Checks if the circuit in dag was a perfect layout in property_set for the given
        coupling_map"""
        self.assertEqual(
            property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.SOLUTION_FOUND
        )

        layout = property_set["post_layout"]
        for gate in dag.two_qubit_ops():
            with self.assertWarns(DeprecationWarning):
                if dag.has_calibration_for(gate):
                    continue
            physical_q0 = layout[gate.qargs[0]]
            physical_q1 = layout[gate.qargs[1]]
            qargs = (physical_q0, physical_q1)
            self.assertTrue(target.instruction_supported(gate.name, qargs))

    def test_no_constraints(self):
        """Test we raise at runtime if no target or coupling graph specified."""
        qc = QuantumCircuit(2)
        empty_pass = VF2PostLayout(strict_direction=False)
        with self.assertRaises(TranspilerError):
            empty_pass.run(circuit_to_dag(qc))

    def test_no_backend_properties(self):
        """Test we raise at runtime if no properties are provided with a coupling graph."""
        qc = QuantumCircuit(2)
        empty_pass = VF2PostLayout(
            coupling_map=CouplingMap([(0, 1), (1, 2)]), strict_direction=False
        )
        with self.assertRaises(TranspilerError):
            empty_pass.run(circuit_to_dag(qc))

    def test_empty_circuit(self):
        """Test no solution found for empty circuit"""
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()

        qc = QuantumCircuit(2, 2)
        cmap = CouplingMap(backend.configuration().coupling_map)
        props = backend.properties()
        vf2_pass = VF2PostLayout(coupling_map=cmap, properties=props, strict_direction=False)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND,
        )

    def test_empty_circuit_v2(self):
        """Test no solution found for empty circuit with v2 backend"""
        qc = QuantumCircuit(2, 2)
        target = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LIMA_CMAP,
            seed=self.seed,
        ).target
        vf2_pass = VF2PostLayout(target=target, strict_direction=False)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND,
        )

    def test_skip_3q_circuit(self):
        """Test that the pass is a no-op on circuits with >2q gates."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()

        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        cmap = CouplingMap(backend.configuration().coupling_map)
        props = backend.properties()
        vf2_pass = VF2PostLayout(coupling_map=cmap, properties=props, strict_direction=False)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.MORE_THAN_2Q,
        )

    def test_skip_3q_circuit_v2(self):
        """Test that the pass is a no-op on circuits with >2q gates with a target."""
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        target = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LIMA_CMAP,
            seed=self.seed,
        ).target
        vf2_pass = VF2PostLayout(target=target, strict_direction=False)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.MORE_THAN_2Q,
        )

    def test_best_mapping_ghz_state_full_device_multiple_qregs(self):
        """Test best mappings with multiple registers"""
        backend = GenericBackendV2(
            num_qubits=5,
            coupling_map=YORKTOWN_CMAP,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            seed=8,
        )
        qr_a = QuantumRegister(2)
        qr_b = QuantumRegister(3)
        qc = QuantumCircuit(qr_a, qr_b)
        qc.h(qr_a[0])
        qc.cx(qr_a[0], qr_a[1])
        qc.cx(qr_a[0], qr_b[0])
        qc.cx(qr_a[0], qr_b[1])
        qc.cx(qr_a[0], qr_b[2])
        qc.measure_all()
        tqc = transpile(qc, seed_transpiler=self.seed, layout_method="trivial")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        cmap = CouplingMap(backend.coupling_map)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed, strict_direction=False)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_best_mapping_ghz_state_full_device_multiple_qregs_v1(self):
        """Test best mappings with multiple registers"""
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()
        qr_a = QuantumRegister(2)
        qr_b = QuantumRegister(3)
        qc = QuantumCircuit(qr_a, qr_b)
        qc.h(qr_a[0])
        qc.cx(qr_a[0], qr_a[1])
        qc.cx(qr_a[0], qr_b[0])
        qc.cx(qr_a[0], qr_b[1])
        qc.cx(qr_a[0], qr_b[2])
        qc.measure_all()
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            tqc = transpile(qc, backend, seed_transpiler=self.seed, layout_method="trivial")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        cmap = CouplingMap(backend.configuration().coupling_map)
        props = backend.properties()
        pass_ = VF2PostLayout(
            coupling_map=cmap, properties=props, seed=self.seed, strict_direction=False
        )
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_2q_circuit_5q_backend(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        backend = GenericBackendV2(
            num_qubits=5,
            coupling_map=BOGOTA_CMAP,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
            seed=42,
        )
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        tqc = transpile(circuit, backend, layout_method="dense")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        cmap = CouplingMap(backend.coupling_map)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed, strict_direction=False)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_2q_circuit_5q_backend_v1(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            tqc = transpile(circuit, backend, layout_method="dense")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        cmap = CouplingMap(backend.configuration().coupling_map)
        props = backend.properties()
        pass_ = VF2PostLayout(
            coupling_map=cmap, properties=props, seed=self.seed, strict_direction=False
        )
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_best_mapping_ghz_state_full_device_multiple_qregs_v2(self):
        """Test best mappings with multiple registers"""

        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LIMA_CMAP,
            seed=self.seed,
        )
        qr_a = QuantumRegister(2)
        qr_b = QuantumRegister(3)
        qc = QuantumCircuit(qr_a, qr_b)
        qc.h(qr_a[0])
        qc.cx(qr_a[0], qr_a[1])
        qc.cx(qr_a[0], qr_b[0])
        qc.cx(qr_a[0], qr_b[1])
        qc.cx(qr_a[0], qr_b[2])
        qc.measure_all()
        tqc = transpile(qc, backend, seed_transpiler=self.seed, layout_method="trivial")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed, strict_direction=False)
        pass_.run(dag)
        self.assertLayoutV2(dag, backend.target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_2q_circuit_5q_backend_v2(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=YORKTOWN_CMAP,
            seed=self.seed,
        )

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        tqc = transpile(circuit, backend, layout_method="dense")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed, strict_direction=False)
        pass_.run(dag)
        self.assertLayoutV2(dag, backend.target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

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

"""
Tests for the ConsolidateBlocks transpiler pass.
"""

import unittest
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, IfElseOp, Gate
from qiskit.circuit.library import U2Gate, SwapGate, CXGate, CZGate, UnitaryGate
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import ConsolidateBlocks
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.operators.measures import process_fidelity
from qiskit.transpiler import PassManager
from qiskit.transpiler import Target
from qiskit.transpiler.passes import Collect1qRuns
from qiskit.transpiler.passes import Collect2qBlocks
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestConsolidateBlocks(QiskitTestCase):
    """
    Tests to verify that consolidating blocks of gates into unitaries
    works correctly.
    """

    def test_consolidate_small_block(self):
        """test a small block of gates can be turned into a unitary on same wires"""
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc.p(0.5, qr[0])
        qc.u(1.5708, 0.2, 0.6, qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)

        pass_ = ConsolidateBlocks(force_consolidate=True)
        pass_.property_set["block_list"] = [list(dag.topological_op_nodes())]
        new_dag = pass_.run(dag)

        unitary = Operator(qc)
        self.assertEqual(len(new_dag.op_nodes()), 1)
        fidelity = process_fidelity(Operator(new_dag.op_nodes()[0].op), unitary)
        self.assertAlmostEqual(fidelity, 1.0, places=7)

    def test_wire_order(self):
        """order of qubits and the corresponding unitary is correct"""
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc.cx(qr[1], qr[0])
        dag = circuit_to_dag(qc)

        pass_ = ConsolidateBlocks(force_consolidate=True)
        pass_.property_set["block_list"] = [dag.op_nodes()]
        new_dag = pass_.run(dag)

        new_node = new_dag.op_nodes()[0]
        self.assertEqual(new_node.qargs, (qr[0], qr[1]))
        unitary = Operator(qc)
        fidelity = process_fidelity(Operator(new_node.op), unitary)
        self.assertAlmostEqual(fidelity, 1.0, places=7)

    def test_topological_order_preserved(self):
        """the original topological order of nodes is preserved
                                                   ______
        q0:--[p]-------.----      q0:-------------|      |--
                       |                 ______   |  U2  |
        q1:--[u2]--(+)-(+)--   =  q1:---|      |--|______|--
                    |                   |  U1  |
        q2:---------.-------      q2:---|______|------------
        """
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.p(0.5, qr[0])
        qc.u(1.5708, 0.2, 0.6, qr[1])
        qc.cx(qr[2], qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)

        pass_ = ConsolidateBlocks(force_consolidate=True)
        topo_ops = list(dag.topological_op_nodes())
        block_1 = [topo_ops[1], topo_ops[2]]
        block_2 = [topo_ops[0], topo_ops[3]]
        pass_.property_set["block_list"] = [block_1, block_2]
        new_dag = pass_.run(dag)

        new_topo_ops = list(new_dag.topological_op_nodes())
        self.assertEqual(len(new_topo_ops), 2)
        self.assertEqual(new_topo_ops[0].qargs, (qr[1], qr[2]))
        self.assertEqual(new_topo_ops[1].qargs, (qr[0], qr[1]))

    def test_3q_blocks(self):
        """blocks of more than 2 qubits work."""

        #             ┌────────┐
        # qr_0: ──────┤ P(0.5) ├────────────■──
        #       ┌─────┴────────┴────┐┌───┐┌─┴─┐
        # qr_1: ┤ U(1.5708,0.2,0.6) ├┤ X ├┤ X ├
        #       └───────────────────┘└─┬─┘└───┘
        # qr_2: ───────────────────────■───────
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.p(0.5, qr[0])
        qc.u(1.5708, 0.2, 0.6, qr[1])
        qc.cx(qr[2], qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)

        pass_ = ConsolidateBlocks(force_consolidate=True)
        pass_.property_set["block_list"] = [list(dag.topological_op_nodes())]
        new_dag = pass_.run(dag)

        unitary = Operator(qc)
        self.assertEqual(len(new_dag.op_nodes()), 1)
        fidelity = process_fidelity(Operator(new_dag.op_nodes()[0].op), unitary)
        self.assertAlmostEqual(fidelity, 1.0, places=7)

    def test_block_spanning_two_regs(self):
        """blocks spanning wires on different quantum registers work."""

        #            ┌────────┐
        # qr0: ──────┤ P(0.5) ├───────■──
        #      ┌─────┴────────┴────┐┌─┴─┐
        # qr1: ┤ U(1.5708,0.2,0.6) ├┤ X ├
        #      └───────────────────┘└───┘
        qr0 = QuantumRegister(1, "qr0")
        qr1 = QuantumRegister(1, "qr1")
        qc = QuantumCircuit(qr0, qr1)
        qc.p(0.5, qr0[0])
        qc.u(1.5708, 0.2, 0.6, qr1[0])
        qc.cx(qr0[0], qr1[0])
        dag = circuit_to_dag(qc)

        pass_ = ConsolidateBlocks(force_consolidate=True)
        pass_.property_set["block_list"] = [list(dag.topological_op_nodes())]
        new_dag = pass_.run(dag)

        unitary = Operator(qc)
        self.assertEqual(len(new_dag.op_nodes()), 1)
        fidelity = process_fidelity(Operator(new_dag.op_nodes()[0].op), unitary)
        self.assertAlmostEqual(fidelity, 1.0, places=7)

    def test_block_spanning_two_regs_different_index(self):
        """blocks spanning wires on different quantum registers work when the wires
        could have conflicting indices. This was raised in #2806 when a CX was applied
        across multiple registers and their indices collided, raising an error."""
        qr0 = QuantumRegister(1, "qr0")
        qr1 = QuantumRegister(2, "qr1")
        qc = QuantumCircuit(qr0, qr1)
        qc.cx(qr0[0], qr1[1])
        dag = circuit_to_dag(qc)

        pass_ = ConsolidateBlocks(force_consolidate=True)
        pass_.property_set["block_list"] = [list(dag.topological_op_nodes())]
        new_dag = pass_.run(dag)

        original_unitary = UnitaryGate(Operator(qc))

        from qiskit.converters import dag_to_circuit

        new_unitary = UnitaryGate(Operator(dag_to_circuit(new_dag)))

        self.assertEqual(original_unitary, new_unitary)

    def test_node_added_before_block(self):
        """Test that a node before a block remains before the block

        This issue was raised in #2737 where the measure was moved
        to be after the 2nd ID gate, as the block was added when the
        first node in the block was seen.

        blocks = [['id', 'cx', 'id']]
        """
        #         ┌────┐┌───┐
        # q_0: |0>┤ Id ├┤ X ├──────
        #         └┬─┬─┘└─┬─┘┌────┐
        # q_1: |0>─┤M├────■──┤ Id ├
        #          └╥┘       └────┘
        # c_0:  0 ══╩══════════════
        qc = QuantumCircuit(2, 1)
        qc.id(0)
        qc.measure(1, 0)
        qc.cx(1, 0)
        qc.id(1)

        # can't just add all the nodes to one block as in other tests
        # as we are trying to test the block gets added in the correct place
        # so use a pass to collect the blocks instead
        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(ConsolidateBlocks())
        qc1 = pass_manager.run(qc)

        self.assertEqual(qc, qc1)

    def test_consolidate_blocks_big(self):
        """Test ConsolidateBlocks with U2(<big numbers>)
        https://github.com/Qiskit/qiskit-terra/issues/3637#issuecomment-612954865
        """
        #      ┌────────────────┐     ┌───┐
        # q_0: ┤ U2(-804.15,pi) ├──■──┤ X ├
        #      ├────────────────┤┌─┴─┐└─┬─┘
        # q_1: ┤ U2(-6433.2,pi) ├┤ X ├──■──
        #      └────────────────┘└───┘
        circuit = QuantumCircuit(2)
        circuit.append(U2Gate(-804.15, np.pi), [0])
        circuit.append(U2Gate(-6433.2, np.pi), [1])
        circuit.cx(0, 1)
        circuit.cx(1, 0)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(ConsolidateBlocks())
        result = pass_manager.run(circuit)

        self.assertEqual(circuit, result)

    def test_node_added_after_block(self):
        """Test that a node after the block remains after the block

        This example was raised in #2764, and checks that the final CX
        stays after the main block, even though one of the nodes in the
        block was declared after it. This occurred when the block was
        added when the last node in the block was seen.

        blocks = [['cx', 'id', 'id']]

        q_0: |0>─────────────■──
                     ┌────┐┌─┴─┐
        q_1: |0>──■──┤ Id ├┤ X ├
                ┌─┴─┐├────┤└───┘
        q_2: |0>┤ X ├┤ Id ├─────
                └───┘└────┘
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 2)
        qc.id(1)
        qc.cx(0, 1)
        qc.id(2)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(ConsolidateBlocks())
        qc1 = pass_manager.run(qc)

        self.assertEqual(qc, qc1)

    def test_node_middle_of_blocks(self):
        """Test that a node surrounded by blocks stays in the same place

        This is a larger test to ensure multiple blocks can all be collected
        and added back in the correct order.

        blocks = [['cx', 'id'], ['cx', 'id'], ['id', 'cx'], ['id', 'cx']]

        q_0: |0>──■───────────────────■──
                ┌─┴─┐┌────┐   ┌────┐┌─┴─┐
        q_1: |0>┤ X ├┤ Id ├─X─┤ Id ├┤ X ├
                ├───┤├────┤ │ ├────┤├───┤
        q_2: |0>┤ X ├┤ Id ├─X─┤ Id ├┤ X ├
                └─┬─┘└────┘   └────┘└─┬─┘
        q_3: |0>──■───────────────────■──

        """
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(3, 2)
        qc.id(1)
        qc.id(2)

        qc.swap(1, 2)

        qc.id(1)
        qc.id(2)
        qc.cx(0, 1)
        qc.cx(3, 2)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(ConsolidateBlocks())
        qc1 = pass_manager.run(qc)

        self.assertEqual(qc, qc1)

    def test_overlapping_block_and_run(self):
        """Test that an overlapping block and run only consolidate once"""

        #      ┌───┐┌───┐┌─────┐
        # q_0: ┤ H ├┤ T ├┤ Sdg ├──■────────────────────────
        #      └───┘└───┘└─────┘┌─┴─┐┌───┐┌─────┐┌───┐┌───┐
        # q_1: ─────────────────┤ X ├┤ T ├┤ Sdg ├┤ Z ├┤ I ├
        #                       └───┘└───┘└─────┘└───┘└───┘
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.t(0)
        qc.sdg(0)
        qc.cx(0, 1)
        qc.t(1)
        qc.sdg(1)
        qc.z(1)
        qc.id(1)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(Collect1qRuns())
        pass_manager.append(ConsolidateBlocks(force_consolidate=True))
        result = pass_manager.run(qc)
        expected = Operator(qc)
        # Assert output circuit is a single unitary gate equivalent to
        # unitary of original circuit
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result.data[0].operation, UnitaryGate)
        self.assertTrue(np.allclose(result.data[0].operation.to_matrix(), expected))

    def test_classical_conditions_maintained(self):
        """Test that consolidate blocks doesn't drop the classical conditions
        This issue was raised in #2752
        """
        qc = QuantumCircuit(1, 1)
        qc.h(0).c_if(qc.cregs[0], 1)
        qc.measure(0, 0)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(ConsolidateBlocks())
        qc1 = pass_manager.run(qc)

        self.assertEqual(qc, qc1)

    def test_no_kak_in_basis(self):
        """Test that pass just returns the input dag without a KAK gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        dag = circuit_to_dag(qc)
        consolidate_blocks_pass = ConsolidateBlocks(basis_gates=["u3"])
        res = consolidate_blocks_pass.run(dag)
        self.assertEqual(res, dag)

    def test_single_gate_block_outside_basis(self):
        """Test that a single gate block outside the configured basis gets converted."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        consolidate_block_pass = ConsolidateBlocks(basis_gates=["id", "cx", "rz", "sx", "x"])
        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(consolidate_block_pass)
        expected = QuantumCircuit(2)
        expected.unitary(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), [0, 1])
        self.assertEqual(expected, pass_manager.run(qc))

    def test_single_gate_block_outside_basis_with_target(self):
        """Test a gate outside basis defined in target gets converted."""
        qc = QuantumCircuit(2)
        target = Target(num_qubits=2)
        # Add ideal basis gates to all qubits
        target.add_instruction(CXGate())
        qc.swap(0, 1)
        consolidate_block_pass = ConsolidateBlocks(target=target)
        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(consolidate_block_pass)
        expected = QuantumCircuit(2)
        expected.unitary(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), [0, 1])
        self.assertEqual(expected, pass_manager.run(qc))

    def test_single_gate_block_outside_local_basis_with_target(self):
        """Test that a gate in basis but outside valid qubits is treated as outside basis with target."""
        qc = QuantumCircuit(2)
        target = Target(num_qubits=2)
        # Add ideal cx to (1, 0) only
        target.add_instruction(CXGate(), {(1, 0): None})
        qc.cx(0, 1)
        consolidate_block_pass = ConsolidateBlocks(target=target)
        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(consolidate_block_pass)
        expected = QuantumCircuit(2)
        expected.unitary(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]), [0, 1])
        self.assertEqual(expected, pass_manager.run(qc))

    def test_single_gate_block_outside_target_with_matching_basis_gates(self):
        """Ensure the target is the source of truth with basis_gates also set."""
        qc = QuantumCircuit(2)
        target = Target(num_qubits=2)
        # Add ideal cx to (1, 0) only
        target.add_instruction(SwapGate())
        qc.swap(0, 1)
        consolidate_block_pass = ConsolidateBlocks(
            basis_gates=["id", "cx", "rz", "sx", "x"], target=target
        )
        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(consolidate_block_pass)
        expected = QuantumCircuit(2)
        expected.swap(0, 1)
        self.assertEqual(expected, pass_manager.run(qc))

    def test_identity_unitary_is_removed(self):
        """Test that a 2q identity unitary is removed without a basis."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.h(0)

        pm = PassManager([Collect2qBlocks(), ConsolidateBlocks()])
        self.assertEqual(QuantumCircuit(5), pm.run(qc))

    def test_identity_1q_unitary_is_removed(self):
        """Test that a 1q identity unitary is removed without a basis."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        qc.h(0)
        pm = PassManager([Collect2qBlocks(), Collect1qRuns(), ConsolidateBlocks()])
        self.assertEqual(QuantumCircuit(5), pm.run(qc))

    def test_descent_into_control_flow(self):
        """Test consolidation in blocks when control flow op is the same as at top level."""

        def circuit_of_test_gates():
            qc = QuantumCircuit(2, 1)
            qc.cx(0, 1)
            qc.cx(1, 0)
            return qc

        def do_consolidation(qc):
            pass_manager = PassManager()
            pass_manager.append(Collect2qBlocks())
            pass_manager.append(ConsolidateBlocks(force_consolidate=True))
            return pass_manager.run(qc)

        result_top = do_consolidation(circuit_of_test_gates())

        qc_control_flow = QuantumCircuit(2, 1)
        ifop = IfElseOp((qc_control_flow.clbits[0], False), circuit_of_test_gates(), None)
        qc_control_flow.append(ifop, qc_control_flow.qubits, qc_control_flow.clbits)

        result_block = do_consolidation(qc_control_flow)
        gate_top = result_top[0].operation
        gate_block = result_block[0].operation.blocks[0][0].operation
        np.testing.assert_allclose(gate_top, gate_block)

    def test_not_crossing_between_control_flow_block_and_parent(self):
        """Test that consolidation does not occur across the boundary between control flow
        blocks and the parent circuit."""
        qc = QuantumCircuit(2, 1)
        qc.cx(0, 1)
        qc_true = QuantumCircuit(2, 1)
        qc_false = QuantumCircuit(2, 1)
        qc_true.cx(0, 1)
        qc_false.cz(0, 1)
        ifop = IfElseOp((qc.clbits[0], True), qc_true, qc_false)
        qc.append(ifop, qc.qubits, qc.clbits)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(ConsolidateBlocks(force_consolidate=True))
        qc_out = pass_manager.run(qc)

        self.assertIsInstance(qc_out[0].operation, UnitaryGate)
        np.testing.assert_allclose(CXGate(), qc_out[0].operation)
        op_true = qc_out[1].operation.blocks[0][0].operation
        op_false = qc_out[1].operation.blocks[1][0].operation
        np.testing.assert_allclose(CXGate(), op_true)
        np.testing.assert_allclose(CZGate(), op_false)

    def test_not_crossing_between_control_flow_ops(self):
        """Test that consolidation does not occur between control flow ops."""
        qc = QuantumCircuit(2, 1)
        qc_true = QuantumCircuit(2, 1)
        qc_false = QuantumCircuit(2, 1)
        qc_true.cx(0, 1)
        qc_false.cz(0, 1)
        ifop1 = IfElseOp((qc.clbits[0], True), qc_true, qc_false)
        qc.append(ifop1, qc.qubits, qc.clbits)
        ifop2 = IfElseOp((qc.clbits[0], True), qc_true, qc_false)
        qc.append(ifop2, qc.qubits, qc.clbits)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(ConsolidateBlocks(force_consolidate=True))
        qc_out = pass_manager.run(qc)

        op_true1 = qc_out[0].operation.blocks[0][0].operation
        op_false1 = qc_out[0].operation.blocks[1][0].operation
        op_true2 = qc_out[1].operation.blocks[0][0].operation
        op_false2 = qc_out[1].operation.blocks[1][0].operation
        np.testing.assert_allclose(CXGate(), op_true1)
        np.testing.assert_allclose(CZGate(), op_false1)
        np.testing.assert_allclose(CXGate(), op_true2)
        np.testing.assert_allclose(CZGate(), op_false2)

    def test_inverted_order(self):
        """Test that the `ConsolidateBlocks` pass creates matrices that are correct under the
        application of qubit binding from the outer circuit to the inner block."""
        body = QuantumCircuit(2, 1)
        body.h(0)
        body.cx(0, 1)

        id_op = Operator(np.eye(4))
        bell = Operator(body)

        qc = QuantumCircuit(2, 1)
        # The first two 'if' blocks here represent exactly the same operation as each other on the
        # outer bits, because in the second, the bit-order of the block is reversed, but so is the
        # order of the bits in the outer circuit that they're bound to, which makes them the same.
        # The second two 'if' blocks also represent the same operation as each other, but the 'first
        # two' and 'second two' pairs represent qubit-flipped operations.
        qc.if_test((0, False), body.copy(), qc.qubits, qc.clbits)
        qc.if_test((0, False), body.reverse_bits(), reversed(qc.qubits), qc.clbits)
        qc.if_test((0, False), body.copy(), reversed(qc.qubits), qc.clbits)
        qc.if_test((0, False), body.reverse_bits(), qc.qubits, qc.clbits)

        # The first two operations represent Bell-state creation on _outer_ qubits (0, 1), the
        # second two represent the same creation, but on outer qubits (1, 0).
        expected = [
            id_op.compose(bell, qargs=(0, 1)),
            id_op.compose(bell, qargs=(0, 1)),
            id_op.compose(bell, qargs=(1, 0)),
            id_op.compose(bell, qargs=(1, 0)),
        ]

        actual = []
        pm = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)])
        for instruction in pm.run(qc).data:
            # For each instruction, the `UnitaryGate` that's been created will always have been made
            # (as an implementation detail of `DAGCircuit.collect_2q_runs` as of commit e5950661) to
            # apply to _inner_ qubits (0, 1).  We need to map that back to the _outer_ qubits that
            # it applies to compare.
            body = instruction.operation.blocks[0]
            wire_map = {
                inner: qc.find_bit(outer).index
                for inner, outer in zip(body.qubits, instruction.qubits)
            }
            actual.append(
                id_op.compose(
                    Operator(body.data[0].operation),
                    qargs=[wire_map[q] for q in body.data[0].qubits],
                )
            )
        self.assertEqual(expected, actual)

    def test_custom_no_target(self):
        """Test pass doesn't fail with custom gate."""

        class MyCustomGate(Gate):
            """Custom gate."""

            def __init__(self):
                super().__init__(name="my_custom", num_qubits=2, params=[])

        qc = QuantumCircuit(2)
        qc.append(MyCustomGate(), [0, 1])

        pm = PassManager([Collect2qBlocks(), ConsolidateBlocks()])
        res = pm.run(qc)

        self.assertEqual(res, qc)


if __name__ == "__main__":
    unittest.main()

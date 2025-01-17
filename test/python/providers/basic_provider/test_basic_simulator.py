# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test basic simulator."""

import os
import unittest
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.qasm2 import dumps
from test import QiskitTestCase  # pylint: disable=wrong-import-order


from . import BasicProviderBackendTestMixin


class TestBasicSimulator(QiskitTestCase, BasicProviderBackendTestMixin):
    """Test the basic provider simulator."""

    def setUp(self):
        super().setUp()
        self.backend = BasicSimulator()
        bell = QuantumCircuit(2, 2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure([0, 1], [0, 1])
        self.circuit = bell

        self.seed = 88
        self.backend = BasicSimulator()
        qasm_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "qasm"
        )
        qasm_filename = os.path.join(qasm_dir, "example.qasm")
        qcirc = QuantumCircuit.from_qasm_file(qasm_filename)
        qcirc.name = "test"
        self.transpiled_circuit = transpile(qcirc, backend=self.backend)

    def test_basic_simulator_single_shot(self):
        """Test single shot run."""
        shots = 1
        result = self.backend.run(
            self.transpiled_circuit, shots=shots, seed_simulator=self.seed
        ).result()
        self.assertEqual(result.success, True)

    def test_measure_sampler_repeated_qubits(self):
        """Test measure sampler if qubits measured more than once."""
        shots = 100
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(4, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
        circuit.measure(qr[1], cr[2])
        circuit.measure(qr[0], cr[3])
        target = {"0110": shots}
        job = self.backend.run(
            transpile(circuit, self.backend), shots=shots, seed_simulator=self.seed
        )
        result = job.result()
        counts = result.get_counts(0)
        self.assertEqual(counts, target)

    def test_measure_sampler_single_qubit(self):
        """Test measure sampler if single-qubit is measured."""
        shots = 100
        num_qubits = 5
        qr = QuantumRegister(num_qubits, "qr")
        cr = ClassicalRegister(1, "cr")

        for qubit in range(num_qubits):
            circuit = QuantumCircuit(qr, cr)
            circuit.x(qr[qubit])
            circuit.measure(qr[qubit], cr[0])
            target = {"1": shots}
            job = self.backend.run(
                transpile(circuit, self.backend), shots=shots, seed_simulator=self.seed
            )
            result = job.result()
            counts = result.get_counts(0)
            self.assertEqual(counts, target)

    def test_measure_sampler_partial_qubit(self):
        """Test measure sampler if single-qubit is measured."""
        shots = 100
        num_qubits = 5
        qr = QuantumRegister(num_qubits, "qr")
        cr = ClassicalRegister(4, "cr")

        #             ░     ░     ░ ┌─┐ ░
        # qr_0: ──────░─────░─────░─┤M├─░────
        #       ┌───┐ ░     ░ ┌─┐ ░ └╥┘ ░
        # qr_1: ┤ X ├─░─────░─┤M├─░──╫──░────
        #       └───┘ ░     ░ └╥┘ ░  ║  ░
        # qr_2: ──────░─────░──╫──░──╫──░────
        #       ┌───┐ ░ ┌─┐ ░  ║  ░  ║  ░ ┌─┐
        # qr_3: ┤ X ├─░─┤M├─░──╫──░──╫──░─┤M├
        #       └───┘ ░ └╥┘ ░  ║  ░  ║  ░ └╥┘
        # qr_4: ──────░──╫──░──╫──░──╫──░──╫─
        #             ░  ║  ░  ║  ░  ║  ░  ║
        # cr: 4/═════════╩═════╩═════╩═════╩═
        #                1     0     2     3
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[3])
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.measure(qr[3], cr[1])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[2])
        circuit.barrier(qr)
        circuit.measure(qr[3], cr[3])
        target = {"1011": shots}
        job = self.backend.run(
            transpile(circuit, self.backend), shots=shots, seed_simulator=self.seed
        )
        result = job.result()
        counts = result.get_counts(0)
        self.assertEqual(counts, target)

    def test_basic_simulator(self):
        """Test data counts output for single circuit run against reference."""
        result = self.backend.run(
            self.transpiled_circuit, shots=1000, seed_simulator=self.seed
        ).result()
        shots = 1024
        threshold = 0.04 * shots
        counts = result.get_counts("test")
        target = {
            "100 100": shots / 8,
            "011 011": shots / 8,
            "101 101": shots / 8,
            "111 111": shots / 8,
            "000 000": shots / 8,
            "010 010": shots / 8,
            "110 110": shots / 8,
            "001 001": shots / 8,
        }
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_if_statement(self):
        """Test if statements."""
        shots = 100
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")

        #       ┌───┐┌─┐          ┌─┐
        # qr_0: ┤ X ├┤M├──────────┤M├──────
        #       ├───┤└╥┘┌─┐       └╥┘┌─┐
        # qr_1: ┤ X ├─╫─┤M├────────╫─┤M├───
        #       └───┘ ║ └╥┘ ┌───┐  ║ └╥┘┌─┐
        # qr_2: ──────╫──╫──┤ X ├──╫──╫─┤M├
        #             ║  ║  └─╥─┘  ║  ║ └╥┘
        #             ║  ║ ┌──╨──┐ ║  ║  ║
        # cr: 3/══════╩══╩═╡ 0x3 ╞═╩══╩══╩═
        #             0  1 └─────┘ 0  1  2
        circuit_if_true = QuantumCircuit(qr, cr)
        circuit_if_true.x(qr[0])
        circuit_if_true.x(qr[1])
        circuit_if_true.measure(qr[0], cr[0])
        circuit_if_true.measure(qr[1], cr[1])
        with self.assertWarns(DeprecationWarning):
            circuit_if_true.x(qr[2]).c_if(cr, 0x3)
        circuit_if_true.measure(qr[0], cr[0])
        circuit_if_true.measure(qr[1], cr[1])
        circuit_if_true.measure(qr[2], cr[2])

        #       ┌───┐┌─┐       ┌─┐
        # qr_0: ┤ X ├┤M├───────┤M├──────
        #       └┬─┬┘└╥┘       └╥┘┌─┐
        # qr_1: ─┤M├──╫─────────╫─┤M├───
        #        └╥┘  ║  ┌───┐  ║ └╥┘┌─┐
        # qr_2: ──╫───╫──┤ X ├──╫──╫─┤M├
        #         ║   ║  └─╥─┘  ║  ║ └╥┘
        #         ║   ║ ┌──╨──┐ ║  ║  ║
        # cr: 3/══╩═══╩═╡ 0x3 ╞═╩══╩══╩═
        #         1   0 └─────┘ 0  1  2
        circuit_if_false = QuantumCircuit(qr, cr)
        circuit_if_false.x(qr[0])
        circuit_if_false.measure(qr[0], cr[0])
        circuit_if_false.measure(qr[1], cr[1])
        with self.assertWarns(DeprecationWarning):
            circuit_if_false.x(qr[2]).c_if(cr, 0x3)
        circuit_if_false.measure(qr[0], cr[0])
        circuit_if_false.measure(qr[1], cr[1])
        circuit_if_false.measure(qr[2], cr[2])
        job = self.backend.run(
            transpile([circuit_if_true, circuit_if_false], self.backend),
            shots=shots,
            seed_simulator=self.seed,
        )
        result = job.result()
        counts_if_true = result.get_counts(circuit_if_true)
        counts_if_false = result.get_counts(circuit_if_false)
        self.assertEqual(counts_if_true, {"111": 100})
        self.assertEqual(counts_if_false, {"001": 100})

    def test_bit_cif_crossaffect(self):
        """Test if bits in a classical register other than
        the single conditional bit affect the conditioned operation."""
        #               ┌───┐          ┌─┐
        # q0_0: ────────┤ H ├──────────┤M├
        #       ┌───┐   └─╥─┘    ┌─┐   └╥┘
        # q0_1: ┤ X ├─────╫──────┤M├────╫─
        #       ├───┤     ║      └╥┘┌─┐ ║
        # q0_2: ┤ X ├─────╫───────╫─┤M├─╫─
        #       └───┘┌────╨─────┐ ║ └╥┘ ║
        # c0: 3/═════╡ c0_0=0x1 ╞═╩══╩══╬═
        #            └──────────┘ 1  2  ║
        # c1: 1/════════════════════════╩═
        #                               0
        shots = 100
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        cr1 = ClassicalRegister(1)
        circuit = QuantumCircuit(qr, cr, cr1)
        circuit.x([qr[1], qr[2]])
        circuit.measure(qr[1], cr[1])
        circuit.measure(qr[2], cr[2])
        with self.assertWarns(DeprecationWarning):
            circuit.h(qr[0]).c_if(cr[0], True)
        circuit.measure(qr[0], cr1[0])
        job = self.backend.run(circuit, shots=shots, seed_simulator=self.seed)
        result = job.result().get_counts()
        target = {"0 110": 100}
        self.assertEqual(result, target)

    def test_teleport(self):
        """Test teleportation as in tutorials"""
        #       ┌─────────┐          ┌───┐ ░ ┌─┐
        # qr_0: ┤ Ry(π/4) ├───────■──┤ H ├─░─┤M├────────────────────
        #       └──┬───┬──┘     ┌─┴─┐└───┘ ░ └╥┘┌─┐
        # qr_1: ───┤ H ├─────■──┤ X ├──────░──╫─┤M├─────────────────
        #          └───┘   ┌─┴─┐└───┘      ░  ║ └╥┘ ┌───┐  ┌───┐ ┌─┐
        # qr_2: ───────────┤ X ├───────────░──╫──╫──┤ Z ├──┤ X ├─┤M├
        #                  └───┘           ░  ║  ║  └─╥─┘  └─╥─┘ └╥┘
        #                                     ║  ║ ┌──╨──┐   ║    ║
        # cr0: 1/═════════════════════════════╩══╬═╡ 0x1 ╞═══╬════╬═
        #                                     0  ║ └─────┘┌──╨──┐ ║
        # cr1: 1/════════════════════════════════╩════════╡ 0x1 ╞═╬═
        #                                        0        └─────┘ ║
        # cr2: 1/═════════════════════════════════════════════════╩═
        #                                                         0
        self.log.info("test_teleport")
        pi = np.pi
        shots = 4000
        qr = QuantumRegister(3, "qr")
        cr0 = ClassicalRegister(1, "cr0")
        cr1 = ClassicalRegister(1, "cr1")
        cr2 = ClassicalRegister(1, "cr2")
        circuit = QuantumCircuit(qr, cr0, cr1, cr2, name="teleport")
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.ry(pi / 4, qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr0[0])
        circuit.measure(qr[1], cr1[0])
        with self.assertWarns(DeprecationWarning):
            circuit.z(qr[2]).c_if(cr0, 1)
        with self.assertWarns(DeprecationWarning):
            circuit.x(qr[2]).c_if(cr1, 1)
        circuit.measure(qr[2], cr2[0])
        job = self.backend.run(
            transpile(circuit, self.backend), shots=shots, seed_simulator=self.seed
        )
        results = job.result()
        data = results.get_counts("teleport")
        alice = {
            "00": data["0 0 0"] + data["1 0 0"],
            "01": data["0 1 0"] + data["1 1 0"],
            "10": data["0 0 1"] + data["1 0 1"],
            "11": data["0 1 1"] + data["1 1 1"],
        }
        bob = {
            "0": data["0 0 0"] + data["0 1 0"] + data["0 0 1"] + data["0 1 1"],
            "1": data["1 0 0"] + data["1 1 0"] + data["1 0 1"] + data["1 1 1"],
        }
        self.log.info("test_teleport: circuit:")
        self.log.info(dumps(circuit))
        self.log.info("test_teleport: data %s", data)
        self.log.info("test_teleport: alice %s", alice)
        self.log.info("test_teleport: bob %s", bob)
        alice_ratio = 1 / np.tan(pi / 8) ** 2
        bob_ratio = bob["0"] / float(bob["1"])
        error = abs(alice_ratio - bob_ratio) / alice_ratio
        self.log.info("test_teleport: relative error = %s", error)
        self.assertLess(error, 0.05)

    def test_memory(self):
        """Test memory."""
        #       ┌───┐        ┌─┐
        # qr_0: ┤ H ├──■─────┤M├───
        #       └───┘┌─┴─┐   └╥┘┌─┐
        # qr_1: ─────┤ X ├────╫─┤M├
        #            └┬─┬┘    ║ └╥┘
        # qr_2: ──────┤M├─────╫──╫─
        #       ┌───┐ └╥┘ ┌─┐ ║  ║
        # qr_3: ┤ X ├──╫──┤M├─╫──╫─
        #       └───┘  ║  └╥┘ ║  ║
        # cr0: 2/══════╬═══╬══╩══╩═
        #              ║   ║  0  1
        #              ║   ║
        # cr1: 2/══════╩═══╩═══════
        #              0   1
        qr = QuantumRegister(4, "qr")
        cr0 = ClassicalRegister(2, "cr0")
        cr1 = ClassicalRegister(2, "cr1")
        circ = QuantumCircuit(qr, cr0, cr1)
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.x(qr[3])
        circ.measure(qr[0], cr0[0])
        circ.measure(qr[1], cr0[1])
        circ.measure(qr[2], cr1[0])
        circ.measure(qr[3], cr1[1])

        shots = 50
        job = self.backend.run(
            transpile(circ, self.backend), shots=shots, seed_simulator=self.seed, memory=True
        )
        result = job.result()
        memory = result.get_memory()
        self.assertEqual(len(memory), shots)
        for mem in memory:
            self.assertIn(mem, ["10 00", "10 11"])

    def test_unitary(self):
        """Test unitary gate instruction"""
        max_qubits = 4
        x_mat = np.array([[0, 1], [1, 0]])
        # Test 1 to max_qubits for random n-qubit unitary gate
        for i in range(max_qubits):
            num_qubits = i + 1
            # Apply X gate to all qubits
            multi_x = x_mat
            for _ in range(i):
                multi_x = np.kron(multi_x, x_mat)
            # Target counts
            shots = 1024
            target_counts = {num_qubits * "1": shots}
            # Test circuit
            qr = QuantumRegister(num_qubits, "qr")
            cr = ClassicalRegister(num_qubits, "cr")
            circuit = QuantumCircuit(qr, cr)
            circuit.unitary(multi_x, qr)
            circuit.measure(qr, cr)
            job = self.backend.run(transpile(circuit, self.backend), shots=shots)
            result = job.result()
            counts = result.get_counts(0)
            self.assertEqual(counts, target_counts)


if __name__ == "__main__":
    unittest.main()

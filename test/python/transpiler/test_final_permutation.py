# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Testing FinalPermutation in transpile flows."""

import unittest

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PermutationGate
from qiskit.circuit.random import random_circuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import ElidePermutations, StarPreRouting
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestFinalPermutationInTranspile(QiskitTestCase):
    """Tests for FinalPermutation in transpile flows."""

    # pylint: disable=unused-argument
    def _check_on_callback(self, pass_, dag, time, property_set, count):
        self.assertEqual(dag.num_qubits(), dag._final_permutation.num_qubits())

    def test_sabre_1(self):
        """Transpile flow includes SabreLayout pass."""
        qc = QuantumCircuit(6)
        qc.cx(0, 2)
        qc.cx(0, 4)
        qc.cx(2, 4)
        qc.cx(1, 3)
        qc.cx(1, 5)
        qc.cx(3, 5)
        op = Operator(qc)
        coupling_map = CouplingMap.from_line(6)
        qct = transpile(
            qc,
            optimization_level=3,
            coupling_map=coupling_map,
            basis_gates=["cx", "u"],
            callback=self._check_on_callback,
        )
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_sabre_2(self):
        """Transpile flow includes SabreLayout pass."""
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        op = Operator(qc)
        coupling_map = CouplingMap.from_line(6)
        qct = transpile(
            qc,
            optimization_level=3,
            coupling_map=coupling_map,
            basis_gates=["cx", "u"],
            callback=self._check_on_callback,
        )
        self.assertTrue(Operator.from_circuit(qct).equiv(op))
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_sabre_3(self):
        """Transpile flow includes SabreLayout pass."""
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        op = Operator(qc)
        coupling_map = CouplingMap.from_line(6)
        qct = transpile(
            qc,
            optimization_level=3,
            coupling_map=coupling_map,
            basis_gates=["cx", "u"],
            callback=self._check_on_callback,
        )
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_elide_sabre_1(self):
        """Transpile flow includes ElidePermutations and SabreLayout passes."""
        qc = QuantumCircuit(6)
        qc.cx(0, 2)
        qc.cx(0, 4)
        qc.cx(2, 4)
        qc.cx(1, 3)
        qc.cx(1, 5)
        qc.cx(3, 5)
        qc.append(PermutationGate([1, 2, 0]), [0, 1, 2])
        op = Operator(qc)
        coupling_map = CouplingMap.from_line(6)
        qct = transpile(
            qc,
            optimization_level=3,
            coupling_map=coupling_map,
            basis_gates=["cx", "u"],
            callback=self._check_on_callback,
        )
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_elide_sabre_2(self):
        """Transpile flow includes ElidePermutations and SabreLayout passes."""
        qc = QuantumCircuit(6)
        qc.cx(0, 2)
        qc.cx(0, 4)
        qc.cx(2, 4)
        qc.cx(1, 3)
        qc.swap(0, 1)
        qc.cx(1, 5)
        qc.cx(3, 5)
        op = Operator(qc)
        coupling_map = CouplingMap.from_line(6)
        qct = transpile(
            qc,
            optimization_level=3,
            coupling_map=coupling_map,
            basis_gates=["cx", "u"],
            callback=self._check_on_callback,
        )
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_star_vf2(self):
        """Transpile flow includes StarPreRouting."""
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        op = Operator(qc)
        coupling_map = CouplingMap.from_line(6)
        spm = generate_preset_pass_manager(
            optimization_level=3,
            seed_transpiler=1234,
            coupling_map=coupling_map,
            basis_gates=["u", "cz"],
        )
        spm.init += StarPreRouting()
        qct = spm.run(qc, callback=self._check_on_callback)
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_elide_star_sabre(self):
        """Transpile flow includes ElidePermutations followed by StarPreRouting."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.append(PermutationGate([0, 2, 1]), [0, 1, 2])
        qc.h(1)
        op = Operator(qc)
        coupling_map = CouplingMap.from_line(5)
        spm = generate_preset_pass_manager(
            optimization_level=3,
            seed_transpiler=1234,
            coupling_map=coupling_map,
            basis_gates=["u", "cz"],
        )
        spm.init += ElidePermutations()
        spm.init += StarPreRouting()
        qct = spm.run(qc, callback=self._check_on_callback)
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_star_elide_sabre_1(self):
        """Transpile flow includes StarPreRouting followed by ElidePermutations."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.append(PermutationGate([0, 2, 1]), [0, 1, 2])
        qc.h(1)
        op = Operator(qc)
        coupling_map = CouplingMap.from_line(5)
        spm = generate_preset_pass_manager(
            optimization_level=3,
            seed_transpiler=1234,
            coupling_map=coupling_map,
            basis_gates=["u", "cz"],
        )
        spm.init += StarPreRouting()
        spm.init += ElidePermutations()
        qct = spm.run(qc, callback=self._check_on_callback)
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_star_elide_sabre_2(self):
        """Transpile flow includes StarPreRouting followed by ElidePermutations."""
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        op = Operator(qc)
        spm = generate_preset_pass_manager(
            optimization_level=3,
            seed_transpiler=1234,
            coupling_map=CouplingMap.from_line(6),
            basis_gates=["u", "cz"],
        )
        spm.init += StarPreRouting()
        spm.init += ElidePermutations()
        qct = spm.run(qc, callback=self._check_on_callback)
        self.assertTrue(Operator._from_circuit_new(qct).equiv(op))

    def test_random_embed(self):
        """Physical circuit has more qubits than the virtual circuit."""
        for seed in range(10):
            num_qubits = 4
            qc = random_circuit(num_qubits, 10, 3, seed=seed)
            coupling_map = CouplingMap.from_line(6)
            extended_qc = QuantumCircuit(6)
            extended_qc = extended_qc.compose(qc, range(4))
            extended_op = Operator(extended_qc)
            qct = transpile(
                qc,
                optimization_level=3,
                coupling_map=coupling_map,
                basis_gates=["cx", "u"],
                seed_transpiler=3,
                callback=self._check_on_callback,
            )
            transpiled_op_new = Operator._from_circuit_new(qct)
            self.assertTrue(transpiled_op_new.equiv(extended_op))

    def test_routing_methods(self):
        """Testing various routing methods."""
        qc = random_circuit(4, 5, 3, seed=1)
        coupling_map = CouplingMap.from_line(6)
        extended_qc = QuantumCircuit(6)
        extended_qc = extended_qc.compose(qc, range(4))
        extended_op = Operator(extended_qc)

        for routing_method in ["stochastic", "lookahead", "basic", "sabre"]:
            qct = transpile(
                qc,
                optimization_level=3,
                coupling_map=coupling_map,
                basis_gates=["cx", "u"],
                seed_transpiler=3,
                routing_method=routing_method,
                callback=self._check_on_callback,
            )
            transpiled_op_new = Operator._from_circuit_new(qct)
            self.assertTrue(transpiled_op_new.equiv(extended_op))

    def test_elide_and_routing_methods(self):
        """Testing ElidePermutations followed by various routing methods."""
        qc = QuantumCircuit(6)
        qc.cx(0, 2)
        qc.cx(0, 4)
        qc.cx(2, 4)
        qc.cx(1, 3)
        qc.cx(1, 5)
        qc.cx(3, 5)
        qc.append(PermutationGate([1, 2, 0]), [0, 1, 2])
        coupling_map = CouplingMap.from_line(8)
        extended_qc = QuantumCircuit(8)
        extended_qc = extended_qc.compose(qc, range(6))
        extended_op = Operator(extended_qc)

        for routing_method in ["stochastic", "lookahead", "basic", "sabre"]:
            qct = transpile(
                qc,
                optimization_level=3,
                coupling_map=coupling_map,
                basis_gates=["cx", "u"],
                seed_transpiler=3,
                routing_method=routing_method,
                callback=self._check_on_callback,
            )
            transpiled_op_new = Operator._from_circuit_new(qct)
            self.assertTrue(transpiled_op_new.equiv(extended_op))

    def test_with_post_layout(self):
        """Test involving VF2PostLayout pass."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        extended_qc = QuantumCircuit(5)
        extended_qc = extended_qc.compose(qc, range(3))
        extended_op = Operator(extended_qc)
        coupling_map = CouplingMap.from_line(5)
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["id", "sx", "x", "cx", "rz"],
            coupling_map=coupling_map,
            seed=0,
        )
        qct = transpile(qc, backend=backend, seed_transpiler=4242, callback=self._check_on_callback)
        transpiled_op_new = Operator._from_circuit_new(qct)
        self.assertTrue(transpiled_op_new.equiv(extended_op))


if __name__ == "__main__":
    unittest.main()

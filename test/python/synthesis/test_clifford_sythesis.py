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

# pylint: disable=invalid-name
"""Tests for Clifford synthesis functions."""

import numpy as np
from ddt import ddt
from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import (
    CXGate,
    CYGate,
    CZGate,
    DCXGate,
    ECRGate,
    HGate,
    IGate,
    SdgGate,
    SGate,
    SXGate,
    SXdgGate,
    SwapGate,
    XGate,
    YGate,
    ZGate,
    iSwapGate,
)
from qiskit.converters.dag_to_circuit import dag_to_circuit
from qiskit.quantum_info.operators import Clifford
from qiskit.synthesis.clifford import (
    synth_clifford_full,
    synth_clifford_ag,
    synth_clifford_bm,
    synth_clifford_greedy,
)

from test import QiskitTestCase  # pylint: disable=wrong-import-order
from test import combine  # pylint: disable=wrong-import-order


class VGate(Gate):
    """V Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new V Gate."""
        super().__init__("v", 1, [])

    def _define(self):
        """V Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.sdg(0)
        qc.h(0)
        self.definition = qc


class WGate(Gate):
    """W Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new W Gate."""
        super().__init__("w", 1, [])

    def _define(self):
        """W Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.append(VGate(), [q[0]], [])
        qc.append(VGate(), [q[0]], [])
        self.definition = qc


def random_clifford_circuit(num_qubits, num_gates, gates="all", seed=None):
    """Generate a pseudo random Clifford circuit."""

    qubits_1_gates = ["i", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "v", "w"]
    qubits_2_gates = ["cx", "cz", "cy", "swap", "iswap", "ecr", "dcx"]
    if gates == "all":
        if num_qubits == 1:
            gates = qubits_1_gates
        else:
            gates = qubits_1_gates + qubits_2_gates

    instructions = {
        "i": (IGate(), 1),
        "x": (XGate(), 1),
        "y": (YGate(), 1),
        "z": (ZGate(), 1),
        "h": (HGate(), 1),
        "s": (SGate(), 1),
        "sdg": (SdgGate(), 1),
        "sx": (SXGate(), 1),
        "sxdg": (SXdgGate(), 1),
        "v": (VGate(), 1),
        "w": (WGate(), 1),
        "cx": (CXGate(), 2),
        "cy": (CYGate(), 2),
        "cz": (CZGate(), 2),
        "swap": (SwapGate(), 2),
        "iswap": (iSwapGate(), 2),
        "ecr": (ECRGate(), 2),
        "dcx": (DCXGate(), 2),
    }

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    samples = rng.choice(gates, num_gates)

    circ = QuantumCircuit(num_qubits)

    for name in samples:
        gate, nqargs = instructions[name]
        qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
        circ.append(gate, qargs)

    return circ


@ddt
class TestCliffordSynthesis(QiskitTestCase):
    """Tests for clifford synthesis functions."""

    @staticmethod
    def _cliffords_1q():
        clifford_dicts = [
            {"stabilizer": ["+Z"], "destabilizer": ["-X"]},
            {"stabilizer": ["-Z"], "destabilizer": ["+X"]},
            {"stabilizer": ["-Z"], "destabilizer": ["-X"]},
            {"stabilizer": ["+Z"], "destabilizer": ["+Y"]},
            {"stabilizer": ["+Z"], "destabilizer": ["-Y"]},
            {"stabilizer": ["-Z"], "destabilizer": ["+Y"]},
            {"stabilizer": ["-Z"], "destabilizer": ["-Y"]},
            {"stabilizer": ["+X"], "destabilizer": ["+Z"]},
            {"stabilizer": ["+X"], "destabilizer": ["-Z"]},
            {"stabilizer": ["-X"], "destabilizer": ["+Z"]},
            {"stabilizer": ["-X"], "destabilizer": ["-Z"]},
            {"stabilizer": ["+X"], "destabilizer": ["+Y"]},
            {"stabilizer": ["+X"], "destabilizer": ["-Y"]},
            {"stabilizer": ["-X"], "destabilizer": ["+Y"]},
            {"stabilizer": ["-X"], "destabilizer": ["-Y"]},
            {"stabilizer": ["+Y"], "destabilizer": ["+X"]},
            {"stabilizer": ["+Y"], "destabilizer": ["-X"]},
            {"stabilizer": ["-Y"], "destabilizer": ["+X"]},
            {"stabilizer": ["-Y"], "destabilizer": ["-X"]},
            {"stabilizer": ["+Y"], "destabilizer": ["+Z"]},
            {"stabilizer": ["+Y"], "destabilizer": ["-Z"]},
            {"stabilizer": ["-Y"], "destabilizer": ["+Z"]},
            {"stabilizer": ["-Y"], "destabilizer": ["-Z"]},
        ]
        return [Clifford.from_dict(i) for i in clifford_dicts]

    def test_decompose_1q(self):
        """Test synthesis for all 1-qubit Cliffords"""
        for cliff in self._cliffords_1q():
            with self.subTest(msg=f"Test circuit {cliff}"):
                target = cliff
                value = Clifford(cliff.to_circuit())
                self.assertEqual(target, value)

    @combine(num_qubits=[2, 3])
    def test_synth_bm(self, num_qubits):
        """Test B&M synthesis for set of {num_qubits}-qubit Cliffords"""
        rng = np.random.default_rng(1234)
        samples = 50
        for use_dag in [True, False]:
            with self.subTest(use_dag=use_dag):
                for _ in range(samples):
                    circ = random_clifford_circuit(num_qubits, 5 * num_qubits, seed=rng)
                    target = Clifford(circ)
                    if use_dag:
                        synth_circ = dag_to_circuit(synth_clifford_bm(target, use_dag=True))
                    else:
                        synth_circ = synth_clifford_bm(target)
                    value = Clifford(synth_circ)
                    self.assertEqual(value, target)

        with self.subTest("check consistency of use_dag"):
            for _ in range(samples):
                circ = random_clifford_circuit(num_qubits, 5 * num_qubits, seed=rng)
                target = Clifford(circ)
                synth_circ_dag = dag_to_circuit(synth_clifford_bm(target, use_dag=True))
                synth_circ = synth_clifford_bm(target)
                self.assertEqual(
                    sorted(synth_circ_dag.count_ops().items(), key=lambda x: x[0]),
                    sorted(synth_circ.count_ops().items(), key=lambda x: x[0]),
                )

    @combine(num_qubits=[2, 3, 4, 5])
    def test_synth_ag(self, num_qubits):
        """Test A&G synthesis for set of {num_qubits}-qubit Cliffords"""
        rng = np.random.default_rng(1234)
        samples = 1
        for use_dag in [True, False]:
            with self.subTest(use_dag=use_dag):
                for _ in range(samples):
                    circ = random_clifford_circuit(num_qubits, 5 * num_qubits, seed=rng)
                    target = Clifford(circ)
                    if use_dag:
                        synth_circ = dag_to_circuit(synth_clifford_ag(target, use_dag=True))
                    else:
                        synth_circ = synth_clifford_ag(target)
                    value = Clifford(synth_circ)
                    self.assertEqual(value, target)

        with self.subTest("check consistency of use_dag"):
            for _ in range(samples):
                circ = random_clifford_circuit(num_qubits, 5 * num_qubits, seed=rng)
                target = Clifford(circ)
                synth_circ_dag = dag_to_circuit(synth_clifford_ag(target, use_dag=True))
                synth_circ = synth_clifford_ag(target)
                self.assertEqual(
                    sorted(synth_circ_dag.count_ops().items(), key=lambda x: x[0]),
                    sorted(synth_circ.count_ops().items(), key=lambda x: x[0]),
                )

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_synth_greedy(self, num_qubits):
        """Test greedy synthesis for set of {num_qubits}-qubit Cliffords"""
        rng = np.random.default_rng(1234)
        samples = 50
        for use_dag in [True, False]:
            with self.subTest(use_dag=use_dag):
                for _ in range(samples):
                    circ = random_clifford_circuit(num_qubits, 5 * num_qubits, seed=rng)
                    target = Clifford(circ)
                    if use_dag:
                        synth_circ = dag_to_circuit(synth_clifford_greedy(target, use_dag=True))
                    else:
                        synth_circ = synth_clifford_greedy(target)
                    value = Clifford(synth_circ)
                    self.assertEqual(value, target)

        with self.subTest("check consistency of use_dag"):
            for _ in range(samples):
                circ = random_clifford_circuit(num_qubits, 5 * num_qubits, seed=rng)
                target = Clifford(circ)
                synth_circ_dag = dag_to_circuit(synth_clifford_greedy(target, use_dag=True))
                synth_circ = synth_clifford_greedy(target)
                self.assertEqual(
                    sorted(synth_circ_dag.count_ops().items(), key=lambda x: x[0]),
                    sorted(synth_circ.count_ops().items(), key=lambda x: x[0]),
                )

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_synth_full(self, num_qubits):
        """Test synthesis for set of {num_qubits}-qubit Cliffords"""
        rng = np.random.default_rng(1234)
        samples = 50
        for use_dag in [True, False]:
            with self.subTest(use_dag=use_dag):
                for _ in range(samples):
                    circ = random_clifford_circuit(num_qubits, 5 * num_qubits, seed=rng)
                    target = Clifford(circ)
                    if use_dag:
                        synth_circ = dag_to_circuit(synth_clifford_full(target, use_dag=True))
                    else:
                        synth_circ = synth_clifford_full(target)
                    value = Clifford(synth_circ)
                    self.assertEqual(value, target)

        with self.subTest("check consistency of use_dag"):
            for _ in range(samples):
                circ = random_clifford_circuit(num_qubits, 5 * num_qubits, seed=rng)
                target = Clifford(circ)
                synth_circ_dag = dag_to_circuit(synth_clifford_full(target, use_dag=True))
                synth_circ = synth_clifford_full(target)
                self.assertEqual(
                    sorted(synth_circ_dag.count_ops().items(), key=lambda x: x[0]),
                    sorted(synth_circ.count_ops().items(), key=lambda x: x[0]),
                )

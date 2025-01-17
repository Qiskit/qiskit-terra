# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring,invalid-name,no-member
# pylint: disable=attribute-defined-outside-init

import os
import itertools

from qiskit.quantum_info import random_clifford
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import efficient_su2, quantum_volume
from .utils import dtc_unitary, multi_control_circuit

SEED = 12345


def build_circuit(width, gates):
    qr = QuantumRegister(width)
    qc = QuantumCircuit(qr)

    while len(qc) < gates:
        for k in range(width):
            qc.h(qr[k])
        for k in range(width - 1):
            qc.cx(qr[k], qr[k + 1])

    return qc


class CircuitConstructionBench:
    params = ([1, 2, 5, 8, 14, 20], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ["width", "gates"]
    timeout = 600

    def setup(self, width, gates):
        self.empty_circuit = build_circuit(width, 0)
        self.sample_circuit = build_circuit(width, gates)

    def time_circuit_construction(self, width, gates):
        build_circuit(width, gates)

    def time_circuit_extend(self, _, __):
        self.empty_circuit.compose(self.sample_circuit, inplace=True)

    def time_circuit_copy(self, _, __):
        self.sample_circuit.copy()


def build_parameterized_circuit(width, gates, param_count):
    params = [Parameter(f"param-{x}") for x in range(param_count)]
    param_iter = itertools.cycle(params)

    qr = QuantumRegister(width)
    qc = QuantumCircuit(qr)

    while len(qc) < gates:
        for k in range(width):
            param = next(param_iter)
            qc.r(0, param, qr[k])
        for k in range(width - 1):
            param = next(param_iter)
            qc.crx(param, qr[k], qr[k + 1])

    return qc, params


class ParameterizedCircuitConstructionBench:
    params = ([20], [8, 128, 2048, 8192, 32768, 131072], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ["width", "gates", "number of params"]
    timeout = 600

    def setup(self, _, gates, params):
        if params > gates:
            raise NotImplementedError

    def time_build_parameterized_circuit(self, width, gates, params):
        build_parameterized_circuit(width, gates, params)


class ParameterizedCircuitBindBench:
    params = ([20], [8, 128, 2048, 8192, 32768, 131072], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ["width", "gates", "number of params"]
    timeout = 600

    def setup(self, width, gates, params):
        if params > gates:
            raise NotImplementedError
        self.circuit, self.params = build_parameterized_circuit(width, gates, params)

    def time_bind_params(self, _, __, ___):
        # TODO: write more complete benchmarks of assign_parameters
        #  that test more of the input formats / combinations
        self.circuit.assign_parameters({x: 3.14 for x in self.params})


class ParamaterizedDifferentCircuit:
    param_names = ["circuit_size", "num_qubits"]
    params = ([10, 50, 100], [10, 50, 150])

    def time_QV100_build(self, circuit_size, num_qubits):
        """Measures an SDKs ability to build a 100Q
        QV circit from scratch.
        """
        return quantum_volume(circuit_size, num_qubits, seed=SEED)

    def time_DTC100_set_build(self, circuit_size, num_qubits):
        """Measures an SDKs ability to build a set
        of 100Q DTC circuits out to 100 layers of
        the underlying unitary
        """
        max_cycles = circuit_size
        initial_state = QuantumCircuit(num_qubits)
        dtc_circuit = dtc_unitary(num_qubits, g=0.95, seed=SEED)

        circs = [initial_state]
        for tt in range(max_cycles):
            qc = circs[tt].compose(dtc_circuit)
            circs.append(qc)
            result = circs[-1]

        return result


class MultiControl:
    param_names = ["width"]
    params = [10, 16, 20]

    def time_multi_control_circuit(self, width):
        """Measures an SDKs ability to build a circuit
        with a multi-controlled X-gate
        """
        out = multi_control_circuit(width)
        return out


class ParameterizedCirc:
    param_names = ["num_qubits"]
    params = [5, 10, 16]

    def time_param_circSU2_100_build(self, num_qubits):
        """Measures an SDKs ability to build a
        parameterized efficient SU2 circuit with circular entanglement
        over 100Q utilizing 4 repetitions.  This will yield a
        circuit with 1000 parameters
        """
        out = efficient_su2(num_qubits, reps=4, entanglement="circular")
        out._build()
        return out


class QasmImport:
    def time_QV100_qasm2_import(self):
        """QASM import of QV100 circuit"""
        self.qasm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qasm"))

        out = QuantumCircuit.from_qasm_file(os.path.join(self.qasm_path, "qv_N100_12345.qasm"))
        ops = out.count_ops()
        assert ops.get("rz", 0) == 120000
        assert ops.get("rx", 0) == 80000
        assert ops.get("cx", 0) == 15000
        return ops


class CliffordSynthesis:
    param_names = ["num_qubits"]
    params = [10, 50, 100]

    def time_clifford_synthesis(self, num_qubits):
        cliff = random_clifford(num_qubits)
        qc = cliff.to_circuit()
        return qc

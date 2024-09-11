# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=missing-module-docstring

import datetime
import itertools
import operator
import unittest
import warnings

from test import combine
from ddt import ddt, data

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import assemble
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import (
    Fake5QV1,
    Fake20QV1,
    Fake7QPulseV1,
    Fake27QPulseV1,
    Fake127QPulseV1,
    FakeOpenPulse2Q,
    GenericBackendV2,
)
from qiskit.providers.backend_compat import BackendV2Converter, convert_to_target
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.providers.models.backendconfiguration import GateConfig
from qiskit.providers.backend import BackendV2
from qiskit.utils import optionals
from qiskit.circuit.library import (
    SXGate,
    MCPhaseGate,
    MCXGate,
    RZGate,
    RXGate,
    U2Gate,
    U1Gate,
    U3Gate,
    YGate,
    ZGate,
    PauliGate,
    SwapGate,
    RGate,
    MCXGrayCode,
    RYGate,
    CZGate,
    ECRGate,
    UnitaryGate,
    UCGate,
    Initialize,
    DiagonalGate,
)
from qiskit.circuit import ControlledGate, Parameter
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.quantum_info.operators.channel import SuperOp
from qiskit.circuit.controlflow import (
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    ContinueLoopOp,
    BreakLoopOp,
    SwitchCaseOp,
)
from qiskit.transpiler.coupling import CouplingMap
from test.utils.base import QiskitTestCase  # pylint: disable=wrong-import-order

with warnings.catch_warnings():
    BACKENDS = [Fake5QV1(), Fake20QV1(), Fake7QPulseV1(), Fake27QPulseV1(), Fake127QPulseV1()]

BACKENDS_V2 = []
for n in [5, 7, 16, 20, 27, 65, 127]:
    cmap = CouplingMap.from_ring(n)
    BACKENDS_V2.append(GenericBackendV2(num_qubits=n, coupling_map=cmap, seed=42))


@ddt
class TestFakeBackends(QiskitTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.circuit = QuantumCircuit(2)
        cls.circuit.h(0)
        cls.circuit.h(1)
        cls.circuit.h(0)
        cls.circuit.h(1)
        cls.circuit.x(0)
        cls.circuit.x(1)
        cls.circuit.measure_all()

    @combine(
        backend=BACKENDS_V2,
        optimization_level=[0, 1, 2, 3],
    )
    def test_circuit_on_fake_backend_v2(self, backend, optimization_level):
        if not optionals.HAS_AER and backend.num_qubits > 20:
            self.skipTest(f"Unable to run fake_backend {backend.name} without qiskit-aer")
        job = backend.run(
            transpile(
                self.circuit, backend, seed_transpiler=42, optimization_level=optimization_level
            ),
            seed_simulator=42,
        )
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    @combine(
        backend=BACKENDS,
        optimization_level=[0, 1, 2, 3],
        dsc="Test execution path on {backend} with optimization level {optimization_level}",
        name="{backend}_opt_level_{optimization_level}",
    )
    def test_circuit_on_fake_backend(self, backend, optimization_level):
        if not optionals.HAS_AER and backend.configuration().num_qubits > 20:
            self.skipTest(
                f"Unable to run fake_backend {backend.configuration().backend_name} without qiskit-aer"
            )
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            transpiled = transpile(
                self.circuit, backend, seed_transpiler=42, optimization_level=optimization_level
            )
        job = backend.run(transpiled, seed_simulator=42)
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    def test_qobj_failure(self):
        backend = BACKENDS[-1]
        with self.assertWarns(DeprecationWarning):
            tqc = transpile(self.circuit, backend)
            qobj = assemble(tqc, backend)
        with self.assertRaises(QiskitError):
            backend.run(qobj)

    @data(*BACKENDS)
    def test_to_dict_properties(self, backend):
        with warnings.catch_warnings():
            # The class QobjExperimentHeader is deprecated
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
            properties = backend.properties()
        if properties:
            self.assertIsInstance(backend.properties().to_dict(), dict)
        else:
            self.assertTrue(backend.configuration().simulator)

    @data(*BACKENDS_V2)
    def test_convert_to_target(self, backend):
        target = backend.target
        if target.dt is not None:
            self.assertLess(target.dt, 1e-6)

    @data(*BACKENDS_V2)
    def test_backend_v2_dtm(self, backend):
        if backend.dtm:
            self.assertLess(backend.dtm, 1e-6)

    @data(*BACKENDS)
    def test_to_dict_configuration(self, backend):
        configuration = backend.configuration()
        if configuration.open_pulse:
            self.assertLess(configuration.dt, 1e-6)
            self.assertLess(configuration.dtm, 1e-6)
            for i in configuration.qubit_lo_range:
                self.assertGreater(i[0], 1e6)
                self.assertGreater(i[1], 1e6)
                self.assertLess(i[0], i[1])

            for i in configuration.meas_lo_range:
                self.assertGreater(i[0], 1e6)
                self.assertGreater(i[0], 1e6)
                self.assertLess(i[0], i[1])

            for i in configuration.rep_times:
                self.assertGreater(i, 0)
                self.assertLess(i, 1)

        self.assertIsInstance(configuration.to_dict(), dict)

    @data(*BACKENDS)
    def test_defaults_to_dict(self, backend):
        if hasattr(backend, "defaults"):
            defaults = backend.defaults()
            self.assertIsInstance(defaults.to_dict(), dict)

            for i in defaults.qubit_freq_est:
                self.assertGreater(i, 1e6)
                self.assertGreater(i, 1e6)

            for i in defaults.meas_freq_est:
                self.assertGreater(i, 1e6)
                self.assertGreater(i, 1e6)
        else:
            self.skipTest(f"Backend {backend} does not have defaults")

    def test_delay_circuit(self):
        with self.assertWarns(DeprecationWarning):
            backend = Fake27QPulseV1()
        backend.configuration().timing_constraints = {
            "acquire_alignment": 1,
            "granularity": 1,
            "min_length": 1,
            "pulse_alignment": 1,
        }
        qc = QuantumCircuit(2)
        qc.delay(502, 0, unit="ns")
        qc.x(1)
        qc.delay(250, 1, unit="ns")
        qc.measure_all()
        with self.assertWarnsRegex(
            DeprecationWarning,
            expected_regex="The `transpile` function will "
            "stop supporting inputs of type `BackendV1`",
        ):
            res = transpile(qc, backend)
        self.assertIn("delay", res.count_ops())

    @data(0, 1, 2, 3)
    def test_converter(self, opt_level):
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()
        backend_v2 = BackendV2Converter(backend)
        self.assertIsInstance(backend_v2, BackendV2)
        res = transpile(self.circuit, backend_v2, optimization_level=opt_level)
        job = backend_v2.run(res)
        with warnings.catch_warnings():
            # TODO remove this catch once Aer stops using QobjDictField
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
            result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    def test_converter_delay_circuit(self):
        with self.assertWarns(DeprecationWarning):
            backend = Fake27QPulseV1()
        backend.configuration().timing_constraints = {
            "acquire_alignment": 1,
            "granularity": 1,
            "min_length": 1,
            "pulse_alignment": 1,
        }
        backend_v2 = BackendV2Converter(backend, add_delay=True)
        self.assertIsInstance(backend_v2, BackendV2)
        qc = QuantumCircuit(2)
        qc.delay(502, 0, unit="ns")
        qc.x(1)
        qc.delay(250, 1, unit="ns")
        qc.measure_all()
        res = transpile(qc, backend_v2)
        self.assertIn("delay", res.count_ops())

    def test_converter_with_missing_gate_property(self):
        """Test converting to V2 model with irregular backend data."""
        with self.assertWarns(DeprecationWarning):
            backend = FakeOpenPulse2Q()
        # The backend includes pulse calibration definition for U2, but its property is gone.
        # Note that u2 is a basis gate of this device.
        # Since gate property is not provided, the gate broadcasts to all qubits as ideal instruction.
        del backend._properties._gates["u2"]

        # This should not raise error
        backend_v2 = BackendV2Converter(backend, add_delay=True)
        self.assertDictEqual(backend_v2.target["u2"], {None: None})

    def test_non_cx_tests(self):
        backend = GenericBackendV2(num_qubits=5, basis_gates=["cz", "x", "sx", "id", "rz"], seed=42)
        self.assertIsInstance(backend.target.operation_from_name("cz"), CZGate)
        backend = GenericBackendV2(
            num_qubits=5, basis_gates=["ecr", "x", "sx", "id", "rz"], seed=42
        )
        self.assertIsInstance(backend.target.operation_from_name("ecr"), ECRGate)

    @unittest.skipUnless(optionals.HAS_AER, "Aer required for this test")
    def test_converter_simulator(self):
        class MCSXGate(ControlledGate):
            def __init__(self, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcsx",
                    1 + num_ctrl_qubits,
                    [],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=SXGate(),
                )

        class MCYGate(ControlledGate):
            def __init__(self, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcy",
                    1 + num_ctrl_qubits,
                    [],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=YGate(),
                )

        class MCZGate(ControlledGate):
            def __init__(self, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcz",
                    1 + num_ctrl_qubits,
                    [],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=ZGate(),
                )

        class MCRXGate(ControlledGate):
            def __init__(self, theta, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcrx",
                    1 + num_ctrl_qubits,
                    [theta],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=RXGate(theta),
                )

        class MCRYGate(ControlledGate):
            def __init__(self, theta, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcry",
                    1 + num_ctrl_qubits,
                    [theta],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=RYGate(theta),
                )

        class MCRZGate(ControlledGate):
            def __init__(self, theta, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcrz",
                    1 + num_ctrl_qubits,
                    [theta],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=RZGate(theta),
                )

        class MCRGate(ControlledGate):
            def __init__(self, theta, phi, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcr",
                    1 + num_ctrl_qubits,
                    [theta, phi],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=RGate(theta, phi),
                )

        class MCU1Gate(ControlledGate):
            def __init__(self, theta, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcu1",
                    1 + num_ctrl_qubits,
                    [theta],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=U1Gate(theta),
                )

        class MCU2Gate(ControlledGate):
            def __init__(self, theta, lam, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcu2",
                    1 + num_ctrl_qubits,
                    [theta, lam],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=U2Gate(theta, lam),
                )

        class MCU3Gate(ControlledGate):
            def __init__(self, theta, lam, phi, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcu3",
                    1 + num_ctrl_qubits,
                    [theta, phi, lam],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=U3Gate(theta, phi, lam),
                )

        class MCUGate(ControlledGate):
            def __init__(self, theta, lam, phi, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcu",
                    1 + num_ctrl_qubits,
                    [theta, phi, lam],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=U3Gate(theta, phi, lam),
                )

        class MCSwapGate(ControlledGate):
            def __init__(self, num_ctrl_qubits, ctrl_state=None):
                super().__init__(
                    "mcswap",
                    2 + num_ctrl_qubits,
                    [],
                    None,
                    num_ctrl_qubits,
                    ctrl_state=ctrl_state,
                    base_gate=SwapGate(),
                )

        from qiskit_aer import AerSimulator
        from qiskit_aer.library import (
            SaveExpectationValue,
            SaveAmplitudes,
            SaveStatevectorDict,
            SaveSuperOp,
            SaveClifford,
            SaveMatrixProductState,
            SaveDensityMatrix,
            SaveProbabilities,
            SaveStatevector,
            SetDensityMatrix,
            SetUnitary,
            SaveState,
            SetMatrixProductState,
            SaveUnitary,
            SetSuperOp,
            SaveExpectationValueVariance,
            SaveStabilizer,
            SetStatevector,
            SetStabilizer,
            SaveAmplitudesSquared,
            SaveProbabilitiesDict,
        )
        from qiskit_aer.noise.errors import ReadoutError
        from qiskit_aer.noise.noise_model import QuantumErrorLocation

        sim = AerSimulator()
        # test only if simulator's backend is V1
        if sim.version > 1:
            return
        phi = Parameter("phi")
        lam = Parameter("lam")
        backend = BackendV2Converter(
            sim,
            name_mapping={
                "mcsx": MCSXGate,
                "mcp": MCPhaseGate,
                "mcphase": MCPhaseGate,
                "quantum_channel": QuantumChannel,
                "initialize": Initialize,
                "save_expval": SaveExpectationValue,
                "diagonal": DiagonalGate,
                "save_amplitudes": SaveAmplitudes,
                "roerror": ReadoutError,
                "mcrx": MCRXGate,
                "kraus": Kraus,
                "save_statevector_dict": SaveStatevectorDict,
                "mcx": MCXGate,
                "mcu1": MCU1Gate,
                "mcu2": MCU2Gate,
                "mcu3": MCU3Gate,
                "save_superop": SaveSuperOp,
                "multiplexer": UCGate,
                "mcy": MCYGate,
                "superop": SuperOp,
                "save_clifford": SaveClifford,
                "save_matrix_product_state": SaveMatrixProductState,
                "save_density_matrix": SaveDensityMatrix,
                "save_probabilities": SaveProbabilities,
                "if_else": IfElseOp,
                "while_loop": WhileLoopOp,
                "for_loop": ForLoopOp,
                "switch_case": SwitchCaseOp,
                "break_loop": BreakLoopOp,
                "continue_loop": ContinueLoopOp,
                "save_statevector": SaveStatevector,
                "mcu": MCUGate,
                "set_density_matrix": SetDensityMatrix,
                "qerror_loc": QuantumErrorLocation,
                "unitary": UnitaryGate,
                "mcz": MCZGate,
                "pauli": PauliGate,
                "set_unitary": SetUnitary,
                "save_state": SaveState,
                "mcswap": MCSwapGate,
                "set_matrix_product_state": SetMatrixProductState,
                "save_unitary": SaveUnitary,
                "mcr": MCRGate,
                "mcx_gray": MCXGrayCode,
                "mcrz": MCRZGate,
                "set_superop": SetSuperOp,
                "save_expval_var": SaveExpectationValueVariance,
                "save_stabilizer": SaveStabilizer,
                "set_statevector": SetStatevector,
                "mcry": MCRYGate,
                "set_stabilizer": SetStabilizer,
                "save_amplitudes_sq": SaveAmplitudesSquared,
                "save_probabilities_dict": SaveProbabilitiesDict,
                "cu2": U2Gate(phi, lam).control(),
            },
        )
        self.assertIsInstance(backend, BackendV2)
        res = transpile(self.circuit, backend)
        job = backend.run(res)
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    def test_filter_faulty_qubits_backend_v2_converter(self):
        """Test faulty qubits in v2 conversion."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake127QPulseV1()
            # Get properties dict to make it easier to work with the properties API
            # is difficult to edit because of the multiple layers of nesting and
            # different object types
            props_dict = backend.properties().to_dict()
        for i in range(62, 67):
            non_operational = {
                "date": datetime.datetime.now(datetime.timezone.utc),
                "name": "operational",
                "unit": "",
                "value": 0,
            }
            props_dict["qubits"][i].append(non_operational)
        with self.assertWarns(DeprecationWarning):
            backend._properties = BackendProperties.from_dict(props_dict)
        v2_backend = BackendV2Converter(backend, filter_faulty=True)
        for i in range(62, 67):
            for qarg in v2_backend.target.qargs:
                self.assertNotIn(i, qarg)

    def test_filter_faulty_qubits_backend_v2_converter_with_delay(self):
        """Test faulty qubits in v2 conversion."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake127QPulseV1()
            # Get properties dict to make it easier to work with the properties API
            # is difficult to edit because of the multiple layers of nesting and
            # different object types
            props_dict = backend.properties().to_dict()
        for i in range(62, 67):
            non_operational = {
                "date": datetime.datetime.now(datetime.timezone.utc),
                "name": "operational",
                "unit": "",
                "value": 0,
            }
            props_dict["qubits"][i].append(non_operational)
        with self.assertWarns(DeprecationWarning):
            backend._properties = BackendProperties.from_dict(props_dict)
        v2_backend = BackendV2Converter(backend, filter_faulty=True, add_delay=True)
        for i in range(62, 67):
            for qarg in v2_backend.target.qargs:
                self.assertNotIn(i, qarg)

    def test_backend_v2_converter_without_delay(self):
        """Test setting :code:`add_delay`argument of :func:`.BackendV2Converter`
        to :code:`False`."""

        expected = {
            (0,),
            (0, 1),
            (0, 2),
            (1,),
            (1, 0),
            (1, 2),
            (2,),
            (2, 0),
            (2, 1),
            (2, 3),
            (2, 4),
            (3,),
            (3, 2),
            (3, 4),
            (4,),
            (4, 2),
            (4, 3),
        }
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()
        backend = BackendV2Converter(backend=backend, filter_faulty=True, add_delay=False)

        self.assertEqual(backend.target.qargs, expected)

    def test_backend_v2_converter_with_meaningless_gate_config(self):
        """Test backend with broken gate config can be converted only with properties data."""
        with self.assertWarns(DeprecationWarning):
            backend_v1 = Fake5QV1()
            backend_v1.configuration().gates = [
                GateConfig(name="NotValidGate", parameters=[], qasm_def="not_valid_gate")
            ]
        backend_v2 = BackendV2Converter(
            backend=backend_v1,
            filter_faulty=True,
            add_delay=False,
        )
        ops_with_measure = backend_v2.target.operation_names
        self.assertCountEqual(
            ops_with_measure,
            backend_v1.configuration().basis_gates + ["measure"],
        )

    def test_filter_faulty_qubits_and_gates_backend_v2_converter(self):
        """Test faulty gates and qubits."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake127QPulseV1()
            # Get properties dict to make it easier to work with the properties API
            # is difficult to edit because of the multiple layers of nesting and
            # different object types
            props_dict = backend.properties().to_dict()
        for i in range(62, 67):
            non_operational = {
                "date": datetime.datetime.now(datetime.timezone.utc),
                "name": "operational",
                "unit": "",
                "value": 0,
            }
            props_dict["qubits"][i].append(non_operational)
        invalid_cx_edges = {
            (113, 114),
            (114, 113),
            (96, 100),
            (100, 96),
            (114, 109),
            (109, 114),
            (24, 34),
            (34, 24),
        }
        non_operational_gate = {
            "date": datetime.datetime.now(datetime.timezone.utc),
            "name": "operational",
            "unit": "",
            "value": 0,
        }
        for gate in props_dict["gates"]:
            if tuple(gate["qubits"]) in invalid_cx_edges:
                gate["parameters"].append(non_operational_gate)

        with self.assertWarns(DeprecationWarning):
            backend._properties = BackendProperties.from_dict(props_dict)
        v2_backend = BackendV2Converter(backend, filter_faulty=True)
        for i in range(62, 67):
            for qarg in v2_backend.target.qargs:
                self.assertNotIn(i, qarg)
        for edge in invalid_cx_edges:
            self.assertNotIn(edge, v2_backend.target["cx"])

    def test_filter_faulty_gates_v2_converter(self):
        """Test just faulty gates in conversion."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake127QPulseV1()
        # Get properties dict to make it easier to work with the properties API
        # is difficult to edit because of the multiple layers of nesting and
        # different object types
        props_dict = backend.properties().to_dict()
        invalid_cx_edges = {
            (113, 114),
            (114, 113),
            (96, 100),
            (100, 96),
            (114, 109),
            (109, 114),
            (24, 34),
            (34, 24),
        }
        non_operational_gate = {
            "date": datetime.datetime.now(datetime.timezone.utc),
            "name": "operational",
            "unit": "",
            "value": 0,
        }
        for gate in props_dict["gates"]:
            if tuple(gate["qubits"]) in invalid_cx_edges:
                gate["parameters"].append(non_operational_gate)

        with self.assertWarns(DeprecationWarning):
            backend._properties = BackendProperties.from_dict(props_dict)
        v2_backend = BackendV2Converter(backend, filter_faulty=True)
        for i in range(62, 67):
            self.assertIn((i,), v2_backend.target.qargs)
        for edge in invalid_cx_edges:
            self.assertNotIn(edge, v2_backend.target["cx"])

    def test_filter_faulty_no_faults_v2_converter(self):
        """Test that faulty qubit filtering does nothing with all operational qubits and gates."""
        with self.assertWarns(DeprecationWarning):
            backend = Fake127QPulseV1()
            v2_backend = BackendV2Converter(backend, filter_faulty=True)
        for i in range(v2_backend.num_qubits):
            self.assertIn((i,), v2_backend.target.qargs)

    @data(0, 1, 2, 3)
    def test_faulty_full_path_transpile_connected_cmap(self, opt_level):
        with self.assertWarns(DeprecationWarning):
            backend = Fake5QV1()
            props = backend.properties().to_dict()

        non_operational_gate = {
            "date": datetime.datetime.now(datetime.timezone.utc),
            "name": "operational",
            "unit": "",
            "value": 0,
        }
        for gate in props["gates"]:
            if tuple(sorted(gate["qubits"])) == (0, 1):
                gate["parameters"].append(non_operational_gate)
        with self.assertWarns(DeprecationWarning):
            backend._properties = BackendProperties.from_dict(props)
        v2_backend = BackendV2Converter(backend, filter_faulty=True)
        qc = QuantumCircuit(5)
        for x, y in itertools.product(range(5), range(5)):
            if x == y:
                continue
            qc.cx(x, y)
        tqc = transpile(qc, v2_backend, seed_transpiler=433, optimization_level=opt_level)
        connections = [tuple(sorted(tqc.find_bit(q).index for q in x.qubits)) for x in tqc.data]
        self.assertNotIn((0, 1), connections)

    def test_convert_to_target_control_flow(self):
        with self.assertWarns(DeprecationWarning):
            backend = Fake27QPulseV1()
        properties = backend.properties()
        configuration = backend.configuration()
        configuration.supported_instructions = [
            "cx",
            "id",
            "delay",
            "measure",
            "reset",
            "rz",
            "sx",
            "x",
            "if_else",
            "for_loop",
            "switch_case",
        ]
        defaults = backend.defaults()
        target = convert_to_target(configuration, properties, defaults)
        self.assertTrue(target.instruction_supported("if_else", ()))
        self.assertFalse(target.instruction_supported("while_loop", ()))
        self.assertTrue(target.instruction_supported("for_loop", ()))
        self.assertTrue(target.instruction_supported("switch_case", ()))

    def test_convert_unrelated_supported_instructions(self):
        with self.assertWarns(DeprecationWarning):
            backend = Fake27QPulseV1()
        properties = backend.properties()
        configuration = backend.configuration()
        configuration.supported_instructions = [
            "cx",
            "id",
            "delay",
            "measure",
            "reset",
            "rz",
            "sx",
            "x",
            "play",
            "u2",
            "u3",
            "u1",
            "shiftf",
            "acquire",
            "setf",
            "if_else",
            "for_loop",
            "switch_case",
        ]
        defaults = backend.defaults()
        target = convert_to_target(configuration, properties, defaults)
        self.assertTrue(target.instruction_supported("if_else", ()))
        self.assertFalse(target.instruction_supported("while_loop", ()))
        self.assertTrue(target.instruction_supported("for_loop", ()))
        self.assertTrue(target.instruction_supported("switch_case", ()))
        self.assertFalse(target.instruction_supported("u3", (0,)))

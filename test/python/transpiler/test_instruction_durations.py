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

# pylint: disable=missing-function-docstring

"""Test InstructionDurations class."""

from qiskit.circuit import Delay, Parameter
from qiskit.providers.fake_provider import Fake27QPulseV1
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestInstructionDurationsClass(QiskitTestCase):
    """Test InstructionDurations class."""

    def test_empty(self):
        durations = InstructionDurations()
        self.assertEqual(durations.dt, None)
        with self.assertRaises(TranspilerError):
            durations.get("cx", [0, 1], "dt")

    def test_fail_if_invalid_dict_is_supplied_when_construction(self):
        invalid_dic = [("cx", [0, 1])]  # no duration
        with self.assertRaises(TranspilerError):
            InstructionDurations(invalid_dic)

    def test_from_backend_for_backend_with_dt(self):
        # Remove context once https://github.com/Qiskit/qiskit/issues/12760 is fixed
        with self.assertWarns(DeprecationWarning):
            backend = Fake27QPulseV1()
        gate = self._find_gate_with_length(backend)
        durations = InstructionDurations.from_backend(backend)
        self.assertGreater(durations.dt, 0)
        self.assertGreater(durations.get(gate, 0), 0)

    def test_from_backend_for_backend_without_dt(self):
        # Remove context once https://github.com/Qiskit/qiskit/issues/12760 is fixed
        with self.assertWarns(DeprecationWarning):
            backend = Fake27QPulseV1()
        delattr(backend.configuration(), "dt")
        gate = self._find_gate_with_length(backend)
        durations = InstructionDurations.from_backend(backend)
        self.assertIsNone(durations.dt)
        self.assertGreater(durations.get(gate, 0, "s"), 0)
        with self.assertRaises(TranspilerError):
            durations.get(gate, 0)

    def test_update_with_parameters(self):
        durations = InstructionDurations(
            [("rzx", (0, 1), 150, (0.5,)), ("rzx", (0, 1), 300, (1.0,))]
        )

        self.assertEqual(durations.get("rzx", [0, 1], parameters=[0.5]), 150)
        self.assertEqual(durations.get("rzx", [0, 1], parameters=[1.0]), 300)

    def _find_gate_with_length(self, backend):
        """Find a gate that has gate length."""
        props = backend.properties()
        for gate in props.gates:
            try:
                if props.gate_length(gate.gate, 0):
                    return gate.gate
            except Exception:  # pylint: disable=broad-except
                pass
        raise ValueError("Unable to find a gate with gate length.")

    def test_can_get_unbounded_duration_without_unit_conversion(self):
        param = Parameter("t")
        parameterized_delay = Delay(param, "dt")
        actual = InstructionDurations().get(parameterized_delay, 0)
        self.assertEqual(actual, param)

    def test_can_get_unbounded_duration_with_unit_conversion_when_dt_is_provided(self):
        param = Parameter("t")
        parameterized_delay = Delay(param, "s")
        actual = InstructionDurations(dt=100).get(parameterized_delay, 0)
        self.assertEqual(actual, param / 100)

    def test_fail_if_get_unbounded_duration_with_unit_conversion_when_dt_is_not_provided(self):
        param = Parameter("t")
        parameterized_delay = Delay(param, "s")
        with self.assertRaises(TranspilerError):
            InstructionDurations().get(parameterized_delay, 0)

    def test_from_backend_with_backendv2(self):
        """Test if `from_backend()` method allows using BackendV2"""
        with self.assertWarns(DeprecationWarning):
            backend = GenericBackendV2(num_qubits=4, calibrate_instructions=True, seed=42)
        inst_durations = InstructionDurations.from_backend(backend)
        self.assertEqual(inst_durations, backend.target.durations())
        self.assertIsInstance(inst_durations, InstructionDurations)

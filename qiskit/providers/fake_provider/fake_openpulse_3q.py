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

"""
Fake backend supporting OpenPulse.
"""
import warnings

from qiskit.providers.models.backendconfiguration import (
    GateConfig,
    PulseBackendConfiguration,
    UchannelLO,
)
from qiskit.providers.models.pulsedefaults import PulseDefaults, Command
from qiskit.qobj import PulseQobjInstruction

from .fake_backend import FakeBackend


class FakeOpenPulse3Q(FakeBackend):
    """Trivial extension of the FakeOpenPulse2Q."""

    def __init__(self):
        configuration = PulseBackendConfiguration(
            backend_name="fake_openpulse_3q",
            backend_version="0.0.0",
            n_qubits=3,
            meas_levels=[0, 1, 2],
            basis_gates=["u1", "u2", "u3", "cx", "id"],
            simulator=False,
            local=True,
            conditional=True,
            open_pulse=True,
            memory=False,
            max_shots=65536,
            gates=[GateConfig(name="TODO", parameters=[], qasm_def="TODO")],
            coupling_map=[[0, 1], [1, 2]],
            n_registers=3,
            n_uchannels=3,
            u_channel_lo=[
                [UchannelLO(q=0, scale=1.0 + 0.0j)],
                [UchannelLO(q=0, scale=-1.0 + 0.0j), UchannelLO(q=1, scale=1.0 + 0.0j)],
                [UchannelLO(q=0, scale=1.0 + 0.0j)],
            ],
            qubit_lo_range=[[4.5, 5.5], [4.5, 5.5], [4.5, 5.5]],
            meas_lo_range=[[6.0, 7.0], [6.0, 7.0], [6.0, 7.0]],
            dt=1.3333,
            dtm=10.5,
            rep_times=[100, 250, 500, 1000],
            meas_map=[[0, 1, 2]],
            channel_bandwidth=[
                [-0.2, 0.4],
                [-0.3, 0.3],
                [-0.3, 0.3],
                [-0.02, 0.02],
                [-0.02, 0.02],
                [-0.02, 0.02],
                [-0.2, 0.4],
                [-0.3, 0.3],
                [-0.3, 0.3],
            ],
            meas_kernels=["kernel1"],
            discriminators=["max_1Q_fidelity"],
            acquisition_latency=[[100, 100], [100, 100], [100, 100]],
            conditional_latency=[
                [100, 1000],
                [1000, 100],
                [100, 1000],
                [100, 1000],
                [1000, 100],
                [100, 1000],
                [1000, 100],
                [100, 1000],
                [1000, 100],
            ],
            channels={
                "acquire0": {"type": "acquire", "purpose": "acquire", "operates": {"qubits": [0]}},
                "acquire1": {"type": "acquire", "purpose": "acquire", "operates": {"qubits": [1]}},
                "acquire2": {"type": "acquire", "purpose": "acquire", "operates": {"qubits": [2]}},
                "d0": {"type": "drive", "purpose": "drive", "operates": {"qubits": [0]}},
                "d1": {"type": "drive", "purpose": "drive", "operates": {"qubits": [1]}},
                "d2": {"type": "drive", "purpose": "drive", "operates": {"qubits": [2]}},
                "m0": {"type": "measure", "purpose": "measure", "operates": {"qubits": [0]}},
                "m1": {"type": "measure", "purpose": "measure", "operates": {"qubits": [1]}},
                "m2": {"type": "measure", "purpose": "measure", "operates": {"qubits": [2]}},
                "u0": {
                    "type": "control",
                    "purpose": "cross-resonance",
                    "operates": {"qubits": [0, 1]},
                },
                "u1": {
                    "type": "control",
                    "purpose": "cross-resonance",
                    "operates": {"qubits": [1, 0]},
                },
                "u2": {
                    "type": "control",
                    "purpose": "cross-resonance",
                    "operates": {"qubits": [2, 1]},
                },
            },
        )
        with warnings.catch_warnings():
            # The class PulseQobjInstruction is deprecated
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
            self._defaults = PulseDefaults.from_dict(
                {
                    "qubit_freq_est": [4.9, 5.0, 4.8],
                    "meas_freq_est": [6.5, 6.6, 6.4],
                    "buffer": 10,
                    "pulse_library": [
                        {"name": "x90p_d0", "samples": 2 * [0.1 + 0j]},
                        {"name": "x90p_d1", "samples": 2 * [0.1 + 0j]},
                        {"name": "x90p_d2", "samples": 2 * [0.1 + 0j]},
                        {"name": "x90m_d0", "samples": 2 * [-0.1 + 0j]},
                        {"name": "x90m_d1", "samples": 2 * [-0.1 + 0j]},
                        {"name": "x90m_d2", "samples": 2 * [-0.1 + 0j]},
                        {"name": "y90p_d0", "samples": 2 * [0.1j]},
                        {"name": "y90p_d1", "samples": 2 * [0.1j]},
                        {"name": "y90p_d2", "samples": 2 * [0.1j]},
                        {"name": "xp_d0", "samples": 2 * [0.2 + 0j]},
                        {"name": "ym_d0", "samples": 2 * [-0.2j]},
                        {"name": "xp_d1", "samples": 2 * [0.2 + 0j]},
                        {"name": "ym_d1", "samples": 2 * [-0.2j]},
                        {"name": "cr90p_u0", "samples": 9 * [0.1 + 0j]},
                        {"name": "cr90m_u0", "samples": 9 * [-0.1 + 0j]},
                        {"name": "cr90p_u1", "samples": 9 * [0.1 + 0j]},
                        {"name": "cr90m_u1", "samples": 9 * [-0.1 + 0j]},
                        {"name": "measure_m0", "samples": 10 * [0.1 + 0j]},
                        {"name": "measure_m1", "samples": 10 * [0.1 + 0j]},
                        {"name": "measure_m2", "samples": 10 * [0.1 + 0j]},
                    ],
                    "cmd_def": [
                        Command.from_dict(
                            {
                                "name": "u1",
                                "qubits": [0],
                                "sequence": [
                                    PulseQobjInstruction(
                                        name="fc", ch="d0", t0=0, phase="-P0"
                                    ).to_dict()
                                ],
                            }
                        ).to_dict(),
                        Command.from_dict(
                            {
                                "name": "u1",
                                "qubits": [1],
                                "sequence": [
                                    PulseQobjInstruction(
                                        name="fc", ch="d1", t0=0, phase="-P0"
                                    ).to_dict()
                                ],
                            }
                        ).to_dict(),
                        Command.from_dict(
                            {
                                "name": "u1",
                                "qubits": [2],
                                "sequence": [
                                    PulseQobjInstruction(
                                        name="fc", ch="d2", t0=0, phase="-P0"
                                    ).to_dict()
                                ],
                            }
                        ).to_dict(),
                        Command.from_dict(
                            {
                                "name": "u2",
                                "qubits": [0],
                                "sequence": [
                                    PulseQobjInstruction(
                                        name="fc", ch="d0", t0=0, phase="-P1"
                                    ).to_dict(),
                                    PulseQobjInstruction(name="y90p_d0", ch="d0", t0=0).to_dict(),
                                    PulseQobjInstruction(
                                        name="fc", ch="d0", t0=2, phase="-P0"
                                    ).to_dict(),
                                ],
                            }
                        ).to_dict(),
                        Command.from_dict(
                            {
                                "name": "u2",
                                "qubits": [1],
                                "sequence": [
                                    PulseQobjInstruction(
                                        name="fc", ch="d1", t0=0, phase="-P1"
                                    ).to_dict(),
                                    PulseQobjInstruction(name="y90p_d1", ch="d1", t0=0).to_dict(),
                                    PulseQobjInstruction(
                                        name="fc", ch="d1", t0=2, phase="-P0"
                                    ).to_dict(),
                                ],
                            }
                        ).to_dict(),
                        Command.from_dict(
                            {
                                "name": "u2",
                                "qubits": [2],
                                "sequence": [
                                    PulseQobjInstruction(
                                        name="fc", ch="d2", t0=0, phase="-P1"
                                    ).to_dict(),
                                    PulseQobjInstruction(name="y90p_d2", ch="d2", t0=0).to_dict(),
                                    PulseQobjInstruction(
                                        name="fc", ch="d2", t0=2, phase="-P0"
                                    ).to_dict(),
                                ],
                            }
                        ).to_dict(),
                        Command.from_dict(
                            {
                                "name": "u3",
                                "qubits": [0],
                                "sequence": [
                                    PulseQobjInstruction(
                                        name="fc", ch="d0", t0=0, phase="-P2"
                                    ).to_dict(),
                                    PulseQobjInstruction(name="x90p_d0", ch="d0", t0=0).to_dict(),
                                    PulseQobjInstruction(
                                        name="fc", ch="d0", t0=2, phase="-P0"
                                    ).to_dict(),
                                    PulseQobjInstruction(name="x90m_d0", ch="d0", t0=2).to_dict(),
                                    PulseQobjInstruction(
                                        name="fc", ch="d0", t0=4, phase="-P1"
                                    ).to_dict(),
                                ],
                            }
                        ).to_dict(),
                        Command.from_dict(
                            {
                                "name": "u3",
                                "qubits": [1],
                                "sequence": [
                                    PulseQobjInstruction(
                                        name="fc", ch="d1", t0=0, phase="-P2"
                                    ).to_dict(),
                                    PulseQobjInstruction(name="x90p_d1", ch="d1", t0=0).to_dict(),
                                    PulseQobjInstruction(
                                        name="fc", ch="d1", t0=2, phase="-P0"
                                    ).to_dict(),
                                    PulseQobjInstruction(name="x90m_d1", ch="d1", t0=2).to_dict(),
                                    PulseQobjInstruction(
                                        name="fc", ch="d1", t0=4, phase="-P1"
                                    ).to_dict(),
                                ],
                            }
                        ).to_dict(),
                        Command.from_dict(
                            {
                                "name": "u3",
                                "qubits": [2],
                                "sequence": [
                                    PulseQobjInstruction(
                                        name="fc", ch="d2", t0=0, phase="-P2"
                                    ).to_dict(),
                                    PulseQobjInstruction(name="x90p_d2", ch="d2", t0=0).to_dict(),
                                    PulseQobjInstruction(
                                        name="fc", ch="d2", t0=2, phase="-P0"
                                    ).to_dict(),
                                    PulseQobjInstruction(name="x90m_d2", ch="d2", t0=2).to_dict(),
                                    PulseQobjInstruction(
                                        name="fc", ch="d2", t0=4, phase="-P1"
                                    ).to_dict(),
                                ],
                            }
                        ).to_dict(),
                        Command.from_dict(
                            {
                                "name": "cx",
                                "qubits": [0, 1],
                                "sequence": [
                                    PulseQobjInstruction(
                                        name="fc", ch="d0", t0=0, phase=1.57
                                    ).to_dict(),
                                    PulseQobjInstruction(name="ym_d0", ch="d0", t0=0).to_dict(),
                                    PulseQobjInstruction(name="xp_d0", ch="d0", t0=11).to_dict(),
                                    PulseQobjInstruction(name="x90p_d1", ch="d1", t0=0).to_dict(),
                                    PulseQobjInstruction(name="cr90p_u0", ch="u0", t0=2).to_dict(),
                                    PulseQobjInstruction(name="cr90m_u0", ch="u0", t0=13).to_dict(),
                                ],
                            }
                        ).to_dict(),
                        Command.from_dict(
                            {
                                "name": "cx",
                                "qubits": [1, 2],
                                "sequence": [
                                    PulseQobjInstruction(
                                        name="fc", ch="d1", t0=0, phase=1.57
                                    ).to_dict(),
                                    PulseQobjInstruction(name="ym_d1", ch="d1", t0=0).to_dict(),
                                    PulseQobjInstruction(name="xp_d1", ch="d1", t0=11).to_dict(),
                                    PulseQobjInstruction(name="x90p_d2", ch="d2", t0=0).to_dict(),
                                    PulseQobjInstruction(name="cr90p_u1", ch="u1", t0=2).to_dict(),
                                    PulseQobjInstruction(name="cr90m_u1", ch="u1", t0=13).to_dict(),
                                ],
                            }
                        ).to_dict(),
                        Command.from_dict(
                            {
                                "name": "measure",
                                "qubits": [0, 1, 2],
                                "sequence": [
                                    PulseQobjInstruction(
                                        name="measure_m0", ch="m0", t0=0
                                    ).to_dict(),
                                    PulseQobjInstruction(
                                        name="measure_m1", ch="m1", t0=0
                                    ).to_dict(),
                                    PulseQobjInstruction(
                                        name="measure_m2", ch="m2", t0=0
                                    ).to_dict(),
                                    PulseQobjInstruction(
                                        name="acquire",
                                        duration=10,
                                        t0=0,
                                        qubits=[0, 1, 2],
                                        memory_slot=[0, 1, 2],
                                    ).to_dict(),
                                ],
                            }
                        ).to_dict(),
                    ],
                }
            )
        super().__init__(configuration)

    def defaults(self):  # pylint: disable=missing-function-docstring
        return self._defaults

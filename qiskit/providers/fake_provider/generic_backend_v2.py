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

"""Generic BackendV2 class that with a simulated ``run``."""

from __future__ import annotations
import warnings

from collections.abc import Iterable
from typing import List, Dict, Any, Union
import numpy as np

from qiskit import pulse
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.controlflow import (
    IfElseOp,
    WhileLoopOp,
    ForLoopOp,
    SwitchCaseOp,
    BreakLoopOp,
    ContinueLoopOp,
)
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap, Target, InstructionProperties, QubitProperties
from qiskit.providers import Options
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.providers.backend import BackendV2
from qiskit.utils import optionals as _optionals
from qiskit.providers.models.pulsedefaults import Command
from qiskit.qobj.converters.pulse_instruction import QobjToInstructionConverter
from qiskit.pulse.calibration_entries import PulseQobjDef
from qiskit.providers.models.pulsedefaults import MeasurementKernel, Discriminator
from qiskit.qobj.pulse_qobj import QobjMeasurementOption
from qiskit.utils.deprecate_pulse import deprecate_pulse_dependency, deprecate_pulse_arg

# Noise default values/ranges for duration and error of supported
# instructions. There are two possible formats:
# - (min_duration, max_duration, min_error, max_error),
#   if the defaults are ranges.
# - (duration, error), if the defaults are fixed values.
_NOISE_DEFAULTS = {
    "cx": (7.992e-08, 8.99988e-07, 1e-5, 5e-3),
    "ecr": (7.992e-08, 8.99988e-07, 1e-5, 5e-3),
    "cz": (7.992e-08, 8.99988e-07, 1e-5, 5e-3),
    "id": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "rz": (0.0, 0.0),
    "sx": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "x": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "measure": (6.99966e-07, 1.500054e-06, 1e-5, 5e-3),
    "delay": (None, None),
    "reset": (None, None),
}

# Fallback values for gates with unknown noise default ranges.
_NOISE_DEFAULTS_FALLBACK = {
    "1-q": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "multi-q": (7.992e-08, 8.99988e-07, 5e-3),
}

# Ranges to sample qubit properties from.
_QUBIT_PROPERTIES = {
    "dt": 0.222e-9,
    "t1": (100e-6, 200e-6),
    "t2": (100e-6, 200e-6),
    "frequency": (5e9, 5.5e9),
}


class PulseDefaults:
    """Internal - Description of default settings for Pulse systems. These are instructions
    or settings that
    may be good starting points for the Pulse user. The user may modify these defaults for custom
    scheduling.
    """

    # Copy from the deprecated from qiskit.providers.models.pulsedefaults.PulseDefaults

    _data = {}

    def __init__(
        self,
        qubit_freq_est: List[float],
        meas_freq_est: List[float],
        buffer: int,
        pulse_library: List[PulseLibraryItem],
        cmd_def: List[Command],
        meas_kernel: MeasurementKernel = None,
        discriminator: Discriminator = None,
        **kwargs: Dict[str, Any],
    ):
        """
        Validate and reformat transport layer inputs to initialize.
        Args:
            qubit_freq_est: Estimated qubit frequencies in GHz.
            meas_freq_est: Estimated measurement cavity frequencies in GHz.
            buffer: Default buffer time (in units of dt) between pulses.
            pulse_library: Pulse name and sample definitions.
            cmd_def: Operation name and definition in terms of Commands.
            meas_kernel: The measurement kernels
            discriminator: The discriminators
            **kwargs: Other attributes for the super class.
        """
        self._data = {}
        self.buffer = buffer
        self.qubit_freq_est = [freq * 1e9 for freq in qubit_freq_est]
        """Qubit frequencies in Hertz."""
        self.meas_freq_est = [freq * 1e9 for freq in meas_freq_est]
        """Measurement frequencies in Hertz."""
        self.pulse_library = pulse_library
        self.cmd_def = cmd_def
        self.instruction_schedule_map = InstructionScheduleMap()
        self.converter = QobjToInstructionConverter(pulse_library)

        for inst in cmd_def:
            entry = PulseQobjDef(converter=self.converter, name=inst.name)
            entry.define(inst.sequence, user_provided=False)
            self.instruction_schedule_map._add(
                instruction_name=inst.name,
                qubits=tuple(inst.qubits),
                entry=entry,
            )

        if meas_kernel is not None:
            self.meas_kernel = meas_kernel
        if discriminator is not None:
            self.discriminator = discriminator

        self._data.update(kwargs)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as ex:
            raise AttributeError(f"Attribute {name} is not defined") from ex

    def to_dict(self):
        """Return a dictionary format representation of the PulseDefaults.
        Returns:
            dict: The dictionary form of the PulseDefaults.
        """
        out_dict = {
            "qubit_freq_est": self.qubit_freq_est,
            "meas_freq_est": self.qubit_freq_est,
            "buffer": self.buffer,
            "pulse_library": [x.to_dict() for x in self.pulse_library],
            "cmd_def": [x.to_dict() for x in self.cmd_def],
        }
        if hasattr(self, "meas_kernel"):
            out_dict["meas_kernel"] = self.meas_kernel.to_dict()
        if hasattr(self, "discriminator"):
            out_dict["discriminator"] = self.discriminator.to_dict()
        for key, value in self.__dict__.items():
            if key not in [
                "qubit_freq_est",
                "meas_freq_est",
                "buffer",
                "pulse_library",
                "cmd_def",
                "meas_kernel",
                "discriminator",
                "converter",
                "instruction_schedule_map",
            ]:
                out_dict[key] = value
        out_dict.update(self._data)

        out_dict["qubit_freq_est"] = [freq * 1e-9 for freq in self.qubit_freq_est]
        out_dict["meas_freq_est"] = [freq * 1e-9 for freq in self.meas_freq_est]
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new PulseDefaults object from a dictionary.

        Args:
            data (dict): A dictionary representing the PulseDefaults
                         to create. It will be in the same format as output by
                         :meth:`to_dict`.
        Returns:
            PulseDefaults: The PulseDefaults from the input dictionary.
        """
        schema = {
            "pulse_library": PulseLibraryItem,  # The class PulseLibraryItem is deprecated
            "cmd_def": Command,
            "meas_kernel": MeasurementKernel,
            "discriminator": Discriminator,
        }

        # Pulse defaults data is nested dictionary.
        # To avoid deepcopy and avoid mutating the source object, create new dict here.
        in_data = {}
        for key, value in data.items():
            if key in schema:
                with warnings.catch_warnings():
                    # The class PulseLibraryItem is deprecated
                    warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
                    if isinstance(value, list):
                        in_data[key] = list(map(schema[key].from_dict, value))
                    else:
                        in_data[key] = schema[key].from_dict(value)
            else:
                in_data[key] = value

        return cls(**in_data)

    def __str__(self):
        qubit_freqs = [freq / 1e9 for freq in self.qubit_freq_est]
        meas_freqs = [freq / 1e9 for freq in self.meas_freq_est]
        qfreq = f"Qubit Frequencies [GHz]\n{qubit_freqs}"
        mfreq = f"Measurement Frequencies [GHz]\n{meas_freqs} "
        return f"<{self.__class__.__name__}({str(self.instruction_schedule_map)}{qfreq}\n{mfreq})>"


def _to_complex(value: Union[List[float], complex]) -> complex:
    """Convert the input value to type ``complex``.
    Args:
        value: Value to be converted.
    Returns:
        Input value in ``complex``.
    Raises:
        TypeError: If the input value is not in the expected format.
    """
    if isinstance(value, list) and len(value) == 2:
        return complex(value[0], value[1])
    elif isinstance(value, complex):
        return value

    raise TypeError(f"{value} is not in a valid complex number format.")


class PulseLibraryItem:
    """INTERNAL - An item in a pulse library."""

    # Copy from the deprecated from qiskit.qobj.PulseLibraryItem
    def __init__(self, name, samples):
        """Instantiate a pulse library item.

        Args:
            name (str): A name for the pulse.
            samples (list[complex]): A list of complex values defining pulse
                shape.
        """
        self.name = name
        if isinstance(samples[0], list):
            self.samples = np.array([complex(sample[0], sample[1]) for sample in samples])
        else:
            self.samples = samples

    def to_dict(self):
        """Return a dictionary format representation of the pulse library item.

        Returns:
            dict: The dictionary form of the PulseLibraryItem.
        """
        return {"name": self.name, "samples": self.samples}

    @classmethod
    def from_dict(cls, data):
        """Create a new PulseLibraryItem object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            PulseLibraryItem: The object from the input dictionary.
        """
        return cls(**data)

    def __repr__(self):
        return f"PulseLibraryItem({self.name}, {repr(self.samples)})"

    def __str__(self):
        return f"Pulse Library Item:\n\tname: {self.name}\n\tsamples: {self.samples}"

    def __eq__(self, other):
        if isinstance(other, PulseLibraryItem):
            if self.to_dict() == other.to_dict():
                return True
        return False


class PulseQobjInstruction:
    """Internal - A class representing a single instruction in a PulseQobj Experiment."""

    # Copy from the deprecated from qiskit.qobj.PulseQobjInstruction

    _COMMON_ATTRS = [
        "ch",
        "conditional",
        "val",
        "phase",
        "frequency",
        "duration",
        "qubits",
        "memory_slot",
        "register_slot",
        "label",
        "type",
        "pulse_shape",
        "parameters",
    ]

    def __init__(
        self,
        name,
        t0,
        ch=None,
        conditional=None,
        val=None,
        phase=None,
        duration=None,
        qubits=None,
        memory_slot=None,
        register_slot=None,
        kernels=None,
        discriminators=None,
        label=None,
        type=None,  # pylint: disable=invalid-name,redefined-builtin
        pulse_shape=None,
        parameters=None,
        frequency=None,
    ):
        """Instantiate a new PulseQobjInstruction object.

        Args:
            name (str): The name of the instruction
            t0 (int): Pulse start time in integer **dt** units.
            ch (str): The channel to apply the pulse instruction.
            conditional (int): The register to use for a conditional for this
                instruction
            val (complex): Complex value to apply, bounded by an absolute value
                of 1.
            phase (float): if a ``fc`` instruction, the frame change phase in
                radians.
            frequency (float): if a ``sf`` instruction, the frequency in Hz.
            duration (int): The duration of the pulse in **dt** units.
            qubits (list): A list of ``int`` representing the qubits the
                instruction operates on
            memory_slot (list): If a ``measure`` instruction this is a list
                of ``int`` containing the list of memory slots to store the
                measurement results in (must be the same length as qubits).
                If a ``bfunc`` instruction this is a single ``int`` of the
                memory slot to store the boolean function result in.
            register_slot (list): If a ``measure`` instruction this is a list
                of ``int`` containing the list of register slots in which to
                store the measurement results (must be the same length as
                qubits). If a ``bfunc`` instruction this is a single ``int``
                of the register slot in which to store the result.
            kernels (list): List of :class:`QobjMeasurementOption` objects
                defining the measurement kernels and set of parameters if the
                measurement level is 1 or 2. Only used for ``acquire``
                instructions.
            discriminators (list): A list of :class:`QobjMeasurementOption`
                used to set the discriminators to be used if the measurement
                level is 2. Only used for ``acquire`` instructions.
            label (str): Label of instruction
            type (str): Type of instruction
            pulse_shape (str): The shape of the parametric pulse
            parameters (dict): The parameters for a parametric pulse
        """
        self.name = name
        self.t0 = t0
        if ch is not None:
            self.ch = ch
        if conditional is not None:
            self.conditional = conditional
        if val is not None:
            self.val = val
        if phase is not None:
            self.phase = phase
        if frequency is not None:
            self.frequency = frequency
        if duration is not None:
            self.duration = duration
        if qubits is not None:
            self.qubits = qubits
        if memory_slot is not None:
            self.memory_slot = memory_slot
        if register_slot is not None:
            self.register_slot = register_slot
        if kernels is not None:
            self.kernels = kernels
        if discriminators is not None:
            self.discriminators = discriminators
        if label is not None:
            self.label = label
        if type is not None:
            self.type = type
        if pulse_shape is not None:
            self.pulse_shape = pulse_shape
        if parameters is not None:
            self.parameters = parameters

    def to_dict(self):
        """Return a dictionary format representation of the Instruction.

        Returns:
            dict: The dictionary form of the PulseQobjInstruction.
        """
        out_dict = {"name": self.name, "t0": self.t0}
        for attr in self._COMMON_ATTRS:
            if hasattr(self, attr):
                out_dict[attr] = getattr(self, attr)
        if hasattr(self, "kernels"):
            out_dict["kernels"] = [x.to_dict() for x in self.kernels]
        if hasattr(self, "discriminators"):
            out_dict["discriminators"] = [x.to_dict() for x in self.discriminators]
        return out_dict

    def __repr__(self):
        out = f'PulseQobjInstruction(name="{self.name}", t0={self.t0}'
        for attr in self._COMMON_ATTRS:
            attr_val = getattr(self, attr, None)
            if attr_val is not None:
                if isinstance(attr_val, str):
                    out += f', {attr}="{attr_val}"'
                else:
                    out += f", {attr}={attr_val}"
        out += ")"
        return out

    def __str__(self):
        out = f"Instruction: {self.name}\n"
        out += f"\t\tt0: {self.t0}\n"
        for attr in self._COMMON_ATTRS:
            if hasattr(self, attr):
                out += f"\t\t{attr}: {getattr(self, attr)}\n"
        return out

    @classmethod
    def from_dict(cls, data):
        """Create a new PulseQobjExperimentConfig object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            PulseQobjInstruction: The object from the input dictionary.
        """
        schema = {
            "discriminators": QobjMeasurementOption,
            "kernels": QobjMeasurementOption,
        }
        skip = ["t0", "name"]

        # Pulse instruction data is nested dictionary.
        # To avoid deepcopy and avoid mutating the source object, create new dict here.
        in_data = {}
        for key, value in data.items():
            if key in skip:
                continue
            if key == "parameters":
                # This is flat dictionary of parametric pulse parameters
                formatted_value = value.copy()
                if "amp" in formatted_value:
                    formatted_value["amp"] = _to_complex(formatted_value["amp"])
                in_data[key] = formatted_value
                continue
            if key in schema:
                if isinstance(value, list):
                    in_data[key] = list(map(schema[key].from_dict, value))
                else:
                    in_data[key] = schema[key].from_dict(value)
            else:
                in_data[key] = value

        return cls(data["name"], data["t0"], **in_data)

    def __eq__(self, other):
        if isinstance(other, PulseQobjInstruction):
            if self.to_dict() == other.to_dict():
                return True
        return False


def _pulse_library():
    # The number of samples determines the pulse durations of the corresponding
    # instructions. This default defines pulses with durations in multiples of
    # 16 dt for consistency with the pulse granularity of real IBM devices, but
    # keeps the number smaller than what would be realistic for
    # manageability. If needed, more realistic durations could be added in the
    # future (order of 160dt for 1q gates, 1760dt for 2q gates and measure).
    return [
        PulseLibraryItem(
            name="pulse_1", samples=np.linspace(0, 1.0, 16, dtype=np.complex128)
        ),  # 16dt
        PulseLibraryItem(
            name="pulse_2", samples=np.linspace(0, 1.0, 32, dtype=np.complex128)
        ),  # 32dt
        PulseLibraryItem(
            name="pulse_3", samples=np.linspace(0, 1.0, 64, dtype=np.complex128)
        ),  # 64dt
    ]


class GenericBackendV2(BackendV2):
    """Generic :class:`~.BackendV2` implementation with a configurable constructor. This class will
    return a :class:`~.BackendV2` instance that runs on a local simulator (in the spirit of fake
    backends) and contains all the necessary information to test backend-interfacing components, such
    as the transpiler. A :class:`.GenericBackendV2` instance can be constructed from as little as a
    specified ``num_qubits``, but users can additionally configure the basis gates, coupling map,
    ability to run dynamic circuits (control flow instructions), instruction calibrations and dtm.
    The remainder of the backend properties are generated by randomly sampling
    from default ranges extracted from historical IBM backend data. The seed for this random
    generation can be fixed to ensure the reproducibility of the backend output.
    This backend only supports gates in the standard library, if you need a more flexible backend,
    there is always the option to directly instantiate a :class:`.Target` object to use for
    transpilation.
    """

    @deprecate_pulse_arg("pulse_channels")
    @deprecate_pulse_arg("calibrate_instructions")
    def __init__(
        self,
        num_qubits: int,
        basis_gates: list[str] | None = None,
        *,
        coupling_map: list[list[int]] | CouplingMap | None = None,
        control_flow: bool = False,
        calibrate_instructions: bool | InstructionScheduleMap | None = None,
        dtm: float | None = None,
        seed: int | None = None,
        pulse_channels: bool = True,
        noise_info: bool = True,
    ):
        """
        Args:
            num_qubits: Number of qubits that will be used to construct the backend's target.
                Note that, while there is no limit in the size of the target that can be
                constructed, this backend runs on local noisy simulators, and these might
                present limitations in the number of qubits that can be simulated.

            basis_gates: List of basis gate names to be supported by
                the target. These must be part of the standard qiskit circuit library.
                The default set of basis gates is ``["id", "rz", "sx", "x", "cx"]``
                The ``"reset"``,  ``"delay"``, and ``"measure"`` instructions are
                always supported by default, even if not specified via ``basis_gates``.

            coupling_map: Optional coupling map
                for the backend. Multiple formats are supported:

                #. :class:`~.CouplingMap` instance
                #. List, must be given as an edge list representing the two qubit interactions
                   supported by the backend, for example:
                   ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``

                If ``coupling_map`` is specified, it must match the number of qubits
                specified in ``num_qubits``. If ``coupling_map`` is not specified,
                a fully connected coupling map will be generated with ``num_qubits``
                qubits.

            control_flow: Flag to enable control flow directives on the target
                (defaults to False).

            calibrate_instructions: DEPRECATED. Instruction calibration settings, this argument
                supports both boolean and :class:`.InstructionScheduleMap` as
                input types, and is ``None`` by default:

                #. If ``calibrate_instructions==None``, no calibrations will be added to the target.
                #. If ``calibrate_instructions==True``, all gates will be calibrated for all
                    qubits using the default pulse schedules generated internally.
                #. If ``calibrate_instructions==False``, all gates will be "calibrated" for
                    all qubits with an empty pulse schedule.
                #. If an :class:`.InstructionScheduleMap` instance is given, the calibrations
                    in this instruction schedule map will be appended to the target
                    instead of the default pulse schedules (this allows for custom calibrations).

            dtm: System time resolution of output signals in nanoseconds.
                None by default.

            seed: Optional seed for generation of default values.

            pulse_channels: DEPRECATED. If true, sets default pulse channel information on the backend.

            noise_info: If true, associates gates and qubits with default noise information.
        """

        super().__init__(
            provider=None,
            name=f"generic_backend_{num_qubits}q",
            description=f"This is a device with {num_qubits} qubits and generic settings.",
            backend_version="",
        )

        self._sim = None
        self._rng = np.random.default_rng(seed=seed)
        self._dtm = dtm
        self._num_qubits = num_qubits
        self._control_flow = control_flow
        self._calibrate_instructions = calibrate_instructions
        self._supported_gates = get_standard_gate_name_mapping()
        self._noise_info = noise_info

        if calibrate_instructions and not noise_info:
            raise QiskitError("Must set parameter noise_info when calibrating instructions.")

        if coupling_map is None:
            self._coupling_map = CouplingMap().from_full(num_qubits)
        else:
            if isinstance(coupling_map, CouplingMap):
                self._coupling_map = coupling_map
            else:
                self._coupling_map = CouplingMap(coupling_map)

            if num_qubits != self._coupling_map.size():
                raise QiskitError(
                    f"The number of qubits (got {num_qubits}) must match "
                    f"the size of the provided coupling map (got {self._coupling_map.size()})."
                )

        self._basis_gates = (
            basis_gates if basis_gates is not None else ["cx", "id", "rz", "sx", "x"]
        )
        for name in ["reset", "delay", "measure"]:
            if name not in self._basis_gates:
                self._basis_gates.append(name)

        self._build_generic_target()
        if pulse_channels:
            self._build_default_channels()
        else:
            self.channels_map = {}

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @property
    def dtm(self) -> float:
        """Return the system time resolution of output signals"""
        # converting `dtm` from nanoseconds to seconds
        return self._dtm * 1e-9 if self._dtm is not None else None

    @property
    def meas_map(self) -> list[list[int]]:
        return self._target.concurrent_measurements

    def _build_default_channels(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=DeprecationWarning)
            # Prevent pulse deprecation warnings from being emitted
            channels_map = {
                "acquire": {(i,): [pulse.AcquireChannel(i)] for i in range(self.num_qubits)},
                "drive": {(i,): [pulse.DriveChannel(i)] for i in range(self.num_qubits)},
                "measure": {(i,): [pulse.MeasureChannel(i)] for i in range(self.num_qubits)},
                "control": {
                    (edge): [pulse.ControlChannel(i)] for i, edge in enumerate(self._coupling_map)
                },
            }
        setattr(self, "channels_map", channels_map)

    def _get_noise_defaults(self, name: str, num_qubits: int) -> tuple:
        """Return noise default values/ranges for duration and error of supported
        instructions. There are two possible formats:
            - (min_duration, max_duration, min_error, max_error),
              if the defaults are ranges.
            - (duration, error), if the defaults are fixed values.
        """
        if name in _NOISE_DEFAULTS:
            return _NOISE_DEFAULTS[name]
        if num_qubits == 1:
            return _NOISE_DEFAULTS_FALLBACK["1-q"]
        return _NOISE_DEFAULTS_FALLBACK["multi-q"]

    def _get_calibration_sequence(
        self, inst: str, num_qubits: int, qargs: tuple[int]
    ) -> list[PulseQobjInstruction]:
        """Return calibration pulse sequence for given instruction (defined by name and num_qubits)
        acting on qargs.
        """

        pulse_library = _pulse_library()
        # Note that the calibration pulses are different for
        # 1q gates vs 2q gates vs measurement instructions.
        if inst == "measure":
            with warnings.catch_warnings():
                # The class PulseQobjInstruction is deprecated
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
                sequence = [
                    PulseQobjInstruction(
                        name="acquire",
                        duration=1792,
                        t0=0,
                        qubits=qargs,
                        memory_slot=qargs,
                    )
                ] + [
                    PulseQobjInstruction(name=pulse_library[1].name, ch=f"m{i}", t0=0)
                    for i in qargs
                ]
            return sequence
        with warnings.catch_warnings():
            # The class PulseQobjInstruction is deprecated
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
            if num_qubits == 1:
                return [
                    PulseQobjInstruction(name="fc", ch=f"u{qargs[0]}", t0=0, phase="-P0"),
                    PulseQobjInstruction(name=pulse_library[0].name, ch=f"d{qargs[0]}", t0=0),
                ]
            return [
                PulseQobjInstruction(name=pulse_library[1].name, ch=f"d{qargs[0]}", t0=0),
                PulseQobjInstruction(name=pulse_library[2].name, ch=f"u{qargs[0]}", t0=0),
                PulseQobjInstruction(name=pulse_library[1].name, ch=f"d{qargs[1]}", t0=0),
                PulseQobjInstruction(name="fc", ch=f"d{qargs[1]}", t0=0, phase=2.1),
            ]

    def _generate_calibration_defaults(self) -> PulseDefaults:
        """Generate pulse calibration defaults as specified with `self._calibrate_instructions`.
        If `self._calibrate_instructions` is True, the pulse schedules will be generated from
        a series of default calibration sequences. If `self._calibrate_instructions` is False,
        the pulse schedules will contain empty calibration sequences, but still be generated and
        added to the target.
        """

        # If self._calibrate_instructions==True, this method
        # will generate default pulse schedules for all gates in self._basis_gates,
        # except for `delay` and `reset`.
        calibration_buffer = self._basis_gates.copy()
        for inst in ["delay", "reset"]:
            calibration_buffer.remove(inst)

        # List of calibration commands (generated from sequences of PulseQobjInstructions)
        # corresponding to each calibrated instruction. Note that the calibration pulses
        # are different for 1q gates vs 2q gates vs measurement instructions.
        cmd_def = []
        for inst in calibration_buffer:
            num_qubits = self._supported_gates[inst].num_qubits
            qarg_set = self._coupling_map if num_qubits > 1 else list(range(self.num_qubits))
            if inst == "measure":
                cmd_def.append(
                    Command(
                        name=inst,
                        qubits=qarg_set,
                        sequence=(
                            self._get_calibration_sequence(inst, num_qubits, qarg_set)
                            if self._calibrate_instructions
                            else []
                        ),
                    )
                )
            else:
                for qarg in qarg_set:
                    qubits = [qarg] if num_qubits == 1 else qarg
                    cmd_def.append(
                        Command(
                            name=inst,
                            qubits=qubits,
                            sequence=(
                                self._get_calibration_sequence(inst, num_qubits, qubits)
                                if self._calibrate_instructions
                                else []
                            ),
                        )
                    )

        qubit_freq_est = np.random.normal(4.8, scale=0.01, size=self.num_qubits).tolist()
        meas_freq_est = np.linspace(6.4, 6.6, self.num_qubits).tolist()
        return PulseDefaults(
            qubit_freq_est=qubit_freq_est,
            meas_freq_est=meas_freq_est,
            buffer=0,
            pulse_library=_pulse_library(),
            cmd_def=cmd_def,
        )

    def _build_generic_target(self):
        """This method generates a :class:`~.Target` instance with
        default qubit, instruction and calibration properties.
        """
        # the qubit properties are sampled from default ranges
        properties = _QUBIT_PROPERTIES
        if not self._noise_info:
            self._target = Target(
                description=f"Generic Target with {self._num_qubits} qubits",
                num_qubits=self._num_qubits,
                dt=properties["dt"],
                qubit_properties=None,
                concurrent_measurements=[list(range(self._num_qubits))],
            )
        else:
            self._target = Target(
                description=f"Generic Target with {self._num_qubits} qubits",
                num_qubits=self._num_qubits,
                dt=properties["dt"],
                qubit_properties=[
                    QubitProperties(
                        t1=self._rng.uniform(properties["t1"][0], properties["t1"][1]),
                        t2=self._rng.uniform(properties["t2"][0], properties["t2"][1]),
                        frequency=self._rng.uniform(
                            properties["frequency"][0], properties["frequency"][1]
                        ),
                    )
                    for _ in range(self._num_qubits)
                ],
                concurrent_measurements=[list(range(self._num_qubits))],
            )

        # Generate instruction schedule map with calibrations to add to target.
        calibration_inst_map = None
        if self._calibrate_instructions is not None:
            if isinstance(self._calibrate_instructions, InstructionScheduleMap):
                calibration_inst_map = self._calibrate_instructions
            else:
                defaults = self._generate_calibration_defaults()
                calibration_inst_map = defaults.instruction_schedule_map

        # Iterate over gates, generate noise params from defaults,
        # and add instructions, noise and calibrations to target.
        for name in self._basis_gates:
            if name not in self._supported_gates:
                raise QiskitError(
                    f"Provided basis gate {name} is not an instruction "
                    f"in the standard qiskit circuit library."
                )
            gate = self._supported_gates[name]
            if self.num_qubits < gate.num_qubits:
                raise QiskitError(
                    f"Provided basis gate {name} needs more qubits than {self.num_qubits}, "
                    f"which is the size of the backend."
                )
            if self._noise_info:
                noise_params = self._get_noise_defaults(name, gate.num_qubits)
                self._add_noisy_instruction_to_target(gate, noise_params, calibration_inst_map)
            else:
                qarg_set = self._coupling_map if gate.num_qubits > 1 else range(self.num_qubits)
                props = {(qarg,) if isinstance(qarg, int) else qarg: None for qarg in qarg_set}
                self._target.add_instruction(gate, properties=props, name=name)

        if self._control_flow:
            self._target.add_instruction(IfElseOp, name="if_else")
            self._target.add_instruction(WhileLoopOp, name="while_loop")
            self._target.add_instruction(ForLoopOp, name="for_loop")
            self._target.add_instruction(SwitchCaseOp, name="switch_case")
            self._target.add_instruction(BreakLoopOp, name="break")
            self._target.add_instruction(ContinueLoopOp, name="continue")

    def _add_noisy_instruction_to_target(
        self,
        instruction: Instruction,
        noise_params: tuple[float, ...] | None,
        calibration_inst_map: InstructionScheduleMap | None,
    ) -> None:
        """Add instruction properties to target for specified instruction.

        Args:
            instruction: Instance of instruction to be added to the target
            noise_params: Error and duration noise values/ranges to
                include in instruction properties.
            calibration_inst_map: Instruction schedule map with calibration defaults
        """
        qarg_set = self._coupling_map if instruction.num_qubits > 1 else range(self.num_qubits)
        props = {}
        for qarg in qarg_set:
            try:
                qargs = tuple(qarg)
            except TypeError:
                qargs = (qarg,)
            duration, error = (
                noise_params
                if len(noise_params) == 2
                else (
                    self._rng.uniform(*noise_params[:2]),
                    self._rng.uniform(*noise_params[2:]),
                )
            )
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=DeprecationWarning)
                # Prevent pulse deprecations from being emitted
                if (
                    calibration_inst_map is not None
                    and instruction.name not in ["reset", "delay"]
                    and qarg in calibration_inst_map.qubits_with_instruction(instruction.name)
                ):
                    # Do NOT call .get method. This parses Qobj immediately.
                    # This operation is computationally expensive and should be bypassed.
                    calibration_entry = calibration_inst_map._get_calibration_entry(
                        instruction.name, qargs
                    )
                else:
                    calibration_entry = None
                if duration is not None and len(noise_params) > 2:
                    # Ensure exact conversion of duration from seconds to dt
                    dt = _QUBIT_PROPERTIES["dt"]
                    rounded_duration = round(duration / dt) * dt
                    # Clamp rounded duration to be between min and max values
                    duration = max(noise_params[0], min(rounded_duration, noise_params[1]))
                props.update({qargs: InstructionProperties(duration, error, calibration_entry)})
        self._target.add_instruction(instruction, props)

        # The "measure" instruction calibrations need to be added qubit by qubit, once the
        # instruction has been added to the target.
        if calibration_inst_map is not None and instruction.name == "measure":
            for qarg in calibration_inst_map.qubits_with_instruction(instruction.name):
                try:
                    qargs = tuple(qarg)
                except TypeError:
                    qargs = (qarg,)
                # Do NOT call .get method. This parses Qobj immediately.
                # This operation is computationally expensive and should be bypassed.
                calibration_entry = calibration_inst_map._get_calibration_entry(
                    instruction.name, qargs
                )
                for qubit in qargs:
                    if qubit < self.num_qubits:
                        self._target[instruction.name][(qubit,)].calibration = calibration_entry

    def run(self, run_input, **options):
        """Run on the backend using a simulator.

        This method runs circuit jobs (an individual or a list of :class:`~.QuantumCircuit`
        ) and pulse jobs (an individual or a list of :class:`~.Schedule` or
        :class:`~.ScheduleBlock`) using :class:`~.BasicSimulator` or Aer simulator and returns a
        :class:`~qiskit.providers.Job` object.

        If qiskit-aer is installed, jobs will be run using the ``AerSimulator`` with
        noise model of the backend. Otherwise, jobs will be run using the
        ``BasicSimulator`` simulator without noise.

        Noisy simulations of pulse jobs are not yet supported in :class:`~.GenericBackendV2`.

        Args:
            run_input (QuantumCircuit or Schedule or ScheduleBlock or list): An
                individual or a list of
                :class:`~qiskit.circuit.QuantumCircuit`,
                :class:`~qiskit.pulse.ScheduleBlock`, or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object, then the expectation is that the value
                specified will be used instead of what's set in the options
                object.

        Returns:
            Job: The job object for the run

        Raises:
            QiskitError: If a pulse job is supplied and qiskit_aer is not installed.
        """
        circuits = run_input
        pulse_job = None
        if isinstance(circuits, (pulse.Schedule, pulse.ScheduleBlock)):
            pulse_job = True
        elif isinstance(circuits, QuantumCircuit):
            pulse_job = False
        elif isinstance(circuits, list):
            if circuits:
                if all(isinstance(x, (pulse.Schedule, pulse.ScheduleBlock)) for x in circuits):
                    pulse_job = True
                elif all(isinstance(x, QuantumCircuit) for x in circuits):
                    pulse_job = False
        if pulse_job is None:  # submitted job is invalid
            raise QiskitError(
                f"Invalid input object {circuits}, must be either a "
                "QuantumCircuit, Schedule, or a list of either"
            )
        if pulse_job:  # pulse job
            raise QiskitError("Pulse simulation is currently not supported for V2 backends.")
        # circuit job
        if not _optionals.HAS_AER:
            warnings.warn("Aer not found using BasicSimulator and no noise", RuntimeWarning)
        if self._sim is None:
            self._setup_sim()
        self._sim._options = self._options
        job = self._sim.run(circuits, **options)
        return job

    def _setup_sim(self) -> None:
        if _optionals.HAS_AER:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel

            self._sim = AerSimulator()
            noise_model = NoiseModel.from_backend(self)
            self._sim.set_options(noise_model=noise_model)
            # Update backend default too to avoid overwriting
            # it when run() is called
            self.set_options(noise_model=noise_model)
        else:
            self._sim = BasicSimulator()

    @classmethod
    def _default_options(cls) -> Options:
        if _optionals.HAS_AER:
            from qiskit_aer import AerSimulator

            return AerSimulator._default_options()
        else:
            return BasicSimulator._default_options()

    @deprecate_pulse_dependency
    def drive_channel(self, qubit: int):
        drive_channels_map = getattr(self, "channels_map", {}).get("drive", {})
        qubits = (qubit,)
        if qubits in drive_channels_map:
            return drive_channels_map[qubits][0]
        return None

    @deprecate_pulse_dependency
    def measure_channel(self, qubit: int):
        measure_channels_map = getattr(self, "channels_map", {}).get("measure", {})
        qubits = (qubit,)
        if qubits in measure_channels_map:
            return measure_channels_map[qubits][0]
        return None

    @deprecate_pulse_dependency
    def acquire_channel(self, qubit: int):
        acquire_channels_map = getattr(self, "channels_map", {}).get("acquire", {})
        qubits = (qubit,)
        if qubits in acquire_channels_map:
            return acquire_channels_map[qubits][0]
        return None

    @deprecate_pulse_dependency
    def control_channel(self, qubits: Iterable[int]):
        control_channels_map = getattr(self, "channels_map", {}).get("control", {})
        qubits = tuple(qubits)
        if qubits in control_channels_map:
            return control_channels_map[qubits]
        return []

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


"""Model and schema for pulse defaults."""
import warnings
from typing import Any, Dict, List

from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap, PulseQobjDef
from qiskit.qobj import PulseLibraryItem, PulseQobjInstruction
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.utils.deprecate_pulse import deprecate_pulse_dependency


class MeasurementKernel:
    """Class representing a Measurement Kernel."""

    def __init__(self, name, params):
        """Initialize a MeasurementKernel object

        Args:
            name (str): The name of the measurement kernel
            params: The parameters of the measurement kernel
        """
        self.name = name
        self.params = params

    def to_dict(self):
        """Return a dictionary format representation of the MeasurementKernel.

        Returns:
            dict: The dictionary form of the MeasurementKernel.
        """
        return {"name": self.name, "params": self.params}

    @classmethod
    def from_dict(cls, data):
        """Create a new MeasurementKernel object from a dictionary.

        Args:
            data (dict): A dictionary representing the MeasurementKernel
                         to create. It will be in the same format as output by
                         :meth:`to_dict`.

        Returns:
            MeasurementKernel: The MeasurementKernel from the input dictionary.
        """
        return cls(**data)


class Discriminator:
    """Class representing a Discriminator."""

    def __init__(self, name, params):
        """Initialize a Discriminator object

        Args:
            name (str): The name of the discriminator
            params: The parameters of the discriminator
        """
        self.name = name
        self.params = params

    def to_dict(self):
        """Return a dictionary format representation of the Discriminator.

        Returns:
            dict: The dictionary form of the Discriminator.
        """
        return {"name": self.name, "params": self.params}

    @classmethod
    def from_dict(cls, data):
        """Create a new Discriminator object from a dictionary.

        Args:
            data (dict): A dictionary representing the Discriminator
                         to create. It will be in the same format as output by
                         :meth:`to_dict`.

        Returns:
            Discriminator: The Discriminator from the input dictionary.
        """
        return cls(**data)


class Command:
    """Class representing a Command.

    Attributes:
        name: Pulse command name.
    """

    _data = {}

    def __init__(self, name: str, qubits=None, sequence=None, **kwargs):
        """Initialize a Command object

        Args:
            name (str): The name of the command
            qubits: The qubits for the command
            sequence (PulseQobjInstruction): The sequence for the Command
            kwargs: Optional additional fields
        """
        self._data = {}
        self.name = name
        if qubits is not None:
            self.qubits = qubits
        if sequence is not None:
            self.sequence = sequence
        self._data.update(kwargs)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as ex:
            raise AttributeError(f"Attribute {name} is not defined") from ex

    def to_dict(self):
        """Return a dictionary format representation of the Command.

        Returns:
            dict: The dictionary form of the Command.
        """
        out_dict = {"name": self.name}
        if hasattr(self, "qubits"):
            out_dict["qubits"] = self.qubits
        if hasattr(self, "sequence"):
            out_dict["sequence"] = [x.to_dict() for x in self.sequence]
        out_dict.update(self._data)
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new Command object from a dictionary.

        Args:
            data (dict): A dictionary representing the ``Command``
                         to create. It will be in the same format as output by
                         :meth:`to_dict`.

        Returns:
            Command: The ``Command`` from the input dictionary.
        """
        # Pulse command data is nested dictionary.
        # To avoid deepcopy and avoid mutating the source object, create new dict here.
        in_data = {}
        for key, value in data.items():
            if key == "sequence":
                in_data[key] = list(map(PulseQobjInstruction.from_dict, value))
            else:
                in_data[key] = value
        return cls(**in_data)


class PulseDefaults:
    """Description of default settings for Pulse systems. These are instructions or settings that
    may be good starting points for the Pulse user. The user may modify these defaults for custom
    scheduling.
    """

    _data = {}

    @deprecate_pulse_dependency
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

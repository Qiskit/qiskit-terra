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

# pylint: disable=invalid-name,redefined-builtin
# pylint: disable=super-init-not-called

"""Module providing definitions of Pulse Qobj classes."""

import copy
import pprint
from typing import Union, List

import numpy as np
from qiskit.qobj.common import QobjDictField
from qiskit.qobj.common import QobjHeader
from qiskit.qobj.common import QobjExperimentHeader


class QobjMeasurementOption:
    """An individual measurement option."""

    def __init__(self, name, params=None):
        """Instantiate a new QobjMeasurementOption object.

        Args:
            name (str): The name of the measurement option
            params (list): The parameters of the measurement option.
        """
        self.name = name
        if params is not None:
            self.params = params

    def to_dict(self):
        """Return a dict format representation of the QobjMeasurementOption.

        Returns:
            dict: The dictionary form of the QasmMeasurementOption.
        """
        out_dict = {"name": self.name}
        if hasattr(self, "params"):
            out_dict["params"] = self.params
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new QobjMeasurementOption object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            QobjMeasurementOption: The object from the input dictionary.
        """
        name = data.pop("name")
        return cls(name, **data)

    def __eq__(self, other):
        if isinstance(other, QobjMeasurementOption):
            if self.to_dict() == other.to_dict():
                return True
        return False


class PulseQobjInstruction:
    """A class representing a single instruction in an PulseQobj Experiment."""

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
        type=None,
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
        out = "Instruction: %s\n" % self.name
        out += "\t\tt0: %s\n" % self.t0
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


class PulseQobjConfig(QobjDictField):
    """A configuration for a Pulse Qobj."""

    def __init__(
        self,
        meas_level,
        meas_return,
        pulse_library,
        qubit_lo_freq,
        meas_lo_freq,
        memory_slot_size=None,
        rep_time=None,
        rep_delay=None,
        shots=None,
        seed_simulator=None,
        memory_slots=None,
        **kwargs,
    ):
        """Instantiate a PulseQobjConfig object.

        Args:
            meas_level (int): The measurement level to use.
            meas_return (int): The level of measurement information to return.
            pulse_library (list): A list of :class:`PulseLibraryItem` objects
                which define the set of primitive pulses
            qubit_lo_freq (list): List of frequencies (as floats) for the qubit
                driver LO's in GHz.
            meas_lo_freq (list): List of frequencies (as floats) for the'
                measurement driver LO's in GHz.
            memory_slot_size (int): Size of each memory slot if the output is
                Level 0.
            rep_time (int): Time per program execution in sec. Must be from the list provided
                by the backend (``backend.configuration().rep_times``). Defaults to the first entry
                in ``backend.configuration().rep_times``.
            rep_delay (float): Delay between programs in sec. Only supported on certain
                backends (``backend.configuration().dynamic_reprate_enabled`` ). If supported,
                ``rep_delay`` will be used instead of ``rep_time`` and must be from the range
                supplied by the backend (``backend.configuration().rep_delay_range``). Default is
                ``backend.configuration().default_rep_delay``.
            shots (int): The number of shots
            seed_simulator (int): the seed to use in the simulator
            memory_slots (list): The number of memory slots on the device
            kwargs: Additional free form key value fields to add to the
                configuration
        """
        self.meas_level = meas_level
        self.meas_return = meas_return
        self.pulse_library = pulse_library
        self.qubit_lo_freq = qubit_lo_freq
        self.meas_lo_freq = meas_lo_freq
        if memory_slot_size is not None:
            self.memory_slot_size = memory_slot_size
        if rep_time is not None:
            self.rep_time = rep_time
        if rep_delay is not None:
            self.rep_delay = rep_delay
        if shots is not None:
            self.shots = int(shots)

        if seed_simulator is not None:
            self.seed_simulator = int(seed_simulator)

        if memory_slots is not None:
            self.memory_slots = int(memory_slots)

        if kwargs:
            self.__dict__.update(kwargs)

    def to_dict(self):
        """Return a dictionary format representation of the Pulse Qobj config.

        Returns:
            dict: The dictionary form of the PulseQobjConfig.
        """
        out_dict = copy.copy(self.__dict__)
        if hasattr(self, "pulse_library"):
            out_dict["pulse_library"] = [x.to_dict() for x in self.pulse_library]

        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new PulseQobjConfig object from a dictionary.

        Args:
            data (dict): A dictionary for the config

        Returns:
            PulseQobjConfig: The object from the input dictionary.
        """
        if "pulse_library" in data:
            pulse_lib = data.pop("pulse_library")
            pulse_lib_obj = [PulseLibraryItem.from_dict(x) for x in pulse_lib]
            data["pulse_library"] = pulse_lib_obj
        return cls(**data)


class PulseQobjExperiment:
    """A Pulse Qobj Experiment.

    Each instance of this class is used to represent an individual Pulse
    experiment as part of a larger Pulse Qobj.
    """

    def __init__(self, instructions, config=None, header=None):
        """Instantiate a PulseQobjExperiment.

        Args:
            config (PulseQobjExperimentConfig): A config object for the experiment
            header (PulseQobjExperimentHeader): A header object for the experiment
            instructions (list): A list of :class:`PulseQobjInstruction` objects
        """
        if config is not None:
            self.config = config
        if header is not None:
            self.header = header
        self.instructions = instructions

    def to_dict(self):
        """Return a dictionary format representation of the Experiment.

        Returns:
            dict: The dictionary form of the PulseQobjExperiment.
        """
        out_dict = {"instructions": [x.to_dict() for x in self.instructions]}
        if hasattr(self, "config"):
            out_dict["config"] = self.config.to_dict()
        if hasattr(self, "header"):
            out_dict["header"] = self.header.to_dict()
        return out_dict

    def __repr__(self):
        instructions_str = [repr(x) for x in self.instructions]
        instructions_repr = "[" + ", ".join(instructions_str) + "]"
        out = "PulseQobjExperiment("
        out += instructions_repr
        if hasattr(self, "config") or hasattr(self, "header"):
            out += ", "
        if hasattr(self, "config"):
            out += "config=" + str(repr(self.config)) + ", "
        if hasattr(self, "header"):
            out += "header=" + str(repr(self.header)) + ", "
        out += ")"
        return out

    def __str__(self):
        out = "\nPulse Experiment:\n"
        if hasattr(self, "config"):
            config = pprint.pformat(self.config.to_dict())
        else:
            config = "{}"
        if hasattr(self, "header"):
            header = pprint.pformat(self.header.to_dict() or {})
        else:
            header = "{}"
        out += "Header:\n%s\n" % header
        out += "Config:\n%s\n\n" % config
        for instruction in self.instructions:
            out += "\t%s\n" % instruction
        return out

    @classmethod
    def from_dict(cls, data):
        """Create a new PulseQobjExperiment object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            PulseQobjExperiment: The object from the input dictionary.
        """
        config = None
        if "config" in data:
            config = PulseQobjExperimentConfig.from_dict(data.pop("config"))
        header = None
        if "header" in data:
            header = QobjExperimentHeader.from_dict(data.pop("header"))
        instructions = None
        if "instructions" in data:
            instructions = [
                PulseQobjInstruction.from_dict(inst) for inst in data.pop("instructions")
            ]
        return cls(instructions, config, header)

    def __eq__(self, other):
        if isinstance(other, PulseQobjExperiment):
            if self.to_dict() == other.to_dict():
                return True
        return False


class PulseQobjExperimentConfig(QobjDictField):
    """A config for a single Pulse experiment in the qobj."""

    def __init__(self, qubit_lo_freq=None, meas_lo_freq=None, **kwargs):
        """Instantiate a PulseQobjExperimentConfig object.

        Args:
            qubit_lo_freq (List[float]): List of qubit LO frequencies in GHz.
            meas_lo_freq (List[float]): List of meas readout LO frequencies in GHz.
            kwargs: Additional free form key value fields to add to the configuration
        """
        if qubit_lo_freq is not None:
            self.qubit_lo_freq = qubit_lo_freq
        if meas_lo_freq is not None:
            self.meas_lo_freq = meas_lo_freq
        if kwargs:
            self.__dict__.update(kwargs)


class PulseLibraryItem:
    """An item in a pulse library."""

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


class PulseQobj:
    """A Pulse Qobj."""

    def __init__(self, qobj_id, config, experiments, header=None):
        """Instantiate a new Pulse Qobj Object.

        Each Pulse Qobj object is used to represent a single payload that will
        be passed to a Qiskit provider. It mirrors the Qobj the published
        `Qobj specification <https://arxiv.org/abs/1809.03452>`_ for Pulse
        experiments.

        Args:
            qobj_id (str): An identifier for the qobj
            config (PulseQobjConfig): A config for the entire run
            header (QobjHeader): A header for the entire run
            experiments (list): A list of lists of :class:`PulseQobjExperiment`
                objects representing an experiment
        """
        self.qobj_id = qobj_id
        self.config = config
        self.header = header or QobjHeader()
        self.experiments = experiments
        self.type = "PULSE"
        self.schema_version = "1.2.0"

    def __repr__(self):
        experiments_str = [repr(x) for x in self.experiments]
        experiments_repr = "[" + ", ".join(experiments_str) + "]"
        out = "PulseQobj(qobj_id='{}', config={}, experiments={}, header={})".format(
            self.qobj_id,
            repr(self.config),
            experiments_repr,
            repr(self.header),
        )
        return out

    def __str__(self):
        out = "Pulse Qobj: %s:\n" % self.qobj_id
        config = pprint.pformat(self.config.to_dict())
        out += "Config: %s\n" % str(config)
        header = pprint.pformat(self.header.to_dict())
        out += "Header: %s\n" % str(header)
        out += "Experiments:\n"
        for experiment in self.experiments:
            out += "%s" % str(experiment)
        return out

    def to_dict(self):
        """Return a dictionary format representation of the Pulse Qobj.

        Note this dict is not in the json wire format expected by IBMQ and qobj
        specification because complex numbers are still of type complex. Also
        this may contain native numpy arrays. When serializing this output
        for use with IBMQ you can leverage a json encoder that converts these
        as expected. For example:

        .. code-block::

            import json
            import numpy as np

            class QobjEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, complex):
                        return (obj.real, obj.imag)
                    return json.JSONEncoder.default(self, obj)

            json.dumps(qobj.to_dict(), cls=QobjEncoder)

        Returns:
            dict: A dictionary representation of the PulseQobj object
        """
        out_dict = {
            "qobj_id": self.qobj_id,
            "header": self.header.to_dict(),
            "config": self.config.to_dict(),
            "schema_version": self.schema_version,
            "type": self.type,
            "experiments": [x.to_dict() for x in self.experiments],
        }
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new PulseQobj object from a dictionary.

        Args:
            data (dict): A dictionary representing the PulseQobj to create. It
                will be in the same format as output by :func:`to_dict`.

        Returns:
            PulseQobj: The PulseQobj from the input dictionary.
        """
        config = None
        if "config" in data:
            config = PulseQobjConfig.from_dict(data["config"])
        experiments = None
        if "experiments" in data:
            experiments = [PulseQobjExperiment.from_dict(exp) for exp in data["experiments"]]
        header = None
        if "header" in data:
            header = QobjHeader.from_dict(data["header"])

        return cls(
            qobj_id=data.get("qobj_id"), config=config, experiments=experiments, header=header
        )

    def __eq__(self, other):
        if isinstance(other, PulseQobj):
            if self.to_dict() == other.to_dict():
                return True
        return False

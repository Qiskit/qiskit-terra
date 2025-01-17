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

"""Module providing definitions of OpenQASM 2 Qobj classes."""

import copy
import pprint
from types import SimpleNamespace
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.qobj.pulse_qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.qobj.common import QobjDictField, QobjHeader
from qiskit.utils import deprecate_func


class QasmQobjInstruction:
    """A class representing a single instruction in an QasmQobj Experiment."""

    @deprecate_func(
        since="1.2",
        removal_timeline="in the 2.0 release",
        additional_msg="The `Qobj` class and related functionality are part of the deprecated "
        "`BackendV1` workflow,  and no longer necessary for `BackendV2`. If a user "
        "workflow requires `Qobj` it likely relies on deprecated functionality and "
        "should be updated to use `BackendV2`.",
    )
    def __init__(
        self,
        name,
        params=None,
        qubits=None,
        register=None,
        memory=None,
        condition=None,
        conditional=None,
        label=None,
        mask=None,
        relation=None,
        val=None,
        snapshot_type=None,
    ):
        """Instantiate a new QasmQobjInstruction object.

        Args:
            name (str): The name of the instruction
            params (list): The list of parameters for the gate
            qubits (list): A list of ``int`` representing the qubits the
                instruction operates on
            register (list): If a ``measure`` instruction this is a list
                of ``int`` containing the list of register slots in which to
                store the measurement results (must be the same length as
                qubits). If a ``bfunc`` instruction this is a single ``int``
                of the register slot in which to store the result.
            memory (list): If a ``measure`` instruction this is a list
                of ``int`` containing the list of memory slots to store the
                measurement results in (must be the same length as qubits).
                If a ``bfunc`` instruction this is a single ``int`` of the
                memory slot to store the boolean function result in.
            condition (tuple): A tuple of the form ``(int, int)`` where the
                first ``int`` is the control register and the second ``int`` is
                the control value if the gate has a condition.
            conditional (int):  The register index of the condition
            label (str): An optional label assigned to the instruction
            mask (int): For a ``bfunc`` instruction the hex value which is
                applied as an ``AND`` to the register bits.
            relation (str): Relational  operator  for  comparing  the  masked
                register to the ``val`` kwarg. Can be either ``==`` (equals) or
                ``!=`` (not equals).
            val (int): Value to which to compare the masked register. In other
                words, the output of the function is ``(register AND mask)``
            snapshot_type (str): For snapshot instructions the type of snapshot
                to use
        """
        self.name = name
        if params is not None:
            self.params = params
        if qubits is not None:
            self.qubits = qubits
        if register is not None:
            self.register = register
        if memory is not None:
            self.memory = memory
        if condition is not None:
            self._condition = condition
        if conditional is not None:
            self.conditional = conditional
        if label is not None:
            self.label = label
        if mask is not None:
            self.mask = mask
        if relation is not None:
            self.relation = relation
        if val is not None:
            self.val = val
        if snapshot_type is not None:
            self.snapshot_type = snapshot_type

    def to_dict(self):
        """Return a dictionary format representation of the Instruction.

        Returns:
            dict: The dictionary form of the QasmQobjInstruction.
        """
        out_dict = {"name": self.name}
        for attr in [
            "params",
            "qubits",
            "register",
            "memory",
            "_condition",
            "conditional",
            "label",
            "mask",
            "relation",
            "val",
            "snapshot_type",
        ]:
            if hasattr(self, attr):
                # TODO: Remove the param type conversion when Aer understands
                # ParameterExpression type
                if attr == "params":
                    params = []
                    for param in list(getattr(self, attr)):
                        if isinstance(param, ParameterExpression):
                            params.append(float(param))
                        else:
                            params.append(param)
                    out_dict[attr] = params
                else:
                    out_dict[attr] = getattr(self, attr)

        return out_dict

    def __repr__(self):
        out = f"QasmQobjInstruction(name='{self.name}'"
        for attr in [
            "params",
            "qubits",
            "register",
            "memory",
            "_condition",
            "conditional",
            "label",
            "mask",
            "relation",
            "val",
            "snapshot_type",
        ]:
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
        for attr in [
            "params",
            "qubits",
            "register",
            "memory",
            "_condition",
            "conditional",
            "label",
            "mask",
            "relation",
            "val",
            "snapshot_type",
        ]:
            if hasattr(self, attr):
                out += f"\t\t{attr}: {getattr(self, attr)}\n"
        return out

    @classmethod
    def from_dict(cls, data):
        """Create a new QasmQobjInstruction object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            QasmQobjInstruction: The object from the input dictionary.
        """
        name = data.pop("name")
        return cls(name, **data)

    def __eq__(self, other):
        if isinstance(other, QasmQobjInstruction):
            if self.to_dict() == other.to_dict():
                return True
        return False


class QasmQobjExperiment:
    """An OpenQASM 2 Qobj Experiment.

    Each instance of this class is used to represent an OpenQASM 2 experiment as
    part of a larger OpenQASM 2 qobj.
    """

    @deprecate_func(
        since="1.2",
        removal_timeline="in the 2.0 release",
        additional_msg="The `Qobj` class and related functionality are part of the deprecated "
        "`BackendV1` workflow,  and no longer necessary for `BackendV2`. If a user "
        "workflow requires `Qobj` it likely relies on deprecated functionality and "
        "should be updated to use `BackendV2`.",
    )
    def __init__(self, config=None, header=None, instructions=None):
        """Instantiate a QasmQobjExperiment.

        Args:
            config (QasmQobjExperimentConfig): A config object for the experiment
            header (QasmQobjExperimentHeader): A header object for the experiment
            instructions (list): A list of :class:`QasmQobjInstruction` objects
        """
        self.config = config or QasmQobjExperimentConfig()
        self.header = header or QasmQobjExperimentHeader()
        self.instructions = instructions or []

    def __repr__(self):
        instructions_str = [repr(x) for x in self.instructions]
        instructions_repr = "[" + ", ".join(instructions_str) + "]"
        return (
            f"QasmQobjExperiment(config={repr(self.config)}, header={repr(self.header)},"
            f" instructions={instructions_repr})"
        )

    def __str__(self):
        out = "\nOpenQASM2 Experiment:\n"
        config = pprint.pformat(self.config.to_dict())
        header = pprint.pformat(self.header.to_dict())
        out += f"Header:\n{header}\n"
        out += f"Config:\n{config}\n\n"
        for instruction in self.instructions:
            out += f"\t{instruction}\n"
        return out

    def to_dict(self):
        """Return a dictionary format representation of the Experiment.

        Returns:
            dict: The dictionary form of the QasmQObjExperiment.
        """
        out_dict = {
            "config": self.config.to_dict(),
            "header": self.header.to_dict(),
            "instructions": [x.to_dict() for x in self.instructions],
        }
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new QasmQobjExperiment object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            QasmQobjExperiment: The object from the input dictionary.
        """
        config = None
        if "config" in data:
            config = QasmQobjExperimentConfig.from_dict(data.pop("config"))
        header = None
        if "header" in data:
            header = QasmQobjExperimentHeader.from_dict(data.pop("header"))
        instructions = None
        if "instructions" in data:
            instructions = [
                QasmQobjInstruction.from_dict(inst) for inst in data.pop("instructions")
            ]
        return cls(config, header, instructions)

    def __eq__(self, other):
        if isinstance(other, QasmQobjExperiment):
            if self.to_dict() == other.to_dict():
                return True
        return False


class QasmQobjConfig(SimpleNamespace):
    """A configuration for an OpenQASM 2 Qobj."""

    @deprecate_func(
        since="1.2",
        removal_timeline="in the 2.0 release",
        additional_msg="The `Qobj` class and related functionality are part of the deprecated "
        "`BackendV1` workflow,  and no longer necessary for `BackendV2`. If a user "
        "workflow requires `Qobj` it likely relies on deprecated functionality and "
        "should be updated to use `BackendV2`.",
    )
    def __init__(
        self,
        shots=None,
        seed_simulator=None,
        memory=None,
        parameter_binds=None,
        meas_level=None,
        meas_return=None,
        memory_slots=None,
        n_qubits=None,
        pulse_library=None,
        calibrations=None,
        rep_delay=None,
        qubit_lo_freq=None,
        meas_lo_freq=None,
        **kwargs,
    ):
        """Model for RunConfig.

        Args:
            shots (int): the number of shots.
            seed_simulator (int): the seed to use in the simulator
            memory (bool): whether to request memory from backend (per-shot readouts)
            parameter_binds (list[dict]): List of parameter bindings
            meas_level (int): Measurement level 0, 1, or 2
            meas_return (str): For measurement level < 2, whether single or avg shots are returned
            memory_slots (int): The number of memory slots on the device
            n_qubits (int): The number of qubits on the device
            pulse_library (list): List of :class:`PulseLibraryItem`.
            calibrations (QasmExperimentCalibrations): Information required for Pulse gates.
            rep_delay (float): Delay between programs in sec. Only supported on certain
                backends (``backend.configuration().dynamic_reprate_enabled`` ). Must be from the
                range supplied by the backend (``backend.configuration().rep_delay_range``). Default
                is ``backend.configuration().default_rep_delay``.
            qubit_lo_freq (list): List of frequencies (as floats) for the qubit driver LO's in GHz.
            meas_lo_freq (list): List of frequencies (as floats) for the measurement driver LO's in
                GHz.
            kwargs: Additional free form key value fields to add to the
                configuration.
        """
        if shots is not None:
            self.shots = int(shots)

        if seed_simulator is not None:
            self.seed_simulator = int(seed_simulator)

        if memory is not None:
            self.memory = bool(memory)

        if parameter_binds is not None:
            self.parameter_binds = parameter_binds

        if meas_level is not None:
            self.meas_level = meas_level

        if meas_return is not None:
            self.meas_return = meas_return

        if memory_slots is not None:
            self.memory_slots = memory_slots

        if n_qubits is not None:
            self.n_qubits = n_qubits

        if pulse_library is not None:
            self.pulse_library = pulse_library

        if calibrations is not None:
            self.calibrations = calibrations

        if rep_delay is not None:
            self.rep_delay = rep_delay

        if qubit_lo_freq is not None:
            self.qubit_lo_freq = qubit_lo_freq

        if meas_lo_freq is not None:
            self.meas_lo_freq = meas_lo_freq

        if kwargs:
            self.__dict__.update(kwargs)

    def to_dict(self):
        """Return a dictionary format representation of the OpenQASM 2 Qobj config.

        Returns:
            dict: The dictionary form of the QasmQobjConfig.
        """
        out_dict = copy.copy(self.__dict__)
        if hasattr(self, "pulse_library"):
            out_dict["pulse_library"] = [x.to_dict() for x in self.pulse_library]

        if hasattr(self, "calibrations"):
            out_dict["calibrations"] = self.calibrations.to_dict()

        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new QasmQobjConfig object from a dictionary.

        Args:
            data (dict): A dictionary for the config

        Returns:
            QasmQobjConfig: The object from the input dictionary.
        """
        if "pulse_library" in data:
            pulse_lib = data.pop("pulse_library")
            pulse_lib_obj = [PulseLibraryItem.from_dict(x) for x in pulse_lib]
            data["pulse_library"] = pulse_lib_obj

        if "calibrations" in data:
            calibrations = data.pop("calibrations")
            data["calibrations"] = QasmExperimentCalibrations.from_dict(calibrations)

        return cls(**data)

    def __eq__(self, other):
        if isinstance(other, QasmQobjConfig):
            if self.to_dict() == other.to_dict():
                return True
        return False


class QasmQobjExperimentHeader(QobjDictField):
    """A header for a single OpenQASM 2 experiment in the qobj."""

    pass


class QasmQobjExperimentConfig(QobjDictField):
    """Configuration for a single OpenQASM 2 experiment in the qobj."""

    @deprecate_func(
        since="1.2",
        removal_timeline="in the 2.0 release",
        additional_msg="The `Qobj` class and related functionality are part of the deprecated "
        "`BackendV1` workflow,  and no longer necessary for `BackendV2`. If a user "
        "workflow requires `Qobj` it likely relies on deprecated functionality and "
        "should be updated to use `BackendV2`.",
    )
    def __init__(self, calibrations=None, qubit_lo_freq=None, meas_lo_freq=None, **kwargs):
        """
        Args:
            calibrations (QasmExperimentCalibrations): Information required for Pulse gates.
            qubit_lo_freq (List[float]): List of qubit LO frequencies in GHz.
            meas_lo_freq (List[float]): List of meas readout LO frequencies in GHz.
            kwargs: Additional free form key value fields to add to the configuration
        """
        if calibrations:
            self.calibrations = calibrations
        if qubit_lo_freq is not None:
            self.qubit_lo_freq = qubit_lo_freq
        if meas_lo_freq is not None:
            self.meas_lo_freq = meas_lo_freq

        super().__init__(**kwargs)

    def to_dict(self):
        out_dict = copy.copy(self.__dict__)
        if hasattr(self, "calibrations"):
            out_dict["calibrations"] = self.calibrations.to_dict()
        return out_dict

    @classmethod
    def from_dict(cls, data):
        if "calibrations" in data:
            calibrations = data.pop("calibrations")
            data["calibrations"] = QasmExperimentCalibrations.from_dict(calibrations)
        return cls(**data)


class QasmExperimentCalibrations:
    """A container for any calibrations data. The gates attribute contains a list of
    GateCalibrations.
    """

    @deprecate_func(
        since="1.2",
        removal_timeline="in the 2.0 release",
        additional_msg="The `Qobj` class and related functionality are part of the deprecated "
        "`BackendV1` workflow,  and no longer necessary for `BackendV2`. If a user "
        "workflow requires `Qobj` it likely relies on deprecated functionality and "
        "should be updated to use `BackendV2`.",
    )
    def __init__(self, gates):
        """
        Initialize a container for calibrations.

        Args:
            gates (list(GateCalibration))
        """
        self.gates = gates

    def to_dict(self):
        """Return a dictionary format representation of the calibrations.

        Returns:
            dict: The dictionary form of the GateCalibration.

        """
        out_dict = copy.copy(self.__dict__)
        out_dict["gates"] = [x.to_dict() for x in self.gates]
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new GateCalibration object from a dictionary.

        Args:
            data (dict): A dictionary representing the QasmExperimentCalibrations to
                         create. It will be in the same format as output by :func:`to_dict`.

        Returns:
            QasmExperimentCalibrations: The QasmExperimentCalibrations from the input dictionary.
        """
        gates = data.pop("gates")
        data["gates"] = [GateCalibration.from_dict(x) for x in gates]
        return cls(**data)


class GateCalibration:
    """Each calibration specifies a unique gate by name, qubits and params, and
    contains the Pulse instructions to implement it."""

    @deprecate_func(
        since="1.2",
        removal_timeline="in the 2.0 release",
        additional_msg="The `Qobj` class and related functionality are part of the deprecated "
        "`BackendV1` workflow,  and no longer necessary for `BackendV2`. If a user "
        "workflow requires `Qobj` it likely relies on deprecated functionality and "
        "should be updated to use `BackendV2`.",
    )
    def __init__(self, name, qubits, params, instructions):
        """
        Initialize a single gate calibration. Instructions may reference waveforms which should be
        made available in the pulse_library.

        Args:
            name (str): Gate name.
            qubits (list(int)): Qubits the gate applies to.
            params (list(complex)): Gate parameter values, if any.
            instructions (list(PulseQobjInstruction)): The gate implementation.
        """
        self.name = name
        self.qubits = qubits
        self.params = params
        self.instructions = instructions

    def __hash__(self):
        return hash(
            (
                self.name,
                tuple(self.qubits),
                tuple(self.params),
                tuple(str(inst) for inst in self.instructions),
            )
        )

    def to_dict(self):
        """Return a dictionary format representation of the Gate Calibration.

        Returns:
            dict: The dictionary form of the GateCalibration.
        """
        out_dict = copy.copy(self.__dict__)
        out_dict["instructions"] = [x.to_dict() for x in self.instructions]
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new GateCalibration object from a dictionary.

        Args:
            data (dict): A dictionary representing the GateCalibration to create. It
                will be in the same format as output by :func:`to_dict`.

        Returns:
            GateCalibration: The GateCalibration from the input dictionary.
        """
        instructions = data.pop("instructions")
        data["instructions"] = [PulseQobjInstruction.from_dict(x) for x in instructions]
        return cls(**data)


class QasmQobj:
    """An OpenQASM 2 Qobj."""

    @deprecate_func(
        since="1.2",
        removal_timeline="in the 2.0 release",
        additional_msg="The `Qobj` class and related functionality are part of the deprecated "
        "`BackendV1` workflow,  and no longer necessary for `BackendV2`. If a user "
        "workflow requires `Qobj` it likely relies on deprecated functionality and "
        "should be updated to use `BackendV2`.",
    )
    def __init__(self, qobj_id=None, config=None, experiments=None, header=None):
        """Instantiate a new OpenQASM 2 Qobj Object.

        Each OpenQASM 2 Qobj object is used to represent a single payload that will
        be passed to a Qiskit provider. It mirrors the Qobj the published
        `Qobj specification <https://arxiv.org/abs/1809.03452>`_ for OpenQASM
        experiments.

        Args:
            qobj_id (str): An identifier for the qobj
            config (QasmQobjRunConfig): A config for the entire run
            header (QobjHeader): A header for the entire run
            experiments (list): A list of lists of :class:`QasmQobjExperiment`
                objects representing an experiment
        """
        self.header = header or QobjHeader()
        self.config = config or QasmQobjConfig()
        self.experiments = experiments or []
        self.qobj_id = qobj_id
        self.type = "QASM"
        self.schema_version = "1.3.0"

    def __repr__(self):
        experiments_str = [repr(x) for x in self.experiments]
        experiments_repr = "[" + ", ".join(experiments_str) + "]"
        return (
            f"QasmQobj(qobj_id='{self.qobj_id}', config={repr(self.config)},"
            f" experiments={experiments_repr}, header={repr(self.header)})"
        )

    def __str__(self):
        out = f"QASM Qobj: {self.qobj_id}:\n"
        config = pprint.pformat(self.config.to_dict())
        out += f"Config: {str(config)}\n"
        header = pprint.pformat(self.header.to_dict())
        out += f"Header: {str(header)}\n"
        out += "Experiments:\n"
        for experiment in self.experiments:
            out += str(experiment)
        return out

    def to_dict(self):
        """Return a dictionary format representation of the OpenQASM 2 Qobj.

        Note this dict is not in the json wire format expected by IBM and Qobj
        specification because complex numbers are still of type complex. Also,
        this may contain native numpy arrays. When serializing this output
        for use with IBM systems, you can leverage a json encoder that converts these
        as expected. For example:

        .. code-block:: python

            import json
            import numpy

            class QobjEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, numpy.ndarray):
                        return obj.tolist()
                    if isinstance(obj, complex):
                        return (obj.real, obj.imag)
                    return json.JSONEncoder.default(self, obj)

            json.dumps(qobj.to_dict(), cls=QobjEncoder)

        Returns:
            dict: A dictionary representation of the QasmQobj object
        """
        out_dict = {
            "qobj_id": self.qobj_id,
            "header": self.header.to_dict(),
            "config": self.config.to_dict(),
            "schema_version": self.schema_version,
            "type": "QASM",
            "experiments": [x.to_dict() for x in self.experiments],
        }
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new QASMQobj object from a dictionary.

        Args:
            data (dict): A dictionary representing the QasmQobj to create. It
                will be in the same format as output by :func:`to_dict`.

        Returns:
            QasmQobj: The QasmQobj from the input dictionary.
        """
        config = None
        if "config" in data:
            config = QasmQobjConfig.from_dict(data["config"])
        experiments = None
        if "experiments" in data:
            experiments = [QasmQobjExperiment.from_dict(exp) for exp in data["experiments"]]
        header = None
        if "header" in data:
            header = QobjHeader.from_dict(data["header"])

        return cls(
            qobj_id=data.get("qobj_id"), config=config, experiments=experiments, header=header
        )

    def __eq__(self, other):
        if isinstance(other, QasmQobj):
            if self.to_dict() == other.to_dict():
                return True
        return False

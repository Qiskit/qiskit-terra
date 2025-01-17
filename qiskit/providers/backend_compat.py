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

"""Backend abstract interface for providers."""

from __future__ import annotations
import logging
import warnings
from typing import List, Iterable, Any, Dict, Optional

from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.providers.backend import QubitProperties
from qiskit.providers.models.backendconfiguration import BackendConfiguration
from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.providers.models.pulsedefaults import PulseDefaults
from qiskit.providers.options import Options
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.deprecate_pulse import deprecate_pulse_arg, deprecate_pulse_dependency


logger = logging.getLogger(__name__)


@deprecate_pulse_arg("defaults")
def convert_to_target(
    configuration: BackendConfiguration,
    properties: BackendProperties = None,
    defaults: PulseDefaults = None,
    custom_name_mapping: Optional[Dict[str, Any]] = None,
    add_delay: bool = True,
    filter_faulty: bool = True,
):
    """Decode transpiler target from backend data set.

    This function generates :class:`.Target`` instance from intermediate
    legacy objects such as :class:`.BackendProperties` and :class:`.PulseDefaults`.
    These objects are usually components of the legacy :class:`.BackendV1` model.

    Args:
        configuration: Backend configuration as ``BackendConfiguration``
        properties: Backend property dictionary or ``BackendProperties``
        defaults: DEPRECATED. Backend pulse defaults dictionary or ``PulseDefaults``
        custom_name_mapping: A name mapping must be supplied for the operation
            not included in Qiskit Standard Gate name mapping, otherwise the operation
            will be dropped in the resulting ``Target`` object.
        add_delay: If True, adds delay to the instruction set.
        filter_faulty: If True, this filters the non-operational qubits.

    Returns:
        A ``Target`` instance.
    """
    return _convert_to_target(
        configuration, properties, defaults, custom_name_mapping, add_delay, filter_faulty
    )


def _convert_to_target(
    configuration: BackendConfiguration,
    properties: BackendProperties = None,
    defaults: PulseDefaults = None,
    custom_name_mapping: Optional[Dict[str, Any]] = None,
    add_delay: bool = True,
    filter_faulty: bool = True,
):
    """An alternative private path to avoid pulse deprecations"""
    # importing packages where they are needed, to avoid cyclic-import.
    # pylint: disable=cyclic-import
    from qiskit.transpiler.target import (
        Target,
        InstructionProperties,
    )
    from qiskit.circuit.controlflow import ForLoopOp, IfElseOp, SwitchCaseOp, WhileLoopOp
    from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
    from qiskit.circuit.parameter import Parameter
    from qiskit.circuit.gate import Gate

    required = ["measure", "delay"]

    # Load Qiskit object representation
    qiskit_inst_mapping = get_standard_gate_name_mapping()
    if custom_name_mapping:
        qiskit_inst_mapping.update(custom_name_mapping)

    qiskit_control_flow_mapping = {
        "if_else": IfElseOp,
        "while_loop": WhileLoopOp,
        "for_loop": ForLoopOp,
        "switch_case": SwitchCaseOp,
    }

    in_data = {"num_qubits": configuration.num_qubits}

    # Parse global configuration properties
    if hasattr(configuration, "dt"):
        in_data["dt"] = configuration.dt
    if hasattr(configuration, "timing_constraints"):
        in_data.update(configuration.timing_constraints)

    # Create instruction property placeholder from backend configuration
    basis_gates = set(getattr(configuration, "basis_gates", []))
    supported_instructions = set(getattr(configuration, "supported_instructions", []))
    gate_configs = {gate.name: gate for gate in configuration.gates}
    all_instructions = set.union(
        basis_gates, set(required), supported_instructions.intersection(CONTROL_FLOW_OP_NAMES)
    )
    inst_name_map = {}  # type: Dict[str, Instruction]

    faulty_ops = set()
    faulty_qubits = set()
    unsupported_instructions = []

    # Create name to Qiskit instruction object repr mapping
    for name in all_instructions:
        if name in qiskit_control_flow_mapping:
            continue
        if name in qiskit_inst_mapping:
            inst_name_map[name] = qiskit_inst_mapping[name]
        elif name in gate_configs:
            # GateConfig model is a translator of QASM opcode.
            # This doesn't have quantum definition, so Qiskit transpiler doesn't perform
            # any optimization in quantum domain.
            # Usually GateConfig counterpart should exist in Qiskit namespace so this is rarely called.
            this_config = gate_configs[name]
            params = list(map(Parameter, getattr(this_config, "parameters", [])))
            coupling_map = getattr(this_config, "coupling_map", [])
            inst_name_map[name] = Gate(
                name=name,
                num_qubits=len(coupling_map[0]) if coupling_map else 0,
                params=params,
            )
        else:
            warnings.warn(
                f"No gate definition for {name} can be found and is being excluded "
                "from the generated target. You can use `custom_name_mapping` to provide "
                "a definition for this operation.",
                RuntimeWarning,
            )
            unsupported_instructions.append(name)

    for name in unsupported_instructions:
        all_instructions.remove(name)

    # Create inst properties placeholder
    # Without any assignment, properties value is None,
    # which defines a global instruction that can be applied to any qubit(s).
    # The None value behaves differently from an empty dictionary.
    # See API doc of Target.add_instruction for details.
    prop_name_map = dict.fromkeys(all_instructions)
    for name in all_instructions:
        if name in gate_configs:
            if coupling_map := getattr(gate_configs[name], "coupling_map", None):
                # Respect operational qubits that gate configuration defines
                # This ties instruction to particular qubits even without properties information.
                # Note that each instruction is considered to be ideal unless
                # its spec (e.g. error, duration) is bound by the properties object.
                prop_name_map[name] = dict.fromkeys(map(tuple, coupling_map))

    # Populate instruction properties
    if properties:

        def _get_value(prop_dict, prop_name):
            if ndval := prop_dict.get(prop_name, None):
                return ndval[0]
            return None

        # is_qubit_operational is a bit of expensive operation so precache the value
        faulty_qubits = {
            q for q in range(configuration.num_qubits) if not properties.is_qubit_operational(q)
        }

        qubit_properties = []
        for qi in range(0, configuration.num_qubits):
            # TODO faulty qubit handling might be needed since
            #  faulty qubit reporting qubit properties doesn't make sense.
            try:
                prop_dict = properties.qubit_property(qubit=qi)
            except KeyError:
                continue
            qubit_properties.append(
                QubitProperties(
                    t1=prop_dict.get("T1", (None, None))[0],
                    t2=prop_dict.get("T2", (None, None))[0],
                    frequency=prop_dict.get("frequency", (None, None))[0],
                )
            )
        in_data["qubit_properties"] = qubit_properties

        for name in all_instructions:
            try:
                for qubits, params in properties.gate_property(name).items():
                    if filter_faulty and (
                        set.intersection(faulty_qubits, qubits)
                        or not properties.is_gate_operational(name, qubits)
                    ):
                        try:
                            # Qubits might be pre-defined by the gate config
                            # However properties objects says the qubits is non-operational
                            del prop_name_map[name][qubits]
                        except KeyError:
                            pass
                        faulty_ops.add((name, qubits))
                        continue
                    if prop_name_map[name] is None:
                        # This instruction is tied to particular qubits
                        # i.e. gate config is not provided, and instruction has been globally defined.
                        prop_name_map[name] = {}
                    prop_name_map[name][qubits] = InstructionProperties(
                        error=_get_value(params, "gate_error"),
                        duration=_get_value(params, "gate_length"),
                    )
                if isinstance(prop_name_map[name], dict) and any(
                    v is None for v in prop_name_map[name].values()
                ):
                    # Properties provides gate properties only for subset of qubits
                    # Associated qubit set might be defined by the gate config here
                    logger.info(
                        "Gate properties of instruction %s are not provided for every qubits. "
                        "This gate is ideal for some qubits and the rest is with finite error. "
                        "Created backend target may confuse error-aware circuit optimization.",
                        name,
                    )
            except BackendPropertyError:
                # This gate doesn't report any property
                continue

        # Measure instruction property is stored in qubit property
        prop_name_map["measure"] = {}

        for qubit_idx in range(configuration.num_qubits):
            if filter_faulty and (qubit_idx in faulty_qubits):
                continue
            qubit_prop = properties.qubit_property(qubit_idx)
            prop_name_map["measure"][(qubit_idx,)] = InstructionProperties(
                error=_get_value(qubit_prop, "readout_error"),
                duration=_get_value(qubit_prop, "readout_length"),
            )

    for op in required:
        # Map required ops to each operational qubit
        if prop_name_map[op] is None:
            prop_name_map[op] = {
                (q,): None
                for q in range(configuration.num_qubits)
                if not filter_faulty or (q not in faulty_qubits)
            }

    if defaults:
        inst_sched_map = defaults.instruction_schedule_map

        for name in inst_sched_map.instructions:
            for qubits in inst_sched_map.qubits_with_instruction(name):
                if not isinstance(qubits, tuple):
                    qubits = (qubits,)
                if (
                    name not in all_instructions
                    or name not in prop_name_map
                    or prop_name_map[name] is None
                    or qubits not in prop_name_map[name]
                ):
                    logger.info(
                        "Gate calibration for instruction %s on qubits %s is found "
                        "in the PulseDefaults payload. However, this entry is not defined in "
                        "the gate mapping of Target. This calibration is ignored.",
                        name,
                        qubits,
                    )
                    continue

                if (name, qubits) in faulty_ops:
                    continue

                entry = inst_sched_map._get_calibration_entry(name, qubits)
                try:
                    prop_name_map[name][qubits]._calibration_prop = entry
                except AttributeError:
                    # if instruction properties are "None", add entry
                    prop_name_map[name].update({qubits: InstructionProperties(None, None, entry)})
                    logger.info(
                        "The PulseDefaults payload received contains an instruction %s on "
                        "qubits %s which is not present in the configuration or properties payload."
                        "A new properties entry will be added to include the new calibration data.",
                        name,
                        qubits,
                    )
    # Add parsed properties to target
    target = Target(**in_data)
    for inst_name in all_instructions:
        if inst_name == "delay" and not add_delay:
            continue
        if inst_name in qiskit_control_flow_mapping:
            # Control flow operator doesn't have gate property.
            target.add_instruction(
                instruction=qiskit_control_flow_mapping[inst_name],
                name=inst_name,
            )
        else:
            target.add_instruction(
                instruction=inst_name_map[inst_name],
                properties=prop_name_map.get(inst_name, None),
                name=inst_name,
            )

    return target


def qubit_props_list_from_props(
    properties: BackendProperties,
) -> List[QubitProperties]:
    """Uses BackendProperties to construct
    and return a list of QubitProperties.
    """
    qubit_props: List[QubitProperties] = []
    for qubit, _ in enumerate(properties.qubits):
        try:
            t_1 = properties.t1(qubit)
        except BackendPropertyError:
            t_1 = None
        try:
            t_2 = properties.t2(qubit)
        except BackendPropertyError:
            t_2 = None
        try:
            frequency = properties.frequency(qubit)
        except BackendPropertyError:
            frequency = None
        qubit_props.append(
            QubitProperties(  # type: ignore[no-untyped-call]
                t1=t_1,
                t2=t_2,
                frequency=frequency,
            )
        )
    return qubit_props


class BackendV2Converter(BackendV2):
    """A converter class that takes a :class:`~.BackendV1` instance and wraps it in a
    :class:`~.BackendV2` interface.

    This class implements the :class:`~.BackendV2` interface and is used to enable
    common access patterns between :class:`~.BackendV1` and :class:`~.BackendV2`. This
    class should only be used if you need a :class:`~.BackendV2` and still need
    compatibility with :class:`~.BackendV1`.

    When using custom calibrations (or other custom workflows) it is **not** recommended
    to mutate the ``BackendV1`` object before applying this converter. For example, in order to
    convert a ``BackendV1`` object with a customized ``defaults().instruction_schedule_map``,
    which has a custom calibration for an operation, the operation name must be in
    ``configuration().basis_gates`` and ``name_mapping`` must be supplied for the operation.
    Otherwise, the operation will be dropped in the resulting ``BackendV2`` object.

    Instead it is typically better to add custom calibrations **after** applying this converter
    instead of updating ``BackendV1.defaults()`` in advance. For example::

        backend_v2 = BackendV2Converter(backend_v1)
        backend_v2.target.add_instruction(
            custom_gate, {(0, 1): InstructionProperties(calibration=custom_sched)}
        )
    """

    def __init__(
        self,
        backend: BackendV1,
        name_mapping: Optional[Dict[str, Any]] = None,
        add_delay: bool = True,
        filter_faulty: bool = True,
    ):
        """Initialize a BackendV2 converter instance based on a BackendV1 instance.

        Args:
            backend: The input :class:`~.BackendV1` based backend to wrap in a
                :class:`~.BackendV2` interface
            name_mapping: An optional dictionary that maps custom gate/operation names in
                ``backend`` to an :class:`~.Operation` object representing that
                gate/operation. By default most standard gates names are mapped to the
                standard gate object from :mod:`qiskit.circuit.library` this only needs
                to be specified if the input ``backend`` defines gates in names outside
                that set.
            add_delay: If set to true a :class:`~qiskit.circuit.Delay` operation
                will be added to the target as a supported operation for all
                qubits
            filter_faulty: If the :class:`~.BackendProperties` object (if present) for
                ``backend`` has any qubits or gates flagged as non-operational filter
                those from the output target.
        """
        self._backend = backend
        self._config = self._backend.configuration()
        super().__init__(
            provider=backend.provider,
            name=backend.name(),
            description=getattr(self._config, "description", None),
            online_date=getattr(self._config, "online_date", None),
            backend_version=self._config.backend_version,
        )
        self._options = self._backend._options
        self._properties = None
        self._defaults = None

        with warnings.catch_warnings():
            # The class QobjExperimentHeader is deprecated
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
            if hasattr(self._backend, "properties"):
                self._properties = self._backend.properties()
            if hasattr(self._backend, "defaults"):
                self._defaults = self._backend.defaults()

        self._target = None
        self._name_mapping = name_mapping
        self._add_delay = add_delay
        self._filter_faulty = filter_faulty

    @property
    def target(self):
        """A :class:`qiskit.transpiler.Target` object for the backend.

        :rtype: Target
        """
        if self._target is None:
            self._target = _convert_to_target(
                configuration=self._config,
                properties=self._properties,
                defaults=self._defaults,
                custom_name_mapping=self._name_mapping,
                add_delay=self._add_delay,
                filter_faulty=self._filter_faulty,
            )
        return self._target

    @property
    def max_circuits(self):
        return self._config.max_experiments

    @classmethod
    def _default_options(cls):
        return Options()

    @property
    def dtm(self) -> float:
        return self._config.dtm

    @property
    def meas_map(self) -> List[List[int]]:
        return self._config.meas_map

    @deprecate_pulse_dependency
    def drive_channel(self, qubit: int):
        return self._config.drive(qubit)

    @deprecate_pulse_dependency
    def measure_channel(self, qubit: int):
        return self._config.measure(qubit)

    @deprecate_pulse_dependency
    def acquire_channel(self, qubit: int):
        return self._config.acquire(qubit)

    @deprecate_pulse_dependency
    def control_channel(self, qubits: Iterable[int]):
        return self._config.control(qubits)

    def run(self, run_input, **options):
        return self._backend.run(run_input, **options)

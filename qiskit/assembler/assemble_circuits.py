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

"""Assemble function for converting a list of circuits into a qobj."""
import copy
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from qiskit.assembler.run_config import RunConfig
from qiskit.assembler.assemble_schedules import _assemble_instructions as _assemble_schedule
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classicalregister import Clbit
from qiskit.exceptions import QiskitError
from qiskit.qobj import (
    QasmQobj,
    QobjExperimentHeader,
    QasmQobjInstruction,
    QasmQobjExperimentConfig,
    QasmQobjExperiment,
    QasmQobjConfig,
    QasmExperimentCalibrations,
    GateCalibration,
    PulseQobjInstruction,
    PulseLibraryItem,
    converters,
    QobjHeader,
)
from qiskit.utils.parallel import parallel_map
from qiskit.utils import deprecate_func


PulseLibrary = Dict[str, List[complex]]


def _assemble_circuit(
    circuit: QuantumCircuit, run_config: RunConfig
) -> Tuple[QasmQobjExperiment, Optional[PulseLibrary]]:
    """Assemble one circuit.

    Args:
        circuit: circuit to assemble
        run_config: configuration of the runtime environment

    Returns:
        One experiment for the QasmQobj, and pulse library for pulse gates (which could be None)

    Raises:
        QiskitError: when the circuit has unit other than 'dt'.
    """
    if circuit.unit != "dt":
        raise QiskitError(
            f"Unable to assemble circuit with unit '{circuit.unit}', which must be 'dt'."
        )

    # header data
    num_qubits = 0
    memory_slots = 0
    qubit_labels = []
    clbit_labels = []

    qreg_sizes = []
    creg_sizes = []
    for qreg in circuit.qregs:
        qreg_sizes.append([qreg.name, qreg.size])
        for j in range(qreg.size):
            qubit_labels.append([qreg.name, j])
        num_qubits += qreg.size
    for creg in circuit.cregs:
        creg_sizes.append([creg.name, creg.size])
        for j in range(creg.size):
            clbit_labels.append([creg.name, j])
        memory_slots += creg.size

    qubit_indices = {qubit: idx for idx, qubit in enumerate(circuit.qubits)}
    clbit_indices = {clbit: idx for idx, clbit in enumerate(circuit.clbits)}

    # TODO: why do we need creq_sizes and qreg_sizes in header
    # TODO: we need to rethink memory_slots as they are tied to classical bit
    metadata = circuit.metadata
    if metadata is None:
        metadata = {}
    with warnings.catch_warnings():
        # The class QobjExperimentHeader is deprecated
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
        header = QobjExperimentHeader(
            qubit_labels=qubit_labels,
            n_qubits=num_qubits,
            qreg_sizes=qreg_sizes,
            clbit_labels=clbit_labels,
            memory_slots=memory_slots,
            creg_sizes=creg_sizes,
            name=circuit.name,
            global_phase=float(circuit.global_phase),
            metadata=metadata,
        )

    # TODO: why do we need n_qubits and memory_slots in both the header and the config
    with warnings.catch_warnings():
        # The class QasmQobjExperimentConfig is deprecated
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
        config = QasmQobjExperimentConfig(n_qubits=num_qubits, memory_slots=memory_slots)
    calibrations, pulse_library = _assemble_pulse_gates(circuit, run_config)
    if calibrations:
        config.calibrations = calibrations

    # Convert conditionals from OpenQASM-2-style (creg ?= int) to qobj-style
    # (register_bit ?= 1), by assuming device has unlimited register slots
    # (supported only for simulators). Map all measures to a register matching
    # their clbit_index, create a new register slot for every conditional gate
    # and add a bfunc to map the creg=val mask onto the gating register bit.

    is_conditional_experiment = any(
        getattr(instruction.operation, "condition", None) for instruction in circuit.data
    )
    max_conditional_idx = 0

    instructions = []
    for op_context in circuit.data:
        instruction = op_context.operation._assemble()

        # Add register attributes to the instruction
        qargs = op_context.qubits
        cargs = op_context.clbits
        if qargs:
            instruction.qubits = [qubit_indices[qubit] for qubit in qargs]
        if cargs:
            instruction.memory = [clbit_indices[clbit] for clbit in cargs]
            # If the experiment has conditional instructions, assume every
            # measurement result may be needed for a conditional gate.
            if instruction.name == "measure" and is_conditional_experiment:
                instruction.register = [clbit_indices[clbit] for clbit in cargs]

        # To convert to a qobj-style conditional, insert a bfunc prior
        # to the conditional instruction to map the creg ?= val condition
        # onto a gating register bit.
        if hasattr(instruction, "_condition"):
            ctrl_reg, ctrl_val = instruction._condition
            mask = 0
            val = 0
            if isinstance(ctrl_reg, Clbit):
                mask = 1 << clbit_indices[ctrl_reg]
                val = (ctrl_val & 1) << clbit_indices[ctrl_reg]
            else:
                for clbit in clbit_indices:
                    if clbit in ctrl_reg:
                        mask |= 1 << clbit_indices[clbit]
                        val |= ((ctrl_val >> list(ctrl_reg).index(clbit)) & 1) << clbit_indices[
                            clbit
                        ]

            conditional_reg_idx = memory_slots + max_conditional_idx
            with warnings.catch_warnings():
                # The class QasmQobjInstruction is deprecated
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
                conversion_bfunc = QasmQobjInstruction(
                    name="bfunc",
                    mask="0x%X" % mask,  # pylint: disable=consider-using-f-string
                    relation="==",
                    val="0x%X" % val,  # pylint: disable=consider-using-f-string
                    register=conditional_reg_idx,
                )
            instructions.append(conversion_bfunc)
            instruction.conditional = conditional_reg_idx
            max_conditional_idx += 1
            # Delete condition attribute now that we have replaced it with
            # the conditional and bfunc
            del instruction._condition

        instructions.append(instruction)
    with warnings.catch_warnings():
        # The class QasmQobjExperiment is deprecated
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
        return (
            QasmQobjExperiment(instructions=instructions, header=header, config=config),
            pulse_library,
        )


def _assemble_pulse_gates(
    circuit: QuantumCircuit, run_config: RunConfig
) -> Tuple[Optional[QasmExperimentCalibrations], Optional[PulseLibrary]]:
    """Assemble and return the circuit calibrations and associated pulse library, if there are any.
    The calibrations themselves may reference the pulse library which is returned as a dict.

    Args:
        circuit: circuit which may have pulse calibrations
        run_config: configuration of the runtime environment

    Returns:
        The calibrations and pulse library, if there are any
    """
    if not circuit.calibrations:
        return None, None
    if not hasattr(run_config, "parametric_pulses"):
        run_config.parametric_pulses = []
    calibrations = []
    pulse_library = {}
    for gate, cals in circuit.calibrations.items():
        for (qubits, params), schedule in cals.items():
            qobj_instructions, _ = _assemble_schedule(
                schedule,
                converters.InstructionToQobjConverter(PulseQobjInstruction),
                run_config,
                pulse_library,
            )
            calibrations.append(
                GateCalibration(str(gate), list(qubits), list(params), qobj_instructions)
            )
    return QasmExperimentCalibrations(gates=calibrations), pulse_library


def _extract_common_calibrations(
    experiments: List[QasmQobjExperiment],
) -> Tuple[List[QasmQobjExperiment], Optional[QasmExperimentCalibrations]]:
    """Given a list of ``QasmQobjExperiment``s, each of which may have calibrations in their
    ``config``, collect common calibrations into a global ``QasmExperimentCalibrations``
    and delete them from their local experiments.

    Args:
        experiments: The list of OpenQASM experiments that are being assembled into one qobj

    Returns:
        The input experiments with modified calibrations, and common calibrations, if there
        are any
    """

    def index_calibrations() -> Dict[int, List[Tuple[int, GateCalibration]]]:
        """Map each calibration to all experiments that contain it."""
        exp_indices = defaultdict(list)
        for exp_idx, exp in enumerate(experiments):
            for gate_cal in exp.config.calibrations.gates:
                # They must be keyed on the hash or identical cals will be indexed separately
                exp_indices[hash(gate_cal)].append((exp_idx, gate_cal))
        return exp_indices

    def collect_common_calibrations() -> List[GateCalibration]:
        """If a gate calibration appears in all experiments, collect it."""
        common_calibrations = []
        for _, exps_w_cal in exp_indices.items():
            if len(exps_w_cal) == len(experiments):
                _, gate_cal = exps_w_cal[0]
                common_calibrations.append(gate_cal)
        return common_calibrations

    def remove_common_gate_calibrations(exps: List[QasmQobjExperiment]) -> None:
        """For calibrations that appear in all experiments, remove them from the individual
        experiment's ``config.calibrations``."""
        for _, exps_w_cal in exp_indices.items():
            if len(exps_w_cal) == len(exps):
                for exp_idx, gate_cal in exps_w_cal:
                    exps[exp_idx].config.calibrations.gates.remove(gate_cal)

    if not (experiments and all(hasattr(exp.config, "calibrations") for exp in experiments)):
        # No common calibrations
        return experiments, None

    exp_indices = index_calibrations()
    common_calibrations = collect_common_calibrations()
    remove_common_gate_calibrations(experiments)

    # Remove the ``calibrations`` attribute if it's now empty
    for exp in experiments:
        if not exp.config.calibrations.gates:
            del exp.config.calibrations

    return experiments, QasmExperimentCalibrations(gates=common_calibrations)


def _configure_experiment_los(
    experiments: List[QasmQobjExperiment],
    lo_converter: converters.LoConfigConverter,
    run_config: RunConfig,
):
    # get per experiment los
    freq_configs = [lo_converter(lo_dict) for lo_dict in getattr(run_config, "schedule_los", [])]

    if len(experiments) > 1 and len(freq_configs) not in [0, 1, len(experiments)]:
        raise QiskitError(
            "Invalid 'schedule_los' setting specified. If specified, it should be "
            "either have a single entry to apply the same LOs for each experiment or "
            "have length equal to the number of experiments."
        )

    if len(freq_configs) > 1:
        if len(experiments) > 1:
            for idx, expt in enumerate(experiments):
                freq_config = freq_configs[idx]
                expt.config.qubit_lo_freq = freq_config.qubit_lo_freq
                expt.config.meas_lo_freq = freq_config.meas_lo_freq
        elif len(experiments) == 1:
            expt = experiments[0]
            experiments = []
            for freq_config in freq_configs:
                expt_config = copy.deepcopy(expt.config)
                expt_config.qubit_lo_freq = freq_config.qubit_lo_freq
                expt_config.meas_lo_freq = freq_config.meas_lo_freq
                experiments.append(
                    QasmQobjExperiment(
                        header=expt.header, instructions=expt.instructions, config=expt_config
                    )
                )

    return experiments


def _assemble_circuits(
    circuits: List[QuantumCircuit], run_config: RunConfig, qobj_id: int, qobj_header: QobjHeader
) -> QasmQobj:
    with warnings.catch_warnings():
        # Still constructs Qobj, that is deprecated. The message is hard to trace to a module,
        # because concurrency is hard.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        experiments_and_pulse_libs = parallel_map(_assemble_circuit, circuits, [run_config])
    experiments = []
    pulse_library = {}
    for exp, lib in experiments_and_pulse_libs:
        experiments.append(exp)
        if lib:
            pulse_library.update(lib)

    # extract common calibrations
    experiments, calibrations = _extract_common_calibrations(experiments)

    # configure LO freqs per circuit
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
        lo_converter = converters.LoConfigConverter(
            QasmQobjExperimentConfig, **run_config.to_dict()
        )
    experiments = _configure_experiment_los(experiments, lo_converter, run_config)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
        qobj_config = QasmQobjConfig()
    if run_config:
        qobj_config_dict = run_config.to_dict()

        # remove LO ranges, not needed in qobj
        qobj_config_dict.pop("qubit_lo_range", None)
        qobj_config_dict.pop("meas_lo_range", None)

        # convert LO frequencies to GHz, if they exist
        if "qubit_lo_freq" in qobj_config_dict:
            qobj_config_dict["qubit_lo_freq"] = [
                freq / 1e9 for freq in qobj_config_dict["qubit_lo_freq"]
            ]
        if "meas_lo_freq" in qobj_config_dict:
            qobj_config_dict["meas_lo_freq"] = [
                freq / 1e9 for freq in qobj_config_dict["meas_lo_freq"]
            ]

        # override default los if single ``schedule_los`` entry set
        schedule_los = qobj_config_dict.pop("schedule_los", [])
        if len(schedule_los) == 1:
            lo_dict = schedule_los[0]
            q_los = lo_converter.get_qubit_los(lo_dict)
            # Hz -> GHz
            if q_los:
                qobj_config_dict["qubit_lo_freq"] = [freq / 1e9 for freq in q_los]
            m_los = lo_converter.get_meas_los(lo_dict)
            if m_los:
                qobj_config_dict["meas_lo_freq"] = [freq / 1e9 for freq in m_los]

        with warnings.catch_warnings():
            # The class QasmQobjConfig is deprecated
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
            qobj_config = QasmQobjConfig(**qobj_config_dict)

    qubit_sizes = []
    memory_slot_sizes = []
    for circ in circuits:
        num_qubits = 0
        memory_slots = 0
        for qreg in circ.qregs:
            num_qubits += qreg.size
        for creg in circ.cregs:
            memory_slots += creg.size
        qubit_sizes.append(num_qubits)
        memory_slot_sizes.append(memory_slots)
    qobj_config.memory_slots = max(memory_slot_sizes)
    qobj_config.n_qubits = max(qubit_sizes)

    if pulse_library:
        qobj_config.pulse_library = [
            PulseLibraryItem(name=name, samples=samples) for name, samples in pulse_library.items()
        ]

    if calibrations and calibrations.gates:
        qobj_config.calibrations = calibrations
    with warnings.catch_warnings():
        # The class QasmQobj is deprecated
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
        return QasmQobj(
            qobj_id=qobj_id, config=qobj_config, experiments=experiments, header=qobj_header
        )


@deprecate_func(
    since="1.2",
    removal_timeline="in the 2.0 release",
    additional_msg="The `Qobj` class and related functionality are part of the deprecated `BackendV1` "
    "workflow,  and no longer necessary for `BackendV2`. If a user workflow requires "
    "`Qobj` it likely relies on deprecated functionality and should be updated to "
    "use `BackendV2`.",
)
def assemble_circuits(
    circuits: List[QuantumCircuit], run_config: RunConfig, qobj_id: int, qobj_header: QobjHeader
) -> QasmQobj:
    """Assembles a list of circuits into a qobj that can be run on the backend.

    Args:
        circuits: circuit(s) to assemble
        run_config: configuration of the runtime environment
        qobj_id: identifier for the generated qobj
        qobj_header: header to pass to the results

    Returns:
        The qobj to be run on the backends

    Examples:

        .. plot::
           :include-source:
           :nofigs:

            from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.assembler import assemble_circuits
            from qiskit.assembler.run_config import RunConfig
            # Build a circuit to convert into a Qobj
            q = QuantumRegister(2)
            c = ClassicalRegister(2)
            qc = QuantumCircuit(q, c)
            qc.h(q[0])
            qc.cx(q[0], q[1])
            qc.measure(q, c)
            # Assemble a Qobj from the input circuit
            qobj = assemble_circuits(circuits=[qc],
                                     qobj_id="custom-id",
                                     qobj_header=[],
                                     run_config=RunConfig(shots=2000, memory=True, init_qubits=True))
    """
    # assemble the circuit experiments
    return _assemble_circuits(circuits, run_config, qobj_id, qobj_header)

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

"""Disassemble function for a qobj into a list of circuits and its config"""
from typing import Any, Dict, List, NewType, Tuple, Union
import collections
import math

from qiskit import pulse
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister

from qiskit.qobj import PulseQobjInstruction
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.utils import deprecate_func

# A ``CircuitModule`` is a representation of a circuit execution on the backend.
# It is currently a list of quantum circuits to execute, a run Qobj dictionary
# and a header dictionary.
CircuitModule = NewType(
    "CircuitModule", Tuple[List[QuantumCircuit], Dict[str, Any], Dict[str, Any]]
)

# A ``PulseModule`` is a representation of a pulse execution on the backend.
# It is currently a list of pulse schedules to execute, a run Qobj dictionary
# and a header dictionary.
PulseModule = NewType("PulseModule", Tuple[List[pulse.Schedule], Dict[str, Any], Dict[str, Any]])


@deprecate_func(
    since="1.2",
    removal_timeline="in the 2.0 release",
    additional_msg="The `Qobj` class and related functionality are part of the deprecated "
    "`BackendV1` workflow,  and no longer necessary for `BackendV2`. If a user "
    "workflow requires `Qobj` it likely relies on deprecated functionality and "
    "should be updated to use `BackendV2`.",
)
def disassemble(qobj) -> Union[CircuitModule, PulseModule]:
    """Disassemble a qobj and return the circuits or pulse schedules, run_config, and user header.

    .. note::

        ``disassemble(assemble(qc))`` is not guaranteed to produce an exactly equal circuit to the
        input, due to limitations in the :obj:`.QasmQobj` format that need to be maintained for
        backend system compatibility.  This is most likely to be the case when using newer features
        of :obj:`.QuantumCircuit`.  In most cases, the output should be equivalent, if not quite
        equal.

    Args:
        qobj (Qobj): The input qobj object to disassemble

    Returns:
        Union[CircuitModule, PulseModule]: The disassembled program which consists of:

            * programs: A list of quantum circuits or pulse schedules
            * run_config: The dict of the run config
            * user_qobj_header: The dict of any user headers in the qobj

    Examples:

        .. plot::
           :include-source:
           :nofigs:

            from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.compiler.assembler import assemble
            from qiskit.assembler.disassemble import disassemble
            # Create a circuit to assemble into a qobj
            q = QuantumRegister(2)
            c = ClassicalRegister(2)
            qc = QuantumCircuit(q, c)
            qc.h(q[0])
            qc.cx(q[0], q[1])
            qc.measure(q, c)
            # Assemble the circuit into a Qobj
            qobj = assemble(qc, shots=2000, memory=True)
            # Disassemble the qobj back into a circuit
            circuits, run_config_out, headers = disassemble(qobj)
    """
    if qobj.type == "PULSE":
        return _disassemble_pulse_schedule(qobj)
    else:
        return _disassemble_circuit(qobj)


def _disassemble_circuit(qobj) -> CircuitModule:
    run_config = qobj.config.to_dict()

    # convert lo freq back to Hz
    qubit_lo_freq = run_config.get("qubit_lo_freq", [])
    if qubit_lo_freq:
        run_config["qubit_lo_freq"] = [freq * 1e9 for freq in qubit_lo_freq]

    meas_lo_freq = run_config.get("meas_lo_freq", [])
    if meas_lo_freq:
        run_config["meas_lo_freq"] = [freq * 1e9 for freq in meas_lo_freq]

    user_qobj_header = qobj.header.to_dict()
    return CircuitModule((_experiments_to_circuits(qobj), run_config, user_qobj_header))


def _qobj_to_circuit_cals(qobj, pulse_lib):
    """Return circuit calibrations dictionary from qobj/exp config calibrations."""
    qobj_cals = qobj.config.calibrations.to_dict()["gates"]
    converter = QobjToInstructionConverter(pulse_lib)

    qc_cals = {}
    for gate in qobj_cals:
        config = (tuple(gate["qubits"]), tuple(gate["params"]))
        cal = {
            config: pulse.Schedule(
                name=f"{gate['name']} {str(gate['params'])} {str(gate['qubits'])}"
            )
        }
        for instruction in gate["instructions"]:
            qobj_instruction = PulseQobjInstruction.from_dict(instruction)
            schedule = converter(qobj_instruction)
            cal[config] = cal[config].insert(schedule.ch_start_time(), schedule)
        if gate["name"] in qc_cals:
            qc_cals[gate["name"]].update(cal)
        else:
            qc_cals[gate["name"]] = cal

    return qc_cals


def _experiments_to_circuits(qobj):
    """Return a list of QuantumCircuit object(s) from a qobj.

    Args:
        qobj (Qobj): The Qobj object to convert to QuantumCircuits

    Returns:
        list: A list of QuantumCircuit objects from the qobj
    """
    if not qobj.experiments:
        return None

    circuits = []
    for exp in qobj.experiments:
        quantum_registers = [QuantumRegister(i[1], name=i[0]) for i in exp.header.qreg_sizes]
        classical_registers = [ClassicalRegister(i[1], name=i[0]) for i in exp.header.creg_sizes]
        circuit = QuantumCircuit(*quantum_registers, *classical_registers, name=exp.header.name)
        qreg_dict = collections.OrderedDict()
        creg_dict = collections.OrderedDict()
        for reg in quantum_registers:
            qreg_dict[reg.name] = reg
        for reg in classical_registers:
            creg_dict[reg.name] = reg
        conditional = {}
        for i in exp.instructions:
            name = i.name
            qubits = []
            params = getattr(i, "params", [])
            try:
                for qubit in i.qubits:
                    qubit_label = exp.header.qubit_labels[qubit]
                    qubits.append(qreg_dict[qubit_label[0]][qubit_label[1]])
            except Exception:  # pylint: disable=broad-except
                pass
            clbits = []
            try:
                for clbit in i.memory:
                    clbit_label = exp.header.clbit_labels[clbit]
                    clbits.append(creg_dict[clbit_label[0]][clbit_label[1]])
            except Exception:  # pylint: disable=broad-except
                pass
            if hasattr(circuit, name):
                instr_method = getattr(circuit, name)
                if i.name == "initialize":
                    _inst = instr_method(params, qubits)
                elif i.name in ["mcx", "mcu1", "mcp"]:
                    _inst = instr_method(*params, qubits[:-1], qubits[-1], *clbits)
                else:
                    _inst = instr_method(*params, *qubits, *clbits)
            elif name == "bfunc":
                conditional["value"] = int(i.val, 16)
                full_bit_size = sum(creg_dict[x].size for x in creg_dict)
                mask_map = {}
                raw_map = {}
                raw = []

                for creg in creg_dict:
                    size = creg_dict[creg].size
                    reg_raw = [1] * size
                    if not raw:
                        raw = reg_raw
                    else:
                        for pos, val in enumerate(raw):
                            if val == 1:
                                raw[pos] = 0
                        raw = reg_raw + raw
                    mask = [0] * (full_bit_size - len(raw)) + raw
                    raw_map[creg] = mask
                    mask_map[int("".join(str(x) for x in mask), 2)] = creg
                if bin(int(i.mask, 16)).count("1") == 1:
                    # The condition is on a single bit.  This might be a single-bit condition, or it
                    # might be a register of length one.  The case that it's a single-bit condition
                    # in a register of length one is ambiguous, and we choose to return a condition
                    # on the register.  This may not match the input circuit exactly, but is at
                    # least equivalent.
                    cbit = int(math.log2(int(i.mask, 16)))
                    for reg in creg_dict.values():
                        size = reg.size
                        if cbit >= size:
                            cbit -= size
                        else:
                            conditional["register"] = reg if reg.size == 1 else reg[cbit]
                            break
                    mask_str = bin(int(i.mask, 16))[2:].zfill(full_bit_size)
                    mask = [int(item) for item in list(mask_str)]
                else:
                    creg = mask_map[int(i.mask, 16)]
                    conditional["register"] = creg_dict[creg]
                    mask = raw_map[creg]
                val = int(i.val, 16)
                for j in reversed(mask):
                    if j == 0:
                        val = val >> 1
                    else:
                        conditional["value"] = val
                        break
            else:
                _inst = temp_opaque_instruction = Instruction(
                    name=name, num_qubits=len(qubits), num_clbits=len(clbits), params=params
                )
                circuit.append(temp_opaque_instruction, qubits, clbits)
            if conditional and name != "bfunc":
                _inst.c_if(conditional["register"], conditional["value"])
                conditional = {}
        pulse_lib = qobj.config.pulse_library if hasattr(qobj.config, "pulse_library") else []
        # The dict update method did not work here; could investigate in the future
        if hasattr(qobj.config, "calibrations"):
            circuit.calibrations = {
                **circuit.calibrations,
                **_qobj_to_circuit_cals(qobj, pulse_lib),
            }
        if hasattr(exp.config, "calibrations"):
            circuit.calibrations = {**circuit.calibrations, **_qobj_to_circuit_cals(exp, pulse_lib)}
        circuits.append(circuit)
    return circuits


def _disassemble_pulse_schedule(qobj) -> PulseModule:
    run_config = qobj.config.to_dict()
    run_config.pop("pulse_library")

    qubit_lo_freq = run_config.get("qubit_lo_freq")
    if qubit_lo_freq:
        run_config["qubit_lo_freq"] = [freq * 1e9 for freq in qubit_lo_freq]

    meas_lo_freq = run_config.get("meas_lo_freq")
    if meas_lo_freq:
        run_config["meas_lo_freq"] = [freq * 1e9 for freq in meas_lo_freq]

    user_qobj_header = qobj.header.to_dict()

    # extract schedule lo settings
    schedule_los = []
    for program in qobj.experiments:
        program_los = {}
        if hasattr(program, "config"):
            if hasattr(program.config, "qubit_lo_freq"):
                for i, lo in enumerate(program.config.qubit_lo_freq):
                    program_los[pulse.DriveChannel(i)] = lo * 1e9

            if hasattr(program.config, "meas_lo_freq"):
                for i, lo in enumerate(program.config.meas_lo_freq):
                    program_los[pulse.MeasureChannel(i)] = lo * 1e9

        schedule_los.append(program_los)

    if any(schedule_los):
        run_config["schedule_los"] = schedule_los

    return PulseModule((_experiments_to_schedules(qobj), run_config, user_qobj_header))


def _experiments_to_schedules(qobj) -> List[pulse.Schedule]:
    """Return a list of :class:`qiskit.pulse.Schedule` object(s) from a qobj.

    Args:
        qobj (Qobj): The Qobj object to convert to pulse schedules.

    Returns:
        A list of :class:`qiskit.pulse.Schedule` objects from the qobj

    Raises:
        pulse.PulseError: If a parameterized instruction is supplied.
    """
    converter = QobjToInstructionConverter(qobj.config.pulse_library)

    schedules = []
    for program in qobj.experiments:
        insts = []
        for inst in program.instructions:
            insts.append(converter(inst))

        schedule = pulse.Schedule(*insts)
        schedules.append(schedule)
    return schedules

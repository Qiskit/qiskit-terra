// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
#![allow(clippy::too_many_arguments)]

use std::f64::consts::PI;
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use approx::relative_eq;
use hashbrown::{HashMap, HashSet};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use ndarray::prelude::*;
use num_complex::{Complex, Complex64};
use numpy::IntoPyArray;
use qiskit_circuit::circuit_instruction::{ExtraInstructionAttributes, OperationFromPython};
use smallvec::{smallvec, SmallVec};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyString};
use pyo3::wrap_pyfunction;
use pyo3::Python;

use qiskit_circuit::converters::{circuit_to_dag, QuantumCircuitData};
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::imports;
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::Qubit;

use crate::euler_one_qubit_decomposer::{
    unitary_to_gate_sequence_inner, EulerBasis, EulerBasisSet, EULER_BASES, EULER_BASIS_NAMES,
};
use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::{NormalOperation, Target};
use crate::two_qubit_decompose::{
    TwoQubitBasisDecomposer, TwoQubitGateSequence, TwoQubitWeylDecomposition,
};
use crate::QiskitError;

const PI2: f64 = PI / 2.;
const PI4: f64 = PI / 4.;

#[derive(Clone, Debug)]
enum DecomposerType {
    TwoQubitBasisDecomposer(Box<TwoQubitBasisDecomposer>),
    XXDecomposer(PyObject),
}

struct DecomposerElement {
    decomposer: DecomposerType,
    gate: NormalOperation,
}

#[derive(Clone, Debug)]
struct TwoQubitUnitarySequence {
    gate_sequence: TwoQubitGateSequence,
    decomp_gate: NormalOperation,
}

// Used in get_2q_decomposers. If the found 2q basis is a subset of GOODBYE_SET,
// then we know TwoQubitBasisDecomposer is an ideal decomposition and there is
// no need to bother trying the XXDecomposer.
static GOODBYE_SET: [&str; 3] = ["cx", "cz", "ecr"];

fn get_target_basis_set(target: &Target, qubit: PhysicalQubit) -> EulerBasisSet {
    let mut target_basis_set: EulerBasisSet = EulerBasisSet::new();
    let target_basis_list = target.operation_names_for_qargs(Some(&smallvec![qubit]));
    match target_basis_list {
        Ok(basis_list) => {
            EULER_BASES
                .iter()
                .enumerate()
                .filter_map(|(idx, gates)| {
                    if !gates.iter().all(|gate| basis_list.contains(gate)) {
                        return None;
                    }
                    let basis = EULER_BASIS_NAMES[idx];
                    Some(basis)
                })
                .for_each(|basis| target_basis_set.add_basis(basis));
        }
        Err(_) => target_basis_set.support_all(),
    }
    if target_basis_set.basis_supported(EulerBasis::U3)
        && target_basis_set.basis_supported(EulerBasis::U321)
    {
        target_basis_set.remove(EulerBasis::U3);
    }
    if target_basis_set.basis_supported(EulerBasis::ZSX)
        && target_basis_set.basis_supported(EulerBasis::ZSXX)
    {
        target_basis_set.remove(EulerBasis::ZSX);
    }
    target_basis_set
}

fn apply_synth_dag(
    py: Python<'_>,
    out_dag: &mut DAGCircuit,
    out_qargs: &[Qubit],
    synth_dag: &DAGCircuit,
) -> PyResult<()> {
    for out_node in synth_dag.topological_op_nodes()? {
        let mut out_packed_instr = synth_dag.dag()[out_node].unwrap_operation().clone();
        let synth_qargs = synth_dag.get_qargs(out_packed_instr.qubits);
        let mapped_qargs: Vec<Qubit> = synth_qargs
            .iter()
            .map(|qarg| out_qargs[qarg.0 as usize])
            .collect();
        out_packed_instr.qubits = out_dag.qargs_interner.insert(&mapped_qargs);
        out_dag.push_back(py, out_packed_instr)?;
    }
    out_dag.add_global_phase(py, &synth_dag.get_global_phase())?;
    Ok(())
}

fn apply_synth_sequence(
    py: Python<'_>,
    out_dag: &mut DAGCircuit,
    out_qargs: &[Qubit],
    sequence: &TwoQubitUnitarySequence,
) -> PyResult<()> {
    let mut instructions = Vec::with_capacity(sequence.gate_sequence.gates().len());
    for (gate, params, qubit_ids) in sequence.gate_sequence.gates() {
        let gate_node = match gate {
            None => sequence.decomp_gate.operation.standard_gate(),
            Some(gate) => *gate,
        };
        let mapped_qargs: Vec<Qubit> = qubit_ids.iter().map(|id| out_qargs[*id as usize]).collect();
        let new_params: Option<Box<SmallVec<[Param; 3]>>> = match gate {
            Some(_) => Some(Box::new(params.iter().map(|p| Param::Float(*p)).collect())),
            None => Some(Box::new(sequence.decomp_gate.params.clone())),
        };
        let instruction = PackedInstruction {
            op: PackedOperation::from_standard(gate_node),
            qubits: out_dag.qargs_interner.insert(&mapped_qargs),
            clbits: out_dag.cargs_interner.get_default(),
            params: new_params,
            extra_attrs: ExtraInstructionAttributes::default(),
            #[cfg(feature = "cache_pygates")]
            py_op: OnceLock::new(),
        };
        instructions.push(instruction);
    }
    out_dag.extend(py, instructions.into_iter())?;
    out_dag.add_global_phase(py, &Param::Float(sequence.gate_sequence.global_phase()))?;
    Ok(())
}

fn synth_error(
    py: Python<'_>,
    synth_circuit: impl Iterator<
        Item = (
            String,
            Option<SmallVec<[Param; 3]>>,
            SmallVec<[PhysicalQubit; 2]>,
        ),
    >,
    target: &Target,
) -> f64 {
    let (lower_bound, upper_bound) = synth_circuit.size_hint();
    let mut gate_fidelities = match upper_bound {
        Some(bound) => Vec::with_capacity(bound),
        None => Vec::with_capacity(lower_bound),
    };
    let mut score_instruction =
        |inst_name: &str,
         inst_params: &Option<SmallVec<[Param; 3]>>,
         inst_qubits: &SmallVec<[PhysicalQubit; 2]>| {
            if let Ok(names) = target.operation_names_for_qargs(Some(inst_qubits)) {
                for name in names {
                    if let Ok(target_op) = target.operation_from_name(name) {
                        let are_params_close = if let Some(params) = inst_params {
                            params.iter().zip(target_op.params.iter()).all(|(p1, p2)| {
                                p1.is_close(py, p2, 1e-10)
                                    .expect("Unexpected parameter expression error.")
                            })
                        } else {
                            false
                        };
                        let is_parametrized = target_op
                            .params
                            .iter()
                            .any(|param| matches!(param, Param::ParameterExpression(_)));
                        if target_op.operation.name() == inst_name
                            && (is_parametrized || are_params_close)
                        {
                            match target[name].get(Some(inst_qubits)) {
                                Some(Some(props)) => {
                                    gate_fidelities.push(1.0 - props.error.unwrap_or(0.0))
                                }
                                _ => gate_fidelities.push(1.0),
                            }
                            break;
                        }
                    }
                }
            }
        };

    for (inst_name, inst_params, inst_qubits) in synth_circuit {
        score_instruction(&inst_name, &inst_params, &inst_qubits);
    }
    1.0 - gate_fidelities.into_iter().product::<f64>()
}

// This is the outer-most run function. It is meant to be called from Python
// in `UnitarySynthesis.run()`.
#[pyfunction]
#[pyo3(name = "run_default_main_loop", signature=(dag, qubit_indices, min_qubits, target, coupling_edges, approximation_degree=None, natural_direction=None))]
fn py_run_main_loop(
    py: Python,
    dag: &mut DAGCircuit,
    qubit_indices: Vec<usize>,
    min_qubits: usize,
    target: &Target,
    coupling_edges: HashSet<[PhysicalQubit; 2]>,
    approximation_degree: Option<f64>,
    natural_direction: Option<bool>,
) -> PyResult<DAGCircuit> {
    // We need to use the python converter because the currently available Rust conversion
    // is lossy. We need `QuantumCircuit` instances to be used in `replace_blocks`.
    let dag_to_circuit = imports::DAG_TO_CIRCUIT.get_bound(py);

    let mut out_dag = dag.copy_empty_like(py, "alike")?;

    // Iterate over dag nodes and determine unitary synthesis approach
    for node in dag.topological_op_nodes()? {
        let mut packed_instr = dag.dag()[node].unwrap_operation().clone();

        if packed_instr.op.control_flow() {
            let OperationRef::Instruction(py_instr) = packed_instr.op.view() else {
                unreachable!("Control flow op must be an instruction")
            };
            let raw_blocks: Vec<PyResult<Bound<PyAny>>> = py_instr
                .instruction
                .getattr(py, "blocks")?
                .bind(py)
                .try_iter()?
                .collect();
            let mut new_blocks = Vec::with_capacity(raw_blocks.len());
            for raw_block in raw_blocks {
                let new_ids = dag
                    .get_qargs(packed_instr.qubits)
                    .iter()
                    .map(|qarg| qubit_indices[qarg.0 as usize])
                    .collect_vec();
                let res = py_run_main_loop(
                    py,
                    &mut circuit_to_dag(
                        py,
                        QuantumCircuitData::extract_bound(&raw_block?)?,
                        false,
                        None,
                        None,
                    )?,
                    new_ids,
                    min_qubits,
                    target,
                    coupling_edges.clone(),
                    approximation_degree,
                    natural_direction,
                )?;
                new_blocks.push(dag_to_circuit.call1((res,))?);
            }
            let new_node = py_instr
                .instruction
                .bind(py)
                .call_method1("replace_blocks", (new_blocks,))?;
            let new_node_op: OperationFromPython = new_node.extract()?;
            packed_instr = PackedInstruction {
                op: new_node_op.operation,
                qubits: packed_instr.qubits,
                clbits: packed_instr.clbits,
                params: (!new_node_op.params.is_empty()).then(|| Box::new(new_node_op.params)),
                extra_attrs: new_node_op.extra_attrs,
                #[cfg(feature = "cache_pygates")]
                py_op: new_node.unbind().into(),
            };
        }
        if !(packed_instr.op.name() == "unitary"
            && packed_instr.op.num_qubits() >= min_qubits as u32)
        {
            out_dag.push_back(py, packed_instr)?;
            continue;
        }
        let unitary: Array<Complex<f64>, Dim<[usize; 2]>> = match packed_instr.op.matrix(&[]) {
            Some(unitary) => unitary,
            None => return Err(QiskitError::new_err("Unitary not found")),
        };
        match unitary.shape() {
            // Run 1q synthesis
            [2, 2] => {
                let qubit = dag.get_qargs(packed_instr.qubits)[0];
                let target_basis_set = get_target_basis_set(target, PhysicalQubit::new(qubit.0));
                let sequence = unitary_to_gate_sequence_inner(
                    unitary.view(),
                    &target_basis_set,
                    qubit.0 as usize,
                    None,
                    true,
                    None,
                );
                match sequence {
                    Some(sequence) => {
                        for (gate, params) in sequence.gates {
                            let new_params: SmallVec<[Param; 3]> =
                                params.iter().map(|p| Param::Float(*p)).collect();
                            out_dag.apply_operation_back(
                                py,
                                gate.into(),
                                &[qubit],
                                &[],
                                Some(new_params),
                                ExtraInstructionAttributes::default(),
                                #[cfg(feature = "cache_pygates")]
                                None,
                            )?;
                        }
                        out_dag.add_global_phase(py, &Param::Float(sequence.global_phase))?;
                    }
                    None => {
                        out_dag.push_back(py, packed_instr)?;
                    }
                }
            }
            // Run 2q synthesis
            [4, 4] => {
                // "out_qargs" is used to append the synthesized instructions to the output dag
                let out_qargs = dag.get_qargs(packed_instr.qubits);
                // "ref_qubits" is used to access properties in the target. It accounts for control flow mapping.
                let ref_qubits: &[PhysicalQubit; 2] = &[
                    PhysicalQubit::new(qubit_indices[out_qargs[0].0 as usize] as u32),
                    PhysicalQubit::new(qubit_indices[out_qargs[1].0 as usize] as u32),
                ];
                let apply_original_op = |out_dag: &mut DAGCircuit| -> PyResult<()> {
                    out_dag.push_back(py, packed_instr.clone())?;
                    Ok(())
                };
                run_2q_unitary_synthesis(
                    py,
                    unitary,
                    ref_qubits,
                    &coupling_edges,
                    target,
                    approximation_degree,
                    natural_direction,
                    &mut out_dag,
                    out_qargs,
                    apply_original_op,
                )?;
            }
            // Run 3q+ synthesis
            _ => {
                let qs_decomposition: &Bound<'_, PyAny> = imports::QS_DECOMPOSITION.get_bound(py);
                let synth_circ = qs_decomposition.call1((unitary.into_pyarray(py),))?;
                let synth_dag = circuit_to_dag(
                    py,
                    QuantumCircuitData::extract_bound(&synth_circ)?,
                    false,
                    None,
                    None,
                )?;
                let out_qargs = dag.get_qargs(packed_instr.qubits);
                apply_synth_dag(py, &mut out_dag, out_qargs, &synth_dag)?;
            }
        }
    }
    Ok(out_dag)
}

fn run_2q_unitary_synthesis(
    py: Python,
    unitary: Array2<Complex64>,
    ref_qubits: &[PhysicalQubit; 2],
    coupling_edges: &HashSet<[PhysicalQubit; 2]>,
    target: &Target,
    approximation_degree: Option<f64>,
    natural_direction: Option<bool>,
    out_dag: &mut DAGCircuit,
    out_qargs: &[Qubit],
    mut apply_original_op: impl FnMut(&mut DAGCircuit) -> PyResult<()>,
) -> PyResult<()> {
    let decomposers = {
        let decomposers_2q =
            get_2q_decomposers_from_target(py, target, ref_qubits, approximation_degree)?;
        decomposers_2q.unwrap_or_default()
    };
    // If there's a single decomposer, avoid computing synthesis score
    if decomposers.len() == 1 {
        let decomposer_item = decomposers.first().unwrap();
        let preferred_dir = preferred_direction(
            decomposer_item,
            ref_qubits,
            natural_direction,
            coupling_edges,
            target,
        )?;
        match decomposer_item.decomposer {
            DecomposerType::TwoQubitBasisDecomposer(_) => {
                let synth = synth_su4_sequence(
                    &unitary,
                    decomposer_item,
                    preferred_dir,
                    approximation_degree,
                )?;
                apply_synth_sequence(py, out_dag, out_qargs, &synth)?;
            }
            DecomposerType::XXDecomposer(_) => {
                let synth = synth_su4_dag(
                    py,
                    &unitary,
                    decomposer_item,
                    preferred_dir,
                    approximation_degree,
                )?;
                apply_synth_dag(py, out_dag, out_qargs, &synth)?;
            }
        }
        return Ok(());
    }

    let mut synth_errors_sequence = Vec::new();
    let mut synth_errors_dag = Vec::new();
    for decomposer in &decomposers {
        let preferred_dir = preferred_direction(
            decomposer,
            ref_qubits,
            natural_direction,
            coupling_edges,
            target,
        )?;
        match &decomposer.decomposer {
            DecomposerType::TwoQubitBasisDecomposer(_) => {
                let sequence =
                    synth_su4_sequence(&unitary, decomposer, preferred_dir, approximation_degree)?;
                let scoring_info =
                    sequence
                        .gate_sequence
                        .gates()
                        .iter()
                        .map(|(gate, params, qubit_ids)| {
                            let inst_qubits =
                                qubit_ids.iter().map(|q| ref_qubits[*q as usize]).collect();
                            match gate {
                                Some(gate) => (
                                    gate.name().to_string(),
                                    Some(params.iter().map(|p| Param::Float(*p)).collect()),
                                    inst_qubits,
                                ),
                                None => (
                                    sequence
                                        .decomp_gate
                                        .operation
                                        .standard_gate()
                                        .name()
                                        .to_string(),
                                    Some(params.iter().map(|p| Param::Float(*p)).collect()),
                                    inst_qubits,
                                ),
                            }
                        });
                let synth_error_from_target = synth_error(py, scoring_info, target);
                synth_errors_sequence.push((sequence, synth_error_from_target));
            }
            DecomposerType::XXDecomposer(_) => {
                let synth_dag = synth_su4_dag(
                    py,
                    &unitary,
                    decomposer,
                    preferred_dir,
                    approximation_degree,
                )?;
                let scoring_info = synth_dag
                    .topological_op_nodes()
                    .expect("Unexpected error in dag.topological_op_nodes()")
                    .map(|node| {
                        let NodeType::Operation(inst) = &synth_dag.dag()[node] else {
                            unreachable!("DAG node must be an instruction")
                        };
                        let inst_qubits = synth_dag
                            .get_qargs(inst.qubits)
                            .iter()
                            .map(|q| ref_qubits[q.0 as usize])
                            .collect();
                        (
                            inst.op.name().to_string(),
                            inst.params.clone().map(|boxed| *boxed),
                            inst_qubits,
                        )
                    });
                let synth_error_from_target = synth_error(py, scoring_info, target);
                synth_errors_dag.push((synth_dag, synth_error_from_target));
            }
        }
    }

    let synth_sequence = synth_errors_sequence
        .iter()
        .enumerate()
        .min_by(|error1, error2| error1.1 .1.partial_cmp(&error2.1 .1).unwrap())
        .map(|(index, _)| &synth_errors_sequence[index]);

    let synth_dag = synth_errors_dag
        .iter()
        .enumerate()
        .min_by(|error1, error2| error1.1 .1.partial_cmp(&error2.1 .1).unwrap())
        .map(|(index, _)| &synth_errors_dag[index]);

    match (synth_sequence, synth_dag) {
        (None, None) => apply_original_op(out_dag)?,
        (Some((sequence, _)), None) => apply_synth_sequence(py, out_dag, out_qargs, sequence)?,
        (None, Some((dag, _))) => apply_synth_dag(py, out_dag, out_qargs, dag)?,
        (Some((sequence, sequence_error)), Some((dag, dag_error))) => {
            if sequence_error > dag_error {
                apply_synth_dag(py, out_dag, out_qargs, dag)?
            } else {
                apply_synth_sequence(py, out_dag, out_qargs, sequence)?
            }
        }
    };
    Ok(())
}

fn get_2q_decomposers_from_target(
    py: Python,
    target: &Target,
    qubits: &[PhysicalQubit; 2],
    approximation_degree: Option<f64>,
) -> PyResult<Option<Vec<DecomposerElement>>> {
    let qubits: SmallVec<[PhysicalQubit; 2]> = SmallVec::from_buf(*qubits);
    let reverse_qubits: SmallVec<[PhysicalQubit; 2]> = qubits.iter().rev().copied().collect();
    let mut available_2q_basis: IndexMap<&str, NormalOperation> = IndexMap::new();
    let mut available_2q_props: IndexMap<&str, (Option<f64>, Option<f64>)> = IndexMap::new();

    let mut qubit_gate_map = IndexMap::new();

    match target.operation_names_for_qargs(Some(&qubits)) {
        Ok(direct_keys) => {
            qubit_gate_map.insert(&qubits, direct_keys);
            if let Ok(reverse_keys) = target.operation_names_for_qargs(Some(&reverse_qubits)) {
                qubit_gate_map.insert(&reverse_qubits, reverse_keys);
            }
        }
        Err(_) => {
            if let Ok(reverse_keys) = target.operation_names_for_qargs(Some(&reverse_qubits)) {
                qubit_gate_map.insert(&reverse_qubits, reverse_keys);
            } else {
                return Err(QiskitError::new_err(
                    "Target has no gates available on qubits to synthesize over.",
                ));
            }
        }
    }

    #[inline]
    fn replace_parametrized_gate(mut op: NormalOperation) -> NormalOperation {
        if let Some(std_gate) = op.operation.try_standard_gate() {
            match std_gate.name() {
                "rxx" => {
                    if let Param::ParameterExpression(_) = op.params[0] {
                        op.params[0] = Param::Float(PI2)
                    }
                }
                "rzx" => {
                    if let Param::ParameterExpression(_) = op.params[0] {
                        op.params[0] = Param::Float(PI4)
                    }
                }
                "rzz" => {
                    if let Param::ParameterExpression(_) = op.params[0] {
                        op.params[0] = Param::Float(PI2)
                    }
                }
                _ => (),
            }
        }
        op
    }

    for (q_pair, gates) in qubit_gate_map {
        for key in gates {
            match target.operation_from_name(key) {
                Ok(op) => {
                    match op.operation.view() {
                        OperationRef::Gate(_) => (),
                        OperationRef::Standard(_) => (),
                        _ => continue,
                    }
                    // Filter out non-2q-gate candidates
                    if op.operation.num_qubits() != 2 {
                        continue;
                    }
                    available_2q_basis.insert(key, replace_parametrized_gate(op.clone()));

                    if target.contains_key(key) {
                        available_2q_props.insert(
                            key,
                            match &target[key].get(Some(q_pair)) {
                                Some(Some(props)) => (props.duration, props.error),
                                _ => (None, None),
                            },
                        );
                    } else {
                        continue;
                    }
                }
                _ => continue,
            }
        }
    }
    if available_2q_basis.is_empty() {
        return Err(QiskitError::new_err(
            "Target has no gates available on qubits to synthesize over.",
        ));
    }

    let target_basis_set = get_target_basis_set(target, qubits[0]);
    let available_1q_basis: IndexSet<&str> =
        IndexSet::from_iter(target_basis_set.get_bases().map(|basis| basis.as_str()));
    let mut decomposers: Vec<DecomposerElement> = Vec::new();

    #[inline]
    fn is_supercontrolled(op: &NormalOperation) -> bool {
        match op.operation.matrix(&op.params) {
            None => false,
            Some(unitary_matrix) => {
                let kak = TwoQubitWeylDecomposition::new_inner(unitary_matrix.view(), None, None)
                    .unwrap();
                relative_eq!(kak.a(), PI4) && relative_eq!(kak.c(), 0.0)
            }
        }
    }

    #[inline]
    fn is_controlled(op: &NormalOperation) -> bool {
        match op.operation.matrix(&op.params) {
            None => false,
            Some(unitary_matrix) => {
                let kak = TwoQubitWeylDecomposition::new_inner(unitary_matrix.view(), None, None)
                    .unwrap();
                relative_eq!(kak.b(), 0.0) && relative_eq!(kak.c(), 0.0)
            }
        }
    }

    // Iterate over 1q and 2q supercontrolled basis, append TwoQubitBasisDecomposers
    let supercontrolled_basis: IndexMap<&str, NormalOperation> = available_2q_basis
        .iter()
        .filter(|(_, v)| is_supercontrolled(v))
        .map(|(k, v)| (*k, v.clone()))
        .collect();

    for basis_1q in &available_1q_basis {
        for (basis_2q, gate) in supercontrolled_basis.iter() {
            let mut basis_2q_fidelity: f64 = match available_2q_props.get(basis_2q) {
                Some(&(_, Some(e))) => 1.0 - e,
                _ => 1.0,
            };
            if let Some(approx_degree) = approximation_degree {
                basis_2q_fidelity *= approx_degree;
            }
            let decomposer = TwoQubitBasisDecomposer::new_inner(
                gate.operation.name().to_string(),
                gate.operation.matrix(&gate.params).unwrap().view(),
                basis_2q_fidelity,
                basis_1q,
                None,
            )?;

            decomposers.push(DecomposerElement {
                decomposer: DecomposerType::TwoQubitBasisDecomposer(Box::new(decomposer)),
                gate: gate.clone(),
            });
        }
    }

    // If our 2q basis gates are a subset of cx, ecr, or cz then we know TwoQubitBasisDecomposer
    // is an ideal decomposition and there is no need to bother calculating the XX embodiments
    // or try the XX decomposer
    let available_basis_set: IndexSet<&str> = available_2q_basis.keys().copied().collect();

    #[inline]
    fn check_goodbye(basis_set: &IndexSet<&str>) -> bool {
        basis_set.iter().all(|gate| GOODBYE_SET.contains(gate))
    }

    if check_goodbye(&available_basis_set) {
        return Ok(Some(decomposers));
    }

    // Let's now look for possible controlled decomposers (i.e. XXDecomposer)
    let controlled_basis: IndexMap<&str, NormalOperation> = available_2q_basis
        .iter()
        .filter(|(_, v)| is_controlled(v))
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    let mut pi2_basis: Option<&str> = None;
    let xx_embodiments: &Bound<'_, PyAny> = imports::XX_EMBODIMENTS.get_bound(py);

    // The xx decomposer args are the interaction strength (f64), basis_2q_fidelity (f64),
    // and embodiments (Bound<'_, PyAny>).
    let xx_decomposer_args = controlled_basis.iter().map(
        |(name, op)| -> PyResult<(f64, f64, pyo3::Bound<'_, pyo3::PyAny>)> {
            let strength = 2.0
                * TwoQubitWeylDecomposition::new_inner(
                    op.operation.matrix(&op.params).unwrap().view(),
                    None,
                    None,
                )
                .unwrap()
                .a();
            let mut fidelity_value = match available_2q_props.get(name) {
                Some(&(_, error)) => 1.0 - error.unwrap_or_default(), // default is 0.0
                None => 1.0,
            };
            if let Some(approx_degree) = approximation_degree {
                fidelity_value *= approx_degree;
            }
            let mut embodiment =
                xx_embodiments.get_item(op.into_pyobject(py)?.getattr("base_class")?)?;

            if embodiment.getattr("parameters")?.len()? == 1 {
                embodiment = embodiment.call_method1("assign_parameters", (vec![strength],))?;
            }
            // basis equivalent to CX are well optimized so use for the pi/2 angle if available
            if relative_eq!(strength, PI2) && supercontrolled_basis.contains_key(name) {
                pi2_basis = Some(op.operation.name());
            }
            Ok((strength, fidelity_value, embodiment))
        },
    );

    let basis_2q_fidelity_dict = PyDict::new(py);
    let embodiments_dict = PyDict::new(py);
    for (strength, fidelity, embodiment) in xx_decomposer_args.flatten() {
        basis_2q_fidelity_dict.set_item(strength, fidelity)?;
        embodiments_dict.set_item(strength, embodiment)?;
    }

    // Iterate over 2q fidelities and select decomposers
    if basis_2q_fidelity_dict.len() > 0 {
        let xx_decomposer: &Bound<'_, PyAny> = imports::XX_DECOMPOSER.get_bound(py);
        for basis_1q in available_1q_basis {
            let pi2_decomposer = if let Some(pi_2_basis) = pi2_basis {
                if pi_2_basis == "cx" && basis_1q == "ZSX" {
                    let fidelity = match approximation_degree {
                        Some(approx_degree) => approx_degree,
                        None => match &target["cx"][Some(&qubits)] {
                            Some(props) => 1.0 - props.error.unwrap_or_default(),
                            None => 1.0,
                        },
                    };
                    Some(TwoQubitBasisDecomposer::new_inner(
                        pi_2_basis.to_string(),
                        StandardGate::CXGate.matrix(&[]).unwrap().view(),
                        fidelity,
                        basis_1q,
                        Some(true),
                    )?)
                } else {
                    None
                }
            } else {
                None
            };

            let decomposer = xx_decomposer.call1((
                &basis_2q_fidelity_dict,
                PyString::new(py, basis_1q),
                &embodiments_dict,
                pi2_decomposer,
            ))?;
            let decomposer_gate = decomposer
                .getattr(intern!(py, "gate"))?
                .extract::<NormalOperation>()?;

            decomposers.push(DecomposerElement {
                decomposer: DecomposerType::XXDecomposer(decomposer.into()),
                gate: decomposer_gate,
            });
        }
    }
    Ok(Some(decomposers))
}

fn preferred_direction(
    decomposer: &DecomposerElement,
    ref_qubits: &[PhysicalQubit; 2],
    natural_direction: Option<bool>,
    coupling_edges: &HashSet<[PhysicalQubit; 2]>,
    target: &Target,
) -> PyResult<Option<bool>> {
    // Returns:
    // * true if gate qubits are in the hardware-native direction
    // * false if gate qubits must be flipped to match hardware-native direction
    let qubits: [PhysicalQubit; 2] = *ref_qubits;
    let mut reverse_qubits: [PhysicalQubit; 2] = qubits;
    reverse_qubits.reverse();

    let compute_cost =
        |lengths: bool, q_tuple: [PhysicalQubit; 2], in_cost: f64| -> PyResult<f64> {
            let cost = match target.qargs_for_operation_name(decomposer.gate.operation.name()) {
                Ok(_) => match target[decomposer.gate.operation.name()].get(Some(
                    &q_tuple
                        .into_iter()
                        .collect::<SmallVec<[PhysicalQubit; 2]>>(),
                )) {
                    Some(Some(_props)) => {
                        if lengths {
                            _props.duration.unwrap_or(in_cost)
                        } else {
                            _props.error.unwrap_or(in_cost)
                        }
                    }
                    _ => in_cost,
                },
                Err(_) => in_cost,
            };
            Ok(cost)
        };

    let preferred_direction = match natural_direction {
        Some(false) => None,
        _ => {
            // None or Some(true)
            let zero_one = coupling_edges.contains(&qubits);
            let one_zero = coupling_edges.contains(&[qubits[1], qubits[0]]);

            match (zero_one, one_zero) {
                (true, false) => Some(true),
                (false, true) => Some(false),
                _ => {
                    let mut cost_0_1: f64 = f64::INFINITY;
                    let mut cost_1_0: f64 = f64::INFINITY;

                    // Try to find the cost in gate_lengths
                    cost_0_1 = compute_cost(true, qubits, cost_0_1)?;
                    cost_1_0 = compute_cost(true, reverse_qubits, cost_1_0)?;

                    // If no valid cost was found in gate_lengths, check gate_errors
                    if !(cost_0_1 < f64::INFINITY || cost_1_0 < f64::INFINITY) {
                        cost_0_1 = compute_cost(false, qubits, cost_0_1)?;
                        cost_1_0 = compute_cost(false, reverse_qubits, cost_1_0)?;
                    }

                    if cost_0_1 < cost_1_0 {
                        Some(true)
                    } else if cost_1_0 < cost_0_1 {
                        Some(false)
                    } else {
                        None
                    }
                }
            }
        }
    };

    if natural_direction == Some(true) && preferred_direction.is_none() {
        return Err(QiskitError::new_err(format!(
            concat!(
                "No preferred direction of gate on qubits {:?} ",
                "could be determined from coupling map or gate lengths / gate errors."
            ),
            qubits
        )));
    }

    Ok(preferred_direction)
}

fn synth_su4_sequence(
    su4_mat: &Array2<Complex64>,
    decomposer_2q: &DecomposerElement,
    preferred_direction: Option<bool>,
    approximation_degree: Option<f64>,
) -> PyResult<TwoQubitUnitarySequence> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    let synth = if let DecomposerType::TwoQubitBasisDecomposer(decomp) = &decomposer_2q.decomposer {
        decomp.call_inner(su4_mat.view(), None, is_approximate, None)?
    } else {
        unreachable!("synth_su4_sequence should only be called for TwoQubitBasisDecomposer.")
    };
    let sequence = TwoQubitUnitarySequence {
        gate_sequence: synth,
        decomp_gate: decomposer_2q.gate.clone(),
    };

    match preferred_direction {
        None => Ok(sequence),
        Some(preferred_dir) => {
            let mut synth_direction: Option<SmallVec<[u8; 2]>> = None;
            // if the gates in synthesis are in the opposite direction of the preferred direction
            // resynthesize a new operator which is the original conjugated by swaps.
            // this new operator is doubly mirrored from the original and is locally equivalent.
            for (gate, _, qubits) in sequence.gate_sequence.gates() {
                if gate.is_none() || gate.unwrap().name() == "cx" {
                    synth_direction = Some(qubits.clone());
                }
            }

            match synth_direction {
                None => Ok(sequence),
                Some(synth_direction) => {
                    let synth_dir = match synth_direction.as_slice() {
                        [0, 1] => true,
                        [1, 0] => false,
                        _ => unreachable!(),
                    };
                    if synth_dir != preferred_dir {
                        reversed_synth_su4_sequence(
                            su4_mat.clone(),
                            decomposer_2q,
                            approximation_degree,
                        )
                    } else {
                        Ok(sequence)
                    }
                }
            }
        }
    }
}

fn reversed_synth_su4_sequence(
    mut su4_mat: Array2<Complex64>,
    decomposer_2q: &DecomposerElement,
    approximation_degree: Option<f64>,
) -> PyResult<TwoQubitUnitarySequence> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    // Swap rows 1 and 2
    let (mut row_1, mut row_2) = su4_mat.multi_slice_mut((s![1, ..], s![2, ..]));
    azip!((x in &mut row_1, y in &mut row_2) (*x, *y) = (*y, *x));

    // Swap columns 1 and 2
    let (mut col_1, mut col_2) = su4_mat.multi_slice_mut((s![.., 1], s![.., 2]));
    azip!((x in &mut col_1, y in &mut col_2) (*x, *y) = (*y, *x));

    let synth = if let DecomposerType::TwoQubitBasisDecomposer(decomp) = &decomposer_2q.decomposer {
        decomp.call_inner(su4_mat.view(), None, is_approximate, None)?
    } else {
        unreachable!(
            "reversed_synth_su4_sequence should only be called for TwoQubitBasisDecomposer."
        )
    };

    let flip_bits: [u8; 2] = [1, 0];
    let mut reversed_gates = Vec::with_capacity(synth.gates().len());
    for (gate, params, qubit_ids) in synth.gates() {
        let new_qubit_ids = qubit_ids
            .into_iter()
            .map(|x| flip_bits[*x as usize])
            .collect::<SmallVec<[u8; 2]>>();
        reversed_gates.push((*gate, params.clone(), new_qubit_ids.clone()));
    }

    let mut reversed_synth: TwoQubitGateSequence = TwoQubitGateSequence::new();
    reversed_synth.set_state((reversed_gates, synth.global_phase()));
    let sequence = TwoQubitUnitarySequence {
        gate_sequence: reversed_synth,
        decomp_gate: decomposer_2q.gate.clone(),
    };
    Ok(sequence)
}

fn synth_su4_dag(
    py: Python,
    su4_mat: &Array2<Complex64>,
    decomposer_2q: &DecomposerElement,
    preferred_direction: Option<bool>,
    approximation_degree: Option<f64>,
) -> PyResult<DAGCircuit> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;
    let synth_dag = if let DecomposerType::XXDecomposer(decomposer) = &decomposer_2q.decomposer {
        let kwargs: HashMap<&str, bool> = [("approximate", is_approximate), ("use_dag", true)]
            .into_iter()
            .collect();
        decomposer
            .call(
                py,
                (su4_mat.clone().into_pyarray(py),),
                Some(&kwargs.into_py_dict(py)?),
            )?
            .extract::<DAGCircuit>(py)?
    } else {
        unreachable!("synth_su4_dag should only be called for XXDecomposer.")
    };

    match preferred_direction {
        None => Ok(synth_dag),
        Some(preferred_dir) => {
            let mut synth_direction: Option<Vec<u32>> = None;
            for node in synth_dag.topological_op_nodes()? {
                let inst = &synth_dag.dag()[node].unwrap_operation();
                if inst.op.num_qubits() == 2 {
                    let qargs = synth_dag.get_qargs(inst.qubits);
                    synth_direction = Some(vec![qargs[0].0, qargs[1].0]);
                }
            }
            match synth_direction {
                None => Ok(synth_dag),
                Some(synth_direction) => {
                    let synth_dir = match synth_direction.as_slice() {
                        [0, 1] => true,
                        [1, 0] => false,
                        _ => unreachable!("There are no more than 2 possible synth directions."),
                    };
                    if synth_dir != preferred_dir {
                        reversed_synth_su4_dag(
                            py,
                            su4_mat.clone(),
                            decomposer_2q,
                            approximation_degree,
                        )
                    } else {
                        Ok(synth_dag)
                    }
                }
            }
        }
    }
}

fn reversed_synth_su4_dag(
    py: Python<'_>,
    mut su4_mat: Array2<Complex64>,
    decomposer_2q: &DecomposerElement,
    approximation_degree: Option<f64>,
) -> PyResult<DAGCircuit> {
    let is_approximate = approximation_degree.is_none() || approximation_degree.unwrap() != 1.0;

    // Swap rows 1 and 2
    let (mut row_1, mut row_2) = su4_mat.multi_slice_mut((s![1, ..], s![2, ..]));
    azip!((x in &mut row_1, y in &mut row_2) (*x, *y) = (*y, *x));

    // Swap columns 1 and 2
    let (mut col_1, mut col_2) = su4_mat.multi_slice_mut((s![.., 1], s![.., 2]));
    azip!((x in &mut col_1, y in &mut col_2) (*x, *y) = (*y, *x));

    let synth_dag = if let DecomposerType::XXDecomposer(decomposer) = &decomposer_2q.decomposer {
        let kwargs: HashMap<&str, bool> = [("approximate", is_approximate), ("use_dag", true)]
            .into_iter()
            .collect();
        decomposer
            .call(
                py,
                (su4_mat.clone().into_pyarray(py),),
                Some(&kwargs.into_py_dict(py)?),
            )?
            .extract::<DAGCircuit>(py)?
    } else {
        unreachable!("reversed_synth_su4_dag should only be called for XXDecomposer")
    };

    let mut target_dag = synth_dag.copy_empty_like(py, "alike")?;
    let flip_bits: [Qubit; 2] = [Qubit(1), Qubit(0)];
    for node in synth_dag.topological_op_nodes()? {
        let mut inst = synth_dag.dag()[node].unwrap_operation().clone();
        let qubits: Vec<Qubit> = synth_dag
            .qargs_interner()
            .get(inst.qubits)
            .iter()
            .map(|x| flip_bits[x.0 as usize])
            .collect();
        inst.qubits = target_dag.qargs_interner.insert_owned(qubits);
        target_dag.push_back(py, inst)?;
    }
    Ok(target_dag)
}

#[pymodule]
pub fn unitary_synthesis(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_main_loop))?;
    Ok(())
}

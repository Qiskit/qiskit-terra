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

use std::cmp::Ordering;
use std::sync::Mutex;

use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use rayon::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::{smallvec, SmallVec};

use qiskit_circuit::circuit_instruction::ExtraInstructionAttributes;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::Qubit;

use crate::convert_2q_block_matrix::blocks_to_matrix;
use crate::euler_one_qubit_decomposer::{
    EulerBasis, EulerBasisSet, EULER_BASES, EULER_BASIS_NAMES,
};
use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::Target;
use crate::two_qubit_decompose::{
    RXXEquivalent, TwoQubitBasisDecomposer, TwoQubitControlledUDecomposer, TwoQubitGateSequence,
};

// The difference between these two types is large where TwoQubitBasisDecomposer
// is 1640 bytes and TwoQubitControlledUDecomposer is only 24 bytes. This means
// each element of ControlledU is wasting > 1600 bytes but that is acceptable in
// this case to avoid the layer of pointer indirection as these are stored
// temporarily in a vec inside a thread to decompose a unitary.
#[allow(clippy::large_enum_variant)]
enum TwoQubitDecomposer {
    Basis(TwoQubitBasisDecomposer),
    ControlledU(TwoQubitControlledUDecomposer),
}

fn get_decomposers_from_target(
    target: &Target,
    qubits: &[Qubit],
    fidelity: f64,
) -> PyResult<Vec<TwoQubitDecomposer>> {
    let physical_qubits = smallvec![PhysicalQubit(qubits[0].0), PhysicalQubit(qubits[1].0)];
    let gate_names = match target.operation_names_for_qargs(Some(&physical_qubits)) {
        Ok(names) => names,
        Err(_) => {
            let reverse_qubits = physical_qubits.iter().rev().copied().collect();
            target
                .operation_names_for_qargs(Some(&reverse_qubits))
                .unwrap()
        }
    };

    let available_kak_gate: Vec<(&str, &PackedOperation, &[Param])> = gate_names
        .iter()
        .filter_map(|name| match target.operation_from_name(name) {
            Ok(raw_op) => match raw_op.operation.view() {
                OperationRef::Standard(_) | OperationRef::Gate(_) => {
                    Some((*name, &raw_op.operation, raw_op.params.as_slice()))
                }
                _ => None,
            },
            Err(_) => None,
        })
        .collect();

    let single_qubit_basis_list =
        target.operation_names_for_qargs(Some(&smallvec![physical_qubits[0]]));
    let mut target_basis_set = EulerBasisSet::new();
    match single_qubit_basis_list {
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

    let euler_bases: Vec<EulerBasis> = target_basis_set.get_bases().collect();

    let decomposers: PyResult<Vec<TwoQubitDecomposer>> = available_kak_gate
        .iter()
        .filter_map(|(two_qubit_name, two_qubit_gate, params)| {
            let matrix = two_qubit_gate.matrix(params);
            matrix.map(|matrix| {
                euler_bases.iter().map(move |euler_basis| {
                    TwoQubitBasisDecomposer::new_inner(
                        two_qubit_name.to_string(),
                        matrix.view(),
                        fidelity,
                        *euler_basis,
                        None,
                    )
                    .map(TwoQubitDecomposer::Basis)
                })
            })
        })
        .flatten()
        .collect();
    let mut decomposers = decomposers?;
    for gate in [
        StandardGate::RXXGate,
        StandardGate::RZZGate,
        StandardGate::RYYGate,
        StandardGate::RZXGate,
    ] {
        if gate_names.contains(gate.name()) {
            decomposers.push(TwoQubitDecomposer::ControlledU(
                TwoQubitControlledUDecomposer::new(RXXEquivalent::Standard(gate))?,
            ));
        }
    }
    Ok(decomposers)
}

/// Score a given sequence using the error rate reported in the target
///
/// Return a tuple of the predicted fidelity and the number of 2q gates in the sequence
#[inline]
fn score_sequence<'a>(
    target: &'a Target,
    kak_gate_name: &str,
    sequence: impl Iterator<Item = (Option<StandardGate>, SmallVec<[Qubit; 2]>)> + 'a,
) -> (f64, usize) {
    let mut gate_count = 0;
    (
        1. - sequence
            .filter_map(|(gate, local_qubits)| {
                let qubits = local_qubits
                    .iter()
                    .map(|qubit| PhysicalQubit(qubit.0))
                    .collect::<Vec<_>>();
                if qubits.len() == 2 {
                    gate_count += 1;
                }
                let name = match gate.as_ref() {
                    Some(g) => g.name(),
                    None => kak_gate_name,
                };
                target
                    .get_error(name, qubits.as_slice())
                    .map(|error| 1. - error)
            })
            .product::<f64>(),
        gate_count,
    )
}

type MappingIterItem = Option<((TwoQubitGateSequence, String), [Qubit; 2])>;

/// This transpiler pass can only run in a context where we've translated the circuit gates (or
/// where we know all gates have a matrix). If any gate identified in the run fails to have a
/// matrix defined (either in rust or python) it will be skipped
#[pyfunction]
pub(crate) fn two_qubit_unitary_peephole_optimize(
    py: Python,
    dag: &DAGCircuit,
    target: &Target,
    fidelity: f64,
) -> PyResult<DAGCircuit> {
    let runs: Vec<Vec<NodeIndex>> = dag.collect_2q_runs().unwrap();
    let node_mapping: HashMap<NodeIndex, usize> =
        HashMap::with_capacity(runs.iter().map(|run| run.len()).sum());
    let locked_node_mapping = Mutex::new(node_mapping);

    let run_mapping: PyResult<Vec<MappingIterItem>> = py.allow_threads(|| {
        // Build a vec of all the best synthesized two qubit gate sequences from the collected runs.
        // This is done in parallel
        runs.par_iter()
            .enumerate()
            .map(|(run_index, node_indices)| {
                let block_qubit_map = node_indices
                    .iter()
                    .find_map(|node_index| {
                        let inst = dag.dag()[*node_index].unwrap_operation();
                        let qubits = dag.get_qargs(inst.qubits);
                        if qubits.len() == 2 {
                            if qubits[0] > qubits[1] {
                                Some([qubits[1], qubits[0]])
                            } else {
                                Some([qubits[0], qubits[1]])
                            }
                        } else {
                            None
                        }
                    })
                    .unwrap();
                let matrix = blocks_to_matrix(dag, node_indices, block_qubit_map)?;
                let decomposers = get_decomposers_from_target(target, &block_qubit_map, fidelity)?;
                let mut decomposer_scores: Vec<Option<(f64, usize)>> =
                    vec![None; decomposers.len()];

                let order_sequence = |(index_a, sequence_a): &(
                    usize,
                    (TwoQubitGateSequence, String),
                ),
                                      (index_b, sequence_b): &(
                    usize,
                    (TwoQubitGateSequence, String),
                )| {
                    let score_a = match decomposer_scores[*index_a] {
                        Some(score) => score,
                        None => {
                            let score: (f64, usize) =
                                score_sequence(
                                    target,
                                    sequence_a.1.as_str(),
                                    sequence_a.0.gates.iter().map(
                                        |(gate, _params, local_qubits)| {
                                            let qubits = local_qubits
                                                .iter()
                                                .map(|qubit| block_qubit_map[*qubit as usize])
                                                .collect();
                                            (*gate, qubits)
                                        },
                                    ),
                                );
                            decomposer_scores[*index_a] = Some(score);
                            score
                        }
                    };

                    let score_b = match decomposer_scores[*index_b] {
                        Some(score) => score,
                        None => {
                            let score: (f64, usize) =
                                score_sequence(
                                    target,
                                    sequence_b.1.as_str(),
                                    sequence_b.0.gates.iter().map(
                                        |(gate, _params, local_qubits)| {
                                            let qubits = local_qubits
                                                .iter()
                                                .map(|qubit| block_qubit_map[*qubit as usize])
                                                .collect();
                                            (*gate, qubits)
                                        },
                                    ),
                                );
                            decomposer_scores[*index_b] = Some(score);
                            score
                        }
                    };
                    score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
                };

                let sequence = decomposers
                    .iter()
                    .map(|decomposer| match decomposer {
                        TwoQubitDecomposer::Basis(decomposer) => (
                            decomposer
                                .call_inner(matrix.view(), None, true, None)
                                .unwrap(),
                            decomposer.gate_name().to_string(),
                        ),
                        TwoQubitDecomposer::ControlledU(decomposer) => (
                            decomposer.call_inner(matrix.view(), 1e-12).unwrap(),
                            match decomposer.rxx_equivalent_gate {
                                RXXEquivalent::Standard(gate) => gate.name().to_string(),
                                RXXEquivalent::CustomPython(_) => {
                                    unreachable!("Decomposer only uses standard gates")
                                }
                            },
                        ),
                    })
                    .enumerate()
                    .min_by(order_sequence)
                    .unwrap();
                let mut original_err: f64 = 1.;
                let mut original_count: usize = 0;
                let mut outside_target = false;
                for node_index in node_indices {
                    let NodeType::Operation(ref inst) = dag.dag()[*node_index] else {
                        unreachable!("All run nodes will be ops")
                    };
                    let qubits = dag
                        .get_qargs(inst.qubits)
                        .iter()
                        .map(|qubit| PhysicalQubit(qubit.0))
                        .collect::<Vec<_>>();
                    if qubits.len() == 2 {
                        original_count += 1;
                    }
                    let name = inst.op.name();
                    let gate_err = match target.get_error(name, qubits.as_slice()) {
                        Some(err) => 1. - err,
                        None => {
                            // If error rate is None this can mean either the gate is not supported
                            // in the target or the gate is ideal. We need to do a second lookup
                            // to determine if the gate is supported, and if it isn't we don't need
                            // to finish scoring because we know we'll use the synthesis output
                            let physical_qargs =
                                qubits.iter().map(|bit| PhysicalQubit(bit.0)).collect();
                            if !target.instruction_supported(name, Some(&physical_qargs)) {
                                outside_target = true;
                                break;
                            }
                            1.
                        }
                    };
                    original_err *= gate_err;
                }
                let original_score = (1. - original_err, original_count);
                let new_score: (f64, usize) = match decomposer_scores[sequence.0] {
                    Some(score) => score,
                    None => score_sequence(
                        target,
                        sequence.1 .1.as_str(),
                        sequence
                            .1
                             .0
                            .gates
                            .iter()
                            .map(|(gate, _params, local_qubits)| {
                                let qubits = local_qubits
                                    .iter()
                                    .map(|qubit| block_qubit_map[*qubit as usize])
                                    .collect();
                                (*gate, qubits)
                            }),
                    ),
                };
                if !outside_target && new_score > original_score {
                    return Ok(None);
                }
                // This is done at the end of the map in some attempt to minimize
                // lock contention. If this were serial code it'd make more sense
                // to do this as part of the iteration building the
                let mut node_mapping = locked_node_mapping.lock().unwrap();
                for node in node_indices {
                    node_mapping.insert(*node, run_index);
                }
                Ok(Some((sequence.1, block_qubit_map)))
            })
            .collect()
    });

    let run_mapping = run_mapping?;
    // After we've computed all the sequences to execute now serially build up a new dag.
    let mut processed_runs: HashSet<usize> = HashSet::with_capacity(run_mapping.len());
    let mut out_dag = dag.copy_empty_like(py, "alike")?;
    let node_mapping = locked_node_mapping.into_inner().unwrap();
    for node in dag.topological_op_nodes()? {
        match node_mapping.get(&node) {
            Some(run_index) => {
                if processed_runs.contains(run_index) {
                    continue;
                }
                if run_mapping[*run_index].is_none() {
                    let NodeType::Operation(ref instr) = dag.dag()[node] else {
                        unreachable!("Must be an op node")
                    };
                    out_dag.push_back(py, instr.clone())?;
                    continue;
                }
                let (sequence, qubit_map) = &run_mapping[*run_index].as_ref().unwrap();
                for (gate, params, local_qubits) in &sequence.0.gates {
                    let qubits: Vec<Qubit> = local_qubits
                        .iter()
                        .map(|index| qubit_map[*index as usize])
                        .collect();
                    let out_params = if params.is_empty() {
                        None
                    } else {
                        Some(params.iter().map(|val| Param::Float(*val)).collect())
                    };
                    match gate {
                        Some(gate) => {
                            #[cfg(feature = "cache_pygates")]
                            {
                                out_dag.apply_operation_back(
                                    py,
                                    PackedOperation::from_standard(*gate),
                                    qubits.as_slice(),
                                    &[],
                                    out_params,
                                    ExtraInstructionAttributes::default(),
                                    None,
                                )
                            }
                            #[cfg(not(feature = "cache_pygates"))]
                            {
                                out_dag.apply_operation_back(
                                    py,
                                    PackedOperation::from_standard(*gate),
                                    qubits.as_slice(),
                                    &[],
                                    out_params,
                                    ExtraInstructionAttributes::default(),
                                )
                            }
                        }
                        None => {
                            let gate = target.operation_from_name(sequence.1.as_str()).unwrap();
                            #[cfg(feature = "cache_pygates")]
                            {
                                out_dag.apply_operation_back(
                                    py,
                                    gate.operation.clone(),
                                    qubits.as_slice(),
                                    &[],
                                    out_params,
                                    ExtraInstructionAttributes::default(),
                                    None,
                                )
                            }
                            #[cfg(not(feature = "cache_pygates"))]
                            {
                                out_dag.apply_operation_back(
                                    py,
                                    gate.operation.clone(),
                                    qubits.as_slice(),
                                    &[],
                                    out_params,
                                    ExtraInstructionAttributes::default(),
                                )
                            }
                        }
                    }?;
                }
                out_dag.add_global_phase(py, &Param::Float(sequence.0.global_phase))?;
                processed_runs.insert(*run_index);
            }
            None => {
                let NodeType::Operation(ref instr) = dag.dag()[node] else {
                    unreachable!("Must be an op node")
                };
                out_dag.push_back(py, instr.clone())?;
            }
        }
    }
    Ok(out_dag)
}

pub fn two_qubit_peephole_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(two_qubit_unitary_peephole_optimize))?;
    Ok(())
}

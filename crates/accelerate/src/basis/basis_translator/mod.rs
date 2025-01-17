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

use compose_transforms::BasisTransformIn;
use compose_transforms::BasisTransformOut;
use compose_transforms::GateIdentifier;

use basis_search::basis_search;
use compose_transforms::compose_transforms;
use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use pyo3::intern;
use pyo3::prelude::*;

mod basis_search;
mod compose_transforms;

use pyo3::types::{IntoPyDict, PyComplex, PyDict, PyTuple};
use pyo3::PyTypeInfo;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::converters::circuit_to_dag;
use qiskit_circuit::imports::DAG_TO_CIRCUIT;
use qiskit_circuit::imports::PARAMETER_EXPRESSION;
use qiskit_circuit::operations::Param;
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{
    circuit_data::CircuitData,
    dag_circuit::DAGCircuit,
    operations::{Operation, OperationRef},
};
use qiskit_circuit::{Clbit, Qubit};
use smallvec::SmallVec;

use crate::equivalence::EquivalenceLibrary;
use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::exceptions::TranspilerError;
use crate::target_transpiler::{Qargs, Target};

type InstMap = HashMap<GateIdentifier, BasisTransformOut>;
type ExtraInstructionMap<'a> = HashMap<&'a Option<Qargs>, InstMap>;

#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "base_run", signature = (dag, equiv_lib, qargs_with_non_global_operation, min_qubits, target_basis=None, target=None, non_global_operations=None))]
fn run(
    py: Python<'_>,
    dag: DAGCircuit,
    equiv_lib: &mut EquivalenceLibrary,
    qargs_with_non_global_operation: HashMap<Option<Qargs>, HashSet<String>>,
    min_qubits: usize,
    target_basis: Option<HashSet<String>>,
    target: Option<&Target>,
    non_global_operations: Option<HashSet<String>>,
) -> PyResult<DAGCircuit> {
    if target_basis.is_none() && target.is_none() {
        return Ok(dag);
    }

    let basic_instrs: HashSet<String>;
    let mut source_basis: HashSet<GateIdentifier> = HashSet::default();
    let mut new_target_basis: HashSet<String>;
    let mut qargs_local_source_basis: HashMap<Option<Qargs>, HashSet<GateIdentifier>> =
        HashMap::default();
    if let Some(target) = target.as_ref() {
        basic_instrs = ["barrier", "snapshot", "store"]
            .into_iter()
            .map(|x| x.to_string())
            .collect();
        let non_global_str: HashSet<&str> = if let Some(operations) = non_global_operations.as_ref()
        {
            operations.iter().map(|x| x.as_str()).collect()
        } else {
            HashSet::default()
        };
        let target_keys = target.keys().collect::<HashSet<_>>();
        new_target_basis = target_keys
            .difference(&non_global_str)
            .map(|x| x.to_string())
            .collect();
        extract_basis_target(
            py,
            &dag,
            &mut source_basis,
            &mut qargs_local_source_basis,
            min_qubits,
            &qargs_with_non_global_operation,
        )?;
    } else {
        basic_instrs = ["measure", "reset", "barrier", "snapshot", "delay", "store"]
            .into_iter()
            .map(|x| x.to_string())
            .collect();
        source_basis = extract_basis(py, &dag, min_qubits)?;
        new_target_basis = target_basis.unwrap();
    }
    new_target_basis = new_target_basis
        .union(&basic_instrs)
        .map(|x| x.to_string())
        .collect();
    // If the source basis is a subset of the target basis and we have no circuit
    // instructions on qargs that have non-global operations there is nothing to
    // translate and we can exit early.
    let source_basis_names: HashSet<String> = source_basis.iter().map(|x| x.0.clone()).collect();
    if source_basis_names.is_subset(&new_target_basis) && qargs_local_source_basis.is_empty() {
        return Ok(dag);
    }
    let basis_transforms = basis_search(equiv_lib, &source_basis, &new_target_basis);
    let mut qarg_local_basis_transforms: HashMap<
        Option<Qargs>,
        Vec<(GateIdentifier, BasisTransformIn)>,
    > = HashMap::default();
    for (qarg, local_source_basis) in qargs_local_source_basis.iter() {
        // For any multiqubit operation that contains a subset of qubits that
        // has a non-local operation, include that non-local operation in the
        // search. This matches with the check we did above to include those
        // subset non-local operations in the check here.
        let mut expanded_target = new_target_basis.clone();
        if qarg.as_ref().is_some_and(|qarg| qarg.len() > 1) {
            let qarg_as_set: HashSet<PhysicalQubit> =
                HashSet::from_iter(qarg.as_ref().unwrap().iter().copied());
            for (non_local_qarg, local_basis) in qargs_with_non_global_operation.iter() {
                if let Some(non_local_qarg) = non_local_qarg {
                    let non_local_qarg_as_set = HashSet::from_iter(non_local_qarg.iter().copied());
                    if qarg_as_set.is_superset(&non_local_qarg_as_set) {
                        expanded_target = expanded_target.union(local_basis).cloned().collect();
                    }
                }
            }
        } else {
            expanded_target = expanded_target
                .union(&qargs_with_non_global_operation[qarg])
                .cloned()
                .collect();
        }
        let local_basis_transforms = basis_search(equiv_lib, local_source_basis, &expanded_target);
        if let Some(local_basis_transforms) = local_basis_transforms {
            qarg_local_basis_transforms.insert(qarg.clone(), local_basis_transforms);
        } else {
            return Err(TranspilerError::new_err(format!(
                "Unable to translate the operations in the circuit: \
            {:?} to the backend's (or manually specified) target \
            basis: {:?}. This likely means the target basis is not universal \
            or there are additional equivalence rules needed in the EquivalenceLibrary being \
            used. For more details on this error see: \
            https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.\
            BasisTranslator#translation-errors",
                local_source_basis
                    .iter()
                    .map(|x| x.0.as_str())
                    .collect_vec(),
                &expanded_target
            )));
        }
    }

    let Some(basis_transforms) = basis_transforms else {
        return Err(TranspilerError::new_err(format!(
            "Unable to translate the operations in the circuit: \
        {:?} to the backend's (or manually specified) target \
        basis: {:?}. This likely means the target basis is not universal \
        or there are additional equivalence rules needed in the EquivalenceLibrary being \
        used. For more details on this error see: \
        https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes. \
        BasisTranslator#translation-errors",
            source_basis.iter().map(|x| x.0.as_str()).collect_vec(),
            &new_target_basis
        )));
    };

    let instr_map: InstMap = compose_transforms(py, &basis_transforms, &source_basis, &dag)?;
    let extra_inst_map: ExtraInstructionMap = qarg_local_basis_transforms
        .iter()
        .map(|(qarg, transform)| -> PyResult<_> {
            Ok((
                qarg,
                compose_transforms(py, transform, &qargs_local_source_basis[qarg], &dag)?,
            ))
        })
        .collect::<PyResult<_>>()?;

    let (out_dag, _) = apply_translation(
        py,
        &dag,
        &new_target_basis,
        &instr_map,
        &extra_inst_map,
        min_qubits,
        &qargs_with_non_global_operation,
    )?;
    Ok(out_dag)
}

/// Method that extracts all non-calibrated gate instances identifiers from a DAGCircuit.
fn extract_basis(
    py: Python,
    circuit: &DAGCircuit,
    min_qubits: usize,
) -> PyResult<HashSet<GateIdentifier>> {
    let mut basis = HashSet::default();
    // Recurse for DAGCircuit
    fn recurse_dag(
        py: Python,
        circuit: &DAGCircuit,
        basis: &mut HashSet<GateIdentifier>,
        min_qubits: usize,
    ) -> PyResult<()> {
        for node in circuit.op_nodes(true) {
            let operation: &PackedInstruction = circuit.dag()[node].unwrap_operation();
            if !circuit.has_calibration_for_index(py, node)?
                && circuit.get_qargs(operation.qubits).len() >= min_qubits
            {
                basis.insert((operation.op.name().to_string(), operation.op.num_qubits()));
            }
            if operation.op.control_flow() {
                let OperationRef::Instruction(inst) = operation.op.view() else {
                    unreachable!("Control flow operation is not an instance of PyInstruction.")
                };
                let inst_bound = inst.instruction.bind(py);
                for block in inst_bound.getattr("blocks")?.iter()? {
                    recurse_circuit(py, block?, basis, min_qubits)?;
                }
            }
        }
        Ok(())
    }

    // Recurse for QuantumCircuit
    fn recurse_circuit(
        py: Python,
        circuit: Bound<PyAny>,
        basis: &mut HashSet<GateIdentifier>,
        min_qubits: usize,
    ) -> PyResult<()> {
        let circuit_data: PyRef<CircuitData> = circuit
            .getattr(intern!(py, "_data"))?
            .downcast_into()?
            .borrow();
        for (index, inst) in circuit_data.iter().enumerate() {
            let instruction_object = circuit.get_item(index)?;
            let has_calibration = circuit
                .call_method1(intern!(py, "_has_calibration_for"), (&instruction_object,))?;
            if !has_calibration.is_truthy()?
                && circuit_data.get_qargs(inst.qubits).len() >= min_qubits
            {
                basis.insert((inst.op.name().to_string(), inst.op.num_qubits()));
            }
            if inst.op.control_flow() {
                let operation_ob = instruction_object.getattr(intern!(py, "operation"))?;
                let blocks = operation_ob.getattr("blocks")?;
                for block in blocks.iter()? {
                    recurse_circuit(py, block?, basis, min_qubits)?;
                }
            }
        }
        Ok(())
    }

    recurse_dag(py, circuit, &mut basis, min_qubits)?;
    Ok(basis)
}

/// Method that extracts a mapping of all the qargs in the local_source basis
/// obtained from the [Target], to all non-calibrated gate instances identifiers from a DAGCircuit.
/// When dealing with `ControlFlowOp` instances the function will perform a recursion call
/// to a variant design to handle instances of `QuantumCircuit`.
fn extract_basis_target(
    py: Python,
    dag: &DAGCircuit,
    source_basis: &mut HashSet<GateIdentifier>,
    qargs_local_source_basis: &mut HashMap<Option<Qargs>, HashSet<GateIdentifier>>,
    min_qubits: usize,
    qargs_with_non_global_operation: &HashMap<Option<Qargs>, HashSet<String>>,
) -> PyResult<()> {
    for node in dag.op_nodes(true) {
        let node_obj: &PackedInstruction = dag.dag()[node].unwrap_operation();
        let qargs: &[Qubit] = dag.get_qargs(node_obj.qubits);
        if dag.has_calibration_for_index(py, node)? || qargs.len() < min_qubits {
            continue;
        }
        // Treat the instruction as on an incomplete basis if the qargs are in the
        // qargs_with_non_global_operation dictionary or if any of the qubits in qargs
        // are a superset for a non-local operation. For example, if the qargs
        // are (0, 1) and that's a global (ie no non-local operations on (0, 1)
        // operation but there is a non-local operation on (1,) we need to
        // do an extra non-local search for this op to ensure we include any
        // single qubit operation for (1,) as valid. This pattern also holds
        // true for > 2q ops too (so for 4q operations we need to check for 3q, 2q,
        // and 1q operations in the same manner)
        let physical_qargs: SmallVec<[PhysicalQubit; 2]> =
            qargs.iter().map(|x| PhysicalQubit(x.0)).collect();
        let physical_qargs_as_set: HashSet<PhysicalQubit> =
            HashSet::from_iter(physical_qargs.iter().copied());
        if qargs_with_non_global_operation.contains_key(&Some(physical_qargs))
            || qargs_with_non_global_operation
                .keys()
                .flatten()
                .any(|incomplete_qargs| {
                    let incomplete_qargs = HashSet::from_iter(incomplete_qargs.iter().copied());
                    physical_qargs_as_set.is_superset(&incomplete_qargs)
                })
        {
            qargs_local_source_basis
                .entry(Some(physical_qargs_as_set.into_iter().collect()))
                .and_modify(|set| {
                    set.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
                })
                .or_insert(HashSet::from_iter([(
                    node_obj.op.name().to_string(),
                    node_obj.op.num_qubits(),
                )]));
        } else {
            source_basis.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
        }
        if node_obj.op.control_flow() {
            let OperationRef::Instruction(op) = node_obj.op.view() else {
                unreachable!("Control flow op is not a control flow op. But control_flow is `true`")
            };
            let bound_inst = op.instruction.bind(py);
            // Use python side extraction instead of the Rust method `op.blocks` due to
            // required usage of a python-space method `QuantumCircuit.has_calibration_for`.
            let blocks = bound_inst.getattr("blocks")?.iter()?;
            for block in blocks {
                extract_basis_target_circ(
                    &block?,
                    source_basis,
                    qargs_local_source_basis,
                    min_qubits,
                    qargs_with_non_global_operation,
                )?;
            }
        }
    }
    Ok(())
}

/// Variant of extract_basis_target that takes an instance of QuantumCircuit.
/// This needs to use a Python instance of `QuantumCircuit` due to it needing
/// to access `has_calibration_for()` which is unavailable through rust. However,
/// this API will be removed with the deprecation of `Pulse`.
fn extract_basis_target_circ(
    circuit: &Bound<PyAny>,
    source_basis: &mut HashSet<GateIdentifier>,
    qargs_local_source_basis: &mut HashMap<Option<Qargs>, HashSet<GateIdentifier>>,
    min_qubits: usize,
    qargs_with_non_global_operation: &HashMap<Option<Qargs>, HashSet<String>>,
) -> PyResult<()> {
    let py = circuit.py();
    let circ_data_bound = circuit.getattr("_data")?.downcast_into::<CircuitData>()?;
    let circ_data = circ_data_bound.borrow();
    for (index, node_obj) in circ_data.iter().enumerate() {
        let qargs = circ_data.get_qargs(node_obj.qubits);
        if circuit
            .call_method1("_has_calibration_for", (circuit.get_item(index)?,))?
            .is_truthy()?
            || qargs.len() < min_qubits
        {
            continue;
        }
        // Treat the instruction as on an incomplete basis if the qargs are in the
        // qargs_with_non_global_operation dictionary or if any of the qubits in qargs
        // are a superset for a non-local operation. For example, if the qargs
        // are (0, 1) and that's a global (ie no non-local operations on (0, 1)
        // operation but there is a non-local operation on (1,) we need to
        // do an extra non-local search for this op to ensure we include any
        // single qubit operation for (1,) as valid. This pattern also holds
        // true for > 2q ops too (so for 4q operations we need to check for 3q, 2q,
        // and 1q operations in the same manner)
        let physical_qargs: SmallVec<[PhysicalQubit; 2]> =
            qargs.iter().map(|x| PhysicalQubit(x.0)).collect();
        let physical_qargs_as_set: HashSet<PhysicalQubit> =
            HashSet::from_iter(physical_qargs.iter().copied());
        if qargs_with_non_global_operation.contains_key(&Some(physical_qargs))
            || qargs_with_non_global_operation
                .keys()
                .flatten()
                .any(|incomplete_qargs| {
                    let incomplete_qargs = HashSet::from_iter(incomplete_qargs.iter().copied());
                    physical_qargs_as_set.is_superset(&incomplete_qargs)
                })
        {
            qargs_local_source_basis
                .entry(Some(physical_qargs_as_set.into_iter().collect()))
                .and_modify(|set| {
                    set.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
                })
                .or_insert(HashSet::from_iter([(
                    node_obj.op.name().to_string(),
                    node_obj.op.num_qubits(),
                )]));
        } else {
            source_basis.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
        }
        if node_obj.op.control_flow() {
            let OperationRef::Instruction(op) = node_obj.op.view() else {
                unreachable!("Control flow op is not a control flow op. But control_flow is `true`")
            };
            let bound_inst = op.instruction.bind(py);
            let blocks = bound_inst.getattr("blocks")?.iter()?;
            for block in blocks {
                extract_basis_target_circ(
                    &block?,
                    source_basis,
                    qargs_local_source_basis,
                    min_qubits,
                    qargs_with_non_global_operation,
                )?;
            }
        }
    }
    Ok(())
}

fn apply_translation(
    py: Python,
    dag: &DAGCircuit,
    target_basis: &HashSet<String>,
    instr_map: &InstMap,
    extra_inst_map: &ExtraInstructionMap,
    min_qubits: usize,
    qargs_with_non_global_operation: &HashMap<Option<Qargs>, HashSet<String>>,
) -> PyResult<(DAGCircuit, bool)> {
    let mut is_updated = false;
    let mut out_dag = dag.copy_empty_like(py, "alike")?;
    for node in dag.topological_op_nodes()? {
        let node_obj = dag.dag()[node].unwrap_operation();
        let node_qarg = dag.get_qargs(node_obj.qubits);
        let node_carg = dag.get_cargs(node_obj.clbits);
        let qubit_set: HashSet<Qubit> = HashSet::from_iter(node_qarg.iter().copied());
        let mut new_op: Option<OperationFromPython> = None;
        if target_basis.contains(node_obj.op.name()) || node_qarg.len() < min_qubits {
            if node_obj.op.control_flow() {
                let OperationRef::Instruction(control_op) = node_obj.op.view() else {
                    unreachable!("This instruction {} says it is of control flow type, but is not an Instruction instance", node_obj.op.name())
                };
                let mut flow_blocks = vec![];
                let bound_obj = control_op.instruction.bind(py);
                let blocks = bound_obj.getattr("blocks")?;
                for block in blocks.iter()? {
                    let block = block?;
                    let dag_block: DAGCircuit =
                        circuit_to_dag(py, block.extract()?, true, None, None)?;
                    let updated_dag: DAGCircuit;
                    (updated_dag, is_updated) = apply_translation(
                        py,
                        &dag_block,
                        target_basis,
                        instr_map,
                        extra_inst_map,
                        min_qubits,
                        qargs_with_non_global_operation,
                    )?;
                    let flow_circ_block = if is_updated {
                        DAG_TO_CIRCUIT
                            .get_bound(py)
                            .call1((updated_dag,))?
                            .extract()?
                    } else {
                        block
                    };
                    flow_blocks.push(flow_circ_block);
                }
                let replaced_blocks = bound_obj.call_method1("replace_blocks", (flow_blocks,))?;
                new_op = Some(replaced_blocks.extract()?);
            }
            if let Some(new_op) = new_op {
                out_dag.apply_operation_back(
                    py,
                    new_op.operation,
                    node_qarg,
                    node_carg,
                    if new_op.params.is_empty() {
                        None
                    } else {
                        Some(new_op.params)
                    },
                    new_op.extra_attrs,
                    #[cfg(feature = "cache_pygates")]
                    None,
                )?;
            } else {
                out_dag.apply_operation_back(
                    py,
                    node_obj.op.clone(),
                    node_qarg,
                    node_carg,
                    if node_obj.params_view().is_empty() {
                        None
                    } else {
                        Some(
                            node_obj
                                .params_view()
                                .iter()
                                .map(|param| param.clone_ref(py))
                                .collect(),
                        )
                    },
                    node_obj.extra_attrs.clone(),
                    #[cfg(feature = "cache_pygates")]
                    None,
                )?;
            }
            continue;
        }
        let node_qarg_as_physical: Option<Qargs> =
            Some(node_qarg.iter().map(|x| PhysicalQubit(x.0)).collect());
        if qargs_with_non_global_operation.contains_key(&node_qarg_as_physical)
            && qargs_with_non_global_operation[&node_qarg_as_physical].contains(node_obj.op.name())
        {
            out_dag.apply_operation_back(
                py,
                node_obj.op.clone(),
                node_qarg,
                node_carg,
                if node_obj.params_view().is_empty() {
                    None
                } else {
                    Some(
                        node_obj
                            .params_view()
                            .iter()
                            .map(|param| param.clone_ref(py))
                            .collect(),
                    )
                },
                node_obj.extra_attrs.clone(),
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
            continue;
        }

        if dag.has_calibration_for_index(py, node)? {
            out_dag.apply_operation_back(
                py,
                node_obj.op.clone(),
                node_qarg,
                node_carg,
                if node_obj.params_view().is_empty() {
                    None
                } else {
                    Some(
                        node_obj
                            .params_view()
                            .iter()
                            .map(|param| param.clone_ref(py))
                            .collect(),
                    )
                },
                node_obj.extra_attrs.clone(),
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
            continue;
        }
        let unique_qargs: Option<Qargs> = if qubit_set.is_empty() {
            None
        } else {
            Some(qubit_set.iter().map(|x| PhysicalQubit(x.0)).collect())
        };
        if extra_inst_map.contains_key(&unique_qargs) {
            replace_node(
                py,
                &mut out_dag,
                node_obj.clone(),
                &extra_inst_map[&unique_qargs],
            )?;
        } else if instr_map
            .contains_key(&(node_obj.op.name().to_string(), node_obj.op.num_qubits()))
        {
            replace_node(py, &mut out_dag, node_obj.clone(), instr_map)?;
        } else {
            return Err(TranspilerError::new_err(format!(
                "BasisTranslator did not map {}",
                node_obj.op.name()
            )));
        }
        is_updated = true;
    }

    Ok((out_dag, is_updated))
}

fn replace_node(
    py: Python,
    dag: &mut DAGCircuit,
    node: PackedInstruction,
    instr_map: &HashMap<GateIdentifier, (SmallVec<[Param; 3]>, DAGCircuit)>,
) -> PyResult<()> {
    let (target_params, target_dag) =
        &instr_map[&(node.op.name().to_string(), node.op.num_qubits())];
    if node.params_view().len() != target_params.len() {
        return Err(TranspilerError::new_err(format!(
            "Translation num_params not equal to op num_params. \
            Op: {:?} {} Translation: {:?}\n{:?}",
            node.params_view(),
            node.op.name(),
            &target_params,
            &target_dag
        )));
    }
    if node.params_view().is_empty() {
        for inner_index in target_dag.topological_op_nodes()? {
            let inner_node = &target_dag.dag()[inner_index].unwrap_operation();
            let old_qargs = dag.get_qargs(node.qubits);
            let old_cargs = dag.get_cargs(node.clbits);
            let new_qubits: Vec<Qubit> = target_dag
                .get_qargs(inner_node.qubits)
                .iter()
                .map(|qubit| old_qargs[qubit.0 as usize])
                .collect();
            let new_clbits: Vec<Clbit> = target_dag
                .get_cargs(inner_node.clbits)
                .iter()
                .map(|clbit| old_cargs[clbit.0 as usize])
                .collect();
            let new_op = if inner_node.op.try_standard_gate().is_none() {
                inner_node.op.py_copy(py)?
            } else {
                inner_node.op.clone()
            };
            if node.condition().is_some() {
                match new_op.view() {
                    OperationRef::Gate(gate) => {
                        gate.gate.setattr(py, "condition", node.condition())?
                    }
                    OperationRef::Instruction(inst) => {
                        inst.instruction
                            .setattr(py, "condition", node.condition())?
                    }
                    OperationRef::Operation(oper) => {
                        oper.operation.setattr(py, "condition", node.condition())?
                    }
                    _ => (),
                }
            }
            let new_params: SmallVec<[Param; 3]> = inner_node
                .params_view()
                .iter()
                .map(|param| param.clone_ref(py))
                .collect();
            let new_extra_props = node.extra_attrs.clone();
            dag.apply_operation_back(
                py,
                new_op,
                &new_qubits,
                &new_clbits,
                if new_params.is_empty() {
                    None
                } else {
                    Some(new_params)
                },
                new_extra_props,
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
        }
        dag.add_global_phase(py, target_dag.global_phase())?;
    } else {
        let parameter_map = target_params
            .iter()
            .zip(node.params_view())
            .into_py_dict_bound(py);
        for inner_index in target_dag.topological_op_nodes()? {
            let inner_node = &target_dag.dag()[inner_index].unwrap_operation();
            let old_qargs = dag.get_qargs(node.qubits);
            let old_cargs = dag.get_cargs(node.clbits);
            let new_qubits: Vec<Qubit> = target_dag
                .get_qargs(inner_node.qubits)
                .iter()
                .map(|qubit| old_qargs[qubit.0 as usize])
                .collect();
            let new_clbits: Vec<Clbit> = target_dag
                .get_cargs(inner_node.clbits)
                .iter()
                .map(|clbit| old_cargs[clbit.0 as usize])
                .collect();
            let new_op = if inner_node.op.try_standard_gate().is_none() {
                inner_node.op.py_copy(py)?
            } else {
                inner_node.op.clone()
            };
            let mut new_params: SmallVec<[Param; 3]> = inner_node
                .params_view()
                .iter()
                .map(|param| param.clone_ref(py))
                .collect();
            if inner_node
                .params_view()
                .iter()
                .any(|param| matches!(param, Param::ParameterExpression(_)))
            {
                new_params = SmallVec::new();
                for param in inner_node.params_view() {
                    if let Param::ParameterExpression(param_obj) = param {
                        let bound_param = param_obj.bind(py);
                        let exp_params = param.iter_parameters(py)?;
                        let bind_dict = PyDict::new_bound(py);
                        for key in exp_params {
                            let key = key?;
                            bind_dict.set_item(&key, parameter_map.get_item(&key)?)?;
                        }
                        let mut new_value: Bound<PyAny>;
                        let comparison = bind_dict.values().iter().any(|param| {
                            param
                                .is_instance(PARAMETER_EXPRESSION.get_bound(py))
                                .is_ok_and(|x| x)
                        });
                        if comparison {
                            new_value = bound_param.clone();
                            for items in bind_dict.items() {
                                new_value = new_value.call_method1(
                                    intern!(py, "assign"),
                                    items.downcast::<PyTuple>()?,
                                )?;
                            }
                        } else {
                            new_value =
                                bound_param.call_method1(intern!(py, "bind"), (&bind_dict,))?;
                        }
                        let eval = new_value.getattr(intern!(py, "parameters"))?;
                        if eval.is_empty()? {
                            new_value = new_value.call_method0(intern!(py, "numeric"))?;
                        }
                        new_params.push(new_value.extract()?);
                    } else {
                        new_params.push(param.clone_ref(py));
                    }
                }
                if new_op.try_standard_gate().is_none() {
                    match new_op.view() {
                        OperationRef::Instruction(inst) => inst
                            .instruction
                            .bind(py)
                            .setattr("params", new_params.clone())?,
                        OperationRef::Gate(gate) => {
                            gate.gate.bind(py).setattr("params", new_params.clone())?
                        }
                        OperationRef::Operation(oper) => oper
                            .operation
                            .bind(py)
                            .setattr("params", new_params.clone())?,
                        _ => (),
                    }
                }
            }
            dag.apply_operation_back(
                py,
                new_op,
                &new_qubits,
                &new_clbits,
                if new_params.is_empty() {
                    None
                } else {
                    Some(new_params)
                },
                inner_node.extra_attrs.clone(),
                #[cfg(feature = "cache_pygates")]
                None,
            )?;
        }

        if let Param::ParameterExpression(old_phase) = target_dag.global_phase() {
            let bound_old_phase = old_phase.bind(py);
            let bind_dict = PyDict::new_bound(py);
            for key in target_dag.global_phase().iter_parameters(py)? {
                let key = key?;
                bind_dict.set_item(&key, parameter_map.get_item(&key)?)?;
            }
            let mut new_phase: Bound<PyAny>;
            if bind_dict.values().iter().any(|param| {
                param
                    .is_instance(PARAMETER_EXPRESSION.get_bound(py))
                    .is_ok_and(|x| x)
            }) {
                new_phase = bound_old_phase.clone();
                for key_val in bind_dict.items() {
                    new_phase =
                        new_phase.call_method1(intern!(py, "assign"), key_val.downcast()?)?;
                }
            } else {
                new_phase = bound_old_phase.call_method1(intern!(py, "bind"), (bind_dict,))?;
            }
            if !new_phase.getattr(intern!(py, "parameters"))?.is_truthy()? {
                new_phase = new_phase.call_method0(intern!(py, "numeric"))?;
                if new_phase.is_instance(&PyComplex::type_object_bound(py))? {
                    return Err(TranspilerError::new_err(format!(
                        "Global phase must be real, but got {}",
                        new_phase.repr()?
                    )));
                }
            }
            let new_phase: Param = new_phase.extract()?;
            dag.add_global_phase(py, &new_phase)?;
        }
    }

    Ok(())
}

#[pymodule]
pub fn basis_translator(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run))?;
    Ok(())
}

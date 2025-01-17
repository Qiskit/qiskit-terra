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

use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::exceptions::TranspilerError;
use crate::target_transpiler::Target;
use hashbrown::HashSet;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{
    circuit_instruction::CircuitInstruction,
    circuit_instruction::ExtraInstructionAttributes,
    converters::{circuit_to_dag, QuantumCircuitData},
    dag_circuit::{DAGCircuit, NodeType},
    dag_node::{DAGNode, DAGOpNode},
    imports,
    imports::get_std_gate_class,
    operations::Operation,
    operations::Param,
    operations::StandardGate,
    packed_instruction::PackedInstruction,
    Qubit,
};
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::{smallvec, SmallVec};
use std::f64::consts::PI;

//#########################################################################
//              CheckGateDirection analysis pass functions
//#########################################################################

/// Check if the two-qubit gates follow the right direction with respect to the coupling map.
///
/// Args:
///     dag: the DAGCircuit to analyze
///
///     coupling_edges: set of edge pairs representing a directed coupling map, against which gate directionality is checked
///
/// Returns:
///     true iff all two-qubit gates comply with the coupling constraints
#[pyfunction]
#[pyo3(name = "check_gate_direction_coupling")]
fn py_check_direction_coupling_map(
    py: Python,
    dag: &DAGCircuit,
    coupling_edges: HashSet<[Qubit; 2]>,
) -> PyResult<bool> {
    let coupling_map_check =
        |_: &PackedInstruction, op_args: &[Qubit]| -> bool { coupling_edges.contains(op_args) };

    check_gate_direction(py, dag, &coupling_map_check, None)
}

/// Check if the two-qubit gates follow the right direction with respect to instructions supported in the given target.
///
/// Args:
///     dag: the DAGCircuit to analyze
///
///     target: the Target against which gate directionality compliance is checked
///
/// Returns:
///     true iff all two-qubit gates comply with the target's coupling constraints
#[pyfunction]
#[pyo3(name = "check_gate_direction_target")]
fn py_check_direction_target(py: Python, dag: &DAGCircuit, target: &Target) -> PyResult<bool> {
    let target_check = |inst: &PackedInstruction, op_args: &[Qubit]| -> bool {
        let qargs = smallvec![
            PhysicalQubit::new(op_args[0].0),
            PhysicalQubit::new(op_args[1].0)
        ];

        target.instruction_supported(inst.op.name(), Some(&qargs))
    };

    check_gate_direction(py, dag, &target_check, None)
}

// The main routine for checking gate directionality.
//
// gate_complies: a function returning true iff the two-qubit gate direction complies with directionality constraints
//
// qubit_mapping: used for mapping the index of a given qubit within an instruction qargs vector to the corresponding qubit index of the
//  original DAGCircuit the pass was called with. This mapping is required since control flow blocks are represented by nested DAGCircuit
//  objects whose instruction qubit indices are relative to the parent DAGCircuit they reside in, thus when we recurse into nested DAGs, we need
//  to carry the mapping context relative to the original DAG.
//  When qubit_mapping is None, the identity mapping is assumed
fn check_gate_direction<T>(
    py: Python,
    dag: &DAGCircuit,
    gate_complies: &T,
    qubit_mapping: Option<&[Qubit]>,
) -> PyResult<bool>
where
    T: Fn(&PackedInstruction, &[Qubit]) -> bool,
{
    for node in dag.op_nodes(false) {
        let NodeType::Operation(packed_inst) = &dag.dag()[node] else {
            panic!("PackedInstruction is expected");
        };

        let inst_qargs = dag.get_qargs(packed_inst.qubits);

        if let OperationRef::Instruction(py_inst) = packed_inst.op.view() {
            if py_inst.control_flow() {
                let circuit_to_dag = imports::CIRCUIT_TO_DAG.get_bound(py);
                let py_inst = py_inst.instruction.bind(py);

                for block in py_inst.getattr("blocks")?.iter()? {
                    let inner_dag: DAGCircuit = circuit_to_dag.call1((block?,))?.extract()?;

                    let block_ok = if let Some(mapping) = qubit_mapping {
                        let mapping = inst_qargs // Create a temp mapping for the recursive call
                            .iter()
                            .map(|q| mapping[q.index()])
                            .collect::<Vec<Qubit>>();

                        check_gate_direction(py, &inner_dag, gate_complies, Some(&mapping))?
                    } else {
                        check_gate_direction(py, &inner_dag, gate_complies, Some(inst_qargs))?
                    };

                    if !block_ok {
                        return Ok(false);
                    }
                }
                continue;
            }
        }

        if inst_qargs.len() == 2
            && !match qubit_mapping {
                // Check gate direction based either on a given custom mapping or the identity mapping
                Some(mapping) => gate_complies(
                    packed_inst,
                    &[
                        mapping[inst_qargs[0].index()],
                        mapping[inst_qargs[1].index()],
                    ],
                ),
                None => gate_complies(packed_inst, inst_qargs),
            }
        {
            return Ok(false);
        }
    }

    Ok(true)
}

//#########################################################################
//              GateDirection transformation pass functions
//#########################################################################

/// Try to swap two-qubit gate directions using pre-defined mapping to follow the right direction with respect to the coupling map.
///
/// Args:
///     dag: the DAGCircuit to analyze
///
///     coupling_edges: set of edge pairs representing a directed coupling map, against which gate directionality is checked
///
/// Returns:
///     the transformed DAGCircuit
#[pyfunction]
#[pyo3(name = "fix_gate_direction_coupling")]
fn py_fix_direction_coupling_map(
    py: Python,
    dag: &mut DAGCircuit,
    coupling_edges: HashSet<[Qubit; 2]>,
) -> PyResult<DAGCircuit> {
    if coupling_edges.is_empty() {
        return Ok(dag.clone());
    }

    let coupling_map_check =
        |_: &PackedInstruction, op_args: &[Qubit]| -> bool { coupling_edges.contains(op_args) };

    fix_gate_direction(py, dag, &coupling_map_check, None).cloned()
}

/// Try to swap two-qubit gate directions using pre-defined mapping to follow the right direction with respect to the given target.
///
/// Args:
///     dag: the DAGCircuit to analyze
///
///     coupling_edges: set of edge pairs representing a directed coupling map, against which gate directionality is checked
///
/// Returns:
///     the transformed DAGCircuit
#[pyfunction]
#[pyo3(name = "fix_gate_direction_target")]
fn py_fix_direction_target(
    py: Python,
    dag: &mut DAGCircuit,
    target: &Target,
) -> PyResult<DAGCircuit> {
    let target_check = |inst: &PackedInstruction, op_args: &[Qubit]| -> bool {
        let qargs = smallvec![
            PhysicalQubit::new(op_args[0].0),
            PhysicalQubit::new(op_args[1].0)
        ];

        // Take this path so Target can check for exact match of the parameterized gate's angle
        if let OperationRef::Standard(std_gate) = inst.op.view() {
            match std_gate {
                StandardGate::RXXGate
                | StandardGate::RYYGate
                | StandardGate::RZZGate
                | StandardGate::RZXGate => {
                    return target
                        .py_instruction_supported(
                            py,
                            None,
                            Some(qargs),
                            Some(
                                get_std_gate_class(py, std_gate)
                                    .expect("These gates should have Python classes")
                                    .bind(py),
                            ),
                            Some(inst.params_view().to_vec()),
                        )
                        .unwrap_or(false)
                }
                _ => {}
            }
        }
        target.instruction_supported(inst.op.name(), Some(&qargs))
    };

    fix_gate_direction(py, dag, &target_check, None).cloned()
}

// The main routine for fixing gate direction. Same parameters are check_gate_direction
fn fix_gate_direction<'a, T>(
    py: Python,
    dag: &'a mut DAGCircuit,
    gate_complies: &T,
    qubit_mapping: Option<&[Qubit]>,
) -> PyResult<&'a DAGCircuit>
where
    T: Fn(&PackedInstruction, &[Qubit]) -> bool,
{
    let mut nodes_to_replace: Vec<(NodeIndex, DAGCircuit)> = Vec::new();
    let mut ops_to_replace: Vec<(NodeIndex, Vec<Bound<PyAny>>)> = Vec::new();

    for node in dag.op_nodes(false) {
        let packed_inst = dag.dag()[node].unwrap_operation();

        let op_args = dag.get_qargs(packed_inst.qubits);

        if let OperationRef::Instruction(py_inst) = packed_inst.op.view() {
            if py_inst.control_flow() {
                let dag_to_circuit = imports::DAG_TO_CIRCUIT.get_bound(py);

                let blocks = py_inst.instruction.bind(py).getattr("blocks")?;
                let blocks = blocks.downcast::<PyTuple>()?;

                let mut blocks_to_replace = Vec::with_capacity(blocks.len());
                for block in blocks {
                    let mut inner_dag = circuit_to_dag(
                        py,
                        QuantumCircuitData::extract_bound(&block)?,
                        false,
                        None,
                        None,
                    )?;

                    let inner_dag = if let Some(mapping) = qubit_mapping {
                        let mapping = op_args // Create a temp mapping for the recursive call
                            .iter()
                            .map(|q| mapping[q.index()])
                            .collect::<Vec<Qubit>>();

                        fix_gate_direction(py, &mut inner_dag, gate_complies, Some(&mapping))?
                    } else {
                        fix_gate_direction(py, &mut inner_dag, gate_complies, Some(op_args))?
                    };

                    let circuit = dag_to_circuit.call1((inner_dag.clone(),))?;
                    blocks_to_replace.push(circuit);
                }

                // Store this for replacement outside the dag.op_nodes loop
                ops_to_replace.push((node, blocks_to_replace));

                continue;
            }
        }

        if op_args.len() != 2 || dag.has_calibration_for_index(py, node)? {
            continue;
        };

        // Take into account qubit index mapping if we're inside a control-flow block
        let (op_args0, op_args1) = if let Some(mapping) = qubit_mapping {
            (mapping[op_args[0].index()], mapping[op_args[1].index()])
        } else {
            (op_args[0], op_args[1])
        };

        if gate_complies(packed_inst, &[op_args0, op_args1]) {
            continue;
        }

        // If the op has a pre-defined replacement - replace if the other direction is supported otherwise error
        // If no pre-defined replacement for the op - if the other direction is supported error saying no pre-defined rule otherwise error saying op is not supported
        if let OperationRef::Standard(std_gate) = packed_inst.op.view() {
            match std_gate {
                StandardGate::CXGate
                | StandardGate::ECRGate
                | StandardGate::CZGate
                | StandardGate::SwapGate
                | StandardGate::RXXGate
                | StandardGate::RYYGate
                | StandardGate::RZZGate
                | StandardGate::RZXGate => {
                    if gate_complies(packed_inst, &[op_args1, op_args0]) {
                        // Store this for replacement outside the dag.op_nodes loop
                        nodes_to_replace.push((node, replace_dag(py, std_gate, packed_inst)?));
                        continue;
                    } else {
                        return Err(TranspilerError::new_err(format!(
                            "The circuit requires a connection between physical qubits {:?} for {}",
                            op_args,
                            packed_inst.op.name()
                        )));
                    }
                }
                _ => {}
            }
        }
        // No matching replacement found
        if gate_complies(packed_inst, &[op_args1, op_args0])
            || has_calibration_for_op_node(py, dag, packed_inst, &[op_args1, op_args0])?
        {
            return Err(TranspilerError::new_err(format!("{} would be supported on {:?} if the direction was swapped, but no rules are known to do that. {:?} can be automatically flipped.", packed_inst.op.name(), op_args, vec!["cx", "cz", "ecr", "swap", "rzx", "rxx", "ryy", "rzz"])));
            // NOTE: Make sure to update the list of the supported gates if adding more replacements
        } else {
            return Err(TranspilerError::new_err(format!(
                "{} with parameters {:?} is not supported on qubits {:?} in either direction.",
                packed_inst.op.name(),
                packed_inst.params_view(),
                op_args
            )));
        }
    }

    for (node, op_blocks) in ops_to_replace {
        let packed_inst = dag.dag()[node].unwrap_operation();
        let OperationRef::Instruction(py_inst) = packed_inst.op.view() else {
            panic!("PyInstruction is expected");
        };
        let new_op = py_inst
            .instruction
            .bind(py)
            .call_method1("replace_blocks", (op_blocks,))?;

        dag.py_substitute_node(dag.get_node(py, node)?.bind(py), &new_op, false, false)?;
    }

    for (node, replacemanet_dag) in nodes_to_replace {
        dag.py_substitute_node_with_dag(
            py,
            dag.get_node(py, node)?.bind(py),
            &replacemanet_dag,
            None,
            true,
        )?;
    }

    Ok(dag)
}

// Check whether the dag as calibration for a DAGOpNode
fn has_calibration_for_op_node(
    py: Python,
    dag: &DAGCircuit,
    packed_inst: &PackedInstruction,
    qargs: &[Qubit],
) -> PyResult<bool> {
    let py_args = PyTuple::new_bound(py, dag.qubits().map_indices(qargs));

    let dag_op_node = Py::new(
        py,
        (
            DAGOpNode {
                instruction: CircuitInstruction {
                    operation: packed_inst.op.clone(),
                    qubits: py_args.unbind(),
                    clbits: PyTuple::empty_bound(py).unbind(),
                    params: packed_inst.params_view().iter().cloned().collect(),
                    extra_attrs: packed_inst.extra_attrs.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: packed_inst.py_op.clone(),
                },
                sort_key: "".into_py(py),
            },
            DAGNode { node: None },
        ),
    )?;

    dag.has_calibration_for(py, dag_op_node.borrow(py))
}

// Return a replacement DAG for the given standard gate in the supported list
// TODO: optimize it by caching the DAGs of the non-parametric gates and caching and
// mutating upon request the DAGs of the parametric gates
fn replace_dag(
    py: Python,
    std_gate: StandardGate,
    inst: &PackedInstruction,
) -> PyResult<DAGCircuit> {
    let replacement_dag = match std_gate {
        StandardGate::CXGate => cx_replacement_dag(py),
        StandardGate::ECRGate => ecr_replacement_dag(py),
        StandardGate::CZGate => cz_replacement_dag(py),
        StandardGate::SwapGate => swap_replacement_dag(py),
        StandardGate::RXXGate => rxx_replacement_dag(py, inst.params_view()),
        StandardGate::RYYGate => ryy_replacement_dag(py, inst.params_view()),
        StandardGate::RZZGate => rzz_replacement_dag(py, inst.params_view()),
        StandardGate::RZXGate => rzx_replacement_dag(py, inst.params_view()),
        _ => panic!("Mismatch in supported gates assumption"),
    };

    replacement_dag
}

//###################################################
// Utility functions to build the replacement dags
//
// TODO: replace this once we have a Rust version of QuantumRegister
#[inline]
fn add_qreg(py: Python, dag: &mut DAGCircuit, num_qubits: u32) -> PyResult<Vec<Qubit>> {
    let qreg = imports::QUANTUM_REGISTER
        .get_bound(py)
        .call1((num_qubits,))?;
    dag.add_qreg(py, &qreg)?;
    let mut qargs = Vec::new();

    for i in 0..num_qubits {
        let qubit = qreg.call_method1(intern!(py, "__getitem__"), (i,))?;
        qargs.push(
            dag.qubits()
                .find(&qubit)
                .expect("Qubit should have been stored in the DAGCircuit"),
        );
    }

    Ok(qargs)
}

#[inline]
fn apply_operation_back(
    py: Python,
    dag: &mut DAGCircuit,
    gate: StandardGate,
    qargs: &[Qubit],
    param: Option<SmallVec<[Param; 3]>>,
) -> PyResult<()> {
    dag.apply_operation_back(
        py,
        PackedOperation::from_standard(gate),
        qargs,
        &[],
        param,
        ExtraInstructionAttributes::default(),
        #[cfg(feature = "cache_pygates")]
        None,
    )?;

    Ok(())
}

fn cx_replacement_dag(py: Python) -> PyResult<DAGCircuit> {
    let new_dag = &mut DAGCircuit::new(py)?;
    let qargs = add_qreg(py, new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(py, new_dag, StandardGate::HGate, &[qargs[0]], None)?;
    apply_operation_back(py, new_dag, StandardGate::HGate, &[qargs[1]], None)?;
    apply_operation_back(
        py,
        new_dag,
        StandardGate::CXGate,
        &[qargs[1], qargs[0]],
        None,
    )?;
    apply_operation_back(py, new_dag, StandardGate::HGate, &[qargs[0]], None)?;
    apply_operation_back(py, new_dag, StandardGate::HGate, &[qargs[1]], None)?;

    Ok(new_dag.clone())
}

fn ecr_replacement_dag(py: Python) -> PyResult<DAGCircuit> {
    let new_dag = &mut DAGCircuit::new(py)?;
    new_dag.add_global_phase(py, &Param::Float(-PI / 2.0))?;
    let qargs = add_qreg(py, new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(py, new_dag, StandardGate::SGate, &[qargs[0]], None)?;
    apply_operation_back(py, new_dag, StandardGate::SXGate, &[qargs[0]], None)?;
    apply_operation_back(py, new_dag, StandardGate::SdgGate, &[qargs[0]], None)?;
    apply_operation_back(py, new_dag, StandardGate::SdgGate, &[qargs[1]], None)?;
    apply_operation_back(py, new_dag, StandardGate::SXGate, &[qargs[1]], None)?;
    apply_operation_back(py, new_dag, StandardGate::SGate, &[qargs[1]], None)?;
    apply_operation_back(
        py,
        new_dag,
        StandardGate::ECRGate,
        &[qargs[1], qargs[0]],
        None,
    )?;
    apply_operation_back(py, new_dag, StandardGate::HGate, &[qargs[0]], None)?;
    apply_operation_back(py, new_dag, StandardGate::HGate, &[qargs[1]], None)?;

    Ok(new_dag.clone())
}

fn cz_replacement_dag(py: Python) -> PyResult<DAGCircuit> {
    let new_dag = &mut DAGCircuit::new(py)?;
    let qargs = add_qreg(py, new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(
        py,
        new_dag,
        StandardGate::CZGate,
        &[qargs[1], qargs[0]],
        None,
    )?;

    Ok(new_dag.clone())
}

fn swap_replacement_dag(py: Python) -> PyResult<DAGCircuit> {
    let new_dag = &mut DAGCircuit::new(py)?;
    let qargs = add_qreg(py, new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(
        py,
        new_dag,
        StandardGate::SwapGate,
        &[qargs[1], qargs[0]],
        None,
    )?;

    Ok(new_dag.clone())
}

fn rxx_replacement_dag(py: Python, param: &[Param]) -> PyResult<DAGCircuit> {
    let new_dag = &mut DAGCircuit::new(py)?;
    let qargs = add_qreg(py, new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(
        py,
        new_dag,
        StandardGate::RXXGate,
        &[qargs[1], qargs[0]],
        Some(SmallVec::from(param)),
    )?;

    Ok(new_dag.clone())
}

fn ryy_replacement_dag(py: Python, param: &[Param]) -> PyResult<DAGCircuit> {
    let new_dag = &mut DAGCircuit::new(py)?;
    let qargs = add_qreg(py, new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(
        py,
        new_dag,
        StandardGate::RYYGate,
        &[qargs[1], qargs[0]],
        Some(SmallVec::from(param)),
    )?;

    Ok(new_dag.clone())
}

fn rzz_replacement_dag(py: Python, param: &[Param]) -> PyResult<DAGCircuit> {
    let new_dag = &mut DAGCircuit::new(py)?;
    let qargs = add_qreg(py, new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(
        py,
        new_dag,
        StandardGate::RZZGate,
        &[qargs[1], qargs[0]],
        Some(SmallVec::from(param)),
    )?;

    Ok(new_dag.clone())
}

fn rzx_replacement_dag(py: Python, param: &[Param]) -> PyResult<DAGCircuit> {
    let new_dag = &mut DAGCircuit::new(py)?;
    let qargs = add_qreg(py, new_dag, 2)?;
    let qargs = qargs.as_slice();

    apply_operation_back(py, new_dag, StandardGate::HGate, &[qargs[0]], None)?;
    apply_operation_back(py, new_dag, StandardGate::HGate, &[qargs[1]], None)?;
    apply_operation_back(
        py,
        new_dag,
        StandardGate::RZXGate,
        &[qargs[1], qargs[0]],
        Some(SmallVec::from(param)),
    )?;
    apply_operation_back(py, new_dag, StandardGate::HGate, &[qargs[0]], None)?;
    apply_operation_back(py, new_dag, StandardGate::HGate, &[qargs[1]], None)?;

    Ok(new_dag.clone())
}

#[pymodule]
pub fn gate_direction(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_check_direction_coupling_map))?;
    m.add_wrapped(wrap_pyfunction!(py_check_direction_target))?;
    m.add_wrapped(wrap_pyfunction!(py_fix_direction_coupling_map))?;
    m.add_wrapped(wrap_pyfunction!(py_fix_direction_target))?;
    Ok(())
}

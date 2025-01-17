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

use hashbrown::HashSet;
use pyo3::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::circuit_instruction::ExtraInstructionAttributes;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::imports::BARRIER;
use qiskit_circuit::operations::{Operation, PyInstruction};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::Qubit;

static FINAL_OP_NAMES: [&str; 2] = ["measure", "barrier"];

#[pyfunction]
#[pyo3(signature=(dag, label=None))]
pub fn barrier_before_final_measurements(
    py: Python,
    dag: &mut DAGCircuit,
    label: Option<String>,
) -> PyResult<()> {
    let final_ops: HashSet<NodeIndex> = dag
        .op_nodes(true)
        .filter(|node| {
            let NodeType::Operation(ref inst) = dag.dag()[*node] else {
                unreachable!();
            };
            if !FINAL_OP_NAMES.contains(&inst.op.name()) {
                return false;
            }
            let is_final_op = dag.bfs_successors(*node).all(|(_, child_successors)| {
                !child_successors.iter().any(|suc| match dag.dag()[*suc] {
                    NodeType::Operation(ref suc_inst) => {
                        !FINAL_OP_NAMES.contains(&suc_inst.op.name())
                    }
                    _ => false,
                })
            });
            is_final_op
        })
        .collect();
    if final_ops.is_empty() {
        return Ok(());
    }
    let ordered_node_indices: Vec<NodeIndex> = dag
        .topological_op_nodes()?
        .filter(|node| final_ops.contains(node))
        .collect();
    let final_packed_ops: Vec<PackedInstruction> = ordered_node_indices
        .into_iter()
        .map(|node| {
            let NodeType::Operation(ref inst) = dag.dag()[node] else {
                unreachable!()
            };
            let res = inst.clone();
            dag.remove_op_node(node);
            res
        })
        .collect();
    let new_barrier = BARRIER
        .get_bound(py)
        .call1((dag.num_qubits(), label.as_deref()))?;

    let new_barrier_py_inst = PyInstruction {
        qubits: dag.num_qubits() as u32,
        clbits: 0,
        params: 0,
        op_name: "barrier".to_string(),
        control_flow: false,
        #[cfg(feature = "cache_pygates")]
        instruction: new_barrier.clone().unbind(),
        #[cfg(not(feature = "cache_pygates"))]
        instruction: new_barrier.unbind(),
    };
    let qargs: Vec<Qubit> = (0..dag.num_qubits() as u32).map(Qubit).collect();
    #[cfg(feature = "cache_pygates")]
    {
        dag.apply_operation_back(
            py,
            PackedOperation::from_instruction(Box::new(new_barrier_py_inst)),
            qargs.as_slice(),
            &[],
            None,
            ExtraInstructionAttributes::new(label, None, None, None),
            Some(new_barrier.unbind()),
        )?;
    }
    #[cfg(not(feature = "cache_pygates"))]
    {
        dag.apply_operation_back(
            py,
            PackedOperation::from_instruction(Box::new(new_barrier_py_inst)),
            qargs.as_slice(),
            &[],
            None,
            ExtraInstructionAttributes::new(label, None, None, None),
        )?;
    }
    for inst in final_packed_ops {
        dag.push_back(py, inst)?;
    }
    Ok(())
}

pub fn barrier_before_final_measurements_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(barrier_before_final_measurements))?;
    Ok(())
}

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
use std::f64::consts::PI;
const PI4: f64 = PI / 4.;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::imports::UNITARY_GATE;
use qiskit_circuit::operations::{Operation, Param};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::Qubit;

use crate::two_qubit_decompose::{Specialization, TwoQubitWeylDecomposition};

/// Extracts the K1r and K1l gates from the decomposition
/// and creates them as new 1-qubit unitary gates.
fn create_k1_gates<'a>(
    decomp: &'a TwoQubitWeylDecomposition,
    py: Python<'a>,
) -> PyResult<(Bound<'a, PyAny>, Bound<'a, PyAny>)> {
    let k1r_arr = decomp.K1r(py);
    let k1l_arr = decomp.K1l(py);
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item(intern!(py, "num_qubits"), 1)?;
    let k1r_gate = UNITARY_GATE
        .get_bound(py)
        .call((k1r_arr, py.None(), false), Some(&kwargs))?;
    let k1l_gate = UNITARY_GATE
        .get_bound(py)
        .call((k1l_arr, py.None(), false), Some(&kwargs))?;
    Ok((k1r_gate, k1l_gate))
}

/// Creates a new instruction and adds it to the DAG.
fn add_new_op(
    new_dag: &mut DAGCircuit,
    new_op: OperationFromPython,
    qargs: Vec<Qubit>,
    mapping: &[usize],
    py: Python,
) -> PyResult<()> {
    let inst = PackedInstruction {
        op: new_op.operation,
        qubits: new_dag.qargs_interner.insert_owned(qargs),
        clbits: new_dag.cargs_interner.get_default(),
        params: (!new_op.params.is_empty()).then(|| Box::new(new_op.params)),
        extra_attrs: new_op.extra_attrs,
        #[cfg(feature = "cache_pygates")]
        py_op: std::sync::OnceLock::new(),
    };
    let qargs = new_dag.get_qargs(inst.qubits);
    let mapped_qargs: Vec<Qubit> = qargs
        .iter()
        .map(|q| Qubit::new(mapping[q.index()]))
        .collect();
    new_dag.apply_operation_back(
        py,
        inst.op.clone(),
        &mapped_qargs,
        &[],
        inst.params.as_deref().cloned(),
        inst.extra_attrs,
        #[cfg(feature = "cache_pygates")]
        None,
    )?;
    Ok(())
}

#[pyfunction]
pub fn split_2q_unitaries(
    py: Python,
    dag: &mut DAGCircuit,
    requested_fidelity: f64,
) -> PyResult<Option<(DAGCircuit, Vec<usize>)>> {
    if !dag.get_op_counts().contains_key("unitary") {
        return Ok(None);
    }
    let nodes: Vec<NodeIndex> = dag.op_nodes(false).collect();
    let mut has_swaps = false;
    for node in nodes {
        if let NodeType::Operation(inst) = &dag.dag()[node] {
            let qubits = dag.get_qargs(inst.qubits).to_vec();
            // We only attempt to split UnitaryGate objects, but this could be extended in future
            // -- however we need to ensure that we can compile the resulting single-qubit unitaries
            // to the supported basis gate set.
            if qubits.len() != 2 || inst.op.name() != "unitary" {
                continue;
            }
            let matrix = inst
                .op
                .matrix(inst.params_view())
                .expect("'unitary' gates should always have a matrix form");
            let decomp = TwoQubitWeylDecomposition::new_inner(
                matrix.view(),
                Some(requested_fidelity),
                None,
            )?;
            if matches!(decomp.specialization, Specialization::SWAPEquiv) {
                has_swaps = true;
            }
            if matches!(decomp.specialization, Specialization::IdEquiv) {
                let (k1r_gate, k1l_gate) = create_k1_gates(&decomp, py)?;
                let insert_fn = |edge: &Wire| -> PyResult<OperationFromPython> {
                    if let Wire::Qubit(qubit) = edge {
                        if *qubit == qubits[0] {
                            k1r_gate.extract()
                        } else {
                            k1l_gate.extract()
                        }
                    } else {
                        unreachable!("This will only be called on ops with no classical wires.");
                    }
                };
                dag.replace_node_with_1q_ops(py, node, insert_fn)?;
                dag.add_global_phase(py, &Param::Float(decomp.global_phase))?;
            }
        }
    }
    if !has_swaps {
        return Ok(None);
    }
    // We have swap-like unitaries, so we create a new DAG in a manner similar to
    // The Elide Permutations pass, while also splitting the unitaries to 1-qubit gates
    let mut mapping: Vec<usize> = (0..dag.num_qubits()).collect();
    let mut new_dag = dag.copy_empty_like(py, "alike")?;
    for node in dag.topological_op_nodes()? {
        if let NodeType::Operation(inst) = &dag.dag()[node] {
            let qubits = dag.get_qargs(inst.qubits).to_vec();
            let mut is_swap = false;
            if qubits.len() == 2 && inst.op.name() == "unitary" {
                let matrix = inst
                    .op
                    .matrix(inst.params_view())
                    .expect("'unitary' gates should always have a matrix form");
                let decomp = TwoQubitWeylDecomposition::new_inner(
                    matrix.view(),
                    Some(requested_fidelity),
                    None,
                )?;
                if matches!(decomp.specialization, Specialization::SWAPEquiv) {
                    // perform the virtual swap
                    is_swap = true;
                    let qargs = dag.get_qargs(inst.qubits);
                    let index0 = qargs[0].index();
                    let index1 = qargs[1].index();
                    mapping.swap(index0, index1);
                    // now add the two 1-qubit gates
                    let (k1r_gate, k1l_gate) = create_k1_gates(&decomp, py)?;
                    add_new_op(
                        &mut new_dag,
                        k1r_gate.extract()?,
                        vec![qubits[0]],
                        &mapping,
                        py,
                    )?;
                    add_new_op(
                        &mut new_dag,
                        k1l_gate.extract()?,
                        vec![qubits[1]],
                        &mapping,
                        py,
                    )?;
                    new_dag.add_global_phase(py, &Param::Float(decomp.global_phase + PI4))?;
                }
            }
            if !is_swap {
                // General instruction
                let qargs = dag.get_qargs(inst.qubits);
                let cargs = dag.get_cargs(inst.clbits);
                let mapped_qargs: Vec<Qubit> = qargs
                    .iter()
                    .map(|q| Qubit::new(mapping[q.index()]))
                    .collect();

                new_dag.apply_operation_back(
                    py,
                    inst.op.clone(),
                    &mapped_qargs,
                    cargs,
                    inst.params.as_deref().cloned(),
                    inst.extra_attrs.clone(),
                    #[cfg(feature = "cache_pygates")]
                    inst.py_op.get().map(|x| x.clone_ref(py)),
                )?;
            }
        }
    }
    Ok(Some((new_dag, mapping)))
}

pub fn split_2q_unitaries_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(split_2q_unitaries))?;
    Ok(())
}

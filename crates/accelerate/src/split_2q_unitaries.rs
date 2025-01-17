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

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::imports::UNITARY_GATE;
use qiskit_circuit::operations::{Operation, Param};

use crate::two_qubit_decompose::{Specialization, TwoQubitWeylDecomposition};

#[pyfunction]
pub fn split_2q_unitaries(
    py: Python,
    dag: &mut DAGCircuit,
    requested_fidelity: f64,
) -> PyResult<()> {
    if !dag.get_op_counts().contains_key("unitary") {
        return Ok(());
    }
    let nodes: Vec<NodeIndex> = dag.op_nodes(false).collect();

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
            if matches!(decomp.specialization, Specialization::IdEquiv) {
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
            // TODO: also look into splitting on Specialization::Swap and just
            // swap the virtual qubits. Doing this we will need to update the
            // permutation like in ElidePermutations
        }
    }
    Ok(())
}

pub fn split_2q_unitaries_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(split_2q_unitaries))?;
    Ok(())
}

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

/// Remove diagonal gates (including diagonal 2Q gates) before a measurement.
use pyo3::prelude::*;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::StandardGate;

/// Run the RemoveDiagonalGatesBeforeMeasure pass on `dag`.
/// Args:
///     dag (DAGCircuit): the DAG to be optimized.
/// Returns:
///     DAGCircuit: the optimized DAG.
#[pyfunction]
#[pyo3(name = "remove_diagonal_gates_before_measure")]
fn run_remove_diagonal_before_measure(dag: &mut DAGCircuit) -> PyResult<()> {
    static DIAGONAL_1Q_GATES: [StandardGate; 8] = [
        StandardGate::RZGate,
        StandardGate::ZGate,
        StandardGate::TGate,
        StandardGate::SGate,
        StandardGate::TdgGate,
        StandardGate::SdgGate,
        StandardGate::U1Gate,
        StandardGate::PhaseGate,
    ];
    static DIAGONAL_2Q_GATES: [StandardGate; 7] = [
        StandardGate::CZGate,
        StandardGate::CRZGate,
        StandardGate::CU1Gate,
        StandardGate::RZZGate,
        StandardGate::CPhaseGate,
        StandardGate::CSGate,
        StandardGate::CSdgGate,
    ];
    static DIAGONAL_3Q_GATES: [StandardGate; 1] = [StandardGate::CCZGate];

    let mut nodes_to_remove = Vec::new();
    for index in dag.op_nodes(true) {
        let node = &dag.dag[index];
        let NodeType::Operation(inst) = node else {
            panic!()
        };

        if inst.op.name() == "measure" {
            let predecessor = (dag.quantum_predecessors(index))
                .next()
                .expect("index is an operation node, so it must have a predecessor.");

            match &dag.dag[predecessor] {
                NodeType::Operation(pred_inst) => match pred_inst.standard_gate() {
                    Some(gate) => {
                        if DIAGONAL_1Q_GATES.contains(&gate) {
                            nodes_to_remove.push(predecessor);
                        } else if DIAGONAL_2Q_GATES.contains(&gate)
                            || DIAGONAL_3Q_GATES.contains(&gate)
                        {
                            let successors = dag.quantum_successors(predecessor);
                            let remove_s = successors
                                .map(|s| {
                                    let node_s = &dag.dag[s];
                                    if let NodeType::Operation(inst_s) = node_s {
                                        inst_s.op.name() == "measure"
                                    } else {
                                        false
                                    }
                                })
                                .all(|ok_to_remove| ok_to_remove);
                            if remove_s {
                                nodes_to_remove.push(predecessor);
                            }
                        }
                    }
                    None => {
                        continue;
                    }
                },
                _ => {
                    continue;
                }
            }
        }
    }

    for node_to_remove in nodes_to_remove {
        if dag.dag.node_weight(node_to_remove).is_some() {
            dag.remove_op_node(node_to_remove);
        }
    }

    Ok(())
}

#[pymodule]
pub fn remove_diagonal_gates_before_measure(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_remove_diagonal_before_measure))?;
    Ok(())
}

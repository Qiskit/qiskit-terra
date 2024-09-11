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

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::packed_instruction::PackedInstruction;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

#[pyfunction]
#[pyo3(name = "filter_op_nodes")]
pub fn py_filter_op_nodes(
    py: Python,
    dag: &mut DAGCircuit,
    predicate: &Bound<PyAny>,
) -> PyResult<()> {
    let callable = |node: NodeIndex| -> PyResult<bool> {
        let dag_op_node = dag.get_node(py, node)?;
        predicate.call1((dag_op_node,))?.extract()
    };
    let mut remove_nodes: Vec<NodeIndex> = Vec::new();
    for node in dag.op_nodes(true) {
        if !callable(node)? {
            remove_nodes.push(node);
        }
    }
    for node in remove_nodes {
        dag.remove_op_node(node);
    }
    Ok(())
}

/// Remove any nodes that have the provided label set
///
/// Args:
///     dag (DAGCircuit): The dag circuit to filter the ops from
///     label (str): The label to filter nodes on
#[pyfunction]
pub fn filter_labeled_op(dag: &mut DAGCircuit, label: String) {
    let predicate = |node: &PackedInstruction| -> bool {
        match node.label() {
            Some(inst_label) => inst_label != label,
            None => false,
        }
    };
    dag.filter_op_nodes(predicate);
}

pub fn filter_op_nodes_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_filter_op_nodes))?;
    m.add_wrapped(wrap_pyfunction!(filter_labeled_op))?;
    Ok(())
}

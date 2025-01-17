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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::PyModule;
use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult, Python};
use qiskit_circuit::Qubit;

use crate::commutation_checker::CommutationChecker;
use hashbrown::HashMap;
use pyo3::prelude::*;

use pyo3::types::{PyDict, PyList};
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use rustworkx_core::petgraph::stable_graph::NodeIndex;

// Custom types to store the commutation sets and node indices,
// see the docstring below for more information.
type CommutationSet = HashMap<Wire, Vec<Vec<NodeIndex>>>;
type NodeIndices = HashMap<(NodeIndex, Wire), usize>;

// the maximum number of qubits we check commutativity for
const MAX_NUM_QUBITS: u32 = 3;

/// Compute the commutation sets for a given DAG.
///
/// We return two HashMaps:
///  * {wire: commutation_sets}: For each wire, we keep a vector of index sets, where each index
///     set contains mutually commuting nodes. Note that these include the input and output nodes
///     which do not commute with anything.
///  * {(node, wire): index}: For each (node, wire) pair we store the index indicating in which
///     commutation set the node appears on a given wire.
///
/// For example, if we have a circuit
///
///     |0> -- X -- SX -- Z (out)
///      0     2    3     4   1   <-- node indices including input (0) and output (1) nodes
///
/// Then we would have
///
///     commutation_set = {0: [[0], [2, 3], [4], [1]]}
///     node_indices = {(0, 0): 0, (1, 0): 3, (2, 0): 1, (3, 0): 1, (4, 0): 2}
///
pub(crate) fn analyze_commutations_inner(
    py: Python,
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
) -> PyResult<(CommutationSet, NodeIndices)> {
    let mut commutation_set: CommutationSet = HashMap::new();
    let mut node_indices: NodeIndices = HashMap::new();

    for qubit in 0..dag.num_qubits() {
        let wire = Wire::Qubit(Qubit(qubit as u32));

        for current_gate_idx in dag.nodes_on_wire(&wire, false) {
            // get the commutation set associated with the current wire, or create a new
            // index set containing the current gate
            let commutation_entry = commutation_set
                .entry(wire.clone())
                .or_insert_with(|| vec![vec![current_gate_idx]]);

            // we can unwrap as we know the commutation entry has at least one element
            let last = commutation_entry.last_mut().unwrap();

            // if the current gate index is not in the set, check whether it commutes with
            // the previous nodes -- if yes, add it to the commutation set
            if !last.contains(&current_gate_idx) {
                let mut all_commute = true;

                for prev_gate_idx in last.iter() {
                    // if the node is an input/output node, they do not commute, so we only
                    // continue if the nodes are operation nodes
                    if let (NodeType::Operation(packed_inst0), NodeType::Operation(packed_inst1)) =
                        (&dag.dag()[current_gate_idx], &dag.dag()[*prev_gate_idx])
                    {
                        let op1 = packed_inst0.op.view();
                        let op2 = packed_inst1.op.view();
                        let params1 = packed_inst0.params_view();
                        let params2 = packed_inst1.params_view();
                        let qargs1 = dag.get_qargs(packed_inst0.qubits);
                        let qargs2 = dag.get_qargs(packed_inst1.qubits);
                        let cargs1 = dag.get_cargs(packed_inst0.clbits);
                        let cargs2 = dag.get_cargs(packed_inst1.clbits);

                        all_commute = commutation_checker.commute_inner(
                            py,
                            &op1,
                            params1,
                            &packed_inst0.extra_attrs,
                            qargs1,
                            cargs1,
                            &op2,
                            params2,
                            &packed_inst1.extra_attrs,
                            qargs2,
                            cargs2,
                            MAX_NUM_QUBITS,
                        )?;
                        if !all_commute {
                            break;
                        }
                    } else {
                        all_commute = false;
                        break;
                    }
                }

                if all_commute {
                    // all commute, add to current list
                    last.push(current_gate_idx);
                } else {
                    // does not commute, create new list
                    commutation_entry.push(vec![current_gate_idx]);
                }
            }

            node_indices.insert(
                (current_gate_idx, wire.clone()),
                commutation_entry.len() - 1,
            );
        }
    }

    Ok((commutation_set, node_indices))
}

#[pyfunction]
#[pyo3(signature = (dag, commutation_checker))]
pub(crate) fn analyze_commutations(
    py: Python,
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
) -> PyResult<Py<PyDict>> {
    // This returns two HashMaps:
    //   * The commuting nodes per wire: {wire: [commuting_nodes_1, commuting_nodes_2, ...]}
    //   * The index in which commutation set a given node is located on a wire: {(node, wire): index}
    // The Python dict will store both of these dictionaries in one.
    let (commutation_set, node_indices) = analyze_commutations_inner(py, dag, commutation_checker)?;

    let out_dict = PyDict::new_bound(py);

    // First set the {wire: [commuting_nodes_1, ...]} bit
    for (wire, commutations) in commutation_set {
        // we know all wires are of type Wire::Qubit, since in analyze_commutations_inner
        // we only iterater over the qubits
        let py_wire = match wire {
            Wire::Qubit(q) => dag.qubits().get(q).unwrap().to_object(py),
            _ => return Err(PyValueError::new_err("Unexpected wire type.")),
        };

        out_dict.set_item(
            py_wire,
            PyList::new_bound(
                py,
                commutations.iter().map(|inner| {
                    PyList::new_bound(
                        py,
                        inner
                            .iter()
                            .map(|node_index| dag.get_node(py, *node_index).unwrap()),
                    )
                }),
            ),
        )?;
    }

    // Then we add the {(node, wire): index} dictionary
    for ((node_index, wire), index) in node_indices {
        let py_wire = match wire {
            Wire::Qubit(q) => dag.qubits().get(q).unwrap().to_object(py),
            _ => return Err(PyValueError::new_err("Unexpected wire type.")),
        };
        out_dict.set_item((dag.get_node(py, node_index)?, py_wire), index)?;
    }

    Ok(out_dict.unbind())
}

pub fn commutation_analysis(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(analyze_commutations))?;
    Ok(())
}

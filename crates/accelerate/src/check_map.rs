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
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::imports::CIRCUIT_TO_DAG;
use qiskit_circuit::operations::{Operation, OperationRef};
use qiskit_circuit::Qubit;

fn recurse<'py>(
    py: Python<'py>,
    dag: &'py DAGCircuit,
    edge_set: &'py HashSet<[u32; 2]>,
    wire_map: Option<&'py [Qubit]>,
) -> PyResult<Option<(String, [u32; 2])>> {
    let check_qubits = |qubits: &[Qubit]| -> bool {
        match wire_map {
            Some(wire_map) => {
                let mapped_bits = [wire_map[qubits[0].index()], wire_map[qubits[1].index()]];
                edge_set.contains(&[mapped_bits[0].into(), mapped_bits[1].into()])
            }
            None => edge_set.contains(&[qubits[0].into(), qubits[1].into()]),
        }
    };
    for node in dag.op_nodes(false) {
        if let NodeType::Operation(inst) = &dag.dag()[node] {
            let qubits = dag.get_qargs(inst.qubits);
            if inst.op.control_flow() {
                if let OperationRef::Instruction(py_inst) = inst.op.view() {
                    let raw_blocks = py_inst.instruction.getattr(py, "blocks")?;
                    let circuit_to_dag = CIRCUIT_TO_DAG.get_bound(py);
                    for raw_block in raw_blocks.bind(py).iter().unwrap() {
                        let block_obj = raw_block?;
                        let block = block_obj
                            .getattr(intern!(py, "_data"))?
                            .downcast::<CircuitData>()?
                            .borrow();
                        let new_dag: DAGCircuit =
                            circuit_to_dag.call1((block_obj.clone(),))?.extract()?;
                        let wire_map = (0..block.num_qubits())
                            .map(|inner| {
                                let outer = qubits[inner];
                                match wire_map {
                                    Some(wire_map) => wire_map[outer.index()],
                                    None => outer,
                                }
                            })
                            .collect::<Vec<_>>();
                        let res = recurse(py, &new_dag, edge_set, Some(&wire_map))?;
                        if res.is_some() {
                            return Ok(res);
                        }
                    }
                }
            } else if qubits.len() == 2
                && (dag.calibrations_empty() || !dag.has_calibration_for_index(py, node)?)
                && !check_qubits(qubits)
            {
                return Ok(Some((
                    inst.op.name().to_string(),
                    [qubits[0].0, qubits[1].0],
                )));
            }
        }
    }
    Ok(None)
}

#[pyfunction]
pub fn check_map(
    py: Python,
    dag: &DAGCircuit,
    edge_set: HashSet<[u32; 2]>,
) -> PyResult<Option<(String, [u32; 2])>> {
    recurse(py, dag, &edge_set, None)
}

pub fn check_map_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(check_map))?;
    Ok(())
}

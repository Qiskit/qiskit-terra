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
use crate::target_transpiler::Target;
use hashbrown::HashSet;
use pyo3::prelude::*;
use qiskit_circuit::imports;
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::{
    dag_circuit::{DAGCircuit, NodeType},
    operations::Operation,
    packed_instruction::PackedInstruction,
    Qubit,
};
use smallvec::smallvec;

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
fn py_check_with_coupling_map(
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
fn py_check_with_target(py: Python, dag: &DAGCircuit, target: &Target) -> PyResult<bool> {
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
        let NodeType::Operation(packed_inst) = &dag.dag[node] else {
            panic!("PackedInstruction is expected");
        };

        let inst_qargs = dag.get_qargs(packed_inst.qubits);

        if let OperationRef::Instruction(py_inst) = packed_inst.op.view() {
            if py_inst.control_flow() {
                let circuit_to_dag = imports::CIRCUIT_TO_DAG.get_bound(py); // TODO: Take out of the recursion
                let py_inst = py_inst.instruction.bind(py);

                for block in py_inst.getattr("blocks")?.iter()? {
                    let inner_dag: DAGCircuit = circuit_to_dag.call1((block?,))?.extract()?;

                    let block_ok = if let Some(mapping) = qubit_mapping {
                        let mapping = inst_qargs // Create a temp mapping for the recursive call
                            .iter()
                            .map(|q| mapping[q.0 as usize])
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
                        mapping[inst_qargs[0].0 as usize],
                        mapping[inst_qargs[1].0 as usize],
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

#[pymodule]
pub fn gate_direction(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_check_with_coupling_map))?;
    m.add_wrapped(wrap_pyfunction!(py_check_with_target))?;
    Ok(())
}

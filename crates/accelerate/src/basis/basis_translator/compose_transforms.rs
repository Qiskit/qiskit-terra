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

use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::imports::{GATE, PARAMETER_VECTOR, QUANTUM_REGISTER};
use qiskit_circuit::parameter_table::ParameterUuid;
use qiskit_circuit::Qubit;
use qiskit_circuit::{
    circuit_data::CircuitData,
    dag_circuit::{DAGCircuit, NodeType},
    operations::{Operation, Param},
};
use smallvec::SmallVec;

use crate::equivalence::CircuitFromPython;

// Custom types
pub type GateIdentifier = (String, u32);
pub type BasisTransformIn = (SmallVec<[Param; 3]>, CircuitFromPython);
pub type BasisTransformOut = (SmallVec<[Param; 3]>, DAGCircuit);

pub(super) fn compose_transforms<'a>(
    py: Python,
    basis_transforms: &'a [(GateIdentifier, BasisTransformIn)],
    source_basis: &'a HashSet<GateIdentifier>,
    source_dag: &'a DAGCircuit,
) -> PyResult<HashMap<GateIdentifier, BasisTransformOut>> {
    let mut gate_param_counts: HashMap<GateIdentifier, usize> = HashMap::default();
    get_gates_num_params(source_dag, &mut gate_param_counts)?;
    let mut mapped_instructions: HashMap<GateIdentifier, BasisTransformOut> = HashMap::new();

    for (gate_name, gate_num_qubits) in source_basis.iter().cloned() {
        let num_params = gate_param_counts[&(gate_name.clone(), gate_num_qubits)];

        let placeholder_params: SmallVec<[Param; 3]> = PARAMETER_VECTOR
            .get_bound(py)
            .call1((&gate_name, num_params))?
            .extract()?;

        let mut dag = DAGCircuit::new(py)?;
        // Create the mock gate and add to the circuit, use Python for this.
        let qubits = QUANTUM_REGISTER.get_bound(py).call1((gate_num_qubits,))?;
        dag.add_qreg(py, &qubits)?;

        let gate = GATE.get_bound(py).call1((
            &gate_name,
            gate_num_qubits,
            placeholder_params
                .iter()
                .map(|x| x.clone_ref(py))
                .collect::<SmallVec<[Param; 3]>>(),
        ))?;
        let gate_obj: OperationFromPython = gate.extract()?;
        let qubits: Vec<Qubit> = (0..dag.num_qubits() as u32).map(Qubit).collect();
        dag.apply_operation_back(
            py,
            gate_obj.operation,
            &qubits,
            &[],
            if gate_obj.params.is_empty() {
                None
            } else {
                Some(gate_obj.params)
            },
            gate_obj.extra_attrs,
            #[cfg(feature = "cache_pygates")]
            Some(gate.into()),
        )?;
        mapped_instructions.insert((gate_name, gate_num_qubits), (placeholder_params, dag));

        for ((gate_name, gate_num_qubits), (equiv_params, equiv)) in basis_transforms {
            for (_, dag) in &mut mapped_instructions.values_mut() {
                let nodes_to_replace = dag
                    .op_nodes(true)
                    .filter_map(|node| {
                        if let Some(NodeType::Operation(op)) = dag.dag().node_weight(node) {
                            if (gate_name.as_str(), *gate_num_qubits)
                                == (op.op.name(), op.op.num_qubits())
                            {
                                Some((
                                    node,
                                    op.params_view()
                                        .iter()
                                        .map(|x| x.clone_ref(py))
                                        .collect::<SmallVec<[Param; 3]>>(),
                                ))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                for (node, params) in nodes_to_replace {
                    let param_mapping: HashMap<ParameterUuid, Param> = equiv_params
                        .iter()
                        .map(|x| ParameterUuid::from_parameter(x.to_object(py).bind(py)))
                        .zip(params)
                        .map(|(uuid, param)| -> PyResult<(ParameterUuid, Param)> {
                            Ok((uuid?, param.clone_ref(py)))
                        })
                        .collect::<PyResult<_>>()?;
                    let mut replacement = equiv.clone();
                    replacement
                        .0
                        .assign_parameters_from_mapping(py, param_mapping)?;
                    let replace_dag: DAGCircuit =
                        DAGCircuit::from_circuit_data(py, replacement.0, true)?;
                    let op_node = dag.get_node(py, node)?;
                    dag.py_substitute_node_with_dag(
                        py,
                        op_node.bind(py),
                        &replace_dag,
                        None,
                        true,
                    )?;
                }
            }
        }
    }
    Ok(mapped_instructions)
}

/// `DAGCircuit` variant.
///
/// Gets the identifier of a gate instance (name, number of qubits) mapped to the
/// number of parameters it contains currently.
fn get_gates_num_params(
    dag: &DAGCircuit,
    example_gates: &mut HashMap<GateIdentifier, usize>,
) -> PyResult<()> {
    for node in dag.op_nodes(true) {
        if let Some(NodeType::Operation(op)) = dag.dag().node_weight(node) {
            example_gates.insert(
                (op.op.name().to_string(), op.op.num_qubits()),
                op.params_view().len(),
            );
            if op.op.control_flow() {
                let blocks = op.op.blocks();
                for block in blocks {
                    get_gates_num_params_circuit(&block, example_gates)?;
                }
            }
        }
    }
    Ok(())
}

/// `CircuitData` variant.
///
/// Gets the identifier of a gate instance (name, number of qubits) mapped to the
/// number of parameters it contains currently.
fn get_gates_num_params_circuit(
    circuit: &CircuitData,
    example_gates: &mut HashMap<GateIdentifier, usize>,
) -> PyResult<()> {
    for inst in circuit.iter() {
        example_gates.insert(
            (inst.op.name().to_string(), inst.op.num_qubits()),
            inst.params_view().len(),
        );
        if inst.op.control_flow() {
            let blocks = inst.op.blocks();
            for block in blocks {
                get_gates_num_params_circuit(&block, example_gates)?;
            }
        }
    }
    Ok(())
}

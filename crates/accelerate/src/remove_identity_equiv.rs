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

use num_complex::Complex64;
use num_complex::ComplexFloat;
use pyo3::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::Target;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit::packed_instruction::PackedInstruction;

#[pyfunction]
#[pyo3(signature=(dag, approx_degree=Some(1.0), target=None))]
fn remove_identity_equiv(
    dag: &mut DAGCircuit,
    approx_degree: Option<f64>,
    target: Option<&Target>,
) {
    let mut remove_list: Vec<NodeIndex> = Vec::new();

    let get_error_cutoff = |inst: &PackedInstruction| -> f64 {
        match approx_degree {
            Some(degree) => {
                if degree == 1.0 {
                    f64::EPSILON
                } else {
                    match target {
                        Some(target) => {
                            let qargs: Vec<PhysicalQubit> = dag
                                .get_qargs(inst.qubits)
                                .iter()
                                .map(|x| PhysicalQubit::new(x.0))
                                .collect();
                            let error_rate = target.get_error(inst.op.name(), qargs.as_slice());
                            match error_rate {
                                Some(err) => err * degree,
                                None => f64::EPSILON.max(1. - degree),
                            }
                        }
                        None => f64::EPSILON.max(1. - degree),
                    }
                }
            }
            None => match target {
                Some(target) => {
                    let qargs: Vec<PhysicalQubit> = dag
                        .get_qargs(inst.qubits)
                        .iter()
                        .map(|x| PhysicalQubit::new(x.0))
                        .collect();
                    let error_rate = target.get_error(inst.op.name(), qargs.as_slice());
                    match error_rate {
                        Some(err) => err,
                        None => f64::EPSILON,
                    }
                }
                None => f64::EPSILON,
            },
        }
    };

    for op_node in dag.op_nodes(false) {
        let inst = dag.dag()[op_node].unwrap_operation();
        match inst.op.view() {
            OperationRef::Standard(gate) => {
                let (dim, trace) = match gate {
                    StandardGate::RXGate | StandardGate::RYGate | StandardGate::RZGate => {
                        if let Param::Float(theta) = inst.params_view()[0] {
                            let trace = (theta / 2.).cos() * 2.;
                            (2., trace)
                        } else {
                            continue;
                        }
                    }
                    StandardGate::RXXGate
                    | StandardGate::RYYGate
                    | StandardGate::RZZGate
                    | StandardGate::RZXGate => {
                        if let Param::Float(theta) = inst.params_view()[0] {
                            let trace = (theta / 2.).cos() * 4.;
                            (4., trace)
                        } else {
                            continue;
                        }
                    }
                    _ => {
                        // Skip global phase gate
                        if gate.num_qubits() < 1 {
                            continue;
                        }
                        if let Some(matrix) = gate.matrix(inst.params_view()) {
                            let dim = matrix.shape()[0] as f64;
                            let trace = matrix.diag().iter().sum::<Complex64>().abs();
                            (dim, trace)
                        } else {
                            continue;
                        }
                    }
                };
                let error = get_error_cutoff(inst);
                let f_pro = (trace / dim).powi(2);
                let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
                if (1. - gate_fidelity).abs() < error {
                    remove_list.push(op_node)
                }
            }
            OperationRef::Gate(gate) => {
                // Skip global phase like gate
                if gate.num_qubits() < 1 {
                    continue;
                }
                if let Some(matrix) = gate.matrix(inst.params_view()) {
                    let error = get_error_cutoff(inst);
                    let dim = matrix.shape()[0] as f64;
                    let trace: Complex64 = matrix.diag().iter().sum();
                    let f_pro = (trace / dim).abs().powi(2);
                    let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
                    if (1. - gate_fidelity).abs() < error {
                        remove_list.push(op_node)
                    }
                }
            }
            _ => continue,
        }
    }
    for node in remove_list {
        dag.remove_op_node(node);
    }
}

pub fn remove_identity_equiv_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(remove_identity_equiv))?;
    Ok(())
}

// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
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
use pyo3::Python;

use num_complex::Complex64;
use numpy::ndarray::linalg::kron;
use numpy::ndarray::{aview2, Array2, ArrayView2};
use numpy::PyReadonlyArray2;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::gate_matrix::ONE_QUBIT_IDENTITY;
use qiskit_circuit::imports::QI_OPERATOR;
use qiskit_circuit::operations::{Operation, OperationRef};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::Qubit;

use crate::euler_one_qubit_decomposer::matmul_1q;
use crate::QiskitError;

#[inline]
pub fn get_matrix_from_inst(py: Python, inst: &PackedInstruction) -> PyResult<Array2<Complex64>> {
    if let Some(mat) = inst.op.matrix(inst.params_view()) {
        Ok(mat)
    } else if inst.op.try_standard_gate().is_some() {
        Err(QiskitError::new_err(
            "Parameterized gates can't be consolidated",
        ))
    } else if let OperationRef::Gate(gate) = inst.op.view() {
        Ok(QI_OPERATOR
            .get_bound(py)
            .call1((gate.gate.clone_ref(py),))?
            .getattr(intern!(py, "data"))?
            .extract::<PyReadonlyArray2<Complex64>>()?
            .as_array()
            .to_owned())
    } else {
        Err(QiskitError::new_err(
            "Can't compute matrix of non-unitary op",
        ))
    }
}

/// Return the matrix Operator resulting from a block of Instructions.
pub fn blocks_to_matrix(
    py: Python,
    dag: &DAGCircuit,
    op_list: &[NodeIndex],
    block_index_map: [Qubit; 2],
) -> PyResult<Array2<Complex64>> {
    let map_bits = |bit: &Qubit| -> u8 {
        if *bit == block_index_map[0] {
            0
        } else {
            1
        }
    };
    let mut qubit_0 = ONE_QUBIT_IDENTITY;
    let mut qubit_1 = ONE_QUBIT_IDENTITY;
    let mut one_qubit_components_modified = false;
    let mut output_matrix: Option<Array2<Complex64>> = None;
    for node in op_list {
        let inst = dag.dag()[*node].unwrap_operation();
        let op_matrix = get_matrix_from_inst(py, inst)?;
        match dag
            .get_qargs(inst.qubits)
            .iter()
            .map(map_bits)
            .collect::<Vec<_>>()
            .as_slice()
        {
            [0] => {
                matmul_1q(&mut qubit_0, op_matrix);
                one_qubit_components_modified = true;
            }
            [1] => {
                matmul_1q(&mut qubit_1, op_matrix);
                one_qubit_components_modified = true;
            }
            [0, 1] => {
                if one_qubit_components_modified {
                    let one_qubits_combined = kron(&aview2(&qubit_1), &aview2(&qubit_0));
                    output_matrix = Some(match output_matrix {
                        None => op_matrix.dot(&one_qubits_combined),
                        Some(current) => {
                            let temp = one_qubits_combined.dot(&current);
                            op_matrix.dot(&temp)
                        }
                    });
                    qubit_0 = ONE_QUBIT_IDENTITY;
                    qubit_1 = ONE_QUBIT_IDENTITY;
                    one_qubit_components_modified = false;
                } else {
                    output_matrix = Some(match output_matrix {
                        None => op_matrix,
                        Some(current) => op_matrix.dot(&current),
                    });
                }
            }
            [1, 0] => {
                let matrix = change_basis(op_matrix.view());
                if one_qubit_components_modified {
                    let one_qubits_combined = kron(&aview2(&qubit_1), &aview2(&qubit_0));
                    output_matrix = Some(match output_matrix {
                        None => matrix.dot(&one_qubits_combined),
                        Some(current) => matrix.dot(&one_qubits_combined.dot(&current)),
                    });
                    qubit_0 = ONE_QUBIT_IDENTITY;
                    qubit_1 = ONE_QUBIT_IDENTITY;
                    one_qubit_components_modified = false;
                } else {
                    output_matrix = Some(match output_matrix {
                        None => matrix,
                        Some(current) => matrix.dot(&current),
                    });
                }
            }
            _ => unreachable!(),
        }
    }
    Ok(match output_matrix {
        Some(matrix) => {
            if one_qubit_components_modified {
                let one_qubits_combined = kron(&aview2(&qubit_1), &aview2(&qubit_0));
                one_qubits_combined.dot(&matrix)
            } else {
                matrix
            }
        }
        None => kron(&aview2(&qubit_1), &aview2(&qubit_0)),
    })
}

/// Switches the order of qubits in a two qubit operation.
#[inline]
pub fn change_basis(matrix: ArrayView2<Complex64>) -> Array2<Complex64> {
    let mut trans_matrix: Array2<Complex64> = matrix.reversed_axes().to_owned();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix = trans_matrix.reversed_axes();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix
}

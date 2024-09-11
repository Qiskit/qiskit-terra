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
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use num_complex::Complex64;
use numpy::ndarray::linalg::kron;
use numpy::ndarray::{aview2, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use smallvec::SmallVec;

use qiskit_circuit::bit_data::BitData;
use qiskit_circuit::circuit_instruction::CircuitInstruction;
use qiskit_circuit::dag_node::DAGOpNode;
use qiskit_circuit::gate_matrix::ONE_QUBIT_IDENTITY;
use qiskit_circuit::imports::QI_OPERATOR;
use qiskit_circuit::operations::Operation;

use crate::QiskitError;

fn get_matrix_from_inst<'py>(
    py: Python<'py>,
    inst: &'py CircuitInstruction,
) -> PyResult<Array2<Complex64>> {
    if let Some(mat) = inst.operation.matrix(&inst.params) {
        Ok(mat)
    } else if inst.operation.try_standard_gate().is_some() {
        Err(QiskitError::new_err(
            "Parameterized gates can't be consolidated",
        ))
    } else {
        Ok(QI_OPERATOR
            .get_bound(py)
            .call1((inst.get_operation(py)?,))?
            .getattr(intern!(py, "data"))?
            .extract::<PyReadonlyArray2<Complex64>>()?
            .as_array()
            .to_owned())
    }
}

/// Return the matrix Operator resulting from a block of Instructions.
#[pyfunction]
#[pyo3(text_signature = "(op_list, /")]
pub fn blocks_to_matrix(
    py: Python,
    op_list: Vec<PyRef<DAGOpNode>>,
    block_index_map_dict: &Bound<PyDict>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    // Build a BitData in block_index_map_dict order. block_index_map_dict is a dict of bits to
    // indices mapping the order of the qargs in the block. There should only be 2 entries since
    // there are only 2 qargs here (e.g. `{Qubit(): 0, Qubit(): 1}`) so we need to ensure that
    // we added the qubits to bit data in the correct index order.
    let mut index_map: Vec<PyObject> = (0..block_index_map_dict.len()).map(|_| py.None()).collect();
    for bit_tuple in block_index_map_dict.items() {
        let (bit, index): (PyObject, usize) = bit_tuple.extract()?;
        index_map[index] = bit;
    }
    let mut bit_map: BitData<u32> = BitData::new(py, "qargs".to_string());
    for bit in index_map {
        bit_map.add(py, bit.bind(py), true)?;
    }
    let identity = aview2(&ONE_QUBIT_IDENTITY);
    let first_node = &op_list[0];
    let input_matrix = get_matrix_from_inst(py, &first_node.instruction)?;
    let mut matrix: Array2<Complex64> = match bit_map
        .map_bits(first_node.instruction.qubits.bind(py).iter())?
        .collect::<Vec<_>>()
        .as_slice()
    {
        [0] => kron(&identity, &input_matrix),
        [1] => kron(&input_matrix, &identity),
        [0, 1] => input_matrix,
        [1, 0] => change_basis(input_matrix.view()),
        [] => Array2::eye(4),
        _ => unreachable!(),
    };
    for node in op_list.into_iter().skip(1) {
        let op_matrix = get_matrix_from_inst(py, &node.instruction)?;
        let q_list = bit_map
            .map_bits(node.instruction.qubits.bind(py).iter())?
            .map(|x| x as u8)
            .collect::<SmallVec<[u8; 2]>>();

        let result = match q_list.as_slice() {
            [0] => Some(kron(&identity, &op_matrix)),
            [1] => Some(kron(&op_matrix, &identity)),
            [1, 0] => Some(change_basis(op_matrix.view())),
            [] => Some(Array2::eye(4)),
            _ => None,
        };
        matrix = match result {
            Some(result) => result.dot(&matrix),
            None => op_matrix.dot(&matrix),
        };
    }
    Ok(matrix.into_pyarray_bound(py).unbind())
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

pub fn convert_2q_block_matrix(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(blocks_to_matrix))?;
    Ok(())
}

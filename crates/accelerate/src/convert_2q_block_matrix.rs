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

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use num_complex::Complex64;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use smallvec::SmallVec;

use crate::common::{
    change_basis, kron_identity_x_matrix, kron_matrix_x_identity, matrix_multiply_4x4,
};

/// Return the matrix Operator resulting from a block of Instructions.
#[pyfunction]
#[pyo3(text_signature = "(op_list, /")]
pub fn blocks_to_matrix(
    py: Python,
    op_list: Vec<(PyReadonlyArray2<Complex64>, SmallVec<[u8; 2]>)>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let input_matrix = op_list[0].0.as_array();
    let mut matrix: Array2<Complex64> = match op_list[0].1.as_slice() {
        [0] => kron_identity_x_matrix(input_matrix),
        [1] => kron_matrix_x_identity(input_matrix),
        [0, 1] => input_matrix.to_owned(),
        [1, 0] => change_basis(input_matrix),
        [] => Array2::eye(4),
        _ => unreachable!(),
    };
    for (op_matrix, q_list) in op_list.into_iter().skip(1) {
        let op_matrix = op_matrix.as_array();

        let result = match q_list.as_slice() {
            [0] => Some(kron_identity_x_matrix(op_matrix.view())),
            [1] => Some(kron_matrix_x_identity(op_matrix.view())),
            [1, 0] => Some(change_basis(op_matrix)),
            [] => Some(Array2::eye(4)),
            _ => None,
        };
        matrix = match result {
            Some(result) => matrix_multiply_4x4(result.view(), matrix.view()),
            None => matrix_multiply_4x4(op_matrix.view(), matrix.view()),
        };
    }
    Ok(matrix.into_pyarray_bound(py).unbind())
}

#[pymodule]
pub fn convert_2q_block_matrix(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(blocks_to_matrix))?;
    Ok(())
}

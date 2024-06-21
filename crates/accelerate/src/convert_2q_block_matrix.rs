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

use faer::prelude::*;
use faer::Mat;
use faer_ext::{IntoFaerComplex, IntoNdarrayComplex};

use std::mem::swap;

use crate::common::{
    change_basis_faer, change_basis_ndarray, faer_kron_identity_x_matrix,
    faer_kron_matrix_x_identity, matmul_to_dst, ndarray_kron_identity_x_matrix,
    ndarray_kron_matrix_x_identity,
};

/// Return the matrix Operator resulting from a block of Instructions.
#[pyfunction]
#[pyo3(text_signature = "(op_list, /")]
pub fn blocks_to_matrix(
    py: Python,
    op_list: Vec<(PyReadonlyArray2<Complex64>, SmallVec<[u8; 2]>)>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let input_matrix = op_list[0].0.as_array().into_faer_complex();

    let mut matrix = match op_list[0].1.as_slice() {
        [0] => faer_kron_identity_x_matrix(input_matrix),
        [1] => faer_kron_matrix_x_identity(input_matrix),
        [0, 1] => input_matrix.to_owned(),
        [1, 0] => change_basis_faer(input_matrix),
        [] => Mat::<c64>::identity(4, 4),
        _ => unreachable!(),
    };

    let mut aux = Mat::<c64>::with_capacity(4, 4);
    // SAFETY: `aux` is a 4x4 matrix whose values are uninitialized and it's used only to store the
    // result of the `matmul` call inside the for loop
    unsafe { aux.set_dims(4, 4) };

    for (op_matrix, q_list) in op_list.into_iter().skip(1) {
        let op_matrix = op_matrix.as_array();

        let result = match q_list.as_slice() {
            [0] => Some(ndarray_kron_identity_x_matrix(op_matrix)),
            [1] => Some(ndarray_kron_matrix_x_identity(op_matrix)),
            [1, 0] => Some(change_basis_ndarray(op_matrix)),
            [] => Some(Array2::<Complex64>::eye(4)),
            _ => None,
        };

        matmul_to_dst(
            aux.as_mut(),
            result
                .as_ref()
                .map(|x| x.view())
                .unwrap_or(op_matrix)
                .into_faer_complex(),
            matrix.as_ref(),
        );

        // Swap values between `aux` and `matrix` to store the result of the `matmul` call
        // in the matrix `matrix` and prepare it for a possible new iteration of the for loop
        swap(&mut aux, &mut matrix);
    }

    Ok(matrix
        .as_ref()
        .into_ndarray_complex()
        .to_owned()
        .into_pyarray_bound(py)
        .unbind())
}

#[pymodule]
pub fn convert_2q_block_matrix(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(blocks_to_matrix))?;
    Ok(())
}

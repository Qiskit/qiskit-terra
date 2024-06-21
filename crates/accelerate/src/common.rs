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
use numpy::ndarray::{array, Array2, ArrayView2};

use faer::modules::core::mul::matmul;
use faer::perm::swap_rows;
use faer::prelude::*;
use faer::{mat, Mat, MatRef, Parallelism};

/*
--------------
     faer
--------------
*/

#[inline]
pub fn matmul_to_dst(dst: MatMut<c64>, lhs: MatRef<c64>, rhs: MatRef<c64>) {
    matmul(dst, lhs, rhs, None, c64::new(1., 0.), Parallelism::None);
}

#[inline]
pub fn faer_kron_matrix_x_identity(lhs: MatRef<c64>) -> Mat<c64> {
    let zero: c64 = c64::new(0., 0.);
    mat![
        [lhs[(0, 0)], zero, lhs[(0, 1)], zero],
        [zero, lhs[(0, 0)], zero, lhs[(0, 1)]],
        [lhs[(1, 0)], zero, lhs[(1, 1)], zero],
        [zero, lhs[(1, 0)], zero, lhs[(1, 1)]],
    ]
}

#[inline]
pub fn faer_kron_identity_x_matrix(rhs: MatRef<c64>) -> Mat<c64> {
    let zero: c64 = c64::new(0., 0.);
    mat![
        [rhs[(0, 0)], rhs[(0, 1)], zero, zero],
        [rhs[(1, 0)], rhs[(1, 1)], zero, zero],
        [zero, zero, rhs[(0, 0)], rhs[(0, 1)]],
        [zero, zero, rhs[(1, 0)], rhs[(1, 1)]],
    ]
}

/// Switches the order of qubits in a two qubit operation.
#[inline]
pub fn change_basis_faer(matrix: MatRef<c64>) -> Mat<c64> {
    let mut trans_matrix: Mat<c64> = matrix.transpose().to_owned();
    let (row1, row2) = trans_matrix.as_mut().two_rows_mut(1, 2);
    swap_rows(row1, row2);

    trans_matrix = trans_matrix.transpose().to_owned();
    let (row1, row2) = trans_matrix.as_mut().two_rows_mut(1, 2);
    swap_rows(row1, row2);

    trans_matrix
}

/*
-----------------
     ndarray
-----------------
*/

#[inline]
pub fn ndarray_kron_matrix_x_identity(lhs: ArrayView2<Complex64>) -> Array2<Complex64> {
    let zero = Complex64::new(0., 0.);
    array![
        [lhs[(0, 0)], zero, lhs[(0, 1)], zero],
        [zero, lhs[(0, 0)], zero, lhs[(0, 1)]],
        [lhs[(1, 0)], zero, lhs[(1, 1)], zero],
        [zero, lhs[(1, 0)], zero, lhs[(1, 1)]],
    ]
}

#[inline]
pub fn ndarray_kron_identity_x_matrix(rhs: ArrayView2<Complex64>) -> Array2<Complex64> {
    let zero = Complex64::new(0., 0.);
    array![
        [rhs[(0, 0)], rhs[(0, 1)], zero, zero],
        [rhs[(1, 0)], rhs[(1, 1)], zero, zero],
        [zero, zero, rhs[(0, 0)], rhs[(0, 1)]],
        [zero, zero, rhs[(1, 0)], rhs[(1, 1)]],
    ]
}

/// Switches the order of qubits in a two qubit operation.
#[inline]
pub fn change_basis_ndarray(matrix: ArrayView2<Complex64>) -> Array2<Complex64> {
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

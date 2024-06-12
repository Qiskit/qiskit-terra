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

#[inline]
pub fn kron_matrix_x_identity(lhs: ArrayView2<Complex64>) -> Array2<Complex64> {
    let zero = Complex64::new(0., 0.);
    array![
        [lhs[(0, 0)], zero, lhs[(0, 1)], zero],
        [zero, lhs[(0, 0)], zero, lhs[(0, 1)]],
        [lhs[(1, 0)], zero, lhs[(1, 1)], zero],
        [zero, lhs[(1, 0)], zero, lhs[(1, 1)]],
    ]
}

#[inline]
pub fn kron_identity_x_matrix(rhs: ArrayView2<Complex64>) -> Array2<Complex64> {
    let zero = Complex64::new(0., 0.);
    array![
        [rhs[(0, 0)], rhs[(0, 1)], zero, zero],
        [rhs[(1, 0)], rhs[(1, 1)], zero, zero],
        [zero, zero, rhs[(0, 0)], rhs[(0, 1)]],
        [zero, zero, rhs[(1, 0)], rhs[(1, 1)]],
    ]
}

#[inline]
pub fn matrix_multiply_2x2(
    a: ArrayView2<Complex64>,
    b: ArrayView2<Complex64>,
) -> Array2<Complex64> {
    let mut result = Array2::uninit((2, 2));

    result[(0, 0)].write(a[(0, 0)] * b[(0, 0)] + a[(0, 1)] * b[(1, 0)]);
    result[(0, 1)].write(a[(0, 0)] * b[(0, 1)] + a[(0, 1)] * b[(1, 1)]);
    result[(1, 0)].write(a[(1, 0)] * b[(0, 0)] + a[(1, 1)] * b[(1, 0)]);
    result[(1, 1)].write(a[(1, 0)] * b[(0, 1)] + a[(1, 1)] * b[(1, 1)]);

    unsafe { result.assume_init() }
}

#[inline]
pub fn matrix_multiply_4x4(
    a: ArrayView2<Complex64>,
    b: ArrayView2<Complex64>,
) -> Array2<Complex64> {
    let mut result = Array2::uninit((4, 4));

    for i in 0..4 {
        for j in 0..4 {
            let mut sum = Complex64::new(0., 0.);
            for k in 0..4 {
                sum += a[(i, k)] * b[(k, j)]
            }
            result[(i, j)].write(sum);
        }
    }

    unsafe { result.assume_init() }
}

#[inline]
pub fn determinant_4x4(a: ArrayView2<Complex64>) -> Complex64 {
    a[(0, 3)] * a[(1, 2)] * a[(2, 1)] * a[(3, 0)]
        - a[(0, 2)] * a[(1, 3)] * a[(2, 1)] * a[(3, 0)]
        - a[(0, 3)] * a[(1, 1)] * a[(2, 2)] * a[(3, 0)]
        + a[(0, 1)] * a[(1, 3)] * a[(2, 2)] * a[(3, 0)]
        + a[(0, 2)] * a[(1, 1)] * a[(2, 3)] * a[(3, 0)]
        - a[(0, 1)] * a[(1, 2)] * a[(2, 3)] * a[(3, 0)]
        - a[(0, 3)] * a[(1, 2)] * a[(2, 0)] * a[(3, 1)]
        + a[(0, 2)] * a[(1, 3)] * a[(2, 0)] * a[(3, 1)]
        + a[(0, 3)] * a[(1, 0)] * a[(2, 2)] * a[(3, 1)]
        - a[(0, 0)] * a[(1, 3)] * a[(2, 2)] * a[(3, 1)]
        - a[(0, 2)] * a[(1, 0)] * a[(2, 3)] * a[(3, 1)]
        + a[(0, 0)] * a[(1, 2)] * a[(2, 3)] * a[(3, 1)]
        + a[(0, 3)] * a[(1, 1)] * a[(2, 0)] * a[(3, 2)]
        - a[(0, 1)] * a[(1, 3)] * a[(2, 0)] * a[(3, 2)]
        - a[(0, 3)] * a[(1, 0)] * a[(2, 1)] * a[(3, 2)]
        + a[(0, 0)] * a[(1, 3)] * a[(2, 1)] * a[(3, 2)]
        + a[(0, 1)] * a[(1, 0)] * a[(2, 3)] * a[(3, 2)]
        - a[(0, 0)] * a[(1, 1)] * a[(2, 3)] * a[(3, 2)]
        - a[(0, 2)] * a[(1, 1)] * a[(2, 0)] * a[(3, 3)]
        + a[(0, 1)] * a[(1, 2)] * a[(2, 0)] * a[(3, 3)]
        + a[(0, 2)] * a[(1, 0)] * a[(2, 1)] * a[(3, 3)]
        - a[(0, 0)] * a[(1, 2)] * a[(2, 1)] * a[(3, 3)]
        - a[(0, 1)] * a[(1, 0)] * a[(2, 2)] * a[(3, 3)]
        + a[(0, 0)] * a[(1, 1)] * a[(2, 2)] * a[(3, 3)]
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

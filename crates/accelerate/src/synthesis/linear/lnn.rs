// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::synthesis::linear::utils::{_add_row_or_col, calc_inverse_matrix_inner};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};
use numpy::PyReadonlyArray2;

use pyo3::prelude::*;

/// Optimize the synthesis of an n-qubit circuit contains only CX gates for
/// linear nearest neighbor (LNN) connectivity.
/// The depth of the circuit is bounded by 5*n, while the gate count is approximately 2.5*n^2
///
/// References:
///     [1]: Kutin, S., Moulton, D. P., Smithline, L. (2007).
///          Computation at a Distance.
///          `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_.

type InstructionList = Vec<(usize, usize)>;
fn _row_sum(row_1: ArrayView1<bool>, row_2: ArrayView1<bool>) -> Option<Array1<bool>> {
    let n = row_1.len();
    if row_2.len() != n {
        return None;
    }
    let result: Array1<bool> = (0..n).map(|i| row_1[i] ^ row_2[i]).collect();
    Some(result)
}
/// Perform ROW operation on a matrix mat
fn _row_op(mat: ArrayViewMut2<bool>, ctrl: usize, trgt: usize) {
    _add_row_or_col(mat, &false, ctrl, trgt);
}

/// Perform COL operation on a matrix mat (in the inverse direction)
fn _col_op(mat: ArrayViewMut2<bool>, ctrl: usize, trgt: usize) {
    _add_row_or_col(mat, &true, trgt, ctrl);
}

/// Add a cx gate to the instructions and update the matrix mat
fn _row_op_update_instructions(
    cx_instructions: &mut InstructionList,
    mat: ArrayViewMut2<bool>,
    a: usize,
    b: usize,
) {
    cx_instructions.push((a, b));
    _row_op(mat, a, b);
}

///     Get the instructions for a lower triangular basis change of a matrix mat.
///     See the proof of Proposition 7.3 in [1].
fn _get_lower_triangular(
    n: usize,
    mat: ArrayView2<bool>,
    mat_inv: ArrayView2<bool>,
) -> (Array2<bool>, Array2<bool>) {
    let mut mat = mat.to_owned();
    let mut mat_t = mat.to_owned();
    let mut mat_inv_t = mat_inv.to_owned();

    let mut cx_instructions_rows: InstructionList = Vec::new();
    // Use the instructions in U, which contains only gates of the form cx(a,b) a>b
    // to transform the matrix to a permuted lower-triangular matrix.
    // The original Matrix is unchanged.
    for i in (0..n).rev() {
        let mut found_first = false;
        let mut first_j = 0;
        // Find the last "1" in row i, use COL operations to the left in order to
        // zero out all other "1"s in that row.
        for j in (0..n).rev() {
            if mat[[i, j]] {
                if !found_first {
                    found_first = true;
                    first_j = j;
                } else {
                    // cx_instructions_cols (L instructions) are not needed
                    _col_op(mat.view_mut(), j, first_j);
                }
            }
        }
        // Use row operations directed upwards to zero out all "1"s above the remaining "1" in row i
        for k in (0..i).rev() {
            if mat[[k, first_j]] {
                _row_op_update_instructions(&mut cx_instructions_rows, mat.view_mut(), i, k)
            }
        }
    }
    // Apply only U instructions to get the permuted L
    for (ctrl, trgt) in cx_instructions_rows {
        _row_op(mat_t.view_mut(), ctrl, trgt);
        _col_op(mat_inv_t.view_mut(), ctrl, trgt);
    }
    (mat_t, mat_inv_t)
}

/// For each row in mat_t, save the column index of the last "1"
fn _get_label_arr(n: usize, mat_t: ArrayView2<bool>) -> Vec<usize> {
    let mut label_arr: Vec<usize> = Vec::new();
    for i in 0..n {
        let mut j = 0;
        while !mat_t[[i, n - 1 - j]] {
            j += 1;
        }
        label_arr.push(j);
    }
    label_arr
}

/// Check if "row" is a linear combination of all rows in mat_inv_t not including the row labeled by k
fn _in_linear_combination(
    label_arr_t: &[usize],
    mat_inv_t: ArrayView2<bool>,
    row: ArrayView1<bool>,
    k: usize,
) -> bool {
    let n = row.len();
    let indx_k = label_arr_t[k];
    let mut w_needed: Array1<bool> = Array1::from_elem(n, false);
    // Find the linear combination of mat_t rows which produces "row"
    for row_l in 0..n {
        if row[row_l] {
            // mat_inv_t can be thought of as a set of instructions. Row l in mat_inv_t
            // indicates which rows from mat_t are necessary to produce the elementary vector e_l
            w_needed = _row_sum(w_needed.view(), mat_inv_t.row(row_l)).unwrap();
        }
    }
    // If the linear combination requires the row labeled by k
    !w_needed[indx_k]
}

/// Returns label_arr_t = label_arr^(-1)
fn _get_label_arr_t(n: usize, label_arr: &[usize]) -> Vec<usize> {
    let mut label_err_t: Vec<usize> = vec![0; n];
    for i in 0..n {
        label_err_t[label_arr[i]] = i
    }
    label_err_t
}

/// Transform an arbitrary boolean invertible matrix to a north-west triangular matrix
/// by Proposition 7.3 in [1]
fn _matrix_to_north_west(
    n: usize,
    mut mat: ArrayViewMut2<bool>,
    mat_inv: ArrayView2<bool>,
) -> InstructionList {
    // The rows of mat_t hold all w_j vectors (see [1]). mat_inv_t is the inverted matrix of mat_t
    let (mat_t, mat_inv_t) = _get_lower_triangular(n, mat.view(), mat_inv.view());
    // Get all pi(i) labels
    let mut label_arr = _get_label_arr(n, mat_t.view());

    // Save the original labels, exchange index <-> value
    let label_arr_t = _get_label_arr_t(n, &label_arr);
    let mut first_qubit = 0;
    let mut empty_layers = 0;
    let mut done = false;
    let mut cx_instructions_rows: InstructionList = Vec::new();
    while !done {
        // At each iteration the values of i switch between even and odd
        let mut at_least_one_needed = false;
        for i in (first_qubit..n - 1).step_by(2) {
            // "If j < k, we do nothing" (see [1])
            // "If j > k, we swap the two labels, and we also perform a box" (see [1])
            if label_arr[i] > label_arr[i + 1] {
                at_least_one_needed = true;
                // iterate on column indices, output rows as Vec<bool>
                let row_sum = _row_sum(mat.row(i), mat.row(i + 1)).unwrap();
                // "Let W be the span of all w_l for l!=k" (see [1])
                // " We can perform a box on <i> and <i + 1> that writes a vector in W to wire <i + 1>."
                // (see [1])
                if _in_linear_combination(
                    &label_arr_t,
                    mat_inv_t.view(),
                    mat.row(i + 1),
                    label_arr[i + 1],
                ) {
                    // do nothing
                } else if _in_linear_combination(
                    &label_arr_t,
                    mat_inv_t.view(),
                    row_sum.view(),
                    label_arr[i + 1],
                ) {
                    _row_op_update_instructions(
                        &mut cx_instructions_rows,
                        mat.view_mut(),
                        i,
                        i + 1,
                    );
                } else if _in_linear_combination(
                    &label_arr_t,
                    mat_inv_t.view(),
                    mat.row(i),
                    label_arr[i + 1],
                ) {
                    _row_op_update_instructions(
                        &mut cx_instructions_rows,
                        mat.view_mut(),
                        i + 1,
                        i,
                    );
                    _row_op_update_instructions(
                        &mut cx_instructions_rows,
                        mat.view_mut(),
                        i,
                        i + 1,
                    );
                }
                (label_arr[i], label_arr[i + 1]) = (label_arr[i + 1], label_arr[i]);
            }
        }
        if !at_least_one_needed {
            empty_layers += 1;
            if empty_layers > 1 {
                // if nothing happened twice in a row, then finished.
                done = true;
            }
        } else {
            empty_layers = 0;
        }
        first_qubit = 1 - first_qubit;
    }
    cx_instructions_rows
}

///    Transform a north-west triangular matrix to identity in depth 3*n by Proposition 7.4 of [1]
fn _north_west_to_identity(n: usize, mut mat: ArrayViewMut2<bool>) -> InstructionList {
    // At start the labels are in reversed order
    let mut label_arr: Vec<usize> = (0..n).rev().collect();
    let mut first_qubit = 0;
    let mut empty_layers = 0;
    let mut done = false;
    let mut cx_instructions_rows: InstructionList = Vec::new();
    while !done {
        let mut at_least_one_needed = false;
        for i in (first_qubit..n - 1).step_by(2) {
            // Exchange the labels if needed
            if label_arr[i] > label_arr[i + 1] {
                at_least_one_needed = true;
                // If row i has "1" in column i+1, swap and remove the "1" (in depth 2)
                // otherwise, only do a swap (in depth 3)
                if !mat[[i, label_arr[i + 1]]] {
                    // Adding this turns the operation to a SWAP
                    _row_op_update_instructions(
                        &mut cx_instructions_rows,
                        mat.view_mut(),
                        i + 1,
                        i,
                    );
                }
                _row_op_update_instructions(&mut cx_instructions_rows, mat.view_mut(), i, i + 1);
                _row_op_update_instructions(&mut cx_instructions_rows, mat.view_mut(), i + 1, i);

                (label_arr[i], label_arr[i + 1]) = (label_arr[i + 1], label_arr[i]);
            }
        }

        if !at_least_one_needed {
            empty_layers += 1;
            if empty_layers > 1 {
                // if nothing happened twice in a row, then finished.
                done = true;
            }
        } else {
            empty_layers = 0;
        }
        first_qubit = 1 - first_qubit;
    }
    cx_instructions_rows
}

///     Optimize CX circuit in depth bounded by 5n for LNN connectivity.
///     The algorithm [1] has two steps:
///     a) transform the original matrix to a north-west matrix (m2nw),
///     b) transform the north-west matrix to identity (nw2id).
///     
///     A square n-by-n matrix A is called north-west if A[i][j]=0 for all i+j>=n
///     For example, the following matrix is north-west:
///     [[0, 1, 0, 1]
///     [1, 1, 1, 0]
///     [0, 1, 0, 0]
///     [1, 0, 0, 0]]

///     According to [1] the synthesis is done on the inverse matrix
///     so the matrix mat is inverted at this step
#[pyfunction]
#[pyo3(signature = (mat))]
pub fn optimize_cx_circ_depth_5n_line(
    _py: Python,
    mat: PyReadonlyArray2<bool>,
) -> PyResult<(InstructionList, InstructionList)> {
    let arrayview = mat.as_array();

    // According to [1] the synthesis is done on the inverse matrix
    // so the matrix mat is inverted at this step

    let mat_inv: Array2<bool> = arrayview.to_owned();
    let mut mat_cpy = calc_inverse_matrix_inner(mat_inv.view(), false).unwrap();

    let n = mat_cpy.nrows();

    // Transform an arbitrary invertible matrix to a north-west triangular matrix
    // by Proposition 7.3 of [1]

    let cx_instructions_rows_m2nw = _matrix_to_north_west(n, mat_cpy.view_mut(), mat_inv.view());
    // Transform a north-west triangular matrix to identity in depth 3*n
    // by Proposition 7.4 of [1]

    let cx_instructions_rows_nw2id = _north_west_to_identity(n, mat_cpy.view_mut());

    let out = (cx_instructions_rows_m2nw, cx_instructions_rows_nw2id);
    Ok(out)
}

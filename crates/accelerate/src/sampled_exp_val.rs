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

use num_complex::Complex64;

use hashbrown::HashMap;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::pauli_exp_val::fast_sum;
use qiskit_circuit::util::c64;

const OPER_TABLE_SIZE: usize = (b'Z' as usize) + 1;
const fn generate_oper_table() -> [[f64; 2]; OPER_TABLE_SIZE] {
    let mut table = [[0.; 2]; OPER_TABLE_SIZE];
    table[b'Z' as usize] = [1., -1.];
    table[b'0' as usize] = [1., 0.];
    table[b'1' as usize] = [0., 1.];
    table
}

static OPERS: [[f64; 2]; OPER_TABLE_SIZE] = generate_oper_table();

fn bitstring_expval(dist: &HashMap<String, f64>, mut oper_str: String) -> f64 {
    let inds: Vec<usize> = oper_str
        .char_indices()
        .filter_map(|(index, oper)| if oper != 'I' { Some(index) } else { None })
        .collect();
    oper_str.retain(|c| !r#"I"#.contains(c));
    let denom: f64 = fast_sum(&dist.values().copied().collect::<Vec<f64>>());
    let exp_val: f64 = dist
        .iter()
        .map(|(bits, val)| {
            let temp_product: f64 = oper_str.bytes().enumerate().fold(1.0, |acc, (idx, oper)| {
                let diagonal: [f64; 2] = OPERS[oper as usize];
                let index_char: char = bits.as_bytes()[inds[idx]] as char;
                let index: usize = index_char.to_digit(10).unwrap() as usize;
                acc * diagonal[index]
            });
            val * temp_product
        })
        .sum();
    exp_val / denom
}

/// Compute the expectation value from a sampled distribution
#[pyfunction]
#[pyo3(text_signature = "(oper_strs, coeff, dist, /)")]
pub fn sampled_expval_float(
    oper_strs: Vec<String>,
    coeff: PyReadonlyArray1<f64>,
    dist: HashMap<String, f64>,
) -> PyResult<f64> {
    let coeff_arr = coeff.as_slice()?;
    let out = oper_strs
        .into_iter()
        .enumerate()
        .map(|(idx, string)| coeff_arr[idx] * bitstring_expval(&dist, string))
        .sum();
    Ok(out)
}

/// Compute the expectation value from a sampled distribution
#[pyfunction]
#[pyo3(text_signature = "(oper_strs, coeff, dist, /)")]
pub fn sampled_expval_complex(
    oper_strs: Vec<String>,
    coeff: PyReadonlyArray1<Complex64>,
    dist: HashMap<String, f64>,
) -> PyResult<Complex64> {
    let coeff_arr = coeff.as_slice()?;
    let out: Complex64 = oper_strs
        .into_iter()
        .enumerate()
        .map(|(idx, string)| coeff_arr[idx] * c64(bitstring_expval(&dist, string), 0.))
        .sum();
    Ok(out)
}

pub fn sampled_exp_val(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sampled_expval_float))?;
    m.add_wrapped(wrap_pyfunction!(sampled_expval_complex))?;
    Ok(())
}

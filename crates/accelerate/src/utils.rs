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

use pyo3::prelude::*;

use faer_ext::IntoFaerComplex;
use num_complex::Complex;
use numpy::{IntoPyArray, PyReadonlyArray2};

/// Return indices that sort partially ordered data.
/// If `data` contains two elements that are incomparable,
/// an error will be thrown.
pub fn arg_sort<T: PartialOrd>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
    indices
}

/// Return the eigenvalues of `unitary` as a one-dimensional `numpy.ndarray`
/// with `dtype(complex128)`.
#[pyfunction]
#[pyo3(text_signature = "(unitary, /")]
pub fn eigenvalues(py: Python, unitary: PyReadonlyArray2<Complex<f64>>) -> PyObject {
    unitary
        .as_array()
        .into_faer_complex()
        .complex_eigenvalues()
        .into_iter()
        .map(|x| Complex::<f64>::new(x.re, x.im))
        .collect::<Vec<_>>()
        .into_pyarray(py)
        .into_any()
        .unbind()
}

pub fn utils(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(eigenvalues))?;
    Ok(())
}

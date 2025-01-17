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

use num_complex::{Complex64, ComplexFloat};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

use faer_ext::{IntoFaerComplex, IntoNdarrayComplex};
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray2};

use crate::euler_one_qubit_decomposer::det_one_qubit;
use qiskit_circuit::util::{c64, C_ZERO, IM};

const EPS: f64 = 1e-10;

// These constants are the non-zero elements of an RZ gate's unitary with an
// angle of pi / 2
const RZ_PI2_11: Complex64 = c64(FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
const RZ_PI2_00: Complex64 = c64(FRAC_1_SQRT_2, FRAC_1_SQRT_2);

/// This method implements the decomposition given in equation (3) in
/// https://arxiv.org/pdf/quant-ph/0410066.pdf.
///
/// The decomposition is used recursively to decompose uniformly controlled gates.
///
/// a,b = single qubit unitaries
/// v,u,r = outcome of the decomposition given in the reference mentioned above
///
/// (see there for the details).
fn demultiplex_single_uc(
    a: ArrayView2<Complex64>,
    b: ArrayView2<Complex64>,
) -> [Array2<Complex64>; 3] {
    let x = a.dot(&b.mapv(|x| x.conj()).t());
    let det_x = det_one_qubit(x.view());
    let x11 = x[[0, 0]] / det_x.sqrt();
    let phi = det_x.arg();

    let r1 = (IM / 2. * (PI / 2. - phi / 2. - x11.arg())).exp();
    let r2 = (IM / 2. * (PI / 2. - phi / 2. + x11.arg() + PI)).exp();

    let r = array![[r1, C_ZERO], [C_ZERO, r2],];

    let decomp = r
        .dot(&x)
        .dot(&r)
        .view()
        .into_faer_complex()
        .complex_eigendecomposition();
    let mut u: Array2<Complex64> = decomp.u().into_ndarray_complex().to_owned();
    let s = decomp.s().column_vector();
    let mut diag: Array1<Complex64> =
        Array1::from_shape_fn(u.shape()[0], |i| s[i].to_num_complex());

    // If d is not equal to diag(i,-i), then we put it into this "standard" form
    // (see eq. (13) in https://arxiv.org/pdf/quant-ph/0410066.pdf) by interchanging
    // the eigenvalues and eigenvectors
    if (diag[0] + IM).abs() < EPS {
        diag = diag.slice(s![..;-1]).to_owned();
        u = u.slice(s![.., ..;-1]).to_owned();
    }
    diag.mapv_inplace(|x| x.sqrt());
    let d = Array2::from_diag(&diag);
    let v = d
        .dot(&u.mapv(|x| x.conj()).t())
        .dot(&r.mapv(|x| x.conj()).t())
        .dot(&b);
    [v, u, r]
}

#[pyfunction]
pub fn dec_ucg_help(
    py: Python,
    sq_gates: Vec<PyReadonlyArray2<Complex64>>,
    num_qubits: u32,
) -> (Vec<PyObject>, PyObject) {
    let mut single_qubit_gates: Vec<Array2<Complex64>> = sq_gates
        .into_iter()
        .map(|x| x.as_array().to_owned())
        .collect();
    let mut diag: Array1<Complex64> = Array1::ones(2_usize.pow(num_qubits));
    let num_controls = num_qubits - 1;
    for dec_step in 0..num_controls {
        let num_ucgs = 2_usize.pow(dec_step);
        // The decomposition works recursively and the followign loop goes over the different
        // UCGates that arise in the decomposition
        for ucg_index in 0..num_ucgs {
            let len_ucg = 2_usize.pow(num_controls - dec_step);
            for i in 0..len_ucg / 2 {
                let shift = ucg_index * len_ucg;
                let a = single_qubit_gates[shift + i].view();
                let b = single_qubit_gates[shift + len_ucg / 2 + i].view();
                // Apply the decomposition for UCGates given in equation (3) in
                // https://arxiv.org/pdf/quant-ph/0410066.pdf
                // to demultiplex one control of all the num_ucgs uniformly-controlled gates
                // with log2(len_ucg) uniform controls
                let [v, u, r] = demultiplex_single_uc(a, b);
                // replace the single-qubit gates with v,u (the already existing ones
                // are not needed any more)
                single_qubit_gates[shift + i] = v;
                single_qubit_gates[shift + len_ucg / 2 + i] = u;
                // Now we decompose the gates D as described in Figure 4 in
                // https://arxiv.org/pdf/quant-ph/0410066.pdf and merge some of the gates
                // into the UCGates and the diagonal at the end of the circuit

                // Remark: The Rz(pi/2) rotation acting on the target qubit and the Hadamard
                // gates arising in the decomposition of D are ignored for the moment (they will
                // be added together with the C-NOT gates at the end of the decomposition
                // (in the method dec_ucg()))
                let r_conj_t = r.mapv(|x| x.conj()).t().to_owned();
                if ucg_index < num_ucgs - 1 {
                    // Absorb the Rz(pi/2) rotation on the control into the UC-Rz gate and
                    // merge the UC-Rz rotation with the following UCGate,
                    // which hasn't been decomposed yet
                    let k = shift + len_ucg + i;

                    single_qubit_gates[k] = single_qubit_gates[k].dot(&r_conj_t);
                    single_qubit_gates[k].mapv_inplace(|x| x * RZ_PI2_00);
                    let k = k + len_ucg / 2;
                    single_qubit_gates[k] = single_qubit_gates[k].dot(&r);
                    single_qubit_gates[k].mapv_inplace(|x| x * RZ_PI2_11);
                } else {
                    // Absorb the Rz(pi/2) rotation on the control into the UC-Rz gate and merge
                    // the trailing UC-Rz rotation into a diagonal gate at the end of the circuit
                    for ucg_index_2 in 0..num_ucgs {
                        let shift_2 = ucg_index_2 * len_ucg;
                        let k = 2 * (i + shift_2);
                        diag[k] *= r_conj_t[[0, 0]] * RZ_PI2_00;
                        diag[k + 1] *= r_conj_t[[1, 1]] * RZ_PI2_00;
                        let k = len_ucg + k;
                        diag[k] *= r[[0, 0]] * RZ_PI2_11;
                        diag[k + 1] *= r[[1, 1]] * RZ_PI2_11;
                    }
                }
            }
        }
    }
    (
        single_qubit_gates
            .into_iter()
            .map(|x| x.into_pyarray(py).into_any().unbind())
            .collect(),
        diag.into_pyarray(py).into_any().unbind(),
    )
}

pub fn uc_gate(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(dec_ucg_help))?;
    Ok(())
}

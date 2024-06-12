// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// In numpy matrices real and imaginary components are adjacent:
//   np.array([1,2,3], dtype='complex').view('float64')
//   array([1., 0., 2., 0., 3., 0.])
// The matrix faer::Mat<c64> has this layout.
// faer::Mat<num_complex::Complex<f64>> instead stores a matrix
// of real components and one of imaginary components.
// In order to avoid copying we want to use `MatRef<c64>` or `MatMut<c64>`.

use approx::{abs_diff_eq, relative_eq};
use num_complex::{Complex, Complex64, ComplexFloat};
use num_traits::Zero;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use smallvec::{smallvec, SmallVec};
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::ops::Deref;

use faer::Side::Lower;
use faer::{prelude::*, ComplexField};
use faer_ext::{IntoFaer, IntoFaerComplex, IntoNdarray};
use ndarray::prelude::*;
use ndarray::Zip;
use numpy::PyReadonlyArray2;
use numpy::{IntoPyArray, ToPyArray};
use pyo3::pybacked::PyBackedStr;

use crate::common::{
    change_basis, determinant_4x4, kron_identity_x_matrix, kron_matrix_x_identity,
    matrix_multiply_2x2, matrix_multiply_4x4,
};
use crate::euler_one_qubit_decomposer::{
    angles_from_unitary, det_one_qubit, unitary_to_gate_sequence_inner, EulerBasis,
    OneQubitGateSequence, ANGLE_ZERO_EPSILON,
};
use crate::utils;
use crate::QiskitError;

use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_pcg::Pcg64Mcg;

use qiskit_circuit::SliceOrInt;

const PI2: f64 = PI / 2.0;
const PI4: f64 = PI / 4.0;
const PI32: f64 = 3.0 * PI2;
const TWO_PI: f64 = 2.0 * PI;

const C1: Complex64 = Complex64 { re: 1.0, im: 0.0 };

pub static ONE_QUBIT_IDENTITY: [[Complex64; 2]; 2] = [
    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
    [Complex64::new(0., 0.), Complex64::new(1., 0.)],
];

static B_NON_NORMALIZED: [[Complex64; 4]; 4] = [
    [
        Complex64::new(1.0, 0.),
        Complex64::new(0., 1.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 1.),
        Complex64::new(1.0, 0.0),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 1.),
        Complex64::new(-1., 0.),
    ],
    [
        Complex64::new(1., 0.),
        Complex64::new(0., -1.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
];

static B_NON_NORMALIZED_DAGGER: [[Complex64; 4]; 4] = [
    [
        Complex64::new(0.5, 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0.5, 0.0),
    ],
    [
        Complex64::new(0., -0.5),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.5),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., -0.5),
        Complex64::new(0., -0.5),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0.5, 0.),
        Complex64::new(-0.5, 0.),
        Complex64::new(0., 0.),
    ],
];

enum MagicBasisTransform {
    Into,
    OutOf,
}

fn magic_basis_transform(
    unitary: ArrayView2<Complex64>,
    direction: MagicBasisTransform,
) -> Array2<Complex64> {
    let _b_nonnormalized = aview2(&B_NON_NORMALIZED);
    let _b_nonnormalized_dagger = aview2(&B_NON_NORMALIZED_DAGGER);
    match direction {
        MagicBasisTransform::OutOf => matrix_multiply_4x4(
            matrix_multiply_4x4(_b_nonnormalized_dagger, unitary).view(),
            _b_nonnormalized,
        ),
        MagicBasisTransform::Into => matrix_multiply_4x4(
            matrix_multiply_4x4(_b_nonnormalized, unitary).view(),
            _b_nonnormalized_dagger,
        ),
    }
}

// faer::c64 and num_complex::Complex<f64> are both structs
// holding two f64's. But several functions are not defined for
// c64. So we implement them here. These things should be contribute
// upstream.

pub trait PowF {
    fn powf(self, pow: f64) -> c64;
}

impl PowF for c64 {
    fn powf(self, pow: f64) -> c64 {
        c64::from(self.to_num_complex().powf(pow))
    }
}

pub trait Arg {
    fn arg(self) -> f64;
}

impl Arg for c64 {
    fn arg(self) -> f64 {
        self.to_num_complex().arg()
    }
}

#[inline(always)]
fn transpose_conjugate(mat: ArrayView2<Complex64>) -> Array2<Complex64> {
    mat.t().mapv(|x| x.conj())
}

pub trait TraceToFidelity {
    /// Average gate fidelity is :math:`Fbar = (d + |Tr (Utarget \\cdot U^dag)|^2) / d(d+1)`
    /// M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)
    fn trace_to_fid(self) -> f64;
}

impl TraceToFidelity for Complex64 {
    fn trace_to_fid(self) -> f64 {
        (4.0 + self.abs().powi(2)) / 20.0
    }
}

impl TraceToFidelity for c64 {
    fn trace_to_fid(self) -> f64 {
        (4.0 + self.faer_abs2()) / 20.0
    }
}

fn decompose_two_qubit_product_gate(
    special_unitary: ArrayView2<Complex64>,
) -> PyResult<(Array2<Complex64>, Array2<Complex64>, f64)> {
    let mut r: Array2<Complex64> = special_unitary.slice(s![..2, ..2]).to_owned();
    let mut det_r = det_one_qubit(r.view());
    if det_r.abs() < 0.1 {
        r = special_unitary.slice(s![2.., ..2]).to_owned();
        det_r = det_one_qubit(r.view());
    }
    if det_r.abs() < 0.1 {
        return Err(QiskitError::new_err(
            "decompose_two_qubit_product_gate: unable to decompose: detR < 0.1",
        ));
    }
    r.mapv_inplace(|x| x / det_r.sqrt());
    let r_t_conj: Array2<Complex64> = transpose_conjugate(r.view());
    let mut temp = kron_identity_x_matrix(r_t_conj.view());
    temp = matrix_multiply_4x4(special_unitary, temp.view());
    let mut l = temp.slice(s![..;2, ..;2]).to_owned();
    let det_l = det_one_qubit(l.view());
    if det_l.abs() < 0.9 {
        return Err(QiskitError::new_err(
            "decompose_two_qubit_product_gate: unable to decompose: detL < 0.9",
        ));
    }
    l.mapv_inplace(|x| x / det_l.sqrt());
    let phase = det_l.arg() / 2.;
    Ok((l, r, phase))
}

fn __weyl_coordinates(unitary: ArrayView2<Complex64>) -> [f64; 3] {
    let uscaled = (C1 / determinant_4x4(unitary).powf(0.25)) * &unitary;
    let uup = magic_basis_transform(uscaled.view(), MagicBasisTransform::OutOf);
    let mut darg: Vec<_> = matrix_multiply_4x4(uup.t(), uup.view())
        .view()
        .into_faer_complex()
        .complex_eigenvalues()
        .into_iter()
        .map(|x: c64| -x.arg() / 2.0)
        .collect();
    darg[3] = -darg[0] - darg[1] - darg[2];
    let mut cs: Vec<_> = (0..3)
        .map(|i| ((darg[i] + darg[3]) / 2.0).rem_euclid(2.0 * PI))
        .collect();
    let cstemp: Vec<f64> = cs
        .iter()
        .map(|x| x.rem_euclid(PI2))
        .map(|x| x.min(PI2 - x))
        .collect();
    let mut order = utils::arg_sort(&cstemp);
    (order[0], order[1], order[2]) = (order[1], order[2], order[0]);
    (cs[0], cs[1], cs[2]) = (cs[order[0]], cs[order[1]], cs[order[2]]);

    // Flip into Weyl chamber
    if cs[0] > PI2 {
        cs[0] -= PI32;
    }
    if cs[1] > PI2 {
        cs[1] -= PI32;
    }
    let mut conjs = 0;
    if cs[0] > PI4 {
        cs[0] = PI2 - cs[0];
        conjs += 1;
    }
    if cs[1] > PI4 {
        cs[1] = PI2 - cs[1];
        conjs += 1;
    }
    if cs[2] > PI2 {
        cs[2] -= PI32;
    }
    if conjs == 1 {
        cs[2] = PI2 - cs[2];
    }
    if cs[2] > PI4 {
        cs[2] -= PI2;
    }
    [cs[1], cs[0], cs[2]]
}

#[pyfunction]
#[pyo3(text_signature = "(basis_b, basis_fidelity, unitary, /")]
pub fn _num_basis_gates(
    basis_b: f64,
    basis_fidelity: f64,
    unitary: PyReadonlyArray2<Complex<f64>>,
) -> usize {
    let u = unitary.as_array();
    __num_basis_gates(basis_b, basis_fidelity, u)
}

fn __num_basis_gates(basis_b: f64, basis_fidelity: f64, unitary: ArrayView2<Complex64>) -> usize {
    let [a, b, c] = __weyl_coordinates(unitary);
    let traces = [
        c64::new(
            4.0 * (a.cos() * b.cos() * c.cos()),
            4.0 * (a.sin() * b.sin() * c.sin()),
        ),
        c64::new(
            4.0 * (PI4 - a).cos() * (basis_b - b).cos() * c.cos(),
            4.0 * (PI4 - a).sin() * (basis_b - b).sin() * c.sin(),
        ),
        c64::new(4.0 * c.cos(), 0.0),
        c64::new(4.0, 0.0),
    ];
    // The originial Python had `np.argmax`, which returns the lowest index in case two or more
    // values have a common maximum value.
    // `max_by` and `min_by` return the highest and lowest indices respectively, in case of ties.
    // So to reproduce `np.argmax`, we use `min_by` and switch the order of the
    // arguments in the comparison.
    traces
        .into_iter()
        .enumerate()
        .map(|(idx, trace)| (idx, trace.trace_to_fid() * basis_fidelity.powi(idx as i32)))
        .min_by(|(_idx1, fid1), (_idx2, fid2)| fid2.partial_cmp(fid1).unwrap())
        .unwrap()
        .0
}

/// A good approximation to the best value x to get the minimum
/// trace distance for :math:`U_d(x, x, x)` from :math:`U_d(a, b, c)`.
fn closest_partial_swap(a: f64, b: f64, c: f64) -> f64 {
    let m = (a + b + c) / 3.;
    let [am, bm, cm] = [a - m, b - m, c - m];
    let [ab, bc, ca] = [a - b, b - c, c - a];
    m + am * bm * cm * (6. + ab * ab + bc * bc + ca * ca) / 18.
}

fn rx_matrix(theta: f64) -> Array2<Complex64> {
    let half_theta = theta / 2.;
    let cos = Complex64::new(half_theta.cos(), 0.);
    let isin = Complex64::new(0., -half_theta.sin());
    array![[cos, isin], [isin, cos]]
}

fn ry_matrix(theta: f64) -> Array2<Complex64> {
    let half_theta = theta / 2.;
    let cos = Complex64::new(half_theta.cos(), 0.);
    let sin = Complex64::new(half_theta.sin(), 0.);
    array![[cos, -sin], [sin, cos]]
}

fn rz_matrix(theta: f64) -> Array2<Complex64> {
    let ilam2 = Complex64::new(0., 0.5 * theta);
    array![
        [(-ilam2).exp(), Complex64::new(0., 0.)],
        [Complex64::new(0., 0.), ilam2.exp()]
    ]
}

static HGATE: [[Complex64; 2]; 2] = [
    [
        Complex64::new(FRAC_1_SQRT_2, 0.),
        Complex64::new(FRAC_1_SQRT_2, 0.),
    ],
    [
        Complex64::new(FRAC_1_SQRT_2, 0.),
        Complex64::new(-FRAC_1_SQRT_2, 0.),
    ],
];

static CXGATE: [[Complex64; 4]; 4] = [
    [
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
];

static SXGATE: [[Complex64; 2]; 2] = [
    [Complex64::new(0.5, 0.5), Complex64::new(0.5, -0.5)],
    [Complex64::new(0.5, -0.5), Complex64::new(0.5, 0.5)],
];

static XGATE: [[Complex64; 2]; 2] = [
    [Complex64::new(0., 0.), Complex64::new(1., 0.)],
    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
];

fn compute_unitary(sequence: &TwoQubitSequenceVec, global_phase: f64) -> Array2<Complex64> {
    let phase = Complex64::new(0., global_phase).exp();
    let mut matrix = Array2::from_diag(&arr1(&[phase, phase, phase, phase]));
    sequence
        .iter()
        .map(|inst| {
            // This only gets called by get_sx_vz_3cx_efficient_euler()
            // which only uses sx, x, rz, and cx gates for the circuit
            // sequence. If we get a different gate this is getting called
            // by something else and is invalid.
            let gate_matrix = match inst.0.as_ref() {
                "sx" => aview2(&SXGATE).to_owned(),
                "rz" => rz_matrix(inst.1[0]),
                "cx" => aview2(&CXGATE).to_owned(),
                "x" => aview2(&XGATE).to_owned(),
                _ => unreachable!("Undefined gate"),
            };
            (gate_matrix, &inst.2)
        })
        .for_each(|(op_matrix, q_list)| {
            let result = match q_list.as_slice() {
                [0] => Some(kron_identity_x_matrix(op_matrix.view())),
                [1] => Some(kron_matrix_x_identity(op_matrix.view())),
                [1, 0] => Some(change_basis(op_matrix.view())),
                [] => Some(Array2::eye(4)),
                _ => None,
            };
            matrix = match result {
                Some(result) => matrix_multiply_4x4(result.view(), matrix.view()),
                None => matrix_multiply_4x4(op_matrix.view(), matrix.view()),
            }
        });
    matrix
}

const DEFAULT_FIDELITY: f64 = 1.0 - 1.0e-9;
const C1_IM: Complex64 = Complex64::new(0.0, 1.0);

#[derive(Clone, Debug, Copy)]
#[pyclass(module = "qiskit._accelerate.two_qubit_decompose")]
enum Specialization {
    General,
    IdEquiv,
    SWAPEquiv,
    PartialSWAPEquiv,
    PartialSWAPFlipEquiv,
    ControlledEquiv,
    MirrorControlledEquiv,
    // These next 3 gates use the definition of fSim from eq (1) in:
    // https://arxiv.org/pdf/2001.08343.pdf
    #[allow(non_camel_case_types)]
    fSimaabEquiv,
    #[allow(non_camel_case_types)]
    fSimabbEquiv,
    #[allow(non_camel_case_types)]
    fSimabmbEquiv,
}

#[pymethods]
impl Specialization {
    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Ideally we'd use the string-only form of `__reduce__` for simplicity, but PyO3 enums
        // don't produce Python singletons, and pickle doesn't like that.
        let val: u8 = match self {
            Self::General => 0,
            Self::IdEquiv => 1,
            Self::SWAPEquiv => 2,
            Self::PartialSWAPEquiv => 3,
            Self::PartialSWAPFlipEquiv => 4,
            Self::ControlledEquiv => 5,
            Self::MirrorControlledEquiv => 6,
            Self::fSimaabEquiv => 7,
            Self::fSimabbEquiv => 8,
            Self::fSimabmbEquiv => 9,
        };
        Ok((py.get_type_bound::<Self>().getattr("_from_u8")?, (val,)).into_py(py))
    }

    #[staticmethod]
    fn _from_u8(val: u8) -> PyResult<Self> {
        match val {
            0 => Ok(Self::General),
            1 => Ok(Self::IdEquiv),
            2 => Ok(Self::SWAPEquiv),
            3 => Ok(Self::PartialSWAPEquiv),
            4 => Ok(Self::PartialSWAPFlipEquiv),
            5 => Ok(Self::ControlledEquiv),
            6 => Ok(Self::MirrorControlledEquiv),
            7 => Ok(Self::fSimaabEquiv),
            8 => Ok(Self::fSimabbEquiv),
            9 => Ok(Self::fSimabmbEquiv),
            x => Err(PyValueError::new_err(format!(
                "unknown specialization discriminant '{x}'"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
#[allow(non_snake_case)]
#[pyclass(module = "qiskit._accelerate.two_qubit_decompose", subclass)]
pub struct TwoQubitWeylDecomposition {
    #[pyo3(get)]
    a: f64,
    #[pyo3(get)]
    b: f64,
    #[pyo3(get)]
    c: f64,
    #[pyo3(get)]
    global_phase: f64,
    K1l: Array2<Complex64>,
    K2l: Array2<Complex64>,
    K1r: Array2<Complex64>,
    K2r: Array2<Complex64>,
    #[pyo3(get)]
    specialization: Specialization,
    default_euler_basis: EulerBasis,
    #[pyo3(get)]
    requested_fidelity: Option<f64>,
    #[pyo3(get)]
    calculated_fidelity: f64,
    unitary_matrix: Array2<Complex64>,
}

impl TwoQubitWeylDecomposition {
    fn weyl_gate(
        &self,
        simplify: bool,
        sequence: &mut TwoQubitSequenceVec,
        atol: f64,
        global_phase: &mut f64,
    ) {
        match self.specialization {
            Specialization::MirrorControlledEquiv => {
                sequence.push(("swap".to_string(), SmallVec::new(), smallvec![0, 1]));
                sequence.push((
                    "rzz".to_string(),
                    smallvec![(PI4 - self.c) * 2.],
                    smallvec![0, 1],
                ));
                *global_phase += PI4
            }
            Specialization::SWAPEquiv => {
                sequence.push(("swap".to_string(), SmallVec::new(), smallvec![0, 1]));
                *global_phase -= 3. * PI / 4.
            }
            _ => {
                if !simplify || self.a.abs() > atol {
                    sequence.push(("rxx".to_string(), smallvec![-self.a * 2.], smallvec![0, 1]));
                }
                if !simplify || self.b.abs() > atol {
                    sequence.push(("ryy".to_string(), smallvec![-self.b * 2.], smallvec![0, 1]));
                }
                if !simplify || self.c.abs() > atol {
                    sequence.push(("rzz".to_string(), smallvec![-self.c * 2.], smallvec![0, 1]));
                }
            }
        }
    }
}

static IPZ: [[Complex64; 2]; 2] = [
    [C1_IM, Complex64::new(0., 0.)],
    [Complex64::new(0., 0.), Complex64::new(0., -1.)],
];
static IPY: [[Complex64; 2]; 2] = [
    [Complex64::new(0., 0.), Complex64::new(1., 0.)],
    [Complex64::new(-1., 0.), Complex64::new(0., 0.)],
];
static IPX: [[Complex64; 2]; 2] = [
    [Complex64::new(0., 0.), C1_IM],
    [C1_IM, Complex64::new(0., 0.)],
];

#[pymethods]
impl TwoQubitWeylDecomposition {
    #[staticmethod]
    fn _from_state(
        angles: [f64; 4],
        matrices: [PyReadonlyArray2<Complex64>; 5],
        specialization: Specialization,
        default_euler_basis: EulerBasis,
        calculated_fidelity: f64,
        requested_fidelity: Option<f64>,
    ) -> Self {
        let [a, b, c, global_phase] = angles;
        Self {
            a,
            b,
            c,
            global_phase,
            K1l: matrices[0].as_array().to_owned(),
            K1r: matrices[1].as_array().to_owned(),
            K2l: matrices[2].as_array().to_owned(),
            K2r: matrices[3].as_array().to_owned(),
            specialization,
            default_euler_basis,
            calculated_fidelity,
            requested_fidelity,
            unitary_matrix: matrices[4].as_array().to_owned(),
        }
    }

    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok((
            py.get_type_bound::<Self>().getattr("_from_state")?,
            (
                [self.a, self.b, self.c, self.global_phase],
                [
                    self.K1l.to_pyarray_bound(py),
                    self.K1r.to_pyarray_bound(py),
                    self.K2l.to_pyarray_bound(py),
                    self.K2r.to_pyarray_bound(py),
                    self.unitary_matrix.to_pyarray_bound(py),
                ],
                self.specialization,
                self.default_euler_basis,
                self.calculated_fidelity,
                self.requested_fidelity,
            ),
        )
            .into_py(py))
    }

    #[new]
    #[pyo3(signature=(unitary_matrix, fidelity=DEFAULT_FIDELITY, _specialization=None))]
    fn new(
        unitary_matrix: PyReadonlyArray2<Complex64>,
        fidelity: Option<f64>,
        _specialization: Option<Specialization>,
    ) -> PyResult<Self> {
        let ipz: ArrayView2<Complex64> = aview2(&IPZ);
        let ipy: ArrayView2<Complex64> = aview2(&IPY);
        let ipx: ArrayView2<Complex64> = aview2(&IPX);

        let mut u = unitary_matrix.as_array().to_owned();
        let unitary_matrix = unitary_matrix.as_array().to_owned();
        let det_u = determinant_4x4(u.view());
        let det_pow = det_u.powf(-0.25);
        u.mapv_inplace(|x| x * det_pow);
        let mut global_phase = det_u.arg() / 4.;
        let u_p = magic_basis_transform(u.view(), MagicBasisTransform::OutOf);
        let m2 = matrix_multiply_4x4(u_p.t(), u_p.view());
        let default_euler_basis = EulerBasis::ZYZ;

        // M2 is a symmetric complex matrix. We need to decompose it as M2 = P D P^T where
        // P ∈ SO(4), D is diagonal with unit-magnitude elements.
        //
        // We can't use raw `eig` directly because it isn't guaranteed to give us real or othogonal
        // eigenvectors. Instead, since `M2` is complex-symmetric,
        //   M2 = A + iB
        // for real-symmetric `A` and `B`, and as
        //   M2^+ @ M2 = A^2 + B^2 + i [A, B] = 1
        // we must have `A` and `B` commute, and consequently they are simultaneously diagonalizable.
        // Mixing them together _should_ account for any degeneracy problems, but it's not
        // guaranteed, so we repeat it a little bit.  The fixed seed is to make failures
        // deterministic; the value is not important.
        let mut state = Pcg64Mcg::seed_from_u64(2023);
        let mut found = false;
        let mut d: Array1<Complex64> = Array1::zeros(0);
        let mut p: Array2<Complex64> = Array2::zeros((0, 0));
        for i in 0..100 {
            let rand_a: f64;
            let rand_b: f64;
            // For debugging the algorithm use the same RNG values from the
            // previous Python implementation for the first random trial.
            // In most cases this loop only executes a single iteration and
            // using the same rng values rules out possible RNG differences
            // as the root cause of a test failure
            if i == 0 {
                rand_a = 1.2602066112249388;
                rand_b = 0.22317849046722027;
            } else {
                rand_a = state.sample(StandardNormal);
                rand_b = state.sample(StandardNormal);
            }
            let m2_real = m2.mapv(|val| rand_a * val.re + rand_b * val.im);
            let p_inner = m2_real
                .view()
                .into_faer()
                .selfadjoint_eigendecomposition(Lower)
                .u()
                .into_ndarray()
                .mapv(Complex64::from);
            let d_inner = matrix_multiply_4x4(
                matrix_multiply_4x4(p_inner.t(), m2.view()).view(),
                p_inner.view(),
            )
            .diag()
            .to_owned();
            let mut diag_d: Array2<Complex64> = Array2::zeros((4, 4));
            diag_d
                .diag_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(index, x)| *x = d_inner[index]);

            let compare = matrix_multiply_4x4(
                matrix_multiply_4x4(p_inner.view(), diag_d.view()).view(),
                p_inner.t(),
            );
            found = abs_diff_eq!(compare.view(), m2, epsilon = 1.0e-13);
            if found {
                p = p_inner;
                d = d_inner;
                break;
            }
        }
        if !found {
            return Err(QiskitError::new_err(format!(
                "TwoQubitWeylDecomposition: failed to diagonalize M2. Please report this at https://github.com/Qiskit/qiskit-terra/issues/4159. Input: {:?}", unitary_matrix
            )));
        }
        let mut d = -d.map(|x| x.arg() / 2.);
        d[3] = -d[0] - d[1] - d[2];
        let mut cs: SmallVec<[f64; 3]> = (0..3)
            .map(|i| ((d[i] + d[3]) / 2.0).rem_euclid(TWO_PI))
            .collect();
        let cstemp: SmallVec<[f64; 3]> = cs
            .iter()
            .map(|x| x.rem_euclid(PI2))
            .map(|x| x.min(PI2 - x))
            .collect();
        let mut order = utils::arg_sort(&cstemp);
        (order[0], order[1], order[2]) = (order[1], order[2], order[0]);
        (cs[0], cs[1], cs[2]) = (cs[order[0]], cs[order[1]], cs[order[2]]);
        (d[0], d[1], d[2]) = (d[order[0]], d[order[1]], d[order[2]]);
        let mut p_orig = p.clone();
        for (i, item) in order.iter().enumerate().take(3) {
            let slice_a = p.slice_mut(s![.., i]);
            let slice_b = p_orig.slice_mut(s![.., *item]);
            Zip::from(slice_a).and(slice_b).for_each(::std::mem::swap);
        }
        if determinant_4x4(p.view()).re < 0. {
            p.slice_mut(s![.., -1]).mapv_inplace(|x| -x);
        }
        let mut temp: Array2<Complex64> = Array2::zeros((4, 4));
        temp.diag_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(index, x)| *x = (C1_IM * d[index]).exp());
        let k1 = magic_basis_transform(
            matrix_multiply_4x4(
                matrix_multiply_4x4(u_p.view(), p.view()).view(),
                temp.view(),
            )
            .view(),
            MagicBasisTransform::Into,
        );
        let k2 = magic_basis_transform(p.t(), MagicBasisTransform::Into);

        #[allow(non_snake_case)]
        let (mut K1l, mut K1r, phase_l) = decompose_two_qubit_product_gate(k1.view())?;
        #[allow(non_snake_case)]
        let (K2l, mut K2r, phase_r) = decompose_two_qubit_product_gate(k2.view())?;
        global_phase += phase_l + phase_r;

        // Flip into Weyl chamber
        if cs[0] > PI2 {
            cs[0] -= PI32;
            K1l = matrix_multiply_2x2(K1l.view(), ipy);
            K1r = matrix_multiply_2x2(K1r.view(), ipy);
            global_phase += PI2;
        }
        if cs[1] > PI2 {
            cs[1] -= PI32;
            K1l = matrix_multiply_2x2(K1l.view(), ipx);
            K1r = matrix_multiply_2x2(K1r.view(), ipx);
            global_phase += PI2;
        }
        let mut conjs = 0;
        if cs[0] > PI4 {
            cs[0] = PI2 - cs[0];
            K1l = matrix_multiply_2x2(K1l.view(), ipy);
            K2r = matrix_multiply_2x2(ipy, K2r.view());
            conjs += 1;
            global_phase -= PI2;
        }
        if cs[1] > PI4 {
            cs[1] = PI2 - cs[1];
            K1l = matrix_multiply_2x2(K1l.view(), ipx);
            K2r = matrix_multiply_2x2(ipx, K2r.view());
            conjs += 1;
            global_phase += PI2;
            if conjs == 1 {
                global_phase -= PI;
            }
        }
        if cs[2] > PI2 {
            cs[2] -= PI32;
            K1l = matrix_multiply_2x2(K1l.view(), ipz);
            K1r = matrix_multiply_2x2(K1r.view(), ipz);
            global_phase += PI2;
            if conjs == 1 {
                global_phase -= PI;
            }
        }
        if conjs == 1 {
            cs[2] = PI2 - cs[2];
            K1l = matrix_multiply_2x2(K1l.view(), ipz);
            K2r = matrix_multiply_2x2(ipz, K2r.view());
            global_phase += PI2;
        }
        if cs[2] > PI4 {
            cs[2] -= PI2;
            K1l = matrix_multiply_2x2(K1l.view(), ipz);
            K1r = matrix_multiply_2x2(K1r.view(), ipz);
            global_phase -= PI2;
        }
        let [a, b, c] = [cs[1], cs[0], cs[2]];
        let is_close = |ap: f64, bp: f64, cp: f64| -> bool {
            let [da, db, dc] = [a - ap, b - bp, c - cp];
            let tr = 4.
                * Complex64::new(
                    da.cos() * db.cos() * dc.cos(),
                    da.sin() * db.sin() * dc.sin(),
                );
            match fidelity {
                Some(fid) => tr.trace_to_fid() >= fid,
                // Set to false here to default to general specialization in the absence of a
                // fidelity and provided specialization.
                None => false,
            }
        };

        let closest_abc = closest_partial_swap(a, b, c);
        let closest_ab_minus_c = closest_partial_swap(a, b, -c);
        let mut flipped_from_original = false;
        let specialization = match _specialization {
            Some(specialization) => specialization,
            None => {
                if is_close(0., 0., 0.) {
                    Specialization::IdEquiv
                } else if is_close(PI4, PI4, PI4) || is_close(PI4, PI4, -PI4) {
                    Specialization::SWAPEquiv
                } else if is_close(closest_abc, closest_abc, closest_abc) {
                    Specialization::PartialSWAPEquiv
                } else if is_close(closest_ab_minus_c, closest_ab_minus_c, -closest_ab_minus_c) {
                    Specialization::PartialSWAPFlipEquiv
                } else if is_close(a, 0., 0.) {
                    Specialization::ControlledEquiv
                } else if is_close(PI4, PI4, c) {
                    Specialization::MirrorControlledEquiv
                } else if is_close((a + b) / 2., (a + b) / 2., c) {
                    Specialization::fSimaabEquiv
                } else if is_close(a, (b + c) / 2., (b + c) / 2.) {
                    Specialization::fSimabbEquiv
                } else if is_close(a, (b - c) / 2., (c - b) / 2.) {
                    Specialization::fSimabmbEquiv
                } else {
                    Specialization::General
                }
            }
        };
        let general = TwoQubitWeylDecomposition {
            a,
            b,
            c,
            global_phase,
            K1l,
            K1r,
            K2l,
            K2r,
            specialization: Specialization::General,
            default_euler_basis,
            requested_fidelity: fidelity,
            calculated_fidelity: -1.0,
            unitary_matrix,
        };
        let mut specialized: TwoQubitWeylDecomposition = match specialization {
            // :math:`U \sim U_d(0,0,0) \sim Id`
            //
            // This gate binds 0 parameters, we make it canonical by setting
            // :math:`K2_l = Id` , :math:`K2_r = Id`.
            Specialization::IdEquiv => TwoQubitWeylDecomposition {
                specialization,
                a: 0.,
                b: 0.,
                c: 0.,
                K1l: matrix_multiply_2x2(general.K1l.view(), general.K2l.view()),
                K1r: matrix_multiply_2x2(general.K1r.view(), general.K2r.view()),
                K2l: Array2::eye(2),
                K2r: Array2::eye(2),
                ..general
            },
            // :math:`U \sim U_d(\pi/4, \pi/4, \pi/4) \sim U(\pi/4, \pi/4, -\pi/4) \sim \text{SWAP}`
            //
            // This gate binds 0 parameters, we make it canonical by setting
            // :math:`K2_l = Id` , :math:`K2_r = Id`.
            Specialization::SWAPEquiv => {
                if c > 0. {
                    TwoQubitWeylDecomposition {
                        specialization,
                        a: PI4,
                        b: PI4,
                        c: PI4,
                        K1l: matrix_multiply_2x2(general.K1l.view(), general.K2r.view()),
                        K1r: matrix_multiply_2x2(general.K1r.view(), general.K2l.view()),
                        K2l: Array2::eye(2),
                        K2r: Array2::eye(2),
                        ..general
                    }
                } else {
                    flipped_from_original = true;
                    TwoQubitWeylDecomposition {
                        specialization,
                        a: PI4,
                        b: PI4,
                        c: PI4,
                        global_phase: global_phase + PI2,
                        K1l: matrix_multiply_2x2(
                            matrix_multiply_2x2(general.K1l.view(), ipz).view(),
                            general.K2r.view(),
                        ),
                        K1r: matrix_multiply_2x2(
                            matrix_multiply_2x2(general.K1r.view(), ipz).view(),
                            general.K2l.view(),
                        ),
                        K2l: Array2::eye(2),
                        K2r: Array2::eye(2),
                        ..general
                    }
                }
            }
            // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, \alpha\pi/4) \sim \text{SWAP}^\alpha`
            //
            // This gate binds 3 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Id`.
            Specialization::PartialSWAPEquiv => {
                let closest = closest_partial_swap(a, b, c);
                let mut k2l_dag = general.K2l.t().to_owned();
                k2l_dag.view_mut().mapv_inplace(|x| x.conj());
                TwoQubitWeylDecomposition {
                    specialization,
                    a: closest,
                    b: closest,
                    c: closest,
                    K1l: matrix_multiply_2x2(general.K1l.view(), general.K2l.view()),
                    K1r: matrix_multiply_2x2(general.K1r.view(), general.K2l.view()),
                    K2r: matrix_multiply_2x2(k2l_dag.view(), general.K2r.view()),
                    K2l: Array2::eye(2),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, -\alpha\pi/4) \sim \text{SWAP}^\alpha`
            //
            // (a non-equivalent root of SWAP from the TwoQubitWeylPartialSWAPEquiv
            // similar to how :math:`x = (\pm \sqrt(x))^2`)
            //
            // This gate binds 3 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Id`
            Specialization::PartialSWAPFlipEquiv => {
                let closest = closest_partial_swap(a, b, -c);
                let mut k2l_dag = general.K2l.t().to_owned();
                k2l_dag.mapv_inplace(|x| x.conj());
                TwoQubitWeylDecomposition {
                    specialization,
                    a: closest,
                    b: closest,
                    c: -closest,
                    K1l: matrix_multiply_2x2(general.K1l.view(), general.K2l.view()),
                    K1r: matrix_multiply_2x2(
                        matrix_multiply_2x2(
                            matrix_multiply_2x2(general.K1r.view(), ipz).view(),
                            general.K2l.view(),
                        )
                        .view(),
                        ipz,
                    ),
                    K2r: matrix_multiply_2x2(
                        matrix_multiply_2x2(matrix_multiply_2x2(ipz, k2l_dag.view()).view(), ipz)
                            .view(),
                        general.K2r.view(),
                    ),
                    K2l: Array2::eye(2),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}`
            //
            // This gate binds 4 parameters, we make it canonical by setting:
            //
            //      :math:`K2_l = Ry(\theta_l) Rx(\lambda_l)` ,
            //      :math:`K2_r = Ry(\theta_r) Rx(\lambda_r)` .
            Specialization::ControlledEquiv => {
                let euler_basis = EulerBasis::XYX;
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), euler_basis);
                let [k2rtheta, k2rphi, k2rlambda, k2rphase] =
                    angles_from_unitary(general.K2r.view(), euler_basis);
                TwoQubitWeylDecomposition {
                    specialization,
                    a,
                    b: 0.,
                    c: 0.,
                    global_phase: global_phase + k2lphase + k2rphase,
                    K1l: matrix_multiply_2x2(general.K1l.view(), rx_matrix(k2lphi).view()),
                    K1r: matrix_multiply_2x2(general.K1r.view(), rx_matrix(k2rphi).view()),
                    K2l: matrix_multiply_2x2(
                        ry_matrix(k2ltheta).view(),
                        rx_matrix(k2llambda).view(),
                    ),
                    K2r: matrix_multiply_2x2(
                        ry_matrix(k2rtheta).view(),
                        rx_matrix(k2rlambda).view(),
                    ),
                    default_euler_basis: euler_basis,
                    ..general
                }
            }
            // :math:`U \sim U_d(\pi/4, \pi/4, \alpha) \sim \text{SWAP} \cdot \text{Ctrl-U}`
            //
            // This gate binds 4 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)` , :math:`K2_r = Ry(\theta_r)\cdot Rz(\lambda_r)`
            Specialization::MirrorControlledEquiv => {
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), EulerBasis::ZYZ);
                let [k2rtheta, k2rphi, k2rlambda, k2rphase] =
                    angles_from_unitary(general.K2r.view(), EulerBasis::ZYZ);
                TwoQubitWeylDecomposition {
                    specialization,
                    a: PI4,
                    b: PI4,
                    c,
                    global_phase: global_phase + k2lphase + k2rphase,
                    K1l: matrix_multiply_2x2(general.K1l.view(), rz_matrix(k2rphi).view()),
                    K1r: matrix_multiply_2x2(general.K1r.view(), rz_matrix(k2lphi).view()),
                    K2l: matrix_multiply_2x2(
                        ry_matrix(k2ltheta).view(),
                        rz_matrix(k2llambda).view(),
                    ),
                    K2r: matrix_multiply_2x2(
                        ry_matrix(k2rtheta).view(),
                        rz_matrix(k2rlambda).view(),
                    ),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, \alpha, \beta), \alpha \geq |\beta|`
            //
            // This gate binds 5 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)`.
            Specialization::fSimaabEquiv => {
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), EulerBasis::ZYZ);
                TwoQubitWeylDecomposition {
                    specialization,
                    a: (a + b) / 2.,
                    b: (a + b) / 2.,
                    c,
                    global_phase: global_phase + k2lphase,
                    K1r: matrix_multiply_2x2(general.K1r.view(), rz_matrix(k2lphi).view()),
                    K1l: matrix_multiply_2x2(general.K1l.view(), rz_matrix(k2lphi).view()),
                    K2l: matrix_multiply_2x2(
                        ry_matrix(k2ltheta).view(),
                        rz_matrix(k2llambda).view(),
                    ),
                    K2r: matrix_multiply_2x2(rz_matrix(-k2lphi).view(), general.K2r.view()),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
            //
            // This gate binds 5 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
            Specialization::fSimabbEquiv => {
                let euler_basis = EulerBasis::XYX;
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), euler_basis);
                TwoQubitWeylDecomposition {
                    specialization,
                    a,
                    b: (b + c) / 2.,
                    c: (b + c) / 2.,
                    global_phase: global_phase + k2lphase,
                    K1r: matrix_multiply_2x2(general.K1r.view(), rx_matrix(k2lphi).view()),
                    K1l: matrix_multiply_2x2(general.K1l.view(), rx_matrix(k2lphi).view()),
                    K2l: matrix_multiply_2x2(
                        ry_matrix(k2ltheta).view(),
                        rx_matrix(k2llambda).view(),
                    ),
                    K2r: matrix_multiply_2x2(rx_matrix(-k2lphi).view(), general.K2r.view()),
                    default_euler_basis: euler_basis,
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
            //
            // This gate binds 5 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
            Specialization::fSimabmbEquiv => {
                let euler_basis = EulerBasis::XYX;
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), euler_basis);
                TwoQubitWeylDecomposition {
                    specialization,
                    a,
                    b: (b - c) / 2.,
                    c: -((b - c) / 2.),
                    global_phase: global_phase + k2lphase,
                    K1l: matrix_multiply_2x2(general.K1l.view(), rx_matrix(k2lphi).view()),
                    K1r: matrix_multiply_2x2(
                        matrix_multiply_2x2(
                            matrix_multiply_2x2(general.K1r.view(), ipz).view(),
                            rx_matrix(k2lphi).view(),
                        )
                        .view(),
                        ipz,
                    ),
                    K2l: matrix_multiply_2x2(
                        ry_matrix(k2ltheta).view(),
                        rx_matrix(k2llambda).view(),
                    ),
                    K2r: matrix_multiply_2x2(
                        matrix_multiply_2x2(
                            matrix_multiply_2x2(ipz, rx_matrix(-k2lphi).view()).view(),
                            ipz,
                        )
                        .view(),
                        general.K2r.view(),
                    ),
                    default_euler_basis: euler_basis,
                    ..general
                }
            }
            // U has no special symmetry.
            //
            // This gate binds all 6 possible parameters, so there is no need to make the single-qubit
            // pre-/post-gates canonical.
            Specialization::General => general,
        };

        let tr = if flipped_from_original {
            let [da, db, dc] = [
                PI2 - a - specialized.a,
                b - specialized.b,
                -c - specialized.c,
            ];
            4. * Complex64::new(
                da.cos() * db.cos() * dc.cos(),
                da.sin() * db.sin() * dc.sin(),
            )
        } else {
            let [da, db, dc] = [a - specialized.a, b - specialized.b, c - specialized.c];
            4. * Complex64::new(
                da.cos() * db.cos() * dc.cos(),
                da.sin() * db.sin() * dc.sin(),
            )
        };
        specialized.calculated_fidelity = tr.trace_to_fid();
        if let Some(fid) = specialized.requested_fidelity {
            if specialized.calculated_fidelity + 1.0e-13 < fid {
                return Err(QiskitError::new_err(format!(
                    "Specialization: {:?} calculated fidelity: {} is worse than requested fidelity: {}",
                    specialized.specialization,
                    specialized.calculated_fidelity,
                    fid
                )));
            }
        }
        specialized.global_phase += tr.arg();
        Ok(specialized)
    }

    #[allow(non_snake_case)]
    #[getter]
    fn K1l(&self, py: Python) -> PyObject {
        self.K1l.to_pyarray_bound(py).into()
    }

    #[allow(non_snake_case)]
    #[getter]
    fn K1r(&self, py: Python) -> PyObject {
        self.K1r.to_pyarray_bound(py).into()
    }

    #[allow(non_snake_case)]
    #[getter]
    fn K2l(&self, py: Python) -> PyObject {
        self.K2l.to_pyarray_bound(py).into()
    }

    #[allow(non_snake_case)]
    #[getter]
    fn K2r(&self, py: Python) -> PyObject {
        self.K2r.to_pyarray_bound(py).into()
    }

    #[getter]
    fn unitary_matrix(&self, py: Python) -> PyObject {
        self.unitary_matrix.to_pyarray_bound(py).into()
    }

    #[pyo3(signature = (euler_basis=None, simplify=false, atol=None))]
    fn circuit(
        &self,
        euler_basis: Option<PyBackedStr>,
        simplify: bool,
        atol: Option<f64>,
    ) -> PyResult<TwoQubitGateSequence> {
        let euler_basis: EulerBasis = match euler_basis {
            Some(basis) => EulerBasis::__new__(basis.deref())?,
            None => self.default_euler_basis,
        };
        let target_1q_basis_list: Vec<EulerBasis> = vec![euler_basis];

        let mut gate_sequence = Vec::new();
        let mut global_phase: f64 = self.global_phase;

        let c2r = unitary_to_gate_sequence_inner(
            self.K2r.view(),
            &target_1q_basis_list,
            0,
            None,
            simplify,
            atol,
        )
        .unwrap();
        for gate in c2r.gates {
            gate_sequence.push((gate.0, gate.1, smallvec![0]))
        }
        global_phase += c2r.global_phase;
        let c2l = unitary_to_gate_sequence_inner(
            self.K2l.view(),
            &target_1q_basis_list,
            1,
            None,
            simplify,
            atol,
        )
        .unwrap();
        for gate in c2l.gates {
            gate_sequence.push((gate.0, gate.1, smallvec![1]))
        }
        global_phase += c2l.global_phase;
        self.weyl_gate(
            simplify,
            &mut gate_sequence,
            atol.unwrap_or(ANGLE_ZERO_EPSILON),
            &mut global_phase,
        );
        let c1r = unitary_to_gate_sequence_inner(
            self.K1r.view(),
            &target_1q_basis_list,
            0,
            None,
            simplify,
            atol,
        )
        .unwrap();
        for gate in c1r.gates {
            gate_sequence.push((gate.0, gate.1, smallvec![0]))
        }
        global_phase += c2r.global_phase;
        let c1l = unitary_to_gate_sequence_inner(
            self.K1l.view(),
            &target_1q_basis_list,
            1,
            None,
            simplify,
            atol,
        )
        .unwrap();
        for gate in c1l.gates {
            gate_sequence.push((gate.0, gate.1, smallvec![1]))
        }
        Ok(TwoQubitGateSequence {
            gates: gate_sequence,
            global_phase,
        })
    }
}

type TwoQubitSequenceVec = Vec<(String, SmallVec<[f64; 3]>, SmallVec<[u8; 2]>)>;

#[pyclass(sequence)]
pub struct TwoQubitGateSequence {
    gates: TwoQubitSequenceVec,
    #[pyo3(get)]
    global_phase: f64,
}

#[pymethods]
impl TwoQubitGateSequence {
    #[new]
    fn new() -> Self {
        TwoQubitGateSequence {
            gates: Vec::new(),
            global_phase: 0.,
        }
    }

    fn __getstate__(&self) -> (TwoQubitSequenceVec, f64) {
        (self.gates.clone(), self.global_phase)
    }

    fn __setstate__(&mut self, state: (TwoQubitSequenceVec, f64)) {
        self.gates = state.0;
        self.global_phase = state.1;
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.gates.len())
    }

    fn __getitem__(&self, py: Python, idx: SliceOrInt) -> PyResult<PyObject> {
        match idx {
            SliceOrInt::Slice(slc) => {
                let len = self.gates.len().try_into().unwrap();
                let indices = slc.indices(len)?;
                let mut out_vec: TwoQubitSequenceVec = Vec::new();
                // Start and stop will always be positive the slice api converts
                // negatives to the index for example:
                // list(range(5))[-1:-3:-1]
                // will return start=4, stop=2, and step=-
                let mut pos: isize = indices.start;
                let mut cond = if indices.step < 0 {
                    pos > indices.stop
                } else {
                    pos < indices.stop
                };
                while cond {
                    if pos < len as isize {
                        out_vec.push(self.gates[pos as usize].clone());
                    }
                    pos += indices.step;
                    if indices.step < 0 {
                        cond = pos > indices.stop;
                    } else {
                        cond = pos < indices.stop;
                    }
                }
                Ok(out_vec.into_py(py))
            }
            SliceOrInt::Int(idx) => {
                let len = self.gates.len() as isize;
                if idx >= len || idx < -len {
                    Err(PyIndexError::new_err(format!("Invalid index, {idx}")))
                } else if idx < 0 {
                    let len = self.gates.len();
                    Ok(self.gates[len - idx.unsigned_abs()].to_object(py))
                } else {
                    Ok(self.gates[idx as usize].to_object(py))
                }
            }
        }
    }
}
#[allow(non_snake_case)]
#[pyclass(module = "qiskit._accelerate.two_qubit_decompose", subclass)]
pub struct TwoQubitBasisDecomposer {
    gate: String,
    basis_fidelity: f64,
    euler_basis: EulerBasis,
    pulse_optimize: Option<bool>,
    basis_decomposer: TwoQubitWeylDecomposition,
    #[pyo3(get)]
    super_controlled: bool,
    u0l: Array2<Complex64>,
    u0r: Array2<Complex64>,
    u1l: Array2<Complex64>,
    u1ra: Array2<Complex64>,
    u1rb: Array2<Complex64>,
    u2la: Array2<Complex64>,
    u2lb: Array2<Complex64>,
    u2ra: Array2<Complex64>,
    u2rb: Array2<Complex64>,
    u3l: Array2<Complex64>,
    u3r: Array2<Complex64>,
    q0l: Array2<Complex64>,
    q0r: Array2<Complex64>,
    q1la: Array2<Complex64>,
    q1lb: Array2<Complex64>,
    q1ra: Array2<Complex64>,
    q1rb: Array2<Complex64>,
    q2l: Array2<Complex64>,
    q2r: Array2<Complex64>,
}
impl TwoQubitBasisDecomposer {
    fn decomp1_inner(
        &self,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Array2<Complex64>; 8]> {
        // FIXME: fix for z!=0 and c!=0 using closest reflection (not always in the Weyl chamber)
        smallvec![
            matrix_multiply_2x2(
                transpose_conjugate(self.basis_decomposer.K2r.view()).view(),
                target.K2r.view()
            ),
            matrix_multiply_2x2(
                transpose_conjugate(self.basis_decomposer.K2l.view()).view(),
                target.K2l.view()
            ),
            matrix_multiply_2x2(
                target.K1r.view(),
                transpose_conjugate(self.basis_decomposer.K1r.view()).view()
            ),
            matrix_multiply_2x2(
                target.K1l.view(),
                transpose_conjugate(self.basis_decomposer.K1l.view()).view()
            ),
        ]
    }

    fn decomp2_supercontrolled_inner(
        &self,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Array2<Complex64>; 8]> {
        smallvec![
            matrix_multiply_2x2(self.q2r.view(), target.K2r.view()),
            matrix_multiply_2x2(self.q2l.view(), target.K2l.view()),
            matrix_multiply_2x2(
                matrix_multiply_2x2(self.q1ra.view(), rz_matrix(2. * target.b).view()).view(),
                self.q1rb.view()
            ),
            matrix_multiply_2x2(
                matrix_multiply_2x2(self.q1la.view(), rz_matrix(-2. * target.a).view()).view(),
                self.q1lb.view()
            ),
            matrix_multiply_2x2(target.K1r.view(), self.q0r.view()),
            matrix_multiply_2x2(target.K1l.view(), self.q0l.view()),
        ]
    }

    fn decomp3_supercontrolled_inner(
        &self,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[Array2<Complex64>; 8]> {
        smallvec![
            matrix_multiply_2x2(self.u3r.view(), target.K2r.view()),
            matrix_multiply_2x2(self.u3l.view(), target.K2l.view()),
            matrix_multiply_2x2(
                matrix_multiply_2x2(self.u2ra.view(), rz_matrix(2. * target.b).view()).view(),
                self.u2rb.view()
            ),
            matrix_multiply_2x2(
                matrix_multiply_2x2(self.u2la.view(), rz_matrix(-2. * target.a).view()).view(),
                self.u2lb.view()
            ),
            matrix_multiply_2x2(
                matrix_multiply_2x2(self.u1ra.view(), rz_matrix(-2. * target.c).view()).view(),
                self.u1rb.view()
            ),
            self.u1l.clone(),
            matrix_multiply_2x2(target.K1r.view(), self.u0r.view()),
            matrix_multiply_2x2(target.K1l.view(), self.u0l.view()),
        ]
    }

    /// Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT gates assuming
    /// two CNOT gates are needed.
    ///
    /// This first decomposes each unitary from the KAK decomposition into ZXZ on the source
    /// qubit of the CNOTs and XZX on the targets in order to commute operators to beginning and
    /// end of decomposition. The beginning and ending single qubit gates are then
    /// collapsed and re-decomposed with the single qubit decomposer. This last step could be avoided
    /// if performance is a concern.
    fn get_sx_vz_2cx_efficient_euler(
        &self,
        decomposition: &SmallVec<[Array2<Complex64>; 8]>,
        target_decomposed: &TwoQubitWeylDecomposition,
    ) -> Option<TwoQubitGateSequence> {
        let mut gates = Vec::new();
        let mut global_phase = target_decomposed.global_phase;
        global_phase -= 2. * self.basis_decomposer.global_phase;
        let euler_q0: Vec<[f64; 3]> = decomposition
            .iter()
            .step_by(2)
            .map(|decomp| {
                let euler_angles = angles_from_unitary(decomp.view(), EulerBasis::ZXZ);
                global_phase += euler_angles[3];
                [euler_angles[2], euler_angles[0], euler_angles[1]]
            })
            .collect();
        let euler_q1: Vec<[f64; 3]> = decomposition
            .iter()
            .skip(1)
            .step_by(2)
            .map(|decomp| {
                let euler_angles = angles_from_unitary(decomp.view(), EulerBasis::XZX);
                global_phase += euler_angles[3];
                [euler_angles[2], euler_angles[0], euler_angles[1]]
            })
            .collect();
        let mut euler_matrix_q0 = matrix_multiply_2x2(
            rx_matrix(euler_q0[0][1]).view(),
            rz_matrix(euler_q0[0][0]).view(),
        );
        euler_matrix_q0 = matrix_multiply_2x2(
            rz_matrix(euler_q0[0][2] + euler_q0[1][0] + PI2).view(),
            euler_matrix_q0.view(),
        );
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q0.view(), 0);
        let mut euler_matrix_q1 = matrix_multiply_2x2(
            rz_matrix(euler_q1[0][1]).view(),
            rx_matrix(euler_q1[0][0]).view(),
        );
        euler_matrix_q1 = matrix_multiply_2x2(
            rx_matrix(euler_q1[0][2] + euler_q1[1][0]).view(),
            euler_matrix_q1.view(),
        );
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q1.view(), 1);
        gates.push(("cx".to_string(), smallvec![], smallvec![0, 1]));
        gates.push(("sx".to_string(), smallvec![], smallvec![0]));
        gates.push((
            "rz".to_string(),
            smallvec![euler_q0[1][1] - PI],
            smallvec![0],
        ));
        gates.push(("sx".to_string(), smallvec![], smallvec![0]));
        gates.push(("rz".to_string(), smallvec![euler_q1[1][1]], smallvec![1]));
        global_phase += PI2;
        gates.push(("cx".to_string(), smallvec![], smallvec![0, 1]));
        let mut euler_matrix_q0 = matrix_multiply_2x2(
            rx_matrix(euler_q0[2][1]).view(),
            rz_matrix(euler_q0[1][2] + euler_q0[2][0] + PI2).view(),
        );
        euler_matrix_q0 =
            matrix_multiply_2x2(rz_matrix(euler_q0[2][2]).view(), euler_matrix_q0.view());
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q0.view(), 0);
        let mut euler_matrix_q1 = matrix_multiply_2x2(
            rz_matrix(euler_q1[2][1]).view(),
            rx_matrix(euler_q1[1][2] + euler_q1[2][0]).view(),
        );
        euler_matrix_q1 =
            matrix_multiply_2x2(rx_matrix(euler_q1[2][2]).view(), euler_matrix_q1.view());
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q1.view(), 1);
        Some(TwoQubitGateSequence {
            gates,
            global_phase,
        })
    }

    /// Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT gates assuming
    /// three CNOT gates are needed.
    ///
    /// This first decomposes each unitary from the KAK decomposition into ZXZ on the source
    /// qubit of the CNOTs and XZX on the targets in order commute operators to beginning and
    /// end of decomposition. Inserting Hadamards reverses the direction of the CNOTs and transforms
    /// a variable Rx -> variable virtual Rz. The beginning and ending single qubit gates are then
    /// collapsed and re-decomposed with the single qubit decomposer. This last step could be avoided
    /// if performance is a concern.
    fn get_sx_vz_3cx_efficient_euler(
        &self,
        decomposition: &SmallVec<[Array2<Complex64>; 8]>,
        target_decomposed: &TwoQubitWeylDecomposition,
    ) -> Option<TwoQubitGateSequence> {
        let mut gates = Vec::new();
        let mut global_phase = target_decomposed.global_phase;
        global_phase -= 3. * self.basis_decomposer.global_phase;
        global_phase = global_phase.rem_euclid(TWO_PI);
        let atol = 1e-10; // absolute tolerance for floats
                          // Decompose source unitaries to zxz
        let euler_q0: Vec<[f64; 3]> = decomposition
            .iter()
            .step_by(2)
            .map(|decomp| {
                let euler_angles = angles_from_unitary(decomp.view(), EulerBasis::ZXZ);
                global_phase += euler_angles[3];
                [euler_angles[2], euler_angles[0], euler_angles[1]]
            })
            .collect();
        // Decompose target unitaries to xzx
        let euler_q1: Vec<[f64; 3]> = decomposition
            .iter()
            .skip(1)
            .step_by(2)
            .map(|decomp| {
                let euler_angles = angles_from_unitary(decomp.view(), EulerBasis::XZX);
                global_phase += euler_angles[3];
                [euler_angles[2], euler_angles[0], euler_angles[1]]
            })
            .collect();

        let x12 = euler_q0[1][2] + euler_q0[2][0];
        let x12_is_non_zero = !abs_diff_eq!(x12, 0., epsilon = atol);
        let mut x12_is_old_mult = None;
        let mut x12_phase = 0.;
        let x12_is_pi_mult = abs_diff_eq!(x12.sin(), 0., epsilon = atol);
        if x12_is_pi_mult {
            x12_is_old_mult = Some(abs_diff_eq!(x12.cos(), -1., epsilon = atol));
            x12_phase = PI * x12.cos();
        }
        let x02_add = x12 - euler_q0[1][0];
        let x12_is_half_pi = abs_diff_eq!(x12, PI2, epsilon = atol);

        let mut euler_matrix_q0 = matrix_multiply_2x2(
            rx_matrix(euler_q0[0][1]).view(),
            rz_matrix(euler_q0[0][0]).view(),
        );
        if x12_is_non_zero && x12_is_pi_mult {
            euler_matrix_q0 = matrix_multiply_2x2(
                rz_matrix(euler_q0[0][2] - x02_add).view(),
                euler_matrix_q0.view(),
            );
        } else {
            euler_matrix_q0 = matrix_multiply_2x2(
                rz_matrix(euler_q0[0][2] + euler_q0[1][0]).view(),
                euler_matrix_q0.view(),
            );
        }
        euler_matrix_q0 = matrix_multiply_2x2(aview2(&HGATE), euler_matrix_q0.view());
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q0.view(), 0);

        let rx_0 = rx_matrix(euler_q1[0][0]);
        let rz = rz_matrix(euler_q1[0][1]);
        let rx_1 = rx_matrix(euler_q1[0][2] + euler_q1[1][0]);
        let mut euler_matrix_q1 = matrix_multiply_2x2(rz.view(), rx_0.view());
        euler_matrix_q1 = matrix_multiply_2x2(rx_1.view(), euler_matrix_q1.view());
        euler_matrix_q1 = matrix_multiply_2x2(aview2(&HGATE), euler_matrix_q1.view());
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix_q1.view(), 1);

        gates.push(("cx".to_string(), smallvec![], smallvec![1, 0]));

        if x12_is_pi_mult {
            // even or odd multiple
            if x12_is_non_zero {
                global_phase += x12_phase;
            }
            if x12_is_non_zero && x12_is_old_mult.unwrap() {
                gates.push(("rz".to_string(), smallvec![-euler_q0[1][1]], smallvec![0]));
            } else {
                gates.push(("rz".to_string(), smallvec![euler_q0[1][1]], smallvec![0]));
                global_phase += PI;
            }
        }
        if x12_is_half_pi {
            gates.push(("sx".to_string(), smallvec![], smallvec![0]));
            global_phase -= PI4;
        } else if x12_is_non_zero && !x12_is_pi_mult {
            if self.pulse_optimize.is_none() {
                self.append_1q_sequence(&mut gates, &mut global_phase, rx_matrix(x12).view(), 0);
            } else {
                return None;
            }
        }
        if abs_diff_eq!(euler_q1[1][1], PI2, epsilon = atol) {
            gates.push(("sx".to_string(), smallvec![], smallvec![1]));
            global_phase -= PI4
        } else if self.pulse_optimize.is_none() {
            self.append_1q_sequence(
                &mut gates,
                &mut global_phase,
                rx_matrix(euler_q1[1][1]).view(),
                1,
            );
        } else {
            return None;
        }
        gates.push((
            "rz".to_string(),
            smallvec![euler_q1[1][2] + euler_q1[2][0]],
            smallvec![1],
        ));
        gates.push(("cx".to_string(), smallvec![], smallvec![1, 0]));
        gates.push(("rz".to_string(), smallvec![euler_q0[2][1]], smallvec![0]));
        if abs_diff_eq!(euler_q1[2][1], PI2, epsilon = atol) {
            gates.push(("sx".to_string(), smallvec![], smallvec![1]));
            global_phase -= PI4;
        } else if self.pulse_optimize.is_none() {
            self.append_1q_sequence(
                &mut gates,
                &mut global_phase,
                rx_matrix(euler_q1[2][1]).view(),
                1,
            );
        } else {
            return None;
        }
        gates.push(("cx".to_string(), smallvec![], smallvec![1, 0]));
        let mut euler_matrix = matrix_multiply_2x2(
            rz_matrix(euler_q0[2][2] + euler_q0[3][0]).view(),
            aview2(&HGATE),
        );
        euler_matrix = matrix_multiply_2x2(rx_matrix(euler_q0[3][1]).view(), euler_matrix.view());
        euler_matrix = matrix_multiply_2x2(rz_matrix(euler_q0[3][2]).view(), euler_matrix.view());
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix.view(), 0);

        let mut euler_matrix = matrix_multiply_2x2(
            rx_matrix(euler_q1[2][2] + euler_q1[3][0]).view(),
            aview2(&HGATE),
        );
        euler_matrix = matrix_multiply_2x2(rz_matrix(euler_q1[3][1]).view(), euler_matrix.view());
        euler_matrix = matrix_multiply_2x2(rx_matrix(euler_q1[3][2]).view(), euler_matrix.view());
        self.append_1q_sequence(&mut gates, &mut global_phase, euler_matrix.view(), 1);

        let out_unitary = compute_unitary(&gates, global_phase);
        // TODO: fix the sign problem to avoid correction here
        if abs_diff_eq!(
            target_decomposed.unitary_matrix[[0, 0]],
            -out_unitary[[0, 0]],
            epsilon = atol
        ) {
            global_phase += PI;
        }
        Some(TwoQubitGateSequence {
            gates,
            global_phase,
        })
    }

    fn append_1q_sequence(
        &self,
        gates: &mut TwoQubitSequenceVec,
        global_phase: &mut f64,
        unitary: ArrayView2<Complex64>,
        qubit: u8,
    ) {
        let target_1q_basis_list = vec![self.euler_basis];
        let sequence = unitary_to_gate_sequence_inner(
            unitary,
            &target_1q_basis_list,
            qubit as usize,
            None,
            true,
            None,
        );
        if let Some(sequence) = sequence {
            *global_phase += sequence.global_phase;
            for gate in sequence.gates {
                gates.push((gate.0, gate.1, smallvec![qubit]));
            }
        }
    }

    fn pulse_optimal_chooser(
        &self,
        best_nbasis: u8,
        decomposition: &SmallVec<[Array2<Complex64>; 8]>,
        target_decomposed: &TwoQubitWeylDecomposition,
    ) -> PyResult<Option<TwoQubitGateSequence>> {
        if self.pulse_optimize.is_some()
            && (best_nbasis == 0 || best_nbasis == 1 || best_nbasis > 3)
        {
            return Ok(None);
        }
        match self.euler_basis {
            EulerBasis::ZSX => (),
            EulerBasis::ZSXX => (),
            _ => {
                if self.pulse_optimize.is_some() {
                    return Err(QiskitError::new_err(format!(
                        "'pulse_optimize' currently only works with ZSX basis ({} used)",
                        self.euler_basis.as_str()
                    )));
                } else {
                    return Ok(None);
                }
            }
        }
        if self.gate != "cx" {
            if self.pulse_optimize.is_some() {
                return Err(QiskitError::new_err(
                    "pulse_optimizer currently only works with CNOT entangling gate",
                ));
            } else {
                return Ok(None);
            }
        }
        let res = if best_nbasis == 3 {
            self.get_sx_vz_3cx_efficient_euler(decomposition, target_decomposed)
        } else if best_nbasis == 2 {
            self.get_sx_vz_2cx_efficient_euler(decomposition, target_decomposed)
        } else {
            None
        };
        if self.pulse_optimize.is_some() && res.is_none() {
            return Err(QiskitError::new_err(
                "Failed to compute requested pulse optimal decomposition",
            ));
        }
        Ok(res)
    }
}

static K12R_ARR: [[Complex64; 2]; 2] = [
    [
        Complex64::new(0., FRAC_1_SQRT_2),
        Complex64::new(FRAC_1_SQRT_2, 0.),
    ],
    [
        Complex64::new(-FRAC_1_SQRT_2, 0.),
        Complex64::new(0., -FRAC_1_SQRT_2),
    ],
];

static K12L_ARR: [[Complex64; 2]; 2] = [
    [Complex64::new(0.5, 0.5), Complex64::new(0.5, 0.5)],
    [Complex64::new(-0.5, 0.5), Complex64::new(0.5, -0.5)],
];

fn decomp0_inner(target: &TwoQubitWeylDecomposition) -> SmallVec<[Array2<Complex64>; 8]> {
    smallvec![
        matrix_multiply_2x2(target.K1r.view(), target.K2r.view()),
        matrix_multiply_2x2(target.K1l.view(), target.K2l.view()),
    ]
}

#[pymethods]
impl TwoQubitBasisDecomposer {
    fn __getnewargs__(&self, py: Python) -> (String, PyObject, f64, &str, Option<bool>) {
        (
            self.gate.clone(),
            self.basis_decomposer
                .unitary_matrix
                .to_pyarray_bound(py)
                .into(),
            self.basis_fidelity,
            self.euler_basis.as_str(),
            self.pulse_optimize,
        )
    }

    #[new]
    #[pyo3(signature=(gate, gate_matrix, basis_fidelity=1.0, euler_basis="U", pulse_optimize=None))]
    fn new(
        gate: String,
        gate_matrix: PyReadonlyArray2<Complex64>,
        basis_fidelity: f64,
        euler_basis: &str,
        pulse_optimize: Option<bool>,
    ) -> PyResult<Self> {
        let ipz: ArrayView2<Complex64> = aview2(&IPZ);
        let basis_decomposer =
            TwoQubitWeylDecomposition::new(gate_matrix, Some(DEFAULT_FIDELITY), None)?;
        let super_controlled = relative_eq!(basis_decomposer.a, PI4, max_relative = 1e-09)
            && relative_eq!(basis_decomposer.c, 0.0, max_relative = 1e-09);

        // Create some useful matrices U1, U2, U3 are equivalent to the basis,
        // expand as Ui = Ki1.Ubasis.Ki2
        let b = basis_decomposer.b;
        let temp = Complex64::new(0.5, -0.5);
        let k11l = array![
            [
                temp * (Complex64::new(0., -1.) * Complex64::new(0., -b).exp()),
                temp * Complex64::new(0., -b).exp()
            ],
            [
                temp * (Complex64::new(0., -1.) * Complex64::new(0., b).exp()),
                temp * -(Complex64::new(0., b).exp())
            ],
        ];
        let k11r = array![
            [
                FRAC_1_SQRT_2 * (Complex64::new(0., 1.) * Complex64::new(0., -b).exp()),
                FRAC_1_SQRT_2 * -Complex64::new(0., -b).exp()
            ],
            [
                FRAC_1_SQRT_2 * Complex64::new(0., b).exp(),
                FRAC_1_SQRT_2 * (Complex64::new(0., -1.) * Complex64::new(0., b).exp())
            ],
        ];
        let k12l = aview2(&K12L_ARR);
        let k12r = aview2(&K12R_ARR);
        let k32l_k21l = array![
            [
                FRAC_1_SQRT_2 * Complex64::new(1., (2. * b).cos()),
                FRAC_1_SQRT_2 * (Complex64::new(0., 1.) * (2. * b).sin())
            ],
            [
                FRAC_1_SQRT_2 * (Complex64::new(0., 1.) * (2. * b).sin()),
                FRAC_1_SQRT_2 * Complex64::new(1., -(2. * b).cos())
            ],
        ];
        let temp = Complex64::new(0.5, 0.5);
        let k21r = array![
            [
                temp * (Complex64::new(0., -1.) * Complex64::new(0., -2. * b).exp()),
                temp * Complex64::new(0., -2. * b).exp()
            ],
            [
                temp * (Complex64::new(0., 1.) * Complex64::new(0., 2. * b).exp()),
                temp * Complex64::new(0., 2. * b).exp()
            ],
        ];
        const K22L_ARR: [[Complex64; 2]; 2] = [
            [
                Complex64::new(FRAC_1_SQRT_2, 0.),
                Complex64::new(-FRAC_1_SQRT_2, 0.),
            ],
            [
                Complex64::new(FRAC_1_SQRT_2, 0.),
                Complex64::new(FRAC_1_SQRT_2, 0.),
            ],
        ];
        let k22l = aview2(&K22L_ARR);
        let k22r_arr: [[Complex64; 2]; 2] = [
            [Complex64::zero(), Complex64::new(1., 0.)],
            [Complex64::new(-1., 0.), Complex64::zero()],
        ];
        let k22r = aview2(&k22r_arr);
        let k31l = array![
            [
                FRAC_1_SQRT_2 * Complex64::new(0., -b).exp(),
                FRAC_1_SQRT_2 * Complex64::new(0., -b).exp()
            ],
            [
                FRAC_1_SQRT_2 * -Complex64::new(0., b).exp(),
                FRAC_1_SQRT_2 * Complex64::new(0., b).exp()
            ],
        ];
        let temp = Complex64::new(0., 1.);
        let k31r = array![
            [temp * Complex64::new(0., b).exp(), Complex64::zero()],
            [Complex64::zero(), temp * -Complex64::new(0., -b).exp()],
        ];
        let temp = Complex64::new(0.5, 0.5);
        let k32r = array![
            [
                temp * Complex64::new(0., b).exp(),
                temp * -Complex64::new(0., -b).exp()
            ],
            [
                temp * (Complex64::new(0., -1.) * Complex64::new(0., b).exp()),
                temp * (Complex64::new(0., -1.) * Complex64::new(0., -b).exp())
            ],
        ];
        let k1ld = transpose_conjugate(basis_decomposer.K1l.view());
        let k1rd = transpose_conjugate(basis_decomposer.K1r.view());
        let k2ld = transpose_conjugate(basis_decomposer.K2l.view());
        let k2rd = transpose_conjugate(basis_decomposer.K2r.view());
        // Pre-build the fixed parts of the matrices used in 3-part decomposition
        let u0l = matrix_multiply_2x2(k31l.view(), k1ld.view());
        let u0r = matrix_multiply_2x2(k31r.view(), k1rd.view());
        let u1l = matrix_multiply_2x2(
            matrix_multiply_2x2(k2ld.view(), k32l_k21l.view()).view(),
            k1ld.view(),
        );
        let u1ra = matrix_multiply_2x2(k2rd.view(), k32r.view());
        let u1rb = matrix_multiply_2x2(k21r.view(), k1rd.view());
        let u2la = matrix_multiply_2x2(k2ld.view(), k22l.view());
        let u2lb = matrix_multiply_2x2(k11l.view(), k1ld.view());
        let u2ra = matrix_multiply_2x2(k2rd.view(), k22r.view());
        let u2rb = matrix_multiply_2x2(k11r.view(), k1rd.view());
        let u3l = matrix_multiply_2x2(k2ld.view(), k12l.view());
        let u3r = matrix_multiply_2x2(k2rd.view(), k12r.view());
        // Pre-build the fixed parts of the matrices used in the 2-part decomposition
        let q0l = matrix_multiply_2x2(transpose_conjugate(k12l.view()).view(), k1ld.view());
        let q0r = matrix_multiply_2x2(
            matrix_multiply_2x2(transpose_conjugate(k12r.view()).view(), ipz.view()).view(),
            k1rd.view(),
        );
        let q1la = matrix_multiply_2x2(k2ld.view(), transpose_conjugate(k11l.view()).view());
        let q1lb = matrix_multiply_2x2(k11l.view(), k1ld.view());
        let q1ra = matrix_multiply_2x2(
            matrix_multiply_2x2(k2rd.view(), ipz.view()).view(),
            transpose_conjugate(k11r.view()).view(),
        );
        let q1rb = matrix_multiply_2x2(k11r.view(), k1rd.view());
        let q2l = matrix_multiply_2x2(k2ld.view(), k12l.view());
        let q2r = matrix_multiply_2x2(k2rd.view(), k12r.view());

        Ok(TwoQubitBasisDecomposer {
            gate,
            basis_fidelity,
            euler_basis: EulerBasis::__new__(euler_basis)?,
            pulse_optimize,
            basis_decomposer,
            super_controlled,
            u0l,
            u0r,
            u1l,
            u1ra,
            u1rb,
            u2la,
            u2lb,
            u2ra,
            u2rb,
            u3l,
            u3r,
            q0l,
            q0r,
            q1la,
            q1lb,
            q1ra,
            q1rb,
            q2l,
            q2r,
        })
    }

    fn traces(&self, target: &TwoQubitWeylDecomposition) -> [Complex64; 4] {
        [
            4. * Complex64::new(
                target.a.cos() * target.b.cos() * target.c.cos(),
                target.a.sin() * target.b.sin() * target.c.sin(),
            ),
            4. * Complex64::new(
                (PI4 - target.a).cos()
                    * (self.basis_decomposer.b - target.b).cos()
                    * target.c.cos(),
                (PI4 - target.a).sin()
                    * (self.basis_decomposer.b - target.b).sin()
                    * target.c.sin(),
            ),
            Complex64::new(4. * target.c.cos(), 0.),
            Complex64::new(4., 0.),
        ]
    }

    /// Decompose target :math:`\sim U_d(x, y, z)` with :math:`0` uses of the basis gate.
    /// Result :math:`U_r` has trace:
    ///
    /// .. math::
    ///
    ///     \Big\vert\text{Tr}(U_r\cdot U_\text{target}^{\dag})\Big\vert =
    ///     4\Big\vert (\cos(x)\cos(y)\cos(z)+ j \sin(x)\sin(y)\sin(z)\Big\vert
    ///
    /// which is optimal for all targets and bases
    #[staticmethod]
    fn decomp0(py: Python, target: &TwoQubitWeylDecomposition) -> SmallVec<[PyObject; 2]> {
        decomp0_inner(target)
            .into_iter()
            .map(|x| x.into_pyarray_bound(py).into())
            .collect()
    }

    /// Decompose target :math:`\sim U_d(x, y, z)` with :math:`1` use of the basis gate
    /// math:`\sim U_d(a, b, c)`.
    /// Result :math:`U_r` has trace:
    ///
    /// .. math::
    ///
    ///     \Big\vert\text{Tr}(U_r \cdot U_\text{target}^{\dag})\Big\vert =
    ///     4\Big\vert \cos(x-a)\cos(y-b)\cos(z-c) + j \sin(x-a)\sin(y-b)\sin(z-c)\Big\vert
    ///
    /// which is optimal for all targets and bases with ``z==0`` or ``c==0``.
    fn decomp1(&self, py: Python, target: &TwoQubitWeylDecomposition) -> SmallVec<[PyObject; 4]> {
        self.decomp1_inner(target)
            .into_iter()
            .map(|x| x.into_pyarray_bound(py).into())
            .collect()
    }

    /// Decompose target :math:`\sim U_d(x, y, z)` with :math:`2` uses of the basis gate.
    ///
    /// For supercontrolled basis :math:`\sim U_d(\pi/4, b, 0)`, all b, result :math:`U_r` has trace
    ///
    /// .. math::
    ///
    ///     \Big\vert\text{Tr}(U_r \cdot U_\text{target}^\dag) \Big\vert = 4\cos(z)
    ///
    /// which is the optimal approximation for basis of CNOT-class :math:`\sim U_d(\pi/4, 0, 0)`
    /// or DCNOT-class :math:`\sim U_d(\pi/4, \pi/4, 0)` and any target. It may
    /// be sub-optimal for :math:`b \neq 0` (i.e. there exists an exact decomposition for any target
    /// using :math:`B \sim U_d(\pi/4, \pi/8, 0)`, but it may not be this decomposition).
    /// This is an exact decomposition for supercontrolled basis and target :math:`\sim U_d(x, y, 0)`.
    /// No guarantees for non-supercontrolled basis.
    fn decomp2_supercontrolled(
        &self,
        py: Python,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[PyObject; 6]> {
        self.decomp2_supercontrolled_inner(target)
            .into_iter()
            .map(|x| x.into_pyarray_bound(py).into())
            .collect()
    }

    /// Decompose target with :math:`3` uses of the basis.
    ///
    /// This is an exact decomposition for supercontrolled basis :math:`\sim U_d(\pi/4, b, 0)`, all b,
    /// and any target. No guarantees for non-supercontrolled basis.
    fn decomp3_supercontrolled(
        &self,
        py: Python,
        target: &TwoQubitWeylDecomposition,
    ) -> SmallVec<[PyObject; 8]> {
        self.decomp3_supercontrolled_inner(target)
            .into_iter()
            .map(|x| x.into_pyarray_bound(py).into())
            .collect()
    }

    /// Decompose a two-qubit ``unitary`` over fixed basis and :math:`SU(2)` using the best
    /// approximation given that each basis application has a finite ``basis_fidelity``.
    #[pyo3(signature = (unitary, basis_fidelity=None, approximate=true, _num_basis_uses=None))]
    fn __call__(
        &self,
        unitary: PyReadonlyArray2<Complex64>,
        basis_fidelity: Option<f64>,
        approximate: bool,
        _num_basis_uses: Option<u8>,
    ) -> PyResult<TwoQubitGateSequence> {
        let basis_fidelity = if !approximate {
            1.0
        } else {
            basis_fidelity.unwrap_or(self.basis_fidelity)
        };
        let target_decomposed =
            TwoQubitWeylDecomposition::new(unitary, Some(DEFAULT_FIDELITY), None)?;
        let traces = self.traces(&target_decomposed);
        let best_nbasis = traces
            .into_iter()
            .enumerate()
            .map(|(idx, trace)| (idx, trace.trace_to_fid() * basis_fidelity.powi(idx as i32)))
            .min_by(|(_idx1, fid1), (_idx2, fid2)| fid2.partial_cmp(fid1).unwrap())
            .unwrap()
            .0;
        let best_nbasis = _num_basis_uses.unwrap_or(best_nbasis as u8);
        let decomposition = match best_nbasis {
            0 => decomp0_inner(&target_decomposed),
            1 => self.decomp1_inner(&target_decomposed),
            2 => self.decomp2_supercontrolled_inner(&target_decomposed),
            3 => self.decomp3_supercontrolled_inner(&target_decomposed),
            _ => unreachable!("Invalid basis to use"),
        };
        let pulse_optimize = self.pulse_optimize.unwrap_or(true);
        let sequence = if pulse_optimize {
            self.pulse_optimal_chooser(best_nbasis, &decomposition, &target_decomposed)?
        } else {
            None
        };
        if let Some(seq) = sequence {
            return Ok(seq);
        }
        let target_1q_basis_list = vec![self.euler_basis];
        let euler_decompositions: SmallVec<[Option<OneQubitGateSequence>; 8]> = decomposition
            .iter()
            .map(|decomp| {
                unitary_to_gate_sequence_inner(
                    decomp.view(),
                    &target_1q_basis_list,
                    0,
                    None,
                    true,
                    None,
                )
            })
            .collect();
        // Worst case length is 5x 1q gates for each 1q decomposition + 1x 2q gate
        // We might overallocate a bit if the euler basis is different but
        // the worst case is just 16 extra elements with just a String and 2 smallvecs
        // each. This is only transient though as the circuit sequences aren't long lived
        // and are just used to create a QuantumCircuit or DAGCircuit when we return to
        // Python space.
        let mut gates = Vec::with_capacity(21);
        let mut global_phase = target_decomposed.global_phase;
        global_phase -= best_nbasis as f64 * self.basis_decomposer.global_phase;
        if best_nbasis == 2 {
            global_phase += PI;
        }
        for i in 0..best_nbasis as usize {
            if let Some(euler_decomp) = &euler_decompositions[2 * i] {
                for gate in &euler_decomp.gates {
                    gates.push((gate.0.clone(), gate.1.clone(), smallvec![0]));
                }
                global_phase += euler_decomp.global_phase
            }
            if let Some(euler_decomp) = &euler_decompositions[2 * i + 1] {
                for gate in &euler_decomp.gates {
                    gates.push((gate.0.clone(), gate.1.clone(), smallvec![1]));
                }
                global_phase += euler_decomp.global_phase
            }
            gates.push((self.gate.clone(), smallvec![], smallvec![0, 1]));
        }
        if let Some(euler_decomp) = &euler_decompositions[2 * best_nbasis as usize] {
            for gate in &euler_decomp.gates {
                gates.push((gate.0.clone(), gate.1.clone(), smallvec![0]));
            }
            global_phase += euler_decomp.global_phase
        }
        if let Some(euler_decomp) = &euler_decompositions[2 * best_nbasis as usize + 1] {
            for gate in &euler_decomp.gates {
                gates.push((gate.0.clone(), gate.1.clone(), smallvec![1]));
            }
            global_phase += euler_decomp.global_phase
        }
        Ok(TwoQubitGateSequence {
            gates,
            global_phase,
        })
    }

    fn num_basis_gates(&self, unitary: PyReadonlyArray2<Complex64>) -> usize {
        _num_basis_gates(self.basis_decomposer.b, self.basis_fidelity, unitary)
    }
}

#[pymodule]
pub fn two_qubit_decompose(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(_num_basis_gates))?;
    m.add_class::<TwoQubitGateSequence>()?;
    m.add_class::<TwoQubitWeylDecomposition>()?;
    m.add_class::<Specialization>()?;
    m.add_class::<TwoQubitBasisDecomposer>()?;
    Ok(())
}

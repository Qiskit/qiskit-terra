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

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4, FRAC_PI_8};

use nalgebra::{Quaternion, UnitQuaternion};
use ndarray::ArrayView2;
use num_complex::Complex64;
use thiserror::Error;

use qiskit_circuit::operations::{Param, StandardGate};

const COS_PI_8: f64 = 0.9238795325112867;
const SIN_PI_8: f64 = 0.3826834323650898;

#[derive(Error, Debug)]
pub enum VersorGateError {
    #[error("cannot act on gates with symbolic parameters")]
    Symbolic,
    #[error("multi-qubit gates have no versor representation")]
    MultiQubit,
    #[error("non-unitary gates have no versor representation")]
    NonUnitary,
}

/// A versor-based (unit quaternion) representation of a single-qubit gate.
///
/// In general, a single-qubit gate is a member of the group $U(2)$, and the group $SU(2)$ is
/// isomoprhic to the versors.  We can keep track of the complex phase separately to the rotation
/// action, to fully describe a member of $U(2)$.
///
/// The convention we use internally here is to associate the versor basis to the Pauli matrices as:
///
///     1 => I,   (where `nalgebra` calls the scalar term `w`)
///     i => i Z,
///     j => i Y,
///     k => i X,
///
/// so, for example, the Pauli Z gate has a possible (phase, versor) representation
/// `(pi/2, [0, -1, 0, 0])`.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct VersorGate {
    pub phase: f64,
    pub action: UnitQuaternion<f64>,
}

impl VersorGate {
    /// Get the identity operator in quaternion form.
    #[inline]
    pub fn identity() -> Self {
        Self {
            phase: 0.,
            action: UnitQuaternion::identity(),
        }
    }

    /// Set the phase of the gate.
    #[inline]
    pub fn with_phase(self, phase: f64) -> Self {
        Self {
            phase,
            action: self.action,
        }
    }

    /// Get the representation of an `RZ(angle)` gate.
    ///
    /// This is guaranteed to be phaseless given the conventions of the surjection from U(2).
    /// Internally this calculates `sin` and `cos`, so discrete-angle forms will be more efficient
    /// to be written explicitly.
    #[inline]
    fn from_rz(angle: f64) -> Self {
        let (sin, cos) = (angle * 0.5).sin_cos();
        Self {
            phase: 0.,
            action: UnitQuaternion::new_unchecked(Quaternion::new(cos, -sin, 0., 0.)),
        }
    }
    /// Get the representation of an `RY(angle)` gate.
    ///
    /// This is guaranteed to be phaseless given the conventions of the surjection from U(2).
    /// Internally this calculates `sin` and `cos`, so discrete-angle forms will be more efficient
    /// to be written explicitly.
    #[inline]
    fn from_ry(angle: f64) -> Self {
        let (sin, cos) = (angle * 0.5).sin_cos();
        Self {
            phase: 0.,
            action: UnitQuaternion::new_unchecked(Quaternion::new(cos, 0., -sin, 0.)),
        }
    }
    /// Get the representation of an `RX(angle)` gate.
    ///
    /// This is guaranteed to be phaseless given the conventions of the surjection from U(2).
    /// Internally this calculates `sin` and `cos`, so discrete-angle forms will be more efficient
    /// to be written explicitly.
    #[inline]
    fn from_rx(angle: f64) -> Self {
        let (sin, cos) = (angle * 0.5).sin_cos();
        Self {
            phase: 0.,
            action: UnitQuaternion::new_unchecked(Quaternion::new(cos, 0., 0., -sin)),
        }
    }

    /// Get the versor representation of a 1q [StandardGate] without constructing a matrix.
    ///
    /// Returns the error state if `gate` is not 1q, or if any of the parameters are symbolic.
    pub fn from_standard(gate: StandardGate, params: &[Param]) -> Result<Self, VersorGateError> {
        match gate {
            StandardGate::GlobalPhaseGate => {
                let &[Param::Float(phase)] = params else {
                    return Err(VersorGateError::Symbolic);
                };
                Ok(Self {
                    phase,
                    action: UnitQuaternion::identity(),
                })
            }
            StandardGate::HGate => {
                debug_assert!(params.is_empty());
                Ok(Self {
                    phase: FRAC_PI_2,
                    action: UnitQuaternion::new_unchecked(Quaternion::new(
                        0.,
                        -FRAC_1_SQRT_2,
                        0.,
                        -FRAC_1_SQRT_2,
                    )),
                })
            }
            StandardGate::IGate => {
                debug_assert!(params.is_empty());
                Ok(Self::identity())
            }
            StandardGate::XGate => {
                debug_assert!(params.is_empty());
                Ok(Self {
                    phase: FRAC_PI_2,
                    action: UnitQuaternion::new_unchecked(Quaternion::new(0., 0., 0., -1.)),
                })
            }
            StandardGate::YGate => {
                debug_assert!(params.is_empty());
                Ok(Self {
                    phase: FRAC_PI_2,
                    action: UnitQuaternion::new_unchecked(Quaternion::new(0., 0., -1., 0.)),
                })
            }
            StandardGate::ZGate => {
                debug_assert!(params.is_empty());
                Ok(Self {
                    phase: FRAC_PI_2,
                    action: UnitQuaternion::new_unchecked(Quaternion::new(0., -1., 0., 0.)),
                })
            }
            StandardGate::PhaseGate | StandardGate::U1Gate => {
                let &[Param::Float(angle)] = params else {
                    return Err(VersorGateError::Symbolic);
                };
                Ok(Self::from_rz(angle).with_phase(angle * 0.5))
            }
            StandardGate::RGate => {
                let &[Param::Float(angle), Param::Float(axis)] = params else {
                    return Err(VersorGateError::Symbolic);
                };
                let (sin_angle, cos_angle) = (angle * 0.5).sin_cos();
                let (sin_axis, cos_axis) = axis.sin_cos();
                Ok(Self {
                    phase: 0.,
                    action: UnitQuaternion::new_unchecked(Quaternion::new(
                        cos_angle,
                        0.,
                        -sin_axis * sin_angle,
                        -cos_axis * sin_angle,
                    )),
                })
            }
            StandardGate::RXGate => {
                let &[Param::Float(angle)] = params else {
                    return Err(VersorGateError::Symbolic);
                };
                Ok(Self::from_rx(angle))
            }
            StandardGate::RYGate => {
                let &[Param::Float(angle)] = params else {
                    return Err(VersorGateError::Symbolic);
                };
                Ok(Self::from_ry(angle))
            }
            StandardGate::RZGate => {
                let &[Param::Float(angle)] = params else {
                    return Err(VersorGateError::Symbolic);
                };
                Ok(Self::from_rz(angle))
            }
            StandardGate::SGate => {
                debug_assert!(params.is_empty());
                Ok(Self {
                    phase: FRAC_PI_4,
                    action: UnitQuaternion::new_unchecked(Quaternion::new(
                        FRAC_1_SQRT_2,
                        -FRAC_1_SQRT_2,
                        0.,
                        0.,
                    )),
                })
            }
            StandardGate::SdgGate => {
                debug_assert!(params.is_empty());
                Ok(Self {
                    phase: -FRAC_PI_4,
                    action: UnitQuaternion::new_unchecked(Quaternion::new(
                        FRAC_1_SQRT_2,
                        FRAC_1_SQRT_2,
                        0.,
                        0.,
                    )),
                })
            }
            StandardGate::SXGate => {
                debug_assert!(params.is_empty());
                Ok(Self {
                    phase: FRAC_PI_4,
                    action: UnitQuaternion::new_unchecked(Quaternion::new(
                        FRAC_1_SQRT_2,
                        0.,
                        0.,
                        -FRAC_1_SQRT_2,
                    )),
                })
            }
            StandardGate::SXdgGate => {
                debug_assert!(params.is_empty());
                Ok(Self {
                    phase: -FRAC_PI_4,
                    action: UnitQuaternion::new_unchecked(Quaternion::new(
                        FRAC_1_SQRT_2,
                        0.,
                        0.,
                        FRAC_1_SQRT_2,
                    )),
                })
            }
            StandardGate::TGate => {
                debug_assert!(params.is_empty());
                Ok(Self {
                    phase: FRAC_PI_8,
                    action: UnitQuaternion::new_unchecked(Quaternion::new(
                        COS_PI_8, -SIN_PI_8, 0., 0.,
                    )),
                })
            }
            StandardGate::TdgGate => {
                debug_assert!(params.is_empty());
                Ok(Self {
                    phase: -FRAC_PI_8,
                    action: UnitQuaternion::new_unchecked(Quaternion::new(
                        COS_PI_8, SIN_PI_8, 0., 0.,
                    )),
                })
            }
            StandardGate::UGate | StandardGate::U3Gate => {
                let &[Param::Float(theta), Param::Float(phi), Param::Float(lambda)] = params else {
                    return Err(VersorGateError::Symbolic);
                };
                Ok(
                    (Self::from_rz(phi) * Self::from_ry(theta) * Self::from_rz(lambda))
                        .with_phase((phi + lambda) * 0.5),
                )
            }
            StandardGate::U2Gate => {
                let &[Param::Float(phi), Param::Float(lambda)] = params else {
                    return Err(VersorGateError::Symbolic);
                };
                let (sin, cos) = (lambda * 0.5).sin_cos();
                // The RY(pi/2).RZ(lambda) part of the decomposition, including the complete
                // corrective phase term.
                let ry_rz = Self {
                    phase: (phi + lambda) * 0.5,
                    action: UnitQuaternion::new_unchecked(
                        FRAC_1_SQRT_2 * Quaternion::new(cos, -sin, -cos, -sin),
                    ),
                };
                Ok(Self::from_rz(phi) * ry_rz)
            }
            _ => Err(VersorGateError::MultiQubit),
        }
    }

    /// Calculate the versor representation of this matrix, assuming it is unitary.
    #[inline(always)]
    fn from_matrix_unchecked<M: Matrix1q>(matrix: &M) -> Self {
        // `matrix` is in U(2), therefore it has some representation
        //
        //   exp(ix) . [[a + bi, c + di], [-c + di, a - bi]],
        //
        // and the matrix is in SU(2) with determinant 1.  We can find the phase angle `x` via the
        // determinant of the given, which we've been promised has magnitude 1, so we can phase it
        // out of the matrix terms by multiplying by the root of the conjugate (since `det(aA) =
        // a^dim(A) det(A)`).  The versor representation is then just `(x/2, [a, b, c, d])`.
        let det = matrix.get(0, 0) * matrix.get(1, 1) - matrix.get(0, 1) * matrix.get(1, 0);
        let inv_rot = det.conj().sqrt();
        Self {
            phase: 0.5 * det.arg(),
            action: UnitQuaternion::new_unchecked(Quaternion::new(
                (inv_rot * matrix.get(0, 0)).re,
                (inv_rot * matrix.get(0, 0)).im,
                (inv_rot * matrix.get(0, 1)).re,
                (inv_rot * matrix.get(0, 1)).im,
            )),
        }
    }

    /// Calculate the versor representation of a unitary matrix, assuming it is unitary.
    pub fn from_contiguous_unchecked(matrix: &[[Complex64; 2]; 2]) -> Self {
        Self::from_matrix_unchecked(matrix)
    }

    /// Calculate the versor representation of a unitary matrix, assuming it is unitary.
    pub fn from_ndarray_unchecked(matrix: &ArrayView2<Complex64>) -> Self {
        Self::from_matrix_unchecked(matrix)
    }

    /// Calculate the versor representation of a unitary matrix.
    ///
    /// Returns the error state if `|| A+.A - 1 ||_2 > tol` for matrix `A`, where the norm is the
    /// Frobenius norm.
    fn from_matrix_with_tol<M: Matrix1q>(matrix: &M, tol: f64) -> Result<Self, VersorGateError> {
        if unitary_frobenius_distance_square(matrix) > tol * tol {
            return Err(VersorGateError::NonUnitary);
        }
        Ok(Self::from_matrix_unchecked(matrix))
    }

    /// Calculate the versor representation of a unitary matrix.
    ///
    /// Returns the error state if `|| A+.A - 1 ||_2 > tol` for matrix `A`, where the norm is the
    /// Frobenius norm.
    pub fn from_contiguous(
        matrix: &[[Complex64; 2]; 2],
        tol: f64,
    ) -> Result<Self, VersorGateError> {
        Self::from_matrix_with_tol(matrix, tol)
    }

    /// Calculate the versor representation of a unitary matrix.
    ///
    /// Returns the error state if `|| A+.A - 1 ||_2 > tol` for matrix `A`, where the norm is the
    /// Frobenius norm.
    pub fn from_ndarray(matrix: &ArrayView2<Complex64>, tol: f64) -> Result<Self, VersorGateError> {
        Self::from_matrix_with_tol(matrix, tol)
    }

    /// Fill a pre-allocated 2x2 Rust-native contiguous array with the Z-basis representation of
    /// this versor.
    ///
    /// The inverse of this function is [from_contiguous].
    #[inline]
    pub fn matrix_contiguous_into(&self, matrix: &mut [[Complex64; 2]; 2]) {
        let phase = Complex64::from_polar(1., self.phase);
        let q = self.action.quaternion();
        matrix[0][0] = phase * Complex64::new(q.w, q.i);
        matrix[0][1] = phase * Complex64::new(q.j, q.k);
        matrix[1][0] = phase * Complex64::new(-q.j, q.k);
        matrix[1][1] = phase * Complex64::new(q.w, -q.i);
    }

    /// Create a new Z-basis representation of this versor as a contiguous Rust-native array.
    ///
    /// The inverse of this function is [from_contiguous].
    #[inline]
    pub fn matrix_contiguous(&self) -> [[Complex64; 2]; 2] {
        let mut out = Default::default();
        self.matrix_contiguous_into(&mut out);
        out
    }
}

impl ::std::ops::Mul for VersorGate {
    type Output = VersorGate;
    fn mul(self, other: VersorGate) -> Self::Output {
        VersorGate {
            phase: self.phase + other.phase,
            action: self.action * other.action,
        }
    }
}
impl ::std::ops::Mul<&VersorGate> for VersorGate {
    type Output = VersorGate;
    fn mul(self, other: &VersorGate) -> VersorGate {
        self * *other
    }
}
impl ::std::ops::Mul<VersorGate> for &VersorGate {
    type Output = VersorGate;
    fn mul(self, other: VersorGate) -> VersorGate {
        *self * other
    }
}
impl ::std::ops::Mul for &VersorGate {
    type Output = VersorGate;
    fn mul(self, other: &VersorGate) -> VersorGate {
        *self * *other
    }
}

/// A module-internal trait to simplify the code-generation of both the dynamic `ndarray` and the
/// static `&[[Complex64; 2]; 2]` and dynamic `ndarray` paths.  Rather than making the user care
/// about importing it, we just expose the concretised methods using it through `VersorGate`.
trait Matrix1q {
    fn get(&self, row: usize, col: usize) -> Complex64;
}
impl<const N: usize, const M: usize> Matrix1q for [[Complex64; N]; M] {
    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> Complex64 {
        self[row][col]
    }
}
impl Matrix1q for ArrayView2<'_, Complex64> {
    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> Complex64 {
        self[(row, col)]
    }
}

/// Calculate the squared Frobenius distance `|| A+ A - 1 ||_2^2`.
///
/// As a rule of thumb, if `A+ A = 1 + eps`, then the Frobenius distance from the identity will be
/// `sqrt(2)*eps`, and this function would return `2 * eps*eps`, up to round-off error.
fn unitary_frobenius_distance_square<M: Matrix1q>(matrix: &M) -> f64 {
    let topleft = matrix.get(0, 0).norm_sqr() + matrix.get(0, 1).norm_sqr();
    let botright = matrix.get(1, 0).norm_sqr() + matrix.get(1, 1).norm_sqr();
    let off =
        matrix.get(0, 0) * matrix.get(1, 0).conj() + matrix.get(0, 1) * matrix.get(1, 1).conj();
    (topleft - 1.).powi(2) + (botright - 1.).powi(2) + 2. * off.norm_sqr()
}

#[cfg(test)]
mod test {
    use super::*;

    use approx::AbsDiffEq;
    use ndarray::aview2;
    use qiskit_circuit::operations::{Operation, Param, StandardGate, STANDARD_GATE_SIZE};

    fn all_1q_gates() -> Vec<StandardGate> {
        (0..STANDARD_GATE_SIZE as u8)
            .filter_map(|x| {
                ::bytemuck::checked::try_cast::<_, StandardGate>(x)
                    .ok()
                    .filter(|gate| gate.num_qubits() == 1)
            })
            .collect()
    }

    #[test]
    fn each_1q_gate_has_correct_matrix() {
        let params = [0.25, -0.75, 1.25, 0.5].map(Param::Float);
        let mut fails = Vec::new();
        for gate in all_1q_gates() {
            let params = &params[0..gate.num_params() as usize];
            let direct_matrix = gate.matrix(params).unwrap();
            let versor_matrix = VersorGate::from_standard(gate, params)
                .unwrap()
                .matrix_contiguous();
            if direct_matrix.abs_diff_ne(&aview2(&versor_matrix), 1e-15) {
                fails.push((gate, direct_matrix, versor_matrix));
            }
        }
        assert_eq!(fails, [])
    }

    #[test]
    fn can_roundtrip_1q_gate_from_matrix() {
        let params = [0.25, -0.75, 1.25, 0.5].map(Param::Float);
        let mut fails = Vec::new();
        for gate in all_1q_gates() {
            let params = &params[0..gate.num_params() as usize];
            let direct_matrix = gate.matrix(params).unwrap();
            let versor_matrix = VersorGate::from_ndarray(&direct_matrix.view(), 1e-15)
                .unwrap()
                .matrix_contiguous();
            if direct_matrix.abs_diff_ne(&aview2(&versor_matrix), 1e-15) {
                fails.push((gate, direct_matrix, versor_matrix));
            }
        }
        assert_eq!(fails, [])
    }

    #[test]
    fn pairwise_multiplication_gives_correct_matrices() {
        // We have two pairs just so in the (x, x) case of iteration we're including two different
        // gates to make sure that any non-commutation is accounted for.
        let left_params = [0.25, -0.75, 1.25, 0.5].map(Param::Float);
        let right_params = [0.5, 1.25, -0.75, 0.25].map(Param::Float);

        let mut fails = Vec::new();
        for (left, right) in all_1q_gates().into_iter().zip(all_1q_gates()) {
            let left_params = &left_params[0..left.num_params() as usize];
            let right_params = &right_params[0..right.num_params() as usize];

            let direct_matrix = left
                .matrix(left_params)
                .unwrap()
                .dot(&right.matrix(right_params).unwrap());
            let versor_matrix = (VersorGate::from_standard(left, left_params).unwrap()
                * VersorGate::from_standard(right, right_params).unwrap())
            .matrix_contiguous();
            if direct_matrix.abs_diff_ne(&aview2(&versor_matrix), 1e-15) {
                fails.push((left, right, direct_matrix, versor_matrix));
            }
        }
        assert_eq!(fails, [])
    }
}

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

use crate::synthesis::linear::utils::calc_inverse_matrix_inner;
use ndarray::{azip, s, Array1, Array2, ArrayView2};
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::Qubit;
use smallvec::{smallvec, SmallVec};

/// Symplectic matrix.
/// Currently this class is internal to the synthesis library.
pub struct SymplecticMatrix {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits).
    pub smat: Array2<bool>,
}

/// Clifford.
/// Currently this class is internal to the synthesis library and
/// has a very different functionality from Qiskit's python-based
/// Clifford class.
#[derive(Clone)]
pub struct Clifford {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits + 1).
    pub tableau: Array2<bool>,
}

impl SymplecticMatrix {
    /// Modifies the matrix in-place by appending S-gate
    #[allow(dead_code)]
    pub fn append_s(&mut self, qubit: usize) {
        let (x, mut z) = self
            .smat
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));
        azip!((z in &mut z, &x in &x) *z ^= x);
    }

    /// Modifies the matrix in-place by prepending S-gate
    pub fn prepend_s(&mut self, qubit: usize) {
        let (x, mut z) = self
            .smat
            .multi_slice_mut((s![self.num_qubits + qubit, ..], s![qubit, ..]));
        azip!((z in &mut z, &x in &x) *z ^= x);
    }

    /// Modifies the matrix in-place by appending H-gate
    #[allow(dead_code)]
    pub fn append_h(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .smat
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));
        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x));
    }

    /// Modifies the matrix in-place by prepending H-gate
    pub fn prepend_h(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .smat
            .multi_slice_mut((s![qubit, ..], s![self.num_qubits + qubit, ..]));
        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x));
    }

    /// Modifies the matrix in-place by appending SWAP-gate
    #[allow(dead_code)]
    pub fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        let (mut x0, mut z0, mut x1, mut z1) = self.smat.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
        ));
        azip!((x0 in &mut x0, x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
        azip!((z0 in &mut z0, z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
    }

    /// Modifies the matrix in-place by prepending SWAP-gate
    pub fn prepend_swap(&mut self, qubit0: usize, qubit1: usize) {
        let (mut x0, mut z0, mut x1, mut z1) = self.smat.multi_slice_mut((
            s![qubit0, ..],
            s![self.num_qubits + qubit0, ..],
            s![qubit1, ..],
            s![self.num_qubits + qubit1, ..],
        ));
        azip!((x0 in &mut x0, x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
        azip!((z0 in &mut z0, z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
    }

    /// Modifies the matrix in-place by appending CX-gate
    #[allow(dead_code)]
    pub fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        let (x0, mut z0, mut x1, z1) = self.smat.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
        ));
        azip!((x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
        azip!((z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
    }

    /// Modifies the matrix in-place by prepending CX-gate
    pub fn prepend_cx(&mut self, qubit0: usize, qubit1: usize) {
        let (x0, mut z0, mut x1, z1) = self.smat.multi_slice_mut((
            s![qubit1, ..],
            s![self.num_qubits + qubit1, ..],
            s![qubit0, ..],
            s![self.num_qubits + qubit0, ..],
        ));
        azip!((x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
        azip!((z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
    }
}

impl Clifford {
    /// Modifies the tableau in-place by appending S-gate
    pub fn append_s(&mut self, qubit: usize) {
        let (x, mut z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
        azip!((z in &mut z, &x in &x) *z ^= x);
    }

    /// Modifies the tableau in-place by appending Sdg-gate
    #[allow(dead_code)]
    pub fn append_sdg(&mut self, qubit: usize) {
        let (x, mut z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & !z);
        azip!((z in &mut z, &x in &x) *z ^= x);
    }

    /// Modifies the tableau in-place by appending SX-gate
    pub fn append_sx(&mut self, qubit: usize) {
        let (mut x, z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= !x & z);
        azip!((&z in &z, x in &mut x) *x ^= z);
    }

    /// Modifies the tableau in-place by appending SXDG-gate
    pub fn append_sxdg(&mut self, qubit: usize) {
        let (mut x, z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
        azip!((&z in &z, x in &mut x) *x ^= z);
    }

    /// Modifies the tableau in-place by appending H-gate
    pub fn append_h(&mut self, qubit: usize) {
        let (mut x, mut z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x));
    }

    /// Modifies the tableau in-place by appending SWAP-gate
    pub fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        let (mut x0, mut z0, mut x1, mut z1) = self.tableau.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
        ));
        azip!((x0 in &mut x0, x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
        azip!((z0 in &mut z0, z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
    }

    /// Modifies the tableau in-place by appending CX-gate
    pub fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        let (x0, mut z0, mut x1, z1, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
            s![.., 2 * self.num_qubits],
        ));
        azip!((p in &mut p, &x0 in &x0, &z0 in &z0, &x1 in &x1, &z1 in &z1) *p ^= (x1 ^ z0 ^ true) & z1 & x0);
        azip!((x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
        azip!((z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
    }

    /// Modifies the tableau in-place by appending W-gate.
    /// This is equivalent to an Sdg gate followed by an H gate.
    pub fn append_v(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .tableau
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));

        azip!((x in &mut x, z in &mut z) (*x, *z) = (*x ^ *z, *x));
    }

    /// Modifies the tableau in-place by appending V-gate.
    /// This is equivalent to two V gates.
    pub fn append_w(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .tableau
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));

        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x ^ *z));
    }

    /// Creates a Clifford from a given sequence of Clifford gates.
    /// In essence, starts from the identity tableau and modifies it
    /// based on the gates in the sequence.
    pub fn from_gate_sequence(
        gate_seq: &CliffordGatesVec,
        num_qubits: usize,
    ) -> Result<Clifford, String> {
        // create the identity
        let mut clifford = Clifford {
            num_qubits,
            tableau: Array2::from_shape_fn((2 * num_qubits, 2 * num_qubits + 1), |(i, j)| i == j),
        };

        gate_seq
            .iter()
            .try_for_each(|(gate, _params, qubits)| match *gate {
                StandardGate::SGate => {
                    clifford.append_s(qubits[0].index());
                    Ok(())
                }
                StandardGate::SdgGate => {
                    clifford.append_sdg(qubits[0].0 as usize);
                    Ok(())
                }
                StandardGate::SXGate => {
                    clifford.append_sx(qubits[0].0 as usize);
                    Ok(())
                }
                StandardGate::SXdgGate => {
                    clifford.append_sxdg(qubits[0].0 as usize);
                    Ok(())
                }
                StandardGate::HGate => {
                    clifford.append_h(qubits[0].index());
                    Ok(())
                }
                StandardGate::CXGate => {
                    clifford.append_cx(qubits[0].index(), qubits[1].index());
                    Ok(())
                }
                StandardGate::SwapGate => {
                    clifford.append_swap(qubits[0].index(), qubits[1].index());
                    Ok(())
                }
                _ => Err(format!("Unsupported gate {:?}", gate)),
            })?;
        Ok(clifford)
    }
}

/// A sequence of Clifford gates.
/// Represents the return type of Clifford synthesis algorithms.
pub type CliffordGatesVec = Vec<(StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>)>;

/// Given a sequence of Clifford gates that correctly implements the symplectic matrix
/// of the target clifford tableau, adds the Pauli gates to also match the phase of
/// the tableau.
pub fn adjust_final_pauli_gates(
    gate_seq: &mut CliffordGatesVec,
    target_tableau: ArrayView2<bool>,
    num_qubits: usize,
) -> Result<(), String> {
    // simulate the clifford circuit that we have constructed
    let simulated_clifford = Clifford::from_gate_sequence(gate_seq, num_qubits)?;

    // compute the phase difference
    let target_phase = target_tableau.column(2 * num_qubits);
    let sim_phase = simulated_clifford.tableau.column(2 * num_qubits);

    let delta_phase: Vec<bool> = target_phase
        .iter()
        .zip(sim_phase.iter())
        .map(|(&a, &b)| a ^ b)
        .collect();

    // compute inverse of the symplectic matrix
    let smat = target_tableau.slice(s![.., ..2 * num_qubits]);
    let smat_inv = calc_inverse_matrix_inner(smat, false)?;

    // compute smat_inv * delta_phase
    let arr1 = smat_inv.map(|v| *v as usize);
    let vec2: Vec<usize> = delta_phase.into_iter().map(|v| v as usize).collect();
    let arr2 = Array1::from(vec2);
    let delta_phase_pre = arr1.dot(&arr2).map(|v| v % 2 == 1);

    // add pauli gates
    for qubit in 0..num_qubits {
        if delta_phase_pre[qubit] && delta_phase_pre[qubit + num_qubits] {
            gate_seq.push((
                StandardGate::YGate,
                smallvec![],
                smallvec![Qubit::new(qubit)],
            ));
        } else if delta_phase_pre[qubit] {
            gate_seq.push((
                StandardGate::ZGate,
                smallvec![],
                smallvec![Qubit::new(qubit)],
            ));
        } else if delta_phase_pre[qubit + num_qubits] {
            gate_seq.push((
                StandardGate::XGate,
                smallvec![],
                smallvec![Qubit::new(qubit)],
            ));
        }
    }

    Ok(())
}

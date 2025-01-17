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

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::getenv_use_multiple_threads;
use faer_ext::{IntoFaerComplex, IntoNdarrayComplex};
use ndarray::prelude::*;
use num_complex::Complex64;
use numpy::IntoPyArray;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::imports::UNITARY_GATE;
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::PyGate;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use smallvec::{smallvec, SmallVec};

type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

#[inline(always)]
fn random_complex(rng: &mut Pcg64Mcg) -> Complex64 {
    Complex64::new(rng.sample(StandardNormal), rng.sample(StandardNormal))
        * std::f64::consts::FRAC_1_SQRT_2
}

// This function's implementation was modeled off of the algorithm used in the
// `scipy.stats.unitary_group.rvs()` function defined here:
//
// https://github.com/scipy/scipy/blob/v1.14.1/scipy/stats/_multivariate.py#L4224-L4256
#[inline]
fn random_unitaries(seed: u64, size: usize) -> impl Iterator<Item = Array2<Complex64>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);

    (0..size).map(move |_| {
        let raw_numbers: [[Complex64; 4]; 4] = [
            [
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
            ],
            [
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
            ],
            [
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
            ],
            [
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
                random_complex(&mut rng),
            ],
        ];

        let qr = aview2(&raw_numbers).into_faer_complex().qr();
        let r = qr.compute_r();
        let diag: [Complex64; 4] = [
            r[(0, 0)].to_num_complex() / r[(0, 0)].abs(),
            r[(1, 1)].to_num_complex() / r[(1, 1)].abs(),
            r[(2, 2)].to_num_complex() / r[(2, 2)].abs(),
            r[(3, 3)].to_num_complex() / r[(3, 3)].abs(),
        ];
        let mut q = qr.compute_q().as_ref().into_ndarray_complex().to_owned();
        q.axis_iter_mut(Axis(0)).for_each(|mut row| {
            row.iter_mut()
                .enumerate()
                .for_each(|(index, val)| *val *= diag[index])
        });
        q
    })
}

const UNITARY_PER_SEED: usize = 50;

#[pyfunction]
#[pyo3(signature=(num_qubits, depth, seed=None))]
pub fn quantum_volume(
    py: Python,
    num_qubits: u32,
    depth: usize,
    seed: Option<u64>,
) -> PyResult<CircuitData> {
    let width = num_qubits as usize / 2;
    let num_unitaries = width * depth;
    let mut permutation: Vec<Qubit> = (0..num_qubits).map(Qubit).collect();

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "num_qubits"), 2)?;
    let mut build_instruction = |(unitary_index, unitary_array): (usize, Array2<Complex64>),
                                 rng: &mut Pcg64Mcg|
     -> PyResult<Instruction> {
        let layer_index = unitary_index % width;
        if layer_index == 0 {
            permutation.shuffle(rng);
        }
        let unitary = unitary_array.into_pyarray(py);

        let unitary_gate = UNITARY_GATE
            .get_bound(py)
            .call((unitary.clone(), py.None(), false), Some(&kwargs))?;
        let instruction = PyGate {
            qubits: 2,
            clbits: 0,
            params: 1,
            op_name: "unitary".to_string(),
            gate: unitary_gate.unbind(),
        };
        let qubit = layer_index * 2;
        Ok((
            PackedOperation::from_gate(Box::new(instruction)),
            smallvec![Param::Obj(unitary.into_any().unbind())],
            vec![permutation[qubit], permutation[qubit + 1]],
            vec![],
        ))
    };

    let mut per_thread = num_unitaries / UNITARY_PER_SEED;
    if per_thread == 0 {
        per_thread = 10;
    }
    let mut outer_rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::from_entropy(),
    };
    let seed_vec: Vec<u64> = rand::distributions::Standard
        .sample_iter(&mut outer_rng)
        .take(num_unitaries)
        .collect();

    let unitaries: Vec<Array2<Complex64>> = if getenv_use_multiple_threads() && num_unitaries > 200
    {
        seed_vec
            .par_chunks(per_thread)
            .flat_map_iter(|seeds| random_unitaries(seeds[0], seeds.len()))
            .collect()
    } else {
        seed_vec
            .chunks(per_thread)
            .flat_map(|seeds| random_unitaries(seeds[0], seeds.len()))
            .collect()
    };
    CircuitData::from_packed_operations(
        py,
        num_qubits,
        0,
        unitaries
            .into_iter()
            .enumerate()
            .map(|x| build_instruction(x, &mut outer_rng)),
        Param::Float(0.),
    )
}

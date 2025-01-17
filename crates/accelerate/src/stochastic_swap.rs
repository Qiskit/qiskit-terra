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

// Needed to pass shared state between functions
// closures don't work because of recurssion
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::sync::RwLock;

use hashbrown::HashSet;

use ndarray::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg64Mcg;

use crate::edge_collections::EdgeCollection;
use crate::getenv_use_multiple_threads;
use crate::nlayout::{NLayout, PhysicalQubit, VirtualQubit};

#[inline]
fn compute_cost(
    dist: &ArrayView2<f64>,
    layout: &NLayout,
    gates: &[VirtualQubit],
    num_gates: usize,
) -> f64 {
    (0..num_gates)
        .map(|gate| {
            dist[[
                gates[2 * gate].to_phys(layout).index(),
                gates[2 * gate + 1].to_phys(layout).index(),
            ]]
        })
        .sum()
}

/// Computes the symmetric random scaling (perturbation) matrix,
/// and places the values in the 'scale' array.
///
/// Args:
///     scale (ndarray): An array of doubles where the values are to be stored.
///     cdist2 (ndarray): Array representing the coupling map distance squared.
///     rand (double *): Array of rands of length num_qubits*(num_qubits+1)//2.
///     num_qubits (int): Number of physical qubits.
#[inline]
fn compute_random_scaling(
    scale: &mut Array2<f64>,
    cdist2: &ArrayView2<f64>,
    rand: &[f64],
    num_qubits: usize,
) {
    let mut idx: usize = 0;
    for ii in 0..num_qubits {
        for jj in 0..ii {
            scale[[ii, jj]] = rand[idx] * cdist2[[ii, jj]];
            scale[[jj, ii]] = scale[[ii, jj]];
            idx += 1
        }
    }
}

fn swap_trial(
    num_qubits: usize,
    int_layout: &NLayout,
    int_qubit_subset: &[VirtualQubit],
    gates: &[VirtualQubit],
    cdist: ArrayView2<f64>,
    cdist2: ArrayView2<f64>,
    edges: &[PhysicalQubit],
    seed: u64,
    trial_num: u64,
    locked_best_possible: Option<&RwLock<&mut Option<(u64, f64, EdgeCollection, NLayout)>>>,
) -> Option<(f64, EdgeCollection, NLayout, usize)> {
    if let Some(locked_best_possible) = locked_best_possible {
        // Return fast if a depth == 1 solution was already found in another parallel
        // trial. However for deterministic results in cases of multiple depth == 1
        // solutions still search for a solution if this trial number is less than
        // the found solution (this mirrors the previous behavior of a serial loop).
        let best_possible = locked_best_possible.read().unwrap();
        if best_possible.is_some() && best_possible.as_ref().unwrap().0 < trial_num {
            return None;
        }
    }
    let mut opt_edges = EdgeCollection::new();
    let mut trial_layout = int_layout.clone();
    let mut optimal_layout = int_layout.clone();

    let num_gates: usize = gates.len() / 2;
    let num_edges: usize = edges.len() / 2;

    let mut cost_reduced;
    let mut depth_step: usize = 1;
    let depth_max: usize = 2 * num_qubits + 1;
    let mut min_cost: f64;
    let mut new_cost: f64;
    let mut dist: f64;

    let mut optimal_start = PhysicalQubit::new(u32::MAX);
    let mut optimal_end = PhysicalQubit::new(u32::MAX);
    let mut optimal_start_qubit = VirtualQubit::new(u32::MAX);
    let mut optimal_end_qubit = VirtualQubit::new(u32::MAX);

    let mut scale = Array2::zeros((num_qubits, num_qubits));

    let distribution = Normal::new(1.0, 1.0 / num_qubits as f64).unwrap();
    let mut rng: Pcg64Mcg = Pcg64Mcg::seed_from_u64(seed);
    let rand_arr: Vec<f64> = distribution
        .sample_iter(&mut rng)
        .take(num_qubits * (num_qubits + 1) / 2)
        .collect();

    compute_random_scaling(&mut scale, &cdist2, &rand_arr, num_qubits);

    let input_qubit_set = int_qubit_subset.iter().copied().collect::<HashSet<_>>();

    while depth_step < depth_max {
        let mut qubit_set = input_qubit_set.clone();
        while !qubit_set.is_empty() {
            min_cost = compute_cost(&scale.view(), &trial_layout, gates, num_gates);
            // Try to decrease the objective function
            cost_reduced = false;
            for idx in 0..num_edges {
                let start_edge = edges[2 * idx];
                let end_edge = edges[2 * idx + 1];
                let start_qubit = start_edge.to_virt(&trial_layout);
                let end_qubit = end_edge.to_virt(&trial_layout);
                if qubit_set.contains(&start_qubit) && qubit_set.contains(&end_qubit) {
                    // Try this edge to reduce cost
                    trial_layout.swap_physical(start_edge, end_edge);
                    // compute objective function
                    new_cost = compute_cost(&scale.view(), &trial_layout, gates, num_gates);
                    // record progress if we succeed
                    if new_cost < min_cost {
                        cost_reduced = true;
                        min_cost = new_cost;
                        optimal_layout = trial_layout.clone();
                        optimal_start = start_edge;
                        optimal_end = end_edge;
                        optimal_start_qubit = start_qubit;
                        optimal_end_qubit = end_qubit;
                    }
                    trial_layout.swap_physical(start_edge, end_edge);
                }
            }
            // After going over all edges
            // Were there any good swap choices?
            if cost_reduced {
                qubit_set.remove(&optimal_start_qubit);
                qubit_set.remove(&optimal_end_qubit);
                trial_layout = optimal_layout.clone();
                opt_edges.add(optimal_start, optimal_end);
            } else {
                break;
            }
        }
        // We have either run out of swap pairs to try or failed to improve
        // the cost

        // Compute the coupling graph distance
        dist = compute_cost(&cdist, &trial_layout, gates, num_gates);
        // If all gates can be applied now we're finished.
        // Otherwise we need to consider a deeper swap circuit
        if dist as usize == num_gates {
            break;
        }
        // increment the depth
        depth_step += 1;
    }
    // Either we have succeeded at some depth d < d_max or failed
    dist = compute_cost(&cdist, &trial_layout, gates, num_gates);
    if let Some(locked_best_possible) = locked_best_possible {
        if dist as usize == num_gates && depth_step == 1 {
            let mut best_possible = locked_best_possible.write().unwrap();
            // In the case an ideal solution has already been found to preserve
            // behavior consistent with the single threaded predecessor to this function
            // we defer to the earlier trial
            if best_possible.is_none() || best_possible.as_ref().unwrap().0 > trial_num {
                **best_possible = Some((trial_num, dist, opt_edges, trial_layout));
            }
            return None;
        }
    }
    Some((dist, opt_edges, trial_layout, depth_step))
}

/// Run the random trials as part of the layer permutation used internally for
/// the stochastic swap algorithm.
///
/// This function is multithreaded and will spawn a thread pool as part of its
/// execution. By default the number of threads will be equal to the number of
/// CPUs. You can tune the number of threads with the RAYON_NUM_THREADS
/// environment variable. For example, setting RAYON_NUM_THREADS=4 would limit
/// the thread pool to 4 threads.
///
/// Args:
///     num_trials (int): The number of random trials to attempt
///     num_qubits (int): The number of qubits
///     int_layout (NLayout): The initial layout for the layer. The layout is a mapping
///         of virtual qubits to physical qubits in the coupling graph
///     int_qubit_subset (ndarray): A 1D array of qubit indices for the set of qubits in the
///         coupling map that we've chosen to map into.
///     int_gates (ndarray): A 1D array of qubit pairs that each 2 qubit gate operates on.
///         The pairs are flattened on the array so that each pair in the list of 2q gates
///         are adjacent in the array. For example, if the 2q interaction list was
///         ``[(0, 1), (2, 1), (3, 2)]``, the input here would be ``[0, 1, 2, 1, 3, 2]``.
///     cdist (ndarray): The distance matrix for the coupling graph of the target
///         backend
///     cdist2 (ndarray): The distance matrix squared for the coupling graph of the
///         target backend
///     edges (ndarray): A flattened 1d array of the edge list of the coupling graph.
///         The pairs are flattened on the array so that each node pair in the edge are
///         adjacent in the array. For example, if the edge list were ``[(0, 1), (1, 2), (2, 3)]``
///         the input array here would be ``[0, 1, 1, 2, 2, 3]``.
///     seed (int): An optional seed for the rng used to generate the random perturbation
///         matrix used in each trial
/// Returns:
///     tuple: If a valid layout permutation is found a tuple of the form:
///         ``(edges, layout, depth)`` is returned. If a solution is not found the output
///         will be ``(None, None, max int)``.
#[pyfunction]
#[pyo3(
    signature = (num_trials, num_qubits, int_layout, int_qubit_subset, int_gates, cdist, cdist2, edges, seed=None)
)]
pub fn swap_trials(
    num_trials: u64,
    num_qubits: usize,
    int_layout: &NLayout,
    int_qubit_subset: PyReadonlyArray1<VirtualQubit>,
    int_gates: PyReadonlyArray1<VirtualQubit>,
    cdist: PyReadonlyArray2<f64>,
    cdist2: PyReadonlyArray2<f64>,
    edges: PyReadonlyArray1<PhysicalQubit>,
    seed: Option<u64>,
) -> PyResult<(Option<EdgeCollection>, Option<NLayout>, usize)> {
    let int_qubit_subset_arr = int_qubit_subset.as_slice()?;
    let int_gates_arr = int_gates.as_slice()?;
    let cdist_arr = cdist.as_array();
    let cdist2_arr = cdist2.as_array();
    let edges_arr = edges.as_slice()?;
    let num_gates: usize = int_gates.len()? / 2;
    let mut best_possible: Option<(u64, f64, EdgeCollection, NLayout)> = None;
    let locked_best_possible: RwLock<&mut Option<(u64, f64, EdgeCollection, NLayout)>> =
        RwLock::new(&mut best_possible);
    let outer_rng: Pcg64Mcg = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::from_entropy(),
    };
    let seed_vec: Vec<u64> = outer_rng
        .sample_iter(&rand::distributions::Standard)
        .take(num_trials as usize)
        .collect();
    // Run in parallel only if we're not already in a multiprocessing context
    // unless force threads is set.
    let run_in_parallel = getenv_use_multiple_threads();

    let mut best_depth = usize::MAX;
    let mut best_edges: Option<EdgeCollection> = None;
    let mut best_layout: Option<NLayout> = None;
    if run_in_parallel {
        let result: Vec<Option<(f64, EdgeCollection, NLayout, usize)>> = (0..num_trials)
            .into_par_iter()
            .map(|trial_num| {
                swap_trial(
                    num_qubits,
                    int_layout,
                    int_qubit_subset_arr,
                    int_gates_arr,
                    cdist_arr,
                    cdist2_arr,
                    edges_arr,
                    seed_vec[trial_num as usize],
                    trial_num,
                    Some(&locked_best_possible),
                )
            })
            .collect();
        match best_possible {
            Some((_trial_num, _dist, edges, layout)) => {
                best_edges = Some(edges);
                best_layout = Some(layout);
                best_depth = 1;
            }
            None => {
                for (dist, edges, layout, depth) in result.into_iter().flatten() {
                    if dist as usize == num_gates && depth < best_depth {
                        best_edges = Some(edges);
                        best_layout = Some(layout);
                        best_depth = depth;
                    }
                }
            }
        };
    } else {
        for trial_num in 0..num_trials {
            let (dist, edges, layout, depth) = swap_trial(
                num_qubits,
                int_layout,
                int_qubit_subset_arr,
                int_gates_arr,
                cdist_arr,
                cdist2_arr,
                edges_arr,
                seed_vec[trial_num as usize],
                trial_num,
                None,
            )
            .unwrap();
            if dist as usize == num_gates && depth < best_depth {
                best_edges = Some(edges);
                best_layout = Some(layout);
                best_depth = depth;
                if depth == 1 {
                    return Ok((best_edges, best_layout, best_depth));
                }
            }
        }
    }
    Ok((best_edges, best_layout, best_depth))
}

pub fn stochastic_swap(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(swap_trials))?;
    m.add_class::<EdgeCollection>()?;
    Ok(())
}

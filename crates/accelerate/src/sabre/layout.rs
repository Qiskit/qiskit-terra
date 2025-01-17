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
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use pyo3::Python;

use hashbrown::HashSet;
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray, PyReadonlyArray2};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use crate::getenv_use_multiple_threads;
use crate::nlayout::{NLayout, PhysicalQubit};

use super::heuristic::Heuristic;
use super::neighbor_table::NeighborTable;
use super::route::{swap_map, swap_map_trial, RoutingTargetView};
use super::sabre_dag::SabreDAG;
use super::swap_map::SwapMap;
use super::{NodeBlockResults, SabreResult};

use crate::dense_layout::best_subset_inner;

#[pyfunction]
#[pyo3(signature = (dag, neighbor_table, distance_matrix, heuristic, max_iterations, num_swap_trials, num_random_trials, seed=None, partial_layouts=vec![]))]
pub fn sabre_layout_and_routing(
    py: Python,
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    distance_matrix: PyReadonlyArray2<f64>,
    heuristic: &Heuristic,
    max_iterations: usize,
    num_swap_trials: usize,
    num_random_trials: usize,
    seed: Option<u64>,
    mut partial_layouts: Vec<Vec<Option<u32>>>,
) -> (NLayout, PyObject, (SwapMap, PyObject, NodeBlockResults)) {
    let run_in_parallel = getenv_use_multiple_threads();
    let target = RoutingTargetView {
        neighbors: neighbor_table,
        coupling: &neighbor_table.coupling_graph(),
        distance: distance_matrix.as_array(),
    };
    let mut starting_layouts: Vec<Vec<Option<u32>>> =
        (0..num_random_trials).map(|_| vec![]).collect();
    starting_layouts.append(&mut partial_layouts);
    // Run a dense layout trial
    starting_layouts.push(compute_dense_starting_layout(
        dag.num_qubits,
        &target,
        run_in_parallel,
    ));
    starting_layouts.push(
        (0..target.neighbors.num_qubits() as u32)
            .map(Some)
            .collect(),
    );
    starting_layouts.push(
        (0..target.neighbors.num_qubits() as u32)
            .rev()
            .map(Some)
            .collect(),
    );
    // This layout targets the largest ring on an IBM eagle device. It has been
    // shown to have good results on some circuits targeting these backends. In
    // all other cases this is no different from an additional random trial,
    // see: https://xkcd.com/221/
    if target.neighbors.num_qubits() == 127 {
        starting_layouts.push(
            [
                0, 1, 2, 3, 4, 5, 6, 15, 22, 23, 24, 25, 34, 43, 42, 41, 40, 53, 60, 59, 61, 62,
                72, 81, 80, 79, 78, 91, 98, 99, 100, 101, 102, 103, 92, 83, 82, 84, 85, 86, 73, 66,
                65, 64, 63, 54, 45, 44, 46, 47, 35, 28, 29, 27, 26, 16, 7, 8, 9, 10, 11, 12, 13,
                17, 30, 31, 32, 36, 51, 50, 49, 48, 55, 68, 67, 69, 70, 74, 89, 88, 87, 93, 106,
                105, 104, 107, 108, 112, 126, 125, 124, 123, 122, 111, 121, 120, 119, 118, 110,
                117, 116, 115, 114, 113, 109, 96, 97, 95, 94, 90, 75, 76, 77, 71, 58, 57, 56, 52,
                37, 38, 39, 33, 20, 21, 19, 18, 14,
            ]
            .into_iter()
            .map(Some)
            .collect(),
        );
    } else if target.neighbors.num_qubits() == 133 {
        // Same for IBM Heron 133 qubit devices. This is the ring computed by using rustworkx's
        // max(simple_cycles(graph), key=len) on the connectivity graph.
        starting_layouts.push(
            [
                108, 107, 94, 88, 89, 90, 75, 71, 70, 69, 56, 50, 51, 52, 37, 33, 32, 31, 18, 12,
                11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 36, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 53, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 66, 67, 74, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 91, 95, 96, 97,
                110, 116, 117, 118, 119, 120, 111, 101, 102, 103, 104, 105, 112, 124, 125, 126,
                127, 128, 113, 109,
            ]
            .into_iter()
            .map(Some)
            .collect(),
        );
    } else if target.neighbors.num_qubits() == 156 {
        // Same for IBM Heron 156 qubit devices. This is the ring computed by using rustworkx's
        // max(simple_cycles(graph), key=len) on the connectivity graph.
        starting_layouts.push(
            [
                136, 123, 122, 121, 116, 101, 102, 103, 96, 83, 82, 81, 76, 61, 62, 63, 56, 43, 42,
                41, 36, 21, 22, 23, 16, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 35, 34,
                33, 32, 31, 30, 29, 28, 27, 26, 25, 37, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                59, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 77, 85, 86, 87, 88, 89, 90, 91, 92,
                93, 94, 95, 99, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 117, 125,
                126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 139, 155, 154, 153, 152, 151,
                150, 149, 148, 147, 146, 145, 144, 143,
            ]
            .into_iter()
            .map(Some)
            .collect(),
        );
    }
    let outer_rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::from_entropy(),
    };
    let seed_vec: Vec<u64> = outer_rng
        .sample_iter(&rand::distributions::Standard)
        .take(starting_layouts.len())
        .collect();
    let res = if run_in_parallel && starting_layouts.len() > 1 {
        seed_vec
            .into_par_iter()
            .enumerate()
            .map(|(index, seed_trial)| {
                (
                    index,
                    layout_trial(
                        &target,
                        dag,
                        heuristic,
                        seed_trial,
                        max_iterations,
                        num_swap_trials,
                        run_in_parallel,
                        &starting_layouts[index],
                    ),
                )
            })
            .min_by_key(|(index, (_, _, result))| {
                (
                    result.map.map.values().map(|x| x.len()).sum::<usize>(),
                    *index,
                )
            })
            .unwrap()
            .1
    } else {
        seed_vec
            .into_iter()
            .enumerate()
            .map(|(index, seed_trial)| {
                layout_trial(
                    &target,
                    dag,
                    heuristic,
                    seed_trial,
                    max_iterations,
                    num_swap_trials,
                    run_in_parallel,
                    &starting_layouts[index],
                )
            })
            .min_by_key(|(_, _, result)| result.map.map.values().map(|x| x.len()).sum::<usize>())
            .unwrap()
    };
    (
        res.0,
        PyArray::from_vec_bound(py, res.1).into(),
        (
            res.2.map,
            res.2.node_order.into_pyarray_bound(py).into(),
            res.2.node_block_results,
        ),
    )
}

fn layout_trial(
    target: &RoutingTargetView,
    dag: &SabreDAG,
    heuristic: &Heuristic,
    seed: u64,
    max_iterations: usize,
    num_swap_trials: usize,
    run_swap_in_parallel: bool,
    starting_layout: &[Option<u32>],
) -> (NLayout, Vec<PhysicalQubit>, SabreResult) {
    let num_physical_qubits: u32 = target.neighbors.num_qubits().try_into().unwrap();
    let mut rng = Pcg64Mcg::seed_from_u64(seed);

    // This is purely for RNG compatibility during a refactor.
    let routing_seed = Pcg64Mcg::seed_from_u64(seed).next_u64();

    // Pick a random initial layout including a full ancilla allocation.
    let mut initial_layout = {
        let physical_qubits: Vec<PhysicalQubit> = if !starting_layout.is_empty() {
            let used_bits: HashSet<u32> = starting_layout
                .iter()
                .filter_map(|x| x.as_ref())
                .copied()
                .collect();
            let mut free_bits: Vec<u32> = (0..num_physical_qubits)
                .filter(|x| !used_bits.contains(x))
                .collect();
            free_bits.shuffle(&mut rng);
            (0..num_physical_qubits)
                .map(|x| {
                    let bit_index = match starting_layout.get(x as usize) {
                        Some(phys) => phys.unwrap_or_else(|| free_bits.pop().unwrap()),
                        None => free_bits.pop().unwrap(),
                    };
                    PhysicalQubit::new(bit_index)
                })
                .collect()
        } else {
            let mut physical_qubits: Vec<PhysicalQubit> =
                (0..num_physical_qubits).map(PhysicalQubit::new).collect();
            physical_qubits.shuffle(&mut rng);
            physical_qubits
        };
        NLayout::from_virtual_to_physical(physical_qubits).unwrap()
    };

    // Sabre routing currently enforces that control-flow blocks return to their starting layout,
    // which means they don't actually affect any heuristics that affect our layout choice.
    let dag_no_control_forward = SabreDAG {
        num_qubits: dag.num_qubits,
        num_clbits: dag.num_clbits,
        dag: dag.dag.clone(),
        nodes: dag.nodes.clone(),
        first_layer: dag.first_layer.clone(),
        node_blocks: dag
            .node_blocks
            .keys()
            .map(|index| (*index, Vec::new()))
            .collect(),
    };
    let dag_no_control_reverse = SabreDAG::new(
        dag_no_control_forward.num_qubits,
        dag_no_control_forward.num_clbits,
        dag_no_control_forward.nodes.iter().rev().cloned().collect(),
        dag_no_control_forward.node_blocks.clone(),
    )
    .unwrap();

    for _iter in 0..max_iterations {
        for dag in [&dag_no_control_forward, &dag_no_control_reverse] {
            let (_result, final_layout) =
                swap_map_trial(target, dag, heuristic, &initial_layout, routing_seed);
            initial_layout = final_layout;
        }
    }

    let (sabre_result, final_layout) = swap_map(
        target,
        dag,
        heuristic,
        &initial_layout,
        Some(seed),
        num_swap_trials,
        Some(run_swap_in_parallel),
    );
    let final_permutation = initial_layout
        .iter_physical()
        .map(|(_, virt)| virt.to_phys(&final_layout))
        .collect();
    (initial_layout, final_permutation, sabre_result)
}

fn compute_dense_starting_layout(
    num_qubits: usize,
    target: &RoutingTargetView,
    run_in_parallel: bool,
) -> Vec<Option<u32>> {
    let mut adj_matrix = target.distance.to_owned();
    if run_in_parallel {
        adj_matrix.par_mapv_inplace(|x| if x == 1. { 1. } else { 0. });
    } else {
        adj_matrix.mapv_inplace(|x| if x == 1. { 1. } else { 0. });
    }
    let [_rows, _cols, map] = best_subset_inner(
        num_qubits,
        adj_matrix.view(),
        0,
        0,
        false,
        true,
        aview2(&[[0.]]),
    );
    map.into_iter().map(|x| Some(x as u32)).collect()
}

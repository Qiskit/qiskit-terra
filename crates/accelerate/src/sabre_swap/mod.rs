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

pub mod layer;
pub mod neighbor_table;
pub mod sabre_dag;
pub mod swap_map;

use std::cmp::Ordering;

use hashbrown::HashMap;
use ndarray::prelude::*;
use numpy::IntoPyArray;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use rand::prelude::SliceRandom;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use rustworkx_core::dictmap::*;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::visit::EdgeRef;
use rustworkx_core::shortest_path::dijkstra;

use crate::getenv_use_multiple_threads;
use crate::nlayout::NLayout;

use layer::{ExtendedSet, FrontLayer};
use neighbor_table::NeighborTable;
use sabre_dag::SabreDAG;
use swap_map::SwapMap;

const BEST_EPSILON: f64 = 1e-10; // Epsilon used in minimum-score calculations.

const EXTENDED_SET_SIZE: usize = 20; // Size of lookahead window.
const DECAY_RATE: f64 = 0.001; // Decay coefficient for penalizing serial swaps.
const DECAY_RESET_INTERVAL: u8 = 5; // How often to reset all decay rates to 1.
const EXTENDED_SET_WEIGHT: f64 = 0.5; // Weight of lookahead window compared to front_layer.

#[pyclass]
pub enum Heuristic {
    Basic,
    Lookahead,
    Decay,
}

struct TrialResult {
    out_map: HashMap<usize, Vec<[usize; 2]>>,
    gate_order: Vec<usize>,
    layout: NLayout,
}

/// Return a set of candidate swaps that affect qubits in front_layer.
///
/// For each virtual qubit in front_layer, find its current location
/// on hardware and the physical qubits in that neighborhood. Every SWAP
/// on virtual qubits that corresponds to one of those physical couplings
/// is a candidate SWAP.
fn obtain_swaps<'a>(
    front_layer: &'a FrontLayer,
    neighbors: &'a NeighborTable,
    layout: &'a NLayout,
) -> impl Iterator<Item = [usize; 2]> + 'a {
    front_layer.iter_active().flat_map(move |&v| {
        neighbors.neighbors[layout.logic_to_phys[v]]
            .iter()
            .filter_map(move |&neighbor| {
                let virtual_neighbor = layout.phys_to_logic[neighbor];
                if virtual_neighbor > v || !front_layer.is_active(virtual_neighbor) {
                    Some([v, virtual_neighbor])
                } else {
                    None
                }
            })
    })
}

/// Fill the given `extended_set` with the next nodes that would be reachable after the front layer
/// (and themselves).  This uses `required_predecessors` as scratch space for efficiency, but
/// returns it to the same state as the input on return.
fn populate_extended_set(
    extended_set: &mut ExtendedSet,
    dag: &SabreDAG,
    front_layer: &FrontLayer,
    required_predecessors: &mut [u32],
) {
    let mut to_visit = front_layer.iter_nodes().copied().collect::<Vec<_>>();
    let mut decremented: HashMap<usize, u32> = HashMap::new();
    let mut i = 0;
    while i < to_visit.len() && extended_set.len() < EXTENDED_SET_SIZE {
        for edge in dag.dag.edges_directed(to_visit[i], Direction::Outgoing) {
            let successor_node = edge.target();
            let successor_index = successor_node.index();
            *decremented.entry(successor_index).or_insert(0) += 1;
            required_predecessors[successor_index] -= 1;
            if required_predecessors[successor_index] == 0 {
                if let [a, b] = dag.dag[successor_node].1[..] {
                    extended_set.insert(successor_node, &[a, b]);
                }
                to_visit.push(successor_node);
            }
        }
        i += 1;
    }
    for (node, amount) in decremented.iter() {
        required_predecessors[*node] += *amount;
    }
}

fn cmap_from_neighor_table(neighbor_table: &NeighborTable) -> DiGraph<(), ()> {
    DiGraph::<(), ()>::from_edges(neighbor_table.neighbors.iter().enumerate().flat_map(
        |(u, targets)| {
            targets
                .iter()
                .map(move |v| (NodeIndex::new(u), NodeIndex::new(*v)))
        },
    ))
}

/// Run sabre swap on a circuit
///
/// Returns:
///     (SwapMap, gate_order): A tuple where the first element is a mapping of
///     DAGCircuit node ids to a list of virtual qubit swaps that should be
///     added before that operation. The second element is a numpy array of
///     node ids that represents the traversal order used by sabre.
#[pyfunction]
pub fn build_swap_map(
    py: Python,
    num_qubits: usize,
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    distance_matrix: PyReadonlyArray2<f64>,
    heuristic: &Heuristic,
    layout: &mut NLayout,
    num_trials: usize,
    seed: Option<u64>,
    run_in_parallel: Option<bool>,
) -> (SwapMap, PyObject) {
    let dist = distance_matrix.as_array();
    let (swap_map, gate_order) = build_swap_map_inner(
        num_qubits,
        dag,
        neighbor_table,
        &dist,
        heuristic,
        seed,
        layout,
        num_trials,
        run_in_parallel,
    );
    (swap_map, gate_order.into_pyarray(py).into())
}

pub fn build_swap_map_inner(
    num_qubits: usize,
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    dist: &ArrayView2<f64>,
    heuristic: &Heuristic,
    seed: Option<u64>,
    layout: &mut NLayout,
    num_trials: usize,
    run_in_parallel: Option<bool>,
) -> (SwapMap, Vec<usize>) {
    let run_in_parallel = match run_in_parallel {
        Some(run_in_parallel) => run_in_parallel,
        None => getenv_use_multiple_threads() && num_trials > 1,
    };
    let coupling_graph: DiGraph<(), ()> = cmap_from_neighor_table(neighbor_table);
    let outer_rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::from_entropy(),
    };
    let seed_vec: Vec<u64> = outer_rng
        .sample_iter(&rand::distributions::Standard)
        .take(num_trials)
        .collect();
    let result = if run_in_parallel {
        seed_vec
            .into_par_iter()
            .enumerate()
            .map(|(index, seed_trial)| {
                (
                    index,
                    swap_map_trial(
                        num_qubits,
                        dag,
                        neighbor_table,
                        dist,
                        &coupling_graph,
                        heuristic,
                        seed_trial,
                        layout.clone(),
                    ),
                )
            })
            .min_by_key(|(index, result)| {
                [
                    result.out_map.values().map(|x| x.len()).sum::<usize>(),
                    *index,
                ]
            })
            .unwrap()
            .1
    } else {
        seed_vec
            .into_iter()
            .map(|seed_trial| {
                swap_map_trial(
                    num_qubits,
                    dag,
                    neighbor_table,
                    dist,
                    &coupling_graph,
                    heuristic,
                    seed_trial,
                    layout.clone(),
                )
            })
            .min_by_key(|result| result.out_map.values().map(|x| x.len()).sum::<usize>())
            .unwrap()
    };
    *layout = result.layout;
    (
        SwapMap {
            map: result.out_map,
        },
        result.gate_order,
    )
}

fn swap_map_trial(
    num_qubits: usize,
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    dist: &ArrayView2<f64>,
    coupling_graph: &DiGraph<(), ()>,
    heuristic: &Heuristic,
    seed: u64,
    mut layout: NLayout,
) -> TrialResult {
    let max_iterations_without_progress = 10 * neighbor_table.neighbors.len();
    let mut out_map: HashMap<usize, Vec<[usize; 2]>> = HashMap::new();
    let mut gate_order = Vec::with_capacity(dag.dag.node_count());
    let mut front_layer = FrontLayer::new(num_qubits);
    let mut extended_set = ExtendedSet::new(num_qubits, EXTENDED_SET_SIZE);
    let mut required_predecessors: Vec<u32> = vec![0; dag.dag.node_count()];
    let mut num_search_steps: u8 = 0;
    let mut qubits_decay: Vec<f64> = vec![1.; num_qubits];
    let mut rng = Pcg64Mcg::seed_from_u64(seed);

    for node in dag.dag.node_indices() {
        for edge in dag.dag.edges(node) {
            required_predecessors[edge.target().index()] += 1;
        }
    }
    route_reachable_nodes(
        &dag.first_layer,
        dag,
        &layout,
        coupling_graph,
        &mut gate_order,
        &mut front_layer,
        &mut required_predecessors,
    );
    populate_extended_set(
        &mut extended_set,
        dag,
        &front_layer,
        &mut required_predecessors,
    );
    // Main logic loop; the front layer only becomes empty when all nodes have been routed.  At
    // each iteration of this loop, we route either one or two gates.
    let mut routable_nodes = Vec::<NodeIndex>::with_capacity(2);
    while !front_layer.is_empty() {
        let mut current_swaps: Vec<[usize; 2]> = Vec::new();
        // Swap-mapping loop.  This is the main part of the algorithm, which we repeat until we
        // either successfully route a node, or exceed the maximum number of attempts.
        while routable_nodes.is_empty() && current_swaps.len() <= max_iterations_without_progress {
            let best_swap = choose_best_swap(
                &front_layer,
                &extended_set,
                &layout,
                neighbor_table,
                dist,
                &qubits_decay,
                heuristic,
                &mut rng,
            );
            front_layer.routable_after(&mut routable_nodes, &best_swap, &layout, coupling_graph);
            current_swaps.push(best_swap);
            layout.swap_logical(best_swap[0], best_swap[1]);
            num_search_steps += 1;
            if num_search_steps >= DECAY_RESET_INTERVAL {
                qubits_decay.fill(1.);
                num_search_steps = 0;
            } else {
                qubits_decay[best_swap[0]] += DECAY_RATE;
                qubits_decay[best_swap[1]] += DECAY_RATE;
            }
        }
        // If we exceeded the number of allowed attempts without successfully routing a node, we
        // reset back to the state we were in last time we routed a node, then find the node in the
        // front layer whose qubits are the closest in the coupling map, and greedily insert swaps
        // to make the node routable.  We could clone the layout each time we route a gate, but
        // this path is only an escape mechansim for the algorithm getting stuck, so it should
        // ideally never be taken, and it doesn't matter if it's not the speediest---it's better to
        // keep the other path faster.
        if routable_nodes.is_empty() {
            undo_swaps(&mut current_swaps, &mut layout);
            let (node, qubits) = closest_operation(&front_layer, &layout, dist);
            swaps_to_route(&mut current_swaps, &qubits, &layout, coupling_graph);
            for &[a, b] in current_swaps.iter() {
                layout.swap_logical(a, b);
            }
            routable_nodes.push(node);
        }
        update_route(
            &routable_nodes,
            current_swaps,
            dag,
            &layout,
            coupling_graph,
            &mut gate_order,
            &mut out_map,
            &mut front_layer,
            &mut extended_set,
            &mut required_predecessors,
        );
        qubits_decay.fill(1.);
        routable_nodes.clear();
    }
    TrialResult {
        out_map,
        gate_order,
        layout,
    }
}

/// Update the system state as the given `nodes` are added to the routing order, preceded by the
/// given `swaps`.  This involves updating the output values `gate_order` and `out_map`, but also
/// the tracking objects `front_layer`, `extended_set` and `required_predecessors` by removing the
/// routed nodes and adding any now-reachable ones.
fn update_route(
    nodes: &[NodeIndex],
    swaps: Vec<[usize; 2]>,
    dag: &SabreDAG,
    layout: &NLayout,
    coupling: &DiGraph<(), ()>,
    gate_order: &mut Vec<usize>,
    out_map: &mut HashMap<usize, Vec<[usize; 2]>>,
    front_layer: &mut FrontLayer,
    extended_set: &mut ExtendedSet,
    required_predecessors: &mut [u32],
) {
    // First node gets the swaps attached.  We don't add to the `gate_order` here because
    // `route_reachable_nodes` is responsible for that part.
    let py_node = dag.dag[nodes[0]].0;
    out_map.insert(py_node, swaps);
    for node in nodes {
        front_layer.remove(node);
    }
    route_reachable_nodes(
        nodes,
        dag,
        layout,
        coupling,
        gate_order,
        front_layer,
        required_predecessors,
    );
    // Ideally we'd know how to mutate the extended set directly, but since its limited size ties
    // its construction strongly to the iteration order through the front layer, it's not easy to
    // do better than just emptying it and rebuilding.
    extended_set.clear();
    populate_extended_set(extended_set, dag, front_layer, required_predecessors);
}

/// Search forwards in the `dag` from all the nodes in `to_visit`, adding them to the `gate_order`
/// or the current `front_layer` as appropriate, and continue inspecting gates until there is
/// nothing further with no required predecessors.
///
/// The nodes in `to_visit` should all already have no further required predecessors.
fn route_reachable_nodes(
    to_visit: &[NodeIndex],
    dag: &SabreDAG,
    layout: &NLayout,
    coupling: &DiGraph<(), ()>,
    gate_order: &mut Vec<usize>,
    front_layer: &mut FrontLayer,
    required_predecessors: &mut [u32],
) {
    let mut to_visit = to_visit.to_vec();
    let mut i = 0;
    // Iterate through `to_visit`, except we often push new nodes onto the end of it.
    while i < to_visit.len() {
        let node = to_visit[i];
        i += 1;
        let (py_node, qubits) = &dag.dag[node];
        match qubits[..] {
            [a, b]
                if !coupling.contains_edge(
                    NodeIndex::new(layout.logic_to_phys[a]),
                    NodeIndex::new(layout.logic_to_phys[b]),
                ) =>
            {
                front_layer.insert(node, [a, b]);
            }
            _ => {
                gate_order.push(*py_node);
                for edge in dag.dag.edges_directed(node, Direction::Outgoing) {
                    let successor_node = edge.target();
                    let successor_index = successor_node.index();
                    required_predecessors[successor_index] -= 1;
                    if required_predecessors[successor_index] == 0 {
                        to_visit.push(successor_node);
                    }
                }
            }
        }
    }
}

/// Walk through the swaps in the given vector, undoing them on the layout and removing them.
fn undo_swaps(swaps: &mut Vec<[usize; 2]>, layout: &mut NLayout) {
    swaps
        .drain(..)
        .rev()
        .for_each(|swap| layout.swap_logical(swap[0], swap[1]));
}

/// Find the node index and its associated virtual qubits that is currently the closest to being
/// routable in terms of number of swaps.
fn closest_operation(
    front_layer: &FrontLayer,
    layout: &NLayout,
    dist: &ArrayView2<f64>,
) -> (NodeIndex, [usize; 2]) {
    let (&node, qubits) = front_layer
        .iter()
        .map(|(node, qubits)| {
            (
                node,
                [
                    layout.logic_to_phys[qubits[0]],
                    layout.logic_to_phys[qubits[1]],
                ],
            )
        })
        .min_by(|(_, qubits_a), (_, qubits_b)| {
            dist[*qubits_a]
                .partial_cmp(&dist[*qubits_b])
                .unwrap_or(Ordering::Equal)
        })
        .unwrap();
    (
        node,
        [
            layout.phys_to_logic[qubits[0]],
            layout.phys_to_logic[qubits[1]],
        ],
    )
}

/// Add the minimal set of swaps to the `swaps` vector that bring the two `qubits` together so that
/// a 2q gate on them could be routed.
fn swaps_to_route(
    swaps: &mut Vec<[usize; 2]>,
    qubits: &[usize; 2],
    layout: &NLayout,
    coupling_graph: &DiGraph<(), ()>,
) {
    let mut shortest_paths: DictMap<NodeIndex, Vec<NodeIndex>> = DictMap::new();
    let u = layout.logic_to_phys[qubits[0]];
    let v = layout.logic_to_phys[qubits[1]];
    (dijkstra(
        coupling_graph,
        NodeIndex::<u32>::new(u),
        Some(NodeIndex::<u32>::new(v)),
        |_| Ok(1.),
        Some(&mut shortest_paths),
    ) as PyResult<Vec<Option<f64>>>)
        .unwrap();
    let shortest_path: Vec<usize> = shortest_paths
        .get(&NodeIndex::new(v))
        .unwrap()
        .iter()
        .map(|n| n.index())
        .collect();
    // Insert greedy swaps along that shortest path
    let split: usize = shortest_path.len() / 2;
    let forwards = &shortest_path[1..split];
    let backwards = &shortest_path[split..shortest_path.len() - 1];
    swaps.reserve(shortest_path.len() - 2);
    for swap in forwards {
        swaps.push([qubits[0], layout.phys_to_logic[*swap]]);
    }
    for swap in backwards.iter().rev() {
        swaps.push([qubits[1], layout.phys_to_logic[*swap]]);
    }
}

/// Return the swap of two virtual qubits that produces the best score of all possible swaps.
fn choose_best_swap(
    layer: &FrontLayer,
    extended_set: &ExtendedSet,
    layout: &NLayout,
    neighbor_table: &NeighborTable,
    dist: &ArrayView2<f64>,
    qubits_decay: &[f64],
    heuristic: &Heuristic,
    rng: &mut Pcg64Mcg,
) -> [usize; 2] {
    let mut min_score = f64::MAX;
    let mut best_swaps: Vec<[usize; 2]> = Vec::new();
    // The decay heuristic is the only one that actually needs the absolute score.
    let absolute_score = match heuristic {
        Heuristic::Decay => {
            layer.total_score(layout, dist)
                + EXTENDED_SET_WEIGHT * extended_set.total_score(layout, dist)
        }
        _ => 0.0,
    };
    for swap in obtain_swaps(layer, neighbor_table, layout) {
        let score = match heuristic {
            Heuristic::Basic => layer.score(swap, layout, dist),
            Heuristic::Lookahead => {
                layer.score(swap, layout, dist)
                    + EXTENDED_SET_WEIGHT * extended_set.score(swap, layout, dist)
            }
            Heuristic::Decay => {
                qubits_decay[swap[0]].max(qubits_decay[swap[1]])
                    * (absolute_score
                        + layer.score(swap, layout, dist)
                        + EXTENDED_SET_WEIGHT * extended_set.score(swap, layout, dist))
            }
        };
        if score < min_score - BEST_EPSILON {
            min_score = score;
            best_swaps.clear();
            best_swaps.push(swap);
        } else if (score - min_score).abs() < BEST_EPSILON {
            best_swaps.push(swap);
        }
        //println!("swap: {}-{}: {}", swap[0], swap[1], score);
    }
    *best_swaps.choose(rng).unwrap()
}

#[pymodule]
pub fn sabre_swap(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(build_swap_map))?;
    m.add_class::<Heuristic>()?;
    m.add_class::<NeighborTable>()?;
    m.add_class::<SabreDAG>()?;
    m.add_class::<SwapMap>()?;
    Ok(())
}

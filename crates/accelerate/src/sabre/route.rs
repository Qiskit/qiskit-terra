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

use std::cmp::Ordering;

use pyo3::prelude::*;
use pyo3::Python;

use hashbrown::HashMap;
use indexmap::IndexMap;
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray, PyReadonlyArray2};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use rustworkx_core::dictmap::*;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::visit::EdgeRef;
use rustworkx_core::shortest_path::dijkstra;
use rustworkx_core::token_swapper::token_swapper;
use smallvec::{smallvec, SmallVec};

use crate::getenv_use_multiple_threads;
use crate::nlayout::{NLayout, PhysicalQubit};

use super::heuristic::{BasicHeuristic, DecayHeuristic, Heuristic, LookaheadHeuristic, SetScaling};
use super::layer::{ExtendedSet, FrontLayer};
use super::neighbor_table::NeighborTable;
use super::sabre_dag::SabreDAG;
use super::swap_map::SwapMap;
use super::{BlockResult, NodeBlockResults, SabreResult};

/// Number of trials for control flow block swap epilogues.
const SWAP_EPILOGUE_TRIALS: usize = 4;

/// A view object onto a full routing target.  This is cheap to clone and to replace components
/// within it; cloning only duplicates the inner references and not the data objects beneath.  This
/// struct doesn't own its data because it's typically a view onto data generated from Python, and
/// we want to avoid copies.
pub struct RoutingTargetView<'a> {
    pub neighbors: &'a NeighborTable,
    pub coupling: &'a DiGraph<(), ()>,
    pub distance: ArrayView2<'a, f64>,
}

/// Long-term internal state of the Sabre routing algorithm.  This includes all the scratch space
/// and tracking that we use over the course of many swap insertions, but doesn't include ephemeral
/// state that never needs to leave the main loop.  This is mostly just a convenience, so we don't
/// have to pass everything from function to function.
struct RoutingState<'a, 'b> {
    target: &'a RoutingTargetView<'b>,
    dag: &'a SabreDAG,
    heuristic: &'a Heuristic,
    /// Mapping of instructions (node indices) to swaps that precede them.
    out_map: HashMap<usize, Vec<[PhysicalQubit; 2]>>,
    /// Order of the instructions (node indices) in the problem DAG in the output.
    gate_order: Vec<usize>,
    /// Map from node index of a control-flow op to its inner result.
    node_block_results: HashMap<usize, Vec<BlockResult>>,
    front_layer: FrontLayer,
    extended_set: ExtendedSet,
    decay: &'a mut [f64],
    /// How many predecessors still need to be satisfied for each node index before it is at the
    /// front of the topological iteration through the nodes as they're routed.
    required_predecessors: &'a mut [u32],
    layout: NLayout,
    /// Reusable allocated storage space for accumulating and scoring swaps.  This is owned as part
    /// of the general state to avoid reallocation costs.
    swap_scores: Vec<([PhysicalQubit; 2], f64)>,
    /// Reusable allocated storage space for tracking the current best swaps.  This is owned as
    /// part of the general state to avoid reallocation costs.
    best_swaps: Vec<[PhysicalQubit; 2]>,
    rng: Pcg64Mcg,
    seed: u64,
}

impl<'a, 'b> RoutingState<'a, 'b> {
    /// Apply a swap to the program-state structures (front layer, extended set and current
    /// layout).
    #[inline]
    fn apply_swap(&mut self, swap: [PhysicalQubit; 2]) {
        self.front_layer.apply_swap(swap);
        self.extended_set.apply_swap(swap);
        self.layout.swap_physical(swap[0], swap[1]);
    }

    /// Return the node, if any, that is on this qubit and is routable with the current layout.
    #[inline]
    fn routable_node_on_qubit(&self, qubit: PhysicalQubit) -> Option<NodeIndex> {
        self.front_layer.qubits()[qubit.index()].and_then(|(node, other)| {
            self.target
                .coupling
                .contains_edge(NodeIndex::new(qubit.index()), NodeIndex::new(other.index()))
                .then_some(node)
        })
    }

    /// Update the system state as the given `nodes` are added to the routing order, preceded by
    /// the given `swaps`.  This involves updating the output values `gate_order` and `out_map`,
    /// but also the tracking objects `front_layer`, `extended_set` and `required_predecessors` by
    /// removing the routed nodes and adding any now-reachable ones.
    fn update_route(&mut self, nodes: &[NodeIndex], swaps: Vec<[PhysicalQubit; 2]>) {
        // First node gets the swaps attached.  We don't add to the `gate_order` here because
        // `route_reachable_nodes` is responsible for that part.
        self.out_map
            .insert(self.dag.dag[nodes[0]].py_node_id, swaps);
        for node in nodes {
            self.front_layer.remove(node);
        }
        self.route_reachable_nodes(nodes);
        // Ideally we'd know how to mutate the extended set directly, but since its limited size
        // ties its construction strongly to the iteration order through the front layer, it's not
        // easy to do better than just emptying it and rebuilding.
        self.extended_set.clear();
        self.populate_extended_set();
    }

    /// Search forwards in the DAG from all the nodes in `nodes`, adding them to the `gate_order`
    /// or the current `front_layer` as appropriate, and continue inspecting gates until there is
    /// nothing further with no required predecessors.
    ///
    /// The nodes in `nodes` should all already have no further required predecessors.
    fn route_reachable_nodes(&mut self, nodes: &[NodeIndex]) {
        let mut to_visit = nodes.to_vec();
        let mut i = 0;
        let dag = &self.dag;
        // Iterate through `to_visit`, except we often push new nodes onto the end of it.
        while i < to_visit.len() {
            let node_id = to_visit[i];
            let node = &dag.dag[node_id];
            i += 1;

            // If the node is a directive that means it can be placed anywhere.
            if !node.directive {
                if let Some(blocks) = dag.node_blocks.get(&node.py_node_id) {
                    let block_results = blocks
                        .iter()
                        .map(|block| self.route_control_flow_block(block))
                        .collect::<Vec<_>>();
                    self.node_block_results
                        .insert(node.py_node_id, block_results);
                } else {
                    match node.qubits[..] {
                        // A gate op whose connectivity must match the device to be placed in the
                        // gate order.
                        [a, b]
                            if !self.target.coupling.contains_edge(
                                NodeIndex::new(a.to_phys(&self.layout).index()),
                                NodeIndex::new(b.to_phys(&self.layout).index()),
                            ) =>
                        {
                            // 2Q op that cannot be placed. Add it to the front layer and move on.
                            self.front_layer.insert(
                                node_id,
                                [a.to_phys(&self.layout), b.to_phys(&self.layout)],
                            );
                            continue;
                        }
                        _ => {}
                    }
                }
            }

            // If we reach here, the node is routable.
            self.gate_order.push(node.py_node_id);
            for edge in dag.dag.edges_directed(node_id, Direction::Outgoing) {
                let successor_node = edge.target();
                let successor_index = successor_node.index();
                self.required_predecessors[successor_index] -= 1;
                if self.required_predecessors[successor_index] == 0 {
                    to_visit.push(successor_node);
                }
            }
        }
    }

    /// Inner worker to route a control-flow block.  Since control-flow blocks are routed to
    /// restore the layout at the end of themselves, and the recursive calls spawn their own
    /// tracking states, this does not affect our own state.
    fn route_control_flow_block(&self, block: &SabreDAG) -> BlockResult {
        let (result, mut block_final_layout) =
            swap_map_trial(self.target, block, self.heuristic, &self.layout, self.seed);
        // For now, we always append a swap circuit that gets the inner block back to the
        // parent's layout.
        let swap_epilogue = {
            // Map physical location in the final layout from the inner routing to the current
            // location in the outer routing.
            let mapping: HashMap<NodeIndex, NodeIndex> = block_final_layout
                .iter_physical()
                .map(|(p, v)| {
                    (
                        NodeIndex::new(p.index()),
                        NodeIndex::new(v.to_phys(&self.layout).index()),
                    )
                })
                .collect();

            let swaps = token_swapper(
                &self.target.coupling,
                mapping,
                Some(SWAP_EPILOGUE_TRIALS),
                Some(self.seed),
                None,
            )
            .unwrap();

            // Convert physical swaps to virtual swaps
            swaps
                .into_iter()
                .map(|(l, r)| {
                    let p_l = PhysicalQubit::new(l.index().try_into().unwrap());
                    let p_r = PhysicalQubit::new(r.index().try_into().unwrap());
                    block_final_layout.swap_physical(p_l, p_r);
                    [p_l, p_r]
                })
                .collect()
        };
        BlockResult {
            result,
            swap_epilogue,
        }
    }

    /// Fill the given `extended_set` with the next nodes that would be reachable after the front
    /// layer (and themselves).  This uses `required_predecessors` as scratch space for efficiency,
    /// but returns it to the same state as the input on return.
    fn populate_extended_set(&mut self) {
        let extended_set_size =
            if let Some(LookaheadHeuristic { size, .. }) = self.heuristic.lookahead {
                size
            } else {
                return;
            };
        let mut to_visit = self.front_layer.iter_nodes().copied().collect::<Vec<_>>();
        let mut decremented: IndexMap<usize, u32, ahash::RandomState> =
            IndexMap::with_hasher(ahash::RandomState::default());
        let mut i = 0;
        let mut visit_now: Vec<NodeIndex> = Vec::new();
        let dag = &self.dag;
        while i < to_visit.len() && self.extended_set.len() < extended_set_size {
            // Visit runs of non-2Q gates fully before moving on to children of 2Q gates. This way,
            // traversal order is a BFS of 2Q gates rather than of all gates.
            visit_now.push(to_visit[i]);
            let mut j = 0;
            while let Some(node) = visit_now.get(j) {
                for edge in dag.dag.edges_directed(*node, Direction::Outgoing) {
                    let successor_node = edge.target();
                    let successor_index = successor_node.index();
                    *decremented.entry(successor_index).or_insert(0) += 1;
                    self.required_predecessors[successor_index] -= 1;
                    if self.required_predecessors[successor_index] == 0 {
                        if !dag.dag[successor_node].directive
                            && !dag.node_blocks.contains_key(&successor_index)
                        {
                            if let [a, b] = dag.dag[successor_node].qubits[..] {
                                self.extended_set
                                    .push([a.to_phys(&self.layout), b.to_phys(&self.layout)]);
                                to_visit.push(successor_node);
                                continue;
                            }
                        }
                        visit_now.push(successor_node);
                    }
                }
                j += 1;
            }
            visit_now.clear();
            i += 1;
        }
        for (node, amount) in decremented.iter() {
            self.required_predecessors[*node] += *amount;
        }
    }

    /// Add swaps to the current set that greedily bring the nearest node together.  This is a
    /// "release valve" mechanism; it ignores all the Sabre heuristics and forces progress, so we
    /// can't get permanently stuck.
    fn force_enable_closest_node(
        &mut self,
        current_swaps: &mut Vec<[PhysicalQubit; 2]>,
    ) -> SmallVec<[NodeIndex; 2]> {
        let (&closest_node, &qubits) = {
            let dist = &self.target.distance;
            self.front_layer
                .iter()
                .min_by(|(_, qubits_a), (_, qubits_b)| {
                    dist[[qubits_a[0].index(), qubits_a[1].index()]]
                        .partial_cmp(&dist[[qubits_b[0].index(), qubits_b[1].index()]])
                        .unwrap_or(Ordering::Equal)
                })
                .unwrap()
        };
        let shortest_path = {
            let mut shortest_paths: DictMap<NodeIndex, Vec<NodeIndex>> = DictMap::new();
            (dijkstra(
                &self.target.coupling,
                NodeIndex::new(qubits[0].index()),
                Some(NodeIndex::new(qubits[1].index())),
                |_| Ok(1.),
                Some(&mut shortest_paths),
            ) as PyResult<Vec<Option<f64>>>)
                .unwrap();
            shortest_paths
                .get(&NodeIndex::new(qubits[1].index()))
                .unwrap()
                .iter()
                .map(|n| PhysicalQubit::new(n.index() as u32))
                .collect::<Vec<_>>()
        };
        // Insert greedy swaps along that shortest path, splitting them between moving the left side
        // and moving the right side to minimise the depth.  One side needs to move up to the split
        // point and the other can stop one short because the gate will be routable then.
        let split: usize = shortest_path.len() / 2;
        current_swaps.reserve(shortest_path.len() - 2);
        for i in 0..split {
            current_swaps.push([shortest_path[i], shortest_path[i + 1]]);
        }
        for i in 0..split - 1 {
            let end = shortest_path.len() - 1 - i;
            current_swaps.push([shortest_path[end], shortest_path[end - 1]]);
        }
        current_swaps.iter().for_each(|&swap| self.apply_swap(swap));

        // If we apply a single swap it could be that we route 2 nodes; that is a setup like
        //  A - B - A - B
        // and we swap the middle two qubits. This cannot happen if we apply 2 or more swaps.
        if current_swaps.len() > 1 {
            smallvec![closest_node]
        } else {
            // check if the closest node has neighbors that are now routable -- for that we get
            // the other physical qubit that was swapped and check whether the node on it
            // is now routable
            let mut possible_other_qubit = current_swaps[0]
                .iter()
                // check if other nodes are in the front layer that are connected by this swap
                .filter_map(|&swap_qubit| self.front_layer.qubits()[swap_qubit.index()])
                // remove the closest_node, which we know we already routed
                .filter(|(node_index, _other_qubit)| *node_index != closest_node)
                .map(|(_node_index, other_qubit)| other_qubit);

            // if there is indeed another candidate, check if that gate is routable
            if let Some(other_qubit) = possible_other_qubit.next() {
                if let Some(also_routed) = self.routable_node_on_qubit(other_qubit) {
                    return smallvec![closest_node, also_routed];
                }
            }
            smallvec![closest_node]
        }
    }

    /// Return the swap of two virtual qubits that produces the best score of all possible swaps.
    fn choose_best_swap(&mut self) -> [PhysicalQubit; 2] {
        // Obtain all candidate swaps from the front layer.  A coupling-map edge is a candidate
        // swap if it involves at least one active qubit (i.e. it must affect the "basic"
        // heuristic), and if it involves two active qubits, we choose the `swap[0] < swap[1]` form
        // to make a canonical choice.
        self.swap_scores.clear();
        for &phys in self.front_layer.iter_active() {
            for &neighbor in self.target.neighbors[phys].iter() {
                if neighbor > phys || !self.front_layer.is_active(neighbor) {
                    self.swap_scores.push(([phys, neighbor], 0.0));
                }
            }
        }

        let dist = &self.target.distance;
        let mut absolute_score = 0.0;

        if let Some(BasicHeuristic { weight, scale }) = self.heuristic.basic {
            let weight = match scale {
                SetScaling::Constant => weight,
                SetScaling::Size => {
                    if self.front_layer.is_empty() {
                        0.0
                    } else {
                        weight / (self.front_layer.len() as f64)
                    }
                }
            };
            absolute_score += weight * self.front_layer.total_score(dist);
            for (swap, score) in self.swap_scores.iter_mut() {
                *score += weight * self.front_layer.score(*swap, dist);
            }
        }

        if let Some(LookaheadHeuristic { weight, scale, .. }) = self.heuristic.lookahead {
            let weight = match scale {
                SetScaling::Constant => weight,
                SetScaling::Size => {
                    if self.extended_set.is_empty() {
                        0.0
                    } else {
                        weight / (self.extended_set.len() as f64)
                    }
                }
            };
            absolute_score += weight * self.extended_set.total_score(dist);
            for (swap, score) in self.swap_scores.iter_mut() {
                *score += weight * self.extended_set.score(*swap, dist);
            }
        }

        if let Some(DecayHeuristic { .. }) = self.heuristic.decay {
            for (swap, score) in self.swap_scores.iter_mut() {
                *score = (absolute_score + *score)
                    * self.decay[swap[0].index()].max(self.decay[swap[1].index()]);
            }
        }

        let mut min_score = f64::INFINITY;
        let epsilon = self.heuristic.best_epsilon;
        for &(swap, score) in self.swap_scores.iter() {
            if score - min_score < -epsilon {
                min_score = score;
                self.best_swaps.clear();
                self.best_swaps.push(swap);
            } else if (score - min_score).abs() <= epsilon {
                self.best_swaps.push(swap);
            }
        }
        *self.best_swaps.choose(&mut self.rng).unwrap()
    }
}

/// Run sabre swap on a circuit
///
/// Returns:
///     (SwapMap, gate_order, node_block_results, final_permutation): A tuple where the first
///     element is a mapping of DAGCircuit node ids to a list of virtual qubit swaps that should be
///     added before that operation. The second element is a numpy array of node ids that
///     represents the traversal order used by sabre.  The third is inner results for the blocks of
///     control flow, and the fourth is a permutation, where `final_permution[i]` is the final
///     logical position of the qubit that began in position `i`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn sabre_routing(
    py: Python,
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    distance_matrix: PyReadonlyArray2<f64>,
    heuristic: &Heuristic,
    initial_layout: &NLayout,
    num_trials: usize,
    seed: Option<u64>,
    run_in_parallel: Option<bool>,
) -> (SwapMap, PyObject, NodeBlockResults, PyObject) {
    let target = RoutingTargetView {
        neighbors: neighbor_table,
        coupling: &neighbor_table.coupling_graph(),
        distance: distance_matrix.as_array(),
    };
    let (res, final_layout) = swap_map(
        &target,
        dag,
        heuristic,
        initial_layout,
        seed,
        num_trials,
        run_in_parallel,
    );
    (
        res.map,
        res.node_order.into_pyarray_bound(py).into(),
        res.node_block_results,
        PyArray::from_iter_bound(
            py,
            (0u32..neighbor_table.num_qubits().try_into().unwrap()).map(|phys| {
                PhysicalQubit::new(phys)
                    .to_virt(initial_layout)
                    .to_phys(&final_layout)
            }),
        )
        .into(),
    )
}

/// Run (potentially in parallel) several trials of the Sabre routing algorithm on the given
/// problem and return the one with fewest swaps.
pub fn swap_map(
    target: &RoutingTargetView,
    dag: &SabreDAG,
    heuristic: &Heuristic,
    initial_layout: &NLayout,
    seed: Option<u64>,
    num_trials: usize,
    run_in_parallel: Option<bool>,
) -> (SabreResult, NLayout) {
    let run_in_parallel = match run_in_parallel {
        Some(run_in_parallel) => run_in_parallel,
        None => getenv_use_multiple_threads() && num_trials > 1,
    };
    let outer_rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::from_entropy(),
    };
    let seed_vec: Vec<u64> = outer_rng
        .sample_iter(&rand::distributions::Standard)
        .take(num_trials)
        .collect();
    if run_in_parallel {
        seed_vec
            .into_par_iter()
            .enumerate()
            .map(|(index, seed_trial)| {
                (
                    index,
                    swap_map_trial(target, dag, heuristic, initial_layout, seed_trial),
                )
            })
            .min_by_key(|(index, (result, _))| {
                [
                    result.map.map.values().map(|x| x.len()).sum::<usize>(),
                    *index,
                ]
            })
            .unwrap()
            .1
    } else {
        seed_vec
            .into_iter()
            .map(|seed_trial| swap_map_trial(target, dag, heuristic, initial_layout, seed_trial))
            .min_by_key(|(result, _)| result.map.map.values().map(|x| x.len()).sum::<usize>())
            .unwrap()
    }
}

/// Run a single trial of the Sabre routing algorithm.
pub fn swap_map_trial(
    target: &RoutingTargetView,
    dag: &SabreDAG,
    heuristic: &Heuristic,
    initial_layout: &NLayout,
    seed: u64,
) -> (SabreResult, NLayout) {
    let num_qubits: u32 = target.neighbors.num_qubits().try_into().unwrap();
    let mut state = RoutingState {
        target,
        dag,
        heuristic,
        out_map: HashMap::new(),
        gate_order: Vec::with_capacity(dag.dag.node_count()),
        node_block_results: HashMap::with_capacity(dag.node_blocks.len()),
        front_layer: FrontLayer::new(num_qubits),
        extended_set: ExtendedSet::new(num_qubits),
        decay: &mut vec![1.; num_qubits as usize],
        required_predecessors: &mut vec![0; dag.dag.node_count()],
        layout: initial_layout.clone(),
        swap_scores: Vec::with_capacity(target.coupling.edge_count()),
        best_swaps: Vec::new(),
        rng: Pcg64Mcg::seed_from_u64(seed),
        seed,
    };
    for node in dag.dag.node_indices() {
        for edge in dag.dag.edges(node) {
            state.required_predecessors[edge.target().index()] += 1;
        }
    }
    state.route_reachable_nodes(&dag.first_layer);
    state.populate_extended_set();

    // Main logic loop; the front layer only becomes empty when all nodes have been routed.  At
    // each iteration of this loop, we route either one or two gates.
    let mut routable_nodes = Vec::<NodeIndex>::with_capacity(2);
    let mut num_search_steps = 0;

    while !state.front_layer.is_empty() {
        let mut current_swaps: Vec<[PhysicalQubit; 2]> = Vec::new();
        // Swap-mapping loop.  This is the main part of the algorithm, which we repeat until we
        // either successfully route a node, or exceed the maximum number of attempts.
        while routable_nodes.is_empty() && current_swaps.len() <= state.heuristic.attempt_limit {
            let best_swap = state.choose_best_swap();
            state.apply_swap(best_swap);
            current_swaps.push(best_swap);
            if let Some(node) = state.routable_node_on_qubit(best_swap[1]) {
                routable_nodes.push(node);
            }
            if let Some(node) = state.routable_node_on_qubit(best_swap[0]) {
                routable_nodes.push(node);
            }
            if let Some(DecayHeuristic { increment, reset }) = state.heuristic.decay {
                num_search_steps += 1;
                if num_search_steps >= reset {
                    state.decay.fill(1.);
                    num_search_steps = 0;
                } else {
                    state.decay[best_swap[0].index()] += increment;
                    state.decay[best_swap[1].index()] += increment;
                }
            }
        }
        if routable_nodes.is_empty() {
            // If we exceeded the max number of heuristic-chosen swaps without making progress,
            // unwind to the last progress point and greedily swap to bring a node together.
            // Efficiency doesn't matter much; this path never gets taken unless we're unlucky.
            current_swaps
                .drain(..)
                .rev()
                .for_each(|swap| state.apply_swap(swap));
            let force_routed = state.force_enable_closest_node(&mut current_swaps);
            routable_nodes.extend(force_routed);
        }
        state.update_route(&routable_nodes, current_swaps);
        if state.heuristic.decay.is_some() {
            state.decay.fill(1.);
        }
        routable_nodes.clear();
    }
    (
        SabreResult {
            map: SwapMap { map: state.out_map },
            node_order: state.gate_order,
            node_block_results: NodeBlockResults {
                results: state.node_block_results,
            },
        },
        state.layout,
    )
}

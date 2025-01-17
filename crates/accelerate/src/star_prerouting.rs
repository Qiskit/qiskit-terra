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

/// Type alias for a node representation.
/// Each node is represented as a tuple containing:
/// - Node id (usize)
/// - List of involved qubit indices (Vec<VirtualQubit>)
/// - Set of involved classical bit indices (HashSet<usize>)
/// - Directive flag (bool)
type Nodes = (usize, Vec<VirtualQubit>, HashSet<usize>, bool);

/// Type alias for a block representation.
/// Each block is represented by a tuple containing:
/// - A boolean indicating the presence of a center (bool)
/// - A list of nodes (Vec<Nodes>)
type Block = (bool, Vec<Nodes>);

use crate::nlayout::PhysicalQubit;
use crate::nlayout::VirtualQubit;
use crate::sabre::sabre_dag::SabreDAG;
use crate::sabre::swap_map::SwapMap;
use crate::sabre::BlockResult;
use crate::sabre::NodeBlockResults;
use crate::sabre::SabreResult;
use hashbrown::HashMap;
use hashbrown::HashSet;
use numpy::IntoPyArray;
use pyo3::prelude::*;

/// Python function to perform star prerouting on a SabreDAG.
/// This function processes star blocks and updates the DAG and qubit mapping.
#[pyfunction]
#[pyo3(text_signature = "(dag, blocks, processing_order, /)")]
fn star_preroute(
    py: Python,
    dag: &mut SabreDAG,
    blocks: Vec<Block>,
    processing_order: Vec<Nodes>,
) -> (SwapMap, PyObject, NodeBlockResults, PyObject) {
    let mut qubit_mapping: Vec<usize> = (0..dag.num_qubits).collect();
    let mut processed_block_ids: HashSet<usize> = HashSet::with_capacity(blocks.len());
    let last_2q_gate = processing_order.iter().rev().find(|node| node.1.len() == 2);
    let mut is_first_star = true;

    // Structures for SabreResult
    let mut out_map: HashMap<usize, Vec<[PhysicalQubit; 2]>> =
        HashMap::with_capacity(dag.dag.node_count());
    let mut gate_order: Vec<usize> = Vec::with_capacity(dag.dag.node_count());
    let node_block_results: HashMap<usize, Vec<BlockResult>> = HashMap::new();

    // Create a HashMap to store the node-to-block mapping
    let mut node_to_block: HashMap<usize, usize> = HashMap::with_capacity(processing_order.len());
    for (block_id, block) in blocks.iter().enumerate() {
        for node in &block.1 {
            node_to_block.insert(node.0, block_id);
        }
    }
    // Store nodes where swaps will be placed.
    let mut swap_locations: Vec<&Nodes> = Vec::with_capacity(processing_order.len());

    // Process blocks, gathering swap locations and updating the gate order
    for node in &processing_order {
        if let Some(&block_id) = node_to_block.get(&node.0) {
            // Skip if the block has already been processed
            if !processed_block_ids.insert(block_id) {
                continue;
            }
            process_block(
                &blocks[block_id],
                last_2q_gate,
                &mut is_first_star,
                &mut gate_order,
                &mut swap_locations,
            );
        } else {
            // Apply operation for nodes not part of any block
            gate_order.push(node.0);
        }
    }

    // Apply the swaps based on the gathered swap locations and gate order
    for (index, node_id) in gate_order.iter().enumerate() {
        for swap_location in &swap_locations {
            if *node_id == swap_location.0 {
                if let Some(next_node_id) = gate_order.get(index + 1) {
                    apply_swap(
                        &mut qubit_mapping,
                        &swap_location.1,
                        *next_node_id,
                        &mut out_map,
                    );
                }
            }
        }
    }

    let res = SabreResult {
        map: SwapMap { map: out_map },
        node_order: gate_order,
        node_block_results: NodeBlockResults {
            results: node_block_results,
        },
    };

    let final_res = (
        res.map,
        res.node_order.into_pyarray(py).into_any().unbind(),
        res.node_block_results,
        qubit_mapping.into_pyarray(py).into_any().unbind(),
    );

    final_res
}

/// Processes a star block, applying operations and handling swaps.
///
/// Args:
///
/// * `block` - A tuple containing a boolean indicating the presence of a center and a vector of nodes representing the star block.
/// * `last_2q_gate` - The last two-qubit gate in the processing order.
/// * `is_first_star` - A mutable reference to a boolean indicating if this is the first star block being processed.
/// * `gate_order` - A mutable reference to the gate order vector.
/// * `swap_locations` - A mutable reference to the nodes where swaps will be placed after
fn process_block<'a>(
    block: &'a Block,
    last_2q_gate: Option<&'a Nodes>,
    is_first_star: &mut bool,
    gate_order: &mut Vec<usize>,
    swap_locations: &mut Vec<&'a Nodes>,
) {
    let (has_center, sequence) = block;

    // If the block contains exactly 2 nodes, apply them directly
    if sequence.len() == 2 {
        for inner_node in sequence {
            gate_order.push(inner_node.0);
        }
        return;
    }

    let mut prev_qargs = None;
    let mut swap_source = false;

    // Process each node in the block
    for inner_node in sequence.iter() {
        // Apply operation directly if it's a single-qubit operation or the same as previous qargs
        if inner_node.1.len() == 1 || prev_qargs == Some(&inner_node.1) {
            gate_order.push(inner_node.0);
            continue;
        }

        // If this is the first star and no swap source has been identified, set swap_source
        if *is_first_star && !swap_source {
            swap_source = *has_center;
            gate_order.push(inner_node.0);
            prev_qargs = Some(&inner_node.1);
            continue;
        }

        // Place 2q-gate and subsequent swap gate
        gate_order.push(inner_node.0);

        if inner_node != last_2q_gate.unwrap() && inner_node.1.len() == 2 {
            swap_locations.push(inner_node);
        }
        prev_qargs = Some(&inner_node.1);
    }
    *is_first_star = false;
}

/// Applies a swap operation to the DAG and updates the qubit mapping.
///
/// # Args:
///
/// * `qubit_mapping` - A mutable reference to the qubit mapping vector.
/// * `qargs` - Qubit indices for the swap operation (node before the swap)
/// * `next_node_id` - ID of the next node in the gate order (node after the swap)
/// * `out_map` - A mutable reference to the output map.
fn apply_swap(
    qubit_mapping: &mut [usize],
    qargs: &[VirtualQubit],
    next_node_id: usize,
    out_map: &mut HashMap<usize, Vec<[PhysicalQubit; 2]>>,
) {
    if qargs.len() == 2 {
        let idx0 = qargs[0].index();
        let idx1 = qargs[1].index();

        // Update the `qubit_mapping` and `out_map` to reflect the swap operation
        qubit_mapping.swap(idx0, idx1);
        out_map.insert(
            next_node_id,
            vec![[
                PhysicalQubit::new(qubit_mapping[idx0].try_into().unwrap()),
                PhysicalQubit::new(qubit_mapping[idx1].try_into().unwrap()),
            ]],
        );
    }
}

pub fn star_prerouting(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(star_preroute))?;
    Ok(())
}

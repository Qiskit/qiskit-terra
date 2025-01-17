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

use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use qiskit_circuit::circuit_data::CircuitData;
use smallvec::SmallVec;

use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::Target;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::Qubit;

#[pyfunction]
fn any_gate_missing_from_target(dag: &DAGCircuit, target: &Target) -> PyResult<bool> {
    #[inline]
    fn is_universal(gate: &PackedInstruction) -> bool {
        matches!(gate.op.name(), "barrier" | "store")
    }

    fn visit_gate(
        target: &Target,
        gate: &PackedInstruction,
        qargs: &[Qubit],
        wire_map: &HashMap<Qubit, PhysicalQubit>,
    ) -> PyResult<bool> {
        let qargs_mapped = SmallVec::from_iter(qargs.iter().map(|q| wire_map[q]));
        if !target.instruction_supported(gate.op.name(), Some(&qargs_mapped)) {
            return Ok(true);
        }

        if gate.op.control_flow() {
            for block in gate.op.blocks() {
                let block_qubits = (0..block.num_qubits()).map(Qubit::new);
                let inner_wire_map = qargs
                    .iter()
                    .zip(block_qubits)
                    .map(|(outer, inner)| (inner, wire_map[outer]))
                    .collect();
                if visit_circuit(target, &block, &inner_wire_map)? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn visit_circuit(
        target: &Target,
        circuit: &CircuitData,
        wire_map: &HashMap<Qubit, PhysicalQubit>,
    ) -> PyResult<bool> {
        for gate in circuit.iter() {
            if is_universal(gate) {
                continue;
            }
            let qargs = circuit.qargs_interner().get(gate.qubits);
            if visit_gate(target, gate, qargs, wire_map)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    // In the outer DAG, virtual and physical bits are the same thing.
    let wire_map: HashMap<Qubit, PhysicalQubit> = HashMap::from_iter(
        (0..dag.num_qubits()).map(|i| (Qubit::new(i), PhysicalQubit::new(i.try_into().unwrap()))),
    );

    // Process the DAG.
    for gate in dag.op_nodes(true) {
        let gate = dag.dag()[gate].unwrap_operation();
        if is_universal(gate) {
            continue;
        }
        let qargs = dag.qargs_interner().get(gate.qubits);
        if visit_gate(target, gate, qargs, &wire_map)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[pyfunction]
fn any_gate_missing_from_basis(
    py: Python,
    dag: &DAGCircuit,
    basis: HashSet<String>,
) -> PyResult<bool> {
    for (gate, _) in dag.count_ops(py, true)? {
        if !basis.contains(gate.as_str()) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn gates_in_basis(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(any_gate_missing_from_target))?;
    m.add_wrapped(wrap_pyfunction!(any_gate_missing_from_basis))?;
    Ok(())
}

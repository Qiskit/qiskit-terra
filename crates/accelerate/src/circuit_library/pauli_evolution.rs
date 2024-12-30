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

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{multiply_param, radd_param, Param, PyInstruction, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{imports, Clbit, Qubit};
use smallvec::{smallvec, SmallVec};

// custom types for a more readable code
type StandardInstruction = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);
type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

/// Return instructions (using only StandardGate operations) to implement a Pauli evolution
/// of a given Pauli string over a given time (as Param).
///
/// Args:
///     pauli: The Pauli string, e.g. "IXYZ".
///     indices: The qubit indices the Pauli acts on, e.g. if given as [0, 1, 2, 3] with the
///         Pauli "IXYZ", then the correspondence is I_0 X_1 Y_2 Z_3.
///     time: The rotation angle. Note that this will directly be used as input of the
///         rotation gate and not be multiplied by a factor of 2 (that should be done before so
///         that this function can remain Rust-only).
///     phase_gate: If ``true``, use the ``PhaseGate`` instead of ``RZGate`` as single-qubit rotation.
///     do_fountain: If ``true``, implement the CX propagation as "fountain" shape, where each
///         CX uses the top qubit as target. If ``false``, uses a "chain" shape, where CX in between
///         neighboring qubits are used.
///
/// Returns:
///     A pointer to an iterator over standard instructions.
pub fn pauli_evolution<'a>(
    pauli: &'a str,
    indices: Vec<usize>,
    time: Param,
    phase_gate: bool,
    do_fountain: bool,
) -> Box<dyn Iterator<Item = StandardInstruction> + 'a> {
    // ensure the Pauli has no identity terms
    let binding = pauli.to_lowercase(); // lowercase for convenience
    let active = binding
        .as_str()
        .chars()
        .zip(indices)
        .filter(|(pauli, _)| *pauli != 'i');
    let (paulis, indices): (Vec<char>, Vec<usize>) = active.unzip();

    match (phase_gate, indices.len()) {
        (_, 0) => Box::new(std::iter::empty()),
        (false, 1) => Box::new(single_qubit_evolution(paulis[0], indices[0], time)),
        (false, 2) => two_qubit_evolution(paulis, indices, time),
        _ => Box::new(multi_qubit_evolution(
            paulis,
            indices,
            time,
            phase_gate,
            do_fountain,
        )),
    }
}

/// Implement a single-qubit Pauli evolution of a Pauli given as char, on a given index and
/// for given time. Note that the time here equals the angle of the rotation and is not
/// multiplied by a factor of 2.
fn single_qubit_evolution(
    pauli: char,
    index: usize,
    time: Param,
) -> impl Iterator<Item = StandardInstruction> {
    let qubit: SmallVec<[Qubit; 2]> = smallvec![Qubit::new(index)];
    let param: SmallVec<[Param; 3]> = smallvec![time];

    std::iter::once(match pauli {
        'x' => (StandardGate::RXGate, param, qubit),
        'y' => (StandardGate::RYGate, param, qubit),
        'z' => (StandardGate::RZGate, param, qubit),
        _ => unreachable!("Unsupported Pauli, at this point we expected one of x, y, z."),
    })
}

/// Implement a 2-qubit Pauli evolution of a Pauli string, on a given indices and
/// for given time. Note that the time here equals the angle of the rotation and is not
/// multiplied by a factor of 2.
///
/// If possible, Qiskit's native 2-qubit Pauli rotations are used. Otherwise, the general
/// multi-qubit evolution is called.
fn two_qubit_evolution<'a>(
    pauli: Vec<char>,
    indices: Vec<usize>,
    time: Param,
) -> Box<dyn Iterator<Item = StandardInstruction> + 'a> {
    let qubits: SmallVec<[Qubit; 2]> = smallvec![Qubit::new(indices[0]), Qubit::new(indices[1])];
    let param: SmallVec<[Param; 3]> = smallvec![time.clone()];
    let paulistring: String = pauli.iter().collect();

    match paulistring.as_str() {
        "xx" => Box::new(std::iter::once((StandardGate::RXXGate, param, qubits))),
        "zx" => Box::new(std::iter::once((StandardGate::RZXGate, param, qubits))),
        "yy" => Box::new(std::iter::once((StandardGate::RYYGate, param, qubits))),
        "zz" => Box::new(std::iter::once((StandardGate::RZZGate, param, qubits))),
        // Note: the CX modes (do_fountain=true/false) give the same circuit for a 2-qubit
        // Pauli, so we just set it to false here
        _ => Box::new(multi_qubit_evolution(pauli, indices, time, false, false)),
    }
}

/// Implement a multi-qubit Pauli evolution. See ``pauli_evolution`` detailed docs.
fn multi_qubit_evolution(
    pauli: Vec<char>,
    indices: Vec<usize>,
    time: Param,
    phase_gate: bool,
    do_fountain: bool,
) -> impl Iterator<Item = StandardInstruction> {
    let active_paulis: Vec<(char, Qubit)> = pauli
        .into_iter()
        .zip(indices.into_iter().map(Qubit::new))
        .collect();

    // get the basis change: x -> HGate, y -> SXdgGate, z -> nothing
    let basis_change: Vec<StandardInstruction> = active_paulis
        .iter()
        .filter(|(p, _)| *p != 'z')
        .map(|(p, q)| match p {
            'x' => (StandardGate::HGate, smallvec![], smallvec![*q]),
            'y' => (StandardGate::SXGate, smallvec![], smallvec![*q]),
            _ => unreachable!("Invalid Pauli string."), // "z" and "i" have been filtered out
        })
        .collect();

    // get the inverse basis change
    let inverse_basis_change: Vec<StandardInstruction> = basis_change
        .iter()
        .map(|(gate, _, qubit)| match gate {
            StandardGate::HGate => (StandardGate::HGate, smallvec![], qubit.clone()),
            StandardGate::SXGate => (StandardGate::SXdgGate, smallvec![], qubit.clone()),
            _ => unreachable!("Invalid basis-changing Clifford."),
        })
        .collect();

    // get the CX propagation up to the first qubit, and down
    let (chain_up, chain_down) = match do_fountain {
        true => (
            cx_fountain(active_paulis.clone()),
            cx_fountain(active_paulis.clone()).rev(),
        ),
        false => (
            cx_chain(active_paulis.clone()),
            cx_chain(active_paulis.clone()).rev(),
        ),
    };

    // get the RZ gate on the first qubit
    let first_qubit = active_paulis.first().unwrap().1;
    let z_rotation = std::iter::once((
        if phase_gate {
            StandardGate::PhaseGate
        } else {
            StandardGate::RZGate
        },
        smallvec![time],
        smallvec![first_qubit],
    ));

    // and finally chain everything together
    basis_change
        .into_iter()
        .chain(chain_down)
        .chain(z_rotation)
        .chain(chain_up)
        .chain(inverse_basis_change)
}

/// Implement a Pauli evolution circuit.
///
/// The Pauli evolution is implemented as a basis transformation to the Pauli-Z basis,
/// followed by a CX-chain and then a single Pauli-Z rotation on the last qubit. Then the CX-chain
/// is uncomputed and the inverse basis transformation applied. E.g. for the evolution under the
/// Pauli string XIYZ we have the circuit
///                 ┌───┐┌───────┐┌───┐
/// 0: ─────────────┤ X ├┤ Rz(2) ├┤ X ├───────────
///    ┌──────┐┌───┐└─┬─┘└───────┘└─┬─┘┌───┐┌────┐
/// 1: ┤ √Xdg ├┤ X ├──■─────────────■──┤ X ├┤ √X ├
///    └──────┘└─┬─┘                   └─┬─┘└────┘
/// 2: ──────────┼───────────────────────┼────────
///     ┌───┐    │                       │  ┌───┐
/// 3: ─┤ H ├────■───────────────────────■──┤ H ├─
///     └───┘                               └───┘
///
/// Args:
///     num_qubits: The number of qubits in the Hamiltonian.
///     sparse_paulis: The Paulis to implement. Given in a sparse-list format with elements
///         ``(pauli_string, qubit_indices, coefficient)``. An element of the form
///         ``("IXYZ", [0,1,2,3], 0.2)``, for example, is interpreted in terms of qubit indices as
///          I_q0 X_q1 Y_q2 Z_q3 and will use a RZ rotation angle of 0.4.
///     insert_barriers: If ``true``, insert a barrier in between the evolution of individual
///         Pauli terms.
///     do_fountain: If ``true``, implement the CX propagation as "fountain" shape, where each
///         CX uses the top qubit as target. If ``false``, uses a "chain" shape, where CX in between
///         neighboring qubits are used.
///
/// Returns:
///     Circuit data for to implement the evolution.
#[pyfunction]
#[pyo3(name = "pauli_evolution", signature = (num_qubits, sparse_paulis, insert_barriers=false, do_fountain=false))]
pub fn py_pauli_evolution(
    num_qubits: usize,
    sparse_paulis: &Bound<PyList>,
    insert_barriers: bool,
    do_fountain: bool,
) -> PyResult<CircuitData> {
    let py = sparse_paulis.py();
    let num_paulis = sparse_paulis.len();
    let mut paulis: Vec<String> = Vec::with_capacity(num_paulis);
    let mut indices: Vec<Vec<usize>> = Vec::with_capacity(num_paulis);
    let mut times: Vec<Param> = Vec::with_capacity(num_paulis);
    let mut global_phase = Param::Float(0.0);
    let mut modified_phase = false; // keep track of whether we modified the phase

    for el in sparse_paulis.iter() {
        let tuple = el.downcast::<PyTuple>()?;
        let pauli = tuple.get_item(0)?.downcast::<PyString>()?.to_string();
        let time = Param::extract_no_coerce(&tuple.get_item(2)?)?;

        if pauli.as_str().chars().all(|p| p == 'i') {
            global_phase = radd_param(global_phase, time, py);
            modified_phase = true;
            continue;
        }

        paulis.push(pauli);
        times.push(time); // note we do not multiply by 2 here, this is done Python side!
        indices.push(tuple.get_item(1)?.extract::<Vec<usize>>()?)
    }

    let barrier = get_barrier(py, num_qubits);

    let evos = paulis.iter().enumerate().zip(indices).zip(times).flat_map(
        |(((i, pauli), qubits), time)| {
            let as_packed = pauli_evolution(pauli, qubits, time, false, do_fountain).map(
                |(gate, params, qubits)| -> PyResult<Instruction> {
                    Ok((gate.into(), params, Vec::from_iter(qubits), Vec::new()))
                },
            );

            // this creates an iterator containing a barrier only if required, otherwise it is empty
            let maybe_barrier = (insert_barriers && i < (num_paulis - 1))
                .then_some(Ok(barrier.clone()))
                .into_iter();
            as_packed.chain(maybe_barrier)
        },
    );

    // When handling all-identity Paulis above, we added the time as global phase.
    // However, the all-identity Paulis should add a negative phase, as they implement
    // exp(-i t I). We apply the negative sign here, to only do a single (-1) multiplication,
    // instead of doing it every time we find an all-identity Pauli.
    if modified_phase {
        global_phase = multiply_param(&global_phase, -1.0, py);
    }

    CircuitData::from_packed_operations(py, num_qubits, 0, evos, global_phase)
}

/// Build a CX chain over the active qubits. E.g. with q_1 inactive, this would return
///
///                    ┌───┐
///     q_0: ──────────┤ X ├
///                    └─┬─┘
///     q_1: ────────────┼──
///               ┌───┐  │
///     q_2: ─────┤ X ├──■──
///          ┌───┐└─┬─┘
///     q_3: ┤ X ├──■───────
///          └─┬─┘
///     q_4: ──■────────────
///
fn cx_chain(
    active_paulis: Vec<(char, Qubit)>,
) -> Box<dyn DoubleEndedIterator<Item = StandardInstruction>> {
    let num_terms = active_paulis.len();
    Box::new(
        (0..num_terms - 1)
            .map(move |i| (active_paulis[i].1, active_paulis[i + 1].1))
            .map(|(target, ctrl)| (StandardGate::CXGate, smallvec![], smallvec![ctrl, target])),
    )
}

/// Build a CX fountain over the active qubits. E.g. with q_1 inactive, this would return
///
///         ┌───┐┌───┐┌───┐
///    q_0: ┤ X ├┤ X ├┤ X ├
///         └─┬─┘└─┬─┘└─┬─┘
///    q_1: ──┼────┼────┼──
///           │    │    │
///    q_2: ──■────┼────┼──
///                │    │
///    q_3: ───────■────┼──
///                     │
///    q_4: ────────────■──
///
fn cx_fountain(
    active_paulis: Vec<(char, Qubit)>,
) -> Box<dyn DoubleEndedIterator<Item = StandardInstruction>> {
    let num_terms = active_paulis.len();
    let first_qubit = active_paulis[0].1;
    Box::new((1..num_terms).rev().map(move |i| {
        let ctrl = active_paulis[i].1;
        (
            StandardGate::CXGate,
            smallvec![],
            smallvec![ctrl, first_qubit],
        )
    }))
}

fn get_barrier(py: Python, num_qubits: usize) -> Instruction {
    let barrier_cls = imports::BARRIER.get_bound(py);
    let barrier = barrier_cls
        .call1((num_qubits,))
        .expect("Could not create Barrier Python-side");
    let barrier_inst = PyInstruction {
        qubits: num_qubits.try_into().unwrap(),
        clbits: 0,
        params: 0,
        op_name: "barrier".to_string(),
        control_flow: false,
        instruction: barrier.into(),
    };
    (
        barrier_inst.into(),
        smallvec![],
        (0..num_qubits).map(Qubit::new).collect(),
        vec![],
    )
}

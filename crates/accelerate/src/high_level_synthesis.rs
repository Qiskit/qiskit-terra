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

use hashbrown::HashMap;
use pyo3::prelude::*;

/// Track global qubits by their state.
/// The global qubits are numbered by consecutive integers starting at `0`,
/// and the states are distinguished into clean (:math:`|0\rangle`)
/// and dirty (unknown).
#[pyclass]
#[derive(Clone, Debug)]
pub struct QubitTracker {
    /// The total number of global qubits
    num_qubits: usize,
    /// Stores the state for each qubit: `true` means clean, `false` means dirty
    state: Vec<bool>,
    /// Stores whether qubits are allowed be used
    enabled: Vec<bool>,
    /// Used internally for keeping the computations in `O(n)`
    ignored: Vec<bool>,
}

#[pymethods]
impl QubitTracker {
    #[new]
    pub fn new(num_qubits: usize) -> Self {
        QubitTracker {
            num_qubits,
            state: vec![false; num_qubits],
            enabled: vec![true; num_qubits],
            ignored: vec![false; num_qubits],
        }
    }

    /// Sets state of the given qubits to dirty
    fn set_dirty(&mut self, qubits: Vec<usize>) {
        for q in qubits {
            self.state[q] = false;
        }
    }

    /// Sets state of the given qubits to clean
    fn set_clean(&mut self, qubits: Vec<usize>) {
        for q in qubits {
            self.state[q] = true;
        }
    }

    /// Disables using the given qubits
    fn disable(&mut self, qubits: Vec<usize>) {
        for q in qubits {
            self.enabled[q] = false;
        }
    }

    /// Enable using the given qubits
    fn enable(&mut self, qubits: Vec<usize>) {
        for q in qubits {
            self.enabled[q] = true;
        }
    }

    /// Returns the number of enabled clean qubits, ignoring the given qubits
    /// ToDo: check if it's faster to avoid using ignored
    fn num_clean(&mut self, ignored_qubits: Vec<usize>) -> usize {
        for q in &ignored_qubits {
            self.ignored[*q] = true;
        }

        let count = (0..self.num_qubits)
            .filter(|q| !self.ignored[*q] && self.enabled[*q] && self.state[*q])
            .count();

        for q in &ignored_qubits {
            self.ignored[*q] = false;
        }

        count
    }

    /// Returns the number of enabled dirty qubits, ignoring the given qubits
    /// ToDo: check if it's faster to avoid using ignored
    fn num_dirty(&mut self, ignored_qubits: Vec<usize>) -> usize {
        for q in &ignored_qubits {
            self.ignored[*q] = true;
        }

        let count = (0..self.num_qubits)
            .filter(|q| !self.ignored[*q] && self.enabled[*q] && !self.state[*q])
            .count();

        for q in &ignored_qubits {
            self.ignored[*q] = false;
        }

        count
    }

    /// Get `num_qubits` enabled qubits, excluding `ignored_qubits`, and returning the
    /// clean qubits first.
    /// ToDo: check if it's faster to avoid using ignored
    fn borrow(&mut self, num_qubits: usize, ignored_qubits: Vec<usize>) -> Vec<usize> {
        for q in &ignored_qubits {
            self.ignored[*q] = true;
        }

        let clean_ancillas = (0..self.num_qubits)
            .filter(|q| !self.ignored[*q] && self.enabled[*q] && self.state[*q]);
        let dirty_ancillas = (0..self.num_qubits)
            .filter(|q| !self.ignored[*q] && self.enabled[*q] && !self.state[*q]);
        let out: Vec<usize> = clean_ancillas
            .chain(dirty_ancillas)
            .take(num_qubits)
            .collect();

        for q in &ignored_qubits {
            self.ignored[*q] = false;
        }
        out
    }

    /// Copies the contents
    fn copy(&self) -> Self {
        QubitTracker {
            num_qubits: self.num_qubits,
            state: self.state.clone(),
            enabled: self.enabled.clone(),
            ignored: self.ignored.clone(),
        }
    }

    /// Replaces the state of the given qubits by their state in the `other` tracker
    fn replace_state(&mut self, other: QubitTracker, qubits: Vec<usize>) {
        for q in qubits {
            self.state[q] = other.state[q]
        }
    }

    /// Pretty-prints
    pub fn __str__(&self) -> String {
        let mut out = String::from("QubitTracker(");
        for q in 0..self.num_qubits {
            out.push_str(&q.to_string());
            out.push(':');
            out.push(' ');
            if !self.enabled[q] {
                out.push('_');
            } else if self.state[q] {
                out.push('0');
            } else {
                out.push('*');
            }
            if q != self.num_qubits - 1 {
                out.push(';');
                out.push(' ');
            } else {
                out.push(')');
            }
        }
        out
    }
}

/// Correspondence between local qubits and global qubits.
///
/// An internal class for handling recursion within `HighLevelSynthesis`.
/// Provides correspondence between the qubit indices of an internal DAG,
/// aka the "local qubits" (for instance, of the definition circuit
/// of a custom gate), and the qubit indices of the original DAG, aka the
/// "global qubits".
///
/// Since the local qubits are consecutive integers starting at zero,
/// i.e. `0`, `1`, `2`, etc., the correspondence is kept using a vector, with
/// the entry in position `k` representing the global qubit that corresponds
/// to the local qubit `k`.
#[pyclass]
#[derive(Clone, Debug)]
pub struct QubitContext {
    /// Mapping from local indices to global indices
    local_to_global: Vec<usize>,
}

#[pymethods]
impl QubitContext {
    #[new]
    pub fn new(local_to_global: Vec<usize>) -> Self {
        QubitContext { local_to_global }
    }

    /// Returns the number of local qubits
    fn num_qubits(&self) -> usize {
        self.local_to_global.len()
    }

    /// Extends the correspondence by an additional qubit that
    /// maps to the given global qubit. Returns the index of the
    /// new local qubit.
    fn add_qubit(&mut self, global_qubit: usize) -> usize {
        let new_local_qubit = self.local_to_global.len();
        self.local_to_global.push(global_qubit);
        new_local_qubit
    }

    /// Returns the local-to-global mapping
    fn to_global_mapping(&self) -> Vec<usize> {
        self.local_to_global.clone()
    }

    /// Returns the global-to-local mapping
    fn to_local_mapping(&self) -> HashMap<usize, usize> {
        HashMap::from_iter(
            self.local_to_global
                .iter()
                .enumerate()
                .map(|(i, j)| (*j, i)),
        )
    }

    /// Restricts the context to a subset of qubits, remapping the indices
    /// to be consecutive integers starting at zero.
    fn restrict(&self, qubits: Vec<usize>) -> Self {
        QubitContext {
            local_to_global: qubits.iter().map(|q| self.local_to_global[*q]).collect(),
        }
    }

    /// Returns the global qubits corresponding to the given local qubit
    fn to_global(&self, qubit: usize) -> usize {
        self.local_to_global[qubit]
    }

    /// Returns the global qubits corresponding to the given local qubit
    fn to_globals(&self, qubits: Vec<usize>) -> Vec<usize> {
        qubits.iter().map(|q| self.local_to_global[*q]).collect()
    }
}

pub fn high_level_synthesis_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<QubitTracker>()?;
    m.add_class::<QubitContext>()?;
    Ok(())
}

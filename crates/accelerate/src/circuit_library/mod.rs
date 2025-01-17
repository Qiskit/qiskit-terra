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

mod blocks;
mod entanglement;
mod iqp;
mod multi_local;
mod parameter_ledger;
mod pauli_evolution;
mod pauli_feature_map;
mod quantum_volume;

pub fn circuit_library(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(pauli_evolution::py_pauli_evolution))?;
    m.add_wrapped(wrap_pyfunction!(pauli_feature_map::pauli_feature_map))?;
    m.add_wrapped(wrap_pyfunction!(entanglement::get_entangler_map))?;
    m.add_wrapped(wrap_pyfunction!(iqp::py_iqp))?;
    m.add_wrapped(wrap_pyfunction!(iqp::py_random_iqp))?;
    m.add_wrapped(wrap_pyfunction!(quantum_volume::quantum_volume))?;
    m.add_wrapped(wrap_pyfunction!(multi_local::py_n_local))?;
    m.add_class::<blocks::Block>()?;

    Ok(())
}

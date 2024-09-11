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

mod clifford;
pub mod linear;
pub mod linear_phase;
mod permutation;

use pyo3::prelude::*;

pub fn synthesis(m: &Bound<PyModule>) -> PyResult<()> {
    let linear_mod = PyModule::new_bound(m.py(), "linear")?;
    linear::linear(&linear_mod)?;
    m.add_submodule(&linear_mod)?;

    let linear_phase_mod = PyModule::new_bound(m.py(), "linear_phase")?;
    linear_phase::linear_phase(&linear_phase_mod)?;
    m.add_submodule(&linear_phase_mod)?;

    let permutation_mod = PyModule::new_bound(m.py(), "permutation")?;
    permutation::permutation(&permutation_mod)?;
    m.add_submodule(&permutation_mod)?;

    let clifford_mod = PyModule::new_bound(m.py(), "clifford")?;
    clifford::clifford(&clifford_mod)?;
    m.add_submodule(&clifford_mod)?;

    Ok(())
}

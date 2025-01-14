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

use crate::synthesis::linear::pmh::synth_pmh;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::{prelude::*, types::PyList};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::Qubit;
use smallvec::{smallvec, SmallVec};
use std::f64::consts::PI;
type Instruction = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>);

fn get_instr(angle: String, qubit_idx: usize) -> Option<Instruction> {
    Some(match angle.as_str() {
        "t" => (
            StandardGate::TGate,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        "tdg" => (
            StandardGate::TdgGate,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        "s" => (
            StandardGate::SGate,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        "sdg" => (
            StandardGate::SdgGate,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        "z" => (
            StandardGate::ZGate,
            smallvec![],
            smallvec![Qubit(qubit_idx as u32)],
        ),
        angles_in_pi => (
            StandardGate::PhaseGate,
            smallvec![Param::Float((angles_in_pi.parse::<f64>().ok()?) % PI)],
            smallvec![Qubit(qubit_idx as u32)],
        ),
    })
}

/// This function is an implementation of the `GraySynth` algorithm which is a heuristic
/// algorithm described in detail in section 4 of [1] for synthesizing small parity networks.
/// It is inspired by Gray codes. Given a set of binary strings :math:`S`
/// (called ``cnots`` bellow), the algorithm synthesizes a parity network for :math:`S` by
/// repeatedly choosing an index :math:`i` to expand and then effectively recursing on
/// the co-factors :math:`S_0` and :math:`S_1`, consisting of the strings :math:`y \in S`,
/// with :math:`y_i = 0` or :math:`1` respectively. As a subset :math:`S` is recursively expanded,
/// ``cx`` gates are applied so that a designated target bit contains the
/// (partial) parity :math:`\chi_y(x)` where :math:`y_i = 1` if and only if :math:`y'_i = 1` for all
/// :math:`y' \in S`. If :math:`S` contains a single element :math:`\{y'\}`, then :math:`y = y'`,
/// and the target bit contains the value :math:`\chi_{y'}(x)` as desired.
/// Notably, rather than uncomputing this sequence of ``cx`` (CNOT) gates when a subset :math:`S`
/// is finished being synthesized, the algorithm maintains the invariant that the remaining
/// parities to be computed are expressed over the current state of bits. This allows the algorithm
/// to avoid the 'backtracking' inherent in uncomputing-based methods.
/// References:
///        1. Matthew Amy, Parsiad Azimzadeh, and Michele Mosca.
///           *On the controlled-NOT complexity of controlled-NOT–phase circuits.*,
///           Quantum Science and Technology 4.1 (2018): 015002.
///           `arXiv:1712.01859 <https://arxiv.org/abs/1712.01859>`_
#[pyfunction]
#[pyo3(signature = (binary_string, angles, section_size=2))]
pub fn synth_cnot_phase_aam(
    py: Python,
    binary_string: PyReadonlyArray2<bool>,
    angles: &Bound<PyList>,
    section_size: Option<i64>,
) -> PyResult<CircuitData> {
    // converting to Option<usize>
    let section_size: Option<usize> =
        section_size.and_then(|num| if num >= 0 { Some(num as usize) } else { None });
    let binary_string = binary_string.as_array().to_owned();
    let num_qubits = binary_string.nrows();

    let mut angles = angles
        .iter()
        .filter_map(|data| {
            data.extract::<String>()
                .or_else(|_| data.extract::<f64>().map(|f| f.to_string()))
                .ok()
        })
        .collect::<Vec<String>>();

    let mut state: Array2<bool> = Array2::from_shape_fn((num_qubits, num_qubits), |(i, j)| i == j);

    let mut binary_str_cpy = binary_string.clone();
    let num_qubits_range = (0..num_qubits).collect::<Vec<usize>>();
    let mut data_stack = vec![(binary_string.clone(), num_qubits_range.clone(), num_qubits)];
    let mut bin_str_ = binary_string;
    let mut qubits_range_ = num_qubits_range;
    let mut qubits_ = num_qubits;

    // The aim is to keep the instructions as iterables
    // Below variables are keeping the state of iteration in
    // `std::iter::from_fn`
    let mut keep_iterating: bool = true;
    let mut cx_gate_done: bool = false;
    let mut phase_loop_on: bool = false;
    let mut phase_done: bool = false;
    let mut cx_phase_done: bool = false;
    let mut qubit_idx: usize = 0;
    let mut index: usize = 0;
    let mut pmh_init_done: bool = false;
    let mut synth_pmh_iter: Option<Box<dyn Iterator<Item = Instruction>>> = None;

    // The instructions are stored as iterables in the variable.
    let cx_phase_iter = std::iter::from_fn(move || {
        // For now, the `gate_instr` stores `None`, but as soon as
        // it gets any instruction, the current loop breaks, and
        // the variable is returned to be stored in `cx_phase_iter`
        // as an iterable.
        let mut gate_instr: Option<Instruction> = None;

        // This while loop corresponds to the first block of code where
        // different phases are applied to the qubits.
        // The variable `phase_done` is a bool switch which is initially
        // set to `false`, so that this while loop gets executed, different
        // phases are applied to the qubits, and then `phase_done` is set
        // to `true` indicating the phases have been applied to the qubits, so
        // the control doesn't enter the while loop again in the execution.
        while !phase_done {
            let bin_str_subset_ = binary_str_cpy.column(index);
            index += 1;
            let target_state = state.row(qubit_idx);

            if bin_str_subset_ == target_state {
                index -= 1;
                binary_str_cpy.remove_index(numpy::ndarray::Axis(1), index);
                let angle = angles.remove(index);
                // instruction applied on the `qubit_idx`
                // according to the `angle` passed.
                gate_instr = get_instr(angle, qubit_idx);
                // as soon, as `gate_instr` gets an instruction, the
                // loop is broken, so that `gate_instr` returns the
                // instruction as iterable to `cx_phase_iter`.
                break;
            }
            if index == binary_str_cpy.ncols() {
                qubit_idx += 1;
                index = 0;
            }
            if qubit_idx == num_qubits {
                phase_done = true;
                index = 0;
                qubit_idx = 0;
            }
        } // end phase applying loop

        // This while loop corresponds to the second block of code
        // where CNOT gate with different phases are applied to the qubits.
        // After the first block is executed `phase_done` is set to true,
        // similarly, `cx_phase_done` is a bool variable indicating if the
        // second block of code which applies CNOTs is complete.
        'outer_loop: while phase_done && !cx_phase_done {
            // Wherever, `gate_instr` gets any instruciton, to return
            // the instruction, the loop has to be broken prematurely,
            // and, the control again strats from line after `std::iter::from_fn`
            // the variable `phase_loop_on` indicates that the loop has been
            // broken prematurely, so that the code blocks which are supposed
            // to get executed after the completion of the loop are not executed.
            if !phase_loop_on && data_stack.is_empty() {
                // If `data_stack` is empty, this marks successfull
                // application of CNOTs, so `cx_phase_done` is set to indicate
                // the second code block applying CNOTs is complete, and, thus
                // the loop gets broken finally.
                cx_phase_done = true;
                break 'outer_loop;
            }
            if !phase_loop_on {
                (bin_str_, qubits_range_, qubits_) = data_stack.pop().unwrap();
            }
            if !phase_loop_on && bin_str_.is_empty() {
                continue 'outer_loop;
            }

            // For every qubit less than `num_qubits`, the condition that all
            // bits of the row corresponding to `qubit_idx` in `bin_str_` is true
            // is checked, even if the condition is fulfilled once, the varibale
            // `keep_iterating` is set, so that after `qubit_idx` reaches `num_qubits`.
            // The entire loop runs at-least once more.
            while keep_iterating && (qubits_ < num_qubits) {
                if !phase_loop_on {
                    keep_iterating = false;
                }
                while qubit_idx < num_qubits {
                    if (qubit_idx != qubits_) && !bin_str_.row(qubit_idx).into_iter().any(|&b| !b) {
                        // To return the `cx` gate stored in `gate_instr` the loop has to be
                        // broke prematurely, the bool `cx_gate_done` indicates that `cx` has been
                        // applied for this particular iteration of loop, so that, when the control
                        // comes back again at this spot, furthur contents of the loop is executed,
                        // and no more `cx` is applied now for this iteration of loop.
                        if !cx_gate_done && !phase_loop_on {
                            keep_iterating = true;
                            cx_gate_done = true;
                            gate_instr = Some((
                                StandardGate::CXGate,
                                smallvec![],
                                smallvec![Qubit((qubit_idx) as u32), Qubit(qubits_ as u32)],
                            ));
                            index = 0;
                            phase_loop_on = true;
                            for col_idx in 0..state.ncols() {
                                state[(qubits_, col_idx)] ^= state[(qubit_idx, col_idx)];
                            }
                            // `gate_instr` contains `cx` gate now, and the loop has to
                            // be borken prematurely to return this gate.
                            break 'outer_loop;
                        }

                        cx_gate_done = false;

                        // This while loop is to apply the phases to qubits after the
                        // `cx` has been applied, again the bool `phase_loop_on` is merely
                        // an indication that the loop is supposed to break prematurely
                        // and whenever the control returns to this block of code it keep
                        // on applying phase until `phase_loop_on` is set to false, which
                        // indicates that all the phases associated to this iteration of loop
                        // has been applied.
                        while phase_loop_on {
                            if index == binary_str_cpy.ncols() {
                                phase_loop_on = false;
                                break;
                            }
                            let bin_str_subset_ = binary_str_cpy.column(index);
                            index += 1;
                            let target_state = state.row(qubits_);
                            if bin_str_subset_ == target_state {
                                index -= 1;
                                binary_str_cpy.remove_index(numpy::ndarray::Axis(1), index);
                                let angle = angles.remove(index);
                                gate_instr = get_instr(angle, qubits_);
                                // loop has to be broken, to return the instruction stored in
                                // `gate_instr`.
                                break 'outer_loop;
                            }
                        }

                        data_stack.push((bin_str_.clone(), qubits_range_.clone(), qubits_));
                        let mut uniq_ele_dat_stack = vec![];

                        for data in data_stack.iter() {
                            if !uniq_ele_dat_stack.contains(data) {
                                let dat_ = (*data).clone();
                                uniq_ele_dat_stack.push(dat_);
                            }
                        }
                        data_stack = uniq_ele_dat_stack;

                        for data in &mut data_stack {
                            let (ref mut temp_bin_str_, _, _) = data;
                            if temp_bin_str_.is_empty() {
                                continue;
                            }

                            for idx in 0..temp_bin_str_.row(qubit_idx).len() {
                                temp_bin_str_[(qubit_idx, idx)] ^= temp_bin_str_[(qubits_, idx)];
                            }
                        }

                        (bin_str_, qubits_range_, qubits_) = data_stack.pop().unwrap();
                    } // end of if qubits_ < num_qubits ...
                    qubit_idx += 1;
                } // end of while check qubit_idx < num_qubits ...
                qubit_idx = 0;
            } // end of while keep iterating ...
            keep_iterating = true;

            if qubits_range_.is_empty() {
                continue 'outer_loop;
            }

            let bin_str_max_0_1: Vec<usize> = bin_str_
                .axis_iter(numpy::ndarray::Axis(0))
                .map(|row| {
                    std::cmp::max(
                        row.iter().filter(|&&x| !x).count(),
                        row.into_iter().filter(|&&x| x).count(),
                    )
                })
                .collect();

            let bin_str_mx_qbt_rng: Vec<usize> = qubits_range_
                .iter()
                .map(|&q_idx| bin_str_max_0_1[q_idx])
                .collect();

            let argmax_ = bin_str_mx_qbt_rng
                .into_iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.cmp(y))
                .map(|(idx, _)| idx)
                .unwrap();

            let argmax_qubit_idx = qubits_range_[argmax_];

            let mut bin_str_subset_0_t = vec![];
            let mut bin_str_subset_1_t = vec![];

            let mut bin_str_subset_0_t_shape = (0_usize, bin_str_.column(0).len());
            let mut bin_str_subset_1_t_shape = (0_usize, 0_usize);
            bin_str_subset_1_t_shape.1 = bin_str_subset_0_t_shape.1;
            for cols in bin_str_.columns() {
                if !cols[argmax_qubit_idx] {
                    bin_str_subset_0_t_shape.0 += 1;
                    bin_str_subset_0_t.append(&mut cols.to_vec());
                } else {
                    bin_str_subset_1_t_shape.0 += 1;
                    bin_str_subset_1_t.append(&mut cols.to_vec());
                }
            }

            let bin_str_subset_0 = Array2::from_shape_vec(
                (bin_str_subset_0_t_shape.0, bin_str_subset_0_t_shape.1),
                bin_str_subset_0_t,
            )
            .unwrap()
            .reversed_axes();
            let bin_str_subset_1 = Array2::from_shape_vec(
                (bin_str_subset_1_t_shape.0, bin_str_subset_1_t_shape.1),
                bin_str_subset_1_t,
            )
            .unwrap()
            .reversed_axes();

            if qubits_ == num_qubits {
                data_stack.push((
                    bin_str_subset_1,
                    qubits_range_
                        .clone()
                        .into_iter()
                        .filter(|&x| x != argmax_qubit_idx)
                        .collect(),
                    argmax_qubit_idx,
                ));
            } else {
                data_stack.push((
                    bin_str_subset_1,
                    qubits_range_
                        .clone()
                        .into_iter()
                        .filter(|&x| x != argmax_qubit_idx)
                        .collect(),
                    qubits_,
                ));
            }
            data_stack.push((
                bin_str_subset_0,
                qubits_range_
                    .clone()
                    .into_iter()
                    .filter(|&x| x != argmax_qubit_idx)
                    .collect(),
                qubits_,
            ));
        } // end 'outer_loop

        // After the phases are applied, then `cx` with phases are applied,
        // now, this is the third block of code which corresponds to appending
        // the output of `synth_pmh` to the iterables stored in `cx_phase_iter`.
        // The bool `pmh_init_done` indicates if the output of `synth_pmh` has been
        // obtained. Since, the size of output from `synth_pmh` depends on the
        // contents of `state` left after applying phases and CNOTs, so the memory
        // required to store the output of `synth_pmh` is not known at compile
        // time, that's why this output which is an iterable has to be stored on the heap!
        if phase_done && cx_phase_done && !pmh_init_done {
            synth_pmh_iter = Some(Box::new(synth_pmh(state.clone(), section_size).rev()));
            // Once, the output has been stored on the heap, setting this
            // bool makes sure the control doesn't enter this if block again.
            pmh_init_done = true;
        }

        // Now, that the output of `synth_pmh` has been stored
        // in `synth_pmh_iter` on the heap, now, all we have to do
        // is to keep calling `next()`, and returning to `cx_phase_iter`.
        // When `next()` yeilds `None` the same is passed to `cx_phase_iter`
        // and that marks the completion of the whole `std::iter::from_fn` logic!
        if pmh_init_done {
            if let Some(ref mut data) = synth_pmh_iter {
                gate_instr = data.next();
            } else {
                gate_instr = None;
            }
        }

        gate_instr
    });

    CircuitData::from_standard_gates(py, num_qubits as u32, cx_phase_iter, Param::Float(0.0))
}

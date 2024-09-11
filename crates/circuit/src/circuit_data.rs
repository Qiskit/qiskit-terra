// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#[cfg(feature = "cache_pygates")]
use std::cell::OnceCell;

use crate::bit_data::BitData;
use crate::circuit_instruction::{CircuitInstruction, OperationFromPython};
use crate::imports::{ANNOTATED_OPERATION, CLBIT, QUANTUM_CIRCUIT, QUBIT};
use crate::interner::{Interned, Interner};
use crate::operations::{Operation, OperationRef, Param, StandardGate};
use crate::packed_instruction::{PackedInstruction, PackedOperation};
use crate::parameter_table::{ParameterTable, ParameterTableError, ParameterUse, ParameterUuid};
use crate::slice::{PySequenceIndex, SequenceIndex};
use crate::{Clbit, Qubit};

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{IntoPyDict, PyDict, PyList, PySet, PyTuple, PyType};
use pyo3::{import_exception, intern, PyTraverseError, PyVisit};

use hashbrown::{HashMap, HashSet};
use indexmap::IndexMap;
use smallvec::SmallVec;

import_exception!(qiskit.circuit.exceptions, CircuitError);

/// A container for :class:`.QuantumCircuit` instruction listings that stores
/// :class:`.CircuitInstruction` instances in a packed form by interning
/// their :attr:`~.CircuitInstruction.qubits` and
/// :attr:`~.CircuitInstruction.clbits` to native vectors of indices.
///
/// Before adding a :class:`.CircuitInstruction` to this container, its
/// :class:`.Qubit` and :class:`.Clbit` instances MUST be registered via the
/// constructor or via :meth:`.CircuitData.add_qubit` and
/// :meth:`.CircuitData.add_clbit`. This is because the order in which
/// bits of the same type are added to the container determines their
/// associated indices used for storage and retrieval.
///
/// Once constructed, this container behaves like a Python list of
/// :class:`.CircuitInstruction` instances. However, these instances are
/// created and destroyed on the fly, and thus should be treated as ephemeral.
///
/// For example,
///
/// .. code-block::
///
///     qubits = [Qubit()]
///     data = CircuitData(qubits)
///     data.append(CircuitInstruction(XGate(), (qubits[0],), ()))
///     assert(data[0] == data[0]) # => Ok.
///     assert(data[0] is data[0]) # => PANICS!
///
/// .. warning::
///
///     This is an internal interface and no part of it should be relied upon
///     outside of Qiskit.
///
/// Args:
///     qubits (Iterable[:class:`.Qubit`] | None): The initial sequence of
///         qubits, used to map :class:`.Qubit` instances to and from its
///         indices.
///     clbits (Iterable[:class:`.Clbit`] | None): The initial sequence of
///         clbits, used to map :class:`.Clbit` instances to and from its
///         indices.
///     data (Iterable[:class:`.CircuitInstruction`]): An initial instruction
///         listing to add to this container. All bits appearing in the
///         instructions in this iterable must also exist in ``qubits`` and
///         ``clbits``.
///     reserve (int): The container's initial capacity. This is reserved
///         before copying instructions into the container when ``data``
///         is provided, so the initialized container's unused capacity will
///         be ``max(0, reserve - len(data))``.
///
/// Raises:
///     KeyError: if ``data`` contains a reference to a bit that is not present
///         in ``qubits`` or ``clbits``.
#[pyclass(sequence, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct CircuitData {
    /// The packed instruction listing.
    data: Vec<PackedInstruction>,
    /// The cache used to intern instruction bits.
    qargs_interner: Interner<[Qubit]>,
    /// The cache used to intern instruction bits.
    cargs_interner: Interner<[Clbit]>,
    /// Qubits registered in the circuit.
    qubits: BitData<Qubit>,
    /// Clbits registered in the circuit.
    clbits: BitData<Clbit>,
    param_table: ParameterTable,
    #[pyo3(get)]
    global_phase: Param,
}

impl CircuitData {
    /// An alternate constructor to build a new `CircuitData` from an iterator
    /// of packed operations. This can be used to build a circuit from a sequence
    /// of `PackedOperation` without needing to involve Python.
    ///
    /// This can be connected with the Python space
    /// QuantumCircuit.from_circuit_data() constructor to build a full
    /// QuantumCircuit from Rust.
    ///
    /// # Arguments
    ///
    /// * py: A GIL handle this is needed to instantiate Qubits in Python space
    /// * num_qubits: The number of qubits in the circuit. These will be created
    ///     in Python as loose bits without a register.
    /// * num_clbits: The number of classical bits in the circuit. These will be created
    ///     in Python as loose bits without a register.
    /// * instructions: An iterator of the (packed operation, params, qubits, clbits) to
    ///     add to the circuit
    /// * global_phase: The global phase to use for the circuit
    pub fn from_packed_operations<I>(
        py: Python,
        num_qubits: u32,
        num_clbits: u32,
        instructions: I,
        global_phase: Param,
    ) -> PyResult<Self>
    where
        I: IntoIterator<
            Item = (
                PackedOperation,
                SmallVec<[Param; 3]>,
                Vec<Qubit>,
                Vec<Clbit>,
            ),
        >,
    {
        let instruction_iter = instructions.into_iter();
        let mut res = Self::with_capacity(
            py,
            num_qubits,
            num_clbits,
            instruction_iter.size_hint().0,
            global_phase,
        )?;
        for (operation, params, qargs, cargs) in instruction_iter {
            let qubits = res.qargs_interner.insert_owned(qargs);
            let clbits = res.cargs_interner.insert_owned(cargs);
            let params = (!params.is_empty()).then(|| Box::new(params));
            res.data.push(PackedInstruction {
                op: operation,
                qubits,
                clbits,
                params,
                extra_attrs: None,
                #[cfg(feature = "cache_pygates")]
                py_op: OnceCell::new(),
            });
            res.track_instruction_parameters(py, res.data.len() - 1)?;
        }
        Ok(res)
    }

    /// An alternate constructor to build a new `CircuitData` from an iterator
    /// of standard gates. This can be used to build a circuit from a sequence
    /// of standard gates, such as for a `StandardGate` definition or circuit
    /// synthesis without needing to involve Python.
    ///
    /// This can be connected with the Python space
    /// QuantumCircuit.from_circuit_data() constructor to build a full
    /// QuantumCircuit from Rust.
    ///
    /// # Arguments
    ///
    /// * py: A GIL handle this is needed to instantiate Qubits in Python space
    /// * num_qubits: The number of qubits in the circuit. These will be created
    ///     in Python as loose bits without a register.
    /// * instructions: An iterator of the standard gate params and qubits to
    ///     add to the circuit
    /// * global_phase: The global phase to use for the circuit
    pub fn from_standard_gates<I>(
        py: Python,
        num_qubits: u32,
        instructions: I,
        global_phase: Param,
    ) -> PyResult<Self>
    where
        I: IntoIterator<Item = (StandardGate, SmallVec<[Param; 3]>, SmallVec<[Qubit; 2]>)>,
    {
        let instruction_iter = instructions.into_iter();
        let mut res = Self::with_capacity(
            py,
            num_qubits,
            0,
            instruction_iter.size_hint().0,
            global_phase,
        )?;
        let no_clbit_index = res.cargs_interner.get_default();
        for (operation, params, qargs) in instruction_iter {
            let qubits = res.qargs_interner.insert(&qargs);
            let params = (!params.is_empty()).then(|| Box::new(params));
            res.data.push(PackedInstruction {
                op: operation.into(),
                qubits,
                clbits: no_clbit_index,
                params,
                extra_attrs: None,
                #[cfg(feature = "cache_pygates")]
                py_op: OnceCell::new(),
            });
            res.track_instruction_parameters(py, res.data.len() - 1)?;
        }
        Ok(res)
    }

    /// Build an empty CircuitData object with an initially allocated instruction capacity
    pub fn with_capacity(
        py: Python,
        num_qubits: u32,
        num_clbits: u32,
        instruction_capacity: usize,
        global_phase: Param,
    ) -> PyResult<Self> {
        let mut res = CircuitData {
            data: Vec::with_capacity(instruction_capacity),
            qargs_interner: Interner::new(),
            cargs_interner: Interner::new(),
            qubits: BitData::new(py, "qubits".to_string()),
            clbits: BitData::new(py, "clbits".to_string()),
            param_table: ParameterTable::new(),
            global_phase,
        };
        if num_qubits > 0 {
            let qubit_cls = QUBIT.get_bound(py);
            for _i in 0..num_qubits {
                let bit = qubit_cls.call0()?;
                res.add_qubit(py, &bit, true)?;
            }
        }
        if num_clbits > 0 {
            let clbit_cls = CLBIT.get_bound(py);
            for _i in 0..num_clbits {
                let bit = clbit_cls.call0()?;
                res.add_clbit(py, &bit, true)?;
            }
        }
        Ok(res)
    }

    /// Append a standard gate to this CircuitData
    pub fn push_standard_gate(
        &mut self,
        operation: StandardGate,
        params: &[Param],
        qargs: &[Qubit],
    ) -> PyResult<()> {
        let no_clbit_index = self.cargs_interner.get_default();
        let params = (!params.is_empty()).then(|| Box::new(params.iter().cloned().collect()));
        let qubits = self.qargs_interner.insert(qargs);
        self.data.push(PackedInstruction {
            op: operation.into(),
            qubits,
            clbits: no_clbit_index,
            params,
            extra_attrs: None,
            #[cfg(feature = "cache_pygates")]
            py_op: OnceCell::new(),
        });
        Ok(())
    }

    /// Add the entries from the `PackedInstruction` at the given index to the internal parameter
    /// table.
    fn track_instruction_parameters(
        &mut self,
        py: Python,
        instruction_index: usize,
    ) -> PyResult<()> {
        for (index, param) in self.data[instruction_index]
            .params_view()
            .iter()
            .enumerate()
        {
            let usage = ParameterUse::Index {
                instruction: instruction_index,
                parameter: index as u32,
            };
            for param_ob in param.iter_parameters(py)? {
                self.param_table.track(&param_ob?, Some(usage))?;
            }
        }
        Ok(())
    }

    /// Remove the entries from the `PackedInstruction` at the given index from the internal
    /// parameter table.
    fn untrack_instruction_parameters(
        &mut self,
        py: Python,
        instruction_index: usize,
    ) -> PyResult<()> {
        for (index, param) in self.data[instruction_index]
            .params_view()
            .iter()
            .enumerate()
        {
            let usage = ParameterUse::Index {
                instruction: instruction_index,
                parameter: index as u32,
            };
            for param_ob in param.iter_parameters(py)? {
                self.param_table.untrack(&param_ob?, usage)?;
            }
        }
        Ok(())
    }

    /// Retrack the entire `ParameterTable`.
    ///
    /// This is necessary each time an insertion or removal occurs on `self.data` other than in the
    /// last position.
    fn reindex_parameter_table(&mut self, py: Python) -> PyResult<()> {
        self.param_table.clear();

        for inst_index in 0..self.data.len() {
            self.track_instruction_parameters(py, inst_index)?;
        }
        for param_ob in self.global_phase.iter_parameters(py)? {
            self.param_table
                .track(&param_ob?, Some(ParameterUse::GlobalPhase))?;
        }
        Ok(())
    }
}

#[pymethods]
impl CircuitData {
    #[new]
    #[pyo3(signature = (qubits=None, clbits=None, data=None, reserve=0, global_phase=Param::Float(0.0)))]
    pub fn new(
        py: Python<'_>,
        qubits: Option<&Bound<PyAny>>,
        clbits: Option<&Bound<PyAny>>,
        data: Option<&Bound<PyAny>>,
        reserve: usize,
        global_phase: Param,
    ) -> PyResult<Self> {
        let mut self_ = CircuitData {
            data: Vec::new(),
            qargs_interner: Interner::new(),
            cargs_interner: Interner::new(),
            qubits: BitData::new(py, "qubits".to_string()),
            clbits: BitData::new(py, "clbits".to_string()),
            param_table: ParameterTable::new(),
            global_phase: Param::Float(0.),
        };
        self_.set_global_phase(py, global_phase)?;
        if let Some(qubits) = qubits {
            for bit in qubits.iter()? {
                self_.add_qubit(py, &bit?, true)?;
            }
        }
        if let Some(clbits) = clbits {
            for bit in clbits.iter()? {
                self_.add_clbit(py, &bit?, true)?;
            }
        }
        if let Some(data) = data {
            self_.reserve(py, reserve);
            self_.extend(py, data)?;
        }
        Ok(self_)
    }

    pub fn __reduce__(self_: &Bound<CircuitData>, py: Python<'_>) -> PyResult<PyObject> {
        let ty: Bound<PyType> = self_.get_type();
        let args = {
            let self_ = self_.borrow();
            (
                self_.qubits.cached().clone_ref(py),
                self_.clbits.cached().clone_ref(py),
                None::<()>,
                self_.data.len(),
            )
        };
        Ok((ty, args, None::<()>, self_.iter()?).into_py(py))
    }

    /// Returns the current sequence of registered :class:`.Qubit` instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`CircuitData` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Qubit`): The current sequence of registered qubits.
    #[getter("qubits")]
    pub fn py_qubits(&self, py: Python<'_>) -> Py<PyList> {
        self.qubits.cached().clone_ref(py)
    }

    /// Return the number of qubits. This is equivalent to the length of the list returned by
    /// :meth:`.CircuitData.qubits`
    ///
    /// Returns:
    ///     int: The number of qubits.
    #[getter]
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Returns the current sequence of registered :class:`.Clbit`
    /// instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`CircuitData` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Clbit`): The current sequence of registered clbits.
    #[getter("clbits")]
    pub fn py_clbits(&self, py: Python<'_>) -> Py<PyList> {
        self.clbits.cached().clone_ref(py)
    }

    /// Return the number of clbits. This is equivalent to the length of the list returned by
    /// :meth:`.CircuitData.clbits`.
    ///
    /// Returns:
    ///     int: The number of clbits.
    #[getter]
    pub fn num_clbits(&self) -> usize {
        self.clbits.len()
    }

    /// Return the number of unbound compile-time symbolic parameters tracked by the circuit.
    pub fn num_parameters(&self) -> usize {
        self.param_table.num_parameters()
    }

    /// Get a (cached) sorted list of the Python-space `Parameter` instances tracked by this circuit
    /// data's parameter table.
    #[getter]
    pub fn get_parameters<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        self.param_table.py_parameters(py)
    }

    pub fn unsorted_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        self.param_table.py_parameters_unsorted(py)
    }

    fn _raw_parameter_table_entry(&self, param: Bound<PyAny>) -> PyResult<Py<PySet>> {
        self.param_table._py_raw_entry(param)
    }

    pub fn get_parameter_by_name(&self, py: Python, name: PyBackedStr) -> Option<Py<PyAny>> {
        self.param_table
            .py_parameter_by_name(&name)
            .map(|ob| ob.clone_ref(py))
    }

    /// Return the width of the circuit. This is the number of qubits plus the
    /// number of clbits.
    ///
    /// Returns:
    ///     int: The width of the circuit.
    pub fn width(&self) -> usize {
        self.num_qubits() + self.num_clbits()
    }

    /// Registers a :class:`.Qubit` instance.
    ///
    /// Args:
    ///     bit (:class:`.Qubit`): The qubit to register.
    ///     strict (bool): When set, raises an error if ``bit`` is already present.
    ///
    /// Raises:
    ///     ValueError: The specified ``bit`` is already present and flag ``strict``
    ///         was provided.
    #[pyo3(signature = (bit, *, strict=true))]
    pub fn add_qubit(&mut self, py: Python, bit: &Bound<PyAny>, strict: bool) -> PyResult<()> {
        self.qubits.add(py, bit, strict)?;
        Ok(())
    }

    /// Registers a :class:`.Clbit` instance.
    ///
    /// Args:
    ///     bit (:class:`.Clbit`): The clbit to register.
    ///     strict (bool): When set, raises an error if ``bit`` is already present.
    ///
    /// Raises:
    ///     ValueError: The specified ``bit`` is already present and flag ``strict``
    ///         was provided.
    #[pyo3(signature = (bit, *, strict=true))]
    pub fn add_clbit(&mut self, py: Python, bit: &Bound<PyAny>, strict: bool) -> PyResult<()> {
        self.clbits.add(py, bit, strict)?;
        Ok(())
    }

    /// Performs a shallow copy.
    ///
    /// Returns:
    ///     CircuitData: The shallow copy.
    #[pyo3(signature = (copy_instructions=true, deepcopy=false))]
    pub fn copy(&self, py: Python<'_>, copy_instructions: bool, deepcopy: bool) -> PyResult<Self> {
        let mut res = CircuitData::new(
            py,
            Some(self.qubits.cached().bind(py)),
            Some(self.clbits.cached().bind(py)),
            None,
            self.data.len(),
            self.global_phase.clone(),
        )?;
        res.qargs_interner = self.qargs_interner.clone();
        res.cargs_interner = self.cargs_interner.clone();
        res.param_table.clone_from(&self.param_table);

        if deepcopy {
            let memo = PyDict::new_bound(py);
            for inst in &self.data {
                res.data.push(PackedInstruction {
                    op: inst.op.py_deepcopy(py, Some(&memo))?,
                    qubits: inst.qubits,
                    clbits: inst.clbits,
                    params: inst.params.clone(),
                    extra_attrs: inst.extra_attrs.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceCell::new(),
                });
            }
        } else if copy_instructions {
            for inst in &self.data {
                res.data.push(PackedInstruction {
                    op: inst.op.py_copy(py)?,
                    qubits: inst.qubits,
                    clbits: inst.clbits,
                    params: inst.params.clone(),
                    extra_attrs: inst.extra_attrs.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceCell::new(),
                });
            }
        } else {
            res.data.extend(self.data.iter().cloned());
        }
        Ok(res)
    }

    /// Reserves capacity for at least ``additional`` more
    /// :class:`.CircuitInstruction` instances to be added to this container.
    ///
    /// Args:
    ///     additional (int): The additional capacity to reserve. If the
    ///         capacity is already sufficient, does nothing.
    pub fn reserve(&mut self, _py: Python<'_>, additional: usize) {
        self.data.reserve(additional);
    }

    /// Returns a tuple of the sets of :class:`.Qubit` and :class:`.Clbit` instances
    /// that appear in at least one instruction's bit lists.
    ///
    /// Returns:
    ///     tuple[set[:class:`.Qubit`], set[:class:`.Clbit`]]: The active qubits and clbits.
    pub fn active_bits(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let qubits = PySet::empty_bound(py)?;
        let clbits = PySet::empty_bound(py)?;
        for inst in self.data.iter() {
            for b in self.qargs_interner.get(inst.qubits) {
                qubits.add(self.qubits.get(*b).unwrap().clone_ref(py))?;
            }
            for b in self.cargs_interner.get(inst.clbits) {
                clbits.add(self.clbits.get(*b).unwrap().clone_ref(py))?;
            }
        }

        Ok((qubits, clbits).into_py(py))
    }

    /// Invokes callable ``func`` with each instruction's operation.
    ///
    /// Args:
    ///     func (Callable[[:class:`~.Operation`], None]):
    ///         The callable to invoke.
    #[pyo3(signature = (func))]
    pub fn foreach_op(&self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for inst in self.data.iter() {
            func.call1((inst.unpack_py_op(py)?,))?;
        }
        Ok(())
    }

    /// Invokes callable ``func`` with the positional index and operation
    /// of each instruction.
    ///
    /// Args:
    ///     func (Callable[[int, :class:`~.Operation`], None]):
    ///         The callable to invoke.
    #[pyo3(signature = (func))]
    pub fn foreach_op_indexed(&self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for (index, inst) in self.data.iter().enumerate() {
            func.call1((index, inst.unpack_py_op(py)?))?;
        }
        Ok(())
    }

    /// Invokes callable ``func`` with each instruction's operation, replacing the operation with
    /// the result, if the operation is not a standard gate without a condition.
    ///
    /// .. warning::
    ///
    ///     This is a shim for while there are still important components of the circuit still
    ///     implemented in Python space.  This method **skips** any instruction that contains an
    ///     non-conditional standard gate (which is likely to be most instructions).
    ///
    /// Args:
    ///     func (Callable[[:class:`~.Operation`], :class:`~.Operation`]):
    ///         A callable used to map original operations to their replacements.
    #[pyo3(signature = (func))]
    pub fn map_nonstandard_ops(&mut self, py: Python<'_>, func: &Bound<PyAny>) -> PyResult<()> {
        for inst in self.data.iter_mut() {
            if inst.op.try_standard_gate().is_some()
                && !inst
                    .extra_attrs
                    .as_ref()
                    .is_some_and(|attrs| attrs.condition.is_some())
            {
                continue;
            }
            let py_op = func.call1((inst.unpack_py_op(py)?,))?;
            let result = py_op.extract::<OperationFromPython>()?;
            inst.op = result.operation;
            inst.params = (!result.params.is_empty()).then(|| Box::new(result.params));
            inst.extra_attrs = result.extra_attrs;
            #[cfg(feature = "cache_pygates")]
            {
                inst.py_op = py_op.unbind().into();
            }
        }
        Ok(())
    }

    /// Replaces the bits of this container with the given ``qubits``
    /// and/or ``clbits``.
    ///
    /// The `:attr:`~.CircuitInstruction.qubits` and
    /// :attr:`~.CircuitInstruction.clbits` of existing instructions are
    /// reinterpreted using the new bit sequences on access.
    /// As such, the primary use-case for this method is to remap a circuit to
    /// a different set of bits in constant time relative to the number of
    /// instructions in the circuit.
    ///
    /// Args:
    ///     qubits (Iterable[:class:`.Qubit] | None):
    ///         The qubit sequence which should replace the container's
    ///         existing qubits, or ``None`` to skip replacement.
    ///     clbits (Iterable[:class:`.Clbit] | None):
    ///         The clbit sequence which should replace the container's
    ///         existing qubits, or ``None`` to skip replacement.
    ///
    /// Raises:
    ///     ValueError: A replacement sequence is smaller than the bit list
    ///         its contents would replace.
    ///
    /// .. note::
    ///
    ///     Instruction operations themselves are NOT adjusted.
    ///     To modify bits referenced by an operation, use
    ///     :meth:`~.CircuitData.foreach_op` or
    ///     :meth:`~.CircuitData.foreach_op_indexed` or
    ///     :meth:`~.CircuitData.map_nonstandard_ops` to adjust the operations manually
    ///     after calling this method.
    ///
    /// Examples:
    ///
    ///     The following :class:`.CircuitData` is reinterpreted as if its bits
    ///     were originally added in reverse.
    ///
    ///     .. code-block::
    ///
    ///         qr = QuantumRegister(3)
    ///         data = CircuitData(qubits=qr, data=[
    ///             CircuitInstruction(XGate(), [qr[0]], []),
    ///             CircuitInstruction(XGate(), [qr[1]], []),
    ///             CircuitInstruction(XGate(), [qr[2]], []),
    ///         ])
    ///
    ///         data.replace_bits(qubits=reversed(qr))
    ///         assert(data == [
    ///             CircuitInstruction(XGate(), [qr[2]], []),
    ///             CircuitInstruction(XGate(), [qr[1]], []),
    ///             CircuitInstruction(XGate(), [qr[0]], []),
    ///         ])
    #[pyo3(signature = (qubits=None, clbits=None))]
    pub fn replace_bits(
        &mut self,
        py: Python<'_>,
        qubits: Option<&Bound<PyAny>>,
        clbits: Option<&Bound<PyAny>>,
    ) -> PyResult<()> {
        let mut temp = CircuitData::new(py, qubits, clbits, None, 0, self.global_phase.clone())?;
        if qubits.is_some() {
            if temp.num_qubits() < self.num_qubits() {
                return Err(PyValueError::new_err(format!(
                    "Replacement 'qubits' of size {:?} must contain at least {:?} bits.",
                    temp.num_qubits(),
                    self.num_qubits(),
                )));
            }
            std::mem::swap(&mut temp.qubits, &mut self.qubits);
        }
        if clbits.is_some() {
            if temp.num_clbits() < self.num_clbits() {
                return Err(PyValueError::new_err(format!(
                    "Replacement 'clbits' of size {:?} must contain at least {:?} bits.",
                    temp.num_clbits(),
                    self.num_clbits(),
                )));
            }
            std::mem::swap(&mut temp.clbits, &mut self.clbits);
        }
        Ok(())
    }

    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    // Note: we also rely on this to make us iterable!
    pub fn __getitem__(&self, py: Python, index: PySequenceIndex) -> PyResult<PyObject> {
        // Get a single item, assuming the index is validated as in bounds.
        let get_single = |index: usize| {
            let inst = &self.data[index];
            let qubits = self.qargs_interner.get(inst.qubits);
            let clbits = self.cargs_interner.get(inst.clbits);
            CircuitInstruction {
                operation: inst.op.clone(),
                qubits: PyTuple::new_bound(py, self.qubits.map_indices(qubits)).unbind(),
                clbits: PyTuple::new_bound(py, self.clbits.map_indices(clbits)).unbind(),
                params: inst.params_view().iter().cloned().collect(),
                extra_attrs: inst.extra_attrs.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: inst.py_op.clone(),
            }
            .into_py(py)
        };
        match index.with_len(self.data.len())? {
            SequenceIndex::Int(index) => Ok(get_single(index)),
            indices => Ok(PyList::new_bound(py, indices.iter().map(get_single)).into_py(py)),
        }
    }

    pub fn __delitem__(&mut self, py: Python, index: PySequenceIndex) -> PyResult<()> {
        self.delitem(py, index.with_len(self.data.len())?)
    }

    pub fn __setitem__(&mut self, index: PySequenceIndex, value: &Bound<PyAny>) -> PyResult<()> {
        fn set_single(slf: &mut CircuitData, index: usize, value: &Bound<PyAny>) -> PyResult<()> {
            let py = value.py();
            slf.untrack_instruction_parameters(py, index)?;
            slf.data[index] = slf.pack(py, &value.downcast::<CircuitInstruction>()?.borrow())?;
            slf.track_instruction_parameters(py, index)?;
            Ok(())
        }

        let py = value.py();
        match index.with_len(self.data.len())? {
            SequenceIndex::Int(index) => set_single(self, index, value),
            indices @ SequenceIndex::PosRange {
                start,
                stop,
                step: 1,
            } => {
                // `list` allows setting a slice with step +1 to an arbitrary length.
                let values = value.iter()?.collect::<PyResult<Vec<_>>>()?;
                for (index, value) in indices.iter().zip(values.iter()) {
                    set_single(self, index, value)?;
                }
                if indices.len() > values.len() {
                    self.delitem(
                        py,
                        SequenceIndex::PosRange {
                            start: start + values.len(),
                            stop,
                            step: 1,
                        },
                    )?
                } else {
                    for value in values[indices.len()..].iter().rev() {
                        self.insert(stop as isize, value.downcast()?.borrow())?;
                    }
                }
                Ok(())
            }
            indices => {
                let values = value.iter()?.collect::<PyResult<Vec<_>>>()?;
                if indices.len() == values.len() {
                    for (index, value) in indices.iter().zip(values.iter()) {
                        set_single(self, index, value)?;
                    }
                    Ok(())
                } else {
                    Err(PyValueError::new_err(format!(
                        "attempt to assign sequence of size {:?} to extended slice of size {:?}",
                        values.len(),
                        indices.len(),
                    )))
                }
            }
        }
    }

    pub fn insert(&mut self, mut index: isize, value: PyRef<CircuitInstruction>) -> PyResult<()> {
        // `list.insert` has special-case extra clamping logic for its index argument.
        let index = {
            if index < 0 {
                // This can't exceed `isize::MAX` because `self.data[0]` is larger than a byte.
                index += self.data.len() as isize;
            }
            if index < 0 {
                0
            } else if index as usize > self.data.len() {
                self.data.len()
            } else {
                index as usize
            }
        };
        let py = value.py();
        let packed = self.pack(py, &value)?;
        self.data.insert(index, packed);
        if index == self.data.len() - 1 {
            self.track_instruction_parameters(py, index)?;
        } else {
            self.reindex_parameter_table(py)?;
        }
        Ok(())
    }

    pub fn pop(&mut self, py: Python<'_>, index: Option<PySequenceIndex>) -> PyResult<PyObject> {
        let index = index.unwrap_or(PySequenceIndex::Int(-1));
        let native_index = index.with_len(self.data.len())?;
        let item = self.__getitem__(py, index)?;
        self.delitem(py, native_index)?;
        Ok(item)
    }

    /// Primary entry point for appending an instruction from Python space.
    pub fn append(&mut self, value: &Bound<CircuitInstruction>) -> PyResult<()> {
        let py = value.py();
        let new_index = self.data.len();
        let packed = self.pack(py, &value.borrow())?;
        self.data.push(packed);
        self.track_instruction_parameters(py, new_index)
    }

    /// Backup entry point for appending an instruction from Python space, in the unusual case that
    /// one of the instruction parameters contains a cyclical reference to the circuit itself.
    ///
    /// In this case, the `params` field should be a list of `(index, parameters)` tuples, where the
    /// index is into the instruction's `params` attribute, and `parameters` is a Python iterable
    /// of `Parameter` objects.
    pub fn append_manual_params(
        &mut self,
        value: &Bound<CircuitInstruction>,
        params: &Bound<PyList>,
    ) -> PyResult<()> {
        let instruction_index = self.data.len();
        let packed = self.pack(value.py(), &value.borrow())?;
        self.data.push(packed);
        for item in params.iter() {
            let (parameter_index, parameters) = item.extract::<(u32, Bound<PyAny>)>()?;
            let usage = ParameterUse::Index {
                instruction: instruction_index,
                parameter: parameter_index,
            };
            for param in parameters.iter()? {
                self.param_table.track(&param?, Some(usage))?;
            }
        }
        Ok(())
    }

    pub fn extend(&mut self, py: Python<'_>, itr: &Bound<PyAny>) -> PyResult<()> {
        if let Ok(other) = itr.downcast::<CircuitData>() {
            let other = other.borrow();
            // Fast path to avoid unnecessary construction of CircuitInstruction instances.
            self.data.reserve(other.data.len());
            for inst in other.data.iter() {
                let qubits = other
                    .qargs_interner
                    .get(inst.qubits)
                    .iter()
                    .map(|b| {
                        Ok(self
                            .qubits
                            .find(other.qubits.get(*b).unwrap().bind(py))
                            .unwrap())
                    })
                    .collect::<PyResult<Vec<Qubit>>>()?;
                let clbits = other
                    .cargs_interner
                    .get(inst.clbits)
                    .iter()
                    .map(|b| {
                        Ok(self
                            .clbits
                            .find(other.clbits.get(*b).unwrap().bind(py))
                            .unwrap())
                    })
                    .collect::<PyResult<Vec<Clbit>>>()?;
                let new_index = self.data.len();
                let qubits_id = self.qargs_interner.insert_owned(qubits);
                let clbits_id = self.cargs_interner.insert_owned(clbits);
                self.data.push(PackedInstruction {
                    op: inst.op.clone(),
                    qubits: qubits_id,
                    clbits: clbits_id,
                    params: inst.params.clone(),
                    extra_attrs: inst.extra_attrs.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: inst.py_op.clone(),
                });
                self.track_instruction_parameters(py, new_index)?;
            }
            return Ok(());
        }
        for v in itr.iter()? {
            self.append(v?.downcast()?)?;
        }
        Ok(())
    }

    /// Assign all the circuit parameters, given an iterable input of `Param` instances.
    fn assign_parameters_iterable(&mut self, sequence: Bound<PyAny>) -> PyResult<()> {
        if let Ok(readonly) = sequence.extract::<PyReadonlyArray1<f64>>() {
            // Fast path for Numpy arrays; in this case we can easily handle them without copying
            // the data across into a Rust-space `Vec` first.
            let array = readonly.as_array();
            if array.len() != self.param_table.num_parameters() {
                return Err(PyValueError::new_err(concat!(
                    "Mismatching number of values and parameters. For partial binding ",
                    "please pass a dictionary of {parameter: value} pairs."
                )));
            }
            let mut old_table = std::mem::take(&mut self.param_table);
            self.assign_parameters_inner(
                sequence.py(),
                array
                    .iter()
                    .map(|value| Param::Float(*value))
                    .zip(old_table.drain_ordered())
                    .map(|(value, (obj, uses))| (obj, value, uses)),
            )
        } else {
            let values = sequence
                .iter()?
                .map(|ob| Param::extract_no_coerce(&ob?))
                .collect::<PyResult<Vec<_>>>()?;
            self.assign_parameters_from_slice(sequence.py(), &values)
        }
    }

    /// Assign all uses of the circuit parameters as keys `mapping` to their corresponding values.
    fn assign_parameters_mapping(&mut self, mapping: Bound<PyAny>) -> PyResult<()> {
        let py = mapping.py();
        let mut items = Vec::new();
        for item in mapping.call_method0("items")?.iter()? {
            let (param_ob, value) = item?.extract::<(Py<PyAny>, AssignParam)>()?;
            let uuid = ParameterUuid::from_parameter(param_ob.bind(py))?;
            items.push((param_ob, value.0, self.param_table.pop(uuid)?));
        }
        self.assign_parameters_inner(py, items)
    }

    pub fn clear(&mut self) {
        std::mem::take(&mut self.data);
        self.param_table.clear();
    }

    /// Counts the number of times each operation is used in the circuit.
    ///
    /// # Parameters
    /// - `self` - A mutable reference to the CircuitData struct.
    ///
    /// # Returns
    /// An IndexMap containing the operation names as keys and their respective counts as values.
    pub fn count_ops(&self) -> IndexMap<&str, usize, ::ahash::RandomState> {
        let mut ops_count: IndexMap<&str, usize, ::ahash::RandomState> = IndexMap::default();
        for instruction in &self.data {
            *ops_count.entry(instruction.op.name()).or_insert(0) += 1;
        }
        ops_count.par_sort_by(|_k1, v1, _k2, v2| v2.cmp(v1));
        ops_count
    }

    // Marks this pyclass as NOT hashable.
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __eq__(slf: &Bound<Self>, other: &Bound<PyAny>) -> PyResult<bool> {
        let slf = slf.as_any();
        if slf.is(other) {
            return Ok(true);
        }
        if slf.len()? != other.len()? {
            return Ok(false);
        }
        // Implemented using generic iterators on both sides
        // for simplicity.
        let mut ours_itr = slf.iter()?;
        let mut theirs_itr = other.iter()?;
        loop {
            match (ours_itr.next(), theirs_itr.next()) {
                (Some(ours), Some(theirs)) => {
                    if !ours?.eq(theirs?)? {
                        return Ok(false);
                    }
                }
                (None, None) => {
                    return Ok(true);
                }
                _ => {
                    return Ok(false);
                }
            }
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for bit in self.qubits.bits().iter().chain(self.clbits.bits().iter()) {
            visit.call(bit)?;
        }

        // Note:
        //   There's no need to visit the native Rust data
        //   structures used for internal tracking: the only Python
        //   references they contain are to the bits in these lists!
        visit.call(self.qubits.cached())?;
        visit.call(self.clbits.cached())?;
        self.param_table.py_gc_traverse(&visit)?;
        Ok(())
    }

    fn __clear__(&mut self) {
        // Clear anything that could have a reference cycle.
        self.data.clear();
        self.qubits.dispose();
        self.clbits.dispose();
        self.param_table.clear();
    }

    /// Set the global phase of the circuit.
    ///
    /// This method assumes that the parameter table is either fully consistent, or contains zero
    /// entries for the global phase, regardless of what value is currently stored there.  It's not
    /// uncommon for subclasses and other parts of Qiskit to have filled in the global phase field
    /// by copies or other means, before making the parameter table consistent.
    #[setter]
    pub fn set_global_phase(&mut self, py: Python, angle: Param) -> PyResult<()> {
        if let Param::ParameterExpression(expr) = &self.global_phase {
            for param_ob in expr.bind(py).getattr(intern!(py, "parameters"))?.iter()? {
                match self.param_table.remove_use(
                    ParameterUuid::from_parameter(&param_ob?)?,
                    ParameterUse::GlobalPhase,
                ) {
                    Ok(_)
                    | Err(ParameterTableError::ParameterNotTracked(_))
                    | Err(ParameterTableError::UsageNotTracked(_)) => (),
                    // Any errors added later might want propagating.
                }
            }
        }
        match angle {
            Param::Float(angle) => {
                self.global_phase = Param::Float(angle.rem_euclid(2. * std::f64::consts::PI));
                Ok(())
            }
            Param::ParameterExpression(_) => {
                for param_ob in angle.iter_parameters(py)? {
                    self.param_table
                        .track(&param_ob?, Some(ParameterUse::GlobalPhase))?;
                }
                self.global_phase = angle;
                Ok(())
            }
            Param::Obj(_) => Err(PyTypeError::new_err("invalid type for global phase")),
        }
    }

    pub fn num_nonlocal_gates(&self) -> usize {
        self.data
            .iter()
            .filter(|inst| inst.op.num_qubits() > 1 && !inst.op.directive())
            .count()
    }
}

impl CircuitData {
    /// Native internal driver of `__delitem__` that uses a Rust-space version of the
    /// `SequenceIndex`.  This assumes that the `SequenceIndex` contains only in-bounds indices, and
    /// panics if not.
    fn delitem(&mut self, py: Python, indices: SequenceIndex) -> PyResult<()> {
        // We need to delete in reverse order so we don't invalidate higher indices with a deletion.
        for index in indices.descending() {
            self.data.remove(index);
        }
        if !indices.is_empty() {
            self.reindex_parameter_table(py)?;
        }
        Ok(())
    }

    fn pack(&mut self, py: Python, inst: &CircuitInstruction) -> PyResult<PackedInstruction> {
        let qubits = self
            .qargs_interner
            .insert_owned(self.qubits.map_bits(inst.qubits.bind(py))?.collect());
        let clbits = self
            .cargs_interner
            .insert_owned(self.clbits.map_bits(inst.clbits.bind(py))?.collect());
        Ok(PackedInstruction {
            op: inst.operation.clone(),
            qubits,
            clbits,
            params: (!inst.params.is_empty()).then(|| Box::new(inst.params.clone())),
            extra_attrs: inst.extra_attrs.clone(),
            #[cfg(feature = "cache_pygates")]
            py_op: inst.py_op.clone(),
        })
    }

    /// Returns an iterator over all the instructions present in the circuit.
    pub fn iter(&self) -> impl Iterator<Item = &PackedInstruction> {
        self.data.iter()
    }

    /// Assigns parameters to circuit data based on a slice of `Param`.
    pub fn assign_parameters_from_slice(&mut self, py: Python, slice: &[Param]) -> PyResult<()> {
        if slice.len() != self.param_table.num_parameters() {
            return Err(PyValueError::new_err(concat!(
                "Mismatching number of values and parameters. For partial binding ",
                "please pass a mapping of {parameter: value} pairs."
            )));
        }
        let mut old_table = std::mem::take(&mut self.param_table);
        self.assign_parameters_inner(
            py,
            slice
                .iter()
                .zip(old_table.drain_ordered())
                .map(|(value, (param_ob, uses))| (param_ob, value.clone_ref(py), uses)),
        )
    }

    /// Assigns parameters to circuit data based on a mapping of `ParameterUuid` : `Param`.
    /// This mapping assumes that the provided `ParameterUuid` keys are instances
    /// of `ParameterExpression`.
    pub fn assign_parameters_from_mapping<I, T>(&mut self, py: Python, iter: I) -> PyResult<()>
    where
        I: IntoIterator<Item = (ParameterUuid, T)>,
        T: AsRef<Param>,
    {
        let mut items = Vec::new();
        for (param_uuid, value) in iter {
            // Assume all the Parameters are already in the circuit
            let param_obj = self.get_parameter_by_uuid(param_uuid);
            if let Some(param_obj) = param_obj {
                // Copy or increase ref_count for Parameter, avoid acquiring the GIL.
                items.push((
                    param_obj.clone_ref(py),
                    value.as_ref().clone_ref(py),
                    self.param_table.pop(param_uuid)?,
                ));
            } else {
                return Err(PyValueError::new_err("An invalid parameter was provided."));
            }
        }
        self.assign_parameters_inner(py, items)
    }

    /// Returns an immutable view of the Interner used for Qargs
    pub fn qargs_interner(&self) -> &Interner<[Qubit]> {
        &self.qargs_interner
    }

    /// Returns an immutable view of the Interner used for Cargs
    pub fn cargs_interner(&self) -> &Interner<[Clbit]> {
        &self.cargs_interner
    }

    /// Returns an immutable view of the Global Phase `Param` of the circuit
    pub fn global_phase(&self) -> &Param {
        &self.global_phase
    }

    /// Returns an immutable view of the Qubits registered in the circuit
    pub fn qubits(&self) -> &BitData<Qubit> {
        &self.qubits
    }

    /// Returns an immutable view of the Classical bits registered in the circuit
    pub fn clbits(&self) -> &BitData<Clbit> {
        &self.clbits
    }

    /// Unpacks from interned value to `[Qubit]`
    pub fn get_qargs(&self, index: Interned<[Qubit]>) -> &[Qubit] {
        self.qargs_interner().get(index)
    }

    /// Unpacks from InternerIndex to `[Clbit]`
    pub fn get_cargs(&self, index: Interned<[Clbit]>) -> &[Clbit] {
        self.cargs_interner().get(index)
    }

    fn assign_parameters_inner<I, T>(&mut self, py: Python, iter: I) -> PyResult<()>
    where
        I: IntoIterator<Item = (Py<PyAny>, T, HashSet<ParameterUse>)>,
        T: AsRef<Param> + Clone,
    {
        let inconsistent =
            || PyRuntimeError::new_err("internal error: circuit parameter table is inconsistent");

        let assign_attr = intern!(py, "assign");
        let assign_parameters_attr = intern!(py, "assign_parameters");
        let _definition_attr = intern!(py, "_definition");
        let numeric_attr = intern!(py, "numeric");
        let parameters_attr = intern!(py, "parameters");
        let params_attr = intern!(py, "params");
        let validate_parameter_attr = intern!(py, "validate_parameter");

        // Bind a single `Parameter` into a Python-space `ParameterExpression`.
        let bind_expr = |expr: Borrowed<PyAny>,
                         param_ob: &Py<PyAny>,
                         value: &Param,
                         coerce: bool|
         -> PyResult<Param> {
            let new_expr = expr.call_method1(assign_attr, (param_ob, value.to_object(py)))?;
            if new_expr.getattr(parameters_attr)?.len()? == 0 {
                let out = new_expr.call_method0(numeric_attr)?;
                if coerce {
                    out.extract()
                } else {
                    Param::extract_no_coerce(&out)
                }
            } else {
                Ok(Param::ParameterExpression(new_expr.unbind()))
            }
        };

        let mut user_operations = HashMap::new();
        let mut uuids = Vec::new();
        for (param_ob, value, uses) in iter {
            debug_assert!(!uses.is_empty());
            uuids.clear();
            for inner_param_ob in value.as_ref().iter_parameters(py)? {
                uuids.push(self.param_table.track(&inner_param_ob?, None)?)
            }
            for usage in uses {
                match usage {
                    ParameterUse::GlobalPhase => {
                        let Param::ParameterExpression(expr) = &self.global_phase else {
                            return Err(inconsistent());
                        };
                        self.set_global_phase(
                            py,
                            bind_expr(expr.bind_borrowed(py), &param_ob, value.as_ref(), true)?,
                        )?;
                    }
                    ParameterUse::Index {
                        instruction,
                        parameter,
                    } => {
                        let parameter = parameter as usize;
                        let previous = &mut self.data[instruction];
                        if let Some(standard) = previous.standard_gate() {
                            let params = previous.params_mut();
                            let Param::ParameterExpression(expr) = &params[parameter] else {
                                return Err(inconsistent());
                            };
                            params[parameter] = match bind_expr(
                                expr.bind_borrowed(py),
                                &param_ob,
                                value.as_ref(),
                                true,
                            )? {
                                Param::Obj(obj) => {
                                    return Err(CircuitError::new_err(format!(
                                        "bad type after binding for gate '{}': '{}'",
                                        standard.name(),
                                        obj.bind(py).repr()?,
                                    )))
                                }
                                param => param,
                            };
                            for uuid in uuids.iter() {
                                self.param_table.add_use(*uuid, usage)?
                            }
                            #[cfg(feature = "cache_pygates")]
                            {
                                // Standard gates can all rebuild their definitions, so if the
                                // cached py_op exists, just clear out any existing cache.
                                if let Some(borrowed) = previous.py_op.get() {
                                    borrowed.bind(py).setattr("_definition", py.None())?
                                }
                            }
                        } else {
                            // Track user operations we've seen so we can rebind their definitions.
                            // Strictly this can add the same binding pair more than once, if an
                            // instruction has the same `Parameter` in several of its `params`, but
                            // we're going to turn that into a `dict` anyway, so it doesn't matter.
                            user_operations
                                .entry(instruction)
                                .or_insert_with(Vec::new)
                                .push((param_ob.clone_ref(py), value.as_ref().clone_ref(py)));

                            let op = previous.unpack_py_op(py)?.into_bound(py);
                            let previous_param = &previous.params_view()[parameter];
                            let new_param = match previous_param {
                                Param::Float(_) => return Err(inconsistent()),
                                Param::ParameterExpression(expr) => {
                                    // For user gates, we don't coerce floats to integers in `Param`
                                    // so that users can use them if they choose.
                                    let new_param = bind_expr(
                                        expr.bind_borrowed(py),
                                        &param_ob,
                                        value.as_ref(),
                                        false,
                                    )?;
                                    // Historically, `assign_parameters` called `validate_parameter`
                                    // only when a `ParameterExpression` became fully bound.  Some
                                    // "generalised" (or user) gates fail without this, though
                                    // arguably, that's them indicating they shouldn't be allowed to
                                    // be parametric.
                                    //
                                    // Our `bind_expr` coercion means that a non-parametric
                                    // `ParameterExperssion` after binding would have been coerced
                                    // to a numeric quantity already, so the match here is
                                    // definitely parameterized.
                                    match new_param {
                                        Param::ParameterExpression(_) => new_param,
                                        new_param => Param::extract_no_coerce(&op.call_method1(
                                            validate_parameter_attr,
                                            (new_param,),
                                        )?)?,
                                    }
                                }
                                Param::Obj(obj) => {
                                    let obj = obj.bind_borrowed(py);
                                    if !obj.is_instance(QUANTUM_CIRCUIT.get_bound(py))? {
                                        return Err(inconsistent());
                                    }
                                    Param::extract_no_coerce(
                                        &obj.call_method(
                                            assign_parameters_attr,
                                            ([(&param_ob, value.as_ref())].into_py_dict_bound(py),),
                                            Some(
                                                &[("inplace", false), ("flat_input", true)]
                                                    .into_py_dict_bound(py),
                                            ),
                                        )?,
                                    )?
                                }
                            };
                            op.getattr(params_attr)?.set_item(parameter, new_param)?;
                            let mut new_op = op.extract::<OperationFromPython>()?;
                            previous.op = new_op.operation;
                            previous.params_mut().swap_with_slice(&mut new_op.params);
                            previous.extra_attrs = new_op.extra_attrs;
                            #[cfg(feature = "cache_pygates")]
                            {
                                previous.py_op = op.into_py(py).into();
                            }
                            for uuid in uuids.iter() {
                                self.param_table.add_use(*uuid, usage)?
                            }
                        }
                    }
                }
            }
        }

        let assign_kwargs = (!user_operations.is_empty()).then(|| {
            [("inplace", true), ("flat_input", true), ("strict", false)].into_py_dict_bound(py)
        });
        for (instruction, bindings) in user_operations {
            // We only put non-standard gates in `user_operations`, so we're not risking creating a
            // previously non-existent Python object.
            let instruction = &self.data[instruction];
            let definition_cache = if matches!(instruction.op.view(), OperationRef::Operation(_)) {
                // `Operation` instances don't have a `definition` as part of their interfaces, but
                // they might be an `AnnotatedOperation`, which is one of our special built-ins.
                // This should be handled more completely in the user-customisation interface by a
                // delegating method, but that's not the data model we currently have.
                let py_op = instruction.unpack_py_op(py)?;
                let py_op = py_op.bind(py);
                if !py_op.is_instance(ANNOTATED_OPERATION.get_bound(py))? {
                    continue;
                }
                py_op
                    .getattr(intern!(py, "base_op"))?
                    .getattr(_definition_attr)?
            } else {
                instruction
                    .unpack_py_op(py)?
                    .bind(py)
                    .getattr(_definition_attr)?
            };
            if !definition_cache.is_none() {
                definition_cache.call_method(
                    assign_parameters_attr,
                    (bindings.into_py_dict_bound(py),),
                    assign_kwargs.as_ref(),
                )?;
            }
        }
        Ok(())
    }

    /// Retrieves the python `Param` object based on its `ParameterUuid`.
    pub fn get_parameter_by_uuid(&self, uuid: ParameterUuid) -> Option<&Py<PyAny>> {
        self.param_table.py_parameter_by_uuid(uuid)
    }
}

/// Helper struct for `assign_parameters` to allow use of `Param::extract_no_coerce` in
/// PyO3-provided `FromPyObject` implementations on containers.
#[repr(transparent)]
struct AssignParam(Param);
impl<'py> FromPyObject<'py> for AssignParam {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Self(Param::extract_no_coerce(ob)?))
    }
}

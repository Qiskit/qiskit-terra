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

#![allow(clippy::too_many_arguments)]

mod errors;
mod instruction_properties;
mod nullable_index_map;

use std::ops::Index;

use ahash::RandomState;

use hashbrown::HashSet;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use nullable_index_map::NullableIndexMap;
use pyo3::{
    exceptions::{PyAttributeError, PyIndexError, PyKeyError, PyValueError},
    prelude::*,
    pyclass,
    types::{PyDict, PyList, PySet, PyTuple},
};

use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::operations::{Operation, Param};
use qiskit_circuit::packed_instruction::PackedOperation;
use smallvec::SmallVec;

use crate::nlayout::PhysicalQubit;

use errors::TargetKeyError;
use instruction_properties::InstructionProperties;

use self::exceptions::TranspilerError;

mod exceptions {
    use pyo3::import_exception_bound;
    import_exception_bound! {qiskit.exceptions, QiskitError}
    import_exception_bound! {qiskit.transpiler.exceptions, TranspilerError}
}

// Custom types
pub type Qargs = SmallVec<[PhysicalQubit; 2]>;
type GateMap = IndexMap<String, PropsMap, RandomState>;
type PropsMap = NullableIndexMap<Qargs, Option<InstructionProperties>>;
type GateMapState = Vec<(String, Vec<(Option<Qargs>, Option<InstructionProperties>)>)>;

/// Represents a Qiskit `Gate` object or a Variadic instruction.
/// Keeps a reference to its Python instance for caching purposes.
#[derive(Debug, Clone, FromPyObject)]
pub(crate) enum TargetOperation {
    Normal(NormalOperation),
    Variadic(PyObject),
}

impl IntoPy<PyObject> for TargetOperation {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Self::Normal(normal) => normal.into_py(py),
            Self::Variadic(variable) => variable,
        }
    }
}

impl ToPyObject for TargetOperation {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        match self {
            Self::Normal(normal) => normal.to_object(py),
            Self::Variadic(variable) => variable.clone_ref(py),
        }
    }
}

impl TargetOperation {
    fn num_qubits(&self) -> u32 {
        match &self {
            Self::Normal(normal) => normal.operation.view().num_qubits(),
            Self::Variadic(_) => {
                unreachable!("'num_qubits' property is reserved for normal operations only.")
            }
        }
    }

    fn params(&self) -> &[Param] {
        match &self {
            TargetOperation::Normal(normal) => normal.params.as_slice(),
            TargetOperation::Variadic(_) => &[],
        }
    }
}

/// Represents a Qiskit `Gate` object, keeps a reference to its Python
/// instance for caching purposes.
#[derive(Debug, Clone)]
pub(crate) struct NormalOperation {
    pub operation: PackedOperation,
    pub params: SmallVec<[Param; 3]>,
    op_object: PyObject,
}

impl<'py> FromPyObject<'py> for NormalOperation {
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        let operation: OperationFromPython = ob.extract()?;
        Ok(Self {
            operation: operation.operation,
            params: operation.params,
            op_object: ob.into(),
        })
    }
}

impl IntoPy<PyObject> for NormalOperation {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.to_object(py)
    }
}

impl ToPyObject for NormalOperation {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.op_object.clone_ref(py)
    }
}

/**
The base class for a Python ``Target`` object. Contains data representing the
constraints of a particular backend.

The intent of this struct is to contain data that can be representable and
accessible through both Rust and Python, so it can be used for rust-based
transpiler processes.

This structure contains duplicates of every element in the Python counterpart of
`gate_map`. Which improves access for Python while sacrificing a small amount of
memory.
 */
#[pyclass(
    mapping,
    subclass,
    name = "BaseTarget",
    module = "qiskit._accelerate.target"
)]
#[derive(Clone, Debug)]
pub(crate) struct Target {
    #[pyo3(get, set)]
    pub description: Option<String>,
    #[pyo3(get)]
    pub num_qubits: Option<usize>,
    #[pyo3(get, set)]
    pub dt: Option<f64>,
    #[pyo3(get, set)]
    pub granularity: u32,
    #[pyo3(get, set)]
    pub min_length: usize,
    #[pyo3(get, set)]
    pub pulse_alignment: u32,
    #[pyo3(get, set)]
    pub acquire_alignment: u32,
    #[pyo3(get, set)]
    pub qubit_properties: Option<Vec<PyObject>>,
    #[pyo3(get, set)]
    pub concurrent_measurements: Option<Vec<Vec<PhysicalQubit>>>,
    gate_map: GateMap,
    #[pyo3(get)]
    _gate_name_map: IndexMap<String, TargetOperation, RandomState>,
    global_operations: IndexMap<u32, HashSet<String>, RandomState>,
    variable_class_operations: IndexSet<String, RandomState>,
    qarg_gate_map: NullableIndexMap<Qargs, Option<HashSet<String>>>,
    non_global_strict_basis: Option<Vec<String>>,
    non_global_basis: Option<Vec<String>>,
}

#[pymethods]
impl Target {
    /// Create a new ``Target`` object
    ///
    ///Args:
    ///    description (str): An optional string to describe the Target.
    ///    num_qubits (int): An optional int to specify the number of qubits
    ///        the backend target has. If not set it will be implicitly set
    ///        based on the qargs when :meth:`~qiskit.Target.add_instruction`
    ///        is called. Note this must be set if the backend target is for a
    ///        noiseless simulator that doesn't have constraints on the
    ///        instructions so the transpiler knows how many qubits are
    ///        available.
    ///    dt (float): The system time resolution of input signals in seconds
    ///    granularity (int): An integer value representing minimum pulse gate
    ///        resolution in units of ``dt``. A user-defined pulse gate should
    ///        have duration of a multiple of this granularity value.
    ///    min_length (int): An integer value representing minimum pulse gate
    ///        length in units of ``dt``. A user-defined pulse gate should be
    ///        longer than this length.
    ///    pulse_alignment (int): An integer value representing a time
    ///        resolution of gate instruction starting time. Gate instruction
    ///        should start at time which is a multiple of the alignment
    ///        value.
    ///    acquire_alignment (int): An integer value representing a time
    ///        resolution of measure instruction starting time. Measure
    ///        instruction should start at time which is a multiple of the
    ///        alignment value.
    ///    qubit_properties (list): A list of :class:`~.QubitProperties`
    ///        objects defining the characteristics of each qubit on the
    ///        target device. If specified the length of this list must match
    ///        the number of qubits in the target, where the index in the list
    ///        matches the qubit number the properties are defined for. If some
    ///        qubits don't have properties available you can set that entry to
    ///        ``None``
    ///    concurrent_measurements(list): A list of sets of qubits that must be
    ///        measured together. This must be provided
    ///        as a nested list like ``[[0, 1], [2, 3, 4]]``.
    ///Raises:
    ///    ValueError: If both ``num_qubits`` and ``qubit_properties`` are both
    ///        defined and the value of ``num_qubits`` differs from the length of
    ///        ``qubit_properties``.
    #[new]
    #[pyo3(signature = (
        description = None,
        num_qubits = None,
        dt = None,
        granularity = None,
        min_length = None,
        pulse_alignment = None,
        acquire_alignment = None,
        qubit_properties = None,
        concurrent_measurements = None,
    ))]
    fn new(
        description: Option<String>,
        num_qubits: Option<usize>,
        dt: Option<f64>,
        granularity: Option<u32>,
        min_length: Option<usize>,
        pulse_alignment: Option<u32>,
        acquire_alignment: Option<u32>,
        qubit_properties: Option<Vec<PyObject>>,
        concurrent_measurements: Option<Vec<Vec<PhysicalQubit>>>,
    ) -> PyResult<Self> {
        let mut num_qubits = num_qubits;
        if let Some(qubit_properties) = qubit_properties.as_ref() {
            if let Some(num_qubits) = num_qubits {
                if num_qubits != qubit_properties.len() {
                    return Err(PyValueError::new_err(
                        "The value of num_qubits specified does not match the \
                            length of the input qubit_properties list",
                    ));
                }
            } else {
                num_qubits = Some(qubit_properties.len())
            }
        }
        Ok(Target {
            description,
            num_qubits,
            dt,
            granularity: granularity.unwrap_or(1),
            min_length: min_length.unwrap_or(1),
            pulse_alignment: pulse_alignment.unwrap_or(1),
            acquire_alignment: acquire_alignment.unwrap_or(0),
            qubit_properties,
            concurrent_measurements,
            gate_map: GateMap::default(),
            _gate_name_map: IndexMap::default(),
            variable_class_operations: IndexSet::default(),
            global_operations: IndexMap::default(),
            qarg_gate_map: NullableIndexMap::default(),
            non_global_basis: None,
            non_global_strict_basis: None,
        })
    }

    /// Add a new instruction to the `Target` after it has been processed in python.
    ///
    /// Args:
    ///     instruction: An instance of `Instruction` or the class representing said instructionm
    ///         if representing a variadic.
    ///     properties: A mapping of qargs and ``InstructionProperties``.
    ///     name: A name assigned to the provided gate.
    /// Raises:
    ///     AttributeError: If gate is already in map
    ///     TranspilerError: If an operation class is passed in for ``instruction`` and no name
    ///         is specified or ``properties`` is set.
    #[pyo3(signature = (instruction, name, properties=None))]
    fn add_instruction(
        &mut self,
        instruction: TargetOperation,
        name: &str,
        properties: Option<PropsMap>,
    ) -> PyResult<()> {
        if self.gate_map.contains_key(name) {
            return Err(PyAttributeError::new_err(format!(
                "Instruction {:?} is already in the target",
                name
            )));
        }
        let mut qargs_val: PropsMap;
        match instruction {
            TargetOperation::Variadic(_) => {
                qargs_val = PropsMap::with_capacity(1);
                qargs_val.extend([(None, None)]);
                self.variable_class_operations.insert(name.to_string());
            }
            TargetOperation::Normal(_) => {
                if let Some(mut properties) = properties {
                    qargs_val = PropsMap::with_capacity(properties.len());
                    let inst_num_qubits = instruction.num_qubits();
                    if properties.contains_key(None) {
                        self.global_operations
                            .entry(inst_num_qubits)
                            .and_modify(|e| {
                                e.insert(name.to_string());
                            })
                            .or_insert(HashSet::from_iter([name.to_string()]));
                    }
                    let property_keys: Vec<Option<Qargs>> =
                        properties.keys().map(|qargs| qargs.cloned()).collect();
                    for qarg in property_keys {
                        if let Some(qarg) = qarg.as_ref() {
                            if qarg.len() != inst_num_qubits as usize {
                                return Err(TranspilerError::new_err(format!(
                                    "The number of qubits for {name} does not match\
                                    the number of qubits in the properties dictionary: {:?}",
                                    qarg
                                )));
                            }
                            self.num_qubits =
                                Some(self.num_qubits.unwrap_or_default().max(
                                    qarg.iter().fold(0, |acc, x| {
                                        if acc > x.index() {
                                            acc
                                        } else {
                                            x.index()
                                        }
                                    }) + 1,
                                ));
                        }
                        let inst_properties = properties.swap_remove(qarg.as_ref()).unwrap();
                        qargs_val.insert(qarg.clone(), inst_properties);
                        if let Some(Some(value)) = self.qarg_gate_map.get_mut(qarg.as_ref()) {
                            value.insert(name.to_string());
                        } else {
                            self.qarg_gate_map
                                .insert(qarg, Some(HashSet::from_iter([name.to_string()])));
                        }
                    }
                } else {
                    qargs_val = PropsMap::with_capacity(0);
                }
            }
        }
        self._gate_name_map.insert(name.to_string(), instruction);
        self.gate_map.insert(name.to_string(), qargs_val);
        self.non_global_basis = None;
        self.non_global_strict_basis = None;
        Ok(())
    }

    /// Update the property object for an instruction qarg pair already in the `Target`
    ///
    /// Args:
    ///     instruction (str): The instruction name to update
    ///     qargs (tuple): The qargs to update the properties of
    ///     properties (InstructionProperties): The properties to set for this instruction
    /// Raises:
    ///     KeyError: If ``instruction`` or ``qarg`` are not in the target
    #[pyo3(text_signature = "(instruction, qargs, properties, /,)")]
    fn update_instruction_properties(
        &mut self,
        instruction: String,
        qargs: Option<Qargs>,
        properties: Option<InstructionProperties>,
    ) -> PyResult<()> {
        if !self.contains_key(&instruction) {
            return Err(PyKeyError::new_err(format!(
                "Provided instruction: '{:?}' not in this Target.",
                &instruction
            )));
        };
        let mut prop_map = self[&instruction].clone();
        if !(prop_map.contains_key(qargs.as_ref())) {
            return Err(PyKeyError::new_err(format!(
                "Provided qarg {:?} not in this Target for {:?}.",
                &qargs.unwrap_or_default(),
                &instruction
            )));
        }
        if let Some(e) = prop_map.get_mut(qargs.as_ref()) {
            *e = properties;
        }
        self.gate_map
            .entry(instruction)
            .and_modify(|e| *e = prop_map);
        Ok(())
    }

    /// Get the qargs for a given operation name
    ///
    /// Args:
    ///     operation (str): The operation name to get qargs for
    /// Returns:
    ///     set: The set of qargs the gate instance applies to.
    #[pyo3(name = "qargs_for_operation_name")]
    pub fn py_qargs_for_operation_name(
        &self,
        py: Python,
        operation: &str,
    ) -> PyResult<Option<Vec<PyObject>>> {
        match self.qargs_for_operation_name(operation) {
            Ok(option_set) => {
                Ok(option_set.map(|qargs| qargs.map(|qargs| qargs.to_object(py)).collect()))
            }
            Err(e) => Err(PyKeyError::new_err(e.message)),
        }
    }

    /// Get the operation class object for a given name
    ///
    /// Args:
    ///     instruction (str): The instruction name to get the
    ///         :class:`~qiskit.circuit.Instruction` instance for
    /// Returns:
    ///     qiskit.circuit.Instruction: The Instruction instance corresponding to the
    ///     name. This also can also be the class for globally defined variable with
    ///     operations.
    #[pyo3(name = "operation_from_name")]
    pub fn py_operation_from_name(&self, py: Python, instruction: &str) -> PyResult<PyObject> {
        match self._operation_from_name(instruction) {
            Ok(instruction) => Ok(instruction.to_object(py)),
            Err(e) => Err(PyKeyError::new_err(e.message)),
        }
    }

    /// Get the operation class object for a specified qargs tuple
    ///
    /// Args:
    ///     qargs (tuple): A qargs tuple of the qubits to get the gates that apply
    ///         to it. For example, ``(0,)`` will return the set of all
    ///         instructions that apply to qubit 0. If set to ``None`` this will
    ///         return any globally defined operations in the target.
    /// Returns:
    ///     list: The list of :class:`~qiskit.circuit.Instruction` instances
    ///     that apply to the specified qarg. This may also be a class if
    ///     a variable width operation is globally defined.
    ///
    /// Raises:
    ///     KeyError: If qargs is not in target
    #[pyo3(name = "operations_for_qargs", signature=(qargs=None, /))]
    pub fn py_operations_for_qargs(
        &self,
        py: Python,
        qargs: Option<Qargs>,
    ) -> PyResult<Vec<PyObject>> {
        // Move to rust native once Gates are in rust
        Ok(self
            .py_operation_names_for_qargs(qargs)?
            .into_iter()
            .map(|x| self._gate_name_map[x].to_object(py))
            .collect())
    }

    /// Get the operation names for a specified qargs tuple
    ///
    /// Args:
    ///     qargs (tuple): A ``qargs`` tuple of the qubits to get the gates that apply
    ///         to it. For example, ``(0,)`` will return the set of all
    ///         instructions that apply to qubit 0. If set to ``None`` this will
    ///         return the names for any globally defined operations in the target.
    /// Returns:
    ///     set: The set of operation names that apply to the specified ``qargs``.
    ///
    /// Raises:
    ///     KeyError: If ``qargs`` is not in target
    #[pyo3(name = "operation_names_for_qargs", signature=(qargs=None, /))]
    pub fn py_operation_names_for_qargs(&self, qargs: Option<Qargs>) -> PyResult<HashSet<&str>> {
        match self.operation_names_for_qargs(qargs.as_ref()) {
            Ok(set) => Ok(set),
            Err(e) => Err(PyKeyError::new_err(e.message)),
        }
    }

    /// Return whether the instruction (operation + qubits) is supported by the target
    ///
    /// Args:
    ///     operation_name (str): The name of the operation for the instruction. Either
    ///         this or ``operation_class`` must be specified, if both are specified
    ///         ``operation_class`` will take priority and this argument will be ignored.
    ///     qargs (tuple): The tuple of qubit indices for the instruction. If this is
    ///         not specified then this method will return ``True`` if the specified
    ///         operation is supported on any qubits. The typical application will
    ///         always have this set (otherwise it's the same as just checking if the
    ///         target contains the operation). Normally you would not set this argument
    ///         if you wanted to check more generally that the target supports an operation
    ///         with the ``parameters`` on any qubits.
    ///     operation_class (Type[qiskit.circuit.Instruction]): The operation class to check whether
    ///         the target supports a particular operation by class rather
    ///         than by name. This lookup is more expensive as it needs to
    ///         iterate over all operations in the target instead of just a
    ///         single lookup. If this is specified it will supersede the
    ///         ``operation_name`` argument. The typical use case for this
    ///         operation is to check whether a specific variant of an operation
    ///         is supported on the backend. For example, if you wanted to
    ///         check whether a :class:`~.RXGate` was supported on a specific
    ///         qubit with a fixed angle. That fixed angle variant will
    ///         typically have a name different from the object's
    ///         :attr:`~.Instruction.name` attribute (``"rx"``) in the target.
    ///         This can be used to check if any instances of the class are
    ///         available in such a case.
    ///     parameters (list): A list of parameters to check if the target
    ///         supports them on the specified qubits. If the instruction
    ///         supports the parameter values specified in the list on the
    ///         operation and qargs specified this will return ``True`` but
    ///         if the parameters are not supported on the specified
    ///         instruction it will return ``False``. If this argument is not
    ///         specified this method will return ``True`` if the instruction
    ///         is supported independent of the instruction parameters. If
    ///         specified with any :class:`~.Parameter` objects in the list,
    ///         that entry will be treated as supporting any value, however parameter names
    ///         will not be checked (for example if an operation in the target
    ///         is listed as parameterized with ``"theta"`` and ``"phi"`` is
    ///         passed into this function that will return ``True``). For
    ///         example, if called with::
    ///
    ///             parameters = [Parameter("theta")]
    ///             target.instruction_supported("rx", (0,), parameters=parameters)
    ///
    ///         will return ``True`` if an :class:`~.RXGate` is supported on qubit 0
    ///         that will accept any parameter. If you need to check for a fixed numeric
    ///         value parameter this argument is typically paired with the ``operation_class``
    ///         argument. For example::
    ///
    ///             target.instruction_supported("rx", (0,), RXGate, parameters=[pi / 4])
    ///
    ///         will return ``True`` if an RXGate(pi/4) exists on qubit 0.
    ///
    /// Returns:
    ///     bool: Returns ``True`` if the instruction is supported and ``False`` if it isn't.
    #[pyo3(
        name = "instruction_supported",
        signature = (operation_name=None, qargs=None, operation_class=None, parameters=None)
    )]
    pub fn py_instruction_supported(
        &self,
        py: Python,
        operation_name: Option<String>,
        qargs: Option<Qargs>,
        operation_class: Option<&Bound<PyAny>>,
        parameters: Option<Vec<Param>>,
    ) -> PyResult<bool> {
        let mut qargs = qargs;
        if self.num_qubits.is_none() {
            qargs = None;
        }
        if let Some(_operation_class) = operation_class {
            for (op_name, obj) in self._gate_name_map.iter() {
                match obj {
                    TargetOperation::Variadic(variable) => {
                        if !_operation_class.eq(variable)? {
                            continue;
                        }
                        // If no qargs operation class is supported
                        if let Some(_qargs) = &qargs {
                            let qarg_set: HashSet<PhysicalQubit> = _qargs.iter().cloned().collect();
                            // If qargs set then validate no duplicates and all indices are valid on device
                            return Ok(_qargs
                                .iter()
                                .all(|qarg| qarg.index() <= self.num_qubits.unwrap_or_default())
                                && qarg_set.len() == _qargs.len());
                        } else {
                            return Ok(true);
                        }
                    }
                    TargetOperation::Normal(normal) => {
                        if python_is_instance(py, normal, _operation_class)? {
                            if let Some(parameters) = &parameters {
                                if parameters.len() != normal.params.len() {
                                    continue;
                                }
                                if !check_obj_params(parameters, normal) {
                                    continue;
                                }
                            }
                            if let Some(_qargs) = &qargs {
                                if self.gate_map.contains_key(op_name) {
                                    let gate_map_name = &self.gate_map[op_name];
                                    if gate_map_name.contains_key(qargs.as_ref()) {
                                        return Ok(true);
                                    }
                                    if gate_map_name.contains_key(None) {
                                        let qubit_comparison =
                                            self._gate_name_map[op_name].num_qubits();
                                        return Ok(qubit_comparison == _qargs.len() as u32
                                            && _qargs.iter().all(|x| {
                                                x.index() < self.num_qubits.unwrap_or_default()
                                            }));
                                    }
                                } else {
                                    let qubit_comparison = obj.num_qubits();
                                    return Ok(qubit_comparison == _qargs.len() as u32
                                        && _qargs.iter().all(|x| {
                                            x.index() < self.num_qubits.unwrap_or_default()
                                        }));
                                }
                            } else {
                                return Ok(true);
                            }
                        }
                    }
                }
            }
            Ok(false)
        } else if let Some(operation_name) = operation_name {
            if let Some(parameters) = parameters {
                if let Some(obj) = self._gate_name_map.get(&operation_name) {
                    if self.variable_class_operations.contains(&operation_name) {
                        if let Some(_qargs) = qargs {
                            let qarg_set: HashSet<PhysicalQubit> = _qargs.iter().cloned().collect();
                            return Ok(_qargs
                                .iter()
                                .all(|qarg| qarg.index() <= self.num_qubits.unwrap_or_default())
                                && qarg_set.len() == _qargs.len());
                        } else {
                            return Ok(true);
                        }
                    }

                    let obj_params = obj.params();
                    if parameters.len() != obj_params.len() {
                        return Ok(false);
                    }
                    for (index, params) in parameters.iter().enumerate() {
                        let mut matching_params = false;
                        let obj_at_index = &obj_params[index];
                        if matches!(obj_at_index, Param::ParameterExpression(_))
                            || python_compare(py, &params, &obj_params[index])?
                        {
                            matching_params = true;
                        }
                        if !matching_params {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }
            }
            Ok(self.instruction_supported(&operation_name, qargs.as_ref()))
        } else {
            Ok(false)
        }
    }

    /// Get the instruction properties for a specific instruction tuple
    ///
    /// This method is to be used in conjunction with the
    /// :attr:`~qiskit.transpiler.Target.instructions` attribute of a
    /// :class:`~qiskit.transpiler.Target` object. You can use this method to quickly
    /// get the instruction properties for an element of
    /// :attr:`~qiskit.transpiler.Target.instructions` by using the index in that list.
    /// However, if you're not working with :attr:`~qiskit.transpiler.Target.instructions`
    /// directly it is likely more efficient to access the target directly via the name
    /// and qubits to get the instruction properties. For example, if
    /// :attr:`~qiskit.transpiler.Target.instructions` returned::
    ///
    ///     [(XGate(), (0,)), (XGate(), (1,))]
    ///
    /// you could get the properties of the ``XGate`` on qubit 1 with::
    ///
    ///     props = target.instruction_properties(1)
    ///
    /// but just accessing it directly via the name would be more efficient::
    ///
    ///     props = target['x'][(1,)]
    ///
    /// (assuming the ``XGate``'s canonical name in the target is ``'x'``)
    /// This is especially true for larger targets as this will scale worse with the number
    /// of instruction tuples in a target.
    ///
    /// Args:
    ///     index (int): The index of the instruction tuple from the
    ///         :attr:`~qiskit.transpiler.Target.instructions` attribute. For, example
    ///         if you want the properties from the third element in
    ///         :attr:`~qiskit.transpiler.Target.instructions` you would set this to be ``2``.
    /// Returns:
    ///     InstructionProperties: The instruction properties for the specified instruction tuple
    pub fn instruction_properties(&self, index: usize) -> PyResult<Option<InstructionProperties>> {
        let mut index_counter = 0;
        for (_operation, props_map) in self.gate_map.iter() {
            let gate_map_oper = props_map.values();
            for inst_props in gate_map_oper {
                if index_counter == index {
                    return Ok(inst_props.clone());
                }
                index_counter += 1;
            }
        }
        Err(PyIndexError::new_err(format!(
            "Index: {:?} is out of range.",
            index
        )))
    }

    /// Return the non-global operation names for the target
    ///
    /// The non-global operations are those in the target which don't apply
    /// on all qubits (for single qubit operations) or all multi-qubit qargs
    /// (for multi-qubit operations).
    ///
    /// Args:
    ///     strict_direction (bool): If set to ``True`` the multi-qubit
    ///         operations considered as non-global respect the strict
    ///         direction (or order of qubits in the qargs is significant). For
    ///         example, if ``cx`` is defined on ``(0, 1)`` and ``ecr`` is
    ///         defined over ``(1, 0)`` by default neither would be considered
    ///         non-global, but if ``strict_direction`` is set ``True`` both
    ///         ``cx`` and ``ecr`` would be returned.
    ///
    /// Returns:
    ///     List[str]: A list of operation names for operations that aren't global in this target
    #[pyo3(name = "get_non_global_operation_names", signature = (/, strict_direction=false,))]
    fn py_get_non_global_operation_names(
        &mut self,
        py: Python<'_>,
        strict_direction: bool,
    ) -> PyObject {
        self.get_non_global_operation_names(strict_direction)
            .to_object(py)
    }

    // Instance attributes

    /// The set of qargs in the target.
    #[getter]
    #[pyo3(name = "qargs")]
    fn py_qargs(&self, py: Python) -> PyResult<PyObject> {
        if let Some(qargs) = self.qargs() {
            let qargs = qargs.map(|qargs| qargs.map(|q| PyTuple::new_bound(py, q)));
            let set = PySet::empty_bound(py)?;
            for qargs in qargs {
                set.add(qargs)?;
            }
            Ok(set.into_any().unbind())
        } else {
            Ok(py.None())
        }
    }

    /// Get the list of tuples ``(:class:`~qiskit.circuit.Instruction`, (qargs))``
    /// for the target
    ///
    /// For globally defined variable width operations the tuple will be of the form
    /// ``(class, None)`` where class is the actual operation class that
    /// is globally defined.
    #[getter]
    #[pyo3(name = "instructions")]
    pub fn py_instructions(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::empty_bound(py);
        for (inst, qargs) in self._instructions() {
            let qargs = qargs.map(|q| PyTuple::new_bound(py, q).unbind());
            list.append((inst, qargs))?;
        }
        Ok(list.unbind())
    }
    /// Get the operation names in the target.
    #[getter]
    #[pyo3(name = "operation_names")]
    fn py_operation_names(&self, py: Python<'_>) -> Py<PyList> {
        PyList::new_bound(py, self.operation_names()).unbind()
    }

    /// Get the operation objects in the target.
    #[getter]
    #[pyo3(name = "operations")]
    fn py_operations(&self, py: Python<'_>) -> Py<PyList> {
        PyList::new_bound(py, self._gate_name_map.values()).unbind()
    }

    /// Returns a sorted list of physical qubits.
    #[getter]
    #[pyo3(name = "physical_qubits")]
    fn py_physical_qubits(&self, py: Python<'_>) -> Py<PyList> {
        PyList::new_bound(py, self.physical_qubits()).unbind()
    }

    // Magic methods:

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.gate_map.len())
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let result_list = PyDict::new_bound(py);
        result_list.set_item("description", self.description.clone())?;
        result_list.set_item("num_qubits", self.num_qubits)?;
        result_list.set_item("dt", self.dt)?;
        result_list.set_item("granularity", self.granularity)?;
        result_list.set_item("min_length", self.min_length)?;
        result_list.set_item("pulse_alignment", self.pulse_alignment)?;
        result_list.set_item("acquire_alignment", self.acquire_alignment)?;
        result_list.set_item("qubit_properties", self.qubit_properties.clone())?;
        result_list.set_item(
            "concurrent_measurements",
            self.concurrent_measurements.clone(),
        )?;
        result_list.set_item(
            "gate_map",
            self.gate_map
                .clone()
                .into_iter()
                .map(|(key, value)| {
                    (
                        key,
                        value
                            .into_iter()
                            .collect::<Vec<(Option<Qargs>, Option<InstructionProperties>)>>(),
                    )
                })
                .collect::<GateMapState>()
                .into_py(py),
        )?;
        result_list.set_item("gate_name_map", self._gate_name_map.to_object(py))?;
        result_list.set_item("global_operations", self.global_operations.clone())?;
        result_list.set_item(
            "qarg_gate_map",
            self.qarg_gate_map.clone().into_iter().collect_vec(),
        )?;
        result_list.set_item("non_global_basis", self.non_global_basis.clone())?;
        result_list.set_item(
            "non_global_strict_basis",
            self.non_global_strict_basis.clone(),
        )?;
        Ok(result_list.unbind())
    }

    fn __setstate__(&mut self, state: Bound<PyDict>) -> PyResult<()> {
        self.description = state
            .get_item("description")?
            .unwrap()
            .extract::<Option<String>>()?;
        self.num_qubits = state
            .get_item("num_qubits")?
            .unwrap()
            .extract::<Option<usize>>()?;
        self.dt = state.get_item("dt")?.unwrap().extract::<Option<f64>>()?;
        self.granularity = state.get_item("granularity")?.unwrap().extract::<u32>()?;
        self.min_length = state.get_item("min_length")?.unwrap().extract::<usize>()?;
        self.pulse_alignment = state
            .get_item("pulse_alignment")?
            .unwrap()
            .extract::<u32>()?;
        self.acquire_alignment = state
            .get_item("acquire_alignment")?
            .unwrap()
            .extract::<u32>()?;
        self.qubit_properties = state
            .get_item("qubit_properties")?
            .unwrap()
            .extract::<Option<Vec<PyObject>>>()?;
        self.concurrent_measurements = state
            .get_item("concurrent_measurements")?
            .unwrap()
            .extract::<Option<Vec<Vec<PhysicalQubit>>>>()?;
        self.gate_map = IndexMap::from_iter(
            state
                .get_item("gate_map")?
                .unwrap()
                .extract::<GateMapState>()?
                .into_iter()
                .map(|(name, prop_map)| (name, PropsMap::from_iter(prop_map))),
        );
        self._gate_name_map = state
            .get_item("gate_name_map")?
            .unwrap()
            .extract::<IndexMap<String, TargetOperation, RandomState>>()?;
        self.global_operations = state
            .get_item("global_operations")?
            .unwrap()
            .extract::<IndexMap<u32, HashSet<String>, RandomState>>()?;
        self.qarg_gate_map = NullableIndexMap::from_iter(
            state
                .get_item("qarg_gate_map")?
                .unwrap()
                .extract::<Vec<(Option<Qargs>, Option<HashSet<String>>)>>()?,
        );
        self.non_global_basis = state
            .get_item("non_global_basis")?
            .unwrap()
            .extract::<Option<Vec<String>>>()?;
        self.non_global_strict_basis = state
            .get_item("non_global_strict_basis")?
            .unwrap()
            .extract::<Option<Vec<String>>>()?;
        Ok(())
    }
}

// Rust native methods
impl Target {
    /// Returns an iterator over all the instructions present in the `Target`
    /// as pair of `&OperationType`, `&SmallVec<[Param; 3]>` and `Option<&Qargs>`.
    // TODO: Remove once `Target` is being consumed.
    #[allow(dead_code)]
    pub fn instructions(&self) -> impl Iterator<Item = (&NormalOperation, Option<&Qargs>)> {
        self._instructions()
            .filter_map(|(operation, qargs)| match &operation {
                TargetOperation::Normal(oper) => Some((oper, qargs)),
                _ => None,
            })
    }

    /// Returns an iterator over all the instructions present in the `Target`
    /// as pair of `&TargetOperation` and `Option<&Qargs>`.
    fn _instructions(&self) -> impl Iterator<Item = (&TargetOperation, Option<&Qargs>)> {
        self.gate_map.iter().flat_map(move |(op, props_map)| {
            props_map
                .keys()
                .map(move |qargs| (&self._gate_name_map[op], qargs))
        })
    }
    /// Returns an iterator over the operation names in the target.
    // TODO: Remove once `Target` is being consumed.
    #[allow(dead_code)]
    pub fn operation_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.gate_map.keys().map(|x| x.as_str())
    }

    /// Get the `OperationType` objects present in the target.
    // TODO: Remove once `Target` is being consumed.
    #[allow(dead_code)]
    pub fn operations(&self) -> impl Iterator<Item = &NormalOperation> {
        return self._gate_name_map.values().filter_map(|oper| match oper {
            TargetOperation::Normal(oper) => Some(oper),
            _ => None,
        });
    }

    /// Get the error rate of a given instruction in the target
    pub fn get_error(&self, name: &str, qargs: &[PhysicalQubit]) -> Option<f64> {
        self.gate_map.get(name).and_then(|gate_props| {
            let qargs_key: Qargs = qargs.iter().cloned().collect();
            match gate_props.get(Some(&qargs_key)) {
                Some(props) => props.as_ref().and_then(|inst_props| inst_props.error),
                None => None,
            }
        })
    }

    /// Get an iterator over the indices of all physical qubits of the target
    pub fn physical_qubits(&self) -> impl ExactSizeIterator<Item = usize> {
        0..self.num_qubits.unwrap_or_default()
    }

    /// Generate non global operations if missing
    fn generate_non_global_op_names(&mut self, strict_direction: bool) -> &[String] {
        let mut search_set: HashSet<Qargs> = HashSet::default();
        if strict_direction {
            // Build search set
            search_set = self.qarg_gate_map.keys().flatten().cloned().collect();
        } else {
            for qarg_key in self.qarg_gate_map.keys().flatten() {
                if qarg_key.len() != 1 {
                    let mut vec = qarg_key.clone();
                    vec.sort_unstable();
                    search_set.insert(vec);
                }
            }
        }
        let mut incomplete_basis_gates: Vec<String> = vec![];
        let mut size_dict: IndexMap<usize, usize, RandomState> = IndexMap::default();
        *size_dict
            .entry(1)
            .or_insert(self.num_qubits.unwrap_or_default()) = self.num_qubits.unwrap_or_default();
        for qarg in &search_set {
            if qarg.len() == 1 {
                continue;
            }
            *size_dict.entry(qarg.len()).or_insert(0) += 1;
        }
        for (inst, qargs_props) in self.gate_map.iter() {
            let mut qarg_len = qargs_props.len();
            let mut qargs_keys = qargs_props.keys().peekable();
            let qarg_sample = qargs_keys.peek().cloned();
            if let Some(qarg_sample) = qarg_sample {
                if qarg_sample.is_none() {
                    continue;
                }
                if !strict_direction {
                    let mut deduplicated_qargs: HashSet<SmallVec<[PhysicalQubit; 2]>> =
                        HashSet::default();
                    for qarg in qargs_keys.flatten() {
                        let mut ordered_qargs = qarg.clone();
                        ordered_qargs.sort_unstable();
                        deduplicated_qargs.insert(ordered_qargs);
                    }
                    qarg_len = deduplicated_qargs.len();
                }
                if let Some(qarg_sample) = qarg_sample {
                    if qarg_len != *size_dict.entry(qarg_sample.len()).or_insert(0) {
                        incomplete_basis_gates.push(inst.clone());
                    }
                }
            }
        }
        if strict_direction {
            self.non_global_strict_basis = Some(incomplete_basis_gates);
            self.non_global_strict_basis.as_ref().unwrap()
        } else {
            self.non_global_basis = Some(incomplete_basis_gates.clone());
            self.non_global_basis.as_ref().unwrap()
        }
    }

    /// Get all non_global operation names.
    pub fn get_non_global_operation_names(&mut self, strict_direction: bool) -> Option<&[String]> {
        if strict_direction {
            if self.non_global_strict_basis.is_some() {
                return self.non_global_strict_basis.as_deref();
            }
        } else if self.non_global_basis.is_some() {
            return self.non_global_basis.as_deref();
        }
        return Some(self.generate_non_global_op_names(strict_direction));
    }

    /// Gets all the operation names that use these qargs. Rust native equivalent of ``BaseTarget.operation_names_for_qargs()``
    pub fn operation_names_for_qargs(
        &self,
        qargs: Option<&Qargs>,
    ) -> Result<HashSet<&str>, TargetKeyError> {
        // When num_qubits == 0 we return globally defined operators
        let mut res: HashSet<&str> = HashSet::default();
        let mut qargs = qargs;
        if self.num_qubits.unwrap_or_default() == 0 || self.num_qubits.is_none() {
            qargs = None;
        }
        if let Some(qargs) = qargs.as_ref() {
            if qargs
                .iter()
                .any(|x| !(0..self.num_qubits.unwrap_or_default()).contains(&x.index()))
            {
                return Err(TargetKeyError::new_err(format!(
                    "{:?} not in Target",
                    qargs
                )));
            }
        }
        if let Some(Some(qarg_gate_map_arg)) = self.qarg_gate_map.get(qargs).as_ref() {
            res.extend(qarg_gate_map_arg.iter().map(|key| key.as_str()));
        }
        for name in self._gate_name_map.keys() {
            if self.variable_class_operations.contains(name) {
                res.insert(name);
            }
        }
        if let Some(qargs) = qargs.as_ref() {
            if let Some(global_gates) = self.global_operations.get(&(qargs.len() as u32)) {
                res.extend(global_gates.iter().map(|key| key.as_str()))
            }
        }
        if res.is_empty() {
            return Err(TargetKeyError::new_err(format!(
                "{:?} not in target",
                qargs
            )));
        }
        Ok(res)
    }

    /// Returns an iterator of `OperationType` instances and parameters present in the Target that affect the provided qargs.
    // TODO: Remove once `Target` is being consumed.
    #[allow(dead_code)]
    pub fn operations_for_qargs(
        &self,
        qargs: Option<&Qargs>,
    ) -> Result<impl Iterator<Item = &NormalOperation>, TargetKeyError> {
        self.operation_names_for_qargs(qargs).map(|operations| {
            operations
                .into_iter()
                .filter_map(|oper| match &self._gate_name_map[oper] {
                    TargetOperation::Normal(normal) => Some(normal),
                    _ => None,
                })
        })
    }

    /// Gets an iterator with all the qargs used by the specified operation name.
    ///
    /// Rust native equivalent of ``BaseTarget.qargs_for_operation_name()``
    pub fn qargs_for_operation_name(
        &self,
        operation: &str,
    ) -> Result<Option<impl Iterator<Item = &Qargs>>, TargetKeyError> {
        if let Some(gate_map_oper) = self.gate_map.get(operation) {
            if gate_map_oper.contains_key(None) {
                return Ok(None);
            }
            let qargs = gate_map_oper.keys().flatten();
            Ok(Some(qargs))
        } else {
            Err(TargetKeyError::new_err(format!(
                "Operation: {operation} not in Target."
            )))
        }
    }

    /// Gets a tuple of Operation object and Parameters based on the operation name if present in the Target.
    // TODO: Remove once `Target` is being consumed.
    #[allow(dead_code)]
    pub fn operation_from_name(
        &self,
        instruction: &str,
    ) -> Result<&NormalOperation, TargetKeyError> {
        match self._operation_from_name(instruction) {
            Ok(TargetOperation::Normal(operation)) => Ok(operation),
            Ok(TargetOperation::Variadic(_)) => Err(TargetKeyError::new_err(format!(
                "Instruction {:?} was found in the target, but the instruction is Varidic.",
                instruction
            ))),
            Err(e) => Err(e),
        }
    }

    /// Gets the instruction object based on the operation name
    fn _operation_from_name(&self, instruction: &str) -> Result<&TargetOperation, TargetKeyError> {
        if let Some(gate_obj) = self._gate_name_map.get(instruction) {
            Ok(gate_obj)
        } else {
            Err(TargetKeyError::new_err(format!(
                "Instruction {:?} not in target",
                instruction
            )))
        }
    }

    /// Returns an iterator over all the qargs of a specific Target object
    pub fn qargs(&self) -> Option<impl Iterator<Item = Option<&Qargs>>> {
        let mut qargs = self.qarg_gate_map.keys().peekable();
        let next_entry = qargs.peek();
        let is_none = next_entry.is_none() || next_entry.unwrap().is_none();
        if qargs.len() == 1 && is_none {
            return None;
        }
        Some(qargs)
    }

    /// Checks whether an instruction is supported by the Target based on instruction name and qargs.
    pub fn instruction_supported(&self, operation_name: &str, qargs: Option<&Qargs>) -> bool {
        if self.gate_map.contains_key(operation_name) {
            if let Some(_qargs) = qargs {
                let qarg_set: HashSet<&PhysicalQubit> = _qargs.iter().collect();
                if let Some(gate_prop_name) = self.gate_map.get(operation_name) {
                    if gate_prop_name.contains_key(qargs) {
                        return true;
                    }
                    if gate_prop_name.contains_key(None) {
                        let obj = &self._gate_name_map[operation_name];
                        if self.variable_class_operations.contains(operation_name) {
                            return qargs.is_none()
                                || _qargs.iter().all(|qarg| {
                                    qarg.index() <= self.num_qubits.unwrap_or_default()
                                }) && qarg_set.len() == _qargs.len();
                        } else {
                            let qubit_comparison = obj.num_qubits();
                            return qubit_comparison == _qargs.len() as u32
                                && _qargs.iter().all(|qarg| {
                                    qarg.index() < self.num_qubits.unwrap_or_default()
                                });
                        }
                    }
                } else {
                    // Duplicate case is if it contains none
                    if self.variable_class_operations.contains(operation_name) {
                        return qargs.is_none()
                            || _qargs
                                .iter()
                                .all(|qarg| qarg.index() <= self.num_qubits.unwrap_or_default())
                                && qarg_set.len() == _qargs.len();
                    } else {
                        let qubit_comparison = self._gate_name_map[operation_name].num_qubits();
                        return qubit_comparison == _qargs.len() as u32
                            && _qargs
                                .iter()
                                .all(|qarg| qarg.index() < self.num_qubits.unwrap_or_default());
                    }
                }
            } else {
                return true;
            }
        }
        false
    }

    // IndexMap methods

    /// Retreive all the gate names in the Target
    // TODO: Remove once `Target` is being consumed.
    #[allow(dead_code)]
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.gate_map.keys().map(|x| x.as_str())
    }

    /// Retrieves an iterator over the property maps stored within the Target
    // TODO: Remove once `Target` is being consumed.
    #[allow(dead_code)]
    pub fn values(&self) -> impl Iterator<Item = &PropsMap> {
        self.gate_map.values()
    }

    /// Checks if a key exists in the Target
    pub fn contains_key(&self, key: &str) -> bool {
        self.gate_map.contains_key(key)
    }
}

// To access the Target's gate map by gate name.
impl Index<&str> for Target {
    type Output = PropsMap;
    fn index(&self, index: &str) -> &Self::Output {
        self.gate_map.index(index)
    }
}

// For instruction_supported
fn check_obj_params(parameters: &[Param], obj: &NormalOperation) -> bool {
    for (index, param) in parameters.iter().enumerate() {
        let param_at_index = &obj.params[index];
        match (param, param_at_index) {
            (Param::Float(p1), Param::Float(p2)) => {
                if p1 != p2 {
                    return false;
                }
            }
            (&Param::ParameterExpression(_), Param::Float(_)) => return false,
            (&Param::ParameterExpression(_), Param::Obj(_)) => return false,
            _ => continue,
        }
    }
    true
}

pub fn python_compare<T, U>(py: Python, obj: &T, other: &U) -> PyResult<bool>
where
    T: ToPyObject,
    U: ToPyObject,
{
    let obj = obj.to_object(py);
    let obj_bound = obj.bind(py);
    obj_bound.eq(other)
}

pub fn python_is_instance<T, U>(py: Python, obj: &T, other: &U) -> PyResult<bool>
where
    T: ToPyObject,
    U: ToPyObject,
{
    let obj = obj.to_object(py);
    let other_obj = other.to_object(py);
    let obj_bound = obj.bind(py);
    obj_bound.is_instance(other_obj.bind(py))
}

pub fn target(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<InstructionProperties>()?;
    m.add_class::<Target>()?;
    Ok(())
}

// TODO: Add rust-based unit testing.

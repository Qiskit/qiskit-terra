// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
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

use numpy::IntoPyArray;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyDeprecationWarning, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple, PyType};
use pyo3::{intern, IntoPy, PyObject, PyResult};

use smallvec::SmallVec;

use crate::imports::{
    CONTROLLED_GATE, CONTROL_FLOW_OP, GATE, INSTRUCTION, OPERATION, WARNINGS_WARN,
};
use crate::operations::{
    Operation, OperationRef, Param, PyGate, PyInstruction, PyOperation, StandardGate,
};
use crate::packed_instruction::PackedOperation;

/// These are extra mutable attributes for a circuit instruction's state. In general we don't
/// typically deal with this in rust space and the majority of the time they're not used in Python
/// space either. To save memory these are put in a separate struct and are stored inside a
/// `Box` on `CircuitInstruction` and `PackedInstruction`.
#[derive(Debug, Clone)]
pub struct ExtraInstructionAttributes {
    pub label: Option<String>,
    pub duration: Option<PyObject>,
    pub unit: Option<String>,
    pub condition: Option<PyObject>,
}

impl ExtraInstructionAttributes {
    /// Construct a new set of the extra attributes if any of the elements are not `None`, or return
    /// `None` if there is no need for an object.
    #[inline]
    pub fn new(
        label: Option<String>,
        duration: Option<Py<PyAny>>,
        unit: Option<String>,
        condition: Option<Py<PyAny>>,
    ) -> Option<Self> {
        if label.is_some() || duration.is_some() || unit.is_some() || condition.is_some() {
            Some(Self {
                label,
                duration,
                unit,
                condition,
            })
        } else {
            None
        }
    }

    /// Get the Python-space version of the stored `unit`.  This evalutes the Python-space default
    /// (`"dt"`) value if we're storing a `None`.
    pub fn py_unit(&self, py: Python) -> Py<PyString> {
        self.unit
            .as_deref()
            .map(|unit| <&str as IntoPy<Py<PyString>>>::into_py(unit, py))
            .unwrap_or_else(|| Self::default_unit(py).clone().unbind())
    }

    /// Get the Python-space default value for the `unit` field.
    pub fn default_unit(py: Python) -> &Bound<PyString> {
        intern!(py, "dt")
    }
}

/// A single instruction in a :class:`.QuantumCircuit`, comprised of the :attr:`operation` and
/// various operands.
///
/// .. note::
///
///     There is some possible confusion in the names of this class, :class:`~.circuit.Instruction`,
///     and :class:`~.circuit.Operation`, and this class's attribute :attr:`operation`.  Our
///     preferred terminology is by analogy to assembly languages, where an "instruction" is made up
///     of an "operation" and its "operands".
///
///     Historically, :class:`~.circuit.Instruction` came first, and originally contained the qubits
///     it operated on and any parameters, so it was a true "instruction".  Over time,
///     :class:`.QuantumCircuit` became responsible for tracking qubits and clbits, and the class
///     became better described as an "operation".  Changing the name of such a core object would be
///     a very unpleasant API break for users, and so we have stuck with it.
///
///     This class was created to provide a formal "instruction" context object in
///     :class:`.QuantumCircuit.data`, which had long been made of ad-hoc tuples.  With this, and
///     the advent of the :class:`~.circuit.Operation` interface for adding more complex objects to
///     circuits, we took the opportunity to correct the historical naming.  For the time being,
///     this leads to an awkward case where :attr:`.CircuitInstruction.operation` is often an
///     :class:`~.circuit.Instruction` instance (:class:`~.circuit.Instruction` implements the
///     :class:`.Operation` interface), but as the :class:`.Operation` interface gains more use,
///     this confusion will hopefully abate.
///
/// .. warning::
///
///     This is a lightweight internal class and there is minimal error checking; you must respect
///     the type hints when using it.  It is the user's responsibility to ensure that direct
///     mutations of the object do not invalidate the types, nor the restrictions placed on it by
///     its context.  Typically this will mean, for example, that :attr:`qubits` must be a sequence
///     of distinct items, with no duplicates.
#[pyclass(freelist = 20, sequence, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct CircuitInstruction {
    pub operation: PackedOperation,
    /// A sequence of the qubits that the operation is applied to.
    #[pyo3(get)]
    pub qubits: Py<PyTuple>,
    /// A sequence of the classical bits that this operation reads from or writes to.
    #[pyo3(get)]
    pub clbits: Py<PyTuple>,
    pub params: SmallVec<[Param; 3]>,
    pub extra_attrs: Option<Box<ExtraInstructionAttributes>>,
    #[cfg(feature = "cache_pygates")]
    pub py_op: OnceCell<Py<PyAny>>,
}

impl CircuitInstruction {
    /// Get the Python-space operation, ensuring that it is mutable from Python space (singleton
    /// gates might not necessarily satisfy this otherwise).
    ///
    /// This returns the cached instruction if valid, but does not replace the cache if it created a
    /// new mutable object; the expectation is that any mutations to the Python object need
    /// assigning back to the `CircuitInstruction` completely to ensure data coherence between Rust
    /// and Python spaces.  We can't protect entirely against that, but we can make it a bit harder
    /// for standard-gate getters to accidentally do the wrong thing.
    pub fn get_operation_mut<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = self.get_operation(py)?.into_bound(py);
        if out.getattr(intern!(py, "mutable"))?.is_truthy()? {
            Ok(out)
        } else {
            out.call_method0(intern!(py, "to_mutable"))
        }
    }

    pub fn condition(&self) -> Option<&PyObject> {
        self.extra_attrs
            .as_ref()
            .and_then(|args| args.condition.as_ref())
    }
}

#[pymethods]
impl CircuitInstruction {
    #[new]
    #[pyo3(signature = (operation, qubits=None, clbits=None))]
    pub fn py_new(
        operation: &Bound<PyAny>,
        qubits: Option<Bound<PyAny>>,
        clbits: Option<Bound<PyAny>>,
    ) -> PyResult<Self> {
        let py = operation.py();
        let op_parts = operation.extract::<OperationFromPython>()?;

        Ok(Self {
            operation: op_parts.operation,
            qubits: as_tuple(py, qubits)?.unbind(),
            clbits: as_tuple(py, clbits)?.unbind(),
            params: op_parts.params,
            extra_attrs: op_parts.extra_attrs,
            #[cfg(feature = "cache_pygates")]
            py_op: operation.into_py(py).into(),
        })
    }

    #[pyo3(signature = (standard, qubits, params, label=None))]
    #[staticmethod]
    pub fn from_standard(
        py: Python,
        standard: StandardGate,
        qubits: Option<Bound<PyAny>>,
        params: SmallVec<[Param; 3]>,
        label: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            operation: standard.into(),
            qubits: as_tuple(py, qubits)?.unbind(),
            clbits: PyTuple::empty_bound(py).unbind(),
            params,
            extra_attrs: label.map(|label| {
                Box::new(ExtraInstructionAttributes {
                    label: Some(label),
                    duration: None,
                    unit: None,
                    condition: None,
                })
            }),
            #[cfg(feature = "cache_pygates")]
            py_op: OnceCell::new(),
        })
    }

    /// Returns a shallow copy.
    ///
    /// Returns:
    ///     CircuitInstruction: The shallow copy.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// The logical operation that this instruction represents an execution of.
    #[getter]
    pub fn get_operation(&self, py: Python) -> PyResult<PyObject> {
        // This doesn't use `get_or_init` because a) the initialiser is fallible and
        // `get_or_try_init` isn't stable, and b) the initialiser can yield to the Python
        // interpreter, which might suspend the thread and allow another to inadvertantly attempt to
        // re-enter the cache setter, which isn't safe.

        #[cfg(feature = "cache_pygates")]
        {
            if let Some(cached_op) = self.py_op.get() {
                return Ok(cached_op.clone_ref(py));
            }
        }

        let out = match self.operation.view() {
            OperationRef::Standard(standard) => standard
                .create_py_op(py, Some(&self.params), self.extra_attrs.as_deref())?
                .into_any(),
            OperationRef::Gate(gate) => gate.gate.clone_ref(py),
            OperationRef::Instruction(instruction) => instruction.instruction.clone_ref(py),
            OperationRef::Operation(operation) => operation.operation.clone_ref(py),
        };

        #[cfg(feature = "cache_pygates")]
        {
            self.py_op.get_or_init(|| out.clone_ref(py));
        }

        Ok(out)
    }

    /// Returns the Instruction name corresponding to the op for this node
    #[getter]
    fn get_name(&self, py: Python) -> PyObject {
        self.operation.name().to_object(py)
    }

    #[getter]
    fn get_params(&self, py: Python) -> PyObject {
        self.params.to_object(py)
    }

    #[getter]
    fn matrix(&self, py: Python) -> Option<PyObject> {
        let matrix = self.operation.view().matrix(&self.params);
        matrix.map(|mat| mat.into_pyarray_bound(py).into())
    }

    #[getter]
    fn label(&self) -> Option<&str> {
        self.extra_attrs
            .as_ref()
            .and_then(|attrs| attrs.label.as_deref())
    }

    #[getter]
    fn get_condition(&self, py: Python) -> Option<PyObject> {
        self.extra_attrs
            .as_ref()
            .and_then(|attrs| attrs.condition.as_ref().map(|x| x.clone_ref(py)))
    }

    #[getter]
    fn duration(&self, py: Python) -> Option<PyObject> {
        self.extra_attrs
            .as_ref()
            .and_then(|attrs| attrs.duration.as_ref().map(|x| x.clone_ref(py)))
    }

    #[getter]
    fn unit(&self, py: Python) -> Py<PyString> {
        // Python space uses `"dt"` as the default, whereas we simply don't store the extra
        // attributes at all if they're none.
        self.extra_attrs
            .as_ref()
            .map(|attrs| attrs.py_unit(py))
            .unwrap_or_else(|| {
                ExtraInstructionAttributes::default_unit(py)
                    .clone()
                    .unbind()
            })
    }

    /// Is the :class:`.Operation` contained in this instruction a Qiskit standard gate?
    pub fn is_standard_gate(&self) -> bool {
        self.operation.try_standard_gate().is_some()
    }

    /// Is the :class:`.Operation` contained in this instruction a subclass of
    /// :class:`.ControlledGate`?
    pub fn is_controlled_gate(&self, py: Python) -> PyResult<bool> {
        match self.operation.view() {
            OperationRef::Standard(standard) => Ok(standard.num_ctrl_qubits() != 0),
            OperationRef::Gate(gate) => gate
                .gate
                .bind(py)
                .is_instance(CONTROLLED_GATE.get_bound(py)),
            _ => Ok(false),
        }
    }

    /// Is the :class:`.Operation` contained in this node a directive?
    pub fn is_directive(&self) -> bool {
        self.operation.directive()
    }

    /// Is the :class:`.Operation` contained in this instruction a control-flow operation (i.e. an
    /// instance of :class:`.ControlFlowOp`)?
    pub fn is_control_flow(&self) -> bool {
        self.operation.control_flow()
    }

    /// Does this instruction contain any :class:`.ParameterExpression` parameters?
    pub fn is_parameterized(&self) -> bool {
        self.params
            .iter()
            .any(|x| matches!(x, Param::ParameterExpression(_)))
    }

    /// Creates a shallow copy with the given fields replaced.
    ///
    /// Returns:
    ///     CircuitInstruction: A new instance with the given fields replaced.
    pub fn replace(
        &self,
        py: Python,
        operation: Option<&Bound<PyAny>>,
        qubits: Option<Bound<PyAny>>,
        clbits: Option<Bound<PyAny>>,
        params: Option<Bound<PyAny>>,
    ) -> PyResult<Self> {
        let qubits = match qubits {
            None => self.qubits.clone_ref(py),
            Some(qubits) => as_tuple(py, Some(qubits))?.unbind(),
        };
        let clbits = match clbits {
            None => self.clbits.clone_ref(py),
            Some(clbits) => as_tuple(py, Some(clbits))?.unbind(),
        };
        let params = params
            .map(|params| params.extract::<SmallVec<[Param; 3]>>())
            .transpose()?;

        if let Some(operation) = operation {
            let op_parts = operation.extract::<OperationFromPython>()?;
            Ok(Self {
                operation: op_parts.operation,
                qubits,
                clbits,
                params: params.unwrap_or(op_parts.params),
                extra_attrs: op_parts.extra_attrs,
                #[cfg(feature = "cache_pygates")]
                py_op: operation.into_py(py).into(),
            })
        } else {
            Ok(Self {
                operation: self.operation.clone(),
                qubits,
                clbits,
                params: params.unwrap_or_else(|| self.params.clone()),
                extra_attrs: self.extra_attrs.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: self.py_op.clone(),
            })
        }
    }

    pub fn __getnewargs__(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok((
            self.get_operation(py)?,
            self.qubits.bind(py),
            self.clbits.bind(py),
        )
            .into_py(py))
    }

    pub fn __repr__(self_: &Bound<Self>, py: Python<'_>) -> PyResult<String> {
        let type_name = self_.get_type().qualname()?;
        let r = self_.try_borrow()?;
        Ok(format!(
            "{}(operation={}, qubits={}, clbits={})",
            type_name,
            r.get_operation(py)?.bind(py).repr()?,
            r.qubits.bind(py).repr()?,
            r.clbits.bind(py).repr()?
        ))
    }

    // Legacy tuple-like interface support.
    //
    // For a best attempt at API compatibility during the transition to using this new class, we need
    // the interface to behave exactly like the old 3-tuple `(inst, qargs, cargs)` if it's treated
    // like that via unpacking or similar.  That means that the `parameters` field is completely
    // absent, and the qubits and clbits must be converted to lists.
    pub fn _legacy_format<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        Ok(PyTuple::new_bound(
            py,
            [
                self.get_operation(py)?,
                self.qubits.bind(py).to_list().into(),
                self.clbits.bind(py).to_list().into(),
            ],
        ))
    }

    pub fn __getitem__(&self, py: Python<'_>, key: &Bound<PyAny>) -> PyResult<PyObject> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        Ok(self._legacy_format(py)?.as_any().get_item(key)?.into_py(py))
    }

    pub fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        Ok(self._legacy_format(py)?.as_any().iter()?.into_py(py))
    }

    pub fn __len__(&self, py: Python) -> PyResult<usize> {
        warn_on_legacy_circuit_instruction_iteration(py)?;
        Ok(3)
    }

    pub fn __richcmp__(
        self_: &Bound<Self>,
        other: &Bound<PyAny>,
        op: CompareOp,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        fn params_eq(py: Python, left: &[Param], right: &[Param]) -> PyResult<bool> {
            if left.len() != right.len() {
                return Ok(false);
            }
            for (left, right) in left.iter().zip(right) {
                let eq = match left {
                    Param::Float(left) => match right {
                        Param::Float(right) => left == right,
                        Param::ParameterExpression(right) | Param::Obj(right) => {
                            right.bind(py).eq(left)?
                        }
                    },
                    Param::ParameterExpression(left) | Param::Obj(left) => match right {
                        Param::Float(right) => left.bind(py).eq(right)?,
                        Param::ParameterExpression(right) | Param::Obj(right) => {
                            left.bind(py).eq(right)?
                        }
                    },
                };
                if !eq {
                    return Ok(false);
                }
            }
            Ok(true)
        }

        fn eq(
            py: Python<'_>,
            self_: &Bound<CircuitInstruction>,
            other: &Bound<PyAny>,
        ) -> PyResult<Option<bool>> {
            if self_.is(other) {
                return Ok(Some(true));
            }

            let self_ = self_.try_borrow()?;

            if other.is_instance_of::<PyTuple>() {
                return Ok(Some(self_._legacy_format(py)?.eq(other)?));
            }
            let Ok(other) = other.downcast::<CircuitInstruction>() else {
                return Ok(None);
            };
            let other = other.try_borrow()?;

            Ok(Some(
                self_.qubits.bind(py).eq(other.qubits.bind(py))?
                    && self_.clbits.bind(py).eq(other.clbits.bind(py))?
                    && self_.operation.py_eq(py, &other.operation)?
                    && (self_.operation.try_standard_gate().is_none()
                        || params_eq(py, &self_.params, &other.params)?),
            ))
        }

        match op {
            CompareOp::Eq => Ok(eq(py, self_, other)?
                .map(|b| b.into_py(py))
                .unwrap_or_else(|| py.NotImplemented())),
            CompareOp::Ne => Ok(eq(py, self_, other)?
                .map(|b| (!b).into_py(py))
                .unwrap_or_else(|| py.NotImplemented())),
            _ => Ok(py.NotImplemented()),
        }
    }
}

/// A container struct that contains the conversion from some `Operation` subclass input, on its way
/// to becoming a `PackedInstruction`.
///
/// This is the primary way of converting an incoming `Gate` / `Instruction` / `Operation` from
/// Python space into Rust-space data.  A typical access pattern is:
///
/// ```rust
/// #[pyfunction]
/// fn accepts_op_from_python(ob: &Bound<PyAny>) -> PyResult<()> {
///     let py_op = ob.extract::<OperationFromPython>()?;
///     // ... use `py_op.operation`, `py_op.params`, etc.
///     Ok(())
/// }
/// ```
///
/// though you can also accept `ob: OperationFromPython` directly, if you don't also need a handle
/// to the Python object that it came from.  The handle is useful for the Python-operation caching.
#[derive(Debug)]
pub struct OperationFromPython {
    pub operation: PackedOperation,
    pub params: SmallVec<[Param; 3]>,
    pub extra_attrs: Option<Box<ExtraInstructionAttributes>>,
}

impl<'py> FromPyObject<'py> for OperationFromPython {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        let ob_type = ob
            .getattr(intern!(py, "base_class"))
            .ok()
            .map(|base| base.downcast_into::<PyType>())
            .transpose()?
            .unwrap_or_else(|| ob.get_type());

        let extract_params = || {
            ob.getattr(intern!(py, "params"))
                .ok()
                .map(|params| params.extract())
                .transpose()
                .map(|params| params.unwrap_or_default())
        };
        let extract_extra = || -> PyResult<_> {
            let unit = {
                // We accept Python-space `None` or `"dt"` as both meaning the default `"dt"`.
                let raw_unit = ob.getattr(intern!(py, "unit"))?;
                (!(raw_unit.is_none()
                    || raw_unit.eq(ExtraInstructionAttributes::default_unit(py))?))
                .then(|| raw_unit.extract::<String>())
                .transpose()?
            };
            Ok(ExtraInstructionAttributes::new(
                ob.getattr(intern!(py, "label"))?.extract()?,
                ob.getattr(intern!(py, "duration"))?.extract()?,
                unit,
                ob.getattr(intern!(py, "condition"))?.extract()?,
            )
            .map(Box::from))
        };

        'standard: {
            let Some(standard) = ob_type
                .getattr(intern!(py, "_standard_gate"))
                .and_then(|standard| standard.extract::<StandardGate>())
                .ok()
            else {
                break 'standard;
            };

            // If the instruction is a controlled gate with a not-all-ones control state, it doesn't
            // fit our definition of standard.  We abuse the fact that we know our standard-gate
            // mapping to avoid an `isinstance` check on `ControlledGate` - a standard gate has
            // nonzero `num_ctrl_qubits` iff it is a `ControlledGate`.
            //
            // `ControlledGate` also has a `base_gate` attribute related to its historical
            // implementation, which technically allows mutations from Python space.  The only
            // mutation of a standard gate's `base_gate` that wouldn't have already broken the
            // Python-space data model is setting a label, so we just catch that case and default
            // back to non-standard-gate handling in that case.
            if standard.num_ctrl_qubits() != 0
                && ((ob.getattr(intern!(py, "ctrl_state"))?.extract::<usize>()?
                    != (1 << standard.num_ctrl_qubits()) - 1)
                    || !ob
                        .getattr(intern!(py, "base_gate"))?
                        .getattr(intern!(py, "label"))?
                        .is_none())
            {
                break 'standard;
            }
            return Ok(OperationFromPython {
                operation: PackedOperation::from_standard(standard),
                params: extract_params()?,
                extra_attrs: extract_extra()?,
            });
        }
        if ob_type.is_subclass(GATE.get_bound(py))? {
            let params = extract_params()?;
            let gate = Box::new(PyGate {
                qubits: ob.getattr(intern!(py, "num_qubits"))?.extract()?,
                clbits: 0,
                params: params.len() as u32,
                op_name: ob.getattr(intern!(py, "name"))?.extract()?,
                gate: ob.into_py(py),
            });
            return Ok(OperationFromPython {
                operation: PackedOperation::from_gate(gate),
                params,
                extra_attrs: extract_extra()?,
            });
        }
        if ob_type.is_subclass(INSTRUCTION.get_bound(py))? {
            let params = extract_params()?;
            let instruction = Box::new(PyInstruction {
                qubits: ob.getattr(intern!(py, "num_qubits"))?.extract()?,
                clbits: ob.getattr(intern!(py, "num_clbits"))?.extract()?,
                params: params.len() as u32,
                op_name: ob.getattr(intern!(py, "name"))?.extract()?,
                control_flow: ob.is_instance(CONTROL_FLOW_OP.get_bound(py))?,
                instruction: ob.into_py(py),
            });
            return Ok(OperationFromPython {
                operation: PackedOperation::from_instruction(instruction),
                params,
                extra_attrs: extract_extra()?,
            });
        }
        if ob_type.is_subclass(OPERATION.get_bound(py))? {
            let params = extract_params()?;
            let operation = Box::new(PyOperation {
                qubits: ob.getattr(intern!(py, "num_qubits"))?.extract()?,
                clbits: ob.getattr(intern!(py, "num_clbits"))?.extract()?,
                params: params.len() as u32,
                op_name: ob.getattr(intern!(py, "name"))?.extract()?,
                operation: ob.into_py(py),
            });
            return Ok(OperationFromPython {
                operation: PackedOperation::from_operation(operation),
                params,
                extra_attrs: None,
            });
        }
        Err(PyTypeError::new_err(format!("invalid input: {}", ob)))
    }
}

/// Convert a sequence-like Python object to a tuple.
fn as_tuple<'py>(py: Python<'py>, seq: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyTuple>> {
    let Some(seq) = seq else {
        return Ok(PyTuple::empty_bound(py));
    };
    if seq.is_instance_of::<PyTuple>() {
        Ok(seq.downcast_into_exact::<PyTuple>()?)
    } else if seq.is_instance_of::<PyList>() {
        Ok(seq.downcast_exact::<PyList>()?.to_tuple())
    } else {
        // New tuple from iterable.
        Ok(PyTuple::new_bound(
            py,
            seq.iter()?
                .map(|o| Ok(o?.unbind()))
                .collect::<PyResult<Vec<PyObject>>>()?,
        ))
    }
}

/// Issue a Python `DeprecationWarning` about using the legacy tuple-like interface to
/// `CircuitInstruction`.
///
/// Beware the `stacklevel` here doesn't work quite the same way as it does in Python as Rust-space
/// calls are completely transparent to Python.
#[inline]
fn warn_on_legacy_circuit_instruction_iteration(py: Python) -> PyResult<()> {
    WARNINGS_WARN
        .get_bound(py)
        .call1((
            intern!(
                py,
                concat!(
                    "Treating CircuitInstruction as an iterable is deprecated legacy behavior",
                    " since Qiskit 1.2, and will be removed in Qiskit 2.0.",
                    " Instead, use the `operation`, `qubits` and `clbits` named attributes."
                )
            ),
            py.get_type_bound::<PyDeprecationWarning>(),
            // Stack level.  Compared to Python-space calls to `warn`, this is unusually low
            // beacuse all our internal call structure is now Rust-space and invisible to Python.
            1,
        ))
        .map(|_| ())
}

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

use approx::relative_eq;
use std::f64::consts::PI;

use crate::circuit_data::CircuitData;
use crate::circuit_instruction::ExtraInstructionAttributes;
use crate::imports::get_std_gate_class;
use crate::imports::{PARAMETER_EXPRESSION, QUANTUM_CIRCUIT};
use crate::{gate_matrix, Qubit};

use ndarray::{aview2, Array2};
use num_complex::Complex64;
use smallvec::smallvec;

use numpy::IntoPyArray;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyFloat, PyIterator, PyTuple};
use pyo3::{intern, IntoPy, Python};

#[derive(Clone, Debug)]
pub enum Param {
    ParameterExpression(PyObject),
    Float(f64),
    Obj(PyObject),
}

impl Param {
    pub fn eq(&self, py: Python, other: &Param) -> PyResult<bool> {
        match [self, other] {
            [Self::Float(a), Self::Float(b)] => Ok(a == b),
            [Self::Float(a), Self::ParameterExpression(b)] => b.bind(py).eq(a),
            [Self::ParameterExpression(a), Self::Float(b)] => a.bind(py).eq(b),
            [Self::ParameterExpression(a), Self::ParameterExpression(b)] => a.bind(py).eq(b),
            [Self::Obj(_), Self::Float(_)] => Ok(false),
            [Self::Float(_), Self::Obj(_)] => Ok(false),
            [Self::Obj(a), Self::ParameterExpression(b)] => a.bind(py).eq(b),
            [Self::Obj(a), Self::Obj(b)] => a.bind(py).eq(b),
            [Self::ParameterExpression(a), Self::Obj(b)] => a.bind(py).eq(b),
        }
    }

    pub fn is_close(&self, py: Python, other: &Param, max_relative: f64) -> PyResult<bool> {
        match [self, other] {
            [Self::Float(a), Self::Float(b)] => Ok(relative_eq!(a, b, max_relative = max_relative)),
            _ => self.eq(py, other),
        }
    }
}

impl<'py> FromPyObject<'py> for Param {
    fn extract_bound(b: &Bound<'py, PyAny>) -> Result<Self, PyErr> {
        Ok(if b.is_instance(PARAMETER_EXPRESSION.get_bound(b.py()))? {
            Param::ParameterExpression(b.clone().unbind())
        } else if let Ok(val) = b.extract::<f64>() {
            Param::Float(val)
        } else {
            Param::Obj(b.clone().unbind())
        })
    }
}

impl IntoPy<PyObject> for Param {
    fn into_py(self, py: Python) -> PyObject {
        match &self {
            Self::Float(val) => val.to_object(py),
            Self::ParameterExpression(val) => val.clone_ref(py),
            Self::Obj(val) => val.clone_ref(py),
        }
    }
}

impl ToPyObject for Param {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            Self::Float(val) => val.to_object(py),
            Self::ParameterExpression(val) => val.clone_ref(py),
            Self::Obj(val) => val.clone_ref(py),
        }
    }
}

impl Param {
    /// Get an iterator over any Python-space `Parameter` instances tracked within this `Param`.
    pub fn iter_parameters<'py>(&self, py: Python<'py>) -> PyResult<ParamParameterIter<'py>> {
        let parameters_attr = intern!(py, "parameters");
        match self {
            Param::Float(_) => Ok(ParamParameterIter(None)),
            Param::ParameterExpression(expr) => Ok(ParamParameterIter(Some(
                expr.bind(py).getattr(parameters_attr)?.iter()?,
            ))),
            Param::Obj(obj) => {
                let obj = obj.bind(py);
                if obj.is_instance(QUANTUM_CIRCUIT.get_bound(py))? {
                    Ok(ParamParameterIter(Some(
                        obj.getattr(parameters_attr)?.iter()?,
                    )))
                } else {
                    Ok(ParamParameterIter(None))
                }
            }
        }
    }

    /// Extract from a Python object without numeric coercion to float.  The default conversion will
    /// coerce integers into floats, but in things like `assign_parameters`, this is not always
    /// desirable.
    pub fn extract_no_coerce(ob: &Bound<PyAny>) -> PyResult<Self> {
        Ok(if ob.is_instance_of::<PyFloat>() {
            Param::Float(ob.extract()?)
        } else if ob.is_instance(PARAMETER_EXPRESSION.get_bound(ob.py()))? {
            Param::ParameterExpression(ob.clone().unbind())
        } else {
            Param::Obj(ob.clone().unbind())
        })
    }

    /// Clones the [Param] object safely by reference count or copying.
    pub fn clone_ref(&self, py: Python) -> Self {
        match self {
            Param::ParameterExpression(exp) => Param::ParameterExpression(exp.clone_ref(py)),
            Param::Float(float) => Param::Float(*float),
            Param::Obj(obj) => Param::Obj(obj.clone_ref(py)),
        }
    }
}

// This impl allows for shared usage between [Param] and &[Param].
// Such blanked impl doesn't exist inherently due to Rust's type system limitations.
// See https://doc.rust-lang.org/std/convert/trait.AsRef.html#reflexivity for more information.
impl AsRef<Param> for Param {
    fn as_ref(&self) -> &Param {
        self
    }
}

/// Struct to provide iteration over Python-space `Parameter` instances within a `Param`.
pub struct ParamParameterIter<'py>(Option<Bound<'py, PyIterator>>);
impl<'py> Iterator for ParamParameterIter<'py> {
    type Item = PyResult<Bound<'py, PyAny>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.as_mut().and_then(|iter| iter.next())
    }
}

/// Trait for generic circuit operations these define the common attributes
/// needed for something to be addable to the circuit struct
pub trait Operation {
    fn name(&self) -> &str;
    fn num_qubits(&self) -> u32;
    fn num_clbits(&self) -> u32;
    fn num_params(&self) -> u32;
    fn control_flow(&self) -> bool;
    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>>;
    fn definition(&self, params: &[Param]) -> Option<CircuitData>;
    fn standard_gate(&self) -> Option<StandardGate>;
    fn directive(&self) -> bool;
}

/// Unpacked view object onto a `PackedOperation`.  This is the return value of
/// `PackedInstruction::op`, and in turn is a view object onto a `PackedOperation`.
///
/// This is the main way that we interact immutably with general circuit operations from Rust space.
#[derive(Debug)]
pub enum OperationRef<'a> {
    Standard(StandardGate),
    Gate(&'a PyGate),
    Instruction(&'a PyInstruction),
    Operation(&'a PyOperation),
}

impl<'a> Operation for OperationRef<'a> {
    #[inline]
    fn name(&self) -> &str {
        match self {
            Self::Standard(standard) => standard.name(),
            Self::Gate(gate) => gate.name(),
            Self::Instruction(instruction) => instruction.name(),
            Self::Operation(operation) => operation.name(),
        }
    }
    #[inline]
    fn num_qubits(&self) -> u32 {
        match self {
            Self::Standard(standard) => standard.num_qubits(),
            Self::Gate(gate) => gate.num_qubits(),
            Self::Instruction(instruction) => instruction.num_qubits(),
            Self::Operation(operation) => operation.num_qubits(),
        }
    }
    #[inline]
    fn num_clbits(&self) -> u32 {
        match self {
            Self::Standard(standard) => standard.num_clbits(),
            Self::Gate(gate) => gate.num_clbits(),
            Self::Instruction(instruction) => instruction.num_clbits(),
            Self::Operation(operation) => operation.num_clbits(),
        }
    }
    #[inline]
    fn num_params(&self) -> u32 {
        match self {
            Self::Standard(standard) => standard.num_params(),
            Self::Gate(gate) => gate.num_params(),
            Self::Instruction(instruction) => instruction.num_params(),
            Self::Operation(operation) => operation.num_params(),
        }
    }
    #[inline]
    fn control_flow(&self) -> bool {
        match self {
            Self::Standard(standard) => standard.control_flow(),
            Self::Gate(gate) => gate.control_flow(),
            Self::Instruction(instruction) => instruction.control_flow(),
            Self::Operation(operation) => operation.control_flow(),
        }
    }
    #[inline]
    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>> {
        match self {
            Self::Standard(standard) => standard.matrix(params),
            Self::Gate(gate) => gate.matrix(params),
            Self::Instruction(instruction) => instruction.matrix(params),
            Self::Operation(operation) => operation.matrix(params),
        }
    }
    #[inline]
    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        match self {
            Self::Standard(standard) => standard.definition(params),
            Self::Gate(gate) => gate.definition(params),
            Self::Instruction(instruction) => instruction.definition(params),
            Self::Operation(operation) => operation.definition(params),
        }
    }
    #[inline]
    fn standard_gate(&self) -> Option<StandardGate> {
        match self {
            Self::Standard(standard) => standard.standard_gate(),
            Self::Gate(gate) => gate.standard_gate(),
            Self::Instruction(instruction) => instruction.standard_gate(),
            Self::Operation(operation) => operation.standard_gate(),
        }
    }
    #[inline]
    fn directive(&self) -> bool {
        match self {
            Self::Standard(standard) => standard.directive(),
            Self::Gate(gate) => gate.directive(),
            Self::Instruction(instruction) => instruction.directive(),
            Self::Operation(operation) => operation.directive(),
        }
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Hash)]
#[repr(u8)]
#[pyclass(module = "qiskit._accelerate.circuit")]
pub enum StandardGate {
    GlobalPhaseGate = 0,
    HGate = 1,
    IGate = 2,
    XGate = 3,
    YGate = 4,
    ZGate = 5,
    PhaseGate = 6,
    RGate = 7,
    RXGate = 8,
    RYGate = 9,
    RZGate = 10,
    SGate = 11,
    SdgGate = 12,
    SXGate = 13,
    SXdgGate = 14,
    TGate = 15,
    TdgGate = 16,
    UGate = 17,
    U1Gate = 18,
    U2Gate = 19,
    U3Gate = 20,
    CHGate = 21,
    CXGate = 22,
    CYGate = 23,
    CZGate = 24,
    DCXGate = 25,
    ECRGate = 26,
    SwapGate = 27,
    ISwapGate = 28,
    CPhaseGate = 29,
    CRXGate = 30,
    CRYGate = 31,
    CRZGate = 32,
    CSGate = 33,
    CSdgGate = 34,
    CSXGate = 35,
    CUGate = 36,
    CU1Gate = 37,
    CU3Gate = 38,
    RXXGate = 39,
    RYYGate = 40,
    RZZGate = 41,
    RZXGate = 42,
    XXMinusYYGate = 43,
    XXPlusYYGate = 44,
    CCXGate = 45,
    CCZGate = 46,
    CSwapGate = 47,
    RCCXGate = 48,
    C3XGate = 49,
    C3SXGate = 50,
    RC3XGate = 51,
}

unsafe impl ::bytemuck::CheckedBitPattern for StandardGate {
    type Bits = u8;

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits < 53
    }
}
unsafe impl ::bytemuck::NoUninit for StandardGate {}

impl ToPyObject for StandardGate {
    fn to_object(&self, py: Python) -> Py<PyAny> {
        (*self).into_py(py)
    }
}

static STANDARD_GATE_NUM_QUBITS: [u32; STANDARD_GATE_SIZE] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 0-9
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 10-19
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, // 20-29
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // 30-39
    2, 2, 2, 2, 2, 3, 3, 3, 3, 4, // 40-49
    4, 4, // 50-51
];

static STANDARD_GATE_NUM_PARAMS: [u32; STANDARD_GATE_SIZE] = [
    1, 0, 0, 0, 0, 0, 1, 2, 1, 1, // 0-9
    1, 0, 0, 0, 0, 0, 0, 3, 1, 2, // 10-19
    3, 0, 0, 0, 0, 0, 0, 0, 0, 1, // 20-29
    1, 1, 1, 0, 0, 0, 4, 1, 3, 1, // 30-39
    1, 1, 1, 2, 2, 0, 0, 0, 0, 0, // 40-49
    0, 0, // 50-51
];

static STANDARD_GATE_NUM_CTRL_QUBITS: [u32; STANDARD_GATE_SIZE] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0-9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 10-19
    0, 1, 1, 1, 1, 0, 0, 0, 0, 1, // 20-29
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, // 30-39
    0, 0, 0, 0, 0, 2, 2, 1, 0, 3, // 40-49
    3, 0, // 50-51
];

static STANDARD_GATE_NAME: [&str; STANDARD_GATE_SIZE] = [
    "global_phase", // 0
    "h",            // 1
    "id",           // 2
    "x",            // 3
    "y",            // 4
    "z",            // 5
    "p",            // 6
    "r",            // 7
    "rx",           // 8
    "ry",           // 9
    "rz",           // 10
    "s",            // 11
    "sdg",          // 12
    "sx",           // 13
    "sxdg",         // 14
    "t",            // 15
    "tdg",          // 16
    "u",            // 17
    "u1",           // 18
    "u2",           // 19
    "u3",           // 20
    "ch",           // 21
    "cx",           // 22
    "cy",           // 23
    "cz",           // 24
    "dcx",          // 25
    "ecr",          // 26
    "swap",         // 27
    "iswap",        // 28
    "cp",           // 29
    "crx",          // 30
    "cry",          // 31
    "crz",          // 32
    "cs",           // 33
    "csdg",         // 34
    "csx",          // 35
    "cu",           // 36
    "cu1",          // 37
    "cu3",          // 38
    "rxx",          // 39
    "ryy",          // 40
    "rzz",          // 41
    "rzx",          // 42
    "xx_minus_yy",  // 43
    "xx_plus_yy",   // 44
    "ccx",          // 45
    "ccz",          // 46
    "cswap",        // 47
    "rccx",         // 48
    "mcx",          // 49 ("c3x")
    "c3sx",         // 50
    "rcccx",        // 51 ("rc3x")
];

impl StandardGate {
    pub fn create_py_op(
        &self,
        py: Python,
        params: Option<&[Param]>,
        extra_attrs: Option<&ExtraInstructionAttributes>,
    ) -> PyResult<Py<PyAny>> {
        let gate_class = get_std_gate_class(py, *self)?;
        let args = match params.unwrap_or(&[]) {
            &[] => PyTuple::empty_bound(py),
            params => PyTuple::new_bound(py, params),
        };
        if let Some(extra) = extra_attrs {
            let kwargs = [
                ("label", extra.label.to_object(py)),
                ("unit", extra.py_unit(py).into_any()),
                ("duration", extra.duration.to_object(py)),
            ]
            .into_py_dict_bound(py);
            let mut out = gate_class.call_bound(py, args, Some(&kwargs))?;
            if let Some(ref condition) = extra.condition {
                out = out.call_method0(py, "to_mutable")?;
                out.setattr(py, "condition", condition)?;
            }
            Ok(out)
        } else {
            gate_class.call_bound(py, args, None)
        }
    }

    pub fn num_ctrl_qubits(&self) -> u32 {
        STANDARD_GATE_NUM_CTRL_QUBITS[*self as usize]
    }
}

#[pymethods]
impl StandardGate {
    pub fn copy(&self) -> Self {
        *self
    }

    // These pymethods are for testing:
    pub fn _to_matrix(&self, py: Python, params: Vec<Param>) -> Option<PyObject> {
        self.matrix(&params)
            .map(|x| x.into_pyarray_bound(py).into())
    }

    pub fn _num_params(&self) -> u32 {
        self.num_params()
    }

    pub fn _get_definition(&self, params: Vec<Param>) -> Option<CircuitData> {
        self.definition(&params)
    }

    #[getter]
    pub fn get_num_qubits(&self) -> u32 {
        self.num_qubits()
    }

    #[getter]
    pub fn get_num_clbits(&self) -> u32 {
        self.num_clbits()
    }

    #[getter]
    pub fn get_num_params(&self) -> u32 {
        self.num_params()
    }

    #[getter]
    pub fn get_name(&self) -> &str {
        self.name()
    }

    pub fn __eq__(&self, other: &Bound<PyAny>) -> Py<PyAny> {
        let py = other.py();
        let Ok(other) = other.extract::<Self>() else {
            return py.NotImplemented();
        };
        (*self == other).into_py(py)
    }

    pub fn __hash__(&self) -> isize {
        *self as isize
    }
}

// This must be kept up-to-date with `StandardGate` when adding or removing
// gates from the enum
//
// Remove this when std::mem::variant_count() is stabilized (see
// https://github.com/rust-lang/rust/issues/73662 )
pub const STANDARD_GATE_SIZE: usize = 52;

impl Operation for StandardGate {
    fn name(&self) -> &str {
        STANDARD_GATE_NAME[*self as usize]
    }

    fn num_qubits(&self) -> u32 {
        STANDARD_GATE_NUM_QUBITS[*self as usize]
    }

    fn num_params(&self) -> u32 {
        STANDARD_GATE_NUM_PARAMS[*self as usize]
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn control_flow(&self) -> bool {
        false
    }

    fn directive(&self) -> bool {
        false
    }

    fn matrix(&self, params: &[Param]) -> Option<Array2<Complex64>> {
        match self {
            Self::GlobalPhaseGate => match params {
                [Param::Float(theta)] => {
                    Some(aview2(&gate_matrix::global_phase_gate(*theta)).to_owned())
                }
                _ => None,
            },
            Self::HGate => match params {
                [] => Some(aview2(&gate_matrix::H_GATE).to_owned()),
                _ => None,
            },
            Self::IGate => match params {
                [] => Some(aview2(&gate_matrix::ONE_QUBIT_IDENTITY).to_owned()),
                _ => None,
            },
            Self::XGate => match params {
                [] => Some(aview2(&gate_matrix::X_GATE).to_owned()),
                _ => None,
            },
            Self::YGate => match params {
                [] => Some(aview2(&gate_matrix::Y_GATE).to_owned()),
                _ => None,
            },
            Self::ZGate => match params {
                [] => Some(aview2(&gate_matrix::Z_GATE).to_owned()),
                _ => None,
            },
            Self::PhaseGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::phase_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::RGate => match params {
                [Param::Float(theta), Param::Float(phi)] => {
                    Some(aview2(&gate_matrix::r_gate(*theta, *phi)).to_owned())
                }
                _ => None,
            },
            Self::RXGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::rx_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::RYGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::ry_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::RZGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::rz_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::SGate => match params {
                [] => Some(aview2(&gate_matrix::S_GATE).to_owned()),
                _ => None,
            },
            Self::SdgGate => match params {
                [] => Some(aview2(&gate_matrix::SDG_GATE).to_owned()),
                _ => None,
            },
            Self::SXGate => match params {
                [] => Some(aview2(&gate_matrix::SX_GATE).to_owned()),
                _ => None,
            },
            Self::SXdgGate => match params {
                [] => Some(aview2(&gate_matrix::SXDG_GATE).to_owned()),
                _ => None,
            },
            Self::TGate => match params {
                [] => Some(aview2(&gate_matrix::T_GATE).to_owned()),
                _ => None,
            },
            Self::TdgGate => match params {
                [] => Some(aview2(&gate_matrix::TDG_GATE).to_owned()),
                _ => None,
            },
            Self::UGate => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::U1Gate => match params[0] {
                Param::Float(val) => Some(aview2(&gate_matrix::u1_gate(val)).to_owned()),
                _ => None,
            },
            Self::U2Gate => match params {
                [Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u2_gate(*phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::U3Gate => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::u3_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::CHGate => match params {
                [] => Some(aview2(&gate_matrix::CH_GATE).to_owned()),
                _ => None,
            },
            Self::CXGate => match params {
                [] => Some(aview2(&gate_matrix::CX_GATE).to_owned()),
                _ => None,
            },
            Self::CYGate => match params {
                [] => Some(aview2(&gate_matrix::CY_GATE).to_owned()),
                _ => None,
            },
            Self::CZGate => match params {
                [] => Some(aview2(&gate_matrix::CZ_GATE).to_owned()),
                _ => None,
            },
            Self::DCXGate => match params {
                [] => Some(aview2(&gate_matrix::DCX_GATE).to_owned()),
                _ => None,
            },
            Self::ECRGate => match params {
                [] => Some(aview2(&gate_matrix::ECR_GATE).to_owned()),
                _ => None,
            },
            Self::SwapGate => match params {
                [] => Some(aview2(&gate_matrix::SWAP_GATE).to_owned()),
                _ => None,
            },
            Self::ISwapGate => match params {
                [] => Some(aview2(&gate_matrix::ISWAP_GATE).to_owned()),
                _ => None,
            },
            Self::CPhaseGate => match params {
                [Param::Float(lam)] => Some(aview2(&gate_matrix::cp_gate(*lam)).to_owned()),
                _ => None,
            },
            Self::CRXGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::crx_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CRYGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::cry_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CRZGate => match params {
                [Param::Float(theta)] => Some(aview2(&gate_matrix::crz_gate(*theta)).to_owned()),
                _ => None,
            },
            Self::CSGate => match params {
                [] => Some(aview2(&gate_matrix::CS_GATE).to_owned()),
                _ => None,
            },
            Self::CSdgGate => match params {
                [] => Some(aview2(&gate_matrix::CSDG_GATE).to_owned()),
                _ => None,
            },
            Self::CSXGate => match params {
                [] => Some(aview2(&gate_matrix::CSX_GATE).to_owned()),
                _ => None,
            },
            Self::CUGate => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam), Param::Float(gamma)] => {
                    Some(aview2(&gate_matrix::cu_gate(*theta, *phi, *lam, *gamma)).to_owned())
                }
                _ => None,
            },
            Self::CU1Gate => match params[0] {
                Param::Float(lam) => Some(aview2(&gate_matrix::cu1_gate(lam)).to_owned()),
                _ => None,
            },
            Self::CU3Gate => match params {
                [Param::Float(theta), Param::Float(phi), Param::Float(lam)] => {
                    Some(aview2(&gate_matrix::cu3_gate(*theta, *phi, *lam)).to_owned())
                }
                _ => None,
            },
            Self::RXXGate => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::rxx_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RYYGate => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::ryy_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RZZGate => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::rzz_gate(theta)).to_owned()),
                _ => None,
            },
            Self::RZXGate => match params[0] {
                Param::Float(theta) => Some(aview2(&gate_matrix::rzx_gate(theta)).to_owned()),
                _ => None,
            },
            Self::XXMinusYYGate => match params {
                [Param::Float(theta), Param::Float(beta)] => {
                    Some(aview2(&gate_matrix::xx_minus_yy_gate(*theta, *beta)).to_owned())
                }
                _ => None,
            },
            Self::XXPlusYYGate => match params {
                [Param::Float(theta), Param::Float(beta)] => {
                    Some(aview2(&gate_matrix::xx_plus_yy_gate(*theta, *beta)).to_owned())
                }
                _ => None,
            },
            Self::CCXGate => match params {
                [] => Some(aview2(&gate_matrix::CCX_GATE).to_owned()),
                _ => None,
            },
            Self::CCZGate => match params {
                [] => Some(aview2(&gate_matrix::CCZ_GATE).to_owned()),
                _ => None,
            },
            Self::CSwapGate => match params {
                [] => Some(aview2(&gate_matrix::CSWAP_GATE).to_owned()),
                _ => None,
            },
            Self::RCCXGate => match params {
                [] => Some(aview2(&gate_matrix::RCCX_GATE).to_owned()),
                _ => None,
            },
            Self::C3XGate => match params {
                [] => Some(aview2(&gate_matrix::C3X_GATE).to_owned()),
                _ => None,
            },
            Self::C3SXGate => match params {
                [] => Some(aview2(&gate_matrix::C3SX_GATE).to_owned()),
                _ => None,
            },
            Self::RC3XGate => match params {
                [] => Some(aview2(&gate_matrix::RC3X_GATE).to_owned()),
                _ => None,
            },
        }
    }

    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        match self {
            Self::GlobalPhaseGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(py, 0, [], params[0].clone())
                        .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::HGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![Param::Float(PI / 2.), FLOAT_ZERO, Param::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::IGate => None,
            Self::XGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![Param::Float(PI), FLOAT_ZERO, Param::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::YGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![
                                Param::Float(PI),
                                Param::Float(PI / 2.),
                                Param::Float(PI / 2.),
                            ],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),

            Self::ZGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(PI)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::PhaseGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![FLOAT_ZERO, FLOAT_ZERO, params[0].clone()],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta_expr = clone_param(&params[0], py);
                let phi_expr1 = add_param(&params[1], -PI / 2., py);
                let phi_expr2 = multiply_param(&phi_expr1, -1.0, py);
                let defparams = smallvec![theta_expr, phi_expr1, phi_expr2];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(Self::UGate, defparams, smallvec![Qubit(0)])],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::RGate,
                            smallvec![theta.clone(), FLOAT_ZERO],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::RGate,
                            smallvec![theta.clone(), Param::Float(PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RZGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![theta.clone()],
                            smallvec![Qubit(0)],
                        )],
                        multiply_param(theta, -0.5, py),
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SdgGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(-PI / 2.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SXGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [
                            (Self::SdgGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::SdgGate, smallvec![], smallvec![Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SXdgGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [
                            (Self::SGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::SGate, smallvec![], smallvec![Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::TGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(PI / 4.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::TdgGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            smallvec![Param::Float(-PI / 4.)],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::UGate => None,
            Self::U1Gate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::PhaseGate,
                            params.iter().cloned().collect(),
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::U2Gate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            smallvec![Param::Float(PI / 2.), params[0].clone(), params[1].clone()],
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::U3Gate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        1,
                        [(
                            Self::UGate,
                            params.iter().cloned().collect(),
                            smallvec![Qubit(0)],
                        )],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CHGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::SGate, smallvec![], q1.clone()),
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::TGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::TdgGate, smallvec![], q1.clone()),
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::SdgGate, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),

            Self::CXGate => None,
            Self::CYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::SdgGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::SGate, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CZGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::HGate, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::DCXGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::ECRGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RZXGate,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(0), Qubit(1)],
                            ),
                            (Self::XGate, smallvec![], smallvec![Qubit(0)]),
                            (
                                Self::RZXGate,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(0), Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::SwapGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::ISwapGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::SGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::SGate, smallvec![], smallvec![Qubit(1)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(0)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(0)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(1)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CPhaseGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::PhaseGate,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                q0,
                            ),
                            (Self::CXGate, smallvec![], q0_1.clone()),
                            (
                                Self::PhaseGate,
                                smallvec![multiply_param(&params[0], -0.5, py)],
                                q1.clone(),
                            ),
                            (Self::CXGate, smallvec![], q0_1),
                            (
                                Self::PhaseGate,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                q1,
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CRXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 2.)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::UGate,
                                smallvec![
                                    multiply_param(theta, -0.5, py),
                                    Param::Float(0.0),
                                    Param::Float(0.0)
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::UGate,
                                smallvec![
                                    multiply_param(theta, 0.5, py),
                                    Param::Float(-PI / 2.),
                                    Param::Float(0.0)
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            Self::CRYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            Self::CRZGate => Python::with_gil(|py| -> Option<CircuitData> {
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RZGate,
                                smallvec![multiply_param(theta, 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::RZGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                        ],
                        Param::Float(0.0),
                    )
                    .expect("Unexpected Qiskit Python bug!"),
                )
            }),
            Self::CSGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::PhaseGate, smallvec![Param::Float(PI / 4.)], q0),
                            (Self::CXGate, smallvec![], q0_1.clone()),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 4.)],
                                q1.clone(),
                            ),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::PhaseGate, smallvec![Param::Float(PI / 4.)], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CSdgGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::PhaseGate, smallvec![Param::Float(-PI / 4.)], q0),
                            (Self::CXGate, smallvec![], q0_1.clone()),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 4.)],
                                q1.clone(),
                            ),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::PhaseGate, smallvec![Param::Float(-PI / 4.)], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CSXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::CPhaseGate, smallvec![Param::Float(PI / 2.)], q0_1),
                            (Self::HGate, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CUGate => Python::with_gil(|py| -> Option<CircuitData> {
                let param_second_p = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], 0.5, py),
                    py,
                );
                let param_third_p = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], -0.5, py),
                    py,
                );
                let param_first_u = radd_param(
                    multiply_param(&params[1], -0.5, py),
                    multiply_param(&params[2], -0.5, py),
                    py,
                );
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::PhaseGate,
                                smallvec![params[3].clone()],
                                smallvec![Qubit(0)],
                            ),
                            (
                                Self::PhaseGate,
                                smallvec![param_second_p],
                                smallvec![Qubit(0)],
                            ),
                            (
                                Self::PhaseGate,
                                smallvec![param_third_p],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::UGate,
                                smallvec![
                                    multiply_param(&params[0], -0.5, py),
                                    FLOAT_ZERO,
                                    param_first_u
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::UGate,
                                smallvec![
                                    multiply_param(&params[0], 0.5, py),
                                    params[1].clone(),
                                    FLOAT_ZERO
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CU1Gate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::U1Gate,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                smallvec![Qubit(0)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U1Gate,
                                smallvec![multiply_param(&params[0], -0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U1Gate,
                                smallvec![multiply_param(&params[0], 0.5, py)],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CU3Gate => Python::with_gil(|py| -> Option<CircuitData> {
                let param_first_u1 = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], 0.5, py),
                    py,
                );
                let param_second_u1 = radd_param(
                    multiply_param(&params[2], 0.5, py),
                    multiply_param(&params[1], -0.5, py),
                    py,
                );
                let param_first_u3 = radd_param(
                    multiply_param(&params[1], -0.5, py),
                    multiply_param(&params[2], -0.5, py),
                    py,
                );
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::U1Gate, smallvec![param_first_u1], smallvec![Qubit(0)]),
                            (
                                Self::U1Gate,
                                smallvec![param_second_u1],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U3Gate,
                                smallvec![
                                    multiply_param(&params[0], -0.5, py),
                                    FLOAT_ZERO,
                                    param_first_u3
                                ],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::U3Gate,
                                smallvec![
                                    multiply_param(&params[0], 0.5, py),
                                    params[1].clone(),
                                    FLOAT_ZERO
                                ],
                                smallvec![Qubit(1)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RXXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::HGate, smallvec![], q0.clone()),
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_q1.clone()),
                            (Self::RZGate, smallvec![theta.clone()], q1.clone()),
                            (Self::CXGate, smallvec![], q0_q1),
                            (Self::HGate, smallvec![], q1),
                            (Self::HGate, smallvec![], q0),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RYYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::RXGate, smallvec![Param::Float(PI / 2.)], q0.clone()),
                            (Self::RXGate, smallvec![Param::Float(PI / 2.)], q1.clone()),
                            (Self::CXGate, smallvec![], q0_q1.clone()),
                            (Self::RZGate, smallvec![theta.clone()], q1.clone()),
                            (Self::CXGate, smallvec![], q0_q1),
                            (Self::RXGate, smallvec![Param::Float(-PI / 2.)], q0),
                            (Self::RXGate, smallvec![Param::Float(-PI / 2.)], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RZZGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::CXGate, smallvec![], q0_q1.clone()),
                            (Self::RZGate, smallvec![theta.clone()], q1),
                            (Self::CXGate, smallvec![], q0_q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RZXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q1 = smallvec![Qubit(1)];
                let q0_q1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::HGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_q1.clone()),
                            (Self::RZGate, smallvec![theta.clone()], q1.clone()),
                            (Self::CXGate, smallvec![], q0_q1),
                            (Self::HGate, smallvec![], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::XXMinusYYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (
                                Self::RZGate,
                                smallvec![multiply_param(beta, -1.0, py)],
                                q1.clone(),
                            ),
                            (Self::RZGate, smallvec![Param::Float(-PI / 2.)], q0.clone()),
                            (Self::SXGate, smallvec![], q0.clone()),
                            (Self::RZGate, smallvec![Param::Float(PI / 2.)], q0.clone()),
                            (Self::SGate, smallvec![], q1.clone()),
                            (Self::CXGate, smallvec![], q0_1.clone()),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, 0.5, py)],
                                q0.clone(),
                            ),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q1.clone(),
                            ),
                            (Self::CXGate, smallvec![], q0_1),
                            (Self::SdgGate, smallvec![], q1.clone()),
                            (Self::RZGate, smallvec![Param::Float(-PI / 2.)], q0.clone()),
                            (Self::SXdgGate, smallvec![], q0.clone()),
                            (Self::RZGate, smallvec![Param::Float(PI / 2.)], q0),
                            (Self::RZGate, smallvec![beta.clone()], q1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::XXPlusYYGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q1_0 = smallvec![Qubit(1), Qubit(0)];
                let theta = &params[0];
                let beta = &params[1];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        2,
                        [
                            (Self::RZGate, smallvec![beta.clone()], q0.clone()),
                            (Self::RZGate, smallvec![Param::Float(-PI / 2.)], q1.clone()),
                            (Self::SXGate, smallvec![], q1.clone()),
                            (Self::RZGate, smallvec![Param::Float(PI / 2.)], q1.clone()),
                            (Self::SGate, smallvec![], q0.clone()),
                            (Self::CXGate, smallvec![], q1_0.clone()),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q1.clone(),
                            ),
                            (
                                Self::RYGate,
                                smallvec![multiply_param(theta, -0.5, py)],
                                q0.clone(),
                            ),
                            (Self::CXGate, smallvec![], q1_0),
                            (Self::SdgGate, smallvec![], q0.clone()),
                            (Self::RZGate, smallvec![Param::Float(-PI / 2.)], q1.clone()),
                            (Self::SXdgGate, smallvec![], q1.clone()),
                            (Self::RZGate, smallvec![Param::Float(PI / 2.)], q1),
                            (Self::RZGate, smallvec![multiply_param(beta, -1.0, py)], q0),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CCXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q0 = smallvec![Qubit(0)];
                let q1 = smallvec![Qubit(1)];
                let q2 = smallvec![Qubit(2)];
                let q0_1 = smallvec![Qubit(0), Qubit(1)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (Self::HGate, smallvec![], q2.clone()),
                            (Self::CXGate, smallvec![], q1_2.clone()),
                            (Self::TdgGate, smallvec![], q2.clone()),
                            (Self::CXGate, smallvec![], q0_2.clone()),
                            (Self::TGate, smallvec![], q2.clone()),
                            (Self::CXGate, smallvec![], q1_2),
                            (Self::TdgGate, smallvec![], q2.clone()),
                            (Self::CXGate, smallvec![], q0_2),
                            (Self::TGate, smallvec![], q1.clone()),
                            (Self::TGate, smallvec![], q2.clone()),
                            (Self::HGate, smallvec![], q2),
                            (Self::CXGate, smallvec![], q0_1.clone()),
                            (Self::TGate, smallvec![], q0),
                            (Self::TdgGate, smallvec![], q1),
                            (Self::CXGate, smallvec![], q0_1),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),

            Self::CCZGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (Self::HGate, smallvec![], smallvec![Qubit(2)]),
                            (
                                Self::CCXGate,
                                smallvec![],
                                smallvec![Qubit(0), Qubit(1), Qubit(2)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(2)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::CSwapGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(1)]),
                            (
                                Self::CCXGate,
                                smallvec![],
                                smallvec![Qubit(0), Qubit(1), Qubit(2)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(1)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),

            Self::RCCXGate => Python::with_gil(|py| -> Option<CircuitData> {
                let q2 = smallvec![Qubit(2)];
                let q0_2 = smallvec![Qubit(0), Qubit(2)];
                let q1_2 = smallvec![Qubit(1), Qubit(2)];
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        3,
                        [
                            (
                                Self::U2Gate,
                                smallvec![FLOAT_ZERO, Param::Float(PI)],
                                q2.clone(),
                            ),
                            (Self::U1Gate, smallvec![Param::Float(PI / 4.)], q2.clone()),
                            (Self::CXGate, smallvec![], q1_2.clone()),
                            (Self::U1Gate, smallvec![Param::Float(-PI / 4.)], q2.clone()),
                            (Self::CXGate, smallvec![], q0_2),
                            (Self::U1Gate, smallvec![Param::Float(PI / 4.)], q2.clone()),
                            (Self::CXGate, smallvec![], q1_2),
                            (Self::U1Gate, smallvec![Param::Float(-PI / 4.)], q2.clone()),
                            (Self::U2Gate, smallvec![FLOAT_ZERO, Param::Float(PI)], q2),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::C3XGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        4,
                        [
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(0)],
                            ),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(1)],
                            ),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(2)],
                            ),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(1)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(2)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(2)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(2)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::PhaseGate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),

            Self::C3SXGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        4,
                        [
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(0), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(1), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(1)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(1), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(2)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(-PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(2)]),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                            (
                                Self::CU1Gate,
                                smallvec![Param::Float(PI / 8.)],
                                smallvec![Qubit(2), Qubit(3)],
                            ),
                            (Self::HGate, smallvec![], smallvec![Qubit(3)]),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
            Self::RC3XGate => Python::with_gil(|py| -> Option<CircuitData> {
                Some(
                    CircuitData::from_standard_gates(
                        py,
                        4,
                        [
                            (
                                Self::U2Gate,
                                smallvec![FLOAT_ZERO, Param::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U2Gate,
                                smallvec![FLOAT_ZERO, Param::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(0), Qubit(3)]),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(1), Qubit(3)]),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U2Gate,
                                smallvec![FLOAT_ZERO, Param::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (Self::CXGate, smallvec![], smallvec![Qubit(2), Qubit(3)]),
                            (
                                Self::U1Gate,
                                smallvec![Param::Float(-PI / 4.)],
                                smallvec![Qubit(3)],
                            ),
                            (
                                Self::U2Gate,
                                smallvec![FLOAT_ZERO, Param::Float(PI)],
                                smallvec![Qubit(3)],
                            ),
                        ],
                        FLOAT_ZERO,
                    )
                    .expect("Unexpected Qiskit python bug"),
                )
            }),
        }
    }

    fn standard_gate(&self) -> Option<StandardGate> {
        Some(*self)
    }
}

const FLOAT_ZERO: Param = Param::Float(0.0);

// Return explicitly requested copy of `param`, handling
// each variant separately.
fn clone_param(param: &Param, py: Python) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta),
        Param::ParameterExpression(theta) => Param::ParameterExpression(theta.clone_ref(py)),
        Param::Obj(_) => unreachable!(),
    }
}

fn multiply_param(param: &Param, mult: f64, py: Python) -> Param {
    match param {
        Param::Float(theta) => Param::Float(theta * mult),
        Param::ParameterExpression(theta) => Param::ParameterExpression(
            theta
                .clone_ref(py)
                .call_method1(py, intern!(py, "__rmul__"), (mult,))
                .expect("Multiplication of Parameter expression by float failed."),
        ),
        Param::Obj(_) => unreachable!(),
    }
}

fn add_param(param: &Param, summand: f64, py: Python) -> Param {
    match param {
        Param::Float(theta) => Param::Float(*theta + summand),
        Param::ParameterExpression(theta) => Param::ParameterExpression(
            theta
                .clone_ref(py)
                .call_method1(py, intern!(py, "__add__"), (summand,))
                .expect("Sum of Parameter expression and float failed."),
        ),
        Param::Obj(_) => unreachable!(),
    }
}

fn radd_param(param1: Param, param2: Param, py: Python) -> Param {
    match [param1, param2] {
        [Param::Float(theta), Param::Float(lambda)] => Param::Float(theta + lambda),
        [Param::ParameterExpression(theta), Param::ParameterExpression(lambda)] => {
            Param::ParameterExpression(
                theta
                    .clone_ref(py)
                    .call_method1(py, intern!(py, "__radd__"), (lambda,))
                    .expect("Parameter expression addition failed"),
            )
        }
        _ => unreachable!(),
    }
}

/// This class is used to wrap a Python side Instruction that is not in the standard library
#[derive(Clone, Debug)]
// We bit-pack pointers to this, so having a known alignment even on 32-bit systems is good.
#[repr(align(8))]
pub struct PyInstruction {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub control_flow: bool,
    pub instruction: PyObject,
}

impl Operation for PyInstruction {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn control_flow(&self) -> bool {
        self.control_flow
    }
    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        None
    }
    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        Python::with_gil(|py| -> Option<CircuitData> {
            match self.instruction.getattr(py, intern!(py, "definition")) {
                Ok(definition) => {
                    let res: Option<PyObject> = definition.call0(py).ok()?.extract(py).ok();
                    match res {
                        Some(x) => {
                            let out: CircuitData =
                                x.getattr(py, intern!(py, "data")).ok()?.extract(py).ok()?;
                            Some(out)
                        }
                        None => None,
                    }
                }
                Err(_) => None,
            }
        })
    }
    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }

    fn directive(&self) -> bool {
        Python::with_gil(|py| -> bool {
            match self.instruction.getattr(py, intern!(py, "_directive")) {
                Ok(directive) => {
                    let res: bool = directive.extract(py).unwrap();
                    res
                }
                Err(_) => false,
            }
        })
    }
}

/// This class is used to wrap a Python side Gate that is not in the standard library
#[derive(Clone, Debug)]
// We bit-pack pointers to this, so having a known alignment even on 32-bit systems is good.
#[repr(align(8))]
pub struct PyGate {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub gate: PyObject,
}

impl Operation for PyGate {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn control_flow(&self) -> bool {
        false
    }
    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        Python::with_gil(|py| -> Option<Array2<Complex64>> {
            match self.gate.getattr(py, intern!(py, "to_matrix")) {
                Ok(to_matrix) => {
                    let res: Option<PyObject> = to_matrix.call0(py).ok()?.extract(py).ok();
                    match res {
                        Some(x) => {
                            let array: PyReadonlyArray2<Complex64> = x.extract(py).ok()?;
                            Some(array.as_array().to_owned())
                        }
                        None => None,
                    }
                }
                Err(_) => None,
            }
        })
    }
    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        Python::with_gil(|py| -> Option<CircuitData> {
            match self.gate.getattr(py, intern!(py, "definition")) {
                Ok(definition) => {
                    let res: Option<PyObject> = definition.call0(py).ok()?.extract(py).ok();
                    match res {
                        Some(x) => {
                            let out: CircuitData =
                                x.getattr(py, intern!(py, "data")).ok()?.extract(py).ok()?;
                            Some(out)
                        }
                        None => None,
                    }
                }
                Err(_) => None,
            }
        })
    }
    fn standard_gate(&self) -> Option<StandardGate> {
        Python::with_gil(|py| -> Option<StandardGate> {
            match self.gate.getattr(py, intern!(py, "_standard_gate")) {
                Ok(stdgate) => stdgate.extract(py).unwrap_or_default(),
                Err(_) => None,
            }
        })
    }
    fn directive(&self) -> bool {
        false
    }
}

/// This class is used to wrap a Python side Operation that is not in the standard library
#[derive(Clone, Debug)]
// We bit-pack pointers to this, so having a known alignment even on 32-bit systems is good.
#[repr(align(8))]
pub struct PyOperation {
    pub qubits: u32,
    pub clbits: u32,
    pub params: u32,
    pub op_name: String,
    pub operation: PyObject,
}

impl Operation for PyOperation {
    fn name(&self) -> &str {
        self.op_name.as_str()
    }
    fn num_qubits(&self) -> u32 {
        self.qubits
    }
    fn num_clbits(&self) -> u32 {
        self.clbits
    }
    fn num_params(&self) -> u32 {
        self.params
    }
    fn control_flow(&self) -> bool {
        false
    }
    fn matrix(&self, _params: &[Param]) -> Option<Array2<Complex64>> {
        None
    }
    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        None
    }
    fn standard_gate(&self) -> Option<StandardGate> {
        None
    }

    fn directive(&self) -> bool {
        Python::with_gil(|py| -> bool {
            match self.operation.getattr(py, intern!(py, "_directive")) {
                Ok(directive) => {
                    let res: bool = directive.extract(py).unwrap();
                    res
                }
                Err(_) => false,
            }
        })
    }
}

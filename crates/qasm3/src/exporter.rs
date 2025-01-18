use crate::ast::{
    Barrier, Break, ClassicalDeclaration, ClassicalType, Constant, Continue, Delay,
    DurationLiteral, DurationUnit, Expression, Float, GateCall, Header, IODeclaration, IOModifier,
    Identifier, Include, IntegerLiteral, Node, Parameter, Program, QuantumBlock,
    QuantumDeclaration, QuantumGateDefinition, QuantumGateModifier, QuantumGateModifierName,
    QuantumGateSignature, QuantumInstruction, QuantumMeasurement, QuantumMeasurementAssignment,
    Reset, Statement, Version,
};
use crate::printer::BasicPrinter;
use crate::symbols::SymbolError;
use crate::symbols::SymbolTable;
use hashbrown::{HashMap, HashSet};
use oq3_semantics::types::{IsConst, Type};
use pyo3::prelude::*;
use pyo3::Python;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Operation, Param};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{Clbit, Qubit};
use std::sync::Mutex;
use thiserror::Error;

use lazy_static::lazy_static;
use regex::Regex;

type ExporterResult<T> = Result<T, QASM3ExporterError>;

#[derive(Error, Debug)]
pub enum QASM3ExporterError {
    #[error("Error: {0}")]
    Error(String),
    #[error("Symbol '{0}' is not found in the table")]
    SymbolNotFound(String),
    #[error("PyError: {0}")]
    PyErr(PyErr),
    #[error("Not in global scope")]
    NotInGlobalScopeError,
}

impl From<PyErr> for QASM3ExporterError {
    fn from(err: PyErr) -> Self {
        QASM3ExporterError::PyErr(err)
    }
}

impl From<SymbolError> for QASM3ExporterError {
    fn from(err: SymbolError) -> Self {
        QASM3ExporterError::Error("Symbol Table error: Missing binding or Already exists".to_string())
    }
}

lazy_static! {
    static ref GLOBAL_COUNTER: Mutex<usize> = Mutex::new(0);
}

fn get_next_counter_value() -> usize {
    let mut counter = GLOBAL_COUNTER.lock().unwrap();
    let val = *counter;
    *counter += 1;
    val
}

lazy_static! {
    static ref RESERVED_KEYWORDS: HashSet<&'static str> = [
        "OPENQASM",
        "angle",
        "array",
        "barrier",
        "bit",
        "bool",
        "box",
        "break",
        "cal",
        "complex",
        "const",
        "continue",
        "creg",
        "ctrl",
        "def",
        "defcal",
        "defcalgrammar",
        "delay",
        "duration",
        "durationof",
        "else",
        "end",
        "extern",
        "float",
        "for",
        "gate",
        "gphase",
        "if",
        "in",
        "include",
        "input",
        "int",
        "inv",
        "let",
        "measure",
        "mutable",
        "negctrl",
        "output",
        "pow",
        "qreg",
        "qubit",
        "reset",
        "return",
        "sizeof",
        "stretch",
        "uint",
        "while"
    ]
    .into_iter()
    .collect();
}

lazy_static! {
    static ref VALID_IDENTIFIER: Regex = Regex::new(r"(^[\w][\w\d]*$|^\$\d+$)").unwrap();
}

fn name_allowed(symbol_table: &SymbolTable, name: &str, unique: bool) -> bool {
    if unique {
        RESERVED_KEYWORDS.contains(name) || symbol_table.all_scopes_contains_name(name)
    } else {
        RESERVED_KEYWORDS.contains(name)
    }
}

fn escape_invalid_identifier(
    symbol_table: &SymbolTable,
    name: &str,
    allow_rename: bool,
    unique: bool,
) -> String {
    let base = if allow_rename {
        format!(
            "_{}",
            name.chars()
                .map(|c| if c.is_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                })
                .collect::<String>()
        )
    } else {
        name.to_string()
    };

    let mut new_name = base.clone();
    while !name_allowed(symbol_table, &new_name, unique) {
        new_name = format!("{}_{}", base, get_next_counter_value());
    }
    new_name
}

pub struct Exporter {
    includes: Vec<&'static str>,
    basis_gates: Vec<&'static str>,
    disable_constants: bool,
    alias_classical_registers: bool,
    allow_aliasing: bool,
    indent: &'static str,
}

impl Exporter {
    pub fn new(
        includes: Vec<&'static str>,
        basis_gates: Vec<&'static str>,
        disable_constants: bool,
        alias_classical_registers: bool,
        allow_aliasing: bool,
        indent: &'static str,
    ) -> Self {
        Self {
            includes,
            basis_gates,
            disable_constants,
            alias_classical_registers,
            allow_aliasing,
            indent,
        }
    }

    pub fn dump(&self, circuit_data: &CircuitData, islayout: bool) -> String {
        let mut builder = QASM3Builder::new(
            circuit_data,
            islayout,
            self.includes.clone(),
            self.basis_gates.clone(),
            self.disable_constants,
            self.allow_aliasing,
        );
        match builder.build_program() {
            Ok(program) => {
                let mut output = String::new();
                BasicPrinter::new(&mut output, self.indent.to_string(), false)
                    .visit(&Node::Program(&program));
                output
            }
            Err(e) => format!("Error: {:?}", e),
        }
    }
}

#[derive(Debug, Clone)]
struct BuildScope<'a> {
    circuit_data: &'a CircuitData,
    qubit_map: HashMap<Qubit, Qubit>,
    clbit_map: HashMap<Clbit, Clbit>,
}

impl<'a> BuildScope<'a> {
    fn new(circuit_data: &'a CircuitData) -> Self {
        let qubit_map = Python::with_gil(|py| {
            let qubits = circuit_data.qubits();
            qubits
                .bits()
                .iter()
                .map(|bit| {
                    let bound_bit = bit.bind(py);
                    let found = qubits.find(bound_bit).expect("Qubit not found");
                    (found, found)
                })
                .collect()
        });

        let clbit_map = Python::with_gil(|py| {
            let clbits = circuit_data.clbits();
            clbits
                .bits()
                .iter()
                .map(|bit| {
                    let bound_bit = bit.bind(py);
                    let found = clbits.find(bound_bit).expect("Clbit not found");
                    (found, found)
                })
                .collect()
        });

        Self {
            circuit_data,
            qubit_map,
            clbit_map,
        }
    }
}

#[derive(Debug, Clone)]
struct Counter {
    current: usize,
}

impl Counter {
    fn new() -> Self {
        Self { current: 0 }
    }
}

impl Iterator for Counter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.current;
        self.current = self.current.wrapping_add(1);
        Some(n)
    }
}

pub struct QASM3Builder<'a> {
    builtin_instr: HashSet<&'static str>,
    loose_bit_prefix: &'static str,
    loose_qubit_prefix: &'static str,
    gate_param_prefix: &'static str,
    gate_qubit_prefix: &'static str,
    circuit_scope: BuildScope<'a>,
    is_layout: bool,
    symbol_table: SymbolTable,
    global_io_decls: Vec<IODeclaration>,
    global_classical_decls: Vec<ClassicalDeclaration>,
    includes: Vec<&'static str>,
    basis_gates: Vec<&'static str>,
    disable_constants: bool,
    allow_aliasing: bool,
    counter: Counter,
}

impl<'a> QASM3Builder<'a> {
    pub fn new(
        circuit_data: &'a CircuitData,
        is_layout: bool,
        includes: Vec<&'static str>,
        basis_gates: Vec<&'static str>,
        disable_constants: bool,
        allow_aliasing: bool,
    ) -> Self {
        Self {
            builtin_instr: [
                "barrier",
                "measure",
                "reset",
                "delay",
                "break_loop",
                "continue_loop",
                "store",
            ]
            .into_iter()
            .collect(),
            loose_bit_prefix: "_bit",
            loose_qubit_prefix: "_qubit",
            gate_param_prefix: "_gate_p",
            gate_qubit_prefix: "_gate_q",
            circuit_scope: BuildScope::new(circuit_data),
            is_layout,
            symbol_table: SymbolTable::new(),
            global_io_decls: Vec::new(),
            global_classical_decls: Vec::new(),
            includes,
            basis_gates,
            disable_constants,
            allow_aliasing,
            counter: Counter::new(),
        }
    }

    pub fn build_program(&mut self) -> ExporterResult<Program> {
        self.register_basis_gates();
        let header = self.build_header();

        self.hoist_global_params()?;
        self.hoist_classical_bits()?;
        let qubit_decls = self.build_qubit_decls()?;
        let main_stmts = self.build_top_level_stmts()?;

        let mut all_stmts = Vec::new();
        for decl in &self.global_io_decls {
            all_stmts.push(Statement::IODeclaration(decl.clone()));
        }
        for decl in &self.global_classical_decls {
            all_stmts.push(Statement::ClassicalDeclaration(decl.clone()));
        }
        for decl in qubit_decls {
            all_stmts.push(Statement::QuantumDeclaration(decl));
        }
        all_stmts.extend(main_stmts);

        Ok(Program {
            header,
            statements: all_stmts,
        })
    }

    fn register_basis_gates(&mut self) {
        for gate in &self.basis_gates {
            let _ = self.symbol_table.new_binding(gate, &Type::Gate(0, 1));
        }
    }

    fn build_header(&mut self) -> Header {
        let includes = self
            .includes
            .iter()
            .map(|fname| {
                if *fname == "stdgates.inc" {
                    self.symbol_table.standard_library_gates();
                }
                Include {
                    filename: fname.to_string(),
                }
            })
            .collect();
        Header {
            version: Some(Version {
                version_number: "3.0".to_string(),
            }),
            includes,
        }
    }

    fn assert_global_scope(&self) -> ExporterResult<()> {
        if !self.symbol_table.in_global_scope() {
            return Err(QASM3ExporterError::NotInGlobalScopeError);
        }
        Ok(())
    }

    fn hoist_global_params(&mut self) -> ExporterResult<()> {
        self.assert_global_scope()?;
        Python::with_gil(|py| {
            for param in self.circuit_scope.circuit_data.get_parameters(py) {
                let raw_name: String = param.getattr("name")?.extract()?;
                let sym = match self
                    .symbol_table
                    .new_binding(&raw_name, &Type::Float(Some(64), IsConst::False))
                {
                    Ok(_) => self.symbol_table.lookup(&raw_name),
                    Err(_) => {
                        let rename =
                            escape_invalid_identifier(&self.symbol_table, &raw_name, true, false);
                        self.symbol_table
                            .new_binding(&rename, &Type::Float(Some(64), IsConst::False))?;
                        self.symbol_table.lookup(&rename)
                    }
                }?;
                self.global_io_decls.push(IODeclaration {
                    modifier: IOModifier::Input,
                    type_: ClassicalType::Float(Float::Double),
                    identifier: Identifier {
                        string: sym.symbol().name().to_string(),
                    },
                });
            }
            Ok(())
        })
    }

    fn hoist_classical_bits(&mut self) -> ExporterResult<()> {
        self.assert_global_scope()?;
        let num_clbits = self.circuit_scope.circuit_data.num_clbits();
        let mut decls = Vec::new();

        for i in 0..num_clbits {
            let raw_name = format!("{}{}", self.loose_bit_prefix, i);
            let sym = match self
                .symbol_table
                .new_binding(&raw_name, &Type::Bit(IsConst::False))
            {
                Ok(_) => self.symbol_table.lookup(&raw_name),
                Err(_) => {
                    let rename =
                        escape_invalid_identifier(&self.symbol_table, &raw_name, true, false);
                    self.symbol_table
                        .new_binding(&rename, &Type::Bit(IsConst::False))?;
                    self.symbol_table.lookup(&rename)
                }
            }?;
            decls.push(ClassicalDeclaration {
                type_: ClassicalType::Bit,
                identifier: Identifier {
                    string: sym.symbol().name().to_string(),
                },
            });
        }
        self.global_classical_decls.extend(decls);
        Ok(())
    }

    fn build_qubit_decls(&mut self) -> ExporterResult<Vec<QuantumDeclaration>> {
        self.assert_global_scope()?;
        let num_qubits = self.circuit_scope.circuit_data.num_qubits();
        let mut decls = Vec::new();

        if self.is_layout {
            self.loose_qubit_prefix = "$";
        }
        for i in 0..num_qubits {
            let raw_name = format!("{}{}", self.loose_qubit_prefix, i);
            let sym = match self.symbol_table.new_binding(&raw_name, &Type::Qubit) {
                Ok(_) => self.symbol_table.lookup(&raw_name),
                Err(_) => {
                    if self.is_layout {
                        // If layout, return error directly
                        return Err(QASM3ExporterError::SymbolNotFound(raw_name));
                    }
                    let rename =
                        escape_invalid_identifier(&self.symbol_table, &raw_name, true, false);
                    self.symbol_table.new_binding(&rename, &Type::Qubit)?;
                    self.symbol_table.lookup(&rename)
                }
            }?;
            decls.push(QuantumDeclaration {
                identifier: Identifier {
                    string: sym.symbol().name().to_string(),
                },
                designator: None,
            });
        }
        Ok(decls)
    }

    fn build_top_level_stmts(&mut self) -> ExporterResult<Vec<Statement>> {
        let mut stmts = Vec::new();
        for instr in self.circuit_scope.circuit_data.data() {
            self.build_instruction(instr, &mut stmts)?;
        }
        Ok(stmts)
    }

    fn build_instruction(
        &mut self,
        instruction: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let name = instruction.op.name();

        if instruction.op.control_flow() {
            return Err(QASM3ExporterError::Error(format!(
                "Control flow is not supported: {}",
                name
            )));
        }

        match name {
            "barrier" => self.handle_barrier(instruction, stmts),
            "measure" => self.handle_measure(instruction, stmts),
            "reset" => self.handle_reset(instruction, stmts),
            "delay" => self.handle_delay(instruction, stmts),
            "break_loop" => {
                stmts.push(Statement::Break(Break {}));
                Ok(())
            }
            "continue_loop" => {
                stmts.push(Statement::Continue(Continue {}));
                Ok(())
            }
            "store" => {
                panic!("Store is not yet supported");
            }
            _ => {
                let gate_call = self.build_gate_call(instruction)?;
                stmts.push(Statement::QuantumInstruction(QuantumInstruction::GateCall(
                    gate_call,
                )));
                Ok(())
            }
        }
    }

    fn handle_barrier(
        &mut self,
        instr: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let mut qubit_ids = Vec::new();
        for q in qargs {
            let name = format!("{}{}", self.loose_qubit_prefix, q.0);
            let symbol = self.symbol_table.lookup(&name)?;
            qubit_ids.push(Identifier {
                string: symbol.symbol().name().to_string(),
            });
        }
        stmts.push(Statement::QuantumInstruction(QuantumInstruction::Barrier(
            Barrier {
                index_identifier_list: qubit_ids,
            },
        )));
        Ok(())
    }

    fn handle_measure(
        &mut self,
        instr: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let mut qubits = Vec::new();
        for q in qargs {
            let name = format!("{}{}", self.loose_qubit_prefix, q.0);
            let symbol = self.symbol_table.lookup(&name)?;
            qubits.push(Identifier {
                string: symbol.symbol().name().to_string(),
            });
        }
        let measurement = QuantumMeasurement {
            identifier_list: qubits,
        };

        let cargs = self
            .circuit_scope
            .circuit_data
            .cargs_interner()
            .get(instr.clbits);
        let c_name = format!("{}{}", self.loose_bit_prefix, cargs[0].0);
        let clbit_symbol = self.symbol_table.lookup(&c_name)?;
        stmts.push(Statement::QuantumMeasurementAssignment(
            QuantumMeasurementAssignment {
                identifier: Identifier {
                    string: clbit_symbol.symbol().name().to_string(),
                },
                quantum_measurement: measurement,
            },
        ));
        Ok(())
    }

    fn handle_reset(
        &mut self,
        instr: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        for q in qargs {
            let name = format!("{}{}", self.loose_qubit_prefix, q.0);
            let sym = self.symbol_table.lookup(&name)?;
            stmts.push(Statement::QuantumInstruction(QuantumInstruction::Reset(
                Reset {
                    identifier: Identifier {
                        string: sym.symbol().name().to_string(),
                    },
                },
            )));
        }
        Ok(())
    }

    fn handle_delay(
        &self,
        instr: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        if instr.op.num_clbits() > 0 {
            return Err(QASM3ExporterError::Error(
                "Delay cannot have classical bits".to_string(),
            ));
        }
        let delay = self.build_delay(instr)?;
        stmts.push(Statement::QuantumInstruction(QuantumInstruction::Delay(
            delay,
        )));
        Ok(())
    }

    fn build_delay(&self, instr: &PackedInstruction) -> ExporterResult<Delay> {
        let duration_value = Python::with_gil(|py| {
            let dur = instr.extra_attrs.duration().ok_or_else(|| {
                QASM3ExporterError::Error("Failed to extract duration".to_string())
            })?;
            dur.bind(py).extract::<f64>().map_err(|_| {
                QASM3ExporterError::Error("Failed to extract duration value".to_string())
            })
        })?;

        let default_unit = "us";
        let unit = instr.extra_attrs.unit().unwrap_or(default_unit);

        let mut map = HashMap::new();
        map.insert("ns", DurationUnit::Nanosecond);
        map.insert("us", DurationUnit::Microsecond);
        map.insert("ms", DurationUnit::Millisecond);
        map.insert("s", DurationUnit::Second);
        map.insert("dt", DurationUnit::Sample);

        let duration_literal = if unit == "ps" {
            DurationLiteral {
                value: duration_value * 1000.0,
                unit: DurationUnit::Nanosecond,
            }
        } else {
            let found = map.get(unit).ok_or_else(|| {
                QASM3ExporterError::Error(format!("Unknown duration unit: {}", unit))
            })?;
            DurationLiteral {
                value: duration_value,
                unit: found.clone(),
            }
        };

        let mut qubits = Vec::new();
        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        for q in qargs {
            let name = format!("{}{}", self.loose_qubit_prefix, q.0);
            let sym = self.symbol_table.lookup(&name)?;
            qubits.push(Identifier {
                string: sym.symbol().name().to_string(),
            });
        }
        Ok(Delay {
            duration: duration_literal,
            qubits,
        })
    }

    fn build_gate_call(&mut self, instr: &PackedInstruction) -> ExporterResult<GateCall> {
        let op_name = instr.op.name();
        if !self.symbol_table.standard_library_gates().contains(&op_name)
            && !self.symbol_table.all_scopes_contains_name(op_name)
        {
            panic!("Non-standard gate calls are not yet supported: {}", op_name);
        }
        let params = if self.disable_constants {
            instr
                .params_view()
                .iter()
                .map(|param| match param {
                    Param::Float(val) => Expression::Parameter(Parameter {
                        obj: val.to_string(),
                    }),
                    Param::ParameterExpression(_) => {
                        panic!("Parameter expressions not supported yet")
                    }
                    Param::Obj(_) => panic!("Objects not supported yet"),
                })
                .collect()
        } else {
            panic!("Constant parameters not supported in this sample");
        };

        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let mut qubit_ids = Vec::new();
        for q in qargs {
            let name = format!("{}{}", self.loose_qubit_prefix, q.0);
            let sym = self.symbol_table.lookup(&name)?;
            qubit_ids.push(Identifier {
                string: sym.symbol().name().to_string(),
            });
        }

        Ok(GateCall {
            quantum_gate_name: Identifier {
                string: op_name.to_string(),
            },
            index_identifier_list: qubit_ids,
            parameters: params,
            modifiers: None,
        })
    }

    #[allow(dead_code)]
    fn define_gate(&mut self, instr: &PackedInstruction) -> ExporterResult<QuantumGateDefinition> {
        let operation = &instr.op;
        if operation.name() == "cx" && operation.num_qubits() == 2 {
            let ctrl_id = Identifier {
                string: "c".to_string(),
            };
            let tgt_id = Identifier {
                string: "t".to_string(),
            };
            let call = GateCall {
                quantum_gate_name: Identifier {
                    string: "U".to_string(),
                },
                index_identifier_list: vec![ctrl_id.clone(), tgt_id.clone()],
                parameters: vec![
                    Expression::Constant(Constant::PI),
                    Expression::IntegerLiteral(IntegerLiteral { value: 0 }),
                    Expression::Constant(Constant::PI),
                ],
                modifiers: Some(vec![QuantumGateModifier {
                    modifier: QuantumGateModifierName::Ctrl,
                    argument: None,
                }]),
            };
            return Ok(QuantumGateDefinition {
                quantum_gate_signature: QuantumGateSignature {
                    name: Identifier {
                        string: "cx".to_string(),
                    },
                    qarg_list: vec![ctrl_id, tgt_id],
                    params: None,
                },
                quantum_block: QuantumBlock {
                    statements: vec![Statement::QuantumInstruction(QuantumInstruction::GateCall(
                        call,
                    ))],
                },
            });
        }
        Err(QASM3ExporterError::Error(
            "Custom gate definition not supported in this sample".to_string(),
        ))
    }
}

// Copyright contributors to the openqasm-parser project
// SPDX-License-Identifier: Apache-2.0

// Defines data structures and api for symbols, scope, and symbol tables.

use hashbrown::HashMap;
use oq3_semantics::types;
use oq3_semantics::types::Type;

// OQ3
// * "The lifetime of each identifier begins when it is declared, and ends
//    at the completion of the scope it was declared in."
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScopeType {
    /// Top-level
    Global,
    /// Body of `gate` and `def`
    Subroutine,
    /// `cal` and `defcal` blocks
    Calibration,
    /// Control flow blocks
    Local,
}

// This wrapped `usize` serves as
// * A unique label for instances of `Symbol`.
// * An index into `all_symbols: Vec<Symbol>`.
// * The values in `SymbolMap`.
//
// I am assuming that we can clone `SymbolId` willy-nilly
// because it is no more expensive than a reference.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymbolId(usize);

impl SymbolId {
    pub fn new() -> SymbolId {
        SymbolId(0)
    }

    /// Post-increment the value, and return the old value.
    /// This is used for getting a new `SymbolId`.
    pub fn post_increment(&mut self) -> SymbolId {
        let old_val = self.clone();
        self.0 += 1;
        old_val
    }
}

impl Default for SymbolId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SymbolError {
    MissingBinding,
    AlreadyBound,
}

pub type SymbolIdResult = Result<SymbolId, SymbolError>;
pub type SymbolRecordResult<'a> = Result<SymbolRecord<'a>, SymbolError>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Symbol {
    name: String,
    typ: Type,
    //    ast_node: SyntaxNode,
}

pub trait SymbolType {
    /// Return the `Type` of `symbol`, which is `Type::Undefined` if
    /// `self` is an `Option<T>` with value `None`.
    fn symbol_type(&self) -> &Type;
}

impl Symbol {
    fn new<T: ToString>(name: T, typ: &Type) -> Symbol {
        Symbol {
            name: name.to_string(),
            typ: typ.clone(),
        }
        // fn new<T: ToString>(name: T, typ: &Type, ast_node: &SyntaxNode) -> Symbol {
        //     Symbol { name: name.to_string(), typ: typ.clone() }
        //        Symbol { name: name.to_string(), typ: typ.clone(), ast_node: ast_node.clone() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl SymbolType for Symbol {
    fn symbol_type(&self) -> &Type {
        &self.typ
    }
}

/// A structure for temporarily collecting information about
/// a symbol.
/// * `Symbol` contains `name: String` and the `Type`.
/// * `symbol_id` wraps a `usize` that serves as
///     * a unique label
///     * the index into the `Vec` of all symbols.
///     * the value in `SymbolMap`: `name` -> `symbol_id`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolRecord<'a> {
    symbol: &'a Symbol,
    symbol_id: SymbolId,
}

impl SymbolRecord<'_> {
    pub fn new(symbol: &Symbol, symbol_id: SymbolId) -> SymbolRecord<'_> {
        SymbolRecord { symbol, symbol_id }
    }

    pub fn symbol_id(&self) -> SymbolId {
        self.symbol_id.clone()
    }

    pub fn symbol(&self) -> &Symbol {
        self.symbol
    }
}

// This trait is a bit heavy weight for what it does.
pub trait SymbolErrorTrait {
    fn to_symbol_id(&self) -> SymbolIdResult;
    fn as_tuple(&self) -> (SymbolIdResult, Type);
}

impl SymbolErrorTrait for SymbolRecordResult<'_> {
    fn to_symbol_id(&self) -> SymbolIdResult {
        self.clone().map(|record| record.symbol_id)
    }

    fn as_tuple(&self) -> (SymbolIdResult, Type) {
        (self.to_symbol_id(), self.symbol_type().clone())
    }
}

impl SymbolType for Option<SymbolRecord<'_>> {
    fn symbol_type(&self) -> &Type {
        match self {
            Some(symbol_record) => symbol_record.symbol_type(),
            None => &Type::Undefined,
        }
    }
}

impl SymbolType for Result<SymbolRecord<'_>, SymbolError> {
    fn symbol_type(&self) -> &Type {
        match self {
            Ok(symbol_record) => symbol_record.symbol_type(),
            Err(_) => &Type::Undefined,
        }
    }
}

impl SymbolType for SymbolRecord<'_> {
    fn symbol_type(&self) -> &Type {
        self.symbol.symbol_type()
    }
}

/// A `SymbolMap` is a map from `names` to `SymbolId` for a single instance
/// of a scope.
/// A `SymbolTable` is a stack of `SymbolMap`s together with a `Vec` mapping
/// `SymbolId as usize` to `Symbol`s.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
struct SymbolMap {
    table: HashMap<String, SymbolId>,
    scope_type: ScopeType,
}

impl SymbolMap {
    fn new(scope_type: ScopeType) -> SymbolMap {
        SymbolMap {
            table: HashMap::<String, SymbolId>::new(),
            scope_type,
        }
    }

    pub fn insert<T: ToString>(&mut self, name: T, sym: SymbolId) {
        self.table.insert(name.to_string(), sym);
    }

    pub fn get_symbol_id(&self, name: &str) -> Option<&SymbolId> {
        self.table.get(name)
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn contains_name(&self, name: &str) -> bool {
        self.table.contains_key(name)
    }

    /// Return the `ScopeType` of the `SymbolMap` of the current, or top-most, scope.
    pub fn scope_type(&self) -> ScopeType {
        self.scope_type.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolTable {
    /// A stack each of whose elements represent a scope mapping `name: String` to `SymbolId`.
    symbol_table_stack: Vec<SymbolMap>,
    /// A list of all `Symbol`s with no explicit scope information. Indices are `SymbolId as usize`.
    all_symbols: Vec<Symbol>,
    /// A counter that is incremented after each new symbol is created.
    symbol_id_counter: SymbolId,
}

impl SymbolTable {
    // This will be called if `include "stdgates.inc"` is encountered. At present we don't have any include guard.
    // FIXME: This function allocates a vector. The caller iterates over the vector.
    //  Would be nice to return the `FlatMap` instead. I tried doing this, but it was super compilcated.
    //  The compiler helps with which trait to use as the return type. But then tons of bugs occur within
    //  the body.
    /// Define gates in standard library "as if" a file of definitions (or declarations) had been read.
    pub(crate) fn standard_library_gates(&mut self) -> Vec<&str> {
        let g1q0p = (
            vec![
                "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", /* 2.0 */ "id",
            ],
            [0, 1],
        );
        let g1q1p = (vec!["p", "rx", "ry", "rz", /* 2.0 */ "phase", "u1"], [1, 1]);
        let g1q2p = (vec![/* 2.0 */ "u2"], [2, 1]);
        let g1q3p = (vec![/* 2.0 */ "u3"], [3, 1]);
        let g2q0p = (vec!["cx", "cy", "cz", "ch", "swap", /* 2.0 */ "CX"], [0, 2]);
        let g2q1p = (vec!["cp", "crx", "cry", "crz", /* 2.0 */ "cphase"], [1, 2]);
        let g2q4p = (vec!["cu"], [4, 2]);
        let g3q0p = (vec!["ccx", "cswap"], [0, 3]);
        let all_gates = vec![g1q0p, g1q1p, g1q2p, g1q3p, g2q0p, g2q1p, g2q4p, g3q0p];
        // If `new_binding` returns `Err`, we push `name` onto a vector which will be
        // used by the caller to record errors. Here flat_map and filter are used to
        // select filter the names.
        all_gates
            .into_iter()
            .flat_map(|(names, [n_cl, n_qu])| {
                names
                    .into_iter()
                    .filter(|name| {
                        // The side effect of the test is important!
                        self.new_binding(name, &Type::Gate(n_cl, n_qu)).is_err()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    /// Return an iterator giving information about all gate declarations. Each element
    /// is a tuple of (gate name, symbol id, num classical params, num quantum params).
    /// `gphase` is not included here. It is treated specially.
    /// `U` is also filtered out, as it is builtin.
    pub fn gates(&self) -> impl Iterator<Item = (&str, SymbolId, usize, usize)> {
        self.all_symbols.iter().enumerate().filter_map(|(n, sym)| {
            if let Type::Gate(num_cl, num_qu) = &sym.symbol_type() {
                if sym.name() == "U" {
                    None
                } else {
                    Some((sym.name(), SymbolId(n), *num_cl, *num_qu))
                }
            } else {
                None
            }
        })
    }

    /// Return a list of hardware qubits referenced in the program as a
    /// `Vec` of 2-tuples. In each tuple the first item is the name and
    /// the second is the `SymbolId`.
    pub fn hardware_qubits(&self) -> Vec<(&str, SymbolId)> {
        self.all_symbols
            .iter()
            .enumerate()
            .filter_map(|(n, sym)| {
                if let Type::HardwareQubit = &sym.symbol_type() {
                    Some((sym.name(), SymbolId(n)))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[allow(dead_code)]
impl SymbolTable {
    /// Create a new `SymbolTable` and initialize with the global scope.
    pub fn new() -> SymbolTable {
        let mut symbol_table = SymbolTable {
            symbol_id_counter: SymbolId::new(),
            symbol_table_stack: Vec::<SymbolMap>::new(),
            all_symbols: Vec::<Symbol>::new(),
        };
        symbol_table.enter_scope(ScopeType::Global);
        // Define global, built-in constants, and the single built-in gate
        for const_name in ["pi", "π", "euler", "ℇ", "tau", "τ"] {
            let _ =
                symbol_table.new_binding(const_name, &Type::Float(Some(64), types::IsConst::True));
        }
        let _ = symbol_table.new_binding("U", &Type::Gate(3, 1)); // U(a, b, c) q
        symbol_table
    }

    fn number_of_scopes(&self) -> usize {
        self.symbol_table_stack.len()
    }

    /// Enter a new scope of type `scope_type`. New bindings will occur in this
    /// scope. This scope will be the first one searched when resolving symbols.
    /// Certain symbols are excepted, such as gate names, which are always global.
    pub(crate) fn enter_scope(&mut self, scope_type: ScopeType) {
        if scope_type == ScopeType::Global && self.number_of_scopes() > 0 {
            panic!("The unique global scope must be the first scope.")
        }
        self.symbol_table_stack.push(SymbolMap::new(scope_type))
    }

    /// Exit the current scope and return to the enclosing scope.
    pub fn exit_scope(&mut self) {
        // Trying to exit the global scope is a programming error.
        assert!(self.symbol_table_stack.len() > 1);
        self.symbol_table_stack.pop();
    }

    // Make a new binding without checking first whether a binding exists in
    // this scope.
    fn new_binding_no_check(&mut self, name: &str, typ: &Type) -> SymbolId {
        // Create new symbol and symbol id.
        //        let symbol = Symbol::new(name, typ, ast_node);
        let symbol = Symbol::new(name, typ);

        // Push the symbol onto list of all symbols (in all scopes). Index
        // to this symbol will be `id_count`.
        self.all_symbols.push(symbol);

        // The "current" SymbolId has not yet been unused.
        // Get the current SymbolId and increment the counter.
        let current_symbol_id = self.symbol_id_counter.post_increment();

        // Map `name` to `symbol_id`.
        self.current_scope_mut()
            .insert(name, current_symbol_id.clone());
        current_symbol_id
    }

    /// If a binding for `name` exists in the current scope, return `Err(SymbolError::AlreadyBound)`.
    /// Otherwise, create a new Symbol from `name` and `typ`, bind `name` to
    /// this Symbol in the current scope, and return the Symbol.
    pub fn new_binding(&mut self, name: &str, typ: &Type) -> Result<SymbolId, SymbolError> {
        // Can't create a binding if it already exists in the current scope.
        if self.current_scope_contains_name(name) {
            return Err(SymbolError::AlreadyBound);
        }
        Ok(self.new_binding_no_check(name, typ))
    }

    // Symbol table for current (latest) scope in stack, mutable ref
    fn current_scope_mut(&mut self) -> &mut SymbolMap {
        self.symbol_table_stack.last_mut().unwrap()
    }

    // Symbol table for current (latest) scope in stack, immutable ref
    fn current_scope(&self) -> &SymbolMap {
        self.symbol_table_stack.last().unwrap()
    }

    /// Return the `ScopeType` of the current, or top-most, scope.
    pub(crate) fn current_scope_type(&self) -> ScopeType {
        self.current_scope().scope_type()
    }

    pub(crate) fn in_global_scope(&self) -> bool {
        self.current_scope_type() == ScopeType::Global
    }

    // Return `true` if `name` is bound in current scope.
    fn current_scope_contains_name(&self, name: &str) -> bool {
        self.current_scope().contains_name(name)
    }

    /// Return `true` if `name` is bound in all the scope.
    pub fn all_scopes_contains_name(&self, name: &str) -> bool {
        self.symbol_table_stack
            .iter()
            .any(|table| table.contains_name(name))
    }

    // /// Look up and return the `SymbolId` that `name` is bound to in the current scope.
    // pub fn lookup_current_scope(&self, name: &str) -> Option<&SymbolId> {
    //     self.current_scope().get(name)
    // }

    // FIXME: Can we make this private?
    // This is only used in tests. The tests are in a different crate, so this function
    // must be public. But it is not needed for anything other than tests.
    /// Return the length (number of bindings) in the current scope.
    pub fn len_current_scope(&self) -> usize {
        self.current_scope().len()
    }

    /// Look up `name` in the stack of symbol tables. Return `SymbolRecord`
    /// if the symbol is found. Otherwise `Err(SymbolError::MissingBinding)`.
    pub fn lookup(&self, name: &str) -> Result<SymbolRecord, SymbolError> {
        for table in self.symbol_table_stack.iter().rev() {
            if let Some(symbol_id) = table.get_symbol_id(name) {
                return Ok(SymbolRecord::new(
                    &self.all_symbols[symbol_id.0],
                    symbol_id.clone(),
                ));
            }
        }
        Err(SymbolError::MissingBinding) // `name` not found in any scope.
    }

    /// Try to lookup `name`. If a binding is found return the `SymbolId`, otherwise create a new binding
    /// and return the new `SymbolId`.
    pub fn lookup_or_new_binding(&mut self, name: &str, typ: &Type) -> SymbolId {
        match self.lookup(name) {
            Ok(symbol_record) => symbol_record.symbol_id,
            Err(_) => self.new_binding_no_check(name, typ),
        }
    }

    /// Simple dump of all symbols. This could be improved.
    pub fn dump(&self) {
        for (n, sym) in self.all_symbols.iter().enumerate() {
            println!("{n} {:?}", sym);
        }
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

use std::ops::Index;
impl Index<&SymbolId> for SymbolTable {
    type Output = Symbol;

    // Interface for retrieving `Symbol`s by `SymbolId`.
    // Indexing into the symbol table with a `SymbolId` (which wraps an integer)
    // returns the `Symbol`, which contains the name and type.
    fn index(&self, symbol_id: &SymbolId) -> &Self::Output {
        &self.all_symbols[symbol_id.0]
    }
}

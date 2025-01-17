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

use itertools::Itertools;

use pyo3::exceptions::PyTypeError;
use qiskit_circuit::parameter_table::ParameterUuid;
use rustworkx_core::petgraph::csr::IndexType;
use rustworkx_core::petgraph::stable_graph::StableDiGraph;
use rustworkx_core::petgraph::visit::IntoEdgeReferences;

use smallvec::SmallVec;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::{error::Error, fmt::Display};

use exceptions::CircuitError;

use ahash::RandomState;
use indexmap::{IndexMap, IndexSet};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyString};

use rustworkx_core::petgraph::{
    graph::{EdgeIndex, NodeIndex},
    visit::EdgeRef,
};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::imports::{ImportOnceCell, QUANTUM_CIRCUIT};
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::{Operation, OperationRef};
use qiskit_circuit::packed_instruction::PackedOperation;

mod exceptions {
    use pyo3::import_exception_bound;
    import_exception_bound! {qiskit.circuit.exceptions, CircuitError}
}
pub static PYDIGRAPH: ImportOnceCell = ImportOnceCell::new("rustworkx", "PyDiGraph");

// Custom Structs

#[pyclass(frozen, sequence, module = "qiskit._accelerate.equivalence")]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub num_qubits: u32,
}

#[pymethods]
impl Key {
    #[new]
    #[pyo3(signature = (name, num_qubits))]
    fn new(name: String, num_qubits: u32) -> Self {
        Self { name, num_qubits }
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        (self.name.to_string(), self.num_qubits).hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(slf: PyRef<Self>) -> String {
        slf.to_string()
    }

    fn __getnewargs__(slf: PyRef<Self>) -> (Bound<PyString>, u32) {
        (
            PyString::new_bound(slf.py(), slf.name.as_str()),
            slf.num_qubits,
        )
    }

    // Ord methods for Python
    fn __lt__(&self, other: &Self) -> bool {
        self.lt(other)
    }
    fn __le__(&self, other: &Self) -> bool {
        self.le(other)
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.eq(other)
    }
    fn __ne__(&self, other: &Self) -> bool {
        self.ne(other)
    }
    fn __ge__(&self, other: &Self) -> bool {
        self.ge(other)
    }
    fn __gt__(&self, other: &Self) -> bool {
        self.gt(other)
    }
}
impl Key {
    fn from_operation(operation: &PackedOperation) -> Self {
        let op_ref: OperationRef = operation.view();
        Key {
            name: op_ref.name().to_string(),
            num_qubits: op_ref.num_qubits(),
        }
    }
}

impl Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Key(name=\'{}\', num_qubits={})",
            self.name, self.num_qubits
        )
    }
}

#[pyclass(frozen, sequence, module = "qiskit._accelerate.equivalence")]
#[derive(Debug, Clone)]
pub struct Equivalence {
    #[pyo3(get)]
    pub params: SmallVec<[Param; 3]>,
    #[pyo3(get)]
    pub circuit: CircuitFromPython,
}

#[pymethods]
impl Equivalence {
    #[new]
    #[pyo3(signature = (params, circuit))]
    fn new(params: SmallVec<[Param; 3]>, circuit: CircuitFromPython) -> Self {
        Self { circuit, params }
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }

    fn __eq__(slf: &Bound<Self>, other: &Bound<PyAny>) -> PyResult<bool> {
        let other_params = other.getattr("params")?;
        let other_circuit = other.getattr("circuit")?;
        Ok(other_params.eq(&slf.getattr("params")?)?
            && other_circuit.eq(&slf.getattr("circuit")?)?)
    }

    fn __getnewargs__<'py>(
        slf: &'py Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        Ok((slf.getattr("params")?, slf.getattr("circuit")?))
    }
}

impl Display for Equivalence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Equivalence(params=[{}], circuit={:?})",
            self.params
                .iter()
                .map(|param| format!("{:?}", param))
                .format(", "),
            self.circuit
        )
    }
}

#[pyclass(frozen, sequence, module = "qiskit._accelerate.equivalence")]
#[derive(Debug, Clone)]
pub struct NodeData {
    #[pyo3(get)]
    pub key: Key,
    #[pyo3(get)]
    pub equivs: Vec<Equivalence>,
}

#[pymethods]
impl NodeData {
    #[new]
    #[pyo3(signature = (key, equivs))]
    fn new(key: Key, equivs: Vec<Equivalence>) -> Self {
        Self { key, equivs }
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }

    fn __eq__(slf: &Bound<Self>, other: &Bound<PyAny>) -> PyResult<bool> {
        Ok(slf.getattr("key")?.eq(other.getattr("key")?)?
            && slf.getattr("equivs")?.eq(other.getattr("equivs")?)?)
    }

    fn __getnewargs__<'py>(
        slf: &'py Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        Ok((slf.getattr("key")?, slf.getattr("equivs")?))
    }
}

impl Display for NodeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NodeData(key={}, equivs=[{}])",
            self.key,
            self.equivs.iter().format(", ")
        )
    }
}

#[pyclass(frozen, sequence, module = "qiskit._accelerate.equivalence")]
#[derive(Debug, Clone)]
pub struct EdgeData {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub num_gates: usize,
    #[pyo3(get)]
    pub rule: Equivalence,
    #[pyo3(get)]
    pub source: Key,
}

#[pymethods]
impl EdgeData {
    #[new]
    #[pyo3(signature = (index, num_gates, rule, source))]
    fn new(index: usize, num_gates: usize, rule: Equivalence, source: Key) -> Self {
        Self {
            index,
            num_gates,
            rule,
            source,
        }
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }

    fn __eq__(slf: &Bound<Self>, other: &Bound<Self>) -> PyResult<bool> {
        let other_borrowed = other.borrow();
        let slf_borrowed = slf.borrow();
        Ok(slf_borrowed.index == other_borrowed.index
            && slf_borrowed.num_gates == other_borrowed.num_gates
            && slf_borrowed.source == other_borrowed.source
            && other.getattr("rule")?.eq(slf.getattr("rule")?)?)
    }

    fn __getnewargs__(slf: PyRef<Self>) -> (usize, usize, Equivalence, Key) {
        (
            slf.index,
            slf.num_gates,
            slf.rule.clone(),
            slf.source.clone(),
        )
    }
}

impl Display for EdgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EdgeData(index={}, num_gates={}, rule={}, source={})",
            self.index, self.num_gates, self.rule, self.source
        )
    }
}

/// Enum that helps extract the Operation and Parameters on a Gate.
/// It is highly derivative of [PackedOperation] while also tracking the specific
/// parameter objects.
#[derive(Debug, Clone)]
pub struct GateOper {
    operation: PackedOperation,
    params: SmallVec<[Param; 3]>,
}

impl<'py> FromPyObject<'py> for GateOper {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let op_struct: OperationFromPython = ob.extract()?;
        Ok(Self {
            operation: op_struct.operation,
            params: op_struct.params,
        })
    }
}

/// Used to extract an instance of [CircuitData] from a [`QuantumCircuit`].
/// It also ensures seamless conversion back to [`QuantumCircuit`] once sent
/// back to Python.
///
/// TODO: Remove this implementation once the [EquivalenceLibrary] is no longer
/// called from Python, or once the API is able to seamlessly accept instances
/// of [CircuitData].
///
/// [`QuantumCircuit`]: https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit
#[derive(Debug, Clone)]
pub struct CircuitFromPython(pub CircuitData);

impl FromPyObject<'_> for CircuitFromPython {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if ob.is_instance(QUANTUM_CIRCUIT.get_bound(ob.py()))? {
            let data: Bound<PyAny> = ob.getattr("_data")?;
            let data_downcast: Bound<CircuitData> = data.downcast_into()?;
            let data_extract: CircuitData = data_downcast.extract()?;
            Ok(Self(data_extract))
        } else {
            Err(PyTypeError::new_err(
                "Provided object was not an instance of QuantumCircuit",
            ))
        }
    }
}

impl IntoPy<PyObject> for CircuitFromPython {
    fn into_py(self, py: Python<'_>) -> PyObject {
        QUANTUM_CIRCUIT
            .get_bound(py)
            .call_method1("_from_circuit_data", (self.0,))
            .unwrap()
            .unbind()
    }
}

impl ToPyObject for CircuitFromPython {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.clone().into_py(py)
    }
}

// Custom Types
type GraphType = StableDiGraph<NodeData, Option<EdgeData>>;
type KTIType = IndexMap<Key, NodeIndex, RandomState>;

/// A library providing a one-way mapping of gates to their equivalent
/// implementations as :class:`.QuantumCircuit` instances.
#[pyclass(
    subclass,
    name = "BaseEquivalenceLibrary",
    module = "qiskit._accelerate.equivalence"
)]
#[derive(Debug, Clone)]
pub struct EquivalenceLibrary {
    graph: GraphType,
    key_to_node_index: KTIType,
    rule_id: usize,
    _graph: Option<PyObject>,
}

#[pymethods]
impl EquivalenceLibrary {
    /// Create a new equivalence library.
    ///
    /// Args:
    ///     base (Optional[EquivalenceLibrary]):  Base equivalence library to
    ///         be referenced if an entry is not found in this library.
    #[new]
    #[pyo3(signature= (base=None))]
    fn new(base: Option<&EquivalenceLibrary>) -> Self {
        if let Some(base) = base {
            Self {
                graph: base.graph.clone(),
                key_to_node_index: base.key_to_node_index.clone(),
                rule_id: base.rule_id,
                _graph: None,
            }
        } else {
            Self {
                graph: GraphType::new(),
                key_to_node_index: KTIType::default(),
                rule_id: 0_usize,
                _graph: None,
            }
        }
    }

    /// Add a new equivalence to the library. Future queries for the Gate
    /// will include the given circuit, in addition to all existing equivalences
    /// (including those from base).
    ///
    /// Parameterized Gates (those including `qiskit.circuit.Parameters` in their
    /// `Gate.params`) can be marked equivalent to parameterized circuits,
    /// provided the parameters match.
    ///
    /// Args:
    ///     gate (Gate): A Gate instance.
    ///     equivalent_circuit (QuantumCircuit): A circuit equivalently
    ///         implementing the given Gate.
    #[pyo3(name = "add_equivalence")]
    fn py_add_equivalence(
        &mut self,
        py: Python,
        gate: GateOper,
        equivalent_circuit: CircuitFromPython,
    ) -> PyResult<()> {
        self.add_equivalence(py, &gate.operation, &gate.params, equivalent_circuit)
    }

    /// Check if a library contains any decompositions for gate.
    ///
    /// Args:
    ///     gate (Gate): A Gate instance.
    ///
    /// Returns:
    ///     Bool: True if gate has a known decomposition in the library.
    ///         False otherwise.
    #[pyo3(name = "has_entry")]
    fn py_has_entry(&self, gate: GateOper) -> bool {
        self.has_entry(&gate.operation)
    }

    /// Set the equivalence record for a Gate. Future queries for the Gate
    /// will return only the circuits provided.
    ///
    /// Parameterized Gates (those including `qiskit.circuit.Parameters` in their
    /// `Gate.params`) can be marked equivalent to parameterized circuits,
    /// provided the parameters match.
    ///
    /// Args:
    ///     gate (Gate): A Gate instance.
    ///     entry (List['QuantumCircuit']) : A list of :class:`.QuantumCircuit` instances, each
    ///         equivalently implementing the given Gate.
    #[pyo3(name = "set_entry")]
    fn py_set_entry(
        &mut self,
        py: Python,
        gate: GateOper,
        entry: Vec<CircuitFromPython>,
    ) -> PyResult<()> {
        self.set_entry(py, &gate.operation, &gate.params, entry)
    }

    /// Gets the set of :class:`.QuantumCircuit` instances circuits from the
    /// library which equivalently implement the given :class:`.Gate`.
    ///
    /// Parameterized circuits will have their parameters replaced with the
    /// corresponding entries from Gate.params.
    ///
    /// Args:
    ///     gate (Gate) - Gate: A Gate instance.
    ///
    /// Returns:
    ///     List[QuantumCircuit]: A list of equivalent :class:`.QuantumCircuit` instances.
    ///         If empty, library contains no known decompositions of Gate.
    ///
    ///         Returned circuits will be ordered according to their insertion in
    ///         the library, from earliest to latest, from top to base. The
    ///         ordering of the StandardEquivalenceLibrary will not generally be
    ///         consistent across Qiskit versions.
    fn get_entry(&self, py: Python, gate: GateOper) -> PyResult<Py<PyList>> {
        let key = Key::from_operation(&gate.operation);
        let query_params = gate.params;

        let bound_equivalencies = self
            ._get_equivalences(&key)
            .into_iter()
            .filter_map(|equivalence| rebind_equiv(py, equivalence, &query_params).ok());
        let return_list = PyList::empty_bound(py);
        for equiv in bound_equivalencies {
            return_list.append(equiv)?;
        }
        Ok(return_list.unbind())
    }

    // TODO: Remove once BasisTranslator is in Rust.
    /// Return graph representing the equivalence library data.
    ///
    /// This property should be treated as read-only as it provides
    /// a reference to the internal state of the :class:`~.EquivalenceLibrary` object.
    /// If the graph returned by this property is mutated it could corrupt the
    /// the contents of the object. If you need to modify the output ``PyDiGraph``
    /// be sure to make a copy prior to any modification.
    ///
    /// Returns:
    ///     PyDiGraph: A graph object with equivalence data in each node.
    #[getter]
    fn get_graph(&mut self, py: Python) -> PyResult<PyObject> {
        if let Some(graph) = &self._graph {
            Ok(graph.clone_ref(py))
        } else {
            self._graph = Some(to_pygraph(py, &self.graph)?);
            Ok(self
                ._graph
                .as_ref()
                .map(|graph| graph.clone_ref(py))
                .unwrap())
        }
    }

    /// Get all the equivalences for the given key
    pub fn _get_equivalences(&self, key: &Key) -> Vec<Equivalence> {
        if let Some(key_in) = self.key_to_node_index.get(key) {
            self.graph[*key_in].equivs.clone()
        } else {
            vec![]
        }
    }

    /// Return list of keys to key to node index map.
    ///
    /// Returns:
    ///     List: Keys to the key to node index map.
    #[pyo3(name = "keys")]
    fn py_keys(slf: PyRef<Self>) -> PyResult<PyObject> {
        let py_dict = PyDict::new_bound(slf.py());
        for key in slf.keys() {
            py_dict.set_item(key.clone().into_py(slf.py()), slf.py().None())?;
        }
        Ok(py_dict.as_any().call_method0("keys")?.into())
    }

    /// Return node index for a given key.
    ///
    /// Args:
    ///     key (Key): Key to an equivalence.
    ///
    /// Returns:
    ///     Int: Index to the node in the graph for the given key.
    #[pyo3(name = "node_index")]
    fn py_node_index(&self, key: &Key) -> usize {
        self.node_index(key).index()
    }

    fn __getstate__(slf: PyRef<Self>) -> PyResult<Bound<PyDict>> {
        let ret = PyDict::new_bound(slf.py());
        ret.set_item("rule_id", slf.rule_id)?;
        let key_to_usize_node: Bound<PyDict> = PyDict::new_bound(slf.py());
        for (key, val) in slf.key_to_node_index.iter() {
            key_to_usize_node.set_item(key.clone().into_py(slf.py()), val.index())?;
        }
        ret.set_item("key_to_node_index", key_to_usize_node)?;
        let graph_nodes: Bound<PyList> = PyList::empty_bound(slf.py());
        for weight in slf.graph.node_weights() {
            graph_nodes.append(weight.clone().into_py(slf.py()))?;
        }
        ret.set_item("graph_nodes", graph_nodes.unbind())?;
        let edges = slf.graph.edge_references().map(|edge| {
            (
                edge.source().index(),
                edge.target().index(),
                edge.weight().clone().into_py(slf.py()),
            )
        });
        let graph_edges = PyList::empty_bound(slf.py());
        for edge in edges {
            graph_edges.append(edge)?;
        }
        ret.set_item("graph_edges", graph_edges.unbind())?;
        Ok(ret)
    }

    fn __setstate__(mut slf: PyRefMut<Self>, state: &Bound<PyDict>) -> PyResult<()> {
        slf.rule_id = state.get_item("rule_id")?.unwrap().extract()?;
        let graph_nodes_ref: Bound<PyAny> = state.get_item("graph_nodes")?.unwrap();
        let graph_nodes: &Bound<PyList> = graph_nodes_ref.downcast()?;
        let graph_edge_ref: Bound<PyAny> = state.get_item("graph_edges")?.unwrap();
        let graph_edges: &Bound<PyList> = graph_edge_ref.downcast()?;
        slf.graph = GraphType::new();
        for node_weight in graph_nodes {
            slf.graph.add_node(node_weight.extract()?);
        }
        for edge in graph_edges {
            let (source_node, target_node, edge_weight) = edge.extract()?;
            slf.graph.add_edge(
                NodeIndex::new(source_node),
                NodeIndex::new(target_node),
                edge_weight,
            );
        }
        slf.key_to_node_index = state
            .get_item("key_to_node_index")?
            .unwrap()
            .extract::<IndexMap<Key, usize>>()?
            .into_iter()
            .map(|(key, val)| (key, NodeIndex::new(val)))
            .collect();
        slf._graph = None;
        Ok(())
    }
}

// Rust native methods
impl EquivalenceLibrary {
    /// Add a new equivalence to the library. Future queries for the Gate
    /// will include the given circuit, in addition to all existing equivalences
    /// (including those from base).
    pub fn add_equivalence(
        &mut self,
        py: Python,
        gate: &PackedOperation,
        params: &[Param],
        equivalent_circuit: CircuitFromPython,
    ) -> PyResult<()> {
        raise_if_shape_mismatch(gate, &equivalent_circuit)?;
        raise_if_param_mismatch(py, params, equivalent_circuit.0.unsorted_parameters(py)?)?;
        let key: Key = Key::from_operation(gate);
        let equiv = Equivalence {
            circuit: equivalent_circuit.clone(),
            params: params.into(),
        };

        let target = self.set_default_node(key);
        if let Some(node) = self.graph.node_weight_mut(target) {
            node.equivs.push(equiv.clone());
        }
        let sources: IndexSet<Key, RandomState> = IndexSet::from_iter(
            equivalent_circuit
                .0
                .iter()
                .map(|inst| Key::from_operation(&inst.op)),
        );
        let edges = Vec::from_iter(sources.iter().map(|source| {
            (
                self.set_default_node(source.clone()),
                target,
                EdgeData {
                    index: self.rule_id,
                    num_gates: sources.len(),
                    rule: equiv.clone(),
                    source: source.clone(),
                },
            )
        }));
        for edge in edges {
            self.graph.add_edge(edge.0, edge.1, Some(edge.2));
        }
        self.rule_id += 1;
        self._graph = None;
        Ok(())
    }

    /// Set the equivalence record for a [PackedOperation]. Future queries for the Gate
    /// will return only the circuits provided.
    pub fn set_entry(
        &mut self,
        py: Python,
        gate: &PackedOperation,
        params: &[Param],
        entry: Vec<CircuitFromPython>,
    ) -> PyResult<()> {
        for equiv in entry.iter() {
            raise_if_shape_mismatch(gate, equiv)?;
            raise_if_param_mismatch(py, params, equiv.0.unsorted_parameters(py)?)?;
        }
        let key = Key::from_operation(gate);
        let node_index = self.set_default_node(key);

        if let Some(graph_ind) = self.graph.node_weight_mut(node_index) {
            graph_ind.equivs.clear();
        }

        let edges: Vec<EdgeIndex> = self
            .graph
            .edges_directed(node_index, rustworkx_core::petgraph::Direction::Incoming)
            .map(|x| x.id())
            .collect();
        for edge in edges {
            self.graph.remove_edge(edge);
        }
        for equiv in entry {
            self.add_equivalence(py, gate, params, equiv)?
        }
        self._graph = None;
        Ok(())
    }

    /// Check if the [EquivalenceLibrary] instance contains any decompositions for gate.
    pub fn has_entry(&self, operation: &PackedOperation) -> bool {
        let key = Key::from_operation(operation);
        self.key_to_node_index.contains_key(&key)
    }

    /// Returns an iterator with all the [Key] instances in the [EquivalenceLibrary].
    pub fn keys(&self) -> impl Iterator<Item = &Key> {
        self.key_to_node_index.keys()
    }

    /// Create a new node if key not found
    pub fn set_default_node(&mut self, key: Key) -> NodeIndex {
        if let Some(value) = self.key_to_node_index.get(&key) {
            *value
        } else {
            let node = self.graph.add_node(NodeData {
                key: key.clone(),
                equivs: vec![],
            });
            self.key_to_node_index.insert(key, node);
            node
        }
    }

    /// Retrieve the [NodeIndex] that represents a [Key].
    pub fn node_index(&self, key: &Key) -> NodeIndex {
        self.key_to_node_index[key]
    }

    /// Expose an immutable view of the inner graph.
    pub fn graph(&self) -> &GraphType {
        &self.graph
    }

    /// Expose a mutable view of the inner graph.
    pub(crate) fn graph_mut(&mut self) -> &mut GraphType {
        &mut self.graph
    }
}

fn raise_if_param_mismatch(
    py: Python,
    gate_params: &[Param],
    circuit_parameters: Bound<PySet>,
) -> PyResult<()> {
    let gate_params_obj = PySet::new_bound(
        py,
        gate_params
            .iter()
            .filter(|param| matches!(param, Param::ParameterExpression(_))),
    )?;
    if !gate_params_obj.eq(&circuit_parameters)? {
        return Err(CircuitError::new_err(format!(
            "Cannot add equivalence between circuit and gate \
            of different parameters. Gate params: {:?}. \
            Circuit params: {:?}.",
            gate_params, circuit_parameters
        )));
    }
    Ok(())
}

fn raise_if_shape_mismatch(gate: &PackedOperation, circuit: &CircuitFromPython) -> PyResult<()> {
    let op_ref = gate.view();
    if op_ref.num_qubits() != circuit.0.num_qubits() as u32
        || op_ref.num_clbits() != circuit.0.num_clbits() as u32
    {
        return Err(CircuitError::new_err(format!(
            "Cannot add equivalence between circuit and gate \
            of different shapes. Gate: {} qubits and {} clbits. \
            Circuit: {} qubits and {} clbits.",
            op_ref.num_qubits(),
            op_ref.num_clbits(),
            circuit.0.num_qubits(),
            circuit.0.num_clbits()
        )));
    }
    Ok(())
}

fn rebind_equiv(
    py: Python,
    equiv: Equivalence,
    query_params: &[Param],
) -> PyResult<CircuitFromPython> {
    let (equiv_params, mut equiv_circuit) = (equiv.params, equiv.circuit);
    let param_mapping: PyResult<IndexMap<ParameterUuid, &Param>> = equiv_params
        .iter()
        .zip(query_params.iter())
        .filter_map(|(param_x, param_y)| match param_x {
            Param::ParameterExpression(param) => {
                let param_uuid = ParameterUuid::from_parameter(param.bind(py));
                Some(param_uuid.map(|uuid| (uuid, param_y)))
            }
            _ => None,
        })
        .collect();
    equiv_circuit
        .0
        .assign_parameters_from_mapping(py, param_mapping?)?;
    Ok(equiv_circuit)
}

// Errors

#[derive(Debug, Clone)]
pub struct EquivalenceError {
    pub message: String,
}

impl EquivalenceError {
    pub fn new_err(message: String) -> Self {
        Self { message }
    }
}

impl Error for EquivalenceError {}

impl Display for EquivalenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

// Conversion helpers

fn to_pygraph<N, E>(py: Python, pet_graph: &StableDiGraph<N, E>) -> PyResult<PyObject>
where
    N: IntoPy<PyObject> + Clone,
    E: IntoPy<PyObject> + Clone,
{
    let graph = PYDIGRAPH.get_bound(py).call0()?;
    let node_weights: Vec<N> = pet_graph.node_weights().cloned().collect();
    graph.call_method1("add_nodes_from", (node_weights,))?;
    let edge_weights: Vec<(usize, usize, E)> = pet_graph
        .edge_references()
        .map(|edge| {
            (
                edge.source().index(),
                edge.target().index(),
                edge.weight().clone(),
            )
        })
        .collect();
    graph.call_method1("add_edges_from", (edge_weights,))?;
    Ok(graph.unbind())
}

#[pymodule]
pub fn equivalence(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EquivalenceLibrary>()?;
    m.add_class::<NodeData>()?;
    m.add_class::<EdgeData>()?;
    m.add_class::<Equivalence>()?;
    m.add_class::<Key>()?;
    Ok(())
}

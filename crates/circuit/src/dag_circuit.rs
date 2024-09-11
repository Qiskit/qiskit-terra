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

use std::hash::{Hash, Hasher};

use ahash::RandomState;

use crate::bit_data::BitData;
use crate::circuit_data::CircuitData;
use crate::circuit_instruction::{
    CircuitInstruction, ExtraInstructionAttributes, OperationFromPython,
};
use crate::converters::QuantumCircuitData;
use crate::dag_node::{DAGInNode, DAGNode, DAGOpNode, DAGOutNode};
use crate::dot_utils::build_dot;
use crate::error::DAGCircuitError;
use crate::imports;
use crate::interner::{Interned, Interner};
use crate::operations::{Operation, OperationRef, Param, PyInstruction, StandardGate};
use crate::packed_instruction::PackedInstruction;
use crate::rustworkx_core_vnext::isomorphism;
use crate::{BitType, Clbit, Qubit, TupleLikeArg};

use hashbrown::{HashMap, HashSet};
use indexmap::IndexMap;
use itertools::Itertools;

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{
    IntoPyDict, PyDict, PyInt, PyIterator, PyList, PySequence, PySet, PyString, PyTuple, PyType,
};

use rustworkx_core::dag_algo::layers;
use rustworkx_core::err::ContractError;
use rustworkx_core::graph_ext::ContractNodesDirected;
use rustworkx_core::petgraph;
use rustworkx_core::petgraph::prelude::StableDiGraph;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::stable_graph::{EdgeReference, NodeIndex};
use rustworkx_core::petgraph::unionfind::UnionFind;
use rustworkx_core::petgraph::visit::{
    EdgeIndexable, IntoEdgeReferences, IntoNodeReferences, NodeFiltered, NodeIndexable,
};
use rustworkx_core::petgraph::Incoming;
use rustworkx_core::traversal::{
    ancestors as core_ancestors, bfs_successors as core_bfs_successors,
    descendants as core_descendants,
};

use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};
use std::convert::Infallible;
use std::f64::consts::PI;

#[cfg(feature = "cache_pygates")]
use std::cell::OnceCell;

static CONTROL_FLOW_OP_NAMES: [&str; 4] = ["for_loop", "while_loop", "if_else", "switch_case"];
static SEMANTIC_EQ_SYMMETRIC: [&str; 4] = ["barrier", "swap", "break_loop", "continue_loop"];

#[derive(Clone, Debug)]
pub enum NodeType {
    QubitIn(Qubit),
    QubitOut(Qubit),
    ClbitIn(Clbit),
    ClbitOut(Clbit),
    VarIn(PyObject),
    VarOut(PyObject),
    Operation(PackedInstruction),
}

#[derive(Clone, Debug)]
pub enum Wire {
    Qubit(Qubit),
    Clbit(Clbit),
    Var(PyObject),
}

impl PartialEq for Wire {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Wire::Qubit(q1), Wire::Qubit(q2)) => q1 == q2,
            (Wire::Clbit(c1), Wire::Clbit(c2)) => c1 == c2,
            (Wire::Var(v1), Wire::Var(v2)) => {
                v1.is(v2) || Python::with_gil(|py| v1.bind(py).eq(v2).unwrap())
            }
            _ => false,
        }
    }
}

impl Eq for Wire {}

impl Hash for Wire {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Qubit(qubit) => qubit.hash(state),
            Self::Clbit(clbit) => clbit.hash(state),
            Self::Var(var) => Python::with_gil(|py| var.bind(py).hash().unwrap().hash(state)),
        }
    }
}

impl Wire {
    fn to_pickle(&self, py: Python) -> PyObject {
        match self {
            Self::Qubit(bit) => (0, bit.0.into_py(py)).into_py(py),
            Self::Clbit(bit) => (1, bit.0.into_py(py)).into_py(py),
            Self::Var(var) => (2, var.clone_ref(py)).into_py(py),
        }
    }

    fn from_pickle(b: &Bound<PyAny>) -> PyResult<Self> {
        let tuple: Bound<PyTuple> = b.extract()?;
        let wire_type: usize = tuple.get_item(0)?.extract()?;
        if wire_type == 0 {
            Ok(Self::Qubit(Qubit(tuple.get_item(1)?.extract()?)))
        } else if wire_type == 1 {
            Ok(Self::Clbit(Clbit(tuple.get_item(1)?.extract()?)))
        } else if wire_type == 2 {
            Ok(Self::Var(tuple.get_item(1)?.unbind()))
        } else {
            Err(PyTypeError::new_err("Invalid wire type"))
        }
    }
}

// TODO: Remove me.
// This is a temporary map type used to store a mapping of
// Var to NodeIndex to hold us over until Var is ported to
// Rust. Currently, we need this because PyObject cannot be
// used as the key to an IndexMap.
//
// Once we've got Var ported, Wire should also become Hash + Eq
// and we can consider combining input/output nodes maps.
#[derive(Clone, Debug)]
struct _VarIndexMap {
    dict: Py<PyDict>,
}

impl _VarIndexMap {
    pub fn new(py: Python) -> Self {
        Self {
            dict: PyDict::new_bound(py).unbind(),
        }
    }

    pub fn keys(&self, py: Python) -> impl Iterator<Item = PyObject> {
        self.dict
            .bind(py)
            .keys()
            .into_iter()
            .map(|k| k.unbind())
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn contains_key(&self, py: Python, key: &PyObject) -> bool {
        self.dict.bind(py).contains(key).unwrap()
    }

    pub fn get(&self, py: Python, key: &PyObject) -> Option<NodeIndex> {
        self.dict
            .bind(py)
            .get_item(key)
            .unwrap()
            .map(|v| NodeIndex::new(v.extract().unwrap()))
    }

    pub fn insert(&mut self, py: Python, key: PyObject, value: NodeIndex) {
        self.dict
            .bind(py)
            .set_item(key, value.index().into_py(py))
            .unwrap()
    }

    pub fn remove(&mut self, py: Python, key: &PyObject) -> Option<NodeIndex> {
        let bound_dict = self.dict.bind(py);
        let res = bound_dict
            .get_item(key.clone_ref(py))
            .unwrap()
            .map(|v| NodeIndex::new(v.extract().unwrap()));
        let _del_result = bound_dict.del_item(key);
        res
    }
    pub fn values<'py>(&self, py: Python<'py>) -> impl Iterator<Item = NodeIndex> + 'py {
        let values = self.dict.bind(py).values();
        values.iter().map(|x| NodeIndex::new(x.extract().unwrap()))
    }

    pub fn iter<'py>(&self, py: Python<'py>) -> impl Iterator<Item = (PyObject, NodeIndex)> + 'py {
        self.dict
            .bind(py)
            .iter()
            .map(|(var, index)| (var.unbind(), NodeIndex::new(index.extract().unwrap())))
    }
}

/// Quantum circuit as a directed acyclic graph.
///
/// There are 3 types of nodes in the graph: inputs, outputs, and operations.
/// The nodes are connected by directed edges that correspond to qubits and
/// bits.
#[pyclass(module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct DAGCircuit {
    /// Circuit name.  Generally, this corresponds to the name
    /// of the QuantumCircuit from which the DAG was generated.
    #[pyo3(get, set)]
    name: Option<PyObject>,
    /// Circuit metadata
    #[pyo3(get, set)]
    metadata: Option<PyObject>,

    calibrations: HashMap<String, Py<PyDict>>,

    pub dag: StableDiGraph<NodeType, Wire>,

    #[pyo3(get)]
    qregs: Py<PyDict>,
    #[pyo3(get)]
    cregs: Py<PyDict>,

    /// The cache used to intern instruction qargs.
    qargs_interner: Interner<[Qubit]>,
    /// The cache used to intern instruction cargs.
    cargs_interner: Interner<[Clbit]>,
    /// Qubits registered in the circuit.
    pub qubits: BitData<Qubit>,
    /// Clbits registered in the circuit.
    pub clbits: BitData<Clbit>,
    /// Global phase.
    global_phase: Param,
    /// Duration.
    #[pyo3(get, set)]
    duration: Option<PyObject>,
    /// Unit of duration.
    #[pyo3(get, set)]
    unit: String,

    // Note: these are tracked separately from `qubits` and `clbits`
    // because it's not yet clear if the Rust concept of a native Qubit
    // and Clbit should correspond directly to the numerical Python
    // index that users see in the Python API.
    /// The index locations of bits, and their positions within
    /// registers.
    qubit_locations: Py<PyDict>,
    clbit_locations: Py<PyDict>,

    /// Map from qubit to input and output nodes of the graph.
    qubit_io_map: Vec<[NodeIndex; 2]>,

    /// Map from clbit to input and output nodes of the graph.
    clbit_io_map: Vec<[NodeIndex; 2]>,

    // TODO: use IndexMap<Wire, NodeIndex> once Var is ported to Rust
    /// Map from var to input nodes of the graph.
    var_input_map: _VarIndexMap,
    /// Map from var to output nodes of the graph.
    var_output_map: _VarIndexMap,

    /// Operation kind to count
    op_names: IndexMap<String, usize, RandomState>,

    // Python modules we need to frequently access (for now).
    control_flow_module: PyControlFlowModule,
    vars_info: HashMap<String, DAGVarInfo>,
    vars_by_type: [Py<PySet>; 3],
}

#[derive(Clone, Debug)]
struct PyControlFlowModule {
    condition_resources: Py<PyAny>,
    node_resources: Py<PyAny>,
}

#[derive(Clone, Debug)]
struct PyLegacyResources {
    clbits: Py<PyTuple>,
    cregs: Py<PyTuple>,
}

impl PyControlFlowModule {
    fn new(py: Python) -> PyResult<Self> {
        let module = PyModule::import_bound(py, "qiskit.circuit.controlflow")?;
        Ok(PyControlFlowModule {
            condition_resources: module.getattr("condition_resources")?.unbind(),
            node_resources: module.getattr("node_resources")?.unbind(),
        })
    }

    fn condition_resources(&self, condition: &Bound<PyAny>) -> PyResult<PyLegacyResources> {
        let res = self
            .condition_resources
            .bind(condition.py())
            .call1((condition,))?;
        Ok(PyLegacyResources {
            clbits: res.getattr("clbits")?.downcast_into_exact()?.unbind(),
            cregs: res.getattr("cregs")?.downcast_into_exact()?.unbind(),
        })
    }

    fn node_resources(&self, node: &Bound<PyAny>) -> PyResult<PyLegacyResources> {
        let res = self.node_resources.bind(node.py()).call1((node,))?;
        Ok(PyLegacyResources {
            clbits: res.getattr("clbits")?.downcast_into_exact()?.unbind(),
            cregs: res.getattr("cregs")?.downcast_into_exact()?.unbind(),
        })
    }
}

struct PyVariableMapper {
    mapper: Py<PyAny>,
}

impl PyVariableMapper {
    fn new(
        py: Python,
        target_cregs: Bound<PyAny>,
        bit_map: Option<Bound<PyDict>>,
        var_map: Option<Bound<PyDict>>,
        add_register: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let kwargs: HashMap<&str, Option<Py<PyAny>>> =
            HashMap::from_iter([("add_register", add_register)]);
        Ok(PyVariableMapper {
            mapper: imports::VARIABLE_MAPPER
                .get_bound(py)
                .call(
                    (target_cregs, bit_map, var_map),
                    Some(&kwargs.into_py_dict_bound(py)),
                )?
                .unbind(),
        })
    }

    fn map_condition<'py>(
        &self,
        condition: &Bound<'py, PyAny>,
        allow_reorder: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = condition.py();
        let kwargs: HashMap<&str, Py<PyAny>> =
            HashMap::from_iter([("allow_reorder", allow_reorder.into_py(py))]);
        self.mapper.bind(py).call_method(
            intern!(py, "map_condition"),
            (condition,),
            Some(&kwargs.into_py_dict_bound(py)),
        )
    }

    fn map_target<'py>(&self, target: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = target.py();
        self.mapper
            .bind(py)
            .call_method1(intern!(py, "map_target"), (target,))
    }
}

impl IntoPy<Py<PyAny>> for PyVariableMapper {
    fn into_py(self, _py: Python<'_>) -> Py<PyAny> {
        self.mapper
    }
}

#[pyfunction]
fn reject_new_register(reg: &Bound<PyAny>) -> PyResult<()> {
    Err(DAGCircuitError::new_err(format!(
        "No register with '{:?}' to map this expression onto.",
        reg.getattr("bits")?
    )))
}

#[pyclass(module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
struct BitLocations {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    registers: Py<PyList>,
}

#[derive(Copy, Clone, Debug)]
enum DAGVarType {
    Input = 0,
    Capture = 1,
    Declare = 2,
}

#[derive(Clone, Debug)]
struct DAGVarInfo {
    var: PyObject,
    type_: DAGVarType,
    in_node: NodeIndex,
    out_node: NodeIndex,
}

#[pymethods]
impl DAGCircuit {
    #[new]
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(DAGCircuit {
            name: None,
            metadata: Some(PyDict::new_bound(py).unbind().into()),
            calibrations: HashMap::new(),
            dag: StableDiGraph::default(),
            qregs: PyDict::new_bound(py).unbind(),
            cregs: PyDict::new_bound(py).unbind(),
            qargs_interner: Interner::new(),
            cargs_interner: Interner::new(),
            qubits: BitData::new(py, "qubits".to_string()),
            clbits: BitData::new(py, "clbits".to_string()),
            global_phase: Param::Float(0.),
            duration: None,
            unit: "dt".to_string(),
            qubit_locations: PyDict::new_bound(py).unbind(),
            clbit_locations: PyDict::new_bound(py).unbind(),
            qubit_io_map: Vec::new(),
            clbit_io_map: Vec::new(),
            var_input_map: _VarIndexMap::new(py),
            var_output_map: _VarIndexMap::new(py),
            op_names: IndexMap::default(),
            control_flow_module: PyControlFlowModule::new(py)?,
            vars_info: HashMap::new(),
            vars_by_type: [
                PySet::empty_bound(py)?.unbind(),
                PySet::empty_bound(py)?.unbind(),
                PySet::empty_bound(py)?.unbind(),
            ],
        })
    }

    #[getter]
    fn input_map(&self, py: Python) -> PyResult<Py<PyDict>> {
        let out_dict = PyDict::new_bound(py);
        for (qubit, indices) in self
            .qubit_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Qubit(idx as u32), indices))
        {
            out_dict.set_item(
                self.qubits.get(qubit).unwrap().clone_ref(py),
                self.get_node(py, indices[0])?,
            )?;
        }
        for (clbit, indices) in self
            .clbit_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Clbit(idx as u32), indices))
        {
            out_dict.set_item(
                self.clbits.get(clbit).unwrap().clone_ref(py),
                self.get_node(py, indices[0])?,
            )?;
        }
        for (var, index) in self.var_input_map.dict.bind(py).iter() {
            out_dict.set_item(
                var,
                self.get_node(py, NodeIndex::new(index.extract::<usize>()?))?,
            )?;
        }
        Ok(out_dict.unbind())
    }

    #[getter]
    fn output_map(&self, py: Python) -> PyResult<Py<PyDict>> {
        let out_dict = PyDict::new_bound(py);
        for (qubit, indices) in self
            .qubit_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Qubit(idx as u32), indices))
        {
            out_dict.set_item(
                self.qubits.get(qubit).unwrap().clone_ref(py),
                self.get_node(py, indices[1])?,
            )?;
        }
        for (clbit, indices) in self
            .clbit_io_map
            .iter()
            .enumerate()
            .map(|(idx, indices)| (Clbit(idx as u32), indices))
        {
            out_dict.set_item(
                self.clbits.get(clbit).unwrap().clone_ref(py),
                self.get_node(py, indices[1])?,
            )?;
        }
        for (var, index) in self.var_output_map.dict.bind(py).iter() {
            out_dict.set_item(
                var,
                self.get_node(py, NodeIndex::new(index.extract::<usize>()?))?,
            )?;
        }
        Ok(out_dict.unbind())
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyDict>> {
        let out_dict = PyDict::new_bound(py);
        out_dict.set_item("name", self.name.as_ref().map(|x| x.clone_ref(py)))?;
        out_dict.set_item("metadata", self.metadata.as_ref().map(|x| x.clone_ref(py)))?;
        out_dict.set_item("calibrations", self.calibrations.clone())?;
        out_dict.set_item("qregs", self.qregs.clone_ref(py))?;
        out_dict.set_item("cregs", self.cregs.clone_ref(py))?;
        out_dict.set_item("global_phase", self.global_phase.clone())?;
        out_dict.set_item(
            "qubit_io_map",
            self.qubit_io_map
                .iter()
                .enumerate()
                .map(|(k, v)| (k, [v[0].index(), v[1].index()]))
                .collect::<IndexMap<_, _, RandomState>>(),
        )?;
        out_dict.set_item(
            "clbit_io_map",
            self.clbit_io_map
                .iter()
                .enumerate()
                .map(|(k, v)| (k, [v[0].index(), v[1].index()]))
                .collect::<IndexMap<_, _, RandomState>>(),
        )?;
        out_dict.set_item("var_input_map", self.var_input_map.dict.clone_ref(py))?;
        out_dict.set_item("var_output_map", self.var_output_map.dict.clone_ref(py))?;
        out_dict.set_item("op_name", self.op_names.clone())?;
        out_dict.set_item(
            "vars_info",
            self.vars_info
                .iter()
                .map(|(k, v)| {
                    (
                        k,
                        (
                            v.var.clone_ref(py),
                            v.type_ as u8,
                            v.in_node.index(),
                            v.out_node.index(),
                        ),
                    )
                })
                .collect::<HashMap<_, _>>(),
        )?;
        out_dict.set_item("vars_by_type", self.vars_by_type.clone())?;
        out_dict.set_item("qubits", self.qubits.bits())?;
        out_dict.set_item("clbits", self.clbits.bits())?;
        let mut nodes: Vec<PyObject> = Vec::with_capacity(self.dag.node_count());
        for node_idx in self.dag.node_indices() {
            let node_data = self.get_node(py, node_idx)?;
            nodes.push((node_idx.index(), node_data).to_object(py));
        }
        out_dict.set_item("nodes", nodes)?;
        out_dict.set_item(
            "nodes_removed",
            self.dag.node_count() != self.dag.node_bound(),
        )?;
        let mut edges: Vec<PyObject> = Vec::with_capacity(self.dag.edge_bound());
        // edges are saved with none (deleted edges) instead of their index to save space
        for i in 0..self.dag.edge_bound() {
            let idx = EdgeIndex::new(i);
            let edge = match self.dag.edge_weight(idx) {
                Some(edge_w) => {
                    let endpoints = self.dag.edge_endpoints(idx).unwrap();
                    (
                        endpoints.0.index(),
                        endpoints.1.index(),
                        edge_w.clone().to_pickle(py),
                    )
                        .to_object(py)
                }
                None => py.None(),
            };
            edges.push(edge);
        }
        out_dict.set_item("edges", edges)?;
        Ok(out_dict.unbind())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let dict_state = state.downcast_bound::<PyDict>(py)?;
        self.name = dict_state.get_item("name")?.unwrap().extract()?;
        self.metadata = dict_state.get_item("metadata")?.unwrap().extract()?;
        self.calibrations = dict_state.get_item("calibrations")?.unwrap().extract()?;
        self.qregs = dict_state.get_item("qregs")?.unwrap().extract()?;
        self.cregs = dict_state.get_item("cregs")?.unwrap().extract()?;
        self.global_phase = dict_state.get_item("global_phase")?.unwrap().extract()?;
        self.op_names = dict_state.get_item("op_name")?.unwrap().extract()?;
        self.var_input_map = _VarIndexMap {
            dict: dict_state.get_item("var_input_map")?.unwrap().extract()?,
        };
        self.var_output_map = _VarIndexMap {
            dict: dict_state.get_item("var_output_map")?.unwrap().extract()?,
        };
        self.vars_by_type = dict_state.get_item("vars_by_type")?.unwrap().extract()?;
        let binding = dict_state.get_item("vars_info")?.unwrap();
        let vars_info_raw = binding.downcast::<PyDict>().unwrap();
        self.vars_info = HashMap::with_capacity(vars_info_raw.len());
        for (key, value) in vars_info_raw.iter() {
            let val_tuple = value.downcast::<PyTuple>()?;
            let info = DAGVarInfo {
                var: val_tuple.get_item(0)?.unbind(),
                type_: match val_tuple.get_item(1)?.extract::<u8>()? {
                    0 => DAGVarType::Input,
                    1 => DAGVarType::Capture,
                    2 => DAGVarType::Declare,
                    _ => return Err(PyValueError::new_err("Invalid var type")),
                },
                in_node: NodeIndex::new(val_tuple.get_item(2)?.extract()?),
                out_node: NodeIndex::new(val_tuple.get_item(3)?.extract()?),
            };
            self.vars_info.insert(key.extract()?, info);
        }

        let binding = dict_state.get_item("qubits")?.unwrap();
        let qubits_raw = binding.downcast::<PyList>().unwrap();
        for bit in qubits_raw.iter() {
            self.qubits.add(py, &bit, false)?;
        }
        let binding = dict_state.get_item("clbits")?.unwrap();
        let clbits_raw = binding.downcast::<PyList>().unwrap();
        for bit in clbits_raw.iter() {
            self.clbits.add(py, &bit, false)?;
        }
        let binding = dict_state.get_item("qubit_io_map")?.unwrap();
        let qubit_index_map_raw = binding.downcast::<PyDict>().unwrap();
        self.qubit_io_map = Vec::with_capacity(qubit_index_map_raw.len());
        for (_k, v) in qubit_index_map_raw.iter() {
            let indices: [usize; 2] = v.extract()?;
            self.qubit_io_map
                .push([NodeIndex::new(indices[0]), NodeIndex::new(indices[1])]);
        }
        let binding = dict_state.get_item("clbit_io_map")?.unwrap();
        let clbit_index_map_raw = binding.downcast::<PyDict>().unwrap();
        self.clbit_io_map = Vec::with_capacity(clbit_index_map_raw.len());

        for (_k, v) in clbit_index_map_raw.iter() {
            let indices: [usize; 2] = v.extract()?;
            self.clbit_io_map
                .push([NodeIndex::new(indices[0]), NodeIndex::new(indices[1])]);
        }
        // Rebuild Graph preserving index holes:
        let binding = dict_state.get_item("nodes")?.unwrap();
        let nodes_lst = binding.downcast::<PyList>()?;
        let binding = dict_state.get_item("edges")?.unwrap();
        let edges_lst = binding.downcast::<PyList>()?;
        let node_removed: bool = dict_state.get_item("nodes_removed")?.unwrap().extract()?;
        self.dag = StableDiGraph::default();
        if !node_removed {
            for item in nodes_lst.iter() {
                let node_w = item.downcast::<PyTuple>().unwrap().get_item(1).unwrap();
                let weight = self.pack_into(py, &node_w)?;
                self.dag.add_node(weight);
            }
        } else if nodes_lst.len() == 1 {
            // graph has only one node, handle logic here to save one if in the loop later
            let binding = nodes_lst.get_item(0).unwrap();
            let item = binding.downcast::<PyTuple>().unwrap();
            let node_idx: usize = item.get_item(0).unwrap().extract().unwrap();
            let node_w = item.get_item(1).unwrap();

            for _i in 0..node_idx {
                self.dag.add_node(NodeType::QubitIn(Qubit(u32::MAX)));
            }
            let weight = self.pack_into(py, &node_w)?;
            self.dag.add_node(weight);
            for i in 0..node_idx {
                self.dag.remove_node(NodeIndex::new(i));
            }
        } else {
            let binding = nodes_lst.get_item(nodes_lst.len() - 1).unwrap();
            let last_item = binding.downcast::<PyTuple>().unwrap();

            // list of temporary nodes that will be removed later to re-create holes
            let node_bound_1: usize = last_item.get_item(0).unwrap().extract().unwrap();
            let mut tmp_nodes: Vec<NodeIndex> =
                Vec::with_capacity(node_bound_1 + 1 - nodes_lst.len());

            for item in nodes_lst {
                let item = item.downcast::<PyTuple>().unwrap();
                let next_index: usize = item.get_item(0).unwrap().extract().unwrap();
                let weight: PyObject = item.get_item(1).unwrap().extract().unwrap();
                while next_index > self.dag.node_bound() {
                    // node does not exist
                    let tmp_node = self.dag.add_node(NodeType::QubitIn(Qubit(u32::MAX)));
                    tmp_nodes.push(tmp_node);
                }
                // add node to the graph, and update the next available node index
                let weight = self.pack_into(py, weight.bind(py))?;
                self.dag.add_node(weight);
            }
            // Remove any temporary nodes we added
            for tmp_node in tmp_nodes {
                self.dag.remove_node(tmp_node);
            }
        }

        // to ensure O(1) on edge deletion, use a temporary node to store missing edges
        let tmp_node = self.dag.add_node(NodeType::QubitIn(Qubit(u32::MAX)));

        for item in edges_lst {
            if item.is_none() {
                // add a temporary edge that will be deleted later to re-create the hole
                self.dag
                    .add_edge(tmp_node, tmp_node, Wire::Qubit(Qubit(u32::MAX)));
            } else {
                let triple = item.downcast::<PyTuple>().unwrap();
                let edge_p: usize = triple.get_item(0).unwrap().extract().unwrap();
                let edge_c: usize = triple.get_item(1).unwrap().extract().unwrap();
                let edge_w = Wire::from_pickle(&triple.get_item(2).unwrap())?;
                self.dag
                    .add_edge(NodeIndex::new(edge_p), NodeIndex::new(edge_c), edge_w);
            }
        }
        self.dag.remove_node(tmp_node);
        Ok(())
    }

    /// Returns the current sequence of registered :class:`.Qubit` instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`DAGCircuit` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Qubit`): The current sequence of registered qubits.
    #[getter]
    pub fn qubits(&self, py: Python<'_>) -> Py<PyList> {
        self.qubits.cached().clone_ref(py)
    }

    /// Returns the current sequence of registered :class:`.Clbit`
    /// instances as a list.
    ///
    /// .. warning::
    ///
    ///     Do not modify this list yourself.  It will invalidate the :class:`DAGCircuit` data
    ///     structures.
    ///
    /// Returns:
    ///     list(:class:`.Clbit`): The current sequence of registered clbits.
    #[getter]
    pub fn clbits(&self, py: Python<'_>) -> Py<PyList> {
        self.clbits.cached().clone_ref(py)
    }

    /// Return a list of the wires in order.
    #[getter]
    fn get_wires(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let wires: Vec<&PyObject> = self
            .qubits
            .bits()
            .iter()
            .chain(self.clbits.bits().iter())
            .collect();
        let out_list = PyList::new_bound(py, wires);
        for var_type_set in &self.vars_by_type {
            for var in var_type_set.bind(py).iter() {
                out_list.append(var)?;
            }
        }
        Ok(out_list.unbind())
    }

    /// Returns the number of nodes in the dag.
    #[getter]
    fn get_node_counter(&self) -> usize {
        self.dag.node_count()
    }

    /// Return the global phase of the circuit.
    #[getter]
    fn get_global_phase(&self) -> Param {
        self.global_phase.clone()
    }

    /// Set the global phase of the circuit.
    ///
    /// Args:
    ///     angle (float, :class:`.ParameterExpression`): The phase angle.
    #[setter]
    fn set_global_phase(&mut self, angle: Param) -> PyResult<()> {
        match angle {
            Param::Float(angle) => {
                self.global_phase = Param::Float(angle.rem_euclid(2. * PI));
            }
            Param::ParameterExpression(angle) => {
                self.global_phase = Param::ParameterExpression(angle);
            }
            Param::Obj(_) => return Err(PyTypeError::new_err("Invalid type for global phase")),
        }
        Ok(())
    }

    /// Return calibration dictionary.
    ///
    /// The custom pulse definition of a given gate is of the form
    ///    {'gate_name': {(qubits, params): schedule}}
    #[getter]
    fn get_calibrations(&self) -> HashMap<String, Py<PyDict>> {
        self.calibrations.clone()
    }

    /// Set the circuit calibration data from a dictionary of calibration definition.
    ///
    ///  Args:
    ///      calibrations (dict): A dictionary of input in the format
    ///          {'gate_name': {(qubits, gate_params): schedule}}
    #[setter]
    fn set_calibrations(&mut self, calibrations: HashMap<String, Py<PyDict>>) {
        self.calibrations = calibrations;
    }

    /// Register a low-level, custom pulse definition for the given gate.
    ///
    /// Args:
    ///     gate (Union[Gate, str]): Gate information.
    ///     qubits (Union[int, Tuple[int]]): List of qubits to be measured.
    ///     schedule (Schedule): Schedule information.
    ///     params (Optional[List[Union[float, Parameter]]]): A list of parameters.
    ///
    /// Raises:
    ///     Exception: if the gate is of type string and params is None.
    fn add_calibration<'py>(
        &mut self,
        py: Python<'py>,
        mut gate: Bound<'py, PyAny>,
        qubits: Bound<'py, PyAny>,
        schedule: Py<PyAny>,
        mut params: Option<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        if gate.is_instance(imports::GATE.get_bound(py))? {
            params = Some(gate.getattr(intern!(py, "params"))?);
            gate = gate.getattr(intern!(py, "name"))?;
        }

        let params_tuple = if let Some(operands) = params {
            let add_calibration = PyModule::from_code_bound(
                py,
                r#"
import numpy as np

def _format(operand):
    try:
        # Using float/complex value as a dict key is not good idea.
        # This makes the mapping quite sensitive to the rounding error.
        # However, the mechanism is already tied to the execution model (i.e. pulse gate)
        # and we cannot easily update this rule.
        # The same logic exists in QuantumCircuit.add_calibration.
        evaluated = complex(operand)
        if np.isreal(evaluated):
            evaluated = float(evaluated.real)
            if evaluated.is_integer():
                evaluated = int(evaluated)
        return evaluated
    except TypeError:
        # Unassigned parameter
        return operand
    "#,
                "add_calibration.py",
                "add_calibration",
            )?;

            let format = add_calibration.getattr("_format")?;
            let mapped: PyResult<Vec<_>> = operands.iter()?.map(|p| format.call1((p?,))).collect();
            PyTuple::new_bound(py, mapped?).into_any()
        } else {
            PyTuple::empty_bound(py).into_any()
        };

        let calibrations = self
            .calibrations
            .entry(gate.extract()?)
            .or_insert_with(|| PyDict::new_bound(py).unbind())
            .bind(py);

        let qubits = if let Ok(qubits) = qubits.downcast::<PySequence>() {
            qubits.to_tuple()?.into_any()
        } else {
            PyTuple::new_bound(py, [qubits]).into_any()
        };
        let key = PyTuple::new_bound(py, &[qubits.unbind(), params_tuple.into_any().unbind()]);
        calibrations.set_item(key, schedule)?;
        Ok(())
    }

    /// Return True if the dag has a calibration defined for the node operation. In this
    /// case, the operation does not need to be translated to the device basis.
    fn has_calibration_for(&self, py: Python, node: PyRef<DAGOpNode>) -> PyResult<bool> {
        if !self
            .calibrations
            .contains_key(node.instruction.operation.name())
        {
            return Ok(false);
        }
        let mut params = Vec::new();
        for p in &node.instruction.params {
            if let Param::ParameterExpression(exp) = p {
                let exp = exp.bind(py);
                if !exp.getattr(intern!(py, "parameters"))?.is_truthy()? {
                    let as_py_float = exp.call_method0(intern!(py, "__float__"))?;
                    params.push(as_py_float.unbind());
                    continue;
                }
            }
            params.push(p.to_object(py));
        }
        let qubits: Vec<BitType> = self
            .qubits
            .map_bits(node.instruction.qubits.bind(py).iter())?
            .map(|bit| bit.0)
            .collect();
        let qubits = PyTuple::new_bound(py, qubits);
        let params = PyTuple::new_bound(py, params);
        self.calibrations[node.instruction.operation.name()]
            .bind(py)
            .contains((qubits, params).to_object(py))
    }

    /// Remove all operation nodes with the given name.
    fn remove_all_ops_named(&mut self, opname: &str) {
        let mut to_remove = Vec::new();
        for (id, weight) in self.dag.node_references() {
            if let NodeType::Operation(packed) = &weight {
                if opname == packed.op.name() {
                    to_remove.push(id);
                }
            }
        }
        for node in to_remove {
            self.remove_op_node(node);
        }
    }

    /// Add individual qubit wires.
    fn add_qubits(&mut self, py: Python, qubits: Vec<Bound<PyAny>>) -> PyResult<()> {
        for bit in qubits.iter() {
            if !bit.is_instance(imports::QUBIT.get_bound(py))? {
                return Err(DAGCircuitError::new_err("not a Qubit instance."));
            }

            if self.qubits.find(bit).is_some() {
                return Err(DAGCircuitError::new_err(format!(
                    "duplicate qubits {}",
                    bit
                )));
            }
        }

        for bit in qubits.iter() {
            self.add_qubit_unchecked(py, bit)?;
        }
        Ok(())
    }

    /// Add individual qubit wires.
    fn add_clbits(&mut self, py: Python, clbits: Vec<Bound<PyAny>>) -> PyResult<()> {
        for bit in clbits.iter() {
            if !bit.is_instance(imports::CLBIT.get_bound(py))? {
                return Err(DAGCircuitError::new_err("not a Clbit instance."));
            }

            if self.clbits.find(bit).is_some() {
                return Err(DAGCircuitError::new_err(format!(
                    "duplicate clbits {}",
                    bit
                )));
            }
        }

        for bit in clbits.iter() {
            self.add_clbit_unchecked(py, bit)?;
        }
        Ok(())
    }

    /// Add all wires in a quantum register.
    fn add_qreg(&mut self, py: Python, qreg: &Bound<PyAny>) -> PyResult<()> {
        if !qreg.is_instance(imports::QUANTUM_REGISTER.get_bound(py))? {
            return Err(DAGCircuitError::new_err("not a QuantumRegister instance."));
        }

        let register_name = qreg.getattr(intern!(py, "name"))?;
        if self.qregs.bind(py).contains(&register_name)? {
            return Err(DAGCircuitError::new_err(format!(
                "duplicate register {}",
                register_name
            )));
        }
        self.qregs.bind(py).set_item(&register_name, qreg)?;

        for (index, bit) in qreg.iter()?.enumerate() {
            let bit = bit?;
            if self.qubits.find(&bit).is_none() {
                self.add_qubit_unchecked(py, &bit)?;
            }
            let locations: PyRef<BitLocations> = self
                .qubit_locations
                .bind(py)
                .get_item(&bit)?
                .unwrap()
                .extract()?;
            locations.registers.bind(py).append((qreg, index))?;
        }
        Ok(())
    }

    /// Add all wires in a classical register.
    fn add_creg(&mut self, py: Python, creg: &Bound<PyAny>) -> PyResult<()> {
        if !creg.is_instance(imports::CLASSICAL_REGISTER.get_bound(py))? {
            return Err(DAGCircuitError::new_err(
                "not a ClassicalRegister instance.",
            ));
        }

        let register_name = creg.getattr(intern!(py, "name"))?;
        if self.cregs.bind(py).contains(&register_name)? {
            return Err(DAGCircuitError::new_err(format!(
                "duplicate register {}",
                register_name
            )));
        }
        self.cregs.bind(py).set_item(register_name, creg)?;

        for (index, bit) in creg.iter()?.enumerate() {
            let bit = bit?;
            if self.clbits.find(&bit).is_none() {
                self.add_clbit_unchecked(py, &bit)?;
            }
            let locations: PyRef<BitLocations> = self
                .clbit_locations
                .bind(py)
                .get_item(&bit)?
                .unwrap()
                .extract()?;
            locations.registers.bind(py).append((creg, index))?;
        }
        Ok(())
    }

    /// Finds locations in the circuit, by mapping the Qubit and Clbit to positional index
    /// BitLocations is defined as: BitLocations = namedtuple("BitLocations", ("index", "registers"))
    ///
    /// Args:
    ///     bit (Bit): The bit to locate.
    ///
    /// Returns:
    ///     namedtuple(int, List[Tuple(Register, int)]): A 2-tuple. The first element (``index``)
    ///         contains the index at which the ``Bit`` can be found (in either
    ///         :obj:`~DAGCircuit.qubits`, :obj:`~DAGCircuit.clbits`, depending on its
    ///         type). The second element (``registers``) is a list of ``(register, index)``
    ///         pairs with an entry for each :obj:`~Register` in the circuit which contains the
    ///         :obj:`~Bit` (and the index in the :obj:`~Register` at which it can be found).
    ///
    ///   Raises:
    ///     DAGCircuitError: If the supplied :obj:`~Bit` was of an unknown type.
    ///     DAGCircuitError: If the supplied :obj:`~Bit` could not be found on the circuit.
    fn find_bit<'py>(&self, py: Python<'py>, bit: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if bit.is_instance(imports::QUBIT.get_bound(py))? {
            return self.qubit_locations.bind(py).get_item(bit)?.ok_or_else(|| {
                DAGCircuitError::new_err(format!(
                    "Could not locate provided bit: {}. Has it been added to the DAGCircuit?",
                    bit
                ))
            });
        }

        if bit.is_instance(imports::CLBIT.get_bound(py))? {
            return self.clbit_locations.bind(py).get_item(bit)?.ok_or_else(|| {
                DAGCircuitError::new_err(format!(
                    "Could not locate provided bit: {}. Has it been added to the DAGCircuit?",
                    bit
                ))
            });
        }

        Err(DAGCircuitError::new_err(format!(
            "Could not locate bit of unknown type: {}",
            bit.get_type()
        )))
    }

    /// Remove classical bits from the circuit. All bits MUST be idle.
    /// Any registers with references to at least one of the specified bits will
    /// also be removed.
    ///
    /// .. warning::
    ///     This method is rather slow, since it must iterate over the entire
    ///     DAG to fix-up bit indices.
    ///
    /// Args:
    ///     clbits (List[Clbit]): The bits to remove.
    ///
    /// Raises:
    ///     DAGCircuitError: a clbit is not a :obj:`.Clbit`, is not in the circuit,
    ///         or is not idle.
    #[pyo3(signature = (*clbits))]
    fn remove_clbits(&mut self, py: Python, clbits: &Bound<PyTuple>) -> PyResult<()> {
        let mut non_bits = Vec::new();
        for bit in clbits.iter() {
            if !bit.is_instance(imports::CLBIT.get_bound(py))? {
                non_bits.push(bit);
            }
        }
        if !non_bits.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "clbits not of type Clbit: {:?}",
                non_bits
            )));
        }

        let bit_iter = match self.clbits.map_bits(clbits.iter()) {
            Ok(bit_iter) => bit_iter,
            Err(_) => {
                return Err(DAGCircuitError::new_err(format!(
                    "clbits not in circuit: {:?}",
                    clbits
                )))
            }
        };
        let clbits: HashSet<Clbit> = bit_iter.collect();
        let mut busy_bits = Vec::new();
        for bit in clbits.iter() {
            if !self.is_wire_idle(py, &Wire::Clbit(*bit))? {
                busy_bits.push(self.clbits.get(*bit).unwrap());
            }
        }

        if !busy_bits.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "clbits not idle: {:?}",
                busy_bits
            )));
        }

        // Remove any references to bits.
        let mut cregs_to_remove = Vec::new();
        for creg in self.cregs.bind(py).values() {
            for bit in creg.iter()? {
                let bit = bit?;
                if clbits.contains(&self.clbits.find(&bit).unwrap()) {
                    cregs_to_remove.push(creg);
                    break;
                }
            }
        }
        self.remove_cregs(py, &PyTuple::new_bound(py, cregs_to_remove))?;

        // Remove DAG in/out nodes etc.
        for bit in clbits.iter() {
            self.remove_idle_wire(py, Wire::Clbit(*bit))?;
        }

        // Copy the current clbit mapping so we can use it while remapping
        // wires used on edges and in operation cargs.
        let old_clbits = self.clbits.clone();

        // Remove the clbit indices, which will invalidate our mapping of Clbit to
        // Python bits throughout the entire DAG.
        self.clbits.remove_indices(py, clbits.clone())?;

        // Update input/output maps to use new Clbits.
        let io_mapping: HashMap<Clbit, [NodeIndex; 2]> = self
            .clbit_io_map
            .drain(..)
            .enumerate()
            .filter_map(|(k, v)| {
                let clbit = Clbit(k as u32);
                if clbits.contains(&clbit) {
                    None
                } else {
                    Some((
                        self.clbits
                            .find(old_clbits.get(Clbit(k as u32)).unwrap().bind(py))
                            .unwrap(),
                        v,
                    ))
                }
            })
            .collect();

        self.clbit_io_map = (0..io_mapping.len())
            .map(|idx| {
                let clbit = Clbit(idx as u32);
                io_mapping[&clbit]
            })
            .collect();

        // Update edges to use the new Clbits.
        for edge_weight in self.dag.edge_weights_mut() {
            if let Wire::Clbit(c) = edge_weight {
                *c = self
                    .clbits
                    .find(old_clbits.get(*c).unwrap().bind(py))
                    .unwrap();
            }
        }

        // Update operation cargs to use the new Clbits.
        for node_weight in self.dag.node_weights_mut() {
            match node_weight {
                NodeType::Operation(op) => {
                    let cargs = self.cargs_interner.get(op.clbits);
                    let carg_bits = old_clbits.map_indices(cargs).map(|b| b.bind(py).clone());
                    op.clbits = self
                        .cargs_interner
                        .insert_owned(self.clbits.map_bits(carg_bits)?.collect());
                }
                NodeType::ClbitIn(c) | NodeType::ClbitOut(c) => {
                    *c = self
                        .clbits
                        .find(old_clbits.get(*c).unwrap().bind(py))
                        .unwrap();
                }
                _ => (),
            }
        }

        // Update bit locations.
        let bit_locations = self.clbit_locations.bind(py);
        for (i, bit) in self.clbits.bits().iter().enumerate() {
            let raw_loc = bit_locations.get_item(bit)?.unwrap();
            let loc = raw_loc.downcast::<BitLocations>().unwrap();
            loc.borrow_mut().index = i;
            bit_locations.set_item(bit, loc)?;
        }
        Ok(())
    }

    /// Remove classical registers from the circuit, leaving underlying bits
    /// in place.
    ///
    /// Raises:
    ///     DAGCircuitError: a creg is not a ClassicalRegister, or is not in
    ///     the circuit.
    #[pyo3(signature = (*cregs))]
    fn remove_cregs(&mut self, py: Python, cregs: &Bound<PyTuple>) -> PyResult<()> {
        let mut non_regs = Vec::new();
        let mut unknown_regs = Vec::new();
        let self_bound_cregs = self.cregs.bind(py);
        for reg in cregs.iter() {
            if !reg.is_instance(imports::CLASSICAL_REGISTER.get_bound(py))? {
                non_regs.push(reg);
            } else if let Some(existing_creg) =
                self_bound_cregs.get_item(&reg.getattr(intern!(py, "name"))?)?
            {
                if !existing_creg.eq(&reg)? {
                    unknown_regs.push(reg);
                }
            } else {
                unknown_regs.push(reg);
            }
        }
        if !non_regs.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "cregs not of type ClassicalRegister: {:?}",
                non_regs
            )));
        }
        if !unknown_regs.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "cregs not in circuit: {:?}",
                unknown_regs
            )));
        }

        for creg in cregs {
            self.cregs
                .bind(py)
                .del_item(creg.getattr(intern!(py, "name"))?)?;
            for (i, bit) in creg.iter()?.enumerate() {
                let bit = bit?;
                let bit_position = self
                    .clbit_locations
                    .bind(py)
                    .get_item(bit)?
                    .unwrap()
                    .downcast_into_exact::<BitLocations>()?;
                bit_position
                    .borrow()
                    .registers
                    .bind(py)
                    .as_any()
                    .call_method1(intern!(py, "remove"), ((&creg, i),))?;
            }
        }
        Ok(())
    }

    /// Remove quantum bits from the circuit. All bits MUST be idle.
    /// Any registers with references to at least one of the specified bits will
    /// also be removed.
    ///
    /// .. warning::
    ///     This method is rather slow, since it must iterate over the entire
    ///     DAG to fix-up bit indices.
    ///
    /// Args:
    ///     qubits (List[~qiskit.circuit.Qubit]): The bits to remove.
    ///
    /// Raises:
    ///     DAGCircuitError: a qubit is not a :obj:`~.circuit.Qubit`, is not in the circuit,
    ///         or is not idle.
    #[pyo3(signature = (*qubits))]
    fn remove_qubits(&mut self, py: Python, qubits: &Bound<PyTuple>) -> PyResult<()> {
        let mut non_qbits = Vec::new();
        for bit in qubits.iter() {
            if !bit.is_instance(imports::QUBIT.get_bound(py))? {
                non_qbits.push(bit);
            }
        }
        if !non_qbits.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "qubits not of type Qubit: {:?}",
                non_qbits
            )));
        }

        let bit_iter = match self.qubits.map_bits(qubits.iter()) {
            Ok(bit_iter) => bit_iter,
            Err(_) => {
                return Err(DAGCircuitError::new_err(format!(
                    "qubits not in circuit: {:?}",
                    qubits
                )))
            }
        };
        let qubits: HashSet<Qubit> = bit_iter.collect();

        let mut busy_bits = Vec::new();
        for bit in qubits.iter() {
            if !self.is_wire_idle(py, &Wire::Qubit(*bit))? {
                busy_bits.push(self.qubits.get(*bit).unwrap());
            }
        }

        if !busy_bits.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "qubits not idle: {:?}",
                busy_bits
            )));
        }

        // Remove any references to bits.
        let mut qregs_to_remove = Vec::new();
        for qreg in self.qregs.bind(py).values() {
            for bit in qreg.iter()? {
                let bit = bit?;
                if qubits.contains(&self.qubits.find(&bit).unwrap()) {
                    qregs_to_remove.push(qreg);
                    break;
                }
            }
        }
        self.remove_qregs(py, &PyTuple::new_bound(py, qregs_to_remove))?;

        // Remove DAG in/out nodes etc.
        for bit in qubits.iter() {
            self.remove_idle_wire(py, Wire::Qubit(*bit))?;
        }

        // Copy the current qubit mapping so we can use it while remapping
        // wires used on edges and in operation qargs.
        let old_qubits = self.qubits.clone();

        // Remove the qubit indices, which will invalidate our mapping of Qubit to
        // Python bits throughout the entire DAG.
        self.qubits.remove_indices(py, qubits.clone())?;

        // Update input/output maps to use new Qubits.
        let io_mapping: HashMap<Qubit, [NodeIndex; 2]> = self
            .qubit_io_map
            .drain(..)
            .enumerate()
            .filter_map(|(k, v)| {
                let qubit = Qubit(k as u32);
                if qubits.contains(&qubit) {
                    None
                } else {
                    Some((
                        self.qubits
                            .find(old_qubits.get(qubit).unwrap().bind(py))
                            .unwrap(),
                        v,
                    ))
                }
            })
            .collect();

        self.qubit_io_map = (0..io_mapping.len())
            .map(|idx| {
                let qubit = Qubit(idx as u32);
                io_mapping[&qubit]
            })
            .collect();

        // Update edges to use the new Qubits.
        for edge_weight in self.dag.edge_weights_mut() {
            if let Wire::Qubit(b) = edge_weight {
                *b = self
                    .qubits
                    .find(old_qubits.get(*b).unwrap().bind(py))
                    .unwrap();
            }
        }

        // Update operation qargs to use the new Qubits.
        for node_weight in self.dag.node_weights_mut() {
            match node_weight {
                NodeType::Operation(op) => {
                    let qargs = self.qargs_interner.get(op.qubits);
                    let qarg_bits = old_qubits.map_indices(qargs).map(|b| b.bind(py).clone());
                    op.qubits = self
                        .qargs_interner
                        .insert_owned(self.qubits.map_bits(qarg_bits)?.collect());
                }
                NodeType::QubitIn(q) | NodeType::QubitOut(q) => {
                    *q = self
                        .qubits
                        .find(old_qubits.get(*q).unwrap().bind(py))
                        .unwrap();
                }
                _ => (),
            }
        }

        // Update bit locations.
        let bit_locations = self.qubit_locations.bind(py);
        for (i, bit) in self.qubits.bits().iter().enumerate() {
            let raw_loc = bit_locations.get_item(bit)?.unwrap();
            let loc = raw_loc.downcast::<BitLocations>().unwrap();
            loc.borrow_mut().index = i;
            bit_locations.set_item(bit, loc)?;
        }
        Ok(())
    }

    /// Remove quantum registers from the circuit, leaving underlying bits
    /// in place.
    ///
    /// Raises:
    ///     DAGCircuitError: a qreg is not a QuantumRegister, or is not in
    ///     the circuit.
    #[pyo3(signature = (*qregs))]
    fn remove_qregs(&mut self, py: Python, qregs: &Bound<PyTuple>) -> PyResult<()> {
        let mut non_regs = Vec::new();
        let mut unknown_regs = Vec::new();
        let self_bound_qregs = self.qregs.bind(py);
        for reg in qregs.iter() {
            if !reg.is_instance(imports::QUANTUM_REGISTER.get_bound(py))? {
                non_regs.push(reg);
            } else if let Some(existing_qreg) =
                self_bound_qregs.get_item(&reg.getattr(intern!(py, "name"))?)?
            {
                if !existing_qreg.eq(&reg)? {
                    unknown_regs.push(reg);
                }
            } else {
                unknown_regs.push(reg);
            }
        }
        if !non_regs.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "qregs not of type QuantumRegister: {:?}",
                non_regs
            )));
        }
        if !unknown_regs.is_empty() {
            return Err(DAGCircuitError::new_err(format!(
                "qregs not in circuit: {:?}",
                unknown_regs
            )));
        }

        for qreg in qregs {
            self.qregs
                .bind(py)
                .del_item(qreg.getattr(intern!(py, "name"))?)?;
            for (i, bit) in qreg.iter()?.enumerate() {
                let bit = bit?;
                let bit_position = self
                    .qubit_locations
                    .bind(py)
                    .get_item(bit)?
                    .unwrap()
                    .downcast_into_exact::<BitLocations>()?;
                bit_position
                    .borrow()
                    .registers
                    .bind(py)
                    .as_any()
                    .call_method1(intern!(py, "remove"), ((&qreg, i),))?;
            }
        }
        Ok(())
    }

    /// Verify that the condition is valid.
    ///
    /// Args:
    ///     name (string): used for error reporting
    ///     condition (tuple or None): a condition tuple (ClassicalRegister, int) or (Clbit, bool)
    ///
    /// Raises:
    ///     DAGCircuitError: if conditioning on an invalid register
    fn _check_condition(&self, py: Python, name: &str, condition: &Bound<PyAny>) -> PyResult<()> {
        if condition.is_none() {
            return Ok(());
        }

        let resources = self.control_flow_module.condition_resources(condition)?;
        for reg in resources.cregs.bind(py) {
            if !self
                .cregs
                .bind(py)
                .contains(reg.getattr(intern!(py, "name"))?)?
            {
                return Err(DAGCircuitError::new_err(format!(
                    "invalid creg in condition for {}",
                    name
                )));
            }
        }

        for bit in resources.clbits.bind(py) {
            if self.clbits.find(&bit).is_none() {
                return Err(DAGCircuitError::new_err(format!(
                    "invalid clbits in condition for {}",
                    name
                )));
            }
        }

        Ok(())
    }

    /// Return a copy of self with the same structure but empty.
    ///
    /// That structure includes:
    ///     * name and other metadata
    ///     * global phase
    ///     * duration
    ///     * all the qubits and clbits, including the registers.
    ///
    /// Returns:
    ///     DAGCircuit: An empty copy of self.
    #[pyo3(signature = (*, vars_mode="alike"))]
    fn copy_empty_like(&self, py: Python, vars_mode: &str) -> PyResult<Self> {
        let mut target_dag = DAGCircuit::with_capacity(
            py,
            self.num_qubits(),
            self.num_clbits(),
            Some(self.num_vars()),
            None,
            None,
        )?;
        target_dag.name = self.name.as_ref().map(|n| n.clone_ref(py));
        target_dag.global_phase = self.global_phase.clone();
        target_dag.duration = self.duration.as_ref().map(|d| d.clone_ref(py));
        target_dag.unit.clone_from(&self.unit);
        target_dag.metadata = self.metadata.as_ref().map(|m| m.clone_ref(py));
        target_dag.qargs_interner = self.qargs_interner.clone();
        target_dag.cargs_interner = self.cargs_interner.clone();

        for bit in self.qubits.bits() {
            target_dag.add_qubit_unchecked(py, bit.bind(py))?;
        }
        for bit in self.clbits.bits() {
            target_dag.add_clbit_unchecked(py, bit.bind(py))?;
        }
        for reg in self.qregs.bind(py).values() {
            target_dag.add_qreg(py, &reg)?;
        }
        for reg in self.cregs.bind(py).values() {
            target_dag.add_creg(py, &reg)?;
        }
        if vars_mode == "alike" {
            for var in self.vars_by_type[DAGVarType::Input as usize]
                .bind(py)
                .iter()
            {
                target_dag.add_var(py, &var, DAGVarType::Input)?;
            }
            for var in self.vars_by_type[DAGVarType::Capture as usize]
                .bind(py)
                .iter()
            {
                target_dag.add_var(py, &var, DAGVarType::Capture)?;
            }
            for var in self.vars_by_type[DAGVarType::Declare as usize]
                .bind(py)
                .iter()
            {
                target_dag.add_var(py, &var, DAGVarType::Declare)?;
            }
        } else if vars_mode == "captures" {
            for var in self.vars_by_type[DAGVarType::Input as usize]
                .bind(py)
                .iter()
            {
                target_dag.add_var(py, &var, DAGVarType::Capture)?;
            }
            for var in self.vars_by_type[DAGVarType::Capture as usize]
                .bind(py)
                .iter()
            {
                target_dag.add_var(py, &var, DAGVarType::Capture)?;
            }
            for var in self.vars_by_type[DAGVarType::Declare as usize]
                .bind(py)
                .iter()
            {
                target_dag.add_var(py, &var, DAGVarType::Capture)?;
            }
        } else if vars_mode != "drop" {
            return Err(PyValueError::new_err(format!(
                "unknown vars_mode: '{}'",
                vars_mode
            )));
        }

        Ok(target_dag)
    }

    #[pyo3(signature=(node, check=false))]
    fn _apply_op_node_back(
        &mut self,
        py: Python,
        node: &Bound<PyAny>,
        check: bool,
    ) -> PyResult<()> {
        if let NodeType::Operation(inst) = self.pack_into(py, node)? {
            if check {
                self.check_op_addition(py, &inst)?;
            }

            self.push_back(py, inst)?;
            Ok(())
        } else {
            Err(PyTypeError::new_err("Invalid node type input"))
        }
    }

    /// Apply an operation to the output of the circuit.
    ///
    /// Args:
    ///     op (qiskit.circuit.Operation): the operation associated with the DAG node
    ///     qargs (tuple[~qiskit.circuit.Qubit]): qubits that op will be applied to
    ///     cargs (tuple[Clbit]): cbits that op will be applied to
    ///     check (bool): If ``True`` (default), this function will enforce that the
    ///         :class:`.DAGCircuit` data-structure invariants are maintained (all ``qargs`` are
    ///         :class:`~.circuit.Qubit`\\ s, all are in the DAG, etc).  If ``False``, the caller *must*
    ///         uphold these invariants itself, but the cost of several checks will be skipped.
    ///         This is most useful when building a new DAG from a source of known-good nodes.
    /// Returns:
    ///     DAGOpNode: the node for the op that was added to the dag
    ///
    /// Raises:
    ///     DAGCircuitError: if a leaf node is connected to multiple outputs
    #[pyo3(name = "apply_operation_back", signature = (op, qargs=None, cargs=None, *, check=true))]
    fn py_apply_operation_back(
        &mut self,
        py: Python,
        op: Bound<PyAny>,
        qargs: Option<TupleLikeArg>,
        cargs: Option<TupleLikeArg>,
        check: bool,
    ) -> PyResult<Py<PyAny>> {
        let py_op = op.extract::<OperationFromPython>()?;
        let qargs = qargs.map(|q| q.value);
        let cargs = cargs.map(|c| c.value);
        let node = {
            let qubits_id = self
                .qargs_interner
                .insert_owned(self.qubits.map_bits(qargs.iter().flatten())?.collect());
            let clbits_id = self
                .cargs_interner
                .insert_owned(self.clbits.map_bits(cargs.iter().flatten())?.collect());
            let instr = PackedInstruction {
                op: py_op.operation,
                qubits: qubits_id,
                clbits: clbits_id,
                params: (!py_op.params.is_empty()).then(|| Box::new(py_op.params)),
                extra_attrs: py_op.extra_attrs,
                #[cfg(feature = "cache_pygates")]
                py_op: op.unbind().into(),
            };

            if check {
                self.check_op_addition(py, &instr)?;
            }
            self.push_back(py, instr)?
        };

        self.get_node(py, node)
    }

    /// Apply an operation to the input of the circuit.
    ///
    /// Args:
    ///     op (qiskit.circuit.Operation): the operation associated with the DAG node
    ///     qargs (tuple[~qiskit.circuit.Qubit]): qubits that op will be applied to
    ///     cargs (tuple[Clbit]): cbits that op will be applied to
    ///     check (bool): If ``True`` (default), this function will enforce that the
    ///         :class:`.DAGCircuit` data-structure invariants are maintained (all ``qargs`` are
    ///         :class:`~.circuit.Qubit`\\ s, all are in the DAG, etc).  If ``False``, the caller *must*
    ///         uphold these invariants itself, but the cost of several checks will be skipped.
    ///         This is most useful when building a new DAG from a source of known-good nodes.
    /// Returns:
    ///     DAGOpNode: the node for the op that was added to the dag
    ///
    /// Raises:
    ///     DAGCircuitError: if initial nodes connected to multiple out edges
    #[pyo3(name = "apply_operation_front", signature = (op, qargs=None, cargs=None, *, check=true))]
    fn py_apply_operation_front(
        &mut self,
        py: Python,
        op: Bound<PyAny>,
        qargs: Option<TupleLikeArg>,
        cargs: Option<TupleLikeArg>,
        check: bool,
    ) -> PyResult<Py<PyAny>> {
        let py_op = op.extract::<OperationFromPython>()?;
        let qargs = qargs.map(|q| q.value);
        let cargs = cargs.map(|c| c.value);
        let node = {
            let qubits_id = self
                .qargs_interner
                .insert_owned(self.qubits.map_bits(qargs.iter().flatten())?.collect());
            let clbits_id = self
                .cargs_interner
                .insert_owned(self.clbits.map_bits(cargs.iter().flatten())?.collect());
            let instr = PackedInstruction {
                op: py_op.operation,
                qubits: qubits_id,
                clbits: clbits_id,
                params: (!py_op.params.is_empty()).then(|| Box::new(py_op.params)),
                extra_attrs: py_op.extra_attrs,
                #[cfg(feature = "cache_pygates")]
                py_op: op.unbind().into(),
            };

            if check {
                self.check_op_addition(py, &instr)?;
            }
            self.push_front(py, instr)?
        };

        self.get_node(py, node)
    }

    /// Compose the ``other`` circuit onto the output of this circuit.
    ///
    /// A subset of input wires of ``other`` are mapped
    /// to a subset of output wires of this circuit.
    ///
    /// ``other`` can be narrower or of equal width to ``self``.
    ///
    /// Args:
    ///     other (DAGCircuit): circuit to compose with self
    ///     qubits (list[~qiskit.circuit.Qubit|int]): qubits of self to compose onto.
    ///     clbits (list[Clbit|int]): clbits of self to compose onto.
    ///     front (bool): If True, front composition will be performed (not implemented yet)
    ///     inplace (bool): If True, modify the object. Otherwise return composed circuit.
    ///     inline_captures (bool): If ``True``, variables marked as "captures" in the ``other`` DAG
    ///         will be inlined onto existing uses of those same variables in ``self``.  If ``False``,
    ///         all variables in ``other`` are required to be distinct from ``self``, and they will
    ///         be added to ``self``.
    ///
    /// ..
    ///     Note: unlike `QuantumCircuit.compose`, there's no `var_remap` argument here.  That's
    ///     because the `DAGCircuit` inner-block structure isn't set up well to allow the recursion,
    ///     and `DAGCircuit.compose` is generally only used to rebuild a DAG from layers within
    ///     itself than to join unrelated circuits.  While there's no strong motivating use-case
    ///     (unlike the `QuantumCircuit` equivalent), it's safer and more performant to not provide
    ///     the option.
    ///
    /// Returns:
    ///    DAGCircuit: the composed dag (returns None if inplace==True).
    ///
    /// Raises:
    ///     DAGCircuitError: if ``other`` is wider or there are duplicate edge mappings.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (other, qubits=None, clbits=None, front=false, inplace=true, *, inline_captures=false))]
    fn compose(
        slf: PyRefMut<Self>,
        py: Python,
        other: &DAGCircuit,
        qubits: Option<Bound<PyList>>,
        clbits: Option<Bound<PyList>>,
        front: bool,
        inplace: bool,
        inline_captures: bool,
    ) -> PyResult<Option<PyObject>> {
        if front {
            return Err(DAGCircuitError::new_err(
                "Front composition not supported yet.",
            ));
        }

        if other.qubits.len() > slf.qubits.len() || other.clbits.len() > slf.clbits.len() {
            return Err(DAGCircuitError::new_err(
                "Trying to compose with another DAGCircuit which has more 'in' edges.",
            ));
        }

        // Number of qubits and clbits must match number in circuit or None
        let identity_qubit_map = other
            .qubits
            .bits()
            .iter()
            .zip(slf.qubits.bits())
            .into_py_dict_bound(py);
        let identity_clbit_map = other
            .clbits
            .bits()
            .iter()
            .zip(slf.clbits.bits())
            .into_py_dict_bound(py);

        let qubit_map: Bound<PyDict> = match qubits {
            None => identity_qubit_map.clone(),
            Some(qubits) => {
                if qubits.len() != other.qubits.len() {
                    return Err(DAGCircuitError::new_err(concat!(
                        "Number of items in qubits parameter does not",
                        " match number of qubits in the circuit."
                    )));
                }

                let self_qubits = slf.qubits.cached().bind(py);
                let other_qubits = other.qubits.cached().bind(py);
                let dict = PyDict::new_bound(py);
                for (i, q) in qubits.iter().enumerate() {
                    let q = if q.is_instance_of::<PyInt>() {
                        self_qubits.get_item(q.extract()?)?
                    } else {
                        q
                    };

                    dict.set_item(other_qubits.get_item(i)?, q)?;
                }
                dict
            }
        };

        let clbit_map: Bound<PyDict> = match clbits {
            None => identity_clbit_map.clone(),
            Some(clbits) => {
                if clbits.len() != other.clbits.len() {
                    return Err(DAGCircuitError::new_err(concat!(
                        "Number of items in clbits parameter does not",
                        " match number of clbits in the circuit."
                    )));
                }

                let self_clbits = slf.clbits.cached().bind(py);
                let other_clbits = other.clbits.cached().bind(py);
                let dict = PyDict::new_bound(py);
                for (i, q) in clbits.iter().enumerate() {
                    let q = if q.is_instance_of::<PyInt>() {
                        self_clbits.get_item(q.extract()?)?
                    } else {
                        q
                    };

                    dict.set_item(other_clbits.get_item(i)?, q)?;
                }
                dict
            }
        };

        let edge_map = if qubit_map.is_empty() && clbit_map.is_empty() {
            // try to do a 1-1 mapping in order
            identity_qubit_map
                .iter()
                .chain(identity_clbit_map.iter())
                .into_py_dict_bound(py)
        } else {
            qubit_map
                .iter()
                .chain(clbit_map.iter())
                .into_py_dict_bound(py)
        };

        // Chck duplicates in wire map.
        {
            let edge_map_values: Vec<_> = edge_map.values().iter().collect();
            if PySet::new_bound(py, edge_map_values.as_slice())?.len() != edge_map.len() {
                return Err(DAGCircuitError::new_err("duplicates in wire_map"));
            }
        }

        // Compose
        let mut dag: PyRefMut<DAGCircuit> = if inplace {
            slf
        } else {
            Py::new(py, slf.clone())?.into_bound(py).borrow_mut()
        };

        dag.global_phase = add_global_phase(py, &dag.global_phase, &other.global_phase)?;

        for (gate, cals) in other.calibrations.iter() {
            let calibrations = match dag.calibrations.get(gate) {
                Some(calibrations) => calibrations,
                None => {
                    dag.calibrations
                        .insert(gate.clone(), PyDict::new_bound(py).unbind());
                    &dag.calibrations[gate]
                }
            };
            calibrations.bind(py).update(cals.bind(py).as_mapping())?;
        }

        // This is all the handling we need for realtime variables, if there's no remapping. They:
        //
        // * get added to the DAG and then operations involving them get appended on normally.
        // * get inlined onto an existing variable, then operations get appended normally.
        // * there's a clash or a failed inlining, and we just raise an error.
        //
        // Notably if there's no remapping, there's no need to recurse into control-flow or to do any
        // Var rewriting during the Expr visits.
        for var in other.iter_input_vars(py)?.bind(py) {
            dag.add_input_var(py, &var?)?;
        }
        if inline_captures {
            for var in other.iter_captured_vars(py)?.bind(py) {
                let var = var?;
                if !dag.has_var(&var)? {
                    return Err(DAGCircuitError::new_err(format!("Variable '{}' to be inlined is not in the base DAG. If you wanted it to be automatically added, use `inline_captures=False`.", var)));
                }
            }
        } else {
            for var in other.iter_captured_vars(py)?.bind(py) {
                dag.add_captured_var(py, &var?)?;
            }
        }
        for var in other.iter_declared_vars(py)?.bind(py) {
            dag.add_declared_var(py, &var?)?;
        }

        let variable_mapper = PyVariableMapper::new(
            py,
            dag.cregs.bind(py).values().into_any(),
            Some(edge_map.clone()),
            None,
            Some(wrap_pyfunction_bound!(reject_new_register, py)?.to_object(py)),
        )?;

        for node in other.topological_nodes()? {
            match &other.dag[node] {
                NodeType::QubitIn(q) => {
                    let bit = other.qubits.get(*q).unwrap().bind(py);
                    let m_wire = edge_map.get_item(bit)?.unwrap_or_else(|| bit.clone());
                    let wire_in_dag = dag.qubits.find(&m_wire);

                    if wire_in_dag.is_none()
                        || (dag.qubit_io_map.len() - 1 < wire_in_dag.unwrap().0 as usize)
                    {
                        return Err(DAGCircuitError::new_err(format!(
                            "wire {} not in self",
                            m_wire,
                        )));
                    }
                    // TODO: Python code has check here if node.wire is in other._wires. Why?
                }
                NodeType::ClbitIn(c) => {
                    let bit = other.clbits.get(*c).unwrap().bind(py);
                    let m_wire = edge_map.get_item(bit)?.unwrap_or_else(|| bit.clone());
                    let wire_in_dag = dag.clbits.find(&m_wire);
                    if wire_in_dag.is_none()
                        || dag.clbit_io_map.len() - 1 < wire_in_dag.unwrap().0 as usize
                    {
                        return Err(DAGCircuitError::new_err(format!(
                            "wire {} not in self",
                            m_wire,
                        )));
                    }
                    // TODO: Python code has check here if node.wire is in other._wires. Why?
                }
                NodeType::Operation(op) => {
                    let m_qargs = {
                        let qubits = other
                            .qubits
                            .map_indices(other.qargs_interner.get(op.qubits));
                        let mut mapped = Vec::with_capacity(qubits.len());
                        for bit in qubits {
                            mapped.push(
                                edge_map
                                    .get_item(bit)?
                                    .unwrap_or_else(|| bit.bind(py).clone()),
                            );
                        }
                        PyTuple::new_bound(py, mapped)
                    };
                    let m_cargs = {
                        let clbits = other
                            .clbits
                            .map_indices(other.cargs_interner.get(op.clbits));
                        let mut mapped = Vec::with_capacity(clbits.len());
                        for bit in clbits {
                            mapped.push(
                                edge_map
                                    .get_item(bit)?
                                    .unwrap_or_else(|| bit.bind(py).clone()),
                            );
                        }
                        PyTuple::new_bound(py, mapped)
                    };

                    // We explicitly create a mutable py_op here since we might
                    // update the condition.
                    let mut py_op = op.unpack_py_op(py)?.into_bound(py);
                    if py_op.getattr(intern!(py, "mutable"))?.extract::<bool>()? {
                        py_op = py_op.call_method0(intern!(py, "to_mutable"))?;
                    }

                    if let Some(condition) = op.condition() {
                        // TODO: do we need to check for condition.is_none()?
                        let condition = variable_mapper.map_condition(condition.bind(py), true)?;
                        if !op.op.control_flow() {
                            py_op = py_op.call_method1(
                                intern!(py, "c_if"),
                                condition.downcast::<PyTuple>()?,
                            )?;
                        } else {
                            py_op.setattr(intern!(py, "condition"), condition)?;
                        }
                    } else if py_op.is_instance(imports::SWITCH_CASE_OP.get_bound(py))? {
                        py_op.setattr(
                            intern!(py, "target"),
                            variable_mapper.map_target(&py_op.getattr(intern!(py, "target"))?)?,
                        )?;
                    };

                    dag.py_apply_operation_back(
                        py,
                        py_op,
                        Some(TupleLikeArg { value: m_qargs }),
                        Some(TupleLikeArg { value: m_cargs }),
                        false,
                    )?;
                }
                // If its a Var wire, we already checked that it exists in the destination.
                NodeType::VarIn(_)
                | NodeType::VarOut(_)
                | NodeType::QubitOut(_)
                | NodeType::ClbitOut(_) => (),
            }
        }

        if !inplace {
            Ok(Some(dag.into_py(py)))
        } else {
            Ok(None)
        }
    }

    /// Reverse the operations in the ``self`` circuit.
    ///
    /// Returns:
    ///     DAGCircuit: the reversed dag.
    fn reverse_ops<'py>(slf: PyRef<Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let qc = imports::DAG_TO_CIRCUIT.get_bound(py).call1((slf,))?;
        let reversed = qc.call_method0("reverse_ops")?;
        imports::CIRCUIT_TO_DAG.get_bound(py).call1((reversed,))
    }

    /// Return idle wires.
    ///
    /// Args:
    ///     ignore (list(str)): List of node names to ignore. Default: []
    ///
    /// Yields:
    ///     Bit: Bit in idle wire.
    ///
    /// Raises:
    ///     DAGCircuitError: If the DAG is invalid
    fn idle_wires(&self, py: Python, ignore: Option<&Bound<PyList>>) -> PyResult<Py<PyIterator>> {
        let mut result: Vec<PyObject> = Vec::new();
        let wires = (0..self.qubit_io_map.len())
            .map(|idx| Wire::Qubit(Qubit(idx as u32)))
            .chain((0..self.clbit_io_map.len()).map(|idx| Wire::Clbit(Clbit(idx as u32))))
            .chain(self.var_input_map.keys(py).map(Wire::Var));
        match ignore {
            Some(ignore) => {
                // Convert the list to a Rust set.
                let ignore_set = ignore
                    .into_iter()
                    .map(|s| s.extract())
                    .collect::<PyResult<HashSet<String>>>()?;
                for wire in wires {
                    let nodes_found = self.nodes_on_wire(py, &wire, true).into_iter().any(|node| {
                        let weight = self.dag.node_weight(node).unwrap();
                        if let NodeType::Operation(packed) = weight {
                            !ignore_set.contains(packed.op.name())
                        } else {
                            false
                        }
                    });

                    if !nodes_found {
                        result.push(match wire {
                            Wire::Qubit(qubit) => self.qubits.get(qubit).unwrap().clone_ref(py),
                            Wire::Clbit(clbit) => self.clbits.get(clbit).unwrap().clone_ref(py),
                            Wire::Var(var) => var,
                        });
                    }
                }
            }
            None => {
                for wire in wires {
                    if self.is_wire_idle(py, &wire)? {
                        result.push(match wire {
                            Wire::Qubit(qubit) => self.qubits.get(qubit).unwrap().clone_ref(py),
                            Wire::Clbit(clbit) => self.clbits.get(clbit).unwrap().clone_ref(py),
                            Wire::Var(var) => var,
                        });
                    }
                }
            }
        }
        Ok(PyTuple::new_bound(py, result).into_any().iter()?.unbind())
    }

    /// Return the number of operations.  If there is control flow present, this count may only
    /// be an estimate, as the complete control-flow path cannot be statically known.
    ///
    /// Args:
    ///     recurse: if ``True``, then recurse into control-flow operations.  For loops with
    ///         known-length iterators are counted unrolled.  If-else blocks sum both of the two
    ///         branches.  While loops are counted as if the loop body runs once only.  Defaults to
    ///         ``False`` and raises :class:`.DAGCircuitError` if any control flow is present, to
    ///         avoid silently returning a mostly meaningless number.
    ///
    /// Returns:
    ///     int: the circuit size
    ///
    /// Raises:
    ///     DAGCircuitError: if an unknown :class:`.ControlFlowOp` is present in a call with
    ///         ``recurse=True``, or any control flow is present in a non-recursive call.
    #[pyo3(signature= (*, recurse=false))]
    fn size(&self, py: Python, recurse: bool) -> PyResult<usize> {
        let mut length = self.dag.node_count() - (self.width() * 2);
        if !self.has_control_flow() {
            return Ok(length);
        }
        if !recurse {
            return Err(DAGCircuitError::new_err(concat!(
                "Size with control flow is ambiguous.",
                " You may use `recurse=True` to get a result",
                " but see this method's documentation for the meaning of this."
            )));
        }

        // Handle recursively.
        let circuit_to_dag = imports::CIRCUIT_TO_DAG.get_bound(py);
        for node in self.dag.node_weights() {
            let NodeType::Operation(node) = node else {
                continue;
            };
            if !node.op.control_flow() {
                continue;
            }
            let OperationRef::Instruction(inst) = node.op.view() else {
                panic!("control flow op must be an instruction");
            };
            let inst_bound = inst.instruction.bind(py);
            if inst_bound.is_instance(imports::FOR_LOOP_OP.get_bound(py))? {
                let blocks = inst_bound.getattr("blocks")?;
                let block_zero = blocks.get_item(0)?;
                let inner_dag: &DAGCircuit = &circuit_to_dag.call1((block_zero,))?.extract()?;
                length += node.params_view().len() * inner_dag.size(py, true)?
            } else if inst_bound.is_instance(imports::WHILE_LOOP_OP.get_bound(py))? {
                let blocks = inst_bound.getattr("blocks")?;
                let block_zero = blocks.get_item(0)?;
                let inner_dag: &DAGCircuit = &circuit_to_dag.call1((block_zero,))?.extract()?;
                length += inner_dag.size(py, true)?
            } else if inst_bound.is_instance(imports::IF_ELSE_OP.get_bound(py))?
                || inst_bound.is_instance(imports::SWITCH_CASE_OP.get_bound(py))?
            {
                let blocks = inst_bound.getattr("blocks")?;
                for block in blocks.iter()? {
                    let inner_dag: &DAGCircuit = &circuit_to_dag.call1((block?,))?.extract()?;
                    length += inner_dag.size(py, true)?;
                }
            } else {
                continue;
            }
            // We don't count a control-flow node itself!
            length -= 1;
        }
        Ok(length)
    }

    /// Return the circuit depth.  If there is control flow present, this count may only be an
    /// estimate, as the complete control-flow path cannot be statically known.
    ///
    /// Args:
    ///     recurse: if ``True``, then recurse into control-flow operations.  For loops
    ///         with known-length iterators are counted as if the loop had been manually unrolled
    ///         (*i.e.* with each iteration of the loop body written out explicitly).
    ///         If-else blocks take the longer case of the two branches.  While loops are counted as
    ///         if the loop body runs once only.  Defaults to ``False`` and raises
    ///         :class:`.DAGCircuitError` if any control flow is present, to avoid silently
    ///         returning a nonsensical number.
    ///
    /// Returns:
    ///     int: the circuit depth
    ///
    /// Raises:
    ///     DAGCircuitError: if not a directed acyclic graph
    ///     DAGCircuitError: if unknown control flow is present in a recursive call, or any control
    ///         flow is present in a non-recursive call.
    #[pyo3(signature= (*, recurse=false))]
    fn depth(&self, py: Python, recurse: bool) -> PyResult<usize> {
        if self.qubits.is_empty() && self.clbits.is_empty() && self.vars_info.is_empty() {
            return Ok(0);
        }
        if !self.has_control_flow() {
            let weight_fn = |_| -> Result<usize, Infallible> { Ok(1) };
            return match rustworkx_core::dag_algo::longest_path(&self.dag, weight_fn).unwrap() {
                Some(res) => Ok(res.1 - 1),
                None => Err(DAGCircuitError::new_err("not a DAG")),
            };
        }
        if !recurse {
            return Err(DAGCircuitError::new_err(concat!(
                "Depth with control flow is ambiguous.",
                " You may use `recurse=True` to get a result",
                " but see this method's documentation for the meaning of this."
            )));
        }

        // Handle recursively.
        let circuit_to_dag = imports::CIRCUIT_TO_DAG.get_bound(py);
        let mut node_lookup: HashMap<NodeIndex, usize> = HashMap::new();
        for (node_index, node) in self.dag.node_references() {
            let NodeType::Operation(node) = node else {
                continue;
            };
            if !node.op.control_flow() {
                continue;
            }
            let OperationRef::Instruction(inst) = node.op.view() else {
                panic!("control flow op must be an instruction")
            };
            let inst_bound = inst.instruction.bind(py);
            let weight = if inst_bound.is_instance(imports::FOR_LOOP_OP.get_bound(py))? {
                node.params_view().len()
            } else {
                1
            };
            if weight == 0 {
                node_lookup.insert(node_index, 0);
            } else {
                let blocks = inst_bound.getattr("blocks")?;
                let mut block_weights: Vec<usize> = Vec::with_capacity(blocks.len()?);
                for block in blocks.iter()? {
                    let inner_dag: &DAGCircuit = &circuit_to_dag.call1((block?,))?.extract()?;
                    block_weights.push(inner_dag.depth(py, true)?);
                }
                node_lookup.insert(node_index, weight * block_weights.iter().max().unwrap());
            }
        }

        let weight_fn = |edge: EdgeReference<'_, Wire>| -> Result<usize, Infallible> {
            Ok(*node_lookup.get(&edge.target()).unwrap_or(&1))
        };
        match rustworkx_core::dag_algo::longest_path(&self.dag, weight_fn).unwrap() {
            Some(res) => Ok(res.1 - 1),
            None => Err(DAGCircuitError::new_err("not a DAG")),
        }
    }

    /// Return the total number of qubits + clbits used by the circuit.
    /// This function formerly returned the number of qubits by the calculation
    /// return len(self._wires) - self.num_clbits()
    /// but was changed by issue #2564 to return number of qubits + clbits
    /// with the new function DAGCircuit.num_qubits replacing the former
    /// semantic of DAGCircuit.width().
    fn width(&self) -> usize {
        self.qubits.len() + self.clbits.len() + self.vars_info.len()
    }

    /// Return the total number of qubits used by the circuit.
    /// num_qubits() replaces former use of width().
    /// DAGCircuit.width() now returns qubits + clbits for
    /// consistency with Circuit.width() [qiskit-terra #2564].
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Return the total number of classical bits used by the circuit.
    pub fn num_clbits(&self) -> usize {
        self.clbits.len()
    }

    /// Compute how many components the circuit can decompose into.
    fn num_tensor_factors(&self) -> usize {
        // This function was forked from rustworkx's
        // number_weekly_connected_components() function as of 0.15.0:
        // https://github.com/Qiskit/rustworkx/blob/0.15.0/src/connectivity/mod.rs#L215-L235

        let mut weak_components = self.dag.node_count();
        let mut vertex_sets = UnionFind::new(self.dag.node_bound());
        for edge in self.dag.edge_references() {
            let (a, b) = (edge.source(), edge.target());
            // union the two vertices of the edge
            if vertex_sets.union(a.index(), b.index()) {
                weak_components -= 1
            };
        }
        weak_components
    }

    fn __eq__(&self, py: Python, other: &DAGCircuit) -> PyResult<bool> {
        // Try to convert to float, but in case of unbound ParameterExpressions
        // a TypeError will be raise, fallback to normal equality in those
        // cases.
        let phase_is_close = |self_phase: f64, other_phase: f64| -> bool {
            ((self_phase - other_phase + PI).rem_euclid(2. * PI) - PI).abs() <= 1.0e-10
        };
        let normalize_param = |param: &Param| {
            if let Param::ParameterExpression(ob) = param {
                ob.bind(py)
                    .call_method0(intern!(py, "numeric"))
                    .ok()
                    .map(|ob| ob.extract::<Param>())
                    .unwrap_or_else(|| Ok(param.clone()))
            } else {
                Ok(param.clone())
            }
        };

        let phase_eq = match [
            normalize_param(&self.global_phase)?,
            normalize_param(&other.global_phase)?,
        ] {
            [Param::Float(self_phase), Param::Float(other_phase)] => {
                Ok(phase_is_close(self_phase, other_phase))
            }
            _ => self.global_phase.eq(py, &other.global_phase),
        }?;
        if !phase_eq {
            return Ok(false);
        }
        if self.calibrations.len() != other.calibrations.len() {
            return Ok(false);
        }

        for (k, v1) in &self.calibrations {
            match other.calibrations.get(k) {
                Some(v2) => {
                    if !v1.bind(py).eq(v2.bind(py))? {
                        return Ok(false);
                    }
                }
                None => {
                    return Ok(false);
                }
            }
        }

        // We don't do any semantic equivalence between Var nodes, as things stand; DAGs can only be
        // equal in our mind if they use the exact same UUID vars.
        for (our_vars, their_vars) in self.vars_by_type.iter().zip(&other.vars_by_type) {
            if !our_vars.bind(py).eq(their_vars)? {
                return Ok(false);
            }
        }

        let self_bit_indices = {
            let indices = self
                .qubits
                .bits()
                .iter()
                .chain(self.clbits.bits())
                .enumerate()
                .map(|(idx, bit)| (bit, idx));
            indices.into_py_dict_bound(py)
        };

        let other_bit_indices = {
            let indices = other
                .qubits
                .bits()
                .iter()
                .chain(other.clbits.bits())
                .enumerate()
                .map(|(idx, bit)| (bit, idx));
            indices.into_py_dict_bound(py)
        };

        // Check if qregs are the same.
        let self_qregs = self.qregs.bind(py);
        let other_qregs = other.qregs.bind(py);
        if self_qregs.len() != other_qregs.len() {
            return Ok(false);
        }
        for (regname, self_bits) in self_qregs {
            let self_bits = self_bits
                .getattr("_bits")?
                .downcast_into_exact::<PyList>()?;
            let other_bits = match other_qregs.get_item(regname)? {
                Some(bits) => bits.getattr("_bits")?.downcast_into_exact::<PyList>()?,
                None => return Ok(false),
            };
            if !self
                .qubits
                .map_bits(self_bits)?
                .eq(other.qubits.map_bits(other_bits)?)
            {
                return Ok(false);
            }
        }

        // Check if cregs are the same.
        let self_cregs = self.cregs.bind(py);
        let other_cregs = other.cregs.bind(py);
        if self_cregs.len() != other_cregs.len() {
            return Ok(false);
        }

        for (regname, self_bits) in self_cregs {
            let self_bits = self_bits
                .getattr("_bits")?
                .downcast_into_exact::<PyList>()?;
            let other_bits = match other_cregs.get_item(regname)? {
                Some(bits) => bits.getattr("_bits")?.downcast_into_exact::<PyList>()?,
                None => return Ok(false),
            };
            if !self
                .clbits
                .map_bits(self_bits)?
                .eq(other.clbits.map_bits(other_bits)?)
            {
                return Ok(false);
            }
        }

        // Check for VF2 isomorphic match.
        let legacy_condition_eq = imports::LEGACY_CONDITION_CHECK.get_bound(py);
        let condition_op_check = imports::CONDITION_OP_CHECK.get_bound(py);
        let switch_case_op_check = imports::SWITCH_CASE_OP_CHECK.get_bound(py);
        let for_loop_op_check = imports::FOR_LOOP_OP_CHECK.get_bound(py);
        let node_match = |n1: &NodeType, n2: &NodeType| -> PyResult<bool> {
            match [n1, n2] {
                [NodeType::Operation(inst1), NodeType::Operation(inst2)] => {
                    if inst1.op.name() != inst2.op.name() {
                        return Ok(false);
                    }
                    let check_args = || -> bool {
                        let node1_qargs = self.qargs_interner.get(inst1.qubits);
                        let node2_qargs = other.qargs_interner.get(inst2.qubits);
                        let node1_cargs = self.cargs_interner.get(inst1.clbits);
                        let node2_cargs = other.cargs_interner.get(inst2.clbits);
                        if SEMANTIC_EQ_SYMMETRIC.contains(&inst1.op.name()) {
                            let node1_qargs =
                                node1_qargs.iter().copied().collect::<HashSet<Qubit>>();
                            let node2_qargs =
                                node2_qargs.iter().copied().collect::<HashSet<Qubit>>();
                            let node1_cargs =
                                node1_cargs.iter().copied().collect::<HashSet<Clbit>>();
                            let node2_cargs =
                                node2_cargs.iter().copied().collect::<HashSet<Clbit>>();
                            if node1_qargs != node2_qargs || node1_cargs != node2_cargs {
                                return false;
                            }
                        } else if node1_qargs != node2_qargs || node1_cargs != node2_cargs {
                            return false;
                        }
                        true
                    };
                    let check_conditions = || -> PyResult<bool> {
                        if let Some(cond1) = inst1
                            .extra_attrs
                            .as_ref()
                            .and_then(|attrs| attrs.condition.as_ref())
                        {
                            if let Some(cond2) = inst2
                                .extra_attrs
                                .as_ref()
                                .and_then(|attrs| attrs.condition.as_ref())
                            {
                                legacy_condition_eq
                                    .call1((cond1, cond2, &self_bit_indices, &other_bit_indices))?
                                    .extract::<bool>()
                            } else {
                                Ok(false)
                            }
                        } else {
                            Ok(inst2
                                .extra_attrs
                                .as_ref()
                                .and_then(|attrs| attrs.condition.as_ref())
                                .is_none())
                        }
                    };

                    match [inst1.op.view(), inst2.op.view()] {
                        [OperationRef::Standard(_op1), OperationRef::Standard(_op2)] => {
                            Ok(inst1.py_op_eq(py, inst2)?
                                && check_args()
                                && check_conditions()?
                                && inst1
                                    .params_view()
                                    .iter()
                                    .zip(inst2.params_view().iter())
                                    .all(|(a, b)| a.is_close(py, b, 1e-10).unwrap()))
                        }
                        [OperationRef::Instruction(op1), OperationRef::Instruction(op2)] => {
                            if op1.control_flow() && op2.control_flow() {
                                let n1 = self.unpack_into(py, NodeIndex::new(0), n1)?;
                                let n2 = other.unpack_into(py, NodeIndex::new(0), n2)?;
                                let name = op1.name();
                                if name == "if_else" || name == "while_loop" {
                                    condition_op_check
                                        .call1((n1, n2, &self_bit_indices, &other_bit_indices))?
                                        .extract()
                                } else if name == "switch_case" {
                                    switch_case_op_check
                                        .call1((n1, n2, &self_bit_indices, &other_bit_indices))?
                                        .extract()
                                } else if name == "for_loop" {
                                    for_loop_op_check
                                        .call1((n1, n2, &self_bit_indices, &other_bit_indices))?
                                        .extract()
                                } else {
                                    Err(PyRuntimeError::new_err(format!(
                                        "unhandled control-flow operation: {}",
                                        name
                                    )))
                                }
                            } else {
                                Ok(inst1.py_op_eq(py, inst2)?
                                    && check_args()
                                    && check_conditions()?)
                            }
                        }
                        [OperationRef::Gate(_op1), OperationRef::Gate(_op2)] => {
                            Ok(inst1.py_op_eq(py, inst2)? && check_args() && check_conditions()?)
                        }
                        [OperationRef::Operation(_op1), OperationRef::Operation(_op2)] => {
                            Ok(inst1.py_op_eq(py, inst2)? && check_args())
                        }
                        // Handle the case we end up with a pygate for a standardgate
                        // this typically only happens if it's a ControlledGate in python
                        // and we have mutable state set.
                        [OperationRef::Standard(_op1), OperationRef::Gate(_op2)] => {
                            Ok(inst1.py_op_eq(py, inst2)? && check_args() && check_conditions()?)
                        }
                        [OperationRef::Gate(_op1), OperationRef::Standard(_op2)] => {
                            Ok(inst1.py_op_eq(py, inst2)? && check_args() && check_conditions()?)
                        }
                        _ => Ok(false),
                    }
                }
                [NodeType::QubitIn(bit1), NodeType::QubitIn(bit2)] => Ok(bit1 == bit2),
                [NodeType::ClbitIn(bit1), NodeType::ClbitIn(bit2)] => Ok(bit1 == bit2),
                [NodeType::QubitOut(bit1), NodeType::QubitOut(bit2)] => Ok(bit1 == bit2),
                [NodeType::ClbitOut(bit1), NodeType::ClbitOut(bit2)] => Ok(bit1 == bit2),
                [NodeType::VarIn(var1), NodeType::VarIn(var2)] => var1.bind(py).eq(var2),
                [NodeType::VarOut(var1), NodeType::VarOut(var2)] => var1.bind(py).eq(var2),
                _ => Ok(false),
            }
        };

        isomorphism::vf2::is_isomorphic(
            &self.dag,
            &other.dag,
            node_match,
            isomorphism::vf2::NoSemanticMatch,
            true,
            Ordering::Equal,
            true,
            None,
        )
        .map_err(|e| match e {
            isomorphism::vf2::IsIsomorphicError::NodeMatcherErr(e) => e,
            _ => {
                unreachable!()
            }
        })
    }

    /// Yield nodes in topological order.
    ///
    /// Args:
    ///     key (Callable): A callable which will take a DAGNode object and
    ///         return a string sort key. If not specified the
    ///         :attr:`~qiskit.dagcircuit.DAGNode.sort_key` attribute will be
    ///         used as the sort key for each node.
    ///
    /// Returns:
    ///     generator(DAGOpNode, DAGInNode, or DAGOutNode): node in topological order
    #[pyo3(name = "topological_nodes")]
    fn py_topological_nodes(
        &self,
        py: Python,
        key: Option<Bound<PyAny>>,
    ) -> PyResult<Py<PyIterator>> {
        let nodes: PyResult<Vec<_>> = if let Some(key) = key {
            self.topological_key_sort(py, &key)?
                .map(|node| self.get_node(py, node))
                .collect()
        } else {
            // Good path, using interner IDs.
            self.topological_nodes()?
                .map(|n| self.get_node(py, n))
                .collect()
        };

        Ok(PyTuple::new_bound(py, nodes?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Yield op nodes in topological order.
    ///
    /// Allowed to pass in specific key to break ties in top order
    ///
    /// Args:
    ///     key (Callable): A callable which will take a DAGNode object and
    ///         return a string sort key. If not specified the
    ///         :attr:`~qiskit.dagcircuit.DAGNode.sort_key` attribute will be
    ///         used as the sort key for each node.
    ///
    /// Returns:
    ///     generator(DAGOpNode): op node in topological order
    #[pyo3(name = "topological_op_nodes")]
    fn py_topological_op_nodes(
        &self,
        py: Python,
        key: Option<Bound<PyAny>>,
    ) -> PyResult<Py<PyIterator>> {
        let nodes: PyResult<Vec<_>> = if let Some(key) = key {
            self.topological_key_sort(py, &key)?
                .filter_map(|node| match self.dag.node_weight(node) {
                    Some(NodeType::Operation(_)) => Some(self.get_node(py, node)),
                    _ => None,
                })
                .collect()
        } else {
            // Good path, using interner IDs.
            self.topological_op_nodes()?
                .map(|n| self.get_node(py, n))
                .collect()
        };

        Ok(PyTuple::new_bound(py, nodes?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Replace a block of nodes with a single node.
    ///
    /// This is used to consolidate a block of DAGOpNodes into a single
    /// operation. A typical example is a block of gates being consolidated
    /// into a single ``UnitaryGate`` representing the unitary matrix of the
    /// block.
    ///
    /// Args:
    ///     node_block (List[DAGNode]): A list of dag nodes that represents the
    ///         node block to be replaced
    ///     op (qiskit.circuit.Operation): The operation to replace the
    ///         block with
    ///     wire_pos_map (Dict[Bit, int]): The dictionary mapping the bits to their positions in the
    ///         output ``qargs`` or ``cargs``. This is necessary to reconstruct the arg order over
    ///         multiple gates in the combined single op node.  If a :class:`.Bit` is not in the
    ///         dictionary, it will not be added to the args; this can be useful when dealing with
    ///         control-flow operations that have inherent bits in their ``condition`` or ``target``
    ///         fields.
    ///     cycle_check (bool): When set to True this method will check that
    ///         replacing the provided ``node_block`` with a single node
    ///         would introduce a cycle (which would invalidate the
    ///         ``DAGCircuit``) and will raise a ``DAGCircuitError`` if a cycle
    ///         would be introduced. This checking comes with a run time
    ///         penalty. If you can guarantee that your input ``node_block`` is
    ///         a contiguous block and won't introduce a cycle when it's
    ///         contracted to a single node, this can be set to ``False`` to
    ///         improve the runtime performance of this method.
    ///
    /// Raises:
    ///     DAGCircuitError: if ``cycle_check`` is set to ``True`` and replacing
    ///         the specified block introduces a cycle or if ``node_block`` is
    ///         empty.
    ///
    /// Returns:
    ///     DAGOpNode: The op node that replaces the block.
    #[pyo3(signature = (node_block, op, wire_pos_map, cycle_check=true))]
    fn replace_block_with_op(
        &mut self,
        py: Python,
        node_block: Vec<PyRef<DAGNode>>,
        op: Bound<PyAny>,
        wire_pos_map: &Bound<PyDict>,
        cycle_check: bool,
    ) -> PyResult<Py<PyAny>> {
        // If node block is empty return early
        if node_block.is_empty() {
            return Err(DAGCircuitError::new_err(
                "Can't replace an empty 'node_block'",
            ));
        }

        let mut qubit_pos_map: HashMap<Qubit, usize> = HashMap::new();
        let mut clbit_pos_map: HashMap<Clbit, usize> = HashMap::new();
        for (bit, index) in wire_pos_map.iter() {
            if bit.is_instance(imports::QUBIT.get_bound(py))? {
                qubit_pos_map.insert(self.qubits.find(&bit).unwrap(), index.extract()?);
            } else if bit.is_instance(imports::CLBIT.get_bound(py))? {
                clbit_pos_map.insert(self.clbits.find(&bit).unwrap(), index.extract()?);
            } else {
                return Err(DAGCircuitError::new_err(
                    "Wire map keys must be Qubit or Clbit instances.",
                ));
            }
        }

        let block_ids: Vec<_> = node_block.iter().map(|n| n.node.unwrap()).collect();

        let mut block_op_names = Vec::new();
        let mut block_qargs: HashSet<Qubit> = HashSet::new();
        let mut block_cargs: HashSet<Clbit> = HashSet::new();
        for nd in &block_ids {
            let weight = self.dag.node_weight(*nd);
            match weight {
                Some(NodeType::Operation(packed)) => {
                    block_op_names.push(packed.op.name().to_string());
                    block_qargs.extend(self.qargs_interner.get(packed.qubits));
                    block_cargs.extend(self.cargs_interner.get(packed.clbits));

                    if let Some(condition) = packed.condition() {
                        block_cargs.extend(
                            self.clbits.map_bits(
                                self.control_flow_module
                                    .condition_resources(condition.bind(py))?
                                    .clbits
                                    .bind(py),
                            )?,
                        );
                        continue;
                    }

                    // Add classical bits from SwitchCaseOp, if applicable.
                    if let OperationRef::Instruction(op) = packed.op.view() {
                        if op.name() == "switch_case" {
                            let op_bound = op.instruction.bind(py);
                            let target = op_bound.getattr(intern!(py, "target"))?;
                            if target.is_instance(imports::CLBIT.get_bound(py))? {
                                block_cargs.insert(self.clbits.find(&target).unwrap());
                            } else if target
                                .is_instance(imports::CLASSICAL_REGISTER.get_bound(py))?
                            {
                                block_cargs.extend(
                                    self.clbits
                                        .map_bits(target.extract::<Vec<Bound<PyAny>>>()?)?,
                                );
                            } else {
                                block_cargs.extend(
                                    self.clbits.map_bits(
                                        self.control_flow_module
                                            .node_resources(&target)?
                                            .clbits
                                            .bind(py),
                                    )?,
                                );
                            }
                        }
                    }
                }
                Some(_) => {
                    return Err(DAGCircuitError::new_err(
                        "Nodes in 'node_block' must be of type 'DAGOpNode'.",
                    ))
                }
                None => {
                    return Err(DAGCircuitError::new_err(
                        "Node in 'node_block' not found in DAG.",
                    ))
                }
            }
        }

        let mut block_qargs: Vec<Qubit> = block_qargs
            .into_iter()
            .filter(|q| qubit_pos_map.contains_key(q))
            .collect();
        block_qargs.sort_by_key(|q| qubit_pos_map[q]);

        let mut block_cargs: Vec<Clbit> = block_cargs
            .into_iter()
            .filter(|c| clbit_pos_map.contains_key(c))
            .collect();
        block_cargs.sort_by_key(|c| clbit_pos_map[c]);

        let py_op = op.extract::<OperationFromPython>()?;

        if py_op.operation.num_qubits() as usize != block_qargs.len() {
            return Err(DAGCircuitError::new_err(format!(
                "Number of qubits in the replacement operation ({}) is not equal to the number of qubits in the block ({})!", py_op.operation.num_qubits(), block_qargs.len()
            )));
        }

        let op_name = py_op.operation.name().to_string();
        let qubits = self.qargs_interner.insert_owned(block_qargs);
        let clbits = self.cargs_interner.insert_owned(block_cargs);
        let weight = NodeType::Operation(PackedInstruction {
            op: py_op.operation,
            qubits,
            clbits,
            params: (!py_op.params.is_empty()).then(|| Box::new(py_op.params)),
            extra_attrs: py_op.extra_attrs,
            #[cfg(feature = "cache_pygates")]
            py_op: op.unbind().into(),
        });

        let new_node = self
            .dag
            .contract_nodes(block_ids, weight, cycle_check)
            .map_err(|e| match e {
                ContractError::DAGWouldCycle => DAGCircuitError::new_err(
                    "Replacing the specified node block would introduce a cycle",
                ),
            })?;

        self.increment_op(op_name.as_str());
        for name in block_op_names {
            self.decrement_op(name.as_str());
        }

        self.get_node(py, new_node)
    }

    /// Replace one node with dag.
    ///
    /// Args:
    ///     node (DAGOpNode): node to substitute
    ///     input_dag (DAGCircuit): circuit that will substitute the node
    ///     wires (list[Bit] | Dict[Bit, Bit]): gives an order for (qu)bits
    ///         in the input circuit. If a list, then the bits refer to those in the ``input_dag``,
    ///         and the order gets matched to the node wires by qargs first, then cargs, then
    ///         conditions.  If a dictionary, then a mapping of bits in the ``input_dag`` to those
    ///         that the ``node`` acts on.
    ///     propagate_condition (bool): If ``True`` (default), then any ``condition`` attribute on
    ///         the operation within ``node`` is propagated to each node in the ``input_dag``.  If
    ///         ``False``, then the ``input_dag`` is assumed to faithfully implement suitable
    ///         conditional logic already.  This is ignored for :class:`.ControlFlowOp`\\ s (i.e.
    ///         treated as if it is ``False``); replacements of those must already fulfill the same
    ///         conditional logic or this function would be close to useless for them.
    ///
    /// Returns:
    ///     dict: maps node IDs from `input_dag` to their new node incarnations in `self`.
    ///
    /// Raises:
    ///     DAGCircuitError: if met with unexpected predecessor/successors
    #[pyo3(signature = (node, input_dag, wires=None, propagate_condition=true))]
    fn substitute_node_with_dag(
        &mut self,
        py: Python,
        node: &Bound<PyAny>,
        input_dag: &DAGCircuit,
        wires: Option<Bound<PyAny>>,
        propagate_condition: bool,
    ) -> PyResult<Py<PyDict>> {
        let (node_index, bound_node) = match node.downcast::<DAGOpNode>() {
            Ok(bound_node) => (bound_node.borrow().as_ref().node.unwrap(), bound_node),
            Err(_) => return Err(DAGCircuitError::new_err("expected node DAGOpNode")),
        };

        let node = match &self.dag[node_index] {
            NodeType::Operation(op) => op.clone(),
            _ => return Err(DAGCircuitError::new_err("expected node")),
        };

        type WireMapsTuple = (HashMap<Qubit, Qubit>, HashMap<Clbit, Clbit>, Py<PyDict>);

        let build_wire_map = |wires: &Bound<PyList>| -> PyResult<WireMapsTuple> {
            let qargs_list = imports::BUILTIN_LIST
                .get_bound(py)
                .call1((bound_node.borrow().get_qargs(py),))?;
            let qargs_list = qargs_list.downcast::<PyList>().unwrap();
            let cargs_list = imports::BUILTIN_LIST
                .get_bound(py)
                .call1((bound_node.borrow().get_cargs(py),))?;
            let cargs_list = cargs_list.downcast::<PyList>().unwrap();
            let cargs_set = imports::BUILTIN_SET.get_bound(py).call1((cargs_list,))?;
            let cargs_set = cargs_set.downcast::<PySet>().unwrap();
            if !propagate_condition && self.may_have_additional_wires(py, &node) {
                let (add_cargs, _add_vars) =
                    self.additional_wires(py, node.op.view(), node.condition())?;
                for wire in add_cargs.iter() {
                    let clbit = &self.clbits.get(*wire).unwrap();
                    if !cargs_set.contains(clbit.clone_ref(py))? {
                        cargs_list.append(clbit)?;
                    }
                }
            }
            let qargs_len = qargs_list.len();
            let cargs_len = cargs_list.len();

            if qargs_len + cargs_len != wires.len() {
                return Err(DAGCircuitError::new_err(format!(
                    "bit mapping invalid: expected {}, got {}",
                    qargs_len + cargs_len,
                    wires.len()
                )));
            }
            let mut qubit_wire_map = HashMap::new();
            let mut clbit_wire_map = HashMap::new();
            let var_map = PyDict::new_bound(py);
            for (index, wire) in wires.iter().enumerate() {
                if wire.is_instance(imports::QUBIT.get_bound(py))? {
                    if index >= qargs_len {
                        unreachable!()
                    }
                    let input_qubit: Qubit = input_dag.qubits.find(&wire).unwrap();
                    let self_qubit: Qubit = self.qubits.find(&qargs_list.get_item(index)?).unwrap();
                    qubit_wire_map.insert(input_qubit, self_qubit);
                } else if wire.is_instance(imports::CLBIT.get_bound(py))? {
                    if index < qargs_len {
                        unreachable!()
                    }
                    clbit_wire_map.insert(
                        input_dag.clbits.find(&wire).unwrap(),
                        self.clbits
                            .find(&cargs_list.get_item(index - qargs_len)?)
                            .unwrap(),
                    );
                } else {
                    return Err(DAGCircuitError::new_err(
                        "`Var` nodes cannot be remapped during substitution",
                    ));
                }
            }
            Ok((qubit_wire_map, clbit_wire_map, var_map.unbind()))
        };

        let (mut qubit_wire_map, mut clbit_wire_map, var_map): (
            HashMap<Qubit, Qubit>,
            HashMap<Clbit, Clbit>,
            Py<PyDict>,
        ) = match wires {
            Some(wires) => match wires.downcast::<PyDict>() {
                Ok(bound_wires) => {
                    let mut qubit_wire_map = HashMap::new();
                    let mut clbit_wire_map = HashMap::new();
                    let var_map = PyDict::new_bound(py);
                    for (source_wire, target_wire) in bound_wires.iter() {
                        if source_wire.is_instance(imports::QUBIT.get_bound(py))? {
                            qubit_wire_map.insert(
                                input_dag.qubits.find(&source_wire).unwrap(),
                                self.qubits.find(&target_wire).unwrap(),
                            );
                        } else if source_wire.is_instance(imports::CLBIT.get_bound(py))? {
                            clbit_wire_map.insert(
                                input_dag.clbits.find(&source_wire).unwrap(),
                                self.clbits.find(&target_wire).unwrap(),
                            );
                        } else {
                            var_map.set_item(source_wire, target_wire)?;
                        }
                    }
                    (qubit_wire_map, clbit_wire_map, var_map.unbind())
                }
                Err(_) => {
                    let wires: Bound<PyList> = match wires.downcast::<PyList>() {
                        Ok(bound_list) => bound_list.clone(),
                        // If someone passes a sequence instead of an exact list (tuple is
                        // occasionally used) cast that to a list and then use it.
                        Err(_) => {
                            let raw_wires = imports::BUILTIN_LIST.get_bound(py).call1((wires,))?;
                            raw_wires.extract()?
                        }
                    };
                    build_wire_map(&wires)?
                }
            },
            None => {
                let raw_wires = input_dag.get_wires(py);
                let binding = raw_wires?;
                let wires = binding.bind(py);
                build_wire_map(wires)?
            }
        };

        let var_iter = input_dag.iter_vars(py)?;
        let raw_set = imports::BUILTIN_SET.get_bound(py).call1((var_iter,))?;
        let input_dag_var_set: &Bound<PySet> = raw_set.downcast()?;

        let node_vars = if self.may_have_additional_wires(py, &node) {
            let (_additional_clbits, additional_vars) =
                self.additional_wires(py, node.op.view(), node.condition())?;
            let var_set = PySet::new_bound(py, &additional_vars)?;
            if input_dag_var_set
                .call_method1(intern!(py, "difference"), (var_set.clone(),))?
                .is_truthy()?
            {
                return Err(DAGCircuitError::new_err(format!(
                    "Cannot replace a node with a DAG with more variables. Variables in node: {:?}. Variables in dag: {:?}",
                    var_set.str(), input_dag_var_set.str(),
                )));
            }
            var_set
        } else {
            PySet::empty_bound(py)?
        };
        let bound_var_map = var_map.bind(py);
        for var in input_dag_var_set.iter() {
            bound_var_map.set_item(var.clone(), var)?;
        }

        for contracted_var in node_vars
            .call_method1(intern!(py, "difference"), (input_dag_var_set,))?
            .downcast::<PySet>()?
            .iter()
        {
            let pred = self
                .dag
                .edges_directed(node_index, Incoming)
                .find(|edge| {
                    if let Wire::Var(var) = edge.weight() {
                        contracted_var.eq(var).unwrap()
                    } else {
                        false
                    }
                })
                .unwrap();
            let succ = self
                .dag
                .edges_directed(node_index, Outgoing)
                .find(|edge| {
                    if let Wire::Var(var) = edge.weight() {
                        contracted_var.eq(var).unwrap()
                    } else {
                        false
                    }
                })
                .unwrap();
            self.dag.add_edge(
                pred.source(),
                succ.target(),
                Wire::Var(contracted_var.unbind()),
            );
        }

        let mut new_input_dag: Option<DAGCircuit> = None;
        // It doesn't make sense to try and propagate a condition from a control-flow op; a
        // replacement for the control-flow op should implement the operation completely.
        let node_map = if propagate_condition && !node.op.control_flow() {
            // Nested until https://github.com/rust-lang/rust/issues/53667 is fixed in a stable
            // release
            if let Some(condition) = node
                .extra_attrs
                .as_ref()
                .and_then(|attrs| attrs.condition.as_ref())
            {
                let mut in_dag = input_dag.copy_empty_like(py, "alike")?;
                // The remapping of `condition` below is still using the old code that assumes a 2-tuple.
                // This is because this remapping code only makes sense in the case of non-control-flow
                // operations being replaced.  These can only have the 2-tuple conditions, and the
                // ability to set a condition at an individual node level will be deprecated and removed
                // in favour of the new-style conditional blocks.  The extra logic in here to add
                // additional wires into the map as necessary would hugely complicate matters if we tried
                // to abstract it out into the `VariableMapper` used elsewhere.
                let wire_map = PyDict::new_bound(py);
                for (source_qubit, target_qubit) in &qubit_wire_map {
                    wire_map.set_item(
                        in_dag.qubits.get(*source_qubit).unwrap().clone_ref(py),
                        self.qubits.get(*target_qubit).unwrap().clone_ref(py),
                    )?
                }
                for (source_clbit, target_clbit) in &clbit_wire_map {
                    wire_map.set_item(
                        in_dag.clbits.get(*source_clbit).unwrap().clone_ref(py),
                        self.clbits.get(*target_clbit).unwrap().clone_ref(py),
                    )?
                }
                wire_map.update(var_map.bind(py).as_mapping())?;

                let reverse_wire_map = wire_map.iter().map(|(k, v)| (v, k)).into_py_dict_bound(py);
                let (py_target, py_value): (Bound<PyAny>, Bound<PyAny>) =
                    condition.bind(py).extract()?;
                let (py_new_target, target_cargs) =
                    if py_target.is_instance(imports::CLBIT.get_bound(py))? {
                        let new_target = reverse_wire_map
                            .get_item(&py_target)?
                            .map(Ok::<_, PyErr>)
                            .unwrap_or_else(|| {
                                // Target was not in node's wires, so we need a dummy.
                                let new_target = imports::CLBIT.get_bound(py).call0()?;
                                in_dag.add_clbit_unchecked(py, &new_target)?;
                                wire_map.set_item(&new_target, &py_target)?;
                                reverse_wire_map.set_item(&py_target, &new_target)?;
                                Ok(new_target)
                            })?;
                        (new_target.clone(), PySet::new_bound(py, &[new_target])?)
                    } else {
                        // ClassicalRegister
                        let target_bits: Vec<Bound<PyAny>> =
                            py_target.iter()?.collect::<PyResult<_>>()?;
                        let mapped_bits: Vec<Option<Bound<PyAny>>> = target_bits
                            .iter()
                            .map(|b| reverse_wire_map.get_item(b))
                            .collect::<PyResult<_>>()?;

                        let mut new_target = Vec::with_capacity(target_bits.len());
                        let target_cargs = PySet::empty_bound(py)?;
                        for (ours, theirs) in target_bits.into_iter().zip(mapped_bits) {
                            if let Some(theirs) = theirs {
                                // Target bit was in node's wires.
                                new_target.push(theirs.clone());
                                target_cargs.add(theirs)?;
                            } else {
                                // Target bit was not in node's wires, so we need a dummy.
                                let theirs = imports::CLBIT.get_bound(py).call0()?;
                                in_dag.add_clbit_unchecked(py, &theirs)?;
                                wire_map.set_item(&theirs, &ours)?;
                                reverse_wire_map.set_item(&ours, &theirs)?;
                                new_target.push(theirs.clone());
                                target_cargs.add(theirs)?;
                            }
                        }
                        let kwargs = [("bits", new_target.into_py(py))].into_py_dict_bound(py);
                        let new_target_register = imports::CLASSICAL_REGISTER
                            .get_bound(py)
                            .call((), Some(&kwargs))?;
                        in_dag.add_creg(py, &new_target_register)?;
                        (new_target_register, target_cargs)
                    };
                let new_condition = PyTuple::new_bound(py, [py_new_target, py_value]);

                qubit_wire_map.clear();
                clbit_wire_map.clear();
                for item in wire_map.items().iter() {
                    let (in_bit, self_bit): (Bound<PyAny>, Bound<PyAny>) = item.extract()?;
                    if in_bit.is_instance(imports::QUBIT.get_bound(py))? {
                        let in_index = in_dag.qubits.find(&in_bit).unwrap();
                        let self_index = self.qubits.find(&self_bit).unwrap();
                        qubit_wire_map.insert(in_index, self_index);
                    } else {
                        let in_index = in_dag.clbits.find(&in_bit).unwrap();
                        let self_index = self.clbits.find(&self_bit).unwrap();
                        clbit_wire_map.insert(in_index, self_index);
                    }
                }
                for in_node_index in input_dag.topological_op_nodes()? {
                    let in_node = &input_dag.dag[in_node_index];
                    if let NodeType::Operation(inst) = in_node {
                        if inst
                            .extra_attrs
                            .as_ref()
                            .and_then(|attrs| attrs.condition.as_ref())
                            .is_some()
                        {
                            return Err(DAGCircuitError::new_err(
                                "cannot propagate a condition to an element that already has one",
                            ));
                        }
                        let cargs = input_dag.cargs_interner.get(inst.clbits);
                        let cargs_bits: Vec<PyObject> = input_dag
                            .clbits
                            .map_indices(cargs)
                            .map(|x| x.clone_ref(py))
                            .collect();
                        if !target_cargs
                            .call_method1(intern!(py, "intersection"), (cargs_bits,))?
                            .downcast::<PySet>()?
                            .is_empty()
                        {
                            return Err(DAGCircuitError::new_err("cannot propagate a condition to an element that acts on those bits"));
                        }
                        let mut new_inst = inst.clone();
                        if new_condition.is_truthy()? {
                            if let Some(ref mut attrs) = new_inst.extra_attrs {
                                attrs.condition = Some(new_condition.as_any().clone().unbind());
                            } else {
                                new_inst.extra_attrs = Some(Box::new(ExtraInstructionAttributes {
                                    condition: Some(new_condition.as_any().clone().unbind()),
                                    label: None,
                                    duration: None,
                                    unit: None,
                                }));
                            }
                            #[cfg(feature = "cache_pygates")]
                            {
                                new_inst.py_op.take();
                            }
                        }
                        in_dag.push_back(py, new_inst)?;
                    }
                }
                let node_map = self.substitute_node_with_subgraph(
                    py,
                    node_index,
                    &in_dag,
                    &qubit_wire_map,
                    &clbit_wire_map,
                    &var_map,
                )?;
                new_input_dag = Some(in_dag);
                node_map
            } else {
                self.substitute_node_with_subgraph(
                    py,
                    node_index,
                    input_dag,
                    &qubit_wire_map,
                    &clbit_wire_map,
                    &var_map,
                )?
            }
        } else {
            self.substitute_node_with_subgraph(
                py,
                node_index,
                input_dag,
                &qubit_wire_map,
                &clbit_wire_map,
                &var_map,
            )?
        };
        self.global_phase = add_global_phase(py, &self.global_phase, &input_dag.global_phase)?;

        let wire_map_dict = PyDict::new_bound(py);
        for (source, target) in clbit_wire_map.iter() {
            let source_bit = match new_input_dag {
                Some(ref in_dag) => in_dag.clbits.get(*source),
                None => input_dag.clbits.get(*source),
            };
            let target_bit = self.clbits.get(*target);
            wire_map_dict.set_item(source_bit, target_bit)?;
        }
        let bound_var_map = var_map.bind(py);

        // Note: creating this list to hold new registers created by the mapper is a temporary
        // measure until qiskit.expr is ported to Rust. It is necessary because we cannot easily
        // have Python call back to DAGCircuit::add_creg while we're currently borrowing
        // the DAGCircuit.
        let new_registers = PyList::empty_bound(py);
        let add_new_register = new_registers.getattr("append")?.unbind();
        let flush_new_registers = |dag: &mut DAGCircuit| -> PyResult<()> {
            for reg in &new_registers {
                dag.add_creg(py, &reg)?;
            }
            new_registers.del_slice(0, new_registers.len())?;
            Ok(())
        };

        let variable_mapper = PyVariableMapper::new(
            py,
            self.cregs.bind(py).values().into_any(),
            Some(wire_map_dict),
            Some(bound_var_map.clone()),
            Some(add_new_register),
        )?;

        for (old_node_index, new_node_index) in node_map.iter() {
            let old_node = match new_input_dag {
                Some(ref in_dag) => &in_dag.dag[*old_node_index],
                None => &input_dag.dag[*old_node_index],
            };
            if let NodeType::Operation(old_inst) = old_node {
                if let OperationRef::Instruction(old_op) = old_inst.op.view() {
                    if old_op.name() == "switch_case" {
                        let raw_target = old_op.instruction.getattr(py, "target")?;
                        let target = raw_target.bind(py);
                        let kwargs = PyDict::new_bound(py);
                        kwargs.set_item(
                            "label",
                            old_inst
                                .extra_attrs
                                .as_ref()
                                .and_then(|attrs| attrs.label.as_ref()),
                        )?;
                        let new_op = imports::SWITCH_CASE_OP.get_bound(py).call(
                            (
                                variable_mapper.map_target(target)?,
                                old_op.instruction.call_method0(py, "cases_specifier")?,
                            ),
                            Some(&kwargs),
                        )?;
                        flush_new_registers(self)?;

                        if let NodeType::Operation(ref mut new_inst) =
                            &mut self.dag[*new_node_index]
                        {
                            new_inst.op = PyInstruction {
                                qubits: old_op.num_qubits(),
                                clbits: old_op.num_clbits(),
                                params: old_op.num_params(),
                                control_flow: old_op.control_flow(),
                                op_name: old_op.name().to_string(),
                                instruction: new_op.clone().unbind(),
                            }
                            .into();
                            #[cfg(feature = "cache_pygates")]
                            {
                                new_inst.py_op = new_op.unbind().into();
                            }
                        }
                    }
                }
                if let Some(condition) = old_inst
                    .extra_attrs
                    .as_ref()
                    .and_then(|attrs| attrs.condition.as_ref())
                {
                    if old_inst.op.name() != "switch_case" {
                        let new_condition: Option<PyObject> = variable_mapper
                            .map_condition(condition.bind(py), false)?
                            .extract()?;
                        flush_new_registers(self)?;

                        if let NodeType::Operation(ref mut new_inst) =
                            &mut self.dag[*new_node_index]
                        {
                            match &mut new_inst.extra_attrs {
                                Some(attrs) => attrs.condition.clone_from(&new_condition),
                                None => {
                                    new_inst.extra_attrs =
                                        Some(Box::new(ExtraInstructionAttributes {
                                            label: None,
                                            condition: new_condition.clone(),
                                            unit: None,
                                            duration: None,
                                        }))
                                }
                            }
                            #[cfg(feature = "cache_pygates")]
                            {
                                new_inst.py_op.take();
                            }
                            match new_inst.op.view() {
                                OperationRef::Instruction(py_inst) => {
                                    py_inst
                                        .instruction
                                        .setattr(py, "condition", new_condition)?;
                                }
                                OperationRef::Gate(py_gate) => {
                                    py_gate.gate.setattr(py, "condition", new_condition)?;
                                }
                                OperationRef::Operation(py_op) => {
                                    py_op.operation.setattr(py, "condition", new_condition)?;
                                }
                                OperationRef::Standard(_) => {}
                            }
                        }
                    }
                }
            }
        }
        let out_dict = PyDict::new_bound(py);
        for (old_index, new_index) in node_map {
            out_dict.set_item(old_index.index(), self.get_node(py, new_index)?)?;
        }
        Ok(out_dict.unbind())
    }

    /// Replace a DAGOpNode with a single operation. qargs, cargs and
    /// conditions for the new operation will be inferred from the node to be
    /// replaced. The new operation will be checked to match the shape of the
    /// replaced operation.
    ///
    /// Args:
    ///     node (DAGOpNode): Node to be replaced
    ///     op (qiskit.circuit.Operation): The :class:`qiskit.circuit.Operation`
    ///         instance to be added to the DAG
    ///     inplace (bool): Optional, default False. If True, existing DAG node
    ///         will be modified to include op. Otherwise, a new DAG node will
    ///         be used.
    ///     propagate_condition (bool): Optional, default True.  If True, a condition on the
    ///         ``node`` to be replaced will be applied to the new ``op``.  This is the legacy
    ///         behaviour.  If either node is a control-flow operation, this will be ignored.  If
    ///         the ``op`` already has a condition, :exc:`.DAGCircuitError` is raised.
    ///
    /// Returns:
    ///     DAGOpNode: the new node containing the added operation.
    ///
    /// Raises:
    ///     DAGCircuitError: If replacement operation was incompatible with
    ///     location of target node.
    #[pyo3(signature = (node, op, inplace=false, propagate_condition=true))]
    fn substitute_node(
        &mut self,
        node: &Bound<PyAny>,
        op: &Bound<PyAny>,
        inplace: bool,
        propagate_condition: bool,
    ) -> PyResult<Py<PyAny>> {
        let mut node: PyRefMut<DAGOpNode> = match node.downcast() {
            Ok(node) => node.borrow_mut(),
            Err(_) => return Err(DAGCircuitError::new_err("Only DAGOpNodes can be replaced.")),
        };
        let py = op.py();
        let node_index = node.as_ref().node.unwrap();
        // Extract information from node that is going to be replaced
        let old_packed = match self.dag.node_weight(node_index) {
            Some(NodeType::Operation(old_packed)) => old_packed.clone(),
            Some(_) => {
                return Err(DAGCircuitError::new_err(
                    "'node' must be of type 'DAGOpNode'.",
                ))
            }
            None => return Err(DAGCircuitError::new_err("'node' not found in DAG.")),
        };
        // Extract information from new op
        let new_op = op.extract::<OperationFromPython>()?;
        let current_wires: HashSet<Wire> = self
            .dag
            .edges(node_index)
            .map(|e| e.weight().clone())
            .collect();
        let mut new_wires: HashSet<Wire> = self
            .qargs_interner
            .get(old_packed.qubits)
            .iter()
            .map(|x| Wire::Qubit(*x))
            .chain(
                self.cargs_interner
                    .get(old_packed.clbits)
                    .iter()
                    .map(|x| Wire::Clbit(*x)),
            )
            .collect();
        let (additional_clbits, additional_vars) = self.additional_wires(
            py,
            new_op.operation.view(),
            new_op
                .extra_attrs
                .as_ref()
                .and_then(|attrs| attrs.condition.as_ref()),
        )?;
        new_wires.extend(additional_clbits.iter().map(|x| Wire::Clbit(*x)));
        new_wires.extend(additional_vars.iter().map(|x| Wire::Var(x.clone_ref(py))));

        if old_packed.op.num_qubits() != new_op.operation.num_qubits()
            || old_packed.op.num_clbits() != new_op.operation.num_clbits()
        {
            return Err(DAGCircuitError::new_err(
                format!(
                    "Cannot replace node of width ({} qubits, {} clbits) with operation of mismatched width ({} qubits, {} clbits)",
                    old_packed.op.num_qubits(), old_packed.op.num_clbits(), new_op.operation.num_qubits(), new_op.operation.num_clbits()
                )));
        }

        #[cfg(feature = "cache_pygates")]
        let mut py_op_cache = Some(op.clone().unbind());

        let mut extra_attrs = new_op.extra_attrs.clone();
        // If either operation is a control-flow operation, propagate_condition is ignored
        if propagate_condition
            && !(node.instruction.operation.control_flow() || new_op.operation.control_flow())
        {
            // if new_op has a condition, the condition can't be propagated from the old node
            if new_op
                .extra_attrs
                .as_ref()
                .and_then(|extra| extra.condition.as_ref())
                .is_some()
            {
                return Err(DAGCircuitError::new_err(
                    "Cannot propagate a condition to an operation that already has one.",
                ));
            }
            if let Some(old_condition) = old_packed.condition() {
                if matches!(new_op.operation.view(), OperationRef::Operation(_)) {
                    return Err(DAGCircuitError::new_err(
                        "Cannot add a condition on a generic Operation.",
                    ));
                }
                if let Some(ref mut extra) = extra_attrs {
                    extra.condition = Some(old_condition.clone_ref(py));
                } else {
                    extra_attrs = ExtraInstructionAttributes::new(
                        None,
                        None,
                        None,
                        Some(old_condition.clone_ref(py)),
                    )
                    .map(Box::new)
                }
                let binding = self
                    .control_flow_module
                    .condition_resources(old_condition.bind(py))?;
                let condition_clbits = binding.clbits.bind(py);
                for bit in condition_clbits {
                    new_wires.insert(Wire::Clbit(self.clbits.find(&bit).unwrap()));
                }
                let op_ref = new_op.operation.view();
                if let OperationRef::Instruction(inst) = op_ref {
                    inst.instruction
                        .bind(py)
                        .setattr(intern!(py, "condition"), old_condition)?;
                } else if let OperationRef::Gate(gate) = op_ref {
                    gate.gate.bind(py).call_method1(
                        intern!(py, "c_if"),
                        old_condition.downcast_bound::<PyTuple>(py)?,
                    )?;
                }
                #[cfg(feature = "cache_pygates")]
                {
                    py_op_cache = None;
                }
            }
        };
        if new_wires != current_wires {
            // The new wires must be a non-strict subset of the current wires; if they add new
            // wires, we'd not know where to cut the existing wire to insert the new dependency.
            return Err(DAGCircuitError::new_err(format!(
                "New operation '{:?}' does not span the same wires as the old node '{:?}'. New wires: {:?}, old_wires: {:?}.", op.str(), old_packed.op.view(), new_wires, current_wires
            )));
        }

        if inplace {
            node.instruction.operation = new_op.operation.clone();
            node.instruction.params = new_op.params.clone();
            node.instruction.extra_attrs = extra_attrs.clone();
            #[cfg(feature = "cache_pygates")]
            {
                node.instruction.py_op = py_op_cache
                    .as_ref()
                    .map(|ob| OnceCell::from(ob.clone_ref(py)))
                    .unwrap_or_default();
            }
        }
        // Clone op data, as it will be moved into the PackedInstruction
        let new_weight = NodeType::Operation(PackedInstruction {
            op: new_op.operation.clone(),
            qubits: old_packed.qubits,
            clbits: old_packed.clbits,
            params: (!new_op.params.is_empty()).then(|| new_op.params.into()),
            extra_attrs,
            #[cfg(feature = "cache_pygates")]
            py_op: py_op_cache.map(OnceCell::from).unwrap_or_default(),
        });
        let node_index = node.as_ref().node.unwrap();
        if let Some(weight) = self.dag.node_weight_mut(node_index) {
            *weight = new_weight;
        }

        // Update self.op_names
        self.decrement_op(old_packed.op.name());
        self.increment_op(new_op.operation.name());

        if inplace {
            Ok(node.into_py(py))
        } else {
            self.get_node(py, node_index)
        }
    }

    /// Decompose the circuit into sets of qubits with no gates connecting them.
    ///
    /// Args:
    ///     remove_idle_qubits (bool): Flag denoting whether to remove idle qubits from
    ///         the separated circuits. If ``False``, each output circuit will contain the
    ///         same number of qubits as ``self``.
    ///
    /// Returns:
    ///     List[DAGCircuit]: The circuits resulting from separating ``self`` into sets
    ///         of disconnected qubits
    ///
    /// Each :class:`~.DAGCircuit` instance returned by this method will contain the same number of
    /// clbits as ``self``. The global phase information in ``self`` will not be maintained
    /// in the subcircuits returned by this method.
    #[pyo3(signature = (remove_idle_qubits=false, *, vars_mode="alike"))]
    fn separable_circuits(
        &self,
        py: Python,
        remove_idle_qubits: bool,
        vars_mode: &str,
    ) -> PyResult<Py<PyList>> {
        let connected_components = rustworkx_core::connectivity::connected_components(&self.dag);
        let dags = PyList::empty_bound(py);

        for comp_nodes in connected_components.iter() {
            let mut new_dag = self.copy_empty_like(py, vars_mode)?;
            new_dag.global_phase = Param::Float(0.);

            // A map from nodes in the this DAGCircuit to nodes in the new dag. Used for adding edges
            let mut node_map: HashMap<NodeIndex, NodeIndex> =
                HashMap::with_capacity(comp_nodes.len());

            // Adding the nodes to the new dag
            let mut non_classical = false;
            for node in comp_nodes {
                match self.dag.node_weight(*node) {
                    Some(w) => match w {
                        NodeType::ClbitIn(b) => {
                            let clbit_in = new_dag.clbit_io_map[b.0 as usize][0];
                            node_map.insert(*node, clbit_in);
                        }
                        NodeType::ClbitOut(b) => {
                            let clbit_out = new_dag.clbit_io_map[b.0 as usize][1];
                            node_map.insert(*node, clbit_out);
                        }
                        NodeType::QubitIn(q) => {
                            let qbit_in = new_dag.qubit_io_map[q.0 as usize][0];
                            node_map.insert(*node, qbit_in);
                            non_classical = true;
                        }
                        NodeType::QubitOut(q) => {
                            let qbit_out = new_dag.qubit_io_map[q.0 as usize][1];
                            node_map.insert(*node, qbit_out);
                            non_classical = true;
                        }
                        NodeType::VarIn(v) => {
                            let var_in = new_dag.var_input_map.get(py, v).unwrap();
                            node_map.insert(*node, var_in);
                        }
                        NodeType::VarOut(v) => {
                            let var_out = new_dag.var_output_map.get(py, v).unwrap();
                            node_map.insert(*node, var_out);
                        }
                        NodeType::Operation(pi) => {
                            let new_node = new_dag.dag.add_node(NodeType::Operation(pi.clone()));
                            new_dag.increment_op(pi.op.name());
                            node_map.insert(*node, new_node);
                            non_classical = true;
                        }
                    },
                    None => panic!("DAG node without payload!"),
                }
            }
            if !non_classical {
                continue;
            }
            let node_filter = |node: NodeIndex| -> bool { node_map.contains_key(&node) };

            let filtered = NodeFiltered(&self.dag, node_filter);

            // Remove the edges added by copy_empty_like (as idle wires) to avoid duplication
            new_dag.dag.clear_edges();
            for edge in filtered.edge_references() {
                let new_source = node_map[&edge.source()];
                let new_target = node_map[&edge.target()];
                new_dag
                    .dag
                    .add_edge(new_source, new_target, edge.weight().clone());
            }
            // Add back any edges for idle wires
            for (qubit, [in_node, out_node]) in new_dag
                .qubit_io_map
                .iter()
                .enumerate()
                .map(|(idx, indices)| (Qubit(idx as u32), indices))
            {
                if new_dag.dag.edges(*in_node).next().is_none() {
                    new_dag
                        .dag
                        .add_edge(*in_node, *out_node, Wire::Qubit(qubit));
                }
            }
            for (clbit, [in_node, out_node]) in new_dag
                .clbit_io_map
                .iter()
                .enumerate()
                .map(|(idx, indices)| (Clbit(idx as u32), indices))
            {
                if new_dag.dag.edges(*in_node).next().is_none() {
                    new_dag
                        .dag
                        .add_edge(*in_node, *out_node, Wire::Clbit(clbit));
                }
            }
            for (var, in_node) in new_dag.var_input_map.iter(py) {
                if new_dag.dag.edges(in_node).next().is_none() {
                    let out_node = new_dag.var_output_map.get(py, &var).unwrap();
                    new_dag
                        .dag
                        .add_edge(in_node, out_node, Wire::Var(var.clone_ref(py)));
                }
            }
            if remove_idle_qubits {
                let idle_wires: Vec<Bound<PyAny>> = new_dag
                    .idle_wires(py, None)?
                    .into_bound(py)
                    .map(|q| q.unwrap())
                    .filter(|e| e.is_instance(imports::QUBIT.get_bound(py)).unwrap())
                    .collect();

                let qubits = PyTuple::new_bound(py, idle_wires);
                new_dag.remove_qubits(py, &qubits)?; // TODO: this does not really work, some issue with remove_qubits itself
            }

            dags.append(pyo3::Py::new(py, new_dag)?)?;
        }

        Ok(dags.unbind())
    }

    /// Swap connected nodes e.g. due to commutation.
    ///
    /// Args:
    ///     node1 (OpNode): predecessor node
    ///     node2 (OpNode): successor node
    ///
    /// Raises:
    ///     DAGCircuitError: if either node is not an OpNode or nodes are not connected
    fn swap_nodes(&mut self, node1: &DAGNode, node2: &DAGNode) -> PyResult<()> {
        let node1 = node1.node.unwrap();
        let node2 = node2.node.unwrap();

        // Check that both nodes correspond to operations
        if !matches!(self.dag.node_weight(node1).unwrap(), NodeType::Operation(_))
            || !matches!(self.dag.node_weight(node2).unwrap(), NodeType::Operation(_))
        {
            return Err(DAGCircuitError::new_err(
                "Nodes to swap are not both DAGOpNodes",
            ));
        }

        // Gather all wires connecting node1 and node2.
        // This functionality was extracted from rustworkx's 'get_edge_data'
        let wires: Vec<Wire> = self
            .dag
            .edges(node1)
            .filter(|edge| edge.target() == node2)
            .map(|edge| edge.weight().clone())
            .collect();

        if wires.is_empty() {
            return Err(DAGCircuitError::new_err(
                "Attempt to swap unconnected nodes",
            ));
        };

        // Closure that finds the first parent/child node connected to a reference node by given wire
        // and returns relevant edge information depending on the specified direction:
        //  - Incoming -> parent -> outputs (parent_edge_id, parent_source_node_id)
        //  - Outgoing -> child -> outputs (child_edge_id, child_target_node_id)
        // This functionality was inspired in rustworkx's 'find_predecessors_by_edge' and 'find_successors_by_edge'.
        let directed_edge_for_wire = |node: NodeIndex, direction: Direction, wire: &Wire| {
            for edge in self.dag.edges_directed(node, direction) {
                if wire == edge.weight() {
                    match direction {
                        Incoming => return Some((edge.id(), edge.source())),
                        Outgoing => return Some((edge.id(), edge.target())),
                    }
                }
            }
            None
        };

        // Vector that contains a tuple of (wire, edge_info, parent_info, child_info) per wire in wires
        let relevant_edges = wires
            .iter()
            .rev()
            .map(|wire| {
                (
                    wire,
                    directed_edge_for_wire(node1, Outgoing, wire).unwrap(),
                    directed_edge_for_wire(node1, Incoming, wire).unwrap(),
                    directed_edge_for_wire(node2, Outgoing, wire).unwrap(),
                )
            })
            .collect::<Vec<_>>();

        // Iterate over relevant edges and modify self.dag
        for (wire, (node1_to_node2, _), (parent_to_node1, parent), (node2_to_child, child)) in
            relevant_edges
        {
            self.dag.remove_edge(parent_to_node1);
            self.dag.add_edge(parent, node2, wire.clone());
            self.dag.remove_edge(node1_to_node2);
            self.dag.add_edge(node2, node1, wire.clone());
            self.dag.remove_edge(node2_to_child);
            self.dag.add_edge(node1, child, wire.clone());
        }
        Ok(())
    }

    /// Get the node in the dag.
    ///
    /// Args:
    ///     node_id(int): Node identifier.
    ///
    /// Returns:
    ///     node: the node.
    fn node(&self, py: Python, node_id: isize) -> PyResult<Py<PyAny>> {
        self.get_node(py, NodeIndex::new(node_id as usize))
    }

    /// Iterator for node values.
    ///
    /// Yield:
    ///     node: the node.
    fn nodes(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let result: PyResult<Vec<_>> = self
            .dag
            .node_references()
            .map(|(node, weight)| self.unpack_into(py, node, weight))
            .collect();
        let tup = PyTuple::new_bound(py, result?);
        Ok(tup.into_any().iter().unwrap().unbind())
    }

    /// Iterator for edge values with source and destination node.
    ///
    /// This works by returning the outgoing edges from the specified nodes. If
    /// no nodes are specified all edges from the graph are returned.
    ///
    /// Args:
    ///     nodes(DAGOpNode, DAGInNode, or DAGOutNode|list(DAGOpNode, DAGInNode, or DAGOutNode):
    ///         Either a list of nodes or a single input node. If none is specified,
    ///         all edges are returned from the graph.
    ///
    /// Yield:
    ///     edge: the edge as a tuple with the format
    ///         (source node, destination node, edge wire)
    fn edges(&self, nodes: Option<Bound<PyAny>>, py: Python) -> PyResult<Py<PyIterator>> {
        let get_node_index = |obj: &Bound<PyAny>| -> PyResult<NodeIndex> {
            Ok(obj.downcast::<DAGNode>()?.borrow().node.unwrap())
        };

        let actual_nodes: Vec<_> = match nodes {
            None => self.dag.node_indices().collect(),
            Some(nodes) => {
                let mut out = Vec::new();
                if let Ok(node) = get_node_index(&nodes) {
                    out.push(node);
                } else {
                    for node in nodes.iter()? {
                        out.push(get_node_index(&node?)?);
                    }
                }
                out
            }
        };

        let mut edges = Vec::new();
        for node in actual_nodes {
            for edge in self.dag.edges_directed(node, Outgoing) {
                edges.push((
                    self.get_node(py, edge.source())?,
                    self.get_node(py, edge.target())?,
                    match edge.weight() {
                        Wire::Qubit(qubit) => self.qubits.get(*qubit).unwrap(),
                        Wire::Clbit(clbit) => self.clbits.get(*clbit).unwrap(),
                        Wire::Var(var) => var,
                    },
                ))
            }
        }

        Ok(PyTuple::new_bound(py, edges)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Get the list of "op" nodes in the dag.
    ///
    /// Args:
    ///     op (Type): :class:`qiskit.circuit.Operation` subclass op nodes to
    ///         return. If None, return all op nodes.
    ///     include_directives (bool): include `barrier`, `snapshot` etc.
    ///
    /// Returns:
    ///     list[DAGOpNode]: the list of dag nodes containing the given op.
    #[pyo3(name= "op_nodes", signature=(op=None, include_directives=true))]
    fn py_op_nodes(
        &self,
        py: Python,
        op: Option<&Bound<PyType>>,
        include_directives: bool,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let mut nodes = Vec::new();
        let filter_is_nonstandard = if let Some(op) = op {
            op.getattr(intern!(py, "_standard_gate")).ok().is_none()
        } else {
            true
        };
        for (node, weight) in self.dag.node_references() {
            if let NodeType::Operation(packed) = &weight {
                if !include_directives && packed.op.directive() {
                    continue;
                }
                if let Some(op_type) = op {
                    // This middle catch is to avoid Python-space operation creation for most uses of
                    // `op`; we're usually just looking for control-flow ops, and standard gates
                    // aren't control-flow ops.
                    if !(filter_is_nonstandard && packed.op.try_standard_gate().is_some())
                        && packed.op.py_op_is_instance(op_type)?
                    {
                        nodes.push(self.unpack_into(py, node, weight)?);
                    }
                } else {
                    nodes.push(self.unpack_into(py, node, weight)?);
                }
            }
        }
        Ok(nodes)
    }

    /// Get a list of "op" nodes in the dag that contain control flow instructions.
    ///
    /// Returns:
    ///     list[DAGOpNode] | None: The list of dag nodes containing control flow ops. If there
    ///         are no control flow nodes None is returned
    fn control_flow_op_nodes(&self, py: Python) -> PyResult<Option<Vec<Py<PyAny>>>> {
        if self.has_control_flow() {
            let result: PyResult<Vec<Py<PyAny>>> = self
                .dag
                .node_references()
                .filter_map(|(node_index, node_type)| match node_type {
                    NodeType::Operation(ref node) => {
                        if node.op.control_flow() {
                            Some(self.unpack_into(py, node_index, node_type))
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect();
            Ok(Some(result?))
        } else {
            Ok(None)
        }
    }

    /// Get the list of gate nodes in the dag.
    ///
    /// Returns:
    ///     list[DAGOpNode]: the list of DAGOpNodes that represent gates.
    fn gate_nodes(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        self.dag
            .node_references()
            .filter_map(|(node, weight)| match weight {
                NodeType::Operation(ref packed) => match packed.op.view() {
                    OperationRef::Gate(_) | OperationRef::Standard(_) => {
                        Some(self.unpack_into(py, node, weight))
                    }
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    /// Get the set of "op" nodes with the given name.
    #[pyo3(signature = (*names))]
    fn named_nodes(&self, py: Python<'_>, names: &Bound<PyTuple>) -> PyResult<Vec<Py<PyAny>>> {
        let mut names_set: HashSet<String> = HashSet::with_capacity(names.len());
        for name_obj in names.iter() {
            names_set.insert(name_obj.extract::<String>()?);
        }
        let mut result: Vec<Py<PyAny>> = Vec::new();
        for (id, weight) in self.dag.node_references() {
            if let NodeType::Operation(ref packed) = weight {
                if names_set.contains(packed.op.name()) {
                    result.push(self.unpack_into(py, id, weight)?);
                }
            }
        }
        Ok(result)
    }

    /// Get list of 2 qubit operations. Ignore directives like snapshot and barrier.
    #[pyo3(name = "two_qubit_ops")]
    pub fn py_two_qubit_ops(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let mut nodes = Vec::new();
        for node in self.two_qubit_ops() {
            let weight = self.dag.node_weight(node).expect("NodeIndex in graph");
            nodes.push(self.unpack_into(py, node, weight)?);
        }
        Ok(nodes)
    }

    /// Get list of 3+ qubit operations. Ignore directives like snapshot and barrier.
    fn multi_qubit_ops(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let mut nodes = Vec::new();
        for (node, weight) in self.dag.node_references() {
            if let NodeType::Operation(ref packed) = weight {
                if packed.op.directive() {
                    continue;
                }

                let qargs = self.qargs_interner.get(packed.qubits);
                if qargs.len() >= 3 {
                    nodes.push(self.unpack_into(py, node, weight)?);
                }
            }
        }
        Ok(nodes)
    }

    /// Returns the longest path in the dag as a list of DAGOpNodes, DAGInNodes, and DAGOutNodes.
    fn longest_path(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let weight_fn = |_| -> Result<usize, Infallible> { Ok(1) };
        match rustworkx_core::dag_algo::longest_path(&self.dag, weight_fn).unwrap() {
            Some(res) => res.0,
            None => return Err(DAGCircuitError::new_err("not a DAG")),
        }
        .into_iter()
        .map(|node_index| self.get_node(py, node_index))
        .collect()
    }

    /// Returns iterator of the successors of a node as DAGOpNodes and DAGOutNodes."""
    fn successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let successors: PyResult<Vec<_>> = self
            .dag
            .neighbors_directed(node.node.unwrap(), Outgoing)
            .unique()
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new_bound(py, successors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of the predecessors of a node as DAGOpNodes and DAGInNodes.
    fn predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .dag
            .neighbors_directed(node.node.unwrap(), Incoming)
            .unique()
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new_bound(py, predecessors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of "op" successors of a node in the dag.
    fn op_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .dag
            .neighbors_directed(node.node.unwrap(), Outgoing)
            .unique()
            .filter_map(|i| match self.dag[i] {
                NodeType::Operation(_) => Some(self.get_node(py, i)),
                _ => None,
            })
            .collect();
        Ok(PyTuple::new_bound(py, predecessors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Returns the iterator of "op" predecessors of a node in the dag.
    fn op_predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .dag
            .neighbors_directed(node.node.unwrap(), Incoming)
            .unique()
            .filter_map(|i| match self.dag[i] {
                NodeType::Operation(_) => Some(self.get_node(py, i)),
                _ => None,
            })
            .collect();
        Ok(PyTuple::new_bound(py, predecessors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Checks if a second node is in the successors of node.
    fn is_successor(&self, node: &DAGNode, node_succ: &DAGNode) -> bool {
        self.dag
            .find_edge(node.node.unwrap(), node_succ.node.unwrap())
            .is_some()
    }

    /// Checks if a second node is in the predecessors of node.
    fn is_predecessor(&self, node: &DAGNode, node_pred: &DAGNode) -> bool {
        self.dag
            .find_edge(node_pred.node.unwrap(), node.node.unwrap())
            .is_some()
    }

    /// Returns iterator of the predecessors of a node that are
    /// connected by a quantum edge as DAGOpNodes and DAGInNodes.
    #[pyo3(name = "quantum_predecessors")]
    fn py_quantum_predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let predecessors: PyResult<Vec<_>> = self
            .quantum_predecessors(node.node.unwrap())
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new_bound(py, predecessors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of the successors of a node that are
    /// connected by a quantum edge as DAGOpNodes and DAGOutNodes.
    #[pyo3(name = "quantum_successors")]
    fn py_quantum_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let successors: PyResult<Vec<_>> = self
            .quantum_successors(node.node.unwrap())
            .map(|i| self.get_node(py, i))
            .collect();
        Ok(PyTuple::new_bound(py, successors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Returns iterator of the predecessors of a node that are
    /// connected by a classical edge as DAGOpNodes and DAGInNodes.
    fn classical_predecessors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let edges = self.dag.edges_directed(node.node.unwrap(), Incoming);
        let filtered = edges.filter_map(|e| match e.weight() {
            Wire::Qubit(_) => None,
            _ => Some(e.source()),
        });
        let predecessors: PyResult<Vec<_>> =
            filtered.unique().map(|i| self.get_node(py, i)).collect();
        Ok(PyTuple::new_bound(py, predecessors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Returns set of the ancestors of a node as DAGOpNodes and DAGInNodes.
    #[pyo3(name = "ancestors")]
    fn py_ancestors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PySet>> {
        let ancestors: PyResult<Vec<PyObject>> = self
            .ancestors(node.node.unwrap())
            .map(|node| self.get_node(py, node))
            .collect();
        Ok(PySet::new_bound(py, &ancestors?)?.unbind())
    }

    /// Returns set of the descendants of a node as DAGOpNodes and DAGOutNodes.
    #[pyo3(name = "descendants")]
    fn py_descendants(&self, py: Python, node: &DAGNode) -> PyResult<Py<PySet>> {
        let descendants: PyResult<Vec<PyObject>> = self
            .descendants(node.node.unwrap())
            .map(|node| self.get_node(py, node))
            .collect();
        Ok(PySet::new_bound(py, &descendants?)?.unbind())
    }

    /// Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
    /// and [DAGNode] is its successors in  BFS order.
    #[pyo3(name = "bfs_successors")]
    fn py_bfs_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let successor_index: PyResult<Vec<(PyObject, Vec<PyObject>)>> = self
            .bfs_successors(node.node.unwrap())
            .map(|(node, nodes)| -> PyResult<(PyObject, Vec<PyObject>)> {
                Ok((
                    self.get_node(py, node)?,
                    nodes
                        .iter()
                        .map(|sub_node| self.get_node(py, *sub_node))
                        .collect::<PyResult<Vec<_>>>()?,
                ))
            })
            .collect();
        Ok(PyList::new_bound(py, successor_index?)
            .into_any()
            .iter()?
            .unbind())
    }

    /// Returns iterator of the successors of a node that are
    /// connected by a classical edge as DAGOpNodes and DAGOutNodes.
    fn classical_successors(&self, py: Python, node: &DAGNode) -> PyResult<Py<PyIterator>> {
        let edges = self.dag.edges_directed(node.node.unwrap(), Outgoing);
        let filtered = edges.filter_map(|e| match e.weight() {
            Wire::Qubit(_) => None,
            _ => Some(e.target()),
        });
        let predecessors: PyResult<Vec<_>> =
            filtered.unique().map(|i| self.get_node(py, i)).collect();
        Ok(PyTuple::new_bound(py, predecessors?)
            .into_any()
            .iter()
            .unwrap()
            .unbind())
    }

    /// Remove an operation node n.
    ///
    /// Add edges from predecessors to successors.
    #[pyo3(name = "remove_op_node")]
    fn py_remove_op_node(&mut self, node: &Bound<PyAny>) -> PyResult<()> {
        let node: PyRef<DAGOpNode> = match node.downcast::<DAGOpNode>() {
            Ok(node) => node.borrow(),
            Err(_) => return Err(DAGCircuitError::new_err("Node not an DAGOpNode")),
        };
        let index = node.as_ref().node.unwrap();
        if self.dag.node_weight(index).is_none() {
            return Err(DAGCircuitError::new_err("Node not in DAG"));
        }
        self.remove_op_node(index);
        Ok(())
    }

    /// Remove all of the ancestor operation nodes of node.
    fn remove_ancestors_of(&mut self, node: &DAGNode) -> PyResult<()> {
        let ancestors: Vec<_> = core_ancestors(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && matches!(self.dag.node_weight(*next), Some(NodeType::Operation(_)))
            })
            .collect();
        for a in ancestors {
            self.dag.remove_node(a);
        }
        Ok(())
    }

    /// Remove all of the descendant operation nodes of node.
    fn remove_descendants_of(&mut self, node: &DAGNode) -> PyResult<()> {
        let descendants: Vec<_> = core_descendants(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && matches!(self.dag.node_weight(*next), Some(NodeType::Operation(_)))
            })
            .collect();
        for d in descendants {
            self.dag.remove_node(d);
        }
        Ok(())
    }

    /// Remove all of the non-ancestors operation nodes of node.
    fn remove_nonancestors_of(&mut self, node: &DAGNode) -> PyResult<()> {
        let ancestors: HashSet<_> = core_ancestors(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && matches!(self.dag.node_weight(*next), Some(NodeType::Operation(_)))
            })
            .collect();
        let non_ancestors: Vec<_> = self
            .dag
            .node_indices()
            .filter(|node_id| !ancestors.contains(node_id))
            .collect();
        for na in non_ancestors {
            self.dag.remove_node(na);
        }
        Ok(())
    }

    /// Remove all of the non-descendants operation nodes of node.
    fn remove_nondescendants_of(&mut self, node: &DAGNode) -> PyResult<()> {
        let descendants: HashSet<_> = core_descendants(&self.dag, node.node.unwrap())
            .filter(|next| {
                next != &node.node.unwrap()
                    && matches!(self.dag.node_weight(*next), Some(NodeType::Operation(_)))
            })
            .collect();
        let non_descendants: Vec<_> = self
            .dag
            .node_indices()
            .filter(|node_id| !descendants.contains(node_id))
            .collect();
        for nd in non_descendants {
            self.dag.remove_node(nd);
        }
        Ok(())
    }

    /// Return a list of op nodes in the first layer of this dag.
    #[pyo3(name = "front_layer")]
    fn py_front_layer(&self, py: Python) -> PyResult<Py<PyList>> {
        let native_front_layer = self.front_layer(py);
        let front_layer_list = PyList::empty_bound(py);
        for node in native_front_layer {
            front_layer_list.append(self.get_node(py, node)?)?;
        }
        Ok(front_layer_list.into())
    }

    /// Yield a shallow view on a layer of this DAGCircuit for all d layers of this circuit.
    ///
    /// A layer is a circuit whose gates act on disjoint qubits, i.e.,
    /// a layer has depth 1. The total number of layers equals the
    /// circuit depth d. The layers are indexed from 0 to d-1 with the
    /// earliest layer at index 0. The layers are constructed using a
    /// greedy algorithm. Each returned layer is a dict containing
    /// {"graph": circuit graph, "partition": list of qubit lists}.
    ///
    /// The returned layer contains new (but semantically equivalent) DAGOpNodes, DAGInNodes,
    /// and DAGOutNodes. These are not the same as nodes of the original dag, but are equivalent
    /// via DAGNode.semantic_eq(node1, node2).
    ///
    /// TODO: Gates that use the same cbits will end up in different
    /// layers as this is currently implemented. This may not be
    /// the desired behavior.
    #[pyo3(signature = (*, vars_mode="captures"))]
    fn layers(&self, py: Python, vars_mode: &str) -> PyResult<Py<PyIterator>> {
        let layer_list = PyList::empty_bound(py);
        let mut graph_layers = self.multigraph_layers(py);
        if graph_layers.next().is_none() {
            return Ok(PyIterator::from_bound_object(&layer_list)?.into());
        }

        for graph_layer in graph_layers {
            let layer_dict = PyDict::new_bound(py);
            // Sort to make sure they are in the order they were added to the original DAG
            // It has to be done by node_id as graph_layer is just a list of nodes
            // with no implied topology
            // Drawing tools rely on _node_id to infer order of node creation
            // so we need this to be preserved by layers()
            // Get the op nodes from the layer, removing any input and output nodes.
            let mut op_nodes: Vec<(&PackedInstruction, &NodeIndex)> = graph_layer
                .iter()
                .filter_map(|node| self.dag.node_weight(*node).map(|dag_node| (dag_node, node)))
                .filter_map(|(node, index)| match node {
                    NodeType::Operation(oper) => Some((oper, index)),
                    _ => None,
                })
                .collect();
            op_nodes.sort_by_key(|(_, node_index)| **node_index);

            if op_nodes.is_empty() {
                return Ok(PyIterator::from_bound_object(&layer_list)?.into());
            }

            let mut new_layer = self.copy_empty_like(py, vars_mode)?;

            new_layer.extend(py, op_nodes.iter().map(|(inst, _)| (*inst).clone()))?;

            let new_layer_op_nodes = new_layer.op_nodes(false).filter_map(|node_index| {
                match new_layer.dag.node_weight(node_index) {
                    Some(NodeType::Operation(ref node)) => Some(node),
                    _ => None,
                }
            });
            let support_iter = new_layer_op_nodes.into_iter().map(|node| {
                PyTuple::new_bound(
                    py,
                    new_layer
                        .qubits
                        .map_indices(new_layer.qargs_interner.get(node.qubits)),
                )
            });
            let support_list = PyList::empty_bound(py);
            for support_qarg in support_iter {
                support_list.append(support_qarg)?;
            }
            layer_dict.set_item("graph", new_layer.into_py(py))?;
            layer_dict.set_item("partition", support_list)?;
            layer_list.append(layer_dict)?;
        }
        Ok(layer_list.into_any().iter()?.into())
    }

    /// Yield a layer for all gates of this circuit.
    ///
    /// A serial layer is a circuit with one gate. The layers have the
    /// same structure as in layers().
    #[pyo3(signature = (*, vars_mode="captures"))]
    fn serial_layers(&self, py: Python, vars_mode: &str) -> PyResult<Py<PyIterator>> {
        let layer_list = PyList::empty_bound(py);
        for next_node in self.topological_op_nodes()? {
            let retrieved_node: &PackedInstruction = match self.dag.node_weight(next_node) {
                Some(NodeType::Operation(node)) => node,
                _ => unreachable!("A non-operation node was obtained from topological_op_nodes."),
            };
            let mut new_layer = self.copy_empty_like(py, vars_mode)?;

            // Save the support of the operation we add to the layer
            let support_list = PyList::empty_bound(py);
            let qubits = PyTuple::new_bound(
                py,
                self.qargs_interner
                    .get(retrieved_node.qubits)
                    .iter()
                    .map(|qubit| self.qubits.get(*qubit)),
            )
            .unbind();
            new_layer.push_back(py, retrieved_node.clone())?;

            if !retrieved_node.op.directive() {
                support_list.append(qubits)?;
            }

            let layer_dict = [
                ("graph", new_layer.into_py(py)),
                ("partition", support_list.into_any().unbind()),
            ]
            .into_py_dict_bound(py);
            layer_list.append(layer_dict)?;
        }

        Ok(layer_list.into_any().iter()?.into())
    }

    /// Yield layers of the multigraph.
    #[pyo3(name = "multigraph_layers")]
    fn py_multigraph_layers(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let graph_layers = self.multigraph_layers(py).map(|layer| -> Vec<PyObject> {
            layer
                .into_iter()
                .filter_map(|index| self.get_node(py, index).ok())
                .collect()
        });
        let list: Bound<PyList> =
            PyList::new_bound(py, graph_layers.collect::<Vec<Vec<PyObject>>>());
        Ok(PyIterator::from_bound_object(&list)?.unbind())
    }

    /// Return a set of non-conditional runs of "op" nodes with the given names.
    ///
    /// For example, "... h q[0]; cx q[0],q[1]; cx q[0],q[1]; h q[1]; .."
    /// would produce the tuple of cx nodes as an element of the set returned
    /// from a call to collect_runs(["cx"]). If instead the cx nodes were
    /// "cx q[0],q[1]; cx q[1],q[0];", the method would still return the
    /// pair in a tuple. The namelist can contain names that are not
    /// in the circuit's basis.
    ///
    /// Nodes must have only one successor to continue the run.
    #[pyo3(name = "collect_runs")]
    fn py_collect_runs(&self, py: Python, namelist: &Bound<PyList>) -> PyResult<Py<PySet>> {
        let mut name_list_set = HashSet::with_capacity(namelist.len());
        for name in namelist.iter() {
            name_list_set.insert(name.extract::<String>()?);
        }
        match self.collect_runs(name_list_set) {
            Some(runs) => {
                let run_iter = runs.map(|node_indices| {
                    PyTuple::new_bound(
                        py,
                        node_indices
                            .into_iter()
                            .map(|node_index| self.get_node(py, node_index).unwrap()),
                    )
                    .unbind()
                });
                let out_set = PySet::empty_bound(py)?;
                for run_tuple in run_iter {
                    out_set.add(run_tuple)?;
                }
                Ok(out_set.unbind())
            }
            None => Err(PyRuntimeError::new_err(
                "Invalid DAGCircuit, cycle encountered",
            )),
        }
    }

    /// Return a set of non-conditional runs of 1q "op" nodes.
    #[pyo3(name = "collect_1q_runs")]
    fn py_collect_1q_runs(&self, py: Python) -> PyResult<Py<PyList>> {
        match self.collect_1q_runs() {
            Some(runs) => {
                let runs_iter = runs.map(|node_indices| {
                    PyList::new_bound(
                        py,
                        node_indices
                            .into_iter()
                            .map(|node_index| self.get_node(py, node_index).unwrap()),
                    )
                    .unbind()
                });
                let out_list = PyList::empty_bound(py);
                for run_list in runs_iter {
                    out_list.append(run_list)?;
                }
                Ok(out_list.unbind())
            }
            None => Err(PyRuntimeError::new_err(
                "Invalid DAGCircuit, cycle encountered",
            )),
        }
    }

    /// Return a set of non-conditional runs of 2q "op" nodes.
    #[pyo3(name = "collect_2q_runs")]
    fn py_collect_2q_runs(&self, py: Python) -> PyResult<Py<PyList>> {
        match self.collect_2q_runs() {
            Some(runs) => {
                let runs_iter = runs.into_iter().map(|node_indices| {
                    PyList::new_bound(
                        py,
                        node_indices
                            .into_iter()
                            .map(|node_index| self.get_node(py, node_index).unwrap()),
                    )
                    .unbind()
                });
                let out_list = PyList::empty_bound(py);
                for run_list in runs_iter {
                    out_list.append(run_list)?;
                }
                Ok(out_list.unbind())
            }
            None => Err(PyRuntimeError::new_err(
                "Invalid DAGCircuit, cycle encountered",
            )),
        }
    }

    /// Iterator for nodes that affect a given wire.
    ///
    /// Args:
    ///     wire (Bit): the wire to be looked at.
    ///     only_ops (bool): True if only the ops nodes are wanted;
    ///                 otherwise, all nodes are returned.
    /// Yield:
    ///      Iterator: the successive nodes on the given wire
    ///
    /// Raises:
    ///     DAGCircuitError: if the given wire doesn't exist in the DAG
    #[pyo3(name = "nodes_on_wire", signature = (wire, only_ops=false))]
    fn py_nodes_on_wire(
        &self,
        py: Python,
        wire: &Bound<PyAny>,
        only_ops: bool,
    ) -> PyResult<Py<PyIterator>> {
        let wire = if wire.is_instance(imports::QUBIT.get_bound(py))? {
            self.qubits.find(wire).map(Wire::Qubit)
        } else if wire.is_instance(imports::CLBIT.get_bound(py))? {
            self.clbits.find(wire).map(Wire::Clbit)
        } else if self.var_input_map.contains_key(py, &wire.clone().unbind()) {
            Some(Wire::Var(wire.clone().unbind()))
        } else {
            None
        }
        .ok_or_else(|| {
            DAGCircuitError::new_err(format!(
                "The given wire {:?} is not present in the circuit",
                wire
            ))
        })?;

        let nodes = self
            .nodes_on_wire(py, &wire, only_ops)
            .into_iter()
            .map(|n| self.get_node(py, n))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new_bound(py, nodes).into_any().iter()?.unbind())
    }

    /// Count the occurrences of operation names.
    ///
    /// Args:
    ///     recurse: if ``True`` (default), then recurse into control-flow operations.  In all
    ///         cases, this counts only the number of times the operation appears in any possible
    ///         block; both branches of if-elses are counted, and for- and while-loop blocks are
    ///         only counted once.
    ///
    /// Returns:
    ///     Mapping[str, int]: a mapping of operation names to the number of times it appears.
    #[pyo3(name = "count_ops", signature = (*, recurse=true))]
    fn py_count_ops(&self, py: Python, recurse: bool) -> PyResult<PyObject> {
        self.count_ops(py, recurse).map(|x| x.into_py(py))
    }

    /// Count the occurrences of operation names on the longest path.
    ///
    /// Returns a dictionary of counts keyed on the operation name.
    fn count_ops_longest_path(&self) -> PyResult<HashMap<&str, usize>> {
        if self.dag.node_count() == 0 {
            return Ok(HashMap::new());
        }
        let weight_fn = |_| -> Result<usize, Infallible> { Ok(1) };
        let longest_path =
            match rustworkx_core::dag_algo::longest_path(&self.dag, weight_fn).unwrap() {
                Some(res) => res.0,
                None => return Err(DAGCircuitError::new_err("not a DAG")),
            };
        // Allocate for worst case where all operations are unique
        let mut op_counts: HashMap<&str, usize> = HashMap::with_capacity(longest_path.len() - 2);
        for node_index in &longest_path[1..longest_path.len() - 1] {
            if let NodeType::Operation(ref packed) = self.dag[*node_index] {
                let name = packed.op.name();
                op_counts
                    .entry(name)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }
        Ok(op_counts)
    }

    /// Returns causal cone of a qubit.
    ///
    /// A qubit's causal cone is the set of qubits that can influence the output of that
    /// qubit through interactions, whether through multi-qubit gates or operations. Knowing
    /// the causal cone of a qubit can be useful when debugging faulty circuits, as it can
    /// help identify which wire(s) may be causing the problem.
    ///
    /// This method does not consider any classical data dependency in the ``DAGCircuit``,
    /// classical bit wires are ignored for the purposes of building the causal cone.
    ///
    /// Args:
    ///     qubit (~qiskit.circuit.Qubit): The output qubit for which we want to find the causal cone.
    ///
    /// Returns:
    ///     Set[~qiskit.circuit.Qubit]: The set of qubits whose interactions affect ``qubit``.
    fn quantum_causal_cone(&self, py: Python, qubit: &Bound<PyAny>) -> PyResult<Py<PySet>> {
        // Retrieve the output node from the qubit
        let output_qubit = self.qubits.find(qubit).ok_or_else(|| {
            DAGCircuitError::new_err(format!(
                "The given qubit {:?} is not present in the circuit",
                qubit
            ))
        })?;
        let output_node_index = self
            .qubit_io_map
            .get(output_qubit.0 as usize)
            .map(|x| x[1])
            .ok_or_else(|| {
                DAGCircuitError::new_err(format!(
                    "The given qubit {:?} is not present in qubit_output_map",
                    qubit
                ))
            })?;

        let mut qubits_in_cone: HashSet<&Qubit> = HashSet::from([&output_qubit]);
        let mut queue: VecDeque<NodeIndex> = self.quantum_predecessors(output_node_index).collect();

        // The processed_non_directive_nodes stores the set of processed non-directive nodes.
        // This is an optimization to avoid considering the same non-directive node multiple
        // times when reached from different paths.
        // The directive nodes (such as barriers or measures) are trickier since when processing
        // them we only add their predecessors that intersect qubits_in_cone. Hence, directive
        // nodes have to be considered multiple times.
        let mut processed_non_directive_nodes: HashSet<NodeIndex> = HashSet::new();

        while !queue.is_empty() {
            let cur_index = queue.pop_front().unwrap();

            if let NodeType::Operation(packed) = self.dag.node_weight(cur_index).unwrap() {
                if !packed.op.directive() {
                    // If the operation is not a directive (in particular not a barrier nor a measure),
                    // we do not do anything if it was already processed. Otherwise, we add its qubits
                    // to qubits_in_cone, and append its predecessors to queue.
                    if processed_non_directive_nodes.contains(&cur_index) {
                        continue;
                    }
                    qubits_in_cone.extend(self.qargs_interner.get(packed.qubits));
                    processed_non_directive_nodes.insert(cur_index);

                    for pred_index in self.quantum_predecessors(cur_index) {
                        if let NodeType::Operation(_pred_packed) =
                            self.dag.node_weight(pred_index).unwrap()
                        {
                            queue.push_back(pred_index);
                        }
                    }
                } else {
                    // Directives (such as barriers and measures) may be defined over all the qubits,
                    // yet not all of these qubits should be considered in the causal cone. So we
                    // only add those predecessors that have qubits in common with qubits_in_cone.
                    for pred_index in self.quantum_predecessors(cur_index) {
                        if let NodeType::Operation(pred_packed) =
                            self.dag.node_weight(pred_index).unwrap()
                        {
                            if self
                                .qargs_interner
                                .get(pred_packed.qubits)
                                .iter()
                                .any(|x| qubits_in_cone.contains(x))
                            {
                                queue.push_back(pred_index);
                            }
                        }
                    }
                }
            }
        }

        let qubits_in_cone_vec: Vec<_> = qubits_in_cone.iter().map(|&&qubit| qubit).collect();
        let elements = self.qubits.map_indices(&qubits_in_cone_vec[..]);
        Ok(PySet::new_bound(py, elements)?.unbind())
    }

    /// Return a dictionary of circuit properties.
    fn properties(&self, py: Python) -> PyResult<HashMap<&str, PyObject>> {
        Ok(HashMap::from_iter([
            ("size", self.size(py, false)?.into_py(py)),
            ("depth", self.depth(py, false)?.into_py(py)),
            ("width", self.width().into_py(py)),
            ("qubits", self.num_qubits().into_py(py)),
            ("bits", self.num_clbits().into_py(py)),
            ("factors", self.num_tensor_factors().into_py(py)),
            ("operations", self.py_count_ops(py, true)?),
        ]))
    }

    /// Draws the dag circuit.
    ///
    /// This function needs `Graphviz <https://www.graphviz.org/>`_ to be
    /// installed. Graphviz is not a python package and can't be pip installed
    /// (the ``graphviz`` package on PyPI is a Python interface library for
    /// Graphviz and does not actually install Graphviz). You can refer to
    /// `the Graphviz documentation <https://www.graphviz.org/download/>`__ on
    /// how to install it.
    ///
    /// Args:
    ///     scale (float): scaling factor
    ///     filename (str): file path to save image to (format inferred from name)
    ///     style (str):
    ///         'plain': B&W graph;
    ///         'color' (default): color input/output/op nodes
    ///
    /// Returns:
    ///     Ipython.display.Image: if in Jupyter notebook and not saving to file,
    ///     otherwise None.
    #[pyo3(signature=(scale=0.7, filename=None, style="color"))]
    fn draw<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        scale: f64,
        filename: Option<&str>,
        style: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let module = PyModule::import_bound(py, "qiskit.visualization.dag_visualization")?;
        module.call_method1("dag_drawer", (slf, scale, filename, style))
    }

    fn _to_dot<'py>(
        &self,
        py: Python<'py>,
        graph_attrs: Option<BTreeMap<String, String>>,
        node_attrs: Option<PyObject>,
        edge_attrs: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyString>> {
        let mut buffer = Vec::<u8>::new();
        build_dot(py, self, &mut buffer, graph_attrs, node_attrs, edge_attrs)?;
        Ok(PyString::new_bound(py, std::str::from_utf8(&buffer)?))
    }

    /// Add an input variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    fn add_input_var(&mut self, py: Python, var: &Bound<PyAny>) -> PyResult<()> {
        if !self.vars_by_type[DAGVarType::Capture as usize]
            .bind(py)
            .is_empty()
        {
            return Err(DAGCircuitError::new_err(
                "cannot add inputs to a circuit with captures",
            ));
        }
        self.add_var(py, var, DAGVarType::Input)
    }

    /// Add a captured variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    fn add_captured_var(&mut self, py: Python, var: &Bound<PyAny>) -> PyResult<()> {
        if !self.vars_by_type[DAGVarType::Input as usize]
            .bind(py)
            .is_empty()
        {
            return Err(DAGCircuitError::new_err(
                "cannot add captures to a circuit with inputs",
            ));
        }
        self.add_var(py, var, DAGVarType::Capture)
    }

    /// Add a declared local variable to the circuit.
    ///
    /// Args:
    ///     var: the variable to add.
    fn add_declared_var(&mut self, py: Python, var: &Bound<PyAny>) -> PyResult<()> {
        self.add_var(py, var, DAGVarType::Declare)
    }

    /// Total number of classical variables tracked by the circuit.
    #[getter]
    fn num_vars(&self) -> usize {
        self.vars_info.len()
    }

    /// Number of input classical variables tracked by the circuit.
    #[getter]
    fn num_input_vars(&self, py: Python) -> usize {
        self.vars_by_type[DAGVarType::Input as usize].bind(py).len()
    }

    /// Number of captured classical variables tracked by the circuit.
    #[getter]
    fn num_captured_vars(&self, py: Python) -> usize {
        self.vars_by_type[DAGVarType::Capture as usize]
            .bind(py)
            .len()
    }

    /// Number of declared local classical variables tracked by the circuit.
    #[getter]
    fn num_declared_vars(&self, py: Python) -> usize {
        self.vars_by_type[DAGVarType::Declare as usize]
            .bind(py)
            .len()
    }

    /// Is this realtime variable in the DAG?
    ///
    /// Args:
    ///     var: the variable or name to check.
    fn has_var(&self, var: &Bound<PyAny>) -> PyResult<bool> {
        match var.extract::<String>() {
            Ok(name) => Ok(self.vars_info.contains_key(&name)),
            Err(_) => {
                let raw_name = var.getattr("name")?;
                let var_name: String = raw_name.extract()?;
                match self.vars_info.get(&var_name) {
                    Some(var_in_dag) => Ok(var_in_dag.var.is(var)),
                    None => Ok(false),
                }
            }
        }
    }

    /// Iterable over the input classical variables tracked by the circuit.
    fn iter_input_vars(&self, py: Python) -> PyResult<Py<PyIterator>> {
        Ok(self.vars_by_type[DAGVarType::Input as usize]
            .bind(py)
            .clone()
            .into_any()
            .iter()?
            .unbind())
    }

    /// Iterable over the captured classical variables tracked by the circuit.
    fn iter_captured_vars(&self, py: Python) -> PyResult<Py<PyIterator>> {
        Ok(self.vars_by_type[DAGVarType::Capture as usize]
            .bind(py)
            .clone()
            .into_any()
            .iter()?
            .unbind())
    }

    /// Iterable over the declared classical variables tracked by the circuit.
    fn iter_declared_vars(&self, py: Python) -> PyResult<Py<PyIterator>> {
        Ok(self.vars_by_type[DAGVarType::Declare as usize]
            .bind(py)
            .clone()
            .into_any()
            .iter()?
            .unbind())
    }

    /// Iterable over all the classical variables tracked by the circuit.
    fn iter_vars(&self, py: Python) -> PyResult<Py<PyIterator>> {
        let out_set = PySet::empty_bound(py)?;
        for var_type_set in &self.vars_by_type {
            for var in var_type_set.bind(py).iter() {
                out_set.add(var)?;
            }
        }
        Ok(out_set.into_any().iter()?.unbind())
    }

    fn _has_edge(&self, source: usize, target: usize) -> bool {
        self.dag
            .contains_edge(NodeIndex::new(source), NodeIndex::new(target))
    }

    fn _is_dag(&self) -> bool {
        rustworkx_core::petgraph::algo::toposort(&self.dag, None).is_ok()
    }

    fn _in_edges(&self, py: Python, node_index: usize) -> Vec<Py<PyTuple>> {
        self.dag
            .edges_directed(NodeIndex::new(node_index), Incoming)
            .map(|wire| {
                (
                    wire.source().index(),
                    wire.target().index(),
                    match wire.weight() {
                        Wire::Qubit(qubit) => self.qubits.get(*qubit).unwrap(),
                        Wire::Clbit(clbit) => self.clbits.get(*clbit).unwrap(),
                        Wire::Var(var) => var,
                    },
                )
                    .into_py(py)
            })
            .collect()
    }

    fn _out_edges(&self, py: Python, node_index: usize) -> Vec<Py<PyTuple>> {
        self.dag
            .edges_directed(NodeIndex::new(node_index), Outgoing)
            .map(|wire| {
                (
                    wire.source().index(),
                    wire.target().index(),
                    match wire.weight() {
                        Wire::Qubit(qubit) => self.qubits.get(*qubit).unwrap(),
                        Wire::Clbit(clbit) => self.clbits.get(*clbit).unwrap(),
                        Wire::Var(var) => var,
                    },
                )
                    .into_py(py)
            })
            .collect()
    }

    fn _in_wires(&self, node_index: usize) -> Vec<&PyObject> {
        self.dag
            .edges_directed(NodeIndex::new(node_index), Incoming)
            .map(|wire| match wire.weight() {
                Wire::Qubit(qubit) => self.qubits.get(*qubit).unwrap(),
                Wire::Clbit(clbit) => self.clbits.get(*clbit).unwrap(),
                Wire::Var(var) => var,
            })
            .collect()
    }

    fn _out_wires(&self, node_index: usize) -> Vec<&PyObject> {
        self.dag
            .edges_directed(NodeIndex::new(node_index), Outgoing)
            .map(|wire| match wire.weight() {
                Wire::Qubit(qubit) => self.qubits.get(*qubit).unwrap(),
                Wire::Clbit(clbit) => self.clbits.get(*clbit).unwrap(),
                Wire::Var(var) => var,
            })
            .collect()
    }

    fn _find_successors_by_edge(
        &self,
        py: Python,
        node_index: usize,
        edge_checker: &Bound<PyAny>,
    ) -> PyResult<Vec<PyObject>> {
        let mut result = Vec::new();
        for e in self
            .dag
            .edges_directed(NodeIndex::new(node_index), Outgoing)
            .unique_by(|e| e.id())
        {
            let weight = match e.weight() {
                Wire::Qubit(q) => self.qubits.get(*q).unwrap(),
                Wire::Clbit(c) => self.clbits.get(*c).unwrap(),
                Wire::Var(v) => v,
            };
            if edge_checker.call1((weight,))?.extract::<bool>()? {
                result.push(self.get_node(py, e.target())?);
            }
        }
        Ok(result)
    }

    fn _edges(&self, py: Python) -> Vec<PyObject> {
        self.dag
            .edge_indices()
            .map(|index| {
                let wire = self.dag.edge_weight(index).unwrap();
                match wire {
                    Wire::Qubit(qubit) => self.qubits.get(*qubit).to_object(py),
                    Wire::Clbit(clbit) => self.clbits.get(*clbit).to_object(py),
                    Wire::Var(var) => var.clone_ref(py),
                }
            })
            .collect()
    }
}

impl DAGCircuit {
    /// Return an iterator of gate runs with non-conditional op nodes of given names
    pub fn collect_runs(
        &self,
        namelist: HashSet<String>,
    ) -> Option<impl Iterator<Item = Vec<NodeIndex>> + '_> {
        let filter_fn = move |node_index: NodeIndex| -> Result<bool, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => Ok(namelist.contains(inst.op.name())
                    && match &inst.extra_attrs {
                        None => true,
                        Some(attrs) => attrs.condition.is_none(),
                    }),
                _ => Ok(false),
            }
        };
        rustworkx_core::dag_algo::collect_runs(&self.dag, filter_fn)
            .map(|node_iter| node_iter.map(|x| x.unwrap()))
    }

    /// Return a set of non-conditional runs of 1q "op" nodes.
    pub fn collect_1q_runs(&self) -> Option<impl Iterator<Item = Vec<NodeIndex>> + '_> {
        let filter_fn = move |node_index: NodeIndex| -> Result<bool, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => Ok(inst.op.num_qubits() == 1
                    && inst.op.num_clbits() == 0
                    && !inst.is_parameterized()
                    && (inst.op.try_standard_gate().is_some()
                        || inst.op.matrix(inst.params_view()).is_some())
                    && inst.condition().is_none()),
                _ => Ok(false),
            }
        };
        rustworkx_core::dag_algo::collect_runs(&self.dag, filter_fn)
            .map(|node_iter| node_iter.map(|x| x.unwrap()))
    }

    /// Return a set of non-conditional runs of 2q "op" nodes.
    pub fn collect_2q_runs(&self) -> Option<Vec<Vec<NodeIndex>>> {
        let filter_fn = move |node_index: NodeIndex| -> Result<Option<bool>, Infallible> {
            let node = &self.dag[node_index];
            match node {
                NodeType::Operation(inst) => match inst.op.view() {
                    OperationRef::Standard(gate) => Ok(Some(
                        gate.num_qubits() <= 2
                            && inst.condition().is_none()
                            && !inst.is_parameterized(),
                    )),
                    OperationRef::Gate(gate) => Ok(Some(
                        gate.num_qubits() <= 2
                            && inst.condition().is_none()
                            && !inst.is_parameterized(),
                    )),
                    _ => Ok(Some(false)),
                },
                _ => Ok(None),
            }
        };

        let color_fn = move |edge_index: EdgeIndex| -> Result<Option<usize>, Infallible> {
            let wire = self.dag.edge_weight(edge_index).unwrap();
            match wire {
                Wire::Qubit(index) => Ok(Some(index.0 as usize)),
                _ => Ok(None),
            }
        };
        rustworkx_core::dag_algo::collect_bicolor_runs(&self.dag, filter_fn, color_fn).unwrap()
    }

    fn increment_op(&mut self, op: &str) {
        match self.op_names.get_mut(op) {
            Some(count) => {
                *count += 1;
            }
            None => {
                self.op_names.insert(op.to_string(), 1);
            }
        }
    }

    fn decrement_op(&mut self, op: &str) {
        match self.op_names.get_mut(op) {
            Some(count) => {
                if *count > 1 {
                    *count -= 1;
                } else {
                    self.op_names.swap_remove(op);
                }
            }
            None => panic!("Cannot decrement something not added!"),
        }
    }

    pub fn quantum_predecessors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.dag
            .edges_directed(node, Incoming)
            .filter_map(|e| match e.weight() {
                Wire::Qubit(_) => Some(e.source()),
                _ => None,
            })
            .unique()
    }

    pub fn quantum_successors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.dag
            .edges_directed(node, Outgoing)
            .filter_map(|e| match e.weight() {
                Wire::Qubit(_) => Some(e.target()),
                _ => None,
            })
            .unique()
    }

    /// Apply a [PackedInstruction] to the back of the circuit.
    ///
    /// The provided `instr` MUST be valid for this DAG, e.g. its
    /// bits, registers, vars, and interner IDs must be valid in
    /// this DAG.
    ///
    /// This is mostly used to apply operations from one DAG to
    /// another that was created from the first via
    /// [DAGCircuit::copy_empty_like].
    fn push_back(&mut self, py: Python, instr: PackedInstruction) -> PyResult<NodeIndex> {
        let op_name = instr.op.name();
        let (all_cbits, vars): (Vec<Clbit>, Option<Vec<PyObject>>) = {
            if self.may_have_additional_wires(py, &instr) {
                let mut clbits: HashSet<Clbit> =
                    HashSet::from_iter(self.cargs_interner.get(instr.clbits).iter().copied());
                let (additional_clbits, additional_vars) =
                    self.additional_wires(py, instr.op.view(), instr.condition())?;
                for clbit in additional_clbits {
                    clbits.insert(clbit);
                }
                (clbits.into_iter().collect(), Some(additional_vars))
            } else {
                (self.cargs_interner.get(instr.clbits).to_vec(), None)
            }
        };

        self.increment_op(op_name);

        let qubits_id = instr.qubits;
        let new_node = self.dag.add_node(NodeType::Operation(instr));

        // Put the new node in-between the previously "last" nodes on each wire
        // and the output map.
        let output_nodes: HashSet<NodeIndex> = self
            .qargs_interner
            .get(qubits_id)
            .iter()
            .map(|q| self.qubit_io_map.get(q.0 as usize).map(|x| x[1]).unwrap())
            .chain(
                all_cbits
                    .iter()
                    .map(|c| self.clbit_io_map.get(c.0 as usize).map(|x| x[1]).unwrap()),
            )
            .chain(
                vars.iter()
                    .flatten()
                    .map(|v| self.var_output_map.get(py, v).unwrap()),
            )
            .collect();

        for output_node in output_nodes {
            let last_edges: Vec<_> = self
                .dag
                .edges_directed(output_node, Incoming)
                .map(|e| (e.source(), e.id(), e.weight().clone()))
                .collect();
            for (source, old_edge, weight) in last_edges.into_iter() {
                self.dag.add_edge(source, new_node, weight.clone());
                self.dag.add_edge(new_node, output_node, weight);
                self.dag.remove_edge(old_edge);
            }
        }

        Ok(new_node)
    }

    /// Apply a [PackedInstruction] to the front of the circuit.
    ///
    /// The provided `instr` MUST be valid for this DAG, e.g. its
    /// bits, registers, vars, and interner IDs must be valid in
    /// this DAG.
    ///
    /// This is mostly used to apply operations from one DAG to
    /// another that was created from the first via
    /// [DAGCircuit::copy_empty_like].
    fn push_front(&mut self, py: Python, inst: PackedInstruction) -> PyResult<NodeIndex> {
        let op_name = inst.op.name();
        let (all_cbits, vars): (Vec<Clbit>, Option<Vec<PyObject>>) = {
            if self.may_have_additional_wires(py, &inst) {
                let mut clbits: HashSet<Clbit> =
                    HashSet::from_iter(self.cargs_interner.get(inst.clbits).iter().copied());
                let (additional_clbits, additional_vars) =
                    self.additional_wires(py, inst.op.view(), inst.condition())?;
                for clbit in additional_clbits {
                    clbits.insert(clbit);
                }
                (clbits.into_iter().collect(), Some(additional_vars))
            } else {
                (self.cargs_interner.get(inst.clbits).to_vec(), None)
            }
        };

        self.increment_op(op_name);

        let qubits_id = inst.qubits;
        let new_node = self.dag.add_node(NodeType::Operation(inst));

        // Put the new node in-between the input map and the previously
        // "first" nodes on each wire.
        let mut input_nodes: Vec<NodeIndex> = self
            .qargs_interner
            .get(qubits_id)
            .iter()
            .map(|q| self.qubit_io_map[q.0 as usize][0])
            .chain(all_cbits.iter().map(|c| self.clbit_io_map[c.0 as usize][0]))
            .collect();
        if let Some(vars) = vars {
            for var in vars {
                input_nodes.push(self.var_input_map.get(py, &var).unwrap());
            }
        }

        for input_node in input_nodes {
            let first_edges: Vec<_> = self
                .dag
                .edges_directed(input_node, Outgoing)
                .map(|e| (e.target(), e.id(), e.weight().clone()))
                .collect();
            for (target, old_edge, weight) in first_edges.into_iter() {
                self.dag.add_edge(input_node, new_node, weight.clone());
                self.dag.add_edge(new_node, target, weight);
                self.dag.remove_edge(old_edge);
            }
        }

        Ok(new_node)
    }

    fn sort_key(&self, node: NodeIndex) -> SortKeyType {
        match &self.dag[node] {
            NodeType::Operation(packed) => (
                self.qargs_interner.get(packed.qubits),
                self.cargs_interner.get(packed.clbits),
            ),
            NodeType::QubitIn(q) => (std::slice::from_ref(q), &[Clbit(u32::MAX)]),
            NodeType::QubitOut(_q) => (&[Qubit(u32::MAX)], &[Clbit(u32::MAX)]),
            NodeType::ClbitIn(c) => (&[Qubit(u32::MAX)], std::slice::from_ref(c)),
            NodeType::ClbitOut(_c) => (&[Qubit(u32::MAX)], &[Clbit(u32::MAX)]),
            _ => (&[], &[]),
        }
    }

    fn topological_nodes(&self) -> PyResult<impl Iterator<Item = NodeIndex>> {
        let key = |node: NodeIndex| -> Result<SortKeyType, Infallible> { Ok(self.sort_key(node)) };
        let nodes =
            rustworkx_core::dag_algo::lexicographical_topological_sort(&self.dag, key, false, None)
                .map_err(|e| match e {
                    rustworkx_core::dag_algo::TopologicalSortError::CycleOrBadInitialState => {
                        PyValueError::new_err(format!("{}", e))
                    }
                    rustworkx_core::dag_algo::TopologicalSortError::KeyError(_) => {
                        unreachable!()
                    }
                })?;
        Ok(nodes.into_iter())
    }

    pub fn topological_op_nodes(&self) -> PyResult<impl Iterator<Item = NodeIndex> + '_> {
        Ok(self.topological_nodes()?.filter(|node: &NodeIndex| {
            matches!(self.dag.node_weight(*node), Some(NodeType::Operation(_)))
        }))
    }

    fn topological_key_sort(
        &self,
        py: Python,
        key: &Bound<PyAny>,
    ) -> PyResult<impl Iterator<Item = NodeIndex>> {
        // This path (user provided key func) is not ideal, since we no longer
        // use a string key after moving to Rust, in favor of using a tuple
        // of the qargs and cargs interner IDs of the node.
        let key = |node: NodeIndex| -> PyResult<String> {
            let node = self.get_node(py, node)?;
            key.call1((node,))?.extract()
        };
        Ok(
            rustworkx_core::dag_algo::lexicographical_topological_sort(&self.dag, key, false, None)
                .map_err(|e| match e {
                    rustworkx_core::dag_algo::TopologicalSortError::CycleOrBadInitialState => {
                        PyValueError::new_err(format!("{}", e))
                    }
                    rustworkx_core::dag_algo::TopologicalSortError::KeyError(ref e) => {
                        e.clone_ref(py)
                    }
                })?
                .into_iter(),
        )
    }

    #[inline]
    fn has_control_flow(&self) -> bool {
        CONTROL_FLOW_OP_NAMES
            .iter()
            .any(|x| self.op_names.contains_key(&x.to_string()))
    }

    fn is_wire_idle(&self, py: Python, wire: &Wire) -> PyResult<bool> {
        let (input_node, output_node) = match wire {
            Wire::Qubit(qubit) => (
                self.qubit_io_map[qubit.0 as usize][0],
                self.qubit_io_map[qubit.0 as usize][1],
            ),
            Wire::Clbit(clbit) => (
                self.clbit_io_map[clbit.0 as usize][0],
                self.clbit_io_map[clbit.0 as usize][1],
            ),
            Wire::Var(var) => (
                self.var_input_map.get(py, var).unwrap(),
                self.var_output_map.get(py, var).unwrap(),
            ),
        };

        let child = self
            .dag
            .neighbors_directed(input_node, Outgoing)
            .next()
            .ok_or_else(|| {
                DAGCircuitError::new_err(format!(
                    "Invalid dagcircuit input node {:?} has no output",
                    input_node
                ))
            })?;

        Ok(child == output_node)
    }

    fn may_have_additional_wires(&self, py: Python, instr: &PackedInstruction) -> bool {
        if instr.condition().is_some() {
            return true;
        }
        let OperationRef::Instruction(inst) = instr.op.view() else {
            return false;
        };
        inst.control_flow()
            || inst
                .instruction
                .bind(py)
                .is_instance(imports::STORE_OP.get_bound(py))
                .unwrap()
    }

    fn additional_wires(
        &self,
        py: Python,
        op: OperationRef,
        condition: Option<&PyObject>,
    ) -> PyResult<(Vec<Clbit>, Vec<PyObject>)> {
        let wires_from_expr = |node: &Bound<PyAny>| -> PyResult<(Vec<Clbit>, Vec<PyObject>)> {
            let mut clbits = Vec::new();
            let mut vars = Vec::new();
            for var in imports::ITER_VARS.get_bound(py).call1((node,))?.iter()? {
                let var = var?;
                let var_var = var.getattr("var")?;
                if var_var.is_instance(imports::CLBIT.get_bound(py))? {
                    clbits.push(self.clbits.find(&var_var).unwrap());
                } else if var_var.is_instance(imports::CLASSICAL_REGISTER.get_bound(py))? {
                    for bit in var_var.iter().unwrap() {
                        clbits.push(self.clbits.find(&bit?).unwrap());
                    }
                } else {
                    vars.push(var.unbind());
                }
            }
            Ok((clbits, vars))
        };

        let mut clbits = Vec::new();
        let mut vars = Vec::new();
        if let Some(condition) = condition {
            let condition = condition.bind(py);
            if !condition.is_none() {
                if condition.is_instance(imports::EXPR.get_bound(py)).unwrap() {
                    let (expr_clbits, expr_vars) = wires_from_expr(condition)?;
                    for bit in expr_clbits {
                        clbits.push(bit);
                    }
                    for var in expr_vars {
                        vars.push(var);
                    }
                } else {
                    for bit in self
                        .control_flow_module
                        .condition_resources(condition)?
                        .clbits
                        .bind(py)
                    {
                        clbits.push(self.clbits.find(&bit).unwrap());
                    }
                }
            }
        }

        if let OperationRef::Instruction(inst) = op {
            let op = inst.instruction.bind(py);
            if inst.control_flow() {
                for var in op.call_method0("iter_captured_vars")?.iter()? {
                    vars.push(var?.unbind())
                }
                if op.is_instance(imports::SWITCH_CASE_OP.get_bound(py))? {
                    let target = op.getattr(intern!(py, "target"))?;
                    if target.is_instance(imports::CLBIT.get_bound(py))? {
                        clbits.push(self.clbits.find(&target).unwrap());
                    } else if target.is_instance(imports::CLASSICAL_REGISTER.get_bound(py))? {
                        for bit in target.iter()? {
                            clbits.push(self.clbits.find(&bit?).unwrap());
                        }
                    } else {
                        let (expr_clbits, expr_vars) = wires_from_expr(&target)?;
                        for bit in expr_clbits {
                            clbits.push(bit);
                        }
                        for var in expr_vars {
                            vars.push(var);
                        }
                    }
                }
            } else if op.is_instance(imports::STORE_OP.get_bound(py))? {
                let (expr_clbits, expr_vars) = wires_from_expr(&op.getattr("lvalue")?)?;
                for bit in expr_clbits {
                    clbits.push(bit);
                }
                for var in expr_vars {
                    vars.push(var);
                }
                let (expr_clbits, expr_vars) = wires_from_expr(&op.getattr("rvalue")?)?;
                for bit in expr_clbits {
                    clbits.push(bit);
                }
                for var in expr_vars {
                    vars.push(var);
                }
            }
        }
        Ok((clbits, vars))
    }

    /// Add a qubit or bit to the circuit.
    ///
    /// Args:
    ///     wire: the wire to be added
    ///
    ///     This adds a pair of in and out nodes connected by an edge.
    ///
    /// Raises:
    ///     DAGCircuitError: if trying to add duplicate wire
    fn add_wire(&mut self, py: Python, wire: Wire) -> PyResult<()> {
        let (in_node, out_node) = match wire {
            Wire::Qubit(qubit) => {
                if (qubit.0 as usize) >= self.qubit_io_map.len() {
                    let input_node = self.dag.add_node(NodeType::QubitIn(qubit));
                    let output_node = self.dag.add_node(NodeType::QubitOut(qubit));
                    self.qubit_io_map.push([input_node, output_node]);
                    Ok((input_node, output_node))
                } else {
                    Err(DAGCircuitError::new_err("qubit wire already exists!"))
                }
            }
            Wire::Clbit(clbit) => {
                if (clbit.0 as usize) >= self.clbit_io_map.len() {
                    let input_node = self.dag.add_node(NodeType::ClbitIn(clbit));
                    let output_node = self.dag.add_node(NodeType::ClbitOut(clbit));
                    self.clbit_io_map.push([input_node, output_node]);
                    Ok((input_node, output_node))
                } else {
                    Err(DAGCircuitError::new_err("classical wire already exists!"))
                }
            }
            Wire::Var(ref var) => {
                if self.var_input_map.contains_key(py, var)
                    || self.var_output_map.contains_key(py, var)
                {
                    return Err(DAGCircuitError::new_err("var wire already exists!"));
                }
                let in_node = self.dag.add_node(NodeType::VarIn(var.clone_ref(py)));
                let out_node = self.dag.add_node(NodeType::VarOut(var.clone_ref(py)));
                self.var_input_map.insert(py, var.clone_ref(py), in_node);
                self.var_output_map.insert(py, var.clone_ref(py), out_node);
                Ok((in_node, out_node))
            }
        }?;

        self.dag.add_edge(in_node, out_node, wire);
        Ok(())
    }

    /// Get the nodes on the given wire.
    ///
    /// Note: result is empty if the wire is not in the DAG.
    pub fn nodes_on_wire(&self, py: Python, wire: &Wire, only_ops: bool) -> Vec<NodeIndex> {
        let mut nodes = Vec::new();
        let mut current_node = match wire {
            Wire::Qubit(qubit) => self.qubit_io_map.get(qubit.0 as usize).map(|x| x[0]),
            Wire::Clbit(clbit) => self.clbit_io_map.get(clbit.0 as usize).map(|x| x[0]),
            Wire::Var(var) => self.var_input_map.get(py, var),
        };

        while let Some(node) = current_node {
            if only_ops {
                let node_weight = self.dag.node_weight(node).unwrap();
                if let NodeType::Operation(_) = node_weight {
                    nodes.push(node);
                }
            } else {
                nodes.push(node);
            }

            let edges = self.dag.edges_directed(node, Outgoing);
            current_node = edges.into_iter().find_map(|edge| {
                if edge.weight() == wire {
                    Some(edge.target())
                } else {
                    None
                }
            });
        }
        nodes
    }

    fn remove_idle_wire(&mut self, py: Python, wire: Wire) -> PyResult<()> {
        let [in_node, out_node] = match wire {
            Wire::Qubit(qubit) => self.qubit_io_map[qubit.0 as usize],
            Wire::Clbit(clbit) => self.clbit_io_map[clbit.0 as usize],
            Wire::Var(var) => [
                self.var_input_map.remove(py, &var).unwrap(),
                self.var_output_map.remove(py, &var).unwrap(),
            ],
        };
        self.dag.remove_node(in_node);
        self.dag.remove_node(out_node);
        Ok(())
    }

    fn add_qubit_unchecked(&mut self, py: Python, bit: &Bound<PyAny>) -> PyResult<Qubit> {
        let qubit = self.qubits.add(py, bit, false)?;
        self.qubit_locations.bind(py).set_item(
            bit,
            Py::new(
                py,
                BitLocations {
                    index: (self.qubits.len() - 1),
                    registers: PyList::empty_bound(py).unbind(),
                },
            )?,
        )?;
        self.add_wire(py, Wire::Qubit(qubit))?;
        Ok(qubit)
    }

    fn add_clbit_unchecked(&mut self, py: Python, bit: &Bound<PyAny>) -> PyResult<Clbit> {
        let clbit = self.clbits.add(py, bit, false)?;
        self.clbit_locations.bind(py).set_item(
            bit,
            Py::new(
                py,
                BitLocations {
                    index: (self.clbits.len() - 1),
                    registers: PyList::empty_bound(py).unbind(),
                },
            )?,
        )?;
        self.add_wire(py, Wire::Clbit(clbit))?;
        Ok(clbit)
    }

    pub fn get_node(&self, py: Python, node: NodeIndex) -> PyResult<Py<PyAny>> {
        self.unpack_into(py, node, self.dag.node_weight(node).unwrap())
    }

    /// Remove an operation node n.
    ///
    /// Add edges from predecessors to successors.
    pub fn remove_op_node(&mut self, index: NodeIndex) {
        let mut edge_list: Vec<(NodeIndex, NodeIndex, Wire)> = Vec::new();
        for (source, in_weight) in self
            .dag
            .edges_directed(index, Incoming)
            .map(|x| (x.source(), x.weight()))
        {
            for (target, out_weight) in self
                .dag
                .edges_directed(index, Outgoing)
                .map(|x| (x.target(), x.weight()))
            {
                if in_weight == out_weight {
                    edge_list.push((source, target, in_weight.clone()));
                }
            }
        }
        for (source, target, weight) in edge_list {
            self.dag.add_edge(source, target, weight);
        }

        match self.dag.remove_node(index) {
            Some(NodeType::Operation(packed)) => {
                let op_name = packed.op.name();
                self.decrement_op(op_name);
            }
            _ => panic!("Must be called with valid operation node!"),
        }
    }

    /// Returns an iterator of the ancestors indices of a node.
    pub fn ancestors(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        core_ancestors(&self.dag, node).filter(move |next| next != &node)
    }

    /// Returns an iterator of the descendants of a node as DAGOpNodes and DAGOutNodes.
    pub fn descendants(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        core_descendants(&self.dag, node).filter(move |next| next != &node)
    }

    /// Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node
    /// and [DAGNode] is its successors in  BFS order.
    pub fn bfs_successors(
        &self,
        node: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, Vec<NodeIndex>)> + '_ {
        core_bfs_successors(&self.dag, node).filter(move |(_, others)| !others.is_empty())
    }

    fn pack_into(&mut self, py: Python, b: &Bound<PyAny>) -> Result<NodeType, PyErr> {
        Ok(if let Ok(in_node) = b.downcast::<DAGInNode>() {
            let in_node = in_node.borrow();
            let wire = in_node.wire.bind(py);
            if wire.is_instance(imports::QUBIT.get_bound(py))? {
                NodeType::QubitIn(self.qubits.find(wire).unwrap())
            } else if wire.is_instance(imports::CLBIT.get_bound(py))? {
                NodeType::ClbitIn(self.clbits.find(wire).unwrap())
            } else {
                NodeType::VarIn(wire.clone().unbind())
            }
        } else if let Ok(out_node) = b.downcast::<DAGOutNode>() {
            let out_node = out_node.borrow();
            let wire = out_node.wire.bind(py);
            if wire.is_instance(imports::QUBIT.get_bound(py))? {
                NodeType::QubitOut(self.qubits.find(wire).unwrap())
            } else if wire.is_instance(imports::CLBIT.get_bound(py))? {
                NodeType::ClbitOut(self.clbits.find(wire).unwrap())
            } else {
                NodeType::VarIn(wire.clone().unbind())
            }
        } else if let Ok(op_node) = b.downcast::<DAGOpNode>() {
            let op_node = op_node.borrow();
            let qubits = self.qargs_interner.insert_owned(
                self.qubits
                    .map_bits(op_node.instruction.qubits.bind(py))?
                    .collect(),
            );
            let clbits = self.cargs_interner.insert_owned(
                self.clbits
                    .map_bits(op_node.instruction.clbits.bind(py))?
                    .collect(),
            );
            let params = (!op_node.instruction.params.is_empty())
                .then(|| Box::new(op_node.instruction.params.clone()));
            let inst = PackedInstruction {
                op: op_node.instruction.operation.clone(),
                qubits,
                clbits,
                params,
                extra_attrs: op_node.instruction.extra_attrs.clone(),
                #[cfg(feature = "cache_pygates")]
                py_op: op_node.instruction.py_op.clone(),
            };
            NodeType::Operation(inst)
        } else {
            return Err(PyTypeError::new_err("Invalid type for DAGNode"));
        })
    }

    fn unpack_into(&self, py: Python, id: NodeIndex, weight: &NodeType) -> PyResult<Py<PyAny>> {
        let dag_node = match weight {
            NodeType::QubitIn(qubit) => Py::new(
                py,
                DAGInNode::new(py, id, self.qubits.get(*qubit).unwrap().clone_ref(py)),
            )?
            .into_any(),
            NodeType::QubitOut(qubit) => Py::new(
                py,
                DAGOutNode::new(py, id, self.qubits.get(*qubit).unwrap().clone_ref(py)),
            )?
            .into_any(),
            NodeType::ClbitIn(clbit) => Py::new(
                py,
                DAGInNode::new(py, id, self.clbits.get(*clbit).unwrap().clone_ref(py)),
            )?
            .into_any(),
            NodeType::ClbitOut(clbit) => Py::new(
                py,
                DAGOutNode::new(py, id, self.clbits.get(*clbit).unwrap().clone_ref(py)),
            )?
            .into_any(),
            NodeType::Operation(packed) => {
                let qubits = self.qargs_interner.get(packed.qubits);
                let clbits = self.cargs_interner.get(packed.clbits);
                Py::new(
                    py,
                    (
                        DAGOpNode {
                            instruction: CircuitInstruction {
                                operation: packed.op.clone(),
                                qubits: PyTuple::new_bound(py, self.qubits.map_indices(qubits))
                                    .unbind(),
                                clbits: PyTuple::new_bound(py, self.clbits.map_indices(clbits))
                                    .unbind(),
                                params: packed.params_view().iter().cloned().collect(),
                                extra_attrs: packed.extra_attrs.clone(),
                                #[cfg(feature = "cache_pygates")]
                                py_op: packed.py_op.clone(),
                            },
                            sort_key: format!("{:?}", self.sort_key(id)).into_py(py),
                        },
                        DAGNode { node: Some(id) },
                    ),
                )?
                .into_any()
            }
            NodeType::VarIn(var) => {
                Py::new(py, DAGInNode::new(py, id, var.clone_ref(py)))?.into_any()
            }
            NodeType::VarOut(var) => {
                Py::new(py, DAGOutNode::new(py, id, var.clone_ref(py)))?.into_any()
            }
        };
        Ok(dag_node)
    }

    /// Returns an iterator over all the indices that refer to an `Operation` node in the `DAGCircuit.`
    pub fn op_nodes<'a>(
        &'a self,
        include_directives: bool,
    ) -> Box<dyn Iterator<Item = NodeIndex> + 'a> {
        let node_ops_iter = self
            .dag
            .node_references()
            .filter_map(|(node_index, node_type)| match node_type {
                NodeType::Operation(ref node) => Some((node_index, node)),
                _ => None,
            });
        if !include_directives {
            Box::new(node_ops_iter.filter_map(|(index, node)| {
                if !node.op.directive() {
                    Some(index)
                } else {
                    None
                }
            }))
        } else {
            Box::new(node_ops_iter.map(|(index, _)| index))
        }
    }

    /// Return an iterator of 2 qubit operations. Ignore directives like snapshot and barrier.
    pub fn two_qubit_ops(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        Box::new(self.op_nodes(false).filter(|index| {
            let weight = self.dag.node_weight(*index).expect("NodeIndex in graph");
            if let NodeType::Operation(ref packed) = weight {
                let qargs = self.qargs_interner.get(packed.qubits);
                qargs.len() == 2
            } else {
                false
            }
        }))
    }

    // Filter any nodes that don't match a given predicate function
    pub fn filter_op_nodes<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&PackedInstruction) -> bool,
    {
        let mut remove_nodes: Vec<NodeIndex> = Vec::new();
        for node in self.op_nodes(true) {
            let NodeType::Operation(op) = &self.dag[node] else {
                unreachable!()
            };
            if !predicate(op) {
                remove_nodes.push(node);
            }
        }
        for node in remove_nodes {
            self.remove_op_node(node);
        }
    }

    pub fn op_nodes_by_py_type<'a>(
        &'a self,
        op: &'a Bound<PyType>,
        include_directives: bool,
    ) -> impl Iterator<Item = NodeIndex> + 'a {
        self.dag
            .node_references()
            .filter_map(move |(node, weight)| {
                if let NodeType::Operation(ref packed) = weight {
                    if !include_directives && packed.op.directive() {
                        None
                    } else if packed.op.py_op_is_instance(op).unwrap() {
                        Some(node)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
    }

    /// Returns an iterator over a list layers of the `DAGCircuit``.
    pub fn multigraph_layers(&self, py: Python) -> impl Iterator<Item = Vec<NodeIndex>> + '_ {
        let mut first_layer: Vec<_> = self.qubit_io_map.iter().map(|x| x[0]).collect();
        first_layer.extend(self.clbit_io_map.iter().map(|x| x[0]));
        first_layer.extend(self.var_input_map.values(py));
        // A DAG is by definition acyclical, therefore unwrapping the layer should never fail.
        layers(&self.dag, first_layer).map(|layer| match layer {
            Ok(layer) => layer,
            Err(_) => unreachable!("Not a DAG."),
        })
    }

    /// Returns an iterator over the first layer of the `DAGCircuit``.
    pub fn front_layer<'a>(&'a self, py: Python) -> Box<dyn Iterator<Item = NodeIndex> + 'a> {
        let mut graph_layers = self.multigraph_layers(py);
        graph_layers.next();

        let next_layer = graph_layers.next();
        match next_layer {
            Some(layer) => Box::new(layer.into_iter().filter(|node| {
                matches!(self.dag.node_weight(*node).unwrap(), NodeType::Operation(_))
            })),
            None => Box::new(vec![].into_iter()),
        }
    }

    fn substitute_node_with_subgraph(
        &mut self,
        py: Python,
        node: NodeIndex,
        other: &DAGCircuit,
        qubit_map: &HashMap<Qubit, Qubit>,
        clbit_map: &HashMap<Clbit, Clbit>,
        var_map: &Py<PyDict>,
    ) -> PyResult<IndexMap<NodeIndex, NodeIndex, RandomState>> {
        if self.dag.node_weight(node).is_none() {
            return Err(PyIndexError::new_err(format!(
                "Specified node {} is not in this graph",
                node.index()
            )));
        }

        // Add wire from pred to succ if no ops on mapped wire on ``other``
        for (in_dag_wire, self_wire) in qubit_map.iter() {
            let [input_node, out_node] = other.qubit_io_map[in_dag_wire.0 as usize];
            if other.dag.find_edge(input_node, out_node).is_some() {
                let pred = self
                    .dag
                    .edges_directed(node, Incoming)
                    .find(|edge| {
                        if let Wire::Qubit(bit) = edge.weight() {
                            bit == self_wire
                        } else {
                            false
                        }
                    })
                    .unwrap();
                let succ = self
                    .dag
                    .edges_directed(node, Outgoing)
                    .find(|edge| {
                        if let Wire::Qubit(bit) = edge.weight() {
                            bit == self_wire
                        } else {
                            false
                        }
                    })
                    .unwrap();
                self.dag
                    .add_edge(pred.source(), succ.target(), Wire::Qubit(*self_wire));
            }
        }
        for (in_dag_wire, self_wire) in clbit_map.iter() {
            let [input_node, out_node] = other.clbit_io_map[in_dag_wire.0 as usize];
            if other.dag.find_edge(input_node, out_node).is_some() {
                let pred = self
                    .dag
                    .edges_directed(node, Incoming)
                    .find(|edge| {
                        if let Wire::Clbit(bit) = edge.weight() {
                            bit == self_wire
                        } else {
                            false
                        }
                    })
                    .unwrap();
                let succ = self
                    .dag
                    .edges_directed(node, Outgoing)
                    .find(|edge| {
                        if let Wire::Clbit(bit) = edge.weight() {
                            bit == self_wire
                        } else {
                            false
                        }
                    })
                    .unwrap();
                self.dag
                    .add_edge(pred.source(), succ.target(), Wire::Clbit(*self_wire));
            }
        }

        let bound_var_map = var_map.bind(py);
        let node_filter = |node: NodeIndex| -> bool {
            match other.dag[node] {
                NodeType::Operation(_) => !other
                    .dag
                    .edges_directed(node, petgraph::Direction::Outgoing)
                    .any(|edge| match edge.weight() {
                        Wire::Qubit(qubit) => !qubit_map.contains_key(qubit),
                        Wire::Clbit(clbit) => !clbit_map.contains_key(clbit),
                        Wire::Var(var) => !bound_var_map.contains(var).unwrap(),
                    }),
                _ => false,
            }
        };
        let reverse_qubit_map: HashMap<Qubit, Qubit> =
            qubit_map.iter().map(|(x, y)| (*y, *x)).collect();
        let reverse_clbit_map: HashMap<Clbit, Clbit> =
            clbit_map.iter().map(|(x, y)| (*y, *x)).collect();
        let reverse_var_map = PyDict::new_bound(py);
        for (k, v) in bound_var_map.iter() {
            reverse_var_map.set_item(v, k)?;
        }
        // Copy nodes from other to self
        let mut out_map: IndexMap<NodeIndex, NodeIndex, RandomState> =
            IndexMap::with_capacity_and_hasher(other.dag.node_count(), RandomState::default());
        for old_index in other.dag.node_indices() {
            if !node_filter(old_index) {
                continue;
            }
            let mut new_node = other.dag[old_index].clone();
            if let NodeType::Operation(ref mut new_inst) = new_node {
                let new_qubit_indices: Vec<Qubit> = other
                    .qargs_interner
                    .get(new_inst.qubits)
                    .iter()
                    .map(|old_qubit| qubit_map[old_qubit])
                    .collect();
                let new_clbit_indices: Vec<Clbit> = other
                    .cargs_interner
                    .get(new_inst.clbits)
                    .iter()
                    .map(|old_clbit| clbit_map[old_clbit])
                    .collect();
                new_inst.qubits = self.qargs_interner.insert_owned(new_qubit_indices);
                new_inst.clbits = self.cargs_interner.insert_owned(new_clbit_indices);
                self.increment_op(new_inst.op.name());
            }
            let new_index = self.dag.add_node(new_node);
            out_map.insert(old_index, new_index);
        }
        // If no nodes are copied bail here since there is nothing left
        // to do.
        if out_map.is_empty() {
            match self.dag.remove_node(node) {
                Some(NodeType::Operation(packed)) => {
                    let op_name = packed.op.name();
                    self.decrement_op(op_name);
                }
                _ => unreachable!("Must be called with valid operation node!"),
            }
            // Return a new empty map to clear allocation from out_map
            return Ok(IndexMap::default());
        }
        // Copy edges from other to self
        for edge in other.dag.edge_references().filter(|edge| {
            out_map.contains_key(&edge.target()) && out_map.contains_key(&edge.source())
        }) {
            self.dag.add_edge(
                out_map[&edge.source()],
                out_map[&edge.target()],
                match edge.weight() {
                    Wire::Qubit(qubit) => Wire::Qubit(qubit_map[qubit]),
                    Wire::Clbit(clbit) => Wire::Clbit(clbit_map[clbit]),
                    Wire::Var(var) => Wire::Var(bound_var_map.get_item(var)?.unwrap().unbind()),
                },
            );
        }
        // Add edges to/from node to nodes in other
        let edges: Vec<(NodeIndex, NodeIndex, Wire)> = self
            .dag
            .edges_directed(node, Incoming)
            .map(|x| (x.source(), x.target(), x.weight().clone()))
            .collect();
        for (source, _target, weight) in edges {
            let wire_input_id = match weight {
                Wire::Qubit(qubit) => other
                    .qubit_io_map
                    .get(reverse_qubit_map[&qubit].0 as usize)
                    .map(|x| x[0]),
                Wire::Clbit(clbit) => other
                    .clbit_io_map
                    .get(reverse_clbit_map[&clbit].0 as usize)
                    .map(|x| x[0]),
                Wire::Var(ref var) => {
                    let index = &reverse_var_map.get_item(var)?.unwrap().unbind();
                    other.var_input_map.get(py, index)
                }
            };
            let old_index =
                wire_input_id.and_then(|x| other.dag.neighbors_directed(x, Outgoing).next());
            let target_out = match old_index {
                Some(old_index) => match out_map.get(&old_index) {
                    Some(new_index) => *new_index,
                    None => {
                        // If the index isn't in the node map we've already added the edges as
                        // part of the idle wire handling at the top of this method so just
                        // move on.
                        continue;
                    }
                },
                None => continue,
            };
            self.dag.add_edge(source, target_out, weight);
        }
        let edges: Vec<(NodeIndex, NodeIndex, Wire)> = self
            .dag
            .edges_directed(node, Outgoing)
            .map(|x| (x.source(), x.target(), x.weight().clone()))
            .collect();
        for (_source, target, weight) in edges {
            let wire_output_id = match weight {
                Wire::Qubit(qubit) => other
                    .qubit_io_map
                    .get(reverse_qubit_map[&qubit].0 as usize)
                    .map(|x| x[1]),
                Wire::Clbit(clbit) => other
                    .clbit_io_map
                    .get(reverse_clbit_map[&clbit].0 as usize)
                    .map(|x| x[1]),
                Wire::Var(ref var) => {
                    let index = &reverse_var_map.get_item(var)?.unwrap().unbind();
                    other.var_output_map.get(py, index)
                }
            };
            let old_index =
                wire_output_id.and_then(|x| other.dag.neighbors_directed(x, Incoming).next());
            let source_out = match old_index {
                Some(old_index) => match out_map.get(&old_index) {
                    Some(new_index) => *new_index,
                    None => {
                        // If the index isn't in the node map we've already added the edges as
                        // part of the idle wire handling at the top of this method so just
                        // move on.
                        continue;
                    }
                },
                None => continue,
            };
            self.dag.add_edge(source_out, target, weight);
        }
        // Remove node
        if let NodeType::Operation(inst) = &self.dag[node] {
            self.decrement_op(inst.op.name().to_string().as_str());
        }
        self.dag.remove_node(node);
        Ok(out_map)
    }

    fn add_var(&mut self, py: Python, var: &Bound<PyAny>, type_: DAGVarType) -> PyResult<()> {
        // The setup of the initial graph structure between an "in" and an "out" node is the same as
        // the bit-related `_add_wire`, but this logically needs to do different bookkeeping around
        // tracking the properties
        if !var.getattr("standalone")?.extract::<bool>()? {
            return Err(DAGCircuitError::new_err(
                "cannot add variables that wrap `Clbit` or `ClassicalRegister` instances",
            ));
        }
        let var_name: String = var.getattr("name")?.extract::<String>()?;
        if let Some(previous) = self.vars_info.get(&var_name) {
            if var.eq(previous.var.clone_ref(py))? {
                return Err(DAGCircuitError::new_err("already present in the circuit"));
            }
            return Err(DAGCircuitError::new_err(
                "cannot add var as its name shadows an existing var",
            ));
        }
        let in_node = NodeType::VarIn(var.clone().unbind());
        let out_node = NodeType::VarOut(var.clone().unbind());
        let in_index = self.dag.add_node(in_node);
        let out_index = self.dag.add_node(out_node);
        self.dag
            .add_edge(in_index, out_index, Wire::Var(var.clone().unbind()));
        self.var_input_map
            .insert(py, var.clone().unbind(), in_index);
        self.var_output_map
            .insert(py, var.clone().unbind(), out_index);
        self.vars_by_type[type_ as usize]
            .bind(py)
            .add(var.clone().unbind())?;
        self.vars_info.insert(
            var_name,
            DAGVarInfo {
                var: var.clone().unbind(),
                type_,
                in_node: in_index,
                out_node: out_index,
            },
        );
        Ok(())
    }

    fn check_op_addition(&self, py: Python, inst: &PackedInstruction) -> PyResult<()> {
        if let Some(condition) = inst.condition() {
            self._check_condition(py, inst.op.name(), condition.bind(py))?;
        }

        for b in self.qargs_interner.get(inst.qubits) {
            if self.qubit_io_map.len() - 1 < b.0 as usize {
                return Err(DAGCircuitError::new_err(format!(
                    "qubit {} not found in output map",
                    self.qubits.get(*b).unwrap()
                )));
            }
        }

        for b in self.cargs_interner.get(inst.clbits) {
            if !self.clbit_io_map.len() - 1 < b.0 as usize {
                return Err(DAGCircuitError::new_err(format!(
                    "clbit {} not found in output map",
                    self.clbits.get(*b).unwrap()
                )));
            }
        }

        if self.may_have_additional_wires(py, inst) {
            let (clbits, vars) = self.additional_wires(py, inst.op.view(), inst.condition())?;
            for b in clbits {
                if !self.clbit_io_map.len() - 1 < b.0 as usize {
                    return Err(DAGCircuitError::new_err(format!(
                        "clbit {} not found in output map",
                        self.clbits.get(b).unwrap()
                    )));
                }
            }
            for v in vars {
                if !self.var_output_map.contains_key(py, &v) {
                    return Err(DAGCircuitError::new_err(format!(
                        "var {} not found in output map",
                        v
                    )));
                }
            }
        }
        Ok(())
    }

    /// Alternative constructor, builds a DAGCircuit with a fixed capacity.
    ///
    /// # Arguments:
    /// - `py`: Python GIL token
    /// - `num_qubits`: Number of qubits in the circuit
    /// - `num_clbits`: Number of classical bits in the circuit.
    /// - `num_vars`: (Optional) number of variables in the circuit.
    /// - `num_ops`: (Optional) number of operations in the circuit.
    /// - `num_edges`: (Optional) If known, number of edges in the circuit.
    pub fn with_capacity(
        py: Python,
        num_qubits: usize,
        num_clbits: usize,
        num_vars: Option<usize>,
        num_ops: Option<usize>,
        num_edges: Option<usize>,
    ) -> PyResult<Self> {
        let num_ops: usize = num_ops.unwrap_or_default();
        let num_vars = num_vars.unwrap_or_default();
        let num_edges = num_edges.unwrap_or(
            num_qubits +    // 1 edge between the input node and the output node or 1st op node.
            num_clbits +    // 1 edge between the input node and the output node or 1st op node.
            num_vars +      // 1 edge between the input node and the output node or 1st op node.
            num_ops, // In Average there will be 3 edges (2 qubits and 1 clbit, or 3 qubits) per op_node.
        );

        let num_nodes = num_qubits * 2 + // One input + One output node per qubit
            num_clbits * 2 +    // One input + One output node per clbit
            num_vars * 2 +  // One input + output node per variable
            num_ops;

        Ok(Self {
            name: None,
            metadata: Some(PyDict::new_bound(py).unbind().into()),
            calibrations: HashMap::default(),
            dag: StableDiGraph::with_capacity(num_nodes, num_edges),
            qregs: PyDict::new_bound(py).unbind(),
            cregs: PyDict::new_bound(py).unbind(),
            qargs_interner: Interner::with_capacity(num_qubits),
            cargs_interner: Interner::with_capacity(num_clbits),
            qubits: BitData::with_capacity(py, "qubits".to_string(), num_qubits),
            clbits: BitData::with_capacity(py, "clbits".to_string(), num_clbits),
            global_phase: Param::Float(0.),
            duration: None,
            unit: "dt".to_string(),
            qubit_locations: PyDict::new_bound(py).unbind(),
            clbit_locations: PyDict::new_bound(py).unbind(),
            qubit_io_map: Vec::with_capacity(num_qubits),
            clbit_io_map: Vec::with_capacity(num_clbits),
            var_input_map: _VarIndexMap::new(py),
            var_output_map: _VarIndexMap::new(py),
            op_names: IndexMap::default(),
            control_flow_module: PyControlFlowModule::new(py)?,
            vars_info: HashMap::with_capacity(num_vars),
            vars_by_type: [
                PySet::empty_bound(py)?.unbind(),
                PySet::empty_bound(py)?.unbind(),
                PySet::empty_bound(py)?.unbind(),
            ],
        })
    }

    /// Get qargs from an intern index
    pub fn get_qargs(&self, index: Interned<[Qubit]>) -> &[Qubit] {
        self.qargs_interner.get(index)
    }

    /// Get cargs from an intern index
    pub fn get_cargs(&self, index: Interned<[Clbit]>) -> &[Clbit] {
        self.cargs_interner.get(index)
    }

    /// Insert a new 1q standard gate on incoming qubit
    pub fn insert_1q_on_incoming_qubit(
        &mut self,
        new_gate: (StandardGate, &[f64]),
        old_index: NodeIndex,
    ) {
        self.increment_op(new_gate.0.name());
        let old_node = &self.dag[old_index];
        let inst = if let NodeType::Operation(old_node) = old_node {
            PackedInstruction {
                op: new_gate.0.into(),
                qubits: old_node.qubits,
                clbits: old_node.clbits,
                params: (!new_gate.1.is_empty())
                    .then(|| Box::new(new_gate.1.iter().map(|x| Param::Float(*x)).collect())),
                extra_attrs: None,
                #[cfg(feature = "cache_pygates")]
                py_op: OnceCell::new(),
            }
        } else {
            panic!("This method only works if provided index is an op node");
        };
        let new_index = self.dag.add_node(NodeType::Operation(inst));
        let (parent_index, edge_index, weight) = self
            .dag
            .edges_directed(old_index, Incoming)
            .map(|edge| (edge.source(), edge.id(), edge.weight().clone()))
            .next()
            .unwrap();
        self.dag.add_edge(parent_index, new_index, weight.clone());
        self.dag.add_edge(new_index, old_index, weight);
        self.dag.remove_edge(edge_index);
    }

    /// Remove a sequence of 1 qubit nodes from the dag
    /// This must only be called if all the nodes operate
    /// on a single qubit with no other wires in or out of any nodes
    pub fn remove_1q_sequence(&mut self, sequence: &[NodeIndex]) {
        let (parent_index, weight) = self
            .dag
            .edges_directed(*sequence.first().unwrap(), Incoming)
            .map(|edge| (edge.source(), edge.weight().clone()))
            .next()
            .unwrap();
        let child_index = self
            .dag
            .edges_directed(*sequence.last().unwrap(), Outgoing)
            .map(|edge| edge.target())
            .next()
            .unwrap();
        self.dag.add_edge(parent_index, child_index, weight);
        for node in sequence {
            match self.dag.remove_node(*node) {
                Some(NodeType::Operation(packed)) => {
                    let op_name = packed.op.name();
                    self.decrement_op(op_name);
                }
                _ => panic!("Must be called with valid operation node!"),
            }
        }
    }

    /// Replace a node with individual operations from a provided callback
    /// function on each qubit of that node.
    #[allow(unused_variables)]
    pub fn replace_node_with_1q_ops<F>(
        &mut self,
        py: Python, // Unused if cache_pygates isn't enabled
        node: NodeIndex,
        insert: F,
    ) -> PyResult<()>
    where
        F: Fn(&Wire) -> PyResult<OperationFromPython>,
    {
        let mut edge_list: Vec<(NodeIndex, NodeIndex, Wire)> = Vec::with_capacity(2);
        for (source, in_weight) in self
            .dag
            .edges_directed(node, Incoming)
            .map(|x| (x.source(), x.weight()))
        {
            for (target, out_weight) in self
                .dag
                .edges_directed(node, Outgoing)
                .map(|x| (x.target(), x.weight()))
            {
                if in_weight == out_weight {
                    edge_list.push((source, target, in_weight.clone()));
                }
            }
        }
        for (source, target, weight) in edge_list {
            let new_op = insert(&weight)?;
            self.increment_op(new_op.operation.name());
            let qubits = if let Wire::Qubit(qubit) = weight {
                vec![qubit]
            } else {
                panic!("This method only works if the gate being replaced has no classical incident wires")
            };
            #[cfg(feature = "cache_pygates")]
            let py_op = match new_op.operation.view() {
                OperationRef::Standard(_) => OnceCell::new(),
                OperationRef::Gate(gate) => OnceCell::from(gate.gate.clone_ref(py)),
                OperationRef::Instruction(instruction) => {
                    OnceCell::from(instruction.instruction.clone_ref(py))
                }
                OperationRef::Operation(op) => OnceCell::from(op.operation.clone_ref(py)),
            };
            let inst = PackedInstruction {
                op: new_op.operation,
                qubits: self.qargs_interner.insert_owned(qubits),
                clbits: self.cargs_interner.get_default(),
                params: (!new_op.params.is_empty()).then(|| Box::new(new_op.params)),
                extra_attrs: new_op.extra_attrs,
                #[cfg(feature = "cache_pygates")]
                py_op: py_op,
            };
            let new_index = self.dag.add_node(NodeType::Operation(inst));
            self.dag.add_edge(source, new_index, weight.clone());
            self.dag.add_edge(new_index, target, weight);
        }

        match self.dag.remove_node(node) {
            Some(NodeType::Operation(packed)) => {
                let op_name = packed.op.name();
                self.decrement_op(op_name);
            }
            _ => panic!("Must be called with valid operation node"),
        }
        Ok(())
    }

    pub fn add_global_phase(&mut self, py: Python, value: &Param) -> PyResult<()> {
        match value {
            Param::Obj(_) => {
                return Err(PyTypeError::new_err(
                    "Invalid parameter type, only float and parameter expression are supported",
                ))
            }
            _ => self.set_global_phase(add_global_phase(py, &self.global_phase, value)?)?,
        }
        Ok(())
    }

    pub fn calibrations_empty(&self) -> bool {
        self.calibrations.is_empty()
    }

    pub fn has_calibration_for_index(&self, py: Python, node_index: NodeIndex) -> PyResult<bool> {
        let node = &self.dag[node_index];
        if let NodeType::Operation(instruction) = node {
            if !self.calibrations.contains_key(instruction.op.name()) {
                return Ok(false);
            }
            let params = match &instruction.params {
                Some(params) => {
                    let mut out_params = Vec::new();
                    for p in params.iter() {
                        if let Param::ParameterExpression(exp) = p {
                            let exp = exp.bind(py);
                            if !exp.getattr(intern!(py, "parameters"))?.is_truthy()? {
                                let as_py_float = exp.call_method0(intern!(py, "__float__"))?;
                                out_params.push(as_py_float.unbind());
                                continue;
                            }
                        }
                        out_params.push(p.to_object(py));
                    }
                    PyTuple::new_bound(py, out_params)
                }
                None => PyTuple::empty_bound(py),
            };
            let qargs = self.qargs_interner.get(instruction.qubits);
            let qubits = PyTuple::new_bound(py, qargs.iter().map(|x| x.0));
            self.calibrations[instruction.op.name()]
                .bind(py)
                .contains((qubits, params).to_object(py))
        } else {
            Err(DAGCircuitError::new_err("Specified node is not an op node"))
        }
    }

    /// Return the op name counts in the circuit
    ///
    /// Args:
    ///     py: The python token necessary for control flow recursion
    ///     recurse: Whether to recurse into control flow ops or not
    pub fn count_ops(
        &self,
        py: Python,
        recurse: bool,
    ) -> PyResult<IndexMap<String, usize, RandomState>> {
        if !recurse || !self.has_control_flow() {
            Ok(self.op_names.clone())
        } else {
            fn inner(
                py: Python,
                dag: &DAGCircuit,
                counts: &mut IndexMap<String, usize, RandomState>,
            ) -> PyResult<()> {
                for (key, value) in dag.op_names.iter() {
                    counts
                        .entry(key.clone())
                        .and_modify(|count| *count += value)
                        .or_insert(*value);
                }
                let circuit_to_dag = imports::CIRCUIT_TO_DAG.get_bound(py);
                for node in dag.dag.node_weights() {
                    let NodeType::Operation(node) = node else {
                        continue;
                    };
                    if !node.op.control_flow() {
                        continue;
                    }
                    let OperationRef::Instruction(inst) = node.op.view() else {
                        panic!("control flow op must be an instruction")
                    };
                    let blocks = inst.instruction.bind(py).getattr("blocks")?;
                    for block in blocks.iter()? {
                        let inner_dag: &DAGCircuit = &circuit_to_dag.call1((block?,))?.extract()?;
                        inner(py, inner_dag, counts)?;
                    }
                }
                Ok(())
            }
            let mut counts =
                IndexMap::with_capacity_and_hasher(self.op_names.len(), RandomState::default());
            inner(py, self, &mut counts)?;
            Ok(counts)
        }
    }

    /// Get an immutable reference to the op counts for this DAGCircuit
    ///
    /// This differs from count_ops() in that it doesn't handle control flow recursion at all
    /// and it returns a reference instead of an owned copy. If you don't need to work with
    /// control flow or ownership of the counts this is a more efficient alternative to
    /// `DAGCircuit::count_ops(py, false)`
    pub fn get_op_counts(&self) -> &IndexMap<String, usize, RandomState> {
        &self.op_names
    }

    /// Extends the DAG with valid instances of [PackedInstruction]
    pub fn extend<I>(&mut self, py: Python, iter: I) -> PyResult<Vec<NodeIndex>>
    where
        I: IntoIterator<Item = PackedInstruction>,
    {
        // Create HashSets to keep track of each bit/var's last node
        let mut qubit_last_nodes: HashMap<Qubit, NodeIndex> = HashMap::default();
        let mut clbit_last_nodes: HashMap<Clbit, NodeIndex> = HashMap::default();
        // TODO: Refactor once Vars are in rust
        // Dict [ Var: (int, VarWeight)]
        let vars_last_nodes: Bound<PyDict> = PyDict::new_bound(py);

        // Consume into iterator to obtain size hint
        let iter = iter.into_iter();
        // Store new nodes to return
        let mut new_nodes = Vec::with_capacity(iter.size_hint().1.unwrap_or_default());
        for instr in iter {
            let op_name = instr.op.name();
            let (all_cbits, vars): (Vec<Clbit>, Option<Vec<PyObject>>) = {
                if self.may_have_additional_wires(py, &instr) {
                    let mut clbits: HashSet<Clbit> =
                        HashSet::from_iter(self.cargs_interner.get(instr.clbits).iter().copied());
                    let (additional_clbits, additional_vars) =
                        self.additional_wires(py, instr.op.view(), instr.condition())?;
                    for clbit in additional_clbits {
                        clbits.insert(clbit);
                    }
                    (clbits.into_iter().collect(), Some(additional_vars))
                } else {
                    (self.cargs_interner.get(instr.clbits).to_vec(), None)
                }
            };

            // Increment the operation count
            self.increment_op(op_name);

            // Get the correct qubit indices
            let qubits_id = instr.qubits;

            // Insert op-node to graph.
            let new_node = self.dag.add_node(NodeType::Operation(instr));
            new_nodes.push(new_node);

            // Check all the qubits in this instruction.
            for qubit in self.qargs_interner.get(qubits_id) {
                // Retrieve each qubit's last node
                let qubit_last_node = *qubit_last_nodes.entry(*qubit).or_insert_with(|| {
                    // If the qubit is not in the last nodes collection, the edge between the output node and its predecessor.
                    // Then, store the predecessor's NodeIndex in the last nodes collection.
                    let output_node = self.qubit_io_map[qubit.0 as usize][1];
                    let (edge_id, predecessor_node) = self
                        .dag
                        .edges_directed(output_node, Incoming)
                        .next()
                        .map(|edge| (edge.id(), edge.source()))
                        .unwrap();
                    self.dag.remove_edge(edge_id);
                    predecessor_node
                });
                qubit_last_nodes
                    .entry(*qubit)
                    .and_modify(|val| *val = new_node);
                self.dag
                    .add_edge(qubit_last_node, new_node, Wire::Qubit(*qubit));
            }

            // Check all the clbits in this instruction.
            for clbit in all_cbits {
                let clbit_last_node = *clbit_last_nodes.entry(clbit).or_insert_with(|| {
                    // If the qubit is not in the last nodes collection, the edge between the output node and its predecessor.
                    // Then, store the predecessor's NodeIndex in the last nodes collection.
                    let output_node = self.clbit_io_map[clbit.0 as usize][1];
                    let (edge_id, predecessor_node) = self
                        .dag
                        .edges_directed(output_node, Incoming)
                        .next()
                        .map(|edge| (edge.id(), edge.source()))
                        .unwrap();
                    self.dag.remove_edge(edge_id);
                    predecessor_node
                });
                clbit_last_nodes
                    .entry(clbit)
                    .and_modify(|val| *val = new_node);
                self.dag
                    .add_edge(clbit_last_node, new_node, Wire::Clbit(clbit));
            }

            // If available, check all the vars in this instruction
            for var in vars.iter().flatten() {
                let var_last_node = if let Some(result) = vars_last_nodes.get_item(var)? {
                    let node: usize = result.extract()?;
                    vars_last_nodes.del_item(var)?;
                    NodeIndex::new(node)
                } else {
                    // If the var is not in the last nodes collection, the edge between the output node and its predecessor.
                    // Then, store the predecessor's NodeIndex in the last nodes collection.
                    let output_node = self.var_output_map.get(py, var).unwrap();
                    let (edge_id, predecessor_node) = self
                        .dag
                        .edges_directed(output_node, Incoming)
                        .next()
                        .map(|edge| (edge.id(), edge.source()))
                        .unwrap();
                    self.dag.remove_edge(edge_id);
                    predecessor_node
                };

                // Because `DAGCircuit::additional_wires` can return repeated instances of vars,
                // we need to make sure to skip those to avoid cycles.
                vars_last_nodes.set_item(var, new_node.index())?;
                if var_last_node == new_node {
                    continue;
                }
                self.dag
                    .add_edge(var_last_node, new_node, Wire::Var(var.clone_ref(py)));
            }
        }

        // Add the output_nodes back to qargs
        for (qubit, node) in qubit_last_nodes {
            let output_node = self.qubit_io_map[qubit.0 as usize][1];
            self.dag.add_edge(node, output_node, Wire::Qubit(qubit));
        }

        // Add the output_nodes back to cargs
        for (clbit, node) in clbit_last_nodes {
            let output_node = self.clbit_io_map[clbit.0 as usize][1];
            self.dag.add_edge(node, output_node, Wire::Clbit(clbit));
        }

        // Add the output_nodes back to vars
        for item in vars_last_nodes.items() {
            let (var, node): (PyObject, usize) = item.extract()?;
            let output_node = self.var_output_map.get(py, &var).unwrap();
            self.dag
                .add_edge(NodeIndex::new(node), output_node, Wire::Var(var));
        }

        Ok(new_nodes)
    }

    /// Alternative constructor to build an instance of [DAGCircuit] from a `QuantumCircuit`.
    pub(crate) fn from_circuit(
        py: Python,
        qc: QuantumCircuitData,
        copy_op: bool,
        qubit_order: Option<Vec<Bound<PyAny>>>,
        clbit_order: Option<Vec<Bound<PyAny>>>,
    ) -> PyResult<DAGCircuit> {
        // Extract necessary attributes
        let qc_data = qc.data;
        let num_qubits = qc_data.num_qubits();
        let num_clbits = qc_data.num_clbits();
        let num_ops = qc_data.__len__();
        let num_vars = qc.declared_vars.len() + qc.input_vars.len() + qc.captured_vars.len();

        // Build DAGCircuit with capacity
        let mut new_dag = DAGCircuit::with_capacity(
            py,
            num_qubits,
            num_clbits,
            Some(num_vars),
            Some(num_ops),
            None,
        )?;

        // Assign other necessary data
        new_dag.name = qc.name.map(|ob| ob.unbind());

        // Avoid manually acquiring the GIL.
        new_dag.global_phase = match qc_data.global_phase() {
            Param::ParameterExpression(exp) => Param::ParameterExpression(exp.clone_ref(py)),
            Param::Float(float) => Param::Float(*float),
            _ => unreachable!("Incorrect parameter assigned for global phase"),
        };

        if let Some(calibrations) = qc.calibrations {
            new_dag.calibrations = calibrations;
        }

        new_dag.metadata = qc.metadata.map(|meta| meta.unbind());

        // Add the qubits depending on order.
        let qubit_map: Option<Vec<Qubit>> = if let Some(qubit_ordering) = qubit_order {
            let mut ordered_vec = Vec::from_iter((0..num_qubits as u32).map(Qubit));
            qubit_ordering
                .into_iter()
                .try_for_each(|qubit| -> PyResult<()> {
                    if new_dag.qubits.find(&qubit).is_some() {
                        return Err(DAGCircuitError::new_err(format!(
                            "duplicate qubits {}",
                            &qubit
                        )));
                    }
                    let qubit_index = qc_data.qubits().find(&qubit).unwrap();
                    ordered_vec[qubit_index.0 as usize] =
                        new_dag.add_qubit_unchecked(py, &qubit)?;
                    Ok(())
                })?;
            Some(ordered_vec)
        } else {
            qc_data
                .qubits()
                .bits()
                .iter()
                .try_for_each(|qubit| -> PyResult<_> {
                    new_dag.add_qubit_unchecked(py, qubit.bind(py))?;
                    Ok(())
                })?;
            None
        };

        // Add the clbits depending on order.
        let clbit_map: Option<Vec<Clbit>> = if let Some(clbit_ordering) = clbit_order {
            let mut ordered_vec = Vec::from_iter((0..num_clbits as u32).map(Clbit));
            clbit_ordering
                .into_iter()
                .try_for_each(|clbit| -> PyResult<()> {
                    if new_dag.clbits.find(&clbit).is_some() {
                        return Err(DAGCircuitError::new_err(format!(
                            "duplicate clbits {}",
                            &clbit
                        )));
                    };
                    let clbit_index = qc_data.clbits().find(&clbit).unwrap();
                    ordered_vec[clbit_index.0 as usize] =
                        new_dag.add_clbit_unchecked(py, &clbit)?;
                    Ok(())
                })?;
            Some(ordered_vec)
        } else {
            qc_data
                .clbits()
                .bits()
                .iter()
                .try_for_each(|clbit| -> PyResult<()> {
                    new_dag.add_clbit_unchecked(py, clbit.bind(py))?;
                    Ok(())
                })?;
            None
        };

        // Add all of the new vars.
        for var in &qc.declared_vars {
            new_dag.add_var(py, var, DAGVarType::Declare)?;
        }

        for var in &qc.input_vars {
            new_dag.add_var(py, var, DAGVarType::Input)?;
        }

        for var in &qc.captured_vars {
            new_dag.add_var(py, var, DAGVarType::Capture)?;
        }

        // Add all the registers
        if let Some(qregs) = qc.qregs {
            for qreg in qregs.iter() {
                new_dag.add_qreg(py, &qreg)?;
            }
        }

        if let Some(cregs) = qc.cregs {
            for creg in cregs.iter() {
                new_dag.add_creg(py, &creg)?;
            }
        }

        // Pre-process and re-intern all indices again.
        let instructions: Vec<PackedInstruction> = qc_data
            .iter()
            .map(|instr| -> PyResult<PackedInstruction> {
                // Re-map the qubits
                let new_qargs = if let Some(qubit_mapping) = &qubit_map {
                    let qargs = qc_data
                        .get_qargs(instr.qubits)
                        .iter()
                        .map(|bit| qubit_mapping[bit.0 as usize])
                        .collect();
                    new_dag.qargs_interner.insert_owned(qargs)
                } else {
                    new_dag
                        .qargs_interner
                        .insert(qc_data.get_qargs(instr.qubits))
                };
                // Remap the clbits
                let new_cargs = if let Some(clbit_mapping) = &clbit_map {
                    let qargs = qc_data
                        .get_cargs(instr.clbits)
                        .iter()
                        .map(|bit| clbit_mapping[bit.0 as usize])
                        .collect();
                    new_dag.cargs_interner.insert_owned(qargs)
                } else {
                    new_dag
                        .cargs_interner
                        .insert(qc_data.get_cargs(instr.clbits))
                };
                // Copy the operations

                Ok(PackedInstruction {
                    op: if copy_op {
                        instr.op.py_deepcopy(py, None)?
                    } else {
                        instr.op.clone()
                    },
                    qubits: new_qargs,
                    clbits: new_cargs,
                    params: instr.params.clone(),
                    extra_attrs: instr.extra_attrs.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceCell::new(),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Finally add all the instructions back
        new_dag.extend(py, instructions)?;

        Ok(new_dag)
    }

    /// Builds a [DAGCircuit] based on an instance of [CircuitData].
    pub fn from_circuit_data(
        py: Python,
        circuit_data: CircuitData,
        copy_op: bool,
    ) -> PyResult<Self> {
        let circ = QuantumCircuitData {
            data: circuit_data,
            name: None,
            calibrations: None,
            metadata: None,
            qregs: None,
            cregs: None,
            input_vars: Vec::new(),
            captured_vars: Vec::new(),
            declared_vars: Vec::new(),
        };
        Self::from_circuit(py, circ, copy_op, None, None)
    }
}

/// Add to global phase. Global phase can only be Float or ParameterExpression so this
/// does not handle the full possibility of parameter values.
fn add_global_phase(py: Python, phase: &Param, other: &Param) -> PyResult<Param> {
    Ok(match [phase, other] {
        [Param::Float(a), Param::Float(b)] => Param::Float(a + b),
        [Param::Float(a), Param::ParameterExpression(b)] => Param::ParameterExpression(
            b.clone_ref(py)
                .call_method1(py, intern!(py, "__radd__"), (*a,))?,
        ),
        [Param::ParameterExpression(a), Param::Float(b)] => Param::ParameterExpression(
            a.clone_ref(py)
                .call_method1(py, intern!(py, "__add__"), (*b,))?,
        ),
        [Param::ParameterExpression(a), Param::ParameterExpression(b)] => {
            Param::ParameterExpression(a.clone_ref(py).call_method1(
                py,
                intern!(py, "__add__"),
                (b,),
            )?)
        }
        _ => panic!("Invalid global phase"),
    })
}

type SortKeyType<'a> = (&'a [Qubit], &'a [Clbit]);

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

// TODO: delete once we move to a version of Rustworkx which includes
//   this implementation as part of rustworkx-core.
//   PR: https://github.com/Qiskit/rustworkx/pull/1235
pub mod isomorphism {
    pub mod vf2 {
        #![allow(clippy::too_many_arguments)]
        // This module was originally forked from petgraph's isomorphism module @ v0.5.0
        // to handle PyDiGraph inputs instead of petgraph's generic Graph. However it has
        // since diverged significantly from the original petgraph implementation.

        use std::cmp::{Ordering, Reverse};
        use std::convert::Infallible;
        use std::error::Error;
        use std::fmt::{Debug, Display, Formatter};
        use std::iter::Iterator;
        use std::marker;
        use std::ops::Deref;

        use hashbrown::HashMap;
        use rustworkx_core::dictmap::*;

        use rustworkx_core::petgraph::data::{Build, Create, DataMap};
        use rustworkx_core::petgraph::stable_graph::NodeIndex;
        use rustworkx_core::petgraph::visit::{
            Data, EdgeCount, EdgeRef, GraphBase, GraphProp, IntoEdgeReferences, IntoEdges,
            IntoEdgesDirected, IntoNeighbors, IntoNeighborsDirected, IntoNodeIdentifiers,
            NodeCount, NodeIndexable,
        };
        use rustworkx_core::petgraph::{Directed, Incoming, Outgoing};

        use rayon::slice::ParallelSliceMut;

        /// Returns `true` if we can map every element of `xs` to a unique
        /// element of `ys` while using `matcher` func to compare two elements.
        fn is_subset<T1: Copy, T2: Copy, F, E>(
            xs: &[T1],
            ys: &[T2],
            matcher: &mut F,
        ) -> Result<bool, E>
        where
            F: FnMut(T1, T2) -> Result<bool, E>,
        {
            let mut valid = vec![true; ys.len()];
            for &a in xs {
                let mut found = false;
                for (&b, free) in ys.iter().zip(valid.iter_mut()) {
                    if *free && matcher(a, b)? {
                        found = true;
                        *free = false;
                        break;
                    }
                }

                if !found {
                    return Ok(false);
                }
            }

            Ok(true)
        }

        #[inline]
        fn sorted<N: std::cmp::PartialOrd>(x: &mut (N, N)) {
            let (a, b) = x;
            if b < a {
                std::mem::swap(a, b)
            }
        }

        /// Returns the adjacency matrix of a graph as a dictionary
        /// with `(i, j)` entry equal to number of edges from node `i` to node `j`.
        fn adjacency_matrix<G>(graph: G) -> HashMap<(NodeIndex, NodeIndex), usize>
        where
            G: GraphProp + GraphBase<NodeId = NodeIndex> + EdgeCount + IntoEdgeReferences,
        {
            let mut matrix = HashMap::with_capacity(graph.edge_count());
            for edge in graph.edge_references() {
                let mut item = (edge.source(), edge.target());
                if !graph.is_directed() {
                    sorted(&mut item);
                }
                let entry = matrix.entry(item).or_insert(0);
                *entry += 1;
            }
            matrix
        }

        /// Returns the number of edges from node `a` to node `b`.
        fn edge_multiplicity<G>(
            graph: &G,
            matrix: &HashMap<(NodeIndex, NodeIndex), usize>,
            a: NodeIndex,
            b: NodeIndex,
        ) -> usize
        where
            G: GraphProp + GraphBase<NodeId = NodeIndex>,
        {
            let mut item = (a, b);
            if !graph.is_directed() {
                sorted(&mut item);
            }
            *matrix.get(&item).unwrap_or(&0)
        }

        /// Nodes `a`, `b` are adjacent if the number of edges
        /// from node `a` to node `b` is greater than `val`.
        fn is_adjacent<G>(
            graph: &G,
            matrix: &HashMap<(NodeIndex, NodeIndex), usize>,
            a: NodeIndex,
            b: NodeIndex,
            val: usize,
        ) -> bool
        where
            G: GraphProp + GraphBase<NodeId = NodeIndex>,
        {
            edge_multiplicity(graph, matrix, a, b) >= val
        }

        trait NodeSorter<G>
        where
            G: GraphBase<NodeId = NodeIndex> + DataMap + NodeCount + EdgeCount + IntoEdgeReferences,
            G::NodeWeight: Clone,
            G::EdgeWeight: Clone,
        {
            type OutputGraph: GraphBase<NodeId = NodeIndex>
                + Create
                + Data<NodeWeight = G::NodeWeight, EdgeWeight = G::EdgeWeight>;

            fn sort(&self, _: G) -> Vec<NodeIndex>;

            fn reorder(&self, graph: G) -> (Self::OutputGraph, HashMap<usize, usize>) {
                let order = self.sort(graph);

                let mut new_graph =
                    Self::OutputGraph::with_capacity(graph.node_count(), graph.edge_count());
                let mut id_map: HashMap<NodeIndex, NodeIndex> =
                    HashMap::with_capacity(graph.node_count());
                for node_index in order {
                    let node_data = graph.node_weight(node_index).unwrap();
                    let new_index = new_graph.add_node(node_data.clone());
                    id_map.insert(node_index, new_index);
                }
                for edge in graph.edge_references() {
                    let edge_w = edge.weight();
                    let p_index = id_map[&edge.source()];
                    let c_index = id_map[&edge.target()];
                    new_graph.add_edge(p_index, c_index, edge_w.clone());
                }
                (
                    new_graph,
                    id_map.iter().map(|(k, v)| (v.index(), k.index())).collect(),
                )
            }
        }

        /// Sort nodes based on node ids.
        struct DefaultIdSorter {}

        impl DefaultIdSorter {
            pub fn new() -> Self {
                Self {}
            }
        }

        impl<G> NodeSorter<G> for DefaultIdSorter
        where
            G: Deref
                + GraphBase<NodeId = NodeIndex>
                + DataMap
                + NodeCount
                + EdgeCount
                + IntoEdgeReferences
                + IntoNodeIdentifiers,
            G::Target: GraphBase<NodeId = NodeIndex>
                + Data<NodeWeight = G::NodeWeight, EdgeWeight = G::EdgeWeight>
                + Create,
            G::NodeWeight: Clone,
            G::EdgeWeight: Clone,
        {
            type OutputGraph = G::Target;
            fn sort(&self, graph: G) -> Vec<NodeIndex> {
                graph.node_identifiers().collect()
            }
        }

        /// Sort nodes based on VF2++ heuristic.
        struct Vf2ppSorter {}

        impl Vf2ppSorter {
            pub fn new() -> Self {
                Self {}
            }
        }

        impl<G> NodeSorter<G> for Vf2ppSorter
        where
            G: Deref
                + GraphProp
                + GraphBase<NodeId = NodeIndex>
                + DataMap
                + NodeCount
                + NodeIndexable
                + EdgeCount
                + IntoNodeIdentifiers
                + IntoEdgesDirected,
            G::Target: GraphBase<NodeId = NodeIndex>
                + Data<NodeWeight = G::NodeWeight, EdgeWeight = G::EdgeWeight>
                + Create,
            G::NodeWeight: Clone,
            G::EdgeWeight: Clone,
        {
            type OutputGraph = G::Target;
            fn sort(&self, graph: G) -> Vec<NodeIndex> {
                let n = graph.node_bound();

                let dout: Vec<usize> = (0..n)
                    .map(|idx| {
                        graph
                            .neighbors_directed(graph.from_index(idx), Outgoing)
                            .count()
                    })
                    .collect();

                let mut din: Vec<usize> = vec![0; n];
                if graph.is_directed() {
                    din = (0..n)
                        .map(|idx| {
                            graph
                                .neighbors_directed(graph.from_index(idx), Incoming)
                                .count()
                        })
                        .collect();
                }

                let mut conn_in: Vec<usize> = vec![0; n];
                let mut conn_out: Vec<usize> = vec![0; n];

                let mut order: Vec<NodeIndex> = Vec::with_capacity(n);

                // Process BFS level
                let mut process = |mut vd: Vec<usize>| -> Vec<usize> {
                    // repeatedly bring largest element in front.
                    for i in 0..vd.len() {
                        let (index, &item) = vd[i..]
                            .iter()
                            .enumerate()
                            .max_by_key(|&(_, &node)| {
                                (
                                    conn_in[node],
                                    dout[node],
                                    conn_out[node],
                                    din[node],
                                    Reverse(node),
                                )
                            })
                            .unwrap();

                        vd.swap(i, i + index);
                        order.push(NodeIndex::new(item));

                        for neigh in graph.neighbors_directed(graph.from_index(item), Outgoing) {
                            conn_in[graph.to_index(neigh)] += 1;
                        }

                        if graph.is_directed() {
                            for neigh in graph.neighbors_directed(graph.from_index(item), Incoming)
                            {
                                conn_out[graph.to_index(neigh)] += 1;
                            }
                        }
                    }
                    vd
                };

                let mut seen: Vec<bool> = vec![false; n];

                // Create BFS Tree from root and process each level.
                let mut bfs_tree = |root: usize| {
                    if seen[root] {
                        return;
                    }

                    let mut next_level: Vec<usize> = Vec::new();

                    seen[root] = true;
                    next_level.push(root);
                    while !next_level.is_empty() {
                        let this_level = next_level;
                        let this_level = process(this_level);

                        next_level = Vec::new();
                        for bfs_node in this_level {
                            for neighbor in
                                graph.neighbors_directed(graph.from_index(bfs_node), Outgoing)
                            {
                                let neigh = graph.to_index(neighbor);
                                if !seen[neigh] {
                                    seen[neigh] = true;
                                    next_level.push(neigh);
                                }
                            }
                        }
                    }
                };

                let mut sorted_nodes: Vec<usize> =
                    graph.node_identifiers().map(|node| node.index()).collect();
                sorted_nodes.par_sort_by_key(|&node| (dout[node], din[node], Reverse(node)));
                sorted_nodes.reverse();

                for node in sorted_nodes {
                    bfs_tree(node);
                }

                order
            }
        }

        #[derive(Debug)]
        pub struct Vf2State<G> {
            pub graph: G,
            /// The current mapping M(s) of nodes from G0 → G1 and G1 → G0,
            /// NodeIndex::end() for no mapping.
            mapping: Vec<NodeIndex>,
            /// out[i] is non-zero if i is in either M_0(s) or Tout_0(s)
            /// These are all the next vertices that are not mapped yet, but
            /// have an outgoing edge from the mapping.
            out: Vec<usize>,
            /// ins[i] is non-zero if i is in either M_0(s) or Tin_0(s)
            /// These are all the incoming vertices, those not mapped yet, but
            /// have an edge from them into the mapping.
            /// Unused if graph is undirected -- it's identical with out in that case.
            ins: Vec<usize>,
            out_size: usize,
            ins_size: usize,
            adjacency_matrix: HashMap<(NodeIndex, NodeIndex), usize>,
            generation: usize,
            _etype: marker::PhantomData<Directed>,
        }

        impl<G> Vf2State<G>
        where
            G: GraphBase<NodeId = NodeIndex> + GraphProp + NodeCount + EdgeCount,
            for<'a> &'a G: GraphBase<NodeId = NodeIndex>
                + GraphProp
                + NodeCount
                + EdgeCount
                + IntoEdgesDirected,
        {
            pub fn new(graph: G) -> Self {
                let c0 = graph.node_count();
                let is_directed = graph.is_directed();
                let adjacency_matrix = adjacency_matrix(&graph);
                Vf2State {
                    graph,
                    mapping: vec![NodeIndex::end(); c0],
                    out: vec![0; c0],
                    ins: vec![0; c0 * (is_directed as usize)],
                    out_size: 0,
                    ins_size: 0,
                    adjacency_matrix,
                    generation: 0,
                    _etype: marker::PhantomData,
                }
            }

            /// Return **true** if we have a complete mapping
            pub fn is_complete(&self) -> bool {
                self.generation == self.mapping.len()
            }

            /// Add mapping **from** <-> **to** to the state.
            pub fn push_mapping(&mut self, from: NodeIndex, to: NodeIndex) {
                self.generation += 1;
                let s = self.generation;
                self.mapping[from.index()] = to;
                // update T0 & T1 ins/outs
                // T0out: Node in G0 not in M0 but successor of a node in M0.
                // st.out[0]: Node either in M0 or successor of M0
                for ix in self.graph.neighbors(from) {
                    if self.out[ix.index()] == 0 {
                        self.out[ix.index()] = s;
                        self.out_size += 1;
                    }
                }
                if self.graph.is_directed() {
                    for ix in self.graph.neighbors_directed(from, Incoming) {
                        if self.ins[ix.index()] == 0 {
                            self.ins[ix.index()] = s;
                            self.ins_size += 1;
                        }
                    }
                }
            }

            /// Restore the state to before the last added mapping
            pub fn pop_mapping(&mut self, from: NodeIndex) {
                let s = self.generation;
                self.generation -= 1;

                // undo (n, m) mapping
                self.mapping[from.index()] = NodeIndex::end();

                // unmark in ins and outs
                for ix in self.graph.neighbors(from) {
                    if self.out[ix.index()] == s {
                        self.out[ix.index()] = 0;
                        self.out_size -= 1;
                    }
                }
                if self.graph.is_directed() {
                    for ix in self.graph.neighbors_directed(from, Incoming) {
                        if self.ins[ix.index()] == s {
                            self.ins[ix.index()] = 0;
                            self.ins_size -= 1;
                        }
                    }
                }
            }

            /// Find the next (least) node in the Tout set.
            pub fn next_out_index(&self, from_index: usize) -> Option<usize> {
                self.out[from_index..]
                    .iter()
                    .enumerate()
                    .find(move |&(index, elt)| {
                        *elt > 0 && self.mapping[from_index + index] == NodeIndex::end()
                    })
                    .map(|(index, _)| index)
            }

            /// Find the next (least) node in the Tin set.
            pub fn next_in_index(&self, from_index: usize) -> Option<usize> {
                self.ins[from_index..]
                    .iter()
                    .enumerate()
                    .find(move |&(index, elt)| {
                        *elt > 0 && self.mapping[from_index + index] == NodeIndex::end()
                    })
                    .map(|(index, _)| index)
            }

            /// Find the next (least) node in the N - M set.
            pub fn next_rest_index(&self, from_index: usize) -> Option<usize> {
                self.mapping[from_index..]
                    .iter()
                    .enumerate()
                    .find(|&(_, elt)| *elt == NodeIndex::end())
                    .map(|(index, _)| index)
            }
        }

        #[derive(Debug)]
        pub enum IsIsomorphicError<NME, EME> {
            NodeMatcherErr(NME),
            EdgeMatcherErr(EME),
        }

        impl<NME: Error, EME: Error> Display for IsIsomorphicError<NME, EME> {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                match self {
                    IsIsomorphicError::NodeMatcherErr(e) => {
                        write!(f, "Node match callback failed with: {}", e)
                    }
                    IsIsomorphicError::EdgeMatcherErr(e) => {
                        write!(f, "Edge match callback failed with: {}", e)
                    }
                }
            }
        }

        impl<NME: Error, EME: Error> Error for IsIsomorphicError<NME, EME> {}

        pub struct NoSemanticMatch;

        pub trait NodeMatcher<G0: GraphBase, G1: GraphBase> {
            type Error;
            fn enabled(&self) -> bool;
            fn eq(
                &mut self,
                _g0: &G0,
                _g1: &G1,
                _n0: G0::NodeId,
                _n1: G1::NodeId,
            ) -> Result<bool, Self::Error>;
        }

        impl<G0: GraphBase, G1: GraphBase> NodeMatcher<G0, G1> for NoSemanticMatch {
            type Error = Infallible;
            #[inline]
            fn enabled(&self) -> bool {
                false
            }
            #[inline]
            fn eq(
                &mut self,
                _g0: &G0,
                _g1: &G1,
                _n0: G0::NodeId,
                _n1: G1::NodeId,
            ) -> Result<bool, Self::Error> {
                Ok(true)
            }
        }

        impl<G0, G1, F, E> NodeMatcher<G0, G1> for F
        where
            G0: GraphBase + DataMap,
            G1: GraphBase + DataMap,
            F: FnMut(&G0::NodeWeight, &G1::NodeWeight) -> Result<bool, E>,
        {
            type Error = E;
            #[inline]
            fn enabled(&self) -> bool {
                true
            }
            #[inline]
            fn eq(
                &mut self,
                g0: &G0,
                g1: &G1,
                n0: G0::NodeId,
                n1: G1::NodeId,
            ) -> Result<bool, Self::Error> {
                if let (Some(x), Some(y)) = (g0.node_weight(n0), g1.node_weight(n1)) {
                    self(x, y)
                } else {
                    Ok(false)
                }
            }
        }

        pub trait EdgeMatcher<G0: GraphBase, G1: GraphBase> {
            type Error;
            fn enabled(&self) -> bool;
            fn eq(
                &mut self,
                _g0: &G0,
                _g1: &G1,
                e0: G0::EdgeId,
                e1: G1::EdgeId,
            ) -> Result<bool, Self::Error>;
        }

        impl<G0: GraphBase, G1: GraphBase> EdgeMatcher<G0, G1> for NoSemanticMatch {
            type Error = Infallible;
            #[inline]
            fn enabled(&self) -> bool {
                false
            }
            #[inline]
            fn eq(
                &mut self,
                _g0: &G0,
                _g1: &G1,
                _e0: G0::EdgeId,
                _e1: G1::EdgeId,
            ) -> Result<bool, Self::Error> {
                Ok(true)
            }
        }

        impl<G0, G1, F, E> EdgeMatcher<G0, G1> for F
        where
            G0: GraphBase + DataMap,
            G1: GraphBase + DataMap,
            F: FnMut(&G0::EdgeWeight, &G1::EdgeWeight) -> Result<bool, E>,
        {
            type Error = E;
            #[inline]
            fn enabled(&self) -> bool {
                true
            }
            #[inline]
            fn eq(
                &mut self,
                g0: &G0,
                g1: &G1,
                e0: G0::EdgeId,
                e1: G1::EdgeId,
            ) -> Result<bool, Self::Error> {
                if let (Some(x), Some(y)) = (g0.edge_weight(e0), g1.edge_weight(e1)) {
                    self(x, y)
                } else {
                    Ok(false)
                }
            }
        }

        /// [Graph] Return `true` if the graphs `g0` and `g1` are (sub) graph isomorphic.
        ///
        /// Using the VF2 algorithm, examining both syntactic and semantic
        /// graph isomorphism (graph structure and matching node and edge weights).
        ///
        /// The graphs should not be multigraphs.
        pub fn is_isomorphic<G0, G1, NM, EM>(
            g0: &G0,
            g1: &G1,
            node_match: NM,
            edge_match: EM,
            id_order: bool,
            ordering: Ordering,
            induced: bool,
            call_limit: Option<usize>,
        ) -> Result<bool, IsIsomorphicError<NM::Error, EM::Error>>
        where
            G0: GraphProp
                + GraphBase<NodeId = NodeIndex>
                + DataMap
                + Create
                + NodeCount
                + EdgeCount,
            for<'a> &'a G0: GraphBase<NodeId = G0::NodeId, EdgeId = G0::EdgeId>
                + Data<NodeWeight = G0::NodeWeight, EdgeWeight = G0::EdgeWeight>
                + NodeIndexable
                + IntoEdgesDirected
                + IntoNodeIdentifiers,
            G0::NodeWeight: Clone,
            G0::EdgeWeight: Clone,
            G1: GraphProp
                + GraphBase<NodeId = NodeIndex>
                + DataMap
                + Create
                + NodeCount
                + EdgeCount,
            for<'a> &'a G1: GraphBase<NodeId = G1::NodeId, EdgeId = G1::EdgeId>
                + Data<NodeWeight = G1::NodeWeight, EdgeWeight = G1::EdgeWeight>
                + NodeIndexable
                + IntoEdgesDirected
                + IntoNodeIdentifiers,
            G1::NodeWeight: Clone,
            G1::EdgeWeight: Clone,
            NM: NodeMatcher<G0, G1>,
            EM: EdgeMatcher<G0, G1>,
        {
            if (g0.node_count().cmp(&g1.node_count()).then(ordering) != ordering)
                || (g0.edge_count().cmp(&g1.edge_count()).then(ordering) != ordering)
            {
                return Ok(false);
            }

            let mut vf2 = Vf2Algorithm::new(
                g0, g1, node_match, edge_match, id_order, ordering, induced, call_limit,
            );

            match vf2.next() {
                Some(Ok(_)) => Ok(true),
                Some(Err(e)) => Err(e),
                None => Ok(false),
            }
        }

        #[derive(Copy, Clone, PartialEq, Debug)]
        enum OpenList {
            Out,
            In,
            Other,
        }

        #[derive(Clone, PartialEq, Debug)]
        enum Frame<N: marker::Copy> {
            Outer,
            Inner { nodes: [N; 2], open_list: OpenList },
            Unwind { nodes: [N; 2], open_list: OpenList },
        }

        /// An iterator which uses the VF2(++) algorithm to produce isomorphic matches
        /// between two graphs, examining both syntactic and semantic graph isomorphism
        /// (graph structure and matching node and edge weights).
        ///
        /// The graphs should not be multigraphs.
        pub struct Vf2Algorithm<G0, G1, NM, EM>
        where
            G0: GraphBase + Data,
            G1: GraphBase + Data,
            NM: NodeMatcher<G0, G1>,
            EM: EdgeMatcher<G0, G1>,
        {
            pub st: (Vf2State<G0>, Vf2State<G1>),
            pub node_match: NM,
            pub edge_match: EM,
            ordering: Ordering,
            induced: bool,
            node_map_g0: HashMap<usize, usize>,
            node_map_g1: HashMap<usize, usize>,
            stack: Vec<Frame<NodeIndex>>,
            call_limit: Option<usize>,
            _counter: usize,
        }

        impl<G0, G1, NM, EM> Vf2Algorithm<G0, G1, NM, EM>
        where
            G0: GraphProp
                + GraphBase<NodeId = NodeIndex>
                + DataMap
                + Create
                + NodeCount
                + EdgeCount,
            for<'a> &'a G0: GraphBase<NodeId = G0::NodeId, EdgeId = G0::EdgeId>
                + Data<NodeWeight = G0::NodeWeight, EdgeWeight = G0::EdgeWeight>
                + NodeIndexable
                + IntoEdgesDirected
                + IntoNodeIdentifiers,
            G0::NodeWeight: Clone,
            G0::EdgeWeight: Clone,
            G1: GraphProp
                + GraphBase<NodeId = NodeIndex>
                + DataMap
                + Create
                + NodeCount
                + EdgeCount,
            for<'a> &'a G1: GraphBase<NodeId = G1::NodeId, EdgeId = G1::EdgeId>
                + Data<NodeWeight = G1::NodeWeight, EdgeWeight = G1::EdgeWeight>
                + NodeIndexable
                + IntoEdgesDirected
                + IntoNodeIdentifiers,
            G1::NodeWeight: Clone,
            G1::EdgeWeight: Clone,
            NM: NodeMatcher<G0, G1>,
            EM: EdgeMatcher<G0, G1>,
        {
            pub fn new(
                g0: &G0,
                g1: &G1,
                node_match: NM,
                edge_match: EM,
                id_order: bool,
                ordering: Ordering,
                induced: bool,
                call_limit: Option<usize>,
            ) -> Self {
                let (g0, node_map_g0) = if id_order {
                    DefaultIdSorter::new().reorder(g0)
                } else {
                    Vf2ppSorter::new().reorder(g0)
                };

                let (g1, node_map_g1) = if id_order {
                    DefaultIdSorter::new().reorder(g1)
                } else {
                    Vf2ppSorter::new().reorder(g1)
                };

                let st = (Vf2State::new(g0), Vf2State::new(g1));
                Vf2Algorithm {
                    st,
                    node_match,
                    edge_match,
                    ordering,
                    induced,
                    node_map_g0,
                    node_map_g1,
                    stack: vec![Frame::Outer],
                    call_limit,
                    _counter: 0,
                }
            }

            fn mapping(&self) -> DictMap<usize, usize> {
                let mut mapping: DictMap<usize, usize> = DictMap::new();
                self.st
                    .1
                    .mapping
                    .iter()
                    .enumerate()
                    .for_each(|(index, val)| {
                        mapping.insert(self.node_map_g0[&val.index()], self.node_map_g1[&index]);
                    });

                mapping
            }

            fn next_candidate(
                st: &mut (Vf2State<G0>, Vf2State<G1>),
            ) -> Option<(NodeIndex, NodeIndex, OpenList)> {
                // Try the out list
                let mut to_index = st.1.next_out_index(0);
                let mut from_index = None;
                let mut open_list = OpenList::Out;

                if to_index.is_some() {
                    from_index = st.0.next_out_index(0);
                    open_list = OpenList::Out;
                }
                // Try the in list
                if to_index.is_none() || from_index.is_none() {
                    to_index = st.1.next_in_index(0);

                    if to_index.is_some() {
                        from_index = st.0.next_in_index(0);
                        open_list = OpenList::In;
                    }
                }
                // Try the other list -- disconnected graph
                if to_index.is_none() || from_index.is_none() {
                    to_index = st.1.next_rest_index(0);
                    if to_index.is_some() {
                        from_index = st.0.next_rest_index(0);
                        open_list = OpenList::Other;
                    }
                }
                match (from_index, to_index) {
                    (Some(n), Some(m)) => Some((NodeIndex::new(n), NodeIndex::new(m), open_list)),
                    // No more candidates
                    _ => None,
                }
            }

            fn next_from_ix(
                st: &mut (Vf2State<G0>, Vf2State<G1>),
                nx: NodeIndex,
                open_list: OpenList,
            ) -> Option<NodeIndex> {
                // Find the next node index to try on the `from` side of the mapping
                let start = nx.index() + 1;
                let cand0 = match open_list {
                    OpenList::Out => st.0.next_out_index(start),
                    OpenList::In => st.0.next_in_index(start),
                    OpenList::Other => st.0.next_rest_index(start),
                }
                .map(|c| c + start); // compensate for start offset.
                match cand0 {
                    None => None, // no more candidates
                    Some(ix) => {
                        debug_assert!(ix >= start);
                        Some(NodeIndex::new(ix))
                    }
                }
            }

            fn pop_state(st: &mut (Vf2State<G0>, Vf2State<G1>), nodes: [NodeIndex; 2]) {
                // Restore state.
                st.0.pop_mapping(nodes[0]);
                st.1.pop_mapping(nodes[1]);
            }

            fn push_state(st: &mut (Vf2State<G0>, Vf2State<G1>), nodes: [NodeIndex; 2]) {
                // Add mapping nx <-> mx to the state
                st.0.push_mapping(nodes[0], nodes[1]);
                st.1.push_mapping(nodes[1], nodes[0]);
            }

            fn is_feasible(
                st: &mut (Vf2State<G0>, Vf2State<G1>),
                nodes: [NodeIndex; 2],
                node_match: &mut NM,
                edge_match: &mut EM,
                ordering: Ordering,
                induced: bool,
            ) -> Result<bool, IsIsomorphicError<NM::Error, EM::Error>> {
                // Check syntactic feasibility of mapping by ensuring adjacencies
                // of nx map to adjacencies of mx.
                //
                // nx == map to => mx
                //
                // R_succ
                //
                // Check that every neighbor of nx is mapped to a neighbor of mx,
                // then check the reverse, from mx to nx. Check that they have the same
                // count of edges.
                //
                // Note: We want to check the lookahead measures here if we can,
                // R_out: Equal for G0, G1: Card(Succ(G, n) ^ Tout); for both Succ and Pred
                // R_in: Same with Tin
                // R_new: Equal for G0, G1: Ñ n Pred(G, n); both Succ and Pred,
                //      Ñ is G0 - M - Tin - Tout
                let end = NodeIndex::end();
                let mut succ_count = [0, 0];
                for n_neigh in st.0.graph.neighbors(nodes[0]) {
                    succ_count[0] += 1;
                    if !induced {
                        continue;
                    }
                    // handle the self loop case; it's not in the mapping (yet)
                    let m_neigh = if nodes[0] != n_neigh {
                        st.0.mapping[n_neigh.index()]
                    } else {
                        nodes[1]
                    };
                    if m_neigh == end {
                        continue;
                    }
                    let val =
                        edge_multiplicity(&st.0.graph, &st.0.adjacency_matrix, nodes[0], n_neigh);

                    let has_edge =
                        is_adjacent(&st.1.graph, &st.1.adjacency_matrix, nodes[1], m_neigh, val);
                    if !has_edge {
                        return Ok(false);
                    }
                }

                for n_neigh in st.1.graph.neighbors(nodes[1]) {
                    succ_count[1] += 1;
                    // handle the self loop case; it's not in the mapping (yet)
                    let m_neigh = if nodes[1] != n_neigh {
                        st.1.mapping[n_neigh.index()]
                    } else {
                        nodes[0]
                    };
                    if m_neigh == end {
                        continue;
                    }
                    let val =
                        edge_multiplicity(&st.1.graph, &st.1.adjacency_matrix, nodes[1], n_neigh);

                    let has_edge =
                        is_adjacent(&st.0.graph, &st.0.adjacency_matrix, nodes[0], m_neigh, val);
                    if !has_edge {
                        return Ok(false);
                    }
                }
                if succ_count[0].cmp(&succ_count[1]).then(ordering) != ordering {
                    return Ok(false);
                }
                // R_pred
                if st.0.graph.is_directed() {
                    let mut pred_count = [0, 0];
                    for n_neigh in st.0.graph.neighbors_directed(nodes[0], Incoming) {
                        pred_count[0] += 1;
                        if !induced {
                            continue;
                        }
                        // the self loop case is handled in outgoing
                        let m_neigh = st.0.mapping[n_neigh.index()];
                        if m_neigh == end {
                            continue;
                        }
                        let val = edge_multiplicity(
                            &st.0.graph,
                            &st.0.adjacency_matrix,
                            n_neigh,
                            nodes[0],
                        );

                        let has_edge = is_adjacent(
                            &st.1.graph,
                            &st.1.adjacency_matrix,
                            m_neigh,
                            nodes[1],
                            val,
                        );
                        if !has_edge {
                            return Ok(false);
                        }
                    }

                    for n_neigh in st.1.graph.neighbors_directed(nodes[1], Incoming) {
                        pred_count[1] += 1;
                        // the self loop case is handled in outgoing
                        let m_neigh = st.1.mapping[n_neigh.index()];
                        if m_neigh == end {
                            continue;
                        }
                        let val = edge_multiplicity(
                            &st.1.graph,
                            &st.1.adjacency_matrix,
                            n_neigh,
                            nodes[1],
                        );

                        let has_edge = is_adjacent(
                            &st.0.graph,
                            &st.0.adjacency_matrix,
                            m_neigh,
                            nodes[0],
                            val,
                        );
                        if !has_edge {
                            return Ok(false);
                        }
                    }
                    if pred_count[0].cmp(&pred_count[1]).then(ordering) != ordering {
                        return Ok(false);
                    }
                }
                macro_rules! field {
                    ($x:ident,     0) => {
                        $x.0
                    };
                    ($x:ident,     1) => {
                        $x.1
                    };
                    ($x:ident, 1 - 0) => {
                        $x.1
                    };
                    ($x:ident, 1 - 1) => {
                        $x.0
                    };
                }
                macro_rules! rule {
                    ($arr:ident, $j:tt, $dir:expr) => {{
                        let mut count = 0;
                        for n_neigh in field!(st, $j).graph.neighbors_directed(nodes[$j], $dir) {
                            let index = n_neigh.index();
                            if field!(st, $j).$arr[index] > 0 && st.$j.mapping[index] == end {
                                count += 1;
                            }
                        }
                        count
                    }};
                }
                // R_out
                if rule!(out, 0, Outgoing)
                    .cmp(&rule!(out, 1, Outgoing))
                    .then(ordering)
                    != ordering
                {
                    return Ok(false);
                }
                if st.0.graph.is_directed()
                    && rule!(out, 0, Incoming)
                        .cmp(&rule!(out, 1, Incoming))
                        .then(ordering)
                        != ordering
                {
                    return Ok(false);
                }
                // R_in
                if st.0.graph.is_directed() {
                    if rule!(ins, 0, Outgoing)
                        .cmp(&rule!(ins, 1, Outgoing))
                        .then(ordering)
                        != ordering
                    {
                        return Ok(false);
                    }

                    if rule!(ins, 0, Incoming)
                        .cmp(&rule!(ins, 1, Incoming))
                        .then(ordering)
                        != ordering
                    {
                        return Ok(false);
                    }
                }
                // R_new
                if induced {
                    let mut new_count = [0, 0];
                    for n_neigh in st.0.graph.neighbors(nodes[0]) {
                        let index = n_neigh.index();
                        if st.0.out[index] == 0 && (st.0.ins.is_empty() || st.0.ins[index] == 0) {
                            new_count[0] += 1;
                        }
                    }
                    for n_neigh in st.1.graph.neighbors(nodes[1]) {
                        let index = n_neigh.index();
                        if st.1.out[index] == 0 && (st.1.ins.is_empty() || st.1.ins[index] == 0) {
                            new_count[1] += 1;
                        }
                    }
                    if new_count[0].cmp(&new_count[1]).then(ordering) != ordering {
                        return Ok(false);
                    }
                    if st.0.graph.is_directed() {
                        let mut new_count = [0, 0];
                        for n_neigh in st.0.graph.neighbors_directed(nodes[0], Incoming) {
                            let index = n_neigh.index();
                            if st.0.out[index] == 0 && st.0.ins[index] == 0 {
                                new_count[0] += 1;
                            }
                        }
                        for n_neigh in st.1.graph.neighbors_directed(nodes[1], Incoming) {
                            let index = n_neigh.index();
                            if st.1.out[index] == 0 && st.1.ins[index] == 0 {
                                new_count[1] += 1;
                            }
                        }
                        if new_count[0].cmp(&new_count[1]).then(ordering) != ordering {
                            return Ok(false);
                        }
                    }
                }
                // semantic feasibility: compare associated data for nodes
                if node_match.enabled()
                    && !node_match
                        .eq(&st.0.graph, &st.1.graph, nodes[0], nodes[1])
                        .map_err(IsIsomorphicError::NodeMatcherErr)?
                {
                    return Ok(false);
                }
                // semantic feasibility: compare associated data for edges
                if edge_match.enabled() {
                    let mut matcher =
                        |g0_edge: (NodeIndex, G0::EdgeId),
                         g1_edge: (NodeIndex, G1::EdgeId)|
                         -> Result<bool, IsIsomorphicError<NM::Error, EM::Error>> {
                            let (nx, e0) = g0_edge;
                            let (mx, e1) = g1_edge;
                            if nx == mx
                                && edge_match
                                .eq(&st.0.graph, &st.1.graph, e0, e1)
                                .map_err(IsIsomorphicError::EdgeMatcherErr)?
                            {
                                return Ok(true);
                            }
                            Ok(false)
                        };

                    // Used to reverse the order of edge args to the matcher
                    // when checking G1 subset of G0.
                    #[inline]
                    fn reverse_args<T1, T2, R, F>(mut f: F) -> impl FnMut(T2, T1) -> R
                    where
                        F: FnMut(T1, T2) -> R,
                    {
                        move |y, x| f(x, y)
                    }

                    // outgoing edges
                    if induced {
                        let e_first: Vec<(NodeIndex, G0::EdgeId)> =
                            st.0.graph
                                .edges(nodes[0])
                                .filter_map(|edge| {
                                    let n_neigh = edge.target();
                                    let m_neigh = if nodes[0] != n_neigh {
                                        st.0.mapping[n_neigh.index()]
                                    } else {
                                        nodes[1]
                                    };
                                    if m_neigh == end {
                                        return None;
                                    }
                                    Some((m_neigh, edge.id()))
                                })
                                .collect();

                        let e_second: Vec<(NodeIndex, G1::EdgeId)> =
                            st.1.graph
                                .edges(nodes[1])
                                .map(|edge| (edge.target(), edge.id()))
                                .collect();

                        if !is_subset(&e_first, &e_second, &mut matcher)? {
                            return Ok(false);
                        };
                    }

                    let e_first: Vec<(NodeIndex, G1::EdgeId)> =
                        st.1.graph
                            .edges(nodes[1])
                            .filter_map(|edge| {
                                let n_neigh = edge.target();
                                let m_neigh = if nodes[1] != n_neigh {
                                    st.1.mapping[n_neigh.index()]
                                } else {
                                    nodes[0]
                                };
                                if m_neigh == end {
                                    return None;
                                }
                                Some((m_neigh, edge.id()))
                            })
                            .collect();

                    let e_second: Vec<(NodeIndex, G0::EdgeId)> =
                        st.0.graph
                            .edges(nodes[0])
                            .map(|edge| (edge.target(), edge.id()))
                            .collect();

                    if !is_subset(&e_first, &e_second, &mut reverse_args(&mut matcher))? {
                        return Ok(false);
                    };

                    // incoming edges
                    if st.0.graph.is_directed() {
                        if induced {
                            let e_first: Vec<(NodeIndex, G0::EdgeId)> =
                                st.0.graph
                                    .edges_directed(nodes[0], Incoming)
                                    .filter_map(|edge| {
                                        let n_neigh = edge.source();
                                        let m_neigh = if nodes[0] != n_neigh {
                                            st.0.mapping[n_neigh.index()]
                                        } else {
                                            nodes[1]
                                        };
                                        if m_neigh == end {
                                            return None;
                                        }
                                        Some((m_neigh, edge.id()))
                                    })
                                    .collect();

                            let e_second: Vec<(NodeIndex, G1::EdgeId)> =
                                st.1.graph
                                    .edges_directed(nodes[1], Incoming)
                                    .map(|edge| (edge.source(), edge.id()))
                                    .collect();

                            if !is_subset(&e_first, &e_second, &mut matcher)? {
                                return Ok(false);
                            };
                        }

                        let e_first: Vec<(NodeIndex, G1::EdgeId)> =
                            st.1.graph
                                .edges_directed(nodes[1], Incoming)
                                .filter_map(|edge| {
                                    let n_neigh = edge.source();
                                    let m_neigh = if nodes[1] != n_neigh {
                                        st.1.mapping[n_neigh.index()]
                                    } else {
                                        nodes[0]
                                    };
                                    if m_neigh == end {
                                        return None;
                                    }
                                    Some((m_neigh, edge.id()))
                                })
                                .collect();

                        let e_second: Vec<(NodeIndex, G0::EdgeId)> =
                            st.0.graph
                                .edges_directed(nodes[0], Incoming)
                                .map(|edge| (edge.source(), edge.id()))
                                .collect();

                        if !is_subset(&e_first, &e_second, &mut reverse_args(&mut matcher))? {
                            return Ok(false);
                        };
                    }
                }
                Ok(true)
            }
        }

        impl<G0, G1, NM, EM> Iterator for Vf2Algorithm<G0, G1, NM, EM>
        where
            G0: GraphProp
                + GraphBase<NodeId = NodeIndex>
                + DataMap
                + Create
                + NodeCount
                + EdgeCount,
            for<'a> &'a G0: GraphBase<NodeId = G0::NodeId, EdgeId = G0::EdgeId>
                + Data<NodeWeight = G0::NodeWeight, EdgeWeight = G0::EdgeWeight>
                + NodeIndexable
                + IntoEdgesDirected
                + IntoNodeIdentifiers,
            G0::NodeWeight: Clone,
            G0::EdgeWeight: Clone,
            G1: GraphProp
                + GraphBase<NodeId = NodeIndex>
                + DataMap
                + Create
                + NodeCount
                + EdgeCount,
            for<'a> &'a G1: GraphBase<NodeId = G1::NodeId, EdgeId = G1::EdgeId>
                + Data<NodeWeight = G1::NodeWeight, EdgeWeight = G1::EdgeWeight>
                + NodeIndexable
                + IntoEdgesDirected
                + IntoNodeIdentifiers,
            G1::NodeWeight: Clone,
            G1::EdgeWeight: Clone,
            NM: NodeMatcher<G0, G1>,
            EM: EdgeMatcher<G0, G1>,
        {
            type Item = Result<DictMap<usize, usize>, IsIsomorphicError<NM::Error, EM::Error>>;

            /// Return Some(mapping) if isomorphism is decided, else None.
            fn next(&mut self) -> Option<Self::Item> {
                if (self
                    .st
                    .0
                    .graph
                    .node_count()
                    .cmp(&self.st.1.graph.node_count())
                    .then(self.ordering)
                    != self.ordering)
                    || (self
                        .st
                        .0
                        .graph
                        .edge_count()
                        .cmp(&self.st.1.graph.edge_count())
                        .then(self.ordering)
                        != self.ordering)
                {
                    return None;
                }

                // A "depth first" search of a valid mapping from graph 1 to graph 2

                // F(s, n, m) -- evaluate state s and add mapping n <-> m

                // Find least T1out node (in st.out[1] but not in M[1])
                while let Some(frame) = self.stack.pop() {
                    match frame {
                        Frame::Unwind {
                            nodes,
                            open_list: ol,
                        } => {
                            Vf2Algorithm::<G0, G1, NM, EM>::pop_state(&mut self.st, nodes);

                            match Vf2Algorithm::<G0, G1, NM, EM>::next_from_ix(
                                &mut self.st,
                                nodes[0],
                                ol,
                            ) {
                                None => continue,
                                Some(nx) => {
                                    let f = Frame::Inner {
                                        nodes: [nx, nodes[1]],
                                        open_list: ol,
                                    };
                                    self.stack.push(f);
                                }
                            }
                        }
                        Frame::Outer => {
                            match Vf2Algorithm::<G0, G1, NM, EM>::next_candidate(&mut self.st) {
                                None => {
                                    if self.st.1.is_complete() {
                                        return Some(Ok(self.mapping()));
                                    }
                                    continue;
                                }
                                Some((nx, mx, ol)) => {
                                    let f = Frame::Inner {
                                        nodes: [nx, mx],
                                        open_list: ol,
                                    };
                                    self.stack.push(f);
                                }
                            }
                        }
                        Frame::Inner {
                            nodes,
                            open_list: ol,
                        } => {
                            let feasible = match Vf2Algorithm::<G0, G1, NM, EM>::is_feasible(
                                &mut self.st,
                                nodes,
                                &mut self.node_match,
                                &mut self.edge_match,
                                self.ordering,
                                self.induced,
                            ) {
                                Ok(f) => f,
                                Err(e) => {
                                    return Some(Err(e));
                                }
                            };

                            if feasible {
                                Vf2Algorithm::<G0, G1, NM, EM>::push_state(&mut self.st, nodes);
                                // Check cardinalities of Tin, Tout sets
                                if self
                                    .st
                                    .0
                                    .out_size
                                    .cmp(&self.st.1.out_size)
                                    .then(self.ordering)
                                    == self.ordering
                                    && self
                                        .st
                                        .0
                                        .ins_size
                                        .cmp(&self.st.1.ins_size)
                                        .then(self.ordering)
                                        == self.ordering
                                {
                                    self._counter += 1;
                                    if let Some(limit) = self.call_limit {
                                        if self._counter > limit {
                                            return None;
                                        }
                                    }
                                    let f0 = Frame::Unwind {
                                        nodes,
                                        open_list: ol,
                                    };

                                    self.stack.push(f0);
                                    self.stack.push(Frame::Outer);
                                    continue;
                                }
                                Vf2Algorithm::<G0, G1, NM, EM>::pop_state(&mut self.st, nodes);
                            }
                            match Vf2Algorithm::<G0, G1, NM, EM>::next_from_ix(
                                &mut self.st,
                                nodes[0],
                                ol,
                            ) {
                                None => continue,
                                Some(nx) => {
                                    let f = Frame::Inner {
                                        nodes: [nx, nodes[1]],
                                        open_list: ol,
                                    };
                                    self.stack.push(f);
                                }
                            }
                        }
                    }
                }
                None
            }
        }
    }
}

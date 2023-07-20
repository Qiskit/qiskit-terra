# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Integer programming model for quantum circuit compilation."""
import copy
import logging
from functools import lru_cache

import numpy as np

from qiskit.transpiler.exceptions import TranspilerError, CouplingError
from qiskit.transpiler.layout import Layout
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.providers.models import BackendProperties
from qiskit.quantum_info import two_qubit_cnot_decompose
from qiskit.quantum_info.synthesis.two_qubit_decompose import (
    TwoQubitWeylDecomposition,
    trace_to_fid,
)
from qiskit.utils import optionals as _optionals
from qiskit.utils.deprecation import deprecate_func

logger = logging.getLogger(__name__)


@_optionals.HAS_DOCPLEX.require_in_instance
class BIPMappingModel:
    """Internal model to create and solve a BIP problem for mapping.

    Attributes:
        problem (Model):
            A CPLEX problem model object, which is set by calling
            :method:`create_cpx_problem`. After calling :method:`solve_cpx_problem`,
            the solution will be stored in :attr:`solution`). None if it's not yet set.
    """

    @deprecate_func(
        since="0.24.0",
        additional_msg="This has been replaced by a new transpiler plugin package: "
        "qiskit-bip-mapper. More details can be found here: "
        "https://github.com/qiskit-community/qiskit-bip-mapper",
    )  # pylint: disable=bad-docstring-quotes
    def __init__(self, dag, coupling_map, qubit_subset, dummy_timesteps=None):
        """
        Args:
            dag (DAGCircuit): DAG circuit to be mapped
            coupling_map (CouplingMap): Coupling map of the device on which the `dag` is mapped.
            qubit_subset (list[int]): Sublist of physical qubits to be used in the mapping.
            dummy_timesteps (int):
                Number of dummy time steps, after each real layer of gates, to
                allow arbitrary swaps between neighbors.

        Raises:
            MissingOptionalLibraryError: If docplex is not installed
            TranspilerError: If size of virtual qubits and physical qubits differ, or
                if coupling_map is not symmetric (bidirectional).
        """

        self._dag = dag
        self._coupling = copy.deepcopy(coupling_map)  # reduced coupling map
        try:
            self._coupling = self._coupling.reduce(qubit_subset)
        except CouplingError as err:
            raise TranspilerError(
                "The 'coupling_map' reduced by 'qubit_subset' must be connected."
            ) from err
        self._coupling.make_symmetric()
        self.global_qubit = qubit_subset  # the map from reduced qubit index to global qubit index

        self.problem = None
        self.solution = None
        self.num_vqubits = len(self._dag.qubits)
        self.num_pqubits = self._coupling.size()
        self._arcs = self._coupling.get_edges()

        if self.num_vqubits != self.num_pqubits:
            raise TranspilerError(
                "BIPMappingModel assumes the same size of virtual and physical qubits."
            )

        self._index_to_virtual = dict(enumerate(dag.qubits))
        self._virtual_to_index = {v: i for i, v in self._index_to_virtual.items()}

        # Construct internal circuit model
        # Extract layers with 2-qubit gates
        self._to_su4layer = []
        self.su4layers = []
        for lay in dag.layers():
            laygates = []
            for node in lay["graph"].two_qubit_ops():
                i1 = self._virtual_to_index[node.qargs[0]]
                i2 = self._virtual_to_index[node.qargs[1]]
                laygates.append(((i1, i2), node))
            if laygates:
                self._to_su4layer.append(len(self.su4layers))
                self.su4layers.append(laygates)
            else:
                self._to_su4layer.append(-1)
        # Add dummy time steps inbetween su4layers. Dummy time steps can only contain SWAPs.
        self.gates = []  # layered 2q-gates with dummy steps
        for k, lay in enumerate(self.su4layers):
            self.gates.append(lay)
            if k == len(self.su4layers) - 1:  # do not add dummy steps after the last layer
                break
            self.gates.extend([[]] * dummy_timesteps)

        self.bprop = None  # Backend properties to compute cx fidelities (set later if necessary)
        self.default_cx_error_rate = (
            None  # Default cx error rate in case backend properties are not available
        )

        logger.info("Num virtual qubits: %d", self.num_vqubits)
        logger.info("Num physical qubits: %d", self.num_pqubits)
        logger.info("Model depth: %d", self.depth)
        logger.info("Dummy steps: %d", dummy_timesteps)

    @property
    def depth(self):
        """Number of time-steps (including dummy steps)."""
        return len(self.gates)

    def is_su4layer(self, depth: int) -> bool:
        """Check if the depth-th layer is su4layer (layer containing 2q-gates) or not.

        Args:
            depth: Depth of the ordinary layer

        Returns:
            True if the depth-th layer is su4layer, otherwise False
        """
        return self._to_su4layer[depth] >= 0

    def to_su4layer_depth(self, depth: int) -> int:
        """Return the depth as a su4layer. If the depth-th layer is not a su4layer, return -1.

        Args:
            depth: Depth of the ordinary layer

        Returns:
            su4layer depth if the depth-th layer is a su4layer, otherwise -1
        """
        return self._to_su4layer[depth]

    # pylint: disable=invalid-name
    def _is_dummy_step(self, t: int):
        """Check if the time-step t is a dummy step or not."""
        return len(self.gates[t]) == 0

    @_optionals.HAS_DOCPLEX.require_in_call
    def create_cpx_problem(
        self,
        objective: str,
        backend_prop: BackendProperties = None,
        line_symm: bool = False,
        depth_obj_weight: float = 0.1,
        default_cx_error_rate: float = 5e-3,
    ):
        """Create integer programming model to compile a circuit.

        Args:
            objective:
                Type of objective function to be minimized:

                * ``'gate_error'``: Approximate gate error of the circuit, which is given as the sum of
                    negative logarithm of CNOT gate fidelities in the circuit. It takes into account
                    only the CNOT gate errors reported in ``backend_prop``.
                * ``'depth'``: Depth (number of timesteps) of the circuit
                * ``'balanced'``: Weighted sum of gate_error and depth

            backend_prop:
                Backend properties storing gate errors, which are required in computing certain
                types of objective function such as ``'gate_error'`` or ``'balanced'``.
                If this is not available, default_cx_error_rate is used instead.

            line_symm:
                Use symmetry breaking constrainst for line topology. Should
                only be True if the hardware graph is a chain/line/path.

            depth_obj_weight:
                Weight of depth objective in ``'balanced'`` objective function.

            default_cx_error_rate:
                Default CX error rate to be used if backend_prop is not available.

        Raises:
            TranspilerError: if unknown objective type is specified or invalid options are specified.
            MissingOptionalLibraryError: If docplex is not installed
        """
        self.bprop = backend_prop
        self.default_cx_error_rate = default_cx_error_rate
        if self.bprop is None and self.default_cx_error_rate is None:
            raise TranspilerError("BackendProperties or default_cx_error_rate must be specified")
        from docplex.mp.model import Model

        mdl = Model()

        # *** Define main variables ***
        # Add w variables
        w = {}
        for t in range(self.depth):
            for q in range(self.num_vqubits):
                for j in range(self.num_pqubits):
                    w[t, q, j] = mdl.binary_var(name=f"w_{t}_{q}_{j}")
        # Add y variables
        y = {}
        for t in range(self.depth):
            for ((p, q), _) in self.gates[t]:
                for (i, j) in self._arcs:
                    y[t, p, q, i, j] = mdl.binary_var(name=f"y_{t}_{p}_{q}_{i}_{j}")
        # Add x variables
        x = {}
        for t in range(self.depth - 1):
            for q in range(self.num_vqubits):
                for i in range(self.num_pqubits):
                    x[t, q, i, i] = mdl.binary_var(name=f"x_{t}_{q}_{i}_{i}")
                    for j in self._coupling.neighbors(i):
                        x[t, q, i, j] = mdl.binary_var(name=f"x_{t}_{q}_{i}_{j}")

        # *** Define main constraints ***
        # Assignment constraints for w variables
        for t in range(self.depth):
            for q in range(self.num_vqubits):
                mdl.add_constraint(
                    sum(w[t, q, j] for j in range(self.num_pqubits)) == 1,
                    ctname=f"assignment_vqubits_{q}_at_{t}",
                )
        for t in range(self.depth):
            for j in range(self.num_pqubits):
                mdl.add_constraint(
                    sum(w[t, q, j] for q in range(self.num_vqubits)) == 1,
                    ctname=f"assignment_pqubits_{j}_at_{t}",
                )
        # Each gate must be implemented
        for t in range(self.depth):
            for ((p, q), _) in self.gates[t]:
                mdl.add_constraint(
                    sum(y[t, p, q, i, j] for (i, j) in self._arcs) == 1,
                    ctname=f"implement_gate_{p}_{q}_at_{t}",
                )
        # Gate can be implemented iff both of its qubits are located at the associated nodes
        for t in range(self.depth - 1):
            for ((p, q), _) in self.gates[t]:
                for (i, j) in self._arcs:
                    # Apply McCormick to y[t, p, q, i, j] == w[t, p, i] * w[t, q, j]
                    mdl.add_constraint(
                        y[t, p, q, i, j] >= w[t, p, i] + w[t, q, j] - 1,
                        ctname=f"McCormickLB_{p}_{q}_{i}_{j}_at_{t}",
                    )
                    # Stronger version of McCormick: gate (p,q) is implemented at (i, j)
                    # if i moves to i or j, and j moves to i or j
                    mdl.add_constraint(
                        y[t, p, q, i, j] <= x[t, p, i, i] + x[t, p, i, j],
                        ctname=f"McCormickUB1_{p}_{q}_{i}_{j}_at_{t}",
                    )
                    mdl.add_constraint(
                        y[t, p, q, i, j] <= x[t, q, j, i] + x[t, q, j, j],
                        ctname=f"McCormickUB2_{p}_{q}_{i}_{j}_at_{t}",
                    )
        # For last time step, use regular McCormick
        for ((p, q), _) in self.gates[self.depth - 1]:
            for (i, j) in self._arcs:
                # Apply McCormick to y[self.depth - 1, p, q, i, j]
                # == w[self.depth - 1, p, i] * w[self.depth - 1, q, j]
                mdl.add_constraint(
                    y[self.depth - 1, p, q, i, j]
                    >= w[self.depth - 1, p, i] + w[self.depth - 1, q, j] - 1,
                    ctname=f"McCormickLB_{p}_{q}_{i}_{j}_at_last",
                )
                mdl.add_constraint(
                    y[self.depth - 1, p, q, i, j] <= w[self.depth - 1, p, i],
                    ctname=f"McCormickUB1_{p}_{q}_{i}_{j}_at_last",
                )
                mdl.add_constraint(
                    y[self.depth - 1, p, q, i, j] <= w[self.depth - 1, q, j],
                    ctname=f"McCormickUB2_{p}_{q}_{i}_{j}_at_last",
                )
        # Logical qubit flow-out constraints
        for t in range(self.depth - 1):  # Flow out; skip last time step
            for q in range(self.num_vqubits):
                for i in range(self.num_pqubits):
                    mdl.add_constraint(
                        w[t, q, i]
                        == x[t, q, i, i] + sum(x[t, q, i, j] for j in self._coupling.neighbors(i)),
                        ctname=f"flow_out_{q}_{i}_at_{t}",
                    )
        # Logical qubit flow-in constraints
        for t in range(1, self.depth):  # Flow in; skip first time step
            for q in range(self.num_vqubits):
                for i in range(self.num_pqubits):
                    mdl.add_constraint(
                        w[t, q, i]
                        == x[t - 1, q, i, i]
                        + sum(x[t - 1, q, j, i] for j in self._coupling.neighbors(i)),
                        ctname=f"flow_in_{q}_{i}_at_{t}",
                    )
        # If a gate is implemented, involved qubits cannot swap with other positions
        for t in range(self.depth - 1):
            for ((p, q), _) in self.gates[t]:
                for (i, j) in self._arcs:
                    mdl.add_constraint(
                        x[t, p, i, j] == x[t, q, j, i], ctname=f"swap_{p}_{q}_{i}_{j}_at_{t}"
                    )
        # Qubit not in gates can flip with their neighbors
        for t in range(self.depth - 1):
            q_no_gate = list(range(self.num_vqubits))
            for ((p, q), _) in self.gates[t]:
                q_no_gate.remove(p)
                q_no_gate.remove(q)
            for (i, j) in self._arcs:
                mdl.add_constraint(
                    sum(x[t, q, i, j] for q in q_no_gate) == sum(x[t, p, j, i] for p in q_no_gate),
                    ctname=f"swap_no_gate_{i}_{j}_at_{t}",
                )

        # *** Define supplemental variables ***
        # Add z variables to count dummy steps (supplemental variables for symmetry breaking)
        z = {}
        for t in range(self.depth):
            if self._is_dummy_step(t):
                z[t] = mdl.binary_var(name=f"z_{t}")

        # *** Define supplemental constraints ***
        # See if a dummy time step is needed
        for t in range(self.depth):
            if self._is_dummy_step(t):
                for q in range(self.num_vqubits):
                    mdl.add_constraint(
                        sum(x[t, q, i, j] for (i, j) in self._arcs) <= z[t],
                        ctname=f"dummy_ts_needed_for_vqubit_{q}_at_{t}",
                    )
        # Symmetry breaking between dummy time steps
        for t in range(self.depth - 1):
            # This is a dummy time step and the next one is dummy too
            if self._is_dummy_step(t) and self._is_dummy_step(t + 1):
                # We cannot use the next time step unless this one is used too
                mdl.add_constraint(z[t] >= z[t + 1], ctname=f"dummy_precedence_{t}")
        # Symmetry breaking on the line -- only works on line topology!
        if line_symm:
            for h in range(1, self.num_vqubits):
                mdl.add_constraint(
                    sum(w[0, p, 0] for p in range(h))
                    + sum(w[0, q, self.num_pqubits - 1] for q in range(h, self.num_vqubits))
                    >= 1,
                    ctname=f"sym_break_line_{h}",
                )

        # *** Define objevtive function ***
        if objective == "depth":
            objexr = sum(z[t] for t in range(self.depth) if self._is_dummy_step(t))
            for t in range(self.depth - 1):
                for q in range(self.num_vqubits):
                    for (i, j) in self._arcs:
                        objexr += 0.01 * x[t, q, i, j]
            mdl.minimize(objexr)
        elif objective in ("gate_error", "balanced"):
            # We add the depth objective with coefficient depth_obj_weight if balanced was selected.
            objexr = 0
            for t in range(self.depth - 1):
                for (p, q), node in self.gates[t]:
                    for (i, j) in self._arcs:
                        # We pay the cost for gate implementation.
                        pbest_fid = -np.log(self._max_expected_fidelity(node, i, j))
                        objexr += y[t, p, q, i, j] * pbest_fid
                        # If a gate is mirrored (followed by a swap on the same qubit pair),
                        # its cost should be replaced with the cost of the combined (mirrored) gate.
                        pbest_fidm = -np.log(self._max_expected_mirrored_fidelity(node, i, j))
                        objexr += x[t, q, i, j] * (pbest_fidm - pbest_fid) / 2
                # Cost of swaps on unused qubits
                for q in range(self.num_vqubits):
                    used_qubits = {q for (pair, _) in self.gates[t] for q in pair}
                    if q not in used_qubits:
                        for i in range(self.num_pqubits):
                            for j in self._coupling.neighbors(i):
                                objexr += x[t, q, i, j] * -3 / 2 * np.log(self._cx_fidelity(i, j))
            # Cost for the last layer (x variables are not defined for depth-1)
            for (p, q), node in self.gates[self.depth - 1]:
                for (i, j) in self._arcs:
                    pbest_fid = -np.log(self._max_expected_fidelity(node, i, j))
                    objexr += y[self.depth - 1, p, q, i, j] * pbest_fid
            if objective == "balanced":
                objexr += depth_obj_weight * sum(
                    z[t] for t in range(self.depth) if self._is_dummy_step(t)
                )
            mdl.minimize(objexr)
        else:
            raise TranspilerError(f"Unknown objective type: {objective}")

        self.problem = mdl
        logger.info("BIP problem stats: %s", self.problem.statistics)

    def _max_expected_fidelity(self, node, i, j):
        return max(
            gfid * self._cx_fidelity(i, j) ** k
            for k, gfid in enumerate(self._gate_fidelities(node))
        )

    def _max_expected_mirrored_fidelity(self, node, i, j):
        return max(
            gfid * self._cx_fidelity(i, j) ** k
            for k, gfid in enumerate(self._mirrored_gate_fidelities(node))
        )

    def _cx_fidelity(self, i, j) -> float:
        # fidelity of cx on global physical qubits
        if self.bprop is not None:
            return 1.0 - self.bprop.gate_error("cx", [self.global_qubit[i], self.global_qubit[j]])
        else:
            return 1.0 - self.default_cx_error_rate

    @staticmethod
    @lru_cache
    def _gate_fidelities(node):
        matrix = node.op.to_matrix()
        target = TwoQubitWeylDecomposition(matrix)
        traces = two_qubit_cnot_decompose.traces(target)
        return [trace_to_fid(traces[i]) for i in range(4)]

    @staticmethod
    @lru_cache
    def _mirrored_gate_fidelities(node):
        matrix = node.op.to_matrix()
        swap = SwapGate().to_matrix()
        targetm = TwoQubitWeylDecomposition(matrix @ swap)
        tracesm = two_qubit_cnot_decompose.traces(targetm)
        return [trace_to_fid(tracesm[i]) for i in range(4)]

    @_optionals.HAS_CPLEX.require_in_call
    def solve_cpx_problem(self, time_limit: float = 60, threads: int = None) -> str:
        """Solve the BIP problem using CPLEX.

        Args:
            time_limit:
                Time limit (seconds) given to CPLEX.

            threads:
                Number of threads to be allowed for CPLEX to use.

        Returns:
            Status string that CPLEX returned after solving the BIP problem.

        Raises:
            MissingOptionalLibraryError: If CPLEX is not installed
        """
        self.problem.set_time_limit(time_limit)
        if threads is not None:
            self.problem.context.cplex_parameters.threads = threads
        self.problem.context.cplex_parameters.randomseed = 777

        self.solution = self.problem.solve()

        status = self.problem.solve_details.status
        logger.info("BIP solution status: %s", status)
        return status

    def get_layout(self, t: int) -> Layout:
        """Get layout at time-step t.

        Args:
            t: Time-step

        Returns:
            Layout
        """
        dic = {}
        for q in range(self.num_vqubits):
            for i in range(self.num_pqubits):
                if self.solution.get_value(f"w_{t}_{q}_{i}") > 0.5:
                    dic[self._index_to_virtual[q]] = self.global_qubit[i]
        layout = Layout(dic)
        for reg in self._dag.qregs.values():
            layout.add_register(reg)
        return layout

    def get_swaps(self, t: int) -> list:
        """Get swaps (pairs of physical qubits) inserted at time-step ``t``.

        Args:
            t: Time-step (<= depth - 1)

        Returns:
            List of swaps (pairs of physical qubits (integers))
        """
        swaps = []
        for (i, j) in self._arcs:
            if i >= j:
                continue
            for q in range(self.num_vqubits):
                if self.solution.get_value(f"x_{t}_{q}_{i}_{j}") > 0.5:
                    swaps.append((self.global_qubit[i], self.global_qubit[j]))
        return swaps

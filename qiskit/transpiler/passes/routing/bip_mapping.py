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

"""Map a DAGCircuit onto a given ``coupling_map``, allocating qubits and adding swap gates."""
import copy
import logging
import math

from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.utils import optionals as _optionals
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.routing.algorithms.bip_model import BIPMappingModel
from qiskit.transpiler.target import target_to_backend_properties, Target

logger = logging.getLogger(__name__)


@_optionals.HAS_CPLEX.require_in_instance("BIP-based mapping pass")
@_optionals.HAS_DOCPLEX.require_in_instance("BIP-based mapping pass")
class BIPMapping(TransformationPass):
    r"""Map a DAGCircuit onto a given ``coupling_map``, allocating qubits and adding swap gates.

    The BIP mapper tries to find the best layout and routing at once by
    solving a BIP (binary integer programming) problem as described in [1].

    The BIP problem represents the layer-by-layer mapping of 2-qubit gates, assuming all the gates
    in a layer can be run on the ``coupling_map``. In the problem, the variables :math:`w` represent
    the layout of qubits for each layer and the variables :math:`x` represent which pair of qubits
    should be swapped in between layers. Based on the values in the solution of the BIP problem,
    the mapped circuit will be constructed.

    The BIP mapper depends on ``docplex`` to represent the BIP problem and CPLEX (``cplex``)
    to solve it. Those packages can be installed with ``pip install qiskit-terra[bip-mapper]``.
    Since the free version of CPLEX can solve only small BIP problems, i.e. mapping of circuits
    with less than about 5 qubits, the paid version of CPLEX may be needed to map larger circuits.

    If you want to fix physical qubits to be used in the mapping (e.g. running Quantum Volume
    circuits), you need to supply ``qubit_subset``, i.e. list of physical qubits to be used
    within the ``coupling_map``.
    Please do not use ``initial_layout`` for that purpose because the BIP mapper gracefully
    ignores ``initial_layout`` (and tries to determines its best layout).

    .. warning::
        The BIP mapper does not scale very well with respect to the number of qubits or gates.
        For example, it may not work with ``qubit_subset`` beyond 10 qubits because
        the BIP solver (CPLEX) may not find any solution within the default time limit.

    **References:**

    [1] G. Nannicini et al. "Optimal qubit assignment and routing via integer programming."
    `arXiv:2106.06446 <https://arxiv.org/abs/2106.06446>`_
    """

    def __init__(
        self,
        coupling_map,
        qubit_subset=None,
        objective="balanced",
        backend_prop=None,
        time_limit=30,
        threads=None,
        max_swaps_inbetween_layers=None,
        depth_obj_weight=0.1,
        default_cx_error_rate=5e-3,
    ):
        """BIPMapping initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): Directed graph represented a coupling map.
            qubit_subset (list[int]): Sublist of physical qubits to be used in the mapping.
                If None, all qubits in the coupling_map will be considered.
            objective (str): Type of objective function to be minimized:

                * ``'gate_error'``: Approximate gate error of the circuit, which is given as the sum of
                    negative logarithm of 2q-gate fidelities in the circuit. It takes into account only
                    the 2q-gate (CNOT) errors reported in ``backend_prop`` and ignores the other errors
                    in such as 1q-gates, SPAMs and idle times.
                * ``'depth'``: Depth (number of 2q-gate layers) of the circuit.
                * ``'balanced'``: [Default] Weighted sum of ``'gate_error'`` and ``'depth'``

            backend_prop (BackendProperties): Backend properties object containing 2q-gate gate errors,
                which are required in computing certain types of objective function
                such as ``'gate_error'`` or ``'balanced'``. If this is not available,
                default_cx_error_rate is used instead.
            time_limit (float): Time limit for solving BIP in seconds
            threads (int): Number of threads to be allowed for CPLEX to solve BIP
            max_swaps_inbetween_layers (int):
                Number of swaps allowed in between layers. If None, automatically set.
                Large value could decrease the probability to build infeasible BIP problem but also
                could reduce the chance of finding a feasible solution within the ``time_limit``.

            depth_obj_weight (float):
                Weight of depth objective in ``'balanced'`` objective. The balanced objective is the
                sum of error_rate + depth_obj_weight * depth.

            default_cx_error_rate (float):
                Default CX error rate to be used if backend_prop is not available.

        Raises:
            MissingOptionalLibraryError: if cplex or docplex are not installed.
            TranspilerError: if invalid options are specified.
        """
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
            self.backend_prop = target_to_backend_properties(self.target)
        else:
            self.target = None
            self.coupling_map = coupling_map
            self.backend_prop = None
        self.qubit_subset = qubit_subset
        self.objective = objective
        if backend_prop is not None:
            self.backend_prop = backend_prop
        self.time_limit = time_limit
        self.threads = threads
        self.max_swaps_inbetween_layers = max_swaps_inbetween_layers
        self.depth_obj_weight = depth_obj_weight
        self.default_cx_error_rate = default_cx_error_rate
        if self.coupling_map is not None and self.qubit_subset is None:
            self.qubit_subset = list(range(self.coupling_map.size()))

    def run(self, dag):
        """Run the BIPMapping pass on `dag`, assuming the number of virtual qubits (defined in
        `dag`) and the number of physical qubits (defined in `coupling_map`) are the same.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG. If there is no 2q-gate in DAG or it fails to map,
                returns the original dag.

        Raises:
            TranspilerError: if the number of virtual and physical qubits are not the same.
            AssertionError: if the final layout is not valid.
        """
        if self.coupling_map is None:
            return dag

        if len(dag.qubits) > len(self.qubit_subset):
            raise TranspilerError("More virtual qubits exist than physical qubits.")

        if len(dag.qubits) != len(self.qubit_subset):
            raise TranspilerError(
                "BIPMapping requires the number of virtual and physical qubits to be the same. "
                "Supply 'qubit_subset' to specify physical qubits to use."
            )

        original_dag = dag

        dummy_steps = math.ceil(math.sqrt(dag.num_qubits()))
        if self.max_swaps_inbetween_layers is not None:
            dummy_steps = max(0, self.max_swaps_inbetween_layers - 1)

        model = BIPMappingModel(
            dag=dag,
            coupling_map=self.coupling_map,
            qubit_subset=self.qubit_subset,
            dummy_timesteps=dummy_steps,
        )

        if len(model.su4layers) == 0:
            logger.info("BIPMapping is skipped due to no 2q-gates.")
            return original_dag

        model.create_cpx_problem(
            objective=self.objective,
            backend_prop=self.backend_prop,
            depth_obj_weight=self.depth_obj_weight,
            default_cx_error_rate=self.default_cx_error_rate,
        )

        status = model.solve_cpx_problem(time_limit=self.time_limit, threads=self.threads)
        if model.solution is None:
            logger.warning("Failed to solve a BIP problem. Status: %s", status)
            return original_dag

        # Get the optimized initial layout
        optimized_layout = model.get_layout(0)

        # Create a layout to track changes in layout for each layer
        layout = copy.deepcopy(optimized_layout)

        # Construct the mapped circuit
        canonical_qreg = QuantumRegister(self.coupling_map.size(), "q")
        mapped_dag = self._create_empty_dagcircuit(dag, canonical_qreg)
        interval = dummy_steps + 1
        for k, layer in enumerate(dag.layers()):
            if model.is_su4layer(k):
                su4dep = model.to_su4layer_depth(k)
                # add swaps between (su4dep-1)-th and su4dep-th su4layer
                from_steps = max(interval * (su4dep - 1), 0)
                to_steps = min(interval * su4dep, model.depth - 1)
                for t in range(from_steps, to_steps):  # pylint: disable=invalid-name
                    for (i, j) in model.get_swaps(t):
                        mapped_dag.apply_operation_back(
                            op=SwapGate(),
                            qargs=[canonical_qreg[i], canonical_qreg[j]],
                        )
                        # update layout, swapping physical qubits (i, j)
                        layout.swap(i, j)

            # map gates in k-th layer
            for node in layer["graph"].nodes():
                if isinstance(node, DAGOpNode):
                    mapped_dag.apply_operation_back(
                        op=copy.deepcopy(node.op),
                        qargs=[canonical_qreg[layout[q]] for q in node.qargs],
                        cargs=node.cargs,
                    )
                # TODO: double check with y values?

        # Check final layout
        final_layout = model.get_layout(model.depth - 1)
        if layout != final_layout:
            raise AssertionError(
                f"Bug: final layout {final_layout} != the layout computed from swaps {layout}"
            )

        self.property_set["layout"] = self._to_full_layout(optimized_layout)
        self.property_set["final_layout"] = self._to_full_layout(final_layout)

        return mapped_dag

    @staticmethod
    def _create_empty_dagcircuit(source_dag: DAGCircuit, canonical_qreg: QuantumRegister):
        target_dag = DAGCircuit()
        target_dag.name = source_dag.name
        target_dag._global_phase = source_dag._global_phase
        target_dag.metadata = source_dag.metadata

        target_dag.add_qreg(canonical_qreg)
        for creg in source_dag.cregs.values():
            target_dag.add_creg(creg)

        return target_dag

    def _to_full_layout(self, layout):
        # fill layout with ancilla qubits (required by drawers)
        idle_physical_qubits = [
            q for q in range(self.coupling_map.size()) if q not in layout.get_physical_bits()
        ]
        if idle_physical_qubits:
            qreg = QuantumRegister(len(idle_physical_qubits), name="ancilla")
            for idx, idle_q in enumerate(idle_physical_qubits):
                layout[idle_q] = qreg[idx]
            layout.add_register(qreg)
        return layout

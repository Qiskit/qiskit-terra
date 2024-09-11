# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rearrange the direction of the cx nodes to match the directed coupling map."""

from math import pi

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit import QuantumRegister, ControlFlowOp
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.library.standard_gates import (
    SGate,
    SdgGate,
    SXGate,
    HGate,
    CXGate,
    CZGate,
    ECRGate,
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
    SwapGate,
)


def _swap_node_qargs(node):
    return DAGOpNode(node.op, node.qargs[::-1], node.cargs)


class GateDirection(TransformationPass):
    """Modify asymmetric gates to match the hardware coupling direction.

    This pass makes use of the following identities::

                             ┌───┐┌───┐┌───┐
        q_0: ──■──      q_0: ┤ H ├┤ X ├┤ H ├
             ┌─┴─┐  =        ├───┤└─┬─┘├───┤
        q_1: ┤ X ├      q_1: ┤ H ├──■──┤ H ├
             └───┘           └───┘     └───┘


                          global phase: 3π/2
             ┌──────┐           ┌───┐ ┌────┐┌─────┐┌──────┐┌───┐
        q_0: ┤0     ├     q_0: ─┤ S ├─┤ √X ├┤ Sdg ├┤1     ├┤ H ├
             │  ECR │  =       ┌┴───┴┐├────┤└┬───┬┘│  Ecr │├───┤
        q_1: ┤1     ├     q_1: ┤ Sdg ├┤ √X ├─┤ S ├─┤0     ├┤ H ├
             └──────┘          └─────┘└────┘ └───┘ └──────┘└───┘


             ┌──────┐          ┌───┐┌──────┐┌───┐
        q_0: ┤0     ├     q_0: ┤ H ├┤1     ├┤ H ├
             │  RZX │  =       ├───┤│  RZX │├───┤
        q_1: ┤1     ├     q_1: ┤ H ├┤0     ├┤ H ├
             └──────┘          └───┘└──────┘└───┘

    This pass assumes that the positions of the qubits in the :attr:`.DAGCircuit.qubits` attribute
    are the physical qubit indices. For example if ``dag.qubits[0]`` is qubit 0 in the
    :class:`.CouplingMap` or :class:`.Target`.
    """

    _KNOWN_REPLACEMENTS = frozenset(["cx", "cz", "ecr", "swap", "rzx", "rxx", "ryy", "rzz"])

    def __init__(self, coupling_map, target=None):
        """GateDirection pass.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
            target (Target): The backend target to use for this pass. If this is specified
                it will be used instead of the coupling map
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.target = target

        # Create the replacement dag and associated register.
        self._cx_dag = DAGCircuit()
        qr = QuantumRegister(2)
        self._cx_dag.add_qreg(qr)
        self._cx_dag.apply_operation_back(HGate(), [qr[0]], [])
        self._cx_dag.apply_operation_back(HGate(), [qr[1]], [])
        self._cx_dag.apply_operation_back(CXGate(), [qr[1], qr[0]], [])
        self._cx_dag.apply_operation_back(HGate(), [qr[0]], [])
        self._cx_dag.apply_operation_back(HGate(), [qr[1]], [])

        # This is done in terms of less-efficient S/SX/Sdg gates instead of the more natural
        # `RY(pi /2)` so we have a chance for basis translation to keep things in a discrete basis
        # during resynthesis, if that's what's being asked for.
        self._ecr_dag = DAGCircuit()
        qr = QuantumRegister(2)
        self._ecr_dag.global_phase = -pi / 2
        self._ecr_dag.add_qreg(qr)
        self._ecr_dag.apply_operation_back(SGate(), [qr[0]], [])
        self._ecr_dag.apply_operation_back(SXGate(), [qr[0]], [])
        self._ecr_dag.apply_operation_back(SdgGate(), [qr[0]], [])
        self._ecr_dag.apply_operation_back(SdgGate(), [qr[1]], [])
        self._ecr_dag.apply_operation_back(SXGate(), [qr[1]], [])
        self._ecr_dag.apply_operation_back(SGate(), [qr[1]], [])
        self._ecr_dag.apply_operation_back(ECRGate(), [qr[1], qr[0]], [])
        self._ecr_dag.apply_operation_back(HGate(), [qr[0]], [])
        self._ecr_dag.apply_operation_back(HGate(), [qr[1]], [])

        self._cz_dag = DAGCircuit()
        qr = QuantumRegister(2)
        self._cz_dag.add_qreg(qr)
        self._cz_dag.apply_operation_back(CZGate(), [qr[1], qr[0]], [])

        self._swap_dag = DAGCircuit()
        qr = QuantumRegister(2)
        self._swap_dag.add_qreg(qr)
        self._swap_dag.apply_operation_back(SwapGate(), [qr[1], qr[0]], [])

        # If adding more replacements (either static or dynamic), also update the class variable
        # `_KNOWN_REPLACMENTS` to include them in the error messages.
        self._static_replacements = {
            "cx": self._cx_dag,
            "cz": self._cz_dag,
            "ecr": self._ecr_dag,
            "swap": self._swap_dag,
        }

    @staticmethod
    def _rzx_dag(parameter):
        _rzx_dag = DAGCircuit()
        qr = QuantumRegister(2)
        _rzx_dag.add_qreg(qr)
        _rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
        _rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
        _rzx_dag.apply_operation_back(RZXGate(parameter), [qr[1], qr[0]], [])
        _rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
        _rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
        return _rzx_dag

    @staticmethod
    def _rxx_dag(parameter):
        _rxx_dag = DAGCircuit()
        qr = QuantumRegister(2)
        _rxx_dag.add_qreg(qr)
        _rxx_dag.apply_operation_back(RXXGate(parameter), [qr[1], qr[0]], [])
        return _rxx_dag

    @staticmethod
    def _ryy_dag(parameter):
        _ryy_dag = DAGCircuit()
        qr = QuantumRegister(2)
        _ryy_dag.add_qreg(qr)
        _ryy_dag.apply_operation_back(RYYGate(parameter), [qr[1], qr[0]], [])
        return _ryy_dag

    @staticmethod
    def _rzz_dag(parameter):
        _rzz_dag = DAGCircuit()
        qr = QuantumRegister(2)
        _rzz_dag.add_qreg(qr)
        _rzz_dag.apply_operation_back(RZZGate(parameter), [qr[1], qr[0]], [])
        return _rzz_dag

    def _run_coupling_map(self, dag, wire_map, edges=None):
        if edges is None:
            edges = set(self.coupling_map.get_edges())
        if not edges:
            return dag
        # Don't include directives to avoid things like barrier, which are assumed always supported.
        for node in dag.op_nodes(include_directives=False):
            if isinstance(node.op, ControlFlowOp):
                new_op = node.op.replace_blocks(
                    dag_to_circuit(
                        self._run_coupling_map(
                            circuit_to_dag(block),
                            {
                                inner: wire_map[outer]
                                for outer, inner in zip(node.qargs, block.qubits)
                            },
                            edges,
                        )
                    )
                    for block in node.op.blocks
                )
                dag.substitute_node(node, new_op, propagate_condition=False)
                continue
            if len(node.qargs) != 2:
                continue
            if dag.has_calibration_for(node):
                continue
            qargs = (wire_map[node.qargs[0]], wire_map[node.qargs[1]])
            if qargs not in edges and (qargs[1], qargs[0]) not in edges:
                raise TranspilerError(
                    f"The circuit requires a connection between physical qubits {qargs}"
                )
            if qargs not in edges:
                replacement = self._static_replacements.get(node.name)
                if replacement is not None:
                    dag.substitute_node_with_dag(node, replacement)
                elif node.name == "rzx":
                    dag.substitute_node_with_dag(node, self._rzx_dag(*node.op.params))
                elif node.name == "rxx":
                    dag.substitute_node_with_dag(node, self._rxx_dag(*node.op.params))
                elif node.name == "ryy":
                    dag.substitute_node_with_dag(node, self._ryy_dag(*node.op.params))
                elif node.name == "rzz":
                    dag.substitute_node_with_dag(node, self._rzz_dag(*node.op.params))
                else:
                    raise TranspilerError(
                        f"'{node.name}' would be supported on '{qargs}' if the direction were"
                        f" swapped, but no rules are known to do that."
                        f" {list(self._KNOWN_REPLACEMENTS)} can be automatically flipped."
                    )
        return dag

    def _run_target(self, dag, wire_map):
        # Don't include directives to avoid things like barrier, which are assumed always supported.
        for node in dag.op_nodes(include_directives=False):
            if isinstance(node.op, ControlFlowOp):
                new_op = node.op.replace_blocks(
                    dag_to_circuit(
                        self._run_target(
                            circuit_to_dag(block),
                            {
                                inner: wire_map[outer]
                                for outer, inner in zip(node.qargs, block.qubits)
                            },
                        )
                    )
                    for block in node.op.blocks
                )
                dag.substitute_node(node, new_op, propagate_condition=False)
                continue
            if len(node.qargs) != 2:
                continue
            if dag.has_calibration_for(node):
                continue
            qargs = (wire_map[node.qargs[0]], wire_map[node.qargs[1]])
            swapped = (qargs[1], qargs[0])
            if node.name in self._static_replacements:
                if self.target.instruction_supported(node.name, qargs):
                    continue
                if self.target.instruction_supported(node.name, swapped):
                    dag.substitute_node_with_dag(node, self._static_replacements[node.name])
                else:
                    raise TranspilerError(
                        f"The circuit requires a connection between physical qubits {qargs}"
                        f" for {node.name}"
                    )
            elif node.name == "rzx":
                if self.target.instruction_supported(
                    qargs=qargs, operation_class=RZXGate, parameters=node.op.params
                ):
                    continue
                if self.target.instruction_supported(
                    qargs=swapped, operation_class=RZXGate, parameters=node.op.params
                ):
                    dag.substitute_node_with_dag(node, self._rzx_dag(*node.op.params))
                else:
                    raise TranspilerError(
                        f"The circuit requires a connection between physical qubits {qargs}"
                        f" for {node.name}"
                    )
            elif node.name == "rxx":
                if self.target.instruction_supported(
                    qargs=qargs, operation_class=RXXGate, parameters=node.op.params
                ):
                    continue
                if self.target.instruction_supported(
                    qargs=swapped, operation_class=RXXGate, parameters=node.op.params
                ):
                    dag.substitute_node_with_dag(node, self._rxx_dag(*node.op.params))
                else:
                    raise TranspilerError(
                        f"The circuit requires a connection between physical qubits {qargs}"
                        f" for {node.name}"
                    )
            elif node.name == "ryy":
                if self.target.instruction_supported(
                    qargs=qargs, operation_class=RYYGate, parameters=node.op.params
                ):
                    continue
                if self.target.instruction_supported(
                    qargs=swapped, operation_class=RYYGate, parameters=node.op.params
                ):
                    dag.substitute_node_with_dag(node, self._ryy_dag(*node.op.params))
                else:
                    raise TranspilerError(
                        f"The circuit requires a connection between physical qubits {qargs}"
                        f" for {node.name}"
                    )
            elif node.name == "rzz":
                if self.target.instruction_supported(
                    qargs=qargs, operation_class=RZZGate, parameters=node.op.params
                ):
                    continue
                if self.target.instruction_supported(
                    qargs=swapped, operation_class=RZZGate, parameters=node.op.params
                ):
                    dag.substitute_node_with_dag(node, self._rzz_dag(*node.op.params))
                else:
                    raise TranspilerError(
                        f"The circuit requires a connection between physical qubits {qargs}"
                        f" for {node.name}"
                    )
            elif self.target.instruction_supported(node.name, qargs):
                continue
            elif self.target.instruction_supported(node.name, swapped) or dag.has_calibration_for(
                _swap_node_qargs(node)
            ):
                raise TranspilerError(
                    f"'{node.name}' would be supported on '{qargs}' if the direction were"
                    f" swapped, but no rules are known to do that."
                    f" {list(self._KNOWN_REPLACEMENTS)} can be automatically flipped."
                )
            else:
                raise TranspilerError(
                    f"'{node.name}' with parameters '{node.op.params}' is not supported on qubits"
                    f" '{qargs}' in either direction."
                )

        return dag

    def run(self, dag):
        """Run the GateDirection pass on `dag`.

        Flips the cx nodes to match the directed coupling map. Modifies the
        input dag.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: The rearranged dag for the coupling map

        Raises:
            TranspilerError: If the circuit cannot be mapped just by flipping the
                cx nodes.
        """
        layout_map = {bit: i for i, bit in enumerate(dag.qubits)}
        if self.target is None:
            return self._run_coupling_map(dag, layout_map)
        return self._run_target(dag, layout_map)

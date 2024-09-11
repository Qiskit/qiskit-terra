# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Objects to represent the information at a node in the DAGCircuit."""
from __future__ import annotations

import typing
import uuid

import qiskit._accelerate.circuit
from qiskit.circuit import (
    Clbit,
    ClassicalRegister,
    IfElseOp,
    WhileLoopOp,
    SwitchCaseOp,
    ForLoopOp,
    Parameter,
    QuantumCircuit,
)
from qiskit.circuit.classical import expr

if typing.TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit


DAGNode = qiskit._accelerate.circuit.DAGNode
DAGOpNode = qiskit._accelerate.circuit.DAGOpNode
DAGInNode = qiskit._accelerate.circuit.DAGInNode
DAGOutNode = qiskit._accelerate.circuit.DAGOutNode


def _legacy_condition_eq(cond1, cond2, bit_indices1, bit_indices2) -> bool:
    if cond1 is cond2 is None:
        return True
    elif None in (cond1, cond2):
        return False
    target1, val1 = cond1
    target2, val2 = cond2
    if val1 != val2:
        return False
    if isinstance(target1, Clbit) and isinstance(target2, Clbit):
        return bit_indices1[target1] == bit_indices2[target2]
    if isinstance(target1, ClassicalRegister) and isinstance(target2, ClassicalRegister):
        return target1.size == target2.size and all(
            bit_indices1[t1] == bit_indices2[t2] for t1, t2 in zip(target1, target2)
        )
    return False


def _circuit_to_dag(circuit: QuantumCircuit, node_qargs, node_cargs, bit_indices) -> DAGCircuit:
    """Get a :class:`.DAGCircuit` of the given :class:`.QuantumCircuit`.  The bits in the output
    will be ordered in a canonical order based on their indices in the outer DAG, as defined by the
    ``bit_indices`` mapping and the ``node_{q,c}args`` arguments."""
    from qiskit.converters import circuit_to_dag  # pylint: disable=cyclic-import

    def sort_key(bits):
        outer, _inner = bits
        return bit_indices[outer]

    return circuit_to_dag(
        circuit,
        copy_operations=False,
        qubit_order=[
            inner for _outer, inner in sorted(zip(node_qargs, circuit.qubits), key=sort_key)
        ],
        clbit_order=[
            inner for _outer, inner in sorted(zip(node_cargs, circuit.clbits), key=sort_key)
        ],
    )


def _make_expr_key(bit_indices):
    def key(var):
        if isinstance(var, Clbit):
            return bit_indices.get(var)
        if isinstance(var, ClassicalRegister):
            return [bit_indices.get(bit) for bit in var]
        return None

    return key


def _condition_op_eq(node1, node2, bit_indices1, bit_indices2):
    cond1 = node1.op.condition
    cond2 = node2.op.condition
    if isinstance(cond1, expr.Expr) and isinstance(cond2, expr.Expr):
        if not expr.structurally_equivalent(
            cond1, cond2, _make_expr_key(bit_indices1), _make_expr_key(bit_indices2)
        ):
            return False
    elif isinstance(cond1, expr.Expr) or isinstance(cond2, expr.Expr):
        return False
    elif not _legacy_condition_eq(cond1, cond2, bit_indices1, bit_indices2):
        return False
    return len(node1.op.blocks) == len(node2.op.blocks) and all(
        _circuit_to_dag(block1, node1.qargs, node1.cargs, bit_indices1)
        == _circuit_to_dag(block2, node2.qargs, node2.cargs, bit_indices2)
        for block1, block2 in zip(node1.op.blocks, node2.op.blocks)
    )


def _switch_case_eq(node1, node2, bit_indices1, bit_indices2):
    target1 = node1.op.target
    target2 = node2.op.target
    if isinstance(target1, expr.Expr) and isinstance(target2, expr.Expr):
        if not expr.structurally_equivalent(
            target1, target2, _make_expr_key(bit_indices1), _make_expr_key(bit_indices2)
        ):
            return False
    elif isinstance(target1, Clbit) and isinstance(target2, Clbit):
        if bit_indices1[target1] != bit_indices2[target2]:
            return False
    elif isinstance(target1, ClassicalRegister) and isinstance(target2, ClassicalRegister):
        if target1.size != target2.size or any(
            bit_indices1[b1] != bit_indices2[b2] for b1, b2 in zip(target1, target2)
        ):
            return False
    else:
        return False
    cases1 = [case for case, _ in node1.op.cases_specifier()]
    cases2 = [case for case, _ in node2.op.cases_specifier()]
    return (
        len(cases1) == len(cases2)
        and all(set(labels1) == set(labels2) for labels1, labels2 in zip(cases1, cases2))
        and len(node1.op.blocks) == len(node2.op.blocks)
        and all(
            _circuit_to_dag(block1, node1.qargs, node1.cargs, bit_indices1)
            == _circuit_to_dag(block2, node2.qargs, node2.cargs, bit_indices2)
            for block1, block2 in zip(node1.op.blocks, node2.op.blocks)
        )
    )


def _for_loop_eq(node1, node2, bit_indices1, bit_indices2):
    indexset1, param1, body1 = node1.op.params
    indexset2, param2, body2 = node2.op.params
    if indexset1 != indexset2:
        return False
    if (param1 is None and param2 is not None) or (param1 is not None and param2 is None):
        return False
    if param1 is not None and param2 is not None:
        sentinel = Parameter(str(uuid.uuid4()))
        body1 = (
            body1.assign_parameters({param1: sentinel}, inplace=False)
            if param1 in body1.parameters
            else body1
        )
        body2 = (
            body2.assign_parameters({param2: sentinel}, inplace=False)
            if param2 in body2.parameters
            else body2
        )
    return _circuit_to_dag(body1, node1.qargs, node1.cargs, bit_indices1) == _circuit_to_dag(
        body2, node2.qargs, node2.cargs, bit_indices2
    )


_SEMANTIC_EQ_CONTROL_FLOW = {
    IfElseOp: _condition_op_eq,
    WhileLoopOp: _condition_op_eq,
    SwitchCaseOp: _switch_case_eq,
    ForLoopOp: _for_loop_eq,
}

_SEMANTIC_EQ_SYMMETRIC = frozenset({"barrier", "swap", "break_loop", "continue_loop"})

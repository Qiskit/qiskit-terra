# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Loop unrolling optimizations"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit import ForLoopOp, Instruction, Parameter
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit import DAGCircuit


class UnrollForLoops(TransformationPass):
    """Unroll for loops."""

    def run(self, dag):
        for node in dag.op_nodes(op=ForLoopOp):
            indexset, loop_parameter, body = node.op.params
            new_circ = body.copy_empty_like()
            if loop_parameter is None:
                for _ in range(len(indexset)):
                    new_circ.compose(body.copy(), inplace=True)
            elif isinstance(loop_parameter, Parameter):
                for index in range(len(indexset)):
                    i_body = body.assign_parameters({loop_parameter: index}, inplace=False)
                    new_circ.compose(i_body, inplace=True)
            new_dag = circuit_to_dag(new_circ)
            dag.substitute_node_with_dag(node, new_dag)
        return dag


class ForLoopBodyOptimizer(TransformationPass):
    """Optimize for loop bodies."""

    def __init__(self, opt_manager=None, pre_opt_limit=0, jamming_limit=0, post_opt_limit=0):
        """
        Initialize for loop unroller.

        Args:
           opt_manager (PassManager): Attempt to optimize loop body across loop
             boundaries using specified PassManager
           pre_opt_limit (int): how many operation back to check
           jamming_cnt (int): how many loops to consider jamming together. This should
             be an integer greater >= 2.
           post_opt_limit (int): how many operations to consider off the back
        """
        super().__init__()
        self._opt = opt_manager
        self._pre_opt_limit = pre_opt_limit
        if jamming_limit >= 2 or jamming_limit == 0:
            self._jamming_limit = jamming_limit
        else:
            raise TranspilerError(f"loop jamming limit set too small: {jamming_limit} < 2")
        self._post_opt_limit = post_opt_limit
        self._global_index_map = None

    def run(self, dag):
        self._global_index_map = {wire: idx for idx, wire in enumerate(dag.qubits)}
        for node in dag.op_nodes(op=ForLoopOp):
            # Optimize order here with pass manager?
            if self._pre_opt_limit:
                self._optimize_pre(dag, node)
            if self._post_opt_limit:
                self._optimize_post(dag, node)
            if self._jamming_limit:
                self._loop_jamming(dag, node)
        return dag

    def _optimize_pre(self, dag, node):
        """optimize body of for-loop with gates before it

        Args:
            dag (DAGCircuit): circuit to optimize
            node (DAGNode): for-loop dag node
        """
        indexset, loop_parameter, body = node.op.params
        trial_dag_in, block = _get_predecessor_dag(dag, node, self._pre_opt_limit)
        # TODO: elliminate this conversion
        trial_dag_out = circuit_to_dag(self._opt.run([dag_to_circuit(trial_dag_in)])[0])
        if sum(trial_dag_out.count_ops().values()) < sum(trial_dag_in.count_ops().values()):
            # trim idle qubits so block qubits matches trial_dag_out
            trial_dag_out.remove_qubits(*trial_dag_out.idle_wires())
            _replace_block_with_dag(dag, block, trial_dag_out)
            if len(indexset) > 1:
                new_for_loop = node.op.copy()
                new_for_loop.params = indexset[:-1], loop_parameter, body
                dag.substitute_node(node, new_for_loop)
            else:
                dag.remove_op_node(node)

    def _optimize_post(self, dag, node):
        """optimize body of for-loop with gates after it

        Args:
            dag (DAGCircuit): circuit to optimize
            node (DAGNode): for-loop dag node
        """
        indexset, loop_parameter, body = node.op.params
        trial_dag_in, block = _get_successor_dag(dag, node, self._post_opt_limit)
        # TODO: elliminate this conversion
        trial_dag_out = circuit_to_dag(self._opt.run([dag_to_circuit(trial_dag_in)])[0])
        if sum(trial_dag_out.count_ops().values()) < sum(trial_dag_in.count_ops().values()):
            # trim idle qubits so block qubits matches trial_dag_out
            trial_dag_out.remove_qubits(*trial_dag_out.idle_wires())
            _replace_block_with_dag(dag, block, trial_dag_out)
            if len(indexset) > 1:
                new_for_loop = node.op.copy()
                new_for_loop.params = indexset[:-1], loop_parameter, body
                dag.substitute_node(node, new_for_loop)
            else:
                dag.remove_op_node(node)

    def _frequency_reduction(self, dag, node):
        """extract invariant operations outside the loop"""
        raise NotImplementedError()

    def _loop_jamming(self, dag, node):
        """fuse two or more loops

        Args:
            dag (DAGCircuit): circuit to optimize
            node (DAGOpNode): for-loop dag node
        """
        # scale up to limit
        indexset, loop_parameter, body = node.op.params
        loop_cnt = len(indexset)
        base_cost = sum(body.count_ops().values())
        trial_records = []  # cost for jamming up to self._jamming_limit loops
        trial_num_loops = list(range(2, min(self._jamming_limit, loop_cnt)))
        for num_loops in trial_num_loops:
            quot, rem = divmod(loop_cnt, num_loops)
            trial_circ_in = body.copy_empty_like()
            for _ in range(num_loops):
                trial_circ_in.compose(body, inplace=True)
            trial_circ_out = self._opt.run([trial_circ_in])[0]
            new_body_cost = sum(trial_circ_out.count_ops().values())
            this_cost = quot * new_body_cost
            if rem:
                # TODO: decide whether to unroll remainder iteration(s) before or after loop
                this_cost += rem * base_cost
            record = {"body": trial_circ_out, "cost": this_cost, "loop_divmod": (quot, rem)}
            trial_records.append(record)
        index_jamming_cost = [record["cost"] for record in trial_records]
        min_trial_cost = min(index_jamming_cost)
        if min_trial_cost < loop_cnt * base_cost:
            best_index = index_jamming_cost.index(min_trial_cost)
            best_record = trial_records[best_index]
            quot, rem = best_record["loop_divmod"]
            cf_op = ForLoopOp(range(quot), loop_parameter, best_record["body"])
            new_dag = DAGCircuit()
            new_dag.add_qubits(body.qubits)
            new_dag.add_clbits(body.clbits)
            new_dag.apply_operation_back(cf_op, qargs=node.qargs, cargs=node.cargs)
            if rem:
                # TODO: use _optimize_pre/_optimize_post to decide whether to apply remainder to
                # front or back
                for cinstr in body.data:
                    new_dag.apply_operation_back(
                        cinstr.operation, qargs=cinstr.qubits, cargs=cinstr.clbits
                    )
            dag.substitute_node_with_dag(node, new_dag)


def _get_predecessor_dag(dag, node, cnt):
    """get the circuit formed from the body of the for-loop node and
    previous cnt nodes"""
    _, _, body = node.op.params

    # extract body of for_loop into new dag
    trial_dag = dag.copy_empty_like()
    trial_dag.global_phase = 0
    qubit_map = dict(zip(body.qubits, node.qargs))
    clbit_map = dict(zip(body.clbits, node.cargs))
    for circ_instr in body.data:
        qargs = [qubit_map[qubit] for qubit in circ_instr.qubits]
        cargs = [clbit_map[clbit] for clbit in circ_instr.clbits]
        trial_dag.apply_operation_back(circ_instr.operation, qargs=qargs, cargs=cargs)
    # add predecessor nodes from dag
    pre_iter = dag.bfs_predecessors(node)
    block = []  # tracks the nodes added from _outside_ the loop
    while cnt > 0:
        try:
            node_rec = next(pre_iter)
        except StopIteration:
            break
        this_node = node_rec[0]
        trial_dag.apply_operation_front(this_node.op, qargs=this_node.qargs, cargs=this_node.cargs)
        block.append(this_node)
        cnt -= 1
    return trial_dag, block


def _get_successor_dag(dag, node, cnt):
    """get the circuit formed from the body of the for-loop node and
    subsequent cnt nodes"""
    _, _, body = node.op.params

    # extract body of for_loop into new dag
    trial_dag = dag.copy_empty_like()
    trial_dag.global_phase = 0
    qubit_map = dict(zip(body.qubits, node.qargs))
    clbit_map = dict(zip(body.clbits, node.cargs))
    for circ_instr in body.data:
        qargs = [qubit_map[qubit] for qubit in circ_instr.qubits]
        cargs = [clbit_map[clbit] for clbit in circ_instr.clbits]
        trial_dag.apply_operation_back(circ_instr.operation, qargs=qargs, cargs=cargs)

    # add predecessor nodes from dag
    post_iter = dag.bfs_successors(node)

    block = []  # tracks the nodes added from _outside_ the loop
    while cnt > 0:
        try:
            node_rec = next(post_iter)
        except StopIteration:
            break
        for child_node in node_rec[1]:
            block.append(child_node)
            trial_dag.apply_operation_back(
                child_node.op, qargs=child_node.qargs, cargs=child_node.cargs
            )
            cnt -= 1
    return trial_dag, block


def _replace_block_with_dag(dag, block_a, dag_b):
    """
    Substitute a block of nodes with a dag
    There should be 1-to-1 correspondance between qubits in block_a and dag_b.
    """
    block_qubit_inds = sorted({dag.qubits.index(bit) for node in block_a for bit in node.qargs})
    num_block_qubits = len(block_qubit_inds)
    if num_block_qubits != dag_b.num_qubits():
        raise TranspilerError(
            f"Number of qubits in block ({num_block_qubits}) "
            f"does not match replacement DAG ({dag_b.num_qubits()})"
        )
    block_clbit_inds = sorted([dag.clbits.index(bit) for node in block_a for bit in node.cargs])
    index_map = {dag.qubits[ind]: ind for ind in block_qubit_inds}
    placeholder_op = Instruction(
        "placeholder", len(block_qubit_inds), len(block_clbit_inds), params=[]
    )
    placeholder_node = dag.replace_block_with_op(block_a, placeholder_op, index_map)
    node_id_map = dag.substitute_node_with_dag(placeholder_node, dag_b)
    node_map = {node: node_id_map[node._node_id] for node in dag_b.op_nodes()}
    return node_map

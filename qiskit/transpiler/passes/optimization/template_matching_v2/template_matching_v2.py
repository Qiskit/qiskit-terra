# This code is part of Qiskit.
#
# (C) Copyright IBM 2020-2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Template matching for all possible qubit configurations and initial matches. It
returns the list of all matches obtained from this algorithm.


**Reference:**

[1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
Exact and practical pattern matching for quantum circuit optimization.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

"""

import itertools

from qiskit.transpiler.passes.optimization.template_matching_v2.template_utils_v2 import (
    get_node,
    get_qindices,
    get_cindices,
    get_descendants,
    get_successors,
)
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.transpiler.passes.optimization.template_matching_v2.forward_match_v2 import ForwardMatch
from qiskit.transpiler.passes.optimization.template_matching_v2.backward_match_v2 import (
    BackwardMatch,
)


class TemplateMatching:
    """
    Class TemplatingMatching allows to apply the full template matching algorithm.
    """

    def __init__(
        self,
        circuit_dag_dep,
        template_dag_dep,
        heuristics_qubits_param=None,
        heuristics_backward_param=None,
    ):
        """
        Create a TemplateMatching object with necessary arguments.
        Args:
            circuit_dag_dep (DAGDependencyV2): circuit dag.
            template_dag_dep (DAGDependencyV2): template dag.
            heuristics_qubits_param (list[int]): [length]
            heuristics_backward_param (list[int]): [length, survivor]
        """
        self.circuit_dag_dep = circuit_dag_dep
        self.template_dag_dep = template_dag_dep
        self.match_list = []
        self.successorstovisit = {}
        self.heuristics_qubits_param = heuristics_qubits_param
        self.heuristics_backward_param = heuristics_backward_param

    def _list_first_match_new(self, node_circuit, node_template, n_qubits_t, n_clbits_t):
        """
        Returns the list of qubits for the circuit given the first match, the unknown qubits are
        replaced by -1.
        Args:
            node_circuit (DAGOpNode): First match node in the circuit.
            node_template (DAGOpNode): First match node in the template.
            n_qubits_t (int): number of qubits in the template.
            n_clbits_t (int): number of classical bits in the template.
        Returns:
            list: list of qubits to consider in circuit (with specific order).
        """
        l_q = []

        # Controlled gate
        if isinstance(node_circuit.op, ControlledGate) and node_template.op.num_ctrl_qubits > 1:
            control = node_template.op.num_ctrl_qubits
            control_qubits_circuit = get_qindices(self.circuit_dag_dep, node_circuit)[:control]
            not_control_qubits_circuit = get_qindices(self.circuit_dag_dep, node_circuit)[control::]

            # Symmetric base gate
            if node_template.op.base_gate.name not in ["rxx", "ryy", "rzz", "swap", "iswap", "ms"]:
                for control_perm_q in itertools.permutations(control_qubits_circuit):
                    control_perm_q = list(control_perm_q)
                    l_q_sub = [-1] * n_qubits_t
                    for q in get_qindices(self.template_dag_dep, node_template):
                        node_circuit_perm = control_perm_q + not_control_qubits_circuit
                        l_q_sub[q] = node_circuit_perm[
                            get_qindices(self.template_dag_dep, node_template).index(q)
                        ]
                    l_q.append(l_q_sub)
            # Not symmetric base gate
            else:
                for control_perm_q in itertools.permutations(control_qubits_circuit):
                    control_perm_q = list(control_perm_q)
                    for not_control_perm_q in itertools.permutations(not_control_qubits_circuit):
                        not_control_perm_q = list(not_control_perm_q)
                        l_q_sub = [-1] * n_qubits_t
                        for q in get_qindices(self.template_dag_dep, node_template):
                            node_circuit_perm = control_perm_q + not_control_perm_q
                            l_q_sub[q] = node_circuit_perm[
                                get_qindices(self.template_dag_dep, node_template).index(q)
                            ]
                        l_q.append(l_q_sub)
        # Not controlled
        else:
            # Symmetric gate
            if node_template.op.name not in ["rxx", "ryy", "rzz", "swap", "iswap", "ms"]:
                l_q_sub = [-1] * n_qubits_t
                for q in get_qindices(self.template_dag_dep, node_template):
                    l_q_sub[q] = get_qindices(self.circuit_dag_dep, node_circuit)[
                        get_qindices(self.template_dag_dep, node_template).index(q)
                    ]
                l_q.append(l_q_sub)
            # Not symmetric
            else:
                for perm_q in itertools.permutations(
                    get_qindices(self.circuit_dag_dep, node_circuit)
                ):
                    l_q_sub = [-1] * n_qubits_t
                    for q in get_qindices(self.template_dag_dep, node_template):
                        l_q_sub[q] = perm_q[
                            get_qindices(self.template_dag_dep, node_template).index(q)
                        ]
                    l_q.append(l_q_sub)

        # Classical control
        if not get_cindices(self.template_dag_dep, node_template) or not get_cindices(
            self.circuit_dag_dep, node_circuit
        ):
            l_c = []
        else:
            l_c = [-1] * n_clbits_t
            for c in get_cindices(self.template_dag_dep, node_template):
                l_c[c] = node_circuit[get_cindices(self.template_dag_dep, node_template).index(c)]

        return l_q, l_c

    def _sublist(self, qubit_id_list, exclude, length):
        """
        Function that returns all possible combinations of a given length, considering an
        excluded list of elements.
        Args:
            qubit_id_list (list): list of qubits indices from the circuit.
            exclude (list): list of qubits from the first matched circuit gate.
            length (int): length of the list to be returned (number of template qubits minus
                number of qubits from the first matched template gate).
        Yield:
            iterator: Iterator of the possible lists.
        """
        for sublist in itertools.combinations(
            [e for e in qubit_id_list if e not in exclude], length
        ):
            yield list(sublist)

    def _list_qubit_clbit_circuit(self, list_first_match, permutation):
        """
        Function that returns the list of the circuit qubits and clbits given a permutation
        and an initial match.
        Args:
            list_first_match (list): list of qubit indices for the initial match.
            permutation (list): possible permutation for the circuit qubits.
        Returns:
            list: list of circuit qubits for the given permutation and initial match.
        """
        list_circuit = []

        counter = 0

        for elem in list_first_match:
            if elem == -1:
                list_circuit.append(permutation[counter])
                counter = counter + 1
            else:
                list_circuit.append(elem)

        return list_circuit

    def _add_match(self, backward_match_list):
        """
        Method to add a match in the list only if it is not already in it.
        If the match is already in the list, the qubit configuration
        is appended to the existing match.
        Args:
            backward_match_list (list): match from the backward part of the
            algorithm.
        """

        already_in = False

        for b_match in backward_match_list:
            for l_match in self.match_list:
                if b_match.match == l_match.match:
                    index = self.match_list.index(l_match)
                    self.match_list[index].qubit.append(b_match.qubit[0])
                    already_in = True
                    break

            if not already_in:
                self.match_list.append(b_match)

    def _explore_circuit(self, node_c, node_t, n_qubits_t, length):
        """
        Explore the descendants of the node_c (up to the given length).
        Args:
            node_c (DAGOpNode): first match in the circuit.
            node_t (DAGOpNode): first match in the template.
            n_qubits_t (int): number of qubits in the template.
            length (int): length for exploration of the descendants.
        Returns:
            list: qubit configuration for the 'length' descendants of node_c.
        """
        template_nodes = range(node_t._node_id + 1, self.template_dag_dep.size())
        circuit_nodes = range(0, self.circuit_dag_dep.size())
        node_t_descs = get_descendants(self.template_dag_dep, node_t._node_id)
        node_c_descs = get_descendants(self.circuit_dag_dep, node_c._node_id)

        counter = 1
        qubit_set = set(get_qindices(self.circuit_dag_dep, node_c))
        if 2 * len(node_t_descs) > len(template_nodes):
            for desc in node_c_descs:
                qarg = get_qindices(get_node(self.circuit_dag_dep, desc))
                if (len(qubit_set | set(qarg))) <= n_qubits_t and counter <= length:
                    qubit_set = qubit_set | set(qarg)
                    counter += 1
                elif (len(qubit_set | set(qarg))) > n_qubits_t:
                    return list(qubit_set)
            return list(qubit_set)

        else:
            not_descendants = list(set(circuit_nodes) - set(node_c_descs))
            candidate = [
                not_descendants[j]
                for j in range(len(not_descendants) - 1, len(not_descendants) - 1 - length, -1)
            ]

            for not_desc in candidate:
                qarg = get_qindices(self.circuit_dag_dep, get_node(self.circuit_dag_dep, not_desc))
                if counter <= length and (len(qubit_set | set(qarg))) <= n_qubits_t:
                    qubit_set = qubit_set | set(qarg)
                    counter += 1
                elif (len(qubit_set | set(qarg))) > n_qubits_t:
                    return list(qubit_set)
            return list(qubit_set)

    def run_template_matching(self):
        """
        Run the complete algorithm for finding all maximal matches for the given template and
        circuit. First it fixes the configuration of the circuit due to the first match.
        Then it explores all compatible qubit configurations of the circuit. For each
        qubit configurations, we apply first the Forward part of the algorithm  and then
        the Backward part of the algorithm. The longest matches for the given configuration
        are stored. Finally, the list of stored matches is sorted.
        """

        # Get the number of qubits/clbits for both circuit and template.
        n_qubits_c = len(self.circuit_dag_dep.qubits)
        n_clbits_c = len(self.circuit_dag_dep.clbits)

        n_qubits_t = len(self.template_dag_dep.qubits)
        n_clbits_t = len(self.template_dag_dep.clbits)

        # # Initialize successorstovisit
        for node in self.circuit_dag_dep.op_nodes():
            self.successorstovisit[node] = get_successors(self.circuit_dag_dep, node._node_id)

        # Loop over the nodes of both template and circuit.
        for node_t in self.template_dag_dep.op_nodes():
            for node_c in self.circuit_dag_dep.op_nodes():

                # Operations match up to ParameterExpressions.
                if node_c.op.soft_compare(node_t.op):
                    qarg_c = get_qindices(self.circuit_dag_dep, node_c)
                    carg_c = get_cindices(self.circuit_dag_dep, node_c)

                    qarg_t = get_qindices(self.template_dag_dep, node_t)
                    carg_t = get_cindices(self.template_dag_dep, node_t)

                    # Fix the qubits and clbits configuration given the first match.
                    all_list_first_match_q, list_first_match_c = self._list_first_match_new(
                        node_c,
                        node_t,
                        n_qubits_t,
                        n_clbits_t,
                    )
                    list_qubits_c = list(range(0, n_qubits_c))
                    list_clbits_c = list(range(0, n_clbits_c))

                    # If the parameter for qubit heuristics is given then extract
                    # the list of qubits for the descendants (length(int)) in the circuit.

                    if self.heuristics_qubits_param:
                        heuristics_qubits = self._explore_circuit(
                            node_c, node_t, n_qubits_t, self.heuristics_qubits_param[0]
                        )
                    else:
                        heuristics_qubits = []

                    for sub_q in self._sublist(list_qubits_c, qarg_c, n_qubits_t - len(qarg_t)):
                        # If the heuristics qubits are a subset of the given qubit configuration,
                        # then this configuration is accepted.
                        if set(heuristics_qubits).issubset(set(sub_q) | set(qarg_c)):
                            # Permute the qubit configuration.
                            for perm_q in itertools.permutations(sub_q):
                                perm_q = list(perm_q)
                                for list_first_match_q in all_list_first_match_q:
                                    list_qubit_circuit = self._list_qubit_clbit_circuit(
                                        list_first_match_q, perm_q
                                    )

                                    # Check for clbit configurations if there are clbits.
                                    if list_clbits_c:
                                        for sub_c in self._sublist(
                                            list_clbits_c, carg_c, n_clbits_t - len(carg_t)
                                        ):
                                            for perm_c in itertools.permutations(sub_c):
                                                perm_c = list(perm_c)

                                                list_clbit_circuit = self._list_qubit_clbit_circuit(
                                                    list_first_match_c, perm_c
                                                )

                                                # Apply the forward match part of the algorithm.
                                                forward = ForwardMatch(
                                                    self.circuit_dag_dep,
                                                    self.template_dag_dep,
                                                    node_c,
                                                    node_t,
                                                    self.successorstovisit,
                                                    list_qubit_circuit,
                                                    list_clbit_circuit,
                                                )
                                                matchedwith, isblocked = forward.run_forward_match()

                                                # Apply the backward match part of the algorithm.
                                                backward = BackwardMatch(
                                                    forward.circuit_dag_dep,
                                                    forward.template_dag_dep,
                                                    forward.match,
                                                    node_c,
                                                    node_t,
                                                    matchedwith,
                                                    isblocked,
                                                    list_qubit_circuit,
                                                    list_clbit_circuit,
                                                    self.heuristics_backward_param,
                                                )

                                                backward.run_backward_match()

                                                # Add the matches to the list.
                                                self._add_match(backward.match_final)
                                    else:
                                        # Apply the forward match part of the algorithm.
                                        forward = ForwardMatch(
                                            self.circuit_dag_dep,
                                            self.template_dag_dep,
                                            node_c,
                                            node_t,
                                            self.successorstovisit,
                                            list_qubit_circuit,
                                        )
                                        matchedwith, isblocked = forward.run_forward_match()

                                        # Apply the backward match part of the algorithm.
                                        backward = BackwardMatch(
                                            forward.circuit_dag_dep,
                                            forward.template_dag_dep,
                                            forward.match,
                                            node_c,
                                            node_t,
                                            matchedwith,
                                            isblocked,
                                            list_qubit_circuit,
                                            [],
                                            self.heuristics_backward_param,
                                        )
                                        backward.run_backward_match()

                                        # Add the matches to the list.
                                        self._add_match(backward.match_final)

        # Sort the list of matches according to the length of the matches (decreasing order).
        self.match_list.sort(key=lambda x: len(x.match), reverse=True)

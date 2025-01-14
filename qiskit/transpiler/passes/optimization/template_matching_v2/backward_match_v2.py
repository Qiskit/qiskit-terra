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
Template matching in the backward direction, it takes an initial match, a
configuration of qubit, both circuit and template as inputs and the list
obtained from forward match. The result is a list of matches between the
template and the circuit.


**Reference:**

[1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
Exact and practical pattern matching for quantum circuit optimization.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

"""
import heapq

from qiskit.transpiler.passes.optimization.template_matching_v2.template_utils_v2 import (
    get_node,
    get_qindices,
    get_cindices,
    get_descendants,
    get_ancestors,
)
from qiskit.circuit.controlledgate import ControlledGate


class Match:
    """
    Object to represent a match and its qubit and clbit configurations.
    """

    def __init__(self, match, qubit, clbit):
        """
        Create a Match class with necessary arguments.
        Args:
            match (list): list of matched gates.
            qubit (list): list of qubits configuration.
            clbit (list): list of clbits configuration.

        """
        # Match list
        self.match = match
        # Qubits list for circuit
        self.qubit = [qubit]
        # Clbits for template
        self.clbit = [clbit]


class MatchingScenarios:
    """
    Class to represent a matching scenario.
    """

    def __init__(
        self, circuit_matched, circuit_blocked, template_matched, template_blocked, matches, counter
    ):
        """
        Create a MatchingScenarios class with necessary arguments.
        Args:
            circuit_matched (list): list of matchedwith attributes in the circuit.
            circuit_blocked (list): list of isblocked attributes in the circuit.
            template_matched (list): list of matchedwith attributes in the template.
            template_blocked (list): list of isblocked attributes in the template.
            matches (list): list of matches.
            counter (int): counter of the number of circuit gates already considered.
        """
        self.circuit_matched = circuit_matched
        self.template_matched = template_matched
        self.circuit_blocked = circuit_blocked
        self.template_blocked = template_blocked
        self.matches = matches
        self.counter = counter


class MatchingScenariosList:
    """
    Object to define a list of MatchingScenarios, with method to append
    """

    def __init__(self):
        """
        Create an empty MatchingScenariosList.
        """
        self.matching_scenarios_list = []

    def append_scenario(self, matching):
        """
        Append a scenario to the list.
        Args:
            matching (MatchingScenarios): a scenario of match.
        """
        self.matching_scenarios_list.append(matching)


class BackwardMatch:
    """
    Class BackwardMatch allows to run backward direction part of template
    matching algorithm.
    """

    def __init__(
        self,
        circuit_dag_dep,
        template_dag_dep,
        forward_matches,
        node_c,
        node_t,
        matchedwith,
        isblocked,
        qubits,
        clbits=None,
        heuristics_backward_param=None,
    ):
        """
        Create a BackwardMatch class with necessary arguments.
        Args:
            circuit_dag_dep (_DAGDependencyV2): circuit in the dag dependency form.
            template_dag_dep (_DAGDependencyV2): template in the dag dependency form.
            forward_matches (list): list of matches obtained in the forward direction.
            node_c (DAGOpNode): node of the first gate matched in the circuit.
            node_t (DAGOpNode): node of the first gate matched in the template.
            matchedwith (dict): per node list of matches
            isblocked (dict): per node indicator of blocked node
            qubits (list): list of considered qubits in the circuit.
            clbits (list): list of considered clbits in the circuit.
            heuristics_backward_param (list): list that contains the two parameters for
                applying the heuristics (length and survivor).
        """
        self.circuit_dag_dep = circuit_dag_dep
        self.template_dag_dep = template_dag_dep
        self.qubits = qubits
        self.clbits = clbits
        self.node_c = node_c
        self.node_t = node_t
        self.forward_matches = forward_matches
        self.match_final = []
        self.heuristics_backward_param = heuristics_backward_param
        self.matching_list = MatchingScenariosList()
        self.matchedwith = matchedwith
        self.isblocked = isblocked

    def _find_backward_candidates(self, template_blocked, matches):
        """
        Function which returns the list of possible backward candidates in the template dag.
        Args:
            template_blocked (list): list of attributes isblocked in the template circuit.
            matches (list): list of matches.
        Returns:
            list: list of backward candidates (id).
        """
        template_block = []

        for node_id in range(self.node_t._node_id, self.template_dag_dep.size()):
            if template_blocked[node_id]:
                template_block.append(node_id)

        matches_template = sorted(match[0] for match in matches)

        node_t_descs = get_descendants(self.template_dag_dep, self.node_t._node_id)
        potential = []
        for index in range(self.node_t._node_id + 1, self.template_dag_dep.size()):
            if (index not in node_t_descs) and (index not in template_block):
                potential.append(index)

        candidates_indices = list(set(potential) - set(matches_template))
        candidates_indices = sorted(candidates_indices)
        candidates_indices.reverse()

        return candidates_indices

    def _is_same_q_conf(self, node_c, node_t, qarg_circuit):
        """
        Check if the qubit configurations are compatible.
        Args:
            node_c (DAGOpNode): node in the circuit.
            node_t (DAGOpNode): node in the template.
            qarg_circuit (list): qubit configuration for the Instruction in the circuit.
        Returns:
            bool: True if possible, False otherwise.
        """
        # If the gate is controlled, then the control qubits have to be compared as sets.
        node_temp_qind = get_qindices(self.template_dag_dep, node_t)
        if isinstance(node_c.op, ControlledGate):
            c_template = node_t.op.num_ctrl_qubits

            if c_template == 1:
                return qarg_circuit == node_temp_qind

            else:
                control_qubits_template = node_temp_qind[:c_template]
                control_qubits_circuit = qarg_circuit[:c_template]

                if set(control_qubits_circuit) == set(control_qubits_template):
                    target_qubits_template = node_temp_qind[c_template::]
                    target_qubits_circuit = qarg_circuit[c_template::]

                    if node_t.op.base_gate.name in [
                        "rxx",
                        "ryy",
                        "rzz",
                        "swap",
                        "iswap",
                        "ms",
                    ]:
                        return set(target_qubits_template) == set(target_qubits_circuit)
                    else:
                        return target_qubits_template == target_qubits_circuit
                else:
                    return False
        # For non controlled gates, the qubit indices for symmetric gates can be compared as sets
        # But for non-symmetric gates the qubits indices have to be compared as lists.
        else:
            if node_t.op.name in ["rxx", "ryy", "rzz", "swap", "iswap", "ms"]:
                return set(qarg_circuit) == set(node_temp_qind)
            else:
                return qarg_circuit == node_temp_qind

    def _is_same_c_conf(self, node_c, node_t, carg_circuit):
        """
        Check if the clbit configurations are compatible.
        Args:
            node_c (DAGOpNode): node in the circuit.
            node_t (DAGOpNode): node in the template.
            carg_circuit (list): clbit configuration for the Instruction in the circuit.
        Returns:
            bool: True if possible, False otherwise.
        """
        if (getattr(node_c.op, "condition", None) and getattr(node_t.op, "condition", None)) and (
            getattr(node_c.op, "condition", None)[1] != getattr(node_t.op, "condition", None)[1]
            or set(carg_circuit) != set(get_cindices(self.template_dag_dep, node_t))
        ):
            return False
        return True

    def _backward_heuristics(self, gate_indices, length, survivor):
        """
        Heuristics to cut the tree in the backward match algorithm
        Args:
            gate_indices (list): list of candidates in the circuit.
            length (int): depth for cutting the tree, cutting operation is repeated every length.
            survivor (int): number of survivor branches.
        """
        # Set the list of the counter for the different scenarios.
        list_counter = []

        for scenario in self.matching_list.matching_scenarios_list:
            list_counter.append(scenario.counter)

        metrics = []
        # If all scenarios have the same counter and the counter is divisible by length.
        if list_counter.count(list_counter[0]) == len(list_counter) and list_counter[0] <= len(
            gate_indices
        ):
            if (list_counter[0] - 1) % length == 0:
                # The list metrics contains metric results for each scenario.
                for scenario in self.matching_list.matching_scenarios_list:
                    metrics.append(len(scenario.matches))
                # Select only the scenarios with higher metrics for the given number of survivors.
                largest = heapq.nlargest(survivor, range(len(metrics)), key=lambda x: metrics[x])
                self.matching_list.matching_scenarios_list = [
                    i
                    for j, i in enumerate(self.matching_list.matching_scenarios_list)
                    if j in largest
                ]

    def run_backward_match(self):
        """
        Apply the backward match algorithm and return the list of matches given an initial match
        and a circuit qubits configuration.
        """
        match_store_list = []
        counter = 1

        # Initialize the list of attributes matchedwith and isblocked.
        circuit_matched = [self.matchedwith.get(node) for node in self.circuit_dag_dep.op_nodes()]
        circuit_blocked = [self.isblocked.get(node) for node in self.circuit_dag_dep.op_nodes()]
        template_matched = [self.matchedwith.get(node) for node in self.template_dag_dep.op_nodes()]
        template_blocked = [self.isblocked.get(node) for node in self.template_dag_dep.op_nodes()]

        # First Scenario is stored in the MatchingScenariosList().
        first_match = MatchingScenarios(
            circuit_matched,
            circuit_blocked,
            template_matched,
            template_blocked,
            self.forward_matches,
            counter,
        )
        self.matching_list = MatchingScenariosList()
        self.matching_list.append_scenario(first_match)

        # Set the circuit indices that can be matched.
        gate_indices = [
            node._node_id
            for node in self.circuit_dag_dep.op_nodes()
            if (not self.matchedwith.get(node) and not self.isblocked.get(node))
        ]
        gate_indices.reverse()

        number_of_gates_to_match = (
            self.template_dag_dep.size() - (self.node_t._node_id - 1) - len(self.forward_matches)
        )

        # While the scenario stack is not empty.
        while self.matching_list.matching_scenarios_list:

            # If parameters are given, the heuristics is applied.
            if self.heuristics_backward_param:
                self._backward_heuristics(
                    gate_indices,
                    self.heuristics_backward_param[0],
                    self.heuristics_backward_param[1],
                )

            scenario = self.matching_list.matching_scenarios_list.pop(0)

            circuit_matched = scenario.circuit_matched
            circuit_blocked = scenario.circuit_blocked
            template_matched = scenario.template_matched
            template_blocked = scenario.template_blocked
            matches_scenario = scenario.matches
            counter_scenario = scenario.counter

            # Part of the match list coming from the backward match.
            match_backward = [
                match for match in matches_scenario if match not in self.forward_matches
            ]

            # Matches are stored if the counter is bigger than the length of the list of
            # candidates in the circuit. Or if number of gate left to match is the same as
            # the length of the backward part of the match.
            if (
                counter_scenario > len(gate_indices)
                or len(match_backward) == number_of_gates_to_match
            ):
                matches_scenario.sort(key=lambda x: x[0])
                match_store_list.append(Match(matches_scenario, self.qubits, self.clbits))
                continue

            # First circuit candidate.
            circuit_id = gate_indices[counter_scenario - 1]
            node_c = get_node(self.circuit_dag_dep, circuit_id)

            # If the circuit candidate is blocked, only the counter is changed.
            if circuit_blocked[circuit_id]:
                matching_scenario = MatchingScenarios(
                    circuit_matched,
                    circuit_blocked,
                    template_matched,
                    template_blocked,
                    matches_scenario,
                    counter_scenario + 1,
                )
                self.matching_list.append_scenario(matching_scenario)
                continue

            # The candidates in the template.
            candidates_indices = self._find_backward_candidates(template_blocked, matches_scenario)

            # Update of the qubits/clbits indices in the circuit in order to be
            # comparable with the one in the template.
            qarg_indices = get_qindices(self.circuit_dag_dep, node_c)
            carg_indices = get_cindices(self.circuit_dag_dep, node_c)

            qarg1 = [self.qubits.index(q) for q in qarg_indices if q in self.qubits]
            if len(qarg1) != len(qarg_indices):
                qarg1 = []

            carg1 = [self.clbits.index(c) for c in carg_indices if c in self.clbits]
            if len(carg1) != len(carg_indices):
                carg1 = []

            global_match = False
            global_broken = []

            # Loop over the template candidates.
            for template_id in candidates_indices:

                node_t = get_node(self.template_dag_dep, template_id)
                qarg2 = get_qindices(
                    self.template_dag_dep, get_node(self.template_dag_dep, template_id)
                )

                # Necessary but not sufficient conditions for a match to happen.
                if (
                    len(qarg1) != len(qarg2)
                    or set(qarg1) != set(qarg2)
                    or node_c.name != node_t.name
                ):
                    continue

                # Check if the qubit, clbit configurations are compatible for a match,
                # also check if the operations are the same.
                if (
                    self._is_same_q_conf(node_c, node_t, qarg1)
                    and self._is_same_c_conf(node_c, node_t, carg1)
                    and node_c.op == node_t.op
                ):
                    block_list = []
                    broken_matches_match = []

                    # Loop to check if the match is not connected, in this case
                    # the descendants matches are blocked and unmatched.
                    for potential_block in get_descendants(self.template_dag_dep, template_id):
                        if not template_matched[potential_block]:
                            template_blocked[potential_block] = True
                            block_list.append(potential_block)
                            for block_id in block_list:
                                for desc_id in get_descendants(self.template_dag_dep, block_id):
                                    template_blocked[desc_id] = True
                                    if template_matched[desc_id]:
                                        new_id = template_matched[desc_id][0]
                                        circuit_matched[new_id] = []
                                        template_matched[desc_id] = []
                                        broken_matches_match.append(desc_id)

                    if broken_matches_match:
                        global_broken.append(True)
                    else:
                        global_broken.append(False)

                    new_matches_scenario = [
                        elem for elem in matches_scenario if elem[0] not in broken_matches_match
                    ]

                    condition = True

                    for back_match in match_backward:
                        if back_match not in new_matches_scenario:
                            condition = False
                            break

                    # First option greedy match.
                    if ([self.node_t._node_id, self.node_c._node_id] in new_matches_scenario) and (
                        condition or not match_backward
                    ):
                        template_matched[template_id] = [circuit_id]
                        circuit_matched[circuit_id] = [template_id]
                        new_matches_scenario.append([template_id, circuit_id])

                        new_matching_scenario = MatchingScenarios(
                            circuit_matched,
                            circuit_blocked,
                            template_matched,
                            template_blocked,
                            new_matches_scenario,
                            counter_scenario + 1,
                        )
                        self.matching_list.append_scenario(new_matching_scenario)

                        global_match = True

            if global_match:
                circuit_matched_block_s = circuit_matched.copy()
                circuit_blocked_block_s = circuit_blocked.copy()

                template_matched_block_s = template_matched.copy()
                template_blocked_block_s = template_blocked.copy()

                matches_scenario_block_s = matches_scenario.copy()
                circuit_blocked_block_s[circuit_id] = True
                broken_matches = []

                # Second option, not a greedy match, block all descendants (push the gate
                # to the right).
                for desc in get_descendants(self.circuit_dag_dep, circuit_id):
                    circuit_blocked_block_s[desc] = True
                    if circuit_matched_block_s[desc]:
                        broken_matches.append(desc)
                        new_id = circuit_matched_block_s[desc][0]
                        template_matched_block_s[new_id] = []
                        circuit_matched_block_s[desc] = []

                new_matches_scenario_block_s = [
                    elem for elem in matches_scenario_block_s if elem[1] not in broken_matches
                ]
                condition_not_greedy = True

                for back_match in match_backward:
                    if back_match not in new_matches_scenario_block_s:
                        condition_not_greedy = False
                        break

                if (
                    [self.node_t._node_id, self.node_c._node_id] in new_matches_scenario_block_s
                ) and (condition_not_greedy or not match_backward):
                    new_matching_scenario = MatchingScenarios(
                        circuit_matched_block_s,
                        circuit_blocked_block_s,
                        template_matched_block_s,
                        template_blocked_block_s,
                        new_matches_scenario_block_s,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(new_matching_scenario)

                # Third option: if blocking the descendants breaks a match, we consider
                # also the possibility to block all ancestors (push the gate to the left).
                if broken_matches and all(global_broken):
                    circuit_blocked[circuit_id] = True
                    for anc in get_ancestors(self.circuit_dag_dep, circuit_id):
                        circuit_blocked[anc] = True

                    matching_scenario = MatchingScenarios(
                        circuit_matched,
                        circuit_blocked,
                        template_matched,
                        template_blocked,
                        matches_scenario,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(matching_scenario)

            # If there is no match then there are three options.
            else:
                circuit_blocked[circuit_id] = True
                following_matches = []
                for desc in get_descendants(self.circuit_dag_dep, circuit_id):
                    if circuit_matched[desc]:
                        following_matches.append(desc)

                # First option, the circuit gate is not disturbing because there are no
                # following match and no ancestors.
                node_c_ancs = get_ancestors(self.circuit_dag_dep, circuit_id)

                if not node_c_ancs or not following_matches:
                    matching_scenario = MatchingScenarios(
                        circuit_matched,
                        circuit_blocked,
                        template_matched,
                        template_blocked,
                        matches_scenario,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(matching_scenario)

                else:
                    # Second option, all ancestors are blocked (circuit gate is
                    # moved to the left).
                    for anc in node_c_ancs:
                        circuit_blocked[anc] = True

                    matching_scenario = MatchingScenarios(
                        circuit_matched,
                        circuit_blocked,
                        template_matched,
                        template_blocked,
                        matches_scenario,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(matching_scenario)

                    # Third option, all descendants are blocked (circuit gate is
                    # moved to the right).

                    broken_matches = []

                    for desc in get_descendants(self.circuit_dag_dep, circuit_id):
                        circuit_blocked[desc] = True
                        if circuit_matched[desc]:
                            broken_matches.append(desc)
                            circuit_matched[desc] = []

                    new_matches_scenario = [
                        elem for elem in matches_scenario if elem[1] not in broken_matches
                    ]
                    condition_block = True

                    for back_match in match_backward:
                        if back_match not in new_matches_scenario:
                            condition_block = False
                            break

                    if ([self.node_t._node_id, self.node_c._node_id] in matches_scenario) and (
                        condition_block or not match_backward
                    ):
                        new_matching_scenario = MatchingScenarios(
                            circuit_matched,
                            circuit_blocked,
                            template_matched,
                            template_blocked,
                            new_matches_scenario,
                            counter_scenario + 1,
                        )
                        self.matching_list.append_scenario(new_matching_scenario)

        length = max(len(m.match) for m in match_store_list)

        # Store the matches with maximal length.
        for scenario in match_store_list:
            if (len(scenario.match) == length) and not any(
                scenario.match == x.match for x in self.match_final
            ):
                self.match_final.append(scenario)

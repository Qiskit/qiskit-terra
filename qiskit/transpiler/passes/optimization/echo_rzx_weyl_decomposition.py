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

"""Weyl decomposition of two-qubit gates in terms of echoed cross-resonance gates."""

from typing import Tuple

from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import RZXGate, HGate, XGate

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes.calibration.rzx_builder import _check_calibration_type, CRCalType
from qiskit.utils.deprecate_pulse import deprecate_pulse_dependency

from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag


class EchoRZXWeylDecomposition(TransformationPass):
    """Rewrite two-qubit gates using the Weyl decomposition.

    This transpiler pass rewrites two-qubit gates in terms of echoed cross-resonance gates according
    to the Weyl decomposition. A two-qubit gate will be replaced with at most six non-echoed RZXGates.
    Each pair of RZXGates forms an echoed RZXGate.
    """

    @deprecate_pulse_dependency
    def __init__(self, instruction_schedule_map=None, target=None):
        """EchoRZXWeylDecomposition pass.

        Args:
            instruction_schedule_map (InstructionScheduleMap): the mapping from circuit
                :class:`~.circuit.Instruction` names and arguments to :class:`.Schedule`\\ s.
            target (Target): The :class:`~.Target` representing the target backend, if both
                ``instruction_schedule_map`` and ``target`` are specified then this argument will take
                precedence and ``instruction_schedule_map`` will be ignored.
        """
        super().__init__()
        self._inst_map = instruction_schedule_map
        if target is not None:
            self._inst_map = target.instruction_schedule_map()

    def _is_native(self, qubit_pair: Tuple) -> bool:
        """Return the direction of the qubit pair that is native."""
        cal_type, _, _ = _check_calibration_type(self._inst_map, qubit_pair)
        return cal_type in [
            CRCalType.ECR_CX_FORWARD,
            CRCalType.ECR_FORWARD,
            CRCalType.DIRECT_CX_FORWARD,
        ]

    @staticmethod
    def _echo_rzx_dag(theta):
        """Return the following circuit

        .. code-block:: text

                 ┌───────────────┐┌───┐┌────────────────┐┌───┐
            q_0: ┤0              ├┤ X ├┤0               ├┤ X ├
                 │  Rzx(theta/2) │└───┘│  Rzx(-theta/2) │└───┘
            q_1: ┤1              ├─────┤1               ├─────
                 └───────────────┘     └────────────────┘
        """
        rzx_dag = DAGCircuit()
        qr = QuantumRegister(2)
        rzx_dag.add_qreg(qr)
        rzx_dag.apply_operation_back(RZXGate(theta / 2), [qr[0], qr[1]], [])
        rzx_dag.apply_operation_back(XGate(), [qr[0]], [])
        rzx_dag.apply_operation_back(RZXGate(-theta / 2), [qr[0], qr[1]], [])
        rzx_dag.apply_operation_back(XGate(), [qr[0]], [])
        return rzx_dag

    @staticmethod
    def _reverse_echo_rzx_dag(theta):
        """Return the following circuit

        .. code-block:: text

                 ┌───┐┌───────────────┐     ┌────────────────┐┌───┐
            q_0: ┤ H ├┤1              ├─────┤1               ├┤ H ├─────
                 ├───┤│  Rzx(theta/2) │┌───┐│  Rzx(-theta/2) │├───┤┌───┐
            q_1: ┤ H ├┤0              ├┤ X ├┤0               ├┤ X ├┤ H ├
                 └───┘└───────────────┘└───┘└────────────────┘└───┘└───┘
        """
        reverse_rzx_dag = DAGCircuit()
        qr = QuantumRegister(2)
        reverse_rzx_dag.add_qreg(qr)
        reverse_rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
        reverse_rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
        reverse_rzx_dag.apply_operation_back(RZXGate(theta / 2), [qr[1], qr[0]], [])
        reverse_rzx_dag.apply_operation_back(XGate(), [qr[1]], [])
        reverse_rzx_dag.apply_operation_back(RZXGate(-theta / 2), [qr[1], qr[0]], [])
        reverse_rzx_dag.apply_operation_back(XGate(), [qr[1]], [])
        reverse_rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
        reverse_rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
        return reverse_rzx_dag

    def run(self, dag: DAGCircuit):
        """Run the EchoRZXWeylDecomposition pass on `dag`.

        Rewrites two-qubit gates in an arbitrary circuit in terms of echoed cross-resonance
        gates by computing the Weyl decomposition of the corresponding unitary. Modifies the
        input dag.

        Args:
            dag (DAGCircuit): DAG to rewrite.

        Returns:
            DAGCircuit: The modified dag.

        Raises:
            TranspilerError: If the circuit cannot be rewritten.
        """

        # pylint: disable=cyclic-import
        from qiskit.quantum_info import Operator
        from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitControlledUDecomposer

        if len(dag.qregs) > 1:
            raise TranspilerError(
                "EchoRZXWeylDecomposition expects a single qreg input DAG,"
                f"but input DAG had qregs: {dag.qregs}."
            )

        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        decomposer = TwoQubitControlledUDecomposer(RZXGate)

        for node in dag.two_qubit_ops():

            unitary = Operator(node.op).data
            dag_weyl = circuit_to_dag(decomposer(unitary))
            dag.substitute_node_with_dag(node, dag_weyl)

        for node in dag.two_qubit_ops():
            if node.name == "rzx":
                control = node.qargs[0]
                target = node.qargs[1]

                physical_q0 = trivial_layout[control]
                physical_q1 = trivial_layout[target]

                is_native = self._is_native((physical_q0, physical_q1))

                theta = node.op.params[0]
                if is_native:
                    dag.substitute_node_with_dag(node, self._echo_rzx_dag(theta))
                else:
                    dag.substitute_node_with_dag(node, self._reverse_echo_rzx_dag(theta))

        return dag

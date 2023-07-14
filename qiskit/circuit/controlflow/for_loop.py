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

"Circuit operation representing a ``for`` loop."

import warnings
from typing import Iterable, Optional, Union

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumcircuit import QuantumCircuit

from .control_flow import ControlFlowOp


class ForLoopOp(ControlFlowOp):
    """A circuit operation which repeatedly executes a subcircuit
    (``body``) parameterized by a parameter ``loop_parameter`` through
    the set of integer values provided in ``indexset``.

    Parameters:
        indexset: A collection of integers to loop over.
        loop_parameter: The placeholder parameterizing ``body`` to which
            the values from ``indexset`` will be assigned.
        body: The loop body to be repeatedly executed.
        label: An optional label for identifying the instruction.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤0          ├
             │           │
        q_1: ┤1          ├
             │  for_loop │
        q_2: ┤2          ├
             │           │
        c_0: ╡0          ╞
             └───────────┘

    """

    def __init__(
        self,
        indexset: Iterable[int],
        loop_parameter: Union[Parameter, None],
        body: QuantumCircuit,
        label: Optional[str] = None,
    ):
        from qiskit.converters.circuit import num_qubits as _num_qubits
        from qiskit.converters.circuit import num_clbits as _num_clbits
        num_qubits = _num_qubits(body)
        num_clbits = _num_clbits(body)

        super().__init__(
            "for_loop", num_qubits, num_clbits, [indexset, loop_parameter, body], label=label
        )

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, parameters):
        indexset, loop_parameter, body = parameters

        if not isinstance(loop_parameter, (Parameter, type(None))):
            raise CircuitError(
                "ForLoopOp expects a loop_parameter parameter to "
                "be either of type Parameter or None, but received "
                f"{type(loop_parameter)}."
            )

        from qiskit.converters.circuit import is_circuit
        from qiskit.converters.circuit import num_qubits as _num_qubits
        from qiskit.converters.circuit import num_clbits as _num_clbits
        if not is_circuit(body):
            raise CircuitError(
                "ForLoopOp expects a body parameter to be of type "
                f"QuantumCircuit or DAGCircuit, but received {type(body)}."
            )

        if _num_qubits(body) != self.num_qubits or _num_clbits(body) != self.num_clbits:
            raise CircuitError(
                "Attempted to assign a body parameter with a num_qubits or "
                "num_clbits different than that of the ForLoopOp. "
                f"ForLoopOp num_qubits/clbits: {self.num_qubits}/{self.num_clbits} "
                f"Supplied body num_qubits/clbits: {_num_qubits(body)}/{_num_clbits(body)}."
            )

        if (
            loop_parameter is not None
            and hasattr(body, "parameters")
            and loop_parameter not in body.parameters
            and loop_parameter.name in (p.name for p in body.parameters)
        ):
            warnings.warn(
                "The Parameter provided as a loop_parameter was not found "
                "on the loop body and so no binding of the indexset to loop "
                "parameter will occur. A different Parameter of the same name "
                f"({loop_parameter.name}) was found. If you intended to loop "
                "over that Parameter, please use that Parameter instance as "
                "the loop_parameter.",
                stacklevel=2,
            )

        # Consume indexset into a tuple unless it was provided as a range.
        # Preserve ranges so that they can be exported as OpenQASM3 ranges.
        indexset = indexset if isinstance(indexset, range) else tuple(indexset)

        self._params = [indexset, loop_parameter, body]

    @property
    def blocks(self):
        return (self._params[2],)

    def replace_blocks(self, blocks):
        (body,) = blocks
        return ForLoopOp(self.params[0], self.params[1], body, label=self.label)


class ForLoopContext:
    """A context manager for building up ``for`` loops onto circuits in a natural order, without
    having to construct the loop body first.

    Within the block, a lot of the bookkeeping is done for you; you do not need to keep track of
    which qubits and clbits you are using, for example, and a loop parameter will be allocated for
    you, if you do not supply one yourself.  All normal methods of accessing the qubits on the
    underlying :obj:`~QuantumCircuit` will work correctly, and resolve into correct accesses within
    the interior block.

    You generally should never need to instantiate this object directly.  Instead, use
    :obj:`.QuantumCircuit.for_loop` in its context-manager form, i.e. by not supplying a ``body`` or
    sets of qubits and clbits.

    Example usage::

        import math
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2, 1)

        with qc.for_loop(range(5)) as i:
            qc.rx(i * math.pi/4, 0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            qc.break_loop().c_if(0, True)

    This context should almost invariably be created by a :meth:`.QuantumCircuit.for_loop` call, and
    the resulting instance is a "friend" of the calling circuit.  The context will manipulate the
    circuit's defined scopes when it is entered (by pushing a new scope onto the stack) and exited
    (by popping its scope, building it, and appending the resulting :obj:`.ForLoopOp`).

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    # Class-level variable keep track of the number of auto-generated loop variables, so we don't
    # get naming clashes.
    _generated_loop_parameters = 0

    __slots__ = (
        "_circuit",
        "_generate_loop_parameter",
        "_loop_parameter",
        "_indexset",
        "_label",
        "_used",
    )

    def __init__(
        self,
        circuit: QuantumCircuit,
        indexset: Iterable[int],
        loop_parameter: Optional[Parameter] = None,
        *,
        label: Optional[str] = None,
    ):
        self._circuit = circuit
        self._generate_loop_parameter = loop_parameter is None
        self._loop_parameter = loop_parameter
        # We can pass through `range` instances because OpenQASM 3 has native support for this type
        # of iterator set.
        self._indexset = indexset if isinstance(indexset, range) else tuple(indexset)
        self._label = label
        self._used = False

    def __enter__(self):
        if self._used:
            raise CircuitError("A for-loop context manager cannot be re-entered.")
        self._used = True
        self._circuit._push_scope()
        if self._generate_loop_parameter:
            self._loop_parameter = Parameter(f"_loop_i_{self._generated_loop_parameters}")
            type(self)._generated_loop_parameters += 1
        return self._loop_parameter

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # If we're leaving the context manager because an exception was raised, there's nothing
            # to do except restore the circuit state.
            self._circuit._pop_scope()
            return False
        scope = self._circuit._pop_scope()
        # Loops do not need to pass any further resources in, because this scope itself defines the
        # extent of ``break`` and ``continue`` statements.
        body = scope.build(scope.qubits, scope.clbits)
        # We always bind the loop parameter if the user gave it to us, even if it isn't actually
        # used, because they requested we do that by giving us a parameter.  However, if they asked
        # us to auto-generate a parameter, then we only add it if they actually used it, to avoid
        # using unnecessary resources.
        if self._generate_loop_parameter and self._loop_parameter not in body.parameters:
            loop_parameter = None
        else:
            loop_parameter = self._loop_parameter
        self._circuit.append(
            ForLoopOp(self._indexset, loop_parameter, body, label=self._label),
            tuple(body.qubits),
            tuple(body.clbits),
        )
        return False

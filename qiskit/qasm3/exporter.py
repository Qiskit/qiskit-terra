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

"""QASM3 Exporter"""

import collections
import re
import io
import itertools
import numbers
from os.path import dirname, join, abspath
from typing import Iterable, List, Sequence, Union

from qiskit.circuit import (
    Barrier,
    CircuitInstruction,
    Clbit,
    Gate,
    Instruction,
    Measure,
    Parameter,
    ParameterExpression,
    QuantumCircuit,
    Qubit,
    Reset,
    Delay,
    Store,
)
from qiskit.circuit.bit import Bit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import (
    IfElseOp,
    ForLoopOp,
    WhileLoopOp,
    SwitchCaseOp,
    ControlFlowOp,
    BreakLoopOp,
    ContinueLoopOp,
    CASE_DEFAULT,
)
from qiskit.circuit.library import standard_gates
from qiskit.circuit.register import Register
from qiskit.circuit.tools import pi_check

from . import ast
from .experimental import ExperimentalFeatures
from .exceptions import QASM3ExporterError
from .printer import BasicPrinter


# Reserved keywords that gates and variables cannot be named.  It is possible that some of these
# _could_ be accepted as variable names by OpenQASM 3 parsers, but it's safer for us to just be very
# conservative.
_RESERVED_KEYWORDS = frozenset(
    {
        "OPENQASM",
        "angle",
        "array",
        "barrier",
        "bit",
        "bool",
        "box",
        "break",
        "cal",
        "complex",
        "const",
        "continue",
        "creg",
        "ctrl",
        "def",
        "defcal",
        "defcalgrammar",
        "delay",
        "duration",
        "durationof",
        "else",
        "end",
        "extern",
        "float",
        "for",
        "gate",
        "gphase",
        "if",
        "in",
        "include",
        "input",
        "int",
        "inv",
        "let",
        "measure",
        "mutable",
        "negctrl",
        "output",
        "pow",
        "qreg",
        "qubit",
        "reset",
        "return",
        "sizeof",
        "stretch",
        "uint",
        "while",
    }
)

# This probably isn't precisely the same as the OQ3 spec, but we'd need an extra dependency to fully
# handle all Unicode character classes, and this should be close enough for users who aren't
# actively _trying_ to break us (fingers crossed).
_VALID_IDENTIFIER = re.compile(r"[\w][\w\d]*", flags=re.U)


def _escape_invalid_identifier(name: str) -> str:
    if name in _RESERVED_KEYWORDS or not _VALID_IDENTIFIER.fullmatch(name):
        name = "_" + re.sub(r"[^\w\d]", "_", name)
    return name


class Exporter:
    """QASM3 exporter main class."""

    def __init__(
        self,
        includes: Sequence[str] = ("stdgates.inc",),
        basis_gates: Sequence[str] = ("U",),
        disable_constants: bool = False,
        alias_classical_registers: bool = None,
        allow_aliasing: bool = None,
        indent: str = "  ",
        experimental: ExperimentalFeatures = ExperimentalFeatures(0),
    ):
        """
        Args:
            includes: the filenames that should be emitted as includes.  These files will be parsed
                for gates, and any objects dumped from this exporter will use those definitions
                where possible.
            basis_gates: the basic defined gate set of the backend.
            disable_constants: if ``True``, always emit floating-point constants for numeric
                parameter values.  If ``False`` (the default), then values close to multiples of
                OpenQASM 3 constants (``pi``, ``euler``, and ``tau``) will be emitted in terms of those
                constants instead, potentially improving accuracy in the output.
            alias_classical_registers: If ``True``, then bits may be contained in more than one
                register.  If so, the registers will be emitted using "alias" definitions, which
                might not be well supported by consumers of OpenQASM 3.

                .. seealso::
                    Parameter ``allow_aliasing``
                        A value for ``allow_aliasing`` overrides any value given here, and
                        supersedes this parameter.
            allow_aliasing: If ``True``, then bits may be contained in more than one register.  If
                so, the registers will be emitted using "alias" definitions, which might not be
                well supported by consumers of OpenQASM 3.  Defaults to ``False`` or the value of
                ``alias_classical_registers``.

                .. versionadded:: 0.25.0
            indent: the indentation string to use for each level within an indented block.  Can be
                set to the empty string to disable indentation.
            experimental: any experimental features to enable during the export.  See
                :class:`ExperimentalFeatures` for more details.
        """
        self.basis_gates = basis_gates
        self.disable_constants = disable_constants
        self.allow_aliasing = (
            allow_aliasing if allow_aliasing is not None else (alias_classical_registers or False)
        )
        self.includes = list(includes)
        self.indent = indent
        self.experimental = experimental

    def dumps(self, circuit):
        """Convert the circuit to OpenQASM 3, returning the result as a string."""
        with io.StringIO() as stream:
            self.dump(circuit, stream)
            return stream.getvalue()

    def dump(self, circuit, stream):
        """Convert the circuit to OpenQASM 3, dumping the result to a file or text stream."""
        builder = QASM3Builder(
            circuit,
            includeslist=self.includes,
            basis_gates=self.basis_gates,
            disable_constants=self.disable_constants,
            allow_aliasing=self.allow_aliasing,
            experimental=self.experimental,
        )
        BasicPrinter(stream, indent=self.indent, experimental=self.experimental).visit(
            builder.build_program()
        )


class GlobalNamespace:
    """Global namespace dict-like."""

    BASIS_GATE = object()

    qiskit_gates = {
        "p": standard_gates.PhaseGate,
        "x": standard_gates.XGate,
        "y": standard_gates.YGate,
        "z": standard_gates.ZGate,
        "h": standard_gates.HGate,
        "s": standard_gates.SGate,
        "sdg": standard_gates.SdgGate,
        "t": standard_gates.TGate,
        "tdg": standard_gates.TdgGate,
        "sx": standard_gates.SXGate,
        "rx": standard_gates.RXGate,
        "ry": standard_gates.RYGate,
        "rz": standard_gates.RZGate,
        "cx": standard_gates.CXGate,
        "cy": standard_gates.CYGate,
        "cz": standard_gates.CZGate,
        "cp": standard_gates.CPhaseGate,
        "crx": standard_gates.CRXGate,
        "cry": standard_gates.CRYGate,
        "crz": standard_gates.CRZGate,
        "ch": standard_gates.CHGate,
        "swap": standard_gates.SwapGate,
        "ccx": standard_gates.CCXGate,
        "cswap": standard_gates.CSwapGate,
        "cu": standard_gates.CUGate,
        "CX": standard_gates.CXGate,
        "phase": standard_gates.PhaseGate,
        "cphase": standard_gates.CPhaseGate,
        "id": standard_gates.IGate,
        "u1": standard_gates.U1Gate,
        "u2": standard_gates.U2Gate,
        "u3": standard_gates.U3Gate,
    }
    include_paths = [abspath(join(dirname(__file__), "..", "qasm", "libs"))]

    def __init__(self, includelist, basis_gates=()):
        self._data = {gate: self.BASIS_GATE for gate in basis_gates}
        self._data["U"] = self.BASIS_GATE

        for includefile in includelist:
            if includefile == "stdgates.inc":
                self._data.update(self.qiskit_gates)
            else:
                # TODO What do if an inc file is not standard?
                # Should it be parsed?
                pass

    def __setitem__(self, name_str, instruction):
        self._data[name_str] = instruction.base_class
        self._data[id(instruction)] = name_str

    def __getitem__(self, key):
        if isinstance(key, Instruction):
            try:
                # Registered gates.
                return self._data[id(key)]
            except KeyError:
                pass
            # Built-in gates.
            if key.name not in self._data:
                raise KeyError(key)
            return key.name
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, instruction):
        if isinstance(instruction, standard_gates.UGate):
            return True
        if id(instruction) in self._data:
            return True
        if self._data.get(instruction.name) is self.BASIS_GATE:
            return True
        if type(instruction) in [Gate, Instruction]:  # user-defined instructions/gate
            return self._data.get(instruction.name, None) == instruction
        type_ = self._data.get(instruction.name)
        if isinstance(type_, type) and isinstance(instruction, type_):
            return True
        return False

    def has_symbol(self, name: str) -> bool:
        """Whether a symbol's name is present in the table."""
        return name in self._data

    def register(self, instruction):
        """Register an instruction in the namespace"""
        # The second part of the condition is a nasty hack to ensure that gates that come with at
        # least one parameter always have their id in the name.  This is a workaround a bug, where
        # gates with parameters do not contain the information required to build the gate definition
        # in symbolic form (unless the parameters are all symbolic).  The exporter currently
        # (2021-12-01) builds gate declarations with parameters in the signature, but then ignores
        # those parameters during the body, and just uses the concrete values from the first
        # instance of the gate it sees, such as:
        #     gate rzx(_gate_p_0) _gate_q_0, _gate_q_1 {
        #         h _gate_q_1;
        #         cx _gate_q_0, _gate_q_1;
        #         rz(0.2) _gate_q_1;        // <- note the concrete value.
        #         cx _gate_q_0, _gate_q_1;
        #         h _gate_q_1;
        #     }
        # This then means that multiple calls to the same gate with different parameters will be
        # incorrect.  By forcing all gates to be defined including their id, we generate a QASM3
        # program that does what was intended, even though the output QASM3 is silly.  See gh-7335.
        if instruction.name in self._data or (
            isinstance(instruction, Gate)
            and not all(isinstance(param, Parameter) for param in instruction.params)
        ):
            key = f"{instruction.name}_{id(instruction)}"
        else:
            key = instruction.name
        self[key] = instruction


# A _Scope is the structure used in the builder to store the contexts and re-mappings of bits from
# the top-level scope where the bits were actually defined.  In the class, 'circuit' is an instance
# of QuantumCircuit that defines this level, and 'bit_map' is a mapping of 'Bit: Bit', where the
# keys are bits in the circuit in this scope, and the values are the Bit in the top-level scope in
# this context that this bit actually represents.  'symbol_map' is a bidirectional mapping of
# '<Terra object>: Identifier' and 'str: <Terra object>', where the string in the second map is the
# name of the identifier.  This is a cheap hack around actually implementing a proper symbol table.
_Scope = collections.namedtuple("_Scope", ("circuit", "bit_map", "symbol_map"))


class QASM3Builder:
    """QASM3 builder constructs an AST from a QuantumCircuit."""

    builtins = (Barrier, Measure, Reset, Delay, BreakLoopOp, ContinueLoopOp, Store)
    loose_bit_prefix = "_bit"
    loose_qubit_prefix = "_qubit"
    gate_parameter_prefix = "_gate_p"
    gate_qubit_prefix = "_gate_q"

    def __init__(
        self,
        quantumcircuit,
        includeslist,
        basis_gates,
        disable_constants,
        allow_aliasing,
        experimental=ExperimentalFeatures(0),
    ):
        # This is a stack of stacks; the outer stack is a list of "outer" look-up contexts, and the
        # inner stack is for scopes within these.  A "outer" look-up context in this sense means
        # the main program body or a gate/subroutine definition, whereas the scopes are for things
        # like the body of a ``for`` loop construct.
        self._circuit_ctx = []
        self.push_context(quantumcircuit)
        self.includeslist = includeslist
        # `_global_io_declarations` and `_global_classical_declarations` are stateful, and any
        # operation that needs a parameter can append to them during the build.  We make all
        # classical declarations global because the IBM qe-compiler stack (our initial consumer of
        # OQ3 strings) prefers declarations to all be global, and it's valid OQ3, so it's not vendor
        # lock-in.  It's possibly slightly memory inefficient, but that's not likely to be a problem
        # in the near term.
        self._global_io_declarations = []
        self._global_classical_forward_declarations = []
        # An arbitrary counter to help with generation of unique ids for symbol names when there are
        # clashes (though we generally prefer to keep user names if possible).
        self._counter = itertools.count()
        self.disable_constants = disable_constants
        self.allow_aliasing = allow_aliasing
        self.global_namespace = GlobalNamespace(includeslist, basis_gates)
        self.experimental = experimental

    def _unique_name(self, prefix: str, scope: _Scope) -> str:
        table = scope.symbol_map
        name = basename = _escape_invalid_identifier(prefix)
        while name in table or name in _RESERVED_KEYWORDS or self.global_namespace.has_symbol(name):
            name = f"{basename}__generated{next(self._counter)}"
        return name

    def _register_gate(self, gate):
        self.global_namespace.register(gate)

    def _register_opaque(self, instruction):
        self.global_namespace.register(instruction)

    def _register_variable(self, variable, scope: _Scope, name=None) -> ast.Identifier:
        """Register a variable in the symbol table for the given scope, returning the name that
        should be used to refer to the variable.  The same name will be returned by subsequent calls
        to :meth:`_lookup_variable` within the same scope.

        If ``name`` is given explicitly, it must not already be defined in the scope.
        """
        # Note that the registration only checks for the existence of a variable that was declared
        # in the current scope, not just one that's available.  This is a rough implementation of
        # the shadowing proposal currently being drafted for OpenQASM 3, though we expect it to be
        # expanded and modified in the future (2022-03-07).
        table = scope.symbol_map
        if name is not None:
            if name in _RESERVED_KEYWORDS:
                raise QASM3ExporterError(f"cannot reserve the keyword '{name}' as a variable name")
            if name in table:
                raise QASM3ExporterError(
                    f"tried to reserve '{name}', but it is already used by '{table[name]}'"
                )
            if self.global_namespace.has_symbol(name):
                raise QASM3ExporterError(
                    f"tried to reserve '{name}', but it is already used by a gate"
                )
        else:
            name = self._unique_name(variable.name, scope)
        identifier = ast.Identifier(name)
        table[identifier.string] = variable
        table[variable] = identifier
        return identifier

    def _reserve_variable_name(self, name: ast.Identifier, scope: _Scope) -> ast.Identifier:
        """Reserve a variable name in the given scope, raising a :class:`.QASM3ExporterError` if
        the name is already in use.

        This is useful for autogenerated names that the exporter itself reserves when dealing with
        objects that have no standard Terra object backing them.

        Returns the same identifier, for convenience in chaining."""
        table = scope.symbol_map
        if name.string in table:
            variable = table[name.string]
            raise QASM3ExporterError(
                f"tried to reserve '{name.string}', but it is already used by '{variable}'"
            )
        table[name.string] = "<internal object>"
        return name

    def _lookup_variable(self, variable) -> ast.Identifier:
        """Lookup a Terra object within the current context, and return the name that should be used
        to represent it in OpenQASM 3 programmes."""
        if isinstance(variable, Bit):
            variable = self.current_scope().bit_map[variable]
        for scope in reversed(self.current_context()):
            if variable in scope.symbol_map:
                return scope.symbol_map[variable]
        raise KeyError(f"'{variable}' is not defined in the current context")

    def build_header(self):
        """Builds a Header"""
        version = ast.Version("3.0")
        includes = self.build_includes()
        return ast.Header(version, includes)

    def build_program(self):
        """Builds a Program"""
        circuit = self.global_scope(assert_=True).circuit
        if circuit.num_captured_vars:
            raise QASM3ExporterError(
                "cannot export an inner scope with captured variables as a top-level program"
            )
        header = self.build_header()

        opaques_to_declare, gates_to_declare = self.hoist_declarations(
            circuit.data, opaques=[], gates=[]
        )
        opaque_definitions = [
            self.build_opaque_definition(instruction) for instruction in opaques_to_declare
        ]
        gate_definitions = [
            self.build_gate_definition(instruction) for instruction in gates_to_declare
        ]

        # Early IBM runtime paramterisation uses unbound `Parameter` instances as `input` variables,
        # not the explicit realtime `Var` variables, so we need this explicit scan.
        self.hoist_global_parameter_declarations()
        # Qiskit's clbits and classical registers need to get mapped to implicit OQ3 variables, but
        # only if they're in the top-level circuit.  The QuantumCircuit data model is that inner
        # clbits are bound to outer bits, and inner registers must be closing over outer ones.
        self.hoist_classical_register_declarations()
        # We hoist registers before new-style vars because registers are an older part of the data
        # model (and used implicitly in PrimitivesV2 outputs) so they get the first go at reserving
        # names in the symbol table.
        self.hoist_classical_io_var_declarations()

        # Similarly, QuantumCircuit qubits/registers are only new variables in the global scope.
        quantum_declarations = self.build_quantum_declarations()
        # This call has side-effects - it can populate `self._global_io_declarations` and
        # `self._global_classical_declarations` as a courtesy to the qe-compiler that prefers our
        # hacky temporary `switch` target variables to be globally defined.
        main_statements = self.build_current_scope()

        statements = [
            statement
            for source in (
                # In older versions of the reference OQ3 grammar, IO declarations had to come before
                # anything else, so we keep doing that as a courtesy.
                self._global_io_declarations,
                opaque_definitions,
                gate_definitions,
                self._global_classical_forward_declarations,
                quantum_declarations,
                main_statements,
            )
            for statement in source
        ]
        return ast.Program(header, statements)

    def hoist_declarations(self, instructions, *, opaques, gates):
        """Walks the definitions in gates/instructions to make a list of gates to declare.

        Mutates ``opaques`` and ``gates`` in-place if given, and returns them."""
        for instruction in instructions:
            operation = instruction.operation
            if isinstance(operation, ControlFlowOp):
                for block in operation.blocks:
                    self.hoist_declarations(block.data, opaques=opaques, gates=gates)
                continue
            if operation in self.global_namespace or isinstance(operation, self.builtins):
                continue

            if isinstance(operation, standard_gates.CXGate):
                # CX gets super duper special treatment because it's the base of Terra's definition
                # tree, but isn't an OQ3 built-in.  We use `isinstance` because we haven't fully
                # fixed what the name/class distinction is (there's a test from the original OQ3
                # exporter that tries a naming collision with 'cx').
                self._register_gate(operation)
                gates.append(operation)
            elif operation.definition is None:
                self._register_opaque(operation)
                opaques.append(operation)
            elif not isinstance(operation, Gate):
                raise QASM3ExporterError("Exporting non-unitary instructions is not yet supported.")
            else:
                self.hoist_declarations(operation.definition.data, opaques=opaques, gates=gates)
                self._register_gate(operation)
                gates.append(operation)
        return opaques, gates

    def global_scope(self, assert_=False):
        """Return the global circuit scope that is used as the basis of the full program.  If
        ``assert_=True``, then this raises :obj:`.QASM3ExporterError` if the current context is not
        the global one."""
        if assert_ and len(self._circuit_ctx) != 1 and len(self._circuit_ctx[0]) != 1:
            # Defensive code to help catch logic errors.
            raise QASM3ExporterError(  # pragma: no cover
                f"Not currently in the global context. Current contexts are: {self._circuit_ctx}"
            )
        return self._circuit_ctx[0][0]

    def current_scope(self):
        """Return the current circuit scope."""
        return self._circuit_ctx[-1][-1]

    def current_context(self):
        """Return the current context (list of scopes)."""
        return self._circuit_ctx[-1]

    def push_scope(self, circuit: QuantumCircuit, qubits: Iterable[Qubit], clbits: Iterable[Clbit]):
        """Push a new scope (like a ``for`` or ``while`` loop body) onto the current context
        stack."""
        current_map = self.current_scope().bit_map
        qubits = tuple(current_map[qubit] for qubit in qubits)
        clbits = tuple(current_map[clbit] for clbit in clbits)
        if circuit.num_qubits != len(qubits):
            raise QASM3ExporterError(  # pragma: no cover
                f"Tried to push a scope whose circuit needs {circuit.num_qubits} qubits, but only"
                f" provided {len(qubits)} qubits to create the mapping."
            )
        if circuit.num_clbits != len(clbits):
            raise QASM3ExporterError(  # pragma: no cover
                f"Tried to push a scope whose circuit needs {circuit.num_clbits} clbits, but only"
                f" provided {len(clbits)} clbits to create the mapping."
            )
        mapping = dict(itertools.chain(zip(circuit.qubits, qubits), zip(circuit.clbits, clbits)))
        self.current_context().append(_Scope(circuit, mapping, {}))

    def pop_scope(self) -> _Scope:
        """Pop the current scope (like a ``for`` or ``while`` loop body) off the current context
        stack."""
        if len(self._circuit_ctx[-1]) <= 1:
            raise QASM3ExporterError(  # pragma: no cover
                "Tried to pop a scope from the current context, but there are no current scopes."
            )
        return self._circuit_ctx[-1].pop()

    def push_context(self, outer_context: QuantumCircuit):
        """Push a new context (like for a ``gate`` or ``def`` body) onto the stack."""
        mapping = {bit: bit for bit in itertools.chain(outer_context.qubits, outer_context.clbits)}
        self._circuit_ctx.append([_Scope(outer_context, mapping, {})])

    def pop_context(self):
        """Pop the current context (like for a ``gate`` or ``def`` body) onto the stack."""
        if len(self._circuit_ctx) == 1:
            raise QASM3ExporterError(  # pragma: no cover
                "Tried to pop the current context, but that is the global context."
            )
        if len(self._circuit_ctx[-1]) != 1:
            raise QASM3ExporterError(  # pragma: no cover
                "Tried to pop the current context while there are still"
                f" {len(self._circuit_ctx[-1]) - 1} unclosed scopes."
            )
        self._circuit_ctx.pop()

    def build_includes(self):
        """Builds a list of included files."""
        return [ast.Include(filename) for filename in self.includeslist]

    def build_opaque_definition(self, instruction):
        """Builds an Opaque gate definition as a CalibrationDefinition"""
        # We can't do anything sensible with this yet, so it's better to loudly say that.
        raise QASM3ExporterError(
            "Exporting opaque instructions with pulse-level calibrations is not yet supported by"
            " the OpenQASM 3 exporter. Received this instruction, which appears opaque:"
            f"\n{instruction}"
        )

    def build_gate_definition(self, gate):
        """Builds a QuantumGateDefinition"""
        if isinstance(gate, standard_gates.CXGate):
            # CX gets super duper special treatment because it's the base of Terra's definition
            # tree, but isn't an OQ3 built-in.  We use `isinstance` because we haven't fully
            # fixed what the name/class distinction is (there's a test from the original OQ3
            # exporter that tries a naming collision with 'cx').
            control, target = ast.Identifier("c"), ast.Identifier("t")
            call = ast.QuantumGateCall(
                ast.Identifier("U"),
                [control, target],
                parameters=[ast.Constant.PI, ast.IntegerLiteral(0), ast.Constant.PI],
                modifiers=[ast.QuantumGateModifier(ast.QuantumGateModifierName.CTRL)],
            )
            return ast.QuantumGateDefinition(
                ast.QuantumGateSignature(ast.Identifier("cx"), [control, target]),
                ast.QuantumBlock([call]),
            )

        self.push_context(gate.definition)
        signature = self.build_gate_signature(gate)
        body = ast.QuantumBlock(self.build_current_scope())
        self.pop_context()
        return ast.QuantumGateDefinition(signature, body)

    def build_gate_signature(self, gate):
        """Builds a QuantumGateSignature"""
        name = self.global_namespace[gate]
        params = []
        definition = gate.definition
        # Dummy parameters
        scope = self.current_scope()
        for num in range(len(gate.params) - len(definition.parameters)):
            param_name = f"{self.gate_parameter_prefix}_{num}"
            params.append(self._reserve_variable_name(ast.Identifier(param_name), scope))
        params += [self._register_variable(param, scope) for param in definition.parameters]
        quantum_arguments = [
            self._register_variable(
                qubit, scope, self._unique_name(f"{self.gate_qubit_prefix}_{i}", scope)
            )
            for i, qubit in enumerate(definition.qubits)
        ]
        return ast.QuantumGateSignature(ast.Identifier(name), quantum_arguments, params or None)

    def hoist_global_parameter_declarations(self):
        """Extend ``self._global_io_declarations`` and ``self._global_classical_declarations`` with
        any implicit declarations used to support the early IBM efforts to use :class:`.Parameter`
        as an input variable."""
        global_scope = self.global_scope(assert_=True)
        for parameter in global_scope.circuit.parameters:
            parameter_name = self._register_variable(parameter, global_scope)
            declaration = _infer_variable_declaration(
                global_scope.circuit, parameter, parameter_name
            )
            if declaration is None:
                continue
            if isinstance(declaration, ast.IODeclaration):
                self._global_io_declarations.append(declaration)
            else:
                self._global_classical_forward_declarations.append(declaration)

    def hoist_classical_register_declarations(self):
        """Extend the global classical declarations with AST nodes declaring all the global-scope
        circuit :class:`.Clbit` and :class:`.ClassicalRegister` instances.  Qiskit's data model
        doesn't involve the declaration of *new* bits or registers in inner scopes; only the
        :class:`.expr.Var` mechanism allows that.

        The behaviour of this function depends on the setting ``allow_aliasing``. If this
        is ``True``, then the output will be in the same form as the output of
        :meth:`.build_classical_declarations`, with the registers being aliases.  If ``False``, it
        will instead return a :obj:`.ast.ClassicalDeclaration` for each classical register, and one
        for the loose :obj:`.Clbit` instances, and will raise :obj:`QASM3ExporterError` if any
        registers overlap.
        """
        scope = self.global_scope(assert_=True)
        if any(len(scope.circuit.find_bit(q).registers) > 1 for q in scope.circuit.clbits):
            # There are overlapping registers, so we need to use aliases to emit the structure.
            if not self.allow_aliasing:
                raise QASM3ExporterError(
                    "Some classical registers in this circuit overlap and need aliases to express,"
                    " but 'allow_aliasing' is false."
                )
            clbits = (
                ast.ClassicalDeclaration(
                    ast.BitType(),
                    self._register_variable(
                        clbit, scope, self._unique_name(f"{self.loose_bit_prefix}{i}", scope)
                    ),
                )
                for i, clbit in enumerate(scope.circuit.clbits)
            )
            self._global_classical_forward_declarations.extend(clbits)
            self._global_classical_forward_declarations.extend(
                self.build_aliases(scope.circuit.cregs)
            )
            return
        # If we're here, we're in the clbit happy path where there are no clbits that are in more
        # than one register.  We can output things very naturally.
        self._global_classical_forward_declarations.extend(
            ast.ClassicalDeclaration(
                ast.BitType(),
                self._register_variable(
                    clbit, scope, self._unique_name(f"{self.loose_bit_prefix}{i}", scope)
                ),
            )
            for i, clbit in enumerate(scope.circuit.clbits)
            if not scope.circuit.find_bit(clbit).registers
        )
        for register in scope.circuit.cregs:
            name = self._register_variable(register, scope)
            for i, bit in enumerate(register):
                scope.symbol_map[bit] = ast.SubscriptedIdentifier(
                    name.string, ast.IntegerLiteral(i)
                )
            self._global_classical_forward_declarations.append(
                ast.ClassicalDeclaration(ast.BitArrayType(len(register)), name)
            )

    def hoist_classical_io_var_declarations(self):
        """Hoist the declarations of classical IO :class:`.expr.Var` nodes into the global state.

        Local :class:`.expr.Var` declarations are handled by the regular local-block scope builder,
        and the :class:`.QuantumCircuit` data model ensures that the only time an IO variable can
        occur is in an outermost block."""
        scope = self.global_scope(assert_=True)
        for var in scope.circuit.iter_input_vars():
            self._global_io_declarations.append(
                ast.IODeclaration(
                    ast.IOModifier.INPUT,
                    _build_ast_type(var.type),
                    self._register_variable(var, scope),
                )
            )

    def build_quantum_declarations(self):
        """Return a list of AST nodes declaring all the qubits in the current scope, and all the
        alias declarations for these qubits."""
        scope = self.global_scope(assert_=True)
        if scope.circuit.layout is not None:
            # We're referring to physical qubits.  These can't be declared in OQ3, but we need to
            # track the bit -> expression mapping in our symbol table.
            for i, bit in enumerate(scope.circuit.qubits):
                scope.symbol_map[bit] = ast.Identifier(f"${i}")
            return []
        if any(len(scope.circuit.find_bit(q).registers) > 1 for q in scope.circuit.qubits):
            # There are overlapping registers, so we need to use aliases to emit the structure.
            if not self.allow_aliasing:
                raise QASM3ExporterError(
                    "Some quantum registers in this circuit overlap and need aliases to express,"
                    " but 'allow_aliasing' is false."
                )
            qubits = [
                ast.QuantumDeclaration(
                    self._register_variable(
                        qubit, scope, self._unique_name(f"{self.loose_qubit_prefix}{i}", scope)
                    )
                )
                for i, qubit in enumerate(scope.circuit.qubits)
            ]
            return qubits + self.build_aliases(scope.circuit.qregs)
        # If we're here, we're in the virtual-qubit happy path where there are no qubits that are in
        # more than one register.  We can output things very naturally.
        loose_qubits = [
            ast.QuantumDeclaration(
                self._register_variable(
                    qubit, scope, self._unique_name(f"{self.loose_qubit_prefix}{i}", scope)
                )
            )
            for i, qubit in enumerate(scope.circuit.qubits)
            if not scope.circuit.find_bit(qubit).registers
        ]
        registers = []
        for register in scope.circuit.qregs:
            name = self._register_variable(register, scope)
            for i, bit in enumerate(register):
                scope.symbol_map[bit] = ast.SubscriptedIdentifier(
                    name.string, ast.IntegerLiteral(i)
                )
            registers.append(
                ast.QuantumDeclaration(name, ast.Designator(ast.IntegerLiteral(len(register))))
            )
        return loose_qubits + registers

    def build_aliases(self, registers: Iterable[Register]) -> List[ast.AliasStatement]:
        """Return a list of alias declarations for the given registers.  The registers can be either
        classical or quantum."""
        scope = self.current_scope()
        out = []
        for register in registers:
            name = self._register_variable(register, scope)
            elements = [self._lookup_variable(bit) for bit in register]
            for i, bit in enumerate(register):
                # This might shadow previous definitions, but that's not a problem.
                scope.symbol_map[bit] = ast.SubscriptedIdentifier(
                    name.string, ast.IntegerLiteral(i)
                )
            out.append(ast.AliasStatement(name, ast.IndexSet(elements)))
        return out

    def build_current_scope(self) -> List[ast.Statement]:
        """Build the instructions that occur in the current scope.

        In addition to everything literally in the circuit's ``data`` field, this also includes
        declarations for any local :class:`.expr.Var` nodes.
        """
        scope = self.current_scope()

        # We forward-declare all local variables uninitialised at the top of their scope. It would
        # be nice to declare the variable at the point of first store (so we can write things like
        # `uint[8] a = 12;`), but there's lots of edge-case logic to catch with that around
        # use-before-definition errors in the OQ3 output, for example if the user has side-stepped
        # the `QuantumCircuit` API protection to produce a circuit that uses an uninitialised
        # variable, or the initial write to a variable is within a control-flow scope.  (It would be
        # easier to see the def/use chain needed to do this cleanly if we were using `DAGCircuit`.)
        statements = [
            ast.ClassicalDeclaration(_build_ast_type(var.type), self._register_variable(var, scope))
            for var in scope.circuit.iter_declared_vars()
        ]
        for instruction in scope.circuit.data:
            operation = instruction.operation
            if isinstance(operation, ControlFlowOp):
                if isinstance(operation, ForLoopOp):
                    statements.append(self.build_for_loop(instruction))
                elif isinstance(operation, WhileLoopOp):
                    statements.append(self.build_while_loop(instruction))
                elif isinstance(operation, IfElseOp):
                    statements.append(self.build_if_statement(instruction))
                elif isinstance(operation, SwitchCaseOp):
                    statements.extend(self.build_switch_statement(instruction))
                else:  # pragma: no cover
                    raise RuntimeError(f"unhandled control-flow construct: {operation}")
                continue
            # Build the node, ignoring any condition.
            if isinstance(operation, Gate):
                nodes = [self.build_gate_call(instruction)]
            elif isinstance(operation, Barrier):
                operands = [self._lookup_variable(operand) for operand in instruction.qubits]
                nodes = [ast.QuantumBarrier(operands)]
            elif isinstance(operation, Measure):
                measurement = ast.QuantumMeasurement(
                    [self._lookup_variable(operand) for operand in instruction.qubits]
                )
                qubit = self._lookup_variable(instruction.clbits[0])
                nodes = [ast.QuantumMeasurementAssignment(qubit, measurement)]
            elif isinstance(operation, Reset):
                nodes = [
                    ast.QuantumReset(self._lookup_variable(operand))
                    for operand in instruction.qubits
                ]
            elif isinstance(operation, Delay):
                nodes = [self.build_delay(instruction)]
            elif isinstance(operation, Store):
                nodes = [
                    ast.AssignmentStatement(
                        self.build_expression(operation.lvalue),
                        self.build_expression(operation.rvalue),
                    )
                ]
            elif isinstance(operation, BreakLoopOp):
                nodes = [ast.BreakStatement()]
            elif isinstance(operation, ContinueLoopOp):
                nodes = [ast.ContinueStatement()]
            else:
                nodes = [self.build_subroutine_call(instruction)]

            if operation.condition is None:
                statements.extend(nodes)
            else:
                body = ast.ProgramBlock(nodes)
                statements.append(
                    ast.BranchingStatement(
                        self.build_expression(_lift_condition(operation.condition)),
                        body,
                    )
                )
        return statements

    def build_if_statement(self, instruction: CircuitInstruction) -> ast.BranchingStatement:
        """Build an :obj:`.IfElseOp` into a :obj:`.ast.BranchingStatement`."""
        operation = instruction.operation
        condition = self.build_expression(_lift_condition(operation.condition))

        true_circuit = operation.blocks[0]
        self.push_scope(true_circuit, instruction.qubits, instruction.clbits)
        true_body = ast.ProgramBlock(self.build_current_scope())
        self.pop_scope()
        if len(operation.blocks) == 1:
            return ast.BranchingStatement(condition, true_body, None)

        false_circuit = operation.blocks[1]
        self.push_scope(false_circuit, instruction.qubits, instruction.clbits)
        false_body = ast.ProgramBlock(self.build_current_scope())
        self.pop_scope()
        return ast.BranchingStatement(condition, true_body, false_body)

    def build_switch_statement(self, instruction: CircuitInstruction) -> Iterable[ast.Statement]:
        """Build a :obj:`.SwitchCaseOp` into a :class:`.ast.SwitchStatement`."""
        operation = instruction.operation
        real_target = self.build_expression(expr.lift(operation.target))
        global_scope = self.global_scope()
        target = self._reserve_variable_name(
            ast.Identifier(self._unique_name("switch_dummy", global_scope)), global_scope
        )
        self._global_classical_forward_declarations.append(
            ast.ClassicalDeclaration(ast.IntType(), target, None)
        )

        if ExperimentalFeatures.SWITCH_CASE_V1 in self.experimental:
            # In this case, defaults can be folded in with other cases (useless as that is).

            def case(values, case_block):
                values = [
                    ast.DefaultCase() if v is CASE_DEFAULT else self.build_integer(v)
                    for v in values
                ]
                self.push_scope(case_block, instruction.qubits, instruction.clbits)
                case_body = ast.ProgramBlock(self.build_current_scope())
                self.pop_scope()
                return values, case_body

            return [
                ast.AssignmentStatement(target, real_target),
                ast.SwitchStatementPreview(
                    target,
                    (case(values, block) for values, block in operation.cases_specifier()),
                ),
            ]

        # Handle the stabilised syntax.
        cases = []
        default = None
        for values, block in operation.cases_specifier():
            self.push_scope(block, instruction.qubits, instruction.clbits)
            case_body = ast.ProgramBlock(self.build_current_scope())
            self.pop_scope()
            if CASE_DEFAULT in values:
                # Even if it's mixed in with other cases, we can skip them and only output the
                # `default` since that's valid and execution will be the same; the evaluation of
                # case labels can't have side effects.
                default = case_body
                continue
            cases.append(([self.build_integer(value) for value in values], case_body))

        return [
            ast.AssignmentStatement(target, real_target),
            ast.SwitchStatement(target, cases, default=default),
        ]

    def build_while_loop(self, instruction: CircuitInstruction) -> ast.WhileLoopStatement:
        """Build a :obj:`.WhileLoopOp` into a :obj:`.ast.WhileLoopStatement`."""
        operation = instruction.operation
        condition = self.build_expression(_lift_condition(operation.condition))
        loop_circuit = operation.blocks[0]
        self.push_scope(loop_circuit, instruction.qubits, instruction.clbits)
        loop_body = ast.ProgramBlock(self.build_current_scope())
        self.pop_scope()
        return ast.WhileLoopStatement(condition, loop_body)

    def build_for_loop(self, instruction: CircuitInstruction) -> ast.ForLoopStatement:
        """Build a :obj:`.ForLoopOp` into a :obj:`.ast.ForLoopStatement`."""
        operation = instruction.operation
        indexset, loop_parameter, loop_circuit = operation.params
        self.push_scope(loop_circuit, instruction.qubits, instruction.clbits)
        scope = self.current_scope()
        if loop_parameter is None:
            # The loop parameter is implicitly declared by the ``for`` loop (see also
            # _infer_parameter_declaration), so it doesn't matter that we haven't declared this.
            loop_parameter_ast = self._reserve_variable_name(ast.Identifier("_"), scope)
        else:
            loop_parameter_ast = self._register_variable(loop_parameter, scope)
        if isinstance(indexset, range):
            # OpenQASM 3 uses inclusive ranges on both ends, unlike Python.
            indexset_ast = ast.Range(
                start=self.build_integer(indexset.start),
                end=self.build_integer(indexset.stop - 1),
                step=self.build_integer(indexset.step) if indexset.step != 1 else None,
            )
        else:
            try:
                indexset_ast = ast.IndexSet([self.build_integer(value) for value in indexset])
            except QASM3ExporterError:
                raise QASM3ExporterError(
                    "The values in OpenQASM 3 'for' loops must all be integers, but received"
                    f" '{indexset}'."
                ) from None
        body_ast = ast.ProgramBlock(self.build_current_scope())
        self.pop_scope()
        return ast.ForLoopStatement(indexset_ast, loop_parameter_ast, body_ast)

    def build_expression(self, node: expr.Expr) -> ast.Expression:
        """Build an expression."""
        return node.accept(_ExprBuilder(self._lookup_variable))

    def build_delay(self, instruction: CircuitInstruction) -> ast.QuantumDelay:
        """Build a built-in delay statement."""
        if instruction.clbits:
            raise QASM3ExporterError(
                f"Found a delay instruction acting on classical bits: {instruction}"
            )
        duration_value, unit = instruction.operation.duration, instruction.operation.unit
        if unit == "ps":
            duration = ast.DurationLiteral(1000 * duration_value, ast.DurationUnit.NANOSECOND)
        else:
            unit_map = {
                "ns": ast.DurationUnit.NANOSECOND,
                "us": ast.DurationUnit.MICROSECOND,
                "ms": ast.DurationUnit.MILLISECOND,
                "s": ast.DurationUnit.SECOND,
                "dt": ast.DurationUnit.SAMPLE,
            }
            duration = ast.DurationLiteral(duration_value, unit_map[unit])
        return ast.QuantumDelay(
            duration, [self._lookup_variable(qubit) for qubit in instruction.qubits]
        )

    def build_integer(self, value) -> ast.IntegerLiteral:
        """Build an integer literal, raising a :obj:`.QASM3ExporterError` if the input is not
        actually an
        integer."""
        if not isinstance(value, numbers.Integral):
            # This is meant to be purely defensive, in case a non-integer slips into the logic
            # somewhere, but no valid Terra object should trigger this.
            raise QASM3ExporterError(f"'{value}' is not an integer")  # pragma: no cover
        return ast.IntegerLiteral(int(value))

    def _rebind_scoped_parameters(self, expression):
        """If the input is a :class:`.ParameterExpression`, rebind any internal
        :class:`.Parameter`\\ s so that their names match their names in the scope.  Other inputs
        are returned unchanged."""
        # This is a little hacky, but the entirety of the Expression handling is essentially
        # missing, pending a new system in Terra to replace it (2022-03-07).
        if not isinstance(expression, ParameterExpression):
            return expression
        return expression.subs(
            {
                param: Parameter(self._lookup_variable(param).string)
                for param in expression.parameters
            }
        )

    def build_gate_call(self, instruction: CircuitInstruction):
        """Builds a QuantumGateCall"""
        operation = instruction.operation
        if isinstance(operation, standard_gates.UGate):
            gate_name = ast.Identifier("U")
        else:
            gate_name = ast.Identifier(self.global_namespace[operation])
        qubits = [self._lookup_variable(qubit) for qubit in instruction.qubits]
        if self.disable_constants:
            parameters = [
                ast.StringifyAndPray(self._rebind_scoped_parameters(param))
                for param in operation.params
            ]
        else:
            parameters = [
                ast.StringifyAndPray(pi_check(self._rebind_scoped_parameters(param), output="qasm"))
                for param in operation.params
            ]

        return ast.QuantumGateCall(gate_name, qubits, parameters=parameters)


def _infer_variable_declaration(
    circuit: QuantumCircuit, parameter: Parameter, parameter_name: ast.Identifier
) -> Union[ast.ClassicalDeclaration, None]:
    """Attempt to infer what type a parameter should be declared as to work with a circuit.

    This is very simplistic; it assumes all parameters are real numbers that need to be input to the
    program, unless one is used as a loop variable, in which case it shouldn't be declared at all,
    because the ``for`` loop declares it implicitly (per the Qiskit/qe-compiler reading of the
    OpenQASM spec at openqasm/openqasm@8ee55ec).

    .. note::

        This is a hack around not having a proper type system implemented in Terra, and really this
        whole function should be removed in favour of proper symbol-table building and lookups.
        This function is purely to try and hack the parameters for ``for`` loops into the exporter
        for now.

    Args:
        circuit: The global-scope circuit, which is the base of the exported program.
        parameter: The parameter to infer the type of.
        parameter_name: The name of the parameter to use in the declaration.

    Returns:
        A suitable :obj:`.ast.ClassicalDeclaration` node, or, if the parameter should *not* be
        declared, then ``None``.
    """

    def is_loop_variable(circuit, parameter):
        """Recurse into the instructions a parameter is used in, checking at every level if it is
        used as the loop variable of a ``for`` loop."""
        # This private access is hacky, and shouldn't need to happen; the type of a parameter
        # _should_ be an intrinsic part of the parameter, or somewhere publicly accessible, but
        # Terra doesn't have those concepts yet.  We can only try and guess at the type by looking
        # at all the places it's used in the circuit.
        for instruction, index in circuit._parameter_table[parameter]:
            if isinstance(instruction, ForLoopOp):
                # The parameters of ForLoopOp are (indexset, loop_parameter, body).
                if index == 1:
                    return True
            if isinstance(instruction, ControlFlowOp):
                if is_loop_variable(instruction.params[index], parameter):
                    return True
        return False

    if is_loop_variable(circuit, parameter):
        return None
    # Arbitrary choice of double-precision float for all other parameters, but it's what we actually
    # expect people to be binding to their Parameters right now.
    return ast.IODeclaration(ast.IOModifier.INPUT, ast.FloatType.DOUBLE, parameter_name)


def _lift_condition(condition):
    if isinstance(condition, expr.Expr):
        return condition
    return expr.lift_legacy_condition(condition)


def _build_ast_type(type_: types.Type) -> ast.ClassicalType:
    if type_.kind is types.Bool:
        return ast.BoolType()
    if type_.kind is types.Uint:
        return ast.UintType(type_.width)
    raise RuntimeError(f"unhandled expr type '{type_}'")  # pragma: no cover


class _ExprBuilder(expr.ExprVisitor[ast.Expression]):
    __slots__ = ("lookup",)

    # This is a very simple, non-contextual converter.  As the type system expands, we may well end
    # up with some places where Terra's abstract type system needs to be lowered to OQ3 rather than
    # mapping 100% directly, which might need a more contextual visitor.

    def __init__(self, lookup):
        self.lookup = lookup

    def visit_var(self, node, /):
        return self.lookup(node) if node.standalone else self.lookup(node.var)

    def visit_value(self, node, /):
        if node.type.kind is types.Bool:
            return ast.BooleanLiteral(node.value)
        if node.type.kind is types.Uint:
            return ast.IntegerLiteral(node.value)
        raise RuntimeError(f"unhandled Value type '{node}'")

    def visit_cast(self, node, /):
        if node.implicit:
            return node.operand.accept(self)
        return ast.Cast(_build_ast_type(node.type), node.operand.accept(self))

    def visit_unary(self, node, /):
        return ast.Unary(ast.Unary.Op[node.op.name], node.operand.accept(self))

    def visit_binary(self, node, /):
        return ast.Binary(
            ast.Binary.Op[node.op.name], node.left.accept(self), node.right.accept(self)
        )

    def visit_index(self, node, /):
        return ast.Index(node.target.accept(self), node.index.accept(self))

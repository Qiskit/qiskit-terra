# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
ParameterExpression Class to enable creating simple expressions of Parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Union

import numbers
import operator

import numpy
import symengine

from qiskit.circuit.exceptions import CircuitError

# This type is redefined at the bottom to insert the full reference to "ParameterExpression", so it
# can safely be used by runtime type-checkers like Sphinx.  Mypy does not need this because it
# handles the references by static analysis.
ParameterValueType = Union["ParameterExpression", float]


class _OPCode(IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    POW = 4
    SIN = 5
    COS = 6
    TAN = 7
    ASIN = 8
    ACOS = 9
    EXP = 10
    LOG = 11
    SIGN = 12
    GRAD = 13
    CONJ = 14
    SUBSTITUTE = 15
    ABS = 16
    ATAN = 17
    RSUB = 18
    RDIV = 19
    RPOW = 20


_OP_CODE_MAP = (
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__pow__",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "exp",
    "log",
    "sign",
    "gradient",
    "conjugate",
    "subs",
    "abs",
    "arctan",
    "__rsub__",
    "__rtruediv__",
    "__rpow__",
)


def op_code_to_method(op_code: _OPCode):
    """Return the method name for a given op_code."""
    return _OP_CODE_MAP[op_code]


@dataclass
class _INSTRUCTION:
    op: _OPCode
    lhs: ParameterValueType | None
    rhs: ParameterValueType | None = None


@dataclass
class _SUBS:
    binds: dict
    op: _OPCode = _OPCode.SUBSTITUTE


class ParameterExpression:
    """ParameterExpression class to enable creating expressions of Parameters."""

    __slots__ = [
        "_parameter_symbols",
        "_parameter_keys",
        "_symbol_expr",
        "_name_map",
        "_qpy_replay",
        "_standalone_param",
    ]

    def __init__(self, symbol_map: dict, expr, *, _qpy_replay=None):
        """Create a new :class:`ParameterExpression`.

        Not intended to be called directly, but to be instantiated via operations
        on other :class:`Parameter` or :class:`ParameterExpression` objects.

        Args:
            symbol_map (Dict[Parameter, [ParameterExpression, float, or int]]):
                Mapping of :class:`Parameter` instances to the :class:`sympy.Symbol`
                serving as their placeholder in expr.
            expr (sympy.Expr): Expression of :class:`sympy.Symbol` s.
        """
        # NOTE: `Parameter.__init__` does not call up to this method, since this method is dependent
        # on `Parameter` instances already being initialized enough to be hashable.  If changing
        # this method, check that `Parameter.__init__` and `__setstate__` are still valid.
        self._parameter_symbols = symbol_map
        self._parameter_keys = frozenset(p._hash_key() for p in self._parameter_symbols)
        self._symbol_expr = expr
        self._name_map: dict | None = None
        self._standalone_param = False
        if _qpy_replay is not None:
            self._qpy_replay = _qpy_replay
        else:
            self._qpy_replay = []

    @property
    def parameters(self) -> set:
        """Returns a set of the unbound Parameters in the expression."""
        return self._parameter_symbols.keys()

    @property
    def _names(self) -> dict:
        """Returns a mapping of parameter names to Parameters in the expression."""
        if self._name_map is None:
            self._name_map = {p.name: p for p in self._parameter_symbols}
        return self._name_map

    def conjugate(self) -> "ParameterExpression":
        """Return the conjugate."""
        if self._standalone_param:
            new_op = _INSTRUCTION(_OPCode.CONJ, self)
        else:
            new_op = _INSTRUCTION(_OPCode.CONJ, None)
        new_replay = self._qpy_replay.copy()
        new_replay.append(new_op)
        conjugated = ParameterExpression(
            self._parameter_symbols, symengine.conjugate(self._symbol_expr), _qpy_replay=new_replay
        )
        return conjugated

    def assign(self, parameter, value: ParameterValueType) -> "ParameterExpression":
        """
        Assign one parameter to a value, which can either be numeric or another parameter
        expression.

        Args:
            parameter (Parameter): A parameter in this expression whose value will be updated.
            value: The new value to bind to.

        Returns:
            A new expression parameterized by any parameters which were not bound by assignment.
        """
        if isinstance(value, ParameterExpression):
            return self.subs({parameter: value})
        return self.bind({parameter: value})

    def bind(
        self, parameter_values: dict, allow_unknown_parameters: bool = False
    ) -> "ParameterExpression":
        """Binds the provided set of parameters to their corresponding values.

        Args:
            parameter_values: Mapping of Parameter instances to the numeric value to which
                              they will be bound.
            allow_unknown_parameters: If ``False``, raises an error if ``parameter_values``
                contains Parameters in the keys outside those present in the expression.
                If ``True``, any such parameters are simply ignored.

        Raises:
            CircuitError:
                - If parameter_values contains Parameters outside those in self.
                - If a non-numeric value is passed in parameter_values.
            ZeroDivisionError:
                - If binding the provided values requires division by zero.

        Returns:
            A new expression parameterized by any parameters which were not bound by
            parameter_values.
        """
        if not allow_unknown_parameters:
            self._raise_if_passed_unknown_parameters(parameter_values.keys())
        self._raise_if_passed_nan(parameter_values)

        new_op = _SUBS(parameter_values)
        symbol_values = {}
        for parameter, value in parameter_values.items():
            if (param_expr := self._parameter_symbols.get(parameter)) is not None:
                symbol_values[param_expr] = value

        bound_symbol_expr = self._symbol_expr.subs(symbol_values)

        # Don't use sympy.free_symbols to count remaining parameters here.
        # sympy will in some cases reduce the expression and remove even
        # unbound symbols.
        # e.g. (sympy.Symbol('s') * 0).free_symbols == set()

        free_parameters = self.parameters - parameter_values.keys()
        free_parameter_symbols = {
            p: s for p, s in self._parameter_symbols.items() if p in free_parameters
        }

        if (
            hasattr(bound_symbol_expr, "is_infinite") and bound_symbol_expr.is_infinite
        ) or bound_symbol_expr == float("inf"):
            raise ZeroDivisionError(
                "Binding provided for expression "
                "results in division by zero "
                f"(Expression: {self}, Bindings: {parameter_values})."
            )

        new_replay = self._qpy_replay.copy()
        new_replay.append(new_op)

        return ParameterExpression(
            free_parameter_symbols, bound_symbol_expr, _qpy_replay=new_replay
        )

    def subs(
        self, parameter_map: dict, allow_unknown_parameters: bool = False
    ) -> "ParameterExpression":
        """Returns a new Expression with replacement Parameters.

        Args:
            parameter_map: Mapping from Parameters in self to the ParameterExpression
                           instances with which they should be replaced.
            allow_unknown_parameters: If ``False``, raises an error if ``parameter_map``
                contains Parameters in the keys outside those present in the expression.
                If ``True``, any such parameters are simply ignored.

        Raises:
            CircuitError:
                - If parameter_map contains Parameters outside those in self.
                - If the replacement Parameters in parameter_map would result in
                  a name conflict in the generated expression.

        Returns:
            A new expression with the specified parameters replaced.
        """
        if not allow_unknown_parameters:
            self._raise_if_passed_unknown_parameters(parameter_map.keys())

        inbound_names = {
            p.name: p
            for replacement_expr in parameter_map.values()
            for p in replacement_expr.parameters
        }
        self._raise_if_parameter_names_conflict(inbound_names, parameter_map.keys())
        new_op = _SUBS(parameter_map)

        # Include existing parameters in self not set to be replaced.
        new_parameter_symbols = {
            p: s for p, s in self._parameter_symbols.items() if p not in parameter_map
        }
        symbol_type = symengine.Symbol

        # If new_param is an expr, we'll need to construct a matching sympy expr
        # but with our sympy symbols instead of theirs.
        symbol_map = {}
        for old_param, new_param in parameter_map.items():
            if (old_symbol := self._parameter_symbols.get(old_param)) is not None:
                symbol_map[old_symbol] = new_param._symbol_expr
                for p in new_param.parameters:
                    new_parameter_symbols[p] = symbol_type(p.name)

        substituted_symbol_expr = self._symbol_expr.subs(symbol_map)
        new_replay = self._qpy_replay.copy()
        new_replay.append(new_op)

        return ParameterExpression(
            new_parameter_symbols, substituted_symbol_expr, _qpy_replay=new_replay
        )

    def _raise_if_passed_unknown_parameters(self, parameters):
        unknown_parameters = parameters - self.parameters
        if unknown_parameters:
            raise CircuitError(
                f"Cannot bind Parameters ({[str(p) for p in unknown_parameters]}) not present in "
                "expression."
            )

    def _raise_if_passed_nan(self, parameter_values):
        nan_parameter_values = {
            p: v for p, v in parameter_values.items() if not isinstance(v, numbers.Number)
        }
        if nan_parameter_values:
            raise CircuitError(
                f"Expression cannot bind non-numeric values ({nan_parameter_values})"
            )

    def _raise_if_parameter_names_conflict(self, inbound_parameters, outbound_parameters=None):
        if outbound_parameters is None:
            outbound_parameters = set()
            outbound_names = {}
        else:
            outbound_names = {p.name: p for p in outbound_parameters}

        inbound_names = inbound_parameters
        conflicting_names = []
        for name, param in inbound_names.items():
            if name in self._names and name not in outbound_names:
                if param != self._names[name]:
                    conflicting_names.append(name)
        if conflicting_names:
            raise CircuitError(
                f"Name conflict applying operation for parameters: {conflicting_names}"
            )

    def _apply_operation(
        self,
        operation: Callable,
        other: ParameterValueType,
        reflected: bool = False,
        op_code: _OPCode = None,
    ) -> "ParameterExpression":
        """Base method implementing math operations between Parameters and
        either a constant or a second ParameterExpression.

        Args:
            operation: An operator, such as add, sub, mul, and truediv.
            other: The second argument to be used with self in operation.
            reflected: Optional - The default ordering is "self operator other".
                       If reflected is True, this is switched to "other operator self".
                       For use in e.g. __radd__, ...

        Raises:
            CircuitError:
                - If parameter_map contains Parameters outside those in self.
                - If the replacement Parameters in parameter_map would result in
                  a name conflict in the generated expression.

        Returns:
            A new expression describing the result of the operation.
        """
        self_expr = self._symbol_expr
        if isinstance(other, ParameterExpression):
            self._raise_if_parameter_names_conflict(other._names)
            parameter_symbols = {**self._parameter_symbols, **other._parameter_symbols}
            other_expr = other._symbol_expr
        elif isinstance(other, numbers.Number) and numpy.isfinite(other):
            parameter_symbols = self._parameter_symbols.copy()
            other_expr = other
        else:
            return NotImplemented

        if reflected:
            expr = operation(other_expr, self_expr)
            if op_code in {_OPCode.RSUB, _OPCode.RDIV, _OPCode.RPOW}:
                if self._standalone_param:
                    new_op = _INSTRUCTION(op_code, self, other)
                else:
                    new_op = _INSTRUCTION(op_code, None, other)
            else:
                if self._standalone_param:
                    new_op = _INSTRUCTION(op_code, other, self)
                else:
                    new_op = _INSTRUCTION(op_code, other, None)
        else:
            expr = operation(self_expr, other_expr)
            if self._standalone_param:
                new_op = _INSTRUCTION(op_code, self, other)
            else:
                new_op = _INSTRUCTION(op_code, None, other)
        new_replay = self._qpy_replay.copy()
        new_replay.append(new_op)

        out_expr = ParameterExpression(parameter_symbols, expr, _qpy_replay=new_replay)
        out_expr._name_map = self._names.copy()
        if isinstance(other, ParameterExpression):
            out_expr._names.update(other._names.copy())

        return out_expr

    def gradient(self, param) -> Union["ParameterExpression", complex]:
        """Get the derivative of a parameter expression w.r.t. a specified parameter expression.

        Args:
            param (Parameter): Parameter w.r.t. which we want to take the derivative

        Returns:
            ParameterExpression representing the gradient of param_expr w.r.t. param
            or complex or float number
        """
        # Check if the parameter is contained in the parameter expression
        if param not in self._parameter_symbols.keys():
            # If it is not contained then return 0
            return 0.0

        if self._standalone_param:
            new_op = _INSTRUCTION(_OPCode.GRAD, self, param)
        else:
            new_op = _INSTRUCTION(_OPCode.GRAD, None, param)
        qpy_replay = self._qpy_replay.copy()
        qpy_replay.append(new_op)

        # Compute the gradient of the parameter expression w.r.t. param
        key = self._parameter_symbols[param]
        expr_grad = symengine.Derivative(self._symbol_expr, key)

        # generate the new dictionary of symbols
        # this needs to be done since in the derivative some symbols might disappear (e.g.
        # when deriving linear expression)
        parameter_symbols = {}
        for parameter, symbol in self._parameter_symbols.items():
            if symbol in expr_grad.free_symbols:
                parameter_symbols[parameter] = symbol
        # If the gradient corresponds to a parameter expression then return the new expression.
        if len(parameter_symbols) > 0:
            return ParameterExpression(parameter_symbols, expr=expr_grad, _qpy_replay=qpy_replay)
        # If no free symbols left, return a complex or float gradient
        expr_grad_cplx = complex(expr_grad)
        if expr_grad_cplx.imag != 0:
            return expr_grad_cplx
        else:
            return float(expr_grad)

    def __add__(self, other):
        return self._apply_operation(operator.add, other, op_code=_OPCode.ADD)

    def __radd__(self, other):
        return self._apply_operation(operator.add, other, reflected=True, op_code=_OPCode.ADD)

    def __sub__(self, other):
        return self._apply_operation(operator.sub, other, op_code=_OPCode.SUB)

    def __rsub__(self, other):
        return self._apply_operation(operator.sub, other, reflected=True, op_code=_OPCode.RSUB)

    def __mul__(self, other):
        return self._apply_operation(operator.mul, other, op_code=_OPCode.MUL)

    def __pos__(self):
        return self._apply_operation(operator.mul, 1, op_code=_OPCode.MUL)

    def __neg__(self):
        return self._apply_operation(operator.mul, -1, op_code=_OPCode.MUL)

    def __rmul__(self, other):
        return self._apply_operation(operator.mul, other, reflected=True, op_code=_OPCode.MUL)

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError("Division of a ParameterExpression by zero.")
        return self._apply_operation(operator.truediv, other, op_code=_OPCode.DIV)

    def __rtruediv__(self, other):
        return self._apply_operation(operator.truediv, other, reflected=True, op_code=_OPCode.RDIV)

    def __pow__(self, other):
        return self._apply_operation(pow, other, op_code=_OPCode.POW)

    def __rpow__(self, other):
        return self._apply_operation(pow, other, reflected=True, op_code=_OPCode.RPOW)

    def _call(self, ufunc, op_code):
        if self._standalone_param:
            new_op = _INSTRUCTION(op_code, self)
        else:
            new_op = _INSTRUCTION(op_code, None)
        new_replay = self._qpy_replay.copy()
        new_replay.append(new_op)
        return ParameterExpression(
            self._parameter_symbols, ufunc(self._symbol_expr), _qpy_replay=new_replay
        )

    def sin(self):
        """Sine of a ParameterExpression"""
        return self._call(symengine.sin, op_code=_OPCode.SIN)

    def cos(self):
        """Cosine of a ParameterExpression"""
        return self._call(symengine.cos, op_code=_OPCode.COS)

    def tan(self):
        """Tangent of a ParameterExpression"""
        return self._call(symengine.tan, op_code=_OPCode.TAN)

    def arcsin(self):
        """Arcsin of a ParameterExpression"""
        return self._call(symengine.asin, op_code=_OPCode.ASIN)

    def arccos(self):
        """Arccos of a ParameterExpression"""
        return self._call(symengine.acos, op_code=_OPCode.ACOS)

    def arctan(self):
        """Arctan of a ParameterExpression"""
        return self._call(symengine.atan, op_code=_OPCode.ATAN)

    def exp(self):
        """Exponential of a ParameterExpression"""
        return self._call(symengine.exp, op_code=_OPCode.EXP)

    def log(self):
        """Logarithm of a ParameterExpression"""
        return self._call(symengine.log, op_code=_OPCode.LOG)

    def sign(self):
        """Sign of a ParameterExpression"""
        return self._call(symengine.sign, op_code=_OPCode.SIGN)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"

    def __str__(self):
        from sympy import sympify, sstr

        return sstr(sympify(self._symbol_expr), full_prec=False)

    def __complex__(self):
        try:
            return complex(self._symbol_expr)
        # TypeError is for sympy, RuntimeError for symengine
        except (TypeError, RuntimeError) as exc:
            if self.parameters:
                raise TypeError(
                    f"ParameterExpression with unbound parameters ({self.parameters}) "
                    "cannot be cast to a complex."
                ) from None
            raise TypeError("could not cast expression to complex") from exc

    def __float__(self):
        try:
            return float(self._symbol_expr)
        # TypeError is for sympy, RuntimeError for symengine
        except (TypeError, RuntimeError) as exc:
            if self.parameters:
                raise TypeError(
                    f"ParameterExpression with unbound parameters ({self.parameters}) "
                    "cannot be cast to a float."
                ) from None
            # In symengine, if an expression was complex at any time, its type is likely to have
            # stayed "complex" even when the imaginary part symbolically (i.e. exactly)
            # cancelled out.  Sympy tends to more aggressively recognize these as symbolically
            # real.  This second attempt at a cast is a way of unifying the behavior to the
            # more expected form for our users.
            cval = complex(self)
            if cval.imag == 0.0:
                return cval.real
            raise TypeError("could not cast expression to float") from exc

    def __int__(self):
        try:
            return int(self._symbol_expr)
        # TypeError is for backwards compatibility, RuntimeError is raised by symengine
        except RuntimeError as exc:
            if self.parameters:
                raise TypeError(
                    f"ParameterExpression with unbound parameters ({self.parameters}) "
                    "cannot be cast to an int."
                ) from None
            raise TypeError("could not cast expression to int") from exc

    def __hash__(self):
        if not self._parameter_symbols:
            # For fully bound expressions, fall back to the underlying value
            return hash(self.numeric())
        return hash((self._parameter_keys, self._symbol_expr))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self

    def __abs__(self):
        """Absolute of a ParameterExpression"""
        return self._call(symengine.Abs, _OPCode.ABS)

    def abs(self):
        """Absolute of a ParameterExpression"""
        return self.__abs__()

    def __eq__(self, other):
        """Check if this parameter expression is equal to another parameter expression
           or a fixed value (only if this is a bound expression).
        Args:
            other (ParameterExpression or a number):
                Parameter expression or numeric constant used for comparison
        Returns:
            bool: result of the comparison
        """
        if isinstance(other, ParameterExpression):
            if self.parameters != other.parameters:
                return False
            from sympy import sympify

            return sympify(self._symbol_expr).equals(sympify(other._symbol_expr))
        elif isinstance(other, numbers.Number):
            return len(self.parameters) == 0 and complex(self._symbol_expr) == other
        return False

    def is_real(self):
        """Return whether the expression is real"""
        if not self._symbol_expr.is_real and self._symbol_expr.is_real is not None:
            # Symengine returns false for is_real on the expression if
            # there is a imaginary component (even if that component is 0),
            # but the parameter will evaluate as real. Check that if the
            # expression's is_real attribute returns false that we have a
            # non-zero imaginary
            if self._symbol_expr.imag == 0.0:
                return True
            return False
        return self._symbol_expr.is_real

    def numeric(self) -> int | float | complex:
        """Return a Python number representing this object, using the most restrictive of
        :class:`int`, :class:`float` and :class:`complex` that is valid for this object.

        In general, an :class:`int` is only returned if the expression only involved symbolic
        integers.  If floating-point values were used during the evaluation, the return value will
        be a :class:`float` regardless of whether the represented value is an integer.  This is
        because floating-point values "infect" symbolic computations by their inexact nature, and
        symbolic libraries will use inexact floating-point semantics not exact real-number semantics
        when they are involved.  If you want to assert that all floating-point calculations *were*
        carried out at infinite precision (i.e. :class:`float` could represent every intermediate
        value exactly), you can use :meth:`float.is_integer` to check if the return float represents
        an integer and cast it using :class:`int` if so.  This would be an unusual pattern;
        typically one requires this by only ever using explicitly :class:`~numbers.Rational` objects
        while working with symbolic expressions.

        This is more reliable and performant than using :meth:`is_real` followed by calling
        :class:`float` or :class:`complex`, as in some cases :meth:`is_real` needs to force a
        floating-point evaluation to determine an accurate result to work around bugs in the
        upstream symbolic libraries.

        Returns:
            A Python number representing the object.

        Raises:
            TypeError: if there are unbound parameters.
        """
        if self._parameter_symbols:
            raise TypeError(
                f"Expression with unbound parameters '{self.parameters}' is not numeric"
            )
        if self._symbol_expr.is_integer:
            # Integer evaluation is reliable, as far as we know.
            return int(self._symbol_expr)
        # We've come across several ways in which symengine's general-purpose evaluators
        # introduce spurious imaginary components can get involved in the output.  The most
        # reliable strategy "try it and see" while forcing real floating-point evaluation.
        try:
            real_expr = self._symbol_expr.evalf(real=True)
        except RuntimeError:
            # Symengine returns `complex` if any imaginary floating-point enters at all, even if
            # the result is zero.  The best we can do is detect that and decay to a float.
            out = complex(self._symbol_expr)
            return out.real if out.imag == 0.0 else out
        return float(real_expr)

    def sympify(self):
        """Return symbolic expression as a raw Sympy or Symengine object.

        Symengine is used preferentially; if both are available, the result will always be a
        ``symengine`` object.  Symengine is a separate library but has integration with Sympy.

        .. note::

            This is for interoperability only.  Qiskit will not accept or work with raw Sympy or
            Symegine expressions in its parameters, because they do not contain the tracking
            information used in circuit-parameter binding and assignment.
        """
        return self._symbol_expr


# Redefine the type so external imports get an evaluated reference; Sphinx needs this to understand
# the type hints.
ParameterValueType = Union[ParameterExpression, float]

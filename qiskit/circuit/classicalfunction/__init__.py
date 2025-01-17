# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
====================================================================
ClassicalFunction compiler (:mod:`qiskit.circuit.classicalfunction`)
====================================================================

.. currentmodule:: qiskit.circuit.classicalfunction

Overview
========

The classical function compiler provides the necessary tools to map a classical
potentially irreversible functions into quantum circuits.  Below is a simple example of
how to synthesize a simple boolean function defined using Python into a
QuantumCircuit:

  .. code-block:: python

      from qiskit.circuit.classicalfunction import classical_function
      from qiskit.circuit.classicalfunction.types import Int1

      @classical_function
      def grover_oracle(a: Int1, b: Int1, c: Int1, d: Int1) -> Int1:
          return (not a and b and not c and d)

      quantum_circuit = grover_oracle.synth(registerless=False)
      quantum_circuit.draw('text')

  .. code-block:: text

           a: ──o──
                │
           b: ──■──
                │
           c: ──o──
                │
           d: ──■──
              ┌─┴─┐
      return: ┤ X ├
              └───┘

Following Qiskit's little-endian bit ordering convention, the left-most bit (``a``) is the most
significant bit and the right-most bit (``d``) is the least significant bit.

.. warning::

    The functionality of `qiskit.circuit.classicalfunction` requires `tweedledum`, 
    which isn't available on all platforms (up to Python version 3.11).
    See `tweedledum installation guide 
    <https://github.com/boschmitt/tweedledum/tree/master?tab=readme-ov-file#installation>`_ 
    for more details.

Supplementary Information
=========================

Tweedledum
----------

Tweedledum is a C++-17 header-only library that implements a large set of
reversible (and quantum) synthesis, optimization, and mapping algorithms.
The classical function compiler relies on it and its dependencies to both represent logic
networks and synthesize them into quantum circuits.

ClassicalFunction data types
----------------------------

At the moment, the only type supported by the classical_function compilers is
``qiskit.circuit.classicalfunction.types.Int1``. The classical function function
to parse *must* include type hints (just ``Int1`` for now). The resulting gate
will be a gate in the size of the sum of all the parameters and the return.

The type ``Int1`` means the classical function will only operate at bit level.


ClassicalFunction compiler API
==============================

classical_function
------------------

Decorator for a classical function that returns a `ClassicalFunction` object.

.. autofunction:: classical_function

ClassicalFunction
-----------------

.. autosummary::
   :toctree: ../stubs/

   ClassicalFunction
   BooleanExpression

Exceptions
----------

.. autosummary::
   :toctree: ../stubs/

   ClassicalFunctionCompilerTypeError
   ClassicalFunctionParseError
   ClassicalFunctionCompilerTypeError

"""

from .classicalfunction import ClassicalFunction
from .exceptions import (
    ClassicalFunctionParseError,
    ClassicalFunctionCompilerError,
    ClassicalFunctionCompilerTypeError,
)
from .boolean_expression import BooleanExpression


def classical_function(func):
    """
    Parses and type checks the callable ``func`` to compile it into an ``ClassicalFunction``
    that can be synthesized into a ``QuantumCircuit``.

    Args:
        func (callable): A callable (with type hints) to compile into an ``ClassicalFunction``.

    Returns:
        ClassicalFunction: An object that can synthesis into a QuantumCircuit (via ``synth()``
        method).
    """
    import inspect
    from textwrap import dedent

    source = dedent(inspect.getsource(func))
    return ClassicalFunction(source, name=func.__name__)

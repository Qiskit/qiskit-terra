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

"""
================================
OpenQASM 3 (:mod:`qiskit.qasm3`)
================================

.. currentmodule:: qiskit.qasm3

Qiskit provides some tools for converting between `OpenQASM 3 <https://openqasm.com>`__
representations of quantum programs, and the :class:`.QuantumCircuit` class.  These will continue to
evolve as Qiskit's support for the dynamic-circuit capabilities expressed by OpenQASM 3 increases.


Exporting to OpenQASM 3
=======================

The high-level functions are simply :func:`dump` and :func:`dumps`, which respectively export to a
file (given as a filename) and to a Python string.

.. autofunction:: dump
.. autofunction:: dumps

Both of these exporter functions are single-use wrappers around the main :class:`Exporter` class.
For more complex exporting needs, including dumping multiple circuits in a single session, it may be
more convenient or faster to use the complete interface.

.. autoclass:: Exporter
    :members:

All of these interfaces will raise :exc:`QASM3ExporterError` on failure.

.. autoexception:: QASM3ExporterError


Importing from OpenQASM 3
=========================

Currently only two high-level functions are offered, as Qiskit support for importing from OpenQASM 3
is in its infancy, and the implementation is expected to change significantly.  The two functions
are :func:`load` and :func:`loads`, which are direct counterparts of :func:`dump` and :func:`dumps`,
respectively loading a program indirectly from a named file and directly from a given string.

.. note::

    While we are still in the exploratory release period, to use either function, the package
    ``qiskit_qasm3_import`` must be installed.  This can be done by installing Qiskit Terra with the
    ``qasm3-import`` extra, such as by:

    .. code-block:: text

        pip install qiskit-terra[qasm3-import]

    We expect that this functionality will eventually be merged into core Terra, and no longer
    require an optional import, but we do not yet have a timeline for this.

.. autofunction:: load
.. autofunction:: loads

Both of these two functions raise :exc:`QASM3ImporterError` on failure.

.. autoexception:: QASM3ImporterError

For example, we can define a quantum program using OpenQASM 3, and use :func:`loads` to directly
convert it into a :class:`.QuantumCircuit`:

.. plot::
    :include-source:

    import qiskit.qasm3

    program = \"\"\"
        OPENQASM 3.0;
        include "stdgates.inc";

        input float[64] a;
        qubit[3] q;
        bit[2] mid;
        bit[3] out;

        let aliased = q[0:1];

        gate my_gate(a) c, t {
          gphase(a / 2);
          ry(a) c;
          cx c, t;
        }
        gate my_phase(a) c {
          ctrl @ inv @ gphase(a) c;
        }

        my_gate(a * 2) aliased[0], q[{1, 2}][0];
        measure q[0] -> mid[0];
        measure q[1] -> mid[1];

        while (mid == "00") {
          reset q[0];
          reset q[1];
          my_gate(a) q[0], q[1];
          my_phase(a - pi/2) q[1];
          mid[0] = measure q[0];
          mid[1] = measure q[1];
        }

        if (mid[0]) {
          let inner_alias = q[{0, 1}];
          reset inner_alias;
        }

        out = measure q;
    \"\"\"
    circuit = qiskit.qasm3.loads(program)
    circuit.draw("mpl")
"""

from qiskit.utils import optionals as _optionals
from .exporter import Exporter
from .exceptions import QASM3Error, QASM3ImporterError, QASM3ExporterError


def dumps(circuit, **kwargs) -> str:
    """Serialize a :class:`~qiskit.circuit.QuantumCircuit` object in an OpenQASM3 string.

    Args:
        circuit (QuantumCircuit): Circuit to serialize.
        **kwargs: Arguments for the :obj:`.Exporter` constructor.

    Returns:
        str: The OpenQASM3 serialization
    """
    return Exporter(**kwargs).dumps(circuit)


def dump(circuit, stream, **kwargs) -> None:
    """Serialize a :class:`~qiskit.circuit.QuantumCircuit` object as a OpenQASM3 stream to file-like
    object.

    Args:
        circuit (QuantumCircuit): Circuit to serialize.
        stream (TextIOBase): stream-like object to dump the OpenQASM3 serialization
        **kwargs: Arguments for the :obj:`.Exporter` constructor.

    """
    Exporter(**kwargs).dump(circuit, stream)


@_optionals.HAS_QASM3_IMPORT.require_in_call("loading from OpenQASM 3")
def load(filename: str):
    """Load an OpenQASM 3 program from the file ``filename``.

    Args:
        filename: the filename to load the program from.

    Returns:
        QuantumCircuit: a circuit representation of the OpenQASM 3 program.

    Raises:
        QASM3ImporterError: if the OpenQASM 3 file is invalid, or cannot be represented by a
            :class:`.QuantumCircuit`.
    """

    import qiskit_qasm3_import

    with open(filename, "r") as fptr:
        program = fptr.read()
    try:
        return qiskit_qasm3_import.parse(program)
    except qiskit_qasm3_import.ConversionError as exc:
        raise QASM3ImporterError(f"Error converting file {filename}.") from exc


@_optionals.HAS_QASM3_IMPORT.require_in_call("loading from OpenQASM 3")
def loads(program: str):
    """Load an OpenQASM 3 program from the given string.

    Args:
        program: the OpenQASM 3 program.

    Returns:
        QuantumCircuit: a circuit representation of the OpenQASM 3 program.

    Raises:
        QASM3ImporterError: if the OpenQASM 3 file is invalid, or cannot be represented by a
            :class:`.QuantumCircuit`.
    """

    import qiskit_qasm3_import

    try:
        return qiskit_qasm3_import.parse(program)
    except qiskit_qasm3_import.ConversionError as exc:
        raise QASM3ImporterError("Error converting OpenQASM program.") from exc

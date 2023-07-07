# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
###########################################################
QPY serialization (:mod:`qiskit.qpy`)
###########################################################

.. currentmodule:: qiskit.qpy

QPY is a binary serialization format for :class:`~.QuantumCircuit` and
:class:`~.ScheduleBlock` objects, designed to be cross-platform,
Python version agnostic, and backwards compatible moving forward. QPY should
be used if you need a mechanism to save or copy between systems a
:class:`~.QuantumCircuit` or :class:`~.ScheduleBlock` that preserves the full
Qiskit object structure (except for custom attributes defined outside of
Qiskit code).


This differs from other serialization formats like
`OpenQASM <https://github.com/openqasm/openqasm>`__ (2.0 or 3.0), which has a
different abstraction model and can result in a loss of information contained
in the original circuit (or is unable to represent some aspects of the
Qiskit objects), or Python's `pickle <https://docs.python.org/3/library/pickle.html>`__,
which will preserve the Qiskit object exactly but will only work for a single Qiskit
version (it is also
`potentially insecure <https://docs.python.org/3/library/pickle.html#module-pickle>`__).

*********
Using QPY
*********

Using QPY is straightforward, as it mirrors the user API of the
serializers in Python's standard library, ``pickle`` and ``json``. There are
two user facing functions: :func:`qiskit.qpy.dump` and
:func:`qiskit.qpy.load`. These are used to dump QPY data
to a file object and load circuits from QPY data in a file object respectively.
For example::

    from qiskit.circuit import QuantumCircuit
    from qiskit import qpy

    qc = QuantumCircuit(2, name='Bell', metadata={'test': True})
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    with open('bell.qpy', 'wb') as fd:
        qpy.dump(qc, fd)

    with open('bell.qpy', 'rb') as fd:
        new_qc = qpy.load(fd)[0]

The :func:`qiskit.qpy.dump` function also lets you
include multiple circuits in a single QPY file::

    with open('twenty_bells.qpy', 'wb') as fd:
        qpy.dump([qc] * 20, fd)

and then loading that file will return a list with all the circuits::

    with open('twenty_bells.qpy', 'rb') as fd:
        twenty_new_bells = qpy.load(fd)

API documentation
=================

.. autosummary::
   :toctree: ../stubs/

   load
   dump

QPY Compatibility
=================

The QPY format is designed to be backwards compatible moving forward. This means
you should be able to load a QPY with any newer Qiskit version than the one
that generated it. However, loading a QPY file with an older Qiskit version is
not supported and may not work.

For example, if you generated a QPY file using qiskit-terra 0.18.1 you could
load that QPY file with qiskit-terra 0.19.0 and a hypothetical qiskit-terra
0.29.0. However, loading that QPY file with 0.18.0 is not supported and may not
work.

QPY Version History
-------------------
The following table keeps track of the ``qiskit-terra`` releases associated to each
``qpy`` version release.

================ =================
``qpy`` version  ``qiskit-terra``
================ =================
QPY v.1          0.18.0
QPY v.2          0.19.0
QPY v.3          0.20.0
QPY v.4          0.20.0
QPY v.5          0.21.0
QPY v.6          0.23.0
QPY v.7          0.24.0
================ =================

.. _qpy_format:

**********
QPY Format
**********

The QPY serialization format is a portable, cross-platform, binary
serialization format for :class:`~qiskit.circuit.QuantumCircuit` and
:class:`~.ScheduleBlock` objects in Qiskit. The basic
file format is as follows:

``'QISKIT' | FILE_HEADER | PAYLOAD_HEADER | PAYLOAD_DATA_BLOCKS``

A QPY file (or memory object) always starts with the following 7
byte UTF8 string: ``QISKIT``, immediately followed by the **file header** block.
The file header is followed by the payload header and the payload data blocks.
These can be **circuit payloads** or **block schedule payloads** (introduced in v.5),
as defined by the **payload header** (also introduced in v.5).


FILE_HEADER
===========

The file header determines the number of objects stored in the file, as well as
information on the Qiskit and QPY versions used to serialize them.

The contents of the file header are defined as a C struct:

.. code-block:: c

    struct {
        uint8_t qpy_version;
        uint8_t qiskit_major_version;
        uint8_t qiskit_minor_version;
        uint8_t qiskit_patch_version;
        uint64_t num_circuits; // number of payload blocks
    }

.. note::
    All values use network byte order [#f1]_ (big endian) for cross-platform
    compatibility.

``num_circuits`` defines the number of consecutive payload blocks that will be found in the file,
with no padding in between data blocks.

PAYLOAD_HEADER
==============

From QPY v.5, the payload header struct:

.. code-block:: c

    struct {
        char type;
    }

immediately follows the file header block to represent the object type stored in the file:

- When ``type==c``, :ref:`qpy_circuit` follow
- When ``type==s``, :ref:`qpy_schedule_block` follow

.. note::
    Each payload type has a distinct representation in QPY. For this reason,
    different types of payloads cannot be packed together in the same file.
    You must create different files for different payload types.
    However, multiple objects with the same payload type **can** be saved in a single file.

.. _qpy_circuit:

:class:`~.QuantumCircuit`  Payloads
===================================

The structure of the circuit payloads is as follows:

``CIRCUIT_HEADER | METADATA | REGISTERS | CUSTOM_DEFINITIONS | INSTRUCTIONS | CIRCUIT_CALIBRATIONS``

From QPY v.5, the :ref:`qpy_circuit_calibrations` payload is
packed after the :ref:`qpy_instructions` block to support the serialization of calibration
and pulse gates.

.. _qpy_circuit_header:

CIRCUIT_HEADER
--------------

The contents of the CIRCUIT_HEADER block are:

``HEADER | NAME | GLOBAL_PHASE_DATA``

The HEADER is defined as a C struct:

.. code-block:: c

    struct {
        uint16_t name_size;
        char global_phase_type; // added in v.2, replacing global_phase
        uint16_t global_phase_size; // added in v.2, replacing global_phase
        uint32_t num_qubits;
        uint32_t num_clbits;
        uint64_t metadata_size;
        uint32_t num_registers;
        uint64_t num_instructions;
        uint64_t num_custom_gates;
    }

The HEADER is immediately followed by ``name_size`` bytes of UTF-8 data containing the NAME
of the circuit. Following this are ``global_phase_size`` bytes
representing the GLOBAL_PHASE_DATA.

The type of the data contained in this block is dictated by the
value of ``global_phase_type``. If it's ``'f'``, the data is a float and is the
size of a ``double``. If it's ``'p'``, it's a :class:`~qiskit.circuit.Parameter`
object represented by a :ref:`qpy_param_struct`. Finally, ``'e'`` defines a
:class:`~qiskit.circuit.ParameterExpression` object (note that this is different from
:class:`~qiskit.circuit.Parameter`), which is represented by a :ref:`qpy_param_expr`.

METADATA
--------

The next field to be parsed is the METADATA field, a UTF-8 encoded JSON string of
``metadata_size`` number of bytes as defined by the :ref:`qpy_circuit_header`'s header.

.. _qpy_registers:

REGISTERS
---------

Immediately following the METADATA, are the REGISTERS. This block contains a **series** of
REGISTER blocks of format:

``HEADER | REGISTER_NAME | REGISTER_MAP``

The number of REGISTER blocks is determined by the :ref:`qpy_circuit_header`'s ``num_registers`` field
(if ``num_registers==0``, no registers will be parsed).

As shown in the format, each REGISTER contains a HEADER struct:

.. code-block:: c

    struct {
        char type;
        _Bool standalone;
        int64_t size; // type changed in v.4 from uint32_t
        uint16_t name_size;
        _Bool in_circuit; // added in v.4
    }

Where ``type`` can be ``'q'`` (for quantum register) or ``'c'`` (for classical register).
The ``standalone`` boolean determines whether the register was constructed as a
standalone register or was created from existing bits. A register is considered
standalone if it has bits constructed solely as part of it, for example, in::

        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

the register ``qr`` would be a standalone register. While for something like::

        bits = [Qubit(), Qubit()]
        qr2 = QuantumRegister(bits=bits)
        qc = QuantumCircuit(qr2)

``qr2`` would have ``standalone`` set to ``False``.

Finally, the ``in_circuit`` boolean represents whether the register is part of the circuit or not.

For each block, immediately following the HEADER struct is the UTF-8 encoded REGISTER_NAME of
size ``name_size``. After the REGISTER_NAME , there is an array of
``int64_t`` values of size ``size`` that contains a map of the register's index to
the circuit's qubit index. For example, array element ``0``'s value is the index
of ``register[0]``'s position in the containing circuit's qubit list.

.. note::
    Prior to QPY v.4, the type of ``size`` (determining the type of the mapping array elements)
    was ``uint32_t``. This was changed to enable negative values, which represent register bits in
    the array not present in the circuit (``in_circuit==False``).

.. _qpy_custom_definition:

CUSTOM_DEFINITIONS
------------------

The block after REGISTERS is CUSTOM_DEFINITIONS. This block allows to specify custom definitions for any
of the instructions in the circuit. It contains a HEADER followed
by a series of CUSTOM_INSTRUCTION blocks:

``HEADER | CUSTOM_INSTRUCTIONS``

The HEADER contents are defined as:

.. code-block:: c

    struct {
        uint64_t size;
    }

If ``size`` is greater than 0, the file contains ``size`` CUSTOM_INSTRUCTION blocks.

Each CUSTOM_INSTRUCTION block starts with a HEADER and the instruction's NAME, followed by data
which varies depending on the instruction represented (and defined by the HEADER):

``HEADER | NAME | INSTRUCTION_DATA``

The CUSTOM_INSTRUCTION header is a C struct defined as:

.. code-block:: c

    struct {
        uint16_t name_size;
        char type;
        uint32_t num_qubits;
        uint32_t num_clbits;
        _Bool custom_definition;
        uint64_t size;
        uint32_t num_ctrl_qubits; // added in QPY v.5
        uint32_t ctrl_state; // added in QPY v.5
        uint64_t base_gate_size // added in QPY v.5
    }

Immediately following the header struct is the UTF-8 encoded name
of size ``name_size``. It should be noted that :class:`~qiskit.circuit.library.PauliEvolutionGate`
entries (added in v.3) have a unique name that is generated with the string
``"###PauliEvolutionGate_"`` followed by a ``uuid`` string.
This gate name is reserved in QPY, and cannot be used for other custom
:class:`~qiskit.circuit.Instruction` objects.

If ``custom_definition`` is ``True``, the immediately following
``size`` bytes contain QPY circuit data that can be used for the custom
definition of that gate. If ``custom_definition`` is ``False``, then the
instruction can be considered opaque (i.e. no custom definition).

The ``type`` field determines what type of object will get
created from the custom definition. If it's ``'g'``, it will be a :class:`~qiskit.circuit.Gate`
object; if it's ``'i'`` it will be a :class:`~qiskit.circuit.Instruction` object
(serialized as :ref:`qpy_instructions`). The type value ``'p'`` represents a
:class:`~qiskit.circuit.library.PauliEvolutionGate` (only after v.3), and the
data payload is defined in :ref:`pauli_evo_qpy`.

Additionally, if ``type`` is ``'c'``, the custom instruction will be a custom :class:`~.ControlledGate`,
and ``base_gate_size`` will be used to define the size of the
:class:`qiskit.circuit.Instruction` object stored in the
:attr:`.ControlledGate.base_gate` attribute. In this case, the next ``base_gate_size`` bytes
will contain the ``INSTRUCTION`` payload for the :attr:`.ControlledGate.base_gate`.

.. _pauli_evo_qpy:

PAULI_EVOLUTION
^^^^^^^^^^^^^^^

This block represents the high level :class:`~qiskit.circuit.library.PauliEvolutionGate` data payload:

``HEADER | SPARSE_PAULI_OPS | TIME | SYNTHESIS_DATA``

The ``PAULI_EVOLUTION`` header is a C struct:

.. code-block:: c

    struct {
        uint64_t operator_count;
        _Bool standalone_op;
        char time_type;
        uint64_t time_size;
        uint64_t synthesis_size;
    }

The header is immediately followed by a ``operator_count`` :ref:`qpy_sparse_pauli_op`
elements (SPARSE_PAULI_OPS). If ``standalone_op`` is ``True``, there must be a single operator.

The following block contains ``time_size`` bytes representing the TIME.
The encoding of these bytes is determined by the value of ``time_type``. Possible values of
``time_type`` are ``'f'``, ``'p'``, and ``'e'``. If ``time_type`` is ``'f'``,
TIME is encoded as a double;
``'p'`` defines a :class:`~qiskit.circuit.Parameter` object (represented by a
:ref:`qpy_param_struct`); and ``e`` defines a :class:`~qiskit.circuit.ParameterExpression` object
(careful, do not confuse with :class:`~qiskit.circuit.Parameter`),
represented by a :ref:`qpy_param_expr`.

Finally, the PAULI_EVOLUTION format contains ``synthesis_size`` bytes of a UTF-8 encoded JSON
payload representing the :class:`.EvolutionSynthesis` class used by the gate.

.. _qpy_sparse_pauli_op:

SPARSE_PAULI_OP_LIST_ELEM
^^^^^^^^^^^^^^^^^^^^^^^^^

This block represents an instance of :class:`.SparsePauliOp` with the format:

``HEADER | PAULI_DATA``

The header struct is quite simple:

.. code-block:: c

    struct {
        uint32_t pauli_op_size;
    }

and immediately followed by ``pauli_op_size`` bytes of ``.npy`` format [#f2]_
data that represents the underlying :class:`~qiskit.quantum_info.SparsePauliOp`.


.. _qpy_instructions:

INSTRUCTIONS
------------

Following the CUSTOM_DEFINITIONS, the :ref:`qpy_circuit` format defines the INSTRUCTIONS
block, which contains a list of INSTRUCTION blocks of format:

``HEADER | NAME | LABEL | CONDITIONAL_REG_NAME | INSTRUCTION_ARGS | INSTRUCTION_PARAMS``

The INSTRUCTION header is a struct:

.. code-block:: c

    struct {
        uint16_t name_size;
        uint16_t label_size;
        uint16_t num_parameters;
        uint32_t num_qargs;
        uint32_t num_cargs;
        _Bool has_conditional;
        uint16_t conditional_reg_name_size;
        int64_t conditional_value;
        uint32_t num_ctrl_qubits; // added in QPY v.5
        uint32_t ctrl_state;  // added in QPY v.5
    }

This object is immediately followed by ``name_size`` bytes of UTF-8 bytes
storing the NAME.

If defined in Qiskit, NAME is populated with the Qiskit class name for the
:class:`~qiskit.circuit.Instruction` class. Else, it falls back to the custom
instruction name.

Following the NAME, if there was a label set for the instruction,
there will be ``label_size`` bytes of UTF-8 data encoding the LABEL.
If ``has_conditional`` is ``True``, these are followed by
``conditional_reg_name_size`` bytes of UTF-8 data for the name of the conditional
register name (CONDITIONAL_REG_NAME). In case of single classical bit conditions,
the register name UTF-8 data will be prefixed with a null character ``"\\x00"``, followed
by a UTF-8 string integer representing the classical bit index in the circuit
that the condition is on.

This is immediately followed by a list of INSTRUCTION_ARG that represent the
arguments of that instruction. They contain ``num_qargs`` quantum arguments followed
by ``num_cargs`` classical arguments, as defined in the INSTRUCTION header.

Each INSTRUCTION_ARG is a represented as a struct:

.. code-block:: c

    struct {
        char type;
        int64_t index; //changed in v.4
    }

Where ``type`` can be ``'q'`` or ``'c'``, and ``index`` represents the
register indices where the instruction is applied. If the values of any of
the array elements of ``index`` are negative,
they represent a register bit that is not present in the circuit.

After all INSTRUCTION_ARGS are defined, and if the instruction is
parametrized, then the parameters will be specified with
``num_parameters`` INSTRUCTION_PARAM blocks of format:

``HEADER | PARAM_DATA``

.. _qpy_instruction_param:

The header defines the format of each INSTRUCTION_PARAM:

.. code-block:: c

    struct {
        char type;
        uint64_t size;
    }

After the header, the next ``size`` bytes will contain PARAM_DATA determined by ``type``.
The ``type`` field can be ``'i'``, ``'f'``, ``'p'``, ``'e'``, ``'s'``, ``'c'``, ``'n'``,
``'v'``, ``'z'``, ``'q'``, ``'r'``, ``'t'``.

``'i'`` stands for integer, ``'f'`` for double, ``'s'`` for string
(encoded as UTF-8), ``'c'`` is a complex type (the data is represented
the :ref:`qpy_param_expr` format),
``'p'`` defines a :class:`~qiskit.circuit.Parameter` object
represented by :ref:`qpy_param_struct`, ``'e'`` defines a
:class:`~qiskit.circuit.ParameterExpression` object (different from
:class:`~qiskit.circuit.Parameter`) represented by :ref:`qpy_param_expr`,
``'n'`` represents an object from numpy (``ndarray`` or other numpy type)
in .npy format [#f2]_, and from QPY v.3 onward, ``'v'`` defines a
:class:`~qiskit.circuit.ParameterVectorElement` represented by
:ref:`qpy_param_vector`.

In QPY v.4, additional types were incorporated: ``'z'`` represents ``None``
(which is encoded as no data), ``'q'`` represents a :class:`.QuantumCircuit`
(which is encoded as a QPY circuit), ``'r'`` represents a ``range`` of
integers (which is encoded as a :ref:`qpy_range_pack`), and ``'t'``
represents a ``sequence`` (which is encoded as
defined by :ref:`qpy_sequence`).

QPY v.5 added support for controlled gates, where the INSTRUCTION header's fields
``num_ctrl_qubits`` and ``ctrl_state`` are used to model the
:attr:`.ControlledGate.num_ctrl_qubits` and
:attr:`.ControlledGate.ctrl_state` attributes.

.. _qpy_param_struct:

PARAMETER
^^^^^^^^^

A PARAMETER represents a :class:`~qiskit.circuit.Parameter` object, it's one of the data types for
INSTRUCTION_PARAMs of format:

``HEADER | PARAM_NAME``

The contents of a PARAMETER are defined by a header:

.. code-block:: c

    struct {
        uint16_t name_size;
        char uuid[16];
    }

which is immediately followed by ``name_size`` UTF-8 bytes representing the
parameter name.

.. _qpy_param_expr:

PARAMETER_EXPR
^^^^^^^^^^^^^^

A PARAMETER_EXPR represents a :class:`~qiskit.circuit.ParameterExpression`
object that the data for an INSTRUCTION_PARAM. The format is

``HEADER | EXPR_STR | MAP_ELEMENTS``


The contents of a PARAMETER_EXPR are defined by a header:

.. code-block:: c

    struct {
        uint64_t map_elements;
        uint64_t expr_size;
    }

Immediately following the header are ``expr_size`` bytes of UTF-8 data containing
the expression string EXPR_STR, which is the ``sympy`` string representation (``srepr``)
of the expression. Following that is a symbol map which contains
``map_elements`` MAP_ELEMENTS with the format:

``HEADER | SYMBOL_MAP_KEY | SYMBOL_DATA``

.. code-block:: c

    struct {
        char symbol_type;
        char type;
        uint64_t size;
    }

The header struct is immediately followed by the SYMBOL_MAP_KEY payload, if
``symbol_type`` is ``'p'``, then it is a :ref:`qpy_param_struct`
object, and if ``symbol_type`` is ``'v'``,
the header is followed by a :ref:`qpy_param_vector`.

This is followed by ``size`` bytes for the
SYMBOL_DATA. The data format is dependent on the value of ``type``. If
``type`` is ``'p'``, it represents a :class:`~qiskit.circuit.Parameter` and
size will be 0, the value will just be the same as the key. Similarly ,if the
``type`` is ``'v'``, then it represents a :class:`~qiskit.circuit.ParameterVectorElement`
and size will be 0 (the value will just be the same as the key). If
``type`` is ``'f'``, then it represents a double precision float. If ``type`` is
``'c'``, it represents a double precision complex (:ref:`qpy_complex`).
Finally, if ``type`` is ``'i'``, it represents an integer in ``int64_t`` format.

.. _qpy_param_vector:

PARAMETER_VECTOR_ELEMENT
^^^^^^^^^^^^^^^^^^^^^^^^

A PARAMETER_VECTOR_ELEMENT represents the :class:`~qiskit.circuit.ParameterVectorElement`,
one of the types supported by INSTRUCTION_PARAM. The contents of the PARAMETER_VECTOR_ELEMENT
are defined as:

``HEADER | NAME``

The header is:

.. code-block:: c

    struct {
        uint16_t vector_name_size;
        uint64_t vector_size;
        char uuid[16];
        uint64_t index;
    }

which is immediately followed by ``vector_name_size`` UTF-8 bytes representing
the parameter vector's  NAME.

.. _qpy_complex:

COMPLEX
^^^^^^^

A COMPLEX is a representation of a double precision complex value object. It is defined as:

.. code-block:: c

    struct {
        double real;
        double imag;
    }

this matches the internal C representation of Python's complex type [#f3]_ .

.. _qpy_range_pack:

RANGE
^^^^^

A RANGE is a representation of a ``range`` object. It is defined as:

.. code-block:: c

    struct {
        int64_t start;
        int64_t stop;
        int64_t step;
    }

.. _qpy_sequence:

SEQUENCE
^^^^^^^^

A Python sequence can be serialized using SEQUENCE as long as its
elements are QPY-serializable types
supported within the INSTRUCTION_PARAM block. The format would be as follows:

``HEADER | SEQUENCE_ELEMENTS``

The header is a struct:

.. code-block:: c

    struct {
        uint64_t size;
    }

and is followed by ``size`` SEQUENCE_ELEMENTS represented as INSTRUCTION_PARAM blocks,
where each of these define an element in the sequence.
The sequence object will be type-casted
into the proper type, e.g. ``tuple``, in the deserialization step.

.. _qpy_circuit_calibrations:

CIRCUIT_CALIBRATIONS
--------------------

The final block in :ref:`qpy_circuit` is the CIRCUIT_CALIBRATIONS block,
a dictionary to define pulse calibrations for the custom
instruction set. This block follows:

``HEADER | CALIBRATION_ENTRIES``

Where the header is:

.. code-block:: c

    struct {
        uint16_t num_cals;
    }

which is followed by the ``num_cals`` length of CALIBRATION_ENTRIES.
Each CALIBRATION_ENTRY follows the format:

``CALIBRATION_DEF | NAME | QUBIT_SEQUENCE | INSTRUCTION_PARAMS``

Where the CALIBRATION_DEF header is:

.. code-block:: c

    struct {
        uint16_t name_size;
        uint16_t num_qubits;
        uint16_t num_params;
        char type;
    }

The calibration definition header is then followed by ``name_size`` UTF-8 bytes of
the gate NAME, ``num_qubits`` length of integers representing a sequence of qubits
(QUBIT_SEQUENCE), and ``num_params`` length of INSTRUCTION_PARAM payload for parameters
associated to the custom instruction.

The ``type`` indicates the class of pulse schedule used, which could in principle be
:class:`~.ScheduleBlock` or :class:`~.Schedule`. As of QPY v.5,
only the :ref:`qpy_schedule_block` payload is supported.

.. _qpy_schedule_block:

:class:`.~ScheduleBlock`  Payloads
===================================

QPY v.5 and above support serialization of :class:`.~ScheduleBlock` payloads.
With the support of :class:`.~ScheduleBlock`, :class:`~.QuantumCircuit` can be
serialized together with :attr:`~.QuantumCircuit.calibrations`, or
`Pulse Gates <https://qiskit.org/documentation/tutorials/circuits_advanced/05_pulse_gates.html>`_.

.. code-block:: python

    from qiskit import pulse, qpy

    with pulse.build() as schedule:
        pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))

    with open('schedule.qpy', 'wb') as fd:
        qpy.dump(qc, fd)

    with open('schedule.qpy', 'rb') as fd:
        new_qc = qpy.load(fd)[0]


Note that circuit and schedule block are serialized and deserialized through
the same QPY interface. Input data type is implicitly analyzed and
no extra option is required to save the schedule block.

A :class:`.~ScheduleBlock` payload will follow:

``SCHEDULE_BLOCK_HEADER | METADATA | SCHEDULE_BLOCK_ALIGNMENTS | SCHEDULE_BLOCK_INSTRUCTIONS | MAPPING``

Version 7 adds support for :class:`.~Reference` instruction and serialization of
a :class:`.~ScheduleBlock` program while keeping its reference to subroutines::

    from qiskit import pulse
    from qiskit import qpy

    with pulse.build() as schedule:
        pulse.reference("cr45p", "q0", "q1")
        pulse.reference("x", "q0")
        pulse.reference("cr45p", "q0", "q1")

    with open('template_ecr.qpy', 'wb') as fd:
        qpy.dump(schedule, fd)

From v.7, SCHEDULE_BLOCK_INSTRUCTIONS is immediately followed by an extra
:ref:`qpy_mapping` UTF-8 bytes block representing the data of the referenced subroutines.

.. _qpy_schedule_block_header:

SCHEDULE_BLOCK_HEADER
---------------------

This first block of the schedule payload is composed in turn of:

``HEADER | SCHEDULE_NAME | METADATA``

The header is:

.. code-block:: c

    struct {
        uint16_t name_size;
        uint64_t metadata_size;
        uint16_t num_element;
    }

which is immediately followed by ``name_size`` UTF-8 bytes of SCHEDULE_NAME and
``metadata_size`` UTF-8 bytes of the JSON serialized metadata dictionary
attached to the schedule.


.. _qpy_schedule_alignments:

SCHEDULE_BLOCK_ALIGNMENTS
-------------------------

After :ref:`qpy_schedule_block_header` comes the alignment context of the schedule block.
The format of this element is:

``ALIGNMENT_TYPE | CONTEXT_PARAMS``

SCHEDULE_BLOCK_ALIGNMENTS starts with a ``char``
representing the supported context type.
The context type char is mapped to each alignment subclass as follows:

- ``l``: :class:`~.AlignLeft`
- ``r``: :class:`~.AlignRight`
- ``s``: :class:`~.AlignSequential`
- ``e``: :class:`~.AlignEquispaced`

This is followed by a :ref:`qpy_sequence` block representing CONTEXT_PARAMS,
the parameters associated with the alignment context :attr:`AlignmentKind._context_params`.

.. note::

    The :class:`~.AlignFunc` context is not supported because of the callback function
    stored in the context parameters.

.. _qpy_schedule_instructions:

SCHEDULE_BLOCK_INSTRUCTIONS
---------------------------

This alignment block is further followed by ``num_element`` block elements which may
consist of nested schedule blocks and schedule instructions.

Each schedule instruction follows this structure:

``INSTRUCTION_TYPE | INSTRUCTION_OPERANDS``

which starts with a  ``char``  representing the INSTRUCTION_TYPE,
followed by the :ref:`qpy_sequence` block representing the instruction
:attr:`~qiskit.pulse.instructions.Instruction.operands` (INSTRUCTION_OPERANDS).
Note that the data structure of pulse :class:`~qiskit.pulse.instructions.Instruction`
is unified so that instance can be uniquely determined by the class and a tuple of operands.
The mapping of ``type`` to the instruction subclass is defined as follows:

- ``a``: :class:`~qiskit.pulse.instructions.Acquire` instruction
- ``p``: :class:`~qiskit.pulse.instructions.Play` instruction
- ``d``: :class:`~qiskit.pulse.instructions.Delay` instruction
- ``f``: :class:`~qiskit.pulse.instructions.SetFrequency` instruction
- ``g``: :class:`~qiskit.pulse.instructions.ShiftFrequency` instruction
- ``q``: :class:`~qiskit.pulse.instructions.SetPhase` instruction
- ``r``: :class:`~qiskit.pulse.instructions.ShiftPhase` instruction
- ``b``: :class:`~qiskit.pulse.instructions.RelativeBarrier` instruction
- ``t``: :class:`~qiskit.pulse.instructions.TimeBlockade` instruction
- ``y``: :class:`~qiskit.pulse.instructions.Reference` instruction (added in v.7)

.. _qpy_schedule_operands:

SCHEDULE_BLOCK_OPERANDS
-----------------------

The operands of these schedule blocks can normally be serialized through the standard QPY value
serialization mechanism. However, there are special object types that only appear in the schedule
operands and require a custom format definition.
Since the operands are serialized as :ref:`qpy_sequence`, each element must be packed with the
:ref:`INSTRUCTION_PARAM <qpy_instruction_param>` pack struct, where each
payload starts with a header block consists of
the ``type`` char  and ``size`` integer (``uint64_t``).

Special objects start with the following type key:

- ``c``: :class:`~qiskit.pulse.channels.Channel`
- ``w``: :class:`~qiskit.pulse.library.Waveform`
- ``s``: :class:`~qiskit.pulse.library.SymbolicPulse`
- ``o``: string (operand string, new in version 0.7)
- ``d``: :data:`.CASE_DEFAULT` for switch-statement support (added in v.7)
- ``R``: :class:`.ClassicalRegister` or :class:`.Clbit` (added in v.7)

Note that ``o`` represents the operand string, which should, in theory, follow same encoding as the
built-in Python string ``s``. However, in the context of pulse instruction operands, this
conflicts with the :class:`~qiskit.pulse.library.SymbolicPulse`, so a different type
character ``o`` has been reserved for this purpose.

``d`` is followed by no data because it represents the literal value :data:`.CASE_DEFAULT`
for switch-statement support,
and ``R`` represents a :class:`.ClassicalRegister` or :class:`.Clbit`,
and is followed by the same format as the description of register or classical bit as used in the
first element of :ref:`the condition of an INSTRUCTION field <qpy_instructions>`.

.. _qpy_schedule_channel:

CHANNEL
-------

The CHANNEL block follows:

``CHANNEL_SUBTYPE | CHANNEL_INDEX``

It starts with channel subtype ``char`` that maps an object data to
:class:`~qiskit.pulse.channels.Channel` subclass. Mapping is defined as follows:

- ``d``: :class:`~qiskit.pulse.channels.DriveChannel`
- ``c``: :class:`~qiskit.pulse.channels.ControlChannel`
- ``m``: :class:`~qiskit.pulse.channels.MeasureChannel`
- ``a``: :class:`~qiskit.pulse.channels.AcquireChannel`
- ``e``: :class:`~qiskit.pulse.channels.MemorySlot`
- ``r``: :class:`~qiskit.pulse.channels.RegisterSlot`

The key is immediately followed by the channel index serialized as an INSTRUCTION_PARAM.

.. _qpy_schedule_waveform:

WAVEFORM
--------

The WAVEFORM block follows

``HEADER | WAVEFORM_DATA | WAVEFORM_NAME``

It starts with a WAVEFORM header:

.. code-block:: c

    struct {
        double epsilon;
        uint32_t data_size;
        _Bool amp_limited;
    }

which is followed by ``data_size`` bytes of complex ``ndarray`` binary generated by numpy.save_.
This represents the complex IQ data points played on a quantum device.
:attr:`~qiskit.pulse.library.Waveform.name` is saved after the samples in the
INSTRUCTION_PARAM pack struct, which can be string or ``None``.

.. _numpy.save: https://numpy.org/doc/stable/reference/generated/numpy.save.html

.. _qpy_schedule_symbolic_pulse:

SYMBOLIC_PULSE
--------------

The SYMBOLIC_PULSE block follows:

``HEADER | PULSE_TYPE | ENVELOPE_EXPR | CONSTRAINTS_EXPR | CONDITIONS_EXPR``

It starts with a SYMBOLIC_PULSE header:

.. code-block:: c

    struct {
        uint16_t type_size;
        uint16_t envelope_size;
        uint16_t constraints_size;
        uint16_t valid_amp_conditions_size;
        _Bool amp_limited;
    }

which is followed by ``type_size`` UTF-8 bytes of :attr:`.SymbolicPulse.pulse_type` string
that represents the class of waveform, such as "Gaussian" or "GaussianSquare".
Then, ``envelope_size``, ``constraints_size``, ``valid_amp_conditions_size`` UTF-8 bytes of
serialized symbolic expressions are generated for :attr:`.SymbolicPulse.envelope`,
:attr:`.SymbolicPulse.constraints`, and :attr:`.SymbolicPulse.valid_amp_conditions`, respectively.
Since string representations of these expressions are usually lengthy,
the expression binary is generated by the python zlib_ module with data compression.

To uniquely specify a pulse instance, we also need to store the associated parameters,
which consist of ``duration`` and the rest of parameters as a dictionary.
Dictionary parameters are first dumped in the :ref:`qpy_mapping` form, and then ``duration``
is dumped with the INSTRUCTION_PARAM pack struct.
Lastly, :attr:`~qiskit.pulse.library.SymbolicPulse.name` is saved also with the
INSTRUCTION_PARAM pack struct, which can be string or ``None``.

.. _zlib: https://docs.python.org/3/library/zlib.html

QPY Version 6 added support for class :class:`.~ScalableSymbolicPulse` through a new
`SymbolicPulse` object with an updated header:

SYMBOLIC_PULSE_V2
-----------------

``HEADER | CLASS_NAME | PULSE_TYPE | ENVELOPE_EXPR | CONSTRAINTS_EXPR | CONDITIONS_EXPR``

The new header is:

.. code-block:: c

    // after QPY v.6
    struct {
        uint16_t class_name_size;
        uint16_t type_size;
        uint16_t envelope_size;
        uint16_t constraints_size;
        uint16_t valid_amp_conditions_size;
        _Bool amp_limited;
    }

The header is then immediately followed by ``class_name_size`` UTF-8 bytes with the name of the class.
Currently, either `SymbolicPulse` or `ScalableSymbolicPulse` are supported.


.. _qpy_mapping:

MAPPING
-------

The MAPPING block is a representation for arbitrary mapping objects. This is a fixed length
:ref:`qpy_sequence` of key-value pairs represented by the MAP_ITEM block defined below.

A MAP_ITEM follows this format:

``HEADER | DICT_KEY | DATA``

It starts with a header defined as:

.. code-block:: c

    struct {
        uint16_t key_size;
        char type;
        uint16_t size;
    }

which is immediately followed by the ``key_size`` UTF-8 bytes representing
the dictionary key and ``size`` UTF-8 bytes of arbitrary object data of a
QPY serializable ``type``.

.. [#f1] https://tools.ietf.org/html/rfc1700
.. [#f2] https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
.. [#f3] https://docs.python.org/3/c-api/complex.html#c.Py_complex

"""

from .interface import dump, load

# For backward compatibility. Provide, Runtime, Experiment call these private functions.
from .binary_io import (
    _write_instruction,
    _read_instruction,
    _write_parameter_expression,
    _read_parameter_expression,
    _read_parameter_expression_v3,
)

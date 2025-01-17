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

"""User interface of qpy serializer."""

from __future__ import annotations

from json import JSONEncoder, JSONDecoder
from typing import Union, List, BinaryIO, Type, Optional
from collections.abc import Iterable
import struct
import warnings
import re

from qiskit.circuit import QuantumCircuit
from qiskit.pulse import ScheduleBlock
from qiskit.exceptions import QiskitError
from qiskit.qpy import formats, common, binary_io, type_keys
from qiskit.qpy.exceptions import QPYLoadingDeprecatedFeatureWarning, QpyError
from qiskit.version import __version__
from qiskit.utils.deprecate_pulse import deprecate_pulse_arg


# pylint: disable=invalid-name
QPY_SUPPORTED_TYPES = Union[QuantumCircuit, ScheduleBlock]

# This version pattern is taken from the pypa packaging project:
# https://github.com/pypa/packaging/blob/21.3/packaging/version.py#L223-L254
# which is dual licensed Apache 2.0 and BSD see the source for the original
# authors and other details
VERSION_PATTERN = (
    "^"
    + r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""
    + "$"
)
VERSION_PATTERN_REGEX = re.compile(VERSION_PATTERN, re.VERBOSE | re.IGNORECASE)


@deprecate_pulse_arg(
    "programs",
    deprecation_description="Passing `ScheduleBlock` to `programs`",
    predicate=lambda p: isinstance(p, ScheduleBlock),
)
def dump(
    programs: Union[List[QPY_SUPPORTED_TYPES], QPY_SUPPORTED_TYPES],
    file_obj: BinaryIO,
    metadata_serializer: Optional[Type[JSONEncoder]] = None,
    use_symengine: bool = True,
    version: int = common.QPY_VERSION,
):
    """Write QPY binary data to a file

    This function is used to save a circuit to a file for later use or transfer
    between machines. The QPY format is backwards compatible and can be
    loaded with future versions of Qiskit.

    For example:

    .. plot::
       :include-source:
       :nofigs:
       :context: reset

        from qiskit.circuit import QuantumCircuit
        from qiskit import qpy

        qc = QuantumCircuit(2, name='Bell', metadata={'test': True})
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

    from this you can write the qpy data to a file:

    .. code-block:: python

        with open('bell.qpy', 'wb') as fd:
            qpy.dump(qc, fd)

    or a gzip compressed file:

    .. code-block:: python

        import gzip

        with gzip.open('bell.qpy.gz', 'wb') as fd:
            qpy.dump(qc, fd)

    Which will save the qpy serialized circuit to the provided file.

    Args:
        programs: QPY supported object(s) to store in the specified file like object.
            QPY supports :class:`.QuantumCircuit` and :class:`.ScheduleBlock`.
            Different data types must be separately serialized.
            Support for :class:`.ScheduleBlock` is deprecated since Qiskit 1.3.0.
        file_obj: The file like object to write the QPY data too
        metadata_serializer: An optional JSONEncoder class that
            will be passed the ``.metadata`` attribute for each program in ``programs`` and will be
            used as the ``cls`` kwarg on the `json.dump()`` call to JSON serialize that dictionary.
        use_symengine: If True, all objects containing symbolic expressions will be serialized
            using symengine's native mechanism. This is a faster serialization alternative,
            but not supported in all platforms. This flag only has an effect if the emitted QPY format
            version is 10, 11, or 12. For QPY format version >= 13 (which is the default starting in
            Qiskit 1.3.0) this flag is no longer used.
        version: The QPY format version to emit. By default this defaults to
            the latest supported format of :attr:`~.qpy.QPY_VERSION`, however for
            compatibility reasons if you need to load the generated QPY payload with an older
            version of Qiskit you can also select an older QPY format version down to the minimum
            supported export version, which only can change during a Qiskit major version release,
            to generate an older QPY format version.  You can access the current QPY version and
            minimum compatible version with :attr:`.qpy.QPY_VERSION` and
            :attr:`.qpy.QPY_COMPATIBILITY_VERSION` respectively.

            .. note::

                If specified with an older version of QPY the limitations and potential bugs stemming
                from the QPY format at that version will persist. This should only be used if
                compatibility with loading the payload with an older version of Qiskit is necessary.

            .. note::

                If serializing a :class:`.QuantumCircuit` or :class:`.ScheduleBlock` that contain
                :class:`.ParameterExpression` objects with ``version`` set low with the intent to
                load the payload using a historical release of Qiskit, it is safest to set the
                ``use_symengine`` flag to ``False``.  Versions of Qiskit prior to 1.2.4 cannot load
                QPY files containing ``symengine``-serialized :class:`.ParameterExpression` objects
                unless the version of ``symengine`` used between the loading and generating
                environments matches.


    Raises:
        QpyError: When multiple data format is mixed in the output.
        TypeError: When invalid data type is input.
        ValueError: When an unsupported version number is passed in for the ``version`` argument
    """
    if not isinstance(programs, Iterable):
        programs = [programs]

    program_types = set()
    for program in programs:
        program_types.add(type(program))

    if len(program_types) > 1:
        raise QpyError(
            "Input programs contain multiple data types. "
            "Different data type must be serialized separately."
        )
    program_type = next(iter(program_types))

    if issubclass(program_type, QuantumCircuit):
        type_key = type_keys.Program.CIRCUIT
        writer = binary_io.write_circuit
    elif program_type is ScheduleBlock:
        type_key = type_keys.Program.SCHEDULE_BLOCK
        writer = binary_io.write_schedule_block
    else:
        raise TypeError(f"'{program_type}' is not supported data type.")

    if version is None:
        version = common.QPY_VERSION
    elif common.QPY_COMPATIBILITY_VERSION > version or version > common.QPY_VERSION:
        raise ValueError(
            f"The specified QPY version {version} is not support for dumping with this version, "
            f"of Qiskit. The only supported versions between {common.QPY_COMPATIBILITY_VERSION} and "
            f"{common.QPY_VERSION}"
        )

    version_match = VERSION_PATTERN_REGEX.search(__version__)
    version_parts = [int(x) for x in version_match.group("release").split(".")]
    encoding = type_keys.SymExprEncoding.assign(use_symengine)
    header = struct.pack(
        formats.FILE_HEADER_V10_PACK,
        b"QISKIT",
        version,
        version_parts[0],
        version_parts[1],
        version_parts[2],
        len(programs),
        encoding,
    )
    file_obj.write(header)
    common.write_type_key(file_obj, type_key)

    pulse_gates = False
    for program in programs:
        if type_key == type_keys.Program.CIRCUIT and program._calibrations_prop:
            pulse_gates = True
        writer(
            file_obj,
            program,
            metadata_serializer=metadata_serializer,
            use_symengine=use_symengine,
            version=version,
        )

    if pulse_gates:
        warnings.warn(
            category=DeprecationWarning,
            message="Pulse gates serialization is deprecated as of Qiskit 1.3. "
            "It will be removed in Qiskit 2.0.",
        )


def load(
    file_obj: BinaryIO,
    metadata_deserializer: Optional[Type[JSONDecoder]] = None,
) -> List[QPY_SUPPORTED_TYPES]:
    """Load a QPY binary file

    This function is used to load a serialized QPY Qiskit program file and create
    :class:`~qiskit.circuit.QuantumCircuit` objects or
    :class:`~qiskit.pulse.schedule.ScheduleBlock` objects from its contents.
    For example:

    .. code-block:: python

        from qiskit import qpy

        with open('bell.qpy', 'rb') as fd:
            circuits = qpy.load(fd)

    or with a gzip compressed file:

    .. code-block:: python

        import gzip
        from qiskit import qpy

        with gzip.open('bell.qpy.gz', 'rb') as fd:
            circuits = qpy.load(fd)

    which will read the contents of the qpy and return a list of
    :class:`~qiskit.circuit.QuantumCircuit` objects or
    :class:`~qiskit.pulse.schedule.ScheduleBlock` objects from the file.

    Args:
        file_obj: A file like object that contains the QPY binary
            data for a circuit or pulse schedule.
        metadata_deserializer: An optional JSONDecoder class
            that will be used for the ``cls`` kwarg on the internal
            ``json.load`` call used to deserialize the JSON payload used for
            the ``.metadata`` attribute for any programs in the QPY file.
            If this is not specified the circuit metadata will
            be parsed as JSON with the stdlib ``json.load()`` function using
            the default ``JSONDecoder`` class.

    Returns:
        The list of Qiskit programs contained in the QPY data.
        A list is always returned, even if there is only 1 program in the QPY data.

    Raises:
        QiskitError: if ``file_obj`` is not a valid QPY file
        TypeError: When invalid data type is loaded.
    """

    # identify file header version
    version = struct.unpack("!6sB", file_obj.read(7))[1]
    file_obj.seek(0)

    if version > common.QPY_VERSION:
        raise QiskitError(
            f"The QPY format version being read, {version}, isn't supported by "
            "this Qiskit version. Please upgrade your version of Qiskit to load this QPY payload"
        )

    if version < 10:
        data = formats.FILE_HEADER._make(
            struct.unpack(
                formats.FILE_HEADER_PACK,
                file_obj.read(formats.FILE_HEADER_SIZE),
            )
        )
    else:
        data = formats.FILE_HEADER_V10._make(
            struct.unpack(
                formats.FILE_HEADER_V10_PACK,
                file_obj.read(formats.FILE_HEADER_V10_SIZE),
            )
        )

    if data.preface.decode(common.ENCODE) != "QISKIT":
        raise QiskitError("Input file is not a valid QPY file")
    version_match = VERSION_PATTERN_REGEX.search(__version__)
    env_qiskit_version = [int(x) for x in version_match.group("release").split(".")]

    qiskit_version = (data.major_version, data.minor_version, data.patch_version)
    # pylint: disable=too-many-boolean-expressions
    if (
        env_qiskit_version[0] < qiskit_version[0]
        or (
            env_qiskit_version[0] == qiskit_version[0] and qiskit_version[1] > env_qiskit_version[1]
        )
        or (
            env_qiskit_version[0] == qiskit_version[0]
            and qiskit_version[1] == env_qiskit_version[1]
            and qiskit_version[2] > env_qiskit_version[2]
        )
    ):
        warnings.warn(
            "The qiskit version used to generate the provided QPY "
            f"file, {'.'.join([str(x) for x in qiskit_version])}, "
            f"is newer than the current qiskit version {__version__}. "
            "This may result in an error if the QPY file uses "
            "instructions not present in this current qiskit "
            "version"
        )

    if data.qpy_version < 5:
        type_key = type_keys.Program.CIRCUIT
    else:
        type_key = common.read_type_key(file_obj)

    if type_key == type_keys.Program.CIRCUIT:
        loader = binary_io.read_circuit
    elif type_key == type_keys.Program.SCHEDULE_BLOCK:
        loader = binary_io.read_schedule_block
        warnings.warn(
            category=QPYLoadingDeprecatedFeatureWarning,
            message="Pulse gates deserialization is deprecated as of Qiskit 1.3 and "
            "will be removed in Qiskit 2.0. This is part of the deprecation plan for "
            "the entire Qiskit Pulse package. Once Pulse is removed, `ScheduleBlock` "
            "sections will be ignored when loading QPY files with pulse data.",
        )

    else:
        raise TypeError(f"Invalid payload format data kind '{type_key}'.")

    if data.qpy_version < 10:
        use_symengine = False
    else:
        use_symengine = data.symbolic_encoding == type_keys.SymExprEncoding.SYMENGINE

    programs = []
    for _ in range(data.num_programs):
        programs.append(
            loader(
                file_obj,
                data.qpy_version,
                metadata_deserializer=metadata_deserializer,
                use_symengine=use_symengine,
            )
        )
    return programs

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

"""Read and write schedule and schedule instructions."""
import json
import struct
import zlib
import warnings

from io import BytesIO

import numpy as np
import symengine as sym

from qiskit.exceptions import QiskitError
from qiskit.pulse import library, channels, instructions
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.qpy import formats, common, type_keys
from qiskit.qpy.binary_io import value
from qiskit.qpy.exceptions import QpyError
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.utils.deprecate_pulse import ignore_pulse_deprecation_warnings


def _read_channel(file_obj, version):
    type_key = common.read_type_key(file_obj)
    index = value.read_value(file_obj, version, {})

    channel_cls = type_keys.ScheduleChannel.retrieve(type_key)

    return channel_cls(index)


def _read_waveform(file_obj, version):
    header = formats.WAVEFORM._make(
        struct.unpack(
            formats.WAVEFORM_PACK,
            file_obj.read(formats.WAVEFORM_SIZE),
        )
    )
    samples_raw = file_obj.read(header.data_size)
    samples = common.data_from_binary(samples_raw, np.load)
    name = value.read_value(file_obj, version, {})

    return library.Waveform(
        samples=samples,
        name=name,
        epsilon=header.epsilon,
        limit_amplitude=header.amp_limited,
    )


def _loads_obj(type_key, binary_data, version, vectors):
    """Wraps `value.loads_value` to deserialize binary data to dictionary
    or list objects which are not supported by `value.loads_value`.
    """
    if type_key == b"D":
        with BytesIO(binary_data) as container:
            return common.read_mapping(
                file_obj=container, deserializer=_loads_obj, version=version, vectors=vectors
            )
    elif type_key == b"l":
        with BytesIO(binary_data) as container:
            return common.read_sequence(
                file_obj=container, deserializer=_loads_obj, version=version, vectors=vectors
            )
    else:
        return value.loads_value(type_key, binary_data, version, vectors)


def _read_kernel(file_obj, version):
    params = common.read_mapping(
        file_obj=file_obj,
        deserializer=_loads_obj,
        version=version,
        vectors={},
    )
    name = value.read_value(file_obj, version, {})
    return Kernel(name=name, **params)


def _read_discriminator(file_obj, version):
    params = common.read_mapping(
        file_obj=file_obj,
        deserializer=_loads_obj,
        version=version,
        vectors={},
    )
    name = value.read_value(file_obj, version, {})
    return Discriminator(name=name, **params)


def _loads_symbolic_expr(expr_bytes, use_symengine=False):
    if expr_bytes == b"":
        return None
    expr_bytes = zlib.decompress(expr_bytes)
    if use_symengine:
        return common.load_symengine_payload(expr_bytes)
    else:
        from sympy import parse_expr

        expr_txt = expr_bytes.decode(common.ENCODE)
        expr = parse_expr(expr_txt)
        return sym.sympify(expr)


def _read_symbolic_pulse(file_obj, version):
    make = formats.SYMBOLIC_PULSE._make
    pack = formats.SYMBOLIC_PULSE_PACK
    size = formats.SYMBOLIC_PULSE_SIZE

    header = make(
        struct.unpack(
            pack,
            file_obj.read(size),
        )
    )
    pulse_type = file_obj.read(header.type_size).decode(common.ENCODE)
    envelope = _loads_symbolic_expr(file_obj.read(header.envelope_size))
    constraints = _loads_symbolic_expr(file_obj.read(header.constraints_size))
    valid_amp_conditions = _loads_symbolic_expr(file_obj.read(header.valid_amp_conditions_size))
    parameters = common.read_mapping(
        file_obj,
        deserializer=value.loads_value,
        version=version,
        vectors={},
    )

    # In the transition to Qiskit Terra 0.23 (QPY version 6), the representation of library pulses
    # was changed from complex "amp" to float "amp" and "angle". The existing library pulses in
    # previous versions are handled here separately to conform with the new representation. To
    # avoid role assumption for "amp" for custom pulses, only the library pulses are handled this
    # way.

    # List of pulses in the library in QPY version 5 and below:
    legacy_library_pulses = ["Gaussian", "GaussianSquare", "Drag", "Constant"]
    class_name = "SymbolicPulse"  # Default class name, if not in the library

    if pulse_type in legacy_library_pulses:
        parameters["angle"] = np.angle(parameters["amp"])
        parameters["amp"] = np.abs(parameters["amp"])
        _amp, _angle = sym.symbols("amp, angle")
        envelope = envelope.subs(_amp, _amp * sym.exp(sym.I * _angle))

        warnings.warn(
            f"Library pulses with complex amp are no longer supported. "
            f"{pulse_type} with complex amp was converted to (amp,angle) representation.",
            UserWarning,
        )
        class_name = "ScalableSymbolicPulse"

    duration = value.read_value(file_obj, version, {})
    name = value.read_value(file_obj, version, {})

    if class_name == "SymbolicPulse":
        return library.SymbolicPulse(
            pulse_type=pulse_type,
            duration=duration,
            parameters=parameters,
            name=name,
            limit_amplitude=header.amp_limited,
            envelope=envelope,
            constraints=constraints,
            valid_amp_conditions=valid_amp_conditions,
        )
    elif class_name == "ScalableSymbolicPulse":
        return library.ScalableSymbolicPulse(
            pulse_type=pulse_type,
            duration=duration,
            amp=parameters["amp"],
            angle=parameters["angle"],
            parameters=parameters,
            name=name,
            limit_amplitude=header.amp_limited,
            envelope=envelope,
            constraints=constraints,
            valid_amp_conditions=valid_amp_conditions,
        )
    else:
        raise NotImplementedError(f"Unknown class '{class_name}'")


def _read_symbolic_pulse_v6(file_obj, version, use_symengine):
    make = formats.SYMBOLIC_PULSE_V2._make
    pack = formats.SYMBOLIC_PULSE_PACK_V2
    size = formats.SYMBOLIC_PULSE_SIZE_V2

    header = make(
        struct.unpack(
            pack,
            file_obj.read(size),
        )
    )
    class_name = file_obj.read(header.class_name_size).decode(common.ENCODE)
    pulse_type = file_obj.read(header.type_size).decode(common.ENCODE)
    envelope = _loads_symbolic_expr(file_obj.read(header.envelope_size), use_symengine)
    constraints = _loads_symbolic_expr(file_obj.read(header.constraints_size), use_symengine)
    valid_amp_conditions = _loads_symbolic_expr(
        file_obj.read(header.valid_amp_conditions_size), use_symengine
    )
    parameters = common.read_mapping(
        file_obj,
        deserializer=value.loads_value,
        version=version,
        vectors={},
    )

    duration = value.read_value(file_obj, version, {})
    name = value.read_value(file_obj, version, {})

    if class_name == "SymbolicPulse":
        return library.SymbolicPulse(
            pulse_type=pulse_type,
            duration=duration,
            parameters=parameters,
            name=name,
            limit_amplitude=header.amp_limited,
            envelope=envelope,
            constraints=constraints,
            valid_amp_conditions=valid_amp_conditions,
        )
    elif class_name == "ScalableSymbolicPulse":
        # Between Qiskit 0.40 and 0.46, the (amp, angle) representation was present,
        # but complex amp was still allowed. In Qiskit 1.0 and beyond complex amp
        # is no longer supported and so the amp needs to be checked and converted.
        # Once QPY version is bumped, a new reader function can be introduced without
        # this check.
        if isinstance(parameters["amp"], complex):
            parameters["angle"] = np.angle(parameters["amp"])
            parameters["amp"] = np.abs(parameters["amp"])
            warnings.warn(
                f"ScalableSymbolicPulse with complex amp are no longer supported. "
                f"{pulse_type} with complex amp was converted to (amp,angle) representation.",
                UserWarning,
            )

        return library.ScalableSymbolicPulse(
            pulse_type=pulse_type,
            duration=duration,
            amp=parameters["amp"],
            angle=parameters["angle"],
            parameters=parameters,
            name=name,
            limit_amplitude=header.amp_limited,
            envelope=envelope,
            constraints=constraints,
            valid_amp_conditions=valid_amp_conditions,
        )
    else:
        raise NotImplementedError(f"Unknown class '{class_name}'")


def _read_alignment_context(file_obj, version):
    type_key = common.read_type_key(file_obj)

    context_params = common.read_sequence(
        file_obj,
        deserializer=value.loads_value,
        version=version,
        vectors={},
    )
    context_cls = type_keys.ScheduleAlignment.retrieve(type_key)

    instance = object.__new__(context_cls)
    instance._context_params = tuple(context_params)

    return instance


# pylint: disable=too-many-return-statements
def _loads_operand(type_key, data_bytes, version, use_symengine):
    if type_key == type_keys.ScheduleOperand.WAVEFORM:
        return common.data_from_binary(data_bytes, _read_waveform, version=version)
    if type_key == type_keys.ScheduleOperand.SYMBOLIC_PULSE:
        if version < 6:
            return common.data_from_binary(data_bytes, _read_symbolic_pulse, version=version)
        else:
            return common.data_from_binary(
                data_bytes, _read_symbolic_pulse_v6, version=version, use_symengine=use_symengine
            )
    if type_key == type_keys.ScheduleOperand.CHANNEL:
        return common.data_from_binary(data_bytes, _read_channel, version=version)
    if type_key == type_keys.ScheduleOperand.OPERAND_STR:
        return data_bytes.decode(common.ENCODE)
    if type_key == type_keys.ScheduleOperand.KERNEL:
        return common.data_from_binary(
            data_bytes,
            _read_kernel,
            version=version,
        )
    if type_key == type_keys.ScheduleOperand.DISCRIMINATOR:
        return common.data_from_binary(
            data_bytes,
            _read_discriminator,
            version=version,
        )

    return value.loads_value(type_key, data_bytes, version, {})


def _read_element(file_obj, version, metadata_deserializer, use_symengine):
    type_key = common.read_type_key(file_obj)

    if type_key == type_keys.Program.SCHEDULE_BLOCK:
        return read_schedule_block(file_obj, version, metadata_deserializer, use_symengine)

    operands = common.read_sequence(
        file_obj, deserializer=_loads_operand, version=version, use_symengine=use_symengine
    )
    name = value.read_value(file_obj, version, {})

    instance = object.__new__(type_keys.ScheduleInstruction.retrieve(type_key))
    instance._operands = tuple(operands)
    instance._name = name
    instance._hash = None

    return instance


def _loads_reference_item(type_key, data_bytes, metadata_deserializer, version):
    if type_key == type_keys.Value.NULL:
        return None
    if type_key == type_keys.Program.SCHEDULE_BLOCK:
        return common.data_from_binary(
            data_bytes,
            deserializer=read_schedule_block,
            version=version,
            metadata_deserializer=metadata_deserializer,
        )

    raise QpyError(
        f"Loaded schedule reference item is neither None nor ScheduleBlock. "
        f"Type key {type_key} is not valid data type for a reference items. "
        "This data cannot be loaded. Please check QPY version."
    )


def _write_channel(file_obj, data, version):
    type_key = type_keys.ScheduleChannel.assign(data)
    common.write_type_key(file_obj, type_key)
    value.write_value(file_obj, data.index, version=version)


def _write_waveform(file_obj, data, version):
    samples_bytes = common.data_to_binary(data.samples, np.save)

    header = struct.pack(
        formats.WAVEFORM_PACK,
        data.epsilon,
        len(samples_bytes),
        data._limit_amplitude,
    )
    file_obj.write(header)
    file_obj.write(samples_bytes)
    value.write_value(file_obj, data.name, version=version)


def _dumps_obj(obj, version):
    """Wraps `value.dumps_value` to serialize dictionary and list objects
    which are not supported by `value.dumps_value`.
    """
    if isinstance(obj, dict):
        with BytesIO() as container:
            common.write_mapping(
                file_obj=container, mapping=obj, serializer=_dumps_obj, version=version
            )
            binary_data = container.getvalue()
        return b"D", binary_data
    elif isinstance(obj, list):
        with BytesIO() as container:
            common.write_sequence(
                file_obj=container, sequence=obj, serializer=_dumps_obj, version=version
            )
            binary_data = container.getvalue()
        return b"l", binary_data
    else:
        return value.dumps_value(obj, version=version)


def _write_kernel(file_obj, data, version):
    name = data.name
    params = data.params
    common.write_mapping(file_obj=file_obj, mapping=params, serializer=_dumps_obj, version=version)
    value.write_value(file_obj, name, version=version)


def _write_discriminator(file_obj, data, version):
    name = data.name
    params = data.params
    common.write_mapping(file_obj=file_obj, mapping=params, serializer=_dumps_obj, version=version)
    value.write_value(file_obj, name, version=version)


def _dumps_symbolic_expr(expr, use_symengine):
    if expr is None:
        return b""
    if use_symengine:
        expr_bytes = expr.__reduce__()[1][0]
    else:
        from sympy import srepr, sympify

        expr_bytes = srepr(sympify(expr)).encode(common.ENCODE)
    return zlib.compress(expr_bytes)


def _write_symbolic_pulse(file_obj, data, use_symengine, version):
    class_name_bytes = data.__class__.__name__.encode(common.ENCODE)
    pulse_type_bytes = data.pulse_type.encode(common.ENCODE)
    envelope_bytes = _dumps_symbolic_expr(data.envelope, use_symengine)
    constraints_bytes = _dumps_symbolic_expr(data.constraints, use_symengine)
    valid_amp_conditions_bytes = _dumps_symbolic_expr(data.valid_amp_conditions, use_symengine)

    header_bytes = struct.pack(
        formats.SYMBOLIC_PULSE_PACK_V2,
        len(class_name_bytes),
        len(pulse_type_bytes),
        len(envelope_bytes),
        len(constraints_bytes),
        len(valid_amp_conditions_bytes),
        data._limit_amplitude,
    )
    file_obj.write(header_bytes)
    file_obj.write(class_name_bytes)
    file_obj.write(pulse_type_bytes)
    file_obj.write(envelope_bytes)
    file_obj.write(constraints_bytes)
    file_obj.write(valid_amp_conditions_bytes)
    common.write_mapping(
        file_obj,
        mapping=data._params,
        serializer=value.dumps_value,
        version=version,
    )
    value.write_value(file_obj, data.duration, version=version)
    value.write_value(file_obj, data.name, version=version)


def _write_alignment_context(file_obj, context, version):
    type_key = type_keys.ScheduleAlignment.assign(context)
    common.write_type_key(file_obj, type_key)
    common.write_sequence(
        file_obj, sequence=context._context_params, serializer=value.dumps_value, version=version
    )


def _dumps_operand(operand, use_symengine, version):
    if isinstance(operand, library.Waveform):
        type_key = type_keys.ScheduleOperand.WAVEFORM
        data_bytes = common.data_to_binary(operand, _write_waveform, version=version)
    elif isinstance(operand, library.SymbolicPulse):
        type_key = type_keys.ScheduleOperand.SYMBOLIC_PULSE
        data_bytes = common.data_to_binary(
            operand, _write_symbolic_pulse, use_symengine=use_symengine, version=version
        )
    elif isinstance(operand, channels.Channel):
        type_key = type_keys.ScheduleOperand.CHANNEL
        data_bytes = common.data_to_binary(operand, _write_channel, version=version)
    elif isinstance(operand, str):
        type_key = type_keys.ScheduleOperand.OPERAND_STR
        data_bytes = operand.encode(common.ENCODE)
    elif isinstance(operand, Kernel):
        type_key = type_keys.ScheduleOperand.KERNEL
        data_bytes = common.data_to_binary(operand, _write_kernel, version=version)
    elif isinstance(operand, Discriminator):
        type_key = type_keys.ScheduleOperand.DISCRIMINATOR
        data_bytes = common.data_to_binary(operand, _write_discriminator, version=version)
    else:
        type_key, data_bytes = value.dumps_value(operand, version=version)

    return type_key, data_bytes


def _write_element(file_obj, element, metadata_serializer, use_symengine, version):
    if isinstance(element, ScheduleBlock):
        common.write_type_key(file_obj, type_keys.Program.SCHEDULE_BLOCK)
        write_schedule_block(file_obj, element, metadata_serializer, use_symengine, version=version)
    else:
        type_key = type_keys.ScheduleInstruction.assign(element)
        common.write_type_key(file_obj, type_key)
        common.write_sequence(
            file_obj,
            sequence=element.operands,
            serializer=_dumps_operand,
            use_symengine=use_symengine,
            version=version,
        )
        value.write_value(file_obj, element.name, version=version)


def _dumps_reference_item(schedule, metadata_serializer, version):
    if schedule is None:
        type_key = type_keys.Value.NULL
        data_bytes = b""
    else:
        type_key = type_keys.Program.SCHEDULE_BLOCK
        data_bytes = common.data_to_binary(
            obj=schedule,
            serializer=write_schedule_block,
            metadata_serializer=metadata_serializer,
            version=version,
        )
    return type_key, data_bytes


@ignore_pulse_deprecation_warnings
def read_schedule_block(file_obj, version, metadata_deserializer=None, use_symengine=False):
    """Read a single ScheduleBlock from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        version (int): QPY version.
        metadata_deserializer (JSONDecoder): An optional JSONDecoder class
            that will be used for the ``cls`` kwarg on the internal
            ``json.load`` call used to deserialize the JSON payload used for
            the :attr:`.ScheduleBlock.metadata` attribute for a schedule block
            in the file-like object. If this is not specified the circuit metadata will
            be parsed as JSON with the stdlib ``json.load()`` function using
            the default ``JSONDecoder`` class.
        use_symengine (bool): If True, symbolic objects will be serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
    Returns:
        ScheduleBlock: The schedule block object from the file.

    Raises:
        TypeError: If any of the instructions is invalid data format.
        QiskitError: QPY version is earlier than block support.
    """
    if version < 5:
        raise QiskitError(f"QPY version {version} does not support ScheduleBlock.")

    data = formats.SCHEDULE_BLOCK_HEADER._make(
        struct.unpack(
            formats.SCHEDULE_BLOCK_HEADER_PACK,
            file_obj.read(formats.SCHEDULE_BLOCK_HEADER_SIZE),
        )
    )
    name = file_obj.read(data.name_size).decode(common.ENCODE)
    metadata_raw = file_obj.read(data.metadata_size)
    metadata = json.loads(metadata_raw, cls=metadata_deserializer)
    context = _read_alignment_context(file_obj, version)

    block = ScheduleBlock(
        name=name,
        metadata=metadata,
        alignment_context=context,
    )
    for _ in range(data.num_elements):
        block_elm = _read_element(file_obj, version, metadata_deserializer, use_symengine)
        block.append(block_elm, inplace=True)

    # Load references
    if version >= 7:
        flat_key_refdict = common.read_mapping(
            file_obj=file_obj,
            deserializer=_loads_reference_item,
            version=version,
            metadata_deserializer=metadata_deserializer,
        )
        ref_dict = {}
        for key_str, schedule in flat_key_refdict.items():
            if schedule is not None:
                composite_key = tuple(key_str.split(instructions.Reference.key_delimiter))
                ref_dict[composite_key] = schedule
        if ref_dict:
            block.assign_references(ref_dict, inplace=True)

    return block


def write_schedule_block(
    file_obj, block, metadata_serializer=None, use_symengine=False, version=common.QPY_VERSION
):
    """Write a single ScheduleBlock object in the file like object.

    Args:
        file_obj (File): The file like object to write the circuit data in.
        block (ScheduleBlock): A schedule block data to write.
        metadata_serializer (JSONEncoder): An optional JSONEncoder class that
            will be passed the :attr:`.ScheduleBlock.metadata` dictionary for
            ``block`` and will be used as the ``cls`` kwarg
            on the ``json.dump()`` call to JSON serialize that dictionary.
        use_symengine (bool): If True, symbolic objects will be serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
        version (int): The QPY format version to use for serializing this circuit block
    Raises:
        TypeError: If any of the instructions is invalid data format.
    """
    metadata = json.dumps(block.metadata, separators=(",", ":"), cls=metadata_serializer).encode(
        common.ENCODE
    )
    block_name = block.name.encode(common.ENCODE)

    # Write schedule block header
    header_raw = formats.SCHEDULE_BLOCK_HEADER(
        name_size=len(block_name),
        metadata_size=len(metadata),
        num_elements=len(block),
    )
    header = struct.pack(formats.SCHEDULE_BLOCK_HEADER_PACK, *header_raw)
    file_obj.write(header)
    file_obj.write(block_name)
    file_obj.write(metadata)

    _write_alignment_context(file_obj, block.alignment_context, version=version)
    for block_elm in block._blocks:
        # Do not call block.blocks. This implicitly assigns references to instruction.
        # This breaks original reference structure.
        _write_element(file_obj, block_elm, metadata_serializer, use_symengine, version=version)

    # Write references
    flat_key_refdict = {}
    for ref_keys, schedule in block._reference_manager.items():
        # Do not call block.reference. This returns the reference of most outer program by design.
        key_str = instructions.Reference.key_delimiter.join(ref_keys)
        flat_key_refdict[key_str] = schedule
    common.write_mapping(
        file_obj=file_obj,
        mapping=flat_key_refdict,
        serializer=_dumps_reference_item,
        metadata_serializer=metadata_serializer,
        version=version,
    )

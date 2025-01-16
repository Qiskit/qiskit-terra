# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Helper functions for packing and unpacking bit arrays.
"""

from __future__ import annotations
from typing import Literal, Any
from collections.abc import Sequence
from numbers import Integral
from dataclasses import dataclass
from functools import lru_cache

from numpy.typing import ArrayLike, NDArray
import numpy as np


def pack_bits(
    array: ArrayLike,
    indices: Sequence[int] | slice | None = None,
    num_bits: int | None = None,
    bitorder: Literal["big", "little"] = "big",
) -> tuple[NDArray[np.uint8], int]:
    """Pack an array of bit values into byte array at specified indices.

    Args:
        array: The boolean array to bit-pack. The last axis of the input array is
            the number of bits.
        indices: The sequence or range of bit indices to set.
            This can be a sequence of unique integers, a range, a slice, or None.
            If None will be treated as a trivial range on the last dimension of
            the input array.
        num_bits: The total number of bits in the bit-packed array. If None
            this will be set from the maximum bit index. Any additional bits
            not in the input array will be set to 0.
        bitorder: The bitorder packing for binary arrays for the specified bits.

    Returns:
        A tuple (packed, num_bits) of the bit-packed array and the number of bits
        in the packed array.

    Raises:
        ValueError: If the input arguments are not a valid packing.

    .. note::

        In qiskit when using indices to refer to a bit we use little-endian
        ordering with respect to bitstrings, so "01" is 1 on bit-0 and 0 on bit-1,
        and corresponds to an outcome 1.
        The bitorder refers to how these bitstrings are represented as sequences
        of boolean values. Using ``bitorder = "big"`` an outcome ``"01" is given by
        a sequence ``[0, 1]``, while for ``bitorder = "little"`` it is [1, 0]``.
        Padding these sequences to whole bytes (8-bits) would be equivalent to
        ``[0, 0, 0, 0, 0, 0, 0, 1]`` (big) and ``[1, 0, 0, 0, 0, 0, 0, 0]`` (little)
        respectively.
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array, dtype=bool)
    array_bits = array.shape[-1]
    indices = _format_indices(array_bits, indices)

    if num_bits is None:
        num_bits = 1 + max(indices)
    elif num_bits < array_bits:
        raise ValueError(
            f"The specified number of bits ({num_bits}) is less than the number "
            f" of bits in the input array ({array_bits})."
        )
    if len(indices) != array_bits:
        raise ValueError(
            f"The number of indices ({len(indices)}) is not equal to the number of bits "
            f" of bits in the input array ({array_bits})."
        )

    packing = _get_bit_packing(indices, num_bits, bitorder)

    if not _is_trivial_bit_packing(packing):
        # Create the sliced bit array on the non-zero bytes
        padded = np.zeros((*array.shape[:-1], 8 * packing.num_byte_indices), dtype=bool)
        padded[..., packing.bit_indices] = array
        array = padded

    # Pack bits for intended bitorder
    packed = _pack(array, packing.bitorder)

    # Pad any empty bytes
    num_bytes = min_num_bytes(packing.num_bits)
    if not _is_trivial_byte_packing(packing):
        padded = np.zeros((*array.shape[:-1], num_bytes), dtype=np.uint8)
        padded[..., packing.byte_indices] = packed
        packed = padded

    return packed, packing.num_bits


def unpack_bits(
    array: NDArray[np.uint8],
    indices: Sequence[int] | slice | None = None,
    num_bits: int | None = None,
    bitorder: Literal["big", "little"] = "big",
) -> NDArray[np.uint8]:
    """Unpack the specified bit indices in packed bit array.

    Args:
        array: The bit-packed byte array to unpack.
        indices: The sequence or range of bit indices to unpack.
            This can be a sequence of unique integers, a range, a slice, or None.
            If None will be treated as a trivial range for the input array.
        num_bits: The total number of bits in the bit-packed array. If None
            this will be set as 8 times the number of bytes in the packed array.
        bitorder: The bitorder packing for binary arrays for the specified bits.

    Returns:
        The unpacked bit array.

    .. note::

        In qiskit when using indices to refer to a bit we use little-endian
        ordering with respect to bit-strings, so "01" is 1 on bit-0 and 0 on bit-1,
        and corresponds to an outcome 1.
        The bitorder refers to how these bit-strings are represented as sequences
        of boolean values. Using ``bitorder = "big"`` an outcome ``"01" is given by
        a sequence ``[0, 1]``, while for ``bitorder = "little"`` it is [1, 0]``.
        Padding these sequences to whole bytes (8-bits) would be equivalent to
        ``[0, 0, 0, 0, 0, 0, 0, 1]`` (big) and ``[1, 0, 0, 0, 0, 0, 0, 0]`` (little)
        respectively.
    """
    if num_bits is None:
        num_bits = 8 * array.shape[-1]
    indices = _format_indices(num_bits, indices)
    if num_bits < len(indices):
        raise ValueError(
            f"The specified number of bits ({num_bits}) is less than the number "
            f" of indexed bits ({len(indices)})."
        )

    packing = _get_bit_packing(indices, num_bits, bitorder)

    # Extract the bytes that span the specified bit indices
    if not _is_trivial_byte_packing(packing):
        array = array[..., packing.byte_indices]

    # Unpack the byte array
    if _is_trivial_bit_packing(packing):
        unpacked = _unpack(array, packing.bitorder, num_bits=packing.num_bit_indices)
    else:
        unpacked = _unpack(array, packing.bitorder)
        unpacked = unpacked[..., packing.bit_indices]

    return unpacked


def slice_packed_bits(
    array: NDArray[np.uint8],
    indices: Sequence[int] | slice | None = None,
    num_bits: int | None = None,
) -> NDArray[np.uint8]:
    """Return a packed-bit array containing only the specified bits.

    Args:
        array: The bit-packed byte array to unpack.
        indices: The sequence or range of bit indices to unpack.
            This can be a sequence of unique integers, a range, a slice, or None.
            If None will be treated as a trivial range for the input array.
        num_bits: The total number of bits in the bit-packed array. If None
            this will be set as 8 times the number of bytes in the packed array.

    Returns:
        The bit-packed array of the specified bit indices of the input array.
    """
    if num_bits is None:
        num_bits = 8 * array.shape[-1]
    indices = _format_indices(num_bits, indices)

    # If indices is not a range(0, N), we need to unpack bits
    if (
        not isinstance(indices, range)
        or indices.start != 0
        or indices.stop < 0
        or indices.step != 1
    ):
        unpacked = unpack_bits(array, indices, num_bits=num_bits, bitorder="little")
        repacked, _ = pack_bits(unpacked, bitorder="little")
        return repacked

    # Extract bytes
    packing = _get_bit_packing(indices, num_bits, "little")
    if not _is_trivial_byte_packing(packing):
        array = array[..., packing.byte_indices]

    # If the bit slice doesn't include whole bytes we need to zero
    # the head and tail
    if tail := packing.num_bit_indices % 8:
        # Make a copy of the array before we modify in place
        # to zero tail bits in the last byte.
        array = array.copy()
        array[..., 0] &= _SLICE_MASKS[tail]

    return array


def min_num_bytes(num_bits: int) -> int:
    """Return the minimum number of bytes needed to store ``num_bits``."""
    return num_bits // 8 + (num_bits % 8 > 0)


@dataclass(frozen=True)
class _BitPacking:
    """Dataclass for storing bit-packing specifications."""

    num_bits: int
    num_byte_indices: int
    byte_indices: Sequence[int] | slice
    num_bit_indices: int
    bit_indices: Sequence[int] | slice
    bitorder: Literal["big", "little"]


def _get_bit_packing(
    indices: Sequence[int],
    num_bits: int | None = None,
    bitorder: Literal["big", "little"] = "big",
) -> _BitPacking:
    """Return dataclass for bit packing and unpacking specifications.

    Args:
        indices: Optional, the sequence or range of bit indices to set.
            This can be a sequence of unique integers, a range.
        num_bits: The total number of bits in the packed array. If None
            this will be set from the maximum bit index.
        bitorder: The bitorder packing for binary arrays for the specified bits.

    Returns:
        A ``_BitPacking`` dataclass.
    """
    if _is_contiguous_range(indices):
        packing = _get_bit_packing_contiguous_range(indices, num_bits, bitorder)
    elif isinstance(indices, (tuple, range)):
        packing = _get_bit_packing_sequence_cached(indices, num_bits, bitorder)
    else:
        packing = _get_bit_packing_sequence(indices, num_bits, bitorder)
    return packing


def _get_bit_packing_sequence(
    indices: Sequence[int],
    num_bits: int,
    bitorder: Literal["big", "little"],
) -> _BitPacking:
    """Return the bit packing specification for general sequence bit indices."""
    num_bytes = min_num_bytes(num_bits)
    if not isinstance(indices, np.ndarray):
        indices = np.asarray(indices)

    # Make negative indices positive
    indices %= num_bits

    # Get bit index positions in the full unpacked array
    # To make computation easier we flip the range around for big ordering
    # so that it refers to the packed array indexes rather than logical
    # bit indexes.
    if bitorder == "little":
        bit_indices = indices
    else:
        bit_indices = 8 * num_bytes - 1 - indices[::-1]

    # Get corresponding byte for each bit and set of containing bytes
    bit_bytes, bit_bytes_inv = np.unique(bit_indices // 8, return_inverse=True)
    if bitorder == "little":
        byte_indices = num_bytes - 1 - bit_bytes[::-1]
    else:
        byte_indices = bit_bytes

    # Compute the shifts for converting bit indices
    # to sliced byte indices so that we don't need to pad on
    # empty bytes
    shifts = np.zeros_like(bit_bytes)
    last_byte = -1
    shift = 0
    for i, b in enumerate(bit_bytes):
        if (diff := b - last_byte - 1) > 0:
            shift += 8 * diff
        shifts[i] = shift
        last_byte = b
    slice_bit_indices = bit_indices - shifts[bit_bytes_inv]

    return _BitPacking(
        num_bits=num_bits,
        num_byte_indices=byte_indices.size,
        byte_indices=byte_indices,
        num_bit_indices=slice_bit_indices.size,
        bit_indices=slice_bit_indices,
        bitorder=bitorder,
    )


@lru_cache(16)
def _get_bit_packing_sequence_cached(
    indices: tuple[int, ...] | range,
    num_bits: int,
    bitorder: Literal["big", "little"],
) -> _BitPacking:
    """Return the bit packing specification for a tuple of indices."""
    return _get_bit_packing_sequence(indices, num_bits, bitorder)


@lru_cache(128)
def _get_bit_packing_contiguous_range(
    indices: range,
    num_bits: int,
    bitorder: Literal["big", "little"],
) -> _BitPacking:
    """Return the bit packing specification for a range of indices."""
    # Replace any negative indexing
    if indices.start < 0 or indices.stop < 0:
        # Need special handling for ranges like range(-K, 0) -> range(N-K, N)
        indices = range(indices.start % num_bits, indices.stop % num_bits or num_bits)

    # Get bit index positions in the full unpacked array
    # To make computation easier we flip the range around for big ordering
    # so that it refers to the packed array indexes rather than logical
    # bit indexes.
    num_bytes = min_num_bytes(num_bits)
    if bitorder == "little":
        bit_indices = indices
    else:
        bit_indices = range(8 * num_bytes - indices.stop, 8 * num_bytes - indices.start)

    # Get byte indices in that contain all the requested bits
    byte_start = bit_indices.start // 8
    byte_stop = 1 + max(bit_indices) // 8
    num_byte_indices = byte_stop - byte_start
    if bitorder == "little":
        # Need to reverse bytes and index from the end
        byte_indices = slice(num_bytes - byte_stop, num_bytes - byte_start, 1)
    else:
        byte_indices = slice(byte_start, byte_stop, 1)

    # Get bit indices within bytes slice as slice
    bit_start = bit_indices.start - 8 * byte_start
    bit_stop = bit_indices.stop - 8 * byte_start
    slice_bit_indices = slice(bit_start, bit_stop, 1)
    num_bit_indices = bit_stop - bit_start

    return _BitPacking(
        num_bits=num_bits,
        num_byte_indices=num_byte_indices,
        byte_indices=byte_indices,
        num_bit_indices=num_bit_indices,
        bit_indices=slice_bit_indices,
        bitorder=bitorder,
    )


def _pack(
    array: NDArray[np.uint8 | np.bool_], bitorder: Literal["big", "little"]
) -> NDArray[np.uint8]:
    """Pack an array of bit values into byte array.

    Args:
        array: The boolean array to bit-pack. The last axis of the input array is
            the number of bits.
        bitorder: The bitorder packing for binary arrays for the specified bits.

    Returns:
        The bit-packed array.
    """
    if bitorder == "big":
        # We need to pad the left of the input with 0 bits to get a
        # whole number of bytes before using packbits.
        packed = np.packbits(_pad_whole_bytes(array, "big"), axis=-1, bitorder="big")
    else:
        # For little bitorder np.packbits pads to bytes correctly
        # but we need to reverse the byte order of the packed array to
        # match our convention used by BitArray
        packed = np.packbits(array, axis=-1, bitorder="little")[..., ::-1]
    return packed


def _unpack(
    array: NDArray[np.uint8], bitorder: Literal["big", "little"], num_bits: int | None = None
) -> NDArray[np.uint8]:
    """Unpack a packed bit array.

    Args:
        array: The bit-packed byte array to unpack.
        bitorder: The bitorder packing for binary arrays for the specified bits.
        num_bits: Optional, the original number of bits in the packed array.

    Returns:
        The unpacked bit array.
    """
    if bitorder == "little":
        # Since BitArray's convention is the reverse of a NumPys packed
        # little we need to reverse the array before unpacking.
        array = array[..., ::-1]
    unpacked = np.unpackbits(array, axis=-1, bitorder=bitorder)
    if num_bits is not None:
        # Slice only the original number of bits from the padded unpacked
        # arrays. For "little" these are the first N bits, for "big" these
        # are the last N bits
        if bitorder == "little":
            unpacked = unpacked[..., :num_bits]
        else:
            unpacked = unpacked[..., unpacked.shape[-1] - num_bits :]
    return unpacked


def _pad_whole_bytes(bits: NDArray[np.uint8], bitorder: str) -> NDArray[np.uint8]:
    """Pad bit array size to a multiple of 8."""
    if remainder := (-bits.shape[-1]) % 8:
        pad = np.zeros((*bits.shape[:-1], remainder), dtype=bits.dtype)
        if bitorder == "big":
            bits = np.concatenate([pad, bits], axis=-1)
        else:
            bits = np.concatenate([bits, pad], axis=-1)
    return bits


def _format_indices(
    bit_length: int,
    bit_indices: Sequence[int] | slice | None,
) -> Sequence[int]:
    """Format bit indices into a range or 1D index array"""
    if bit_indices is None:
        return range(bit_length)
    if isinstance(bit_indices, slice):
        return _slice_to_range(bit_indices, bit_length)
    if isinstance(bit_indices, Integral):
        return range(bit_indices)
    return bit_indices


def _slice_to_range(value: slice, default_length: int | None = None) -> range:
    """Convert a slice to a range of the specified size.

    Args:
        value: The slice to convert to a range.
        default_length: Used to convert a slice with a stop value of None.

    Returns:
        The range equivalent to the input slice.

    Raises:
        ValueError: If the input slice and size are not compatible.
    """
    # This helper function is intended to handle ambiguity inherent
    # in slices with None for its stop (and possible other) values.
    start = 0 if value.start is None else value.start
    step = 1 if value.step is None else value.step
    if (stop := value.stop) is None:
        if default_length is None:
            raise ValueError("A slice with stop value of None requires a default length.")
        stop = start + step * default_length
    srange = range(start, stop, step)
    return srange


def _is_contiguous_range(value: Any) -> bool:
    """Return whether we can treat the bit indices as a contiguous range"""
    return isinstance(value, range) and value.step == 1


def _is_trivial_bit_packing(packing: _BitPacking) -> bool:
    """Return whether the bit_indices in the packing is a trivial packing slice."""
    indices = packing.bit_indices
    if not isinstance(indices, slice):
        return False
    # NOTE: Because of the way base _pack works a trivial bit range
    # differs for little and big bitorder. A trivial range for little
    # is a range(0, k), while a trivial range for big is range(n - k, n)
    # for n = 8*bytes for the required number of bytes to pack the bits
    if packing.bitorder == "little":
        trivial_start = 0
        trivial_stop = packing.num_bit_indices
    else:
        trivial_stop = 8 * packing.num_byte_indices
        trivial_start = trivial_stop - packing.num_bit_indices
    return (
        (indices.start is None or indices.start == trivial_start)
        and (indices.stop is None or indices.stop == trivial_stop)
        and (indices.step is None or indices.step == 1)
    )


def _is_trivial_byte_packing(packing: _BitPacking) -> bool:
    """Return whether the byte_indices in the packing is a trivial packing slice."""
    indices = packing.byte_indices
    if not isinstance(indices, slice):
        return False
    num_bytes = min_num_bytes(packing.num_bits)
    return (
        (indices.start is None or indices.start == 0)
        and (indices.stop is None or indices.stop == num_bytes)
        and (indices.step is None or indices.step == 1)
    )


# Bytes for masking the lowest number of bits in a byte for slicing
_SLICE_MASKS = np.array([0, 1, 3, 7, 15, 31, 63, 127, 255], dtype=np.uint8)

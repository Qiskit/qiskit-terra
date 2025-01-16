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

"""Tests of bit packing utility functions."""

from __future__ import annotations
from test import QiskitTestCase
import itertools as it
import ddt
import numpy as np
from numpy.typing import NDArray

from qiskit.primitives.containers.bit_packing import (
    pack_bits,
    unpack_bits,
    slice_packed_bits,
    min_num_bytes,
    _pack,
    _unpack,
    _get_bit_packing,
    _pad_whole_bytes,
)


def _switch_bitorder(bitorder: str, switch: bool) -> str:
    """Switch bitorder big <-> little"""
    if not switch:
        return bitorder
    if bitorder == "big":
        return "little"
    return "big"


def _ref_pack_bits(
    bits: NDArray[np.uint8],
    indices: range | list[int] | NDArray[np.int64],
    bitorder: str,
    num_bits: int | None = None,
) -> NDArray[np.uint8]:
    """Reference padding of a bit array on specified indices and bitorder.

    This assumes the base `_pack` function works correctly on whole byte bit
    registers.
    """
    if isinstance(indices, range):
        indices = list(indices)
    if num_bits is None:
        num_bits = 1 + np.max(indices)
    byte_bits = 8 * min_num_bytes(num_bits)
    padded = np.zeros((*bits.shape[:-1], byte_bits), dtype=np.uint8)
    if bitorder == "big":
        padded[:, indices] = bits[..., ::-1]
        padded = padded[..., ::-1]
    else:
        padded[:, indices] = bits
    return _pack(padded, bitorder)


@ddt.ddt
class BitPackingTestCase(QiskitTestCase):
    """Tests bit-packing util functions."""

    @ddt.idata(it.product(["big", "little"]))
    @ddt.unpack
    def test_base_pack_byte(self, bitorder):
        """Test base packing of whole single bytes"""
        target = np.arange(2**8, dtype=np.uint8).reshape(-1, 1)
        bits = np.array(
            [[int(char) for char in bin(i)[2:].zfill(8)] for i in range(2**8)], dtype=np.uint8
        )
        if bitorder == "little":
            bits = bits[..., ::-1]
        value = _pack(bits, bitorder=bitorder)
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [2, 8, 13, 24, 199]))
    @ddt.unpack
    def test_base_pack(self, bitorder, num_bits):
        """Test packing pads bits correctly"""
        rng = np.random.default_rng(seed=1234)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        target = _pack(_pad_whole_bytes(bits, bitorder), bitorder=bitorder)
        value = _pack(bits, bitorder=bitorder)
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [False, True], [2, 8, 13, 24, 199]))
    @ddt.unpack
    def test_base_unpack(self, bitorder, switch_order, num_bits):
        """Test packing pads bits correctly"""
        rng = np.random.default_rng(seed=1234)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        padded = _pad_whole_bytes(bits, bitorder)
        unpack_bitorder = _switch_bitorder(bitorder, switch_order)

        # Use num_bits kwarg of _unpack
        value = _unpack(_pack(padded, bitorder), unpack_bitorder, num_bits)
        target = bits[..., ::-1] if switch_order else bits
        self.assertTrue(np.all(value == target))

        # Don't use num_bits kwarg of _unpack so it returns padded
        pad_value = _unpack(_pack(padded, bitorder), unpack_bitorder)
        pad_target = padded[..., ::-1] if switch_order else padded
        self.assertTrue(np.all(pad_value == pad_target))

    @ddt.idata(it.product([False, True], ["big", "little"], [0, 1, 8, 99], [2, 8, 24, 199]))
    @ddt.unpack
    def test_get_bit_packing_range(self, as_sequence, bitorder, start_bit, num_bits):
        """Test pack bits on range(start, start + num_bits)."""
        stop_bit = num_bits + start_bit
        indices = range(start_bit, stop_bit)
        if as_sequence:
            indices = list(indices)
        packing = _get_bit_packing(indices, stop_bit, bitorder)
        # Check number of bytes needed in range
        first_byte = start_bit // 8
        last_byte = (start_bit + num_bits - 1) // 8
        num_byte_indices = 1 + last_byte - first_byte
        index_type = np.ndarray if as_sequence else slice
        self.assertIsInstance(packing.bit_indices, index_type, msg="Incorrect bit_indices type")
        self.assertIsInstance(packing.byte_indices, index_type, msg="Incorrect byte_indices type")
        self.assertEqual(packing.num_bits, start_bit + num_bits, msg="Incorrect number of bits")
        self.assertEqual(packing.num_bit_indices, num_bits, msg="Incorrect number of bit indices")
        self.assertEqual(
            packing.num_byte_indices, num_byte_indices, msg="Incorrect number of byte indices"
        )

    @ddt.idata(it.product(["big", "little"], [2, 8, 13, 24, 199]))
    @ddt.unpack
    def test_pack_bits_default(self, bitorder, num_bits):
        """Test pack bits on a default range(num_bits)"""
        rng = np.random.default_rng(seed=1234)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        target = _pack(bits, bitorder)
        value, value_bits = pack_bits(bits, bitorder=bitorder)
        self.assertEqual(value_bits, num_bits)
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [False, True], [2, 8, 13, 24, 199]))
    @ddt.unpack
    def test_unpack_bits_default(self, bitorder, switch_order, num_bits):
        """Test pack bits on a default range(num_bits)"""
        rng = np.random.default_rng(seed=1234)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        packed = _pack(bits, bitorder)
        value = unpack_bits(packed, num_bits, bitorder=_switch_bitorder(bitorder, switch_order))
        target = bits[..., ::-1] if switch_order else bits
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [0, 1, 8, 99], [2, 8, 24, 199]))
    @ddt.unpack
    def test_pack_bits_range(self, bitorder, start_bit, num_bits):
        """Test pack bits on range(start, start + num_bits)."""
        rng = np.random.default_rng(seed=1234)
        stop_bit = num_bits + start_bit
        indices = range(start_bit, stop_bit)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        target = _ref_pack_bits(bits, indices, bitorder)
        value, value_bits = pack_bits(bits, indices=indices, bitorder=bitorder)
        self.assertEqual(value_bits, stop_bit)
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [False, True], [0, 1, 8, 99], [2, 8, 24, 199]))
    @ddt.unpack
    def test_unpack_bits_range(self, bitorder, switch_order, start_bit, num_bits):
        """Test pack bits on range(start, start + num_bits)."""
        rng = np.random.default_rng(seed=1234)
        stop_bit = num_bits + start_bit
        indices = range(start_bit, stop_bit)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        packed = _ref_pack_bits(bits, indices, bitorder)
        value = unpack_bits(
            packed, indices=indices, bitorder=_switch_bitorder(bitorder, switch_order)
        )
        target = bits[..., ::-1] if switch_order else bits
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [1, 5, 8, 32, 29, 151]))
    @ddt.unpack
    def test_pack_bits_sequence(self, bitorder, num_bits):
        """Test pack bits on a random selected sequence of indices"""
        rng = np.random.default_rng(seed=12345)
        indices = rng.choice(2 * num_bits, size=num_bits, replace=False)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        target = _ref_pack_bits(bits, indices, bitorder)
        value, value_bits = pack_bits(bits, indices=indices, bitorder=bitorder)
        self.assertEqual(value_bits, 1 + np.max(indices))
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [False, True], [1, 5, 8, 32, 29, 151]))
    @ddt.unpack
    def test_unpack_bits_sequence(self, bitorder, switch_order, num_bits):
        """Test pack bits on a random selected sequence of indices"""
        rng = np.random.default_rng(seed=12345)
        indices = rng.choice(2 * num_bits, size=num_bits, replace=False)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        packed = _ref_pack_bits(bits, indices, bitorder)
        value = unpack_bits(
            packed, indices=indices, bitorder=_switch_bitorder(bitorder, switch_order)
        )
        target = bits[..., ::-1] if switch_order else bits
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [1, 5, 8, 32, 29, 151]))
    @ddt.unpack
    def test_pack_bits_negative_sequence(self, bitorder, num_bits):
        """Test pack bits on a random selected sequence of indices"""
        rng = np.random.default_rng(seed=12345)
        total_bits = 2 * num_bits
        indices = -rng.choice(total_bits, size=num_bits, replace=False)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        ref_indices = indices % total_bits
        target = _ref_pack_bits(bits, ref_indices, bitorder, num_bits=total_bits)
        value, value_bits = pack_bits(bits, indices=indices, num_bits=total_bits, bitorder=bitorder)
        self.assertEqual(value_bits, total_bits)
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [False, True], [1, 5, 8, 32, 29, 151]))
    @ddt.unpack
    def test_unpack_bits_negative_sequence(self, bitorder, switch_order, num_bits):
        """Test pack bits on a random selected sequence of indices"""
        rng = np.random.default_rng(seed=12345)
        total_bits = 2 * num_bits
        indices = -rng.choice(total_bits, size=num_bits, replace=False)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        ref_indices = indices % total_bits
        packed = _ref_pack_bits(bits, ref_indices, bitorder, num_bits=total_bits)
        value = unpack_bits(
            packed,
            indices=indices,
            num_bits=total_bits,
            bitorder=_switch_bitorder(bitorder, switch_order),
        )
        target = bits[..., ::-1] if switch_order else bits
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [1, 5, 8, 32]))
    @ddt.unpack
    def test_pack_bits_sparse_sequence(self, bitorder, num_bits):
        """Test pack bits on a random selected sequence of sparse byte block indices"""
        rng = np.random.default_rng(seed=12345)
        span = (
            list(range(16, 16 + num_bits // 3))
            + list(range(64, 64 + num_bits // 3))
            + list(range(128, 128 + num_bits // 3 + num_bits % 3))
        )
        indices = rng.choice(span, size=num_bits, replace=False)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        target = _ref_pack_bits(bits, indices, bitorder)
        value, value_bits = pack_bits(bits, indices=indices, bitorder=bitorder)
        self.assertEqual(value_bits, 1 + np.max(indices))
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [False, True], [1, 5, 8, 32]))
    @ddt.unpack
    def test_unpack_bits_sparse_sequence(self, bitorder, switch_order, num_bits):
        """Test pack bits on a random selected sequence of sparse byte block indices"""
        rng = np.random.default_rng(seed=12345)
        span = (
            list(range(16, 16 + num_bits // 3))
            + list(range(64, 64 + num_bits // 3))
            + list(range(128, 128 + num_bits // 3 + num_bits % 3))
        )
        indices = rng.choice(span, size=num_bits, replace=False)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        packed = _ref_pack_bits(bits, indices, bitorder)
        value = unpack_bits(
            packed, indices=indices, bitorder=_switch_bitorder(bitorder, switch_order)
        )
        target = bits[..., ::-1] if switch_order else bits
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [1, 3, 8, 20, 24], [1, 8, 19, 33, 200]))
    @ddt.unpack
    def test_pack_bits_strided_range(self, bitorder, stride, num_bits):
        """Test round-trip bit packing of random strided range of bits"""
        rng = np.random.default_rng(seed=12345)
        start = rng.integers(num_bits)
        indices = range(start, start + num_bits * stride, stride)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        target = _ref_pack_bits(bits, indices, bitorder)
        value, value_bits = pack_bits(bits, indices, bitorder=bitorder)
        self.assertEqual(value_bits, 1 + np.max(indices))
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [False, True], [1, 3, 8, 20, 24], [1, 8, 19, 33, 200]))
    @ddt.unpack
    def test_unpack_bits_strided_range(self, bitorder, switch_order, stride, num_bits):
        """Test round-trip bit packing of random strided range of bits"""
        rng = np.random.default_rng(seed=12345)
        start = rng.integers(num_bits)
        indices = range(start, start + num_bits * stride, stride)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        packed = _ref_pack_bits(bits, indices, bitorder)
        value = unpack_bits(
            packed, indices=indices, bitorder=_switch_bitorder(bitorder, switch_order)
        )
        target = bits[..., ::-1] if switch_order else bits
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [1, 8, 99], [2, 8, 24, 199]))
    @ddt.unpack
    def test_pack_bits_negative_range(self, bitorder, start_bit, num_bits):
        """Test pack bits on range(start, start + num_bits)."""
        rng = np.random.default_rng(seed=1234)
        total_bits = num_bits + start_bit
        ref_indices = range(start_bit, total_bits)
        indices = range(-num_bits, 0)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        target = _ref_pack_bits(bits, ref_indices, bitorder, num_bits=total_bits)
        value, _ = pack_bits(bits, indices=indices, num_bits=total_bits, bitorder=bitorder)
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [False, True], [1, 8, 99], [2, 8, 24, 199]))
    @ddt.unpack
    def test_unpack_bits_negative_range(self, bitorder, switch_order, start_bit, num_bits):
        """Test pack bits on range(start, start + num_bits)."""
        rng = np.random.default_rng(seed=1234)
        total_bits = num_bits + start_bit
        ref_indices = range(start_bit, total_bits)
        indices = range(-num_bits, 0)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        packed = _ref_pack_bits(bits, ref_indices, bitorder, num_bits=total_bits)
        value = unpack_bits(
            packed,
            indices=indices,
            num_bits=total_bits,
            bitorder=_switch_bitorder(bitorder, switch_order),
        )
        target = bits[..., ::-1] if switch_order else bits
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [1, 4, 8, 28, 199]))
    @ddt.unpack
    def test_slicing_trivial(self, bitorder, num_bits):
        """Test trivial packed bit slicing of full array"""
        rng = np.random.default_rng(seed=54321)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        target = _pack(bits, bitorder)
        value = slice_packed_bits(target)
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [0, 1, 8, 99], [1, 8, 19, 24, 33, 101]))
    @ddt.unpack
    def test_slicing_range(self, bitorder, start_bit, num_bits):
        """Test packed bit slicing of a range of bits"""
        rng = np.random.default_rng(seed=54321)
        stop_bit = num_bits + start_bit
        indices = range(start_bit, stop_bit)
        bits = rng.integers(2, size=(10, num_bits), dtype=np.uint8)
        target = _pack(bits, bitorder)
        packed, _ = pack_bits(bits, indices=indices, bitorder=bitorder)
        value = slice_packed_bits(packed, indices)
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [1, 5, 8, 32, 29, 151]))
    @ddt.unpack
    def test_slicing_sequence(self, bitorder, num_bits):
        """Test packed bit slicing of a sequence of bits"""
        rng = np.random.default_rng(seed=54321)
        indices = rng.choice(2 * num_bits, size=num_bits, replace=False)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        target = _pack(bits, bitorder)
        packed, _ = pack_bits(bits, indices=indices, bitorder=bitorder)
        value = slice_packed_bits(packed, indices)
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [1, 5, 8, 32]))
    @ddt.unpack
    def test_slicing_sparse_sequence(self, bitorder, num_bits):
        """Test packed bit slicing of a sequence bits of sparse byte blocks"""
        rng = np.random.default_rng(seed=54321)
        span = (
            list(range(16, 16 + num_bits // 3))
            + list(range(64, 64 + num_bits // 3))
            + list(range(128, 128 + num_bits // 3 + num_bits % 3))
        )
        indices = rng.choice(span, size=num_bits, replace=False)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        target = _pack(bits, bitorder)
        packed, _ = pack_bits(bits, indices=indices, bitorder=bitorder)
        value = slice_packed_bits(packed, indices)
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [1, 5, 8, 32]))
    @ddt.unpack
    def test_slicing_negative_sequence(self, bitorder, num_bits):
        """Test packed bit slicing of a sequence bits of sparse byte blocks"""
        rng = np.random.default_rng(seed=54321)
        total_bits = 2 * num_bits
        indices = -rng.choice(total_bits, size=num_bits, replace=False)
        ref_indices = indices % total_bits
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        target = _pack(bits, bitorder)
        packed, _ = pack_bits(bits, indices=ref_indices, num_bits=total_bits, bitorder=bitorder)
        value = slice_packed_bits(packed, indices, num_bits=total_bits)
        self.assertTrue(np.all(target == value))

    @ddt.idata(it.product(["big", "little"], [1, 8, 99], [2, 8, 24, 199]))
    @ddt.unpack
    def test_slicing_negative_range(self, bitorder, start_bit, num_bits):
        """Test pack bits on range(start, start + num_bits)."""
        rng = np.random.default_rng(seed=1234)
        total_bits = num_bits + start_bit
        ref_indices = range(start_bit, total_bits)
        indices = range(-num_bits, 0)
        bits = rng.integers(2, size=(1000, num_bits), dtype=np.uint8)
        target = _pack(bits, bitorder)
        packed, _ = pack_bits(bits, indices=ref_indices, num_bits=total_bits, bitorder=bitorder)
        value = slice_packed_bits(packed, indices, num_bits=total_bits)
        self.assertTrue(np.all(target == value))

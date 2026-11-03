"""Typed numeric-array archive without per-array ZIP/.npy member overhead.

This is reversible container encoding, equally applicable to every baseline and
candidate.  Array names use UTF-8;
dtype strings explicitly record endianness; shape and all C-order raw bytes are
stored.  Object/structured/non-numeric dtypes are rejected.  No pickle, digest,
fingerprint or data approximation is used.  Container bytes include all metadata.
"""
from __future__ import annotations

import math
import struct


VERSION = 1
MAGIC = b"KQAR"
_HEAD = struct.Struct("<4sHQ")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")


def _deps():
    import numpy as np
    return np


def _check_dtype(dtype):
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None or dtype.kind not in "biufc":
        raise TypeError("Packed array archive supports plain numeric/bool dtypes only")


class _Reader:
    def __init__(self, data):
        self.data, self.position = memoryview(data), 0

    def take(self, count):
        if count < 0 or count > len(self.data)-self.position:
            raise ValueError("Truncated packed array archive")
        start = self.position
        self.position += count
        return self.data[start:self.position]

    def number(self, record):
        return record.unpack(self.take(record.size))[0]


def pack_arrays(mapping):
    """Return version scalar and one uint8 payload, preserving dtype/shape/bytes.

    Named values must be ndarrays.  Their logical C-order bytes are preserved;
    input strides are not part of this array-value representation.  Keys are
    stored in ascending UTF-8 byte order.  All serialized metadata lives inside
    the returned payload; the external version scalar is an optional convenience.
    """
    np = _deps()
    entries = []
    for name, array in mapping.items():
        if not isinstance(name, str):
            raise TypeError("Packed array names must be strings")
        if not isinstance(array, np.ndarray):
            raise TypeError("Packed array values must be numpy ndarrays")
        _check_dtype(array.dtype)
        encoded_name = name.encode("utf-8", errors="strict")
        dtype_text = array.dtype.str.encode("ascii", errors="strict")
        if len(encoded_name) > (1 << 32)-1 or len(dtype_text) > 65535 or array.ndim > 64:
            raise ValueError("Packed array metadata exceeds the format limits")
        if any(size < 0 or size > (1 << 64)-1 for size in array.shape) or array.nbytes > (1 << 64)-1:
            raise ValueError("Packed array dimensions exceed the format limits")
        entries.append((encoded_name, dtype_text, array))
    entries.sort(key=lambda entry: entry[0])
    if any(entries[i][0] == entries[i-1][0] for i in range(1, len(entries))):
        raise ValueError("Duplicate packed array name")
    stream = bytearray(_HEAD.pack(MAGIC, VERSION, len(entries)))
    for name, dtype_text, array in entries:
        stream.extend(_U32.pack(len(name)))
        stream.extend(name)
        stream.extend(_U16.pack(len(dtype_text)))
        stream.extend(dtype_text)
        stream.extend(_U16.pack(array.ndim))
        for size in array.shape:
            stream.extend(_U64.pack(size))
        raw = array.tobytes(order="C")
        stream.extend(_U64.pack(len(raw)))
        stream.extend(raw)
    return dict(packed_arrays_version=np.asarray(VERSION, dtype=np.uint16),
                packed_arrays_bytes=np.frombuffer(bytes(stream), dtype=np.uint8).copy())


def unpack_arrays(actual):
    """Decode an actual uint8 archive into independent original numeric arrays.

    An outer mapping containing only packed_arrays_bytes is supported, allowing
    np.savez_compressed(..., packed_arrays_bytes=payload) to use one ZIP member.
    If packed_arrays_version is also stored it must agree with the internal tag.
    Unknown outer fields are unused; callers must still count their actual bytes.
    """
    np = _deps()
    payload = np.asarray(actual["packed_arrays_bytes"])
    if payload.dtype != np.uint8 or payload.ndim != 1:
        raise ValueError("Packed array payload must be a one-dimensional uint8 array")
    if "packed_arrays_version" in actual:
        external_version = np.asarray(actual["packed_arrays_version"])
        if external_version.shape != () or external_version.dtype.kind not in "iu" or int(external_version) != VERSION:
            raise ValueError("Unsupported external packed array archive version")
    reader = _Reader(payload.tobytes(order="C"))
    magic, version, count = _HEAD.unpack(reader.take(_HEAD.size))
    if magic != MAGIC or version != VERSION:
        raise ValueError("Unsupported packed array archive magic or version")
    if count > len(reader.data)//16:
        raise ValueError("Packed array count is inconsistent with payload length")
    result = {}
    previous_name = None
    for _ in range(count):
        name_bytes = reader.take(reader.number(_U32)).tobytes()
        if previous_name is not None and name_bytes <= previous_name:
            raise ValueError("Packed array names must be unique and ordered")
        previous_name = name_bytes
        try:
            name = name_bytes.decode("utf-8", errors="strict")
            dtype_text = reader.take(reader.number(_U16)).tobytes().decode("ascii", errors="strict")
            dtype = np.dtype(dtype_text)
        except (UnicodeError, TypeError, ValueError) as error:
            raise ValueError("Invalid packed array name or dtype metadata") from error
        _check_dtype(dtype)
        if dtype.str != dtype_text:
            raise ValueError("Packed dtype must be explicit and canonically encode endianness")
        ndim = reader.number(_U16)
        if ndim > 64:
            raise ValueError("Packed array has too many dimensions")
        shape = tuple(reader.number(_U64) for _ in range(ndim))
        nbytes = reader.number(_U64)
        expected_bytes = math.prod(shape)*dtype.itemsize
        if expected_bytes != nbytes:
            raise ValueError("Packed shape, dtype and raw byte length disagree")
        raw = reader.take(nbytes)
        try:
            result[name] = np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
        except (ValueError, OverflowError) as error:
            raise ValueError("Packed array shape cannot be represented on this runtime") from error
    if reader.position != len(reader.data):
        raise ValueError("Packed array archive contains trailing data")
    return result

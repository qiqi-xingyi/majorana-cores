"""Lossless adaptive storage for L @ R factors and sparse Pauli inputs.

Packing preserves every nonzero numerical value; no thresholding or quantization.
Array-byte choices are independent of ZIP compression. Importing this module
imports no numerical package.
"""
from pathlib import Path
import time


def _deps():
    import numpy as np
    from scipy import sparse
    return np, sparse


def _values(values, np):
    values = np.asarray(values)
    if values.dtype.kind not in 'fc':
        raise ValueError('values must have a floating or complex dtype')
    if not np.all(np.isfinite(values)):
        raise ValueError('nonfinite values cannot be stored as scientific evidence')
    if values.dtype.kind == 'c' and np.all(values.imag == 0):
        values = values.real
    return np.ascontiguousarray(values)


def _metadata(identity, n_qubits, cut_nA, kind, np):
    n, cut = int(n_qubits), int(cut_nA)
    if n != n_qubits or cut != cut_nA or not 0 < cut < n:
        raise ValueError('cut must lie inside an integer qubit register')
    identity = complex(identity)
    if not (np.isfinite(identity.real) and np.isfinite(identity.imag)):
        raise ValueError('identity must be finite')
    return dict(format_version=np.asarray(1, dtype=np.uint8),
                representation=np.asarray(kind, dtype=np.uint8),
                dimensions=np.asarray([n, cut], dtype=np.uint64),
                identity=np.asarray([identity.real, identity.imag], dtype=np.float64))


def _pack_labels(labels, width, np):
    labels = [str(s) for s in labels]
    if any(len(s) != width or any(p not in 'IXYZ' for p in s) for s in labels):
        raise ValueError('Pauli labels must match the register width and IXYZ alphabet')
    packed = np.zeros((len(labels), (width + 3)//4), dtype=np.uint8)
    for i, label in enumerate(labels):
        for q, char in enumerate(label):
            packed[i, q//4] |= 'IXYZ'.index(char) << (2*(q % 4))
    return packed


def _unpack_labels(values, width, np):
    values = np.asarray(values)
    if values.dtype != np.uint8 or values.ndim != 2 or values.shape[1] != (width+3)//4:
        raise ValueError('invalid packed label shape or dtype')
    return np.asarray([''.join('IXYZ'[(int(row[q//4]) >> (2*(q % 4))) & 3]
                               for q in range(width)) for row in values], dtype=f'U{width}')


def _bytes(arrays):
    return sum(int(value.nbytes) for value in arrays.values())


def _matrix_options(values, prefix, orientation, np, sparse):
    dense = {prefix+'_format': np.asarray(0, dtype=np.uint8), prefix+'_values': values}
    matrix = sparse.csc_matrix(values) if orientation == 'csc' else sparse.csr_matrix(values)
    # Index widths depend on actual bounds, not on SciPy's internal default.
    index_type = np.uint32 if max(values.shape, default=0) <= 2**32 else np.uint64
    pointer_type = np.uint32 if matrix.nnz < 2**32 else np.uint64
    encoded = {prefix+'_format': np.asarray(1, dtype=np.uint8),
               prefix+'_values': matrix.data,
               prefix+'_indices': matrix.indices.astype(index_type),
               prefix+'_indptr': matrix.indptr.astype(pointer_type)}
    choice = dense if _bytes(dense) <= _bytes(encoded) else encoded
    return choice, dense, encoded


def pack_factors(payload, identity, n_qubits, cut_nA):
    """Return (array mapping, metrics); left/right encode L @ R, without conjugation.

    Left chooses dense or CSC; right chooses dense or CSR independently. Real
    conversion is allowed only for an exactly zero imaginary part. Ties use dense.
    """
    np, sparse = _deps()
    started = time.perf_counter()
    arrays = _metadata(identity, n_qubits, cut_nA, 1, np)
    left, right = (_values(payload[key], np) for key in ('left', 'right'))
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError('factor dimensions must represent left @ right')
    aa = _pack_labels(payload['a_labels'], int(cut_nA), np)
    bb = _pack_labels(payload['b_labels'], int(n_qubits)-int(cut_nA), np)
    if len(aa) != left.shape[0] or len(bb) != right.shape[1]:
        raise ValueError('label counts must match factor dimensions')
    arrays.update(a_labels=aa, b_labels=bb,
                  shape=np.asarray([left.shape[0], left.shape[1], right.shape[1]], dtype=np.uint64))
    shared_bytes = _bytes(arrays)
    lc, ld, ls = _matrix_options(left, 'left', 'csc', np, sparse)
    rc, rd, rs = _matrix_options(right, 'right', 'csr', np, sparse)
    arrays.update(lc); arrays.update(rc)
    metrics = dict(storage_representation='adaptive_factors',
        left_format='dense' if int(lc['left_format']) == 0 else 'csc',
        right_format='dense' if int(rc['right_format']) == 0 else 'csr',
        left_dtype=str(left.dtype), right_dtype=str(right.dtype),
        packed_payload_bytes=_bytes(arrays), shared_metadata_and_labels_bytes=shared_bytes,
        all_dense_payload_bytes=shared_bytes+_bytes(ld)+_bytes(rd),
        all_sparse_payload_bytes=shared_bytes+_bytes(ls)+_bytes(rs),
        actual_stored_numeric_entries=int(lc['left_values'].size+rc['right_values'].size),
        factor_entries=int(left.size+right.size),
        factor_nonzero_entries=int(np.count_nonzero(left)+np.count_nonzero(right)),
        encoding_sec=time.perf_counter()-started,
        storage_scope='array payload including labels, identity, dimensions and format tags; excludes Python containers',
        lossless_numeric_values=True, factor_product='left @ right', threshold_applied=False)
    return arrays, metrics


def pack_pauli(payload, identity, n_qubits, cut_nA):
    """Pack raw sparse Pauli labels/values with the same common metadata as factors."""
    np, _ = _deps()
    started = time.perf_counter()
    arrays = _metadata(identity, n_qubits, cut_nA, 0, np)
    coefficients = _values(payload['coefficients'], np)
    labels = _pack_labels(payload['pauli_labels'], int(n_qubits), np)
    if coefficients.ndim != 1 or len(coefficients) != len(labels):
        raise ValueError('Pauli labels and coefficients must have equal length')
    common_bytes = _bytes(arrays)
    arrays.update(labels=labels, coefficients=coefficients)
    return arrays, dict(storage_representation='sparse_pauli',
        coefficient_dtype=str(coefficients.dtype), packed_payload_bytes=_bytes(arrays),
        common_metadata_bytes=common_bytes, actual_stored_numeric_entries=int(coefficients.size),
        encoding_sec=time.perf_counter()-started,
        storage_scope='array payload including labels, identity, dimensions and format tags; excludes Python containers',
        lossless_numeric_values=True, threshold_applied=False)


def unpack(arrays):
    """Decode an array mapping (including an open np.load file), preserving values."""
    np, sparse = _deps()
    if int(arrays['format_version']) != 1:
        raise ValueError('unsupported storage format version')
    n, cut = map(int, arrays['dimensions'])
    if not 0 < cut < n:
        raise ValueError('invalid recorded cut')
    identity = np.asarray(arrays['identity'])
    if identity.shape != (2,):
        raise ValueError('invalid recorded identity')
    result = dict(identity=complex(float(identity[0]), float(identity[1])), n_qubits=n, cut_nA=cut)
    kind = int(arrays['representation'])
    if kind == 0:
        result.update(pauli_labels=_unpack_labels(arrays['labels'], n, np),
                      coefficients=np.array(arrays['coefficients'], copy=True))
        if result['coefficients'].ndim != 1 or len(result['coefficients']) != len(result['pauli_labels']):
            raise ValueError('recorded Pauli dimensions disagree')
    elif kind == 1:
        a, rank, b = map(int, arrays['shape'])
        for name, shape, constructor in [('left', (a, rank), sparse.csc_matrix),
                                          ('right', (rank, b), sparse.csr_matrix)]:
            mode = int(arrays[name+'_format'])
            if mode == 0:
                value = np.array(arrays[name+'_values'], copy=True)
                if value.shape != shape:
                    raise ValueError('recorded dense factor shape disagrees')
            elif mode == 1:
                value = constructor((arrays[name+'_values'], arrays[name+'_indices'],
                                     arrays[name+'_indptr']), shape=shape).toarray()
            else:
                raise ValueError('unsupported factor encoding')
            result[name] = value
        result.update(a_labels=_unpack_labels(arrays['a_labels'], cut, np),
                      b_labels=_unpack_labels(arrays['b_labels'], n-cut, np))
        if len(result['a_labels']) != a or len(result['b_labels']) != b:
            raise ValueError('recorded label counts disagree')
    else:
        raise ValueError('unsupported representation')
    return result


def serialize(path, arrays):
    """Write a new compressed NPZ; report its actual file size, not an estimate.

    An existing file is retained: callers should use a fresh attempt directory.
    """
    np, _ = _deps()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return dict(compressed_npz_bytes=path.stat().st_size,
                serialization_sec=time.perf_counter()-started,
                packed_payload_bytes=_bytes(arrays), serialized_path=str(path))

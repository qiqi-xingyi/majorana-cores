"""Error-budgeted storage polishing of certified neutral-ID packets.

Only the projected N=2 Hermitian task P_N Hhat P_N is certified.  Rounding or
dropping common Pauli coefficients can break exact charge conservation, which
is not asserted for the polished common remainder.  Its exact coefficient L1
error is a full-Fock upper bound, usable with the original N=2 native certificate
by the triangle inequality.  No commutativity is required.

This is a storage-encoding/error-allocation control, not a new structure theorem.
"""
from __future__ import annotations
from fractions import Fraction as Q
import itertools
import math
from pathlib import Path


TARGET = Q(1, 625)
_COMMON_KEYS = ("format_version", "representation", "dimensions", "identity", "labels", "coefficients")
_SCOPE = "Projected N=2 Hermitian cached-H task P_N Hhat P_N only; common charge conservation is not asserted"


def _deps():
    import numpy as np
    import factor_storage as fs
    return np, fs


def _check(condition, message):
    if not condition:
        raise RuntimeError(message)


def _fraction(record, prefix="bound"):
    stem = prefix+"_" if prefix else ""
    return Q(int(record[stem+"numerator"]), int(record[stem+"denominator"]))


def _record(value):
    return dict(numerator=str(value.numerator), denominator=str(value.denominator),
                display_ha=float(value))


def _group(items):
    result = []
    for label, group in itertools.groupby(sorted(items, key=lambda x: x[0]), key=lambda x: x[0]):
        value = sum((entry[1] for entry in group), Q(0))
        if value:
            result.append((label, value))
    return result


def _terms(payload):
    items = []
    for label, value in zip(payload["pauli_labels"], payload["coefficients"]):
        value = complex(value)
        _check(value.imag == 0 and math.isfinite(value.real), "Real finite Pauli coefficients required")
        items.append((str(label), Q.from_float(value.real)))
    return _group(items)


def _l1_difference(first, second):
    return sum((abs(value) for _, value in
                _group(first+[(label, -value) for label, value in second])), Q(0))


def _payload(terms, identity, n):
    np, _ = _deps()
    return dict(pauli_labels=np.asarray([label for label, _ in terms], dtype="U"+str(n)),
                coefficients=np.asarray([float(value) for _, value in terms], dtype=np.float64),
                identity=identity, n_qubits=n)


def coalesce_round_common(payload, budget):
    """Coalesce exactly, round once per label, then greedily drop rounded terms.

    Selection uses rounding_L1 + dropped_rounded_L1 <= budget.  The final actual
    exact residual L1 is recomputed independently and can be smaller.
    """
    exact = _terms(payload)
    rounded = []
    rounding = Q(0)
    for label, value in exact:
        number = float(value)
        _check(math.isfinite(number), "Coalesced coefficient cannot fit float64")
        actual = Q.from_float(number)
        rounding += abs(value-actual)
        if actual:
            rounded.append((label, actual))
    _check(rounding <= budget, "One-round common error exceeds allocated budget")
    removed = []
    drop_bound = Q(0)
    for index, (label, value) in sorted(enumerate(rounded), key=lambda row: (abs(row[1][1]), row[1][0])):
        if rounding+drop_bound+abs(value) <= budget:
            removed.append(index)
            drop_bound += abs(value)
    removed.sort()
    kept, position = [], 0
    for index, term in enumerate(rounded):
        if position < len(removed) and index == removed[position]:
            position += 1
        else:
            kept.append(term)
    n = int(payload["n_qubits"])
    round_payload = _payload(rounded, payload["identity"], n)
    polish_payload = _payload(kept, payload["identity"], n)
    actual_error = _l1_difference(exact, _terms(polish_payload))
    _check(actual_error <= rounding+drop_bound <= budget, "Exact common L1 check failed")
    return round_payload, polish_payload, dict(
        source_components=len(payload["coefficients"]), source_unique_terms=len(exact),
        round_only_terms=len(rounded), polished_terms=len(kept), terms_dropped=len(removed),
        rounding_l1=_record(rounding), dropped_rounded_l1=_record(drop_bound),
        selection_triangle_bound=_record(rounding+drop_bound),
        actual_common_l1=_record(actual_error), allocated_budget=_record(budget),
        round_once_per_coalesced_label=True, charge_preservation_of_common_not_asserted=True)


def _weights(q, constant):
    return [Q(1)]*q + [2*abs(Q.from_float(float(constant)))]*(q*q-q)


def _rational_matrix(matrix):
    np, _ = _deps()
    _check(matrix.dtype == np.float64 and matrix.ndim == 2 and np.all(np.isfinite(matrix)),
           "Finite float64 factors are required")
    return [[Q.from_float(float(value)) for value in row] for row in matrix]


def _factor_perturbation_bound(left, right, altered_left, altered_right, constant, qa, qb):
    """Exact sum_k [w1(delta A_k) w1(B_k)+w1(Ahat_k) w1(delta B_k)]."""
    aa, bb, ah, bh = map(_rational_matrix, (left, right, altered_left, altered_right))
    wa, wb = _weights(qa, constant), _weights(qb, constant)
    _check(left.shape[0] == altered_left.shape[0] == qa*qa and
           right.shape[1] == altered_right.shape[1] == qb*qb,
           "Factor rows/columns disagree with one-body coordinates")
    rank = left.shape[1]
    _check(left.shape == altered_left.shape and right.shape == altered_right.shape and
           right.shape[0] == rank, "Factor shapes disagree")
    left_error = right_error = Q(0)
    for k in range(rank):
        delta_a = sum((wa[i]*abs(aa[i][k]-ah[i][k]) for i in range(len(wa))), Q(0))
        original_b = sum((wb[j]*abs(bb[k][j]) for j in range(len(wb))), Q(0))
        altered_a = sum((wa[i]*abs(ah[i][k]) for i in range(len(wa))), Q(0))
        delta_b = sum((wb[j]*abs(bb[k][j]-bh[k][j]) for j in range(len(wb))), Q(0))
        left_error += delta_a*original_b
        right_error += altered_a*delta_b
    return left_error, right_error


def prune_factors(left, right, constant, qa, qb, budget):
    """Greedy coordinate deletion with separately bounded left/right errors."""
    np, _ = _deps()
    original_a, original_b = _rational_matrix(left), _rational_matrix(right)
    wa, wb = _weights(qa, constant), _weights(qb, constant)
    _check(left.shape[0] == qa*qa and right.shape[1] == qb*qb and
           left.shape[1] == right.shape[0], "Invalid one-body factor dimensions")
    rank = left.shape[1]
    ah, bh = left.copy(), right.copy()
    b_norms = [sum((wb[j]*abs(original_b[k][j]) for j in range(qb*qb)), Q(0))
               for k in range(rank)]
    left_cost = Q(0)
    candidates = [(wa[i]*abs(original_a[i][k]), i, k)
                  for i in range(qa*qa) for k in range(rank) if original_a[i][k]]
    for weighted, i, k in sorted(candidates):
        cost = weighted*b_norms[k]
        if left_cost+cost <= budget/2:
            ah[i, k] = 0.
            left_cost += cost
    ah_exact = _rational_matrix(ah)
    a_norms = [sum((wa[i]*abs(ah_exact[i][k]) for i in range(qa*qa)), Q(0))
               for k in range(rank)]
    right_cost = Q(0)
    candidates = [(wb[j]*abs(original_b[k][j]), k, j)
                  for k in range(rank) for j in range(qb*qb) if original_b[k][j]]
    for weighted, k, j in sorted(candidates):
        cost = a_norms[k]*weighted
        if right_cost+cost <= budget/2:
            bh[k, j] = 0.
            right_cost += cost
    independent_left, independent_right = _factor_perturbation_bound(
        left, right, ah, bh, constant, qa, qb)
    _check((independent_left, independent_right) == (left_cost, right_cost) and
           left_cost+right_cost <= budget, "Independent factor perturbation bound failed")
    return ah, bh, dict(
        factor_budget=_record(budget), left_budget=_record(budget/2), right_budget=_record(budget/2),
        left_error_bound=_record(left_cost), right_error_bound=_record(right_cost),
        total_factor_error_bound=_record(left_cost+right_cost),
        left_entries_before=int(left.size), left_nonzeros_before=int(np.count_nonzero(left)),
        left_entries_after=int(ah.size), left_nonzeros_after=int(np.count_nonzero(ah)),
        right_entries_before=int(right.size), right_nonzeros_before=int(np.count_nonzero(right)),
        right_entries_after=int(bh.size), right_nonzeros_after=int(np.count_nonzero(bh)),
        coordinate_weights_diagonal=_record(Q(1)),
        coordinate_weights_offdiagonal=_record(2*abs(Q.from_float(float(constant)))),
        product_monotonicity_assumed=False,
        proof="A B - Ahat Bhat = (A-Ahat) B + Ahat (B-Bhat); each B(F) norm <= weighted coefficient L1")


def pack_coo(matrix, prefix):
    np, _ = _deps()
    _check(matrix.dtype == np.float64 and matrix.ndim == 2 and np.all(np.isfinite(matrix)),
           "Finite float64 COO matrix required")
    _check(max(matrix.shape, default=0) <= 65535, "COO shape does not fit uint16")
    rows, columns = np.nonzero(matrix)
    index_type = np.uint8 if max(matrix.shape, default=0) <= 256 else np.uint16
    return {
        prefix+"_shape": np.asarray(matrix.shape, dtype=np.uint16),
        prefix+"_row": rows.astype(index_type), prefix+"_column": columns.astype(index_type),
        prefix+"_value": np.asarray(matrix[rows, columns], dtype=np.float64),
    }


def unpack_coo(arrays, prefix):
    np, _ = _deps()
    shape = arrays[prefix+"_shape"]
    _check(shape.dtype == np.uint16 and shape.shape == (2,), "Invalid sparse shape metadata")
    shape = tuple(map(int, shape))
    rows, columns, values = (arrays[prefix+suffix] for suffix in ("_row", "_column", "_value"))
    _check(rows.ndim == columns.ndim == values.ndim == 1 and
           rows.shape == columns.shape == values.shape and
           rows.dtype in (np.dtype("uint8"), np.dtype("uint16")) and
           columns.dtype == rows.dtype and values.dtype == np.float64,
           "Invalid COO array shape/dtype")
    _check(np.all(rows < shape[0]) and np.all(columns < shape[1]) and
           np.all(np.isfinite(values)) and np.all(values != 0), "Invalid COO coordinate/value")
    if len(rows) > 1:
        order = np.lexsort((columns, rows))
        rr, cc = rows[order], columns[order]
        _check(not np.any((rr[1:] == rr[:-1]) & (cc[1:] == cc[:-1])),
               "Duplicate sparse coordinates")
    matrix = np.zeros(shape, dtype=np.float64)
    matrix[rows, columns] = values
    return matrix


def _save(path, arrays):
    np, _ = _deps()
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        loaded = {key: archive[key] for key in archive.files}
    _check(sorted(loaded) == sorted(arrays) and all(
        loaded[key].dtype == arrays[key].dtype and loaded[key].shape == arrays[key].shape and
        np.array_equal(loaded[key], arrays[key]) for key in arrays),
        "Actual NPZ round trip changed a value or metadata")
    return loaded, dict(path=str(path), payload_bytes=int(sum(value.nbytes for value in loaded.values())),
                        npz_bytes=int(Path(path).stat().st_size), actual_roundtrip_passed=True)


def _common_arrays(packet):
    return {key: packet[key] for key in _COMMON_KEYS}


def _hybrid(common, native, constant, encoding):
    np, _ = _deps()
    arrays = dict(common)
    arrays.update({"native_"+key: value for key, value in native.items()})
    arrays["offdiag_dictionary_constant"] = np.asarray(constant, dtype=np.float64)
    arrays["hybrid_neutral_encoding"] = np.asarray(encoding, dtype=np.uint8)
    return arrays


def _certificate(base, common, factor=Q(0)):
    total = base+common+factor
    return dict(base_strict_n2_upper=_record(base), common_exact_pauli_l1=_record(common),
                factor_weighted_perturbation_upper=_record(factor),
                total_strict_n2_upper=_record(total), target=_record(TARGET),
                target_pass=bool(total <= TARGET), charge_preservation_of_common_not_asserted=True,
                scope=_SCOPE, triangle_inequality_no_commutativity_assumption=True)


def _actual_common_error(packet, exact_terms, identity):
    _, fs = _deps()
    loaded = fs.unpack(_common_arrays(packet))
    _check(complex(loaded["identity"]) == complex(identity), "Actual packet identity changed")
    return _l1_difference(exact_terms, _terms(loaded))


def _pack_common(payload, n, cut):
    _, fs = _deps()
    return fs.pack_pauli(payload, payload["identity"], n, cut)[0]


def _source_payload(path):
    np, _ = _deps()
    with np.load(path, allow_pickle=False) as archive:
        return dict(n_qubits=20, pauli_labels=np.array(archive["pauli_labels"], copy=True),
                    coefficients=np.array(archive["coefficients"], copy=True),
                    identity=float(archive["identity"]))


def test_dyadic_common_coalescing_and_rounding_error():
    np, _ = _deps()
    payload = dict(n_qubits=2, identity=0.,
        pauli_labels=np.asarray(["XX", "XX", "YY", "YY", "YY"]),
        coefficients=np.asarray([1., 2.**-54, 1e16, 1., -1e16], dtype=np.float64))
    rounded, polished, info = coalesce_round_common(payload, Q(1, 1 << 50))
    assert _terms(rounded) == [("XX", Q(1)), ("YY", Q(1))]
    assert _terms(polished) == _terms(rounded)
    assert _fraction(info["rounding_l1"], "") == Q(1, 1 << 54)
    assert _l1_difference(_terms(payload), _terms(rounded)) == Q(1, 1 << 54)


def test_sparse_actual_packet_roundtrip(tmp_path):
    np, _ = _deps()
    left = np.asarray([[0., .25], [1., 0.], [-.125, 0.]], dtype=np.float64)
    right = np.asarray([[.5, 0., 0.], [0., -.75, 0.]], dtype=np.float64)
    arrays = pack_coo(left, "left")
    arrays.update(pack_coo(right, "right"))
    saved, storage = _save(tmp_path/"actual_sparse.npz", arrays)
    assert np.array_equal(unpack_coo(saved, "left"), left)
    assert np.array_equal(unpack_coo(saved, "right"), right)
    assert saved["left_shape"].dtype == np.uint16
    assert saved["left_row"].dtype == np.uint8
    assert storage["actual_roundtrip_passed"]


def _record_value(record):
    return Q(int(record["numerator"]), int(record["denominator"]))

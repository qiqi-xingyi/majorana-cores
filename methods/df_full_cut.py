"""Actual DF/CD factors to the COMPLETE Pauli-cut Majorana core.

This is not the Coulomb K spectrum.  Every local, even-cross and odd-cross
channel of the completed whole Hamiltonian is retained at the requested cut.
No ERI four-index tensor, full Pauli expansion or Fock matrix is constructed.
The first engine materializes O(q**4) Majorana channel/Gram data; it is NOT an
implementation of an O(q**3) implicit Wick contraction scheme.

Default floating compilation is a candidate only.  A separate actual-output
certificate against the exact original operator must cover all its rounding.
Optional compile_exact verifies construction rounding on bounded small inputs.
Coefficient coordinates use fixed combinadic arrays, never coefficient hashes.
"""
from __future__ import annotations

from fractions import Fraction as Q
import itertools
import math
from numbers import Integral
import time

import numpy as np

import fair_coulomb_baselines as df
import scaled_coulomb_probe as scale
import universal_core as uc
from compact_factor_queries import pair_index

NUMERIC_FIELDS = ("a2", "b2", "a4", "b4", "J11", "J13", "J31", "J22")


def _record(value):
    return dict(numerator=str(value.numerator), denominator=str(value.denominator), float=float(value))


def _choose(maximum):
    table = np.zeros((maximum+1, 5), dtype=np.int64)
    for n in range(maximum+1):
        for degree in range(5):
            table[n, degree] = math.comb(n, degree) if n >= degree else 0
    return table


def _ranks(labels, n, choose):
    """Vectorized lexicographic combinadic coordinates, with no lookup hashes."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError("Combination labels must form a two-dimensional array")
    result = np.zeros(len(labels), dtype=np.int64)
    previous = np.full(len(labels), -1, dtype=np.int64)
    degree = labels.shape[1]
    for position in range(degree):
        current = labels[:, position]
        result += choose[n-previous-1, degree-position]-choose[n-current, degree-position]
        previous = current
    return result


def _labels(n, degree):
    count = math.comb(n, degree) if n >= degree else 0
    return np.fromiter((index for item in itertools.combinations(range(n), degree) for index in item),
                       dtype=np.uint64, count=count*degree).reshape(count, degree)


def _empty_channels(n, cut, exact):
    na, nb = 2*cut, 2*(n-cut)
    dtype = object if exact else np.float64
    pa, pb = math.comb(na, 2), math.comb(nb, 2)
    ca, cb = math.comb(na, 3), math.comb(nb, 3)
    return dict(n_qubits=n, cut=cut, nA=na, nB=nb,
        a2=np.zeros(pa, dtype), b2=np.zeros(pb, dtype),
        a4_labels=_labels(na, 4), b4_labels=_labels(nb, 4),
        a4=np.zeros(math.comb(na, 4), dtype), b4=np.zeros(math.comb(nb, 4), dtype),
        J11=np.zeros((na, nb), dtype), J13=np.zeros((na, cb), dtype),
        J31=np.zeros((ca, nb), dtype), J22=np.zeros((pa, pb), dtype))


def _order(order, n):
    supplied = list(range(n)) if order is None else list(order)
    if (any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in supplied)
            or sorted(map(int, supplied)) != list(range(n))):
        raise ValueError("order must be a new-to-old permutation of every retained spin mode")
    return np.asarray(supplied, np.int64)


def _factor_matrices(packet, order, exact):
    q, constant, vacuum, h = scale.read_header(packet)
    left, _ = df.decode_packet(packet)
    if left.shape[0] != q*q:
        raise ValueError("Actual factor and header coordinate dimensions differ")
    imaginary = [pair_index(i, j, q)+1 for i in range(q) for j in range(i+1, q)]
    if imaginary and np.any(left[imaginary] != 0):
        raise ValueError("This first compiler supports real spatial F only; imaginary coordinates are nonzero")
    n, width = 2*q, left.shape[1]
    permutation = _order(order, n)
    dtype = object if exact else np.float64
    factors = np.zeros((q, q, width), dtype)
    c = Q.from_float(float(constant)) if exact else constant
    for i in range(q):
        factors[i, i] = ([Q.from_float(float(value)) for value in left[i]] if exact else left[i])
        for j in range(i+1, q):
            values = left[pair_index(i, j, q)]
            actual = ([c*Q.from_float(float(value)) for value in values] if exact else c*values)
            factors[i, j] = actual
            factors[j, i] = actual
    one_body = (np.asarray([[Q.from_float(float(value)) for value in row] for row in h], object)
                if exact else h)
    full_factors = np.zeros((n, n, width), dtype)
    full_h = np.zeros((n, n), dtype)
    full_factors[:q, :q], full_factors[q:, q:] = factors, factors
    full_h[:q, :q], full_h[q:, q:] = one_body, one_body
    full_factors = full_factors[np.ix_(permutation, permutation, np.arange(width))]
    full_h = full_h[np.ix_(permutation, permutation)]
    # Only q-by-q one-body squares are formed, never a four-index ERI tensor.
    if exact:
        counter = np.zeros((q, q), object)
        for column in range(width):
            counter += factors[:, :, column]@factors[:, :, column]
    else:
        counter = np.einsum("ijr,jkr->ik", factors, factors, optimize=True)
    full_counter = np.zeros((n, n), dtype)
    full_counter[:q, :q], full_counter[q:, q:] = counter, counter
    full_counter = full_counter[np.ix_(permutation, permutation)]
    vacuum_value = (sum((Q.from_float(float(value)) for value in vacuum), Q(0))
                    if exact else math.fsum(map(float, vacuum)))
    return q, width, permutation, factors, full_factors, full_h, full_counter, vacuum_value, left.nbytes


def _one_body_majorana(matrix, pair_table, exact):
    """Q(M)=tr(M)/2+sum a_ij (i gamma_i gamma_j), for real symmetric M."""
    n = len(matrix)
    half = Q(1, 2) if exact else .5
    result = np.zeros(math.comb(2*n, 2), object if exact else np.float64)
    for p in range(n):
        result[pair_table[2*p, 2*p+1]] = half*matrix[p, p]
        for t in range(p+1, n):
            value = half*matrix[p, t]
            result[pair_table[2*p, 2*t+1]] = value
            result[pair_table[2*p+1, 2*t]] = -value
    return half*sum((matrix[p, p] for p in range(n)), Q(0) if exact else 0.), result


def _put_quadratic(channels, values, pair_labels, choose):
    na, nb = channels["nA"], channels["nB"]
    for value, (i, j) in zip(values, pair_labels):
        i, j = int(i), int(j)
        if j < na:
            channels["a2"][uc._comb_rank((i, j), na)] = value
        elif i >= na:
            channels["b2"][uc._comb_rank((i-na, j-na), nb)] = value
        else:
            channels["J11"][i, j-na] = value


def _put_quartic_block(channels, i, j, k, l, values, choose):
    """Place one lexicographic batch with the UC twisted-left odd signs."""
    na, nb = channels["nA"], channels["nB"]
    degree_left = int(i < na)+int(j < na)+(k < na).astype(int)+(l < na).astype(int)
    for degree in range(5):
        take = degree_left == degree
        if not np.any(take):
            continue
        kk, ll, vv = k[take], l[take], values[take]
        ii, jj = np.full(len(kk), i), np.full(len(kk), j)
        if degree == 4:
            labels = np.column_stack((ii, jj, kk, ll))
            channels["a4"][_ranks(labels, na, choose)] = vv
        elif degree == 0:
            labels = np.column_stack((ii-na, jj-na, kk-na, ll-na))
            channels["b4"][_ranks(labels, nb, choose)] = vv
        elif degree == 2:
            row = uc._comb_rank((i, j), na)
            cols = _ranks(np.column_stack((kk-na, ll-na)), nb, choose)
            channels["J22"][row, cols] = vv
        elif degree == 3:
            rows = _ranks(np.column_stack((ii, jj, kk)), na, choose)
            channels["J31"][rows, ll-na] = -vv
        else:
            cols = _ranks(np.column_stack((jj-na, kk-na, ll-na)), nb, choose)
            channels["J13"][i, cols] = -vv


def _compile(packet, cut, order, exact, max_exact_monomials, max_exact_integer_ops):
    started = time.perf_counter()
    q, _, _, _ = scale.read_header(packet)
    n, m = 2*q, 4*q
    if isinstance(cut, bool) or not isinstance(cut, Integral) or not 0 < int(cut) < n:
        raise ValueError("cut must lie strictly inside the full spin-mode register")
    if exact and math.comb(m, 2)+math.comb(m, 4) > max_exact_monomials:
        raise ValueError("Exact compiler monomial resource guard exceeded")
    (q, width, permutation, factors, matrices, h, counter, vacuum,
     decoded_left_bytes) = _factor_matrices(packet, order, exact)
    pair_labels = _labels(m, 2).astype(np.int64)
    pair_table = np.full((m, m), -1, np.int64)
    pair_table[pair_labels[:, 0], pair_labels[:, 1]] = np.arange(len(pair_labels))
    dtype = object if exact else np.float64
    quadratic_factors = np.zeros((len(pair_labels), width), dtype)
    scalar_factors = np.zeros(width, dtype)
    for column in range(width):
        scalar_factors[column], quadratic_factors[:, column] = _one_body_majorana(
            matrices[:, :, column], pair_table, exact)
    h_scalar, h_quadratic = _one_body_majorana(h, pair_table, exact)
    counter_scalar, counter_quadratic = _one_body_majorana(counter, pair_table, exact)
    quadratic = h_quadratic+4*(quadratic_factors@scalar_factors)-2*counter_quadratic
    identity = vacuum+h_scalar+2*(scalar_factors@scalar_factors
        +sum((value*value for value in quadratic_factors.flat), Q(0) if exact else 0.)-counter_scalar)
    active = np.flatnonzero(np.any(quadratic_factors != 0, axis=1))
    if exact and len(active)*len(active)*width > max_exact_integer_ops:
        raise ValueError("Exact compiler arithmetic resource guard exceeded")
    owner = np.zeros(len(pair_labels), np.int64)
    owner[active] = np.arange(1, len(active)+1)
    gram = np.zeros((len(active)+1, len(active)+1), dtype)
    gram[1:, 1:] = quadratic_factors[active]@quadratic_factors[active].T
    channels = _empty_channels(n, int(cut), exact)
    choose = _choose(m)
    _put_quadratic(channels, quadratic, pair_labels, choose)
    if len(active):
        for i in range(m-3):
            for j in range(i+1, m-2):
                k, l = np.triu_indices(m-j-1, k=1)
                k, l = k+j+1, l+j+1
                aij = owner[pair_table[i, j]]
                akl = owner[pair_table[k, l]]
                aik, ajl = owner[pair_table[i, k]], owner[pair_table[j, l]]
                ail, ajk = owner[pair_table[i, l]], owner[pair_table[j, k]]
                values = 4*(gram[aij, akl]-gram[aik, ajl]+gram[ail, ajk])
                _put_quartic_block(channels, i, j, k, l, values, choose)
    error = Q(0)
    if exact:
        rounded_identity = float(identity)
        error += abs(identity-Q.from_float(rounded_identity))
        for name in NUMERIC_FIELDS:
            source = channels[name]
            actual = np.asarray(source, np.float64)
            error += sum((abs(Q(value)-Q.from_float(float(rounded)))
                          for value, rounded in zip(source.flat, actual.flat)), Q(0))
            channels[name] = actual
        identity = rounded_identity
    if not math.isfinite(float(identity)) or not all(np.all(np.isfinite(channels[name])) for name in NUMERIC_FIELDS):
        raise FloatingPointError("Nonfinite compiled full-cut coefficients")
    channels["source"] = dict(n_qubits=n, identity=float(identity),
        labels=np.asarray([], dtype="U"+str(n)), coefficients=np.empty(0, np.float64),
        not_a_materialized_Pauli_source=True)
    channels["parsing_sec"] = time.perf_counter()-started
    channels["kernel_numeric_bytes"] = sum(channels[name].nbytes for name in NUMERIC_FIELDS)
    channels["quartic_index_bytes"] = channels["a4_labels"].nbytes+channels["b4_labels"].nbytes
    metadata = dict(n_spatial_modes=q, n_qubits=n, cut=int(cut), total_factor_width=width,
        new_to_old_spin_mode_order=permutation.tolist(), all_spin_modes_retained=True,
        exact_fermionic_mode_relabelling=True, occupations_fixed=0,
        source_model="actual saved LL.T, native c, bare h and Evac, with full shared Coulomb completion",
        complete_local_even_odd_Majorana_channels=True, local_quartic_common_included=True,
        native_Coulomb_K_spectrum_substituted_for_full_cut_spectrum=False,
        floating_backend=not exact, strict_compiler_bound_available=exact,
        compiler_rounding_full_Fock_L1_upper=_record(error) if exact else None,
        compiler_bound_scope="rounding from exact actual-factor whole-H coefficients to returned float channels" if exact else None,
        independent_original_reference_actual_output_certificate_required=True,
        numeric_core_spectra_are_not_strict_intervals=True,
        no_ERI_four_index_tensor=True, no_full_Pauli_expansion=True, no_Fock_matrix=True,
        coefficient_accumulation="fixed lexicographic combinadic arrays and direct Gram/wedge algebra",
        intermediate_space_order="O(q^4) Majorana channels and active quadratic Gram; not O(q^3) implicit Wick",
        intermediate_array_bytes=dict(decoded_left=decoded_left_bytes, spatial_one_body_factors=factors.nbytes,
            spin_one_body_factors=matrices.nbytes, quadratic_Majorana_factors=quadratic_factors.nbytes,
            active_quadratic_Gram=gram.nbytes, channel_numeric=channels["kernel_numeric_bytes"],
            local_quartic_labels=channels["quartic_index_bytes"]),
        exact_object_array_nbytes_excludes_Python_integer_heap=exact,
        no_coefficient_hash_or_frozen_occupation_strategy=True,
        compile_seconds=time.perf_counter()-started)
    return dict(channels=channels, source_identity=float(identity), metadata=metadata)


def compile_packet(packet, cut, order=None):
    """Fast float candidate compilation; strict actual-output certificate is external."""
    return _compile(packet, cut, order, False, 0, 0)


def compile_exact(packet, cut, order=None, *, max_monomials=200_000, max_integer_ops=20_000_000):
    """Bounded exact-dyadic/Fraction construction with a strict final-rounding L1 bound."""
    return _compile(packet, cut, order, True, max_monomials, max_integer_ops)


def spectrum(compiled):
    """Complete numerical coefficient spectrum descriptor, with implicit zeros."""
    descriptor, core = uc.coefficient_spectrum(compiled["channels"])
    return dict(descriptor=descriptor, core=core, source_identity=compiled["source_identity"],
                metadata=compiled["metadata"])


def factor_svd(compiled):
    """Shared UC frames and factor SVD for the complete whole-H cut channels."""
    descriptor, core = uc.factor_svd(compiled["channels"])
    return descriptor, core

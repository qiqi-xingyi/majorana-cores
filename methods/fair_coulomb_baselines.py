"""Standard DF/CD controls for one finite shared-Coulomb source.

The proposed compact Coulomb factor construction already uses standard pivoted
Cholesky.  ``make_cd_packet`` therefore deliberately calls that SAME constructor
and production codec.  Its equality is a control/tie, not an algorithmic win.

DF retains positive eigenmodes of exactly the same binary64 source support
blocks and stores ONE real factor L, never independent copies of L and L.T.
``compact_df_packet`` gives DF the same trapezoidal storage opportunity by a
standard pivoted thin QR gauge change.  Its stored rounded R is a new actual
candidate and requires its own strict residual certificate.

Queries produce the complete singular spectrum of the ACTUAL supplied L L.T,
including structural zeros.  Source eigenspectra, numerical factors, QR, Gram
and answers are never cached by a prepared plan.  An exactly preformed block
dense control is available with its construction outside the timer.
"""
from __future__ import annotations

from fractions import Fraction as Q
import math
from numbers import Integral
import time

import numpy as np
from scipy import linalg

import near_geometry_factors as nf
import pivoted_cholesky_factors as pc
import coulomb_nuclear_certificate as cn
import dyadic_spectral_certificate as ds

PREFIX = "fair_df_"


def _source(kernel, support_tolerance):
    matrix, graph = pc._source(kernel, support_tolerance)
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("DF/CD comparison requires the same bitwise-symmetric canonical source")
    return matrix, graph


def _caps(max_rank, graph):
    if max_rank is None:
        return [len(rows) for rows, _ in graph]
    if isinstance(max_rank, (Integral, np.integer)) and not isinstance(max_rank, (bool, np.bool_)):
        if max_rank < 0:
            raise ValueError("Nonnegative maximum rank required")
        return [min(int(max_rank), len(rows)) for rows, _ in graph]
    values = list(max_rank)
    if len(values) != len(graph) or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (Integral, np.integer))
            or not 0 <= int(value) <= len(rows) for value, (rows, _) in zip(values, graph)):
        raise ValueError("One admissible integer rank cap per support component required")
    return list(map(int, values))


def make_source_packet(kernel, support_tolerance=1e-12):
    """Minimal shared structured full-K storage; caller adds the SAME h/Evac."""
    _source(kernel, support_tolerance)
    return pc.source_storage_packet(pc.make_source_triangle_packet(kernel, support_tolerance))


def make_cd_packet(kernel, tolerance, max_rank=None, support_tolerance=1e-12, psd_tolerance=1e-12):
    """Standard pivoted-CD alias: an intentional identical-algorithm control."""
    _source(kernel, support_tolerance)
    return pc.storage_packet(pc.make_packet(kernel, tolerance, max_rank,
        support_tolerance=support_tolerance, psd_tolerance=psd_tolerance))


def make_df_packet(kernel, eigenvalue_tolerance=0., max_rank=None,
                   support_tolerance=1e-12, psd_tolerance=1e-12):
    """Positive-eigenvalue truncated DF, one dense L per common support block.

    Tolerances are coefficient-space construction choices, NEVER Ha accuracy.
    Small negative numerical eigenvalues are dropped, with every perturbation
    retained in the certificate against the original canonical source.
    Materially indefinite blocks reject rather than inventing a PSD model.
    Construction policies/diagnostics belong in the external configuration; this
    returned dictionary contains only the minimal actual decoding information.
    """
    source, graph = _source(kernel, support_tolerance)
    tolerance, psd_tolerance = float(eigenvalue_tolerance), float(psd_tolerance)
    if (not math.isfinite(tolerance) or tolerance < 0
            or not math.isfinite(psd_tolerance) or psd_tolerance < 0):
        raise ValueError("Nonnegative finite eigenvalue and PSD tolerances required")
    maxima = _caps(max_rank, graph)
    packet = {PREFIX+"version": np.asarray(1, np.uint8),
              PREFIX+"dimension": np.asarray(len(source), np.uint16)}
    retained = 0
    for (rows, _), maximum in zip(graph, maxima):
        values, vectors = linalg.eigh(source[np.ix_(rows, rows)], check_finite=False)
        if len(values) and values[0] < -psd_tolerance:
            raise ValueError("Source block is materially indefinite for positive DF")
        indices = np.flatnonzero(values > tolerance)[::-1][:maximum]
        if not len(indices):
            continue
        left = vectors[:, indices]*np.sqrt(values[indices])[None, :]
        # Exact sign flips make equivalent construction runs deterministic;
        # they are no content hash, numerical approximation or occupation rule.
        for column in range(left.shape[1]):
            pivot = int(np.argmax(np.abs(left[:, column])))
            if left[pivot, column] < 0:
                left[:, column] *= -1
        stem = PREFIX+"block_%03d_"%retained
        packet[stem+"rows"] = rows.astype(np.uint16)
        packet[stem+"width"] = np.asarray(len(indices), np.uint16)
        packet[stem+"left_values"] = np.asarray(left, np.float64).ravel(order="C").copy()
        retained += 1
    packet[PREFIX+"count"] = np.asarray(retained, np.uint16)
    return packet


def _header(packet):
    scalars = []
    for name in ("version", "dimension", "count"):
        value = np.asarray(packet[PREFIX+name])
        if value.shape or value.dtype.kind not in "iu":
            raise ValueError("Invalid single-factor DF header")
        scalars.append(int(value))
    version, dimension, count = scalars
    if version != 1 or not 0 <= dimension <= 65535 or not 0 <= count <= dimension:
        raise ValueError("Unsupported single-factor DF header")
    return dimension, count


def _records(packet):
    dimension, count = _header(packet)
    used = []
    for index in range(count):
        stem = PREFIX+"block_%03d_"%index
        rows, width, values = (np.asarray(packet[stem+name])
                              for name in ("rows", "width", "left_values"))
        if (rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows) or np.any(rows < 0)
                or np.any(rows >= dimension) or np.any(np.diff(rows.astype(int)) <= 0)
                or any(int(row) in used for row in rows)):
            raise ValueError("Invalid, overlapping or unordered DF block coordinates")
        if (width.shape or width.dtype.kind not in "iu" or not 1 <= int(width) <= len(rows)
                or values.dtype != np.float64 or values.ndim != 1
                or len(values) != len(rows)*int(width) or not np.all(np.isfinite(values))):
            raise ValueError("Invalid actual single-factor DF block")
        used.extend(map(int, rows))
        yield rows.astype(int), int(width), values, stem


def compact_df_packet(packet):
    """Standard pivoted thin-QR gauge compression of actual DF factor bytes.

    L[p,:].T=Q R gives L[p,:] L[p,:].T=R.T R in exact arithmetic.
    Only the actual rounded trapezoidal R.T is stored.  Every floating QR
    perturbation is subject to a NEW original-source dyadic certificate.
    This is a standard gauge/encoding optimization, not a new DF algorithm.
    """
    dimension, count = _header(packet)
    result = {pc.PREFIX+"version": np.asarray(1, np.uint8),
              pc.PREFIX+"dimension": np.asarray(dimension, np.uint16),
              pc.PREFIX+"count": np.asarray(count, np.uint16)}
    for index, (rows, width, values, _) in enumerate(_records(packet)):
        left = values.reshape(len(rows), width)
        _, right, permutation = linalg.qr(left.T, mode="economic", pivoting=True, check_finite=False)
        stem = pc.PREFIX+"block_%03d_"%index
        result[stem+"rows_in_pivot_order"] = rows[permutation].astype(np.uint16)
        result[stem+"width"] = np.asarray(width, np.uint16)
        result[stem+"trapezoid_columns"] = np.concatenate(
            [right[column, column:] for column in range(width)])
    return result


def decode_packet(packet):
    if PREFIX+"version" not in packet:
        return pc.decode_packet(packet)
    dimension, _ = _header(packet)
    records = list(_records(packet))
    left = np.zeros((dimension, sum(width for _, width, _, _ in records)), np.float64)
    offset = 0
    for rows, width, values, _ in records:
        left[np.ix_(rows, np.arange(offset, offset+width))] = values.reshape(len(rows), width)
        offset += width
    return left, left.T.copy()


def prepare_query(packet):
    """Only validated structure and actual-value views survive preparation."""
    if PREFIX+"version" not in packet:
        return dict(fair_baseline_prepared_query=True, kind="standard_compact_CD_or_DF",
                    native=pc.prepare_query(packet), cached_numeric_factors=False,
                    cached_gram=False, cached_QR=False, cached_spectra=False)
    dimension, _ = _header(packet)
    blocks = [dict(size=len(rows), width=width, values=values)
              for rows, width, values, _ in _records(packet)]
    return dict(fair_baseline_prepared_query=True, kind="standard_single_L_DF",
                dimension=dimension, blocks=blocks, cached_numeric_factors=False,
                cached_gram=False, cached_QR=False, cached_spectra=False,
                dense_left_reshape_is_zero_copy_decode=True, compact_values_are_views=True,
                transient_structural_index_bytes=0)


def fresh_svd_prepared(plan, solver="gram"):
    """Fresh actual decode plus Gram/eigh, or fresh thin-QR plus small SVD.

    ``qr`` squares singular values of a freshly computed R rather than squaring
    a numerical condition number in Gram formation.  Selection of a faster
    kernel must use separate pilots and be disclosed, not pick lucky repeats.
    Both solvers return every global singular value, with zero padding.
    """
    if not plan.get("fair_baseline_prepared_query") or solver not in ("gram", "qr"):
        raise ValueError("Validated supplied-factor plan and gram/qr solver required")
    if plan["kind"] == "standard_compact_CD_or_DF":
        native = plan["native"]
        if solver == "gram":
            return pc.fresh_svd_prepared(native)
        blocks, dimension = native["blocks"], native["dimension"]
        lefts = (pc._decode_planned_block(block) for block in blocks)
    else:
        blocks, dimension = plan["blocks"], plan["dimension"]
        lefts = (block["values"].reshape(block["size"], block["width"]) for block in blocks)
    spectra = []
    for left in lefts:
        if not left.shape[1]:
            continue
        if solver == "gram":
            spectra.append(np.abs(linalg.eigvalsh(left.T@left, check_finite=False)))
        else:
            right = linalg.qr(left, mode="r", check_finite=False)[0][:left.shape[1], :]
            spectra.append(linalg.svdvals(right, check_finite=False)**2)
    return pc._spectrum_union(spectra, dimension)


def prepare_dense(packet):
    """Exact same-actual-kernel support-block control; prepare outside timer."""
    if PREFIX+"version" not in packet:
        return pc.prepare_dense(packet)
    dimension, _ = _header(packet)
    blocks = []
    for rows, width, values, _ in _records(packet):
        left = values.reshape(len(rows), width)
        blocks.append(left@left.T)
    return dict(pchol_prepared_dense=True, dimension=dimension, blocks=blocks,
                preparation="actual same single-L block products outside query timer")


def dense_svd(plan):
    return pc.dense_svd(plan)


def metadata(packet):
    if PREFIX+"version" not in packet:
        result = pc.metadata(packet)
        result.update(algorithm_family="standard DF/CD compact Gram factor",
                      renamed_CD_is_independent_algorithm=False)
        return result
    dimension, count = _header(packet)
    blocks = [dict(size=len(rows), width=width, stored_numeric_values=len(values))
              for rows, width, values, _ in _records(packet)]
    return dict(algorithm_family="standard spectral-truncated DF, single dense L",
                dimension=dimension, block_count=count, blocks=blocks,
                total_width=sum(row["width"] for row in blocks),
                actual_array_payload_bytes=sum(np.asarray(value).nbytes for value in packet.values()),
                same_compact_QR_gauge_opportunity_available=True,
                source_eigensystem_or_query_answer_cached=False)


def identical_native_arrays(first, second):
    """Direct actual-array comparison for a same-algorithm control, no hashes."""
    return sorted(first) == sorted(second) and all(
        np.asarray(first[name]).dtype == np.asarray(second[name]).dtype
        and np.asarray(first[name]).shape == np.asarray(second[name]).shape
        and np.asarray(first[name]).tobytes(order="C") == np.asarray(second[name]).tobytes(order="C")
        for name in first)


def _exact_df_residual(source, rows, records):
    real, imag, source_power = ds.float_to_dyadic(source[np.ix_(rows, rows)])
    if np.any(imag):
        raise AssertionError("Real canonical source acquired imaginary components")
    converted = []
    for actual_rows, width, values, _ in records:
        factor, imaginary, power = ds.float_to_dyadic(values.reshape(len(actual_rows), width))
        if np.any(imaginary):
            raise AssertionError("Real actual DF factor acquired imaginary components")
        local = np.searchsorted(rows, actual_rows)
        if np.any(local >= len(rows)) or not np.array_equal(rows[local], actual_rows):
            raise ValueError("Actual DF factor crosses the canonical source support graph")
        converted.append((local, factor, power))
    factor_power = max((entry[2] for entry in converted), default=0)
    width = sum(entry[1].shape[1] for entry in converted)
    factor = np.zeros((len(rows), width), dtype=object)
    offset = 0
    for local, block, power in converted:
        factor[np.ix_(local, np.arange(offset, offset+block.shape[1]))] = block*(1 << (factor_power-power))
        offset += block.shape[1]
    actual = factor@factor.T
    common_power = max(source_power, 2*factor_power)
    return real*(1 << (common_power-source_power))-actual*(1 << (common_power-2*factor_power)), common_power


def certify_packet(source_kernel, actual_packet, constant, n_electrons=2, target=Q(1, 625), *,
                   support_tolerance=1e-12, max_block_dimension=ds.MAX_DIMENSION,
                   max_integer_multiply_ops=100_000_000, max_integer_bits=4096, N=None):
    """Same strict dyadic Coulomb certificate for dense DF or compact DF/CD.

    No floating L@L.T surrogate is certified.  Actual persisted dense L bytes
    are multiplied as exact dyadic integers, using the same polar residual
    theorem and physical prefactor as ``coulomb_nuclear_certificate``.
    The caller must independently verify identical h/Evac and add any common
    ERI-to-K bridge error before accepting the declared total Ha threshold.
    """
    if PREFIX+"version" not in actual_packet:
        return cn.certify_packet(source_kernel, actual_packet, constant,
            n_electrons=n_electrons, target=target, support_tolerance=support_tolerance,
            max_block_dimension=max_block_dimension, max_integer_multiply_ops=max_integer_multiply_ops,
            max_integer_bits=max_integer_bits, N=N)
    started = time.perf_counter()
    source, graph = _source(source_kernel, support_tolerance)
    dimension = len(source)
    q = math.isqrt(dimension)
    if not q or q*q != dimension or _header(actual_packet)[0] != dimension:
        raise ValueError("Equal square spatial-coordinate dimensions required")
    if N is not None:
        if n_electrons != 2 and n_electrons != N:
            raise ValueError("Conflicting N and n_electrons")
        n_electrons = N
    if isinstance(n_electrons, bool) or not isinstance(n_electrons, Integral) or not 0 <= n_electrons <= 2*q:
        raise ValueError("Electron count outside the full retained register")
    c = np.asarray(constant)
    if c.shape or c.dtype != np.float64 or not np.isfinite(c):
        raise ValueError("Actual finite float64 dictionary constant required")
    c_exact = Q.from_float(float(c))
    if "shared_coulomb_constant" in actual_packet and Q.from_float(float(actual_packet["shared_coulomb_constant"])) != c_exact:
        raise ValueError("Actual completion header uses a different dictionary constant")
    if "shared_coulomb_spatial_modes" in actual_packet and int(actual_packet["shared_coulomb_spatial_modes"]) != q:
        raise ValueError("Actual completion header uses a different spatial-mode count")
    if isinstance(target, Integral):
        target = Q(int(target))
    if not isinstance(target, Q) or target < 0:
        raise ValueError("Nonnegative exact rational target required")
    for value in (max_block_dimension, max_integer_multiply_ops, max_integer_bits):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError("Positive integer resource guards required")
    guard = min(int(max_block_dimension), ds.MAX_DIMENSION)
    records = list(_records(actual_packet))
    owner = np.full(dimension, -1, dtype=int)
    for index, (rows, _) in enumerate(graph):
        owner[rows] = index
    groups = cn._partition_records(records, owner, len(graph))
    off_upper, off_row, off_count = cn._offblock_nuclear(source, owner)
    total, blocks, censored = off_upper, [], False
    for index, ((rows, _), group) in enumerate(zip(graph, groups)):
        n, rank = len(rows), sum(record[1] for record in group)
        operations = n*n*rank+4*n*n*n
        info = dict(index=index, dimension=n, actual_total_factor_width=rank,
                    estimated_integer_multiply_ops=operations, source_rows=rows.tolist())
        if n > guard or operations > max_integer_multiply_ops:
            info.update(evaluated=False, resource_censored=True,
                        reason="block dimension or integer-work guard exceeded")
            censored = True
            blocks.append(info)
            continue
        residual, power = _exact_df_residual(source, rows, group)
        bits = max((abs(int(value)).bit_length() for value in residual.flat), default=0)
        info.update(actual_residual_dyadic_exponent=power, actual_residual_integer_max_bits=bits,
                    residual_built_with_exact_integer_products=True)
        if bits > max_integer_bits:
            info.update(evaluated=False, resource_censored=True, reason="exact residual integer-size guard exceeded")
            censored = True
            blocks.append(info)
            continue
        if not np.any(residual):
            nuclear, witness, method = Q(0), None, "exact zero residual"
        else:
            witness = ds.certify_hermitian(residual, np.zeros_like(residual), power, target=target)
            if witness["polar_certificate_available"]:
                nuclear = Q(int(witness["trace_norm_upper_numerator"]), int(witness["trace_norm_upper_denominator"]))
                method = "sum absolute dyadic candidate eigenvalues plus dimension times exact polar eta"
            else:
                nuclear = n*Q(int(witness["bound_numerator"]), int(witness["bound_denominator"]))
                method = "dimension times independently strict spectral bound; polar witness unavailable"
        info.update(evaluated=True, resource_censored=False, exact_zero_residual_verified=not np.any(residual),
                    spectral_witness=witness, nuclear_method=method, nuclear_upper=cn._record(nuclear))
        blocks.append(info)
        total += nuclear
    mu = max(Q(1), 2*c_exact*c_exact)
    prefactor = 4*math.comb(int(n_electrons), 2)*mu
    bound = None if censored else prefactor*total
    return dict(schema="exact_dyadic_spin_shared_Coulomb_nuclear_certificate",
        evaluated=not censored, resource_censored=censored,
        target_pass=bool(bound is not None and bound <= target),
        n_spatial_modes=q, n_spin_modes=2*q, n_electrons=int(n_electrons), coefficient_dimension=dimension,
        actual_native_encoding="actual_single_dense_L_DF", source_support_tolerance=float(support_tolerance),
        source_support_block_count=len(graph), source_support_block_dimensions=[len(rows) for rows, _ in graph],
        source_symmetry_verified_bitwise=True, actual_factor_columns_confined_to_source_blocks=True,
        actual_product_computed_by_exact_dyadic_integer_arithmetic=True, residual_PSD_assumed=False,
        dictionary_constant=cn._record(c_exact), dictionary_frobenius_norm_squared_multiplier=cn._record(mu),
        physical_error_prefactor=cn._record(prefactor), offblock_source_nonzero_entries=off_count,
        offblock_exact_symmetric_row_sum_upper=cn._record(off_row), offblock_nuclear_upper=cn._record(off_upper),
        offblock_method="coefficient dimension times exact symmetric absolute row sum", block_certificates=blocks,
        nuclear_upper=cn._record(total) if not censored else None,
        physical_bound=cn._record(bound) if bound is not None else None,
        bound_numerator=str(bound.numerator) if bound is not None else None,
        bound_denominator=str(bound.denominator) if bound is not None else None,
        bound_ha=cn._record(bound)["float"] if bound is not None else None,
        target_numerator=str(target.numerator), target_denominator=str(target.denominator),
        full_fock_matrix_materialized=False, many_electron_sector_matrix_materialized=False,
        max_certified_block_dimension=guard, identical_h_and_Evac_header_required=True,
        header_error_included=False, upstream_integral_or_source_normalization_error_certified=False,
        scope="N-electron norm versus actual finite K; caller verifies identical h/Evac and any common source bridge",
        elapsed_seconds=time.perf_counter()-started)


FAIRNESS_AUDIT = dict(
    proposed_CD_and_optimized_CD_are_same_algorithm=True,
    same_actual_packet_same_query_must_be_reported_as_control_tie=True,
    standard_DF_allowed_single_L_storage=True,
    standard_DF_allowed_pivoted_QR_trapezoid_gauge=True,
    dense_and_compact_DF_are_separate_encodings_not_independent_methods=True,
    common_one_body_and_vacuum_bytes_must_be_identical=True,
    source_K_native_constant_and_support_policy_common_to_all_methods=True,
    all_selected_actual_packets_require_strict_original_source_certificate=True,
    Pauli_storage_comparison_must_include_whole_H_in_same_container=True,
    Pauli_to_C_preparation_must_not_be_charged_when_factor_preparation_is_excluded=True,
    source_spectra_and_generation_eigenvalues_excluded_from_query_cache=True,
    task_is_complete_spectrum_of_each_actual_chemically_certified_coefficient_kernel=True,
    factor_vs_own_preformed_actual_dense_is_same_matrix_comparison=True,
    different_DF_CD_actual_approximants_must_not_be_called_identical_matrices=True,
    DF_CD_algorithmic_originality_claimed=False)

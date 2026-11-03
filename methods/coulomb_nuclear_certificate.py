"""Exact-dyadic Coulomb residual certificates without an N=2 matrix.

Let F(v) use diagonal coordinates v_i and offdiagonal real/imaginary
coordinates c*v_mu. Then ||F(v)||_op <= ||F(v)||_F <= sqrt(mu)||v||_2,
mu=max(1,2*c**2). For a real symmetric coefficient residual E, its spectral
decomposition (with signed eigenvalues) and the first-quantized pair form give
  ||Delta V||_{N electrons} <= 4*binom(N,2)*mu*||E||_*.
Indeed the eigenmode lambda,v contributes 4*lambda*sum_{a<b}F(v)_a F(v)_b.
The same spatial F acts in both spin copies, with no larger operator norm.
PSD of E is NOT required. This is a posteriori error control, not a rank bound.

Every actual compact float is converted to an exact dyadic integer BEFORE
forming L L.T. Source support blocks are only a computational partition. All
discarded off-block source entries are bounded by m*||E_off||_infinity, which
dominates their nuclear norm for a symmetric m-by-m residual. Within each block,
the existing exact polar eigensystem witness supplies sum|lambda_hat|+n*eta.
No full-Fock or many-electron sector matrix is formed, and no upstream integral,
source normalization or one-body/vacuum-header equality is established here.
The caller must retain exactly the same h/Evac header on source and candidate.
"""
from __future__ import annotations
from fractions import Fraction as Q
import math
from numbers import Integral
import time

import numpy as np
import near_geometry_factors as nf
import pivoted_cholesky_factors as pc
import dyadic_spectral_certificate as ds


def _record(value):
    try:
        display = float(value)
        if not math.isfinite(display):
            display = None
    except OverflowError:
        display = None
    return dict(numerator=str(value.numerator),denominator=str(value.denominator),float=display)


def _scalar_int(packet,key):
    value = np.asarray(packet[key])
    if value.shape or value.dtype.kind not in "iu" or int(value) < 0:
        raise ValueError("Nonnegative scalar integer required: "+key)
    return int(value)


def _actual_records(packet,dimension):
    """Return validated block arrays; no decoded floating product is accepted."""
    is_factor = pc.PREFIX+"version" in packet
    is_source = pc.SOURCE+"version" in packet
    if is_factor == is_source:
        raise ValueError("Exactly one compact-Cholesky or symmetric-source native encoding required")
    if is_factor:
        actual_dimension,_ = pc._header(packet)
        if actual_dimension != dimension:
            raise ValueError("Source and actual compact dimensions differ")
        return "actual_compact_Cholesky",list(pc._records(packet))
    if _scalar_int(packet,pc.SOURCE+"version") != 1:
        raise ValueError("Unknown source-triangle version")
    if _scalar_int(packet,pc.SOURCE+"dimension") != dimension:
        raise ValueError("Source and actual source-triangle dimensions differ")
    result,used = [],[]
    for i in range(_scalar_int(packet,pc.SOURCE+"count")):
        stem = pc.SOURCE+"block_%03d_"%i
        rows = np.asarray(packet[stem+"rows"])
        values = np.asarray(packet[stem+"upper"])
        if (rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows)
                or np.any(rows < 0) or np.any(rows >= dimension)
                or np.any(np.diff(np.sort(rows)) <= 0)
                or any(int(j) in used for j in rows)):
            raise ValueError("Invalid, duplicate or overlapping source-triangle indices")
        if (values.dtype != np.float64 or values.ndim != 1
                or len(values) != len(rows)*(len(rows)+1)//2 or not np.all(np.isfinite(values))):
            raise ValueError("Invalid actual source-triangle values")
        used.extend(map(int,rows))
        result.append((rows.astype(int),None,values,stem))
    return "actual_symmetric_source_triangles",result


def _partition_records(records,owner,count):
    groups = [[] for _ in range(count)]
    for record in records:
        rows,width,_,_ = record
        if width == 0:
            continue
        labels = owner[rows]
        if np.any(labels < 0) or not np.all(labels == labels[0]):
            raise ValueError("Actual native block crosses source support blocks or inactive coordinates")
        groups[int(labels[0])].append(record)
    return groups


def _exact_block_residual(source,rows,records,encoding):
    source_real,source_imag,source_power = ds.float_to_dyadic(source[np.ix_(rows,rows)])
    if np.any(source_imag):
        raise AssertionError("Finite real source became complex")
    converted = []
    for actual_rows,width,values,_ in records:
        real,imag,power = ds.float_to_dyadic(values)
        if np.any(imag):
            raise AssertionError("Finite real native values became complex")
        local = np.searchsorted(rows,actual_rows)
        if np.any(local >= len(rows)) or not np.array_equal(rows[local],actual_rows):
            raise ValueError("Native row not in its recorded source component")
        converted.append((local,width,real,power))
    if encoding == "actual_compact_Cholesky":
        factor_power = max((entry[3] for entry in converted),default=0)
        total_width = sum(entry[1] for entry in converted)
        integer_left = np.zeros((len(rows),total_width),object)
        offset = 0
        for local,width,values,power in converted:
            cursor = 0
            for col in range(width):
                length = len(local)-col
                integer_left[local[col:],offset+col] = values[cursor:cursor+length]*(1 << (factor_power-power))
                cursor += length
            if cursor != len(values):
                raise ValueError("Actual compact trapezoid length mismatch")
            offset += width
        actual = integer_left@integer_left.T
        actual_power = 2*factor_power
    else:
        actual_power = max((entry[3] for entry in converted),default=0)
        actual = np.zeros((len(rows),len(rows)),object)
        for local,_,values,power in converted:
            upper = np.triu_indices(len(local))
            scaled = values*(1 << (actual_power-power))
            actual[(local[upper[0]],local[upper[1]])] = scaled
            actual[(local[upper[1]],local[upper[0]])] = scaled
    power = max(source_power,actual_power)
    residual = source_real*(1 << (power-source_power))-actual*(1 << (power-actual_power))
    if not np.array_equal(residual,residual.T):
        raise ArithmeticError("Exact coefficient residual is not symmetric")
    return residual,power


def _offblock_nuclear(source,owner):
    row_upper = Q(0)
    nonzero = 0
    for i in range(len(source)):
        indices = np.flatnonzero(source[i])
        total = Q(0)
        for j in indices:
            if owner[i] < 0 or owner[i] != owner[j]:
                total += abs(Q.from_float(float(source[i,j])))
                nonzero += 1
        row_upper = max(row_upper,total)
    return len(source)*row_upper,row_upper,nonzero


def certify_packet(source_kernel,actual_packet,constant,n_electrons=2,target=Q(1,625),*,
                   support_tolerance=1e-12,max_block_dimension=ds.MAX_DIMENSION,
                   max_integer_multiply_ops=100_000_000,max_integer_bits=4096,N=None):
    """Strict all-N-electron Coulomb approximation bound, conditional on same h/Evac.

    Actual native data must be a pchol minimal packet or pchol_source triangle
    packet, possibly combined with a separate common header. N is an optional
    alias for n_electrons. Source support tolerance changes only the proof's
    partition: every dropped finite entry remains in the certified residual.
    Blocks exceeding a resource guard are marked censored, never accepted.
    """
    started = time.perf_counter()
    source = np.asarray(source_kernel)
    if (source.dtype != np.float64 or source.ndim != 2 or source.shape[0] != source.shape[1]
            or not np.all(np.isfinite(source)) or not np.array_equal(source,source.T)):
        raise ValueError("Actual finite real bitwise-symmetric float64 source kernel required")
    dimension = len(source)
    q = math.isqrt(dimension)
    if q < 1 or q*q != dimension:
        raise ValueError("Coefficient dimension must be the square of spatial-mode count")
    if N is not None:
        if n_electrons != 2 and n_electrons != N:
            raise ValueError("Conflicting N and n_electrons")
        n_electrons = N
    if (isinstance(n_electrons,bool) or not isinstance(n_electrons,Integral)
            or not 0 <= int(n_electrons) <= 2*q):
        raise ValueError("Electron count must lie between zero and all retained spin modes")
    n_electrons = int(n_electrons)
    c = np.asarray(constant)
    if c.shape or c.dtype != np.float64 or not np.isfinite(c):
        raise ValueError("Actual finite float64 dictionary constant required")
    c_exact = Q.from_float(float(c))
    # If a completion header is present, its coordinate map must match exactly.
    header_constant = "shared_coulomb_constant"
    if header_constant in actual_packet and Q.from_float(float(actual_packet[header_constant])) != c_exact:
        raise ValueError("Actual completion header uses a different dictionary constant")
    if "shared_coulomb_spatial_modes" in actual_packet and int(actual_packet["shared_coulomb_spatial_modes"]) != q:
        raise ValueError("Actual completion header has a different spatial-mode count")
    if isinstance(target,Integral):
        target = Q(int(target))
    if not isinstance(target,Q) or target < 0:
        raise ValueError("Nonnegative exact rational target required")
    for value in (max_block_dimension,max_integer_multiply_ops,max_integer_bits):
        if isinstance(value,bool) or not isinstance(value,Integral) or value < 1:
            raise ValueError("Positive integer resource guards required")
    effective_dimension_guard = min(int(max_block_dimension),ds.MAX_DIMENSION)
    encoding,records = _actual_records(actual_packet,dimension)
    graph = nf.components(source,support_tolerance,symmetric=True)
    owner = np.full(dimension,-1,int)
    for index,(rows,_) in enumerate(graph):
        owner[rows] = index
    groups = _partition_records(records,owner,len(graph))
    off_upper,off_row,off_count = _offblock_nuclear(source,owner)
    total_nuclear = off_upper
    blocks,censored = [],False
    for index,((rows,_),group) in enumerate(zip(graph,groups)):
        n = len(rows)
        rank = sum(record[1] for record in group) if encoding == "actual_compact_Cholesky" else 0
        operations = n*n*rank+4*n*n*n
        info = dict(index=index,dimension=n,actual_total_factor_width=rank,
                    estimated_integer_multiply_ops=operations,source_rows=rows.tolist())
        if n > effective_dimension_guard or operations > max_integer_multiply_ops:
            info.update(evaluated=False,resource_censored=True,
                        reason="block dimension or integer-work guard exceeded")
            censored = True
            blocks.append(info)
            continue
        residual,power = _exact_block_residual(source,rows,group,encoding)
        bits = max((abs(int(value)).bit_length() for value in residual.flat),default=0)
        info.update(actual_residual_dyadic_exponent=power,actual_residual_integer_max_bits=bits,
                    residual_built_with_exact_integer_products=True)
        if bits > max_integer_bits:
            info.update(evaluated=False,resource_censored=True,reason="exact residual integer-size guard exceeded")
            censored = True
            blocks.append(info)
            continue
        if not np.any(residual):
            nuclear = Q(0)
            info.update(exact_zero_residual_verified=True,spectral_witness=None,
                        nuclear_method="exact zero residual")
        else:
            witness = ds.certify_hermitian(residual,np.zeros_like(residual),power,target=target)
            if witness["polar_certificate_available"]:
                nuclear = Q(int(witness["trace_norm_upper_numerator"]),int(witness["trace_norm_upper_denominator"]))
                method = "sum absolute dyadic candidate eigenvalues plus dimension times exact polar eta"
            else:
                nuclear = n*Q(int(witness["bound_numerator"]),int(witness["bound_denominator"]))
                method = "dimension times independently strict spectral bound; polar witness unavailable"
            info.update(exact_zero_residual_verified=False,spectral_witness=witness,nuclear_method=method)
        info.update(evaluated=True,resource_censored=False,nuclear_upper=_record(nuclear))
        total_nuclear += nuclear
        blocks.append(info)
    mu = max(Q(1),2*c_exact*c_exact)
    prefactor = 4*math.comb(n_electrons,2)*mu
    nuclear = None if censored else total_nuclear
    bound = None if censored else prefactor*total_nuclear
    result = dict(schema="exact_dyadic_spin_shared_Coulomb_nuclear_certificate",
        evaluated=not censored,resource_censored=censored,target_pass=bool(bound is not None and bound <= target),
        n_spatial_modes=q,n_spin_modes=2*q,n_electrons=n_electrons,coefficient_dimension=dimension,
        actual_native_encoding=encoding,source_support_tolerance=float(support_tolerance),
        source_support_block_count=len(graph),source_support_block_dimensions=[len(rows) for rows,_ in graph],
        source_symmetry_verified_bitwise=True,actual_factor_columns_confined_to_source_blocks=True,
        actual_product_computed_by_exact_dyadic_integer_arithmetic=True,residual_PSD_assumed=False,
        dictionary_constant=_record(c_exact),dictionary_frobenius_norm_squared_multiplier=_record(mu),
        physical_error_prefactor=_record(prefactor),offblock_source_nonzero_entries=off_count,
        offblock_exact_symmetric_row_sum_upper=_record(off_row),offblock_nuclear_upper=_record(off_upper),
        offblock_method="coefficient dimension times exact symmetric absolute row sum",
        block_certificates=blocks,nuclear_upper=(_record(nuclear) if nuclear is not None else None),
        physical_bound=(_record(bound) if bound is not None else None),
        bound_numerator=(str(bound.numerator) if bound is not None else None),
        bound_denominator=(str(bound.denominator) if bound is not None else None),
        bound_ha=(_record(bound)["float"] if bound is not None else None),
        target_numerator=str(target.numerator),target_denominator=str(target.denominator),
        full_fock_matrix_materialized=False,many_electron_sector_matrix_materialized=False,
        max_certified_block_dimension=effective_dimension_guard,
        identical_h_and_Evac_header_required=True,header_error_included=False,
        upstream_integral_or_source_normalization_error_certified=False,
        scope="N-electron norm of the spin-shared Coulomb operator from actual finite coefficient source minus actual packet; conditional on identical h and Evac header; no upstream/source-normalization guarantee",
        elapsed_seconds=time.perf_counter()-started)
    return result

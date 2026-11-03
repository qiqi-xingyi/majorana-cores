"""Pivoted-Cholesky factors stored as actual trapezoidal arrays.

Rows are recorded IN PIVOT ORDER, so no second permutation array is needed.
The packet stores n*r-r*(r-1)/2 doubles per block, never a dense left factor.
Generation diagnostics do not certify chemical accuracy or exact PSD of an
upstream cache; the original-H complete-sector certificate remains mandatory.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import linalg
import near_geometry_factors as nf

PREFIX = "pchol_"
SOURCE = "pchol_source_"


def _source(kernel, support_tolerance):
    matrix = np.asarray(kernel)
    if (matrix.dtype != np.float64 or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix))):
        raise ValueError("Actual finite real float64 square kernel is required")
    if not math.isfinite(float(support_tolerance)) or support_tolerance < 0:
        raise ValueError("Finite nonnegative support tolerance required")
    if len(matrix) > 65535:
        raise ValueError("Global coordinate indices exceed uint16 encoding")
    return matrix, nf.components(matrix,support_tolerance,symmetric=True)


def _pivoted(block,tolerance,maximum,psd_tolerance):
    """Finite floating construction, with explicit rejection/shift policy."""
    symmetric = .5*block+.5*block.T
    eigenvalues = linalg.eigvalsh(symmetric,check_finite=False)
    minimum = float(eigenvalues[0]) if len(eigenvalues) else 0.
    if minimum < -psd_tolerance:
        raise ValueError("Source block is materially indefinite: minimum eigenvalue "
                         +repr(minimum)+" below -psd_tolerance="+repr(psd_tolerance))
    # A tiny negative eigendiagnostic is not silently clipped or treated as PSD.
    # Construct a shifted candidate instead; all of this perturbation is retained
    # in the final original-source physical residual.
    shift = -minimum+psd_tolerance if minimum < 0 else 0.
    target = symmetric+shift*np.eye(len(block))
    n = len(block)
    order = np.arange(n)
    diagonal = np.diag(target).copy()
    left = np.zeros((n,maximum),np.float64)
    width = 0
    for k in range(maximum):
        pivot = k+int(np.argmax(diagonal[k:]))
        if diagonal[pivot] <= tolerance:
            break
        if pivot != k:
            order[[k,pivot]] = order[[pivot,k]]
            diagonal[[k,pivot]] = diagonal[[pivot,k]]
            left[[k,pivot],:] = left[[pivot,k],:]
        if diagonal[k] <= 0:
            raise ArithmeticError("A nonpositive pivot survived a positive stopping test")
        left[k,k] = math.sqrt(float(diagonal[k]))
        if k+1 < n:
            column = target[order[k+1:],order[k]]-left[k+1:,:k]@left[k,:k]
            left[k+1:,k] = column/left[k,k]
            diagonal[k+1:] -= left[k+1:,k]**2
            if float(np.min(diagonal[k+1:])) < -max(psd_tolerance,1e-14)*max(1.,float(linalg.norm(target))):
                raise ArithmeticError("Pivoted-Cholesky residual lost numerical PSD beyond stated policy")
        diagonal[k] = 0
        width = k+1
    left = left[:,:width]
    values = np.concatenate([left[j:,j] for j in range(width)]) if width else np.empty(0,np.float64)
    if len(values) != n*width-width*(width-1)//2:
        raise AssertionError("Trapezoidal storage count is inconsistent")
    residual = target[np.ix_(order,order)]-left@left.T
    diagnostic = dict(minimum_source_eigenvalue_float=minimum,diagonal_shift=shift,
                      numerical_residual_max_diagonal=float(max(np.diag(residual),default=0.)),
                      shifted_source_residual_fro=float(linalg.norm(residual)),
                      original_symmetric_source_residual_fro=float(linalg.norm(
                          symmetric[np.ix_(order,order)]-left@left.T)),
                      original_source_skew_fro=float(linalg.norm(block-block.T)))
    return order,values,width,diagnostic


def make_packet(kernel,tolerance,max_rank=None,support_tolerance=1e-12,psd_tolerance=1e-12):
    """Actual compact candidate; tolerance is residual-diagonal stopping, NOT Ha.

    max_rank=None means no per-block cap, an integer caps every block, and an
    integer sequence specifies caps separately. Small numerical negative source
    eigenvalues trigger an explicitly stored diagonal shift; larger ones reject.
    No theorem of low numerical rank or exact PSD is inferred from this screen.
    """
    matrix,graph = _source(kernel,support_tolerance)
    tolerance,psd_tolerance = float(tolerance),float(psd_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0 or not math.isfinite(psd_tolerance) or psd_tolerance < 0:
        raise ValueError("Nonnegative finite stopping/PSD tolerances required")
    if max_rank is None:
        maxima = [len(rr) for rr,_ in graph]
    elif isinstance(max_rank,(int,np.integer)) and not isinstance(max_rank,(bool,np.bool_)):
        if max_rank < 0:
            raise ValueError("Nonnegative rank cap required")
        maxima = [min(int(max_rank),len(rr)) for rr,_ in graph]
    else:
        maxima = list(max_rank)
        if len(maxima) != len(graph) or any(
                not isinstance(r,(int,np.integer)) or isinstance(r,(bool,np.bool_))
                or not 0 <= r <= len(rr) for r,(rr,_) in zip(maxima,graph)):
            raise ValueError("One admissible integer cap per support block required")
        maxima = list(map(int,maxima))
    packet = {PREFIX+"version":np.asarray(1,np.uint8),
              PREFIX+"dimension":np.asarray(len(matrix),np.uint16),
              PREFIX+"count":np.asarray(len(graph),np.uint16),
              PREFIX+"stop_tolerance":np.asarray(tolerance,np.float64),
              PREFIX+"support_tolerance":np.asarray(support_tolerance,np.float64),
              PREFIX+"psd_tolerance":np.asarray(psd_tolerance,np.float64)}
    for i,((rows,cols),maximum) in enumerate(zip(graph,maxima)):
        order,values,width,diagnostic = _pivoted(
            matrix[np.ix_(rows,cols)],tolerance,maximum,psd_tolerance)
        stem = PREFIX+"block_%03d_"%i
        packet[stem+"rows_in_pivot_order"] = rows[order].astype(np.uint16)
        packet[stem+"width"] = np.asarray(width,np.uint16)
        packet[stem+"trapezoid_columns"] = values
        # These fields are counted in storage and describe every source change.
        for name,number in diagnostic.items():
            packet[stem+name] = np.asarray(number,np.float64)
    return packet


def _header(packet):
    version = np.asarray(packet[PREFIX+"version"])
    dimension,count = np.asarray(packet[PREFIX+"dimension"]),np.asarray(packet[PREFIX+"count"])
    if (version.shape or version.dtype.kind not in "iu" or int(version) != 1
            or dimension.shape or dimension.dtype.kind not in "iu"
            or count.shape or count.dtype.kind not in "iu"):
        raise ValueError("Invalid pivoted-Cholesky packet header")
    return int(dimension),int(count)


def _records(packet):
    dimension,count = _header(packet)
    used = []
    for i in range(count):
        stem = PREFIX+"block_%03d_"%i
        rows = np.asarray(packet[stem+"rows_in_pivot_order"])
        width_record = np.asarray(packet[stem+"width"])
        values = np.asarray(packet[stem+"trapezoid_columns"])
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
            raise ValueError("Invalid pivot-ordered coordinate indices")
        rows = rows.astype(int)
        if (np.any(rows < 0) or np.any(rows >= dimension) or np.any(np.diff(np.sort(rows)) <= 0)
                or any(int(j) in used for j in rows)):
            raise ValueError("Duplicate, overlapping or out-of-bounds factor rows")
        used.extend(map(int,rows))
        if width_record.shape or width_record.dtype.kind not in "iu":
            raise ValueError("Invalid factor width scalar")
        width = int(width_record)
        if not 0 <= width <= len(rows):
            raise ValueError("Factor width exceeds block size")
        expected = len(rows)*width-width*(width-1)//2
        if values.dtype != np.float64 or values.ndim != 1 or len(values) != expected or not np.all(np.isfinite(values)):
            raise ValueError("Invalid actual compact trapezoid values")
        yield rows,width,values,stem


def decode_block(rows,width,values):
    """Required fresh compact decode; implicit above-pivot entries are exactly zero."""
    left = np.zeros((len(rows),width),np.float64)
    cursor = 0
    for col in range(width):
        count = len(rows)-col
        left[col:,col] = values[cursor:cursor+count]
        cursor += count
    if cursor != len(values):
        raise ValueError("Compact trapezoid length mismatch")
    return left


def decode_packet(packet):
    """Compatibility output for actual full-H certification, never cached by fresh_svd."""
    dimension,_ = _header(packet)
    records = list(_records(packet))
    total = sum(width for _,width,_,_ in records)
    left = np.zeros((dimension,total),np.float64)
    offset = 0
    for rows,width,values,_ in records:
        left[np.ix_(rows,np.arange(offset,offset+width))] = decode_block(rows,width,values)
        offset += width
    return left,left.T.copy()


def storage_packet(packet):
    """Minimal lossless production factor mapping, with no construction diagnostics.

    Save metadata(packet) to the screening JSON BEFORE calling this function.
    The diagonal shift is already represented in the stored numerical factor;
    decoding and original-source certification depend only on the actual factor.
    """
    _header(packet)
    result = {PREFIX+name:packet[PREFIX+name] for name in ("version","dimension")}
    retained = 0
    for _,width,_,stem in _records(packet):
        if not width:
            continue
        production_stem = PREFIX+"block_%03d_"%retained
        for name in ("rows_in_pivot_order","width","trapezoid_columns"):
            result[production_stem+name] = packet[stem+name]
        retained += 1
    result[PREFIX+"count"] = np.asarray(retained,np.uint16)
    return result


def _spectrum_union(spectra,dimension):
    joined = np.concatenate(spectra) if spectra else np.empty(0,float)
    if len(joined) > dimension:
        raise ArithmeticError("Compact block spectra exceed global output size")
    return np.sort(np.concatenate((joined,np.zeros(dimension-len(joined)))))[::-1]


def fresh_svd(packet):
    """Every query includes compact decode, new small Gram, eigensolve and padding.

    For exact real L, nonzero singular values of PSD L L.T are eigenvalues of
    L.T L. Floating Gram formation can lose accuracy near zero; independently
    verify the ENTIRE returned spectrum against the same actual dense operator.
    No cached L, QR, Gram, Cholesky source pivots or eigenvalues are used.
    """
    dimension,_ = _header(packet)
    spectra = []
    for rows,width,values,_ in _records(packet):
        left = decode_block(rows,width,values)
        if width:
            gram = left.T@left
            spectra.append(np.abs(linalg.eigvalsh(gram,check_finite=False)))
    return _spectrum_union(spectra,dimension)


def prepare_query(packet):
    """Validate once; retain ONLY structure indices and actual compact value views.

    No L, Gram, dense block or eigenvalue is prepared. Mutating existing value
    arrays is visible to subsequent queries; replacing arrays requires a new
    plan. This optional supplied-packet plan must be prepared outside timing
    for both methods. Its temporary index-array RAM is reported separately
    from the actual serialized archive payload.
    """
    dimension,_ = _header(packet)
    blocks = []
    for rows,width,values,_ in _records(packet):
        n = len(rows)
        positions = (np.concatenate([np.arange(j,n,dtype=np.intp)*width+j
                                     for j in range(width)])
                     if width else np.empty(0,np.intp))
        blocks.append(dict(size=n,width=width,positions=positions,values=values))
    return dict(pchol_prepared_query=True,dimension=dimension,blocks=blocks,
                cached_numeric_factors=False,cached_gram=False,cached_spectra=False,
                compact_values_are_views=True,
                transient_structural_index_bytes=sum(b["positions"].nbytes for b in blocks))


def _decode_planned_block(block):
    left = np.zeros((block["size"],block["width"]),np.float64)
    left.ravel()[block["positions"]] = block["values"]
    return left


def fresh_svd_prepared(plan):
    """Timed query: fresh compact decode, Gram, eigenvalues, sort and full padding."""
    if not plan.get("pchol_prepared_query",False):
        raise ValueError("Prepare a validated compact query structure plan first")
    spectra = []
    for block in plan["blocks"]:
        left = _decode_planned_block(block)
        if block["width"]:
            spectra.append(np.abs(linalg.eigvalsh(left.T@left,check_finite=False)))
    return _spectrum_union(spectra,plan["dimension"])


def prepare_dense(packet):
    """Strong same-actual-Ck block PSD dense baseline, materialization OUTSIDE timer."""
    dimension,_ = _header(packet)
    blocks = []
    for rows,width,values,_ in _records(packet):
        left = decode_block(rows,width,values)
        blocks.append(left@left.T)
    return dict(pchol_prepared_dense=True,dimension=dimension,blocks=blocks,
                preparation="actual compact decode plus per-block L L.T, outside timed dense query")


def dense_svd(prepared):
    if not prepared.get("pchol_prepared_dense",False):
        raise ValueError("Prepare actual dense blocks before timing")
    spectra = [np.abs(linalg.eigvalsh(block,check_finite=False)) for block in prepared["blocks"] if len(block)]
    return _spectrum_union(spectra,prepared["dimension"])


def metadata(packet):
    """Actual factor cost vs source symmetric triangle, with rank-gap accounting."""
    dimension,count = _header(packet)
    blocks = []
    for rows,width,values,stem in _records(packet):
        n = len(rows)
        dense_count = n*(n+1)//2
        deficit = n-width
        expected_saving = deficit*(deficit+1)//2
        if dense_count-len(values) != expected_saving:
            raise ArithmeticError("Trapezoid-versus-triangle count identity failed")
        blocks.append(dict(size=n,width=width,stored_numeric_values=len(values),
                           source_triangle_numeric_values=dense_count,
                           numerical_payload_saving_bytes=8*expected_saving,
                           rank_deficit=deficit,
                           row_permutation_reuses_global_index_array=True,
                           separate_permutation_array_stored=False,
                           diagonal_shift=(float(packet[stem+"diagonal_shift"])
                                           if stem+"diagonal_shift" in packet else None),
                           minimum_source_eigenvalue_float=(float(packet[stem+"minimum_source_eigenvalue_float"])
                                           if stem+"minimum_source_eigenvalue_float" in packet else None),
                           construction_diagnostics={name:(float(packet[stem+name])
                                                    if stem+name in packet else None)
                               for name in ("minimum_source_eigenvalue_float","diagonal_shift",
                                   "numerical_residual_max_diagonal","shifted_source_residual_fro",
                                   "original_symmetric_source_residual_fro","original_source_skew_fro")}))
    return dict(dimension=dimension,block_count=count,blocks=blocks,
                total_width=sum(row["width"] for row in blocks),
                actual_array_payload_bytes=sum(value.nbytes for key,value in packet.items() if key.startswith(PREFIX)),
                total_numeric_saving_vs_source_triangle_bytes=sum(
                    row["numerical_payload_saving_bytes"] for row in blocks),
                exact_PSD_source_certified=False,
                approximation_requires_original_source_physical_certificate=True,
                construction_tolerances={name:(float(packet[PREFIX+name])
                                               if PREFIX+name in packet else None)
                                        for name in ("stop_tolerance","support_tolerance","psd_tolerance")},
                construction_diagnostics_in_packet=all(
                    PREFIX+"block_%03d_diagonal_shift"%i in packet for i in range(count)),
                full_rank_numeric_saving_is_zero=True,
                fresh_query_compact_decode_in_timer=True)


def make_source_triangle_packet(kernel,support_tolerance=1e-12):
    """Actual structured SOURCE storage control; sym/off-support error still needs cert."""
    matrix,graph = _source(kernel,support_tolerance)
    packet = {SOURCE+"version":np.asarray(1,np.uint8),
              SOURCE+"dimension":np.asarray(len(matrix),np.uint16),
              SOURCE+"count":np.asarray(len(graph),np.uint16),
              SOURCE+"support_tolerance":np.asarray(support_tolerance,np.float64)}
    for i,(rows,cols) in enumerate(graph):
        stem = SOURCE+"block_%03d_"%i
        symmetric = .5*matrix[np.ix_(rows,cols)]+.5*matrix[np.ix_(rows,cols)].T
        packet[stem+"rows"] = rows.astype(np.uint16)
        packet[stem+"upper"] = symmetric[np.triu_indices(len(rows))].copy()
    return packet


def decode_source_triangle(packet):
    if int(packet[SOURCE+"version"]) != 1:
        raise ValueError("Unknown source-triangle format")
    dimension = int(packet[SOURCE+"dimension"])
    result = np.zeros((dimension,dimension),np.float64)
    for i in range(int(packet[SOURCE+"count"])):
        stem = SOURCE+"block_%03d_"%i
        rows = np.asarray(packet[stem+"rows"],int)
        sub = np.zeros((len(rows),len(rows)),np.float64)
        sub[np.triu_indices(len(rows))] = packet[stem+"upper"]
        sub = sub+sub.T-np.diag(np.diag(sub))
        result[np.ix_(rows,rows)] = sub
    return result


def source_storage_packet(packet):
    """Minimal source-triangle production mapping; screen policy belongs in JSON."""
    if int(packet[SOURCE+"version"]) != 1:
        raise ValueError("Unknown source-triangle format")
    result = {SOURCE+name:packet[SOURCE+name] for name in ("version","dimension","count")}
    for i in range(int(packet[SOURCE+"count"])):
        stem = SOURCE+"block_%03d_"%i
        for name in ("rows","upper"):
            result[stem+name] = packet[stem+name]
    return result


"""Fermionic mode permutations and block factor primitives.

The permutation is exact on finite stored Pauli coefficients and retains every
mode/occupation. Support thresholding and symmetrization are approximations whose
physical error must be certified against the original cached Hamiltonian.
"""
from __future__ import annotations
import argparse
import itertools
import math
from pathlib import Path
import numpy as np
from scipy import linalg
import theory_channels as tc

PREFIX = "near_factor_"


def alpha_beta_order(q):
    """new_to_old for a spin-interleaved, even-sized spin-orbital register."""
    if not isinstance(q, (int, np.integer)) or q < 2 or q % 2:
        raise ValueError("An even positive number of retained spin orbitals is required")
    return list(range(0, q, 2))+list(range(1, q, 2))


def _parity(sequence):
    return -1 if sum(sequence[i] > sequence[j] for i in range(len(sequence))
                     for j in range(i+1, len(sequence))) % 2 else 1


def mode_permute(case, new_to_old):
    """Return permuted case, metadata; char0 is mode0, never Qiskit label order.

    If U a_old_i^dagger U^dagger = a_new_old_to_new[i]^dagger, the output is
    U H U^dagger, including the fermionic reordering signs. Every coefficient
    is copied or sign-flipped, with no threshold, summation, hash, or rounding.
    """
    n = int(case["n_qubits"])
    supplied_order = list(new_to_old)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           for i in supplied_order):
        raise ValueError("Integer mode indices required")
    order = [int(i) for i in supplied_order]
    if sorted(order) != list(range(n)):
        raise ValueError("new_to_old must be a permutation of all retained modes")
    inverse = [0]*n
    for new, old in enumerate(order):
        inverse[old] = new
    labels, coefficients = [], []
    signs = []
    for label, coefficient in zip(case["pauli_labels"], case["coefficients"]):
        indices, old_phase = tc.pauli_majorana(str(label))
        mapped = [2*inverse[j//2]+j % 2 for j in indices]
        parity = _parity(mapped)
        new_label, new_phase = tc.majorana_pauli(tuple(sorted(mapped)), n)
        sign = complex(parity*new_phase/old_phase)
        if sign.imag != 0 or sign.real not in (-1, 1):
            raise ArithmeticError("Hermitian fermionic mode permutation has a non-real sign")
        labels.append(new_label)
        coefficients.append(coefficient*int(sign.real))
        signs.append(int(sign.real))
    result = dict(case)
    result["pauli_labels"] = np.asarray(labels, dtype="U"+str(n))
    result["coefficients"] = np.asarray(coefficients, dtype=np.asarray(case["coefficients"]).dtype)
    result["name"] = str(case.get("name", "cached"))+"_mode_permuted"
    if not np.array_equal(np.abs(result["coefficients"]), np.abs(case["coefficients"])):
        raise ArithmeticError("Signed permutation altered an actual coefficient magnitude")
    metadata = dict(new_to_old=order, old_to_new=inverse, label_character_zero_is_mode_zero=True,
                    fermionic_majorana_inversion_signs_included=True,
                    coefficient_sign_changes=sum(s < 0 for s in signs),
                    identity_unchanged=True, modes_retained=n, orbitals_removed=0,
                    no_occupation_fixed=True, no_scf_or_orbital_rotation=True,
                    exact_signed_coefficient_map=True,
                    scope="Exact unitary fermionic relabelling of the saved finite Hamiltonian")
    return result, metadata


def components(kernel, support_tolerance=0.0, symmetric=False):
    """Connected support components; no content hashes or inferred symmetries.

    Generic components use a bipartite row/column graph. symmetric=True uses
    an undirected single-index graph on a square matrix, allowing a signed
    symmetric one-factor representation. Cross-component source entries,
    including every nonzero entry below tolerance, remain error to certify.
    """
    matrix = np.asarray(kernel)
    if matrix.ndim != 2 or np.iscomplexobj(matrix) or not np.all(np.isfinite(matrix)):
        raise ValueError("Finite real two-dimensional kernel required")
    tolerance = float(support_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Nonnegative finite support tolerance required")
    m, n = matrix.shape
    if symmetric and m != n:
        raise ValueError("Signed symmetric components require a square kernel")
    size = m if symmetric else m+n
    parents = list(range(size))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def join(i, j):
        ri, rj = root(i), root(j)
        if ri != rj:
            parents[max(ri, rj)] = min(ri, rj)
    active = [False]*size
    rows, columns = np.nonzero(np.abs(matrix) > tolerance)
    for i, j in zip(rows, columns):
        x, y = int(i), int(j) if symmetric else m+int(j)
        active[x] = active[y] = True
        join(x, y)
    labels = sorted((root(i), i) for i in range(size) if active[i])
    output = []
    for _, group in itertools.groupby(labels, key=lambda row: row[0]):
        indices = [i for _, i in group]
        if symmetric:
            rr = np.asarray(indices, dtype=np.int64)
            cc = rr.copy()
        else:
            rr = np.asarray([i for i in indices if i < m], dtype=np.int64)
            cc = np.asarray([i-m for i in indices if i >= m], dtype=np.int64)
        if len(rr) and len(cc):
            output.append((rr, cc))
    return output


def block_shapes(kernel, support_tolerance=0.0, right_signs=None):
    return [(len(rows), len(cols)) for rows, cols in
            components(kernel, support_tolerance, symmetric=right_signs is not None)]


def _widths(ranks, shapes):
    if ranks is None:
        return [min(shape) for shape in shapes]
    if isinstance(ranks, (float, np.floating)):
        if not math.isfinite(float(ranks)) or not 0 <= ranks <= 1:
            raise ValueError("A relative retained width must lie in [0,1]")
        return [int(math.ceil(float(ranks)*min(shape))) for shape in shapes]
    if isinstance(ranks, (int, np.integer)) and not isinstance(ranks, (bool, np.bool_)):
        if ranks < 0:
            raise ValueError("Nonnegative per-block width required")
        return [min(int(ranks), min(shape)) for shape in shapes]
    widths = list(ranks)
    if len(widths) != len(shapes) or any(
            isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer))
            or not 0 <= int(r) <= min(shape) for r, shape in zip(widths, shapes)):
        raise ValueError("One admissible integer retained width per support block is required")
    return list(map(int, widths))


def make_block_packet(kernel, ranks=None, support_tolerance=0.0, right_signs=None):
    """Return NPZ arrays under near_factor_ only, compatible with other codecs.

    ranks=None means full width of every retained support block. An integer is
    a per-block cap, a float in [0,1] a per-block relative width, and a list
    specifies the width separately. right_signs=None uses independent SVD L/R;
    a fixed +/-1 vector requests one-factor signed symmetric eigendecomposition.
    No symmetry, threshold error, or low-rank accuracy is accepted implicitly.
    """
    matrix = np.asarray(kernel)
    if matrix.dtype != np.float64 or matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("Provide the actual real float64 coefficient kernel")
    m, n = matrix.shape
    signs = None
    if right_signs is not None:
        signs = np.asarray(right_signs)
        if m != n or signs.shape != (n,) or not np.all((signs == 1) | (signs == -1)):
            raise ValueError("Signed symmetry requires a square kernel and fixed +/-1 signs")
        signs = signs.astype(np.int8)
    graph = components(matrix, support_tolerance, symmetric=signs is not None)
    shapes = [(len(rr), len(cc)) for rr, cc in graph]
    widths = _widths(ranks, shapes)
    if max(m, n, len(graph)) > 65535:
        raise ValueError("Dimensions do not fit the declared uint16 packet")
    packet = {
        PREFIX+"version": np.asarray(1, np.uint8),
        PREFIX+"shape": np.asarray([m, n], np.uint16),
        PREFIX+"count": np.asarray(len(graph), np.uint16),
        PREFIX+"support_tolerance": np.asarray(support_tolerance, np.float64),
    }
    offblock = matrix.copy()
    for i, ((rows, columns), width) in enumerate(zip(graph, widths)):
        stem = PREFIX+"block_%03d_" % i
        block = matrix[np.ix_(rows, columns)]
        offblock[np.ix_(rows, columns)] = 0
        packet[stem+"rows"] = rows.astype(np.uint16)
        packet[stem+"columns"] = columns.astype(np.uint16)
        packet[stem+"width"] = np.asarray(width, np.uint16)
        if signs is None:
            u, s, vh = linalg.svd(block, full_matrices=False, check_finite=False)
            packet[stem+"kind"] = np.asarray(0, np.uint8)
            packet[stem+"left"] = np.asarray(u[:, :width]*s[:width], np.float64)
            packet[stem+"right"] = np.asarray(vh[:width], np.float64)
        else:
            d = signs[columns]
            symmetric = block*d[None, :]
            symmetric = 0.5*symmetric+0.5*symmetric.T
            eigenvalues, vectors = linalg.eigh(symmetric, check_finite=False)
            order = sorted(range(len(eigenvalues)),
                           key=lambda j: (-abs(float(eigenvalues[j])), -float(eigenvalues[j]), j))[:width]
            selected = eigenvalues[order]
            left = vectors[:, order]*np.sqrt(np.abs(selected))[None, :]
            packet[stem+"kind"] = np.asarray(1, np.uint8)
            packet[stem+"left"] = np.asarray(left, np.float64)
            packet[stem+"eigen_signs"] = np.where(selected < 0, -1, 1).astype(np.int8)
            packet[stem+"right_signs"] = d
    packet[PREFIX+"offblock_source_fro_diagnostic"] = np.asarray(linalg.norm(offblock), np.float64)
    left, right = decode_block_packet(packet)
    packet[PREFIX+"complete_source_residual_fro_diagnostic"] = np.asarray(
        linalg.norm(matrix-left@right), np.float64)
    return packet


def _header(packet):
    version, shape, count = packet[PREFIX+"version"], packet[PREFIX+"shape"], packet[PREFIX+"count"]
    if np.asarray(version).shape or int(version) != 1:
        raise ValueError("Unsupported block-factor packet version")
    shape = np.asarray(shape)
    if shape.shape != (2,) or shape.dtype.kind not in "iu":
        raise ValueError("Invalid global block-factor shape")
    if np.asarray(count).shape or int(count) < 0:
        raise ValueError("Invalid block count")
    return tuple(map(int, shape)), int(count)


def _read_blocks(packet, check_finite=True, decode_shared_right=True):
    shape, count = _header(packet)
    occupied_rows, occupied_cols = [], []
    for i in range(count):
        stem = PREFIX+"block_%03d_" % i
        rr, cc = np.asarray(packet[stem+"rows"]), np.asarray(packet[stem+"columns"])
        width, kind = int(packet[stem+"width"]), int(packet[stem+"kind"])
        left = np.asarray(packet[stem+"left"])
        if rr.ndim != 1 or cc.ndim != 1 or rr.dtype.kind not in "iu" or cc.dtype.kind not in "iu":
            raise ValueError("Invalid support index arrays")
        rows, cols = rr.astype(int), cc.astype(int)
        if not len(rows) or not len(cols) or not 0 <= width <= min(len(rows), len(cols)):
            raise ValueError("Invalid block dimensions/width")
        if (np.any(np.diff(rows) <= 0) or np.any(np.diff(cols) <= 0)
                or np.any(rows >= shape[0]) or np.any(cols >= shape[1])):
            raise ValueError("Unsorted, duplicate, or out-of-bounds block coordinates")
        if any(int(r) in occupied_rows for r in rows) or any(int(c) in occupied_cols for c in cols):
            raise ValueError("Support blocks overlap in a global row/column")
        occupied_rows.extend(map(int, rows))
        occupied_cols.extend(map(int, cols))
        if left.dtype != np.float64 or left.shape != (len(rows), width):
            raise ValueError("Invalid actual stored block left factor")
        if kind == 0:
            right = np.asarray(packet[stem+"right"])
            if right.dtype != np.float64 or right.shape != (width, len(cols)):
                raise ValueError("Invalid actual stored block right factor")
        elif kind == 1:
            sig = np.asarray(packet[stem+"eigen_signs"])
            ds = np.asarray(packet[stem+"right_signs"])
            if (not np.array_equal(rows, cols) or sig.shape != (width,) or ds.shape != (len(cols),)
                    or not np.all((sig == 1) | (sig == -1)) or not np.all((ds == 1) | (ds == -1))):
                raise ValueError("Invalid signed one-factor block")
            right = sig[:, None]*left.T*ds[None, :] if decode_shared_right else None
        else:
            raise ValueError("Unknown block representation")
        if check_finite and not (np.all(np.isfinite(left)) and
                                 (right is None or np.all(np.isfinite(right)))):
            raise ValueError("Nonfinite actual stored factor")
        yield rows, cols, left, right, kind, stem


def packet_metadata(packet):
    shape, count = _header(packet)
    blocks = []
    for rr, cc, left, right, kind, stem in _read_blocks(packet):
        blocks.append(dict(rows=rr.tolist(), columns=cc.tolist(), shape=[len(rr), len(cc)],
                           width=left.shape[1], encoding="signed_one_factor" if kind else "independent_LR"))
    return dict(shape=list(shape), block_count=count, blocks=blocks,
                total_retained_width=sum(b["width"] for b in blocks),
                total_dense_block_entries=sum(math.prod(b["shape"]) for b in blocks),
                stored_array_bytes=sum(x.nbytes for key, x in packet.items() if key.startswith(PREFIX)),
                actual_source_residual_fro_diagnostic=float(packet[PREFIX+"complete_source_residual_fro_diagnostic"]),
                physical_accuracy_certified=False)


def decode_block_packet(packet):
    """CONTROL/compatibility decode to global L/R; production spectrum needs no decode."""
    shape, _ = _header(packet)
    blocks = list(_read_blocks(packet))
    width = sum(left.shape[1] for _, _, left, *_ in blocks)
    left = np.zeros((shape[0], width), np.float64)
    right = np.zeros((width, shape[1]), np.float64)
    offset = 0
    for rows, cols, ll, rr, _, _ in blocks:
        count = ll.shape[1]
        left[np.ix_(rows, np.arange(offset, offset+count))] = ll
        right[np.ix_(np.arange(offset, offset+count), cols)] = rr
        offset += count
    return left, right


def _assemble_spectrum(values, shape):
    count = min(shape)
    joined = np.concatenate(values) if values else np.empty(0, float)
    if len(joined) > count:
        raise ArithmeticError("Block spectra exceed the global singular-value output size")
    return np.sort(np.concatenate([joined, np.zeros(count-len(joined))]))[::-1]


def prepare_block_dense(packet):
    """Fair baseline preparation, OUTSIDE the timed supplied-factor SVD query."""
    shape, _ = _header(packet)
    dense = []
    for rows, cols, left, right, kind, stem in _read_blocks(packet):
        sub = left@right
        symmetric = sub*packet[stem+"right_signs"][None, :] if kind else None
        valid = True
        skew = 0.0
        if symmetric is not None:
            skew = float(linalg.norm(symmetric-symmetric.T))
            valid = skew <= 1e-12*max(1.0, float(linalg.norm(symmetric)))
        dense.append(dict(matrix=sub, symmetric_matrix=symmetric, shape=[len(rows), len(cols)],
                          symmetric_skew_fro_diagnostic=skew,
                          symmetric_baseline_control_valid=valid))
    return dict(prepared_dense_blocks=True, shape=shape, blocks=dense,
                source="same actual decoded block products; materialization excluded from query timer")


def block_dense_svd(prepared, exploit_symmetry=False):
    """Strong same-task structure baseline on PREMATERIALIZED actual blocks.

    Pass prepare_block_dense(packet) once before timing. exploit_symmetry=True
    uses dense symmetric eigenvalues on signed blocks, standard SVD otherwise.
    """
    if not prepared.get("prepared_dense_blocks", False):
        raise ValueError("Pre-materialize with prepare_block_dense outside the query timer")
    spectra = []
    for block in prepared["blocks"]:
        if exploit_symmetry and block["symmetric_matrix"] is not None:
            # Floating product construction need not be bitwise symmetric.
            # Do not silently average its skew. A proposed eigensolver baseline
            # is valid only after a separately bounded negligible skew check.
            symmetric = block["symmetric_matrix"]
            if not block["symmetric_baseline_control_valid"]:
                raise ArithmeticError("Actual dense signed block is materially asymmetric")
            spectra.append(np.sort(np.abs(linalg.eigvalsh(symmetric, check_finite=False)))[::-1])
        else:
            spectra.append(linalg.svdvals(block["matrix"], check_finite=False))
    return _assemble_spectrum(spectra, prepared["shape"])


def support_diagnostics(kernel, tolerances=(0.0, 1e-15, 1e-12, 1e-9, 1e-6),
                        right_signs=None):
    """Descriptive support and spectra only; final physical certificates required."""
    rows = []
    for tolerance in tolerances:
        graph = components(kernel, tolerance, symmetric=right_signs is not None)
        discarded = np.array(kernel, copy=True)
        blocks = []
        for rr, cc in graph:
            block = kernel[np.ix_(rr, cc)]
            singular = linalg.svdvals(block, check_finite=False)
            tails = np.sqrt(np.concatenate([np.cumsum((singular[::-1]**2))[::-1], [0.0]]))
            blocks.append(dict(rows=rr.tolist(), columns=cc.tolist(),
                               shape=[len(rr), len(cc)], singular_values=singular.tolist(),
                               frobenius_best_rank_tails=tails.tolist()))
            discarded[np.ix_(rr, cc)] = 0
        rows.append(dict(support_tolerance=float(tolerance), blocks=blocks,
                         block_count=len(graph), sum_dense_entries=sum(
                             len(rr)*len(cc) for rr, cc in graph),
                         discarded_offblock_fro_diagnostic=float(linalg.norm(discarded)),
                         physical_operator_error_certified=False))
    return rows

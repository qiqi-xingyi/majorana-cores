"""Finite two-basis H2 panel for a shared whole-Coulomb primitive.

H2 at 0.74 Angstrom, aug-cc-pVDZ and cc-pVTZ, all spherical AOs retained.
No SCF or occupation restriction is performed.  The finite reference is an
explicitly pair-symmetrized real binary64 K with the ACTUAL native constant c.
Source and factors share exactly the same binary h/Evac header.  A strict
dyadic nuclear-norm certificate controls the full two-electron Coulomb error
without constructing a Pauli expansion or a many-electron sector matrix.
"""
from __future__ import annotations
from fractions import Fraction as Q
import itertools
import math
from pathlib import Path
import time

import numpy as np
from scipy import linalg
from io_helpers import write_json, write_rows
import compact_packet_polish as cp
import packed_array_archive as pa
import near_geometry_factors as nf
import pivoted_cholesky_factors as pc
import coulomb_nuclear_certificate as cnc
from compact_factor_queries import pair_index

TARGET = Q(1, 625)
CONSTANT = 1/math.sqrt(2)
BASES = (("aug-cc-pVDZ", 18), ("cc-pVTZ", 28))
PIVOT_TOLERANCES = (0., 1e-9, 1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
HEADER = "shared_coulomb_"


def _pairs(q):
    return [(i, i, i) for i in range(q)]+[(i, j, pair_index(i, j, q))
        for i in range(q) for j in range(i+1, q)]


def coulomb_kernel(eri, constant=CONSTANT):
    """Return bitwise-symmetric K, using K_mu,nu=(pq|rs)/(4*s_mu*s_nu).

    s_diag=1 and s_off=c because F_off=c(E_pq+E_qp).  Using 1/c in this
    denominator would be a different, incorrectly normalized physical operator.
    The eight chemists-integral permutations are averaged explicitly, and each
    symmetric K pair is assigned from one shared floating result.
    """
    eri = np.asarray(eri)
    if (eri.dtype != np.float64 or eri.ndim != 4 or not all(size == eri.shape[0] for size in eri.shape)
            or not np.all(np.isfinite(eri))):
        raise ValueError("Finite real float64 square four-index ERI required")
    q = eri.shape[0]
    c = float(constant)
    if q < 1 or not math.isfinite(c) or c == 0:
        raise ValueError("Positive mode dimension and finite nonzero c required")
    entries = _pairs(q)
    kernel = np.zeros((q*q, q*q), dtype=np.float64)
    maximum_spread = 0.
    for first, (p, r, mu) in enumerate(entries):
        smu = 1. if p == r else c
        for s, t, nu in entries[first:]:
            snu = 1. if s == t else c
            values = [float(eri[p, r, s, t]), float(eri[r, p, s, t]),
                      float(eri[p, r, t, s]), float(eri[r, p, t, s]),
                      float(eri[s, t, p, r]), float(eri[t, s, p, r]),
                      float(eri[s, t, r, p]), float(eri[t, s, r, p])]
            value = math.fsum(values)/8.
            maximum_spread = max(maximum_spread, max(values)-min(values))
            coefficient = value/(4*smu*snu)
            kernel[mu, nu] = kernel[nu, mu] = coefficient
    if not np.all(np.isfinite(kernel)) or not np.array_equal(kernel, kernel.T):
        raise ArithmeticError("Finite bitwise-symmetric Coulomb reference construction failed")
    return kernel, dict(chemists_integral_order="(p q | r s)",
        eightfold_pair_average_explicit=True, maximum_input_permutation_spread=maximum_spread,
        native_real_offdiagonal_F_constant=c, kernel_normalization="ERI/(4*s_mu*s_nu), s_diag=1,s_off=c",
        imaginary_coordinate_rows_exactly_zero=True, finite_source_defined_by_actual_K=True,
        upstream_integral_or_pair_averaging_error_certified=False)


def tensor_from_kernel(kernel, constant=CONSTANT):
    """Independent-use floating reconstruction of the finite spatial ERI tensor."""
    matrix = np.asarray(kernel)
    q = math.isqrt(len(matrix))
    if matrix.shape != (q*q, q*q):
        raise ValueError("Kernel dimension must be q squared")
    entries = _pairs(q)
    tensor = np.zeros((q, q, q, q), dtype=np.float64)
    for p, r, mu in entries:
        for s, t, nu in entries:
            value = 4*(1. if p == r else constant)*(1. if s == t else constant)*matrix[mu, nu]
            for index in ((p,r,s,t),(r,p,s,t),(p,r,t,s),(r,p,t,s),
                          (s,t,p,r),(t,s,p,r),(s,t,r,p),(t,s,r,p)):
                tensor[index] = value
    return tensor


def make_header(h, vacuum, constant=CONSTANT):
    h = np.asarray(h)
    if (h.dtype != np.float64 or h.ndim != 2 or h.shape[0] != h.shape[1]
            or not np.all(np.isfinite(h)) or not np.array_equal(h, h.T)):
        raise ValueError("Actual real bitwise-symmetric one-body h required")
    q = len(h)
    if not 1 <= q <= 65535 or not math.isfinite(float(vacuum)) or not math.isfinite(float(constant)):
        raise ValueError("Finite header values and encoded spatial dimension required")
    return {HEADER+"version": np.asarray(1, np.uint8),
            HEADER+"spatial_modes": np.asarray(q, np.uint16),
            HEADER+"constant": np.asarray(constant, np.float64),
            HEADER+"vacuum_components": np.asarray([vacuum], np.float64),
            HEADER+"h_upper": h[np.triu_indices(q)].copy(),
            HEADER+"semantics": np.asarray(1, np.uint8)}


def identical_header(reference, actual):
    for name, value in reference.items():
        candidate = np.asarray(actual[name])
        if (candidate.dtype != value.dtype or candidate.shape != value.shape
                or candidate.tobytes(order="C") != value.tobytes(order="C")):
            raise ArithmeticError("Actual source/factor h/Evac header differs: "+name)
    return True


def read_header(packet):
    """Decode the scalable whole-operator header without a Fock-space matrix."""
    if int(packet[HEADER+"version"]) != 1 or int(packet[HEADER+"semantics"]) != 1:
        raise ValueError("Unsupported scalable Coulomb header semantics")
    q = int(packet[HEADER+"spatial_modes"])
    c = float(packet[HEADER+"constant"])
    vacuum = np.asarray(packet[HEADER+"vacuum_components"])
    upper = np.asarray(packet[HEADER+"h_upper"])
    if (not 1 <= q <= 65535 or not math.isfinite(c) or c == 0
            or vacuum.dtype != np.float64 or vacuum.ndim != 1 or not np.all(np.isfinite(vacuum))
            or upper.dtype != np.float64 or upper.shape != (q*(q+1)//2,) or not np.all(np.isfinite(upper))):
        raise ValueError("Invalid scalable actual one-body/vacuum header")
    h = np.zeros((q, q), dtype=np.float64)
    indices = np.triu_indices(q)
    h[indices] = upper
    h[(indices[1], indices[0])] = upper
    return q, c, vacuum.copy(), h


def build_source(basis, folder=None):
    """All-AO symmetric Lowdin construction, with no SCF or orbital removal."""
    import pyscf
    from pyscf import gto, ao2mo
    started = time.monotonic()
    mol = gto.M(atom=[("H", (0., 0., 0.)), ("H", (0., 0., .74))],
                basis=basis, unit="Angstrom", charge=0, spin=0, cart=False, verbose=0)
    q = mol.nao_nr()
    if mol.nelectron != 2:
        raise AssertionError("Unexpected H2 electron count")
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=np.float64)
    h_ao = mol.intor_symmetric("int1e_kin")+mol.intor_symmetric("int1e_nuc")
    values, vectors = linalg.eigh(overlap, check_finite=False)
    if np.min(values) <= 0:
        raise ArithmeticError("Nonpositive overlap; orbital deletion is prohibited")
    x = (vectors/np.sqrt(values)[None, :])@vectors.T
    raw_h = x.T@h_ao@x
    h = np.empty_like(raw_h)
    for i in range(q):
        for j in range(i, q):
            h[i, j] = h[j, i] = .5*raw_h[i, j]+.5*raw_h[j, i]
    eri = np.asarray(ao2mo.kernel(mol, x, compact=False), dtype=np.float64).reshape((q,)*4)
    if not all(np.all(np.isfinite(array)) for array in (x, h, eri)):
        raise FloatingPointError("Nonfinite all-orbital transformed input")
    kernel, normalization = coulomb_kernel(eri, CONSTANT)
    header = make_header(h, float(mol.energy_nuc()), CONSTANT)
    metadata = dict(basis=basis, spatial_modes=q, n_qubits=2*q, n_electrons=2,
        distance_angstrom=.74, orbital_basis="symmetric Lowdin S**(-1/2), every spherical AO",
        scf_performed=False, canonical_orbitals_used=False, orbitals_removed=0,
        occupations_fixed=0, overlap_minimum_eigenvalue=float(values[0]),
        h_pair_averaging_max_input_asymmetry=float(np.max(np.abs(raw_h-raw_h.T))),
        normalization=normalization, pyscf_version=pyscf.__version__,
        build_seconds=time.monotonic()-started,
        full_sector_dimension=math.comb(2*q, 2), full_sector_materialized=False,
        input_and_basis_error_certified=False)
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/"input_reference.npz", overlap_ao=overlap,
            hcore_ao=h_ao, lowdin_x=x, hcore_lowdin=h, eri_lowdin=eri,
            reference_kernel=kernel, vacuum_energy=np.asarray(mol.energy_nuc()))
        write_json(folder/"input_metadata.json", metadata)
    return dict(kernel=kernel, header=header, h=h, eri=eri, metadata=metadata)


def load_actual(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return pa.unpack_arrays(arrays) if "packed_arrays_bytes" in arrays else arrays


def save_encodings(folder, stem, header, native):
    folder.mkdir(parents=True, exist_ok=True)
    arrays = dict(native)
    arrays.update(header)
    records = []
    for container in ("ordinary", "packed"):
        encoded = arrays if container == "ordinary" else dict(
            packed_arrays_bytes=pa.pack_arrays(arrays)["packed_arrays_bytes"])
        actual_outer, storage = cp._save(folder/(stem+"_"+container+".npz"), encoded)
        actual = actual_outer if container == "ordinary" else pa.unpack_arrays(actual_outer)
        if sorted(actual) != sorted(arrays):
            raise ArithmeticError("Actual archive changed the declared array names")
        for name, value in arrays.items():
            if (actual[name].dtype != value.dtype or actual[name].shape != value.shape
                    or actual[name].tobytes(order="C") != value.tobytes(order="C")):
                raise ArithmeticError("Actual whole-Coulomb archive changed a value or metadata")
        identical_header(header, actual)
        records.append(dict(container=container, storage=storage,
                            exact_bytes_roundtrip_verified=True, identical_h_and_Evac_verified=True))
    return records


def best_storage(records):
    return dict(payload=min(records, key=lambda row: row["storage"]["payload_bytes"]),
                npz=min(records, key=lambda row: row["storage"]["npz_bytes"]))


def floating_screen(kernel, native, graph, off_upper):
    """Sufficient floating diagnostic only, not the final exact certificate."""
    left, right = pc.decode_packet(native)
    nuclear = off_upper
    for rows, _ in graph:
        residual = kernel[np.ix_(rows, rows)]-left[rows]@right[:, rows]
        residual = .5*residual+.5*residual.T
        nuclear += float(np.sum(np.abs(linalg.eigvalsh(residual, check_finite=False))))
    return 4*max(1., 2*CONSTANT**2)*nuclear


def screen_candidates(kernel):
    graph = nf.components(kernel, 1e-12, symmetric=True)
    owner = np.full(len(kernel), -1, dtype=int)
    for index, (rows, _) in enumerate(graph):
        owner[rows] = index
    off_row = 0.
    for i in range(len(kernel)):
        off_row = max(off_row, math.fsum(abs(float(kernel[i, j])) for j in range(len(kernel))
                     if owner[i] < 0 or owner[i] != owner[j]))
    off_upper = len(kernel)*off_row
    options, screens = [], []
    for tolerance in PIVOT_TOLERANCES:
        name = "pivot_"+str(tolerance)
        try:
            generated = pc.make_packet(kernel, tolerance=tolerance,
                support_tolerance=1e-12, psd_tolerance=1e-12)
        except ValueError as error:
            if "materially indefinite" not in str(error):
                raise
            screens.append(dict(candidate=name, tolerance=tolerance,
                                scientifically_inapplicable=True, reason=str(error)))
            continue
        diagnostics = pc.metadata(generated)
        native = pc.storage_packet(generated)
        error = floating_screen(kernel, native, graph, off_upper)
        row = dict(candidate=name, tolerance=tolerance, total_width=diagnostics["total_width"],
            native_payload_bytes=sum(value.nbytes for value in native.values()),
            float_physical_nuclear_bound_ha=error, rigorous=False,
            construction_diagnostics=diagnostics)
        screens.append(row)
        options.append((row, native))
    selected = []
    for threshold in (.0011, .0015):
        valid = [pair for pair in options if pair[0]["float_physical_nuclear_bound_ha"] <= threshold]
        if valid:
            choice = min(valid, key=lambda pair: (pair[0]["native_payload_bytes"], pair[0]["candidate"]))
            if not any(old[0]["candidate"] == choice[0]["candidate"] for old in selected):
                selected.append(choice)
    return screens, selected


def benchmark_factor(packet):
    """Complete same-Ck spectra; compact decode is INSIDE every fresh query."""
    plan = pc.prepare_query(packet)
    left, right = pc.decode_packet(packet)
    dense = left@right
    structured = pc.prepare_dense(packet)
    expected = linalg.svdvals(dense, check_finite=False)
    comparisons = {}
    for baseline in ("global_dense", "structured_dense"):
        pc.fresh_svd_prepared(plan)
        (linalg.svdvals(dense, check_finite=False) if baseline == "global_dense" else pc.dense_svd(structured))
        pairs = []
        for repeat in range(5):
            measured = {}
            order = ("factor", "baseline") if repeat % 2 == 0 else ("baseline", "factor")
            for method in order:
                start = time.perf_counter()
                values = (pc.fresh_svd_prepared(plan) if method == "factor" else
                          linalg.svdvals(dense, check_finite=False) if baseline == "global_dense" else
                          pc.dense_svd(structured))
                measured[method] = time.perf_counter()-start
                if not np.allclose(values, expected, rtol=2e-10, atol=2e-12):
                    raise ArithmeticError("Complete actual-Ck singular spectra differ")
            pairs.append(dict(repeat=repeat, order=list(order), **measured,
                              ratio=measured["baseline"]/measured["factor"]))
        comparisons[baseline] = dict(repeats=5, pairs=pairs,
            median_paired_ratio=float(np.median([row["ratio"] for row in pairs])))
    return dict(task="complete singular values of the same actual supplied Ck",
        precision="float64", shape=list(dense.shape), comparisons=comparisons,
        fresh_compact_decode_and_Gram_inside_every_query=True,
        source_pivots_or_L_or_Gram_or_spectra_cached=False,
        transient_structural_index_bytes=plan["transient_structural_index_bytes"],
        exact_same_actual_support_blocks_for_strong_dense=True,
        dense_controls_prepared_outside_timer=True,
        generation_certificate_header_IO_excluded=True)


def _comparison(best, source_best):
    return dict(payload_ratio=best["payload"]["storage"]["payload_bytes"]/source_best["payload"]["storage"]["payload_bytes"],
                npz_ratio=best["npz"]["storage"]["npz_bytes"]/source_best["npz"]["storage"]["npz_bytes"])


def run_case(basis, expected_modes, folder):
    folder.mkdir()
    source = build_source(basis, folder)
    kernel, header = source["kernel"], source["header"]
    q = len(source["h"])
    if q != expected_modes:
        raise AssertionError("Unexpected spherical all-AO basis dimension")
    native_source = pc.source_storage_packet(pc.make_source_triangle_packet(kernel, support_tolerance=1e-12))
    source_encodings = save_encodings(folder/"source", "whole_source", header, native_source)
    actual_source = load_actual(source_encodings[0]["storage"]["path"])
    identical_header(header, actual_source)
    source_cert = cnc.certify_packet(kernel, actual_source, np.float64(CONSTANT),
                                    n_electrons=2, target=TARGET, max_block_dimension=190)
    source_best = best_storage(source_encodings)
    source_record = dict(certificate=source_cert, target_pass=source_cert["target_pass"],
        encodings=source_encodings, best_storage=source_best,
        representation="minimal same-support source K upper triangles plus shared h/Evac",
        identical_source_reference_and_header=True)
    screens, selected = screen_candidates(kernel)
    write_rows(folder/"screens.csv", screens)
    rows = []
    for index, (screen, native) in enumerate(selected):
        encodings = save_encodings(folder/("candidate_%02d"%index), "whole_factor", header, native)
        actual = load_actual(encodings[0]["storage"]["path"])
        identical_header(header, actual)
        cert = cnc.certify_packet(kernel, actual, np.float64(CONSTANT),
                                 n_electrons=2, target=TARGET, max_block_dimension=190)
        timing = benchmark_factor(actual)
        best = best_storage(encodings)
        comparison = _comparison(best, source_best)
        speed = timing["comparisons"]["structured_dense"]["median_paired_ratio"]
        global_speed = timing["comparisons"]["global_dense"]["median_paired_ratio"]
        valid = bool(cert["target_pass"] and source_cert["target_pass"])
        row = dict(candidate=screen["candidate"], total_width=screen["total_width"],
            construction_diagnostics=screen["construction_diagnostics"], certificate=cert,
            target_pass=cert["target_pass"], encodings=encodings, best_storage=best,
            primary_timing=timing, storage_vs_shared_whole_source=comparison,
            source_baseline_target_pass=source_cert["target_pass"],
            joint_payload_and_strong_speed=bool(valid and comparison["payload_ratio"] < 1 and speed > 1),
            joint_npz_and_strong_speed=bool(valid and comparison["npz_ratio"] < 1 and speed > 1),
            joint_payload_and_global_speed=bool(valid and comparison["payload_ratio"] < 1 and global_speed > 1),
            identical_h_Evac_and_coefficient_dictionary_verified=True)
        rows.append(row)
        write_json(folder/("candidate_%02d_record.json"%index), row)
    result = dict(basis=basis, passed=True, spatial_modes=q, n_qubits=2*q,
        input=source["metadata"], source=source_record, bestsource=source_best,
        records=rows, screens=screens, selected_candidates=len(selected),
        finite_source_normalization_explicitly_controlled=True,
        strong_storage_baseline="shared whole full K+h/Evac in the same basis, precision and semantic representation",
        source_and_candidate_container_opportunities=["ordinary NPZ", "single uint8 typed-packed NPZ"],
        common_Pauli_codecs_applicable=False, no_Pauli_expansion_required=True,
        comparison_to_best_arbitrary_Pauli_representation_claimed=False,
        all_AOs_retained=True, no_SCF=True, no_fixed_occupations=True,
        strict_scope="N=2 Coulomb error versus finite pair-symmetrized K with identical h/Evac; upstream integral and basis error excluded",
        scientific_negative_and_censored_results_retained=True,
        real_quantum_hardware_jobs_submitted=0)
    write_json(folder/"summary.json", result)
    return result

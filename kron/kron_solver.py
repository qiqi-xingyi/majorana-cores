# --*-- coding:utf-8 --*--
#
# KronSolver: Kronecker decomposition-based solver for molecular Hamiltonians.
#
# Given H (n qubits), decompose H ≈ sum_r alpha_r * (A_r ⊗ B_r) + c0*I
# and leverage the factored form to:
#   1. Approximate ground-state energy via sub-system diagonalization
#   2. Estimate quantum circuit cutting feasibility & resource savings
#   3. Enable classical simulation of systems beyond exact diag limits

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from openfermion import get_sparse_operator, jordan_wigner
from openfermion.ops import QubitOperator
from scipy.sparse.linalg import eigsh

from .kron_decomposition import (
    KronBoundaryAnalyzer,
    CutConfig,
    BuildConfig,
    PowerIterConfig,
    DenseDatasetConfig,
    _build_sparse_C_from_qubit_op,
    _matvec_C,
    _matvec_CH,
    _power_iteration_top_singular,
    _dense_traceless,
    _dense_top1_schmidt_via_matvec,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class KronSolverConfig:
    """Configuration for the Kronecker solver."""
    # Decomposition
    max_rank: int = 5                  # Maximum Kronecker rank to compute
    rho_threshold: float = 0.95        # Cumulative rho threshold for early stopping
    traceless: bool = True             # Remove identity before decomposition

    # Power iteration
    piter_max_iter: int = 300
    piter_tol: float = 1e-10
    piter_seed: int = 0

    # Cut strategy
    cut_nA: Optional[int] = None       # None = balanced

    # Energy solver
    n_eigvals: int = 4                 # Number of eigenvalues to compute per sub-system
    energy_method: str = "kron_diag"   # "kron_diag" or "perturbative"

    # Build
    coeff_tol: float = 1e-12

    # Optional ARPACK fallback; the default performs a single svds attempt.
    svds_failure_policy: str = "raise"  # or "arpack_ncv64_once"


# ============================================================
# Rank-k Deflated Kronecker Decomposition
# ============================================================

@dataclass
class KronFactor:
    """A single rank-1 Kronecker factor: alpha * (A ⊗ B)."""
    rank_index: int
    alpha: float                       # singular value = scaling factor
    rho_k: float                       # fraction of norm captured by this component
    rho_cumulative: float              # cumulative fraction up to this rank

    # Pauli-space factors (sparse representation)
    A_pauli: Dict[str, complex]        # {pauli_string: coefficient} for system A
    B_pauli: Dict[str, complex]        # {pauli_string: coefficient} for system B

    # Actual full coefficient-space residual after this prefix.
    residual_fro2: float = 0.0

    # Dense factors (if available, for small systems)
    A_matrix: Optional[np.ndarray] = None   # (dim_a, dim_a)
    B_matrix: Optional[np.ndarray] = None   # (dim_b, dim_b)


@dataclass
class KronDecomposition:
    """Complete rank-k Kronecker decomposition result."""
    n_qubits: int
    cut_nA: int
    cut_nB: int
    identity_coeff: complex            # c0 * I⊗I term
    identity_weight: float             # |c0|^2 / ||H||^2
    factors: List[KronFactor] = field(default_factory=list)
    total_fro2: float = 0.0            # total Frobenius norm squared of coefficient matrix
    residual_fro2: float = 0.0         # full retained-input coefficient residual
    discarded_fro2: float = 0.0        # input terms dropped by coeff_tol
    coefficient_input_fro2: float = 0.0
    residual_metric_version: str = "unspecified"
    certificate_target: str = "unspecified"
    input_representation: str = "supplied_qubit_operator"
    coefficient_tolerance: float = 0.0
    factors_are_hermitian: bool = False
    solver_backend: str = "sparse_implicit_subspace_svds"
    solver_policy: str = "raise"
    solver_policy_version: str = "svds_failure_policy"
    svds_events: List[Dict[str, Any]] = field(default_factory=list)
    solver_cost_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return len(self.factors)

    @property
    def rho_cumulative(self) -> float:
        if not self.factors:
            return 0.0
        return self.factors[-1].rho_cumulative

    @property
    def alphas(self) -> List[float]:
        return [f.alpha for f in self.factors]


def _full_coefficient_residual_fro2(
    coefficient_matrix,
    left: np.ndarray,
    right: np.ndarray,
    total_fro2: float,
) -> float:
    """Evaluate ||C-left@right||_F^2, including entries outside sparse support.

    The Gram identity is inexpensive away from cancellation. Near roundoff we
    evaluate bounded row blocks explicitly, never materializing the dense C.
    """
    if left.shape[1] == 0:
        return float(total_fro2)
    coo = coefficient_matrix.tocoo()
    cross = 0.0 + 0.0j
    for lo in range(0, coo.nnz, 4096):
        hi = min(lo + 4096, coo.nnz)
        predicted = np.einsum(
            "ij,ji->i", left[coo.row[lo:hi]], right[:, coo.col[lo:hi]],
        )
        cross += np.vdot(coo.data[lo:hi], predicted)
    left_gram = left.conjugate().T @ left
    right_gram = right @ right.conjugate().T
    approximation_fro2 = float(np.real(np.sum(left_gram * right_gram.T)))
    residual = float(total_fro2 - 2.0 * np.real(cross) + approximation_fro2)
    cancellation_scale = (
        abs(total_fro2) + 2.0 * abs(cross) + abs(approximation_fro2)
    )
    roundoff_floor = 64.0 * np.finfo(float).eps * cancellation_scale
    if residual > roundoff_floor:
        return residual

    # A clipped cancellation result must never become an artificial zero
    # certificate. Only this explicit residual is used near machine precision.
    n_rows, n_cols = coefficient_matrix.shape
    block_rows = max(1, min(64, 262144 // max(1, n_cols)))
    residual = 0.0
    for lo in range(0, n_rows, block_rows):
        hi = min(lo + block_rows, n_rows)
        block = left[lo:hi] @ right
        sparse_block = coefficient_matrix[lo:hi].tocoo()
        np.add.at(block, (sparse_block.row, sparse_block.col), -sparse_block.data)
        residual += float(np.vdot(block, block).real)
    return residual


def _orthogonalize_twice(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Modified projection with a second pass for nearly exhausted subspaces."""
    result = np.asarray(vector).reshape(-1).copy()
    for _ in range(2):
        if basis.shape[1]:
            result -= basis @ (basis.conjugate().T @ result)
    length = float(np.linalg.norm(result))
    if length <= 16.0 * np.finfo(float).eps:
        raise RuntimeError("Unable to find an independent residual direction.")
    return result / length


SVDS_POLICY_VERSION = "svds_failure_policy"
SVDS_EVENT_VERSION = "kron_svds_events"


class KronSVDSCandidateError(RuntimeError):
    """A fallback singular triplet failed the local quality rule."""


def _attach_svds_evidence(exc, policy, events, costs):
    # Keep the raised exception type, including interruption BaseExceptions.
    for key, value in (("solver_policy", policy), ("solver_policy_version", SVDS_POLICY_VERSION),
                       ("svds_events", list(events)), ("solver_cost_summary", dict(costs))):
        try:
            setattr(exc, key, value)
        except Exception:
            pass


def _checked_recovery_triplet(operator, answer, tol, residual_fro):
    left, values, right = answer
    left, values, right = np.asarray(left), np.asarray(values), np.asarray(right)
    m, n = operator.shape
    if left.shape != (m, 1) or values.shape != (1,) or right.shape != (1, n):
        raise KronSVDSCandidateError("Fallback singular-triplet shapes are invalid")
    if not all(np.all(np.isfinite(a)) for a in (left, values, right)):
        raise KronSVDSCandidateError("Fallback singular triplet is nonfinite")
    if np.any(np.imag(values) != 0.0) or float(np.real(values[0])) <= 0.0:
        raise KronSVDSCandidateError("Fallback singular value must be positive and real")
    sigma = float(np.real(values[0]))
    u, v = left[:, 0], right[0, :].conjugate()
    rtol = max(float(tol), 128.0 * np.finfo(float).eps)
    unorm, vnorm = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if (not math.isfinite(rtol) or not math.isfinite(unorm) or not math.isfinite(vnorm)
            or abs(unorm - 1.0) > rtol or abs(vnorm - 1.0) > rtol):
        exc = KronSVDSCandidateError("Fallback singular vectors are not unit vectors")
        exc.candidate_quality = dict(sigma=sigma, u_norm=unorm, v_norm=vnorm, quality_rtol=rtol)
        raise exc
    left_residual = float(np.linalg.norm(operator.matvec(v) - sigma * u))
    right_residual = float(np.linalg.norm(operator.rmatvec(u) - sigma * v))
    metrics = dict(sigma=sigma, u_norm=unorm, v_norm=vnorm,
        left_triplet_residual_abs=left_residual, right_triplet_residual_abs=right_residual,
        left_triplet_residual_relative_sigma=left_residual / sigma,
        right_triplet_residual_relative_sigma=right_residual / sigma,
        left_triplet_residual_relative_D_fro=left_residual / residual_fro if residual_fro > 0 else None,
        right_triplet_residual_relative_D_fro=right_residual / residual_fro if residual_fro > 0 else None,
        quality_rtol=rtol, quality_abs_threshold=rtol * sigma,
        quality_rule="finite, valid shape, unit vectors and both implicit-D triplet residuals <= max(tol,128eps)*sigma",
        quality_scope="declared local numerical rule; not the svds documented guarantee or proof of global optimality")
    if (not math.isfinite(left_residual) or not math.isfinite(right_residual)
            or not math.isfinite(rtol * sigma) or max(left_residual, right_residual) > rtol * sigma):
        exc = KronSVDSCandidateError("Fallback triplet fails the finite declared residual rule")
        exc.candidate_quality = metrics
        raise exc
    return metrics


def _svds_event_metadata(record):
    """Keep rejected nonfinite diagnostics visible without invalid JSON numbers."""
    nonfinite = {}
    def clean(value, path):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite[path] = repr(value)
            return None
        if isinstance(value, dict):
            return {key: clean(item, path + "." + key if path else key) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item, path + "[" + str(index) + "]") for index, item in enumerate(value)]
        return value
    result = clean(record, "")
    if nonfinite:
        result["nonfinite_metadata_fields"] = list(nonfinite)
        result["nonfinite_metadata_repr"] = nonfinite
    return result


def _svds_with_failure_policy(operator, start, config, stage, residual_fro, svds,
                              no_convergence, events, costs, event_sink):
    """First svds attempt, followed only on its ARPACK limit by one ncv64 attempt."""
    began = time.perf_counter()
    policy = config.svds_failure_policy
    original_v0 = np.array(start, copy=True)  # Before a library may mutate v0.
    tol = max(0.0, float(config.piter_tol))
    maxiter = max(300, int(config.piter_max_iter))
    for key in ("initial_svds_sec", "additional_svds_sec", "quality_check_sec", "event_sink_sec", "recovery_protocol_wall_sec"):
        costs.setdefault(key, 0.0)
    costs.update(instrumented=True, scope="top-1 calls, recovery checks and event management only; protocol wall includes components and is not additive with them; full decomposition wall is measured by caller")
    base = dict(event_schema_version=SVDS_EVENT_VERSION, solver_policy=policy,
        solver_policy_version=SVDS_POLICY_VERSION, stage=int(stage),
        operator_shape=list(operator.shape), dtype=str(operator.dtype), k=1, which="LM", solver="arpack",
        tol=tol, maxiter=maxiter, seed=int(config.piter_seed), v0_real=original_v0.real.tolist(), v0_imag=original_v0.imag.tolist(),
        v0_scope="actual initial array before original call; fallback copies it without another RNG draw",
        budget_scope="each attempt has its own maxiter; a fallback and larger ncv add work beyond the original single-attempt budget")

    def emit(event, attempt, ncv, **values):
        record = _svds_event_metadata(dict(base, event=event, attempt=attempt, ncv=ncv, **values))
        events.append(record)
        if event_sink is not None:
            started = time.perf_counter()
            try:
                event_sink(dict(record, operator_shape=list(record["operator_shape"])))
            finally:
                costs["event_sink_sec"] += time.perf_counter() - started

    initial_arpack_failure = None

    def attempt(number, ncv, vector):
        nonlocal initial_arpack_failure
        emit("attempt_started", number, ncv, status="started")
        started = time.perf_counter()
        kwargs = dict(k=1, which="LM", solver="arpack", v0=vector, tol=tol,
                      maxiter=maxiter, return_singular_vectors=True)
        if ncv is not None:
            kwargs["ncv"] = ncv
        try:
            answer = svds(operator, **kwargs)
        except BaseException as exc:
            if number == 1 and isinstance(exc, no_convergence):
                initial_arpack_failure = exc
            elapsed = time.perf_counter() - started
            costs["initial_svds_sec" if number == 1 else "additional_svds_sec"] += elapsed
            emit("attempt_finished", number, ncv,
                 status="failed" if isinstance(exc, Exception) else "interrupted",
                 error_type=type(exc).__name__, message=str(exc), svds_call_sec=elapsed)
            raise
        elapsed = time.perf_counter() - started
        costs["initial_svds_sec" if number == 1 else "additional_svds_sec"] += elapsed
        emit("attempt_finished", number, ncv, status="ok", svds_call_sec=elapsed)
        return answer

    try:
        try:
            return attempt(1, None, start)
        except no_convergence as exc:
            if (exc is not initial_arpack_failure or policy != "arpack_ncv64_once"
                    or min(operator.shape) <= 64):
                raise
        answer = attempt(2, 64, original_v0.copy())
        emit("candidate_check_started", 2, 64, status="started")
        started = time.perf_counter()
        try:
            metrics = _checked_recovery_triplet(operator, answer, tol, residual_fro)
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            costs["quality_check_sec"] += elapsed
            emit("candidate_check_finished", 2, 64,
                 status="rejected" if isinstance(exc, Exception) else "interrupted",
                 error_type=type(exc).__name__, message=str(exc), quality_check_sec=elapsed,
                 candidate_quality=getattr(exc, "candidate_quality", None))
            raise
        elapsed = time.perf_counter() - started
        costs["quality_check_sec"] += elapsed
        emit("candidate_check_finished", 2, 64, status="ok", quality_check_sec=elapsed, candidate_quality=metrics)
        return answer
    finally:
        costs["recovery_protocol_wall_sec"] += time.perf_counter() - began


def decompose_rank_k(
    qubit_op: QubitOperator,
    n_qubits: int,
    config: KronSolverConfig,
    *,
    event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> KronDecomposition:
    """Full-residual sparse decomposition; recovery is explicit and finite.

    Default 'raise' without a sink performs a single svds call per stage.
    The optional ncv64 policy adds one ARPACK retry with a larger Krylov space.
    Event sinks must persist their own journal; failures never become factors.
    """
    policy = config.svds_failure_policy
    if policy not in ("raise", "arpack_ncv64_once"):
        raise ValueError("svds_failure_policy must be 'raise' or 'arpack_ncv64_once'")
    if event_sink is not None and not callable(event_sink):
        raise TypeError("event_sink must be callable or None")
    events: List[Dict[str, Any]] = []
    costs: Dict[str, Any] = {}
    try:
        return _decompose_rank_k_impl(qubit_op, n_qubits, config, event_sink=event_sink,
                                     svds_events=events, solver_cost_summary=costs)
    except BaseException as exc:
        _attach_svds_evidence(exc, policy, events, costs)
        raise


def _decompose_rank_k_impl(
    qubit_op: QubitOperator,
    n_qubits: int,
    config: KronSolverConfig,
    *,
    event_sink,
    svds_events,
    solver_cost_summary,
) -> KronDecomposition:
    """Factor sparse C using an implicit *full* residual and orthogonal bases.

    For a left basis Q, B=Q^H C and D=C-QB. Sparse products with C plus
    low-rank products implement D and D^H, so off-support entries are retained
    implicitly. A final SVD of the small factor B produces ordered, orthogonal
    factors; only B, never the dense coefficient matrix, is materialized.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, svds
    from .kron_decomposition import _term_to_ab_keys

    cut_cfg = CutConfig(cut_nA=config.cut_nA)
    build_cfg = BuildConfig(coeff_tol=config.coeff_tol)
    n_a = config.cut_nA if config.cut_nA is not None else n_qubits // 2
    n_b = n_qubits - n_a
    if not 0 <= n_a <= n_qubits:
        raise ValueError("cut_nA must lie between zero and n_qubits.")

    rows, cols, vals, n_rows, n_cols, total_fro2, _ = _build_sparse_C_from_qubit_op(
        qubit_op=qubit_op, n_qubits=n_qubits, cut_cfg=cut_cfg,
        build_cfg=build_cfg, remove_identity=config.traceless,
    )
    original_identity = complex(qubit_op.terms.get((), 0.0))
    identity_coeff = original_identity if config.traceless else 0.0 + 0.0j
    input_fro2 = 0.0
    discarded_fro2 = 0.0
    for term, coeff in qubit_op.terms.items():
        if config.traceless and not term:
            continue
        weight = abs(complex(coeff)) ** 2
        input_fro2 += weight
        if abs(complex(coeff)) <= config.coeff_tol:
            discarded_fro2 += weight
    full_input_fro2 = sum(abs(complex(c)) ** 2 for c in qubit_op.terms.values())
    real_coefficients = bool(np.all(np.imag(vals) == 0.0))
    hermitian_input = real_coefficients and original_identity.imag == 0.0
    decomp = KronDecomposition(
        n_qubits=n_qubits, cut_nA=n_a, cut_nB=n_b,
        identity_coeff=identity_coeff,
        identity_weight=(
            float(abs(original_identity) ** 2 / full_input_fro2)
            if full_input_fro2 > 0.0 else 0.0
        ),
        total_fro2=float(total_fro2), residual_fro2=float(total_fro2),
        discarded_fro2=float(discarded_fro2),
        coefficient_input_fro2=float(input_fro2),
        residual_metric_version="full_coefficient_residual",
        certificate_target="original_pauli_input",
        input_representation="supplied_qubit_operator",
        coefficient_tolerance=float(config.coeff_tol),
        factors_are_hermitian=hermitian_input,
        solver_policy=config.svds_failure_policy, solver_policy_version=SVDS_POLICY_VERSION,
        svds_events=svds_events, solver_cost_summary=solver_cost_summary,
    )
    requested_rank = max(0, min(int(config.max_rank), n_rows, n_cols))
    if requested_rank == 0 or total_fro2 == 0.0:
        return decomp

    # Match the builder's compact row/column indexing exactly. Compact zero
    # rows/columns are harmless; projecting interior zero entries is not.
    a_index: Dict[str, int] = {}
    b_index: Dict[str, int] = {}
    for term, coeff in qubit_op.terms.items():
        if not term or abs(complex(coeff)) <= config.coeff_tol:
            continue
        a_key, b_key = _term_to_ab_keys(term, n_a=n_a, n_b=n_b)
        if a_key not in a_index:
            a_index[a_key] = len(a_index)
        if b_key not in b_index:
            b_index[b_key] = len(b_index)
    if not config.traceless and abs(original_identity) > config.coeff_tol:
        a_index.setdefault("I" * n_a, len(a_index))
        b_index.setdefault("I" * n_b, len(b_index))
    a_keys = [""] * n_rows
    b_keys = [""] * n_cols
    for key, index in a_index.items():
        a_keys[index] = key
    for key, index in b_index.items():
        b_keys[index] = key

    dtype = np.float64 if real_coefficients else np.complex128
    data = vals.real.astype(dtype) if real_coefficients else vals.astype(dtype)
    coefficient_matrix = coo_matrix(
        (data, (rows, cols)), shape=(n_rows, n_cols),
    ).tocsr()
    # Exhausting the smaller left space guarantees full recovery at full rank.
    transposed = n_rows > n_cols
    work = (
        coefficient_matrix.conjugate().T.tocsr()
        if transposed else coefficient_matrix
    )
    m, n = work.shape
    basis = np.empty((m, 0), dtype=dtype)
    factor_rows = np.empty((0, n), dtype=dtype)
    rng = np.random.default_rng(config.piter_seed)
    current_residual = float(total_fro2)

    for _ in range(requested_rank):
        if m - basis.shape[1] == 1:
            # svds(k=1) cannot act on a one-dimensional remaining space.
            # Pick a stable coordinate and project it to that complement.
            pivot = int(np.argmax(1.0 - np.sum(np.abs(basis) ** 2, axis=1)))
            direction = np.zeros(m, dtype=dtype)
            direction[pivot] = 1.0
        elif m <= 2:
            # Complex ARPACK requires k < N-1, so k=1 is invalid for N=2.
            # Work is oriented with m=min(shape). Form only its <=2 by <=2
            # left Gram matrix; never densify a possibly very wide work matrix.
            gram = (work @ work.conjugate().T).toarray()
            if basis.shape[1]:
                complement = np.eye(m, dtype=dtype) - basis @ basis.conjugate().T
                gram = complement @ gram @ complement
            eigenvalues, eigenvectors = np.linalg.eigh(gram)
            leading_singular = math.sqrt(max(0.0, float(eigenvalues[-1])))
            if leading_singular <= np.finfo(float).eps * math.sqrt(total_fro2):
                break
            direction = eigenvectors[:, -1]
        else:
            def matvec(x):
                vector = np.asarray(x).reshape(-1)
                return work @ vector - basis @ (factor_rows @ vector)

            def rmatvec(x):
                vector = np.asarray(x).reshape(-1)
                return (
                    work.conjugate().T @ vector
                    - factor_rows.conjugate().T @ (basis.conjugate().T @ vector)
                )

            residual_operator = LinearOperator(
                (m, n), matvec=matvec, rmatvec=rmatvec, dtype=dtype,
            )
            start = rng.standard_normal(m)
            if not real_coefficients:
                start = start + 1j * rng.standard_normal(m)
            if config.svds_failure_policy == "raise" and event_sink is None:
                left, singular_values, _ = svds(
                    residual_operator, k=1, which="LM", solver="arpack",
                    v0=start, tol=max(0.0, float(config.piter_tol)),
                    maxiter=max(300, int(config.piter_max_iter)),
                    return_singular_vectors=True,
                )
            else:
                left, singular_values, _ = _svds_with_failure_policy(
                    residual_operator, start, config, basis.shape[1] + 1,
                    math.sqrt(max(0.0, current_residual)), svds, ArpackNoConvergence,
                    svds_events, solver_cost_summary, event_sink,
                )
            if singular_values[0] <= np.finfo(float).eps * math.sqrt(total_fro2):
                break
            direction = left[:, 0]

        try:
            direction = _orthogonalize_twice(direction, basis)
        except RuntimeError:
            # At numerical exhaustion ARPACK may select cancellation noise in
            # the already captured span. Stop only after an explicit complete
            # residual audit; never replace that residual by zero.
            current_residual = _full_coefficient_residual_fro2(
                work, basis, factor_rows, float(total_fro2),
            )
            backward_error = 64.0 * np.finfo(float).eps
            if current_residual <= backward_error ** 2 * total_fro2:
                break
            # A substantial residual must not be silently dropped. Select an
            # independent complement coordinate and continue the subspace.
            pivot = int(np.argmax(1.0 - np.sum(np.abs(basis) ** 2, axis=1)))
            direction = np.zeros(m, dtype=dtype)
            direction[pivot] = 1.0
            direction = _orthogonalize_twice(direction, basis)
        new_row = np.asarray(work.conjugate().T @ direction).conjugate()
        basis = np.column_stack((basis, direction))
        factor_rows = np.vstack((factor_rows, new_row))
        current_residual = _full_coefficient_residual_fro2(
            work, basis, factor_rows, float(total_fro2),
        )
        capture = float(1.0 - current_residual / total_fro2)
        if current_residual <= max(0.0, 1.0 - config.rho_threshold) * total_fro2:
            break
        if current_residual <= (
            np.finfo(float).eps ** 2 * total_fro2
        ):
            break

    if basis.shape[1] == 0:
        return decomp

    small_left, singular_values, right = np.linalg.svd(
        factor_rows, full_matrices=False,
    )
    left = basis @ small_left
    if transposed:
        left, right = right.conjugate().T, left.conjugate().T

    previous_residual = float(total_fro2)
    for j, singular_value in enumerate(singular_values):
        # Keep every computed coefficient. Dropping tiny factor entries here
        # would create a discrepancy between the certified and returned factors.
        a_pauli = {
            key: complex(value) for key, value in zip(a_keys, left[:, j])
            if value != 0.0
        }
        b_pauli = {
            key: complex(value) for key, value in zip(b_keys, right[j, :])
            if value != 0.0
        }
        residual = _full_coefficient_residual_fro2(
            coefficient_matrix, left[:, :j + 1],
            singular_values[:j + 1, None] * right[:j + 1],
            float(total_fro2),
        )
        rho = float(1.0 - residual / total_fro2)
        decomp.factors.append(KronFactor(
            rank_index=j, alpha=float(singular_value),
            rho_k=float((previous_residual - residual) / total_fro2),
            rho_cumulative=rho, A_pauli=a_pauli, B_pauli=b_pauli,
            residual_fro2=float(residual),
        ))
        decomp.residual_fro2 = float(residual)
        previous_residual = residual
        if residual <= max(0.0, 1.0 - config.rho_threshold) * total_fro2:
            break
    return decomp


def decompose_rank_k_dense(
    H: np.ndarray,
    n_qubits: int,
    config: KronSolverConfig,
) -> Tuple[KronDecomposition, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Dense rank-k Kronecker decomposition with matrix factors.

    Returns decomposition + list of (A_matrix, B_matrix) pairs.
    """
    n_a = config.cut_nA if config.cut_nA is not None else (n_qubits // 2)
    dim = H.shape[0]
    dim_a = 1 << n_a
    dim_b = dim // dim_a

    piter_cfg = PowerIterConfig(
        max_iter=config.piter_max_iter,
        tol=config.piter_tol,
        seed=config.piter_seed,
    )

    H_work = H.copy()
    c0 = 0.0 + 0.0j
    if config.traceless:
        H_work, c0 = _dense_traceless(H)

    fro2_total = float(np.linalg.norm(H_work, 'fro') ** 2)

    decomp = KronDecomposition(
        n_qubits=n_qubits,
        cut_nA=n_a,
        cut_nB=n_qubits - n_a,
        identity_coeff=c0,
        identity_weight=0.0,  # will compute below
        total_fro2=fro2_total,
    )

    matrix_factors = []
    cumulative_rho = 0.0

    for rank_idx in range(config.max_rank):
        current_fro2 = float(np.linalg.norm(H_work, 'fro') ** 2)
        if current_fro2 < 1e-24:
            break

        s1, u, v = _dense_top1_schmidt_via_matvec(H_work, n_a=n_a, piter=piter_cfg)

        if s1 < 1e-15:
            break

        rho_k = (s1 * s1) / fro2_total if fro2_total > 0 else 0.0
        cumulative_rho += rho_k

        # Balanced scaling
        sroot = math.sqrt(max(s1, 0.0))
        A_mat = (sroot * u).reshape(dim_a, dim_a)
        B_mat = (sroot * np.conjugate(v)).reshape(dim_b, dim_b)

        factor = KronFactor(
            rank_index=rank_idx,
            alpha=s1,
            rho_k=rho_k,
            rho_cumulative=cumulative_rho,
            A_pauli={},
            B_pauli={},
            A_matrix=A_mat,
            B_matrix=B_mat,
        )
        decomp.factors.append(factor)
        matrix_factors.append((A_mat, B_mat))

        # Deflate: H_work -= kron(A, B)
        H_work = H_work - np.kron(A_mat, B_mat)

        if cumulative_rho >= config.rho_threshold:
            break

    decomp.residual_fro2 = float(np.linalg.norm(H_work, 'fro') ** 2)
    return decomp, matrix_factors


# ============================================================
# Accelerated Ground-State Energy Solver
# ============================================================

@dataclass
class EnergyResult:
    """Result from Kronecker-accelerated energy computation."""
    # Energies
    e_kron_rank1: float                  # Rank-1 Kronecker energy estimate
    e_kron_best: float                   # Best rank-k energy estimate
    e_exact: Optional[float] = None      # Exact diag energy (if computed)

    # Errors (relative to exact, if available)
    error_rank1: Optional[float] = None
    error_best: Optional[float] = None

    # Decomposition info
    rank_used: int = 1
    rho_cumulative: float = 0.0

    # Timing
    time_decompose_sec: float = 0.0
    time_solve_sec: float = 0.0
    time_exact_sec: float = 0.0

    # Sub-system info
    dim_original: int = 0               # 2^n
    dim_subsystem_a: int = 0            # 2^(n/2)
    dim_subsystem_b: int = 0            # 2^(n/2)
    classical_speedup: float = 0.0      # dim_original / (dim_a + dim_b)


def _kron_energy_rank1_dense(
    A: np.ndarray,
    B: np.ndarray,
    c0: complex,
    n_eigvals: int = 4,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute ground-state energy from rank-1 Kronecker decomposition.

    If H ≈ kron(A, B) + c0*I, we use the eigenvalue identity:
        eig(kron(A, B)) = {eig(A)_i * eig(B)_j}

    IMPORTANT: A, B from the balanced Schmidt decomposition are generally
    NOT Hermitian (they are reshaped singular vectors). However, kron(A,B)
    IS Hermitian. We use np.linalg.eig (general eigenvalues) for A, B
    and then take real parts of the products (which must be real since
    kron(A,B) is Hermitian).

    This is the KEY speedup: diag two dim_sub matrices O(dim_sub^3)
    instead of one dim matrix O(dim^3).

    Returns (e_ground, eigvals_A, eigvals_B).
    """
    dim_a, dim_b = A.shape[0], B.shape[0]

    # General eigenvalues (A, B are not necessarily Hermitian)
    if dim_a <= 1024:
        eigvals_A = np.linalg.eigvals(A)
    else:
        # For large systems, use Lanczos on kron(A,B) directly
        # since eigvals of non-Hermitian matrices are not easily partial-computed
        eigvals_A = np.linalg.eigvals(A)

    if dim_b <= 1024:
        eigvals_B = np.linalg.eigvals(B)
    else:
        eigvals_B = np.linalg.eigvals(B)

    # All pairwise products: eig(kron(A,B)) = {eig(A)_i * eig(B)_j}
    # Since kron(A,B) + c0*I is Hermitian, these products + c0 must be real.
    products = np.outer(eigvals_A, eigvals_B).ravel()
    products_shifted = products + c0

    # Take real parts (imaginary parts should be ~0 for Hermitian kron)
    e_min = float(np.min(products_shifted.real))

    return e_min, eigvals_A, eigvals_B


def _kron_energy_rankk_dense(
    factors: List[Tuple[np.ndarray, np.ndarray]],
    c0: complex,
    dim: int,
    n_eigvals: int = 4,
) -> float:
    """
    Compute ground-state energy from rank-k Kronecker decomposition.

    H ≈ sum_r kron(A_r, B_r) + c0*I

    For rank > 1, we reconstruct H_approx and diagonalize it.
    This is still cheaper than exact diag if rank is small and we
    can use the Kronecker structure for matvec.
    """
    if len(factors) == 0:
        return c0.real

    dim_a = factors[0][0].shape[0]
    dim_b = factors[0][1].shape[0]

    # For small dim, reconstruct and diag directly
    if dim <= 4096:  # 12 qubits
        H_approx = np.zeros((dim, dim), dtype=np.complex128)
        for A, B in factors:
            H_approx += np.kron(A, B)
        H_approx += c0 * np.eye(dim, dtype=np.complex128)
        eigvals = np.linalg.eigvalsh(H_approx)
        return float(eigvals[0])

    # For larger systems, use Kronecker-structured matvec with Lanczos
    # H_approx |v> = sum_r kron(A_r, B_r) |v> + c0 |v>
    from scipy.sparse.linalg import LinearOperator

    def kron_matvec(x):
        """Compute H_approx @ x using Kronecker structure."""
        y = c0 * x
        X = x.reshape(dim_a, dim_b)
        for A, B in factors:
            # kron(A, B) @ vec(X) = vec(A @ X @ B.T)
            Y = A @ X @ B.T
            y += Y.reshape(-1)
        return y

    H_op = LinearOperator(shape=(dim, dim), matvec=kron_matvec, dtype=np.complex128)
    k = min(n_eigvals, dim - 2)
    eigvals = eigsh(H_op, k=k, which='SA', return_eigenvectors=False)
    return float(np.min(eigvals))


def solve_ground_state(
    qubit_op: QubitOperator,
    n_qubits: int,
    config: KronSolverConfig,
    compute_exact: bool = False,
) -> EnergyResult:
    """
    Compute ground-state energy using Kronecker decomposition.

    For system with n qubits:
    - Exact diag requires O(2^n) memory and O(8^n) time
    - Rank-1 Kron requires O(2^(n/2)) memory and O(8^(n/2)) time
    - Rank-k Kron matvec requires O(k * 4^(n/2)) per iteration

    Parameters
    ----------
    qubit_op : QubitOperator
        Hamiltonian in Pauli-sum form.
    n_qubits : int
        Number of qubits.
    config : KronSolverConfig
        Solver configuration.
    compute_exact : bool
        If True and system is small enough, also compute exact energy.

    Returns
    -------
    EnergyResult
        Energy estimates with timing and error info.
    """
    dim = 1 << n_qubits
    n_a = config.cut_nA if config.cut_nA is not None else (n_qubits // 2)
    dim_a = 1 << n_a
    dim_b = dim // dim_a

    result = EnergyResult(
        e_kron_rank1=0.0,
        e_kron_best=0.0,
        dim_original=dim,
        dim_subsystem_a=dim_a,
        dim_subsystem_b=dim_b,
    )

    # === Step 1: Kronecker decomposition ===
    t0 = time.time()

    # Choose dense or sparse path based on system size
    use_dense = n_qubits <= 20  # dense path feasible up to ~20 qubits

    if use_dense:
        sparse_H = get_sparse_operator(qubit_op, n_qubits=n_qubits)
        H_dense = sparse_H.toarray().astype(np.complex128)

        decomp, matrix_factors = decompose_rank_k_dense(H_dense, n_qubits, config)
    else:
        decomp = decompose_rank_k(qubit_op, n_qubits, config)
        matrix_factors = []

    t_decompose = time.time() - t0
    result.time_decompose_sec = t_decompose
    result.rank_used = decomp.rank
    result.rho_cumulative = decomp.rho_cumulative

    # === Step 2: Solve for energy ===
    t0 = time.time()

    if use_dense and matrix_factors:
        # Rank-1 energy
        A1, B1 = matrix_factors[0]
        e_rank1, eigvals_A, eigvals_B = _kron_energy_rank1_dense(
            A1, B1, decomp.identity_coeff, config.n_eigvals,
        )
        result.e_kron_rank1 = e_rank1

        # Rank-k energy (best)
        e_best = _kron_energy_rankk_dense(
            matrix_factors, decomp.identity_coeff, dim, config.n_eigvals,
        )
        result.e_kron_best = e_best
    else:
        # Sparse path: construct sub-operators from Pauli factors
        # For now, build dense sub-system matrices from Pauli decomposition
        if decomp.factors:
            A_ops, B_ops = _pauli_factors_to_matrices(
                decomp.factors, n_a, n_qubits - n_a,
            )
            if A_ops and B_ops:
                e_rank1, _, _ = _kron_energy_rank1_dense(
                    A_ops[0], B_ops[0], decomp.identity_coeff, config.n_eigvals,
                )
                result.e_kron_rank1 = e_rank1

                # Rank-k energy
                factors_list = list(zip(A_ops, B_ops))
                e_best = _kron_energy_rankk_dense(
                    factors_list, decomp.identity_coeff, dim, config.n_eigvals,
                )
                result.e_kron_best = e_best

    t_solve = time.time() - t0
    result.time_solve_sec = t_solve

    # === Step 3: Exact energy (optional, for validation) ===
    if compute_exact and n_qubits <= 20:
        t0 = time.time()
        if use_dense:
            eigvals = np.linalg.eigvalsh(H_dense)
        else:
            sparse_H = get_sparse_operator(qubit_op, n_qubits=n_qubits)
            H_dense = sparse_H.toarray().astype(np.complex128)
            eigvals = np.linalg.eigvalsh(H_dense)
        result.e_exact = float(eigvals[0])
        result.time_exact_sec = time.time() - t0

        result.error_rank1 = abs(result.e_kron_rank1 - result.e_exact)
        result.error_best = abs(result.e_kron_best - result.e_exact)

    # Speedup estimate
    # Exact: diag O(dim^3). Kron rank-1: diag O(dim_a^3 + dim_b^3)
    exact_cost = dim ** 3
    kron_cost = dim_a ** 3 + dim_b ** 3
    result.classical_speedup = exact_cost / kron_cost if kron_cost > 0 else 0.0

    return result


def _pauli_factors_to_matrices(
    factors: List[KronFactor],
    n_a: int,
    n_b: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert Pauli-space factors to dense matrices."""
    PAULI = {
        "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }

    def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
        """Convert 'IXYZ' to tensor product matrix."""
        mat = PAULI[pauli_str[0]]
        for ch in pauli_str[1:]:
            mat = np.kron(mat, PAULI[ch])
        return mat

    dim_a = 1 << n_a
    dim_b = 1 << n_b

    A_mats = []
    B_mats = []

    for factor in factors:
        # Build A matrix from Pauli decomposition
        A = np.zeros((dim_a, dim_a), dtype=np.complex128)
        for pauli_str, coeff in factor.A_pauli.items():
            A += coeff * pauli_string_to_matrix(pauli_str)

        # Build B matrix from Pauli decomposition
        B = np.zeros((dim_b, dim_b), dtype=np.complex128)
        for pauli_str, coeff in factor.B_pauli.items():
            B += coeff * pauli_string_to_matrix(pauli_str)

        # Apply balanced scaling: alpha = s1, A and B are unit-norm
        # So the actual factor is alpha * kron(A, B) = s1 * kron(A_unit, B_unit)
        # Balanced: sqrt(s1) * A, sqrt(s1) * B
        sroot = math.sqrt(max(factor.alpha, 0.0))
        A_mats.append(sroot * A)
        B_mats.append(sroot * B)

    return A_mats, B_mats


# ============================================================
# Quantum Resource Estimator
# ============================================================

@dataclass
class QuantumResourceEstimate:
    """Quantum resource estimates with and without Kronecker decomposition."""
    # System info
    n_qubits: int
    n_qubits_subsystem_a: int
    n_qubits_subsystem_b: int

    # Separability metrics
    rho1_total: float
    rho1_traceless: float
    identity_weight: float
    rank_for_99pct: int                # Kronecker rank needed for 99% norm capture
    rank_for_999pct: int               # ... for 99.9%

    # Quantum circuit resources (uncut circuit)
    original_cnot_depth_estimate: int   # ~ O(n^4) for molecular Hamiltonians
    original_qubit_count: int

    # Quantum circuit resources (with Kronecker cutting)
    cut_cnot_depth_a: int
    cut_cnot_depth_b: int
    cut_qubit_count_a: int
    cut_qubit_count_b: int
    cut_n_subcircuits: int              # 2 for rank-1, 2k for rank-k
    cut_classical_overhead: float       # O(k) classical post-processing

    # Savings
    qubit_reduction_factor: float       # n / max(n_a, n_b)
    depth_reduction_factor: float       # est. depth ratio
    feasibility: str                    # "excellent" / "good" / "marginal" / "not_recommended"
    recommendation: str                 # Human-readable recommendation


def estimate_quantum_resources(
    qubit_op: QubitOperator,
    n_qubits: int,
    config: KronSolverConfig,
) -> QuantumResourceEstimate:
    """
    Estimate quantum computing resources with and without Kronecker decomposition.

    This provides practical guidance on whether circuit cutting via Kronecker
    decomposition is beneficial for a given molecular system.
    """
    cut_cfg = CutConfig(cut_nA=config.cut_nA)
    build_cfg = BuildConfig(coeff_tol=config.coeff_tol)
    piter_cfg = PowerIterConfig(
        max_iter=config.piter_max_iter,
        tol=config.piter_tol,
        seed=config.piter_seed,
    )

    n_a = config.cut_nA if config.cut_nA is not None else (n_qubits // 2)
    n_b = n_qubits - n_a

    # Compute rho1 metrics
    analyzer = KronBoundaryAnalyzer(
        cut_cfg=cut_cfg,
        build_cfg=build_cfg,
        piter_cfg=piter_cfg,
        dense_cfg=DenseDatasetConfig(enabled=False),
        verbose=False,
    )

    total = analyzer.compute_rho1_from_paulisum(qubit_op, n_qubits, remove_identity=False)
    tr = analyzer.compute_rho1_from_paulisum(qubit_op, n_qubits, remove_identity=True)

    rho1_total = total["rho1"]
    rho1_traceless = tr["rho1"]
    fro2_total = total.get("fro2_coeff", 0.0)
    id_abs2 = total.get("id_abs2", 0.0)
    identity_weight = float(id_abs2 / fro2_total) if fro2_total > 0 else 0.0

    # Compute rank-k decomposition to find rank needed for 99%/99.9%
    decomp = decompose_rank_k(qubit_op, n_qubits, KronSolverConfig(
        max_rank=min(20, config.max_rank * 2),
        rho_threshold=0.999,
        traceless=config.traceless,
        piter_max_iter=config.piter_max_iter,
        piter_tol=config.piter_tol,
        piter_seed=config.piter_seed,
        svds_failure_policy=config.svds_failure_policy,
        cut_nA=config.cut_nA,
        coeff_tol=config.coeff_tol,
    ))

    rank_99 = decomp.rank
    rank_999 = decomp.rank
    for f in decomp.factors:
        if f.rho_cumulative >= 0.99 and rank_99 == decomp.rank:
            rank_99 = f.rank_index + 1
        if f.rho_cumulative >= 0.999:
            rank_999 = f.rank_index + 1
            break

    # --- Resource estimates ---

    # CNOT depth for molecular Hamiltonian simulation:
    # Trotterized: O(n^4) for 2-body terms
    # UCCSD ansatz: O(n^3 * n_occ) ~ O(n^4) for VQE
    original_cnot_depth = n_qubits ** 4
    original_qubit_count = n_qubits

    # With Kronecker cutting:
    # Each sub-circuit operates on n/2 qubits
    # CNOT depth scales as O((n/2)^4)
    cut_cnot_a = n_a ** 4
    cut_cnot_b = n_b ** 4

    # Number of subcircuits: 2 per rank component
    n_subcircuits = 2 * rank_99

    # Classical overhead: O(rank * 4^max(n_a,n_b)) for combining results
    # But this is exponentially cheaper than running full circuit
    classical_overhead = rank_99

    # Savings
    qubit_reduction = n_qubits / max(n_a, n_b) if max(n_a, n_b) > 0 else 1.0
    depth_reduction = original_cnot_depth / (max(cut_cnot_a, cut_cnot_b) + 1)

    # Feasibility assessment
    if rho1_traceless >= 0.95 and rank_99 <= 3:
        feasibility = "excellent"
        rec = (f"Highly recommended. rho1_traceless={rho1_traceless:.3f} indicates near-perfect "
               f"Kronecker separability. Circuit cutting to {n_a}+{n_b} qubits will preserve "
               f"chemical accuracy with rank-{rank_99} decomposition.")
    elif rho1_traceless >= 0.85 and rank_99 <= 5:
        feasibility = "good"
        rec = (f"Recommended. rho1_traceless={rho1_traceless:.3f} with rank-{rank_99} "
               f"captures 99% of the Hamiltonian norm. Moderate classical overhead "
               f"for combining {n_subcircuits} sub-circuits.")
    elif rho1_traceless >= 0.70:
        feasibility = "marginal"
        rec = (f"Possible but with caveats. rho1_traceless={rho1_traceless:.3f} requires "
               f"rank-{rank_99} for 99% accuracy. Higher rank increases classical overhead "
               f"significantly. Consider alternative partitioning strategies.")
    else:
        feasibility = "not_recommended"
        rec = (f"Not recommended. rho1_traceless={rho1_traceless:.3f} indicates strong "
               f"entanglement between subsystems. Kronecker-based circuit cutting would "
               f"introduce large errors. Consider full-circuit simulation or alternative methods.")

    return QuantumResourceEstimate(
        n_qubits=n_qubits,
        n_qubits_subsystem_a=n_a,
        n_qubits_subsystem_b=n_b,
        rho1_total=rho1_total,
        rho1_traceless=rho1_traceless,
        identity_weight=identity_weight,
        rank_for_99pct=rank_99,
        rank_for_999pct=rank_999,
        original_cnot_depth_estimate=original_cnot_depth,
        original_qubit_count=original_qubit_count,
        cut_cnot_depth_a=cut_cnot_a,
        cut_cnot_depth_b=cut_cnot_b,
        cut_qubit_count_a=n_a,
        cut_qubit_count_b=n_b,
        cut_n_subcircuits=n_subcircuits,
        cut_classical_overhead=classical_overhead,
        qubit_reduction_factor=qubit_reduction,
        depth_reduction_factor=depth_reduction,
        feasibility=feasibility,
        recommendation=rec,
    )


# ============================================================
# High-Level API: KronSolver
# ============================================================

class KronSolver:
    """
    All-in-one Kronecker decomposition solver for molecular Hamiltonians.

    Provides:
    1. Separability analysis (rho1 metrics)
    2. Rank-k Kronecker decomposition
    3. Accelerated ground-state energy estimation
    4. Quantum resource estimation for circuit cutting
    5. Classical simulation of beyond-exact-diag systems

    Usage
    -----
    >>> solver = KronSolver()
    >>> result = solver.analyze("H 0 0 0; H 0 0 0.7", "sto-3g")
    >>> print(result["energy"])
    >>> print(result["quantum_resources"])
    """

    def __init__(self, config: Optional[KronSolverConfig] = None):
        self.config = config or KronSolverConfig()
        self._analyzer = KronBoundaryAnalyzer(
            cut_cfg=CutConfig(cut_nA=self.config.cut_nA),
            build_cfg=BuildConfig(coeff_tol=self.config.coeff_tol),
            piter_cfg=PowerIterConfig(
                max_iter=self.config.piter_max_iter,
                tol=self.config.piter_tol,
                seed=self.config.piter_seed,
            ),
            dense_cfg=DenseDatasetConfig(enabled=False),
            verbose=False,
        )

    def build_hamiltonian(
        self,
        atom: str,
        basis: str,
        charge: int = 0,
        spin: int = 0,
    ) -> Tuple[QubitOperator, int]:
        """Build qubit Hamiltonian from molecular specification."""
        return self._analyzer.build_qubit_operator(atom, basis, charge, spin)

    def analyze(
        self,
        atom: str,
        basis: str,
        charge: int = 0,
        spin: int = 0,
        compute_exact: bool = False,
    ) -> Dict[str, Any]:
        """
        Complete analysis of a molecular system.

        Returns dict with keys:
            - system: basic system info
            - decomposition: KronDecomposition object
            - energy: EnergyResult object
            - quantum_resources: QuantumResourceEstimate object
        """
        t_total = time.time()

        # Build Hamiltonian
        t0 = time.time()
        qubit_op, n_qubits = self.build_hamiltonian(atom, basis, charge, spin)
        t_build = time.time() - t0

        # Decomposition
        decomp_result = decompose_rank_k(qubit_op, n_qubits, self.config)

        # Energy
        energy_result = solve_ground_state(
            qubit_op, n_qubits, self.config, compute_exact=compute_exact,
        )

        # Quantum resources
        qr_result = estimate_quantum_resources(qubit_op, n_qubits, self.config)

        return {
            "system": {
                "atom": atom,
                "basis": basis,
                "charge": charge,
                "spin": spin,
                "n_qubits": n_qubits,
                "dim": 1 << n_qubits,
            },
            "decomposition": decomp_result,
            "energy": energy_result,
            "quantum_resources": qr_result,
            "timing": {
                "build_hamiltonian_sec": t_build,
                "total_sec": time.time() - t_total,
            },
        }

    def analyze_qubit_op(
        self,
        qubit_op: QubitOperator,
        n_qubits: int,
        compute_exact: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze a pre-built QubitOperator (skip PySCF step).
        Useful for external Hamiltonians or custom operators.
        """
        decomp_result = decompose_rank_k(qubit_op, n_qubits, self.config)
        energy_result = solve_ground_state(
            qubit_op, n_qubits, self.config, compute_exact=compute_exact,
        )
        qr_result = estimate_quantum_resources(qubit_op, n_qubits, self.config)

        return {
            "system": {"n_qubits": n_qubits, "dim": 1 << n_qubits},
            "decomposition": decomp_result,
            "energy": energy_result,
            "quantum_resources": qr_result,
        }

    def quick_separability_check(
        self,
        atom: str,
        basis: str,
        charge: int = 0,
        spin: int = 0,
    ) -> Dict[str, Any]:
        """
        Fast check: is this molecule amenable to Kronecker decomposition?

        Returns rho1 metrics + feasibility assessment without full decomposition.
        """
        qubit_op, n_qubits = self.build_hamiltonian(atom, basis, charge, spin)

        total = self._analyzer.compute_rho1_from_paulisum(qubit_op, n_qubits, remove_identity=False)
        tr = self._analyzer.compute_rho1_from_paulisum(qubit_op, n_qubits, remove_identity=True)

        fro2_total = total.get("fro2_coeff", 0.0)
        id_abs2 = total.get("id_abs2", 0.0)
        identity_weight = float(id_abs2 / fro2_total) if fro2_total > 0 else 0.0

        rho1_tr = tr["rho1"]
        if rho1_tr >= 0.95:
            verdict = "EXCELLENT - near-perfect Kronecker separability"
        elif rho1_tr >= 0.85:
            verdict = "GOOD - suitable for rank-1 to rank-3 decomposition"
        elif rho1_tr >= 0.70:
            verdict = "MARGINAL - may need rank-5+ decomposition"
        else:
            verdict = "POOR - strong entanglement, decomposition may not help"

        return {
            "n_qubits": n_qubits,
            "rho1_total": total["rho1"],
            "rho1_traceless": rho1_tr,
            "identity_weight": identity_weight,
            "verdict": verdict,
        }

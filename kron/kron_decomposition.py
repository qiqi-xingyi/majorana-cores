# --*-- coding:utf-8 --*--


from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openfermion import get_sparse_operator, jordan_wigner  # InteractionOperator no longer needed
from openfermion.ops import FermionOperator, QubitOperator
from pyscf import ao2mo, gto, scf


# ============================================================
# Spatial-orbital integrals -> FermionOperator (spin-orbital)
# ============================================================

def _build_fermion_op_from_spatial_integrals(
    h1_mo: np.ndarray,
    h2_mo: np.ndarray,
    nuclear_repulsion: float,
    tol: float = 1e-12,
) -> FermionOperator:
    """
    Build a FermionOperator in the spin-orbital basis directly from
    spatial-orbital MO integrals produced by PySCF.

    Spin-orbital convention:  p = 2*i + sigma,  sigma=0 (alpha), 1 (beta).

    Parameters
    ----------
    h1_mo : (n_spatial, n_spatial) array
        One-electron integrals in the MO basis.
    h2_mo : (n_spatial, n_spatial, n_spatial, n_spatial) array
        Two-electron integrals in **chemist** notation: (ij|kl).
    nuclear_repulsion : float
        Nuclear repulsion energy (constant term).
    tol : float
        Drop terms with |coeff| <= tol.

    Returns
    -------
    FermionOperator
        H = E_nuc + sum_{pq} h1[p,q] a†_p a_q
                  + 0.5 * sum_{pqrs} <pq|rs> a†_p a†_q a_s a_r
        where <pq|rs> is the physicist two-electron integral.
    """
    n_spatial = int(h1_mo.shape[0])
    n_so = 2 * n_spatial

    fermion_op = FermionOperator((), nuclear_repulsion)

    # --- one-body: h1_so[p,q] = h1_mo[i,j] * delta(sigma_p, sigma_q) ---
    for i in range(n_spatial):
        for j in range(n_spatial):
            coeff = h1_mo[i, j]
            if abs(coeff) <= tol:
                continue
            for sigma in (0, 1):
                p = 2 * i + sigma
                q = 2 * j + sigma
                fermion_op += FermionOperator(((p, 1), (q, 0)), coeff)

    # --- two-body: 0.5 * <pq|rs> * a†_p a†_q a_s a_r ---
    # Chemist-to-physicist: <pq|rs> = (pr|qs)
    # h2_mo is in chemist notation (ij|kl), so:
    #   <pq|rs>_spin = delta(sp,sr) * delta(sq,ss) * h2_mo[ip, ir, iq, is]
    for i in range(n_spatial):
        for k in range(n_spatial):
            for j in range(n_spatial):
                for l in range(n_spatial):
                    # (ik|jl)_chemist = <ij|kl>_physicist
                    val = h2_mo[i, k, j, l]
                    if abs(val) <= tol:
                        continue
                    for s1 in (0, 1):       # spin of p and r
                        p = 2 * i + s1      # spatial i
                        r = 2 * k + s1      # spatial k
                        for s2 in (0, 1):   # spin of q and s
                            q = 2 * j + s2  # spatial j
                            s = 2 * l + s2  # spatial l
                            # 0.5 * <pq|rs> * a†_p a†_q a_s a_r
                            fermion_op += FermionOperator(
                                ((p, 1), (q, 1), (s, 0), (r, 0)), 0.5 * val
                            )

    return fermion_op


# ============================================================
# Configs
# ============================================================

@dataclass
class CutConfig:
    cut_nA: Optional[int] = None  # None => balanced split


@dataclass
class BuildConfig:
    coeff_tol: float = 1e-12  # drop tiny Pauli coefficients when building


@dataclass
class PowerIterConfig:
    max_iter: int = 200
    tol: float = 1e-8
    seed: int = 0


@dataclass
class DenseDatasetConfig:
    enabled: bool = True
    max_dense_qubits: int = 12                 # only materialize dense H if n_qubits <= this
    out_dir: str = "dense_kron_dataset"        # per-system npz goes here
    save_factors: bool = True                 # save A,B,alpha in addition to H and H_approx
    remove_identity_before_decomp: bool = True # do traceless decomp for dataset (recommended)
    save_h_traceless: bool = False            # optionally also store H_traceless
    npz_compress: bool = True                 # use savez_compressed
    overwrite: bool = False                   # if False and file exists -> skip saving


# ============================================================
# Pauli-sum -> sparse coefficient matrix C (operator-Schmidt)
# ============================================================

def _term_to_ab_keys(
    term: Tuple[Tuple[int, str], ...],
    n_a: int,
    n_b: int,
) -> Tuple[str, str]:
    a_ops = ["I"] * n_a
    b_ops = ["I"] * n_b
    for q, op in term:
        if q < n_a:
            a_ops[q] = op
        else:
            b_ops[q - n_a] = op
    return ("".join(a_ops), "".join(b_ops))


def _build_sparse_C_from_qubit_op(
    qubit_op: QubitOperator,
    n_qubits: int,
    cut_cfg: CutConfig,
    build_cfg: BuildConfig,
    remove_identity: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, float]:
    n_a = cut_cfg.cut_nA if cut_cfg.cut_nA is not None else (n_qubits // 2)
    n_b = n_qubits - n_a

    a_index: Dict[str, int] = {}
    b_index: Dict[str, int] = {}
    pair_acc: Dict[Tuple[int, int], complex] = {}

    id_a = "I" * n_a
    id_b = "I" * n_b
    id_coeff: complex = 0.0 + 0.0j

    for term, coeff in qubit_op.terms.items():
        c = complex(coeff)
        if abs(c) <= build_cfg.coeff_tol:
            continue

        if len(term) == 0:
            # identity
            if not remove_identity:
                id_coeff += c
            continue

        a_key, b_key = _term_to_ab_keys(term, n_a=n_a, n_b=n_b)
        if remove_identity and (a_key == id_a) and (b_key == id_b):
            continue

        ia = a_index.get(a_key)
        if ia is None:
            ia = len(a_index)
            a_index[a_key] = ia

        ib = b_index.get(b_key)
        if ib is None:
            ib = len(b_index)
            b_index[b_key] = ib

        key = (ia, ib)
        pair_acc[key] = pair_acc.get(key, 0.0 + 0.0j) + c

    if (not remove_identity) and (abs(id_coeff) > build_cfg.coeff_tol):
        ia = a_index.get(id_a)
        if ia is None:
            ia = len(a_index)
            a_index[id_a] = ia
        ib = b_index.get(id_b)
        if ib is None:
            ib = len(b_index)
            b_index[id_b] = ib
        key = (ia, ib)
        pair_acc[key] = pair_acc.get(key, 0.0 + 0.0j) + id_coeff

    nnz = len(pair_acc)
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    vals = np.empty(nnz, dtype=np.complex128)

    fro2 = 0.0
    for k, ((r, c), v) in enumerate(pair_acc.items()):
        rows[k] = r
        cols[k] = c
        vals[k] = v
        fro2 += (v.real * v.real + v.imag * v.imag)

    id_abs2 = float(id_coeff.real * id_coeff.real + id_coeff.imag * id_coeff.imag)
    return rows, cols, vals, len(a_index), len(b_index), float(fro2), float(id_abs2)


def _matvec_C(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n_rows: int, x: np.ndarray) -> np.ndarray:
    y = np.zeros(n_rows, dtype=np.complex128)
    # IMPORTANT: rows contains repeats; fancy indexing does NOT accumulate correctly for repeats.
    np.add.at(y, rows, vals * x[cols])
    return y


def _matvec_CH(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n_cols: int, x: np.ndarray) -> np.ndarray:
    y = np.zeros(n_cols, dtype=np.complex128)
    # IMPORTANT: cols contains repeats; use np.add.at for correct accumulation.
    np.add.at(y, cols, np.conjugate(vals) * x[rows])
    return y


def _power_iteration_top_singular(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    n_rows: int,
    n_cols: int,
    cfg: PowerIterConfig,
) -> Tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    v = (rng.standard_normal(n_cols) + 1j * rng.standard_normal(n_cols)).astype(np.complex128)
    v_norm = float(np.linalg.norm(v))
    if v_norm == 0.0:
        v[0] = 1.0 + 0.0j
        v_norm = 1.0
    v /= v_norm

    s_prev = 0.0
    u = np.zeros(n_rows, dtype=np.complex128)

    for _ in range(cfg.max_iter):
        u_raw = _matvec_C(rows, cols, vals, n_rows, v)
        s = float(np.linalg.norm(u_raw))
        if s == 0.0:
            return 0.0, u, v
        u = u_raw / s

        v_raw = _matvec_CH(rows, cols, vals, n_cols, u)
        t = float(np.linalg.norm(v_raw))
        if t == 0.0:
            return 0.0, u, v
        v = v_raw / t

        if s_prev > 0.0 and abs(s - s_prev) / s_prev < cfg.tol:
            s_prev = s
            break
        s_prev = s

    return float(s_prev), u, v


# ============================================================
# Dense operator-Schmidt top-1 without explicitly forming M
# ============================================================

def _dense_traceless(H: np.ndarray) -> Tuple[np.ndarray, complex]:
    """H_tr = H - c0 I, where c0 = Tr(H)/dim."""
    dim = H.shape[0]
    c0 = np.trace(H) / complex(dim)
    H_tr = H - c0 * np.eye(dim, dtype=H.dtype)
    return H_tr, c0


def _dense_top1_schmidt_via_matvec(
    H: np.ndarray,
    n_a: int,
    piter: PowerIterConfig,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute top singular of M induced by reshape:
      H(iA,iB,jA,jB) -> M((iA,jA),(iB,jB)).
    We avoid building M by contracting the 4-tensor.
    Returns s1, u, v with u in C^{dim_a^2}, v in C^{dim_b^2}.
    """
    n = int(round(math.log2(H.shape[0])))
    dim = 1 << n
    assert H.shape == (dim, dim)
    dim_a = 1 << n_a
    dim_b = dim // dim_a

    # T shape: (iA, iB, jA, jB) then permute to (iA, jA, iB, jB)
    T = H.reshape(dim_a, dim_b, dim_a, dim_b).transpose(0, 2, 1, 3)
    # Now M is reshape(T, (dim_a^2, dim_b^2)) but we won't materialize it.

    def matvec_M(x: np.ndarray) -> np.ndarray:
        X = x.reshape(dim_b, dim_b)  # (iB, jB)
        Y = np.tensordot(T, X, axes=([2, 3], [0, 1]))  # (iA, jA)
        return Y.reshape(dim_a * dim_a)

    def matvec_MH(x: np.ndarray) -> np.ndarray:
        X = x.reshape(dim_a, dim_a)  # (iA, jA)
        Y = np.tensordot(np.conjugate(T), X, axes=([0, 1], [0, 1]))  # (iB, jB)
        return Y.reshape(dim_b * dim_b)

    rng = np.random.default_rng(piter.seed)
    v = (rng.standard_normal(dim_b * dim_b) + 1j * rng.standard_normal(dim_b * dim_b)).astype(np.complex128)
    v_norm = float(np.linalg.norm(v))
    if v_norm == 0.0:
        v[0] = 1.0 + 0.0j
        v_norm = 1.0
    v /= v_norm

    s_prev = 0.0
    u = np.zeros(dim_a * dim_a, dtype=np.complex128)

    for _ in range(piter.max_iter):
        u_raw = matvec_M(v)
        s = float(np.linalg.norm(u_raw))
        if s == 0.0:
            return 0.0, u, v
        u = u_raw / s

        v_raw = matvec_MH(u)
        t = float(np.linalg.norm(v_raw))
        if t == 0.0:
            return 0.0, u, v
        v = v_raw / t

        if s_prev > 0.0 and abs(s - s_prev) / s_prev < piter.tol:
            s_prev = s
            break
        s_prev = s

    return float(s_prev), u, v




# ============================================================
# Main class
# ============================================================

class KronBoundaryAnalyzer:
    """
    Pipeline:
      - Build qubit Hamiltonian in Pauli-sum form (QubitOperator) using PySCF + OpenFermion + JW
      - Compute rho1 diagnostics (total / traceless)
      - Compute rank-1 decomposition factors in Pauli space:
            H ≈ alpha * (A ⊗ B)
        where A,B are returned as Pauli-key dictionaries (unit-norm in coefficient space)
      - Optional: If n_qubits <= max_dense_qubits, materialize dense H and save dataset:
            input: H_dense
            output: H_approx_dense = kron(A_dense, B_dense)  (balanced split + balanced scaling)
    """

    def __init__(
        self,
        cut_cfg: Optional[CutConfig] = None,
        build_cfg: Optional[BuildConfig] = None,
        piter_cfg: Optional[PowerIterConfig] = None,
        dense_cfg: Optional[DenseDatasetConfig] = None,
        max_qubits: Optional[int] = None,
        verbose: bool = True,
    ):
        self.cut_cfg = cut_cfg if cut_cfg is not None else CutConfig()
        self.build_cfg = build_cfg if build_cfg is not None else BuildConfig()
        self.piter_cfg = piter_cfg if piter_cfg is not None else PowerIterConfig()
        self.dense_cfg = dense_cfg if dense_cfg is not None else DenseDatasetConfig()
        self.max_qubits = max_qubits
        self.verbose = verbose

    # -----------------------
    # Step 1: build qubit_op
    # -----------------------
    def build_qubit_operator(
        self,
        atom: str,
        basis: str,
        charge: int,
        spin: int,
    ) -> Tuple[QubitOperator, int]:
        if self.verbose:
            print(f"[build] basis='{basis}', charge={charge}, spin={spin}")

        mol = gto.Mole()
        mol.atom = atom
        mol.basis = basis
        mol.charge = int(charge)
        mol.spin = int(spin)
        mol.build()

        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.run()

        if not mf.converged:
            print(f"[WARN] RHF SCF did NOT converge for atom='{atom}', basis='{basis}'")

        # Spatial orbitals in RHF (MO count)
        n_spatial_orb = int(mf.mo_coeff.shape[1])
        # Spin-orbital modes = 2 * spatial orbitals
        n_spin_orb = 2 * n_spatial_orb
        n_qubits = n_spin_orb

        if self.max_qubits is not None and n_qubits > int(self.max_qubits):
            raise ValueError(f"n_qubits={n_qubits} exceeds max_qubits={self.max_qubits}")

        # MO-basis integrals (spatial orbitals)
        h1 = mf.get_hcore()
        mo_coeff = mf.mo_coeff
        h1_mo = np.einsum("pi,pq,qj->ij", mo_coeff, h1, mo_coeff)

        eri_mo = ao2mo.kernel(mol, mo_coeff)
        h2_mo = ao2mo.restore(1, eri_mo, n_spatial_orb)  # chemist notation (ij|kl)

        constant = float(mol.energy_nuc())

        # Build FermionOperator directly from spatial integrals
        # (correct chemist→physicist index mapping + proper destruction operator ordering)
        fermion_op = _build_fermion_op_from_spatial_integrals(
            h1_mo=h1_mo,
            h2_mo=h2_mo,
            nuclear_repulsion=constant,
            tol=self.build_cfg.coeff_tol,
        )

        qubit_op = jordan_wigner(fermion_op)
        return qubit_op, int(n_qubits)

    # -----------------------
    # Dense materialization (only for small systems)
    # -----------------------
    def materialize_dense_H(
        self,
        qubit_op: QubitOperator,
        n_qubits: int,
        dtype: np.dtype = np.complex128,
    ) -> np.ndarray:
        sparse_H = get_sparse_operator(qubit_op, n_qubits=n_qubits)
        H = sparse_H.toarray().astype(dtype, copy=False)
        return H

    # -----------------------
    # rho1 diagnostics on Pauli-sum
    # -----------------------
    def compute_rho1_from_paulisum(
        self,
        qubit_op: QubitOperator,
        n_qubits: int,
        remove_identity: bool,
    ) -> Dict[str, Any]:
        rows, cols, vals, n_rows, n_cols, fro2, id_abs2 = _build_sparse_C_from_qubit_op(
            qubit_op=qubit_op,
            n_qubits=n_qubits,
            cut_cfg=self.cut_cfg,
            build_cfg=self.build_cfg,
            remove_identity=remove_identity,
        )

        if fro2 <= 0.0 or len(vals) == 0:
            return {
                "rho1": 0.0,
                "s1": 0.0,
                "fro2_coeff": float(fro2),
                "n_rows": int(n_rows),
                "n_cols": int(n_cols),
                "nnz": int(len(vals)),
                "id_abs2": float(id_abs2),
            }

        s1, u, v = _power_iteration_top_singular(rows, cols, vals, n_rows, n_cols, self.piter_cfg)
        rho1 = float((s1 * s1) / fro2) if fro2 > 0 else 0.0

        return {
            "rho1": float(rho1),
            "s1": float(s1),
            "fro2_coeff": float(fro2),
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "nnz": int(len(vals)),
            "id_abs2": float(id_abs2),
            "u": u,
            "v": v,
        }

    # -----------------------
    # rank-1 factors in Pauli space (balanced scaling)
    # -----------------------
    def decompose_rank1_pauli_factors(
        self,
        qubit_op: QubitOperator,
        n_qubits: int,
        traceless: bool = True,
        drop_tol: float = 0.0,
        max_terms: Optional[int] = None,
    ) -> Dict[str, Any]:
        n_a = self.cut_cfg.cut_nA if self.cut_cfg.cut_nA is not None else (n_qubits // 2)
        n_b = n_qubits - n_a
        id_a = "I" * n_a
        id_b = "I" * n_b

        a_index: Dict[str, int] = {}
        b_index: Dict[str, int] = {}
        pair_acc: Dict[Tuple[int, int], complex] = {}

        id_coeff: complex = 0.0 + 0.0j
        for term, coeff in qubit_op.terms.items():
            c = complex(coeff)
            if abs(c) <= self.build_cfg.coeff_tol:
                continue

            if len(term) == 0:
                id_coeff += c
                continue

            a_key, b_key = _term_to_ab_keys(term, n_a=n_a, n_b=n_b)
            if traceless and (a_key == id_a) and (b_key == id_b):
                continue

            ia = a_index.get(a_key)
            if ia is None:
                ia = len(a_index)
                a_index[a_key] = ia

            ib = b_index.get(b_key)
            if ib is None:
                ib = len(b_index)
                b_index[b_key] = ib

            key = (ia, ib)
            pair_acc[key] = pair_acc.get(key, 0.0 + 0.0j) + c

        if (not traceless) and (abs(id_coeff) > self.build_cfg.coeff_tol):
            ia = a_index.get(id_a)
            if ia is None:
                ia = len(a_index)
                a_index[id_a] = ia
            ib = b_index.get(id_b)
            if ib is None:
                ib = len(b_index)
                b_index[id_b] = ib
            key = (ia, ib)
            pair_acc[key] = pair_acc.get(key, 0.0 + 0.0j) + id_coeff

        nnz = len(pair_acc)
        if nnz == 0:
            return {
                "alpha": 0.0,
                "rho1": 0.0,
                "A_pauli_unit": {},
                "B_pauli_unit": {},
                "cut_nA": int(n_a),
                "cut_nB": int(n_b),
            }

        rows = np.empty(nnz, dtype=np.int64)
        cols = np.empty(nnz, dtype=np.int64)
        vals = np.empty(nnz, dtype=np.complex128)
        fro2 = 0.0
        for k, ((r, c), v) in enumerate(pair_acc.items()):
            rows[k] = r
            cols[k] = c
            vals[k] = v
            fro2 += (v.real * v.real + v.imag * v.imag)

        n_rows = len(a_index)
        n_cols = len(b_index)

        s1, u, v = _power_iteration_top_singular(rows, cols, vals, n_rows, n_cols, self.piter_cfg)
        rho1 = float((s1 * s1) / fro2) if fro2 > 0 else 0.0

        # index->key lists
        a_keys = [None] * n_rows
        for k, i in a_index.items():
            a_keys[i] = k
        b_keys = [None] * n_cols
        for k, i in b_index.items():
            b_keys[i] = k

        # balanced factorization: C ≈ s1 * u * conj(v)^T
        # A uses u, B uses conj(v), both unit-norm in coefficient space
        def vec_to_dict(keys: List[str], vec: np.ndarray) -> Dict[str, complex]:
            pairs: List[Tuple[str, complex]] = []
            for i, c in enumerate(vec):
                z = complex(c)
                if drop_tol > 0.0 and abs(z) <= drop_tol:
                    continue
                pairs.append((keys[i], z))
            if max_terms is not None and len(pairs) > int(max_terms):
                pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
                pairs = pairs[: int(max_terms)]
            return {k: v for k, v in pairs}

        A_dict = vec_to_dict(a_keys, u)
        B_dict = vec_to_dict(b_keys, np.conjugate(v))

        return {
            "alpha": float(s1),
            "rho1": float(rho1),
            "A_pauli_unit": A_dict,
            "B_pauli_unit": B_dict,
            "cut_nA": int(n_a),
            "cut_nB": int(n_b),
        }

    # -----------------------
    # Dense rank-1 Kronecker approx (balanced split + balanced scaling)
    # -----------------------
    def decompose_dense_rank1_kron(
        self,
        H: np.ndarray,
        n_qubits: int,
        traceless: bool,
    ) -> Dict[str, Any]:
        n_a = self.cut_cfg.cut_nA if self.cut_cfg.cut_nA is not None else (n_qubits // 2)
        dim = H.shape[0]
        assert H.shape == (dim, dim)
        assert (dim & (dim - 1)) == 0, "dim must be power of 2"
        assert (1 << n_qubits) == dim

        H_use = H
        c0 = 0.0 + 0.0j
        if traceless:
            H_use, c0 = _dense_traceless(H)

        s1, u, v = _dense_top1_schmidt_via_matvec(H_use, n_a=n_a, piter=self.piter_cfg)

        dim_a = 1 << n_a
        dim_b = dim // dim_a

        # Balanced scaling:
        # M ≈ s1 * u * v^H  =>  vec(A)=sqrt(s1)*u, vec(B)=sqrt(s1)*conj(v)
        # so ||A||_F ≈ ||B||_F ≈ sqrt(s1)
        sroot = math.sqrt(max(s1, 0.0))
        A = (sroot * u).reshape(dim_a, dim_a)
        B = (sroot * np.conjugate(v)).reshape(dim_b, dim_b)

        H_approx = np.kron(A, B)

        # If we decomposed traceless part, compare in the original space by adding back c0 I.
        if traceless:
            H_approx_full = H_approx + c0 * np.eye(dim, dtype=H.dtype)
        else:
            H_approx_full = H_approx

        fro = float(np.linalg.norm(H - H_approx_full))
        fro_rel = float(fro / (np.linalg.norm(H) + 1e-30))

        return {
            "cut_nA": int(n_a),
            "cut_nB": int(n_qubits - n_a),
            "c0_identity": complex(c0),
            "s1": float(s1),
            "A": A.astype(np.complex128, copy=False),
            "B": B.astype(np.complex128, copy=False),
            "H_approx": H_approx_full.astype(np.complex128, copy=False),
            "frob_abs": float(fro),
            "frob_rel": float(fro_rel),
        }

    # -----------------------
    # Save dense dataset sample
    # -----------------------
    @staticmethod
    def _sanitize(name: str) -> str:
        return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in name)

    def maybe_save_dense_dataset(
        self,
        name: str,
        H: np.ndarray,
        n_qubits: int,
        dense_decomp: Dict[str, Any],
    ) -> Optional[str]:
        cfg = self.dense_cfg
        if not cfg.enabled:
            return None
        if n_qubits > int(cfg.max_dense_qubits):
            return None

        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = self._sanitize(name)
        fp = out_dir / f"{safe}.npz"

        if fp.exists() and (not cfg.overwrite):
            return str(fp)

        payload: Dict[str, Any] = {
            "name": np.array(name),
            "n_qubits": np.array(int(n_qubits)),
            "dim": np.array(int(H.shape[0])),
            "cut_nA": np.array(int(dense_decomp["cut_nA"])),
            "cut_nB": np.array(int(dense_decomp["cut_nB"])),
            "H": H.astype(np.complex128, copy=False),
            "H_approx": dense_decomp["H_approx"].astype(np.complex128, copy=False),
            "s1": np.array(float(dense_decomp["s1"])),
            "frob_abs": np.array(float(dense_decomp["frob_abs"])),
            "frob_rel": np.array(float(dense_decomp["frob_rel"])),
            "c0_identity": np.array(dense_decomp["c0_identity"]),
        }

        if cfg.save_h_traceless and cfg.remove_identity_before_decomp:
            H_tr, _ = _dense_traceless(H.astype(np.complex128, copy=False))
            payload["H_traceless"] = H_tr

        if cfg.save_factors:
            payload["A"] = dense_decomp["A"].astype(np.complex128, copy=False)
            payload["B"] = dense_decomp["B"].astype(np.complex128, copy=False)

        if cfg.npz_compress:
            np.savez_compressed(fp, **payload)
        else:
            np.savez(fp, **payload)

        return str(fp)

    # -----------------------
    # One-shot API
    # -----------------------
    def analyze_system(
        self,
        name: str,
        atom: str,
        basis: str,
        charge: int,
        spin: int,
        factor_drop_tol: float = 0.0,
        factor_max_terms: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Build Pauli-sum Hamiltonian
        qubit_op, n_qubits = self.build_qubit_operator(atom=atom, basis=basis, charge=charge, spin=spin)

        # rho1 (total & traceless) from Pauli-sum
        total = self.compute_rho1_from_paulisum(qubit_op, n_qubits, remove_identity=False)
        tr = self.compute_rho1_from_paulisum(qubit_op, n_qubits, remove_identity=True)

        fro2_total = float(total.get("fro2_coeff", 0.0))
        id_abs2 = float(total.get("id_abs2", 0.0))
        identity_weight = float(id_abs2 / fro2_total) if fro2_total > 0.0 else 0.0

        # Rank-1 factors in Pauli space (traceless by default)
        factors = self.decompose_rank1_pauli_factors(
            qubit_op=qubit_op,
            n_qubits=n_qubits,
            traceless=True,
            drop_tol=factor_drop_tol,
            max_terms=factor_max_terms,
        )

        out: Dict[str, Any] = {
            "name": name,
            "atom": atom,
            "basis": basis,
            "charge": int(charge),
            "spin": int(spin),
            "n_qubits": int(n_qubits),

            "rho1_total": float(total["rho1"]),
            "rho1_traceless": float(tr["rho1"]),
            "identity_weight": float(identity_weight),

            "rank1_alpha_traceless": float(factors["alpha"]),
            "cut_nA": int(factors["cut_nA"]),
            "cut_nB": int(factors["cut_nB"]),
            "A_pauli_unit": factors["A_pauli_unit"],
            "B_pauli_unit": factors["B_pauli_unit"],

            # dense dataset path if saved
            "dense_dataset_path": "",
        }

        # Optional dense dataset generation for small n_qubits
        if self.dense_cfg.enabled and n_qubits <= int(self.dense_cfg.max_dense_qubits):
            H_dense = self.materialize_dense_H(qubit_op, n_qubits, dtype=np.complex128)
            dense_decomp = self.decompose_dense_rank1_kron(
                H=H_dense,
                n_qubits=n_qubits,
                traceless=bool(self.dense_cfg.remove_identity_before_decomp),
            )
            fp = self.maybe_save_dense_dataset(name=name, H=H_dense, n_qubits=n_qubits, dense_decomp=dense_decomp)
            out["dense_dataset_path"] = fp if fp is not None else ""

            # optional: also return a couple of scalar errors for quick inspection
            out["dense_frob_rel"] = float(dense_decomp["frob_rel"])
            out["dense_s1"] = float(dense_decomp["s1"])

        return out


# ============================================================
# Minimal example
# ============================================================
if __name__ == "__main__":
    analyzer = KronBoundaryAnalyzer(
        cut_cfg=CutConfig(cut_nA=None),  # balanced
        build_cfg=BuildConfig(coeff_tol=1e-12),
        piter_cfg=PowerIterConfig(max_iter=200, tol=1e-8, seed=0),
        dense_cfg=DenseDatasetConfig(
            enabled=True,
            max_dense_qubits=12,
            out_dir="dense_kron_dataset",
            save_factors=True,
            remove_identity_before_decomp=True,
            save_h_traceless=False,
            npz_compress=True,
            overwrite=False,
        ),
        max_qubits=None,
        verbose=True,
    )

    rec = analyzer.analyze_system(
        name="H2_STO3G_R0.7",
        atom="H 0.000 0.000 0.000; H 0.000 0.000 0.700",
        basis="sto-3g",
        charge=0,
        spin=0,
        factor_drop_tol=1e-10,
        factor_max_terms=5000,
    )

    print(f"\nname={rec['name']}")
    print(f"n_qubits={rec['n_qubits']}")
    print(
        f"rho1_total={rec['rho1_total']:.6f} | "
        f"rho1_traceless={rec['rho1_traceless']:.6f} | "
        f"identity_weight={rec['identity_weight']:.6f}"
    )
    print(f"alpha(traceless)={rec['rank1_alpha_traceless']:.6f} | cut=({rec['cut_nA']},{rec['cut_nB']})")
    print(f"|A_terms|={len(rec['A_pauli_unit'])}, |B_terms|={len(rec['B_pauli_unit'])}")
    if rec.get("dense_dataset_path", ""):
        print(f"dense_dataset_path={rec['dense_dataset_path']}")
        print(f"dense_frob_rel={rec.get('dense_frob_rel', float('nan')):.6e}")

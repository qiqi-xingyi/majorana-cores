# Exact Majorana-degree cores of quantum-chemistry Hamiltonians

Code for "Exact Majorana-degree cores explain the cut spectra of quantum-chemistry Hamiltonians" (npj Quantum Information)
by Yuqi Zhang, Sixu Chen, Bo Fang, Feixiong Cheng and Qiang Guan.

The repository holds the core method scripts, in Python: the coefficient-space solver and the modules that implement the
exact cores, their channels, the certificates, the integral factorizations and the storage formats. The inputs and results
of the calculations are in the data record cited in the paper.

## Environment

Python 3.11 with NumPy, SciPy, PySCF and OpenFermion (versions in `requirements.txt`). The modules in `methods/` import
each other by name: put `methods/` and the repository root on `PYTHONPATH`.

## Contents

| Path | Contents |
|---|---|
| `kron/kron_solver.py`, `kron/kron_decomposition.py` | Coefficient-space solver: truncated decomposition of the coefficient matrix by sparse deflation with the residual applied implicitly, returned-factor residual and certificates |
| `methods/universal_core.py` | Exact core of the coefficient matrix at a contiguous cut: coefficient SVD from shared implicit factors |
| `methods/theory_channels.py`, `methods/near_geometry_factors.py` | Majorana-degree and parity channels, fermionic mode permutations and block factor primitives |
| `methods/pivoted_cholesky_factors.py`, `methods/df_full_cut.py`, `methods/fair_coulomb_baselines.py`, `methods/scaled_coulomb_probe.py` | Cholesky and density-fitting factors of the two-electron integrals and their complete Pauli-cut cores; standard DF/CD controls |
| `methods/dyadic_spectral_certificate.py`, `methods/coulomb_nuclear_certificate.py`, `methods/compact_factor_queries.py` | Exact dyadic a posteriori certificates for small Hermitian matrices and Coulomb residuals |
| `methods/energy_numerics.py` | Sector-energy queries and floating-point enclosures for the energy error bounds |
| `methods/factor_storage.py`, `methods/packed_array_archive.py`, `methods/compact_packet_polish.py` | Storage formats for factors and sparse Pauli inputs |
| `methods/io_helpers.py` | JSON and CSV output helpers |

## License

MIT License (`LICENSE`).

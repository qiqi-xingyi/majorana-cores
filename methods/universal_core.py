"""Universal finite-body coefficient SVD using shared implicit factors.

No charge dictionary, full Fock factor,
coefficient threshold, randomized sketch, hashing strategy, or fixed orbital is
used. The theorem is exact; floating-point spectra below are diagnostics, not
validated singular-value intervals. Actual saved packets separately receive
an exact full-Fock Gram residual certificate after readback.
"""
from __future__ import annotations
from bisect import bisect_left
import itertools
import math
import time



def _deps():
    import numpy as np
    from scipy import linalg, sparse
    return np, linalg, sparse


def _unique_sorted(items):
    return [key for key, _ in itertools.groupby(sorted(items))]


def _comb_rank(indices, n):
    """Lexicographic combination coordinate, no coefficient lookup table."""
    d = len(indices)
    rank = 0
    previous = -1
    for position, index in enumerate(indices):
        rank += math.comb(n-previous-1, d-position) - math.comb(n-index, d-position)
        previous = index
    return rank


def canonical_pauli(case):
    """Combine duplicate real Hermitian Pauli terms by fsum; retain every nonzero."""
    np, _, _ = _deps()
    n = int(case['n_qubits'])
    labels = list(map(str, case['pauli_labels']))
    coefficients = list(case['coefficients'])
    if len(labels) != len(coefficients):
        raise ValueError('Pauli labels and coefficients have different lengths')
    if any(len(label) != n or any(ch not in 'IXYZ' for ch in label) for label in labels):
        raise ValueError('Pauli labels do not match the declared register')
    records = sorted(zip(labels, coefficients), key=lambda row: row[0])
    output = []
    identity = complex(case.get('identity', 0.))
    if not math.isfinite(identity.real) or not math.isfinite(identity.imag):
        raise ValueError('Nonfinite identity coefficient')
    for label, group in itertools.groupby(records, key=lambda row: row[0]):
        vals = [complex(row[1]) for row in group]
        if any(not math.isfinite(z.real) or not math.isfinite(z.imag) for z in vals):
            raise ValueError('Nonfinite Pauli coefficient')
        value = complex(math.fsum(z.real for z in vals), math.fsum(z.imag for z in vals))
        if value.imag != 0:
            raise ValueError('Hermitian Pauli coefficients must be exactly real')
        if label == 'I'*n:
            identity += value
        elif value.real != 0:
            output.append((label, value.real))
    if identity.imag != 0:
        raise ValueError('Identity coefficient must be exactly real')
    return dict(n_qubits=n, identity=float(identity.real),
                labels=np.asarray([row[0] for row in output], dtype=f'U{n}'),
                coefficients=np.asarray([row[1] for row in output], dtype=float),
                duplicate_combination='math.fsum; exactly zero sums only removed')


def build_channels(case, cut):
    """Parse the universal 0/2/4 Majorana channels directly from cached Pauli."""
    np, _, _ = _deps()
    import theory_channels as tc
    started = time.perf_counter()
    source = canonical_pauli(case)
    n = source['n_qubits']
    cut = int(cut)
    if not 0 < cut < n:
        raise ValueError('Cut must lie strictly inside the register')
    na, nb = 2*cut, 2*(n-cut)
    pa, pb = math.comb(na, 2), math.comb(nb, 2)
    ca = math.comb(na, 3) if na >= 3 else 0
    cb = math.comb(nb, 3) if nb >= 3 else 0
    a2, b2 = np.zeros(pa), np.zeros(pb)
    j11, j13 = np.zeros((na, nb)), np.zeros((na, cb))
    j31, j22 = np.zeros((ca, nb)), np.zeros((pa, pb))
    a4, b4 = [], []
    for label, coefficient in zip(source['labels'], source['coefficients']):
        indices, phase = tc.pauli_majorana(str(label))
        value = complex(float(coefficient))/phase
        if value.imag != 0:
            raise AssertionError('Hermitian Majorana coefficient is not real')
        value = float(value.real)
        degree = len(indices)
        if degree not in (2, 4):
            raise ValueError('Nonzero term outside one/two-body Majorana degree class: '
                             +str((label, indices, value)))
        left = tuple(j for j in indices if j < na)
        right = tuple(j-na for j in indices if j >= na)
        if not right:
            if degree == 2:
                a2[_comb_rank(left, na)] = value
            else:
                a4.append((left, value))
            continue
        if not left:
            if degree == 2:
                b2[_comb_rank(right, nb)] = value
            else:
                b4.append((right, value))
            continue
        degrees = (len(left), len(right))
        i, j = _comb_rank(left, na), _comb_rank(right, nb)
        if degrees == (1, 1):
            j11[i, j] = value
        elif degrees == (1, 3):
            j13[i, j] = -value
        elif degrees == (3, 1):
            j31[i, j] = -value
        elif degrees == (2, 2):
            j22[i, j] = value
        else:
            raise AssertionError('Unexpected allowed-degree split')
    a4.sort(key=lambda row: row[0])
    b4.sort(key=lambda row: row[0])
    result = dict(n_qubits=n, cut=cut, nA=na, nB=nb, a2=a2, b2=b2,
        J11=j11, J13=j13, J31=j31, J22=j22,
        a4_labels=np.asarray([row[0] for row in a4], dtype=np.uint64).reshape(-1, 4),
        b4_labels=np.asarray([row[0] for row in b4], dtype=np.uint64).reshape(-1, 4),
        a4=np.asarray([row[1] for row in a4], dtype=float),
        b4=np.asarray([row[1] for row in b4], dtype=float),
        source=source, parsing_sec=time.perf_counter()-started)
    result['kernel_numeric_bytes'] = sum(result[key].nbytes for key in
        ('a2', 'b2', 'a4', 'b4', 'J11', 'J13', 'J31', 'J22'))
    result['quartic_index_bytes'] = result['a4_labels'].nbytes+result['b4_labels'].nbytes
    return result


def small_cores(channels):
    """Fresh economy QR and exact-structure cores; no threshold-based rank choice.

    Economy frames may include orthogonal complements for exactly dependent
    columns. This preserves the exact intertwining identity and dimension bound;
    it does not claim a numerical decision about the exact cubic rank.
    """
    np, la, _ = _deps()
    ch = channels
    pa, pb = len(ch['a2']), len(ch['b2'])
    alpha, beta = float(la.norm(ch['a4'])), float(la.norm(ch['b4']))
    ea, eb = int(alpha != 0.), int(beta != 0.)
    ke = np.zeros((1+pa+ea, 1+pb+eb))
    ke[0, 1:1+pb] = ch['b2']
    ke[1:1+pa, 0] = ch['a2']
    ke[1:1+pa, 1:1+pb] = ch['J22']
    if ea:
        ke[-1, 0] = alpha
    if eb:
        ke[0, -1] = beta
    def qr_or_empty(matrix):
        if min(matrix.shape) == 0 or np.count_nonzero(matrix) == 0:
            return np.empty((matrix.shape[0], 0)), np.empty((0, matrix.shape[1]))
        return la.qr(matrix, mode='economic', check_finite=False)
    qa, ra = qr_or_empty(ch['J31'])
    qb, rb = qr_or_empty(ch['J13'].T)
    lb = rb.T
    na, nb = ch['nA'], ch['nB']
    ko = np.zeros((na+qa.shape[1], nb+qb.shape[1]))
    ko[:na, :nb] = ch['J11']
    ko[:na, nb:] = lb
    ko[na:, :nb] = ra
    return dict(K_E=ke, K_O=ko, Q_A=qa, Q_B=qb, R_A=ra, L_B=lb,
                alpha=alpha, beta=beta, epsilon_A=ea, epsilon_B=eb)


def _svd_values(matrix, la, np):
    return (la.svdvals(matrix, check_finite=False)
            if min(matrix.shape) else np.empty(0))


def coefficient_spectrum(channels):
    """Entire spectrum descriptor with implicit zeros; numerical, no cutoff."""
    np, la, _ = _deps()
    core = small_cores(channels)
    se, so = _svd_values(core['K_E'], la, np), _svd_values(core['K_O'], la, np)
    values = np.sort(np.concatenate((se, so)))[::-1]
    candidates = values[values != 0.]
    ambient = 4**min(channels['cut'], channels['n_qubits']-channels['cut'])
    structural_cap = min(ambient, min(len(channels['a2'])+2,
        len(channels['b2'])+2)+channels['nA']+channels['nB'])
    descriptor = dict(candidate_nonzero_singular_values=candidates,
        candidate_nonzero_count=len(candidates),
        candidate_zero_multiplicity=ambient-len(candidates),
        ambient_singular_value_count=ambient,
        exact_structural_zero_lower_bound=ambient-structural_cap,
        sorted_core_singular_values=values, even_core_shape=core['K_E'].shape,
        odd_core_shape=core['K_O'].shape, no_exponential_zero_array=True,
        identity_removed=True, tolerance_threshold_used=False,
        arithmetic='float64 ordinary QR/SVD; candidates are not strict spectral intervals',
        zero_scope='Numerical exact-zero test only; candidate multiplicity is not certified rank')
    return descriptor, core


def factor_svd(channels):
    """Implicit factor frames for reconstruction and matrix-element queries."""
    np, la, _ = _deps()
    descriptor, core = coefficient_spectrum(channels)
    for block in ('E', 'O'):
        matrix = core['K_'+block]
        if min(matrix.shape):
            u, s, vh = la.svd(matrix, full_matrices=False, check_finite=False,
                            lapack_driver='gesdd')
        else:
            u, s, vh = np.empty((matrix.shape[0], 0)), np.empty(0), np.empty((0, matrix.shape[1]))
        core['U_'+block], core['s_'+block], core['Vh_'+block] = u, s, vh
    return descriptor, core


def active_original_matrix(source, cut):
    """Independent sparse reshape of actual Pauli labels, including union labels."""
    np, _, sparse = _deps()
    left = _unique_sorted(str(label[:cut]) for label in source['labels'])
    right = _unique_sorted(str(label[cut:]) for label in source['labels'])
    rows = [bisect_left(left, str(label[:cut])) for label in source['labels']]
    cols = [bisect_left(right, str(label[cut:])) for label in source['labels']]
    matrix = sparse.coo_matrix((source['coefficients'], (rows, cols)),
                              shape=(len(left), len(right))).tocsr()
    return matrix, left, right


def verify_structure(ch, core):
    """Check shared lifting including zero blocks, without a large coefficient grid."""
    np, la, _ = _deps()
    qa, qb = core['Q_A'], core['Q_B']
    errors = [float(la.norm(ch['J31']-qa@core['R_A'])),
              float(la.norm(ch['J13']-core['L_B']@qb.T)),
              float(la.norm(qa.T@qa-np.eye(qa.shape[1]))),
              float(la.norm(qb.T@qb-np.eye(qb.shape[1])))]
    if core['epsilon_A']:
        errors.append(float(la.norm(ch['a4']-
            core['alpha']*(ch['a4']/core['alpha']))))
    if core['epsilon_B']:
        errors.append(float(la.norm(ch['b4']-
            core['beta']*(ch['b4']/core['beta']))))
    if 'U_E' in core:
        for block in ('E', 'O'):
            rebuilt = (core['U_'+block]*core['s_'+block])@core['Vh_'+block]
            errors.append(float(la.norm(rebuilt-core['K_'+block])))
    na, nb = ch['nA'], ch['nB']
    odd_zero = float(la.norm(core['K_O'][na:, nb:]))
    pa, pb = len(ch['a2']), len(ch['b2'])
    even_zeros = [float(la.norm(core['K_E'][1:1+pa, 1+pb:])),
                  float(la.norm(core['K_E'][1+pa:, 1:1+pb])),
                  float(la.norm(core['K_E'][1+pa:, 1+pb:]))]
    source_fro2 = math.fsum(float(x)**2 for x in ch['source']['coefficients'])
    core_fro2 = math.fsum(float(x)**2 for x in core['K_E'].ravel()) + \
                math.fsum(float(x)**2 for x in core['K_O'].ravel())
    scale = max(1., math.sqrt(source_fro2))
    if max(errors+[odd_zero]+even_zeros, default=0.) > 2e-10*scale:
        raise AssertionError('Universal isometric reconstruction failed')
    if abs(core_fro2-source_fro2) > 2e-10*max(1., source_fro2):
        raise AssertionError('Independent sparse source/core second moment differs')
    return dict(lifting_residual_and_orthogonality_max=max(errors, default=0.),
        forbidden_zero_core_norm=max([odd_zero]+even_zeros),
        source_frobenius_squared=source_fro2, core_frobenius_squared=core_fro2,
        second_moment_absolute_error=abs(core_fro2-source_fro2),
        reconstruction_scope='Every nonzero channel and every forbidden block; float diagnostic',
        passed=True)


def _monomial_element(indices, q, bra, ket, twisted=False):
    phase = (1j)**(len(indices)*(len(indices)-1)//2)
    if twisted:
        phase *= 1j*((-1)**ket.bit_count())
    mask = ket
    for index in reversed(indices):
        p = index//2
        bit = (mask >> p) & 1
        phase *= (-1)**((mask & ((1 << p)-1)).bit_count())
        if index % 2:
            phase *= 1j if bit == 0 else -1j
        mask ^= 1 << p
    return phase if mask == bra else 0j


def factor_element(ch, core, block, index, side, bra, ket):
    """Matrix element of one normalized implicit factor; no Fock matrix."""
    np, _, _ = _deps()
    if block not in ('E', 'O') or side not in ('A', 'B'):
        raise ValueError('block must be E/O and side A/B')
    q = ch['cut'] if side == 'A' else ch['n_qubits']-ch['cut']
    if not (0 <= bra < 1 << q and 0 <= ket < 1 << q):
        raise ValueError('Occupation bit string exceeds subsystem')
    coords = core['U_'+block][:, index] if side == 'A' else core['Vh_'+block][index, :]
    vals = []
    if block == 'E':
        vals.append(complex(coords[0]) if bra == ket else 0j)
        count = math.comb(2*q, 2)
        for coefficient, indices in zip(coords[1:1+count], itertools.combinations(range(2*q), 2)):
            if coefficient != 0.:
                vals.append(coefficient*_monomial_element(indices, q, bra, ket))
        norm = core['alpha'] if side == 'A' else core['beta']
        if norm != 0:
            labels = ch['a4_labels'] if side == 'A' else ch['b4_labels']
            coefficients = ch['a4'] if side == 'A' else ch['b4']
            for indices, coefficient in zip(labels, coefficients):
                vals.append(coords[-1]*(coefficient/norm)*
                    _monomial_element(tuple(map(int, indices)), q, bra, ket))
    else:
        count = 2*q
        for i, coefficient in enumerate(coords[:count]):
            if coefficient != 0.:
                vals.append(coefficient*_monomial_element((i,), q, bra, ket, side == 'A'))
        frame = core['Q_A'] if side == 'A' else core['Q_B']
        cubic = frame@coords[count:]
        for coefficient, indices in zip(cubic, itertools.combinations(range(2*q), 3)):
            if coefficient != 0.:
                vals.append(coefficient*_monomial_element(indices, q, bra, ket, side == 'A'))
    return complex(math.fsum(float(complex(z).real) for z in vals),
                   math.fsum(float(complex(z).imag) for z in vals))


def hamiltonian_element(ch, core, bra, ket, include_identity=True):
    """Implicit full reconstruction query, used for small independent controls."""
    cut = ch['cut']
    mask = (1 << cut)-1
    vals = []
    for block in ('E', 'O'):
        for index, singular in enumerate(core['s_'+block]):
            if singular != 0.:
                vals.append(singular*
                    factor_element(ch, core, block, index, 'A', bra & mask, ket & mask)*
                    factor_element(ch, core, block, index, 'B', bra >> cut, ket >> cut))
    if include_identity and bra == ket:
        vals.append(ch['source']['identity'])
    return complex(math.fsum(float(complex(z).real) for z in vals),
                   math.fsum(float(complex(z).imag) for z in vals))


def representation_packet(ch, core):
    """Shared factor descriptor; quartic directions stored once per side."""
    np, _, _ = _deps()
    arrays = dict(format_version=np.asarray(1, dtype=np.uint8),
        dimensions=np.asarray([ch['n_qubits'], ch['cut']], dtype=np.uint64),
        identity=np.asarray(ch['source']['identity'], dtype=float),
        a4_labels=ch['a4_labels'], b4_labels=ch['b4_labels'],
        a4_direction=ch['a4']/core['alpha'] if core['alpha'] else ch['a4'],
        b4_direction=ch['b4']/core['beta'] if core['beta'] else ch['b4'],
        odd_Q_A=core['Q_A'], odd_Q_B=core['Q_B'])
    for block in ('E', 'O'):
        for prefix in ('U_', 's_', 'Vh_'):
            arrays[prefix+block] = core[prefix+block]
    return arrays


def benchmark(ch, dense, repetitions=5):
    """Fresh QR + values-only core SVD versus prepared same-C dense SVD."""
    np, la, _ = _deps()
    coefficient_spectrum(ch)
    baseline = _svd_values(dense, la, np)
    rows = []
    for repeat in range(repetitions):
        order = ('core', 'dense') if repeat % 2 == 0 else ('dense', 'core')
        elapsed = {}
        calculated = None
        for name in order:
            started = time.perf_counter()
            if name == 'core':
                calculated, _ = coefficient_spectrum(ch)
            else:
                baseline = _svd_values(dense, la, np)
            elapsed[name] = time.perf_counter()-started
        values = calculated['sorted_core_singular_values']
        length = max(len(values), len(baseline))
        error = float(la.norm(np.pad(values, (0, length-len(values)))-
                              np.pad(baseline, (0, length-len(baseline)))))
        if error > 2e-10*max(1., float(la.norm(baseline))):
            raise AssertionError('Paired full-spectrum targets differ')
        rows.append(dict(repeat=repeat, order='/'.join(order), core_seconds=elapsed['core'],
            dense_seconds=elapsed['dense'], dense_over_core=elapsed['dense']/elapsed['core'],
            complete_spectrum_l2_error=error, scope='Prepared original C; fresh QR and cores per query; float64 values only'))
    return rows

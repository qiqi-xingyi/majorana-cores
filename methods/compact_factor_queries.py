"""Compact queries for centered Hermitian one-body factors.

Input order: q diagonal coordinates, then real/imaginary coordinates for i<j.
Production functions never build the q**2-by-q-by-q operator dictionary or the
2**q physical matrix. Exact certificates interpret coefficient and dictionary
constant floats as exact dyadic rationals before multiplying them.
"""
from __future__ import annotations
import math
import time
from fractions import Fraction as Q
from numbers import Integral

import numpy as np
from scipy import linalg
from dyadic_spectral_certificate import float_to_dyadic, certify_hermitian


def _view(coefficients, q, offdiag_constant):
    if not isinstance(q, Integral) or isinstance(q, bool) or q < 1:
        raise ValueError("q must be a positive integer")
    if not isinstance(coefficients, np.ndarray):
        raise TypeError("Provide the already loaded ndarray; implicit O(q**2) conversion is disallowed")
    coefficients = np.asarray(coefficients)
    if coefficients.dtype != np.float64 or coefficients.shape != (q*q,):
        raise ValueError("A stored float64 coefficient vector of length q**2 is required")
    constant = float(offdiag_constant)
    if not math.isfinite(constant):
        raise ValueError("The actual stored dictionary constant must be finite")
    return coefficients, int(q), constant


def validate_coefficients(coefficients, q, offdiag_constant):
    """One-time full finite scan; outside an individual O(1)/O(q) element query."""
    coefficients, q, constant = _view(coefficients, q, offdiag_constant)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Nonfinite coefficient")
    return coefficients, q, constant


def pair_index(i, j, q):
    """First of the real/imaginary coordinates for i<j, in O(1) arithmetic."""
    if not (0 <= i < j < q):
        raise ValueError("Pair must satisfy 0 <= i < j < q")
    return q + 2 * (i*(2*q-i-1)//2 + j-i-1)


def one_body_matrix(coefficients, q, offdiag_constant):
    """Construct only the q-by-q Hermitian F of B=2*dGamma(F)-tr(F)*I."""
    coefficients, q, constant = validate_coefficients(coefficients, q, offdiag_constant)
    matrix = np.zeros((q, q), dtype=np.complex128)
    for i in range(q):
        matrix[i, i] = coefficients[i]
    index = q
    for i in range(q):
        for j in range(i+1, q):
            value = constant * complex(coefficients[index], coefficients[index+1])
            matrix[i, j], matrix[j, i] = value, value.conjugate()
            index += 2
    if index != q*q:
        raise AssertionError("Implicit dictionary coordinate count mismatch")
    return matrix


def norm_float(coefficients, q, offdiag_constant):
    """Floating full-Fock norm estimate from the q-by-q trace norm."""
    matrix = one_body_matrix(coefficients, q, offdiag_constant)
    return float(np.sum(np.abs(linalg.eigvalsh(matrix, check_finite=False))))


def element_exact(coefficients, q, offdiag_constant, bra, ket):
    """Exact Fractions of <bra|B|ket; q diagonal or two offdiagonal coefficient accesses.

    Inputs must already be stored float64. Shape checks do not scan q**2
    entries. Only accessed coefficients are converted to exact rational values.
    A separate validate_coefficients call checks the full packet once.
    Determinant bit-count and parity operations have their usual integer bit cost;
    O(1) here refers to coefficient accesses, not arbitrary-length integer work.
    """
    coefficients, q, constant = _view(coefficients, q, offdiag_constant)
    if (not isinstance(bra, Integral) or not isinstance(ket, Integral)
            or bra < 0 or ket < 0 or bra >= 1 << q or ket >= 1 << q):
        raise ValueError("Invalid Slater determinant")
    bra, ket = int(bra), int(ket)
    if bra.bit_count() != ket.bit_count():
        return Q(0), Q(0)
    if bra == ket:
        diagonal = sum(((1 if (ket >> i) & 1 else -1)
                        * Q.from_float(float(coefficients[i])) for i in range(q)), Q(0))
        return diagonal, Q(0)
    removed, added = ket & ~bra, bra & ~ket
    if removed.bit_count() != 1 or added.bit_count() != 1:
        return Q(0), Q(0)
    k, j = removed.bit_length()-1, added.bit_length()-1
    index = pair_index(min(j, k), max(j, k), q)
    middle = ket ^ (1 << k)
    sign = (-1) ** (((ket & ((1 << k)-1)).bit_count())
                    + ((middle & ((1 << j)-1)).bit_count()))
    scale = 2 * sign * Q.from_float(constant)
    real = scale * Q.from_float(float(coefficients[index]))
    imag = scale * Q.from_float(float(coefficients[index+1])) * (1 if j < k else -1)
    return real, imag


def exact_one_body(coefficients, q, offdiag_constant):
    """Exact q-by-q integer F matrices and a shared dyadic exponent."""
    coefficients, q, constant = validate_coefficients(coefficients, q, offdiag_constant)
    real_coeff, imag_coeff, cp = float_to_dyadic(coefficients)
    if np.any(imag_coeff):
        raise AssertionError("Real coefficient vector required")
    cn, cd = constant.as_integer_ratio()
    ce = cd.bit_length()-1
    if cd != 1 << ce:
        raise AssertionError("Dictionary constant is not dyadic")
    real = np.zeros((q, q), dtype=object)
    imag = np.zeros((q, q), dtype=object)
    for i in range(q):
        real[i, i] = int(real_coeff[i]) * (1 << ce)
    index = q
    for i in range(q):
        for j in range(i+1, q):
            x, y = int(real_coeff[index])*cn, int(real_coeff[index+1])*cn
            real[i, j] = real[j, i] = x
            imag[i, j], imag[j, i] = y, -y
            index += 2
    return real, imag, cp+ce


def certified_norm(coefficients, q, offdiag_constant):
    """Strict full-Fock B norm interval from exact stored-dyadic q-by-q F."""
    started = time.perf_counter()
    real, imag, exponent = exact_one_body(coefficients, q, offdiag_constant)
    certificate = certify_hermitian(real, imag, exponent, target=Q(10**6))
    if not certificate["polar_certificate_available"]:
        raise ArithmeticError("No strict trace-norm interval available")
    certificate.update(
        full_factor_norm_lower_numerator=certificate["trace_norm_lower_numerator"],
        full_factor_norm_lower_denominator=certificate["trace_norm_lower_denominator"],
        full_factor_norm_upper_numerator=certificate["trace_norm_upper_numerator"],
        full_factor_norm_upper_denominator=certificate["trace_norm_upper_denominator"],
        full_factor_norm_identity="||2*dGamma(F)-tr(F)*I||_fullFock = ||F||_trace",
        coefficient_entries=int(q*q), largest_constructed_matrix_entries=int(q*q),
        implicit_dictionary=True, full_dictionary_materialized=False,
        full_physical_matrix_materialized=False,
        query_certificate_seconds=float(time.perf_counter()-started),
        scope="Exact actual stored coefficient vector and actual offdiagonal dictionary constant; no q**4 dictionary or 2**q physical factor")
    return certificate

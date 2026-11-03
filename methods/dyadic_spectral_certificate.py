"""Exact dyadic a posteriori upper certificates for small Hermitian matrices.

All floating eigensolver output is a candidate only.  For E and the returned
binary floating Q and Lambda, integer arithmetic forms Q^* Q - I and
E - Q Lambda Q^* exactly.  Thus the certificate does not inherit a floating
matrix-product residual.

If beta >= ||Q^*Q-I||_2 and rho >= ||E-Q Lambda Q^*||_2, then

    ||E||_2 <= rho + (1+beta) max_j |Lambda_j|.

This follows from the triangle inequality and
||Q||_2^2 = ||Q^*Q||_2 <= 1+beta; Q need not be invertible.
This is an upper certificate for the supplied exact dyadic matrix, not for
unrecorded upstream integral or projection errors.
"""
from __future__ import annotations

from fractions import Fraction
from numbers import Integral
import math
import time



# Full N=2 validation on 20 retained modes has binom(20,2)=190 states.
# This is an explicit resource guard, not a restriction in the certificate.
MAX_DIMENSION = 190


def _imports():
    import numpy as np
    return np


def _integer_array(values, name):
    np = _imports()
    array = np.asarray(values, dtype=object)
    output = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        value = array[index]
        if not isinstance(value, Integral):
            raise TypeError(name + " must contain integers")
        output[index] = int(value)
    return output


def _integer_exponent(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(name + " must be an integer")
    return int(value)


def _dyadic_fraction(numerator, exponent):
    numerator = int(numerator)
    if exponent >= 0:
        return Fraction(numerator, 1 << exponent)
    return Fraction(numerator * (1 << (-exponent)), 1)


def _fraction_record(value):
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        "float": _finite_float(value),
    }


def _finite_float(value):
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _validate_matrix(real_int, imag_int, exponent):
    np = _imports()
    real_int = _integer_array(real_int, "real_int")
    imag_int = _integer_array(imag_int, "imag_int")
    exponent = _integer_exponent(exponent, "p")
    if real_int.ndim != 2 or real_int.shape[0] != real_int.shape[1]:
        raise ValueError("E must be a square matrix")
    if real_int.shape != imag_int.shape:
        raise ValueError("real_int and imag_int must have the same shape")
    dimension = real_int.shape[0]
    if not 1 <= dimension <= MAX_DIMENSION:
        raise ValueError("matrix dimension must be between 1 and " + str(MAX_DIMENSION))
    if not np.array_equal(real_int, real_int.T):
        raise ValueError("E is not exactly Hermitian: real part is not symmetric")
    if not np.array_equal(imag_int, -imag_int.T):
        raise ValueError("E is not exactly Hermitian: imaginary part is not skew-symmetric")
    return real_int, imag_int, exponent


def float_to_dyadic(arr):
    """Return exact integer real/imaginary arrays and a common 2-adic exponent.

    Finite real or complex IEEE floating values of at most binary64 precision
    are accepted.  Conversion of binary32 values to Python float is exact.
    Extended precision is rejected rather than silently rounded.  The result
    represents the values actually supplied, not their decimal spellings.
    """
    np = _imports()
    array = np.asarray(arr)
    if array.dtype.kind not in ("f", "c"):
        raise TypeError("arr must have a real or complex floating dtype")
    component_size = array.dtype.itemsize // (2 if array.dtype.kind == "c" else 1)
    if component_size > 8:
        raise TypeError("extended precision floats require an explicit exact conversion")
    pairs = []
    common_exponent = 0
    for index in np.ndindex(array.shape):
        value = array[index]
        real = float(value.real)
        imag = float(value.imag) if array.dtype.kind == "c" else 0.0
        if not math.isfinite(real) or not math.isfinite(imag):
            raise ValueError("arr must contain finite floating values")
        real_num, real_den = real.as_integer_ratio()
        imag_num, imag_den = imag.as_integer_ratio()
        real_p = real_den.bit_length() - 1
        imag_p = imag_den.bit_length() - 1
        # Python's finite binary float denominators are powers of two.
        if real_den != 1 << real_p or imag_den != 1 << imag_p:
            raise AssertionError("a binary float had a non-dyadic denominator")
        common_exponent = max(common_exponent, real_p, imag_p)
        pairs.append((index, real_num, real_p, imag_num, imag_p))
    real_int = np.empty(array.shape, dtype=object)
    imag_int = np.empty(array.shape, dtype=object)
    for index, real_num, real_p, imag_num, imag_p in pairs:
        real_int[index] = int(real_num) << (common_exponent - real_p)
        imag_int[index] = int(imag_num) << (common_exponent - imag_p)
    return real_int, imag_int, common_exponent


def _row_bound(real_int, imag_int, exponent):
    """Strict spectral upper bound for an exactly Hermitian dyadic matrix."""
    np = _imports()
    if not np.array_equal(real_int, real_int.T):
        raise AssertionError("the certified real residual lost Hermiticity")
    if not np.array_equal(imag_int, -imag_int.T):
        raise AssertionError("the certified imaginary residual lost Hermiticity")
    largest = 0
    for row in range(real_int.shape[0]):
        total = sum(abs(int(real_int[row, col])) + abs(int(imag_int[row, col]))
                    for col in range(real_int.shape[1]))
        largest = max(largest, total)
    # For Hermitian A, ||A||_2 <= sqrt(||A||_1 ||A||_inf) = ||A||_inf.
    # Each |Re(A_ij)|+|Im(A_ij)| dominates the complex entry modulus.
    return _dyadic_fraction(largest, exponent)


def _candidate_certificate(real_int, imag_int, p, q, eigenvalues,
                           eigenvalue_scale_exponent=0):
    """Certify supplied binary-float candidates by exact integer arithmetic.

    This internal entry is deliberately available to adversarial validation
    tests.  eigenvalues * 2**eigenvalue_scale_exponent is interpreted exactly.
    No numerical orthogonality or eigendecomposition accuracy is assumed.
    """
    np = _imports()
    real_int, imag_int, p = _validate_matrix(real_int, imag_int, p)
    eigenvalue_scale_exponent = _integer_exponent(
        eigenvalue_scale_exponent, "eigenvalue_scale_exponent")
    q = np.asarray(q)
    eigenvalues = np.asarray(eigenvalues)
    dimension = real_int.shape[0]
    if q.shape != (dimension, dimension) or eigenvalues.shape != (dimension,):
        raise ValueError("Q and eigenvalues have incompatible dimensions")
    if eigenvalues.dtype.kind != "f":
        raise TypeError("candidate eigenvalues must be real floating values")
    qr, qi, qp = float_to_dyadic(q)
    lr, li, lp = float_to_dyadic(eigenvalues)
    if any(int(value) for value in li.flat):
        raise AssertionError("candidate eigenvalues must be real")
    lp -= eigenvalue_scale_exponent

    gram_real = qr.T @ qr + qi.T @ qi
    gram_imag = qr.T @ qi - qi.T @ qr
    gram_p = 2 * qp
    identity_scale = 1 << gram_p
    for index in range(dimension):
        gram_real[index, index] -= identity_scale
    beta = _row_bound(gram_real, gram_imag, gram_p)

    # Multiplication by lr scales columns; every operation below is on Python
    # integers in object arrays.  Form the exact product of the stored floats.
    weighted_real = qr * lr[None, :]
    weighted_imag = qi * lr[None, :]
    product_real = weighted_real @ qr.T + weighted_imag @ qi.T
    product_imag = weighted_imag @ qr.T - weighted_real @ qi.T
    product_p = 2 * qp + lp
    residual_p = max(p, product_p)
    input_scale = 1 << (residual_p - p)
    product_scale = 1 << (residual_p - product_p)
    residual_real = real_int * input_scale - product_real * product_scale
    residual_imag = imag_int * input_scale - product_imag * product_scale
    rho = _row_bound(residual_real, residual_imag, residual_p)
    largest_lambda = _dyadic_fraction(
        max(abs(int(value)) for value in lr.flat), lp)
    bound = rho + (1 + beta) * largest_lambda
    return bound, beta, rho, largest_lambda


def _scaled_float_candidate(real_int, imag_int, p):
    np = _imports()
    largest_bits = max(
        max(abs(int(value)).bit_length() for value in real_int.flat),
        max(abs(int(value)).bit_length() for value in imag_int.flat),
    )
    if largest_bits == 0:
        return np.zeros(real_int.shape, dtype=np.complex128), 0
    # Scale E by 2**(-scale), so every component of E_scaled has modulus < 1.
    # Fraction.__float__ avoids overflow from converting a huge Python int
    # before the exact power-of-two division.
    scale = largest_bits - p
    denominator = 1 << largest_bits
    scaled = np.empty(real_int.shape, dtype=np.complex128)
    for index in np.ndindex(real_int.shape):
        scaled[index] = complex(
            float(Fraction(int(real_int[index]), denominator)),
            float(Fraction(int(imag_int[index]), denominator)),
        )
    return scaled, scale


def certify_hermitian(real_int, imag_int, p, target=Fraction(1, 625)):
    """Return a strict, JSON-compatible spectral upper certificate.

    E = (real_int + 1j * imag_int) / 2**p is interpreted exactly, including for
    negative p.  target must be a nonnegative Fraction (or integer), so a decimal
    tolerance is never silently replaced by its binary floating approximation.
    The decision compares exact rational numbers.  Float fields are diagnostic
    only, can underflow, and are None on overflow.
    """
    started = time.perf_counter()
    real_int, imag_int, p = _validate_matrix(real_int, imag_int, p)
    if isinstance(target, Integral):
        target = Fraction(int(target), 1)
    if not isinstance(target, Fraction):
        raise TypeError("target must be an exact Fraction or integer")
    if target < 0:
        raise ValueError("target must be nonnegative")
    import scipy.linalg

    scaled, scale = _scaled_float_candidate(real_int, imag_int, p)
    eigenvalues, q = scipy.linalg.eigh(
        scaled, check_finite=False, overwrite_a=False, driver="evr")
    bound, beta, rho, largest_lambda = _candidate_certificate(
        real_int, imag_int, p, q, eigenvalues, scale)
    lambda_real, _, lambda_p = float_to_dyadic(eigenvalues)
    sum_abs_lambda = _dyadic_fraction(
        sum(abs(int(value)) for value in lambda_real.flat), lambda_p - scale)
    polar_available = beta < 1
    if polar_available:
        # Q = U S with U unitary because beta < 1 implies Q is invertible.
        # ||S-I|| <= beta and ||S|| <= 1+beta imply
        # ||S Lambda S - Lambda|| <= max|Lambda| * beta * (2+beta).
        # Weyl then bounds every sorted eigenvalue error by eta.
        eta = rho + largest_lambda * beta * (2 + beta)
        trace_lower = max(Fraction(0), sum_abs_lambda - real_int.shape[0] * eta)
        trace_upper = sum_abs_lambda + real_int.shape[0] * eta
    else:
        eta = trace_lower = trace_upper = None
    float_diagnostic = {
        "normalized_eigenvalue_abs_max": float(max(abs(value) for value in eigenvalues)),
        "candidate_eigenvalue_abs_max": _finite_float(largest_lambda),
        "bound": _finite_float(bound),
        "candidate_scale_exponent": int(scale),
        "used_for_target_decision": False,
    }
    return {
        "schema": "exact_dyadic_hermitian_spectral_upper",
        "dimension": int(real_int.shape[0]),
        "input_dyadic_exponent": int(p),
        "bound_numerator": str(bound.numerator),
        "bound_denominator": str(bound.denominator),
        "target_numerator": str(target.numerator),
        "target_denominator": str(target.denominator),
        "target_pass": bool(bound <= target),
        "beta": _fraction_record(beta),
        "rho": _fraction_record(rho),
        "eigenvalue_abs_max": _fraction_record(largest_lambda),
        "max_abs_lambda_numerator": str(largest_lambda.numerator),
        "max_abs_lambda_denominator": str(largest_lambda.denominator),
        "sum_abs_lambda_numerator": str(sum_abs_lambda.numerator),
        "sum_abs_lambda_denominator": str(sum_abs_lambda.denominator),
        "polar_certificate_available": bool(polar_available),
        "eta_numerator": str(eta.numerator) if eta is not None else None,
        "eta_denominator": str(eta.denominator) if eta is not None else None,
        "trace_norm_lower_numerator": str(trace_lower.numerator) if trace_lower is not None else None,
        "trace_norm_lower_denominator": str(trace_lower.denominator) if trace_lower is not None else None,
        "trace_norm_upper_numerator": str(trace_upper.numerator) if trace_upper is not None else None,
        "trace_norm_upper_denominator": str(trace_upper.denominator) if trace_upper is not None else None,
        "polar_scope": "eta bounds all sorted eigenvalue errors; trace norm interval uses n times eta",
        "float_diagnostic": float_diagnostic,
        "norm_bound": "exact Hermitian real/imaginary absolute row sum",
        "elapsed_seconds": float(time.perf_counter() - started),
        "scope": "supplied exact dyadic matrix only",
    }

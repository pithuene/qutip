# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
from libc.stdint cimport int64_t

cdef extern from "<complex>" namespace "std" nogil:
    double complex exp(double complex x)
    double complex complex_conjugate "conj"(double complex x)

np_fact = np.zeros(21, dtype=float)
cdef double[:] inv_factorial = np_fact
cdef Py_ssize_t factorial_index
inv_factorial[0] = 1
for factorial_index in range(1, 21):
    inv_factorial[factorial_index] = (
        inv_factorial[factorial_index - 1] / factorial_index
    )


cdef inline double complex _integer_power(
    double complex value,
    int64_t exponent,
) noexcept nogil:
    """Return a non-negative integer power without a generic complex pow."""
    cdef double complex result = 1.0
    while exponent > 0:
        result *= value
        exponent -= 1
    return result


cdef inline double complex _control_monomial_derivative(
    const double complex[:, :] rotated_amplitudes,
    const int64_t[:, :] positive_counts,
    const int64_t[:, :] negative_counts,
    Py_ssize_t step,
    Py_ssize_t monomial,
    Py_ssize_t target_drive,
    bint conjugate_derivative,
) noexcept nogil:
    """Differentiate one monomial without dividing by its amplitude."""
    cdef Py_ssize_t drive
    cdef int64_t positive_count, negative_count
    cdef double complex amplitude, value = 1.0
    for drive in range(rotated_amplitudes.shape[0]):
        amplitude = rotated_amplitudes[drive, step]
        positive_count = positive_counts[monomial, drive]
        negative_count = negative_counts[monomial, drive]
        if drive == target_drive and conjugate_derivative:
            if positive_count == 0:
                return 0.0
            value *= <double>positive_count * _integer_power(
                complex_conjugate(amplitude), positive_count - 1
            )
            value *= _integer_power(amplitude, negative_count)
        elif drive == target_drive:
            if negative_count == 0:
                return 0.0
            value *= _integer_power(
                complex_conjugate(amplitude), positive_count
            )
            value *= <double>negative_count * _integer_power(
                amplitude, negative_count - 1
            )
        else:
            value *= _integer_power(
                complex_conjugate(amplitude), positive_count
            )
            value *= _integer_power(amplitude, negative_count)
    return value


cdef void _matrix_multiply(
    const double complex[:, :] left,
    const double complex[:, :] right,
    double complex[:, :] output,
) noexcept nogil:
    """Multiply two small dense matrices."""
    cdef Py_ssize_t row, column, inner
    cdef double complex value
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            value = 0.0
            for inner in range(left.shape[1]):
                value += left[row, inner] * right[inner, column]
            output[row, column] = value


cpdef complex cy_compute_integrals(double[:] ws, double dt, double a_tol=1e-10):
    """
        Computes the value of the nested integrals for a given array of
        effective omegas. See eq. (7) in Ref.

        Parameters
        ----------
        ws : double[:]
            An array of effective omegas. ws[0] is the omega for the rightmost
            integral.

        dt : double
            The time increment.

        a_tol : double, default = 1e-10
            The absolute tolerance used.

        Returns
        -------
        value : complex
            The value of the nested integrals.

        Notes
        -----
        Integrals are done analytically from right to left with integration
        by parts.

    """
    cdef double[:] ws_prime
    if len(ws) == 1:
        if abs(ws[0]) < a_tol:
            return dt
        else:
            return (-1.j / ws[0]) * (exp(1j * ws[0] * dt) - 1.)
    else:
        if abs(ws[0]) < a_tol:
            return cy_compute_tn_integrals(ws[1:], 1, dt)
        else:
            ws_prime = ws[1:].copy()
            ws_prime[0] += ws[0]
            return (-1j / ws[0]) * (
                cy_compute_integrals(ws_prime, dt)
                - cy_compute_integrals(ws[1:], dt)
            )


cpdef object cy_compute_Sn(
    double[:, :] omega_vectors,
    long[:, :] ket_bra_idx,
    double[:, :] diff_lambdas,
    double complex[:] matrix_elements,
    double dt,
    int length,
    double a_tol=1e-10,
):
    """
        Accumulate one Dyson-order tensor before order prefactors.

        This moves the Python loop over frequency vectors and nonzero
        matrix-element paths into Cython.  Matrix multiplication by exp(H0 dt)
        is intentionally left to NumPy in the Python caller.
    """
    cdef Py_ssize_t n_omega = omega_vectors.shape[0]
    cdef Py_ssize_t n_paths = diff_lambdas.shape[0]
    cdef Py_ssize_t order = omega_vectors.shape[1]
    cdef Py_ssize_t i, j, k
    cdef long row, col
    cdef object Sn_arr = np.zeros(
        (n_omega, length, length), dtype=np.complex128
    )
    cdef double complex[:, :, :] Sn = Sn_arr
    cdef object ws_arr = np.empty(order, dtype=float)
    cdef double[:] ws = ws_arr

    for i in range(n_omega):
        for j in range(n_paths):
            for k in range(order):
                ws[k] = omega_vectors[i, k] + diff_lambdas[j, k]
            row = ket_bra_idx[j, 0]
            col = ket_bra_idx[j, 1]
            Sn[i, row, col] += (
                cy_compute_integrals(ws, dt, a_tol) * matrix_elements[j]
            )

    return Sn_arr


cpdef tuple cy_control_polynomial_vjp(
    const double complex[:, :] rotated_amplitudes,
    const double complex[:, :] rotations,
    const int64_t[:, :] positive_counts,
    const int64_t[:, :] negative_counts,
    const double complex[:, :, :] coefficients,
    const double complex[:, :] free_propagator,
    const double complex[:, :] cotangent,
    const double complex[:, :] boundary_prefix,
    const double complex[:, :] running_suffix,
):
    """Evaluate one polynomial batch and run its exact chronological VJP.

    This fuses polynomial evaluation, prefix and suffix scans, and contraction
    with analytic monomial derivatives. It preserves every collected Dyson
    coefficient and does not divide by amplitudes, including at zero.
    """
    cdef Py_ssize_t drive_count = rotated_amplitudes.shape[0]
    cdef Py_ssize_t step_count = rotated_amplitudes.shape[1]
    cdef Py_ssize_t monomial_count = coefficients.shape[0]
    cdef Py_ssize_t dimension = free_propagator.shape[0]
    if rotations.shape[0] != drive_count or rotations.shape[1] != step_count:
        raise ValueError("rotations must match rotated_amplitudes")
    if (
        positive_counts.shape[0] != monomial_count
        or negative_counts.shape[0] != monomial_count
        or positive_counts.shape[1] != drive_count
        or negative_counts.shape[1] != drive_count
    ):
        raise ValueError("control counts must match amplitudes and coefficients")
    if (
        free_propagator.shape[1] != dimension
        or coefficients.shape[1] != dimension
        or coefficients.shape[2] != dimension
        or cotangent.shape[0] != dimension
        or cotangent.shape[1] != dimension
        or boundary_prefix.shape[0] != dimension
        or boundary_prefix.shape[1] != dimension
        or running_suffix.shape[0] != dimension
        or running_suffix.shape[1] != dimension
    ):
        raise ValueError("VJP matrices must have equal square dimensions")

    cdef object subpropagators_array = np.empty(
        (step_count, dimension, dimension), dtype=np.complex128
    )
    cdef object prefixes_array = np.empty(
        (step_count + 1, dimension, dimension), dtype=np.complex128
    )
    cdef object suffix_array = np.array(
        running_suffix, dtype=np.complex128, order="C", copy=True
    )
    cdef object next_suffix_array = np.empty_like(suffix_array)
    cdef object temporary_array = np.empty_like(suffix_array)
    cdef object local_cotangent_array = np.empty_like(suffix_array)
    cdef object effective_cotangent_array = np.empty(
        (drive_count, step_count), dtype=np.complex128
    )
    cdef double complex[:, :, :] subpropagators = subpropagators_array
    cdef double complex[:, :, :] prefixes = prefixes_array
    cdef double complex[:, :] suffix = suffix_array
    cdef double complex[:, :] next_suffix = next_suffix_array
    cdef double complex[:, :] temporary = temporary_array
    cdef double complex[:, :] local_cotangent = local_cotangent_array
    cdef double complex[:, :] effective_cotangent = (
        effective_cotangent_array
    )
    cdef Py_ssize_t step, monomial, drive, row, column, inner
    cdef double complex value, sensitivity, conjugate_derivative, plain_derivative
    cdef double complex derivative_x, derivative_y, rotation
    cdef double gradient_x, gradient_y

    with nogil:
        # Materialize this bounded batch and its boundary-aware prefixes.
        for step in range(step_count):
            for row in range(dimension):
                for column in range(dimension):
                    subpropagators[step, row, column] = free_propagator[
                        row, column
                    ]
            for monomial in range(monomial_count):
                value = 1.0
                for drive in range(drive_count):
                    value *= _integer_power(
                        complex_conjugate(
                            rotated_amplitudes[drive, step]
                        ),
                        positive_counts[monomial, drive],
                    )
                    value *= _integer_power(
                        rotated_amplitudes[drive, step],
                        negative_counts[monomial, drive],
                    )
                for row in range(dimension):
                    for column in range(dimension):
                        subpropagators[step, row, column] += (
                            value * coefficients[monomial, row, column]
                        )
        for row in range(dimension):
            for column in range(dimension):
                prefixes[0, row, column] = boundary_prefix[row, column]
        for step in range(step_count):
            _matrix_multiply(
                subpropagators[step], prefixes[step], prefixes[step + 1]
            )

        # Pull the propagator cotangent through the chronological product.
        for step in range(step_count - 1, -1, -1):
            for row in range(dimension):
                for column in range(dimension):
                    value = 0.0
                    for inner in range(dimension):
                        value += (
                            complex_conjugate(suffix[inner, row])
                            * cotangent[inner, column]
                        )
                    temporary[row, column] = value
            for row in range(dimension):
                for column in range(dimension):
                    value = 0.0
                    for inner in range(dimension):
                        value += (
                            temporary[row, inner]
                            * complex_conjugate(
                                prefixes[step, column, inner]
                            )
                        )
                    local_cotangent[row, column] = value
            for drive in range(drive_count):
                effective_cotangent[drive, step] = 0.0
            for monomial in range(monomial_count):
                sensitivity = 0.0
                for row in range(dimension):
                    for column in range(dimension):
                        sensitivity += (
                            complex_conjugate(local_cotangent[row, column])
                            * coefficients[monomial, row, column]
                        )
                for drive in range(drive_count):
                    conjugate_derivative = _control_monomial_derivative(
                        rotated_amplitudes,
                        positive_counts,
                        negative_counts,
                        step,
                        monomial,
                        drive,
                        True,
                    )
                    plain_derivative = _control_monomial_derivative(
                        rotated_amplitudes,
                        positive_counts,
                        negative_counts,
                        step,
                        monomial,
                        drive,
                        False,
                    )
                    rotation = rotations[drive, step]
                    derivative_x = (
                        conjugate_derivative
                        * complex_conjugate(rotation)
                        + plain_derivative * rotation
                    )
                    derivative_y = 1j * (
                        plain_derivative * rotation
                        - conjugate_derivative
                        * complex_conjugate(rotation)
                    )
                    gradient_x = (sensitivity * derivative_x).real
                    gradient_y = (sensitivity * derivative_y).real
                    effective_cotangent[drive, step] += (
                        gradient_x + 1j * gradient_y
                    )
            _matrix_multiply(suffix, subpropagators[step], next_suffix)
            for row in range(dimension):
                for column in range(dimension):
                    suffix[row, column] = next_suffix[row, column]

    return effective_cotangent_array, suffix_array


cdef complex cy_compute_tn_integrals(
    double[:] ws,
    int n,
    double dt,
    double a_tol=1e-10,
):
    """
        Helper function to compute nested integrals when the function to
        integrate is t^n/factorial(n) * exp(1j*omega*t). This happens when
        some effective omegas are 0. In that case, the recursion differs a
        bit from _compute_integrals(). See eq. (7) in Ref.

        Paramaters
        ----------
        ws : double[:]
            An array of effective omegas. ws[0] is the omega for the rightmost
            integral.

        n : int
            The variable in t^n/factorial(n).

        dt : double
            The time increment.

        a_tol : double, default = 1e-10
            The absolute tolerance used.

        Returns
        -------
        value : complex
            The value of the nested integrals when the function to integrate is
            t^n/factorial(n) * exp(1j*omega*t).

        Notes
        -----
        Integrals are done analytically from right to left with integration
        by parts.

    """
    cdef complex factor, term1, term2
    cdef double[:] ws_prime
    cdef int j

    if n == 0:
        return cy_compute_integrals(ws, dt)

    if n == 20:
        # Max supported n, order of 1e-18
        return 0.

    if len(ws) == 1:
        if abs(ws[0]) < a_tol:
            return (dt ** (n + 1)) * inv_factorial[n + 1]
        else:
            factor = (-1j/ws[0]) * exp(1j*ws[0]*dt)
            term1 = 0
            for j in range(n+1):
                term1 += ((1j/ws[0])**j) * (dt**(n-j) * inv_factorial[n-j])
            term2 = (1j / ws[0])**(n+1)
            return factor * term1 + term2
    else:
        if abs(ws[0]) < a_tol:
            return cy_compute_tn_integrals(ws[1:], n + 1, dt)
        else:
            factor = -1j / ws[0]
            ws_prime = ws[1:].copy()
            ws_prime[0] += ws[0]
            term1 = cy_compute_tn_integrals(ws_prime, n, dt)
            term2 = cy_compute_tn_integrals(ws, n - 1, dt)
            return factor * (term1 - term2)

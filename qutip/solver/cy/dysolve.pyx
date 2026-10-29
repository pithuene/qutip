# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
from libc.stdint cimport int64_t

cdef extern from "<complex>" namespace "std" nogil:
    double complex exp(double complex x)
    double complex complex_conjugate "conj"(double complex x)

# Largest supported Dyson order. One order adds one node to the divided difference.
cdef enum:
    MAX_INTEGRAL_ORDER = 20
    MAX_TAYLOR_TERMS = 40

np_fact = np.zeros(MAX_INTEGRAL_ORDER + MAX_TAYLOR_TERMS + 2, dtype=float)
cdef double[:] inv_factorial = np_fact
cdef Py_ssize_t factorial_index
inv_factorial[0] = 1
for factorial_index in range(1, np_fact.shape[0]):
    inv_factorial[factorial_index] = (
        inv_factorial[factorial_index - 1] / factorial_index
    )

# Nodes closer than this (in units of phase) are evaluated by a Taylor series
# instead of the divided-difference recursion, which would divide by their
# separation and amplify rounding errors.
cdef double CLUSTER_PHASE_SPREAD = 1.0
cdef double TAYLOR_RELATIVE_TOLERANCE = 1e-17


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


cdef double complex _clustered_exponential_divided_difference(
    const double* phases,
    Py_ssize_t node_count,
) noexcept nogil:
    """Divided difference of ``exp(1j * x)`` at nearby real phases.

    Around the cluster midpoint ``c``, ``exp[i x_0, ..., i x_k]`` equals
    ``exp(i c) * sum_m h_m(i (x - c)) / (m + k)!``, where ``h_m`` is the
    complete homogeneous symmetric polynomial. No term divides by a node
    separation, so coincident and nearly coincident nodes are exact.
    """
    cdef Py_ssize_t k = node_count - 1
    cdef double center = 0.5 * (phases[0] + phases[k])
    cdef double radius = 0.5 * (phases[k] - phases[0])
    cdef double complex homogeneous[MAX_TAYLOR_TERMS + 1]
    cdef double complex shifted, total
    cdef double term_bound = 1.0
    cdef Py_ssize_t term_count = 0
    cdef Py_ssize_t m, node

    # |h_m| / (m + k)! <= radius**m / (m! k!), so stop once that bound is negligible.
    while term_count < MAX_TAYLOR_TERMS and term_bound > TAYLOR_RELATIVE_TOLERANCE:
        term_count += 1
        term_bound *= radius / term_count
    homogeneous[0] = 1.0
    for m in range(1, term_count + 1):
        homogeneous[m] = 0.0
    for node in range(node_count):
        shifted = 1j * (phases[node] - center)
        for m in range(1, term_count + 1):
            homogeneous[m] += shifted * homogeneous[m - 1]
    total = 0.0
    for m in range(term_count, -1, -1):
        total += homogeneous[m] * inv_factorial[m + k]
    return exp(1j * center) * total


cdef double complex _nested_exponential_integral(
    const double* ws,
    Py_ssize_t order,
    double dt,
) noexcept nogil:
    """Time-ordered integral of ``exp(1j * sum_k ws[k] * t_k)`` over ``[0, dt]``.

    ``ws[0]`` belongs to the rightmost (innermost) integral. The integral equals
    ``dt**order`` times the divided difference of ``exp(1j * x)`` at the phases
    ``0, ws[-1] dt, (ws[-1] + ws[-2]) dt, ..., sum(ws) dt``. The table is built
    from sorted phases, so every recursion step divides by the spread of its
    nodes. Spreads below ``CLUSTER_PHASE_SPREAD`` use a Taylor series instead.
    This stays accurate when a branch frequency times ``dt`` is tiny, which
    happens for near-resonant low-frequency drives.
    """
    cdef double phases[MAX_INTEGRAL_ORDER + 1]
    cdef double complex table[MAX_INTEGRAL_ORDER + 1]
    cdef double phase
    cdef Py_ssize_t node, level, i, j

    phases[0] = 0.0
    for node in range(1, order + 1):
        phases[node] = phases[node - 1] + ws[order - node] * dt
    # Insertion sort, the node count is at most MAX_INTEGRAL_ORDER + 1.
    for i in range(1, order + 1):
        phase = phases[i]
        j = i - 1
        while j >= 0 and phases[j] > phase:
            phases[j + 1] = phases[j]
            j -= 1
        phases[j + 1] = phase

    if phases[order] - phases[0] < CLUSTER_PHASE_SPREAD:
        return dt ** order * _clustered_exponential_divided_difference(
            phases, order + 1
        )
    for node in range(order + 1):
        table[node] = exp(1j * phases[node])
    # After level ``level``, table[i] holds exp[i x_i, ..., i x_{i + level}].
    for level in range(1, order + 1):
        for i in range(order + 1 - level):
            if phases[i + level] - phases[i] < CLUSTER_PHASE_SPREAD:
                table[i] = _clustered_exponential_divided_difference(
                    &phases[i], level + 1
                )
            else:
                table[i] = (table[i + 1] - table[i]) / (
                    1j * (phases[i + level] - phases[i])
                )
    return dt ** order * table[0]


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
            Unused. Zero and nearly zero omegas need no special treatment.

        Returns
        -------
        value : complex
            The value of the nested integrals.
    """
    cdef double contiguous_ws[MAX_INTEGRAL_ORDER]
    cdef Py_ssize_t k
    if not 1 <= ws.shape[0] <= MAX_INTEGRAL_ORDER:
        raise ValueError(
            f"Dyson order must be between 1 and {MAX_INTEGRAL_ORDER}, "
            f"got {ws.shape[0]}"
        )
    for k in range(ws.shape[0]):
        contiguous_ws[k] = ws[k]
    return _nested_exponential_integral(contiguous_ws, ws.shape[0], dt)


cpdef object cy_compute_Sn(
    double[:, :] omega_vectors,
    long[:, :] ket_bra_idx,
    double[:, :] diff_lambdas,
    double complex[:] matrix_elements,
    double dt,
    int length,
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
    cdef double ws[MAX_INTEGRAL_ORDER]
    if not 1 <= order <= MAX_INTEGRAL_ORDER:
        raise ValueError(
            f"Dyson order must be between 1 and {MAX_INTEGRAL_ORDER}, "
            f"got {order}"
        )

    for i in range(n_omega):
        for j in range(n_paths):
            for k in range(order):
                ws[k] = omega_vectors[i, k] + diff_lambdas[j, k]
            row = ket_bra_idx[j, 0]
            col = ket_bra_idx[j, 1]
            Sn[i, row, col] += (
                _nested_exponential_integral(ws, order, dt) * matrix_elements[j]
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

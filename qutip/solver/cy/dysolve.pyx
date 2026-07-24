#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport cython

cdef extern from "<complex>" namespace "std" nogil:
    double complex exp(double complex x)

np_fact = np.zeros(21, dtype=float)
cdef double[:] inv_factorial = np_fact
inv_factorial[0] = 1
for i in range(1, 21):
    inv_factorial[i] = inv_factorial[i-1] / i


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


cdef void _compute_integral_frequency_derivatives(
    double[:] frequencies,
    double dt,
    double a_tol,
    double complex[:] derivatives,
):
    """Differentiate one nested integral with respect to each frequency."""
    cdef Py_ssize_t order = frequencies.shape[0]
    cdef Py_ssize_t insertion_index, source_index
    cdef object augmented_arr = np.empty(order + 1, dtype=float)
    cdef double[:] augmented = augmented_arr
    cdef double complex cumulative_integral = 0.

    # Inserting a zero-frequency integration at position r contributes the
    # interval between time-ordering variables r - 1 and r. Their cumulative
    # sum is the time multiplying each differentiated exponential.
    for insertion_index in range(order):
        for source_index in range(insertion_index):
            augmented[source_index] = frequencies[source_index]
        augmented[insertion_index] = 0.
        for source_index in range(insertion_index, order):
            augmented[source_index + 1] = frequencies[source_index]
        cumulative_integral += cy_compute_integrals(augmented, dt, a_tol)
        derivatives[insertion_index] = 1j * cumulative_integral


cpdef object cy_compute_integral_frequency_derivatives(
    double[:] frequencies,
    double dt,
    double a_tol=1e-10,
):
    """Differentiate a nested Dyson integral by its frequency entries."""
    cdef object derivatives_arr = np.empty(
        frequencies.shape[0], dtype=np.complex128
    )
    cdef double complex[:] derivatives = derivatives_arr
    _compute_integral_frequency_derivatives(
        frequencies, dt, a_tol, derivatives
    )
    return derivatives_arr


cdef double complex _compute_moment_integral_series(
    double[:] frequencies,
    long derivative_subset,
    double dt,
):
    """Evaluate short-step moments as a series in ``frequency * dt``.

    The general polynomial-exponential recurrence contains terms divided by
    frequency sums which are subtracted again at the integration boundaries.
    When ``abs(frequency * dt)`` is small, those individually large terms are
    nearly equal even though their final difference is proportional to a high
    power of ``dt``. Floating-point subtraction then loses the significant
    digits of the physical result (catastrophic cancellation).

    Rescaling every integration time as ``t = dt * x`` factors the exact
    ``dt ** (Dyson order + number of time factors)`` dependence out first.
    Expanding the remaining exponentials in the dimensionless variables
    ``frequency * dt`` evaluates only naturally scaled terms on ``0 <= x <= 1``
    and therefore remains accurate as ``dt`` approaches the timing-grid floor.
    """
    cdef int max_degree = 48
    cdef Py_ssize_t order = frequencies.shape[0]
    cdef Py_ssize_t position, exponential_degree, current_degree
    cdef Py_ssize_t selected_count = 0
    cdef Py_ssize_t integrated_power
    cdef double complex exponential_term, result
    cdef double scale = 1.
    cdef object current_arr = np.zeros(max_degree + 1, dtype=np.complex128)
    cdef object next_arr = np.zeros(max_degree + 1, dtype=np.complex128)
    cdef double complex[:] current = current_arr
    cdef double complex[:] next_coefficients = next_arr

    current[0] = 1.
    for position in range(order):
        if derivative_subset & (1 << position):
            selected_count += 1
        integrated_power = position + 1 + selected_count
        next_arr.fill(0.)
        exponential_term = 1.
        for exponential_degree in range(max_degree + 1):
            if exponential_degree > 0:
                exponential_term *= (
                    1j * frequencies[position] * dt / exponential_degree
                )
            for current_degree in range(
                max_degree - exponential_degree + 1
            ):
                next_coefficients[current_degree + exponential_degree] += (
                    current[current_degree]
                    * exponential_term
                    / (
                        integrated_power
                        + current_degree
                        + exponential_degree
                    )
                )
        for current_degree in range(max_degree + 1):
            current[current_degree] = next_coefficients[current_degree]

    result = 0.
    for current_degree in range(max_degree + 1):
        result += current[current_degree]
    for position in range(order + selected_count):
        scale *= dt
    return scale * result


cdef double complex _compute_moment_integral(
    double[:] frequencies,
    long derivative_subset,
    double dt,
    double a_tol,
):
    """Integrate selected time factors with nested exponentials.

    Short steps use the dimensionless series above because direct boundary
    cancellation is ill-conditioned there. Longer steps use the more compact
    polynomial-exponential recurrence, where the boundary terms are separated
    enough for stable floating-point subtraction.
    """
    cdef Py_ssize_t order = frequencies.shape[0]
    cdef Py_ssize_t scaled_frequency_index
    cdef double max_scaled_frequency = 0.
    cdef Py_ssize_t position, term_index, coefficient_index, search_index
    cdef Py_ssize_t current_count = 1
    cdef Py_ssize_t next_count, polynomial_degree, integrated_degree
    cdef Py_ssize_t zero_term_index
    cdef bint include_time
    cdef double combined_frequency
    cdef double complex boundary_constant
    cdef double complex polynomial_value
    cdef object current_frequencies_arr = np.zeros(order + 1, dtype=float)
    cdef object next_frequencies_arr = np.zeros(order + 1, dtype=float)
    cdef double[:] current_frequencies = current_frequencies_arr
    cdef double[:] next_frequencies = next_frequencies_arr
    cdef object current_polynomials_arr = np.zeros(
        (order + 1, 2 * order + 1), dtype=np.complex128
    )
    cdef object next_polynomials_arr = np.zeros(
        (order + 1, 2 * order + 1), dtype=np.complex128
    )
    cdef double complex[:, :] current_polynomials = current_polynomials_arr
    cdef double complex[:, :] next_polynomials = next_polynomials_arr
    cdef object current_degrees_arr = np.zeros(order + 1, dtype=np.int_)
    cdef object next_degrees_arr = np.zeros(order + 1, dtype=np.int_)
    cdef long[:] current_degrees = current_degrees_arr
    cdef long[:] next_degrees = next_degrees_arr

    for scaled_frequency_index in range(order):
        combined_frequency = abs(
            frequencies[scaled_frequency_index] * dt
        )
        if combined_frequency > max_scaled_frequency:
            max_scaled_frequency = combined_frequency
    # Avoid subtracting nearly equal exponential boundary terms on short steps.
    if max_scaled_frequency <= 2.:
        return _compute_moment_integral_series(
            frequencies, derivative_subset, dt
        )

    current_polynomials[0, 0] = 1.
    for position in range(order):
        include_time = (derivative_subset & (1 << position)) != 0
        next_count = 0
        boundary_constant = 0.
        next_polynomials_arr.fill(0.)
        next_degrees_arr.fill(0)
        for term_index in range(current_count):
            combined_frequency = (
                current_frequencies[term_index] + frequencies[position]
            )
            polynomial_degree = current_degrees[term_index] + include_time
            if abs(combined_frequency) < a_tol:
                zero_term_index = -1
                for search_index in range(next_count):
                    if abs(next_frequencies[search_index]) < a_tol:
                        zero_term_index = search_index
                        break
                if zero_term_index < 0:
                    zero_term_index = next_count
                    next_frequencies[next_count] = 0.
                    next_count += 1
                integrated_degree = polynomial_degree + 1
                if integrated_degree > next_degrees[zero_term_index]:
                    next_degrees[zero_term_index] = integrated_degree
                for coefficient_index in range(polynomial_degree + 1):
                    if coefficient_index >= include_time:
                        next_polynomials[
                            zero_term_index, coefficient_index + 1
                        ] += (
                            current_polynomials[
                                term_index, coefficient_index - include_time
                            ]
                            / (coefficient_index + 1)
                        )
            else:
                next_frequencies[next_count] = combined_frequency
                next_degrees[next_count] = polynomial_degree
                next_polynomials[next_count, polynomial_degree] = (
                    current_polynomials[
                        term_index, polynomial_degree - include_time
                    ]
                    / (1j * combined_frequency)
                )
                for coefficient_index in range(
                    polynomial_degree - 1, -1, -1
                ):
                    polynomial_value = 0.
                    if coefficient_index >= include_time:
                        polynomial_value = current_polynomials[
                            term_index, coefficient_index - include_time
                        ]
                    next_polynomials[next_count, coefficient_index] = (
                        polynomial_value
                        - (coefficient_index + 1)
                        * next_polynomials[next_count, coefficient_index + 1]
                    ) / (1j * combined_frequency)
                boundary_constant -= next_polynomials[next_count, 0]
                next_count += 1

        if boundary_constant != 0.:
            zero_term_index = -1
            for search_index in range(next_count):
                if abs(next_frequencies[search_index]) < a_tol:
                    zero_term_index = search_index
                    break
            if zero_term_index < 0:
                zero_term_index = next_count
                next_frequencies[next_count] = 0.
                next_count += 1
            next_polynomials[zero_term_index, 0] += boundary_constant

        current_count = next_count
        for term_index in range(current_count):
            current_frequencies[term_index] = next_frequencies[term_index]
            current_degrees[term_index] = next_degrees[term_index]
            for coefficient_index in range(current_degrees[term_index] + 1):
                current_polynomials[term_index, coefficient_index] = (
                    next_polynomials[term_index, coefficient_index]
                )

    polynomial_value = 0.
    for term_index in range(current_count):
        boundary_constant = 0.
        for coefficient_index in range(
            current_degrees[term_index], -1, -1
        ):
            boundary_constant = (
                boundary_constant * dt
                + current_polynomials[term_index, coefficient_index]
            )
        polynomial_value += (
            exp(1j * current_frequencies[term_index] * dt)
            * boundary_constant
        )
    return polynomial_value


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


cpdef object cy_compute_Sn_frequency_derivative_subsets(
    double[:, :] omega_vectors,
    long[:, :] ket_bra_idx,
    double[:, :] diff_lambdas,
    double complex[:] matrix_elements,
    double dt,
    int length,
    double a_tol=1e-10,
):
    """Differentiate one unscaled Dyson tensor for every frequency subset."""
    cdef Py_ssize_t n_omega = omega_vectors.shape[0]
    cdef Py_ssize_t n_paths = diff_lambdas.shape[0]
    cdef Py_ssize_t order = omega_vectors.shape[1]
    cdef Py_ssize_t subset_count = 1 << order
    cdef Py_ssize_t omega_index, path_index, frequency_index, subset
    cdef Py_ssize_t derivative_count
    cdef long row, col
    cdef object output_arr = np.zeros(
        (n_omega, subset_count, length, length), dtype=np.complex128
    )
    cdef double complex[:, :, :, :] output = output_arr
    cdef object frequencies_arr = np.empty(order, dtype=float)
    cdef double[:] frequencies = frequencies_arr
    cdef object derivative_positions_arr = np.empty(order, dtype=np.int_)
    cdef long[:] derivative_positions = derivative_positions_arr
    cdef double complex derivative

    for omega_index in range(n_omega):
        for path_index in range(n_paths):
            for frequency_index in range(order):
                frequencies[frequency_index] = (
                    omega_vectors[omega_index, frequency_index]
                    + diff_lambdas[path_index, frequency_index]
                )
            row = ket_bra_idx[path_index, 0]
            col = ket_bra_idx[path_index, 1]
            output[omega_index, 0, row, col] += (
                cy_compute_integrals(frequencies, dt, a_tol)
                * matrix_elements[path_index]
            )
            for subset in range(1, subset_count):
                derivative_count = 0
                for frequency_index in range(order):
                    if subset & (1 << frequency_index):
                        derivative_positions[derivative_count] = frequency_index
                        derivative_count += 1
                derivative = _compute_moment_integral(
                    frequencies, subset, dt, a_tol
                )
                for frequency_index in range(derivative_count):
                    derivative *= 1j
                output[omega_index, subset, row, col] += (
                    derivative * matrix_elements[path_index]
                )

    return output_arr


cpdef object cy_compute_Sn_frequency_derivatives(
    double[:, :] omega_vectors,
    long[:, :] ket_bra_idx,
    double[:, :] diff_lambdas,
    double complex[:] matrix_elements,
    double dt,
    int length,
    double a_tol=1e-10,
):
    """Differentiate one unscaled Dyson tensor by each branch frequency."""
    cdef Py_ssize_t n_omega = omega_vectors.shape[0]
    cdef Py_ssize_t n_paths = diff_lambdas.shape[0]
    cdef Py_ssize_t order = omega_vectors.shape[1]
    cdef Py_ssize_t omega_index, path_index, frequency_index
    cdef long row, col
    cdef object output_arr = np.zeros(
        (n_omega, order, length, length), dtype=np.complex128
    )
    cdef double complex[:, :, :, :] output = output_arr
    cdef object frequencies_arr = np.empty(order, dtype=float)
    cdef double[:] frequencies = frequencies_arr
    cdef object derivatives_arr = np.empty(order, dtype=np.complex128)
    cdef double complex[:] derivatives = derivatives_arr

    for omega_index in range(n_omega):
        for path_index in range(n_paths):
            for frequency_index in range(order):
                frequencies[frequency_index] = (
                    omega_vectors[omega_index, frequency_index]
                    + diff_lambdas[path_index, frequency_index]
                )
            _compute_integral_frequency_derivatives(
                frequencies, dt, a_tol, derivatives
            )
            row = ket_bra_idx[path_index, 0]
            col = ket_bra_idx[path_index, 1]
            for frequency_index in range(order):
                output[omega_index, frequency_index, row, col] += (
                    derivatives[frequency_index] * matrix_elements[path_index]
                )

    return output_arr


cdef complex cy_compute_tn_integrals(double[:] ws, int n, double dt, double a_tol=1e-10):
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
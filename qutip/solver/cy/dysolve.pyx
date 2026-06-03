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
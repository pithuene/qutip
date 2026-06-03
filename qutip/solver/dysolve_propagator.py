from qutip import Qobj, qeye_like
from .cy.dysolve import cy_compute_Sn
from numpy.typing import ArrayLike
import numpy as np
from numbers import Number
import itertools


__all__ = ['DysolvePropagator', 'dysolve_propagator']


class DysolvePropagator:
    """
    A generator of propagator using Dysolve.
    https://arxiv.org/abs/2012.09282

    Parameters
    ----------
    H_0 : Qobj
        The base hamiltonian of the system.

    X : Qobj
        A cosine perturbation applied on the system.

    omega : float
        The frequency of the cosine perturbation.

    options : dict, optional
        Extra parameters.

        - "max_order"

            A given integer to indicate the highest order of
            approximation used to compute the propagators (default is 4).
            This corresponds to n in eq. (4) of Ref.

        - "a_tol"

            The absolute tolerance used when computing the propagators
            (default is 1e-10).

        - "max_dt"

            The maximum time increment used when computing propagators
            (default is 0.1).

        - "batch_size"

            Number of same-dt substeps to contract at once when materializing
            subpropagators (default is 10). Larger values reduce Python
            overhead for small Hilbert spaces, but use more memory.

    Notes
    -----
    The system's hamiltonian must be of the form
    H = H_0 + cos(omega*t)X for Dysolve to work.

    For the moment, only a cosine perturbation is allowed. Dysolve can
    manage more exotic perturbations, but this is not implemented yet.

    .. note:: Experimental.

    """

    def __init__(
        self,
        H_0: Qobj,
        X: Qobj,
        omega: float,
        options: dict[str] = None,
    ):
        # System
        self._eigenenergies, self._basis = H_0.eigenstates()
        self._H_0 = H_0.transform(self._basis)
        self._X = X.transform(self._basis)
        self._elems = self._X.full().flatten()
        self._omega = omega

        # Options
        if options is None:
            self.max_order = 4
            self.a_tol = 1e-10
            self.max_dt = 0.1
            self.batch_size = 10
        else:
            self.max_order = options.get('max_order', 4)
            self.max_dt = options.get('max_dt', 0.1)
            self.a_tol = options.get('a_tol', 1e-10)
            self.batch_size = options.get('batch_size', 10)

        # Memoization
        self._dt_key_decimals = (
            options.get('dt_key_decimals', 15) if options is not None else 15
        )
        self._dt_Sns = {}
        self._omega_vectors = {}
        self._omega_sums = {}
        self._matrix_element_paths = {}

        # Time propagator
        self.U = None

    def __call__(self, t_f: float, t_i: float = 0.0) -> Qobj:
        """
        Computes the propagator from t_i to t_f. If t_i is not provided,
        computes the propagator from 0 to t_f.

        Parameters
        ----------
        t_f : float
            Final time of the evolution.

        t_i : float, default = 0.0
            Initial time of the evolution.

        Returns
        -------
        U : Qobj
            The propagator U(t_f, t_i) from t_i to t_f.

        Notes
        -----
        If t_f - t_i > max_dt, splits the evolution into smaller ones
        to then reconstruct U(t_f, t_i).

        Memoization is used. First call may be slow but the next calls
        should be faster.

        """
        U = self._compute_interval(t_i, t_f)

        self.U = Qobj(U, self._H_0._dims, copy=False).transform(
            self._basis, True
        )

        return self.U

    def propagators(self, t: ArrayLike) -> list[Qobj]:
        """
        Compute cumulative propagators for a list of times.

        Returns ``[U(t[i], t[0])]`` for all times in ``t``.  Unlike repeated
        calls to :meth:`__call__`, this keeps the cumulative propagator in the
        eigenbasis and only converts to ``Qobj`` at requested output times.
        Consecutive substeps with the same ``dt`` are contracted in batches.
        """
        t = np.asarray(t, dtype=float)
        if len(t) <= 1:
            return [qeye_like(self._H_0)]

        intervals = np.diff(t)
        direction = np.sign(intervals[0])
        if (
            direction == 0
            or np.any(np.sign(intervals) != direction)
            or np.any(np.abs(intervals) <= self.a_tol)
        ):
            return self._propagators_by_interval(t)

        step_times = []
        step_dts = []
        output_after_step = []
        for t_i, time_diff in zip(t[:-1], intervals):
            dt = self.max_dt * np.sign(time_diff)
            n_steps = abs(int(time_diff / self.max_dt))
            for j in range(n_steps):
                step_times.append(t_i + j*dt)
                step_dts.append(dt)
                output_after_step.append(False)
            remaining = time_diff - n_steps*dt
            if abs(remaining) > self.a_tol:
                step_times.append(t_i + n_steps*dt)
                step_dts.append(remaining)
                output_after_step.append(False)
            output_after_step[-1] = True

        return self._propagators_from_steps(
            np.asarray(step_times), np.asarray(step_dts), output_after_step
        )

    def _propagators_by_interval(self, t: ArrayLike) -> list[Qobj]:
        Us = [qeye_like(self._H_0).transform(self._basis, True)]
        U = np.eye(len(self._eigenenergies), dtype=np.complex128)
        for t_i, t_f in zip(t[:-1], t[1:]):
            U = self._compute_interval(t_i, t_f, U)
            Us.append(
                Qobj(U.copy(), self._H_0._dims, copy=False).transform(
                    self._basis, True
                )
            )
        return Us

    def _propagators_from_steps(
        self,
        step_times: ArrayLike,
        step_dts: ArrayLike,
        output_after_step: list[bool],
    ) -> list[Qobj]:
        Us = [qeye_like(self._H_0).transform(self._basis, True)]
        U = np.eye(len(self._eigenenergies), dtype=np.complex128)
        batch_size = self.batch_size
        step = 0

        while step < len(step_times):
            dt = step_dts[step]
            run_stop = step + 1
            while (
                run_stop < len(step_times)
                and abs(step_dts[run_stop] - dt) <= self.a_tol
            ):
                run_stop += 1

            for start in range(step, run_stop, batch_size):
                stop = min(start + batch_size, run_stop)
                subprops = self._compute_subprops(step_times[start:stop], dt)
                for offset, subpropagator in enumerate(subprops):
                    current_step = start + offset
                    U = subpropagator @ U
                    if output_after_step[current_step]:
                        Us.append(
                            Qobj(
                                U.copy(), self._H_0._dims, copy=False
                            ).transform(self._basis, True)
                        )
            step = run_stop

        return Us

    def _get_omega_vectors(self, n: int) -> ArrayLike:
        """
        Get all frequency sign combinations for a Dyson order.
        """
        if n not in self._omega_vectors:
            self._omega_vectors[n] = np.fromiter(
                itertools.product([self._omega, -self._omega], repeat=n),
                np.dtype((float, (n,)))
            )
        return self._omega_vectors[n]

    def _get_omega_sums(self, n: int) -> ArrayLike:
        """
        Get sums of all frequency sign combinations for a Dyson order.
        """
        if n not in self._omega_sums:
            self._omega_sums[n] = np.ascontiguousarray(
                np.sum(self._get_omega_vectors(n), axis=1), dtype=float
            )
        return self._omega_sums[n]

    def _get_matrix_element_paths(self, n: int) -> tuple[ArrayLike, ArrayLike]:
        """
        Get nonzero matrix-element paths for a Dyson order.

        For an order ``n`` path ``[k_n, ..., k_0]``, the stored value is
        ``X[k_n, k_{n-1}] * ... * X[k_1, k_0]``.  Paths with a zero product
        never contribute to the Dyson operators, so pruning them avoids the
        ``N ** (n + 1)`` all-path loop for sparse drive operators.
        """
        if n in self._matrix_element_paths:
            return self._matrix_element_paths[n]

        length = self._X.shape[0]
        rows, cols = np.nonzero(self._elems.reshape(length, length))
        values = self._elems.reshape(length, length)[rows, cols]

        if n == 1:
            paths = np.column_stack((rows, cols)).astype(np.int64)
            path_values = values
        else:
            previous_paths, previous_values = self._get_matrix_element_paths(
                n - 1
            )
            paths = []
            path_values = []
            previous_final = previous_paths[:, 0]
            for row, col, value in zip(rows, cols, values):
                matching = np.nonzero(previous_final == col)[0]
                if len(matching):
                    paths.append(
                        np.column_stack((
                            np.full(len(matching), row, dtype=np.int64),
                            previous_paths[matching],
                        ))
                    )
                    path_values.append(value * previous_values[matching])
            if paths:
                paths = np.vstack(paths)
                path_values = np.concatenate(path_values)
            else:
                paths = np.empty((0, n + 1), dtype=np.int64)
                path_values = np.empty(0, dtype=np.complex128)

        self._matrix_element_paths[n] = (paths, path_values)
        return paths, path_values

    def _dt_cache_key(self, dt: float) -> float:
        """
        Normalize time-step cache keys.

        Pulse programs often produce mathematically identical durations with
        tiny floating-point differences. Exact float keys defeat the expensive
        Dyson-operator cache, so cache at femtosecond-like precision by
        default while still using the rounded value consistently.
        """
        return round(float(dt), self._dt_key_decimals)

    def _compute_Sns(self, dt: float) -> dict:
        """
        Computes Sns for each omega vector. This implements a similar equation
        to eq. (14) in Ref, but the function "f" is not used to avoid dealing
        explicitly with limits.

        Parameters
        ----------
        dt : float
            The time increment.

        Returns
        -------
        Sns : dict
            Sns for each omega vector. key = order with the result for each
            omega vector.

        """
        dt_key = self._dt_cache_key(dt)
        if dt_key in self._dt_Sns:
            return self._dt_Sns[dt_key]

        else:
            dt = dt_key
            Sns = {}
            length = len(self._eigenenergies)
            exp_H_0 = (-1j*dt*self._H_0).expm().full()
            eigenenergies = np.asarray(self._eigenenergies)

            Sns[0] = exp_H_0

            for n in range(1, self.max_order + 1):
                omega_vectors = self._get_omega_vectors(n)
                paths, matrix_elements = self._get_matrix_element_paths(n)
                path_energies = eigenenergies[paths]
                diff_lambdas = -np.diff(path_energies)[:, ::-1]
                ket_bra_idx = paths[:, [0, -1]]

                Sn = cy_compute_Sn(
                    np.ascontiguousarray(omega_vectors, dtype=float),
                    np.ascontiguousarray(ket_bra_idx, dtype=np.int_),
                    np.ascontiguousarray(diff_lambdas, dtype=float),
                    np.ascontiguousarray(matrix_elements, dtype=np.complex128),
                    dt,
                    length,
                    self.a_tol,
                )
                Sn *= (-1j / 2) ** n
                Sn = exp_H_0 @ Sn

                Sns[n] = Sn

            self._dt_Sns[dt_key] = Sns
            return Sns

    def _envelope_branch_factors(
        self,
        amplitudes: ArrayLike,
        n: int,
        *,
        gradient: bool | str = False,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Return branch amplitude factors for a piecewise-constant envelope.

        ``amplitudes`` may be real, for a cosine envelope, or complex.  A
        complex amplitude ``a = x + 1j*y`` represents the real drive
        ``x*cos(omega*t) + y*sin(omega*t)`` multiplying ``X``.  Since the
        prepared Dyson tensors already include the ``1/2`` factors from the
        cosine decomposition, each positive-frequency branch is weighted by
        ``conj(a)`` and each negative-frequency branch by ``a``.
        """
        amplitudes = np.asarray(amplitudes)
        omega_vectors = self._get_omega_vectors(n)
        n_positive = np.count_nonzero(
            np.isclose(omega_vectors, self._omega), axis=1
        )
        n_negative = n - n_positive

        plus = np.conjugate(amplitudes)[:, None]
        minus = amplitudes[:, None]
        factors = plus ** n_positive[None, :] * minus ** n_negative[None, :]
        if not gradient:
            return factors

        d_plus_dx = np.ones_like(plus)
        d_minus_dx = np.ones_like(minus)
        d_plus_dy = -1j * np.ones_like(plus)
        d_minus_dy = 1j * np.ones_like(minus)

        with np.errstate(divide='ignore', invalid='ignore'):
            d_dx = np.zeros_like(factors, dtype=np.result_type(factors, complex))
            d_dy = np.zeros_like(d_dx)
            pos_mask = n_positive > 0
            neg_mask = n_negative > 0
            if np.any(pos_mask):
                d_dx[:, pos_mask] += (
                    n_positive[pos_mask][None, :]
                    * plus ** (n_positive[pos_mask][None, :] - 1)
                    * minus ** n_negative[pos_mask][None, :]
                    * d_plus_dx
                )
                d_dy[:, pos_mask] += (
                    n_positive[pos_mask][None, :]
                    * plus ** (n_positive[pos_mask][None, :] - 1)
                    * minus ** n_negative[pos_mask][None, :]
                    * d_plus_dy
                )
            if np.any(neg_mask):
                d_dx[:, neg_mask] += (
                    n_negative[neg_mask][None, :]
                    * plus ** n_positive[neg_mask][None, :]
                    * minus ** (n_negative[neg_mask][None, :] - 1)
                    * d_minus_dx
                )
                d_dy[:, neg_mask] += (
                    n_negative[neg_mask][None, :]
                    * plus ** n_positive[neg_mask][None, :]
                    * minus ** (n_negative[neg_mask][None, :] - 1)
                    * d_minus_dy
                )
        return factors, d_dx, d_dy

    def _compute_envelope_subprops(
        self,
        amplitudes: ArrayLike,
        dt: float,
        t0: float = 0.0,
        *,
        gradient: bool | str = False,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Compute all subpixel propagators for a shaped envelope.
        """
        amplitudes = np.asarray(amplitudes)
        if amplitudes.ndim != 1:
            raise ValueError("amplitudes must be a one-dimensional array")
        if len(amplitudes) == 0:
            raise ValueError("amplitudes must contain at least one value")

        current_times = t0 + np.arange(len(amplitudes), dtype=float) * dt
        Sns = self._compute_Sns(dt)
        length = len(self._eigenenergies)
        subpropagators = np.broadcast_to(
            Sns[0], (len(amplitudes), length, length)
        ).astype(np.complex128, copy=True)
        if gradient:
            dsubprops_dx = np.zeros_like(subpropagators)
            dsubprops_dy = np.zeros_like(subpropagators)

        for n in range(1, self.max_order + 1):
            omega_sums = self._get_omega_sums(n)
            phases = np.exp(1j * np.outer(current_times, omega_sums))
            if gradient:
                amp_factors, d_amp_dx, d_amp_dy = self._envelope_branch_factors(
                    amplitudes, n, gradient=True
                )
                dsubprops_dx += np.tensordot(
                    phases * d_amp_dx, Sns[n], axes=(1, 0)
                )
                dsubprops_dy += np.tensordot(
                    phases * d_amp_dy, Sns[n], axes=(1, 0)
                )
            else:
                amp_factors = self._envelope_branch_factors(amplitudes, n)
            subpropagators += np.tensordot(
                phases * amp_factors, Sns[n], axes=(1, 0)
            )

        if gradient:
            return subpropagators, dsubprops_dx, dsubprops_dy
        return subpropagators

    def envelope_propagator(
        self,
        amplitudes: ArrayLike,
        dt: float,
        t0: float = 0.0,
        *,
        gradient: bool | str = False,
    ):
        """
        Propagator for a piecewise-constant shaped drive envelope.

        The Hamiltonian represented by this method is
        ``H = H0 + X * (x_l*cos(omega*t) + y_l*sin(omega*t))`` during the
        subpixel ``l``, where ``amplitudes[l] = x_l + 1j*y_l``.  Real
        amplitudes therefore produce a shaped cosine drive.  The operator
        ``X`` passed to :class:`DysolvePropagator` should be the drive
        operator per unit envelope amplitude.

        If ``gradient`` is false, return ``U(T, t0)`` as a ``Qobj``.  If
        ``gradient='real'``, also return a tuple of derivatives with respect
        to the real cosine amplitudes.  If ``gradient='quadratures'``, return
        derivatives with respect to the real and imaginary quadratures as
        ``(U, dU_dx, dU_dy)``.
        """
        if gradient not in (False, 'real', 'quadratures'):
            raise ValueError(
                "gradient must be False, 'real', or 'quadratures'"
            )
        want_gradient = bool(gradient)
        if want_gradient:
            step_data = self._compute_envelope_subprops(
                amplitudes, dt, t0, gradient=True
            )
            subprops, dsubprops_dx, dsubprops_dy = step_data
        else:
            subprops = self._compute_envelope_subprops(amplitudes, dt, t0)

        length = len(self._eigenenergies)
        prefixes = [np.eye(length, dtype=np.complex128)]
        for subprop in subprops:
            prefixes.append(subprop @ prefixes[-1])
        total = prefixes[-1]
        U = Qobj(total, self._H_0._dims, copy=False).transform(
            self._basis, True
        )
        if not want_gradient:
            return U

        suffix = np.eye(length, dtype=np.complex128)
        dU_dx = [None] * len(subprops)
        dU_dy = [None] * len(subprops)
        for index in range(len(subprops) - 1, -1, -1):
            dU_dx[index] = suffix @ dsubprops_dx[index] @ prefixes[index]
            dU_dy[index] = suffix @ dsubprops_dy[index] @ prefixes[index]
            suffix = suffix @ subprops[index]

        dU_dx = tuple(
            Qobj(dU, self._H_0._dims, copy=False).transform(
                self._basis, True
            )
            for dU in dU_dx
        )
        dU_dy = tuple(
            Qobj(dU, self._H_0._dims, copy=False).transform(
                self._basis, True
            )
            for dU in dU_dy
        )
        if gradient == 'real':
            return U, dU_dx
        return U, dU_dx, dU_dy

    def _compute_interval(
        self,
        t_i: float,
        t_f: float,
        U: ArrayLike = None,
    ) -> ArrayLike:
        """
        Apply the propagator from ``t_i`` to ``t_f`` to ``U`` in eigenbasis.
        """
        time_diff = t_f - t_i
        if U is None:
            U = np.eye(len(self._eigenenergies), dtype=np.complex128)
        if abs(time_diff) <= self.a_tol:
            return U

        dt = self.max_dt * np.sign(time_diff)
        n_steps = abs(int(time_diff / self.max_dt))
        batch_size = self.batch_size

        for start in range(0, n_steps, batch_size):
            stop = min(start + batch_size, n_steps)
            current_times = t_i + np.arange(start, stop) * dt
            for subpropagator in self._compute_subprops(current_times, dt):
                U = subpropagator @ U

        remaining = time_diff - n_steps*dt
        if abs(remaining) > self.a_tol:
            U = self._compute_subprop(t_i + n_steps*dt, remaining) @ U

        return U

    def _compute_subprops(
        self, current_times: ArrayLike, dt: float
    ) -> ArrayLike:
        """
        Computes a batch of subpropagators.
        """
        current_times = np.asarray(current_times, dtype=float)
        Sns = self._compute_Sns(dt)
        length = len(self._eigenenergies)

        subpropagators = np.broadcast_to(
            Sns[0], (len(current_times), length, length)
        ).astype(np.complex128, copy=True)

        for n in range(1, self.max_order + 1):
            omega_sums = self._get_omega_sums(n)
            phases = np.exp(1j * np.outer(current_times, omega_sums))
            subpropagators += np.tensordot(phases, Sns[n], axes=(1, 0))

        return subpropagators

    def _compute_subprop(self, current_time: float, dt: float) -> ArrayLike:
        """
        Computes a subpropagator U(current_time + dt, current_time).

        Parameters
        ----------
        current_time : float
            The starting time of the evolution. Can be positive or negative.

        dt : float
            The time increment.

        Returns
        -------
        subpropagator : ArrayLike
            U(current_time + dt, current_time).

        """
        return self._compute_subprops([current_time], dt)[0]


def dysolve_propagator(
        H_0: Qobj,
        X: Qobj,
        omega: float,
        t: float | list[float],
        options: dict[str] = None
) -> Qobj | list[Qobj]:
    """
    A generator of propagator(s) using Dysolve.
    https://arxiv.org/abs/2012.09282.

    Parameters
    ----------
    H_0 : Qobj
        The hamiltonian of the system.

    X : Qobj
        A cosine perturbation applied on the system.

    omega : float
        The frequency of the cosine perturbation.

    t : float | list[float]
        Time or list of times for which to evaluate the propagator(s). If t
        is a single number, the propagator from 0 to t is computed. When
        t is a list, the propagators from the first time to each elements in
        t is returned. In that case, the first output will always be the
        identity matrix. Also, in that case, have the same time increment in
        between elements for better performance.

    options : dict, optional
        Extra parameters.

        - "max_order"

            A given integer to indicate the highest order of
            approximation used to compute the propagators (default is 4).
            This corresponds to n in eq. (4) of Ref.

        - "a_tol"

            The absolute tolerance used when computing the propagators
            (default is 1e-10).

        - "max_dt"

            The maximum time increment used when computing propagators
            (default is 0.1).

    Returns
    -------
    Us : Qobj | list[Qobj]
        The time evolution propagator U(t,0) if t is a single number or else
        a list of propagators [U(t[i], t[0])] for all elements t[i] in t.

    Notes
    -----
    The system's hamiltonian must be of the form
    H = H_0 + cos(omega*t)X for Dysolve to work.

    For the moment, only a cosine perturbation is allowed. Dysolve can
    manage more exotic perturbations, but this is not implemented yet.

    .. note:: Experimental.

    """
    if isinstance(t, Number):
        dysolve = DysolvePropagator(H_0, X, omega, options)
        return dysolve(t)

    else:
        dysolve = DysolvePropagator(H_0, X, omega, options)
        Us = dysolve.propagators(t)

    return Us

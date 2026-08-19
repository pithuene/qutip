import itertools
from collections import OrderedDict
from numbers import Number

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erf

from qutip import Qobj, qeye_like

from .cy.dysolve import cy_compute_Sn

__all__ = ['DysolvePropagator', 'PreparedDysolveEnvelope', 'dysolve_propagator', 'gaussian_filter_matrix']


def gaussian_filter_matrix(
    n_pixels: int,
    subpixels_per_pixel: int,
    pixel_dt: float,
    bandwidth: float,
) -> ArrayLike:
    """Return a Gaussian filter matrix from pixels to subpixels.

    The returned matrix has shape ``(n_subpixels, n_pixels)``, where
    ``n_subpixels = n_pixels * subpixels_per_pixel``.  Multiplying pixel
    values by ``T.T`` gives the filtered subpixel amplitudes.  Rows are
    normalized so a constant input envelope remains constant near the finite
    pulse boundaries.
    """
    if n_pixels <= 0:
        raise ValueError("n_pixels must be positive")
    if subpixels_per_pixel <= 0:
        raise ValueError("subpixels_per_pixel must be positive")
    if pixel_dt <= 0.0:
        raise ValueError("pixel_dt must be positive")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")

    n_subpixels = n_pixels * subpixels_per_pixel
    subpixel_dt = pixel_dt / subpixels_per_pixel
    subpixel_times = (np.arange(n_subpixels, dtype=float) + 0.5) * subpixel_dt
    pixel_starts = np.arange(n_pixels, dtype=float) * pixel_dt
    pixel_ends = pixel_starts + pixel_dt
    scaled_starts = 0.5 * bandwidth * (
        subpixel_times[:, None] - pixel_starts[None, :]
    )
    scaled_ends = 0.5 * bandwidth * (
        subpixel_times[:, None] - pixel_ends[None, :]
    )
    matrix = 0.5 * (erf(scaled_starts) - erf(scaled_ends))
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.divide(
            matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0.0
        )
    return matrix


class PreparedDysolveEnvelope:
    """Prepared t0-independent Dysolve contraction for a fixed envelope.

    The expensive envelope-amplitude contraction is prepared once for a fixed
    subpixel amplitude vector and timestep.  Absolute pulse start time remains
    a runtime input through carrier phase factors, so this object can be reused
    by identical pulses placed at different times.
    """

    def __init__(self, solver, amplitudes: ArrayLike, dt: float):
        self._solver = solver
        self.dt = solver.canonical_time_step(dt)
        self._amplitudes = np.array(
            solver._as_drive_amplitudes(amplitudes),
            dtype=np.complex128,
            order='C',
            copy=True,
        )
        self.n_subpixels = int(self._amplitudes.shape[1])
        self._length = len(solver._eigenenergies)
        self._S0 = solver._compute_Sns(self.dt)[0]
        self._terms = self._prepare_terms()

    @property
    def nbytes(self) -> int:
        """Return bytes in arrays owned only by this prepared envelope.

        Shared solver tensors such as ``_S0`` and cached Dyson tensors are
        excluded.
        """
        return self._amplitudes.nbytes + sum(
            omega_sums.nbytes + contractions.nbytes
            for omega_sums, contractions in self._terms
        )

    def _prepare_terms(self):
        solver = self._solver
        local_times = np.arange(self.n_subpixels, dtype=float) * self.dt
        Sns = solver._compute_Sns(self.dt)
        terms = []
        for n in range(1, solver.max_order + 1):
            omega_sums = solver._get_omega_sums(n)
            amp_factors = solver._envelope_branch_factors(self._amplitudes, n)
            branch_weights = np.exp(1j * np.outer(local_times, omega_sums)) * amp_factors
            unique_omega_sums, inverse = np.unique(omega_sums, return_inverse=True)
            contractions = np.zeros(
                (len(unique_omega_sums), self.n_subpixels, self._length, self._length),
                dtype=np.complex128,
            )
            for group_index in range(len(unique_omega_sums)):
                branch_indices = inverse == group_index
                contractions[group_index] = np.tensordot(
                    branch_weights[:, branch_indices],
                    Sns[n][branch_indices],
                    axes=(1, 0),
                )
            terms.append((unique_omega_sums, contractions))
        return tuple(terms)

    def _subpropagators_slice(self, start: int, stop: int, t0: float) -> ArrayLike:
        """Materialize one chronological slice of prepared subpropagators."""
        subpropagators = np.broadcast_to(
            self._S0, (stop - start, self._length, self._length)
        ).astype(np.complex128, copy=True)
        for omega_sums, contractions in self._terms:
            phases = np.exp(1j * omega_sums * float(t0))
            subpropagators += np.tensordot(
                phases, contractions[:, start:stop], axes=(0, 0)
            )
        return subpropagators

    def subpropagators(self, t0: float = 0.0) -> ArrayLike:
        """Return subpixel propagators for a pulse starting at ``t0``."""
        return self._subpropagators_slice(0, self.n_subpixels, t0)

    def propagator(self, t0: float = 0.0) -> Qobj:
        """Return the full envelope propagator for a pulse starting at ``t0``."""
        total = np.eye(self._length, dtype=np.complex128)
        for start in range(0, self.n_subpixels, self._solver.batch_size):
            stop = min(start + self._solver.batch_size, self.n_subpixels)
            for subpropagator in self._subpropagators_slice(start, stop, t0):
                total = subpropagator @ total
        return Qobj(total, self._solver._H_0._dims, copy=False).transform(
            self._solver._basis, True
        )


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

        - "dt_cache_size"

            Maximum number of canonical time-step tensors retained by each
            solver (default is 64). Zero disables this cache.

    Notes
    -----
    The standard propagator methods support Hamiltonians of the form
    ``H = H_0 + sum_m cos(omega_m*t) X_m``.  Use
    :meth:`from_drives` for multiple drive terms.  Piecewise-constant real or
    complex envelopes are supported by :meth:`envelope_propagator`, where
    complex amplitudes represent I/Q quadratures.  Gaussian-filtered optimizer
    pixels are supported by :meth:`filtered_envelope_propagator`.

    More general perturbations described in the Dysolve paper, such as
    arbitrary modulation functions or linear interpolation within subpixels,
    are not implemented.

    .. note:: Experimental.

    """

    def __init__(
        self,
        H_0: Qobj,
        X: Qobj,
        omega: float,
        options: dict[str] = None,
    ):
        self._initialize(H_0, [(X, omega)], options)

    @classmethod
    def from_drives(
        cls,
        H_0: Qobj,
        drives,
        options: dict[str] = None,
    ):
        """Create a Dysolve propagator with multiple drive terms.

        ``drives`` is an iterable of ``(X, omega)`` pairs.  The represented
        Hamiltonian is ``H0 + sum_m X_m cos(omega_m t)`` for the standard
        propagator methods.  Complex envelopes in :meth:`envelope_propagator`
        provide independent I/Q quadratures for each drive.
        """
        obj = cls.__new__(cls)
        obj._initialize(H_0, drives, options)
        return obj

    def _initialize(self, H_0: Qobj, drives, options: dict[str] = None):
        drives = tuple(drives)
        if len(drives) == 0:
            raise ValueError("at least one drive must be supplied")

        # System
        self._eigenenergies, self._basis = H_0.eigenstates()
        self._H_0 = H_0.transform(self._basis)
        self._Xs = tuple(X.transform(self._basis) for X, _ in drives)
        self._omegas = np.asarray([omega for _, omega in drives], dtype=float)
        self._n_drives = len(self._Xs)

        # Backward-compatible single-drive attributes.
        self._X = self._Xs[0]
        self._omega = float(self._omegas[0])
        self._elems_by_drive = tuple(X.full().flatten() for X in self._Xs)
        self._elems = self._elems_by_drive[0]

        # Options
        if options is None:
            self.max_order = 4
            self.a_tol = 1e-10
            self.max_dt = 0.1
            self.batch_size = 10
            self.dt_cache_size = 64
        else:
            self.max_order = options.get('max_order', 4)
            self.max_dt = options.get('max_dt', 0.1)
            self.a_tol = options.get('a_tol', 1e-10)
            self.batch_size = options.get('batch_size', 10)
            self.dt_cache_size = options.get('dt_cache_size', 64)
        if type(self.dt_cache_size) is not int or self.dt_cache_size < 0:
            raise ValueError("dt_cache_size must be a non-negative integer")

        # Memoization
        self._dt_key_decimals = (
            options.get('dt_key_decimals', 15) if options is not None else 15
        )
        self._dt_Sns = OrderedDict()
        self._omega_vectors = {}
        self._omega_sums = {}
        self._branch_drive_indices = {}
        self._branch_signs = {}
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

    def _get_branch_metadata(self, n: int) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Get drive indices, signs and frequencies for a Dyson order.

        The branch slots are ordered from the rightmost time-ordered integral
        to the leftmost one, matching the convention used by
        ``cy_compute_integrals``.
        """
        if n not in self._omega_vectors:
            alphabet = [
                (drive_index, sign, sign * self._omegas[drive_index])
                for drive_index in range(self._n_drives)
                for sign in (1, -1)
            ]
            branches = tuple(itertools.product(alphabet, repeat=n))
            self._branch_drive_indices[n] = np.asarray(
                [[entry[0] for entry in branch] for branch in branches],
                dtype=np.int_,
            )
            self._branch_signs[n] = np.asarray(
                [[entry[1] for entry in branch] for branch in branches],
                dtype=np.int_,
            )
            self._omega_vectors[n] = np.asarray(
                [[entry[2] for entry in branch] for branch in branches],
                dtype=float,
            )
        return (
            self._branch_drive_indices[n],
            self._branch_signs[n],
            self._omega_vectors[n],
        )

    def _get_omega_vectors(self, n: int) -> ArrayLike:
        """
        Get all drive-frequency sign combinations for a Dyson order.
        """
        return self._get_branch_metadata(n)[2]

    def _get_omega_sums(self, n: int) -> ArrayLike:
        """
        Get sums of all drive-frequency sign combinations for a Dyson order.
        """
        if n not in self._omega_sums:
            self._omega_sums[n] = np.ascontiguousarray(
                np.sum(self._get_omega_vectors(n), axis=1), dtype=float
            )
        return self._omega_sums[n]

    def _get_matrix_element_paths(
        self, drive_indices: tuple[int, ...]
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Get nonzero matrix-element paths for an ordered drive sequence.

        ``drive_indices`` is ordered from the rightmost Dyson integral to the
        leftmost one.  For an order ``n`` path ``[k_n, ..., k_0]``, the stored
        value is ``X_left[k_n, k_{n-1}] * ... * X_right[k_1, k_0]``.
        """
        drive_indices = tuple(int(index) for index in drive_indices)
        if drive_indices in self._matrix_element_paths:
            return self._matrix_element_paths[drive_indices]

        length = self._Xs[0].shape[0]
        drive = drive_indices[-1]
        elems = self._elems_by_drive[drive].reshape(length, length)
        rows, cols = np.nonzero(elems)
        values = elems[rows, cols]

        if len(drive_indices) == 1:
            paths = np.column_stack((rows, cols)).astype(np.int64)
            path_values = values.astype(np.complex128, copy=False)
        else:
            previous_paths, previous_values = self._get_matrix_element_paths(
                drive_indices[:-1]
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
                paths = np.empty((0, len(drive_indices) + 1), dtype=np.int64)
                path_values = np.empty(0, dtype=np.complex128)

        self._matrix_element_paths[drive_indices] = (paths, path_values)
        return paths, path_values

    def canonical_time_step(self, dt: float) -> float:
        """Return the time step used for preparation and solver caches.

        Pulse programs often produce mathematically identical durations with
        tiny floating-point differences. The configured precision normalizes
        those differences before any prepared or shared tensors use the step.
        """
        return round(float(dt), self._dt_key_decimals)

    def _compute_Sns(self, dt: float) -> dict:
        """
        Computes Sns for each branch vector. This implements a similar equation
        to eq. (14) in Ref, but the function "f" is not used to avoid dealing
        explicitly with limits.

        Parameters
        ----------
        dt : float
            The time increment.

        Returns
        -------
        Sns : dict
            Sns for each branch vector, keyed by Dyson order.
        """
        dt_key = self.canonical_time_step(dt)
        if dt_key in self._dt_Sns:
            Sns = self._dt_Sns[dt_key]
            self._dt_Sns.move_to_end(dt_key)
            return Sns

        dt = dt_key
        Sns = {}
        length = len(self._eigenenergies)
        exp_H_0 = (-1j*dt*self._H_0).expm().full()
        eigenenergies = np.asarray(self._eigenenergies)

        Sns[0] = exp_H_0

        for n in range(1, self.max_order + 1):
            drive_indices, _, omega_vectors = self._get_branch_metadata(n)
            Sn = np.zeros(
                (len(omega_vectors), length, length), dtype=np.complex128
            )
            unique_drive_indices = np.unique(drive_indices, axis=0)
            for drive_sequence in unique_drive_indices:
                mask = np.all(drive_indices == drive_sequence[None, :], axis=1)
                paths, matrix_elements = self._get_matrix_element_paths(
                    tuple(drive_sequence)
                )
                if len(paths) == 0:
                    continue
                path_energies = eigenenergies[paths]
                diff_lambdas = -np.diff(path_energies)[:, ::-1]
                ket_bra_idx = paths[:, [0, -1]]

                Sn_group = cy_compute_Sn(
                    np.ascontiguousarray(omega_vectors[mask], dtype=float),
                    np.ascontiguousarray(ket_bra_idx, dtype=np.int_),
                    np.ascontiguousarray(diff_lambdas, dtype=float),
                    np.ascontiguousarray(matrix_elements, dtype=np.complex128),
                    dt,
                    length,
                    self.a_tol,
                )
                Sn_group *= (-1j / 2) ** n
                Sn[mask] = exp_H_0 @ Sn_group

            Sns[n] = Sn

        if self.dt_cache_size > 0:
            self._dt_Sns[dt_key] = Sns
            while len(self._dt_Sns) > self.dt_cache_size:
                self._dt_Sns.popitem(last=False)
        return Sns

    def _as_drive_amplitudes(self, amplitudes: ArrayLike) -> ArrayLike:
        """Return amplitudes as an ``(n_drives, n_subpixels)`` array."""
        amplitudes = np.asarray(amplitudes)
        if amplitudes.ndim == 1 and self._n_drives == 1:
            amplitudes = amplitudes[None, :]
        elif amplitudes.ndim != 2:
            raise ValueError(
                "amplitudes must be one-dimensional for one drive or "
                "two-dimensional with shape (n_drives, n_subpixels)"
            )
        if amplitudes.shape[0] != self._n_drives:
            raise ValueError(
                "amplitudes first dimension must match the number of drives"
            )
        if amplitudes.shape[1] == 0:
            raise ValueError("amplitudes must contain at least one subpixel")
        return amplitudes

    def _format_drive_gradients(self, gradients):
        """Preserve the single-drive gradient return format."""
        if self._n_drives == 1:
            return tuple(gradients[0])
        return tuple(tuple(drive_gradients) for drive_gradients in gradients)

    def _envelope_branch_factors(
        self,
        amplitudes: ArrayLike,
        n: int,
        *,
        gradient: bool | str = False,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Return branch amplitude factors for a piecewise-constant envelope.

        ``amplitudes`` has shape ``(n_drives, n_subpixels)``.  A complex
        amplitude ``a = x + 1j*y`` represents the real drive
        ``x*cos(omega*t) + y*sin(omega*t)`` multiplying that drive's operator.
        Since the prepared Dyson tensors already include the ``1/2`` factors
        from the cosine decomposition, each positive-frequency branch is
        weighted by ``conj(a)`` and each negative-frequency branch by ``a``.
        """
        amplitudes = np.asarray(amplitudes)
        drive_indices, signs, _ = self._get_branch_metadata(n)
        n_subpixels = amplitudes.shape[1]
        n_branches = drive_indices.shape[0]

        factors = np.ones((n_subpixels, n_branches), dtype=np.result_type(amplitudes, complex))
        values_by_position = []
        for position in range(n):
            values = np.empty_like(factors)
            for drive_index in range(self._n_drives):
                drive_mask = drive_indices[:, position] == drive_index
                if not np.any(drive_mask):
                    continue
                positive = drive_mask & (signs[:, position] > 0)
                negative = drive_mask & (signs[:, position] < 0)
                if np.any(positive):
                    values[:, positive] = np.conjugate(amplitudes[drive_index])[:, None]
                if np.any(negative):
                    values[:, negative] = amplitudes[drive_index, :, None]
            values_by_position.append(values)
            factors *= values

        if not gradient:
            return factors

        d_dx = np.zeros(
            (self._n_drives, n_subpixels, n_branches),
            dtype=np.result_type(amplitudes, complex),
        )
        d_dy = np.zeros_like(d_dx)
        for position in range(n):
            product_except = np.ones_like(factors)
            for other_position, values in enumerate(values_by_position):
                if other_position != position:
                    product_except *= values
            for drive_index in range(self._n_drives):
                drive_mask = drive_indices[:, position] == drive_index
                if not np.any(drive_mask):
                    continue
                positive = drive_mask & (signs[:, position] > 0)
                negative = drive_mask & (signs[:, position] < 0)
                if np.any(positive):
                    branch_indices = np.nonzero(positive)[0]
                    d_dx[drive_index][:, branch_indices] += product_except[:, branch_indices]
                    d_dy[drive_index][:, branch_indices] += -1j * product_except[:, branch_indices]
                if np.any(negative):
                    branch_indices = np.nonzero(negative)[0]
                    d_dx[drive_index][:, branch_indices] += product_except[:, branch_indices]
                    d_dy[drive_index][:, branch_indices] += 1j * product_except[:, branch_indices]
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
        amplitudes = self._as_drive_amplitudes(amplitudes)
        n_subpixels = amplitudes.shape[1]

        current_times = t0 + np.arange(n_subpixels, dtype=float) * dt
        Sns = self._compute_Sns(dt)
        length = len(self._eigenenergies)
        subpropagators = np.broadcast_to(
            Sns[0], (n_subpixels, length, length)
        ).astype(np.complex128, copy=True)
        if gradient:
            dsubprops_dx = np.zeros(
                (self._n_drives, n_subpixels, length, length), dtype=np.complex128
            )
            dsubprops_dy = np.zeros_like(dsubprops_dx)

        for n in range(1, self.max_order + 1):
            omega_sums = self._get_omega_sums(n)
            phases = np.exp(1j * np.outer(current_times, omega_sums))
            if gradient:
                amp_factors, d_amp_dx, d_amp_dy = self._envelope_branch_factors(
                    amplitudes, n, gradient=True
                )
                for drive_index in range(self._n_drives):
                    dsubprops_dx[drive_index] += np.tensordot(
                        phases * d_amp_dx[drive_index], Sns[n], axes=(1, 0)
                    )
                    dsubprops_dy[drive_index] += np.tensordot(
                        phases * d_amp_dy[drive_index], Sns[n], axes=(1, 0)
                    )
            else:
                amp_factors = self._envelope_branch_factors(amplitudes, n)
            subpropagators += np.tensordot(
                phases * amp_factors, Sns[n], axes=(1, 0)
            )

        if gradient:
            return subpropagators, dsubprops_dx, dsubprops_dy
        return subpropagators

    def estimate_prepared_envelope_nbytes(self, n_subpixels: int) -> int:
        """Estimate arrays owned by a prepared complex128 envelope.

        Shared solver tensors are excluded, matching
        :attr:`PreparedDysolveEnvelope.nbytes`.
        """
        if n_subpixels <= 0:
            raise ValueError("n_subpixels must be positive")
        length = len(self._eigenenergies)
        amplitude_bytes = self._n_drives * n_subpixels * np.dtype(np.complex128).itemsize
        term_bytes = 0
        for order in range(1, self.max_order + 1):
            unique_omega_count = len(np.unique(self._get_omega_sums(order)))
            term_bytes += unique_omega_count * (
                np.dtype(float).itemsize
                + n_subpixels * length * length * np.dtype(np.complex128).itemsize
            )
        return amplitude_bytes + term_bytes

    def prepare_envelope(self, amplitudes: ArrayLike, dt: float) -> PreparedDysolveEnvelope:
        """Prepare a reusable t0-independent envelope contraction."""
        return PreparedDysolveEnvelope(self, amplitudes, dt)

    def envelope_parameter_gradients(
        self,
        amplitudes: ArrayLike,
        amplitude_derivatives: ArrayLike,
        dt: float,
        t0: float = 0.0,
    ):
        """Return an envelope propagator and gradients for arbitrary parameters.

        ``amplitude_derivatives`` contains ``d amplitudes / d parameter``.  For
        one drive it may have shape ``(n_parameters, n_subpixels)``.  For one or
        more drives it may have shape ``(n_parameters, n_drives, n_subpixels)``.
        Complex derivatives are interpreted in the same I/Q convention as
        ``amplitudes``.
        """
        amplitudes = self._as_drive_amplitudes(amplitudes)
        derivatives = np.asarray(amplitude_derivatives, dtype=np.complex128)
        if derivatives.ndim == 2 and self._n_drives == 1:
            derivatives = derivatives[:, None, :]
        if derivatives.ndim != 3:
            raise ValueError(
                "amplitude_derivatives must have shape (n_parameters, n_subpixels) "
                "for one drive or (n_parameters, n_drives, n_subpixels)"
            )
        n_parameters = int(derivatives.shape[0])
        if derivatives.shape[1:] != amplitudes.shape:
            raise ValueError(
                "amplitude_derivatives trailing dimensions must match amplitudes"
            )

        length = len(self._eigenenergies)
        total = np.eye(length, dtype=np.complex128)
        parameter_gradients = np.zeros(
            (n_parameters, length, length), dtype=np.complex128
        )
        for start in range(0, amplitudes.shape[1], self.batch_size):
            stop = min(start + self.batch_size, amplitudes.shape[1])
            subprops, dsubprops_dx, dsubprops_dy = self._compute_envelope_subprops(
                amplitudes[:, start:stop],
                dt,
                t0 + start * dt,
                gradient=True,
            )
            batch_derivatives = derivatives[:, :, start:stop]
            for index, subpropagator in enumerate(subprops):
                subpropagator_derivatives = np.einsum(
                    'pd,dij->pij',
                    batch_derivatives[:, :, index].real,
                    dsubprops_dx[:, index],
                ) + np.einsum(
                    'pd,dij->pij',
                    batch_derivatives[:, :, index].imag,
                    dsubprops_dy[:, index],
                )
                parameter_gradients = (
                    subpropagator @ parameter_gradients
                    + subpropagator_derivatives @ total
                )
                total = subpropagator @ total

        U = Qobj(total, self._H_0._dims, copy=False).transform(
            self._basis, True
        )
        dU_dp = tuple(
            Qobj(gradient, self._H_0._dims, copy=False).transform(
                self._basis, True
            )
            for gradient in parameter_gradients
        )
        return U, dU_dp

    def envelope_propagator(
        self,
        amplitudes: ArrayLike,
        dt: float,
        t0: float = 0.0,
        *,
        gradient: bool | str = False,
    ):
        """
        Propagator for piecewise-constant shaped drive envelopes.

        For one drive, ``amplitudes`` may be one-dimensional.  For multiple
        drives, it must have shape ``(n_drives, n_subpixels)``.  The Hamiltonian
        represented by this method is
        ``H = H0 + sum_m X_m * (x_m,l*cos(omega_m*t) + y_m,l*sin(omega_m*t))``
        during subpixel ``l``, where ``amplitudes[m, l] = x_m,l + 1j*y_m,l``.

        If ``gradient`` is false, return ``U(T, t0)`` as a ``Qobj``.  If
        ``gradient='real'``, also return derivatives with respect to the real
        cosine amplitudes.  If ``gradient='quadratures'``, return derivatives
        with respect to the real and imaginary quadratures.
        """
        if gradient not in (False, 'real', 'quadratures'):
            raise ValueError(
                "gradient must be False, 'real', or 'quadratures'"
            )
        want_gradient = bool(gradient)
        if not want_gradient:
            amplitudes = self._as_drive_amplitudes(amplitudes)
            length = len(self._eigenenergies)
            total = np.eye(length, dtype=np.complex128)
            for start in range(0, amplitudes.shape[1], self.batch_size):
                stop = min(start + self.batch_size, amplitudes.shape[1])
                subpropagators = self._compute_envelope_subprops(
                    amplitudes[:, start:stop], dt, t0 + start * dt
                )
                for subpropagator in subpropagators:
                    total = subpropagator @ total
            return Qobj(total, self._H_0._dims, copy=False).transform(
                self._basis, True
            )

        step_data = self._compute_envelope_subprops(
            amplitudes, dt, t0, gradient=True
        )
        subprops, dsubprops_dx, dsubprops_dy = step_data

        length = len(self._eigenenergies)
        prefixes = [np.eye(length, dtype=np.complex128)]
        for subprop in subprops:
            prefixes.append(subprop @ prefixes[-1])
        total = prefixes[-1]
        U = Qobj(total, self._H_0._dims, copy=False).transform(
            self._basis, True
        )
        suffix = np.eye(length, dtype=np.complex128)
        n_subpixels = len(subprops)
        dU_dx = [[None] * n_subpixels for _ in range(self._n_drives)]
        dU_dy = [[None] * n_subpixels for _ in range(self._n_drives)]
        for index in range(n_subpixels - 1, -1, -1):
            for drive_index in range(self._n_drives):
                dU_dx[drive_index][index] = (
                    suffix @ dsubprops_dx[drive_index, index] @ prefixes[index]
                )
                dU_dy[drive_index][index] = (
                    suffix @ dsubprops_dy[drive_index, index] @ prefixes[index]
                )
            suffix = suffix @ subprops[index]

        for drive_index in range(self._n_drives):
            dU_dx[drive_index] = tuple(
                Qobj(dU, self._H_0._dims, copy=False).transform(
                    self._basis, True
                )
                for dU in dU_dx[drive_index]
            )
            dU_dy[drive_index] = tuple(
                Qobj(dU, self._H_0._dims, copy=False).transform(
                    self._basis, True
                )
                for dU in dU_dy[drive_index]
            )
        dU_dx = self._format_drive_gradients(dU_dx)
        dU_dy = self._format_drive_gradients(dU_dy)
        if gradient == 'real':
            return U, dU_dx
        return U, dU_dx, dU_dy

    def filtered_envelope_propagator(
        self,
        pixels: ArrayLike,
        pixel_dt: float,
        subpixels_per_pixel: int,
        bandwidth: float,
        t0: float = 0.0,
        *,
        gradient: bool | str = False,
    ):
        """Propagator for Gaussian-filtered optimizer pixels.

        ``pixels`` follows the same drive-axis convention as
        :meth:`envelope_propagator`, but indexes optimizer pixels instead of
        delivered subpixels.  A Gaussian filter maps pixels to subpixels before
        propagation.  Requested gradients are returned with respect to the
        unfiltered optimizer pixels.
        """
        if gradient not in (False, 'real', 'quadratures'):
            raise ValueError(
                "gradient must be False, 'real', or 'quadratures'"
            )
        pixels = self._as_drive_amplitudes(pixels)
        n_pixels = pixels.shape[1]
        filter_matrix = gaussian_filter_matrix(
            n_pixels, subpixels_per_pixel, pixel_dt, bandwidth
        )
        subpixel_dt = pixel_dt / subpixels_per_pixel
        subpixels = pixels @ filter_matrix.T
        if not gradient:
            return self.envelope_propagator(subpixels, subpixel_dt, t0)

        if gradient == 'real':
            U, dU_ds = self.envelope_propagator(
                subpixels, subpixel_dt, t0, gradient='real'
            )
            dU_ds = (dU_ds,) if self._n_drives == 1 else dU_ds
            dU_du = []
            for drive_index in range(self._n_drives):
                drive_gradients = []
                for pixel_index in range(n_pixels):
                    gradient_sum = sum(
                        filter_matrix[subpixel_index, pixel_index]
                        * dU_ds[drive_index][subpixel_index]
                        for subpixel_index in range(filter_matrix.shape[0])
                    )
                    drive_gradients.append(gradient_sum)
                dU_du.append(tuple(drive_gradients))
            return U, self._format_drive_gradients(dU_du)

        U, dU_dx_ds, dU_dy_ds = self.envelope_propagator(
            subpixels, subpixel_dt, t0, gradient='quadratures'
        )
        dU_dx_ds = (dU_dx_ds,) if self._n_drives == 1 else dU_dx_ds
        dU_dy_ds = (dU_dy_ds,) if self._n_drives == 1 else dU_dy_ds
        dU_dx_du = []
        dU_dy_du = []
        for drive_index in range(self._n_drives):
            drive_gradients_x = []
            drive_gradients_y = []
            for pixel_index in range(n_pixels):
                gradient_sum_x = sum(
                    filter_matrix[subpixel_index, pixel_index]
                    * dU_dx_ds[drive_index][subpixel_index]
                    for subpixel_index in range(filter_matrix.shape[0])
                )
                gradient_sum_y = sum(
                    filter_matrix[subpixel_index, pixel_index]
                    * dU_dy_ds[drive_index][subpixel_index]
                    for subpixel_index in range(filter_matrix.shape[0])
                )
                drive_gradients_x.append(gradient_sum_x)
                drive_gradients_y.append(gradient_sum_y)
            dU_dx_du.append(tuple(drive_gradients_x))
            dU_dy_du.append(tuple(drive_gradients_y))
        return (
            U,
            self._format_drive_gradients(dU_dx_du),
            self._format_drive_gradients(dU_dy_du),
        )

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
    This helper supports the single-drive Hamiltonian
    ``H = H_0 + cos(omega*t) X``.  For multiple drive terms or shaped
    envelopes, instantiate :class:`DysolvePropagator` directly.

    .. note:: Experimental.

    """
    if isinstance(t, Number):
        dysolve = DysolvePropagator(H_0, X, omega, options)
        return dysolve(t)

    else:
        dysolve = DysolvePropagator(H_0, X, omega, options)
        Us = dysolve.propagators(t)

    return Us

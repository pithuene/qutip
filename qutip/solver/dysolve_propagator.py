from collections.abc import Callable

from qutip import Qobj, qeye_like
from .cy.dysolve import (
    cy_compute_Sn,
    cy_compute_Sn_frequency_derivatives,
    cy_compute_Sn_frequency_derivative_subsets,
)
from numpy.typing import ArrayLike
from scipy.special import erf
import numpy as np
from numbers import Number
import itertools


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
        self.dt = float(dt)
        self._amplitudes = solver._as_drive_amplitudes(amplitudes)
        self.n_subpixels = int(self._amplitudes.shape[1])
        self._length = len(solver._eigenenergies)
        self._S0 = solver._compute_Sns(self.dt)[0]
        self._terms = self._prepare_terms()

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

    def subpropagators(self, t0: float = 0.0) -> ArrayLike:
        """Return subpixel propagators for a pulse starting at ``t0``."""
        subpropagators = np.broadcast_to(
            self._S0, (self.n_subpixels, self._length, self._length)
        ).astype(np.complex128, copy=True)
        for omega_sums, contractions in self._terms:
            phases = np.exp(1j * omega_sums * float(t0))
            subpropagators += np.tensordot(phases, contractions, axes=(0, 0))
        return subpropagators

    def propagator(self, t0: float = 0.0) -> Qobj:
        """Return the full envelope propagator for a pulse starting at ``t0``."""
        total = np.eye(self._length, dtype=np.complex128)
        for subpropagator in self.subpropagators(t0):
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

        - "min_timestep"

            Finest timing-grid increment for adaptive dyadic propagation.
            Absolute start and final times are rounded to this grid. Must be
            combined with ``error_tolerance_per_time``.

        - "error_tolerance_per_time"

            Maximum accepted full-step versus two-half-step error divided by
            step duration. Enables adaptive dyadic propagation when combined
            with ``min_timestep``.

    Notes
    -----
    The standard propagator methods support Hamiltonians of the form
    ``H = H_0 + sum_m cos(omega_m*t) X_m``.  Use
    :meth:`from_drives` for multiple drive terms.  Piecewise-constant real or
    complex envelopes are supported by :meth:`envelope_propagator`, where
    complex amplitudes represent I/Q quadratures.  Gaussian-filtered optimizer
    pixels are supported by :meth:`filtered_envelope_propagator`.

    Adaptive envelopes use integral-preserving linear interpolation within
    each selected subpixel. General arbitrary modulation functions described
    in the Dysolve paper are not implemented.

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
        options = {} if options is None else options
        if 'max_dt' in options:
            raise ValueError(
                "max_dt is no longer supported; use min_timestep and "
                "error_tolerance_per_time"
            )

        self.max_order = options.get('max_order', 4)
        self.a_tol = options.get('a_tol', 1e-10)
        self.min_timestep = options.get('min_timestep')
        self.error_tolerance_per_time = options.get(
            'error_tolerance_per_time'
        )
        if self.max_order > 0:
            if self.min_timestep is None or self.error_tolerance_per_time is None:
                raise ValueError(
                    "min_timestep and error_tolerance_per_time are required"
                )
            if self.min_timestep <= 0:
                raise ValueError("min_timestep must be positive")
            if self.error_tolerance_per_time <= 0:
                raise ValueError("error_tolerance_per_time must be positive")

        # Memoization
        self._dt_key_decimals = options.get('dt_key_decimals', 15)
        self._dt_Sns = {}
        self._dt_Sn_frequency_derivatives = {}
        self._dt_Sn_frequency_derivative_subsets = {}
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
        Adaptive propagation selects a dyadic timestep and Dyson order from
        ``min_timestep`` and ``error_tolerance_per_time``.

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
        return self._propagators_by_interval(t)

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

    def _dt_cache_key(self, dt: float) -> float:
        """
        Normalize time-step cache keys.

        Pulse programs often produce mathematically identical durations with
        tiny floating-point differences. Exact float keys defeat the expensive
        Dyson-operator cache, so cache at femtosecond-like precision by
        default while still using the rounded value consistently.
        """
        return round(float(dt), self._dt_key_decimals)

    def _compute_Sn(self, dt: float, order: int) -> ArrayLike:
        """Return one cached Dyson-order tensor for a time increment."""
        if order < 0 or order > self.max_order:
            raise ValueError(
                f"order must be between 0 and max_order={self.max_order}"
            )

        dt_key = self._dt_cache_key(dt)
        cached_orders = self._dt_Sns.setdefault(dt_key, {})
        if order in cached_orders:
            return cached_orders[order]

        exp_H_0 = cached_orders.get(0)
        if exp_H_0 is None:
            exp_H_0 = (-1j * dt_key * self._H_0).expm().full()
            cached_orders[0] = exp_H_0
        if order == 0:
            return exp_H_0

        length = len(self._eigenenergies)
        eigenenergies = np.asarray(self._eigenenergies)
        drive_indices, _, omega_vectors = self._get_branch_metadata(order)
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
                dt_key,
                length,
                self.a_tol,
            )
            Sn_group *= (-1j / 2) ** order
            Sn[mask] = exp_H_0 @ Sn_group

        cached_orders[order] = Sn
        return Sn

    def _compute_Sn_frequency_derivatives(
        self, dt: float, order: int
    ) -> ArrayLike:
        """Return cached derivatives with respect to branch frequencies."""
        if order < 1 or order > self.max_order:
            raise ValueError(
                f"order must be between 1 and max_order={self.max_order}"
            )

        dt_key = self._dt_cache_key(dt)
        cached_orders = self._dt_Sn_frequency_derivatives.setdefault(dt_key, {})
        if order in cached_orders:
            return cached_orders[order]

        exp_H_0 = self._compute_Sn(dt_key, 0)
        length = len(self._eigenenergies)
        eigenenergies = np.asarray(self._eigenenergies)
        drive_indices, _, omega_vectors = self._get_branch_metadata(order)
        derivatives = np.zeros(
            (len(omega_vectors), order, length, length), dtype=np.complex128
        )
        for drive_sequence in np.unique(drive_indices, axis=0):
            mask = np.all(drive_indices == drive_sequence[None, :], axis=1)
            paths, matrix_elements = self._get_matrix_element_paths(
                tuple(drive_sequence)
            )
            if len(paths) == 0:
                continue
            path_energies = eigenenergies[paths]
            diff_lambdas = -np.diff(path_energies)[:, ::-1]
            ket_bra_idx = paths[:, [0, -1]]
            derivative_group = cy_compute_Sn_frequency_derivatives(
                np.ascontiguousarray(omega_vectors[mask], dtype=float),
                np.ascontiguousarray(ket_bra_idx, dtype=np.int_),
                np.ascontiguousarray(diff_lambdas, dtype=float),
                np.ascontiguousarray(matrix_elements, dtype=np.complex128),
                dt_key,
                length,
                self.a_tol,
            )
            derivative_group *= (-1j / 2) ** order
            derivatives[mask] = exp_H_0 @ derivative_group

        cached_orders[order] = derivatives
        return derivatives

    def _compute_Sn_frequency_derivative_subsets(
        self, dt: float, order: int
    ) -> ArrayLike:
        """Return cached mixed derivatives for every frequency subset."""
        if order < 1 or order > self.max_order:
            raise ValueError(
                f"order must be between 1 and max_order={self.max_order}"
            )

        dt_key = self._dt_cache_key(dt)
        cached_orders = self._dt_Sn_frequency_derivative_subsets.setdefault(
            dt_key, {}
        )
        if order in cached_orders:
            return cached_orders[order]

        exp_H_0 = self._compute_Sn(dt_key, 0)
        length = len(self._eigenenergies)
        eigenenergies = np.asarray(self._eigenenergies)
        drive_indices, _, omega_vectors = self._get_branch_metadata(order)
        derivative_subsets = np.zeros(
            (len(omega_vectors), 1 << order, length, length),
            dtype=np.complex128,
        )
        for drive_sequence in np.unique(drive_indices, axis=0):
            mask = np.all(drive_indices == drive_sequence[None, :], axis=1)
            paths, matrix_elements = self._get_matrix_element_paths(
                tuple(drive_sequence)
            )
            if len(paths) == 0:
                continue
            path_energies = eigenenergies[paths]
            diff_lambdas = -np.diff(path_energies)[:, ::-1]
            ket_bra_idx = paths[:, [0, -1]]
            derivative_group = cy_compute_Sn_frequency_derivative_subsets(
                np.ascontiguousarray(omega_vectors[mask], dtype=float),
                np.ascontiguousarray(ket_bra_idx, dtype=np.int_),
                np.ascontiguousarray(diff_lambdas, dtype=float),
                np.ascontiguousarray(matrix_elements, dtype=np.complex128),
                dt_key,
                length,
                self.a_tol,
            )
            derivative_group *= (-1j / 2) ** order
            derivative_subsets[mask] = exp_H_0 @ derivative_group

        cached_orders[order] = derivative_subsets
        return derivative_subsets

    def _compute_Sns(self, dt: float, max_order: int = None) -> dict:
        """Return cached Dyson tensors through ``max_order``."""
        if max_order is None:
            max_order = self.max_order
        if max_order < 0 or max_order > self.max_order:
            raise ValueError(
                f"max_order must be between 0 and {self.max_order}"
            )
        return {
            order: self._compute_Sn(dt, order)
            for order in range(max_order + 1)
        }

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

    def _as_drive_values(self, values: ArrayLike, *, name: str) -> ArrayLike:
        """Return one complex value per configured drive."""
        values = np.asarray(values, dtype=np.complex128)
        if values.ndim == 0 and self._n_drives == 1:
            values = values.reshape(1)
        if values.shape != (self._n_drives,):
            raise ValueError(
                f"{name} must contain one value per drive; expected "
                f"shape ({self._n_drives},), got {values.shape}"
            )
        return values

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
        max_order: int = None,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Compute all subpixel propagators for a shaped envelope.
        """
        amplitudes = self._as_drive_amplitudes(amplitudes)
        n_subpixels = amplitudes.shape[1]

        current_times = t0 + np.arange(n_subpixels, dtype=float) * dt
        if max_order is None:
            max_order = self.max_order
        Sns = self._compute_Sns(dt, max_order=max_order)
        length = len(self._eigenenergies)
        subpropagators = np.broadcast_to(
            Sns[0], (n_subpixels, length, length)
        ).astype(np.complex128, copy=True)
        if gradient:
            dsubprops_dx = np.zeros(
                (self._n_drives, n_subpixels, length, length), dtype=np.complex128
            )
            dsubprops_dy = np.zeros_like(dsubprops_dx)

        for n in range(1, max_order + 1):
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

    def _compute_linear_envelope_subpropagator(
        self,
        start_amplitudes: ArrayLike,
        end_amplitudes: ArrayLike,
        dt: float,
        t0: float = 0.0,
        *,
        max_order: int = None,
        gradient: bool = False,
    ) -> ArrayLike | tuple[ArrayLike, ...]:
        """Propagate one linear subpixel and optionally its endpoint gradients."""
        if dt == 0.0:
            identity = np.eye(len(self._eigenenergies), dtype=np.complex128)
            if not gradient:
                return identity
            zero_gradient = np.zeros(
                (self._n_drives, *identity.shape), dtype=np.complex128
            )
            return identity, *(zero_gradient.copy() for _ in range(4))
        if max_order is None:
            max_order = self.max_order
        if max_order < 0 or max_order > self.max_order:
            raise ValueError(
                f"max_order must be between 0 and {self.max_order}"
            )

        start_amplitudes = self._as_drive_values(
            start_amplitudes, name="start_amplitudes"
        )
        end_amplitudes = self._as_drive_values(
            end_amplitudes, name="end_amplitudes"
        )
        dt = self._dt_cache_key(dt)
        slopes = (end_amplitudes - start_amplitudes) / dt
        length = len(self._eigenenergies)
        subpropagator = self._compute_Sn(dt, 0).copy()
        if gradient:
            endpoint_gradients = np.zeros(
                (4, self._n_drives, length, length), dtype=np.complex128
            )

        for order in range(1, max_order + 1):
            drive_indices, signs, _ = self._get_branch_metadata(order)
            n_branches = drive_indices.shape[0]
            branch_starts = np.empty(
                (n_branches, order), dtype=np.complex128
            )
            branch_slopes = np.empty_like(branch_starts)
            for position in range(order):
                for drive_index in range(self._n_drives):
                    drive_mask = drive_indices[:, position] == drive_index
                    positive = drive_mask & (signs[:, position] > 0)
                    negative = drive_mask & (signs[:, position] < 0)
                    branch_starts[positive, position] = np.conjugate(
                        start_amplitudes[drive_index]
                    )
                    branch_slopes[positive, position] = np.conjugate(
                        slopes[drive_index]
                    )
                    branch_starts[negative, position] = start_amplitudes[
                        drive_index
                    ]
                    branch_slopes[negative, position] = slopes[drive_index]

            phases = np.exp(1j * self._get_omega_sums(order) * float(t0))
            derivative_subsets = self._compute_Sn_frequency_derivative_subsets(
                dt, order
            )
            branch_indices = np.arange(n_branches)
            quadrature_derivatives = np.where(signs > 0, -1j, 1j)
            for subset in range(1 << order):
                values_by_position = [
                    -1j * branch_slopes[:, position]
                    if subset & (1 << position)
                    else branch_starts[:, position]
                    for position in range(order)
                ]
                coefficients = np.prod(values_by_position, axis=0)
                subset_tensor = derivative_subsets[:, subset]
                subpropagator += np.tensordot(
                    phases * coefficients, subset_tensor, axes=(0, 0)
                )
                if not gradient:
                    continue

                coefficient_gradients = np.zeros(
                    (4, self._n_drives, n_branches), dtype=np.complex128
                )
                for position in range(order):
                    product_except_position = np.ones(
                        n_branches, dtype=np.complex128
                    )
                    for other_position, values in enumerate(
                        values_by_position
                    ):
                        if other_position != position:
                            product_except_position *= values
                    quadrature_derivative = quadrature_derivatives[:, position]
                    if subset & (1 << position):
                        derivative_values = (
                            1j * product_except_position / dt,
                            1j * quadrature_derivative
                            * product_except_position
                            / dt,
                            -1j * product_except_position / dt,
                            -1j * quadrature_derivative
                            * product_except_position
                            / dt,
                        )
                    else:
                        derivative_values = (
                            product_except_position,
                            quadrature_derivative * product_except_position,
                            np.zeros(n_branches),
                            np.zeros(n_branches),
                        )
                    position_drives = drive_indices[:, position]
                    for endpoint_index, derivative_value in enumerate(
                        derivative_values
                    ):
                        coefficient_gradients[
                            endpoint_index, position_drives, branch_indices
                        ] += derivative_value
                endpoint_gradients += np.tensordot(
                    coefficient_gradients * phases,
                    subset_tensor,
                    axes=(2, 0),
                )

        assert subpropagator.shape == (length, length)
        if gradient:
            return subpropagator, *endpoint_gradients
        return subpropagator

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

        subprops, dsubprops_dx, dsubprops_dy = self._compute_envelope_subprops(
            amplitudes, dt, t0, gradient=True
        )
        length = len(self._eigenenergies)
        prefixes = [np.eye(length, dtype=np.complex128)]
        for subprop in subprops:
            prefixes.append(subprop @ prefixes[-1])
        total = prefixes[-1]
        U = Qobj(total, self._H_0._dims, copy=False).transform(
            self._basis, True
        )

        parameter_gradients = np.zeros(
            (n_parameters, length, length), dtype=np.complex128
        )
        suffix = np.eye(length, dtype=np.complex128)
        for index in range(len(subprops) - 1, -1, -1):
            for drive_index in range(self._n_drives):
                gradient_x = (
                    suffix @ dsubprops_dx[drive_index, index] @ prefixes[index]
                )
                gradient_y = (
                    suffix @ dsubprops_dy[drive_index, index] @ prefixes[index]
                )
                real_derivatives = derivatives[:, drive_index, index].real
                imaginary_derivatives = derivatives[:, drive_index, index].imag
                parameter_gradients += (
                    real_derivatives[:, None, None] * gradient_x
                    + imaginary_derivatives[:, None, None] * gradient_y
                )
            suffix = suffix @ subprops[index]

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
            return self.prepare_envelope(amplitudes, dt).propagator(t0)

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

    @staticmethod
    def _adaptive_boundaries(
        t_i: float,
        t_f: float,
        breakpoints: ArrayLike,
    ) -> tuple[float, ...]:
        """Return ordered interval boundaries inside the evolution span."""
        lower, upper = sorted((float(t_i), float(t_f)))
        interior = sorted(
            float(point)
            for point in np.asarray(breakpoints, dtype=float)
            if lower < point < upper
        )
        if t_f < t_i:
            interior.reverse()
        return (float(t_i), *interior, float(t_f))

    @staticmethod
    def _adaptive_envelope_sample_times(
        current_time: float, dt: float
    ) -> ArrayLike:
        """Sample a subpixel midpoint and both one-sided boundaries."""
        end_time = current_time + dt
        return np.asarray([
            np.nextafter(current_time, end_time),
            current_time + dt / 2,
            np.nextafter(end_time, current_time),
        ])

    @staticmethod
    def _integral_preserving_linear_endpoints(
        samples: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Return linear endpoints whose mean equals the midpoint sample."""
        sample_delta = samples[..., 2] - samples[..., 0]
        return (
            samples[..., 1] - sample_delta / 2,
            samples[..., 1] + sample_delta / 2,
        )

    def _compute_adaptive_linear_envelope_step(
        self,
        amplitude: Callable[[ArrayLike], ArrayLike],
        current_time: float,
        dt: float,
        order: int,
    ) -> ArrayLike:
        """Propagate one adaptive integral-preserving linear subpixel."""
        amplitudes = self._as_drive_amplitudes(
            amplitude(self._adaptive_envelope_sample_times(current_time, dt))
        )
        linear_start, linear_end = self._integral_preserving_linear_endpoints(
            amplitudes
        )
        if np.array_equal(linear_start, linear_end):
            return self._compute_envelope_subprops(
                linear_start[:, None],
                dt,
                current_time,
                max_order=order,
            )[0]
        return self._compute_linear_envelope_subpropagator(
            linear_start,
            linear_end,
            dt,
            current_time,
            max_order=order,
        )

    def _adaptive_frozen_envelope_split_error_rate(
        self,
        amplitude: Callable[[ArrayLike], ArrayLike],
        current_time: float,
        dt: float,
        order: int,
    ) -> float:
        """Estimate Dyson error with the local envelope value held fixed."""
        amplitudes = self._as_drive_amplitudes(
            amplitude(np.asarray([current_time + dt / 2]))
        )

        def frozen_step(step_time, step_dt, selected_order):
            return self._compute_envelope_subprops(
                amplitudes,
                step_dt,
                step_time,
                max_order=selected_order,
            )[0]

        return self._adaptive_split_error_rate(
            current_time, dt, order, frozen_step
        )

    def adaptive_envelope_propagator(
        self,
        amplitude: Callable[[ArrayLike], ArrayLike],
        t_f: float,
        t_i: float = 0.0,
        *,
        breakpoints: ArrayLike = (),
    ) -> Qobj:
        """Propagate an absolute-time amplitude envelope adaptively."""
        def step_propagator(current_time, dt, order):
            return self._compute_adaptive_linear_envelope_step(
                amplitude, current_time, dt, order
            )

        def order_error_rate(current_time, dt, order):
            return self._adaptive_frozen_envelope_split_error_rate(
                amplitude, current_time, dt, order
            )

        propagator = np.eye(len(self._eigenenergies), dtype=np.complex128)
        boundaries = self._adaptive_boundaries(t_i, t_f, breakpoints)
        for interval_start, interval_end in zip(
            boundaries[:-1], boundaries[1:], strict=True
        ):
            order, step_ticks = self._adaptive_interval_plan(
                interval_start,
                interval_end,
                step_propagator,
                order_error_rate=order_error_rate,
            )
            current_tick = round(interval_start / self.min_timestep)
            for step_tick in step_ticks:
                current_time = current_tick * self.min_timestep
                dt = step_tick * self.min_timestep
                propagator = step_propagator(
                    current_time, dt, order
                ) @ propagator
                current_tick += step_tick
        return Qobj(
            propagator, self._H_0._dims, copy=False
        ).transform(self._basis, True)

    def adaptive_envelope_parameter_gradients(
        self,
        amplitude: Callable[[ArrayLike], ArrayLike],
        amplitude_derivatives: Callable[[ArrayLike], ArrayLike],
        t_f: float,
        t_i: float = 0.0,
        *,
        breakpoints: ArrayLike = (),
    ) -> tuple[Qobj, tuple[Qobj, ...]]:
        """Propagate an envelope and its parameter derivatives adaptively."""
        def step_propagator(current_time, dt, order):
            return self._compute_adaptive_linear_envelope_step(
                amplitude, current_time, dt, order
            )

        def order_error_rate(current_time, dt, order):
            return self._adaptive_frozen_envelope_split_error_rate(
                amplitude, current_time, dt, order
            )

        propagator = np.eye(len(self._eigenenergies), dtype=np.complex128)
        propagator_gradients = None
        boundaries = self._adaptive_boundaries(t_i, t_f, breakpoints)
        for interval_start, interval_end in zip(
            boundaries[:-1], boundaries[1:], strict=True
        ):
            order, step_ticks = self._adaptive_interval_plan(
                interval_start,
                interval_end,
                step_propagator,
                order_error_rate=order_error_rate,
            )
            if not step_ticks:
                continue
            current_tick = round(interval_start / self.min_timestep)
            for step_tick in step_ticks:
                current_time = current_tick * self.min_timestep
                dt = step_tick * self.min_timestep
                sample_times = self._adaptive_envelope_sample_times(
                    current_time, dt
                )
                amplitudes = self._as_drive_amplitudes(amplitude(sample_times))
                linear_start, linear_end = (
                    self._integral_preserving_linear_endpoints(amplitudes)
                )
                step_data = self._compute_linear_envelope_subpropagator(
                    linear_start,
                    linear_end,
                    dt,
                    current_time,
                    max_order=order,
                    gradient=True,
                )
                (
                    step_propagator_matrix,
                    start_gradients_x,
                    start_gradients_y,
                    end_gradients_x,
                    end_gradients_y,
                ) = step_data

                derivatives = np.asarray(
                    amplitude_derivatives(sample_times), dtype=np.complex128
                )
                if self._n_drives == 1 and derivatives.ndim == 2:
                    derivatives = derivatives[:, None, :]
                expected_shape = (self._n_drives, 3)
                if (
                    derivatives.ndim != 3
                    or derivatives.shape[1:] != expected_shape
                ):
                    raise ValueError(
                        "amplitude_derivatives must have shape "
                        "(n_parameters, n_drives, n_times)"
                    )
                if propagator_gradients is None:
                    propagator_gradients = np.zeros(
                        (derivatives.shape[0], *propagator.shape),
                        dtype=np.complex128,
                    )

                linear_start_derivatives, linear_end_derivatives = (
                    self._integral_preserving_linear_endpoints(derivatives)
                )
                step_parameter_gradients = (
                    np.tensordot(
                        linear_start_derivatives.real,
                        start_gradients_x,
                        axes=(1, 0),
                    )
                    + np.tensordot(
                        linear_start_derivatives.imag,
                        start_gradients_y,
                        axes=(1, 0),
                    )
                    + np.tensordot(
                        linear_end_derivatives.real,
                        end_gradients_x,
                        axes=(1, 0),
                    )
                    + np.tensordot(
                        linear_end_derivatives.imag,
                        end_gradients_y,
                        axes=(1, 0),
                    )
                )

                previous_propagator = propagator
                propagator = step_propagator_matrix @ previous_propagator
                propagator_gradients = np.asarray([
                    step_gradient @ previous_propagator
                    + step_propagator_matrix @ propagator_gradient
                    for step_gradient, propagator_gradient in zip(
                        step_parameter_gradients,
                        propagator_gradients,
                        strict=True,
                    )
                ])
                current_tick += step_tick

        if propagator_gradients is None:
            raise ValueError("adaptive envelope interval must span at least one timestep")

        transformed_propagator = Qobj(
            propagator, self._H_0._dims, copy=False
        ).transform(self._basis, True)
        transformed_gradients = tuple(
            Qobj(gradient, self._H_0._dims, copy=False).transform(
                self._basis, True
            )
            for gradient in propagator_gradients
        )
        return transformed_propagator, transformed_gradients

    def _adaptive_split_error_rate(
        self,
        current_time: float,
        dt: float,
        order: int,
        step_propagator: Callable[[float, float, int], ArrayLike] = None,
    ) -> float:
        """Estimate one adaptive step's error per unit evolution time."""
        if step_propagator is None:
            step_propagator = self._compute_subprop
        half_dt = dt / 2
        full_step = step_propagator(current_time, dt, order)
        first_half = step_propagator(current_time, half_dt, order)
        second_half = step_propagator(
            current_time + half_dt, half_dt, order
        )
        split_step = second_half @ first_half
        return float(
            np.linalg.norm(full_step - split_step, ord='nuc') / abs(dt)
        )

    def _adaptive_plan_for_order(
        self,
        start_tick: int,
        stop_tick: int,
        order: int,
        piece_limit: int = None,
        step_propagator: Callable[[float, float, int], ArrayLike] = None,
        step_error_rate: Callable[[float, float, int], float] = None,
    ) -> tuple[int, ...] | None:
        """Build actual-time-checked dyadic steps for one Dyson order."""
        def evaluate_step_error(current_time, dt):
            if step_error_rate is not None:
                return step_error_rate(current_time, dt, order)
            return self._adaptive_split_error_rate(
                current_time, dt, order, step_propagator
            )

        direction = 1 if stop_tick > start_tick else -1
        current_tick = start_tick
        step_ticks = []
        accepted_tick_hint = None

        while current_tick != stop_tick:
            remaining_ticks = abs(stop_tick - current_tick)
            largest_remaining_tick = 1 << (remaining_ticks.bit_length() - 1)
            candidate_tick = min(
                accepted_tick_hint or largest_remaining_tick,
                largest_remaining_tick,
            )
            accepted_tick = None
            while candidate_tick >= 1:
                dt = direction * candidate_tick * self.min_timestep
                current_time = current_tick * self.min_timestep
                error_rate = evaluate_step_error(current_time, dt)
                if error_rate <= self.error_tolerance_per_time:
                    accepted_tick = candidate_tick
                    break
                candidate_tick //= 2

            if accepted_tick is None:
                return None

            larger_tick = accepted_tick * 2
            while accepted_tick_hint is not None and larger_tick <= remaining_ticks:
                dt = direction * larger_tick * self.min_timestep
                current_time = current_tick * self.min_timestep
                error_rate = evaluate_step_error(current_time, dt)
                if not error_rate <= self.error_tolerance_per_time:
                    break
                accepted_tick = larger_tick
                larger_tick *= 2
            accepted_tick_hint = accepted_tick
            signed_tick = direction * accepted_tick
            step_ticks.append(signed_tick)
            current_tick += signed_tick
            if piece_limit is not None and len(step_ticks) > piece_limit:
                return tuple(step_ticks)

        return tuple(step_ticks)

    def _adaptive_interval_plan(
        self,
        t_i: float,
        t_f: float,
        step_propagator: Callable[[float, float, int], ArrayLike] = None,
        *,
        order_error_rate: Callable[[float, float, int], float] = None,
    ) -> tuple[int, tuple[int, ...]]:
        """Select a Dyson order and dyadic step plan for one interval."""
        start_tick = round(float(t_i) / self.min_timestep)
        stop_tick = round(float(t_f) / self.min_timestep)
        if start_tick == stop_tick:
            return 1, ()

        escalation_thresholds = {1: 1000, 2: 200, 3: 100}
        fallback_orders = []
        selected_plan = None
        for order in range(1, self.max_order + 1):
            piece_limit = (
                escalation_thresholds.get(order)
                if order < self.max_order
                else None
            )
            step_ticks = self._adaptive_plan_for_order(
                start_tick,
                stop_tick,
                order,
                piece_limit,
                step_propagator,
                order_error_rate,
            )
            if step_ticks is None:
                continue
            fallback_orders.append(order)
            if piece_limit is None or len(step_ticks) <= piece_limit:
                selected_plan = (order, step_ticks)
                break

        if selected_plan is None:
            for fallback_order in reversed(fallback_orders):
                step_ticks = self._adaptive_plan_for_order(
                    start_tick,
                    stop_tick,
                    fallback_order,
                    step_propagator=step_propagator,
                    step_error_rate=order_error_rate,
                )
                if step_ticks is not None:
                    selected_plan = (fallback_order, step_ticks)
                    break

        if selected_plan is None:
            raise ValueError(
                "Requested error_tolerance_per_time cannot be met with "
                "max_order and min_timestep"
            )

        order, step_ticks = selected_plan
        if order_error_rate is not None:
            step_ticks = self._adaptive_plan_for_order(
                start_tick,
                stop_tick,
                order,
                step_propagator=step_propagator,
            )
            if step_ticks is None:
                raise ValueError(
                    "Requested error_tolerance_per_time cannot be met with "
                    "max_order and min_timestep"
                )
        return order, step_ticks

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
        if self.max_order == 0:
            return self._compute_subprop(
                t_i, time_diff, max_order=0
            ) @ U

        order, step_ticks = self._adaptive_interval_plan(t_i, t_f)
        current_tick = round(float(t_i) / self.min_timestep)
        for step_tick in step_ticks:
            current_time = current_tick * self.min_timestep
            dt = step_tick * self.min_timestep
            U = self._compute_subprop(
                current_time, dt, max_order=order
            ) @ U
            current_tick += step_tick
        return U

    def _compute_subprops(
        self,
        current_times: ArrayLike,
        dt: float,
        max_order: int = None,
    ) -> ArrayLike:
        """
        Computes a batch of subpropagators.
        """
        current_times = np.asarray(current_times, dtype=float)
        if max_order is None:
            max_order = self.max_order
        Sns = self._compute_Sns(dt, max_order=max_order)
        length = len(self._eigenenergies)

        subpropagators = np.broadcast_to(
            Sns[0], (len(current_times), length, length)
        ).astype(np.complex128, copy=True)

        for n in range(1, max_order + 1):
            omega_sums = self._get_omega_sums(n)
            phases = np.exp(1j * np.outer(current_times, omega_sums))
            subpropagators += np.tensordot(phases, Sns[n], axes=(1, 0))

        return subpropagators

    def _compute_subprop(
        self,
        current_time: float,
        dt: float,
        max_order: int = None,
    ) -> ArrayLike:
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
        return self._compute_subprops(
            [current_time], dt, max_order=max_order
        )[0]


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

        - "min_timestep", "error_tolerance_per_time"

            Required adaptive timing-grid and error-rate settings.

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

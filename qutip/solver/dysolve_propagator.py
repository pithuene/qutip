import itertools
from collections import OrderedDict
from math import comb
from numbers import Number
from typing import NamedTuple

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


class _DysolveControlPolynomial(NamedTuple):
    """Static Dyson matrices collected by equal control monomial."""

    free_propagator: ArrayLike
    positive_counts: ArrayLike
    negative_counts: ArrayLike
    phase_charges: ArrayLike
    coefficients: ArrayLike


class PreparedDysolveEnvelope:
    """Prepared t0-independent control monomials for a fixed envelope.

    Envelope and local-time factors are prepared once. Absolute start time and
    carrier phases remain runtime inputs, so identical pulses can be reused at
    different placements without storing subpropagator matrices.
    """

    def __init__(self, solver, amplitudes: ArrayLike, dt: float):
        self._solver = solver
        self.dt = solver.canonical_time_step(dt)
        amplitudes = np.array(
            solver._as_drive_amplitudes(amplitudes),
            dtype=np.complex128,
            order='C',
            copy=True,
        )
        self.n_subpixels = int(amplitudes.shape[1])
        self._length = len(solver._eigenenergies)
        self._polynomial = solver._compute_control_polynomial(self.dt)
        local_times = np.arange(self.n_subpixels, dtype=float) * self.dt
        rotations = np.exp(-1j * np.outer(solver._omegas, local_times))
        drive_factors = solver._control_monomial_factors(
            amplitudes * rotations,
            self._polynomial,
        )
        self._control_values = np.prod(drive_factors, axis=0)

    @property
    def nbytes(self) -> int:
        """Return bytes in arrays owned only by this prepared envelope."""
        return self._control_values.nbytes

    def _subpropagators_slice(
        self,
        start: int,
        stop: int,
        t0: float,
        carrier_phases: ArrayLike,
    ) -> ArrayLike:
        """Materialize one chronological slice of prepared subpropagators."""
        polynomial = self._polynomial
        subpropagators = np.broadcast_to(
            polynomial.free_propagator,
            (stop - start, self._length, self._length),
        ).astype(np.complex128, copy=True)
        effective_phases = self._solver._omegas * float(t0) + carrier_phases
        monomial_phases = np.exp(1j * (polynomial.phase_charges @ effective_phases))
        subpropagators += np.tensordot(
            self._control_values[start:stop] * monomial_phases,
            polynomial.coefficients,
            axes=(1, 0),
        )
        return subpropagators

    def subpropagators(
        self,
        t0: float = 0.0,
        carrier_phases: ArrayLike | None = None,
    ) -> ArrayLike:
        """Return subpixel propagators with runtime carrier phases."""
        phases = self._solver._as_carrier_phases(carrier_phases)
        return self._subpropagators_slice(0, self.n_subpixels, t0, phases)

    def propagator(
        self,
        t0: float = 0.0,
        carrier_phases: ArrayLike | None = None,
    ) -> Qobj:
        """Return the full envelope propagator with runtime carrier phases."""
        phases = self._solver._as_carrier_phases(carrier_phases)
        total = np.eye(self._length, dtype=np.complex128)
        batch_size = self._solver._fixed_order_batch_size()
        for start in range(0, self.n_subpixels, batch_size):
            stop = min(start + batch_size, self.n_subpixels)
            subpropagators = self._subpropagators_slice(start, stop, t0, phases)
            batch_product, _ = self._solver._chronological_product(subpropagators)
            total, _ = self._solver._chronological_product(
                np.stack((total, batch_product))
            )
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

            Number of same-dt substeps to contract at once in gradient paths
            (default is 10).

        - "fixed_order_workspace_bytes"

            Transient workspace ceiling for fixed-order envelope propagation
            and parameter gradients (default is 256 MiB). The batch size is
            estimated conservatively from the solver and derivative shapes.

        - "fixed_order_batch_size"

            Performance tile for fixed-order envelope propagation and
            parameter gradients (default is 512 steps). The workspace ceiling
            can reduce this tile for larger systems or derivative sets.

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
            self.fixed_order_workspace_bytes = 256 * 1024 * 1024
            self.fixed_order_batch_size = 512
            self.dt_cache_size = 64
        else:
            self.max_order = options.get('max_order', 4)
            self.max_dt = options.get('max_dt', 0.1)
            self.a_tol = options.get('a_tol', 1e-10)
            self.batch_size = options.get('batch_size', 10)
            self.fixed_order_workspace_bytes = options.get(
                'fixed_order_workspace_bytes', 256 * 1024 * 1024
            )
            self.fixed_order_batch_size = options.get(
                'fixed_order_batch_size', 512
            )
            self.dt_cache_size = options.get('dt_cache_size', 64)
        positive_integer_options = {
            "fixed_order_workspace_bytes": self.fixed_order_workspace_bytes,
            "fixed_order_batch_size": self.fixed_order_batch_size,
        }
        for name, value in positive_integer_options.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.dt_cache_size) is not int or self.dt_cache_size < 0:
            raise ValueError("dt_cache_size must be a non-negative integer")

        # Memoization
        self._dt_key_decimals = (
            options.get('dt_key_decimals', 15) if options is not None else 15
        )
        self._dt_Sns = OrderedDict()
        self._control_polynomials = {}
        self._omega_vectors = {}
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

    def _fixed_order_batch_size(self, n_parameters: int = 0) -> int:
        """Return a workspace-bounded envelope batch size."""
        length = len(self._eigenenergies)
        matrix_bytes = length * length * np.dtype(np.complex128).itemsize
        matrix_count = 3
        if n_parameters:
            matrix_count += 2 * self._n_drives + 4 * n_parameters
        branch_workspace_bytes = max(
            (
                (order + 4 + (2 * self._n_drives if n_parameters else 0))
                * (2 * self._n_drives) ** order
                * np.dtype(np.complex128).itemsize
                for order in range(1, self.max_order + 1)
            ),
            default=0,
        )
        bytes_per_step = (
            matrix_count * matrix_bytes
            + branch_workspace_bytes
            + np.dtype(float).itemsize
        )
        workspace_batch_size = max(
            1,
            self.fixed_order_workspace_bytes // bytes_per_step,
        )
        return min(self.fixed_order_batch_size, workspace_batch_size)

    @staticmethod
    def _chronological_product(
        matrices: ArrayLike,
        derivatives: ArrayLike | None = None,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Reduce chronological propagators and parameter derivatives."""
        product = np.asarray(matrices)
        if derivatives is None:
            product_derivatives = np.empty(
                (0, len(product), product.shape[-2], product.shape[-1]),
                dtype=np.complex128,
            )
        else:
            product_derivatives = np.asarray(derivatives)
            if product_derivatives.shape[1:] != product.shape:
                raise ValueError(
                    "derivatives must have shape "
                    "(n_parameters, n_steps, dimension, dimension)"
                )
        while len(product) > 1:
            pair_count = len(product) // 2
            earlier = product[:2 * pair_count:2]
            later = product[1:2 * pair_count:2]
            reduced = later @ earlier
            reduced_derivatives = (
                product_derivatives[:, 1:2 * pair_count:2] @ earlier
                + later @ product_derivatives[:, :2 * pair_count:2]
            )
            if len(product) % 2:
                reduced = np.concatenate((reduced, product[-1:]))
                reduced_derivatives = np.concatenate(
                    (reduced_derivatives, product_derivatives[:, -1:]),
                    axis=1,
                )
            product = reduced
            product_derivatives = reduced_derivatives
        return product[0], product_derivatives[:, 0]

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

    def _compute_control_polynomial(self, dt: float) -> _DysolveControlPolynomial:
        r"""Collect ordered Dyson branches with equal control dependence.

        Equation (B2) of the Dysolve paper weights every ordered branch by a
        product of complex drive amplitudes. For a piecewise-constant subpixel,
        that product depends only on how often each positive- and
        negative-frequency drive branch occurs, not on their order. If
        ``z_d = a_d exp(-i omega_d t)``, all branches with counts ``p_d`` and
        ``q_d`` share the monomial

        ``prod_d conj(z_d)**p_d * z_d**q_d``.

        Their ordered operator integrals remain distinct and are summed into
        one static matrix coefficient. This only reassociates the finite Dyson
        sum; it does not change its order, timestep, or counter-rotating terms.
        """
        dt_key = self.canonical_time_step(dt)
        if dt_key in self._control_polynomials:
            self._dt_Sns.move_to_end(dt_key)
            return self._control_polynomials[dt_key]

        Sns = self._compute_Sns(dt_key)
        positive_counts = []
        negative_counts = []
        coefficients = []
        for order in range(1, self.max_order + 1):
            drive_indices, signs, _ = self._get_branch_metadata(order)
            branch_positive_counts = np.zeros(
                (len(drive_indices), self._n_drives), dtype=np.int_
            )
            branch_negative_counts = np.zeros_like(branch_positive_counts)
            for drive_index in range(self._n_drives):
                drive_mask = drive_indices == drive_index
                branch_positive_counts[:, drive_index] = np.sum(
                    drive_mask & (signs > 0), axis=1
                )
                branch_negative_counts[:, drive_index] = np.sum(
                    drive_mask & (signs < 0), axis=1
                )
            branch_counts = np.column_stack(
                (branch_positive_counts, branch_negative_counts)
            )
            unique_counts, inverse = np.unique(
                branch_counts, axis=0, return_inverse=True
            )
            for group_index, counts in enumerate(unique_counts):
                positive_counts.append(counts[:self._n_drives])
                negative_counts.append(counts[self._n_drives:])
                coefficients.append(
                    np.sum(Sns[order][inverse == group_index], axis=0)
                )

        length = len(self._eigenenergies)
        positive_counts = np.asarray(positive_counts, dtype=np.int_).reshape(
            -1, self._n_drives
        )
        negative_counts = np.asarray(negative_counts, dtype=np.int_).reshape(
            -1, self._n_drives
        )
        polynomial = _DysolveControlPolynomial(
            free_propagator=Sns[0],
            positive_counts=positive_counts,
            negative_counts=negative_counts,
            phase_charges=positive_counts - negative_counts,
            coefficients=np.asarray(coefficients, dtype=np.complex128).reshape(
                -1, length, length
            ),
        )
        if self.dt_cache_size > 0:
            self._control_polynomials[dt_key] = polynomial
        return polynomial

    @staticmethod
    def _control_monomial_factors(
        rotated_amplitudes: ArrayLike,
        polynomial: _DysolveControlPolynomial,
    ) -> ArrayLike:
        """Return one control-monomial factor array per drive."""
        return np.asarray(
            [
                np.conj(rotated_amplitudes[drive_index])[:, None]
                ** polynomial.positive_counts[None, :, drive_index]
                * rotated_amplitudes[drive_index, :, None]
                ** polynomial.negative_counts[None, :, drive_index]
                for drive_index in range(rotated_amplitudes.shape[0])
            ]
        )

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
                evicted_dt, _ = self._dt_Sns.popitem(last=False)
                self._control_polynomials.pop(evicted_dt, None)
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

    def _as_carrier_phases(self, carrier_phases: ArrayLike | None) -> ArrayLike:
        """Return one explicit carrier phase for every drive."""
        if carrier_phases is None:
            return np.zeros(self._n_drives, dtype=float)
        phases = np.asarray(carrier_phases, dtype=float)
        if phases.ndim == 0 and self._n_drives == 1:
            phases = phases[None]
        if phases.shape != (self._n_drives,):
            raise ValueError("carrier_phases must contain one phase per drive")
        return phases

    def _format_drive_gradients(self, gradients):
        """Preserve the single-drive gradient return format."""
        if self._n_drives == 1:
            return tuple(gradients[0])
        return tuple(tuple(drive_gradients) for drive_gradients in gradients)

    def _compute_control_subprops(
        self,
        amplitudes: ArrayLike,
        current_times: ArrayLike,
        dt: float,
        *,
        gradient: bool | str = False,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Evaluate the collected control polynomial at each requested time."""
        amplitudes = self._as_drive_amplitudes(amplitudes)
        current_times = np.asarray(current_times, dtype=float)
        if amplitudes.shape[1] != len(current_times):
            raise ValueError("amplitudes and current_times must have equal lengths")

        polynomial = self._compute_control_polynomial(dt)
        rotations = np.exp(-1j * np.outer(self._omegas, current_times))
        rotated_amplitudes = amplitudes * rotations
        drive_factors = self._control_monomial_factors(
            rotated_amplitudes,
            polynomial,
        )
        control_values = np.prod(drive_factors, axis=0)
        length = len(self._eigenenergies)
        subpropagators = np.broadcast_to(
            polynomial.free_propagator,
            (len(current_times), length, length),
        ).astype(np.complex128, copy=True)
        subpropagators += np.tensordot(
            control_values,
            polynomial.coefficients,
            axes=(1, 0),
        )
        if not gradient:
            return subpropagators

        dsubprops_dx = np.empty(
            (self._n_drives, len(current_times), length, length),
            dtype=np.complex128,
        )
        dsubprops_dy = np.empty_like(dsubprops_dx)
        for drive_index in range(self._n_drives):
            product_except_drive = np.ones_like(control_values)
            for other_drive_index, factor in enumerate(drive_factors):
                if other_drive_index != drive_index:
                    product_except_drive *= factor

            rotated_amplitude = rotated_amplitudes[drive_index, :, None]
            rotation = rotations[drive_index, :, None]
            positive_counts = polynomial.positive_counts[None, :, drive_index]
            negative_counts = polynomial.negative_counts[None, :, drive_index]
            positive_derivative = (
                positive_counts
                * np.conj(rotated_amplitude)
                ** np.maximum(positive_counts - 1, 0)
                * rotated_amplitude**negative_counts
            )
            negative_derivative = (
                negative_counts
                * np.conj(rotated_amplitude)**positive_counts
                * rotated_amplitude
                ** np.maximum(negative_counts - 1, 0)
            )
            derivative_x = product_except_drive * (
                positive_derivative * np.conj(rotation)
                + negative_derivative * rotation
            )
            derivative_y = 1j * product_except_drive * (
                negative_derivative * rotation
                - positive_derivative * np.conj(rotation)
            )
            dsubprops_dx[drive_index] = np.tensordot(
                derivative_x,
                polynomial.coefficients,
                axes=(1, 0),
            )
            dsubprops_dy[drive_index] = np.tensordot(
                derivative_y,
                polynomial.coefficients,
                axes=(1, 0),
            )
        return subpropagators, dsubprops_dx, dsubprops_dy

    def _compute_envelope_subprops(
        self,
        amplitudes: ArrayLike,
        dt: float,
        t0: float = 0.0,
        *,
        gradient: bool | str = False,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Compute all subpixel propagators for a shaped envelope."""
        amplitudes = self._as_drive_amplitudes(amplitudes)
        current_times = t0 + np.arange(amplitudes.shape[1], dtype=float) * dt
        return self._compute_control_subprops(
            amplitudes,
            current_times,
            dt,
            gradient=gradient,
        )

    def estimate_prepared_envelope_nbytes(self, n_subpixels: int) -> int:
        """Estimate arrays owned by a prepared complex128 envelope.

        Shared solver tensors are excluded, matching
        :attr:`PreparedDysolveEnvelope.nbytes`.
        """
        if n_subpixels <= 0:
            raise ValueError("n_subpixels must be positive")
        monomial_count = sum(
            comb(order + 2 * self._n_drives - 1, 2 * self._n_drives - 1)
            for order in range(1, self.max_order + 1)
        )
        return monomial_count * n_subpixels * np.dtype(np.complex128).itemsize

    def prepare_envelope(self, amplitudes: ArrayLike, dt: float) -> PreparedDysolveEnvelope:
        """Prepare reusable t0-independent control monomials."""
        return PreparedDysolveEnvelope(self, amplitudes, dt)

    def envelope_propagator_vjp(
        self,
        amplitudes: ArrayLike,
        propagator_cotangent: Qobj,
        dt: float,
        t0: float = 0.0,
        *,
        carrier_phases: ArrayLike | None = None,
    ) -> tuple[Qobj, ArrayLike, ArrayLike]:
        """Return a propagator and its reverse-mode envelope gradients.

        The cotangent follows the real scalar convention
        ``dF = real(trace(cotangent.dag() * dU))``. The returned complex
        envelope cotangent satisfies ``dF = real(vdot(cotangent, dA))``.
        Carrier-phase gradients contain one real value per drive.
        """
        amplitudes = self._as_drive_amplitudes(amplitudes)
        phases = self._as_carrier_phases(carrier_phases)
        carrier_factors = np.exp(-1j * phases)[:, None]
        effective_amplitudes = amplitudes * carrier_factors
        length = len(self._eigenenergies)
        step_count = effective_amplitudes.shape[1]
        batch_size = self._fixed_order_batch_size(1)
        batch_starts = tuple(range(0, step_count, batch_size))

        boundary_prefixes = [np.eye(length, dtype=np.complex128)]
        for start in batch_starts:
            stop = min(start + batch_size, step_count)
            subpropagators = self._compute_envelope_subprops(
                effective_amplitudes[:, start:stop],
                dt,
                t0 + start * dt,
            )
            batch_product, _ = self._chronological_product(subpropagators)
            boundary_prefixes.append(batch_product @ boundary_prefixes[-1])

        cotangent = propagator_cotangent.transform(self._basis).full()
        effective_cotangent = np.empty_like(effective_amplitudes, dtype=np.complex128)
        running_suffix = np.eye(length, dtype=np.complex128)
        for batch_index in range(len(batch_starts) - 1, -1, -1):
            start = batch_starts[batch_index]
            stop = min(start + batch_size, step_count)
            subpropagators, derivatives_x, derivatives_y = self._compute_envelope_subprops(
                effective_amplitudes[:, start:stop],
                dt,
                t0 + start * dt,
                gradient=True,
            )
            local_step_count = stop - start
            prefixes = np.empty((local_step_count + 1, length, length), dtype=np.complex128)
            prefixes[0] = boundary_prefixes[batch_index]
            for step_index, subpropagator in enumerate(subpropagators):
                prefixes[step_index + 1] = subpropagator @ prefixes[step_index]

            suffixes = np.empty((local_step_count, length, length), dtype=np.complex128)
            local_suffix = running_suffix
            for step_index in range(local_step_count - 1, -1, -1):
                suffixes[step_index] = local_suffix
                local_suffix = local_suffix @ subpropagators[step_index]

            local_cotangents = np.empty_like(subpropagators)
            for step_index in range(local_step_count):
                local_cotangents[step_index] = (
                    suffixes[step_index].conj().T @ cotangent @ prefixes[step_index].conj().T
                )
            gradients_x = np.real(np.einsum('kij,dkij->dk', local_cotangents.conj(), derivatives_x))
            gradients_y = np.real(np.einsum('kij,dkij->dk', local_cotangents.conj(), derivatives_y))
            effective_cotangent[:, start:stop] = gradients_x + 1j * gradients_y
            running_suffix = local_suffix
        amplitude_cotangent = effective_cotangent * carrier_factors.conj()
        carrier_phase_gradients = np.real(np.sum(np.conj(effective_cotangent) * (-1j * effective_amplitudes), axis=1))

        propagator = Qobj(boundary_prefixes[-1], self._H_0._dims, copy=False).transform(self._basis, True)
        return propagator, amplitude_cotangent, carrier_phase_gradients

    def envelope_parameter_gradients(
        self,
        amplitudes: ArrayLike,
        amplitude_derivatives: ArrayLike,
        dt: float,
        t0: float = 0.0,
        *,
        carrier_phases: ArrayLike | None = None,
        carrier_phase_derivatives: ArrayLike | None = None,
    ):
        """Return an envelope propagator and gradients for arbitrary parameters.

        ``amplitude_derivatives`` contains ``d amplitudes / d parameter``.  For
        one drive it may have shape ``(n_parameters, n_subpixels)``.  For one or
        more drives it may have shape ``(n_parameters, n_drives, n_subpixels)``.
        Complex derivatives are interpreted in the same I/Q convention as
        ``amplitudes``. ``carrier_phase_derivatives`` has shape
        ``(n_parameters, n_drives)`` and applies the chain rule for explicit
        runtime carrier phases.
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

        phases = self._as_carrier_phases(carrier_phases)
        if carrier_phase_derivatives is None:
            phase_derivatives = np.zeros((n_parameters, self._n_drives), dtype=float)
        else:
            phase_derivatives = np.asarray(carrier_phase_derivatives, dtype=float)
            if phase_derivatives.shape != (n_parameters, self._n_drives):
                raise ValueError(
                    "carrier_phase_derivatives must have shape (n_parameters, n_drives)"
                )
        carrier_factors = np.exp(-1j * phases)[:, None]
        derivatives = derivatives * carrier_factors[None] - (
            1j * amplitudes[None] * carrier_factors[None] * phase_derivatives[:, :, None]
        )
        amplitudes = amplitudes * carrier_factors

        length = len(self._eigenenergies)
        total = np.eye(length, dtype=np.complex128)
        parameter_gradients = np.zeros(
            (n_parameters, length, length), dtype=np.complex128
        )
        batch_size = self._fixed_order_batch_size(n_parameters)
        for start in range(0, amplitudes.shape[1], batch_size):
            stop = min(start + batch_size, amplitudes.shape[1])
            subpropagators, dsubprops_dx, dsubprops_dy = self._compute_envelope_subprops(
                amplitudes[:, start:stop],
                dt,
                t0 + start * dt,
                gradient=True,
            )
            batch_derivatives = derivatives[:, :, start:stop]
            subpropagator_derivatives = np.einsum(
                'pdb,dbij->pbij',
                batch_derivatives.real,
                dsubprops_dx,
            ) + np.einsum(
                'pdb,dbij->pbij',
                batch_derivatives.imag,
                dsubprops_dy,
            )
            batch_product, batch_parameter_gradients = self._chronological_product(
                subpropagators,
                subpropagator_derivatives,
            )
            total, parameter_gradients = self._chronological_product(
                np.stack((total, batch_product)),
                np.stack((parameter_gradients, batch_parameter_gradients), axis=1),
            )

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
        carrier_phases: ArrayLike | None = None,
    ) -> Qobj:
        """Return the propagator with explicit runtime carrier phases."""
        amplitudes = self._as_drive_amplitudes(amplitudes)
        amplitudes = amplitudes * np.exp(-1j * self._as_carrier_phases(carrier_phases))[:, None]
        length = len(self._eigenenergies)
        total = np.eye(length, dtype=np.complex128)
        batch_size = self._fixed_order_batch_size()
        for start in range(0, amplitudes.shape[1], batch_size):
            stop = min(start + batch_size, amplitudes.shape[1])
            subpropagators = self._compute_envelope_subprops(
                amplitudes[:, start:stop], dt, t0 + start * dt
            )
            batch_product, _ = self._chronological_product(subpropagators)
            total, _ = self._chronological_product(
                np.stack((total, batch_product))
            )
        return Qobj(total, self._H_0._dims, copy=False).transform(
            self._basis, True
        )

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

        parameter_count = self._n_drives * n_pixels
        derivative_count = parameter_count if gradient == 'real' else 2 * parameter_count
        amplitude_derivatives = np.zeros(
            (derivative_count, self._n_drives, subpixels.shape[1]),
            dtype=np.complex128,
        )
        for drive_index in range(self._n_drives):
            parameter_slice = slice(
                drive_index * n_pixels,
                (drive_index + 1) * n_pixels,
            )
            amplitude_derivatives[parameter_slice, drive_index] = filter_matrix.T
            if gradient == 'quadratures':
                quadrature_slice = slice(
                    parameter_count + drive_index * n_pixels,
                    parameter_count + (drive_index + 1) * n_pixels,
                )
                amplitude_derivatives[quadrature_slice, drive_index] = 1j * filter_matrix.T

        propagator, parameter_gradients = self.envelope_parameter_gradients(
            subpixels,
            amplitude_derivatives,
            subpixel_dt,
            t0,
        )
        real_gradients = tuple(
            parameter_gradients[drive_index * n_pixels:(drive_index + 1) * n_pixels]
            for drive_index in range(self._n_drives)
        )
        if gradient == 'real':
            return propagator, self._format_drive_gradients(real_gradients)
        imaginary_gradients = tuple(
            parameter_gradients[
                parameter_count + drive_index * n_pixels:
                parameter_count + (drive_index + 1) * n_pixels
            ]
            for drive_index in range(self._n_drives)
        )
        return (
            propagator,
            self._format_drive_gradients(real_gradients),
            self._format_drive_gradients(imaginary_gradients),
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
        amplitudes = np.ones(
            (self._n_drives, len(current_times)), dtype=np.complex128
        )
        return self._compute_control_subprops(amplitudes, current_times, dt)

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

import numpy as np
import pytest
from scipy.special import factorial

from qutip import (
    CoreOptions,
    Qobj,
    enr_destroy,
    qeye,
    qeye_like,
    sigmax,
    sigmay,
    sigmaz,
    tensor,
)
from qutip.solver import propagator
from qutip.solver.cy.dysolve import cy_compute_integrals
from qutip.solver.dysolve_propagator import (
    DysolvePropagator,
    dysolve_propagator,
    gaussian_filter_matrix,
)


def _enr_xx():
    a, b = enr_destroy([2, 2], 1)
    return (a + a.dag()) @ (b + b.dag())

def _enr_xz():
    a, b = enr_destroy([2, 2], 1)
    return (a + a.dag()) @ (b.dag() @ b)

def _enr_zz():
    a, b = enr_destroy([2, 2], 1)
    return (a.dag() @ a) @ (b.dag() @ b)


def _qobj_data(obj):
    return obj.full()


def test_envelope_propagator_matches_constant_amplitude_evolution():
    dt = 0.025
    amplitudes = np.ones(5)
    solver = DysolvePropagator(
        0.37 * sigmaz(),
        0.23 * sigmax(),
        1.7,
        options={"max_order": 3, "max_dt": dt, "a_tol": 1e-12},
    )

    shaped = solver.envelope_propagator(amplitudes, dt)
    constant = solver(len(amplitudes) * dt)

    np.testing.assert_allclose(
        _qobj_data(shaped), _qobj_data(constant), rtol=1e-12, atol=1e-12
    )


def test_prepared_envelope_matches_direct_subpropagator_contraction():
    dt = 0.02
    amplitudes = np.array([0.7 + 0.1j, -0.2 + 0.3j, 0.4 - 0.2j])
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options={"max_order": 3, "max_dt": dt, "a_tol": 1e-12},
    )
    prepared = solver.prepare_envelope(amplitudes, dt)

    for t0 in (0.0, 0.13, 1.7):
        np.testing.assert_allclose(
            prepared.subpropagators(t0),
            solver._compute_envelope_subprops(amplitudes, dt, t0),
            rtol=1e-12,
            atol=1e-12,
        )


def test_prepared_envelope_nbytes_matches_owned_arrays():
    amplitudes = np.array([0.7 + 0.1j, -0.2 + 0.3j, 0.4 - 0.2j])
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options={"max_order": 3},
    )

    prepared = solver.prepare_envelope(amplitudes, 0.02)

    assert prepared.nbytes == prepared._control_values.nbytes
    assert prepared.nbytes == solver.estimate_prepared_envelope_nbytes(len(amplitudes))


def test_prepared_envelope_stores_control_monomials():
    amplitudes = np.array(
        [[0.7 + 0.1j, -0.2 + 0.3j], [0.4 - 0.2j, 0.1 + 0.5j]]
    )
    solver = DysolvePropagator.from_drives(
        0.31 * sigmaz(),
        [(0.19 * sigmax(), 1.3), (0.13 * sigmay(), 1.9)],
        options={"max_order": 2},
    )

    prepared = solver.prepare_envelope(amplitudes, 0.02)
    polynomial = prepared._polynomial

    assert polynomial.coefficients.shape == (14, 2, 2)
    assert polynomial.positive_counts.shape == (14, amplitudes.shape[0])
    assert polynomial.negative_counts.shape == (14, amplitudes.shape[0])
    assert prepared._control_values.shape == (amplitudes.shape[1], 14)


def test_control_polynomial_matches_explicit_ordered_branch_sum():
    dt = 0.015
    current_times = np.array([0.07, 0.11, 0.19])
    amplitudes = np.array(
        [
            [0.7 + 0.1j, -0.2 + 0.3j, 0.4 - 0.2j],
            [0.2 - 0.4j, 0.6 + 0.2j, -0.1 + 0.3j],
        ]
    )
    solver = DysolvePropagator.from_drives(
        0.31 * sigmaz(),
        [(0.19 * sigmax(), 1.3), (0.13 * sigmay(), 1.9)],
        options={"max_order": 4},
    )
    Sns = solver._compute_Sns(dt)
    expected = np.broadcast_to(Sns[0], (len(current_times), 2, 2)).astype(
        np.complex128, copy=True
    )
    for order in range(1, solver.max_order + 1):
        drive_indices, signs, omega_vectors = solver._get_branch_metadata(order)
        branch_values = np.ones(
            (len(current_times), len(drive_indices)), dtype=np.complex128
        )
        for position in range(order):
            for drive_index in range(solver._n_drives):
                drive_mask = drive_indices[:, position] == drive_index
                positive = drive_mask & (signs[:, position] > 0)
                negative = drive_mask & (signs[:, position] < 0)
                branch_values[:, positive] *= np.conj(amplitudes[drive_index])[:, None]
                branch_values[:, negative] *= amplitudes[drive_index, :, None]
        branch_phases = np.exp(
            1j * np.outer(current_times, np.sum(omega_vectors, axis=1))
        )
        expected += np.tensordot(
            branch_phases * branch_values,
            Sns[order],
            axes=(1, 0),
        )

    actual = solver._compute_control_subprops(amplitudes, current_times, dt)
    polynomial = solver._compute_control_polynomial(dt)

    assert len(polynomial.coefficients) == 69
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_control_polynomial_gradients_are_finite_at_zero_amplitude():
    dt = 0.015
    current_times = np.array([0.07])
    amplitudes = np.zeros((2, 1), dtype=np.complex128)
    solver = DysolvePropagator.from_drives(
        0.31 * sigmaz(),
        [(0.19 * sigmax(), 1.3), (0.13 * sigmay(), 1.9)],
        options={"max_order": 4},
    )

    _, derivatives_x, derivatives_y = solver._compute_control_subprops(
        amplitudes,
        current_times,
        dt,
        gradient=True,
    )

    assert np.all(np.isfinite(derivatives_x))
    assert np.all(np.isfinite(derivatives_y))
    epsilon = 1e-7
    for drive_index in range(2):
        real_direction = np.zeros_like(amplitudes)
        real_direction[drive_index, 0] = epsilon
        finite_difference_x = (
            solver._compute_control_subprops(
                amplitudes + real_direction, current_times, dt
            )
            - solver._compute_control_subprops(
                amplitudes - real_direction, current_times, dt
            )
        ) / (2 * epsilon)
        imaginary_direction = 1j * real_direction
        finite_difference_y = (
            solver._compute_control_subprops(
                amplitudes + imaginary_direction, current_times, dt
            )
            - solver._compute_control_subprops(
                amplitudes - imaginary_direction, current_times, dt
            )
        ) / (2 * epsilon)
        np.testing.assert_allclose(
            derivatives_x[drive_index, 0],
            finite_difference_x[0],
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            derivatives_y[drive_index, 0],
            finite_difference_y[0],
            rtol=1e-10,
            atol=1e-10,
        )


def test_timestep_tensor_cache_evicts_by_lru_and_recomputes():
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options={"max_order": 1, "dt_cache_size": 2},
    )
    first = solver._compute_Sns(0.01)
    second = solver._compute_Sns(0.02)

    assert solver._compute_Sns(0.01) is first
    solver._compute_Sns(0.03)

    assert len(solver._dt_Sns) <= 2
    assert tuple(solver._dt_Sns) == (0.01, 0.03)
    recomputed_second = solver._compute_Sns(0.02)

    assert recomputed_second is not second
    for order in second:
        np.testing.assert_allclose(recomputed_second[order], second[order])
    assert len(solver._dt_Sns) <= 2


def test_control_polynomial_cache_follows_timestep_tensor_lru():
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options={"max_order": 2, "dt_cache_size": 2},
    )
    first = solver._compute_control_polynomial(0.01)
    second = solver._compute_control_polynomial(0.02)

    assert solver._compute_control_polynomial(0.01) is first
    solver._compute_control_polynomial(0.03)

    assert tuple(solver._dt_Sns) == (0.01, 0.03)
    assert set(solver._control_polynomials) == {0.01, 0.03}
    assert solver._compute_control_polynomial(0.02) is not second
    assert tuple(solver._dt_Sns) == (0.03, 0.02)
    assert set(solver._control_polynomials) == {0.02, 0.03}


def test_zero_timestep_tensor_cache_size_bypasses_storage():
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options={"max_order": 0, "dt_cache_size": 0},
    )

    first = solver._compute_Sns(0.01)
    second = solver._compute_Sns(0.01)

    assert first is not second
    np.testing.assert_allclose(first[0], second[0])
    assert not solver._dt_Sns


@pytest.mark.parametrize("invalid_limit", [-1, 1.5, True])
def test_timestep_tensor_cache_size_must_be_non_negative_integer(invalid_limit):
    with pytest.raises(ValueError, match="dt_cache_size"):
        DysolvePropagator(
            0.31 * sigmaz(),
            0.19 * sigmax(),
            1.3,
            options={"dt_cache_size": invalid_limit},
        )


def test_timestep_tensor_cache_size_defaults_to_64():
    solver = DysolvePropagator(0.31 * sigmaz(), 0.19 * sigmax(), 1.3)

    assert solver.dt_cache_size == 64


def test_batched_envelope_matches_full_prepared_result_with_absolute_t0():
    dt = 0.02
    t0 = 0.17
    amplitudes = np.array([
        0.7 + 0.1j,
        -0.2 + 0.3j,
        0.4 - 0.2j,
        -0.1 - 0.6j,
        0.2 + 0.5j,
    ])
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options={
            "max_order": 3,
            "batch_size": 2,
            "fixed_order_workspace_bytes": 2200,
        },
    )
    prepared = solver.prepare_envelope(amplitudes, dt)
    full_total = np.eye(2, dtype=np.complex128)
    for subpropagator in prepared.subpropagators(t0):
        full_total = subpropagator @ full_total
    full_result = Qobj(full_total, solver._H_0._dims, copy=False).transform(
        solver._basis, True
    )

    batched = solver.envelope_propagator(amplitudes, dt, t0=t0)

    np.testing.assert_allclose(_qobj_data(batched), _qobj_data(full_result), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        _qobj_data(prepared.propagator(t0)), _qobj_data(full_result), rtol=1e-12, atol=1e-12
    )


def test_chronological_product_matches_sequential_reduction_for_odd_batch():
    random = np.random.default_rng(98123)
    matrices = random.normal(size=(7, 3, 3)) + 1j * random.normal(size=(7, 3, 3))
    sequential = np.eye(3, dtype=np.complex128)
    for matrix in matrices:
        sequential = matrix @ sequential

    pairwise, derivatives = DysolvePropagator._chronological_product(matrices)

    np.testing.assert_allclose(pairwise, sequential, rtol=1e-13, atol=1e-13)
    assert derivatives.shape == (0, 3, 3)


def test_chronological_product_reduces_parameter_derivatives():
    random = np.random.default_rng(7124)
    matrices = np.eye(3)[None, :, :] + 0.01 * (
        random.normal(size=(7, 3, 3)) + 1j * random.normal(size=(7, 3, 3))
    )
    derivatives = 0.01 * (
        random.normal(size=(2, 7, 3, 3)) + 1j * random.normal(size=(2, 7, 3, 3))
    )
    sequential = np.eye(3, dtype=np.complex128)
    sequential_derivatives = np.zeros((2, 3, 3), dtype=np.complex128)
    for matrix, matrix_derivatives in zip(matrices, derivatives.transpose(1, 0, 2, 3), strict=True):
        sequential_derivatives = matrix @ sequential_derivatives + matrix_derivatives @ sequential
        sequential = matrix @ sequential

    pairwise, pairwise_derivatives = DysolvePropagator._chronological_product(matrices, derivatives)

    np.testing.assert_allclose(pairwise, sequential, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(pairwise_derivatives, sequential_derivatives, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    "option_name, invalid_value",
    [
        ("fixed_order_workspace_bytes", 0),
        ("fixed_order_workspace_bytes", 1.5),
        ("fixed_order_workspace_bytes", True),
        ("fixed_order_batch_size", 0),
        ("fixed_order_batch_size", 1.5),
        ("fixed_order_batch_size", True),
    ],
)
def test_fixed_order_workspace_options_require_positive_integers(option_name, invalid_value):
    with pytest.raises(ValueError, match=option_name):
        DysolvePropagator(
            0.31 * sigmaz(),
            0.19 * sigmax(),
            1.3,
            options={option_name: invalid_value},
        )


def test_fixed_order_batch_respects_performance_tile_and_workspace():
    tile_limited_solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options={
            "max_order": 3,
            "fixed_order_workspace_bytes": 1024**2,
            "fixed_order_batch_size": 7,
        },
    )
    assert tile_limited_solver._fixed_order_batch_size() == 7
    assert tile_limited_solver._fixed_order_batch_size(3) == 7

    workspace_limited_solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options={"max_order": 3, "fixed_order_workspace_bytes": 10_000},
    )
    assert workspace_limited_solver._fixed_order_batch_size() > 1
    assert workspace_limited_solver._fixed_order_batch_size(3) < workspace_limited_solver._fixed_order_batch_size()

    limited_solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options={"max_order": 3, "fixed_order_workspace_bytes": 1},
    )
    assert limited_solver._fixed_order_batch_size() == 1
    assert limited_solver._fixed_order_batch_size(3) == 1


def test_fixed_order_batch_defaults_to_measured_performance_tile():
    solver = DysolvePropagator(0.31 * sigmaz(), 0.19 * sigmax(), 1.3)

    assert solver.fixed_order_workspace_bytes == 256 * 1024**2
    assert solver.fixed_order_batch_size == 512
    assert solver._fixed_order_batch_size() == 512


def test_multi_drive_fixed_order_batches_match_sequential_reduction():
    dt = 0.015
    t0 = 0.23
    amplitudes = np.array(
        [
            [0.7 + 0.1j, -0.2 + 0.3j, 0.4 - 0.2j, 0.1 + 0.5j, -0.3j],
            [0.2 - 0.4j, 0.6 + 0.2j, -0.1 + 0.3j, 0.5 - 0.2j, 0.4j],
        ]
    )
    solver = DysolvePropagator.from_drives(
        0.31 * sigmaz(),
        [(0.19 * sigmax(), 1.3), (0.13 * sigmay(), 1.9)],
        options={"max_order": 3, "fixed_order_workspace_bytes": 15000},
    )
    sequential = np.eye(2, dtype=np.complex128)
    for subpropagator in solver._compute_envelope_subprops(amplitudes, dt, t0):
        sequential = subpropagator @ sequential
    expected = Qobj(sequential, solver._H_0._dims, copy=False).transform(solver._basis, True)

    streaming = solver.envelope_propagator(amplitudes, dt, t0=t0)
    prepared = solver.prepare_envelope(amplitudes, dt).propagator(t0)

    np.testing.assert_allclose(_qobj_data(streaming), _qobj_data(expected), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_qobj_data(prepared), _qobj_data(expected), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("frequencies", [(1.3, 1.9), (0.0, 1.9)])
def test_prepared_envelope_applies_multi_drive_carrier_phases_at_runtime(frequencies):
    dt = 0.015
    t0 = 0.23
    amplitudes = np.array(
        [
            [0.7 + 0.1j, -0.2 + 0.3j, 0.4 - 0.2j],
            [0.2 - 0.4j, 0.6 + 0.2j, -0.1 + 0.3j],
        ]
    )
    carrier_phases = np.array([0.37, -1.21])
    solver = DysolvePropagator.from_drives(
        0.31 * sigmaz(),
        [(0.19 * sigmax(), frequencies[0]), (0.13 * sigmay(), frequencies[1])],
        options={"max_order": 3},
    )

    expected = solver.envelope_propagator(
        amplitudes * np.exp(-1j * carrier_phases[:, None]),
        dt,
        t0=t0,
    )
    actual = solver.prepare_envelope(amplitudes, dt).propagator(
        t0,
        carrier_phases=carrier_phases,
    )

    np.testing.assert_allclose(_qobj_data(actual), _qobj_data(expected), rtol=1e-12, atol=1e-12)


def test_envelope_parameter_gradients_match_finite_difference():
    dt = 0.02
    t0 = 0.11
    amplitudes = np.array([0.7 + 0.1j, -0.2 + 0.3j, 0.4 - 0.2j])
    derivatives = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, 0.5j, -1.0j],
        ],
        dtype=complex,
    )
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options={"max_order": 3, "max_dt": dt, "a_tol": 1e-12, "batch_size": 2},
    )

    propagator, parameter_gradients = solver.envelope_parameter_gradients(
        amplitudes,
        derivatives,
        dt,
        t0=t0,
    )
    reference_propagator = solver.envelope_propagator(amplitudes, dt, t0=t0)
    np.testing.assert_allclose(_qobj_data(propagator), _qobj_data(reference_propagator), rtol=1e-12, atol=1e-12)

    epsilon = 1e-6
    for derivative, parameter_gradient in zip(derivatives, parameter_gradients, strict=True):
        finite_difference = (
            _qobj_data(solver.envelope_propagator(amplitudes + epsilon * derivative, dt, t0=t0))
            - _qobj_data(solver.envelope_propagator(amplitudes - epsilon * derivative, dt, t0=t0))
        ) / (2 * epsilon)
        np.testing.assert_allclose(_qobj_data(parameter_gradient), finite_difference, rtol=1e-8, atol=1e-8)


def test_envelope_propagator_vjp_matches_forward_parameter_gradients():
    dt = 0.015
    t0 = 0.07
    amplitudes = np.array(
        [[0.7 + 0.1j, -0.2 + 0.3j], [0.4 - 0.2j, 0.1 + 0.5j]]
    )
    amplitude_derivatives = np.array(
        [
            [[0.3 + 0.2j, 0.1 - 0.4j], [0.0, 0.2j]],
            [[0.0, -0.1j], [0.5 + 0.1j, -0.2]],
        ]
    )
    carrier_phases = np.array([0.2, -0.3])
    phase_derivatives = np.array([[0.1, -0.2], [0.0, 0.4]])
    solver = DysolvePropagator.from_drives(
        0.31 * sigmaz(),
        [(0.19 * sigmax(), 1.3), (0.13 * sigmay(), 1.9)],
        options={
            "max_order": 3,
            "max_dt": dt,
            "a_tol": 1e-12,
            "fixed_order_batch_size": 1,
        },
    )
    cotangent = 0.4 * qeye(2) + (0.2 + 0.1j) * sigmax() - 0.3j * sigmay()

    forward_propagator, forward_gradients = solver.envelope_parameter_gradients(
        amplitudes,
        amplitude_derivatives,
        dt,
        t0=t0,
        carrier_phases=carrier_phases,
        carrier_phase_derivatives=phase_derivatives,
    )
    reverse_propagator, amplitude_cotangent, phase_gradients = solver.envelope_propagator_vjp(
        amplitudes,
        cotangent,
        dt,
        t0=t0,
        carrier_phases=carrier_phases,
    )

    np.testing.assert_allclose(_qobj_data(reverse_propagator), _qobj_data(forward_propagator), atol=1e-12)
    for parameter_index, forward_gradient in enumerate(forward_gradients):
        expected = np.real(np.vdot(_qobj_data(cotangent), _qobj_data(forward_gradient)))
        actual = np.real(np.vdot(amplitude_cotangent, amplitude_derivatives[parameter_index]))
        actual += phase_gradients @ phase_derivatives[parameter_index]
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-11)


def test_multi_drive_propagator_matches_qutip_propagator():
    H_0 = 0.37 * sigmaz()
    X_0 = 0.11 * sigmax()
    X_1 = 0.07 * sigmay()
    omega_0 = 1.3
    omega_1 = 2.1
    t = 0.12
    solver = DysolvePropagator.from_drives(
        H_0,
        [(X_0, omega_0), (X_1, omega_1)],
        options={"max_order": 4, "max_dt": 0.02, "a_tol": 1e-12},
    )

    dysolve_U = solver(t)

    def coeff_0(t, omega_0):
        return np.cos(omega_0 * t)

    def coeff_1(t, omega_1):
        return np.cos(omega_1 * t)

    qutip_U = propagator(
        [H_0, [X_0, coeff_0], [X_1, coeff_1]],
        t,
        args={"omega_0": omega_0, "omega_1": omega_1},
        options={"atol": 1e-11, "rtol": 1e-10},
    )
    np.testing.assert_allclose(
        _qobj_data(dysolve_U), _qobj_data(qutip_U), rtol=1e-5, atol=1e-7
    )


def test_multi_drive_envelope_parameter_gradients_match_finite_difference():
    dt = 0.015
    amplitudes = np.array(
        [[0.7 + 0.1j, -0.2 + 0.3j], [0.4 - 0.2j, 0.1 + 0.5j]]
    )
    derivatives = np.array(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0j, 0.0]],
        ]
    )
    solver = DysolvePropagator.from_drives(
        0.31 * sigmaz(),
        [(0.19 * sigmax(), 1.3), (0.13 * sigmay(), 1.9)],
        options={"max_order": 3, "max_dt": dt, "a_tol": 1e-12},
    )

    _, parameter_gradients = solver.envelope_parameter_gradients(amplitudes, derivatives, dt)

    epsilon = 1e-6
    for derivative, parameter_gradient in zip(derivatives, parameter_gradients, strict=True):
        finite_difference = (
            _qobj_data(solver.envelope_propagator(amplitudes + epsilon * derivative, dt))
            - _qobj_data(solver.envelope_propagator(amplitudes - epsilon * derivative, dt))
        ) / (2 * epsilon)
        np.testing.assert_allclose(_qobj_data(parameter_gradient), finite_difference, rtol=1e-8, atol=1e-8)


def test_gaussian_filter_matrix_preserves_constant_envelope():
    matrix = gaussian_filter_matrix(
        n_pixels=4,
        subpixels_per_pixel=3,
        pixel_dt=0.2,
        bandwidth=20.0,
    )
    assert matrix.shape == (12, 4)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(matrix @ np.ones(4), np.ones(12))


def test_filtered_envelope_gradient_matches_finite_difference():
    pixels = np.array([0.7 + 0.1j, -0.2 + 0.3j, 0.4 - 0.2j])
    pixel_dt = 0.02
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options={"max_order": 3, "max_dt": pixel_dt / 2, "a_tol": 1e-12},
    )

    _, gradients_x, gradients_y = solver.filtered_envelope_propagator(
        pixels,
        pixel_dt,
        subpixels_per_pixel=2,
        bandwidth=80.0,
        gradient="quadratures",
    )
    eps = 1e-6
    index = 1
    plus = pixels.copy(); plus[index] += eps
    minus = pixels.copy(); minus[index] -= eps
    finite_difference_x = (
        _qobj_data(
            solver.filtered_envelope_propagator(
                plus, pixel_dt, 2, 80.0
            )
        )
        - _qobj_data(
            solver.filtered_envelope_propagator(
                minus, pixel_dt, 2, 80.0
            )
        )
    ) / (2 * eps)
    np.testing.assert_allclose(
        _qobj_data(gradients_x[index]), finite_difference_x, rtol=1e-8, atol=1e-8
    )

    plus = pixels.copy(); plus[index] += 1j * eps
    minus = pixels.copy(); minus[index] -= 1j * eps
    finite_difference_y = (
        _qobj_data(
            solver.filtered_envelope_propagator(
                plus, pixel_dt, 2, 80.0
            )
        )
        - _qobj_data(
            solver.filtered_envelope_propagator(
                minus, pixel_dt, 2, 80.0
            )
        )
    ) / (2 * eps)
    np.testing.assert_allclose(
        _qobj_data(gradients_y[index]), finite_difference_y, rtol=1e-8, atol=1e-8
    )


def test_multi_drive_filtered_gradient_matches_finite_difference():
    pixels = np.array(
        [[0.7 + 0.1j, -0.2 + 0.3j], [0.4 - 0.2j, 0.1 + 0.5j]]
    )
    pixel_dt = 0.02
    solver = DysolvePropagator.from_drives(
        0.31 * sigmaz(),
        [(0.19 * sigmax(), 1.3), (0.13 * sigmay(), 1.9)],
        options={"max_order": 3, "max_dt": pixel_dt / 2, "a_tol": 1e-12},
    )

    _, gradients_x, gradients_y = solver.filtered_envelope_propagator(
        pixels,
        pixel_dt,
        subpixels_per_pixel=2,
        bandwidth=80.0,
        gradient="quadratures",
    )
    eps = 1e-6
    drive_index = 1
    pixel_index = 0
    plus = pixels.copy(); plus[drive_index, pixel_index] += eps
    minus = pixels.copy(); minus[drive_index, pixel_index] -= eps
    finite_difference_x = (
        _qobj_data(solver.filtered_envelope_propagator(plus, pixel_dt, 2, 80.0))
        - _qobj_data(solver.filtered_envelope_propagator(minus, pixel_dt, 2, 80.0))
    ) / (2 * eps)
    np.testing.assert_allclose(
        _qobj_data(gradients_x[drive_index][pixel_index]),
        finite_difference_x,
        rtol=1e-8,
        atol=1e-8,
    )

    plus = pixels.copy(); plus[drive_index, pixel_index] += 1j * eps
    minus = pixels.copy(); minus[drive_index, pixel_index] -= 1j * eps
    finite_difference_y = (
        _qobj_data(solver.filtered_envelope_propagator(plus, pixel_dt, 2, 80.0))
        - _qobj_data(solver.filtered_envelope_propagator(minus, pixel_dt, 2, 80.0))
    ) / (2 * eps)
    np.testing.assert_allclose(
        _qobj_data(gradients_y[drive_index][pixel_index]),
        finite_difference_y,
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.parametrize("eff_omega", [-10.0, -1.0, -0.1, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("dt", [-10.0, -1.0, -0.1, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("ws, answer", [
    # First part of tuple is "ws", second part is "answer"
    (
        np.array([0.0]),
        lambda _, dt: dt
    ),
    (
        np.array([1e-12]),
        lambda _, dt: dt
    ),
    (
        lambda eff_omega: np.array([eff_omega]),
        lambda eff_omega, dt: (-1j/eff_omega) * (np.exp(1j*eff_omega*dt) - 1)
    ),
    (
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        lambda _, dt: (dt**5) / factorial(5)
    ),
    (
        np.array([1e-12, 1e-12, 1e-12]),
        lambda _, dt: (dt**3) / factorial(3)
    ),
    (
        lambda eff_omega: np.array([eff_omega, 0.0]),
        lambda eff_omega, dt: (-1j/eff_omega) * (
            (-1j/eff_omega) * (np.exp(1j*eff_omega*dt) - 1) - dt
        )
    ),
    (
        lambda eff_omega: np.array([0.0, eff_omega]),
        lambda eff_omega, dt: (-1j*dt/eff_omega) * np.exp(1j*eff_omega*dt) -
        ((1j/eff_omega)**2) * (np.exp(1j*eff_omega*dt)-1))
])
def test_integrals_1(eff_omega, dt, ws, answer):
    if callable(ws):
        ws = ws(eff_omega)
    if callable(answer):
        answer = answer(eff_omega, dt)

    integrals = cy_compute_integrals(ws, dt)

    assert np.isclose(integrals, answer, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("eff_omega_1", [-25.0, -5.0, -0.5, 0.5, 5.0, 25.0])
@pytest.mark.parametrize("eff_omega_2", [-25.0, -5.0, -0.5, 0.5, 5.0, 25.0])
@pytest.mark.parametrize("dt", [-10.0, -1.0, -0.1, 0.1, 1.0, 10.0])
def test_integrals_2(eff_omega_1, eff_omega_2, dt):
    ws = np.array([eff_omega_1, eff_omega_2])
    integrals = cy_compute_integrals(ws, dt)

    if eff_omega_1 + eff_omega_2 == 0:
        answer = (-1j*dt/eff_omega_1) + \
            (np.exp(1j*eff_omega_2*dt)-1)/(eff_omega_1*eff_omega_2)
    else:
        exp_1 = np.exp(1j*(eff_omega_1+eff_omega_2)*dt)
        exp_2 = np.exp(1j*eff_omega_2*dt)
        answer = -(exp_1-1)/(eff_omega_1*(eff_omega_1+eff_omega_2)) + \
            (exp_2-1)/(eff_omega_1*eff_omega_2)

    assert np.isclose(integrals, answer, rtol=1e-10, atol=1e-10)



@pytest.mark.parametrize("H_0", [
    sigmaz(), sigmay(), sigmaz(), qeye(2), tensor(sigmax(), sigmaz()),
    tensor(sigmax(), sigmaz()) + tensor(qeye(2), sigmay()),
    _enr_xz()
])
@pytest.mark.parametrize("t_i, t_f", [
    (0, 0.1), (0, 0.5), (0, 1), (0, 10), (0, -1),
    (-0.1, 0.1), (-0.5, 0.5), (-1, 1), (-10, 10), (1, -1)
])
def test_zeroth_order(H_0, t_i, t_f):
    # self.X and self.omega don't matter
    dysolve = DysolvePropagator(
        H_0, qeye_like(H_0), 0, options={'max_order': 0}
    )
    U = dysolve(t_f, t_i)

    exp = (-1j*H_0*(t_f - t_i)).expm()

    with CoreOptions(atol=1e-10, rtol=1e-10):
        assert U == exp


@pytest.mark.parametrize("t_i, t_f", [
    (0, 0.001), (0.001, 0), (-0.001, 0.001)
])
def test_zeroth_order_shorter_than_max_dt(t_i, t_f):
    H_0 = tensor(sigmax(), sigmaz())
    dysolve = DysolvePropagator(
        H_0, qeye_like(H_0), 0,
        options={'max_order': 0, 'max_dt': 0.1}
    )
    U = dysolve(t_f, t_i)

    exp = (-1j*H_0*(t_f - t_i)).expm()

    with CoreOptions(atol=1e-10, rtol=1e-10):
        assert U == exp


def test_integral_a_tol_option_is_used(monkeypatch):
    seen_a_tols = []

    def fake_compute_Sn(
        omega_vectors, ket_bra_idx, diff_lambdas, matrix_elements,
        dt, length, a_tol=1e-10
    ):
        seen_a_tols.append(a_tol)
        return np.zeros((len(omega_vectors), length, length), dtype=complex)

    monkeypatch.setitem(
        DysolvePropagator._compute_Sns.__globals__,
        "cy_compute_Sn",
        fake_compute_Sn,
    )
    dysolve = DysolvePropagator(
        sigmaz(), sigmax(), 1,
        options={'max_order': 1, 'a_tol': 1e-4}
    )
    dysolve._compute_Sns(0.1)

    assert seen_a_tols
    assert set(seen_a_tols) == {1e-4}


@pytest.mark.parametrize("H_0", [sigmax(), sigmay(), sigmaz()])
@pytest.mark.parametrize("X", [sigmax(), sigmay(), sigmaz()])
@pytest.mark.parametrize("t", [-0.15, -0.1, 0, 0.1, 0.15])
@pytest.mark.parametrize("omega", [0, 1, 10])
def test_2x2_propagators_single_time(H_0, X, t, omega):
    # Dysolve
    options = {'max_order': 3, 'max_dt': 0.05}
    U = dysolve_propagator(H_0, X, omega, t, options=options)

    # Qutip.solver.propagator
    def H1_coeff(t, omega):
        return np.cos(omega * t)

    H = [H_0, [X, H1_coeff]]
    args = {'omega': omega}
    prop = propagator(
        H, t, args=args, options={"atol": 1e-10, "rtol": 1e-8}
    )

    with CoreOptions(atol=1e-10, rtol=1e-6):
        assert U == prop


@pytest.mark.parametrize("H_0", [sigmay(), sigmaz()])
@pytest.mark.parametrize("X", [sigmay(), sigmaz()])
@pytest.mark.parametrize("ts", [
    [0, 0.25, 0.5],
    [0, -0.25, -0.5],
    [-0.1, 0, 0.1]
])
@pytest.mark.parametrize("omega", [0, 10])
def test_2x2_propagators_list_times(H_0, X, ts, omega):
    options = {'max_order': 3, 'max_dt': 0.01}
    Us = dysolve_propagator(H_0, X, omega, ts, options=options)

    # Qutip.solver.propagator
    def H1_coeff(t, omega):
        return np.cos(omega * t)

    H = [H_0, [X, H1_coeff]]
    args = {'omega': omega}
    props = propagator(
        H, ts, args=args, options={"atol": 1e-10, "rtol": 1e-8}
    )

    with CoreOptions(atol=1e-10, rtol=1e-6):
        assert Us == props


@pytest.mark.parametrize("H_0", [
    tensor(sigmax(), sigmaz()) + tensor(qeye(2), sigmay()),
    tensor(sigmaz(), qeye(2))
])
@pytest.mark.parametrize("X", [
    tensor(qeye(2), sigmaz()),
    tensor(sigmaz(), sigmax()) + tensor(sigmay(), qeye(2))
])
@pytest.mark.parametrize("omega", [
    5, 10
])
@pytest.mark.parametrize("t_f", [
    1, -1
])
def test_4x4_propagators_single_time(H_0, X, omega, t_f):
    options = {'max_order': 3, 'max_dt': 0.01}
    U = dysolve_propagator(H_0, X, omega, t_f, options=options)

    # Qutip.solver.propagator
    def H1_coeff(t, omega):
        return np.cos(omega * t)

    H = [H_0, [X, H1_coeff]]
    args = {'omega': omega}
    prop = propagator(
        H, t_f, args=args, options={"atol": 1e-10, "rtol": 1e-8}
    )

    with CoreOptions(atol=1e-10, rtol=1e-5):
        assert U == prop


@pytest.mark.parametrize("H_0", [
    tensor(sigmax(), sigmaz()) + tensor(qeye(2), sigmay()),
    tensor(sigmaz(), qeye(2))
])
@pytest.mark.parametrize("X", [
    tensor(qeye(2), sigmaz()),
    tensor(sigmaz(), sigmax()) + tensor(sigmay(), qeye(2))
])
@pytest.mark.parametrize("omega", [
    0, 10
])
@pytest.mark.parametrize("ts", [
    [0, 0.25, 0.5],
    [0, -0.25, -0.5],
    [-0.1, 0, 0.1]
])
def test_4x4_propagators_list_times(H_0, X, omega, ts):
    options = {'max_order': 3, 'max_dt': 0.01}
    Us = dysolve_propagator(H_0, X, omega, ts, options=options)

    # Qutip.solver.propagator
    def H1_coeff(t, omega):
        return np.cos(omega * t)

    H = [H_0, [X, H1_coeff]]
    args = {'omega': omega}
    props = propagator(
        H, ts, args=args, options={"atol": 1e-10, "rtol": 1e-8}
    )

    with CoreOptions(atol=1e-10, rtol=1e-6):
        assert Us == props


@pytest.mark.parametrize("omega", [5, 10])
@pytest.mark.parametrize("t_f", [1, -1])
def test_enr_propagators_single_time(omega, t_f):
    # reuses other test with both H_0 and X set to enr space operators
    H_0 = _enr_zz()
    X = _enr_xz()
    test_4x4_propagators_single_time(H_0, X, omega, t_f)


@pytest.mark.parametrize("omega", [0, 10])
@pytest.mark.parametrize("ts", [
    [0, 0.25, 0.5],
    [0, -0.25, -0.5],
    [-0.1, 0, 0.1]
])
def test_enr_propagators_list_times(omega, ts):
    # reuses other test with both H_0 and X set to enr space operators
    H_0 = _enr_zz()
    X = _enr_xz()
    test_4x4_propagators_list_times(H_0, X, omega, ts)



@pytest.mark.parametrize("H_0, X", [
    (
        sigmaz(), sigmax(),
    ),
    (
        tensor(sigmaz(), sigmaz()), tensor(sigmax(), sigmax()),
    ),
    (
        tensor(sigmaz(), sigmaz(), sigmaz()),
        tensor(sigmax(), sigmax(), sigmax())
    ),
    (
        tensor(sigmaz(), sigmaz(), sigmaz(), sigmaz()),
        tensor(sigmax(), sigmax(), sigmax(), sigmax())
    ),
    (
        _enr_zz(), _enr_xx()
    )
])
def test_dims(H_0, X):
    dysolve = DysolvePropagator(
        H_0, X, 1, {'max_order': 0}
    )
    U = dysolve(0.001)
    assert (dysolve._H_0.dims == dysolve._X.dims == U.dims)

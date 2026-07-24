from qutip.solver.dysolve_propagator import (
    DysolvePropagator,
    dysolve_propagator,
    gaussian_filter_matrix,
)
from qutip.solver import propagator
from qutip.solver.cy.dysolve import (
    cy_compute_integrals,
    cy_compute_integral_frequency_derivatives,
    cy_compute_Sn,
    cy_compute_Sn_frequency_derivatives,
    cy_compute_Sn_frequency_derivative_subsets,
)
from qutip import (
    CoreOptions, Qobj, sigmax, sigmay, sigmaz, qeye,
    qeye_like, tensor, enr_destroy,
)
from scipy.linalg import expm_frechet
from scipy.special import factorial
import numpy as np
import pytest


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


def _adaptive_options(
    min_timestep=0.001,
    *,
    max_order=3,
    error_tolerance_per_time=1e-6,
    a_tol=1e-12,
):
    return {
        "max_order": max_order,
        "min_timestep": min_timestep,
        "error_tolerance_per_time": error_tolerance_per_time,
        "a_tol": a_tol,
    }


def test_prepared_envelope_matches_direct_subpropagator_contraction():
    dt = 0.02
    amplitudes = np.array([0.7 + 0.1j, -0.2 + 0.3j, 0.4 - 0.2j])
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options=_adaptive_options(dt),
    )
    prepared = solver.prepare_envelope(amplitudes, dt)

    for t0 in (0.0, 0.13, 1.7):
        np.testing.assert_allclose(
            prepared.subpropagators(t0),
            solver._compute_envelope_subprops(amplitudes, dt, t0),
            rtol=1e-12,
            atol=1e-12,
        )



def test_linear_envelope_subpropagator_matches_constant_envelope():
    dt = 0.04
    t0 = 0.17
    amplitude = 0.7 + 0.2j
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options=_adaptive_options(dt, max_order=3),
    )

    linear = solver._compute_linear_envelope_subpropagator(
        amplitude,
        amplitude,
        dt,
        t0,
        max_order=3,
    )
    constant = solver._compute_envelope_subprops(
        np.array([amplitude]),
        dt,
        t0,
        max_order=3,
    )[0]

    np.testing.assert_allclose(linear, constant, rtol=1e-13, atol=1e-13)


def test_linear_envelope_subpropagator_improves_smooth_drive_accuracy():
    H_0 = 0.31 * sigmaz()
    drive_operator = 0.19 * sigmax()
    omega = 1.3
    dt = 0.3
    start_amplitude = 0.1 + 0.05j
    end_amplitude = 0.9 - 0.25j
    solver = DysolvePropagator(
        H_0,
        drive_operator,
        omega,
        options=_adaptive_options(dt, max_order=4),
    )

    linear_data = solver._compute_linear_envelope_subpropagator(
        start_amplitude,
        end_amplitude,
        dt,
        max_order=4,
    )
    midpoint_data = solver._compute_envelope_subprops(
        np.array([(start_amplitude + end_amplitude) / 2]),
        dt,
        max_order=4,
    )[0]
    linear = Qobj(linear_data, H_0.dims, copy=False).transform(
        solver._basis, True
    )
    midpoint = Qobj(midpoint_data, H_0.dims, copy=False).transform(
        solver._basis, True
    )

    def coefficient(time, _=None):
        amplitude = start_amplitude + (end_amplitude - start_amplitude) * time / dt
        return amplitude.real * np.cos(omega * time) + amplitude.imag * np.sin(omega * time)

    reference = propagator(
        [H_0, [drive_operator, coefficient]],
        dt,
        args={},
        options={"atol": 1e-13, "rtol": 1e-13},
    )
    linear_error = np.linalg.norm(_qobj_data(linear - reference))
    midpoint_error = np.linalg.norm(_qobj_data(midpoint - reference))

    assert linear_error < 1e-7
    assert linear_error < midpoint_error / 100


def test_linear_envelope_endpoint_gradients_match_finite_difference():
    dt = 0.04
    start_amplitudes = np.array([0.1 + 0.05j, -0.2 + 0.1j])
    end_amplitudes = np.array([0.9 - 0.25j, 0.3 + 0.2j])
    solver = DysolvePropagator.from_drives(
        0.31 * sigmaz(),
        [(0.19 * sigmax(), 1.3), (0.13 * sigmay(), 1.9)],
        options=_adaptive_options(dt, max_order=3),
    )

    _, *endpoint_gradients = solver._compute_linear_envelope_subpropagator(
        start_amplitudes,
        end_amplitudes,
        dt,
        0.17,
        max_order=3,
        gradient=True,
    )
    epsilon = 1e-6
    for endpoint_index in range(2):
        for quadrature_index, direction in enumerate((1.0, 1j)):
            for drive_index in range(2):
                plus = [start_amplitudes.copy(), end_amplitudes.copy()]
                minus = [start_amplitudes.copy(), end_amplitudes.copy()]
                plus[endpoint_index][drive_index] += epsilon * direction
                minus[endpoint_index][drive_index] -= epsilon * direction
                finite_difference = (
                    solver._compute_linear_envelope_subpropagator(
                        *plus, dt, 0.17, max_order=3
                    )
                    - solver._compute_linear_envelope_subpropagator(
                        *minus, dt, 0.17, max_order=3
                    )
                ) / (2 * epsilon)
                actual = endpoint_gradients[
                    2 * endpoint_index + quadrature_index
                ][drive_index]
                np.testing.assert_allclose(
                    actual, finite_difference, rtol=1e-8, atol=2e-9
                )


def test_envelope_propagator_real_gradient_matches_finite_difference():
    dt = 0.02
    amplitudes = np.array([0.7, -0.2, 0.4])
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options=_adaptive_options(dt),
    )

    _, gradients = solver.envelope_propagator(amplitudes, dt, gradient="real")
    eps = 1e-6
    for index, gradient in enumerate(gradients):
        plus = amplitudes.copy()
        minus = amplitudes.copy()
        plus[index] += eps
        minus[index] -= eps
        finite_difference = (
            _qobj_data(solver.envelope_propagator(plus, dt))
            - _qobj_data(solver.envelope_propagator(minus, dt))
        ) / (2 * eps)
        np.testing.assert_allclose(
            _qobj_data(gradient), finite_difference, rtol=1e-8, atol=1e-8
        )


def test_envelope_propagator_quadrature_gradients_match_finite_difference():
    dt = 0.02
    amplitudes = np.array([0.7 + 0.1j, -0.2 + 0.3j, 0.4 - 0.2j])
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options=_adaptive_options(dt),
    )

    _, gradients_x, gradients_y = solver.envelope_propagator(
        amplitudes, dt, gradient="quadratures"
    )
    eps = 1e-6
    for index, (gradient_x, gradient_y) in enumerate(
        zip(gradients_x, gradients_y, strict=True)
    ):
        plus = amplitudes.copy()
        minus = amplitudes.copy()
        plus[index] += eps
        minus[index] -= eps
        finite_difference_x = (
            _qobj_data(solver.envelope_propagator(plus, dt))
            - _qobj_data(solver.envelope_propagator(minus, dt))
        ) / (2 * eps)
        np.testing.assert_allclose(
            _qobj_data(gradient_x), finite_difference_x, rtol=1e-8, atol=1e-8
        )

        plus = amplitudes.copy()
        minus = amplitudes.copy()
        plus[index] += 1j * eps
        minus[index] -= 1j * eps
        finite_difference_y = (
            _qobj_data(solver.envelope_propagator(plus, dt))
            - _qobj_data(solver.envelope_propagator(minus, dt))
        ) / (2 * eps)
        np.testing.assert_allclose(
            _qobj_data(gradient_y), finite_difference_y, rtol=1e-8, atol=1e-8
        )



def test_envelope_parameter_gradients_match_quadrature_contraction():
    dt = 0.02
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
        options=_adaptive_options(dt),
    )

    propagator, parameter_gradients = solver.envelope_parameter_gradients(
        amplitudes,
        derivatives,
        dt,
    )
    reference_propagator, gradients_x, gradients_y = solver.envelope_propagator(
        amplitudes,
        dt,
        gradient="quadratures",
    )
    expected = []
    for derivative in derivatives:
        gradient = 0 * reference_propagator
        for sample_derivative, gradient_x, gradient_y in zip(
            derivative,
            gradients_x,
            gradients_y,
            strict=True,
        ):
            gradient += sample_derivative.real * gradient_x + sample_derivative.imag * gradient_y
        expected.append(gradient)

    np.testing.assert_allclose(_qobj_data(propagator), _qobj_data(reference_propagator), rtol=1e-12, atol=1e-12)
    for actual, expected_gradient in zip(parameter_gradients, expected, strict=True):
        np.testing.assert_allclose(_qobj_data(actual), _qobj_data(expected_gradient), rtol=1e-12, atol=1e-12)



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
        options=_adaptive_options(0.001, max_order=4),
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


def test_multi_drive_envelope_gradients_match_finite_difference():
    dt = 0.015
    amplitudes = np.array(
        [[0.7 + 0.1j, -0.2 + 0.3j], [0.4 - 0.2j, 0.1 + 0.5j]]
    )
    solver = DysolvePropagator.from_drives(
        0.31 * sigmaz(),
        [(0.19 * sigmax(), 1.3), (0.13 * sigmay(), 1.9)],
        options=_adaptive_options(dt),
    )

    _, gradients_x, gradients_y = solver.envelope_propagator(
        amplitudes, dt, gradient="quadratures"
    )
    eps = 1e-6
    for drive_index, pixel_index in [(0, 1), (1, 0)]:
        plus = amplitudes.copy()
        minus = amplitudes.copy()
        plus[drive_index, pixel_index] += eps
        minus[drive_index, pixel_index] -= eps
        finite_difference_x = (
            _qobj_data(solver.envelope_propagator(plus, dt))
            - _qobj_data(solver.envelope_propagator(minus, dt))
        ) / (2 * eps)
        np.testing.assert_allclose(
            _qobj_data(gradients_x[drive_index][pixel_index]),
            finite_difference_x,
            rtol=1e-8,
            atol=1e-8,
        )

        plus = amplitudes.copy()
        minus = amplitudes.copy()
        plus[drive_index, pixel_index] += 1j * eps
        minus[drive_index, pixel_index] -= 1j * eps
        finite_difference_y = (
            _qobj_data(solver.envelope_propagator(plus, dt))
            - _qobj_data(solver.envelope_propagator(minus, dt))
        ) / (2 * eps)
        np.testing.assert_allclose(
            _qobj_data(gradients_y[drive_index][pixel_index]),
            finite_difference_y,
            rtol=1e-8,
            atol=1e-8,
        )


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
        options=_adaptive_options(pixel_dt / 2),
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
        options=_adaptive_options(pixel_dt / 2),
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


@pytest.mark.parametrize(
    "frequencies",
    [
        np.array([0.7]),
        np.array([0.7, 1.2]),
        np.array([0.7, 1.2, -0.3]),
    ],
)
def test_integral_frequency_derivatives_match_finite_difference(frequencies):
    dt = 0.4
    actual = cy_compute_integral_frequency_derivatives(frequencies, dt, 1e-12)
    expected = []
    epsilon = 1e-6
    for frequency_index in range(len(frequencies)):
        plus = frequencies.copy()
        minus = frequencies.copy()
        plus[frequency_index] += epsilon
        minus[frequency_index] -= epsilon
        expected.append(
            (
                cy_compute_integrals(plus, dt, 1e-12)
                - cy_compute_integrals(minus, dt, 1e-12)
            )
            / (2 * epsilon)
        )

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize(
    "frequencies",
    [
        np.array([0.0, 1.2, -0.3]),
        np.array([0.7, 0.0, -0.3]),
        np.array([0.0, 0.0]),
    ],
)
def test_integral_frequency_derivatives_handle_degenerate_frequencies(frequencies):
    dt = 0.4
    actual = cy_compute_integral_frequency_derivatives(frequencies, dt, 1e-12)
    cumulative_frequencies = np.cumsum(frequencies[::-1])[::-1]
    generator = np.diag(np.append(1j * cumulative_frequencies, 0.0))
    generator += np.diag(np.ones(len(frequencies)), 1)
    expected = []
    for frequency_index in range(len(frequencies)):
        generator_derivative = np.zeros_like(generator)
        diagonal_indices = np.arange(frequency_index + 1)
        generator_derivative[diagonal_indices, diagonal_indices] = 1j
        expected.append(
            expm_frechet(
                dt * generator,
                dt * generator_derivative,
                compute_expm=False,
            )[0, -1]
        )

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_Sn_frequency_derivatives_match_finite_difference():
    omega_vectors = np.array([[0.7, 1.2], [-0.7, 1.2]])
    ket_bra_indices = np.array([[0, 1], [1, 0]], dtype=int)
    diff_lambdas = np.array([[0.11, -0.23], [-0.19, 0.31]])
    matrix_elements = np.array([0.4 + 0.2j, -0.1 + 0.3j])
    dt = 0.4
    actual = cy_compute_Sn_frequency_derivatives(
        omega_vectors,
        ket_bra_indices,
        diff_lambdas,
        matrix_elements,
        dt,
        2,
        1e-12,
    )
    epsilon = 1e-6
    for frequency_index in range(omega_vectors.shape[1]):
        plus = omega_vectors.copy()
        minus = omega_vectors.copy()
        plus[:, frequency_index] += epsilon
        minus[:, frequency_index] -= epsilon
        expected = (
            cy_compute_Sn(
                plus,
                ket_bra_indices,
                diff_lambdas,
                matrix_elements,
                dt,
                2,
                1e-12,
            )
            - cy_compute_Sn(
                minus,
                ket_bra_indices,
                diff_lambdas,
                matrix_elements,
                dt,
                2,
                1e-12,
            )
        ) / (2 * epsilon)
        np.testing.assert_allclose(
            actual[:, frequency_index], expected, rtol=2e-6, atol=1e-9
        )


def test_Sn_frequency_derivative_subsets_include_base_and_singletons():
    omega_vectors = np.array([[0.7, 1.2], [-0.7, 1.2]])
    ket_bra_indices = np.array([[0, 1], [1, 0]], dtype=int)
    diff_lambdas = np.array([[0.11, -0.23], [-0.19, 0.31]])
    matrix_elements = np.array([0.4 + 0.2j, -0.1 + 0.3j])
    dt = 0.4
    subsets = cy_compute_Sn_frequency_derivative_subsets(
        omega_vectors,
        ket_bra_indices,
        diff_lambdas,
        matrix_elements,
        dt,
        2,
        1e-12,
    )
    base = cy_compute_Sn(
        omega_vectors,
        ket_bra_indices,
        diff_lambdas,
        matrix_elements,
        dt,
        2,
        1e-12,
    )
    first_derivatives = cy_compute_Sn_frequency_derivatives(
        omega_vectors,
        ket_bra_indices,
        diff_lambdas,
        matrix_elements,
        dt,
        2,
        1e-12,
    )

    np.testing.assert_allclose(subsets[:, 0], base, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        subsets[:, 1], first_derivatives[:, 0], rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        subsets[:, 2], first_derivatives[:, 1], rtol=1e-13, atol=1e-13
    )


def test_Sn_frequency_derivative_subsets_handle_zero_frequencies():
    dt = 0.3
    subsets = cy_compute_Sn_frequency_derivative_subsets(
        np.zeros((1, 2)),
        np.array([[0, 0]], dtype=int),
        np.zeros((1, 2)),
        np.ones(1, dtype=complex),
        dt,
        1,
        1e-12,
    )

    np.testing.assert_allclose(
        subsets[0, :, 0, 0],
        [dt**2 / 2, 1j * dt**3 / 6, 1j * dt**3 / 3, -(dt**4) / 8],
        rtol=1e-13,
        atol=1e-13,
    )


def test_frequency_derivative_tensors_are_cached_lazily():
    solver = DysolvePropagator(
        0.31 * sigmaz(),
        0.19 * sigmax(),
        1.3,
        options=_adaptive_options(0.02, max_order=3),
    )

    derivatives = solver._compute_Sn_frequency_derivatives(0.02, 2)

    assert derivatives.shape == (4, 2, 2, 2)
    assert solver._compute_Sn_frequency_derivatives(0.02, 2) is derivatives
    assert set(solver._dt_Sns[0.02]) == {0}
    assert set(solver._dt_Sn_frequency_derivatives[0.02]) == {2}


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
def test_short_zeroth_order_interval(t_i, t_f):
    H_0 = tensor(sigmax(), sigmaz())
    dysolve = DysolvePropagator(
        H_0, qeye_like(H_0), 0,
        options={'max_order': 0}
    )
    U = dysolve(t_f, t_i)

    exp = (-1j*H_0*(t_f - t_i)).expm()

    with CoreOptions(atol=1e-10, rtol=1e-10):
        assert U == exp


@pytest.mark.parametrize("options, message", [
    ({}, "min_timestep and error_tolerance_per_time are required"),
    ({'min_timestep': 0.01}, "min_timestep and error_tolerance_per_time are required"),
    ({'error_tolerance_per_time': 1e-5}, "min_timestep and error_tolerance_per_time are required"),
    ({'max_dt': 0.1}, "max_dt is no longer supported"),
    (
        {
            'min_timestep': 0.0,
            'error_tolerance_per_time': 1e-5,
        },
        "min_timestep must be positive",
    ),
    (
        {
            'min_timestep': 0.01,
            'error_tolerance_per_time': 0.0,
        },
        "error_tolerance_per_time must be positive",
    ),
])
def test_adaptive_options_are_validated(options, message):
    with pytest.raises(ValueError, match=message):
        DysolvePropagator(sigmaz(), sigmax(), 1, options=options)


def test_adaptive_order_escalates_by_piece_count(monkeypatch):
    dysolve = DysolvePropagator(
        sigmaz(), sigmax(), 1,
        options={
            'min_timestep': 1.0,
            'error_tolerance_per_time': 1.0,
            'max_order': 4,
        },
    )
    largest_ticks_by_order = {1: 1, 2: 4, 3: 16, 4: 64}

    def controlled_error_rate(
        self, current_time, dt, order, step_propagator=None
    ):
        del current_time, step_propagator
        tick = round(abs(dt) / self.min_timestep)
        return 0.0 if tick <= largest_ticks_by_order[order] else 2.0

    monkeypatch.setattr(
        DysolvePropagator,
        "_adaptive_split_error_rate",
        controlled_error_rate,
    )

    order, step_ticks = dysolve._adaptive_interval_plan(0.0, 1200.0)

    assert order == 3
    assert len(step_ticks) == 75
    assert set(step_ticks) == {16}


def test_adaptive_envelope_piece_count_does_not_escalate_dyson_order(
    monkeypatch,
):
    dysolve = DysolvePropagator(
        sigmaz(),
        sigmax(),
        1,
        options={
            'min_timestep': 1.0,
            'error_tolerance_per_time': 1.0,
            'max_order': 4,
        },
    )
    order_checks = []

    def interpolation_error_rate(
        self, current_time, dt, order, step_propagator=None
    ):
        del self, current_time, order, step_propagator
        return 0.0 if abs(dt) <= 1.0 else 2.0

    def dyson_error_rate(current_time, dt, order):
        del current_time, dt
        order_checks.append(order)
        return 0.0

    monkeypatch.setattr(
        DysolvePropagator,
        "_adaptive_split_error_rate",
        interpolation_error_rate,
    )
    order, step_ticks = dysolve._adaptive_interval_plan(
        0.0,
        1001.0,
        lambda current_time, dt, order: np.eye(2),
        order_error_rate=dyson_error_rate,
    )

    assert order == 1
    assert len(step_ticks) == 1001
    assert set(step_ticks) == {1}
    assert set(order_checks) == {1}


def test_adaptive_rejects_one_tick_when_split_check_fails(monkeypatch):
    dysolve = DysolvePropagator(
        sigmaz(), sigmax(), 1,
        options={
            'min_timestep': 1.0,
            'error_tolerance_per_time': 1.0,
            'max_order': 2,
        },
    )
    def rejected_error_rate(
        self, current_time, dt, order, step_propagator=None
    ):
        del self, current_time, dt, order, step_propagator
        return 2.0

    monkeypatch.setattr(
        DysolvePropagator,
        "_adaptive_split_error_rate",
        rejected_error_rate,
    )

    with pytest.raises(ValueError, match="cannot be met"):
        dysolve._adaptive_interval_plan(0.0, 1.0)


def test_adaptive_one_tick_still_checks_two_half_steps(monkeypatch):
    dysolve = DysolvePropagator(
        sigmaz(), sigmax(), 1,
        options={
            'min_timestep': 1.0,
            'error_tolerance_per_time': 1.0,
            'max_order': 1,
        },
    )
    evaluated_steps = []

    def record_subprop(current_time, dt, max_order=None):
        del current_time, max_order
        evaluated_steps.append(dt)
        return np.eye(2, dtype=complex)

    monkeypatch.setattr(dysolve, "_compute_subprop", record_subprop)

    assert dysolve._adaptive_split_error_rate(0.0, 1.0, 1) == 0.0
    assert evaluated_steps == [1.0, 0.5, 0.5]


@pytest.mark.parametrize("t_i, t_f", [
    (0.037, 0.237),
    (0.237, 0.037),
])
def test_adaptive_propagator_matches_fine_reference_lazily(t_i, t_f):
    H_0 = 0.7 * sigmaz()
    X = 0.35 * sigmax()
    omega = 5.0
    tolerance_per_time = 1e-5
    adaptive = DysolvePropagator(
        H_0, X, omega,
        options={
            'min_timestep': 0.001,
            'error_tolerance_per_time': tolerance_per_time,
            'max_order': 4,
            'a_tol': 1e-12,
        },
    )
    reference = DysolvePropagator(
        H_0, X, omega,
        options=_adaptive_options(
            0.00025,
            max_order=4,
            error_tolerance_per_time=1e-9,
        ),
    )

    order, step_ticks = adaptive._adaptive_interval_plan(t_i, t_f)
    actual = adaptive(t_f, t_i)
    expected = reference(t_f, t_i)

    assert order == 2
    assert step_ticks
    assert all(abs(tick) & (abs(tick) - 1) == 0 for tick in step_ticks)
    assert (actual - expected).norm() < 3e-6
    assert max(max(cached_orders) for cached_orders in adaptive._dt_Sns.values()) == order


def test_adaptive_propagator_supports_time_lists():
    H_0 = 0.7 * sigmaz()
    X = 0.35 * sigmax()
    times = [0.037, 0.137, 0.237]
    adaptive = DysolvePropagator(
        H_0, X, 5.0,
        options={
            'min_timestep': 0.001,
            'error_tolerance_per_time': 1e-5,
            'max_order': 4,
            'a_tol': 1e-12,
        },
    )
    reference = DysolvePropagator(
        H_0, X, 5.0,
        options=_adaptive_options(
            0.00025,
            max_order=4,
            error_tolerance_per_time=1e-9,
        ),
    )

    actual = adaptive.propagators(times)
    expected = reference.propagators(times)

    assert len(actual) == len(times)
    assert all(
        (actual_propagator - expected_propagator).norm() < 3e-6
        for actual_propagator, expected_propagator in zip(
            actual, expected, strict=True
        )
    )


def test_adaptive_constant_envelope_matches_constant_drive():
    options = _adaptive_options(
        0.001,
        max_order=4,
        error_tolerance_per_time=1e-5,
    )
    dysolve = DysolvePropagator(
        0.7 * sigmaz(), 0.35 * sigmax(), 5.0, options=options
    )

    shaped = dysolve.adaptive_envelope_propagator(
        lambda times: np.ones_like(times), 0.237, 0.037
    )
    constant = dysolve(0.237, 0.037)

    np.testing.assert_allclose(
        _qobj_data(shaped), _qobj_data(constant), rtol=1e-12, atol=1e-12
    )


def test_adaptive_shaped_envelope_matches_fine_reference():
    start = 0.037
    duration = 0.2
    dysolve = DysolvePropagator(
        0.7 * sigmaz(),
        0.35 * sigmax(),
        5.0,
        options=_adaptive_options(
            0.001,
            max_order=4,
            error_tolerance_per_time=1e-5,
        ),
    )

    def amplitude(times):
        return 0.7 + 0.15j * np.sin(2 * np.pi * (times - start) / duration)

    actual = dysolve.adaptive_envelope_propagator(
        amplitude, start + duration, start
    )
    reference_step_count = 1600
    reference_dt = duration / reference_step_count
    reference_times = start + (
        np.arange(reference_step_count) + 0.5
    ) * reference_dt
    expected = dysolve.envelope_propagator(
        amplitude(reference_times), reference_dt, t0=start
    )

    assert (actual - expected).norm() < 2e-7


def test_adaptive_envelope_parameter_gradients_match_finite_difference():
    start = 0.037
    duration = 0.2
    parameters = np.array([0.5, 0.2])
    dysolve = DysolvePropagator(
        0.7 * sigmaz(),
        0.35 * sigmax(),
        5.0,
        options=_adaptive_options(
            0.001,
            max_order=4,
            error_tolerance_per_time=1e-5,
        ),
    )

    def amplitude(times, values=parameters):
        tau = (times - start) / duration
        return values[0] + values[1] * tau

    def amplitude_derivatives(times):
        tau = (times - start) / duration
        return np.stack([np.ones_like(tau), tau])

    _, gradients = dysolve.adaptive_envelope_parameter_gradients(
        amplitude,
        amplitude_derivatives,
        start + duration,
        start,
    )

    epsilon = 1e-6
    for parameter_index, gradient in enumerate(gradients):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[parameter_index] += epsilon
        minus[parameter_index] -= epsilon
        plus_propagator = dysolve.adaptive_envelope_propagator(
            lambda times, values=plus: amplitude(times, values),
            start + duration,
            start,
        )
        minus_propagator = dysolve.adaptive_envelope_propagator(
            lambda times, values=minus: amplitude(times, values),
            start + duration,
            start,
        )
        finite_difference = (
            plus_propagator - minus_propagator
        ) / (2 * epsilon)
        assert (gradient - finite_difference).norm() < 1e-8


def test_dyson_orders_are_prepared_lazily(monkeypatch):
    computed_orders = []
    compute_Sn = DysolvePropagator._compute_Sn.__globals__["cy_compute_Sn"]

    def record_order(omega_vectors, *args, **kwargs):
        computed_orders.append(omega_vectors.shape[1])
        return compute_Sn(omega_vectors, *args, **kwargs)

    monkeypatch.setitem(
        DysolvePropagator._compute_Sn.__globals__,
        "cy_compute_Sn",
        record_order,
    )
    dysolve = DysolvePropagator(
        sigmaz(), sigmax(), 1,
        options=_adaptive_options(0.01, a_tol=1e-10),
    )

    assert set(dysolve._compute_Sns(0.1, max_order=1)) == {0, 1}
    assert computed_orders == [1]

    dysolve._compute_Sns(0.1, max_order=1)
    assert computed_orders == [1]

    assert set(dysolve._compute_Sns(0.1)) == {0, 1, 2, 3}
    assert computed_orders == [1, 2, 3]


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
        options=_adaptive_options(0.01, max_order=1, a_tol=1e-4),
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
    options = _adaptive_options(0.001)
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
    options = _adaptive_options(0.001)
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
    options = _adaptive_options(0.001)
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
    options = _adaptive_options(0.001)
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

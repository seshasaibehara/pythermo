import os
import json
import pytest
import numpy as np
import pythermo.clex.clex as pyclex


@pytest.fixture
def correlation_matrix(root_pytest_dir: str) -> np.ndarray:
    """TODO: Docstring for hull_data.

    Returns
    -------
    TODO

    """
    with open(
        os.path.join(root_pytest_dir, "tests", "input_files", "hull_data.json"), "r"
    ) as f:
        hull_data = json.load(f)

    correlation_matrix = pyclex.get_correlation_matrix(hull_data)

    return correlation_matrix


@pytest.fixture
def fake_eci_sets() -> np.ndarray:
    """Fake eci sets for testing

    Returns
    -------
    np.ndarray
        Fake ECI sets

    """
    return np.array([[0.45, 0.75], [0.45, 0.75]])


def test_mean_eci_set(fake_eci_sets: np.ndarray) -> None:
    """Tests mean eci set

    Parameters
    ----------
    fake_eci_sets : np.ndarray
        ECI sets

    Returns
    -------
    None

    """
    mean_eci = pyclex.get_mean_eci_set(fake_eci_sets)
    expected_mean_eci = np.array([0.6, 0.6])

    assert np.allclose(mean_eci, expected_mean_eci)


def test_get_correlation_matrix(correlation_matrix: np.ndarray) -> None:
    """Tests correlation matrix extraction

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Correlation matrix

    Returns
    -------
    None

    """
    expected_correlation_matrix = np.array([[0.0, 1.0]] * 9)

    assert np.allclose(correlation_matrix, expected_correlation_matrix)


def test_get_predicted_formation_energies(
    fake_eci_sets: np.ndarray, correlation_matrix: np.ndarray
) -> None:
    """Tests predicted formation energy extraction

    Parameters
    ----------
    fake_eci_sets : np.ndarray
        ECI sets
    correlation_matrix : np.ndarray
        Correlation matrix

    Returns
    -------
    None

    """
    formation_energy_sets = pyclex.get_predicted_formation_energies(
        correlation_matrix, fake_eci_sets
    )
    expected_formation_energy_sets = np.array([[0.45, 0.75]] * 9)

    assert np.allclose(formation_energy_sets, expected_formation_energy_sets)


def test_get_mean_predicted_formation_energies(
    fake_eci_sets: np.ndarray, correlation_matrix: np.ndarray
) -> None:
    """Tests mean predicted formation energy extraction

    Parameters
    ----------
    fake_eci_sets : np.ndarray
        ECI sets
    correlation_matrix : np.ndarray
        Correlation matrix

    Returns
    -------
    None

    """
    formation_energy_sets = pyclex.get_predicted_formation_energies(
        correlation_matrix, fake_eci_sets
    )
    mean_predicted_formation_energies = pyclex.get_mean_predicted_formation_energies(
        formation_energy_sets
    )
    expected_mean_formation_energies = np.array([0.6] * 9)

    assert np.allclose(
        mean_predicted_formation_energies, expected_mean_formation_energies
    )


def test_get_predicted_formation_energy_standard_deviations(
    fake_eci_sets: np.ndarray, correlation_matrix: np.ndarray
) -> None:
    """Tests formation energy standard deviation function

    Parameters
    ----------
    fake_eci_sets : np.ndarray
        ECI sets
    correlation_matrix : np.ndarray
        Correlation matrix

    Returns
    -------
    None

    """
    formation_energy_sets = pyclex.get_predicted_formation_energies(
        correlation_matrix, fake_eci_sets
    )
    formation_energy_standard_deviations = (
        pyclex.get_predicted_formation_energy_standard_deviations(formation_energy_sets)
    )
    expected_formation_energy_standard_deviations = np.array([0.15] * 9)

    assert np.allclose(
        formation_energy_standard_deviations,
        expected_formation_energy_standard_deviations,
    )


def test_get_eci_standard_deviations(fake_eci_sets: np.ndarray) -> None:
    """Tests eci standard deviations

    Parameters
    ----------
    fake_eci_sets : np.ndarray
        ECI sets

    Returns
    -------
    None

    """
    eci_standard_deviations = pyclex.get_eci_standard_deviations(fake_eci_sets)
    expected_eci_standard_deivations = np.array([0.15, 0.15])

    assert np.allclose(eci_standard_deviations, expected_eci_standard_deivations)


def test_get_rms_error_of_fit(
    correlation_matrix: np.ndarray, fake_eci_sets: np.ndarray
) -> None:
    """Test rms error of fit

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Correlation matrix
    fake_eci_sets : np.ndarray
        ECI sets

    Returns
    -------
    None

    """
    mean_formation_energies = pyclex.get_mean_predicted_formation_energies(
        pyclex.get_predicted_formation_energies(correlation_matrix, fake_eci_sets)
    )
    dft_formation_energies = np.array([0.1] * 9)
    rms_error_of_fit = pyclex.get_rms_error_of_fit(
        mean_formation_energies, dft_formation_energies
    )

    assert rms_error_of_fit == 0.5

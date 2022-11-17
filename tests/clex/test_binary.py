import os
import json
import pytest
import numpy as np
import pythermo.clex.binary as pybinary


@pytest.fixture
def libi_data(root_pytest_dir: str) -> list[dict]:
    """Read libi data

    Parameters
    ----------
    root_pytest_dir : str
        root directory for pytest

    Returns
    -------
    list[dict]
        ccasm query json style configs

    """
    with open(
        os.path.join(root_pytest_dir, "tests", "input_files", "libi_data.json"), "r"
    ) as f:
        return json.load(f)


def test_ground_state_indices(libi_data: list[dict]) -> None:
    """Test ground state indices

    Parameters
    ----------
    libi_data : list[dict]
        ccasm query json style configs

    Returns
    -------
    None

    """
    comps = np.array([config["comp"] for config in libi_data])
    formation_energies = np.array([config["formation_energy"] for config in libi_data])

    assert pybinary.ground_state_indices(comps, formation_energies) == [0, 4, 1, 7, 6]


def test_ground_state_configs(libi_data: list[dict]) -> None:
    """Test ground state configs

    Parameters
    ----------
    libi_data : list[dict]
        ccasm query json style configs

    Returns
    -------
    None

    """
    assert pybinary.ground_state_configs(libi_data) == [
        libi_data[i] for i in [0, 4, 1, 7, 6]
    ]


def test_ground_state_comps_and_energies(libi_data: list[dict]) -> None:
    """Test ground state comps and energies

    Parameters
    ----------
    libi_data : list[dict]
        ccasm query json style configs

    Returns
    -------
    None

    """
    comps = np.array([config["comp"] for config in libi_data])
    formation_energies = np.array([config["formation_energy"] for config in libi_data])

    (
        ground_state_comps,
        ground_state_energies,
    ) = pybinary.ground_state_comps_and_energies(comps, formation_energies)

    expected_ground_state_comps = np.array([[0.0], [0.25], [0.333333333], [0.5], [1.0]])
    expected_ground_state_energies = np.array(
        [0.0, -0.548678, -0.5002834, -0.39904486, 0.0]
    )

    assert np.allclose(ground_state_comps, expected_ground_state_comps) is True
    assert np.allclose(ground_state_energies, expected_ground_state_energies) is True


def test_indices_and_hull_distances_within_given_params(libi_data: list[dict]) -> None:
    """Test retrieval of indices and hull distances of configs
    given max and min hull distances

    Parameters
    ----------
    libi_data : list[dict]
        ccasm query json style configs

    Returns
    -------
    None

    """
    comps = np.array([config["comp"] for config in libi_data])
    formation_energies = np.array([config["formation_energy"] for config in libi_data])
    index, hull_distance = pybinary.indices_and_hull_distances_within_given_parameters(
        comps, formation_energies, 0.02
    )

    assert index == [3]
    assert np.allclose(hull_distance, [0.0122516]) is True


def test_required_property_from_a_given_list_of_indices(libi_data: list[dict]) -> None:
    """Test retrieval of required property from configs
    given a list of indices

    Parameters
    ----------
    libi_data : list[dict]
        ccasm query json style configs

    Returns
    -------
    None

    """
    comps = pybinary.get_required_property_from_formation_energy_dict_from_given_list_of_indices(
        libi_data, "comp", [3, 6]
    )

    assert np.allclose(comps, [[0.5], [1.0]]) is True


def test_convert_hull_distances_to_formation_energies(libi_data: list[dict]) -> None:
    """Test convert hull distances to formation energies

    Parameters
    ----------
    libi_data : list[dict]
        ccasm query json style configs

    Returns
    -------
    None

    """
    comps = np.array([config["comp"] for config in libi_data])
    formation_energies = np.array([config["formation_energy"] for config in libi_data])
    formation_energies_of_hull_distances = (
        pybinary.convert_hull_distances_to_formation_energies(
            np.array([0.0122516]), np.array([[0.5]]), comps, formation_energies
        )
    )

    assert np.allclose(formation_energies_of_hull_distances, [-0.386793266500]) is True


def test_chemical_potentials(libi_data: list[dict]) -> None:
    """Test chemical potentials

    Parameters
    ----------
    libi_data : list[dict]
        ccasm query json style configs

    Returns
    -------
    None

    """

    comps = np.array([config["comp"] for config in libi_data])
    formation_energies = np.array([config["formation_energy"] for config in libi_data])
    li_chem_pots, bi_chem_pots = pybinary.get_chemical_potentials(
        comps, formation_energies, [-1.9042108, -3.885612667000], "s"
    )
    expected_li_chem_pots = np.array(
        [[-1.9042108], [-2.5980765], [-2.6069713], [-2.70230053]]
    )
    expected_bi_chem_pots = np.array(
        [[-6.0803286], [-3.9987315], [-3.9809419], [-3.88561267]]
    )

    assert np.allclose(li_chem_pots, expected_li_chem_pots) is True
    assert np.allclose(bi_chem_pots, expected_bi_chem_pots) is True


def test_voltages(libi_data: list[dict]) -> None:
    """Test voltages

    Parameters
    ----------
    libi_data : list[dict]
        ccasm query json style configs

    Returns
    -------
    None

    """
    expected_li_chem_pots = np.array(
        [[-1.9042108], [-2.5980765], [-2.6069713], [-2.70230053]]
    )
    expected_li_voltages = -(expected_li_chem_pots - (-1.9042108))

    comps = np.array([config["comp"] for config in libi_data])
    formation_energies = np.array([config["formation_energy"] for config in libi_data])
    li_chem_pots, _ = pybinary.get_chemical_potentials(
        comps, formation_energies, [-1.9042108, -3.885612667000], "s"
    )
    li_voltages = pybinary.get_voltages(li_chem_pots, -1.9042108, 1)

    assert np.allclose(li_voltages, expected_li_voltages) is True


def test_end_state_configs(libi_data: list[dict]) -> None:
    """Test end state configs

    Parameters
    ----------
    libi_data : list[dict]
        ccasm query json style configs

    Returns
    -------
    None

    """
    ground_states = pybinary.ground_state_configs(libi_data)
    end_states = pybinary.end_state_configs(ground_states)

    assert end_states == [libi_data[i] for i in [0, 6]]

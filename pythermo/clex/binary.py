import thermocore
import numpy as np
import scipy.spatial
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


def binary_convex_hull(
    comps: np.ndarray, energies: np.ndarray
) -> scipy.spatial.ConvexHull:
    """Returns binary convex hull from comps and energies

    Parameters
    ----------
    comps : np.ndarray
    energies : np.ndarray

    Returns
    -------
    scipy.spatial.ConvexHull
        ConvexHull

    """
    return scipy.spatial.ConvexHull(np.column_stack((comps, energies)))


def ground_state_indices(comps, energies):
    """TODO: Docstring for ground_state_indices.

    Parameters
    ----------
    comps : TODO
    energies : TODO

    Returns
    -------
    TODO

    """
    binary_hull = binary_convex_hull(comps, energies)
    lower_hull = thermocore.geometry.hull.lower_hull(binary_hull)

    return lower_hull[0].tolist()


def order_ground_state_comps_and_energies(
    ground_state_comps, ground_state_formation_energies
):
    """TODO: Docstring for order_ground_state_comps_and_energies.

    Parameters
    ----------
    ground_state_comps : TODO
    ground_state_formation_energies : TODO

    Returns
    -------
    TODO

    """
    zipped_comps_eneriges = list(
        zip(ground_state_comps, ground_state_formation_energies)
    )
    zipped_comps_eneriges.sort(key=lambda x: x[0])

    return (
        np.array([entry[0] for entry in zipped_comps_eneriges]),
        np.array([entry[1] for entry in zipped_comps_eneriges]),
    )


def ground_state_comps_and_energies(
    comps: np.ndarray, energies: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Ground state energies and comps sorted in order of increasing composition

    Parameters
    ----------
    comps : np.ndarray
    energies : np.ndarray

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Ground state energies and compositions sorted in order of increasing composition

    """
    # convex hull
    hull = binary_convex_hull(comps, energies)

    # lower hull
    lower_hull = thermocore.geometry.hull.lower_hull(hull)

    # get ground state energies and comps
    ground_state_comps = [comps.tolist()[index] for index in lower_hull[0].tolist()]
    ground_state_energies = [
        energies.tolist()[index] for index in lower_hull[0].tolist()
    ]

    # sort them by comps
    return order_ground_state_comps_and_energies(
        ground_state_comps, ground_state_energies
    )


def plot_binary_convex_hull(
    ax: plt.axis, comps: np.ndarray, energies: np.ndarray, **kwargs
) -> plt.axis:
    """Plot binary convex hull along with the lower hull

    Parameters
    ----------
    ax : plt.axis
    comp : np.ndarray
    energies : np.ndarray

    Returns
    -------
    ax : plt.axis
        Returns matplotlib axis after plotting

    """
    # clean up *args and **kwargs and setup default args
    on_hull_options = default_on_hull_plotting_options()
    not_on_hull_options = default_not_on_hull_plotting_options()
    lower_hull_options = default_lower_hull_plotting_options()

    for key, value in kwargs.items():
        if key == "on_hull_options":
            for on_hull_key, on_hull_value in value.items():
                on_hull_options[on_hull_key] = on_hull_value
        if key == "not_on_hull_options":
            for not_on_hull_key, not_on_hull_value in value.items():
                not_on_hull_options[not_on_hull_key] = not_on_hull_value
        if key == "lower_hull_options":
            for lower_hull_key, lower_hull_value in value.items():
                lower_hull_options[lower_hull_key] = lower_hull_value

    # get convex hull
    hull = binary_convex_hull(comps, energies)

    # get the lower_hull
    lower_hull = thermocore.geometry.hull.lower_hull(hull)

    # plt comps, energies
    for index, entry in enumerate(zip(comps.tolist(), energies.tolist())):
        if index in lower_hull[0].tolist():
            ax.scatter(entry[0], entry[1], **on_hull_options)
        else:
            ax.scatter(entry[0], entry[1], **not_on_hull_options)

    # plot lower hull
    ground_state_comps, ground_state_energies = ground_state_comps_and_energies(
        comps, energies
    )
    ax.plot(ground_state_comps, ground_state_energies, **lower_hull_options)

    return ax


def indices_and_hull_distances_within_given_parameters(
    comps: np.ndarray,
    energies: np.ndarray,
    max_distance: float,
    min_distance: float = 1e-4,
) -> List[Tuple[int, float]]:
    """Indices and hull distances from lower convex hull of compositions and energies within
    given min and max hull distances

    Parameters
    ----------
    comps : np.ndarray
    energies : np.ndarray
    max_distance : float
    min_distance : float, optional

    Returns
    -------
    List[Tuple[int, float]]
        Indices and hull distances of compositions and energies close to convex hull
        within given parameters

    """
    lower_hull_distances = thermocore.geometry.hull.lower_hull_distances(
        comps[:, np.newaxis], energies
    )

    indices_and_hull_distnaces = [
        (index, hull_distance)
        for index, hull_distance in enumerate(lower_hull_distances)
        if hull_distance < max_distance and hull_distance > min_distance
    ]

    return [entry[0] for entry in indices_and_hull_distnaces], [
        entry[1] for entry in indices_and_hull_distnaces
    ]


def default_on_hull_plotting_options() -> Dict:
    """Default plotting options for points on hull

    Returns
    -------
    Dict

    """
    return {
        "color": "tab:green",
        "marker": "s",
        "facecolors": "none",
        "linewidth": 2.0,
        "s": 54,
    }


def default_not_on_hull_plotting_options() -> Dict:
    """Default plotting options for points not on hull

    Returns
    -------
    Dict

    """
    return {
        "color": "tab:green",
        "alpha": 0.75,
        "s": 54,
    }


def default_lower_hull_plotting_options() -> Dict:
    """Default plotting options for points on lower hull

    Returns
    -------
    Dict

    """
    return {
        "color": "black",
        "linestyle": "--",
        "linewidth": 2,
    }


def get_required_property_from_formation_energy_dict_from_given_list_of_indices(
    formation_energy_dict: Dict, required_property: str, indices_list: List[int]
) -> List:
    """Get config names for a given list of indices

    Parameters
    ----------
    formation_energy_dict : Dict
    indices_list : List[int]

    Returns
    -------
    List

    """
    return [
        entry[required_property]
        for index, entry in enumerate(formation_energy_dict)
        if index in indices_list
    ]


def convert_hull_distances_to_formation_energies(
    hull_distances: np.ndarray,
    comps_of_hull_distances: np.ndarray,
    comps: np.ndarray,
    formation_energies: np.ndarray,
) -> np.ndarray:
    """TODO: Docstring for convert_hull_distances_to_formation_energies.

    Parameters
    ----------
    hull_distances : TODO
    comps_of_hull_distances : TODO
    comps : TODO
    formation_energies : TODO

    Returns
    -------
    TODO

    """
    convex_hull = binary_convex_hull(comps, formation_energies)
    lower_hull_energies_at_hull_distance_comps = (
        thermocore.geometry.hull.lower_hull_energies(
            comps_of_hull_distances, convex_hull
        )
    )

    if len(hull_distances.shape) == 1:
        return hull_distances + lower_hull_energies_at_hull_distance_comps
    else:
        return (
            hull_distances + lower_hull_energies_at_hull_distance_comps[:, np.newaxis]
        )


def _intercepts_on_axes(ground_state_comps, ground_state_formation_energies):
    """TODO: Docstring for get_chemical_potentials_for_a_substitutioal_solid.

    Parameters
    ----------
    ground_state_comps : TODO
    ground_state_formation_energies : TODO

    Returns
    -------
    TODO

    """

    ordered_ground_state_comps_and_formation_energies = (
        order_ground_state_comps_and_energies(
            ground_state_comps, ground_state_formation_energies
        )
    )

    comps = ordered_ground_state_comps_and_formation_energies[0]
    fes = ordered_ground_state_comps_and_formation_energies[1]

    # {(fe(i+1) - fe(i)) / (comp(i+1) - comp(i))}*(-comp(i)) + fe(i)
    intercepts_on_left_axis = []
    for i in range(len(comps) - 1):
        intercept = (
            ((fes[i + 1] - fes[i]) / (comps[i + 1] - comps[i])) * (-comps[i])
        ) + fes[i]

        intercepts_on_left_axis.append(intercept)

    # { (fe(i+1) - fe(i))/ (comp(i+1) - comp(i)) *(1-comp(i))} + fe(i)
    intercepts_on_right_axis = []
    for i in range(len(comps) - 1):
        intercept = (
            ((fes[i + 1] - fes[i]) / (comps[i + 1] - comps[i])) * (1 - comps[i])
        ) + fes[i]
        intercepts_on_right_axis.append(intercept)

    return (np.array(intercepts_on_left_axis), np.array(intercepts_on_right_axis))


def get_chemical_potentials(
    ground_state_comps,
    ground_state_formation_energies,
    reference_energies: List,
    type_of_compound: str,
):
    """TODO: Docstring for get_chemical_potentials.

    Parameters
    ----------
    ground_state_comps : TODO
    ground_state_formation_energies : TODO
    reference_energies : TODO
    type_of_compound : TODO

    Returns
    -------
    TODO

    """
    assert len(ground_state_comps) == len(ground_state_formation_energies)

    first_character = type_of_compound.lower()[0]
    if first_character == "s":
        intercepts_on_left_axis, intercepts_on_right_axis = _intercepts_on_axes(
            ground_state_comps, ground_state_formation_energies
        )
        chemical_potentials_on_left_axis = (
            intercepts_on_left_axis + reference_energies[0]
        )
        chemical_potentials_on_right_axis = (
            intercepts_on_right_axis + reference_energies[1]
        )

        return chemical_potentials_on_left_axis, chemical_potentials_on_right_axis

    elif first_character == "i":
        raise NotImplementedError(
            "Chemical potentials for interstitial solids are not implemented yet"
        )

    else:
        raise RuntimeError(
            "Type of compound not known. Should be either substitutional or interstitial"
        )


def get_voltages(chemical_potentials, reference_state, number_of_electrons):
    """TODO: Docstring for get_voltages.

    Parameters
    ----------
    chemical_potentials : TODO
    reference_state : TODO
     : TODO

    Returns
    -------
    TODO

    """
    return -(chemical_potentials - reference_state) / number_of_electrons

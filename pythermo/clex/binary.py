import numpy as np
import matplotlib.pyplot as plt
import thermocore.geometry.hull as thull


def ground_state_indices(comps: np.ndarray, energies: np.ndarray) -> list[int]:
    """Given comps and energies, returns a list of ground state indices
    of comps and energies

    Parameters
    ----------
    comps : np.ndarray
    energies : np.ndarray

    Returns
    -------
    list[int]
        Indices of ground states

    """
    binary_hull = thull.full_hull(comps, energies)
    lower_hull = thull.lower_hull(binary_hull)

    return lower_hull[0].tolist()


def order_ground_state_comps_and_energies(
    ground_state_comps: np.ndarray, ground_state_formation_energies: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Order ground states by composition

    Parameters
    ----------
    ground_state_comps : np.ndarray
    ground_state_formation_energies : np.ndarray

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Ordered ground state comps and energies

    """
    zipped_comps_eneriges = list(
        zip(ground_state_comps, ground_state_formation_energies)
    )
    zipped_comps_eneriges.sort(key=lambda comp: comp[0])

    return (
        np.array([entry[0] for entry in zipped_comps_eneriges]),
        np.array([entry[1] for entry in zipped_comps_eneriges]),
    )


def ground_state_comps_and_energies(
    comps: np.ndarray, energies: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ground state energies and comps sorted in order of increasing composition

    Parameters
    ----------
    comps : np.ndarray
    energies : np.ndarray

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Ground state energies and compositions sorted in order of increasing composition

    """
    indices = ground_state_indices(comps, energies)

    # get ground state energies and comps
    ground_state_comps = [comps[index] for index in indices]
    ground_state_energies = [energies[index] for index in indices]

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

    # plt comps, energies
    on_hull_comps, on_hull_energies = ground_state_comps_and_energies(comps, energies)
    ax.scatter(on_hull_comps, on_hull_energies, **on_hull_options)
    ax.plot(on_hull_comps, on_hull_energies, **lower_hull_options)

    on_hull_indices = ground_state_indices(comps, energies)
    not_on_hull_comps = [
        comp for i, comp in enumerate(comps) if i not in on_hull_indices
    ]
    not_on_hull_energies = [
        energy for i, energy in enumerate(energies) if i not in on_hull_indices
    ]

    ax.scatter(not_on_hull_comps, not_on_hull_energies, **not_on_hull_options)

    return ax


def indices_and_hull_distances_within_given_parameters(
    comps: np.ndarray,
    energies: np.ndarray,
    max_distance: float,
    min_distance: float = 1e-4,
) -> list[tuple[int, float]]:
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
    list[tuple[int, float]]
        Indices and hull distances of compositions and energies close to convex hull
        within given parameters

    """
    lower_hull_distances = thull.lower_hull_distances(comps, energies)

    indices_and_hull_distnaces = [
        (index, hull_distance)
        for index, hull_distance in enumerate(lower_hull_distances)
        if hull_distance < max_distance and hull_distance > min_distance
    ]

    return [entry[0] for entry in indices_and_hull_distnaces], [
        entry[1] for entry in indices_and_hull_distnaces
    ]


def default_on_hull_plotting_options() -> dict:
    """Default plotting options for points on hull

    Returns
    -------
    dict

    """
    return dict(
        color="tab:green",
        marker="s",
        facecolors="none",
        linewidth=2.0,
        s=54,
    )


def default_not_on_hull_plotting_options() -> dict:
    """Default plotting options for points not on hull

    Returns
    -------
    dict

    """
    return dict(
        color="tab:green",
        alpha=0.75,
        s=54,
    )


def default_lower_hull_plotting_options() -> dict:
    """Default plotting options for points on lower hull

    Returns
    -------
    dict

    """
    return dict(
        color="black",
        linestyle="--",
        linewidth=2.0,
    )


def get_required_property_from_formation_energy_dict_from_given_list_of_indices(
    formation_energy_dict: dict, required_property: str, indices_list: list[int]
) -> list:
    """Get config names for a given list of indices

    Parameters
    ----------
    formation_energy_dict : dict
    indices_list : list[int]

    Returns
    -------
    list

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
    """Given hull distances and comps, converts the hull
    distances to formation energies

    Parameters
    ----------
    hull_distances : np.ndarray
    comps_of_hull_distances : np.ndarray
    comps : np.ndarray
    formation_energies : np.ndarray

    Returns
    -------
    np.ndarray

    """
    convex_hull = thull.full_hull(comps, formation_energies)
    lower_hull_energies_at_hull_distance_comps = thull.lower_hull_energies(
        comps_of_hull_distances, convex_hull
    )

    if len(hull_distances.shape) == 1:
        return hull_distances + lower_hull_energies_at_hull_distance_comps
    else:
        return (
            hull_distances + lower_hull_energies_at_hull_distance_comps[:, np.newaxis]
        )


def _intercepts_on_axes(
    ground_state_comps: np.ndarray, ground_state_formation_energies: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Given ground_state_comps and ground_state_formation_energies,
    calculates intercepts of ground states on both the axes of
    convex hull

    Parameters
    ----------
    ground_state_comps : np.ndarray
    ground_state_formation_energies : np.ndarray

    Returns
    -------
    tuple[np.ndarray, np.ndarray]

    """

    comps, fes = order_ground_state_comps_and_energies(
        ground_state_comps, ground_state_formation_energies
    )

    # { [(fe(i+1) - fe(i)) / (comp(i+1) - comp(i))] *(-comp(i))} + fe(i)
    intercepts_on_left_axis = np.array(
        [
            (((fes[i + 1] - fes[i]) / (comps[i + 1] - comps[i])) * (-comps[i])) + fes[i]
            for i in range(len(comps) - 1)
        ]
    )

    # { [(fe(i+1) - fe(i))/ (comp(i+1) - comp(i))] *(1-comp(i)) } + fe(i)
    intercepts_on_right_axis = np.array(
        [
            (((fes[i + 1] - fes[i]) / (comps[i + 1] - comps[i])) * (1 - comps[i]))
            + fes[i]
            for i in range(len(comps) - 1)
        ]
    )

    return intercepts_on_left_axis, intercepts_on_right_axis


def get_chemical_potentials(
    ground_state_comps: np.ndarray,
    ground_state_formation_energies: np.ndarray,
    reference_energies: list,
    type_of_compound: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates chemical potentials of both the elements
    in a binary system. If ``type_of_compound`` is
    "substitutional", gives the intercepts on left and
    right axis respectively.

    Parameters
    ----------
    ground_state_comps : np.ndarray
    ground_state_formation_energies : np.ndarray
    reference_energies : list
    type_of_compound : str

    Returns
    -------
    tuple[np.ndarray, np.ndarray]

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


def get_voltages(
    chemical_potentials: np.ndarray,
    reference_state: np.ndarray,
    number_of_electrons: int,
) -> np.ndarray:
    """Given reference_state and chemical potentials,
    calculates voltages

    Parameters
    ----------
    chemical_potentials : np.ndarray
    reference_state : np.ndarray
    number_of_electrons : int

    Returns
    -------
    np.ndarray

    """
    return -(chemical_potentials - reference_state) / number_of_electrons

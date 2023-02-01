import os
import numpy as np
import matplotlib.pyplot as plt
import thermocore.geometry.hull as thull


def ground_state_indices(
    comps: np.ndarray, formation_energies: np.ndarray
) -> list[int]:
    """Given comps and formation_energies, returns a
    list of ground state indices of comps and energies
    ordered by ascending order of comps

    Parameters
    ----------
    comps : np.ndarray
        compositions where each row corresponds to
        a composition of a config
    formation_energies : np.ndarray
        formation energies as a vector

    Returns
    -------
    list[int]
        Indices of ground states ordered by
        ascending order of compositions

    """
    binary_hull = thull.full_hull(comps, formation_energies)
    lower_hull = thull.lower_hull(binary_hull)

    ground_state_indices = lower_hull[0].tolist()
    ground_state_indices.sort(key=lambda index: comps[index])
    return ground_state_indices


def ground_state_configs(selected_configurations: list[dict]) -> list[dict]:
    """Given a list of selected configurations,
    returns ground states among them

    Parameters
    ----------
    selected_configurations : list[dict]
        ccasm query json style selection

    Returns
    -------
    list[dict]
        Ground state configs

    """

    comps = np.array([config["comp"] for config in selected_configurations])
    energies = np.array(
        [config["formation_energy"] for config in selected_configurations]
    )
    ground_states = ground_state_indices(comps, energies)

    return [selected_configurations[i] for i in ground_states]


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
    ground_state_comps = np.array([comps[index] for index in indices])
    ground_state_energies = np.array([energies[index] for index in indices])

    return ground_state_comps, ground_state_energies


def plot_binary_convex_hull(
    ax: plt.axis,
    comps: np.ndarray,
    energies: np.ndarray,
    plot_on_hull: bool = True,
    plot_not_on_hull: bool = True,
    plot_lower_hull: bool = True,
    max_not_on_hull_energy=1000,
    **kwargs,
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
    on_hull_options, not_on_hull_options, lower_hull_options = get_plotting_options(
        kwargs
    )

    # plt comps, energies
    on_hull_comps, on_hull_energies = ground_state_comps_and_energies(comps, energies)
    if plot_on_hull:
        ax.scatter(on_hull_comps, on_hull_energies, **on_hull_options)

    if plot_lower_hull:
        ax.plot(on_hull_comps, on_hull_energies, **lower_hull_options)

    on_hull_indices = ground_state_indices(comps, energies)
    not_on_hull_comps = np.array(
        [comp for i, comp in enumerate(comps) if i not in on_hull_indices]
    )
    not_on_hull_energies = np.array(
        [energy for i, energy in enumerate(energies) if i not in on_hull_indices]
    )

    if plot_not_on_hull:
        ax.scatter(
            not_on_hull_comps,
            np.where(
                not_on_hull_energies < max_not_on_hull_energy,
                not_on_hull_energies,
                np.nan,
            ),
            **not_on_hull_options,
        )

    return ax


def get_plotting_options(
    user_options: dict,
) -> tuple[dict, dict, dict]:
    """
    Get plotting options as dictionaries by reading
    in user_options

    Parameters
    ----------
    user_options : dict
        User provided options as dictionary

    Returns
    -------
    tuple[dict, dict, dict]
        on_hull_options, not_on_hull_options, lower_hull_options
        as dictionaries

    """
    on_hull_options = default_on_hull_plotting_options()
    not_on_hull_options = default_not_on_hull_plotting_options()
    lower_hull_options = default_lower_hull_plotting_options()

    for key, value in user_options.items():
        if key == "on_hull_options":
            for on_hull_key, on_hull_value in value.items():
                on_hull_options[on_hull_key] = on_hull_value
        if key == "not_on_hull_options":
            for not_on_hull_key, not_on_hull_value in value.items():
                not_on_hull_options[not_on_hull_key] = not_on_hull_value
        if key == "lower_hull_options":
            for lower_hull_key, lower_hull_value in value.items():
                lower_hull_options[lower_hull_key] = lower_hull_value

    return on_hull_options, not_on_hull_options, lower_hull_options


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

    indices_and_hull_distances = [
        (index, hull_distance)
        for index, hull_distance in enumerate(lower_hull_distances)
        if hull_distance < max_distance and hull_distance > min_distance
    ]

    return [entry[0] for entry in indices_and_hull_distances], [
        entry[1] for entry in indices_and_hull_distances
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
    comps: np.ndarray, fes: np.ndarray
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
    comps: np.ndarray,
    formation_energies: np.ndarray,
    reference_energies: list,
    type_of_compound: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates chemical potentials of both the elements
    in a binary system. If ``type_of_compound`` is
    "substitutional", gives the intercepts on left and
    right axis respectively.

    Parameters
    ----------
    comps : np.ndarray
    formation_energies : np.ndarray
    reference_energies : list
    type_of_compound : str

    Returns
    -------
    tuple[np.ndarray, np.ndarray]

    """
    (
        ground_state_comps,
        ground_state_formation_energies,
    ) = ground_state_comps_and_energies(comps, formation_energies)

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


def end_state_configs(ground_state_configs: list[dict]) -> list[dict]:
    """TODO: Docstring for find_end_states.

    Parameters
    ----------
    ground_state_configs : TODO

    Returns
    -------
    TODO

    """

    return [
        config
        for config in ground_state_configs
        if np.isclose(config["comp"][0], 0.0) or np.isclose(config["comp"][0], 1.0)
    ]


def fetch_ground_state_relaxed_structure_paths(
    selected_configurations: list[dict],
    calctype: str = "default",
    exclude_end_states: bool = True,
) -> list[str]:
    """Given a list of ``selected_configurations``, computes
    the ground states and makes a list of ground states and
    returns a path to CONTCAR by reading the config ``name``
    in ``selected_configurations``

    IF THERE ARE CONFLICTING MAPS OR MAPS FROM DIFFERENT CONFIGURATIONS
    THIS FUNCTION DOES NOT KNOW ABOUT IT. PLEASE MAKE SURE TO CHECK
    THAT THE CONFIG NAME ACTUALLY HAS THE FORMATION ENERGY CORRESPONDING
    TO IT'S STRUCTURE (NOT THE BEST MAP).

    Parameters
    ----------
    selected_configurations : list[dict]
        ccasm query json style list of configurations
        with comp and formation_energy queries present
    calctype : str, optional
        ccasm calctype (default = "default")
    exclude_end_states: bool, optional
        If True (default), excludes end state
        (where comp is 0 or 1) contcars


    Returns
    -------
    list[str]
        Paths of the ground state structures (CONTCARS) in a
        casm project. For example if "SCEL1_1_1_1_0_0_0/0" is
        a ground state, this function returns
        ["training_data/SCEL1_1_1_1_0_0_0/0/calctype.``calctype``/run.final/CONTCAR"]

    """
    ground_states = ground_state_configs(selected_configurations)
    end_states = end_state_configs(ground_states)
    if exclude_end_states:
        return [
            os.path.join(
                "training_data",
                config["name"],
                "calctype." + calctype,
                "run.final",
                "CONTCAR",
            )
            for config in ground_states
            if config not in end_states
        ]

    return [
        os.path.join(
            "training_data",
            config["name"],
            "calctype." + calctype,
            "run.final",
            "CONTCAR",
        )
        for config in ground_states
    ]


def fetch_ground_states_info(
    selected_configurations: list[dict],
    type_of_compound: str,
    reference_energies: list[float],
    reference_state_names: list[str] | None = None,
    exclude_end_states: bool = True,
) -> list[dict]:
    """TODO: Docstring for fetch_ground_states_info.

    Parameters
    ----------
    selected_configurations : TODO
    type_of_compound : TODO
    reference_energies : TODO
    reference_state_names : TODO
    exclude_end_states : TODO

    Returns
    -------
    TODO

    """
    comps = np.array([config["comp"] for config in selected_configurations])
    formation_energies = np.array(
        [config["formation_energy"] for config in selected_configurations]
    )

    ground_states = ground_state_configs(selected_configurations)
    end_states = end_state_configs(ground_states)

    type_of_compound = type_of_compound.lower()[0]
    if type_of_compound == "s":
        left_chem_pot, right_chem_pot = get_chemical_potentials(
            comps, formation_energies, reference_energies, "s"
        )
    else:
        raise NotImplementedError("Interstitial compounds not implemented")

    if reference_state_names is None:
        reference_state_names = ["left_chem_pot_range", "right_chem_pot_range"]

    ground_states_info = []
    for i, config in enumerate(ground_states):
        # if ground states
        if exclude_end_states:
            if config in end_states:
                continue

        ground_state_info = dict()
        ground_state_info.update(config)

        if np.isclose(config["comp"][0], 0.0):
            ground_state_info[reference_state_names[0]] = [
                left_chem_pot[0][0],
                left_chem_pot[0][0],
            ]
            ground_state_info[reference_state_names[1]] = [
                right_chem_pot[0][0],
                right_chem_pot[0][0],
            ]

        elif np.isclose(config["comp"][0], 1.0):
            ground_state_info[reference_state_names[0]] = [
                left_chem_pot[0][0],
                left_chem_pot[0][0],
            ]
            ground_state_info[reference_state_names[1]] = [
                right_chem_pot[0][0],
                right_chem_pot[0][0],
            ]

        else:
            ground_state_info[reference_state_names[0]] = [
                left_chem_pot[i - 1][0],
                left_chem_pot[i][0],
            ]
            ground_state_info[reference_state_names[1]] = [
                right_chem_pot[i - 1][0],
                right_chem_pot[i][0],
            ]

        ground_states_info.append(ground_state_info)
    return ground_states_info

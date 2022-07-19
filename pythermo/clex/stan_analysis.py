import thermocore
import numpy as np


def get_eci_samples(stan_results):
    """Assumes stan_results contain eci

    Parameters
    ----------
    stan_results : TODO

    Returns
    -------
    TODO

    """
    return stan_results["eci"]


def get_mean_ecis(eci_samples):
    """TODO: Docstring for get_mean_ecis_from_eci_samples.

    Parameters
    ----------
    eci_samples : TODO

    Returns
    -------
    TODO

    """
    return np.mean(eci_samples, axis=1)


def get_correlation_matrix(casm_queried_corr_dictionary):
    """TODO: Docstring for get_correlation_matrix_from_correlation_dictionary.

    Parameters
    ----------
    corr_dictionary : TODO

    Returns
    -------
    TODO

    """

    correlations_sorted_by_property = (
        thermocore.io.casm.regroup_query_by_config_property(
            casm_queried_corr_dictionary
        )
    )
    return np.array(correlations_sorted_by_property["corr"])


def get_formation_energy_samples(correlation_matrix, eci_samples):
    """TODO: Docstring for get_mean_formation_energy.

    Parameters
    ----------
    arg1 : TODO

    Returns
    -------
    TODO

    """
    return correlation_matrix @ eci_samples


def get_mean_formation_energies(formation_energy_samples):
    """TODO: Docstring for get_mean_formation_energy.

    Parameters
    ----------
    formation_energy_samples : TODO

    Returns
    -------
    TODO

    """
    return np.mean(formation_energy_samples, axis=1)


def get_formation_energy_standard_deviations(formation_energy_samples):
    """TODO: Docstring for get_formation_energy_standard_deviations.

    Parameters
    ----------
    formation_energy_samples : TODO

    Returns
    -------
    TODO

    """
    return np.std(formation_energy_samples, axis=1)


def get_eci_standard_deviations(eci_samples):
    """TODO: Docstring for get_eci_standard_deviations.

    Parameters
    ----------
    eci_samples : TODO

    Returns
    -------
    TODO

    """
    return np.std(eci_samples, axis=1)


def get_rms_error_of_fit(predicted_formation_energies, dft_formation_energies):
    """TODO: Docstring for get_rms_error_of_fit.

    Parameters
    ----------
    arg1 : TODO

    Returns
    -------
    TODO

    """
    return np.sqrt(
        np.mean(np.square(predicted_formation_energies - dft_formation_energies))
    )


def hull_distance_correlation(corr: np.ndarray, comp: np.ndarray, hull) -> np.ndarray:
    """Calculated the effective correlations to predict hull distance instead of absolute formation energy.
    Parameters:
    -----------
    corr: np.array
        nxk correlation matrix, where n is the number of configurations and k is the number of ECI.
    comp: np.array
        nxc matrix of compositions, where n is the number of configurations and c is the number of composition axes.
    formation_energy: np.array
        nx1 matrix of formation energies.

    Returns:
    --------
    hulldist_corr: np.array
        nxk matrix of effective correlations describing hull distance instead of absolute formation energy.
        n is the number of configurations and k is the number of ECI.
    """

    # Get convex hull simplices
    lower_vertices, lower_simplices = thermocore.geometry.hull.lower_hull(hull)

    hulldist_corr = np.zeros(corr.shape)

    for config_index in list(range(corr.shape[0])):

        # Find the simplex that contains the current configuration's composition, and find the hull energy for that composition
        (
            relevant_simplex_index,
            _,
        ) = thermocore.geometry.hull.lower_hull_simplex_containing(
            compositions=comp[config_index].reshape(1, -1),
            convex_hull=hull,
            lower_hull_simplex_indices=lower_simplices,
        )

        relevant_simplex_index = relevant_simplex_index[0]

        # Find vectors defining the corners of the simplex which contains the curent configuration's composition.
        simplex_corners = comp[hull.simplices[relevant_simplex_index]]
        interior_point = np.array(comp[config_index]).reshape(1, -1)

        # Enforce that the sum of weights is equal to 1.
        simplex_corners = np.hstack(
            (simplex_corners, np.ones((simplex_corners.shape[0], 1)))
        )
        interior_point = np.hstack((interior_point, np.ones((1, 1))))

        # Project the interior point onto the vectors that define the simplex corners.
        weights = interior_point @ np.linalg.pinv(simplex_corners)

        # Form the hull distance correlations by taking a linear combination of simplex corners.

        hulldist_corr[config_index] = (
            corr[config_index] - weights @ corr[hull.simplices[relevant_simplex_index]]
        )

    return hulldist_corr


def plot_error_bar(ax, xprop, yprop, yerr, **kwargs):
    """TODO: Docstring for plot_error_bar.

    Parameters
    ----------
    ax : TODO
    xprop : TODO
    yprop : TODO
    yerr : TODO
    **kwargs : TODO

    Returns
    -------
    TODO

    """
    keys = list(kwargs.keys())
    if "error_bar_options" not in keys:
        error_bar_options = default_error_bar_options()

    for key, value in kwargs.items():
        if key == "error_bar_options":
            error_bar_options = value
        else:
            raise RuntimeError("Not a valid key")

    ax.errorbar(xprop, yprop, yerr=yerr, **error_bar_options)

    return ax


def default_error_bar_options():
    """TODO: Docstring for default_error_bar_options.
    Returns
    -------
    TODO

    """
    return {"fmt": "none", "capsize": 3.0, "elinewidth": 2.0}

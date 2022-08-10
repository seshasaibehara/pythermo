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

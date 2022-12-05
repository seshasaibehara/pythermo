import numpy as np
import matplotlib.pyplot as plt


def get_mean_eci_set(eci_sets: np.ndarray) -> np.ndarray:
    """Given a matrix of ``eci_sets`` where
    each column corresponds to one set of ecis,
    returns the mean of all the sets

    Parameters
    ----------
    eci_sets : np.ndarray
        Matrix where each column is one set of ecis

    Returns
    -------
    np.ndarray
        A vector of mean ecis

    """

    return np.mean(eci_sets, axis=1)


def get_correlation_matrix(configs: list[dict]) -> np.ndarray:
    """Given ccasm query json style list of
    ``configs`` containing correlation information,
    makes a numpy matrix of correlations where
    each row corresponds to correlations of one
    config

    Parameters
    ----------
    configs : list[dict]
        ccasm query json style list
        containing correlation info

    Returns
    -------
    np.ndarray
        Correlation matrix where each row
        corresponds to correlation values
        of one config

    """

    return np.array([config["corr"] for config in configs])


def get_predicted_formation_energies(
    correlation_matrix: np.ndarray, eci_sets: np.ndarray
) -> np.ndarray:
    """Given a ``correlation_matrix`` and ``eci_sets``,
    returns predicted formation energies by multiplying
    correlation matrix with eci sets

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Correlation matrix where each row is correlation
        values of one config
    eci_sets : np.ndarray
        eci values where each column is a set of ecis

    Returns
    -------
    np.ndarray
        predicted formation energies where each
        column corresponds to formation energies
        for a set of ecis

    """
    return correlation_matrix @ eci_sets


def get_mean_predicted_formation_energies(
    predicted_formation_energies: np.ndarray,
) -> np.ndarray:
    """Given a matrix of formation energies
    where each column corresponds to formation
    energies of configs for a set of ecis,
    computes the mean of all the sets

    Parameters
    ----------
    predicted_formation_energies : np.ndarray
        Matrix of formation energies where each
        column is formation energy values for a
        set of ecis

    Returns
    -------
    np.ndarray
        A vector of mean formation energies over
        formation energies corresponding to various
        sets of ecis

    """
    return np.mean(predicted_formation_energies, axis=1)


def get_predicted_formation_energy_standard_deviations(
    predicted_formation_energies: np.ndarray,
) -> np.ndarray:
    """Given a matrix of formation energies
    where each column corresponds to formation
    energies of configs for a set of ecis,
    computes the standard deviation of all
    the sets

    Parameters
    ----------
    predicted_formation_energies : np.ndarray
        Matrix of formation energies where each
        column is formation energy values for a
        set of ecis

    Returns
    -------
    np.ndarray
        A vector of standard deviations of formation
        energies corresponding to various
        sets of ecis

    """
    return np.std(predicted_formation_energies, axis=1)


def get_eci_standard_deviations(eci_sets: np.ndarray) -> np.ndarray:
    """Given a matrix of ``eci_sets`` where
    each column corresponds to one set of ecis,
    returns the standard deviations of all the sets

    Parameters
    ----------
    eci_sets : np.ndarray
        Matrix where each column is one set of ecis

    Returns
    -------
    np.ndarray
        A vector of standard deviations of ecis

    """
    return np.std(eci_sets, axis=1)


def get_rms_error_of_fit(
    predicted_formation_energies: np.ndarray, dft_formation_energies: np.ndarray
) -> np.ndarray:
    """Given ``predicted_formation_energies`` and
    ``dft_formation_energies`` returns root mean square
    error of the fit

    Parameters
    ----------
    predicted_formation_energies : np.ndarray
        Predicted formation energies from a fit
    dft_formation_energies : np.ndarray
        DFT computed formation energies

    Returns
    -------
    np.ndarray
        RMS error of the fit

    """
    return np.sqrt(
        np.mean(np.square(predicted_formation_energies - dft_formation_energies))
    )


def plot_error_bar(
    ax: plt.axis,
    xprop: np.ndarray,
    yprop: np.ndarray,
    yerr: np.ndarray,
    **kwargs: float | str,
) -> plt.axis:
    """Makes an error bar on ``yprop`` with ``yerr`` at
    ``xprop`` on a matplotlib ``ax``

    Parameters
    ----------
    ax : plt.axis
        Matplotlib axis
    xprop : np.ndarray
        xprop such as comp
    yprop : np.ndarray
        yprop such as energies
    yerr : np.ndarray
        error on yprop such as standard deviations of
        energies
    **kwargs : float | str
        List of kwargs to pass to  ``ax.errorbar`` routine

    Returns
    -------
    plt.axis
        Matplotlib axis with error bar plotted

    """
    error_bar_options = default_error_bar_options()

    for key, value in kwargs.items():
        error_bar_options[key] = value

    ax.errorbar(xprop, yprop, yerr=yerr, **error_bar_options)

    return ax


def default_error_bar_options() -> dict:
    """Default error bar options

    Returns
    -------
    dict
        Default error bar options

    """
    return dict(fmt="none", capsize=3.0, elinewidth=2.0)

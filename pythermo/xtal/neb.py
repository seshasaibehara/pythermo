import numpy as np
import pandas as pd
import scipy.interpolate as sci
import matplotlib.pyplot as plt


def spline_fit_neb_images(
    neb_distances: np.ndarray, neb_energies: np.ndarray, number_of_spline_images=1000
) -> tuple[np.ndarray, np.ndarray]:
    """Do a spline fit of neb_distances and neb_energies
    and return a smoother grid of neb_distances and
    neb_energies

    Parameters
    ----------
    neb_distances : np.ndarray
        Neb distances as a numpy array
    neb_energies : np.ndarray
        Neb energies as a numpy array

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Smoother grid of neb_distances and neb_energies

    """
    spline_fit = sci.interp1d(neb_distances, neb_energies, kind="cubic")

    spline_distances = np.linspace(
        neb_distances[0], neb_distances[-1], number_of_spline_images
    )
    spline_energies = spline_fit(spline_distances)

    return spline_distances, spline_energies


def plot_neb_data(
    ax: plt.axis,
    neb_data: pd.DataFrame,
    spline_data: pd.DataFrame | None = None,
    number_of_spline_images: int = 1000,
    normalize_reaction_coordinate: bool = True,
    **kwargs: dict,
) -> plt.axis:
    """Plots neb_data and spline_data

    Parameters
    ----------
    ax : plt.axis
        Matplotlib axis
    neb_data : pd.DataFrame
        Neb data as pandas DataFrame
    spline_data : pd.DataFrame | None, optional
        Spline data. Can be provided as a pandas
        DataFrame. If None (default) will be interpolated
    number_of_spline_images: int, optional
        Number of interpolated images if spline
        data is not provided (default=1000)
    normalize_reaction_coordinate : bool, optional
        Normalizes reaction coordinate (default=True)
    **kwargs : dict
        Dict containing info about neb plotting options
        (as "neb_options") or spline plotting options
        (as "spline_options")

    Returns
    -------
    plt.axis
        Matplotlib axis with neb and splines plotted

    """
    # sort keyword arguments
    keys = list(kwargs.keys())
    if "neb_options" not in keys:
        neb_options = default_neb_plotting_options()
    else:
        neb_options = kwargs["neb_options"]

    if "spline_options" not in keys:
        spline_options = default_spline_plotting_options()
    else:
        spline_options = kwargs["spline_options"]

    # get neb data
    neb_distances = np.array(neb_data[1].to_list())
    neb_energies = np.array(neb_data[2].to_list())

    # get spline data
    if spline_data is None:
        spline_distances, spline_energies = spline_fit_neb_images(
            neb_distances, neb_energies, number_of_spline_images
        )
    else:
        spline_distances = np.array(spline_data[1].to_list())
        spline_energies = np.array(spline_data[2].to_list())

    # normalize reaction coordinate
    if normalize_reaction_coordinate:
        neb_distances = neb_distances / neb_distances[-1]
        spline_distances = spline_distances / spline_distances[-1]

    # scatter plot neb data
    ax.scatter(neb_distances, neb_energies, **neb_options)

    # plot spline data
    ax.plot(spline_distances, spline_energies, **spline_options)

    return ax


def default_neb_plotting_options() -> dict:
    """Default neb plotting options

    Returns
    -------
    dict

    """
    return dict(c="black", s=54, marker="o")


def default_spline_plotting_options() -> dict:
    """Default spline plotting options

    Returns
    -------
    dict

    """
    return dict(color="black", linewidth=2.0)

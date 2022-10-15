import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate


def cubic_strain_order_parameters(hencky_strain: np.ndarray) -> np.ndarray:
    """Given hencky strain parameters as a 6 by 1 vector
    where the ordering is Exx, Eyy, Ezz, Eyz, Exz, Exy,
    computes the cubic strain orders parameters

    Parameters
    ----------
    hencky_strain : np.ndarray
        Hencky strain parameters

    Returns
    -------
    np.ndarray
        Cubic strain order parameters

    """
    Exx, Eyy, Ezz, Eyz, Exz, Exy = (
        hencky_strain[0],
        hencky_strain[1],
        hencky_strain[2],
        hencky_strain[3],
        hencky_strain[4],
        hencky_strain[5],
    )
    q1 = (Exx + Eyy + Ezz) / np.sqrt(3)
    q2 = (Exx - Eyy) / np.sqrt(2)
    q3 = ((2 * Ezz) - Exx - Eyy) / np.sqrt(6)
    q4 = Eyz / np.sqrt(2)
    q5 = Exz / np.sqrt(2)
    q6 = Exy / np.sqrt(2)
    return np.array([q1, q2, q3, q4, q5, q6])


def symmetrically_equivalent_e2e3s(
    e2e3s: np.ndarray,
) -> np.ndarray:
    """Given a list of e2e3 vectors, returns symmetrically
    equivalent e2e3s by rotating the vectors
    by 120 and 240 degress respectively

    Parameters
    ----------
    e2e3s : np.ndarray
        list of e2e3 vectors

    Returns
    -------
    np.ndarray
        e2e3s equivalents

    """
    rot_120 = np.array(
        [
            [-1 / 2, -np.sqrt(3) / 2],
            [np.sqrt(3) / 2, -1 / 2],
        ]
    )
    rot_240 = np.array(
        [
            [-1 / 2, np.sqrt(3) / 2],
            [-np.sqrt(3) / 2, -1 / 2],
        ]
    )

    mirror_e2e3s = np.array([[-e2e3[0], e2e3[1]] for e2e3 in e2e3s])
    mirror_e2e3_120s = np.array([rot_120 @ e2e3 for e2e3 in mirror_e2e3s])
    mirror_e2e3_240s = np.array([rot_240 @ e2e3 for e2e3 in mirror_e2e3s])
    e2e3_120s = np.array([rot_120 @ e2e3 for e2e3 in e2e3s])
    e2e3_240s = np.array([rot_240 @ e2e3 for e2e3 in e2e3s])
    return np.vstack(
        (e2e3_120s, e2e3_240s, mirror_e2e3s, mirror_e2e3_120s, mirror_e2e3_240s)
    )


def plot_e2e3_strain_energies_along_with_equivalents(
    ax: plt.axis,
    e2e3s: np.ndarray,
    energies: np.ndarray,
    grid_points_along_one_axis=1000,
    cut_off_energy: float = None,
    **kwargs,
):
    """
    Plot e2-e3 strains along with their energies
    Also includes symmetrically equivalent strains
    in the e2-e3 space

    Parameters
    ----------
    ax : plt.axis
        matplotlib axis
    e2e3s : np.ndarray
        Each row of the matrix is e2-e3 values
        of one config
    energies : np.ndarray
        Energies of each config
    grid_points_along_one_axis : int, optional
        Number of grid points to extrapolate over
        on both axes
    cut_off_energy : float, optional
        If not None (default), replace energies
        with np.nan before plotting
    kwargs : dict
        Plotting options that will be passed
        onto the matplotlib axis

    Returns
    -------
    QuadContourSet
        Return of ax.contourf()

    Raises
    ------
    RuntimeError
        If number of rows of energies is not
        same as number of rows of e2e3s
    """
    if len(energies) != len(e2e3s):
        raise RuntimeError(
            "Number of energies should be equal to number of e2-e3 values"
        )

    plotting_options = default_e2e3_plotting_options()
    for key, value in kwargs.items():
        plotting_options[key] = value

    e2e3s_equiv = symmetrically_equivalent_e2e3s(e2e3s)
    all_e2e3s = np.vstack((e2e3s, e2e3s_equiv))
    all_e2s = all_e2e3s[:, 0]
    all_e3s = all_e2e3s[:, 1]
    all_energies = np.hstack(
        (energies, energies, energies, energies, energies, energies)
    )

    E2, E3 = np.meshgrid(
        np.linspace(np.min(all_e2s), np.max(all_e2s), grid_points_along_one_axis),
        np.linspace(np.min(all_e3s), np.max(all_e3s), grid_points_along_one_axis),
    )

    grid_energies = interpolate.griddata(
        all_e2e3s, all_energies, np.column_stack((np.ravel(E2), np.ravel(E3)))
    )

    grid_energies = grid_energies.reshape(
        (grid_points_along_one_axis, grid_points_along_one_axis)
    )
    if cut_off_energy is not None:
        grid_energies = np.where(grid_energies < cut_off_energy, grid_energies, np.nan)

    return ax.contourf(E2, E3, grid_energies, **plotting_options)


def default_e2e3_plotting_options() -> dict:
    """Default e2-e3 plotting options

    Returns
    -------
    dict

    """
    return dict(cmap="viridis")

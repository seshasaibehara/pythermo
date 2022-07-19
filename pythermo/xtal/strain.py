import numpy as np


def cubic_strain_order_parameters(Exx, Eyy, Ezz, Eyz, Exz, Exy):
    """TODO: Docstring for cubic_strain_order_parameters.

    Parameters
    ----------
    Exx : TODO
    Eyy : TODO
    Ezz : TODO
    Eyz : TODO
    Exz : TODO
    Exy : TODO

    Returns
    -------
    TODO

    """
    q1 = (Exx + Eyy + Ezz) / np.sqrt(3)
    q2 = (Exx - Eyy) / np.sqrt(2)
    q3 = ((2 * Ezz) - Exx - Eyy) / np.sqrt(6)
    q4 = Eyz / np.sqrt(2)
    q5 = Exz / np.sqrt(2)
    q6 = Exy / np.sqrt(2)
    return (q1, q2, q3, q4, q5, q6)

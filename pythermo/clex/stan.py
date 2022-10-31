import os
import json
import numpy as np
import sklearn.model_selection as sk


def get_eci_sets(stan_results: dict, eci_key: str = "eci") -> np.ndarray:
    """Get eci sets from stan_results dictionary
    as a matrix where each column corresponds to
    one set of eci

    Parameters
    ----------
    stan_results : dict
        Stan results dictionary
    eci_key : str, optional
        ECI key in stan_results dictionary
        (default="eci")

    Returns
    -------
    np.ndarray
        A matrix where each column corresponds
        to one set of eci

    """
    return np.array(stan_results[eci_key])


def default_stan_model() -> str:
    """Stan model file with alpha and sigma
    as placeholders

    Returns
    -------
    str

    """
    stan_model = """
    data {
            int n_eci;
            int n_configs;
            matrix[n_configs, n_eci] corr;
            vector[n_configs] energies;
    }
    parameters{
        vector[n_eci] eci;
        }
    model{
        eci ~ normal(0, alpha);
        energies ~ normal(corr * eci, sigma);
    }
    """

    return stan_model


def grid_up_sigma_for_convergence(
    sigmas: np.ndarray,
    alpha: float,
    configs: list[dict],
    kfold_splits: int = 5,
    convergence_dir: str = "./sigma_convergence",
    init_sigma_dir: int = 0,
) -> None:
    """For a given list of sigmas, this function
    goes into the ``convergence_dir``, creates
    sigma directories starting from ``init_sigma_dir``.
    Also divides the given ``configs`` data into kfold
    cross validation and writes json files for test and
    train data in each split folder.

    Parameters
    ----------
    sigmas : np.ndarray
        list of sigma values to grid up
    alpha : float
        alpha value at which to grid up
        sigma values
    configs : list[dict]
        ccasm query json style list of configs
        with correlation information
    kfold_split : int, optional
        Number of KFold splits (default=5)
    convergence_dir : str, optional
        Directory in which to grid up sigmas
        (default="./sigma_convergence")
    init_sigma_dir : int, optional
        Starting index of sigma folders (default=0)

    Returns
    -------
    None

    """
    if not os.path.isdir(convergence_dir):
        os.mkdir(convergence_dir)

    # default stan model with alpha replaced
    stan_model = default_stan_model()
    stan_model = stan_model.replace("alpha", str(alpha))

    # start gridding up sigmas
    for i, sigma in enumerate(sigmas):
        i += init_sigma_dir

        # make one sigma dir
        sigma_dir = os.path.join(convergence_dir, str(i))
        os.mkdir(sigma_dir)

        # get stan model and write a json file
        # with stan model info and alpha value
        ce_model = stan_model.replace("sigma", str(sigma))
        with open(os.path.join(sigma_dir, "info.json"), "w") as f:
            json.dump(dict(stan_model=ce_model, sigma=sigma, alpha=alpha), f)

        # do a kfold split to the data
        # and write test, train data into
        # each split
        _split_and_write_kfold_data(configs, kfold_splits, sigma_dir)

    return None


def grid_up_alpha_for_convergence(
    alphas: np.ndarray,
    sigma: float,
    configs: list[dict],
    kfold_splits: int = 5,
    convergence_dir: str = "./alpha_convergence",
    init_alpha_dir: int = 0,
) -> None:
    """For a given list of alphas, this function
    goes into the ``convergence_dir``, creates
    alpha directories starting from ``init_alpha_dir``.
    Also divides the given ``configs`` data into kfold
    cross validation and writes json files for test and
    train data in each split folder.

    Parameters
    ----------
    alphas : np.ndarray
        list of alpha values to grid up
    sigma : float
        sigma value at which to grid up alphas
    configs : list[dict]
        ccasm query json style configs containing
        correlation information
    kfold_split : int, optional
        Number of kfold splits (default = 5)
    convergence_dir : str, optional
        Convergence directory where alpha
        folders need to be created (default="./alpha_convergence")
    init_alpha_dir : int, optional
        Starting index of alpha folders (default=0)

    Returns
    -------
    None

    """
    if not os.path.isdir(convergence_dir):
        os.mkdir(convergence_dir)

    # default stan model with sigma replaced
    stan_model = default_stan_model()
    stan_model = stan_model.replace("sigma", str(sigma))

    # start gridding up alphas
    for i, alpha in enumerate(alphas):
        i += init_alpha_dir

        # make one alpha dir
        alpha_dir = os.path.join(convergence_dir, str(i))
        os.mkdir(alpha_dir)

        # get stan model and write a json file
        # with stan model info and alpha value
        ce_model = stan_model.replace("alpha", str(alpha))
        with open(os.path.join(alpha_dir, "info.json"), "w") as f:
            json.dump(dict(stan_model=ce_model, alpha=alpha, sigma=sigma), f)

        # do a kfold split to the data
        # and write test, train data into
        # each split
        _split_and_write_kfold_data(configs, kfold_splits, alpha_dir)

    return None


def _split_and_write_kfold_data(
    configs: list[dict], kfold_splits: int, base_dir: str
) -> None:
    """Splits the given config into ``kfold_splits``
    and writes the test and train data into the
    respective split folders

    Parameters
    ----------
    configs : list[dict]
        ccasm query json style configs
        containing correlation info
    kfold_splits : int
        Number of kfold splits
    base_dir : str
        Base dir where "kfold_" directories
        will be written. For example ("alpha_convergence/1")

    Returns
    -------
    None

    """

    corrs = np.array([config["corr"] for config in configs])

    kfold = sk.KFold(kfold_splits)

    for j, (train_indices, test_indices) in enumerate(kfold.split(corrs)):

        train_configs = [configs[train_index] for train_index in train_indices]
        test_configs = [configs[test_index] for test_index in test_indices]

        kfold_dir = os.path.join(base_dir, "kfold_" + str(j))
        os.mkdir(kfold_dir)

        with open(os.path.join(kfold_dir, "train_data.json"), "w") as f:
            json.dump(train_configs, f)

        with open(os.path.join(kfold_dir, "test_data.json"), "w") as f:
            json.dump(test_configs, f)

    return None

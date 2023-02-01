import os
import stan
import json
import random
import pickle
import numpy as np
import thermocore as tc
import pythermo.clex.clex as pyclex
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
    num_chains: int = 4,
    num_samples: int = 1000,
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
            json.dump(
                dict(
                    stan_model=ce_model,
                    sigma=sigma,
                    alpha=alpha,
                    num_chains=num_chains,
                    num_samples=num_samples,
                ),
                f,
            )

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
    num_chains: int = 4,
    num_samples: int = 1000,
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
            json.dump(
                dict(
                    stan_model=ce_model,
                    alpha=alpha,
                    sigma=sigma,
                    num_chains=num_chains,
                    num_samples=num_samples,
                ),
                f,
            )

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


def run_gridded_up_stan_runs(run_dir: str, init_dir: int, end_dir: int) -> None:
    """TODO: Docstring for run_gridded_up_stan_runs.

    Parameters
    ----------
    run_dir : TODO
    init_dir : TODO
    end_dir : TODO

    Returns
    -------
    TODO

    """

    for dir_num in range(init_dir, end_dir + 1):
        dir = os.path.join(run_dir, str(dir_num))

        with open(os.path.join(dir, "info.json"), "r") as f:
            model_info = json.load(f)

        stan_model = model_info["stan_model"]
        num_chains = model_info["num_chains"]
        num_samples = model_info["num_samples"]

        kfold_dir_names = [
            kfold_dir for kfold_dir in os.listdir(dir) if "kfold_" in kfold_dir
        ]

        for kfold_dir_name in kfold_dir_names:
            kfold_dir = os.path.join(dir, kfold_dir_name)

            with open(os.path.join(kfold_dir, "train_data.json"), "r") as f:
                train_configs = json.load(f)

            corr_train = [config["corr"] for config in train_configs]
            energies_train = [config["formation_energy"] for config in train_configs]
            ce_data = {
                "n_eci": len(corr_train[0]),
                "n_configs": len(energies_train),
                "corr": corr_train,
                "energies": energies_train,
            }
            posterior = stan.build(stan_model, data=ce_data)
            fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)
            with open(os.path.join(kfold_dir, "train_results.pkl"), "wb") as f:
                pickle.dump(fit, f, protocol=pickle.HIGHEST_PROTOCOL)

    return None


def mean_kfold_rms_from_gridded_up_stan_runs(run_dir: str, init_dir: int, end_dir: int):
    """TODO: Docstring for analyze_gridded_up_stan_runs.

    Parameters
    ----------
    arg1 : TODO

    Returns
    -------
    TODO

    """
    dir_range = range(init_dir, end_dir + 1)

    mean_rms_errors = []
    for dir_num in dir_range:
        dir = os.path.join(run_dir, str(dir_num))
        kfold_dir_names = [
            kfold_dir for kfold_dir in os.listdir(dir) if "kfold_" in kfold_dir
        ]
        kfold = len(kfold_dir_names)

        mean_rms_at_one_dir = 0
        for kfold_dir in kfold_dir_names:
            with open(os.path.join(dir, kfold_dir, "test_data.json"), "r") as f:
                test_data = json.load(f)

            with open(os.path.join(dir, kfold_dir, "train_results.pkl"), "rb") as f:
                train_results = pickle.load(f)

            test_corrs = pyclex.get_correlation_matrix(test_data)
            test_formation_energies = np.array(
                [config["formation_energy"] for config in test_data]
            )

            eci_sets = get_eci_sets(train_results)
            test_predicted_formation_energies = pyclex.get_predicted_formation_energies(
                test_corrs, eci_sets
            )
            test_mean_predicted_formation_energies = (
                pyclex.get_mean_predicted_formation_energies(
                    test_predicted_formation_energies
                )
            )
            rms_of_fit = pyclex.get_rms_error_of_fit(
                test_mean_predicted_formation_energies, test_formation_energies
            )

            mean_rms_at_one_dir += rms_of_fit

        mean_rms_at_one_dir = mean_rms_at_one_dir / kfold

        mean_rms_errors.append(mean_rms_at_one_dir)

    return mean_rms_errors


def sample_random_eci_sets_for_mc(
    casm_root_dir: str,
    eci_sets: np.ndarray,
    basis_dict: dict,
    num_of_ecis,
    init_eci_dir_number: int = 0,
    include_mean_eci=True,
    casm_clex="formation_energy",
    calctype="default",
    reftype="default",
    bsettype="default",
):
    """TODO: Docstring for write_random_eci_sets.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO

    """
    clex_dir = os.path.join(
        casm_root_dir,
        "cluster_expansions",
        "clex." + casm_clex,
        "calctype." + calctype,
        "ref." + reftype,
        "bset." + bsettype,
    )

    if include_mean_eci:
        mean_eci_set = pyclex.get_mean_eci_set(eci_sets)
        mean_eci_dict = tc.io.casm.append_ECIs_to_basis_data(mean_eci_set, basis_dict)
        mean_eci_dir = os.path.join(clex_dir, "eci.mean")
        os.mkdir(mean_eci_dir)

        with open(
            os.path.join(mean_eci_dir, "eci.json"),
            "w",
        ) as f:
            json.dump(mean_eci_dict, f)

    for eci_number in range(init_eci_dir_number, init_eci_dir_number + num_of_ecis + 1):
        eci_dir = os.path.join(clex_dir, "eci." + str(eci_number))
        os.mkdir(eci_dir)

        # TODO: Should they be truly random, instead of a single column?
        eci_column_number = random.randint(0, len(eci_sets) - 1)
        random_eci_set = eci_sets[:, eci_column_number]
        random_eci_dict = tc.io.casm.append_ECIs_to_basis_data(
            random_eci_set, basis_dict
        )

        with open(os.path.join(eci_dir, "eci.json"), "w") as f:
            json.dump(random_eci_dict, f)

    return None

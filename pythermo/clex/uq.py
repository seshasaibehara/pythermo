import json
import numpy as np
import pythermo.clex.clex as pyclex
from sklearn.decomposition import PCA
import pythermo.clex.binary as pybinary


def get_orbit_branch_index(basis_dict):
    """TODO: Docstring for get_orbit_branch_index.

    Parameters
    ----------
    basis_dict : TODO

    Returns
    -------
    TODO

    """
    return [len(orbit["prototype"]["sites"]) for orbit in basis_dict["orbits"]]


def PCA_of_eci_set(eci_set: np.ndarray):
    """TODO: Docstring for PCA_at_one_iteration.

    Parameters
    ----------
    eci_set : np.ndarray[M, N]
        where M corresponds to number of ECI sets with correct ground states
        at that iteration. N corresponds to number ECI sets

    Returns
    -------
    TODO

    """
    # construct PCA. Not giving number of required components
    # because you don't want to dimension reduction here
    pca = PCA()

    # fit to eci_set - do PCA on eci_set
    pca.fit(eci_set)

    # pca components ordered by highest variance first
    # higher variance -> lower precision
    # lower variance -> higher precision (implies any small deviation
    # of these lower variance components will push it out of the ground state region)
    pca_components = pca.components_

    # get explained variances. Variances of each of the
    # principal components ordered by highest varince first
    pca_explained_variances = pca.explained_variance_

    # get noise variances - something to do with bishop algo
    # finding it to be zero. which is good
    pca_noise_variance = pca.noise_variance_

    # covariance matrix in the original space? (I'm pretty sure)
    # It is equivalent to pca_components.T @ explained_variance @ pca_components
    # (see sklearn documentation)
    # because it has pca_components.T @ explained_variance @ pca_components
    # it is something you can directly pass to when you are sampling gaussians
    pca_covariance_matrix = pca.get_covariance()

    return (
        pca_components,
        pca_explained_variances,
        pca_covariance_matrix,
        pca_noise_variance,
    )


def spurious_and_missing_ground_states(
    true_comps,
    true_energies,
    true_names,
    predicted_comps,
    predicted_energies,
    predicted_names,
):
    """
    Parameters
    ----------
    true_comps: np.array
        True comps used to compute DFT ground states
    true_energies : np.array
        True formation energies used to compute DFT ground states
    true_names : list[str]
        List of all config names used to compute DFT hull
    predicted_comps : np.array
        Large set of comps used by ecis to predict clex hull, which shouldn't break the hull
    predicted_energies : np.array
        Large set of predicted energies used by ecis to predict clex hull, which shouldn't break the hull
    predicted_names : list[str]
        Large set of predicted names used by ecis to predict clex hull, which shouldn't break the hull

    Returns
    -------
    tuple[list[int], list[str], list[int], list[str]]
        Returns a tuple in the following order:
        spurious ground state indices - indices in predicted energies which break the hull
        spurious ground state names - names of configs in predicted energies which break the hull
        missing ground state indices - indices in true energies which do not show up in the predicted hull
            but should
        missing ground state names - names of configs in true energies which do not show up in the
            predicted hull but should

    """
    # see what are the spurious ground states
    predicted_ground_state_indices = np.array(
        pybinary.ground_state_indices(predicted_comps, predicted_energies)
    )
    predicted_ground_state_names = [
        predicted_names[i] for i in predicted_ground_state_indices
    ]
    ground_state_indices = np.array(
        pybinary.ground_state_indices(true_comps, true_energies)
    )
    config_ground_state_names = [true_names[i] for i in ground_state_indices]

    spurious_ground_states = [
        (i, uncalc_ground_name)
        for i, uncalc_ground_name in enumerate(predicted_ground_state_names)
        if uncalc_ground_name not in config_ground_state_names
    ]

    missing_ground_states = [
        (i, ground_name)
        for i, ground_name in enumerate(config_ground_state_names)
        if ground_name not in predicted_ground_state_names
    ]

    spurious_ground_state_indices = predicted_ground_state_indices[
        [i for i, _ in spurious_ground_states]
    ]
    missing_ground_state_indices = ground_state_indices[
        [i for i, _ in missing_ground_states]
    ]

    spurious_ground_state_names = [name for _, name in spurious_ground_states]
    missing_ground_state_names = [name for _, name in missing_ground_states]

    return (
        spurious_ground_state_indices,
        spurious_ground_state_names,
        missing_ground_state_indices,
        missing_ground_state_names,
    )


def sample_from_pca_pancake_and_write_ecis_with_perfect_ground_states(
    pca_components_excluding_mean,
    mean_vector,
    num_samples,
    eci_dict_name,
    prior_covariance,
    avg_eci,
    true_comps,
    true_corrs,
    true_formation_energies,
    true_names,
    uncalculated_comps,
    uncalculated_corrs,
    uncalculated_names,
):
    """Sample and write ecis with perfect ground states

    Parameters
    ----------
    pca_components_excluding_mean : np.ndarray
        Each row is pca component and it is assumed that you already
        excluded the row with the mean vector
    mean_vector : np.ndarray
        sometimes last pca component might not exactly be the mean_vector
        it can also be -last row. hence provide it explicitly
    num_samples : int
        Number eci samples to sample
    eci_dict_name : str
        Name to which ecis should be written to
    prior_covariance : np.array
        prior covariance matrix of rest of the pca components
        should be a num of pca components - 1 x pca_components -1 shape
        will be a diagonal matrix
    avg_eci : np.array
        avg eci around which ecis should be sampled from
        in pca space. generally will be a zero vector of pca_components - 1 shape
    true_comps: np.array
        True comps used to compute DFT ground states
    true_corrs: np.array
        True corrs used to compute predicted energies of configs fit to
    true_formation_energies : np.array
        True formation energies used to compute DFT ground states
    true_names : list[str]
        List of all config names used to compute DFT hull
    uncalculated_comps : np.array
        Large set of comps used by ecis to predict clex hull, which shouldn't break the hull
    uncalculated_corrs : np.array
        Large set of corrs used by ecis to predict clex hull, which shouldn't break the hull
    uncalculated_names : list[str]
        Large set of config names used by ecis to predict clex hull, which shouldn't break the hull

    Returns
    -------
    None

    """
    # sample 10000 eci given eci and prior covariance
    prior_samples = np.random.multivariate_normal(
        avg_eci, prior_covariance, num_samples
    )

    holy_grail_eci = []
    holy_grail_rms = []
    holy_grail_count = 0

    # collect only holy grail ecis
    for pca_sample in prior_samples:
        # multiply each of the coefficient obtained by sampling with the corresponding pca component
        sample = pca_sample[:, np.newaxis] * pca_components_excluding_mean

        sample = np.sum(sample, axis=0)
        sample = sample + mean_vector

        uncalculated_predicted_energies = uncalculated_corrs @ sample
        predicted_energies = true_corrs @ sample

        (
            spurious_ground_state_indices,
            _,
            missing_ground_state_indices,
            _,
        ) = spurious_and_missing_ground_states(
            true_comps,
            true_formation_energies,
            true_names,
            uncalculated_comps,
            uncalculated_predicted_energies,
            uncalculated_names,
        )
        rms = pyclex.get_rms_error_of_fit(predicted_energies, true_formation_energies)

        # print(spurious_ground_state_names)
        # print(missing_ground_state_names)

        # spurious_ground_state_comps = np.array(
        #    [
        #        1 - uncalculated_configs[i]["comp"][0]
        #        for i in spurious_ground_state_indices
        #    ]
        # )

        if (
            len(spurious_ground_state_indices) == 0
            and len(missing_ground_state_indices) == 0
        ):
            #    print("NO SPURIOUS GROUND STATES AND NO MISSING GROUND STATES: HOLY GRAIL")
            #    print("RMS: ", rms)
            #    print(spurious_ground_state_names)
            #    print(missing_ground_state_names)
            holy_grail_eci.append(sample.tolist())
            holy_grail_rms.append(rms)
            # non_spurious_eci.append(sample)
            # non_spurious_rms.append(rms)
            holy_grail_count += 1
            # print("----------------")

    print("Number of samples with HOLY GRAIL: ", holy_grail_count)

    with open(eci_dict_name, "w") as f:
        json.dump(
            dict(
                holy_grail_eci=holy_grail_eci,
                holy_grail_rms=holy_grail_rms,
                holy_grail_count=holy_grail_count,
            ),
            f,
        )


def sample_and_write_ecis_with_perfect_ground_states(
    num_samples,
    eci_dict_name,
    prior_covariance,
    avg_eci,
    true_comps,
    true_corrs,
    true_formation_energies,
    true_names,
    uncalculated_comps,
    uncalculated_corrs,
    uncalculated_names,
):
    """Sample and write ecis with perfect ground states

    Parameters
    ----------
    num_samples : int
        Number eci samples to sample
    eci_dict_name : str
        Name to which ecis should be written to
    prior_covariance : np.array
        prior covariance matrix
    avg_eci : np.array
        avg eci around which ecis should be sampled from
    true_comps: np.array
        True comps used to compute DFT ground states
    true_corrs: np.array
        True corrs used to compute predicted energies of configs fit to
    true_formation_energies : np.array
        True formation energies used to compute DFT ground states
    true_names : list[str]
        List of all config names used to compute DFT hull
    uncalculated_comps : np.array
        Large set of comps used by ecis to predict clex hull, which shouldn't break the hull
    uncalculated_corrs : np.array
        Large set of corrs used by ecis to predict clex hull, which shouldn't break the hull
    uncalculated_names : list[str]
        Large set of config names used by ecis to predict clex hull, which shouldn't break the hull

    Returns
    -------
    None

    """
    # sample 10000 eci given eci and prior covariance
    prior_samples = np.random.multivariate_normal(
        avg_eci, prior_covariance, num_samples
    )

    holy_grail_eci = []
    holy_grail_rms = []
    holy_grail_count = 0

    # collect only holy grail ecis
    for sample in prior_samples:
        uncalculated_predicted_energies = uncalculated_corrs @ sample
        predicted_energies = true_corrs @ sample

        (
            spurious_ground_state_indices,
            _,
            missing_ground_state_indices,
            _,
        ) = spurious_and_missing_ground_states(
            true_comps,
            true_formation_energies,
            true_names,
            uncalculated_comps,
            uncalculated_predicted_energies,
            uncalculated_names,
        )
        rms = pyclex.get_rms_error_of_fit(predicted_energies, true_formation_energies)

        # print(spurious_ground_state_names)
        # print(missing_ground_state_names)

        # spurious_ground_state_comps = np.array(
        #    [
        #        1 - uncalculated_configs[i]["comp"][0]
        #        for i in spurious_ground_state_indices
        #    ]
        # )

        if (
            len(spurious_ground_state_indices) == 0
            and len(missing_ground_state_indices) == 0
        ):
            #    print("NO SPURIOUS GROUND STATES AND NO MISSING GROUND STATES: HOLY GRAIL")
            #    print("RMS: ", rms)
            #    print(spurious_ground_state_names)
            #    print(missing_ground_state_names)
            holy_grail_eci.append(sample.tolist())
            holy_grail_rms.append(rms)
            # non_spurious_eci.append(sample)
            # non_spurious_rms.append(rms)
            holy_grail_count += 1
            # print("----------------")

    print("Number of samples with HOLY GRAIL: ", holy_grail_count)

    with open(eci_dict_name, "w") as f:
        json.dump(
            dict(
                holy_grail_eci=holy_grail_eci,
                holy_grail_rms=holy_grail_rms,
                holy_grail_count=holy_grail_count,
            ),
            f,
        )


def _loop_through_min_max_vals_to_adjust_means(
    mean_eci_vector,
    pca_component,
    min_val,
    max_val,
    grid_num,
    comps,
    formation_energies,
    names,
    predicted_comps,
    predicted_corrs,
    predicted_names,
):
    """TODO: Docstring for _loop_through_min_max_vals_to_adjust_means.

    Returns
    -------
    TODO

    """
    vals_with_no_spurious_and_missing_states = []
    indices_with_no_spurious_and_missing_states = []
    index_to_pad = []
    vals_to_pad = []
    for ind, val in enumerate(np.linspace(min_val, max_val, grid_num)):
        # this new_eci_vector doesn't fall on the unit sphere
        # the approximation is that this is enough to determine
        # when projected whether it will break the bounds or not
        new_eci_vector = mean_eci_vector + (val * pca_component)

        (
            spurious_indices,
            _,
            missing_indices,
            _,
        ) = spurious_and_missing_ground_states(
            true_comps=comps,
            true_energies=formation_energies,
            true_names=names,
            predicted_comps=predicted_comps,
            predicted_energies=predicted_corrs @ new_eci_vector,
            predicted_names=predicted_names,
        )

        if len(spurious_indices) == 0 and len(missing_indices) == 0:
            vals_with_no_spurious_and_missing_states.append(val)
            indices_with_no_spurious_and_missing_states.append(ind)
            if ind == 0:
                index_to_pad.append(ind)
                vals_to_pad.append(val)
                break
            if ind == grid_num - 1:
                index_to_pad.append(ind)
                vals_to_pad.append(val)
                break

    return (
        indices_with_no_spurious_and_missing_states,
        vals_with_no_spurious_and_missing_states,
        index_to_pad,
        vals_to_pad,
    )


def readjust_mean_vector_by_gridding_pca_vector(
    mean_eci_vector,
    pca_component,
    min_val,
    max_val,
    grid_num,
    pad,
    pad_scale,
    comps,
    formation_energies,
    names,
    predicted_comps,
    predicted_corrs,
    predicted_names,
):
    """TODO: Docstring for readjust_mean_vector.

    Parameters
    ----------
    pca_component : TODO
    comps : TODO
    formation_energies : TODO
    names : TODO
    uncalculated_comps : TODO
    uncalculated_corrs : TODO
    uncalculated_names : TODO

    Returns
    -------
    TODO

    """

    # check if the indices are a continuous list. If not something weird is happening
    # raise a warning if it happens
    (
        indices_with_no_spurious_and_missing_states,
        vals_with_no_spurious_and_missing_states,
        index_to_pad,
        vals_to_pad,
    ) = _loop_through_min_max_vals_to_adjust_means(
        mean_eci_vector=mean_eci_vector,
        pca_component=pca_component,
        min_val=min_val,
        max_val=max_val,
        grid_num=grid_num,
        comps=comps,
        formation_energies=formation_energies,
        names=names,
        predicted_comps=predicted_comps,
        predicted_corrs=predicted_corrs,
        predicted_names=predicted_names,
    )

    while len(index_to_pad) != 0:
        if not pad:
            print("WARNING: Please turn on pad to True!")
            break
        else:
            if index_to_pad[0] == 0:
                min_val = vals_to_pad[0] * pad_scale
            if index_to_pad[0] == grid_num - 1:
                max_val = vals_to_pad[0] * pad_scale
            print("Padding with new values: ")
            print("New min val: ", min_val)
            print("New max val: ", max_val)
            (
                indices_with_no_spurious_and_missing_states,
                vals_with_no_spurious_and_missing_states,
                index_to_pad,
                vals_to_pad,
            ) = _loop_through_min_max_vals_to_adjust_means(
                mean_eci_vector=mean_eci_vector,
                pca_component=pca_component,
                min_val=min_val,
                max_val=max_val,
                grid_num=grid_num,
                comps=comps,
                formation_energies=formation_energies,
                names=names,
                predicted_comps=predicted_comps,
                predicted_corrs=predicted_corrs,
                predicted_names=predicted_names,
            )

    indices_with_no_spurious_and_missing_states = np.array(
        indices_with_no_spurious_and_missing_states
    )
    vals_with_no_spurious_and_missing_states = np.array(
        vals_with_no_spurious_and_missing_states
    )
    test_indices = (
        indices_with_no_spurious_and_missing_states
        - indices_with_no_spurious_and_missing_states[0]
    )

    if not np.allclose(test_indices, list(range(0, len(test_indices)))):
        raise RuntimeWarning(
            "Please check your results. Something weird happened in gridding"
        )

    # reorient your mean

    adjusted_mean = (mean_eci_vector) + (
        (
            vals_with_no_spurious_and_missing_states[0]
            + vals_with_no_spurious_and_missing_states[-1]
        )
        / 2
    ) * pca_component

    # normalize your adjusted mean
    adjusted_mean_normalized = adjusted_mean / np.linalg.norm(adjusted_mean)

    # angle between adjusted mean and original mean
    angle_between_adjusted_mean_and_original = np.arccos(
        np.dot(adjusted_mean_normalized, mean_eci_vector)
    ) * (180 / np.pi)

    return (
        adjusted_mean,
        adjusted_mean_normalized,
        angle_between_adjusted_mean_and_original,
        vals_with_no_spurious_and_missing_states,
        indices_with_no_spurious_and_missing_states,
    )


def grid_up_all_pca_components(
    mean_eci_vector, pca_gridded_components, pca_gridded_bounds, num_samples
):
    """Given a mean eci vector, and rest of the pca components
    which form the null space of the mean eci vector, make num_samples
    eci vectors which is mean eci vector + sum_i (coeff_i * pca_component_i)
    where coeff_i is determined by the bounds by pca_gridded_bounds

    Jitting it increases the efficiency after 1000 samples or so

    Parameters
    ----------
    mean_eci_vector : np.ndarray
    pca_gridded_components :  np.ndarray
    pca_gridded_bounds : np.ndarray
    num_samplse : int

    Returns
    -------
    all_samples : np.ndarray
        Each row in all_samples determines a new eci vector in the grid

    """
    all_samples = []

    for _ in range(num_samples):
        new_sample = mean_eci_vector
        # for pca_component, pca_bound in zip(
        #    [pca_gridded_components[0]], [pca_gridded_bounds[0]]
        # ):
        for pca_component, pca_bounds in zip(
            pca_gridded_components, pca_gridded_bounds
        ):
            coeff = np.random.uniform(pca_bounds[0], pca_bounds[1], 1)
            new_sample = new_sample + (coeff * pca_component)

        all_samples.append(new_sample)

    return all_samples


def avg_and_normalize_eci_sets_and_do_sanity_checks(eci_set_dict, verbose=True):
    """TODO: Docstring for avg_and_normalize_eci_sets_and_do_sanity_checks.

    Parameters
    ----------
    eci_set_dict : TODO

    Returns
    -------
    TODO

    """
    ecis_set = np.array(eci_set_dict["holy_grail_eci"])
    ecis_set_count = eci_set_dict["holy_grail_count"]
    ecis_set_rms = eci_set_dict["holy_grail_rms"]

    # get the average of eci of all the 2351 samples
    ecis_set_avg = np.average(ecis_set, axis=0)

    # normalize the average
    ecis_set_avg_normalized = ecis_set_avg / np.linalg.norm(ecis_set_avg)
    # check if it's normalized

    # take all the ecis and normalize them individually
    ecis_set_norms = np.linalg.norm(ecis_set, axis=1)
    ecis_set_normalized = ecis_set / ecis_set_norms[:, np.newaxis]

    ecis_set_avg_normalized_dot_products = np.dot(
        ecis_set_normalized, ecis_set_avg_normalized
    )
    ecis_set_angles = np.arccos(ecis_set_avg_normalized_dot_products) * (180 / np.pi)

    # doing PCA on normalized vectors
    # once you normalized all the vectors -> Do PCA on it
    (
        pca_components_normalized,
        pca_explained_variances_normalized,
        pca_covariance_matrix_normalized,
        pca_noise_variance_normalized,
    ) = PCA_of_eci_set(ecis_set_normalized)

    if verbose:
        # check the dot products before normalizing also
        print("Number of samples: ", ecis_set_count)
        print(
            "Norm of averaged and normalized vector should be 1: ",
            np.linalg.norm(ecis_set_avg_normalized),
        )
        # once normalized check each row of ECI has norm equal to 1
        ecis_set_normalized_norms = np.linalg.norm(ecis_set_normalized, axis=1)
        print(
            "Are all the norms of ecis 1: ",
            np.all(np.isclose(ecis_set_normalized_norms, 1.0)),
        )
        # check if the average normalized is the same as avg found after normalizing each of them individually
        ecis_set_normalized_avg = np.average(ecis_set_normalized, axis=0)
        print(
            "Norm of normalized and averaged vector should not be 1: ",
            np.linalg.norm(ecis_set_normalized_avg),
        )
        # check the angle between avg normalized and normalized average
        # do the dot product
        ecis_set_normalized_avg_and_avg_normalized_dot_product = np.dot(
            ecis_set_avg_normalized, ecis_set_normalized_avg
        )
        print(
            "Angle between normalized avg and avg normalized vector is: ",
            np.arccos(
                ecis_set_normalized_avg_and_avg_normalized_dot_product
                / (
                    np.linalg.norm(ecis_set_normalized_avg)
                    * np.linalg.norm(ecis_set_avg_normalized)
                )
            )
            * (180 / np.pi),
        )
        print(
            "Difference between average normalized and normalized avg (after normalizing to make avg fall on the unit sphere): ",
            np.linalg.norm(
                ecis_set_avg_normalized
                - ecis_set_normalized_avg / np.linalg.norm(ecis_set_normalized_avg)
            ),
        )
        print("Noise variance should be 0: ", pca_noise_variance_normalized)

    return (
        ecis_set_avg,
        ecis_set_avg_normalized,
        ecis_set_normalized,
        pca_components_normalized,
        pca_explained_variances_normalized,
        pca_covariance_matrix_normalized,
        pca_noise_variance_normalized,
        ecis_set_count,
        ecis_set_rms,
        ecis_set_angles,
    )

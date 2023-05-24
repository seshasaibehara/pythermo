import os
import ase
import json
import pandas as pd
import pythermo.jobs as pyjobs


def get_required_data_for_ml_fitting_from_casm_configs(
    selected_configurations: list[dict], calctype: str = "default"
):
    """TODO: Docstring for get_required_data_for_ml_fitting_from_casm_configs.

    Parameters
    ----------
    selected_configurations : TODO
    calctype : TODO, optional

    Returns
    -------
    TODO

    """

    calc_dirs = pyjobs.casm_jobs._get_calctype_dirs(selected_configurations, calctype)

    all_configs_data = pd.DataFrame(columns=["path", "energy", "forces", "ase_atoms"])

    for calc_dir in calc_dirs:
        # read proerties.calc.json
        properties_path = os.path.join(calc_dir, "properties.calc.json")
        with open(properties_path, "r") as f:
            properties = json.load(f)

        # get properties dataframe and add path column
        config_data = get_required_data_for_ml_fitting_casm_properties_json(properties)
        config_data["path"] = [properties_path]

        # append to already existing config data
        all_configs_data = pd.concat([all_configs_data, config_data], ignore_index=True)

    return all_configs_data


def get_required_data_for_ml_fitting_casm_properties_json(properties: dict):
    """TODO: Docstring for get_required_data_for_ml_fitting_casm_properties_json.

    Parameters
    ----------
    properties_json : TODO

    Returns
    -------
    TODO

    """

    lattice = properties["lattice_vectors"]
    forces = properties["atom_properties"]["force"]["value"]
    coords = properties["atom_coords"]
    coord_mode = properties["coordinate_mode"]
    energy = properties["global_properties"]["energy"]["value"]

    if coord_mode == "Direct":
        ase_struc = ase.Atoms(scaled_positions=coords, pbc=True, cell=lattice)

    else:
        ase_struc = ase.Atoms(positions=coords, pbc=True, cell=lattice)

    return pd.DataFrame(
        [[energy, forces, ase_struc.todict()]],
        columns=["energy", "forces", "ase_atoms"],
    )

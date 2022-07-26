import os
import json
import numpy as np
import pymatgen.core as pmgcore
import pymatgen.io.vasp as pmgvasp
from pymatgen.io.cif import CifWriter


def _get_file_paths_as_a_string(file_type, calctype_dirs):
    """TODO: Docstring for _get_file_paths_as_a_string.

    Parameters
    ----------
    file_type : TODO
    calctype_dirs : TODO

    Returns
    -------
    TODO

    """
    file_paths = [
        os.path.join(calctype_dir, file_type) for calctype_dir in calctype_dirs
    ]
    file_path_str = ""
    for file_path in file_paths:
        file_path_str += file_path + "\n"

    return file_path_str


def _get_config_names(selected_configurations):
    """TODO: Docstring for _get_config_names.

    Parameters
    ----------
    selected_configurations : TODO

    Returns
    -------
    TODO

    """
    return [config["name"] for config in selected_configurations]


def _get_calctype_dirs(selected_configurations, calctype):
    """TODO: Docstring for _get_calctype_dirs.

    Parameters
    ----------
    selected_configurations : TODO
    calctype : TODO

    Returns
    -------
    TODO

    """
    return [
        os.path.join("training_data", config_name, "calctype." + calctype)
        for config_name in _get_config_names(selected_configurations)
    ]


def toss_file_str(
    selected_configurations,
    calctype="default",
    write_incar=True,
    write_poscar=True,
    write_kpoints=True,
    write_potcar=True,
    write_relaxandstatic=True,
):
    """TODO: Docstring for write_toss_file.

    Parameters
    ----------
    selected_configurations : TODO
    calctype : TODO, optional
    write_incar : TODO, optional
    write_poscar : TODO, optional
    write_kpoints : TODO, optional
    write_potcar : TODO, optional
    write_relaxandstatic : TODO, optional

    Returns
    -------
    TODO

    """
    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)

    toss_file_str = ""
    if write_incar:
        toss_file_str += _get_file_paths_as_a_string("INCAR", calctype_dirs)
    if write_kpoints:
        toss_file_str += _get_file_paths_as_a_string("KPOINTS", calctype_dirs)
    if write_poscar:
        toss_file_str += _get_file_paths_as_a_string("POSCAR", calctype_dirs)
    if write_potcar:
        toss_file_str += _get_file_paths_as_a_string("POTCAR", calctype_dirs)
    if write_relaxandstatic:
        toss_file_str += _get_file_paths_as_a_string("relaxandstatic.sh", calctype_dirs)

    return toss_file_str


def submit_script_str(selected_configurations, queue_type: str, calctype="default"):
    """TODO: Docstring for submit_script_str.

    Parameters
    ----------
    selected_configurations : TODO
    queue_type : TODO
    calctype : TODO, optional

    Returns
    -------
    TODO

    """
    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)

    submit_str = "#!/bin/bash\n"
    submit_str += "configs=(\n"
    for calctype_dir in calctype_dirs:
        submit_str += " " + calctype_dir + "\n"
    submit_str += ")\n"

    submit_str += 'for i in "${configs[@]}"; do\n'
    submit_str += " cd $i\n"
    if queue_type == "pbs":
        submit_str += " qsub relaxandstatic.sh\n"
    if queue_type == "slurm":
        submit_str += " sbatch relaxandstatic.sh\n"
    submit_str += " cd ../../../../\n"
    submit_str += "done\n"

    return submit_str


def _job_names(config_names):
    """TODO: Docstring for _job_names.

    Parameters
    ----------
    config_names : TODO

    Returns
    -------
    TODO

    """
    return [name.replace("/", ".") for name in config_names]


def copy_relaxandstatic_cmds(
    selected_configurations, relaxandstatic_path=None, calctype="default"
):
    """TODO: Docstring for copy_relaxandstatic_cmd.

    Parameters
    ----------
    selected_configurations : TODO
    calctype : TODO, optional

    Returns
    -------
    TODO

    """
    if relaxandstatic_path is None:
        relaxandstatic_path = os.path.join(
            "training_data", "settings", "calctype." + calctype, "relaxandstatic.sh"
        )

    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)

    return [
        "cp " + relaxandstatic_path + " " + calctype_dir + "/"
        for calctype_dir in calctype_dirs
    ]


def change_job_names_cmds(selected_configurations, queue, calctype="default"):
    """TODO: Docstring for change_job_names_cmds.

    Parameters
    ----------
    selected_configurations : TODO
    queue : TODO
    calctype : TODO, optional

    Returns
    -------
    TODO

    """
    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)
    job_names = _job_names(_get_config_names(selected_configurations))

    if queue == "pbs":
        return [
            'sed -i "s/.*#PBS -N.*/#PBS -N '
            + job_name
            + '/g" '
            + os.path.join(calctype_dir, "relaxandstatic.sh")
            for job_name, calctype_dir in zip(job_names, calctype_dirs)
        ]
    elif queue == "slurm":
        return [
            'sed -i "s/.*#SBATCH -J.*/#SBATCH -J '
            + job_name
            + '/g" '
            + os.path.join(calctype_dir, "relaxandstatic.sh")
            for job_name, calctype_dir in zip(job_names, calctype_dirs)
        ]
    else:
        raise ValueError(str(queue) + " is not a valid queue.")


def write_incars(selected_configurations, incars, calctype):
    """TODO: Docstring for write_incar_cmds.

    Parameters
    ----------
    selected_configurations : TODO
    incars : TODO
    calctype : TODO

    Returns
    -------
    TODO

    """
    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)
    [
        incar.write_file(os.path.join(calctype_dir, "INCAR"))
        for incar, calctype_dir in zip(incars, calctype_dirs)
    ]
    return None


def modify_incar_magmoms(
    selected_configurations, new_magmoms: dict, calctype="default", noncollinear=False
):
    """TODO: Docstring for modify_incar_magmoms.

    Parameters
    ----------
    selected_configurations : TODO
    new_magmoms : TODO
    calctype : TODO, optional

    Returns
    -------
    TODO

    """
    if noncollinear:
        raise NotImplementedError("NONCOLLINEAR calculations not yet implemented")

    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)
    incars = [
        pmgvasp.Incar.from_file(os.path.join(calctype_dir, "INCAR"))
        for calctype_dir in calctype_dirs
    ]
    poscars = [
        pmgvasp.Poscar.from_file(os.path.join(calctype_dir, "POSCAR"))
        for calctype_dir in calctype_dirs
    ]

    provided_atom_types = new_magmoms["atom_types"]
    provided_magmoms = new_magmoms["magmoms"]

    modifed_incars = []
    for incar, poscar in zip(incars, poscars):

        # if noncollinear calculations are found in INCAR, STOP!
        incar_keys = list(incar.keys())
        if "LNONCOLLINEAR" in incar_keys or "LSORBIT" in incar_keys:
            raise NotImplementedError("NONCOLLINEAR calculations not yet implemented")

        # get the updated magnetic moments for each atom type (assumes each atom type has the same value of magmom)
        # if magenetic moment for an atom type is not provided, use the one already in incar
        modified_magmoms = []
        index = 0
        for n, atom_type in zip(poscar.natoms, poscar.site_symbols):
            # change magmom
            if atom_type in provided_atom_types:
                magmom = [provided_magmoms[provided_atom_types.index(atom_type)]] * n
            else:
                magmom = incar["MAGMOM"][index : index + n]
            index += n

            modified_magmoms.extend(magmom)

        # Update the value in INCAR
        incar["MAGMOM"] = modified_magmoms

        modifed_incars.append(incar)

    return modifed_incars


def get_casm_commands_to_turn_off_given_configurations(
    configurations_info: dict, casm_executable="ccasm"
):
    """TODO: Docstring for get_casm_commands_to_turn_off_given_configurations.

    Parameters
    ----------
    configurations_info : TODO

    Returns
    -------
    TODO

    """
    scels = configurations_info["scelnames"]

    config_range_for_scel = [
        list(range(config_range["start"], config_range["stop"] + 1))
        if type(config_range) is dict
        else config_range
        for config_range in configurations_info["range"]
    ]

    config_names = [
        "" + str(scel) + "/" + str(config_number)
        for scel, config_range in zip(scels, config_range_for_scel)
        for config_number in config_range
    ]

    casm_commands = [
        casm_executable
        + " select --set-off 're(configname, "
        + '"'
        + config_name
        + '"'
        + ")'"
        for config_name in config_names
    ]

    return casm_commands


def visualize_magnetic_moments_from_casm_structure(
    selected_configurations, noncollinear=False
):
    """TODO: Docstring for visualize_magnetic_moments.

    Parameters
    ----------
    selected_configurations : TODO
    calctype : TODO, optional

    Returns
    -------
    TODO

    """
    config_names = _get_config_names(selected_configurations)
    casm_structure_files = [
        os.path.join("training_data", config_name, "structure.json")
        for config_name in config_names
    ]

    casm_structures = []
    for structure_file in casm_structure_files:
        with open(structure_file, "r") as f:
            casm_structures.append(json.load(f))

    lattices_of_all_configs = [
        np.array(casm_structure["lattice_vectors"])
        for casm_structure in casm_structures
    ]

    cart_coords_of_all_configs = [
        np.array(casm_structure["atom_coords"]) for casm_structure in casm_structures
    ]

    atom_types_of_all_configs = [
        casm_structure["atom_type"] for casm_structure in casm_structures
    ]

    if not noncollinear:
        magspin_values_for_all_configs = [
            np.append(
                np.zeros((len(casm_structure["atom_type"]), 2)),
                np.array(casm_structure["mol_properties"]["Cmagspin"]["value"]),
                axis=1,
            )
            for casm_structure in casm_structures
        ]
    else:
        magspin_values_for_all_configs = [
            np.array(casm_structure["mol_properties"]["Cmagspin"]["value"])
            for casm_structure in casm_structures
        ]

    pymatgen_structures = [
        pmgcore.Structure(
            lattice,
            atom_types,
            cart_coords,
            site_properties={"magmom": magspins},
            coords_are_cartesian=True,
        )
        for lattice, cart_coords, atom_types, magspins in zip(
            lattices_of_all_configs,
            cart_coords_of_all_configs,
            atom_types_of_all_configs,
            magspin_values_for_all_configs,
        )
    ]

    mcifs = [
        CifWriter(pmg_structure, write_magmoms=True)
        for pmg_structure in pymatgen_structures
    ]

    return mcifs

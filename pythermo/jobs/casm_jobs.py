import os
import json
import shlex
import numpy as np
import pymatgen.core as pmgcore
import pymatgen.io.vasp as pmgvasp
from pymatgen.io.cif import CifWriter


def _get_file_paths_as_a_string(file_type: str, calctype_dirs: list[str]) -> str:
    """Return file paths in casm configurations' calctype directory
    filetype can be anything like INCAR, KPOINTS, POTCAR, etc

    Parameters
    ----------
    file_type : str
        file_type like INCAR, KPOINTS, POTCAR etc.
    calctype_dirs : list[str]
        list of all configurations calctype directories

    Returns
    -------
    str
        A long string with all the file paths

    """
    file_paths = [
        os.path.join(calctype_dir, file_type) for calctype_dir in calctype_dirs
    ]
    file_path_str = ""
    for file_path in file_paths:
        file_path_str += file_path + "\n"

    return file_path_str


def _get_config_names(selected_configurations: list[dict]) -> list[str]:
    """Given a list of configurations in ccasm query json format,
    return a list of config names example ["SCEL1_1_1_1_0_0_0/0"]

    Parameters
    ----------
    selected_configurations : list[dict]
        list of configurations in ccasm query json format

    Returns
    -------
    list[str]
        List of confignames

    """
    return [config["name"] for config in selected_configurations]


def _get_calctype_dirs(selected_configurations: list[dict], calctype: str) -> list[str]:
    """Given a list of configurations in ccasm query json format and calctype,
    returns list of configurations' calctype directories
    (training_data/configname/calctype.calctype)

    Parameters
    ----------
    selected_configurations : list[dict]
        list of configurations in ccasm query json format
    calctype : str
        ccasm query calctype

    Returns
    -------
    list[str]
        list of configurations' calctype directories

    """
    return [
        os.path.join("training_data", config_name, "calctype." + calctype)
        for config_name in _get_config_names(selected_configurations)
    ]


def toss_file_str(
    selected_configurations: list[dict],
    calctype: str = "default",
    write_incar: bool = True,
    write_poscar: bool = True,
    write_kpoints: bool = True,
    write_potcar: bool = True,
    write_relaxandstatic: bool = True,
) -> str:
    """For a given list of configurations in ccasm query json format,
    return a string which can be used in conjuction with rsync --files-from
    By default, contains configuration directories with INCAR, KPOINTS,
    POSCAR, POTCAR, and relaxandstatic.sh

    Parameters
    ----------
    selected_configurations : list[dict]
        list of configurations in ccasm query json format
    calctype : str, optional
        ccasm calctype
    write_incar : bool, optional
        whether to include incar (default=True)
    write_poscar : bool, optional
        whether to include poscar (default=True)
    write_kpoints : bool, optional
        whether to include kpoints (default=True)
    write_potcar : bool, optional
        whether to include potcar (default=True)
    write_relaxandstatic : bool, optional
        whether to include relaxandstatic.sh (default=True)

    Returns
    -------
    str
        When written to a file can be used with rsync --files-from

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


def submit_script_str(
    selected_configurations: list[dict], queue_type: str, calctype: str = "default"
) -> str:
    """Given a list of configurations in ccasm query json format and a queue type
    (slurm or pbs), return a submit string which can be executed as bash script,
    when written to a file to submit the calculations.

    Parameters
    ----------
    selected_configurations : list[dict]
        configurations in ccasm query json format
    queue_type : str
        HPC queue type (slurm or pbs)
    calctype : str, optional
        ccasm calctype

    Returns
    -------
    str
        A string when written to a file can be executed as bash file
        to submit calculations

    Raises
    ------
    RuntimeError
        When ``queue_type`` is neither pbs or slurm

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
    elif queue_type == "slurm":
        submit_str += " sbatch relaxandstatic.sh\n"
    else:
        raise RuntimeError("Invalid queue (" + queue_type + ")")
    submit_str += " cd ../../../../\n"
    submit_str += "done\n"

    return submit_str


def _job_names(config_names: list[str]) -> list[str]:
    """Given a list of configuration names, replace
    "/" to "." which can be used a job name

    Parameters
    ----------
    config_names : list[str]
        list of configuration names

    Returns
    -------
    list[str]
        list of configuration names where "/" is replaced by "."

    """
    return [name.replace("/", ".") for name in config_names]


def copy_relaxandstatic_cmds(
    selected_configurations: list[dict],
    relaxandstatic_path: str = None,
    calctype: str = "default",
) -> list[str]:
    """For a given list of configurations in ccasm query json format,
    write out copy commands for relaxandstatic.sh file from training_data/givencalctype/settings/
    (if an explicit path is not given using ``relaxandstatic_path``) to
    each configuration's calctype folder

    Parameters
    ----------
    selected_configurations : list[dict]
        list of configurations in ccasm query json format
    relaxandstatic_path : str, optional
        path to relaxandstatic.sh (default=training_data/calctype.given/settings/relaxandstatic.sh
    calctype : str, optional
        ccasm calctype

    Returns
    -------
    list[str]
        List of copy commands to copy relaxandstatic.sh to configuration's calctype
        directory

    """
    if relaxandstatic_path is None:
        relaxandstatic_path = os.path.join(
            "training_data", "settings", "calctype." + calctype, "relaxandstatic.sh"
        )

    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)

    return [
        shlex.split("cp " + relaxandstatic_path + " " + calctype_dir + "/")
        for calctype_dir in calctype_dirs
    ]


def change_job_names_cmds(
    selected_configurations: list[dict], queue: str, calctype: str = "default"
) -> list[str]:
    """List of sed commands to change job names for calculations in a given list of
    selected configurations and calctype. If ``queue`` is ``slurm``, the sed
    command will replace the ``#SBATCH -J``, line with job name as configname.
    And if the ``queue`` is ``pbs``, ``#PBS -N``, line will be replaced.

    Parameters
    ----------
    selected_configurations : list[dict]
        list of configurations in ccasm query json format
    queue : str
        cluster queue type, accepted arguments pbs or slurm
    calctype : str, optional
        ccasm calctype (default="default")

    Returns
    -------
    list[str]
        List of sed commands, when executed with os.system will change
        job names in relaxandstatic.sh file

    Raises
    ------
    RuntimeError
        If a queue type provided is neither slurm or pbs

    """
    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)
    job_names = _job_names(_get_config_names(selected_configurations))

    if queue == "pbs":
        return [
            shlex.split(
                'sed -i "s/.*#PBS -N.*/#PBS -N '
                + job_name
                + '/g" '
                + os.path.join(calctype_dir, "relaxandstatic.sh")
            )
            for job_name, calctype_dir in zip(job_names, calctype_dirs)
        ]
    elif queue == "slurm":
        return [
            shlex.split(
                'sed -i "s/.*#SBATCH -J.*/#SBATCH -J '
                + job_name
                + '/g" '
                + os.path.join(calctype_dir, "relaxandstatic.sh")
            )
            for job_name, calctype_dir in zip(job_names, calctype_dirs)
        ]
    else:
        raise RuntimeError(str(queue) + " is not a valid queue.")


def write_incars(
    selected_configurations: list[dict], incars: list[pmgvasp.Incar], calctype: str
) -> None:
    """Write pymatgen Incars for ``selected_configurations`` in a given ``calctype``
    directory

    Parameters
    ----------
    selected_configurations : list[dict]
        list of configurations in ccasm query json format
    incars : list[pmgvasp.Incar]
        list of pymatgen Incars
    calctype : str
        ccasm calctype where you want to write INCARs

    Returns
    -------
    None

    """
    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)
    [
        incar.write_file(os.path.join(calctype_dir, "INCAR"))
        for incar, calctype_dir in zip(incars, calctype_dirs)
    ]
    return None


# TODO: Allow to change magnetic moments of each element individually
# TODO: Allow for non-collinear calculations
def modify_incar_magmoms(
    selected_configurations: list[dict],
    new_magmoms: dict,
    calctype: str = "default",
    noncollinear: bool = False,
) -> list[pmgvasp.Incar]:
    """For a given list of configurations in ccasm query json format,
    this function returns a list of pymatgen INCARs with magnetic moments
    modified using ``new_magmoms``. For example, if you want to modify
    magnetic moments of Li, O (magmoms of Mn, Ni will be kept)
    in Li2MnNiO8 structure:\n
    Example ``new_magmoms``:\n
    {\n
        "atom_types": ["Li", "O"],\n
        "magmoms" : [0.6, 0.4],\n
    }\n

    Parameters
    ----------
    selected_configurations : list[dict]
        List of configurations in ccasm query json format
    new_magmoms : dict
        dictionary containing new ``magmoms`` for given ``atom_types``
    calctype : str, optional
        ccasm calctype (default="default")
    noncollinear : bool, optional
        If the calculations are noncollinear (default=False)

    Returns
    -------
    list[pymgvasp.Incar]
        List of pymatgen Incars with modified magnetic moments
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

        # get the updated magnetic moments for each atom type
        # (assumes each atom type has the same value of magmom)
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
) -> list[str]:
    """Given ``configurations_info``, return a list of ccasm commands to
    turn them off. \n

    Example ``configurations_info``:\n
    {\n
     "scelnames" : ["SCEL1..", "SCEL2...", ....]\n
     "range": [{"start":0, "stop":4}, [1, 3, 4,], ....]\n
    }\n

    "scelnames" is a list containing ccasm supercell names for which configuration list will
    be constructed. "range" is a list with each element corresponding to configurations in
    a supercell corresponding to the same index in "scelnames". The configurations can either
    be given as a dictionary with "start" and "stop" numbers (stop is included) or just a
    list of numbers

    Parameters
    ----------
    configurations_info : dict
        Read the example above to understand how it's interpreted.

    Returns
    -------
    list[str]
        A list of ccasm commands which can be executed to turn off
        given configurations

    Raises
    ------
    RuntimeError
        If length of ``configurations_info["scelnames"]`` doesn't match
        with length of ``configurations_info["range"]``

    """
    if len(configurations_info["scelnames"]) != len(configurations_info["range"]):
        raise RuntimeError("length of scelnames doesn't match with length of range")

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
        shlex.split(
            casm_executable
            + " select --set-off 're(configname, "
            + '"'
            + config_name
            + '"'
            + ")'"
        )
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


def visualize_magmoms_from_outcar(outcar_path: str, contcar_path: str) -> CifWriter:
    """TODO: Docstring for visualize_magmoms_from_outcar.

    Parameters
    ----------
    outcar_path : TODO
    contcar_path : TODO

    Returns
    -------
    TODO

    """
    magmoms_dict = pmgvasp.Outcar(outcar_path).magnetization
    magmoms = [magmom_dict["tot"] for magmom_dict in magmoms_dict]

    contcar_structure = pmgvasp.Poscar.from_file(contcar_path).structure
    contcar_structure.add_site_property("magmom", magmoms)

    return CifWriter(contcar_structure, write_magmoms=True)


def remove_completed_calculations(
    selected_configurations: list[dict], calctype: str = "default"
) -> str:
    """For a given list of configurations (in ccasm query json format),
    a status.json file will be read from training_data/configname/calctype/status.json.
    If status["status"] is "complete", a line will be added to the output as
    "rm -rf training_data/configname/calctype"

    WARNING: USE THIS STRING TO WRITE A BASH FILE AND EXECUTE IT ONLY IF YOU WANT
    TO REMOVE COMPLETED CALCULATIONS ON A REMOTE CLUSTER. MAKE SURE THE DATA IS
    SYNCED TO YOUR COMPUTER. DON'T USE IT ON YOUR OWN COMPUTER.

    Parameters
    ----------
    selected_configurations : list[dict]
        List of configurations in ccasm query json format
    calctype : str, optional
        ccasm ``calctype`` of your DFT calcuations

    Returns
    -------
    str
        Write this string to a bash file and execute it to remove completed
        calculations

    """
    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)

    # read status files
    # if status is complete, add the corresponding config dir to be removed
    remove_str = "#!/bin/bash\n"
    remove_str += "#WARNING: USE THIS FILE ONLY TO REMOVE COMPLETED CALCULATIONS\n"
    remove_str += (
        "#WARNING: USE IT ON A REMOTE CLUSTER AFTER THE CALCULATIONS ARE SYNCED\n"
    )
    for calctype_dir in calctype_dirs:
        with open(os.path.join(calctype_dir, "status.json"), "r") as f:
            status = json.load(f)

        if status["status"] == "complete":
            remove_str += "rm -rf " + str(calctype_dir) + "/\n"

    return remove_str


def write_initial_status_files(
    selected_configurations: list[dict], calctype: str = "default"
) -> None:
    """TODO: Docstring for write_initial_status_files.

    Parameters
    ----------
    selected_configurations : TODO
    calctype : TODO, optional

    Returns
    -------
    TODO

    """
    status = {"status": "unsubmitted"}

    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)

    for calctype_dir in calctype_dirs:
        with open(os.path.join(calctype_dir, "status.json"), "w") as f:
            json.dump(status, f)

    return None

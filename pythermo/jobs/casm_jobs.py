import os
import json
import shlex
import subprocess
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
    for incar, calctype_dir in zip(incars, calctype_dirs):
        incar_string = str(incar)
        modified_incar_string = incar_string.replace("True", ".TRUE.")
        modified_incar_string = modified_incar_string.replace("False", ".FALSE.")

        with open(os.path.join(calctype_dir, "INCAR"), "w") as f:
            f.write(modified_incar_string)

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
        "atom_types": ["Li", "O", "Mn"],\n
        "magmoms" : [0.6, 0.4, {"values":[-1, 1],\n
                                "new_values":[-5, 5]}\n
                    ],\n
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
        # if magenetic moment for an atom type is not provided, use the one already in incar
        modified_magmoms = []
        index = 0
        for n, atom_type in zip(poscar.natoms, poscar.site_symbols):
            # get old magmoms for a particular atom type
            magmom = np.array(incar["MAGMOM"][index : index + n])

            # if atom type is what you want to change
            if atom_type in provided_atom_types:
                new_magmom = provided_magmoms[provided_atom_types.index(atom_type)]

                # if all magmoms of a particular atom type needs to be replaced
                if isinstance(new_magmom, float):
                    magmom = [
                        provided_magmoms[provided_atom_types.index(atom_type)]
                    ] * n

                # if magmoms of a particular atom type needs to be changed value by value
                if isinstance(new_magmom, dict):
                    for value, new_value in zip(
                        new_magmom["values"], new_magmom["new_values"]
                    ):
                        magmom = np.where(np.isclose(magmom, value), new_value, magmom)

            index += n

            modified_magmoms.extend(magmom)

        # Update the value in INCAR
        incar["MAGMOM"] = modified_magmoms

        modifed_incars.append(incar)

    return modifed_incars


def write_properties_json(
    selected_configurations: list[dict],
    config_properties: list[dict],
    calctype: str = "default",
) -> None:
    """Given configuration list and properties.calc.json
    dictionaries, writes properties.calc.json in the
    casm project

    Parameters
    ----------
    selected_configurations : list[dict]
        ccasm style configurations
    properties_json : list[dict]
        ccasm style properties.calc.json
    calctype : str, optional
        ccasm calctype (default = "default")

    Returns
    -------
    None
        Writes properties.calc.json for all the
        ``selected_configurations``

    """
    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)
    for calctype_dir, config_property in zip(calctype_dirs, config_properties):
        with open(os.path.join(calctype_dir, "properties.calc.json"), "w") as f:
            json.dump(config_property, f, indent=2)

    return None


def modify_magmoms_in_properties_json(
    selected_configurations: list[dict],
    elements_with_dof: list[str],
    calctype: str = "default",
) -> list[dict]:
    """Given a list of configurations and calctype,
    changes the Cunitmagspin values of ``elements_with_dof``
    to either +1 or -1 and remaining values to zero

    Parameters
    ----------
    selected_configurations : list[dict]
        ccasm style configurations
    elements_with_dof : list[str]
        List of elements which have Cunitmagpsin dof
    calctype : str, optional
        ccasm calctype (default = "default")

    Returns
    -------
    list[dict]
        List of modified properties.calc.json dictionaries
        for all the ``selected_configurations``

    """
    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)

    config_properties = []
    for calctype_dir in calctype_dirs:
        with open(os.path.join(calctype_dir, "properties.calc.json"), "r") as f:
            properties_json = json.load(f)

        atom_types = properties_json["atom_type"]
        magmoms = properties_json["atom_properties"]["Cunitmagspin"]["value"]

        new_magmoms = []
        for atom_type, magmom in zip(atom_types, magmoms):
            if atom_type in elements_with_dof:
                new_magmoms.append([magmom[0] / abs(magmom[0])])

            else:
                new_magmoms.append([0.0])

        properties_json["atom_properties"]["Cunitmagspin"]["value"] = new_magmoms

        config_properties.append(properties_json)

    return config_properties


def get_casm_config_list_from_config_info(configurations_info: dict) -> list[dict]:
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

    config_list = [
        dict(name=config_name, selected=True) for config_name in config_names
    ]

    return config_list


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


def visualize_magmoms_from_outcar(
    outcar_path: str, contcar_path: str, non_collinear=False
) -> CifWriter:
    """TODO: Docstring for visualize_magmoms_from_outcar.

    Parameters
    ----------
    outcar_path : TODO
    contcar_path : TODO

    Returns
    -------
    TODO

    """
    if non_collinear:
        raise NotImplementedError("Non collinear calculations not implemented yet")

    magmoms_dict = pmgvasp.Outcar(outcar_path).magnetization
    magmoms = [
        magmom_dict["tot"] / abs(magmom_dict["tot"])
        if abs(float(magmom_dict["tot"])) >= 0.5
        else 0
        for magmom_dict in magmoms_dict
    ]
    contcar_structure = pmgvasp.Poscar.from_file(contcar_path).structure

    magmoms_xyz_cart = np.array(
        [np.append(np.zeros(2), magmom_z) for magmom_z in magmoms]
    )
    magmoms_xyz_frac = (
        np.linalg.inv(contcar_structure.lattice.matrix.transpose())
        @ magmoms_xyz_cart.transpose()
    )
    magmoms_xyz_frac = magmoms_xyz_frac.transpose()

    contcar_structure.add_site_property("magmom", magmoms_xyz_cart)

    return CifWriter(contcar_structure, write_magmoms=True)


def get_config_list_of_completed_vasp_runs(
    selected_configurations: list[dict], calctype: str = "default"
) -> list[dict]:
    """Given a list of configs, this function returns
    a list of cofigurations with completed vasp runs
    by reading the status.json in each config/calctype dir

    Parameters
    ----------
    selected_configurations : list[dict]
        List of configurations in ccasm query json format
    calctype : str, optional
        ccasm ``calctype`` of your DFT calcuations

    Returns
    -------
    list[dict]
        List of configurations in ccasm query json format

    """
    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)

    completed_configurations = [
        config
        for config, calctype_dir in zip(selected_configurations, calctype_dirs)
        if is_vasp_relaxandstatic_run_complete(calctype_dir)
    ]

    return completed_configurations


def setup_continuing_vasp_relaxandstatic_run(vasp_run_dir: str) -> None:
    """Helper function to deal with setting up continuing vasp
    run. If the calculation is complete, nothing is done and
    the function returns True. Else, all the old vasp files will
    be move into a new run_ folder and VASP input files will be
    copied over for a new calculation. CONTCAR from the max run.folder
    will be the new POSCAR.

    Parameters
    ----------
    vasp_run_dir : str
        vasp run dir

    Returns
    -------
    bool
        Whether the calculation is finished or not

    """

    # get relaxandstatic_name
    relaxandstatic_name = [
        file for file in os.listdir(vasp_run_dir) if "relaxandstatic" in file
    ][0]

    # figure out the max run dir; so that you can copy contcar from there
    max_run_dir = ""
    run_dirs = []
    for dir in os.listdir(vasp_run_dir):
        if "run." in dir:
            if "final" in dir:
                max_run_dir = "run.final"
            else:
                run_dirs.append(int(str(dir).replace("run.", "")))

    if max_run_dir != "run.final":
        max_run_dir = "run." + str(max(run_dirs))

    # make the new_calc dir and copy the current calculation there
    calc_dirs = [
        int(dir.replace("run_", ""))
        for dir in os.listdir(vasp_run_dir)
        if "run_" in dir
    ]
    if len(calc_dirs) == 0:
        new_calc_dir = os.path.join(vasp_run_dir, "run_0")
    else:
        new_calc_dir = os.path.join(vasp_run_dir, "run_" + str(max(calc_dirs) + 1))

    # make the new_calc_dir
    os.mkdir(new_calc_dir)

    # move all the files to the new_calc_dir
    # figure out files to move
    move_files = [dir for dir in os.listdir(vasp_run_dir) if "run_" not in dir]
    for f in move_files:
        move_args = "mv " + os.path.join(vasp_run_dir, f) + " " + new_calc_dir + "/"
        move = subprocess.Popen(move_args, shell=True, stdout=subprocess.PIPE)
        move.communicate()

    # copy incar kpoints potcar contcar relaxandstatic.sh
    copy_incar_args = shlex.split("cp " + new_calc_dir + "/INCAR " + vasp_run_dir + "/")
    copy_kpoints_args = shlex.split(
        "cp " + new_calc_dir + "/KPOINTS " + vasp_run_dir + "/"
    )
    copy_potcar_args = shlex.split(
        "cp " + new_calc_dir + "/POTCAR " + vasp_run_dir + "/"
    )
    copy_contcar_args = shlex.split(
        "cp "
        + new_calc_dir
        + "/"
        + max_run_dir
        + "/CONTCAR "
        + vasp_run_dir
        + "/POSCAR"
    )
    copy_relaxandstatic_args = shlex.split(
        "cp " + new_calc_dir + "/" + relaxandstatic_name + " " + vasp_run_dir + "/"
    )

    copy_incar = subprocess.Popen(copy_incar_args, stdout=subprocess.PIPE)
    copy_incar.communicate()
    copy_kpoints = subprocess.Popen(copy_kpoints_args, stdout=subprocess.PIPE)
    copy_kpoints.communicate()
    copy_potcar = subprocess.Popen(copy_potcar_args, stdout=subprocess.PIPE)
    copy_potcar.communicate()
    copy_contcar = subprocess.Popen(copy_contcar_args, stdout=subprocess.PIPE)
    copy_contcar.communicate()
    copy_relaxandstatic = subprocess.Popen(
        copy_relaxandstatic_args, stdout=subprocess.PIPE
    )
    copy_relaxandstatic.communicate()

    return None


def resubmit_vasp_runs_for_given_configs(
    selected_configurations: list[dict], calctype: str = "default"
):
    """Given a list of configs, this function sets up
    a continuing vasp run in each config/calctype_dir

    Parameters
    ----------
    selected_configurations : list[dict]
        List of configurations in ccasm query json format
    calctype : str, optional
        ccasm ``calctype`` of your DFT calcuations

    Returns
    -------
    None

    """

    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)
    [
        setup_continuing_vasp_relaxandstatic_run(calctype_dir)
        for calctype_dir in calctype_dirs
    ]
    return None


def remove_completed_calculations(
    selected_configurations: list[dict], calctype: str = "default"
) -> str:
    """For a given list of configurations (in ccasm query json format),
    a status.json file will be read from training_data/configname/calctype/status.json.
    If status["status"] is "complete", a line will be added to the output as
    "rm -rf training_data/configname/calctype"
    If status["status"] is "started", but the calculation termintated because
    it hit the time limit, status["status"] will be updated to ["failed"]
    and this calculation will also be remove

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
    # calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)

    # read status files
    # if status is complete, add the corresponding config dir to be removed
    remove_str = "#!/bin/bash\n"
    remove_str += "#WARNING: USE THIS FILE ONLY TO REMOVE COMPLETED CALCULATIONS\n"
    remove_str += (
        "#WARNING: USE IT ON A REMOTE CLUSTER AFTER THE CALCULATIONS ARE SYNCED\n"
    )

    completed_configs = get_config_list_of_completed_vasp_runs(
        selected_configurations, calctype
    )
    completed_calctype_dirs = _get_calctype_dirs(completed_configs, calctype)

    remove_str += "".join(
        (
            "rm -rf " + str(calctype_dir) + "/\n"
            for calctype_dir in completed_calctype_dirs
        )
    )

    terminated_configs = get_config_list_of_vasp_runs_terminated_due_to_time_limit(
        selected_configurations, calctype
    )
    terminated_calctype_dirs = _get_calctype_dirs(terminated_configs, calctype)
    remove_str += "".join(
        (
            "rm -rf " + str(calctype_dir) + "/\n"
            for calctype_dir in terminated_calctype_dirs
        )
    )

    # update status.json to failed for terminated configs
    update_status_json_for_given_configs(terminated_configs, calctype, "failed")

    return remove_str


def update_status_json_for_given_configs(
    selected_configurations: list[dict], calctype: str, status: str
) -> None:
    """Given a list of selected configs and calctype,
    update status.json in the config/calctype by the
    provided status

    Parameters
    ----------
    selected_configurations : list[dict]
        List of configurations in ccasm query json format
    calctype : str, optional
        ccasm ``calctype`` of your DFT calcuations
    status : str
        status to be updated in status.json in config/calctype

    Returns
    -------
    None

    """

    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)

    for calctype_dir in calctype_dirs:
        # update status with failed
        with open(os.path.join(calctype_dir, "status.json"), "w") as f:
            json.dump(dict(status=status), f)

    return None


def get_config_list_of_vasp_runs_terminated_due_to_time_limit(
    selected_configurations: list[dict], calctype: str
) -> list[dict]:
    """Given a list of configurations, this function returns
    a list of configurations which were killed due to time limit
    by checking in the corresponding calctype dir if
    "process killed (SIGTERM)" string exists in stdout in any
    of the run.* dirs (or) by reading the slurm or pbs out files

    Parameters
    ----------
    selected_configurations : list[dict]
        List of configurations in ccasm query json format
    calctype : str, optional
        ccasm ``calctype`` of your DFT calcuations

    Returns
    -------
    list[dict]
        List of calculations which are terminated due to
        time limit
    """

    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)
    terminated_configs = []

    for config, calctype_dir in zip(selected_configurations, calctype_dirs):
        with open(os.path.join(calctype_dir, "status.json"), "r") as f:
            status = json.load(f)

        if status["status"] == "started":
            # in case of pbs, read "*.e" file to see if time limit exceeded
            pbs_error_file = [file for file in os.listdir(calctype_dir) if ".e" in file]

            # in case of slurm, read "*.out" file to see if time limit exceeded
            slurm_out_file = [
                file for file in os.listdir(calctype_dir) if ".out" in file
            ]

            if len(slurm_out_file) != 0:
                with open(os.path.join(calctype_dir, slurm_out_file[0]), "r") as f:
                    if "DUE TO TIME LIMIT" in f.read():
                        terminated_configs.append(config)
                        continue

            if len(pbs_error_file) != 0:
                with open(os.path.join(calctype_dir, pbs_error_file[0]), "r") as f:
                    if "walltime" in f.read() and "exceeded limit" in f.read():
                        terminated_configs.append(config)
                        continue

            # if the above pbs and slurm out files do not catch the error,
            # try this by reading stdouts
            run_dirs = [dir for dir in os.listdir(calctype_dir) if "run." in dir]
            for run_dir in run_dirs:
                with open(os.path.join(calctype_dir, run_dir, "stdout"), "r") as f:
                    if "process killed (SIGTERM)" in f.read():
                        terminated_configs.append(config)
                        continue

    return terminated_configs


def write_initial_status_files(
    selected_configurations: list[dict], calctype: str = "default"
) -> None:
    """Writes initial status files when generating the
    submit script.

    Parameters
    ----------
    selected_configurations : list[dict]
        List of configurations in ccasm query json format
    calctype : str, optional
        ccasm ``calctype`` of your DFT calcuations

    Returns
    -------
    None
        Writes status.json with "unsubmitted" status

    """
    status = {"status": "unsubmitted"}

    calctype_dirs = _get_calctype_dirs(selected_configurations, calctype)

    for calctype_dir in calctype_dirs:
        with open(os.path.join(calctype_dir, "status.json"), "w") as f:
            json.dump(status, f)

    return None


def properties_json_from_relaxation_dir(relaxation_dir: str) -> dict:
    """Given vasp relaxation directory, returns casm
    style properties dictionary

    Parameters
    ----------
    relaxation_dir : str
        Vasp relaxation_dir where CONTCAR, OUTCAR, OSZICAR
        are present

    Returns
    -------
    dict
        properties dictionary formatted in casm style
    """

    contcar_path = os.path.join(relaxation_dir, "CONTCAR")

    contcar = pmgvasp.Poscar.from_file(contcar_path)
    relaxed_structure = contcar.structure

    properties = dict()

    # write structure info
    atom_types = []
    for atom_symbol, natom in zip(contcar.site_symbols, contcar.natoms):
        atom_types += [atom_symbol] * natom

    properties["atom_type"] = atom_types
    properties["coordinate_mode"] = "Direct"
    properties["atom_coords"] = relaxed_structure.frac_coords.tolist()
    properties["lattice_vectors"] = relaxed_structure.lattice.matrix.tolist()

    outcar_path = os.path.join(relaxation_dir, "OUTCAR")
    outcar = pmgvasp.Outcar(outcar_path)

    # write energy
    properties["global_properties"] = dict()
    properties["global_properties"]["energy"] = dict()
    properties["global_properties"]["energy"]["value"] = outcar.final_energy

    # write forces
    forces = outcar.read_table_pattern(
        header_pattern=r"\sPOSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s-+",
        row_pattern=r"\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
        footer_pattern=r"\s--+",
        postprocess=lambda x: float(x),
        last_one_only=False,
    )[0]
    properties["atom_properties"] = dict()
    properties["atom_properties"]["force"] = dict()
    properties["atom_properties"]["force"]["value"] = forces

    return properties


def toss_file_str_for_a_vasp_run(calc_dir: str, relaxandstatic_name: str) -> str:
    """Returns a toss file str with INCAR, KPOINTS,
    POTCAR, POSCAR, relaxandstatic_name which can
    be used with rsync --files-from. Attaches calc_dir
    to all the vasp input files

    THIS STRING SHOULD BE USED FROM DIRECTORY RELATIVE TO
    WHICH calc_dir IS DEFINED. FOR EXAMPLE, IF calc_dir is
    './0/init', THIS TOSS STRING IS VALID FROM ./

    Parameters
    ----------
    calc_dir : str
        dir which needs to be attached to the head of
        VASP input files
    relaxandstatic_name : str
        relaxandstatic.sh name

    Returns
    -------
    str
        toss file string which can be used from rsync --files-from

    """
    incar_path = os.path.join(calc_dir, "INCAR")
    kpoints_path = os.path.join(calc_dir, "KPOINTS")
    potcar_path = os.path.join(calc_dir, "POTCAR")
    poscar_path = os.path.join(calc_dir, "POSCAR")
    relaxandstatic_path = os.path.join(calc_dir, relaxandstatic_name)

    toss_str = (
        incar_path
        + "\n"
        + kpoints_path
        + "\n"
        + potcar_path
        + "\n"
        + poscar_path
        + "\n"
        + relaxandstatic_path
        + "\n"
    )

    return toss_str


def toss_file_str_for_a_neb_run(calc_dir: str, relaxandstatic_name: str) -> str:
    """Returns a toss file str with INCAR, KPOINTS,
    POTCAR, relaxandstatic_name which can
    be used with rsync --files-from. Attaches calc_dir
    to all the vasp input files

    THIS STRING SHOULD BE USED FROM DIRECTORY RELATIVE TO
    WHICH calc_dir IS DEFINED. FOR EXAMPLE, IF calc_dir is
    './0/init', THIS TOSS STRING IS VALID FROM ./

    Parameters
    ----------
    calc_dir : str
        dir which needs to be attached to the head of
        VASP input files
    relaxandstatic_name : str
        relaxandstatic.sh name

    Returns
    -------
    str
        toss file string which can be used from rsync --files-from

    """
    incar_path = os.path.join(calc_dir, "INCAR")
    kpoints_path = os.path.join(calc_dir, "KPOINTS")
    potcar_path = os.path.join(calc_dir, "POTCAR")
    relaxandstatic_path = os.path.join(calc_dir, relaxandstatic_name)

    toss_str = (
        incar_path
        + "\n"
        + kpoints_path
        + "\n"
        + potcar_path
        + "\n"
        + relaxandstatic_path
        + "\n"
    )
    for i in range(7):
        toss_str += os.path.join(calc_dir, "0" + str(i) + "/POSCAR\n")

    return toss_str


# TODO: allow for submit.sh & toss_files.txt file names as args
def setup_vasp_relax_runs_for_hop_envs(
    hop_env_dirs: list[str], vasp_input_file_dir: str, dry_run: False
) -> None:
    """Sets up vasp relax runs for given hop env dirs
    In the hop env dir, you should have POSCAR_init.vasp,
    and POSCAR_final.vasp describing the initial and final
    hop geometries. This function then makes init, and final
    directories in each of the hop env dir and sets up vasp
    runs given a path to a directory that contains vasp input
    files

    Also writes a toss file list, submit.sh that can be used to
    communicate with hpc system. This assumes that the hpc system
    uses slurm. Contact me if you want it for another

    ASSUMES ALL THE PATHS TO DIR ARE RELATIVE TO WHERE
    YOU ARE EXECUTING THE FUNCTION

    Parameters
    ----------
    hop_env_dirs : list[str]
        list of hop env dirs
    vasp_input_file_dir : str
        path to dir containing INCAR, KPOINTS,
        POTCAR, and relaxandstatic file
    dry_run : False
        only writes toss_file and submit script

    Returns
    -------
    None

    """
    incar = os.path.join(vasp_input_file_dir, "INCAR")
    kpoints = os.path.join(vasp_input_file_dir, "KPOINTS")
    potcar = os.path.join(vasp_input_file_dir, "POTCAR")
    relaxandstatic_name = [
        file for file in os.listdir(vasp_input_file_dir) if "relaxandstatic" in file
    ][0]
    relaxandstatic = os.path.join(vasp_input_file_dir, relaxandstatic_name)

    print("Dry run: ", dry_run)

    # write toss and submit script
    toss_str = ""
    submit_str = "#!/bin/bash\n"
    submit_str += "configs=(\n"

    for hop_dir in hop_env_dirs:
        # make init dir and copy POSCAR_init to init/POSCAR
        init_dir = os.path.join(hop_dir, "init")
        if not dry_run:
            os.mkdir(init_dir)
        copy_poscar_init_args = shlex.split(
            "cp "
            + os.path.join(hop_dir, "POSCAR_init.vasp")
            + " "
            + os.path.join(init_dir, "POSCAR")
        )
        copy_poscar_init = subprocess.Popen(
            copy_poscar_init_args, stdout=subprocess.PIPE
        )
        if not dry_run:
            copy_poscar_init.communicate()

        # make final dir and copy POSCAR_final to final/POSCAR
        final_dir = os.path.join(hop_dir, "final")
        if not dry_run:
            os.mkdir(final_dir)
        copy_poscar_final_args = shlex.split(
            "cp "
            + os.path.join(hop_dir, "POSCAR_final.vasp")
            + " "
            + os.path.join(final_dir, "POSCAR")
        )
        copy_poscar_final = subprocess.Popen(
            copy_poscar_final_args, stdout=subprocess.PIPE
        )
        if not dry_run:
            copy_poscar_final.communicate()

        # copy incar to init and final
        copy_incar_to_init_args = shlex.split("cp " + incar + " " + init_dir)
        copy_incar_to_final_args = shlex.split("cp " + incar + " " + final_dir)
        copy_incar_to_init = subprocess.Popen(
            copy_incar_to_init_args, stdout=subprocess.PIPE
        )
        copy_incar_to_final = subprocess.Popen(
            copy_incar_to_final_args, stdout=subprocess.PIPE
        )
        if not dry_run:
            copy_incar_to_init.communicate()
            copy_incar_to_final.communicate()

        # copy kpoints to init and final
        copy_kpoints_to_init_args = shlex.split("cp " + kpoints + " " + init_dir)
        copy_kpoints_to_final_args = shlex.split("cp " + kpoints + " " + final_dir)
        copy_kpoints_to_init = subprocess.Popen(
            copy_kpoints_to_init_args, stdout=subprocess.PIPE
        )
        copy_kpoints_to_final = subprocess.Popen(
            copy_kpoints_to_final_args, stdout=subprocess.PIPE
        )
        if not dry_run:
            copy_kpoints_to_init.communicate()
            copy_kpoints_to_final.communicate()

        # copy potcar to init and final
        copy_potcar_to_init_args = shlex.split("cp " + potcar + " " + init_dir)
        copy_potcar_to_final_args = shlex.split("cp " + potcar + " " + final_dir)
        copy_potcar_to_init = subprocess.Popen(
            copy_potcar_to_init_args, stdout=subprocess.PIPE
        )
        copy_potcar_to_final = subprocess.Popen(
            copy_potcar_to_final_args, stdout=subprocess.PIPE
        )
        if not dry_run:
            copy_potcar_to_init.communicate()
            copy_potcar_to_final.communicate()

        # copy relaxandstatic to init and final
        sed_init_args = shlex.split(
            'sed "s/.*#SBATCH -J.*/#SBATCH -J '
            + "initrelax"
            + hop_dir
            + '/g" '
            + relaxandstatic
        )
        sed_final_args = shlex.split(
            'sed "s/.*#SBATCH -J.*/#SBATCH -J '
            + "finalrelax"
            + hop_dir
            + '/g" '
            + relaxandstatic
        )
        if not dry_run:
            sed_init = subprocess.Popen(sed_init_args, stdout=subprocess.PIPE)
            sed_final = subprocess.Popen(sed_final_args, stdout=subprocess.PIPE)
            sed_init_out, _ = sed_init.communicate()
            with open(os.path.join(hop_dir, "init", relaxandstatic_name), "w") as f:
                f.write(sed_init_out.decode("utf-8"))

            sed_final_out, _ = sed_final.communicate()
            with open(os.path.join(hop_dir, "final", relaxandstatic_name), "w") as f:
                f.write(sed_final_out.decode("utf-8"))

        # write toss file
        toss_str += toss_file_str_for_a_vasp_run(
            os.path.join(hop_dir, "init"), relaxandstatic_name
        )
        toss_str += toss_file_str_for_a_vasp_run(
            os.path.join(hop_dir, "final"), relaxandstatic_name
        )

        # submit script
        submit_str += " " + init_dir + "\n"
        submit_str += " " + final_dir + "\n"

        print("Finished setting up init and final vasp runs for ", hop_dir)
        print("===========================================================")

    submit_str += ")\n"
    submit_str += 'for i in "${configs[@]}"; do\n'
    submit_str += " cd $i\n"
    submit_str += " sbatch " + relaxandstatic_name + "\n"
    submit_str += " cd ../../\n"
    submit_str += "done\n"

    with open("./submit.sh", "w") as f:
        f.write(submit_str)

    with open("./toss_files.txt", "w") as f:
        f.write(toss_str)

    return None


def is_vasp_relaxandstatic_run_complete(vasp_run_dir: str) -> bool:
    """Tells whether a vasp run is complete or
    not by reading the status.json

    Parameters
    ----------
    vasp_run_dir : str

    Returns
    -------
    bool
        Is vasp run done

    """

    with open(os.path.join(vasp_run_dir, "status.json"), "r") as f:
        status = json.load(f)

    if status["status"] == "complete":
        return True

    return False


def resubmit_incomplete_vasp_runs_for_hop_envs(
    hop_env_dirs: list[str], dry_run: True
) -> None:
    """Resubmit incomplete vasp runs in init
    and final dirs for hop env dirs. Incompleteness
    is read from hop_env_dir/init_or_final/status.json
    If calculation is complete nothing is done. Else
    a new run_ will be made and old calculation files
    will be moved there and a new calculation is setup
    by copying vasp input files and latest CONTCAR

    ASSUMES ALL THE PATHS TO DIR ARE RELATIVE TO WHERE
    YOU ARE EXECUTING THE FUNCTION

    Parameters
    ----------
    hop_env_dirs : list[dir]
        all the hop env dirs
    dry_run : bool
        If true only prints out statuses
        of each vasp calculation

    Returns
    -------
    None

    """
    toss_str = ""
    submit_str = "#!/bin/bash\n"
    submit_str += "configs=(\n"
    for hop_dir in hop_env_dirs:
        relaxandstatic_name = [
            file for file in os.listdir(hop_dir) if "relaxandstatic" in file
        ][0]

        is_init_done = is_vasp_relaxandstatic_run_complete(
            os.path.join(hop_dir, "init")
        )
        is_final_done = is_vasp_relaxandstatic_run_complete(
            os.path.join(hop_dir, "final")
        )

        if dry_run:
            print(hop_dir, " init status is: ", is_init_done)
            print(hop_dir, " final status is: ", is_final_done)
            continue

        if not is_init_done:
            setup_continuing_vasp_relaxandstatic_run(os.path.join(hop_dir, "init"))
            toss_str += toss_file_str_for_a_vasp_run(
                os.path.join(hop_dir, "init"), relaxandstatic_name
            )
            submit_str += " " + os.path.join(hop_dir, "init") + "\n"

        if not is_final_done:
            setup_continuing_vasp_relaxandstatic_run(os.path.join(hop_dir, "final"))
            toss_str += toss_file_str_for_a_vasp_run(
                os.path.join(hop_dir, "init"), relaxandstatic_name
            )
            submit_str += " " + os.path.join(hop_dir, "final") + "\n"

        print("Done with init and final of ", hop_dir)
        print("=====================================")

    submit_str += ")\n"
    submit_str += 'for i in "${configs[@]}"; do\n'
    submit_str += " cd $i\n"
    submit_str += " sbatch " + relaxandstatic_name + "\n"
    submit_str += " cd ../../\n"
    submit_str += "done\n"

    if not toss_str == "":
        with open("./toss_files.txt", "w") as f:
            f.write(toss_str)
        with open("./submit.sh", "w") as f:
            f.write(submit_str)

    return None


def setup_nebs_for_hops(hop_env_dirs: list[str], neb_input_file_dir: str) -> None:
    """Setup NEB calculations for hop_env_dir by
    making images from CONTCARs from init/run.final and
    final/run.final. Also copy input files INCAR, KPOINTS,
    POTCAR, relaxandstatic from neb_input_file_dir/

    ASSUMES ALL THE PATHS TO DIR ARE RELATIVE TO WHERE
    YOU ARE EXECUTING THE FUNCTION

    Parameters
    ----------
    hop_env_dirs : list[dir]
        all the hop env dirs
    neb_input_file_dir : str
        Path to dir where there are input files
        for NEB calculations

    Returns
    -------
    None

    """
    cwd = os.getcwd()

    neb_incar = os.path.join(neb_input_file_dir, "INCAR")
    neb_kpoints = os.path.join(neb_input_file_dir, "KPOINTS")
    neb_potcar = os.path.join(neb_input_file_dir, "POTCAR")
    neb_relaxandstatic_name = [
        file for file in os.listdir(neb_input_file_dir) if "relaxandstatic" in file
    ][0]
    neb_relaxandstatic = os.path.join(neb_input_file_dir, neb_relaxandstatic_name)

    toss_str = ""
    submit_str = "#!/bin/bash\n"
    submit_str += "configs=(\n"
    for hop_dir in hop_env_dirs:
        init_dir = os.path.join(hop_dir, "init")
        final_dir = os.path.join(hop_dir, "final")

        # copy init CONTCAR
        copy_init_contcar_args = shlex.split(
            "cp "
            + os.path.join(init_dir, "run.final", "CONTCAR")
            + " "
            + os.path.join(hop_dir, "init_relaxed.vasp")
        )
        copy_init_contcar = subprocess.Popen(
            copy_init_contcar_args, stdout=subprocess.PIPE
        )
        copy_init_contcar.communicate()

        # copy final CONTCAR
        copy_final_contcar_args = shlex.split(
            "cp "
            + os.path.join(final_dir, "run.final", "CONTCAR")
            + " "
            + os.path.join(hop_dir, "final_relaxed.vasp")
        )
        copy_final_contcar = subprocess.Popen(
            copy_final_contcar_args, stdout=subprocess.PIPE
        )
        copy_final_contcar.communicate()

        # make images
        os.chdir(os.path.join(cwd, hop_dir))
        setup_neb_images_args = "nebmake.pl init_relaxed.vasp final_relaxed.vasp 5"
        setup_neb_images = subprocess.Popen(
            setup_neb_images_args, shell=True, stdout=subprocess.PIPE
        )
        setup_neb_images.communicate()
        print("Done setting up NEB images for ", hop_dir)
        os.chdir(cwd)

        # copy vasp input files
        copy_incar_args = shlex.split("cp " + neb_incar + " " + hop_dir + "/")
        copy_kpoints_args = shlex.split("cp " + neb_kpoints + " " + hop_dir + "/")
        copy_potcar_args = shlex.split("cp " + neb_potcar + " " + hop_dir + "/")
        sed_relaxandstatic_args = shlex.split(
            'sed "s/.*#SBATCH -J.*/#SBATCH -J '
            + "hop"
            + hop_dir
            + '/g" '
            + neb_relaxandstatic
        )

        copy_incar = subprocess.Popen(copy_incar_args, stdout=subprocess.PIPE)
        copy_incar.communicate()

        copy_kpoints = subprocess.Popen(copy_kpoints_args, stdout=subprocess.PIPE)
        copy_kpoints.communicate()

        copy_potcar = subprocess.Popen(copy_potcar_args, stdout=subprocess.PIPE)
        copy_potcar.communicate()

        sed_relaxandstatic = subprocess.Popen(
            sed_relaxandstatic_args, stdout=subprocess.PIPE
        )
        sed_output, _ = sed_relaxandstatic.communicate()

        with open(os.path.join(hop_dir, neb_relaxandstatic_name), "w") as f:
            f.write(sed_output.decode("utf-8"))

        # write toss and submit str ------
        toss_str += toss_file_str_for_a_neb_run(hop_dir, neb_relaxandstatic_name)

        submit_str += " " + hop_dir + "\n"

        print("Done setting up VASP files for ", hop_dir)
        print("=========================================")

    submit_str += ")\n"
    submit_str += 'for i in "${configs[@]}"; do\n'
    submit_str += " cd $i\n"
    submit_str += " sbatch " + neb_relaxandstatic_name + "\n"
    submit_str += " cd ../\n"
    submit_str += "done\n"

    with open("./submit.sh", "w") as f:
        f.write(submit_str)

    with open("./toss_files.txt", "w") as f:
        f.write(toss_str)

    return None


def analyze_and_resubmit_nebs_for_hops(
    hop_env_dirs: list[str], dry_run: bool = False
) -> None:
    """Resubmit incomplete NEB calculations for hop_env_dir
    If NEB calculation is finished, it will be analyzed by
    doing nebresults.pl. Else CONTCARs from each image
    are copied over as POSCARs for continuing NEB run

    Parameters
    ----------
    hop_env_dirs : list[str]
        list of hop environment directories
    dry_run : bool
        If True, will only print what calculations
        are done and what are not done

    Returns
    -------
    None

    """
    cwd = os.getcwd()
    toss_str = ""
    submit_str = "#!/bin/bash\n"
    submit_str += "configs=(\n"

    neb_relaxandstatic_name = [
        file for file in os.listdir(hop_env_dirs[0]) if "relaxandstatic" in file
    ][0]
    for hop_dir in hop_env_dirs:
        # get status of neb calculations
        if os.path.isfile(os.path.join(hop_dir, "status.json")):
            with open(os.path.join(hop_dir, "status.json"), "r") as f:
                status = json.load(f)["status"]
        else:
            with open(os.path.join(hop_dir, "stdout"), "r") as f:
                if "reached required accuracy" in f.read():
                    status = "complete"
                else:
                    status = "started"

        if dry_run:
            print("NEB calculation for ", hop_dir, " is ", status)
            continue

        if status == "complete":
            # copy init OUTCAR to 00/
            copy_init_outcar_args = shlex.split(
                "cp "
                + os.path.join(hop_dir, "init", "run.final", "OUTCAR")
                + " "
                + os.path.join(hop_dir, "00")
            )
            copy_init_outcar = subprocess.Popen(
                copy_init_outcar_args, stdout=subprocess.PIPE
            )
            copy_init_outcar.communicate()

            # copy final OUTCAR to 06/
            copy_final_outcar_args = shlex.split(
                "cp "
                + os.path.join(hop_dir, "final", "run.final", "OUTCAR")
                + " "
                + os.path.join(hop_dir, "06")
            )
            copy_final_outcar = subprocess.Popen(
                copy_final_outcar_args, stdout=subprocess.PIPE
            )
            copy_final_outcar.communicate()

            # run nebresults.pl
            os.chdir(os.path.join(cwd, hop_dir))
            nebresult = subprocess.Popen("nebresults.pl", stdout=subprocess.PIPE)
            nebresult.communicate()

            print("Finished analyzing NEB run for ", hop_dir)
            print("=========================================")

            os.chdir(cwd)

        else:
            # Figure out max run_ dir
            run_dirs = [
                int(dir.replace("run_", ""))
                for dir in os.listdir(hop_dir)
                if "run_" in dir
            ]
            if len(run_dirs) == 0:
                max_run_dir = -1

            else:
                max_run_dir = max(run_dirs)

            new_run_dir = max_run_dir + 1
            new_run_dir = os.path.join("run_" + str(new_run_dir))

            # go to hop dir and setup NEB stuff
            os.chdir(os.path.join(cwd, hop_dir))
            os.mkdir(new_run_dir)

            # move 0* to max_run_dir + 1
            move_args = "mv 00 01 02 03 04 05 06 slurm-* stdout " + new_run_dir
            move = subprocess.Popen(move_args, shell=True, stdout=subprocess.PIPE)
            move.communicate()

            # copy POSCAR/CONTCAR to newly made image dirs
            for i in ["00", "01", "02", "03", "04", "05", "06"]:
                os.mkdir(i)
                if i == "00" or i == "06":
                    copy_args = shlex.split(
                        "cp "
                        + os.path.join(new_run_dir, i, "POSCAR")
                        + " "
                        + os.path.join(i, "POSCAR")
                    )
                else:
                    copy_args = shlex.split(
                        "cp "
                        + os.path.join(new_run_dir, i, "CONTCAR")
                        + " "
                        + os.path.join(i, "POSCAR")
                    )

                copy = subprocess.Popen(copy_args, stdout=subprocess.PIPE)
                copy.communicate()

            os.chdir(cwd)
            toss_str += toss_file_str_for_a_neb_run(hop_dir, neb_relaxandstatic_name)

            submit_str += " " + hop_dir + "\n"

            print("Done setting up VASP files for a new NEB run in ", hop_dir)
            print("=========================================")

    submit_str += ")\n"
    submit_str += 'for i in "${configs[@]}"; do\n'
    submit_str += " cd $i\n"
    submit_str += " sbatch " + neb_relaxandstatic_name + "\n"
    submit_str += " cd ../\n"
    submit_str += "done\n"

    if toss_str != "":
        with open("./submit.sh", "w") as f:
            f.write(submit_str)

        with open("./toss_files.txt", "w") as f:
            f.write(toss_str)

    return None


def get_transf_mat_to_scel(prim_lattice: np.ndarray, scel_lattice: np.ndarray):
    """TODO: Docstring for get_transf_mat_to_scel_from_properties_dict.

    Parameters
    ----------
    prim_lattice : TODO
     : TODO

    Returns
    -------
    TODO

    """
    return np.rint(np.linalg.inv(prim_lattice) @ scel_lattice)


def get_all_properties_json_from_vacancy_calc_dir(
    vacancy_calc_dir: str, is_hop_env_dir=False
) -> list[dict]:
    """From a given vacancy calculation dir, assuming
    vacancy calculations are in config_* in the calculation
    dir, this function goes into run.final of all the
    config_* and gets properties json in ccasm format
    which can be used to import into ccasm project

    Parameters
    ----------
    vacancy_calc_dir : str
        Path to directory where config_* relaxations
        are present

    Returns
    -------
    list[dict]
        list of all properties jsons

    """

    # assuming all the vacancy configs calcuations are in config_
    properties_jsons = []

    if is_hop_env_dir:
        for dirname, dirs, _ in os.walk(vacancy_calc_dir):
            if "/init" in dirname or "/final" in dirname:
                for dir in dirs:
                    if "run.final" == dir:
                        properties_json = properties_json_from_relaxation_dir(
                            os.path.join(dirname, dir)
                        )
                        properties_json["path"] = os.path.join(dirname, dir)
                        properties_jsons.append(properties_json)

        return properties_jsons

    vacancy_config_dirs = [
        os.path.join(vacancy_calc_dir, dir)
        for dir in os.listdir(vacancy_calc_dir)
        if os.path.isdir(os.path.join(vacancy_calc_dir, dir)) and "config_" in dir
    ]

    for vacancy_config_dir in vacancy_config_dirs:
        run_final_dir = [
            os.path.join(vacancy_calc_dir, vacancy_config_dir, dir)
            for dir in os.listdir(os.path.join(vacancy_calc_dir, vacancy_config_dir))
            if "run.final" in dir
            and os.path.isdir(os.path.join(vacancy_calc_dir, vacancy_config_dir, dir))
        ][0]

        properties_json = properties_json_from_relaxation_dir(run_final_dir)
        properties_json["path"] = run_final_dir
        properties_jsons.append(properties_json)

    return properties_jsons

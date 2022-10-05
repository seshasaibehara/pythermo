import os
import json
import argparse
import subprocess
import pythermo.jobs as pyjobs


def _sanitize_bool_args(arg: str) -> bool:
    """Given an argument, return the corresponding boolean value

    Parameters
    ----------
    arg : str
        bool argument passed a string

    Returns
    -------
    bool
        If arg starts with 't', returns True
        If arg starts with 'f', returns False

    Raises
    ------
    RuntimeError
        If an argument starts with any other letter other than 't' or 'f'

    """
    if arg.lower()[0] == "t":
        return True
    elif arg.lower()[0] == "f":
        return False
    else:
        raise RuntimeError("Could not convert " + arg + " to a boolean value.")


def _configuration_list(filename: str) -> list[dict]:
    """Returns a configuration list from
    a given filename

    Parameters
    ----------
    filename : str
        filename

    Returns
    -------
    list[dict]
        List of configuration dictionaries
        in ccasm style

    """
    with open(filename, "r") as f:
        selection = json.load(f)

    return selection


def _add_infile_argument(parser: argparse.ArgumentParser) -> None:
    """Adds input file argument to a given SubParser

    Parameters
    ----------
    parser : argparse.ArgumentParser
        SubParser for which you want to add
        input file argument

    Returns
    -------
    None

    """
    parser.add_argument(
        "--infile",
        "-i",
        type=str,
        required=True,
        help="Input file path with list of configurations (ccasm query json format)",
    )

    return None


def _add_outfile_argument(parser: argparse.ArgumentParser) -> None:
    """Adds output file argument to a given SubParser

    Parameters
    ----------
    parser : argparse.ArgumentParser
        SubParser for which you want to add
        output file argument

    Returns
    -------
    None

    """
    parser.add_argument(
        "--outfile",
        "-o",
        type=str,
        required=True,
        help="output file path",
    )
    return None


def _add_calctype_argument(parser: argparse.ArgumentParser) -> None:
    """Adds calctype argument for a given parser

    Parameters
    ----------
    parser : argparse.ArgumentParser
        SubParser for which you want to add
        calctype argument

    Returns
    -------
    None

    """
    parser.add_argument(
        "--calctype",
        nargs="?",
        default="default",
        type=str,
        help='casm calctype from which configurations need to be read (default="default")',
    )
    return None


def add_toss_arguments(subparser: argparse.ArgumentParser) -> None:
    """Add toss arguments to the given subparser

    Parameters
    ----------
    subparser : argparse.ArgumentParser
        subparser

    Returns
    -------
    None

    """
    _add_infile_argument(subparser)
    _add_outfile_argument(subparser)
    _add_calctype_argument(subparser)

    subparser.add_argument(
        "--incar",
        nargs="?",
        default="True",
        help="Whether to write incar (default=True)",
    )
    subparser.add_argument(
        "--poscar",
        nargs="?",
        default="True",
        help="Whether to write poscar (default=True)",
    )
    subparser.add_argument(
        "--kpoints",
        nargs="?",
        default="True",
        help="Whether to write kpoints (default=True)",
    )
    subparser.add_argument(
        "--potcar",
        nargs="?",
        default="True",
        help="Whether to write potcar (default=True)",
    )
    subparser.add_argument(
        "--relaxandstatic",
        nargs="?",
        default="True",
        help="Whether to write relaxandstatic.sh (default=True)",
    )


def add_submit_arguments(subparser: argparse.ArgumentParser) -> None:
    """Add submit arguments to the given subparser

    Parameters
    ----------
    subparser : argparse.ArgumentParser
        subparser

    Returns
    -------
    None

    """
    _add_infile_argument(subparser)
    _add_outfile_argument(subparser)
    subparser.add_argument(
        "--queue",
        "-q",
        type=str,
        choices=["slurm", "pbs"],
        required=True,
        help="HPC cluster queue type",
    )
    _add_calctype_argument(subparser)
    subparser.add_argument(
        "--relaxandstatic",
        nargs="?",
        default=None,
        help="path to relaxandstatic.sh file (default=training_data/settings/calctype.default/relaxandstatic.sh)",
    )


def add_modify_magmom_arguments(subparser: argparse.ArgumentParser) -> None:
    """Add modify magmom arguments to the given subparser

    Parameters
    ----------
    subparser : subparser.ArgumentParser
        subparser

    Returns
    -------
    None

    """
    _add_infile_argument(subparser)
    subparser.add_argument(
        "--magmoms",
        "-m",
        type=str,
        required=True,
        help="Path to magnetic moments file in json format (each atom type SHOULD have the same value of magmom)",
    )
    _add_calctype_argument(subparser)


def add_configs_arguments(subparser: argparse.ArgumentParser) -> None:
    """Add configs arguments to the given subparser

    Parameters
    ----------
    subparser : argparse.ArgumentParser
        subparser

    Returns
    -------
    None

    """
    subparser.add_argument(
        "--infile", "-i", required=True, type=str, help="Path to input file"
    )
    subparser.add_argument(
        "--outfile", "-o", required=True, type=str, help="Path to output file"
    )


def add_visualize_magmoms_arguments(subparser: argparse.ArgumentParser) -> None:
    """Add visualize magmoms to the given subparser

    Parameters
    ----------
    subparser : argparse.ArgumentParser
        subparser

    Returns
    -------
    None

    """
    _add_infile_argument(subparser)

    subparser.add_argument(
        "--contcar",
        "-c",
        default=None,
        type=str,
        help="Path to contcar if visualizing through outcar",
    )
    subparser.add_argument(
        "--outfile", "-o", type=str, default=None, help="Path to outcar mcif file"
    )


def add_remove_arguments(subparser: argparse.ArgumentParser) -> None:
    """Add remove arguments to the given subparser

    Parameters
    ----------
    subparser : argparse.ArgumentParser
        subparser

    Returns
    -------
    None

    """
    _add_infile_argument(subparser)
    _add_outfile_argument(subparser)
    _add_calctype_argument(subparser)

    return None


def execute_toss(args: argparse.ArgumentParser) -> None:
    """Execute toss given arguments

    Parameters
    ----------
    args : argparse.ArgumentParser
        Toss arguments

    Returns
    -------
    None

    """
    selection = _configuration_list(args.infile)
    toss_str = pyjobs.casm_jobs.toss_file_str(
        selection,
        args.calctype,
        _sanitize_bool_args(args.incar),
        _sanitize_bool_args(args.poscar),
        _sanitize_bool_args(args.kpoints),
        _sanitize_bool_args(args.potcar),
        _sanitize_bool_args(args.relaxandstatic),
    )

    with open(args.outfile, "w") as f:
        f.write(toss_str)

    return None


def execute_submit(args: argparse.ArgumentParser) -> None:
    """Execute submit given arguments

    Parameters
    ----------
    args : argparse.ArgumentParser
        Submit arguments

    Returns
    -------
    None

    """
    selection = _configuration_list(args.infile)

    # copy relaxandstatic.sh to given configurations
    copy_commands = pyjobs.casm_jobs.copy_relaxandstatic_cmds(
        selection, args.relaxandstatic, args.calctype
    )
    for copy_command_args in copy_commands:
        copy = subprocess.Popen(copy_command_args, stdout=subprocess.PIPE)
        copy.communicate()

    # sed relaxandstatic.sh and update job names
    job_name_cmds = pyjobs.casm_jobs.change_job_names_cmds(
        selection, args.queue, args.calctype
    )
    for job_name_cmd_args in job_name_cmds:
        sed = subprocess.Popen(job_name_cmd_args, stdout=subprocess.PIPE)
        sed.communicate()

    # write initial status file
    pyjobs.casm_jobs.write_initial_status_files(selection, args.calctype)

    # generate submit script
    submit_str = pyjobs.casm_jobs.submit_script_str(
        selection, args.queue, args.calctype
    )
    with open(args.outfile, "w") as f:
        f.write(submit_str)

    return None


def execute_modify_magmoms(args: argparse.ArgumentParser) -> None:
    """Execute modify magmoms given arguments

    Parameters
    ----------
    args : argparse.ArgumentParser
        modify magmom arguments

    Returns
    -------
    None

    """
    selection = _configuration_list(args.infile)

    with open(args.magmoms, "r") as f:
        new_magmoms = json.load(f)

    modified_incars = pyjobs.casm_jobs.modify_incar_magmoms(
        selection, new_magmoms, args.calctype, False
    )

    pyjobs.casm_jobs.write_incars(selection, modified_incars, args.calctype)

    return None


def execute_configs(args: argparse.ArgumentParser) -> None:
    """Exectue setoff given arguments

    Parameters
    ----------
    args : argparse.ArgumentParser
        setoff arguments

    Returns
    -------
    None

    """

    # read config info
    with open(args.infile, "r") as f:
        configurations_info = json.load(f)

    # get casm commands
    config_list = pyjobs.casm_jobs.get_casm_config_list_from_config_info(
        configurations_info
    )

    with open(args.outfile, "w") as f:
        json.dump(config_list, f)

    return None


def execute_visualize_magmoms(args: argparse.ArgumentParser) -> None:
    """Exectue visualize magmoms given arguments

    Parameters
    ----------
    args : argparse.ArgumentParser
        Visualize magmom arguments

    Returns
    -------
    None

    """
    if os.path.basename(args.infile) == "OUTCAR":
        if args.outfile is None or args.contcar is None:
            raise RuntimeError("Please provide outfile and contcar path")
        mcif = pyjobs.casm_jobs.visualize_magmoms_from_outcar(args.infile, args.contcar)
        mcif.write_file(args.outfile)

    else:
        selection = _configuration_list(args.infile)
        file_names = [
            os.path.join("training_data", config_name, "structure.mcif")
            for config_name in pyjobs.casm_jobs._get_config_names(selection)
        ]

        mcif_writers = pyjobs.casm_jobs.visualize_magnetic_moments_from_casm_structure(
            selection, False
        )

        [
            mcif.write_file(file_name)
            for mcif, file_name in zip(mcif_writers, file_names)
        ]
    return None


def exectue_remove(args: argparse.ArgumentParser) -> None:
    """Exectue remove command given the arguments

    Parameters
    ----------
    args : argparse.ArgumentParser
        Remove arguments

    Returns
    -------
    None

    """
    selection = _configuration_list(args.infile)

    remove_str = pyjobs.casm_jobs.remove_completed_calculations(
        selection, args.calctype
    )

    with open(args.outfile, "w") as f:
        f.write(remove_str)

    return None


def main():
    parser = argparse.ArgumentParser("Helpful functions for use with casm")
    subparser = parser.add_subparsers(dest="command")

    # Toss command
    toss = subparser.add_parser(
        "toss", help="Creates a file which can be used with rsync --files-from"
    )
    add_toss_arguments(toss)

    # submit script
    submit = subparser.add_parser(
        "submit",
        help="Generates a submit script that can be used for submitting jobs on slurm/pbs queues. Also copies relaxandstatic.sh to configuration directory and updates the jobname.",
    )
    add_submit_arguments(submit)

    # modify incars
    modify_magmoms = subparser.add_parser(
        "modify_magmoms",
        help="Change magmoms in INCAR for a given list of configurations",
    )
    add_modify_magmom_arguments(modify_magmoms)

    # setoff given list of configurations
    configs = subparser.add_parser(
        "configs",
        help="Get casm style selection file from provided configuration information",
    )
    add_configs_arguments(configs)

    # visualize magmoms
    visualize_magmoms = subparser.add_parser(
        "visualize_magmoms", help="Writes .mcif file for given configurations"
    )
    add_visualize_magmoms_arguments(visualize_magmoms)

    # remove completed calculations
    remove_completed_calculations = subparser.add_parser(
        "remove", help="Remove completed calculations"
    )
    add_remove_arguments(remove_completed_calculations)

    # parse all args
    args = parser.parse_args()

    if args.command == "toss":
        execute_toss(args)

    if args.command == "submit":
        execute_submit(args)

    if args.command == "modify_magmoms":
        execute_modify_magmoms(args)

    if args.command == "configs":
        execute_configs(args)

    if args.command == "visualize_magmoms":
        execute_visualize_magmoms(args)

    if args.command == "remove":
        exectue_remove(args)


if __name__ == "__main__":
    main()

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


def _add_verbose_argument(parser: argparse.ArgumentParser) -> None:
    """Adds verbose argument for a given parser

    Parameters
    ----------
    parser : argparse.ArgumentParser
        SubParser for which you want to add
        verbose argument

    Returns
    -------
    None

    """
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Helps increase verbosity"
    )
    return None


def print_verbosity(is_verbose: bool, *args: str) -> None:
    """If ``is_verbose`` is ``True``,
    prints ``verbosity``

    Parameters
    ----------
    is_verbose : bool
        Whether you want to print ``verbosity``
    *args : str
        strings to print

    Returns
    -------
    None

    """
    if is_verbose:
        [print(arg) for arg in args]

    return None


def execute_visualize_magmoms(args):
    """TODO: Docstring for execute_visualize_magmoms.

    Parameters
    ----------
    args : TODO

    Returns
    -------
    TODO

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


def main():
    parser = argparse.ArgumentParser("Helpful functions for use with casm")
    subparser = parser.add_subparsers(dest="command")

    # Toss command
    toss = subparser.add_parser(
        "toss", help="Creates a file which can be used with rsync --files-from"
    )

    _add_infile_argument(toss)
    _add_outfile_argument(toss)
    _add_calctype_argument(toss)
    _add_verbose_argument(toss)

    toss.add_argument(
        "--incar",
        nargs="?",
        default="True",
        help="Whether to write incar (default=True)",
    )
    toss.add_argument(
        "--poscar",
        nargs="?",
        default="True",
        help="Whether to write poscar (default=True)",
    )
    toss.add_argument(
        "--kpoints",
        nargs="?",
        default="True",
        help="Whether to write kpoints (default=True)",
    )
    toss.add_argument(
        "--potcar",
        nargs="?",
        default="True",
        help="Whether to write potcar (default=True)",
    )
    toss.add_argument(
        "--relaxandstatic",
        nargs="?",
        default="True",
        help="Whether to write relaxandstatic.sh (default=True)",
    )

    # submit script
    submit = subparser.add_parser(
        "submit",
        help="Generates a submit script that can be used for submitting jobs on slurm/pbs queues. Also copies relaxandstatic.sh to configuration directory and updates the jobname.",
    )
    _add_infile_argument(submit)
    _add_outfile_argument(submit)
    submit.add_argument(
        "--queue",
        "-q",
        type=str,
        choices=["slurm", "pbs"],
        required=True,
        help="HPC cluster queue type",
    )
    _add_calctype_argument(submit)
    submit.add_argument(
        "--relaxandstatic",
        nargs="?",
        default=None,
        help="path to relaxandstatic.sh file (default=training_data/settings/calctype.default/relaxandstatic.sh)",
    )

    # modify incars
    modify_magmoms = subparser.add_parser(
        "modify_magmoms",
        help="Change magmoms in INCAR for a given list of configurations",
    )
    _add_infile_argument(modify_magmoms)
    modify_magmoms.add_argument(
        "--magmoms",
        "-m",
        type=str,
        required=True,
        help="Path to magnetic moments file in json format (each atom type SHOULD have the same value of magmom)",
    )
    _add_calctype_argument(modify_magmoms)

    # setoff given list of configurations
    setoff = subparser.add_parser(
        "setoff",
        help="ccasm select --set-off given range of configurations in a supercell",
    )
    setoff.add_argument(
        "--infile", "-i", required=True, type=str, help="Path to input file"
    )
    setoff.add_argument(
        "--executable",
        nargs="?",
        default="ccasm",
        help='path to casm executable (default = "ccasm")',
    )

    # visualize magmoms
    visualize_magmoms = subparser.add_parser(
        "visualize_magmoms", help="Writes .mcif file for given configurations"
    )
    _add_infile_argument(visualize_magmoms)

    visualize_magmoms.add_argument(
        "--contcar",
        "-c",
        default=None,
        type=str,
        help="Path to contcar if visualizing through outcar",
    )
    visualize_magmoms.add_argument(
        "--outfile", "-o", type=str, default=None, help="Path to outcar mcif file"
    )

    # remove completed calculations
    remove_completed_calculations = subparser.add_parser(
        "remove", help="Remove completed calculations"
    )
    _add_infile_argument(remove_completed_calculations)
    _add_outfile_argument(remove_completed_calculations)
    _add_calctype_argument(remove_completed_calculations)

    # parse all args
    args = parser.parse_args()

    if args.command == "toss":
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
        # deal with verbosity
        [
            print_verbosity(
                args.verbose, "Writing INCAR to toss file for " + config["name"]
            )
            for config in selection
            if _sanitize_bool_args(args.incar)
        ]
        [
            print_verbosity(
                args.verbose, "Writing POSCAR to toss file for " + config["name"]
            )
            for config in selection
            if _sanitize_bool_args(args.poscar)
        ]
        [
            print_verbosity(
                args.verbose, "Writing KPOINTS to toss file for " + config["name"]
            )
            for config in selection
            if _sanitize_bool_args(args.kpoints)
        ]
        [
            print_verbosity(
                args.verbose, "Writing POTCAR to toss file for " + config["name"]
            )
            for config in selection
            if _sanitize_bool_args(args.potcar)
        ]
        [
            print_verbosity(
                args.verbose,
                "Writing relaxandstatic.sh to toss file for " + config["name"],
            )
            for config in selection
            if _sanitize_bool_args(args.relaxandstatic)
        ]

        with open(args.outfile, "w") as f:
            f.write(toss_str)

    if args.command == "submit":
        selection = _configuration_list(args.infile)

        # copy relaxandstatic.sh to given configurations
        copy_commands = pyjobs.casm_jobs.copy_relaxandstatic_cmds(
            selection, args.relaxandstatic, args.calctype
        )
        for copy_command_args in copy_commands:
            copy = subprocess.Popen(copy_command_args, stdout=subprocess.PIPE)
            copy_output, copy_error = copy.communicate()

        # sed relaxandstatic.sh and update job names
        job_name_cmds = pyjobs.casm_jobs.change_job_names_cmds(
            selection, args.queue, args.calctype
        )
        for job_name_cmd_args in job_name_cmds:
            sed = subprocess.Popen(job_name_cmd_args, stdout=subprocess.PIPE)
            sed_output, sed_error = sed.communicate()

        # write initial status file
        pyjobs.casm_jobs.write_initial_status_files(selection, args.calctype)

        # generate submit script
        submit_str = pyjobs.casm_jobs.submit_script_str(
            selection, args.queue, args.calctype
        )
        with open(args.outfile, "w") as f:
            f.write(submit_str)

    if args.command == "modify_magmoms":
        selection = _configuration_list(args.infile)

        with open(args.magmoms, "r") as f:
            new_magmoms = json.load(f)

        modified_incars = pyjobs.casm_jobs.modify_incar_magmoms(
            selection, new_magmoms, args.calctype, False
        )

        pyjobs.casm_jobs.write_incars(selection, modified_incars, args.calctype)

    if args.command == "setoff":

        # read config info
        with open(args.infile, "r") as f:
            configurations_info = json.load(f)

        # get casm commands
        casm_commands = (
            pyjobs.casm_jobs.get_casm_commands_to_turn_off_given_configurations(
                configurations_info, args.executable
            )
        )

        for casm_cmd_args in casm_commands:
            casm = subprocess.Popen(casm_cmd_args, stdout=subprocess.PIPE)
            casm_output, casm_error = casm.communicate()

    if args.command == "visualize_magmoms":
        execute_visualize_magmoms(args)

    if args.command == "remove":
        selection = _configuration_list(args.infile)

        remove_str = pyjobs.casm_jobs.remove_completed_calculations(
            selection, args.calctype
        )

        with open(args.outfile, "w") as f:
            f.write(remove_str)


if __name__ == "__main__":
    main()

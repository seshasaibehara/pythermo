import os
import json
import argparse
import pythermo.jobs as pyjobs


def _sanitize_bool_args(arg: str) -> bool:
    """Given an argument, return the corresponding boolean value

    Parameters
    ----------
    arg : TODO

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


def _configuration_list(filename):
    """TODO: Docstring for _.

    Parameters
    ----------
    filename : TODO

    Returns
    -------
    TODO

    """
    try:
        with open(filename, "r") as f:
            selection = json.load(f)
    except ValueError:
        raise ValueError(
            "Could not open " + filename + " file. Please check the file exists"
        )

    return selection


def _add_infile_argument(parser):
    """TODO: Docstring for _add_infile_argument.

    Parameters
    ----------
    parser : TODO

    Returns
    -------
    TODO

    """
    parser.add_argument(
        "--infile",
        "-i",
        type=str,
        required=True,
        help="Input file path with list of configurations (ccasm query json format)",
    )


def _add_outfile_argument(parser):
    """TODO: Docstring for _add_outfile_argument.

    Parameters
    ----------
    parser : TODO

    Returns
    -------
    TODO

    """
    parser.add_argument(
        "--outfile",
        "-o",
        type=str,
        required=True,
        help="output file path",
    )


def _add_calctype_argument(parser):
    """TODO: Docstring for _add_calctype_argument.

    Parameters
    ----------
    parser : TODO

    Returns
    -------
    TODO

    """
    parser.add_argument(
        "--calctype",
        nargs="?",
        default="default",
        type=str,
        help='casm calctype from which configurations need to be read (default="default")',
    )


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
    magmoms = subparser.add_parser(
        "magmom", help="Change magmoms in INCAR for a given list of configurations"
    )
    _add_infile_argument(magmoms)
    magmoms.add_argument(
        "--magmoms",
        "-m",
        type=str,
        required=True,
        help="Path to magnetic moments file in json format (each atom type SHOULD have the same value of magmom)",
    )
    _add_calctype_argument(magmoms)

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

        with open(args.outfile, "w") as f:
            f.write(toss_str)

    if args.command == "submit":
        selection = _configuration_list(args.infile)

        # copy relaxandstatic.sh to given configurations
        copy_commands = pyjobs.casm_jobs.copy_relaxandstatic_cmds(
            selection, args.relaxandstatic, args.calctype
        )
        [os.system(cp) for cp in copy_commands]

        # sed relaxandstatic.sh and update job names
        job_names_cmds = pyjobs.casm_jobs.change_job_names_cmds(
            selection, args.queue, args.calctype
        )
        [os.system(sed) for sed in job_names_cmds]

        # generate submit script
        submit_str = pyjobs.casm_jobs.submit_script_str(
            selection, args.queue, args.calctype
        )

        with open(args.outfile, "w") as f:
            f.write(submit_str)

    if args.command == "magmom":
        selection = _configuration_list(args.infile)

        with open(args.magmoms, "r") as f:
            new_magmoms = json.load(f)

        modified_incars = pyjobs.casm_jobs.modify_incar_magmoms(
            selection, new_magmoms, args.calctype, False
        )

        pyjobs.casm_jobs.write_incars(selection, modified_incars, args.calctype)


if __name__ == "__main__":
    main()

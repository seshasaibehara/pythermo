import os
import json
import shlex
import subprocess


def run_mc_for_different_eci_sets(mc_settings: dict, eci_names: list[str]) -> None:
    """TODO: Docstring for  run_mc_for_different_eci_sets.

    Parameters
    ----------
    mc_settings : TODO
    eci_dirs : TODO
    quiet: TODO

    Returns
    -------
    TODO

    """
    cwd = os.getcwd()

    status = dict()
    status["complete"] = []
    status["unsubmitted"] = eci_names

    for eci in eci_names:
        status["running"] = eci
        status["unsubmitted"].remove(eci)

        with open(os.path.join(cwd, "status.json"), "w") as f:
            json.dump(status, f)

        # set eci in casm proj
        casm_eci_commands = shlex.split("ccasm settings --set-eci " + eci)
        casm_set_eci = subprocess.Popen(casm_eci_commands, stdout=subprocess.PIPE)
        casm_set_eci.communicate()

        # make mc dir
        mc_dir = os.path.join(cwd, "eci." + eci)
        os.mkdir(mc_dir)

        # copy mc_settings
        with open(os.path.join(mc_dir, "input_settings.json"), "w") as f:
            json.dump(mc_settings, f)

        # run mc
        os.chdir(mc_dir)
        mc_log = open("stdout", "a")
        casm_mc_commands = shlex.split("ccasm monte -s input_settings.json")
        casm_mc = subprocess.Popen(casm_mc_commands, stdout=mc_log, stderr=mc_log)
        casm_mc.communicate()
        mc_log.close()
        os.chdir(cwd)

        status["complete"].append(eci)

    return None

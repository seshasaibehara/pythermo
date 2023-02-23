import os
import json
import pythermo.clex.ternary as pyternary


def test_ternary_corr_indices_corresponding_to_binary_corrs(root_pytest_dir: str):
    """TODO: Docstring for test_ternary_corr_indices_corresponding_to_binary_corrs.

    Parameters
    ----------
    root_pytest_dir : TODO

    Returns
    -------
    TODO

    """
    with open(
        os.path.join(root_pytest_dir, "tests", "input_files", "binary_basis.json"), "r"
    ) as f:
        binary_basis_dict = json.load(f)
    with open(
        os.path.join(root_pytest_dir, "tests", "input_files", "ternary_basis.json"), "r"
    ) as f:
        ternary_basis_dict = json.load(f)

    (
        ternary_corr_indices_corresponding_to_binary_corr,
        _,
    ) = pyternary.ternary_corr_indices_corresponding_to_binary_corr(
        ternary_basis_dict, binary_basis_dict
    )

    assert ternary_corr_indices_corresponding_to_binary_corr == [
        0,
        1,
        3,
        6,
        9,
        12,
        15,
        18,
        24,
        30,
        36,
        40,
        48,
        56,
        62,
        68,
        74,
        80,
        88,
        96,
        102,
        108,
        117,
        129,
        135,
        143,
        151,
        156,
        166,
        182,
        198,
        208,
        220,
        232,
        244,
        256,
        272,
        284,
        291,
        303,
        315,
        327,
        343,
        355,
        371,
        379,
        385,
        394,
        410,
        420,
        432,
        448,
        464,
        480,
        490,
        502,
        514,
        530,
        546,
        562,
        578,
        594,
        606,
        613,
    ]

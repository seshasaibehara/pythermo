import pytest


@pytest.fixture
def root_pytest_dir(request: pytest.FixtureRequest) -> str:
    """Returns the path of root_dir (where pytest.ini lies) from which pytests run.
    Useful for resolving absolute path of input files that
    need to be supplied to test functions

    Parameters
    ----------
    Request : pytest.FixtureRequest

    Returns
    -------
    str

    """
    return str(request.config.rootdir)

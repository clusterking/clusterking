#!/usr/bin/env python3

# std
import os
import pathlib

# 3rd party
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ENV_VAR_TESTING_MODE = "B_decays_clustering_TESTMODE"


def set_testing_mode(testing_mode: bool) -> None:
    """
    Set an environment variable signalling if we are in testing mode.
    
    Args:
        testing_mode (bool): True if we are in testing mode 

    Returns:
        None
    """
    if testing_mode:
        os.environ[ENV_VAR_TESTING_MODE] = "true"
    else:
        os.environ[ENV_VAR_TESTING_MODE] = "false"


def is_testing_mode():
    testing_mode = os.environ[ENV_VAR_TESTING_MODE]
    if testing_mode == "true":
        return True
    elif testing_mode == "false":
        return False
    else:
        raise ValueError("{} set to invalid value {}.".format(
            ENV_VAR_TESTING_MODE, testing_mode))


def test_jupyter_notebook(path) -> None:
    """ Runs jupyter notebook. A ValueError is raised if the file was 
    not found. """
    set_testing_mode(True)
    path = pathlib.Path(path)
    if not path.exists():
        raise ValueError("Notebook '{}' wasn't even found!".format(path))
    run_path_str = str(path.parent.resolve())
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        with path.open() as f:
            nb = nbformat.read(f, as_version=4)
            ep.preprocess(nb, {'metadata': {'path': run_path_str}})
    except Exception as e:
        set_testing_mode(False)
        raise e

#!/usr/bin/env python3

# std
import os
import pathlib
import unittest

# 3rd party
import logging
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import numpy as np

# ours
from clusterking.util.log import set_global_log_level

ENV_VAR_TESTING_MODE = "CLUSTERKING_TESTMODE"


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
        raise ValueError(
            "Environment variable {} set to invalid value {}.".format(
            ENV_VAR_TESTING_MODE, testing_mode)
        )


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


class MyTestCase(unittest.TestCase):
    """ Implements an additional general testing methods. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        set_testing_mode(True)
        set_global_log_level(logging.WARNING)

    def assertAllClose(self, a, b):
        """ Compares two numpy arrays """
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        if not isinstance(b, np.ndarray):
            b = np.array(b)
        almost_same = (np.allclose(
            a, b
        ))
        if not almost_same:
            self.assertTrue(False, "Not the same: {} and {}.".format(a, b))

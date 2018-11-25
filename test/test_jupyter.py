#!/usr/bin/python3

""" This file tests all jupyter notbooks """

import unittest
import os.path
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# todo: use new python path class, rather than clumsy methods


def test_jupyter_notebook(path: str) -> bool:
    """ Runs jupyter notebook and returns True if it executed without
    error and false otherwise. """
    if not os.path.exists(path):
        raise ValueError("Notebook '{}' wasn't even found!".format(path))
    run_path = os.path.dirname(path)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
        ep.preprocess(nb, {'metadata': {'path': run_path}})


class TestJupyter(unittest.TestCase):
    pass


def test_generator(path):
    def test(self):
        test_jupyter_notebook(path)
    return test


def failed_test_generator(path):
    def test(self):
        with self.assertRaises(Exception):
            test_jupyter_notebook(path)
    return test


def underscore_string(path: str):
    ret = ""
    for letter in path:
        if letter.isalnum():
            ret += letter
        else:
            ret += "_"
    return ret

# note: Do NOT move that in an __file__ == "__main__" check!
this_dir = os.path.dirname(os.path.abspath(__file__))
notebook_dir = os.path.join(this_dir, "..", "jupyter")
notebooks = [ fn for fn in os.listdir(notebook_dir) if fn.endswith(".ipynb") ]
notebook_paths = [os.path.join(notebook_dir, notebook) for notebook in notebooks]
for path in notebook_paths:
    test_name = "test_" + underscore_string(path)
    if os.path.basename(path) == "unittest_jupyter_exception.ipynb":
        test = failed_test_generator(path)
    else:
        test = test_generator(path)
    setattr(TestJupyter, test_name, test)

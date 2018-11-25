#!/usr/bin/python3

""" This file tests all jupyter notbooks """

import unittest
import os.path
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError

# todo: use new python path class, rather than clumsy methods


def test_jupyter_notebook(path: str) -> bool:
    """ Runs jupyter notebook and returns True if it executed without
    error and false otherwise. """
    if not os.path.exists(path):
        print("Notebook '{}' wasn't even found!".format(path),
              file=sys.stderr)
        return False
    run_path = os.path.dirname(path)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
        try:
            ep.preprocess(nb, {'metadata': {'path': run_path}})
        except:
            return False
    return True


class TestJupyter(unittest.TestCase):
    pass


def test_generator(path):
    def test(self):
        self.assertTrue(test_jupyter_notebook(path))
    return test


def underscore_string(path: str):
    ret = ""
    for letter in path:
        if letter.isalnum():
            ret += letter
        else:
            ret += "_"
    return ret


if __name__ == "__main__":
    notebook_dir = os.path.join("..", "jupyter")
    notebooks = [ fn for fn in os.listdir(notebook_dir) if fn.endswith(".ipynb") ]
    notebook_paths = [os.path.join(notebook_dir, notebook) for notebook in notebooks]
    for path in notebook_paths:
        test = test_generator(path)
        test_name = "test_" + underscore_string(path)
        setattr(TestJupyter, test_name, test)
    # print(dir(TestJupyter))
    unittest.main()
#!/usr/bin/python3

""" This file executes all jupyter notbooks and tests if they run 
successfully. """

import unittest
import pathlib
import nbformat
from typing import Union
from nbconvert.preprocessors import ExecutePreprocessor


def test_jupyter_notebook(path: Union[str, pathlib.Path]) -> None:
    """ Runs jupyter notebook. A ValueError is raised if the file was 
    not found. """
    path = pathlib.Path(path)
    if not path.exists():
        raise ValueError("Notebook '{}' wasn't even found!".format(path))
    run_path_str = str(path.parent.resolve())
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    with path.open() as f:
        nb = nbformat.read(f, as_version=4)
        ep.preprocess(nb, {'metadata': {'path': run_path_str}})

def test_function_generator(path):
    # path = pathlib.Path(path)
    def test_fct():
        return test_jupyter_notebook(path)
    # name = underscore_string(path.name)
    # test_fct.__name__ = "test_jupyter_{}".format(name)
    return test_fct

#
# # Will attach all tests to this class below.
# class TestJupyter(unittest.TestCase):
#     pass
#
#
# def test_generator(path):
#     path = pathlib.Path(path)
#     # noinspection PyUnusedLocal
#     def test(self):
#         test_jupyter_notebook(path)
#     return test
#
#
# def failed_test_generator(path):
#     def test(self):
#         with self.assertRaises(Exception):
#             test_jupyter_notebook(path)
#     return test
#
#

def underscore_string(path) -> str:
    path = str(path)
    ret = ""
    for letter in path:
        if letter.isalnum():
            ret += letter
        else:
            ret += "_"
    return ret
#
#
def setup_tests():
    this_dir = pathlib.Path(__file__).resolve().parent
    notebook_base_dir = this_dir / ".." / ".."/ "jupyter"
    # noinspection PyUnresolvedReferences
    notebooks = [
        fn for fn in notebook_base_dir.iterdir() if fn.name.endswith(".ipynb")
    ]
    notebook_paths = sorted([
        notebook_base_dir / notebook for notebook in notebooks
    ])

    for path in notebook_paths:
        test_name = "test_jupyter_{}".format(underscore_string(path.name))
        if path.name == "unittest_jupyter_exception.ipynb":
            pass
            # test = failed_test_generator(path)
        else:
            fct = test_function_generator(path)
            globals()[test_name] = fct

setup_tests()

# print(globals())

#
#
# # note: Do NOT move that in an __file__ == "__main__" check!
# # Else this won't work with python3 -m unittest discover
# setup_tests()
#
#
# if __name__ == '__main__':
#     unittest.main(verbosity=2)

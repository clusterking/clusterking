#!/usr/bin/env python3

import pathlib


file_stub = """#!/usr/bin/python3

\"\"\" Automatically generated file. All changes will be lost.
Generated by generate_tests.py.\"\"\"

from bclustering.util.testing import test_jupyter_notebook


def {function_name}():
    test_jupyter_notebook('{notebook_path}')

if __name__ == "__main__":
    {function_name}()
"""


def underscore_string(string) -> str:
    string = str(string)
    ret = ""
    for letter in string:
        if letter.isalnum():
            ret += letter
        else:
            ret += "_"
    return ret


class TestGenerator(object):
    def __init__(self):
        self.this_dir = pathlib.Path(__file__).resolve().parent
        self.jupyter_dir = self.this_dir / ".."
        self.notebooks = [
            candidate
            for candidate in self.jupyter_dir.iterdir()
            if candidate.suffix == ".ipynb"
        ]
        self.generated_test_dir = self.this_dir / "test_auto_generated_tests"
        self.generated_test_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def notebook_path_to_test_name(notebook_path: pathlib.Path) -> str:
        name = underscore_string(notebook_path.name)
        if not name.startswith("test_"):
            name = "test_" + name
        return name

    def notebook_path_to_test_path(
        self, notebook_path: pathlib.Path
    ) -> pathlib.Path:
        name = self.notebook_path_to_test_name(notebook_path)
        name += ".py"
        return self.generated_test_dir / name

    def generate_test(self, notebook_path):
        p = self.notebook_path_to_test_path(notebook_path)
        name = self.notebook_path_to_test_path(notebook_path)
        with p.open("w") as outfile:
            outfile.write(
                file_stub.format(
                    function_name=name, notebook_path=notebook_path.resolve()
                )
            )

    def generate_all(self):
        for nb_path in self.notebooks:
            self.generate_test(nb_path)


def test():
    """Small hack so that running testing also regenerates the
    test files. Even though"""
    tg = TestGenerator()
    tg.generate_all()


if __name__ == "__main__":
    my_tg = TestGenerator()
    my_tg.generate_all()

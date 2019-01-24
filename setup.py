#!/usr/bin/env python3

from distutils.core import setup
# noinspection PyUnresolvedReferences
import setuptools  # (1)
import pathlib

# (1) see https://stackoverflow.com/questions/8295644/
# Without this, install_requires is ignored.

# todo: perhaps split up in install_requires and extras_require
install_requires = [
    'pandas',
    'numpy',
    'scipy',
    'flavio',
    'gitpython',
    'nbconvert',
    'nbformat',
    'jupyter_client',
    'matplotlib',
    'sklearn',
    'colorlog',
    'ipykernel'
]

this_dir = pathlib.Path(__file__).resolve().parent
scripts_dir = this_dir / "bclustering" / "bin"
scripts = [str(s.resolve()) for s in scripts_dir.iterdir()]
print(scripts)

setup(
    name='bclustering',
    version='dev',
    packages=['bclustering',],
    install_requires = install_requires,
    url="https://github.com/RD-clustering/B_decays_clustering",
    scripts=scripts,
)
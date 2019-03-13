#!/usr/bin/env python3

# std
from distutils.core import setup
# noinspection PyUnresolvedReferences
import setuptools  # see below (1)
import pathlib

# ours
import bclustering.util.metadata

# (1) see https://stackoverflow.com/questions/8295644/
# Without this import, install_requires won't work.

bclustering.util.metadata.save_git_info()


# todo: perhaps read from requirements.txt
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
    'ipykernel',
    'wilson',
    'tqdm'
]

this_dir = pathlib.Path(__file__).resolve().parent

packages = setuptools.find_packages()
print(packages)

setup(
    name='bclustering',
    version='dev',
    packages=packages,
    install_requires=install_requires,
    url="https://github.com/RD-clustering/B_decays_clustering",
    package_data={
        'bclustering': ['git_info.json'],
    }
)

#!/usr/bin/env python3

# std
from distutils.core import setup
# noinspection PyUnresolvedReferences
import setuptools  # see below (1)
import pathlib

# (1) see https://stackoverflow.com/questions/8295644/
# Without this import, install_requires won't work.

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

keywords = [
    "clustering",
    "cluster",
    "kinematics",
    "cluster-analysis",
    "machine-learning",
    "ml",
    "hep",
    "hep-ml",
    "hep-ex",
    "hep-ph",
    "wilson",
]

description = "Cluster sets of histograms/curves, in particular kinematic" \
              "distributions in high energy physics."

this_dir = pathlib.Path(__file__).resolve().parent

packages = setuptools.find_packages()
print(packages)

setup(
    name='clusterking',
    version='1.0.dev',
    packages=packages,
    install_requires=install_requires,
    url="https://github.com/clusterking/clusterking",
    project_urls={
        "Bug Tracker": "https://github.com/clusterking/clusterking/issues",
        "Documentation": "https://clusterking.readthedocs.io/",
        "Source Code": "https://github.com/clusterking/clusterking/",
    },
    package_data={
        'clusterking': ['git_info.json'],
    },
    license="MIT",
    keywords=keywords,
    description=description
)

#
# # Can only do this after installation of course
# todo: can actually do that with setup_requires
# print("Will this really be run in the end?")
# import clusterking.util.metadata
# clusterking.util.metadata.save_git_info()

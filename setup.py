#!/usr/bin/env python3

# std
from distutils.core import setup
import site
import sys

# noinspection PyUnresolvedReferences
import setuptools  # see below (1)
from pathlib import Path

# (1) see https://stackoverflow.com/questions/8295644/
# Without this import, install_requires won't work.

# Sometimes editable install fails with an error message about user site
# being not writeable. The following line can fix that, see
# https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

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

description = (
    "Cluster sets of histograms/curves, in particular kinematic"
    " distributions in high energy physics."
)

this_dir = Path(__file__).resolve().parent

packages = setuptools.find_packages()

with (this_dir / "README.rst").open() as fh:
    long_description = fh.read()

with (this_dir / "clusterking" / "version.txt").open() as vf:
    version = vf.read()

install_requires = [
    "pandas",
    "numpy",
    "scipy",
    "gitpython",
    "scikit-learn",
    "colorlog",
    "wilson != 2.2",
    "tqdm",
    "sqlalchemy",
]

extras_require = {
    "plotting": ["matplotlib"],
    "testing": [
        "pytest>=4.4.0",
        "pytest-subtests",
        "pytest-cov",
        "nbconvert",
    ],
    "dev": [
        "pytest>=4.4.0",
        "pytest-subtests",
        "pytest-cov",
        "nbstripout",
        "nbconvert",
        "nbformat",
        "jupyter_client",
        "ipykernel",
        "twine",
        "pre-commit",
        "sphinx",
        "sphinx_book_theme",
        "coveralls",
    ],
}


setup(
    name="clusterking",
    version=version,
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    url="https://github.com/clusterking/clusterking",
    project_urls={
        "Bug Tracker": "https://github.com/clusterking/clusterking/issues",
        "Documentation": "https://clusterking.readthedocs.io/",
        "Source Code": "https://github.com/clusterking/clusterking/",
    },
    package_data={"clusterking": ["git_info.json", "version.txt"]},
    license="MIT",
    keywords=keywords,
    description=description,
    long_description=long_description,
    long_description_content_type="text/x-rst",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)

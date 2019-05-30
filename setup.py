#!/usr/bin/env python3

# std
from distutils.core import setup

# noinspection PyUnresolvedReferences
import setuptools  # see below (1)
from pathlib import Path

# (1) see https://stackoverflow.com/questions/8295644/
# Without this import, install_requires won't work.


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

with (this_dir / "requirements.txt").open() as rf:
    binder_requires = [
        req.strip()
        for req in rf.readlines()
        if req.strip() and not req.startswith("#")
    ]

install_requires = [
    "pandas",
    "numpy",
    "scipy",
    "gitpython",
    "sklearn",
    "colorlog",
    "wilson",
    "tqdm",
    "sqlalchemy",
]

extras_require = {
    "plotting": ["matplotlib"],
    "dev": [
        "pytest>=4.4.0",
        "pytest-subtests",
        "pytest-cov",
        "nbstripout",
        "nbconvert",
        "nbformat",
        "jupyter_client",
        "ipykernel",
    ],
}

binder_additional = sorted(list(set(binder_requires) - set(install_requires)))
binder_missing = sorted(list(set(install_requires) - set(binder_requires)))
if binder_missing:
    raise ValueError(
        "The requirements given in setup.py don't match these given in"
        " requirements.txt. Please check that you applied you changes to "
        "both files.\n"
        "requirements.txt misses the following dependencies: {}".format(
            ", ".join(binder_missing)
        )
    )

if binder_additional:
    print(
        "NOTE: requirements.txt has the following additional "
        "dependencies: {}".format(", ".join(binder_additional))
    )

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

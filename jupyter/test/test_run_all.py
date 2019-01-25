#!/usr/bin/python3

""" This file executes all jupyter notbooks and tests if they run 
successfully. """

# std
import pathlib

# 3rd party

# ours
from bclustering.util.testing import test_jupyter_notebook

this_dir = pathlib.Path(__file__).resolve().parent
jupyter_dir = this_dir / ".."


# Yes, we could dynamically create these functions, however this would
# make it impossible to run them in parallel with nose2.
# At least to my knowledge anyway.
# If you have any idea how to do this, please write me ;)


def test_tutorial_basics():
    test_jupyter_notebook(jupyter_dir / "001_tutorial_basics.ipynb")


def test_plot_bundles():
    test_jupyter_notebook(jupyter_dir / "plot_bundles.ipynb")


def test_plot_clusters():
    test_jupyter_notebook(jupyter_dir / "plot_clusters.ipynb")


def test_clustering():
    test_jupyter_notebook(jupyter_dir / "test_clustering.ipynb")


def test_plotcluster():
    test_jupyter_notebook(jupyter_dir / "test_PlotCluster.ipynb")


def test_voxel_plot():
    test_jupyter_notebook(jupyter_dir / "test_voxel_plot.ipynb")

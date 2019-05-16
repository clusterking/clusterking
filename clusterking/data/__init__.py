#!/usr/bin/env python3

"""
This page describes the main data object that are used by ClusterKinG.
If you do not need to include errors in your analysis, use
:py:class:`~clusterking.data.Data`, else
:py:class:`~clusterking.data.DataWithErrors` (which inherits from
:py:class:`~clusterking.data.Data` but adds additional methods to it).

Both classes inherit from a very basic class,
:py:class:`~clusterking.data.DFMD`, which provides basic input and output
methods.

.. warning::

    We will switch to a SQL database as output format soon, so there will be
    some slight interface changes. In particular, all information (which is
    currently divided between two files) will be in only one file.

"""

from clusterking.data.dwe import DataWithErrors
from clusterking.data.data import Data
from clusterking.data.dfmd import DFMD

#!/usr/bin/env python3

# std
from abc import abstractmethod

# 3rd
import numpy as np

# ours
from clusterking.data.data import Data
from clusterking.util.metadata import nested_dict
from clusterking.util.log import get_logger
from clusterking.result import Result
from clusterking.worker import Worker


class AbstractBenchmark(Worker):
    """Subclass this class to implement algorithms to choose benchmark
    points from all the points (in parameter space) that correspond to one
    cluster.
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.bpoints = None
        self.md = nested_dict()
        self.log = get_logger("Benchmark")
        self.set_cluster_column("cluster")

    @property
    def cluster_column(self) -> str:
        return self.md["cluster_column"]

    # **************************************************************************
    # Settings
    # **************************************************************************

    def set_cluster_column(self, column):
        self.md["cluster_column"] = column

    # **************************************************************************
    # Run
    # **************************************************************************

    @abstractmethod
    def _run(self, data):
        pass


class BenchmarkResult(Result):
    def __init__(self, data, bpoints, md):
        super().__init__(data=data)
        self._bpoints = bpoints
        self._md = md

    def _write(self, bpoint_column="bpoint") -> None:
        """ Write benchmark points to a column in the dataframe of the data
        object.

        Args:
            bpoint_column: Column to write to

        Returns:
            None
        """
        self._data.df[bpoint_column] = self._bpoints
        self._data.md["bpoint"][bpoint_column] = self._md

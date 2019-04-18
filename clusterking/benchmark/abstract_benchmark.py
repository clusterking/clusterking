#!/usr/bin/env python3

# std

# 3rd
import numpy as np

# ours
from clusterking.data.data import Data
from clusterking.util.metadata import nested_dict
from clusterking.util.log import get_logger


class AbstractBenchmark(object):
    """Subclass this class to implement algorithms to choose benchmark
    points from all the points (in parameter space) that correspond to one
    cluster.
    """
    def __init__(self, data: Data, cluster_column="cluster"):
        """

        Args:
            data: :py:class:`~clusterking.data.data.Data` object
            cluster_column: Column name of the clusters
        """
        self.data = data
        self.bpoints = None
        self.md = nested_dict()
        self.log = get_logger("Benchmark")
        self.md["cluster_column"] = cluster_column

    @property
    def cluster_column(self):
        """ The column from which we read the cluster information.
        Defaults to 'cluster'. """
        return self.md["cluster_column"]

    @property
    def _clusters(self):
        return self.data.df[self.cluster_column]

    def select_bpoints(self) -> None:
        """ Select one benchmark point for each cluster.
        """
        self.bpoints = self._select_bpoints()

    def _select_bpoints(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def write(self, bpoint_column="bpoint") -> None:
        """ Write benchmark points to a column in the dataframe of the data
        object.

        Args:
            bpoint_column: Column to write to

        Returns:
            None
        """
        self.data.df[bpoint_column] = self.bpoints
        self.data.md["bpoint"][bpoint_column] = self.md

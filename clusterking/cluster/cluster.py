#!/usr/bin/env python3

"""Read the results from scan.py and get_clusters them.
"""

# std
import time
from abc import abstractmethod

# us
from clusterking.util.metadata import version_info, nested_dict
from clusterking.util.log import get_logger
from clusterking.data.data import Data
from clusterking.worker import DataWorker
from clusterking.result import DataResult


class Cluster(DataWorker):
    """ Abstract baseclass of the Cluster classes. This class is subclassed to
    implement specific clustering algorithms and defines common functions.
    """

    def __init__(self):
        """
        Args:
            data: :py:class:`~clusterking.data.Data` object
        """
        super().__init__()
        self.log = get_logger("Scanner")

        self.clusters = None
        # self.bpoints = None

        #: Metadata
        self.md = nested_dict()

        self.md["git"] = version_info(self.log)
        self.md["time"] = time.strftime("%a %_d %b %Y %H:%M", time.gmtime())

    @abstractmethod
    def run(self, data, **kwargs):
        """ Implementation of the clustering. Should return an array-like object
        with the cluster number.
        """
        pass


# todo: add back n_clusters
class ClusterResult(DataResult):
    def __init__(self, data, md, clusters):
        super().__init__(data=data)
        self._md = md
        self._clusters = clusters
        self._md["n_clusters"] = len(set(self._clusters))

    def write(self, cluster_column="cluster"):
        """ Write results back in the :py:class:`~clusterking.data.Data`
        object. """
        self._data.df[cluster_column] = self._clusters
        self._data.md["cluster"][cluster_column] = self._md
        self._data.rename_clusters(column=cluster_column)

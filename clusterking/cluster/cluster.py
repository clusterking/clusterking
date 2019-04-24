#!/usr/bin/env python3

"""Read the results from scan.py and get_clusters them.
"""

# std
import time

# us
from clusterking.util.metadata import git_info, nested_dict
from clusterking.util.log import get_logger
from clusterking.data.data import Data


class Cluster(object):
    """ Abstract baseclass of the Cluster classes. This class is subclassed to
    implement specific clustering algorithms and defines common functions.
    """
    def __init__(self, data: Data):
        """
        Args:
            data: :py:class:`~clusterking.data.Data` object
        """
        self.log = get_logger("Scanner")

        self.data = data
        self.clusters = None
        # self.bpoints = None

        #: Metadata
        self.md = nested_dict()

        self.md["git"] = git_info(self.log)
        self.md["time"] = time.strftime(
            "%a %_d %b %Y %H:%M", time.gmtime()
        )

    def cluster(self, **kwargs):
        """ Performs the clustering.
        This method is a wrapper around the _cluster implementation in the
        subclasses. See there for additional arguments.
        """
        self.log.info("Performing clustering.")

        self.md["cluster_args"] = kwargs

        self.clusters = self._cluster(**kwargs)

        n_clusters = len(set(self.clusters))
        self.log.info(
            "Clustering resulted in {} get_clusters.".format(n_clusters)
        )
        self.md["n_clusters"] = n_clusters

        self.log.info("Done")

    def _cluster(self, **kwargs):
        """ Implementation of the clustering. Should return an array-like object
        with the cluster number.
        """
        raise NotImplementedError

    # todo: overwrite argument?
    def write(self, cluster_column="cluster"):
        """ Write results back in the :py:class:`~clusterking.data.Data`
        object. """
        self.data.df[cluster_column] = self.clusters
        self.data.md["cluster"][cluster_column] = self.md
        self.data.rename_clusters(column=cluster_column)

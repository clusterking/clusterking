#!/usr/bin/env python3

"""Read the results from scan.py and get_clusters them.
"""

# std
import time

# us
from bclustering.util.metadata import git_info, nested_dict
from bclustering.util.log import get_logger


# todo: allow initializing from either file or directly the dataframe and the
#  metadata
class Cluster(object):
    def __init__(self):
        """ This class is subclassed to implement specific clustering
        algorithms and defines common functions.

        Args:
            # todo: write something here
        """
        self.log = get_logger("Scanner")

        #: Metadata
        self.md = nested_dict()
        self.md["git"] = git_info(self.log)
        self.md["time"] = time.strftime("%a %_d %b %Y %H:%M",
                                                   time.gmtime())

    def cluster(self, data, column="cluster", **kwargs):
        """ Performs the clustering. 
        This method is a wrapper around the _cluster implementation in the 
        subclasses. See there for additional arguments. 
        
        Args:
            column: Column to which the get_clusters should be appended.
            
        Returns:
            None
        """
        self.log.info("Performing clustering.")

        # Taking the column name as additional key means that we can easily
        # save the configuration values of different clusterings
        md = self.md[column]

        for key, value in kwargs.items():
            md[key] = value

        clusters = self._cluster(data, **kwargs)

        n_clusters = len(set(clusters))
        self.log.info(
            "Clustering resulted in {} get_clusters.".format(n_clusters)
        )
        md["n_clusters"] = n_clusters

        data.df[column] = clusters
        data.md["cluster"] = self.md
        data.rename_clusters(column=column)

        self.log.info("Done")

    def _cluster(self, data, **kwargs):
        """ Implmentation of the clustering. Should return an array-like object
        with the cluster number.
        """
        raise NotImplementedError

#!/usr/bin/env python3

# std
import logging

# 3rd party

# ours
from clusterking.util.log import get_logger


class ColorScheme(object):
    """ Class holding color scheme. Subclass and overwrite color lists
    to implement different schemes.
    """
    def __init__(self, clusters=None):

        self.log = get_logger("BundlePlot", sh_level=logging.WARNING)

        if clusters is not None:
            self.clusters = list(clusters)
        else:
            self.clusters = []

        self.cluster_colors = [
            "red",
            "green",
            "blue",
            "black",
            "orange",
            "pink"
        ]

    def _get_cluster_color(self, cluster, listname):
        """ Try to pick a unique element of a list corresponding to cluster.
        
        Args:
            cluster: Name of cluster (or index)
            listname: Name of a list which is attribute of this class
                to pick from
        
        Returns:
            Element of that list
        """
        if self.clusters:
            if cluster in self.clusters:
                index = self.clusters.index(cluster)
            else:
                self.log.error(
                    "Cluster {} is not in the list of clusters. ".format(
                        cluster
                    ))
                index = 0
        else:
            assert(isinstance(cluster, int))
            index = cluster

        pick_list = getattr(self, listname)

        # fixme: this should only be displayed once!
        if index > len(pick_list):
            self.log.warning(
                "Not enough elements in self.{}. Some clusters might end up"
                "with identical elements."
            )

        return pick_list[index % len(pick_list)]

    def get_cluster_color(self, cluster):
        """
        Get color for cluster
        
        Args:
            cluster: Name of cluster.

        Returns:
            Color as something that matplotlib understands.
        """
        return self._get_cluster_color(cluster, "cluster_colors")

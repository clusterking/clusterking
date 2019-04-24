#!/usr/bin/env python3

""" This subpackage provides classes to perform the actual clustering.

Different clustering algorithms correspond to different subclasses of the
base class :class:`clusterking.cluster.Cluster` (and inherit all of its
methods).

Currently implemented:

* :class:`~clusterking.cluster.HierarchyCluster`: Hierarchical clustering
  (https://en.wikipedia.org/wiki/Hierarchical_clustering/)
* :class:`~clusterking.cluster.KmeansCluster`: Kmeans clustering
  (https://en.wikipedia.org/wiki/K-means_clustering/)

"""

from clusterking.cluster.hierarchy_cluster import HierarchyCluster
from clusterking.cluster.kmeans_cluster import KmeansCluster
from clusterking.cluster.cluster import Cluster

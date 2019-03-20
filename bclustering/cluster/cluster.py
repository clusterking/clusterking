#!/usr/bin/env python3

"""Read the results from scan.py and get_clusters them.
"""

# standard
import atexit
import time

# 3rd party
import matplotlib.pyplot as plt

# us
from bclustering.data.dfmd import DFMD
from bclustering.util.metadata import git_info


# todo: allow initializing from either file or directly the dataframe and the
#  metadata
class Cluster(DFMD):
    # **************************************************************************
    # A:  Setup
    # **************************************************************************

    def __init__(self, *args, **kwargs):
        """ This class is subclassed to implement specific clustering
        algorithms and defines common functions.

        Args:
            # todo: write something here
        """
        if "log" not in kwargs:
            kwargs["log"] = "Cluster"
        # todo: write something about DFMD here
        super().__init__(*args, **kwargs)

        #: Metadata
        self.md["cluster"]["git"] = git_info(self.log)
        self.md["cluster"]["time"] = time.strftime("%a %_d %b %Y %H:%M",
                                                   time.gmtime())

        #: Should we wait for plots to be shown?
        self._wait_plots = False
        # call self.close() when this script exits
        atexit.register(self.close)

    # *************************************************************************
    # B:  Cluster
    # *************************************************************************

    def cluster(self, column="cluster", **kwargs):
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
        md = self.md["cluster"][column]

        for key, value in kwargs.items():
            md[key] = value

        clusters = self._cluster(**kwargs)

        n_clusters = len(set(clusters))
        self.log.info(
            "Clustering resulted in {} get_clusters.".format(n_clusters)
        )
        md["n_clusters"] = n_clusters

        self.df[column] = clusters

        self.rename_clusters_auto(column)

        self.log.info("Done")

    def _cluster(self, **kwargs):
        """ Implmentation of the clustering. Should return an array-like object
        with the cluster number.
        """
        raise NotImplementedError

    # **************************************************************************
    # C:  Utility
    # **************************************************************************

    def rename_clusters(self, old2new, column="cluster", new_column=None):
        """Renames the get_clusters. This also allows to merge several
        get_clusters by assigning them the same name.
        
        Args:
            old2new: Dictionary old name -> new name. If no mapping is defined
                for a key, it remains unchanged.
            column: The column with the original cluster numbers. 
            new_column: Write out as a new column with name `new_columns`, 
                e.g. when merging get_clusters with this method
        """
        clusters_old_unique = self.df[column].unique()
        # If a key doesn't appear in old2new, this means we don't change it.
        for cluster in clusters_old_unique:
            if cluster not in old2new:
                old2new[cluster] = cluster
        self.rename_clusters_apply(
            lambda name: old2new[name],
            column,
            new_column
        )

    def rename_clusters_apply(self, funct, column="cluster", new_column=None):
        """Apply method to cluster names. 
        
        Example:  Suppose your get_clusters are numbered from 1 to 10, but you
        want to start counting at 0:
        
        .. code-block:: python
            
            self.rename_clusters_apply(lambda i: i-1)
        
        Args:
            funct: Function to be applied to each cluster name (taking one 
                argument)
            column: The column with the original cluster numbers. 
            new_column: Write out as a new column with new name
            
        Returns:
            None
        """
        if not new_column:
            new_column = column
        self.df[new_column] = \
            [funct(cluster) for cluster in self.df[column].values]

    def rename_clusters_auto(self, column="cluster", new_column=None):
        """Try to name get_clusters in a way that doesn't depend on the 
        clustering algorithm (e.g. hierarchy clustering assigns names from 1 
        to n, whereas other cluster methods assign names from 0, etc.). 
        Right now, we simply change the names of the get_clusters in such a 
        way, that they are numbered from 0 to n-1 in an 'ascending' way with 
        respect to the order of rows in the dataframe. 
        
        Args:
            column: Column containing the cluster names
            new_column: Write out as a new column with new name
            
        Returns:
            None
        """
        old_cluster_names = self.df[column].unique()
        new_cluster_names = range(len(old_cluster_names))
        old2new = dict(zip(old_cluster_names, new_cluster_names))
        self.rename_clusters(old2new, column, new_column)

    # **************************************************************************
    # D:  MISC
    # **************************************************************************

    def close(self):
        """This method is called when this script exits. A corresponding
        hook has been set up in the __init__ method.
        We use that to wait for interactive plots/plotting windows to close
        if we made any. """
        if self._wait_plots:
            # this will block until all plotting windows were closed
            plt.show()

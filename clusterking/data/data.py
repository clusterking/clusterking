#!/usr/bin/env python3

# 3d
import numpy as np
from typing import Callable

# ours
from clusterking.data.dfmd import DFMD


# todo: docstrings
class Data(DFMD):
    """ A class which adds more convenience methods to DFMD. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # **************************************************************************
    # Property shortcuts
    # **************************************************************************

    @property
    def bin_cols(self):
        """ All columns that correspond to the bins of the
        distribution. This is automatically read from the
        metadata as set in e.g. the Scan.run. """
        columns = list(self.df.columns)
        # todo: more general?
        return [c for c in columns if c.startswith("bin")]

    @property
    def par_cols(self):
        """ All columns that correspond to the parameters (e.g. Wilso
        parameters). This is automatically read from the
        metadata as set in e.g. the Scan.run.
        """
        return self.md["scan"]["wpoints"]["coeffs"]

    @property
    def n(self):
        """ Number of points in parameter space that were sampled. """
        return len(self.df)

    @property
    def nbins(self):
        """ Number of bins of the distribution. """
        return len(self.bin_cols)

    @property
    def npars(self):
        """ Number of parameters that were sampled (i.e. number of dimensions
        of the sampled parameter space.
        """
        return len(self.par_cols)

    # **************************************************************************
    # Returning things
    # **************************************************************************

    def data(self, normalize=False) -> np.ndarray:
        """ Returns all histograms as a large matrix.

        Args:
            normalize: Normalize all histograms

        Returns:
            numpy.ndarray of shape self.n x self.nbins
        """
        data = self.df[self.bin_cols].values
        if normalize:
            # Reshaping here is important!
            return data / np.sum(data, axis=1).reshape((self.n, 1))
        else:
            return data

    def norms(self):
        """ Returns a vector of all normalizations of all histograms (where
        each histogram corresponds to one sampled point in parameter space).

        Returns:
            numpy.ndarray of shape self.n
        """
        return np.sum(self.data(), axis=1)

    def clusters(self, cluster_column="cluster"):
        return self.df[cluster_column].unique()

    # **************************************************************************
    # C:  Manipulating things
    # **************************************************************************

    # --------------------------------------------------------------------------
    # Renaming clusters
    # --------------------------------------------------------------------------

    # todo: doc
    # fixme: perhaps don't allow new_column but rather give copy method
    def rename_clusters(self, arg=None, column="cluster", new_column=None):
        if arg is None:
            self._rename_clusters_auto(column=column, new_column=new_column)
        elif isinstance(arg, dict):
            self._rename_clusters_dict(
                old2new=arg, column=column, new_column=new_column
            )
        elif isinstance(arg, Callable):
            self._rename_clusters_func(
                funct=arg, column=column, new_column=new_column
            )
        else:
            raise ValueError("Unsupported type ({}) for argument.".format(
                type(arg))
            )

    def _rename_clusters_dict(self, old2new, column="cluster", new_column=None):
        """Renames the clusters. This also allows to merge several
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
        self._rename_clusters_func(
            lambda name: old2new[name],
            column,
            new_column
        )

    def _rename_clusters_func(self, funct, column="cluster", new_column=None):
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

    def _rename_clusters_auto(self, column="cluster", new_column=None):
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

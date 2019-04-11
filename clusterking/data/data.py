#!/usr/bin/env python3

# std
import copy

# 3d
import numpy as np
from typing import Callable, Union, Iterable

# ours
from clusterking.data.dfmd import DFMD
from clusterking.plots.plot_bundles import BundlePlot
from clusterking.util.misc import check_matplot_inline


# todo: docstrings
class Data(DFMD):
    """ This class inherits from the :py:class:`~clusterking.data.dfmd.DFMD`
    class and adds additional methods to it. It is the basic container,
    that contains the

    * The distributions to cluster
    * The cluster numbers after clustering
    * The benchmark points after they are selected.
    """
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
        """ All columns that correspond to the parameters (e.g. Wilson
        parameters). This is automatically read from the
        metadata as set in e.g. the
        :py:meth:`clusterking.scan.scanner.Scanner.run`.
        """
        return self.md["scan"]["spoints"]["coeffs"]

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

    # todo: sorting
    def clusters(self, cluster_column="cluster"):
        """ Return numpy array of all cluster names (unique)

        Args:
            cluster_column: Column that contains the cluster names
        """
        return self.df[cluster_column].unique()

    # todo: test me
    def get_param_values(self, param: Union[None, str] = None):
        """ Return all unique values of this parameter

        Args:
            param: Name of parameter. If none is given, instead return a
                dictionary mapping of parameters to their values.

        Returns:

        """
        if param is None:
            return {
                param: self.get_param_values(param)
                for param in self.par_cols
            }
        return self.df[param].unique()

    # **************************************************************************
    # Subsample
    # **************************************************************************

    def only_bpoints(self, column="bpoint", inplace=False):
        """ Keep only the benchmark points as sample points.

        Args:
            column: benchmark point column (boolean)
            inplace: If True, the current Data object is modified, if False,
                a new copy of the Data object is returned.

        Returns:
            None or Data
        """
        if inplace:
            self.df = self.df[self.df[column]]
        else:
            new_obj = copy.deepcopy(self)
            new_obj.only_bpoints(inplace=True)
            return new_obj

    def _bpoint_slices(self, bpoint_column="bpoint"):
        """ See docstring of only_bpoint_slices. """
        bpoint_df = self.only_bpoints(column=bpoint_column)
        return {
            param: bpoint_df.df[param].unique()
            for param in self.par_cols
        }

    def only_bpoint_slices(self, bpoint_column="bpoint", inplace=False):
        """ For each sampled parameter, keep only those values that are attained
        by at least one benchmark point.

        For example, if you sample two parameters
        ``a`` and ``b`` as 10 values between 0 and 1 and you have 3 benchmark
        points ``(0, 0)``, ``(0.5, 0.5)`` and ``(0.5, 1)``, we only keep the
        set of sample points :math:`(a,b)\\in \{0, 0.5\}\\times \{0, 0.5, 1\}`.


        Args:
            bpoint_column: Column with benchmark points (default 'bpoints')
            inplace: Modify this Data object instead of returning a new one

        Returns:
            New data object if inplace==False, else None.
        """
        return self.fix_param(
            inplace=inplace,
            bpoints=False,
            **self._bpoint_slices(bpoint_column=bpoint_column)
        )

    # todo: test me
    # todo: order dict to avoid changing results
    def fix_param(self, inplace=False, bpoints=False,
                  bpoint_slices=False,
                  bpoint_column="bpoint",
                  **kwargs):
        """ Fix some parameter values to get a subset of sample points.

        Examples:

        .. code-block:: python

            d = Data("/path/to/tutorial/csv/folder", "tutorial_basics")

            # Return a new Data object, keeping the two values CT_bctaunutau
            # closest to -0.75 or 0.5
            d.fix_param(CT_bctaunutau=[-.75, 0.5])

            # Return a new Data object, where we also fix CSL_bctaunutau to the
            # value closest to -1.0
            d.fix_param(CT_bctaunutau=[-.75, 0.5], CSL_bctaunutau=-1.0)

        Args:
            inplace: Modify this Data object instead of returning a new one
            bpoints: Keep bpoints (no matter if they are selected by the other
                selection or not)
            bpoint_slices: Keep all parameter values that are 'used' by
                benchmark points (see
                :py:meth:`~clusterking.data.data.Data.only_bpoint_slices`).
            bpoint_column: Column with benchmark points (default 'bpoints')
                (for use with the ``bpoints`` option)
            **kwargs: Specify parameter values:
                Use ``<parameter name>=<value>`` or
                ``<parameter name>=[<value1>, ..., <valuen>]``.

        Returns:
            If ``inplace == True``, return new Data with subset of sample
            points.
        """
        if not inplace:
            new_obj = copy.deepcopy(self)
            new_obj.fix_param(inplace=True, bpoints=bpoints, **kwargs)
            return new_obj

        # From here on, we apply everything in place.

        if bpoint_slices:
            bpoint_slices = self._bpoint_slices(bpoint_column=bpoint_column)
        else:
            bpoint_slices = {param: [] for param in self.par_cols}

        # Prepare values
        values_dict = {}
        for param, values in kwargs.items():
            if not isinstance(values, Iterable):
                values_dict[param] = [values]
            if bpoint_slices:
                values_dict[param].extend(bpoint_slices[param])

        # Get selector
        selector = np.full(self.n, True, bool)
        for param, values in values_dict:
            param_selector = np.full(self.n, False, bool)
            for value in values:
                available_values = self.df[param].values
                idx = (np.abs(available_values - value)).argmin()
                nearest_value = available_values[idx]
                param_selector |= \
                    np.isclose(self.df[param].values, nearest_value)
            selector &= param_selector
        if bpoints:
            selector |= self.df[bpoint_column].astype(bool)

        # Apply selector to dataframe
        self.df = self.df[selector]

    # todo: test
    # todo: add usage example to docstring
    def sample_param(self,
                     bpoints=False,
                     bpoint_slices=False,
                     bpoint_column="bpoints",
                     inplace=False,
                     **kwargs):
        """ Return a Data object that contains a subset of the sample points
        (points in parameter space). Similar to Data.fix_param.

        Args:
            inplace: Modify this Data object instead of returning a new one
            bpoints: Keep bpoints (no matter if they are selected by the other
                selection or not)
            bpoint_slices: Keep all parameter values that are 'used' by
                benchmark points (see
                :py:meth:`~clusterking.data.data.Data.only_bpoint_slices`).
            bpoint_column: Column with benchmark points (default 'bpoints')
                (for use with the ``bpoints`` option)
            **kwargs: Specify parameter ranges:
                ``<coeff name>=(min, max, npoints)`` or
                ``<coeff name>=npoints``
                For each coeff (identified by <coeff name>), select (at most)
                npoints points between min and max.
                In total this will therefore result in npoints_{coeff_1} x ...
                x npoints_{coeff_npar} sample points (provided that there are
                enough sample points available).
                If a coefficient isn't contained in the dictionary, this
                dimension of the sample remains untouched.

        """
        fix_kwargs = {}
        for param, value in kwargs.items():
            if isinstance(value, Iterable):
                param_min, param_max, param_npoints = value
            elif isinstance(value, (int, float)):
                param_min = self.df[param].min()
                param_max = self.df[param].max()
                param_npoints = value
            else:
                raise ValueError(
                    "Incompatible type {} of {}".format(type(value), value)
                )
            fix_kwargs[param] = np.linspace(param_min, param_max, param_npoints)

        return self.fix_param(
            inplace=inplace,
            bpoints=bpoints,
            bpoint_slices=bpoint_slices,
            bpoint_column=bpoint_column,
            **fix_kwargs
        )

    # **************************************************************************
    # Manipulating things
    # **************************************************************************

    # --------------------------------------------------------------------------
    # Renaming clusters
    # --------------------------------------------------------------------------

    # todo: doc
    # fixme: perhaps don't allow new_column but rather give copy method
    def rename_clusters(self, arg=None, column="cluster", new_column=None):
        """ Rename clusters based on either

        1. A dictionary of the form ``{<old cluster name>: <new cluster name>}``
        2. A function that maps the old cluster name to the new cluster name

        Example for 2: Say our ``Data`` object ``d`` contains clusters 1 to 10
        in the default column ``cluster``. The following method call
        will instead use the numbers 0 to 9:

        .. code-block:: python

            d.rename_clusters(lambda x: x-1)

        Args:
            arg: Dictionary or function as described above.
            column: Column that contains the cluster names
            new_column: New column to write to (default None, i.e. rename in
                place)

        Returns:
            None
        """
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

    # **************************************************************************
    # Quick plots
    # **************************************************************************

    # todo: ideally, we could just copy the docstrings from plot_bundles etc,
    #  but that doesn't work here, because some of the arguments are attributes
    #  currently
    def plot_dist(self,
                  cluster_column="cluster",
                  bpoint_column="bpoint",
                  title=None,
                  clusters=None,
                  nlines=0,
                  bpoints=True,
                  legend=True):
        """Plot several examples of distributions for each cluster specified.

        Args:
            cluster_column: Column with the cluster names (default 'cluster')
            bpoint_column: Column with bpoints (default 'bpoint')
            title: Plot title (``None``: automatic)
            clusters: List of clusters to selected or single cluster.
                If None (default), all clusters are chosen.
            nlines: Number of example distributions of each cluster to be
                plotted (default 0)
            bpoints: Draw benchmark points (default True)
            legend: Draw legend? (default True)
            return_figure:

        Returns:
            Figure
        """
        bp = BundlePlot(self)
        bp.cluster_column = cluster_column
        bp.bpoint_column = bpoint_column
        bp.title = title
        bp.draw_legend = legend
        bp.plot_bundles(
            clusters=clusters,
            nlines=nlines,
            bpoints=bpoints
        )
        return bp.fig

    def plot_dist_minmax(self,
                         cluster_column="cluster",
                         bpoint_column="bpoint",
                         title=None,
                         clusters=None,
                         bpoints=True,
                         legend=True):
        """

        Args:
            cluster_column: Column with the cluster names (default 'cluster')
            bpoint_column: Column with bpoints (default 'bpoint')
            title: Plot title (``None``: automatic)
            clusters: List of clusters to selected or single cluster.
                If None (default), all clusters are chosen.
            bpoints: Draw benchmark points (default True)
            legend: Draw legend? (default True)

        Returns:
            Figure
        """
        bp = BundlePlot(self)
        bp.cluster_column = cluster_column
        bp.bpoint_column = bpoint_column
        bp.title = title
        bp.draw_legend = legend
        bp.plot_minmax(
            clusters=clusters,
            bpoints=bpoints
        )
        return bp.fig

    def plot_dist_box(self,
                      cluster_column="cluster",
                      bpoint_column="bpoint",
                      title=None,
                      clusters=None,
                      bpoints=True,
                      whiskers=2.5,
                      legend=True):
        """
        Box plot of the bin contents of the distributions corresponding
        to selected clusters.

        Args:
            cluster_column: Column with the cluster names (default 'cluster')
            bpoint_column: Column with bpoints (default 'bpoint')
            title: Plot title (``None``: automatic)
            clusters: List of clusters to selected or single cluster.
                If None (default), all clusters are chosen.
            bpoints: Draw benchmark points (default True)
            whiskers: Length of the whiskers of the box plot in units of IQR
                (interquartile range, containing 50% of all values). Default
                2.5.
            legend: Draw legend? (default True)

        Returns:
            Figure

        """
        bp = BundlePlot(self)
        bp.cluster_column = cluster_column
        bp.bpoint_column = bpoint_column
        bp.title = title
        bp.draw_legend = legend
        bp.box_plot(
            clusters=clusters,
            bpoints=bpoints,
            whiskers=whiskers
        )
        return bp.fig

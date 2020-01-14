#!/usr/bin/env python3

# std
import copy

# 3d
import numpy as np
from typing import Callable, Union, Iterable, List, Any, Optional, Dict

# ours
from clusterking.data.dfmd import DFMD
from clusterking.maths.metric_utils import (
    uncondense_distance_matrix,
    metric_selection,
)


class Data(DFMD):
    """ This class inherits from the :py:class:`~clusterking.data.DFMD`
    class and adds additional methods to it. It is the basic container,
    that contains

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
    def bin_cols(self) -> List[str]:
        """ All columns that correspond to the bins of the
        distribution. This is automatically read from the
        metadata as set in e.g. :meth:`clusterking.scan.Scanner.run`.
        """
        columns = list(self.df.columns)
        # todo: more general?
        return [c for c in columns if c.startswith("bin")]

    @property
    def par_cols(self) -> List[str]:
        """ All columns that correspond to the parameters (e.g. Wilson
        parameters). This is automatically read from the
        metadata as set in e.g. the
        :meth:`clusterking.scan.Scanner.run`.
        """
        return self.md["scan"]["spoints"]["coeffs"]

    @property
    def n(self) -> int:
        """ Number of points in parameter space that were sampled. """
        return len(self.df)

    @property
    def nbins(self) -> int:
        """ Number of bins of the distribution. """
        return len(self.bin_cols)

    @property
    def npars(self) -> int:
        """ Number of parameters that were sampled (i.e. number of dimensions
        of the sampled parameter space.
        """
        return len(self.par_cols)

    # todo: make not private?
    @property
    def _dist_xrange(self):
        """ Return minimum and maximum of x axis of the distributions.
        If not set, this will be 0 and number of bins.
        """
        binning = self.md["scan"]["dfunction"]["binning"]
        if not binning:
            # Fall back and give number of bins
            mini = 0
            maxi = self.nbins
        else:
            mini = min(binning)
            maxi = max(binning)
        return mini, maxi

    @property
    def _dist_vars(self):
        """ Return name of variables on x axis and y axis of the
        distribution. """
        xvar = self.md["scan"]["dfunction"]["xvar"]
        yvar = self.md["scan"]["dfunction"]["yvar"]
        if not xvar:
            xvar = None
        if not yvar:
            yvar = None
        return xvar, yvar

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

    def norms(self) -> np.ndarray:
        """ Returns a vector of all normalizations of all histograms (where
        each histogram corresponds to one sampled point in parameter space).

        Returns:
            numpy.ndarray of shape self.n
        """
        return np.sum(self.data(), axis=1)

    def clusters(self, cluster_column="cluster") -> List[Any]:
        """ Return list of all cluster names (unique)

        Args:
            cluster_column: Column that contains the cluster names
        """
        if cluster_column not in self.df.columns:
            raise ValueError(
                "The column '{}', which should contain the cluster names"
                " does not exist in the dataframe. Perhaps your data isn't "
                "clustered yet? Or you used a different name for the column "
                "(in this case, you can usually specify it with a "
                "'cluster_column' parameter).".format(cluster_column)
            )
        return sorted(list(self.df[cluster_column].unique()))

    # todo: test me
    def get_param_values(self, param: Optional[Union[None, str]] = None):
        """ Return all unique values of this parameter

        Args:
            param: Name of parameter. If none is given, instead return a
                dictionary mapping of parameters to their values.

        Returns:

        """
        if param is None:
            return {
                param: self.get_param_values(param) for param in self.par_cols
            }
        return self.df[param].unique()

    # **************************************************************************
    # Subsample
    # **************************************************************************

    def only_bpoints(self, bpoint_column="bpoint", inplace=False):
        """ Keep only the benchmark points as sample points.

        Args:
            bpoint_column: benchmark point column (boolean)
            inplace: If True, the current Data object is modified, if False,
                a new copy of the Data object is returned.

        Returns:
            None or Data
        """
        if inplace:
            self.df = self.df[self.df[bpoint_column]]
        else:
            # todo: this is inefficient
            new_obj = copy.deepcopy(self)
            new_obj.only_bpoints(inplace=True, bpoint_column=bpoint_column)
            return new_obj

    def _bpoint_slices(self, bpoint_column="bpoint"):
        """ See docstring of only_bpoint_slices. """
        bpoint_df = self.only_bpoints(bpoint_column=bpoint_column)
        return {param: bpoint_df.df[param].unique() for param in self.par_cols}

    # todo: test me
    # todo: order dict to avoid changing results
    def fix_param(
        self,
        inplace=False,
        bpoints=False,
        bpoint_slices=False,
        bpoint_column="bpoint",
        **kwargs
    ):
        """ Fix some parameter values to get a subset of sample points.

        Args:
            inplace: Modify this Data object instead of returning a new one
            bpoints: Keep bpoints (no matter if they are selected by the other
                selection or not)
            bpoint_slices: Keep all parameter values that are attained by
                benchmark points.
            bpoint_column: Column with benchmark points (default 'bpoints')
                (for use with the ``bpoints`` option)
            **kwargs: Specify parameter values:
                Use ``<parameter name>=<value>`` or
                ``<parameter name>=[<value1>, ..., <valuen>]``.

        Returns:
            If ``inplace == False``, return new Data with subset of sample
            points.

        Examples:

        .. code-block:: python

            d = Data("/path/to/tutorial/csv/folder", "tutorial_basics")

        Return a new Data object, keeping the two values ``CT_bctaunutau``
        closest to -0.75 or 0.5

        .. code-block:: python

            d.fix_param(CT_bctaunutau=[-.75, 0.5])

        Return a new Data object, where we also fix ``CSL_bctaunutau`` to the
        value closest to -1.0:

        .. code-block:: python

            d.fix_param(CT_bctaunutau=[-.75, 0.5], CSL_bctaunutau=-1.0)

        Return a new Data object, keeping the two values ``CT_bctaunutau``
        closest to -0.75 or 0.5, but make sure we do not discard any
        benchmark points in that process:

        .. code-block:: python

            d.fix_param(CT_bctaunutau=[-.75, 0.5], bpoints=True)

        Return a new Data object, keeping the two values ``CT_bctaunutau``
        closest to -0.75 or 0.5, but keep all values of ``CT_bctaunutau``
        that are attained by at least one benchmark point:

        .. code-block:: python

            d.fix_param(CT_bctaunutau=[-.75, 0.5], bpoint_slices=True)

        Return a new Data object, keeping only those values of
        ``CT_bctaunutau``, that are attained by at least one benchmark point:

        .. code-block:: python

            d.fix_param(CT_bctaunutau=[], bpoint_slice=True)

        """
        if not inplace:
            # todo: this is inefficient
            new_obj = copy.deepcopy(self)
            new_obj.fix_param(
                inplace=True,
                bpoints=bpoints,
                bpoint_slices=bpoint_slices,
                bpoint_column=bpoint_column,
                **kwargs
            )
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
            else:
                values_dict[param] = list(values)
            values_dict[param].extend(bpoint_slices[param])

        # Get selector
        selector = np.full(self.n, True, bool)
        for param, values in values_dict.items():
            param_selector = np.full(self.n, False, bool)
            for value in values:
                available_values = self.df[param].values
                idx = (np.abs(available_values - value)).argmin()
                nearest_value = available_values[idx]
                param_selector |= np.isclose(
                    self.df[param].values, nearest_value
                )
            selector &= param_selector
        if bpoints:
            selector |= self.df[bpoint_column].astype(bool)

        # Apply selector to dataframe
        self.df = self.df[selector]

    # todo: test
    def sample_param(
        self,
        bpoints=False,
        bpoint_slices=False,
        bpoint_column="bpoint",
        inplace=False,
        **kwargs
    ):
        """ Return a Data object that contains a subset of the sample points
        (points in parameter space). Similar to Data.fix_param.

        Args:
            inplace: Modify this Data object instead of returning a new one
            bpoints: Keep bpoints (no matter if they are selected by the other
                selection or not)
            bpoint_slices: Keep all parameter values that are attained by
                benchmark points
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

        Returns:
            If ``inplace == False``, return new Data with subset of sample
            points.

        Examples:

        .. code-block:: python

            d = Data("/path/to/tutorial/csv/folder", "tutorial_basics")

        Return a new Data object, keeping subsampling ``CT_bctaunutau``
        closest to 5 values between -1 and 1:

        .. code-block:: python

            d.sample_param(CT_bctaunutau=(-1, 1, 10))

        The same in shorter syntax
        (because -1 and 1 are the minimum and maximum of the parameter)

        .. code-block:: python

            d.sample_param(CT_bctaunutau=10)

        For the ``bpoints`` and ``bpoint_slices`` syntax, see the documenation
        of :py:meth:`clusterking.data.Data.fix_param`.
        """
        fix_kwargs = {}
        for param, value in kwargs.items():
            if isinstance(value, Iterable):
                try:
                    param_min, param_max, param_npoints = value
                except ValueError:
                    raise ValueError(
                        "Please specify minimum, maximum and number of points."
                    )
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

    def sample_param_random(
        self, inplace=False, bpoints=False, bpoint_column="bpoint", **kwargs
    ):
        """ Random subsampling in parameter space.

        Args:
            inplace: Modify this Data object instead of returning a new one
            bpoints: Keep bpoints (no matter if they are selected by the other
                selection or not)
            bpoint_column: Column with benchmark points (default 'bpoints')
                (for use with the ``bpoints`` option)
            **kwargs: Arguments for :meth:`pandas.DataFrame.sample`

        Returns:
            If ``inplace == False``, return new Data with subset of sample
            points.
        """
        if not inplace:
            # todo: this is inefficient, why do we have to copy everything
            #  first?
            new = self.copy()
            new.sample_param_random(
                inplace=True,
                bpoints=bpoints,
                bpoint_column=bpoint_column,
                **kwargs
            )
            return new
        if not bpoints:
            self.df = self.df.sample(**kwargs)
        else:
            bpoint_df = self.df[self.df[bpoint_column]]
            self.df = self.df[~self.df[bpoint_column]].sample(**kwargs)
            self.df = self.df.append(bpoint_df)

    def find_closest_spoints(self, point: Dict[str, float], n=10) -> "Data":
        """ Given a point in parameter space, find the closest sampling
        points to it and return them as a :py:class:`Data` object with the
        corresponding subset of spoints.
        The order of the rows in the dataframe :py:attr:`Data.df` will be in
        order of increasing parameter space distance from the given point.

        Args:
            point: Dictionary of parameter name to value
            n: Maximal number of rows to return

        Returns:
            :py:class:`Data` object with subset of rows of dataframe
            corresponding to the closest points in parameter space.
        """
        if not set(point.keys()) == set(self.par_cols):
            raise ValueError(
                "Invalid specification of a point: Please give values"
                " exactly for the following keys: {}".format(
                    ", ".join(self.par_cols)
                )
            )
        if n <= 0:
            raise ValueError("n has to be an integer >= 1.")

        # argpartition will throw if we request more or equal rows than we have,
        # so we have to be careful
        n_max = self.n
        if n_max == 0:
            raise ValueError("Not enough rows available.")
        if n_max < n:
            n_max = n

        distances = np.sqrt(
            np.sum(
                np.array(
                    [
                        np.square(self.df[param].values - point[param])
                        for param in self.par_cols
                    ]
                ),
                axis=0,
            )
        )

        if n < n_max:
            closest = np.argpartition(distances, n)[:n]
        else:
            # n == n_max
            closest = np.arange(0, len(distances))
        # note that argpartition did not sort these yet, so we do this now
        closest = closest[np.argsort(distances[closest])]

        new = self.copy(data=False)
        new.df = self.df.iloc[closest]
        return new

    def find_closest_bpoints(
        self, point: Dict[str, float], n=10, bpoint_column="bpoint"
    ):
        """ Given a point in parameter space, find the closest benchmark
        points to it and return them as a :py:class:`Data` object with the
        corresponding subset of benchmark points.
        The order of the rows in the dataframe :py:attr:`Data.df` will be in
        order of increasing parameter space distance from the given point.

        Args:
            point: Dictionary of parameter name to value
            n: Maximal number of rows to return
            bpoint_column: Column name of the benchmark column

        Returns:
            :py:class:`Data` object with subset of rows of dataframe
            corresponding to the closest points in parameter space.
        """
        if not set(point.keys()) == set(self.par_cols):
            raise ValueError(
                "Invalid specification of a point: Please give values"
                " exactly for the following keys: {}".format(
                    ", ".join(self.par_cols)
                )
            )
        if n <= 0:
            raise ValueError("n has to be an integer >= 1.")

        # argpartition will throw if we request more or equal rows than we have,
        # so we have to be careful
        n_max = len(self.df[self.df[bpoint_column]])
        if n_max == 0:
            raise ValueError("Not enough rows available.")
        if n_max < n:
            n_max = n

        distances = np.sqrt(
            np.sum(
                np.array(
                    [
                        np.square(
                            self.df[self.df[bpoint_column]][param].values
                            - point[param]
                        )
                        for param in self.par_cols
                    ]
                ),
                axis=0,
            )
        )

        if n < n_max:
            closest = np.argpartition(distances, n)[:n]
        else:
            # n == n_max
            closest = np.arange(0, len(distances))
        # note that argpartition did not sort these yet, so we do this now
        closest = closest[np.argsort(distances[closest])]

        new = self.copy(data=False)
        new.df = self.df[self.df[bpoint_column]].iloc[closest]
        return new

    # **************************************************************************
    # Manipulating things
    # **************************************************************************

    def configure_variable(self, variable, axis_label=None):
        """ Set additional information for variables, e.g. the variable on the
        x axis of the plots of the distribution or the parameters.

        Args:
            variable: Name of the variable
            axis_label: An alternate name which will be used on the axes of
                plots.
        """
        if axis_label is not None:
            self.md["variables"][variable]["axis_label"] = axis_label

    def _get_axis_label(self, variable):
        r = self.md["variables"][variable]["axis_label"]
        if r:
            return r
        else:
            return variable

    # --------------------------------------------------------------------------
    # Renaming clusters
    # --------------------------------------------------------------------------

    # todo: Test this
    # todo: inplace?
    # fixme: perhaps don't allow new_column but rather give copy method
    def rename_clusters(self, arg=None, column="cluster", new_column=None):
        """ Rename clusters based on either

        1. A dictionary of the form ``{<old cluster name>: <new cluster name>}``
        2. A function that maps the old cluster name to the new cluster name

        Example for 2: Say our :py:class:`~clusterking.data.Data`
        object ``d`` contains clusters 1 to 10
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
            raise ValueError(
                "Unsupported type ({}) for argument.".format(type(arg))
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
            lambda name: old2new[name], column, new_column
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
        self.df[new_column] = [
            funct(cluster) for cluster in self.df[column].values
        ]

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
        old_cluster_names = sorted(list(self.df[column].unique()))
        new_cluster_names = range(len(old_cluster_names))
        old2new = dict(zip(old_cluster_names, new_cluster_names))
        self.rename_clusters(old2new, column, new_column)

    # **************************************************************************
    # Quick plots
    # **************************************************************************

    def plot_dist(
        self,
        cluster_column="cluster",
        bpoint_column="bpoint",
        title: Optional[str] = None,
        clusters: Optional[List[int]] = None,
        nlines=None,
        bpoints=True,
        legend=True,
        ax=None,
        hist_kwargs: Optional[Dict[str, Any]] = None,
        hist_kwargs_bp: Optional[Dict[str, Any]] = None,
    ):
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
            ax: Instance of `matplotlib.axes.Axes` to plot on. If None, a new
                one is instantiated.
            hist_kwargs: Keyword arguments passed on to
                :meth:`~clusterking.plots.plot_histogram.plot_histogram`
            hist_kwargs_bp: Like ``hist_kwargs`` but used for benchmark points.
                If ``None``, ``hist_kwargs`` is used.

        Note: To customize these kind of plots further, check the
        :py:class:`~clusterking.plots.plot_bundles.BundlePlot` class and the
        :py:meth:`~clusterking.plots.plot_bundles.BundlePlot.plot_bundles`
        method thereof.

        Returns:
            Figure
        """
        from clusterking.plots.plot_bundles import BundlePlot

        bp = BundlePlot(self)
        bp.cluster_column = cluster_column
        bp.bpoint_column = bpoint_column
        bp.title = title
        bp.draw_legend = legend
        bp.plot_bundles(
            ax=ax,
            clusters=clusters,
            nlines=nlines,
            bpoints=bpoints,
            hist_kwargs=hist_kwargs,
            hist_kwargs_bp=hist_kwargs_bp,
        )
        return bp.fig

    def plot_dist_minmax(
        self,
        cluster_column="cluster",
        bpoint_column="bpoint",
        title: Optional[str] = None,
        clusters: Optional[List[int]] = None,
        bpoints=True,
        legend=True,
        ax=None,
        hist_kwargs: Optional[Dict[str, Any]] = None,
        fill_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """ Plot the minimum and maximum of each bin for the specified
        clusters.

        Args:
            cluster_column: Column with the cluster names (default 'cluster')
            bpoint_column: Column with bpoints (default 'bpoint')
            title: Plot title (``None``: automatic)
            clusters: List of clusters to selected or single cluster.
                If None (default), all clusters are chosen.
            bpoints: Draw benchmark points (default True)
            legend: Draw legend? (default True)
            ax: Instance of `matplotlib.axes.Axes` to plot on. If None, a new
                one is instantiated.
            hist_kwargs: Keyword arguments to
                :meth:`~clusterking.plots.plot_histogram.plot_histogram`
            fill_kwargs: Keyword arguments to`matplotlib.pyplot.fill_between`

        Note: To customize these kind of plots further, check the
        :py:class:`~clusterking.plots.plot_bundles.BundlePlot` class and the
        :py:meth:`~clusterking.plots.plot_bundles.BundlePlot.plot_minmax`
        method thereof.

        Returns:
            Figure
        """
        from clusterking.plots.plot_bundles import BundlePlot

        bp = BundlePlot(self)
        bp.cluster_column = cluster_column
        bp.bpoint_column = bpoint_column
        bp.title = title
        bp.draw_legend = legend
        bp.plot_minmax(
            clusters=clusters,
            bpoints=bpoints,
            hist_kwargs=hist_kwargs,
            fill_kwargs=fill_kwargs,
            ax=ax,
        )
        return bp.fig

    def plot_dist_box(
        self,
        cluster_column="cluster",
        bpoint_column="bpoint",
        title: Optional[str] = None,
        clusters: Optional[List[int]] = None,
        bpoints=True,
        whiskers=2.5,
        legend=True,
        ax=None,
        boxplot_kwargs: Optional[Dict[str, Any]] = None,
        hist_kwargs: Optional[Dict[str, Any]] = None,
    ):
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
            boxplot_kwargs: Arguments to `matplotlib.pyplot.boxplot`
            ax: Instance of `matplotlib.axes.Axes` to plot on. If None, a new
                one is instantiated.
            boxplot_kwargs: Keyword arguments to `matplotlib.pyplot.boxplot`
            hist_kwargs: Keyword arguments to
                :meth:`~clusterking.plots.plot_histogram.plot_histogram`

        Note: To customize these kind of plots further, check the
        :py:class:`~clusterking.plots.plot_bundles.BundlePlot` class and the
        :py:meth:`~clusterking.plots.plot_bundles.BundlePlot.box_plot`
        method thereof.

        Returns:
            Figure

        """
        from clusterking.plots.plot_bundles import BundlePlot

        bp = BundlePlot(self)
        bp.cluster_column = cluster_column
        bp.bpoint_column = bpoint_column
        bp.title = title
        bp.draw_legend = legend
        bp.box_plot(
            clusters=clusters,
            bpoints=bpoints,
            whiskers=whiskers,
            boxplot_kwargs=boxplot_kwargs,
            hist_kwargs=hist_kwargs,
            ax=ax,
        )
        return bp.fig

    def plot_clusters_scatter(
        self,
        params=None,
        clusters=None,
        cluster_column="cluster",
        bpoint_column="bpoint",
        legend=True,
        max_subplots=16,
        max_cols=4,
        markers=("o", "v", "^", "v", "<", ">"),
        figsize=4,
        aspect_ratio=None,
    ):
        """
        Create scatter plot, specifying the columns to be on the axes of the
        plot. If 3 column are specified, 3D scatter plots
        are presented, else 2D plots. If the dataframe contains more columns,
        such that each row is not only specified by the columns on the axes,
        a selection of subplots is created, showing 'cuts'.
        Benchmark points are marked by enlarged plot markers.

        Args:
            params: The names of the columns to be shown on the x, (y, (z))
               axis of the plots.
            clusters: The get_clusters to be plotted (default: all)
            cluster_column: Column with the cluster names (default 'cluster')
            bpoint_column: Column with bpoints (default 'bpoint')
            legend: Draw legend? (default True)
            max_subplots: Maximal number of subplots
            max_cols: Maximal number of columns of the subplot grid
            markers: List of markers of the get_clusters
            figsize: Base size of each subplot
            aspect_ratio: Aspect ratio of 2D plots. If None, will be chosen
                automatically based on data ranges.

        Returns:
            Figure
        """
        from clusterking.plots.plot_clusters import ClusterPlot

        if params is None:
            if len(self.par_cols) in [1, 2, 3]:
                params = self.par_cols[:]
            else:
                raise ValueError("Please specify parameter 'params'.")

        cp = ClusterPlot(self)
        cp.cluster_column = cluster_column
        cp.bpoint_column = bpoint_column
        cp.draw_legend = legend
        cp.max_subplots = max_subplots
        cp.max_cols = max_cols
        cp.markers = markers
        cp.fig_base_size = figsize
        cp.aspect_ratio = aspect_ratio
        cp.scatter(params, clusters=clusters)
        return cp.fig

    def plot_clusters_fill(
        self,
        params=None,
        cluster_column="cluster",
        bpoint_column="bpoint",
        legend=True,
        max_subplots=16,
        max_cols=4,
        figsize=4,
        aspect_ratio=None,
    ):
        """
        Call this method with two column names, x and y. The results are
        similar to those of 2D scatter plots as created by the scatter
        method, except that the coloring is expanded to the whole xy plane.
        Note: This method only works with uniformly sampled NP!

        Args:
            params: The names of the columns to be shown on the x, y (and z)
               axis of the plots.
            cluster_column: Column with the cluster names (default 'cluster')
            bpoint_column: Column with bpoints (default 'bpoint')
            legend: Draw legend? (default True)
            max_subplots: Maximal number of subplots
            max_cols: Maximal number of columns of the subplot grid
            figsize: Base size of each subplot
            aspect_ratio: Aspect ratio of 2D plots. If None, will be chosen
                automatically based on data ranges.

        Returns:
            Figure
        """
        from clusterking.plots.plot_clusters import ClusterPlot

        if params is None:
            if len(self.par_cols) in [2]:
                params = self.par_cols[:]
            else:
                raise ValueError("Please specify parameter 'params'.")

        cp = ClusterPlot(self)
        cp.cluster_column = cluster_column
        cp.bpoint_column = bpoint_column
        cp.draw_legend = legend
        cp.max_subplots = max_subplots
        cp.max_cols = max_cols
        cp.fig_base_size = figsize
        cp.aspect_ratio = aspect_ratio
        cp.fill(params)
        return cp.fig

    def plot_bpoint_distance_matrix(
        self,
        cluster_column="cluster",
        bpoint_column="bpoint",
        metric="euclidean",
        ax=None,
    ):
        """ Plot the pairwise distances of all benchmark points.

        Args:
            cluster_column: Column with the cluster names (default 'cluster')
            bpoint_column: Column with bpoints (default 'bpoint')
            metric: String or function. See
                :func:`clusterking.maths.metric.metric_selection`. Default: Euclidean
                distance.
            ax: Matplotlib axes or None (automatic)

        Returns:
            Figure
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        bpoints = self.only_bpoints(bpoint_column=bpoint_column).copy(True)
        bpoints.df.sort_values(cluster_column, inplace=True)
        metric_funct = metric_selection(metric)
        distance_matrix = uncondense_distance_matrix(metric_funct(bpoints))
        cax = ax.matshow(distance_matrix)
        fig.colorbar(cax)

        cluster_labels = bpoints.df[cluster_column].tolist()
        ax.set_xticklabels([""] + cluster_labels)
        ax.set_yticklabels([""] + cluster_labels)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Cluster")
        ax.set_title("Pairwise distances of benchmark points")
        ax.tick_params(
            axis="x", bottom=True, top=True, labelbottom=True, labeltop=False
        )

        return fig

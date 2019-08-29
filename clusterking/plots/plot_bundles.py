#!/usr/bin/env python3

# std
import logging
import random
from typing import List, Iterable, Union, Optional, Dict, Any
from distutils.version import StrictVersion

# 3rd party
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.animation as animation

# ours
from clusterking.util.log import get_logger
from clusterking.plots.plot_histogram import plot_histogram, plot_histogram_fill
from clusterking.plots.colors import ColorScheme


def get_random_indizes(start: int, stop: int, n: int) -> List[int]:
    """ Generate a list of n distinct (!) random integers.

    Args:
        start: Minimum of index (start <= index)
        stop: Maximum of index (index < stop)
        n: Number of distinct random indizes to be generated

    Returns:
        List `number` many (different) random indizes
    """
    indizes = set()
    iterations = 0
    while len(indizes) < n:
        indizes.add(random.randrange(start, stop))
        if iterations >= 10 * n:
            print(
                "Did not manage to generate enough different random "
                "integers (only {} of {}).".format(len(indizes), n)
            )
            break
    return sorted(list(indizes))


class BundlePlot(object):
    """ Plotting class to plot distributions by cluster in order to analyse
    which distributions get assigned to which cluster. """

    def __init__(self, data):
        """

        Args:
            data: :py:class:`~clusterking.data.data.Data` object
        """

        #: logging.Logger object
        self.log = get_logger("BundlePlot", sh_level=logging.WARNING)

        #: pandas dataframe
        self.data = data

        #: Name of the column holding the cluster number
        self.cluster_column = "cluster"

        self.bpoint_column = "bpoint"

        #: Color scheme
        # fixme: this will be problematic if I reinitialize this
        if self._has_clusters:
            self.color_scheme = ColorScheme(self._clusters)
        else:
            self.color_scheme = ColorScheme([0])

        #: Draw legend?
        self.draw_legend = True

        #: Override default titles with this title. If None, the default title
        #:  is used.
        self.title = None

        #: Instance of matplotlib.axes.Axes
        self.ax = None

    @property
    def fig(self):
        """ Instance of matplotlib.pyplot.figure """
        return self.ax.get_figure()

    @property
    def xrange(self):
        """ Range of the xaxis """
        return self.data._dist_xrange

    @property
    def xlabel(self):
        if self.data._dist_vars[0]:
            return self.data._get_axis_label(self.data._dist_vars[0])
        return None

    @property
    def ylabel(self):
        if self.data._dist_vars[1]:
            return self.data._get_axis_label(self.data._dist_vars[1])
        return None

    @property
    def _bins(self):
        if self.data.md["scan"]["dfunction"]["binning_mode"] == "integrate":
            return self.data.md["scan"]["dfunction"]["binning"]
        return np.array(range(0, self.data.nbins + 1))

    # **************************************************************************
    # Internal helpers
    # **************************************************************************

    @property
    def _has_bpoints(self):
        """ Do we have benchmark points? """
        return self.bpoint_column in self.data.df.columns

    @property
    def _has_clusters(self):
        """ Do we have clustered data? """
        return self.cluster_column in self.data.df.columns

    @property
    def _clusters(self):
        """ Return array of all distinct clusters. """
        return self.data.clusters(cluster_column=self.cluster_column)

    def _filter_clusters(self, clusters: Iterable[int]) -> List[int]:
        """ Return list of existing clusters only. """
        clusters = list(set(clusters))
        selection = [c for c in clusters if c in self._clusters]
        removed = [c for c in clusters if c not in self._clusters]
        if removed:
            self.log.warning(
                "The cluster(s) {} does not exist in data, "
                "so I removed them.".format(
                    ", ".join(map(str, sorted(removed)))
                )
            )
        return selection

    def _interpret_cluster_input(self, clusters=None) -> List[int]:
        """ Flexible handling of user specifications for clusters.

        Args:
            clusters: Either None (all clusters) a single int (just that
                cluster or a list of clusters).

        Returns:
            List of selected clusters. If no clusters are available at all, an
            empty list is returned.

        Raises:
            If clusters are requested but None are available, a ValueError is
            raised
        """
        if not self._has_clusters:
            if clusters is None:
                return []
            else:
                raise ValueError(
                    "No cluster information available, but individual clusters"
                    " were requested."
                )
        if isinstance(clusters, int):
            clusters = [clusters]
        if not clusters:
            clusters = self._clusters
        return self._filter_clusters(clusters)

    # todo: getting the bpoint should be a different function
    def _get_df_cluster(
        self, cluster: Union[None, int], bpoint=None, bpoint_return_index=False
    ) -> pd.DataFrame:
        """ Return only the rows corresponding to one cluster in the
        dataframe and only the columns that correspond to the bins.

        Args:
            cluster: Name of the cluster. If ``None``, no cluster selection
                will be done.
            bpoint: If True, return benchmark point, if False, return all non-
                benchmark points, if None, return everything.

        Returns:
            pandas.DataFrame as described above
        """
        # to avoid long line:
        cc = self.cluster_column
        bc = self.data.bin_cols
        if cluster is not None:
            df = self.data.df[self.data.df[cc] == cluster]
        else:
            df = self.data.df
        if bpoint is None:
            return df[bc]
        elif bpoint is False:
            if self._has_bpoints:
                return df[df[self.bpoint_column] == False][bc]
            else:
                return df[bc]
        elif bpoint is True:
            if self._has_bpoints:
                df_bp = df[df[self.bpoint_column] == True]
                assert len(df_bp) == 1
                if not bpoint_return_index:
                    return df_bp[bc]
                else:
                    return df_bp[bc], df_bp.index[0]

            else:
                return pd.DataFrame()
        else:
            raise ValueError("Invalid argument bpoint=={}".format(bpoint))

    def _get_df_cluster_err_high(self, index):
        indices = list(self.data.df.index)
        loc = indices.index(index)
        # todo: this is horribly inefficient
        return self.data.err()[loc]

    def _get_df_cluster_err_low(self, *args, **kwargs):
        return self._get_df_cluster_err_high(*args, **kwargs)

    def _set_ax(self, ax, title):
        """ Set up axes. """
        if len(self.data.df) == 0:
            raise ValueError(
                "No data to plot. Please check if your dataframe contains "
                "any row."
            )
        if self.title is not None:
            title = self.title
        if not ax:
            fig, ax = plt.subplots()
            self.ax = ax
        ax.set_title(title)
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)

    # **************************************************************************
    # Plots
    # **************************************************************************

    # --------------------------------------------------------------------------
    # Legend
    # --------------------------------------------------------------------------

    def _draw_legend(self, clusters=None):
        # todo: Should be multi column legend if we have too many patches...
        if not self._has_clusters:
            return
        if not self.draw_legend:
            return
        clusters = self._interpret_cluster_input(clusters)
        if len(clusters) <= 1:
            return
        legend_elements = []
        for cluster in clusters:
            color = self.color_scheme.get_cluster_color(cluster)
            # pycharm can't seem to find patches:
            # noinspection PyUnresolvedReferences
            p = matplotlib.patches.Patch(
                facecolor=color, edgecolor=color, label=cluster
            )
            legend_elements.append(p)
        self.ax.legend(
            handles=legend_elements, loc="best", title="Clusters", frameon=False
        )

    # --------------------------------------------------------------------------
    # Benchmark points + more lines
    # --------------------------------------------------------------------------

    def _plot_bundles(
        self,
        cluster: Union[None, int],
        nlines=0,
        benchmark=True,
        hist_kwargs: Optional[Dict[str, Any]] = None,
        hist_kwargs_bp: Optional[Dict[str, Any]] = None,
    ) -> None:
        """ Main implementation of self.plot_bundles (private method).
        This method will be called for each cluster in self.plot_bundles.

        Args:
            cluster: Number of cluster to be plotted
            nlines: Number of example distributions of the cluster to be
                plotted
            hist_kwargs: See :meth:`plot_bundles`,
            hist_kwargs_bp: See :meth:`plot_bundles`

        Returns:
            None
        """

        if hist_kwargs is None:
            hist_kwargs = {}
        if hist_kwargs_bp is None:
            hist_kwargs = hist_kwargs.copy()

        df_cluster_no_bp = self._get_df_cluster(cluster, bpoint=False)
        if len(df_cluster_no_bp) < nlines:
            self.log.warning(
                "Not enough rows for cluster {} "
                "Only plotting {} lines.".format(cluster, len(df_cluster_no_bp))
            )
            nlines = len(df_cluster_no_bp)
        df_cluster_bp = self._get_df_cluster(cluster, bpoint=True)

        indizes = get_random_indizes(0, len(df_cluster_no_bp), nlines)
        if cluster is None:
            color = self.color_scheme.get_cluster_color(0)
            colors = self.color_scheme.get_cluster_colors_faded(0, nlines)
        else:
            color = self.color_scheme.get_cluster_color(cluster)
            colors = self.color_scheme.get_cluster_colors_faded(cluster, nlines)
        if nlines == 1 and not benchmark:
            # Do not use faded out color if we just plot one line
            colors = [color]

        for i, index in enumerate(indizes):
            data = np.squeeze(df_cluster_no_bp.iloc[[index]].values)
            plot_histogram(
                self.ax,
                self._bins,
                data,
                color=colors[i],
                linestyle="-",
                **hist_kwargs
            )
        if self._has_bpoints and benchmark:
            plot_histogram(
                self.ax,
                self._bins,
                df_cluster_bp.values,
                color=color,
                **hist_kwargs
            )

    def plot_bundles(
        self,
        clusters: Optional[Union[None, int, Iterable[int]]] = None,
        nlines=None,
        ax=None,
        bpoints=True,
        hist_kwargs: Optional[Dict[str, Any]] = None,
        hist_kwargs_bp: Optional[Dict[str, Any]] = None,
    ) -> None:
        """ Plot several examples of distributions for each cluster specified

        Args:
            clusters: List of clusters to selected or single cluster.
                If None (default), all clusters are chosen.
            nlines: Number of example distributions of each cluster to be
                plotted. Defaults to 0 if we plot benchmark points and 3
                otherwise.
            ax: Instance of matplotlib.axes.Axes to be plotted on. If None
                (default), a new axes object and figure is initialized and
                saved as self.ax and self.fig.
            bpoints: Draw benchmark curve
            hist_kwargs: Keyword arguments passed on to
                :meth:`~clusterking.plots.plot_histogram.plot_histogram`
            hist_kwargs_bp: Like ``hist_kwargs`` but used for benchmark points.
                If ``None``, ``hist_kwargs`` is used.

        Returns:
            None
        """
        clusters = self._interpret_cluster_input(clusters)

        if nlines is None:
            if self._has_bpoints and bpoints:
                nlines = 0
            else:
                nlines = 3

        _title = []
        if self._has_bpoints:
            _title.append("benchmark point(s)")
        if nlines:
            if self._has_bpoints:
                _title.append("+")
            _title.append("{} sample point(s) ".format(nlines))
        if clusters:
            _title.append(
                "for cluster(s) {}".format(
                    ", ".join(map(str, sorted(clusters)))
                )
            )
        self._set_ax(ax, " ".join(_title))

        # pycharm might be confused about the type of `clusters`:
        # noinspection PyTypeChecker
        for cluster in clusters:
            self._plot_bundles(
                cluster,
                nlines=nlines,
                benchmark=bpoints,
                hist_kwargs=hist_kwargs,
                hist_kwargs_bp=hist_kwargs_bp,
            )
        if not clusters:
            self._plot_bundles(
                cluster=None,
                nlines=nlines,
                benchmark=False,
                hist_kwargs=hist_kwargs,
                hist_kwargs_bp=hist_kwargs_bp,
            )

        self._draw_legend(clusters)

    # todo: doc
    def animate_bundle(self, cluster, n, benchmark=True):
        # There seems to be some underlying magic here with fig
        fig = plt.figure()
        ax = fig.gca()
        self.ax = ax
        linestyle = "-"
        if benchmark:
            self._plot_bundles(cluster, 0, benchmark=True)
            linestyle = "--"
        ims = []
        df_cluster_no_bp = self._get_df_cluster(cluster, bpoint=False)
        color = self.color_scheme.get_cluster_color(cluster)
        for i in range(n):
            index = random.randrange(0, len(df_cluster_no_bp))
            contents = np.squeeze(df_cluster_no_bp.iloc[[index]].values)
            contents = np.append(contents, contents[-1])
            edges = np.arange(len(contents))

            ims.append(
                plt.step(
                    edges,
                    contents,
                    where="post",
                    color=color,
                    linestyle=linestyle,
                )
            )

        # self._set_ax(None, "Animated sample points")
        anim = animation.ArtistAnimation(
            fig, ims, interval=500, repeat_delay=3000, blit=True
        )
        # In order to display this in the notebook, use
        # from IPython.display import HTML
        # HTML(anim.to_html5_video())
        return anim

    # --------------------------------------------------------------------------
    # Minima/Maxima of bin content for each cluster
    # --------------------------------------------------------------------------

    def _plot_minmax(
        self,
        cluster: Union[None, int],
        bpoints=True,
        hist_kwargs: Optional[Dict[str, Any]] = None,
        fill_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """ Main implementation of self.plot_minmax.
        This method will be called for each cluster in self.plot_minmax.

        Args:
            cluster: Name of cluster to be plotted or None if there are no
                clusters
            bpoints: Plot benchmark points
            hist_kwargs: See :meth:`plot_minmax`
            fill_kwargs: See :meth:`plot_minmax`

        Returns:
            None
        """
        df_cluster = self._get_df_cluster(cluster)
        maxima = list(df_cluster.max().values)
        minima = list(df_cluster.min().values)

        if fill_kwargs is None:
            fill_kwargs = {}

        if cluster is not None:
            color = self.color_scheme.get_cluster_color(cluster)
        else:
            color = self.color_scheme.get_cluster_color(0)

        for i in range(len(maxima)):
            x = self._bins[i : (i + 2)]
            y1 = [minima[i], minima[i]]
            y2 = [maxima[i], maxima[i]]
            fb_kwargs = dict(
                facecolor=color,
                interpolate=False,
                alpha=0.3,
                hatch="////",
                color=color,
            )
            fb_kwargs.update(fill_kwargs)
            self.ax.fill_between(x, y1, y2, **fb_kwargs)
        if bpoints:
            self._plot_bundles(
                cluster, nlines=0, benchmark=True, hist_kwargs=hist_kwargs
            )

    def plot_minmax(
        self,
        clusters: Optional[Union[int, Iterable[int]]] = None,
        ax=None,
        bpoints=True,
        hist_kwargs: Optional[Dict[str, Any]] = None,
        fill_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """ Plot the minimum and maximum of each bin for the specified
        clusters.

        Args:
            clusters:  List of clusters to selected or single cluster.
                If None (default), all clusters are chosen.
            ax: Instance of ``matplotlib.axes.Axes`` to plot on. If None, a new
                one is instantiated.
            bpoints: Plot benchmark points
            hist_kwargs: Keyword arguments to
                :meth:`~clusterking.plots.plot_histogram.plot_histogram`
            fill_kwargs: Keyword arguments to`matplotlib.pyplot.fill_between`

        Returns:
            None
        """
        clusters = self._interpret_cluster_input(clusters)

        _title = ["Minima and maxima of the bin contents"]
        if self._has_clusters:
            _title.append(
                "for cluster(s) {}".format(
                    ", ".join(map(str, sorted(clusters)))
                )
            )
        self._set_ax(ax, " ".join(_title))

        # pycharm might be confused about the type of `clusters`:
        # noinspection PyTypeChecker
        for cluster in clusters:
            self._plot_minmax(
                cluster,
                bpoints=bpoints,
                hist_kwargs=hist_kwargs,
                fill_kwargs=fill_kwargs,
            )
        if not clusters:
            self._plot_minmax(
                None,
                bpoints=bpoints,
                hist_kwargs=hist_kwargs,
                fill_kwargs=fill_kwargs,
            )

        self._draw_legend(clusters)

    # --------------------------------------------------------------------------
    # Plot with errors
    # --------------------------------------------------------------------------

    def _err_plot(
        self,
        cluster: Union[None, int],
        bpoints=True,
        hist_kwargs: Optional[Dict[str, Any]] = None,
        hist_fill_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """ Main implementation of :meth:`err_plot``

        Args:
            cluster: Name of cluster to be plotted or None if there are no
                clusters
            bpoints: Plot benchmark points? If False or benchmark points are
                not available, distributions correponding to random sample
                points are chosen.
            hist_kwargs: See :meth:`err_plot`
            hist_fill_kwargs: See :meth:`err_plot`

        Returns:
            None
        """
        if hist_kwargs is None:
            hist_kwargs = {}
        if hist_fill_kwargs is None:
            hist_fill_kwargs = {}

        if bpoints and self._has_bpoints:
            data, index = self._get_df_cluster(
                cluster, bpoint=True, bpoint_return_index=True
            )
        else:
            df_cluster_no_bp = self._get_df_cluster(cluster, bpoint=False)
            i = get_random_indizes(0, len(df_cluster_no_bp), 1)[0]
            data = df_cluster_no_bp.iloc[[i]]
            index = data.index[0]
            data = np.squeeze(data.values)

        err_high = self._get_df_cluster_err_high(index=index)
        err_low = self._get_df_cluster_err_low(index=index)

        if cluster is not None:
            color = self.color_scheme.get_cluster_color(cluster)
            light_color = self.color_scheme.get_err_color(cluster)
        else:
            color = self.color_scheme.get_cluster_color(0)
            light_color = self.color_scheme.get_err_color(0)

        hist_kw = dict(color=color, linestyle="-")
        hist_kw.update(hist_kwargs)

        plot_histogram(self.ax, self._bins, data, **hist_kw)

        hf_kw = dict(color=light_color)
        hf_kw.update(hist_fill_kwargs)

        plot_histogram_fill(
            self.ax, self._bins, data - err_low, data + err_high, **hf_kw
        )

    def err_plot(
        self,
        clusters: Optional[Union[None, int, Iterable[int]]] = None,
        ax=None,
        bpoints=True,
        hist_kwargs: Optional[Dict[str, Any]] = None,
        hist_fill_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """ Plot distributions with errors.

        Args:
            clusters:  List of clusters to selected or single cluster.
                If None (default), all clusters are chosen.
            ax: Instance of `matplotlib.axes.Axes` to plot on. If None, a new
                one is instantiated.
            bpoints: Plot benchmark points? If False or benchmark points are
                not available, distributions correponding to random sample
                points are chosen.
            hist_kwargs: Keyword arguments to
                :meth:`~clusterking.plots.plot_histogram.plot_histogram`
            hist_fill_kwargs: Keyword arguments to
                :meth:`~clusterking.plots.plot_histogram.plot_histogram_fill`

        Returns:
            None
        """
        clusters = self._interpret_cluster_input(clusters)
        _title = []
        if self._has_bpoints and bpoints:
            _title.append("Benchmark point")
        else:
            _title.append("Random sample point")
        if clusters:
            _title.append(
                "for cluster(s) {}".format(
                    ", ".join(map(str, sorted(clusters)))
                )
            )
        self._set_ax(ax, " ".join(_title))

        # pycharm might be confused about the type of `clusters`:
        # noinspection PyTypeChecker
        for cluster in clusters:
            self._err_plot(
                cluster,
                bpoints=bpoints,
                hist_kwargs=hist_kwargs,
                hist_fill_kwargs=hist_fill_kwargs,
            )
        if not clusters:
            self._err_plot(
                cluster=None,
                bpoints=False,
                hist_kwargs=hist_kwargs,
                hist_fill_kwargs=hist_fill_kwargs,
            )

        self._draw_legend(clusters)

    # --------------------------------------------------------------------------
    # Box plots
    # --------------------------------------------------------------------------

    def _box_plot(
        self,
        cluster,
        whiskers=1.5,
        bpoints=True,
        boxplot_kwargs=None,
        hist_kwargs=None,
    ) -> None:
        """ Main implementation of :meth:`box_plot`.
        Gets called for every cluster specified in :meth:`box_plot`.

        Args:
            cluster: Name of cluster to be plotted
            whiskers: Length of the whiskers of the box plot.
                See self.box_plot for more information.
                Default: 1.5 (matplotlib default)
            boxplot_kwargs: See :meth:`box_plot`
            hist_kwargs: See :meth:`box_plot`

        Returns:
            None
        """
        if boxplot_kwargs is None:
            boxplot_kwargs = {}
        if hist_kwargs is None:
            hist_kwargs = {}

        df_cluster = self._get_df_cluster(cluster)
        data = df_cluster.values

        if cluster is not None:
            color = self.color_scheme.get_cluster_color(cluster)
        else:
            color = self.color_scheme.get_cluster_color(0)

        bins = self._bins
        positions = 1 / 2 * (np.array(bins[1:]) + np.array(bins[:-1]))

        boxplot_options = dict(
            notch=False,
            positions=positions,  # np.array(range(len(data.T))) + 0.5,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor=color, color=color, alpha=0.3),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color),
            medianprops=dict(color=color),
            whis=whiskers,  # extend the range of the whiskers
        )
        boxplot_options.update(boxplot_kwargs)

        if StrictVersion(matplotlib.__version__) < StrictVersion("3.1"):
            boxplot_options["manage_xticks"] = False
        else:
            boxplot_options["manage_ticks"] = False

        self.ax.boxplot(data, **boxplot_options)
        if bpoints:
            self._plot_bundles(cluster, nlines=0, hist_kwargs=hist_kwargs)

    def box_plot(
        self,
        clusters: Optional[Union[int, Iterable[int]]] = None,
        ax=None,
        whiskers=2.5,
        bpoints=True,
        boxplot_kwargs: Optional[Dict[str, Any]] = None,
        hist_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """ Box plot of the bin contents of the distributions corresponding
        to selected clusters.

        Args:
            clusters:  List of clusters to selected or single cluster.
                If None (default), all clusters are chosen.
            ax: Instance of matplotlib.axes.Axes to plot on. If None, a new
                one is instantiated.
            whiskers: Length of the whiskers of the box plot in units of IQR
                (interquartile range, containing 50% of all values). Default
                2.5.
            bpoints: Draw benchmarks?
            boxplot_kwargs: Arguments to `matplotlib.pyplot.boxplot`
            hist_kwargs: Keyword arguments to
                :meth:`~clusterking.plots.plot_histogram.plot_histogram`
        """
        clusters = self._interpret_cluster_input(clusters)
        _title = ["Box plot of the bin contents"]
        if self._has_clusters:
            _title.append(
                "for cluster(s) {}".format(
                    ", ".join(map(str, sorted(clusters)))
                )
            )
        _title.append("\nWhisker length set to {}*IQR".format(whiskers))
        self._set_ax(ax, " ".join(_title))
        # pycharm might be confused about the type of `clusters`:
        # noinspection PyTypeChecker
        for cluster in clusters:
            self._box_plot(
                cluster,
                whiskers=whiskers,
                bpoints=bpoints,
                boxplot_kwargs=boxplot_kwargs,
            )
        if not clusters:
            self._box_plot(
                None,
                whiskers=whiskers,
                bpoints=bpoints,
                hist_kwargs=hist_kwargs,
            )

        self._draw_legend(clusters)

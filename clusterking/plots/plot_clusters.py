#!/usr/bin/env python3

# std
from math import ceil
import logging
from typing import List

# 3d party
import matplotlib.pyplot as plt
import matplotlib
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D  # NOTE BELOW (*)
import numpy as np
import pandas as pd

# ours
from clusterking.util.log import get_logger
from clusterking.plots.colors import ColorScheme

# (*) This import line is not explicitly used, but do not remove it!
# It is nescessary to load the 3d support!


# fixme: Maybe not take _setup_all?
# todo: also have the 3d equivalent of ClusterPlot.fill (using voxels)
# todo: option to disable legend
class ClusterPlot(object):
    """ Plot clusters in parameter space.

    After initialization, use the 'scatter' or 'fill' method for plotting.

    You can modify the attributes of this class to tweak some properties
    of the plots.

    Args:
        data: Data object
    """
    def __init__(self, data):
        #: logging.Logger object
        self.log = get_logger("ClusterPlot", sh_level=logging.WARNING)

        #: Instance of pandas.DataFrame
        self.data = data

        # (Advanced) config values
        # Documented in docstring of this class

        #: Color scheme
        self.color_scheme = ColorScheme()

        #: List of markers of the get_clusters (scatter plot only).
        self.markers = None
        if not self.markers:
            self.markers = ["o", "v", "^", "v", "<", ">"]

        #: Maximal number of subplots
        self.max_subplots = 16

        #: Maximal number of columns of the subplot grid
        self.max_cols = 4

        #: figure size of each subplot
        self.figsize = (4, 4)

        #: The name of the column that holds the cluster index
        self.cluster_column = "cluster"

        #: The name of the column that holds the benchmark yes/no information
        self.bpoint_column = "bpoint"

        #: Default marker size
        self.default_marker_size = \
            1/2 * matplotlib.rcParams['lines.markersize'] ** 2
        #: Marker size of benchmark points
        self.bpoint_marker_size = 6 * self.default_marker_size

        #: If true, a legend is drawn
        self.draw_legend = True

        # Internal values: Do not modify
        # ----------------------------------------------------------------------

        # Names of the columns to be on the axes of
        # the plot.
        self._axis_columns = None

        self._clusters = None

        self._df_dofs = None

        self._fig = None
        self._axs = None

    # ==========================================================================
    # User access via property
    # ==========================================================================

    @property
    def fig(self):
        """ The figure. """
        return self._fig

    # ==========================================================================
    # Quick access of simple logic
    # ==========================================================================

    @property
    def _axli(self):
        """Note: axs contains all axes (subplots) as a 2D grid, axsli contains
        the same objects but as a simple list (easier to iterate over)"""
        return self._axs.flatten()

    @property
    def _has_bpoints(self):
        """ True if we have benchmark points. """
        return self.bpoint_column in self.data.df.columns

    @property
    def _dofs(self):
        """ Column names of the degrees of freedom. """
        return list(self._df_dofs.columns)

    @property
    def _nsubplots(self):
        """ Number of subplots. """
        # +1 to have space for legend!
        return max(1, len(self._df_dofs)) + int(self.draw_legend)

    @property
    def _ncols(self):
        """ Number of columns of the subplot grid. """
        return min(self.max_cols, self._nsubplots)

    @property
    def _nrows(self):
        """ Number of rows of the subplot grid. """
        # the ``int`` technically does not make a difference, but pycharm
        # thinks that ``ceil`` returns floats and therefore complains
        # otherwise
        return int(ceil(self._nsubplots / self._ncols))

    # ==========================================================================
    # Helper functions
    # ==========================================================================

    def _find_dofs(self):
        """ Find all relevant wilson coefficients that are not axes on
        the plots (called _dofs). """

        dofs = []
        # 'Index columns' are by default the columns that hold the wilson
        # coefficients.
        for col in self.data.par_cols:
            if col not in self._axis_columns:
                if len(self.data.df[col].unique()) >= 2:
                    dofs.append(col)
        self.log.debug("dofs = {}".format(dofs))

        if not dofs:
            df_dofs = pd.DataFrame([])
        else:
            df_dofs = self.data.df[dofs].drop_duplicates().sort_values(dofs)
            df_dofs.reset_index(inplace=True, drop=True)

        self.log.debug("number of subplots = {}".format(len(df_dofs)))

        # Reduce the number of subplots by only sampling several points of
        # the Wilson coeffs that aren't on the axes

        if len(df_dofs) > self.max_subplots:
            steps_per_dof = int(self.max_subplots **
                                (1 / len(dofs)))
            self.log.debug("number of steps per dof", steps_per_dof)
            for col in dofs:
                allowed_values = df_dofs[col].unique()
                indizes = list(set(np.linspace(0, len(allowed_values)-1,
                                               steps_per_dof).astype(int)))
                allowed_values = allowed_values[indizes]
                df_dofs = df_dofs[df_dofs[col].isin(allowed_values)]
            self.log.debug("number of subplots left after "
                           "subsampling = {}".format(len(df_dofs)))

        self._df_dofs = df_dofs

    def _setup_subplots(self):
        """ Set up the subplot grid"""

        # squeeze keyword: https://stackoverflow.com/questions/44598708/
        # do not share axes, that makes problems if the grid is incomplete
        subplots_args = {
            "nrows": self._nrows,
            "ncols": self._ncols,
            "figsize": (self._ncols * self.figsize[0],
                        self._nrows * self.figsize[1]),
            "squeeze": False,
        }
        if len(self._axis_columns) == 3:
            subplots_args["subplot_kw"] = {'projection': '3d'}
        self._fig, self._axs = plt.subplots(**subplots_args)
        # this also sets self._axli

        ihidden = self._nrows * self._ncols - self._nsubplots + 1
        icol_hidden = self._ncols - ihidden
        self.log.debug("ihidden = {}".format(ihidden))
        self.log.debug("icol_hidden = {}".format(icol_hidden))

        if len(self._axis_columns) == 2:
            for isubplot in range(self._nrows * self._ncols):
                irow = isubplot//self._ncols
                icol = isubplot % self._ncols

                if isubplot >= self._nsubplots:
                    self.log.debug("hiding", irow, icol)
                    self._axli[isubplot].set_visible(False)

                if icol == 0:
                    self._axli[isubplot].set_ylabel(self._axis_columns[1])
                else:
                    self._axli[isubplot].set_yticklabels([])

                if irow == self._nrows - 2 and icol >= icol_hidden:
                    self._axli[isubplot].set_xlabel(self._axis_columns[0])
                elif irow == self._nrows - 1 and icol <= icol_hidden:
                    self._axli[isubplot].set_xlabel(self._axis_columns[0])
                else:
                    self._axli[isubplot].set_xticklabels([])

        else:
            for isubplot in range(self._nsubplots):
                self._axli[isubplot].set_xlabel(self._axis_columns[0])
                self._axli[isubplot].set_ylabel(self._axis_columns[1])
                self._axli[isubplot].set_zlabel(self._axis_columns[2])

        for isubplot in range(self._nsubplots - 1):
            title = " ".join(
                "{}={:.2f}".format(key, self._df_dofs.iloc[isubplot][key])
                for key in self._dofs
            )
            self._axli[isubplot].set_title(title)

        # set the xrange explicitly in order to not depend
        # on which get_clusters are shown etc.

        for isubplot in range(self._nsubplots):
            self._axli[isubplot].set_xlim(self._get_lims(0))
            self._axli[isubplot].set_ylim(self._get_lims(1))
            if len(self._axis_columns) == 3:
                self._axli[isubplot].set_zlim(self._get_lims(2))

    # todo: if scatter: use proper markers!
    def _add_legend(self):
        if not self.draw_legend:
            return
        legend_elements = []
        for cluster in self._clusters:
            color = self.color_scheme.get_cluster_color(cluster)
            # pycharm can't seem to find patches:
            # noinspection PyUnresolvedReferences
            p = matplotlib.patches.Patch(
                facecolor=color,
                edgecolor=color,
                label=cluster,
            )
            legend_elements.append(p)
        self._axli[self._nsubplots - 1].legend(
            handles=legend_elements,
            loc='lower left',
            title="Clusters",
            frameon=False
        )
        # todo: this shouldn't be necessary if setup axes worked as expected
        self._axli[self._nsubplots - 1].set_axis_off()

    def _get_lims(self, ax_no: int, stretch=0.1):
        """ Get lower and upper limit of axis (including padding)

        Args:
            ax_no: 0 for x-axis, 1 for y-axis etc.
            stretch: Fraction of total value span to add as padding.

        Returns:
            (minimum of plotrange, maximum of plotrange)
        """
        mi = min(self.data.df[self._axis_columns[ax_no]].values)
        ma = max(self.data.df[self._axis_columns[ax_no]].values)
        d = ma-mi
        pad = stretch * d
        return mi-pad, ma+pad

    def _setup_all(self, cols: List[str], clusters=None) -> None:
        """ Performs all setups.

        Args:
            cols: Names of the columns to be on the plot axes
            clusters: Clusters to plot

        Returns:
            None
            """
        assert(2 <= len(cols) <= 3)
        self._clusters = clusters
        self._axis_columns = cols
        if not self._clusters:
            self._clusters = list(self.data.df[self.cluster_column].unique())
        # Careful: Use the real number of clusters when initializing the
        # color scheme!
        self.color_scheme = ColorScheme(self.data.clusters(self.cluster_column))
        self._find_dofs()
        self._setup_subplots()

    def _set_fill_colors(self, matrix: np.ndarray) \
            -> np.ndarray:
        """ A helper function for the fill method. Given a n x m matrix of
        cluster numbers, this returns a n x m x 3 matrix, where the last 3
        dimensions contain the rgb value of the color that this cluster
        should be colored with.

        Args:
            matrix: m x n matrix containing cluster numbers

        Returns:
            n x m x 3 matrix with last 3 dimensions containing RGB codes
        """
        rows, cols = matrix.shape
        matrix_colored = np.zeros((rows, cols, 3))
        for irow in range(rows):
            for icol in range(cols):
                cluster = int(matrix[irow, icol])
                color = self.color_scheme.get_cluster_color(cluster)
                # pycharm doesn't find ``colors`` in matplotlib:
                # noinspection PyUnresolvedReferences
                rgb = matplotlib.colors.hex2color(
                    matplotlib.colors.cnames[color]
                )
                matrix_colored[irow, icol] = rgb
        return matrix_colored

    # ==========================================================================
    # Plotting methods
    # ==========================================================================

    # todo: **kwargs
    # todo: factor out the common part of scatter and fill into its own method?
    def scatter(self, cols: List[str], clusters=None):
        """ Create scatter plot, specifying the columns to be on the axes of the
        plot. If 3 column are specified, 3D scatter plots
        are presented, else 2D plots. If the dataframe contains more columns,
        such that each row is not only specified by the columns on the axes,
        a selection of subplots is created, showing 'cuts'.

        Args:
            cols: The names of the columns to be shown on the x, y (and z)
               axis of the plots.
            clusters: The get_clusters to be plotted (default: all)

        Returns:
            The figure (unless the 'inline' setting of matplotllib is
            detected).
        """
        self._setup_all(cols, clusters)

        for isubplot in range(self._nsubplots - int(self.draw_legend)):
            for cluster in self._clusters:
                df_cluster = \
                    self.data.df[self.data.df[self.cluster_column] == cluster]
                for col in self._dofs:
                    df_cluster = df_cluster[
                        df_cluster[col] == self._df_dofs.iloc[isubplot][col]
                    ]

                if self._has_bpoints:
                    df_cluster_no_bp = df_cluster[
                        df_cluster[self.bpoint_column] == False
                    ]
                    df_cluster_bp = df_cluster[
                        df_cluster[self.bpoint_column] == True
                    ]
                else:
                    df_cluster_no_bp = df_cluster
                    df_cluster_bp = pd.DataFrame()
                # df_cluster_non_bpoint = df_cluster[]]

                self._axli[isubplot].scatter(
                    *[df_cluster_no_bp[col] for col in self._axis_columns],
                    color=self.color_scheme.get_cluster_color(cluster),
                    marker=self.markers[cluster-1 % len(self.markers)],
                    label=cluster,
                    s=self.default_marker_size
                )
                if self._has_bpoints:
                    self._axli[isubplot].scatter(
                        *[df_cluster_bp[col] for col in self._axis_columns],
                        color=self.color_scheme.get_cluster_color(cluster),
                        marker=self.markers[cluster-1 % len(self.markers)],
                        label=cluster,
                        s=self.bpoint_marker_size
                    )

        self._add_legend()

        if 'inline' not in matplotlib.get_backend():
            return self._fig

    # todo: implement interpolation
    # todo: **kwargs
    def fill(self, cols: List[str]):
        """ Call this method with two column names, x and y. The results are
        similar to those of 2D scatter plots as created by the scatter
        method, except that the coloring is expanded to the whole xy plane.
        Note: This method only works with uniformly sampled NP!

        Args:
            cols: List of name of column to be plotted on x-axis and on y-axis

        Returns:
            The figure (unless the 'inline' setting of matplotllib is
            detected).
        """
        assert(len(cols) == 2)
        self._setup_all(cols)

        for isubplot in range(self._nsubplots - int(self.draw_legend)):
            df_subplot = self.data.df.copy()
            for col in self._dofs:
                df_subplot = df_subplot[
                    df_subplot[col] == self._df_dofs.iloc[isubplot][col]
                ]
            x = df_subplot[cols[0]].unique()
            y = df_subplot[cols[1]].unique()
            df_subplot.sort_values(by=[cols[1], cols[0]],
                                   ascending=[False, True],
                                   inplace=True)
            z = df_subplot[self.cluster_column].values
            # check if this makes sense
            z_matrix = z.reshape(y.shape[0], int(len(z) / y.shape[0]))
            self._axli[isubplot].imshow(
                self._set_fill_colors(z_matrix),
                interpolation='none',
                extent=[min(x), max(x), min(y), max(y)],
                aspect='auto'
            )

        self._add_legend()
        if 'inline' not in matplotlib.get_backend():
            return self._fig

    # ==========================================================================
    # Shortcuts for user
    # ==========================================================================

    def savefig(self, *args, **kwargs):
        """ Equivalent to ClusterPlot.fig.savefig(*args, **kwargs): Saves
        figure to file, e.g. ClusterPlot.savefig("test.pdf"). """
        self._fig.savefig(*args, **kwargs)

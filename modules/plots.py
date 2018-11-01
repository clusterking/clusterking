import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import matplotlib

# This import line is not explicitly used, but do not remove it!
# It is nescessary to be able to perform
from mpl_toolkits.mplot3d import Axes3D

# todo: move to a more elaborate plotting concept
# like https://scipy-cookbook.readthedocs.io/items/Matplotlib_UnfilledHistograms.html for unfilled histograms
# todo: more flexible signature?
def plot_histogram(ax: plt.axes,
                   binning: np.array,
                   contents: np.array,
                   normalized=False,
                   *args,
                   **kwargs) -> None:
    """ Plots histogram

    Args:
        ax: Axes of a plot
        binning: numpy array of bin edges
        contents: numpy array of bin contents
        normalized: If true, the plotted histogram will be normalized

    Returns:
        None
    """
    assert len(binning.shape) == len(contents.shape) == 1
    n = binning.shape[0] - 1  # number of bins
    assert n == contents.shape[0]
    assert n >= 1
    mpoints = (binning[1:] + binning[:-1]) / 2

    if normalized:
        values = contents / sum(contents)
    else:
        values = contents

    return ax.hist(mpoints,
                   bins=binning,
                   weights=values,
                   linewidth=0,
                   *args,
                   **kwargs
    )


# todo: warning if not enough colors for all clusters
# todo: also have the 3d equivalent of ClusterPlot.fill (using voxels)
class ClusterPlot(object):
    def __init__(self, df):
        # from arguments
        self.df = df

        # config values: Save to configure by user

        self.colors = None
        if not self.colors:
            self.colors = ["red", "green", "blue", "black", "orange", "pink", ]
        self.markers = None
        if not self.markers:
            self.markers = ["o", "v", "^", "v", "<", ">"]
        self.max_subplots = 16
        self.max_cols = 4
        self.figsize = (4, 4)
        self.debug = False

        # internal values: Do not modify

        self._cols = None
        self._clusters = None

        self._dofs = None
        self._relevant_dofs = None
        self._df_dofs = None

        self._nsubplots = None
        self._nrows = None
        self._ncols = None

        self._fig = None
        self._axs = None
        self._axli = None

    def _d(self, *args, **kwargs):
        """ For debugging this class """
        if self.debug:
            print(*args, **kwargs)

    def _find_dofs(self):
        """ find all relevant wilson coefficients that are not axes on
        the plots (called _dofs) """

        self._dofs = []
        self._relevant_dofs = []
        for col in ['l', 'r', 'sl', 'sr', 't']:
            if col not in self._cols:
                self._dofs.append(col)
                if len(self.df[col].unique()) >= 2:
                    self._relevant_dofs.append(col)
        self._d("_dofs = {}, relevant_dofs = {}".format(self._dofs, self._relevant_dofs))

        # find all unique value combinations of these columns
        self._df_dofs = self.df[self._dofs].drop_duplicates().sort_values(self._dofs)
        self._df_dofs.reset_index(inplace=True)
        self._d("number of subplots = {}".format(len(self._df_dofs)))

    def _sample_dofs(self):
        """Reduce the number of subplots by only sampling several points of
        the Wilson coeffs that aren't on the axes"""

        if len(self._df_dofs) > self.max_subplots:
            steps_per_dof = int(self.max_subplots ** (1 / len(self._relevant_dofs)))
            self._d("number of steps per dof", steps_per_dof)
            for col in self._relevant_dofs:
                allowed_values = self._df_dofs[col].unique()
                indizes = list(set(np.linspace(0, len(allowed_values)-1,
                                               steps_per_dof).astype(int)))
                allowed_values = allowed_values[indizes]
                self._df_dofs = self._df_dofs[self._df_dofs[col].isin(allowed_values)]
            self._d("number of subplots left after "
                   "subsampling = {}".format(len(self._df_dofs)))

        self.nsubplots = len(self._df_dofs)
        self.ncols = min(self.max_cols, len(self._df_dofs))
        self.nrows = ceil(len(self._df_dofs) / self.ncols)
        self._d("nrows = {}, ncols = {}".format(self.nrows, self.ncols))

    def _setup_subplots(self):
        """ Set up the subplot grid"""

        # squeeze keyword: https://stackoverflow.com/questions/44598708/
        # do not share axes, that makes problems if the grid is incomplete
        subplots_args = {
            "nrows": self.nrows,
            "ncols": self.ncols,
            "figsize": (self.ncols*self.figsize[0], self.nrows*self.figsize[1]),
            "squeeze": False,
        }
        if len(self._cols) == 3:
            subplots_args["subplot_kw"] = {'projection': '3d'}
        self.fig, self.axs = plt.subplots(**subplots_args)
        self.axli = self.axs.flatten()

        # note: axs contains all axes (subplots) as a 2D grid,
        #       axsli contains the same objects but as a
        #       simple list (easier to iterate over)

        ihidden = self.nrows*self.ncols - self.nsubplots
        icol_hidden = self.ncols - ihidden
        self._d("ihidden = {}".format(ihidden))
        self._d("icol_hidden = {}".format(icol_hidden))

        if len(self._cols) == 2:
            for isubplot in range(self.nrows * self.ncols):
                irow = isubplot//self.ncols
                icol = isubplot % self.ncols

                if isubplot >= self.nsubplots:
                    self._d("hiding", irow, icol)
                    self.axli[isubplot].set_visible(False)

                if icol == 0:
                    self.axli[isubplot].set_ylabel(self._cols[1])
                else:
                    self.axli[isubplot].set_yticklabels([])

                if irow == self.nrows - 2 and icol >= icol_hidden:
                    self.axli[isubplot].set_xlabel(self._cols[0])
                elif irow == self.nrows - 1 and icol <= icol_hidden:
                    self.axli[isubplot].set_xlabel(self._cols[0])
                else:
                    self.axli[isubplot].set_xticklabels([])

        else:
            for isubplot in range(self.nsubplots):
                self.axli[isubplot].set_xlabel(self._cols[0])
                self.axli[isubplot].set_ylabel(self._cols[1])
                self.axli[isubplot].set_zlabel(self._cols[2])

        for isubplot in range(self.nsubplots):
            title = " ".join("{}={:.2f}".format(key, self._df_dofs.iloc[isubplot][key])
                             for key in self._relevant_dofs)
            self.axli[isubplot].set_title(title)

        # set the xrange explicitly in order to not depend
        # on which clusters are shown etc.

        for isubplot in range(self.nsubplots):
            self.axli[isubplot].set_xlim(self._get_lims(0))
            self.axli[isubplot].set_ylim(self._get_lims(1))
            if len(self._cols) == 3:
                self.axli[isubplot].set_zlim(self._get_lims(2))

    def _get_lims(self, ax_no, stretch=0.1):
        """ Get lower and upper limit of axis (including padding) """
        mi = min(self.df[self._cols[ax_no]].values)
        ma = max(self.df[self._cols[ax_no]].values)
        d = ma-mi
        pad = stretch * d
        return mi-pad, ma+pad

    def _setup_all(self, cols, clusters=None):
        """ Performs all setups"""
        assert(2 <= len(cols) <= 3)
        self._clusters = clusters
        self._cols = cols
        if not self._clusters:
            self._clusters = list(self.df['cluster'].unique())

        self._find_dofs()
        self._sample_dofs()
        self._setup_subplots()

    def scatter(self, cols, clusters=None):
        """ Do a scatter plot """
        self._setup_all(cols, clusters)

        for isubplot in range(self.nsubplots):
            for cluster in self._clusters:
                df_cluster = self.df[self.df['cluster'] == cluster]
                for col in self._relevant_dofs:
                    df_cluster = df_cluster[df_cluster[col] ==
                                                 self._df_dofs.iloc[isubplot][col]]
                self.axli[isubplot].scatter(
                    *[df_cluster[col] for col in self._cols],
                    color=self.colors[cluster-1 % len(self.colors)],
                    marker=self.markers[cluster-1 % len(self.markers)],
                    label=cluster
                )
        if 'inline' not in matplotlib.get_backend():
            return self.fig

    def _set_fill_colors(self, matrix, color_offset=-1):
        rows, cols = matrix.shape
        matrix_colored = np.zeros((rows, cols, 3))
        # todo: this is slow
        for irow in range(rows):
            for icol in range(cols):
                value = int(matrix[irow, icol]) + color_offset
                color = self.colors[value % len(self.colors)]
                rgb = matplotlib.colors.hex2color(matplotlib.colors.cnames[color])
                matrix_colored[irow, icol] = rgb
        return matrix_colored


    def fill(self, cols):
        print("This method only works with uniformly sampled NP and has not "
              "been tested much either.")

        assert( len(cols) == 2)
        self._setup_all(cols)

        for isubplot in range(self.nsubplots):
            df_subplot = self.df.copy()
            for col in self._relevant_dofs:
                df_subplot = df_subplot[df_subplot[col] ==
                                        self._df_dofs.iloc[isubplot][col]]
            x = df_subplot[cols[0]].unique()
            y = df_subplot[cols[1]].unique()
            df_subplot.sort_values(by=[cols[1], cols[0]],
                                   ascending=[False, True],
                                   inplace=True)
            z = df_subplot['cluster'].values
            Z = z.reshape(y.shape[0], x.shape[0])
            self.axli[isubplot].imshow(
                self._set_fill_colors(Z, color_offset=-1),
                interpolation='none',
                extent=[min(x), max(x), min(y), max(y)]
            )

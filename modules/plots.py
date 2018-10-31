import numpy as np
from math import ceil
import matplotlib.pyplot as plt
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


def plot_clusters(df,
                     cols,
                     clusters=None,
                     colors=None,
                     markers=None,
                     max_levels=16,
                     debug=False,
                     **kwargs):
    """Creates 2D plots (slices) of the clusters.

    Args:
        df: Dataframe
        x_col: Name of wilson coeff to be plotted on x axis
        y_col: Name of wilson coeff to be plotted on y axis
        clusters: List of clusters to include
        colors: List of colors to use for the different clusters
        markers: List of colors to use for the different clusters
        max_levels: Maximal number of different plots/slices
        debug: debug enabled?
        kwargs: arguments for plt.scatter

    Returns:

    """
    def deb(*args, **kwargs):
        """ For debugging this function """
        if debug:
            print(*args, **kwargs)

    assert(2 <= len(cols) <= 3)

    # *** 1. find all relevant wilson coefficients that are not ***
    # ***    axes on the plots                                  ***

    _rem_cols = [col for col in ['l', 'r', 'sl', 'sr', 't']
                 if col not in cols]
    deb("remaining columns", _rem_cols)
    df_levels = df[_rem_cols].drop_duplicates().sort_values(_rem_cols)
    rem_cols = []
    for col in _rem_cols:
        if len(df_levels[col].unique()) >= 2:
            rem_cols.append(col)
    df_levels = df_levels[rem_cols]
    deb("remaining columns after cleaning", rem_cols)
    deb("number of levels", len(df_levels))

    # *** 2. reduce the number of subplots by only samplling **
    # ***    several points of the above Wilson coeffs       **

    if max_levels < len(df_levels):
        # still too many levels?
        steps_per_dof = int(max_levels ** (1/len(rem_cols)))
        deb("number of steps per dof", steps_per_dof)
        for col in rem_cols:
            allowed_values = df_levels[col].unique()
            indizes = list(set(np.linspace(0, len(allowed_values)-1,
                                  steps_per_dof).astype(int)))
            allowed_values = allowed_values[indizes]
            df_levels = df_levels[df_levels[col].isin(allowed_values)]
    deb("number of levels left after cleaning", len(df_levels))

    # *** 3. Set up subplots ***

    max_cols = 4
    ncols = min(max_cols, len(df_levels))
    nrows = ceil(len(df_levels)/ncols)

    deb("nrows", nrows, "ncols", ncols)

    subplots_args = {
        "nrows":nrows,
        "ncols": ncols,
        "sharex": "col",
        "sharey": "row",
        "figsize": (ncols*4, nrows*4),
        "squeeze": False
    }
    if len(cols) == 3:
        subplots_args["subplot_kw"] = {'projection': '3d'}
    fig, axs = plt.subplots(**subplots_args)
    # squeeze keyword: https://stackoverflow.com/questions/44598708/

    if len(cols) == 2:
        for irow in range(nrows):
            axs[irow, 0].set_ylabel(cols[1])
        for icol in range(ncols):
            axs[nrows-1, icol].set_xlabel(cols[0])
    else:
        for irow in range(nrows):
            for icol in range(ncols):
                axs[irow, icol].set_xlabel(cols[0])
                axs[irow, icol].set_ylabel(cols[1])
                axs[irow, icol].set_zlabel(cols[2])

    # *** 4. MISC preparations ***

    if not colors:
        colors = ["red", "green", "blue", "pink"]
    if not markers:
        markers = ["o", "v", "^"]
    if not clusters:
        # plot all
        clusters = list(df['cluster'].unique())

    # *** 5. Plot ***

    ilevel = 0
    for _, level_rows in df_levels.iterrows():
        irow = ilevel//ncols
        icol = ceil(ilevel % ncols)
        title = " ".join("{}={:.2f}".format(key, level_rows[key]) for key in rem_cols)
        axs[irow, icol].set_title(title)
        for icluster, cluster in enumerate(clusters):
            df_cluster = df[df['cluster'] == cluster]
            for col in rem_cols:
                df_cluster = df_cluster[df_cluster[col] == level_rows[col]]
            axs[irow, icol].scatter(
                *[df_cluster[col] for col in cols],
                color=colors[icluster % len(colors)],
                marker=markers[icluster % len(markers)],
                label=cluster,
                **kwargs
            )
        ilevel += 1
    # ax.legend(bbox_to_anchor=(1.2, 1.05))
    # return fig

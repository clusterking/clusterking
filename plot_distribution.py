#!/usr/bin/env python3

from modules.distribution import bin_function, dGq2, q2min, q2max
from modules.util.log import get_logger
import numpy as np
from modules.inputs import Wilson
# from typing import List

import matplotlib.pyplot as plt

log = get_logger("Plot")


# todo: move to a more elaborate plotting concept like https://scipy-cookbook.readthedocs.io/items/Matplotlib_UnfilledHistograms.html for unfilled histograms
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

    return ax.hist(mpoints, bins=binning, weights=values, linewidth=0, *args, **kwargs)


if __name__ == "__main__":
    w = Wilson(0, 0, 0, 0, 0)

    log.info("Calculating integrals")
    bins = np.linspace(q2min, q2max, 4)
    more_bins = np.linspace(q2min, q2max, 8)
    even_more_bins = np.linspace(q2min, q2max, 20)

    values = bin_function(lambda x: dGq2(w, x), bins)
    more_values = bin_function(lambda x: dGq2(w, x), more_bins)
    even_more_values = bin_function(lambda x: dGq2(w, x), even_more_bins)
    log.info("Calculation done")

    fig, axs = plt.subplots(2, 3)

    plot_histogram(axs[0][0], bins, np.array(values), color="red", normalized=True)
    plot_histogram(axs[0][1], more_bins, np.array(more_values), color="black", normalized=True)
    plot_histogram(axs[0][2], even_more_bins, np.array(even_more_values), color="green", normalized=True)
    plot_histogram(axs[1][0], bins, np.array(values), color="red", normalized=False)
    plot_histogram(axs[1][1], more_bins, np.array(more_values), color="black", normalized=False)
    plot_histogram(axs[1][2], even_more_bins, np.array(even_more_values), color="green", normalized=False)

    fig.show()

    # wait till we press a button
    input("Press any key to end program.")

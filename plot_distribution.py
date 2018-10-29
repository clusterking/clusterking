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
    print("Calc")
    bins = np.linspace(q2min, q2max, 4)

    # todo: use np.array rather than list!
    values_single = bin_function(lambda x: dGq2(w, x), list(bins))
    print("Calc done")

    fig, axs = plt.subplots(1, 2)
    plot_histogram(axs[0], bins, np.array(values_single), color="red")
    plot_histogram(axs[1], bins, np.array(values_single), normalized=True)
    # plot_histogram(axs[1], np.array([1,2,3]), np.array([1, 3]), normalized=True)
    fig.show()
    input()


#!/usr/bin/env python3

# std
from math import sqrt
from typing import Optional, Dict

# 3rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_histogram(ax, edges, contents, normalize=False, **kwargs):
    """
    Plot a histogram.

    Args:
        ax: Instance of `matplotlib.axes.Axes` to plot on. If ``None``, a new
            figure will be initialized.
        edges: Edges of the bins or None (to use bin numbers on the x axis)
        contents: bin contents
        normalize (bool): Normalize histogram. Default False.
        **kwargs: passed on to `matplotlib.pyplot.step`

    Returns:
        Instance of `matplotlib.axes.Axes`
    """

    if not ax:
        fig, ax = plt.subplots()

    # "flatten" contents
    _contents = contents
    contents = np.array(contents)
    contents = contents.squeeze()
    if not len(contents.shape) == 1:
        raise ValueError(
            "The supplied contents array of shape {} can't be squeezed"
            "into a one dimensional array.".format(_contents.shape)
        )

    # bin numbers for the x axis if no x values are supplied
    if edges:
        edges = np.array(edges)
        assert len(edges.shape) == 1
        if not len(edges) == (len(contents) + 1):
            raise ValueError(
                "Invalid number of bin edges ({}) supplied for "
                "{} bins.".format(len(edges), len(contents))
            )
    else:
        edges = np.arange(len(contents) + 1)
        # force to have only integers on the x axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Normalize contents?
    if normalize:
        bin_widths = edges[:-1] - edges[1:]
        norm = sum(contents * bin_widths)
        contents /= norm

    # We need to repeat the last entry of the contents
    # because we use the 'post' plotting method for matplotlib.pyplot.step
    contents = np.append(contents, contents[-1])

    step_kwargs = dict(where="post")
    step_kwargs.update(kwargs)
    ax.step(edges, contents, **step_kwargs)

    return ax


def plot_histogram_fill(ax, edges, content_low, content_high, **kwargs):
    """
    Plot two histograms with area shaded in between.

    Args:
        ax: Instance of `matplotlib.axes.Axes` to plot on. If ``None``, a new
            figure will be initialized.
        edges: Edges of the bins or None (to use bin numbers on the x axis)
        content_low: Bin contents of lower histogram
        content_high: Bin contents of higher histogram
        **kwargs: Keyword argumenets to `matplotlib.pyplot.fill_between`

    Returns:
        Instance of `matplotlib.axes.Axes`
    """

    if not ax:
        fig, ax = plt.subplots()

    content_low = np.squeeze(np.array(content_low))
    content_high = np.squeeze(np.array(content_high))

    if not len(content_high) == len(content_low) == len(edges) - 1:
        raise ValueError(
            "Lenghts don't match: content_high: {}, content_low: {}, "
            "edges -1: {}".format(
                len(content_high), len(content_low), len(edges) - 1
            )
        )

    content_low = np.concatenate((content_low, np.atleast_1d(content_low[-1])))
    content_high = np.concatenate(
        (content_high, np.atleast_1d(content_high[-1]))
    )

    fb_kwargs = dict(step="post", linewidth=0)
    fb_kwargs.update(kwargs)

    ax.fill_between(edges, content_low, content_high, **fb_kwargs)

    return ax


def plot_hist_with_mean(
    series,
    ax=None,
    hist_kwargs: Optional[Dict] = None,
    draw_line=True,
    line_kwargs: Optional[Dict] = None,
):
    """ Plot histogram together with a line that designates the mean.

    Args:
        series: Anything that can be converted to `pandas.Series`
        ax: Instance of `matplotlib.axes.Axes` to plot on. If ``None``, a new
            figure will be initialized.
        hist_kwargs: Keyword arguments to `pandas.Series.hist`
        draw_line: Draw line for mean
        line_kwargs:

    Returns:
        Instance of `matplotlib.axes.Axes`

    Example:

    .. code-block:: python

        import numpy as np
        from clusterking.plots.plot_histogram import plot_hist_with_mean
        plot_hist_with_mean(np.random.normal(size=100))
    """

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    if ax is None:
        _, ax = plt.subplots()

    if hist_kwargs is None:
        hist_kwargs = {}
    all_hist_kwargs = dict(bins=int(sqrt(len(series))))
    all_hist_kwargs.update(hist_kwargs)

    series.hist(ax=ax, **all_hist_kwargs)

    if draw_line:
        if line_kwargs is None:
            line_kwargs = {}
        all_line_kwargs = dict(linestyle="dashed", color="k", linewidth=1)
        all_line_kwargs.update(line_kwargs)
        ax.axvline(series.mean(), **all_line_kwargs)
        min_ylim, max_ylim = ax.get_ylim()
        ax.text(
            series.mean(), max_ylim * 0.9, "Mean: {:.2f}".format(series.mean())
        )

    ax.set_title(series.name)
    return ax

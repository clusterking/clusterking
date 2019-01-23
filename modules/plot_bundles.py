#!/usr/bin/env python3

# std
import matplotlib.pyplot as plt
import random
from typing import List

# 3rd party
import numpy as np
import pandas as pd

# ours
from modules.util.log import get_logger


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
        indizes.add(random.randint(start, stop))
        if iterations >= 10 * n:
            print(
                "Did not manage to generate enough different random "
                "integers (only {} of {}).".format(len(indizes), n)
            )
            break
    return sorted(list(indizes))


class PlotBundles(object):
    def __init__(self, df: pd.DataFrame, bin_column_prefix="bin", cluster_column="cluster"):
        self.log = get_logger("PlotBundles")

        self.df = df

        # The names of the columns that hold the bin contents
        # Can be redefined by the user afterwards.
        self.bin_columns = [
            col for col in self.df.columns if col.startswith(bin_column_prefix)
        ]
        if not self.bin_columns:
            self.log.warning("Did not bin columns. Please set them manually.")

        self.cluster_column = cluster_column
        self.clusters = list(self.df[self.cluster_column].unique())

        self.cluster_colors = ["red", "green", "blue", "black", "orange",
                               "pink"]
        if len(self.cluster_colors) < len(self.clusters):
            self.log.warning(
                "Warning: Not enough colors for all clusters. Some of the "
                "colors will be identical."
            )

        self.fig = None
        self.ax = None

    def get_cluster_color(self, cluster):
        return self.cluster_colors[cluster % len(self.cluster_colors)]

    def get_df_cluster(self, cluster):
        return self.df[self.df[self.cluster_column] == cluster][self.bin_columns]

    def _plot_bundles(self, ax, cluster, nlines=3):

        linestyles = ['-', '--', '-.', ':']

        df_cluster = self.get_df_cluster(cluster)
        if len(df_cluster) < nlines:
            self.log.warning(
                "Not enough rows in dataframe. "
                "Only plotting {} lines.".format(len(df_cluster))
            )
            nlines = len(df_cluster)

        indizes = get_random_indizes(0, len(df_cluster), nlines)

        bin_numbers = np.array(range(1, len(self.bin_columns) + 1))

        color = self.get_cluster_color(cluster)

        # todo: use post
        for i, index in enumerate(indizes):
            data = df_cluster.iloc[[index]].values.reshape(len(self.bin_columns))
            ax.step(
                bin_numbers,
                data,
                where="mid",
                color=color,
                linestyle=linestyles[i % len(linestyles)]
            )

    def plot_bundles(self, clusters=None, nlines=1, ax=None):
        if not clusters:
            clusters = self.clusters
        if isinstance(clusters, int):
            clusters = [clusters]
        if not ax:
            fig, ax = plt.subplots()
            ax.set_title(
                "{} example(s) of distributions for cluster(s) {}".format(
                    nlines, ", ".join(map(str, sorted(clusters)))
                )
            )
            self.fig = fig
            self.ax = ax
        for cluster in clusters:
            self._plot_bundles(ax, cluster, nlines=nlines)

    def _plot_minmax(self, ax, cluster):
        df_cluster = self.get_df_cluster(cluster)
        maxima = list(df_cluster.max().values)
        minima = list(df_cluster.min().values)

        bin_numbers = np.array(range(1, len(self.bin_columns) + 2))

        color = self.get_cluster_color(cluster)
        for i in range(len(maxima)):
            x = bin_numbers[i:i+2]
            y1 = [minima[i], minima[i]]
            y2 = [maxima[i], maxima[i]]
            ax.fill_between(
                x,
                y1,
                y2,
                facecolor=color,
                interpolate=False,
                alpha=0.3,
                hatch="////",
                color=color
            )

    def plot_minmax(self, clusters=None, ax=None):
        if not clusters:
            clusters = self.clusters
        if isinstance(clusters, int):
            clusters = [clusters]
        if not ax:
            fig, ax = plt.subplots()
            ax.set_title(
                "Minima and maxima of the bin contents for "
                "cluster(s) {}".format(', '.join(map(str, sorted(clusters))))
            )
            self.fig = fig
            self.ax = ax

        for cluster in clusters:
            self._plot_minmax(ax, cluster)

    def _box_plot(self, ax, cluster, whiskers=1.5):
        df_cluster = self.get_df_cluster(cluster)
        data = df_cluster.values

        c = self.get_cluster_color(cluster)

        ax.boxplot(
            data,
            notch=False,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor=c, color=c, alpha=0.3),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c),
            #facecolor=None,
            whis=whiskers  # extend the range of the whiskers
        )

    def box_plot(self, clusters=None, ax=None):
        if not clusters:
            clusters = self.clusters
        if isinstance(clusters, int):
            clusters = [clusters]
        whiskers = 2.5
        if not ax:
            fig, ax = plt.subplots()
            ax.set_title(
                "Box plot of the bin contents for cluster(s) {}\n"
                "Whisker length set to {}*IQR".format(
                    ", ".join(map(str, sorted(clusters))),
                    whiskers
                )
            )
        for cluster in clusters:
            self._box_plot(ax, cluster, whiskers=whiskers)


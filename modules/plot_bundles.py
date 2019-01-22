#!/usr/bin/env python3

# std
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

# ours
from modules.util.log import get_logger


def get_random_indizes(maximum: int, number: int):
    indizes = set()
    iterations = 0
    while len(indizes) < number:
        indizes.add(random.randint(0, maximum))
        if iterations >= 10 * number:
            print(
                "Did not manage to generate enough different random "
                "integers (only {} of {}).".format(len(indizes), number)
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

        self.cluster_colors = ["red", "green", "blue", "black", "orange", "pink", ]
        if len(self.cluster_colors) < len(self.clusters):
            print("Warning: Not enough colors for all clusters. Some of the "
                  "colors will be identical.")

    def get_cluster_color(self, cluster):
        return self.cluster_colors[cluster % len(self.cluster_colors)]

    def get_clusters(self, clusters=None):
        if clusters:
            return clusters
        return self.clusters

    def _plot_bundles(self, cluster, nlines=3, ax=None):

        linestyles = ['-', '--', '-.', ':']

        df_cluster = self.df[self.df[self.cluster_column] == cluster]
        if len(df_cluster) < nlines:
            self.log.warning(
                "Not enough rows in dataframe. "
                "Only plotting {} lines.".format(len(df_cluster))
            )
            nlines = len(df_cluster)

        indizes = get_random_indizes(len(df_cluster), nlines)

        if not ax:
            fig, ax = plt.subplots()
            ax.set_title(
                "{} examples of distributions for cluster {}".format(
                    nlines, cluster
                )
            )

        bin_numbers = np.array(range(1, len(self.bin_columns) + 1))

        color = self.get_cluster_color(cluster)

        for i, index in enumerate(indizes):
            data = df_cluster.iloc[[index]][self.bin_columns].values.reshape(len(self.bin_columns))
            ax.step(bin_numbers, data, where="mid", color=color, linestyle=linestyles[i % len(linestyles)])

    def plot_bundles(self, clusters=None, nlines=1):
        if not clusters:
            clusters = self.get_clusters()
        if isinstance(clusters, int):
            clusters = [clusters]
        fig, ax = plt.subplots()
        for cluster in clusters:
            self._plot_bundles(cluster, nlines=nlines, ax=ax)

    def _plot_minmax(self, cluster, ax=None):
        df_cluster = self.df[self.df[self.cluster_column] == cluster][self.bin_columns]
        maxima = list(df_cluster.max().values)
        minima = list(df_cluster.min().values)

        if not ax:
            fig, ax = plt.subplots()
            ax.set_title("Minima and maxima of the bin contents for "
              "cluster {}".format(cluster))

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

    def plot_minmax(self, clusters=None):
        if not clusters:
            self.get_clusters()
        if isinstance(clusters, int):
            clusters = [clusters]
        fig, ax = plt.subplots()
        clusters = self.get_clusters(clusters)
        for cluster in clusters:
            self._plot_minmax(cluster, ax=ax)


    def _box_plot(self, cluster, ax=None):
        df_cluster = self.df[self.df[self.cluster_column] == cluster][self.bin_columns]
        data = df_cluster.values

        whiskers=2.5

        if not ax:
            fig, ax = plt.subplots()
            ax.set_title("Box plot of the bin contents for cluster 5\n"
                     "Whisker length set to {}*IQR".format(whiskers))
        c=self.get_cluster_color(cluster)

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

    def box_plot(self, clusters=None):
        if not clusters:
            self.get_clusters()
        if isinstance(clusters, int):
            clusters = [clusters]
        clusters = self.get_clusters(clusters)
        fig, ax = plt.subplots()
        for cluster in clusters:
            self._box_plot(cluster, ax=ax)


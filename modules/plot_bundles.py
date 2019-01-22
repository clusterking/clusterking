#!/usr/bin/env python3

# std
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

# ours
from modules.util.log import get_logger


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

        # self.clusters = list(self.df[self.cluster_column].unique())

        self.cluster_colors = ["red", "green", "blue", "black", "orange", "pink", ]

    def plot_bundles(self, cluster, nlines=4, seed=None):

        df_cluster = self.df[self.df[self.cluster_column] == cluster]

        if seed:
            random.seed(seed)

        if len(df_cluster) < nlines:
            self.log.warning(
                "Not enough rows in dataframe. "
                "Only plotting {} lines.".format(len(df_cluster))
            )
            nlines = len(df_cluster)

        indizes = set()
        iterations = 0
        while len(indizes) < nlines:
            indizes.add(random.randint(0, len(df_cluster)))
            if iterations >= 10 * nlines:
                self.log.warning(
                    "Did not manage to generate enough different random "
                    "integers (only {} of {}).".format(len(indizes), nlines))
                break

        fig, ax = plt.subplots()
        bin_numbers = np.array(range(1, len(self.bin_columns) + 1))

        for index in indizes:
            data = df_cluster.iloc[[index]][self.bin_columns].values.reshape(len(self.bin_columns))
            ax.step(bin_numbers, data, where="mid")

        ax.set_title(
            "{} examples of distributions for cluster {}".format(
                nlines, cluster
            )
        )

    def plot_min_max(self, cluster):
        df_cluster = self.df[self.df[self.cluster_column] == cluster][self.bin_columns]
        maxima = list(df_cluster.max().values)
        minima = list(df_cluster.min().values)


        # return

        fig, ax = plt.subplots()
        bin_numbers = np.array(range(1, len(self.bin_columns) + 2))

        for i in range(len(maxima)):
            x = bin_numbers[i:i+2]
            y1 = [minima[i], minima[i]]
            y2 = [maxima[i], maxima[i]]
            ax.fill_between(x, y1, y2, facecolor="red", interpolate=False, alpha=0.3, hatch="////")

        # minima_tweaked = minima[:]
        # minima_tweaked.append(minima[-1])
        # maxima_tweaked = maxima[:]
        # maxima_tweaked.append(maxima[-1])

        ax.set_title("Minima and maxima of the bin contents for "
                     "cluster {}".format(cluster))

        # ax.step(bin_numbers, minima_tweaked, where="post", color="black")
        # ax.step(bin_numbers, maxima_tweaked, where="post", color="black")

    def box_plot(self, cluster):
        df_cluster = self.df[self.df[self.cluster_column] == cluster][self.bin_columns]
        data = df_cluster.values

        whiskers=2.5

        fig, ax = plt.subplots()
        ax.boxplot(
            data,
            notch=False,
            vert=True,
            patch_artist=False,
            whis=whiskers  # extend the range of the whiskers
        )

        ax.set_title("Box plot of the bin contents for cluster 5\n"
                     "Whisker length set to {}*IQR".format(whiskers))

    def plot_bundles_overlaid(self, clusters=None, seed=None):

        if seed:
            random.seed(seed)

        if not clusters:
            clusters = list(self.df[self.cluster_column].unique())

        bin_numbers = np.array(range(1, len(self.bin_columns) + 1))

        fig, ax = plt.subplots()
        for cluster in clusters:
            df_cluster = self.df[self.df[self.cluster_column] == cluster][self.bin_columns]
            index = random.randint(0, len(df_cluster))
            contents = df_cluster.iloc[[index]].values.reshape(len(self.bin_columns))
            # print(contents)
            ax.step(bin_numbers, contents, where="mid", label="cluster {}".format(cluster))
        ax.legend(frameon=False)

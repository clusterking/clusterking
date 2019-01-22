#!/usr/bin/env python3

from modules.util.log import get_logger

import matplotlib.pyplot as plt

from .plots import plot_histogram

import pandas as pd

import random

import numpy as np

class PlotBundles(object):
    def __init__(self, df: pd.DataFrame, bin_column_prefix="bin"):
        self.log = get_logger("PlotBundles")

        self.df = df

        # The names of the columns that hold the bin contents
        # Can be redefined by the user afterwards.
        self.bin_columns = [
            col for col in self.df.columns if col.startswith(bin_column_prefix)
        ]
        if not self.bin_columns:
            self.log.warning("Did not bin columns. Please set them manually.")

        self.cluster_column = "cluster"

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

    # def plot_min_mx(self, cluster):
    #     df_cluster = self.df[self.df[self.cluster_column] == cluster]

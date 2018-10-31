#!/usr/bin/env python3

"""Read the results from distance_matrix.py and cluster them.
"""

# standard
import argparse
import json
import os.path
import sys

# 3rd party
import pandas as pd
import matplotlib.pyplot as plt

# us
from modules.util.log import get_logger
from modules.util.cli import yn_prompt
from scan import Scanner
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


class Cluster(object):
    def __init__(self, input_path):
        self.log = get_logger("Cluster")

        self.log.info("Input file basename: '{}'.".format(input_path))
        self.input_path = input_path

        self.scan_metadata = None
        self.scan_df = None
        self.scan_index2wilson = None

        self._get_scan_data()
        self._get_scan_metadata()
        self._get_scan_index2wilson()

        self.hierarchy = None

    def _get_scan_data(self):
        path = Scanner.data_output_path(self.input_path)
        self.log.debug("Loading scanner data from '{}'.".format(path))
        with open(path, 'r') as data_file:
            self.scan_df = pd.read_csv(data_file)
        self.log.debug("Done.")

    def _get_scan_metadata(self):
        path = Scanner.config_output_path(self.input_path)
        self.log.debug("Loading scanner metadata from '{}'.".format(path))
        with open(path, 'r') as metadata_file:
            self.scan_metadata = json.load(metadata_file)
        self.log.debug("Done.")

    # todo: probably we don't need that separately actually
    def _get_scan_index2wilson(self):
        path = Scanner.config_output_path(self.input_path)
        self.log.debug("Loading index2wilson from '{}'.".format(path))
        with open(path, 'r') as index2wilson_file:
            self.scan_index2wilson = json.load(index2wilson_file)
        self.log.debug("Done.")

    def build_hierarchy(self, metric="euclidean", method="complete"):
        # values =
        # matrix =
        nbins = self.scan_metadata["scan"]["q2points"]["nbins"]
        # only the q2 bins without any other information in the dataframe
        data = self.scan_df[["bin{}".format(i) for i in range(nbins)]]
        self.hierarchy = linkage(data, metric=metric, method=method)

    def dendogram(self, output_path=None):
        if self.hierarchy is None:
            self.log.error("Hierarchy not yet set up. Returning without "
                           "doing anything.")
            return
        labelsize=20
        ticksize=15
        plt.title('Hierarchical Clustering Dendrogram', fontsize=labelsize)
        plt.xlabel('ID', fontsize=labelsize)
        plt.ylabel('distance', fontsize=labelsize)
        den = dendrogram(
            self.hierarchy , color_threshold = 0.02,
            leaf_rotation=90., # rotates the x axis labels
            leaf_font_size=8, # font size for the x axis labels
        )
        if not output_path:
            plt.show()



def cli():
    """Command line interface to run the integration jobs from the command
    line with additional options.

    Simply run this script with '--help' to see all options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="Input file/distance matrix.",
                        default="output/scan/global_results",
                        dest="input_path")
    parser.add_argument("-o", "--output",
                        help="Output file.",
                        default="output/cluster/global_results.out",
                        dest="output_path")
    args = parser.parse_args()

    c = Cluster(args.input_path)

    # todo: this doesn't make sense yet
    if os.path.exists(args.output_path):
        agree = yn_prompt("Output path '{}' already exists and will be "
                          "overwritten. Proceed?".format(args.output_path))
        if not agree:
            c.log.critical("User abort.")
            sys.exit(1)

    c.log.info("Output file: '{}'".format(args.output_path))

    print(c)

    c.build_hierarchy()
    c.dendogram()
    input()

if __name__ == "__main__":
    # Run command line interface
    cli()
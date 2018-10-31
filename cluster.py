#!/usr/bin/env python3

"""Read the results from distance_matrix.py and cluster them.
"""

# standard
import argparse
import json
import os.path
import sys
from typing import Union

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

        self._get_scan_data()
        self._get_scan_metadata()

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

    def build_hierarchy(self, metric="euclidean", method="complete"):
        nbins = self.scan_metadata["scan"]["q2points"]["nbins"]
        # only the q2 bins without any other information in the dataframe
        data = self.scan_df[["bin{}".format(i) for i in range(nbins)]]
        self.hierarchy = linkage(data, metric=metric, method=method)

    def dendogram(
            self,
            output: Union[None, str]=None,
            ax=None,
            show=False,
            **kwargs
    ) -> plt.Axes:
        """Creates dendogram

        Args:
            output: If supplied, we save the dendogram there
            ax: An axes object if you want to add the dendogram to an existing
                axes rather than creating a new one
            show: If true, the dendogram is shown in a viewer.

        Returns:
            The matplotlib.pyplot.Axes object
        """
        if self.hierarchy is None:
            self.log.error("Hierarchy not yet set up. Returning without "
                           "doing anything.")
            return

        # do we add to a plot or generate a whole new figure?
        if ax:
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots()

        labelsize = 20
        ticksize = 15
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=labelsize)
        ax.set_xlabel('ID', fontsize=labelsize)
        ax.set_ylabel('Distance', fontsize=labelsize)

        # set defaults for dendogram plotting options here
        # (this way we can overwrite them with additional arguments)
        den_config = {
            "color_threshold": "default",
            "leaf_rotation": 90.,  # rotates the x axis labels
            "leaf_font_size": 8,   # font size for the x axis labels
        }
        den_config.update(kwargs)

        den = dendrogram(
            self.hierarchy,
            ax=ax,
            **den_config
        )

        if show:
            # Note: we do not block here, so make sure that you're including
            # plt.show() or an input statement somewhere in this script so
            # that it doesn't just exit.
            fig.show()
        if output:
            assert(isinstance(output, str))
            dirname = os.path.dirname(output)
            if dirname and not os.path.exists(dirname):
                self.log.debug("Creating dir '{}'.".format(dirname))
                os.makedirs(dirname)
            fig.savefig(output, bbox_inches="tight")
            self.log.info("Wrote dendogram to '{}'.".format(output))

        return ax


def cli():
    """Command line interface to run the integration jobs from the command
    line with additional options.

    Simply run this script with '--help' to see all options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="Input file basename",
                        default="output/scan/global_results",
                        dest="input_path")
    parser.add_argument("-o", "--output",
                        help="Output file basename",
                        default="output/cluster/global_results",
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

    c.build_hierarchy()
    c.dendogram(show=True, output="test.pdf")
    plt.show()

if __name__ == "__main__":
    # Run command line interface
    cli()
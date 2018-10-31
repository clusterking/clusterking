#!/usr/bin/env python3

"""Read the results from distance_matrix.py and cluster them.
"""

# standard
import argparse
import json
import os.path
import sys
import time
from typing import Union

# 3rd party
import matplotlib.pyplot as plt
import pandas as pd

# us
from modules.util.log import get_logger
from modules.util.cli import yn_prompt
from modules.util.misc import nested_dict, git_info
from scan import Scanner
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


class Cluster(object):
    def __init__(self, input_path):
        self.log = get_logger("Cluster")

        self.log.info("Input file basename: '{}'.".format(input_path))
        self.input_path = input_path

        # will load the one metadata from scan and then add our own
        self.metadata = nested_dict()
        self._get_scan_metadata()
        self.metadata["cluster"]["git"] = git_info(self.log)
        self.metadata["cluster"]["time"] = time.strftime("%a %d %b %Y %H:%M",
                                                         time.gmtime())

        # dataframe from scanner
        self.df = None
        self._get_scan_data()

        self.hierarchy = None

    def _get_scan_data(self):
        path = Scanner.data_output_path(self.input_path)
        self.log.debug("Loading scanner data from '{}'.".format(path))
        with open(path, 'r') as data_file:
            self.df = pd.read_csv(data_file)
        self.log.debug("Done.")

    def _get_scan_metadata(self):
        path = Scanner.config_output_path(self.input_path)
        self.log.debug("Loading scanner metadata from '{}'.".format(path))
        with open(path, 'r') as metadata_file:
            scan_metadata = json.load(metadata_file)
        self.metadata.update(scan_metadata)
        self.log.debug("Done.")

    # todo: switch to more flexible keyword approach as below
    def build_hierarchy(self, **kwargs):
        self.log.debug("Building hierarchy.")
        nbins = self.metadata["scan"]["q2points"]["nbins"]
        # only the q2 bins without any other information in the dataframe
        data = self.df[["bin{}".format(i) for i in range(nbins)]]

        # set defaults for linkage options here
        # (this way we can overwrite them with additional arguments)
        linkage_config = {
            "metric": "euclidean",
            "method": "complete"
        }
        linkage_config.update(kwargs)

        self.hierarchy = linkage(data, **linkage_config)
        config = self.metadata["cluster"]["hierarchy"]
        config["metric"] = linkage_config["metric"]
        config["method"] = linkage_config["method"]
        self.log.debug("Done")

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
            **kwargs: Additional keyword options to
                scipy.cluster.hierarchy.dendogram

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

    def cluster(self, max_d=0.02, **kwargs):
        """Performs the actual clustering
        Args:
            max_d:
            **kwargs:

        Returns:
            None
        """
        self.log.debug("Performing clustering.")
        if self.hierarchy is None:
            self.log.error("Hierarchy not yet set up. Returning without "
                           "doing anything")
            return

        # set up defaults for clustering here
        # (this way we can overwrite them with additional arguments)
        fcluster_config = {
            "criterion": "distance"
        }
        fcluster_config.update(kwargs)
        clusters = fcluster(self.hierarchy, max_d, **fcluster_config)
        self.df["cluster"] = pd.Series(clusters, index=self.df.index)

        config = self.metadata["cluster"]["cluster"]
        config["max_d"] = max_d
        config["criterion"] = fcluster_config["criterion"]

    @staticmethod
    def data_output_path(general_output_path):
        """ Taking the general output path, return the path to the data file.
        """
        return os.path.join(
            os.path.dirname(general_output_path),
            os.path.basename(general_output_path) + "_data.csv"
        )

    @staticmethod
    def metadata_output_path(general_output_path):
        """ Taking the general output path, return the path to the config file.
        """
        return os.path.join(
            os.path.dirname(general_output_path),
            os.path.basename(general_output_path) + "_metadata.json"
        )

    def write(self, general_output_path):
        pass

        # *** 1. Clean files and make sure the folders exist ***

        config_path = self.metadata_output_path(general_output_path)
        data_path = self.data_output_path(general_output_path)

        self.log.info("Will write config to '{}'.".format(config_path))
        self.log.info("Will write data to '{}'.".format(data_path))

        paths = [config_path, data_path]
        for path in paths:
            dirname = os.path.dirname(path)
            if dirname and not os.path.exists(dirname):
                self.log.debug("Creating directory '{}'.".format(dirname))
                os.makedirs(dirname)
            if os.path.exists(path):
                self.log.debug("Removing file '{}'.".format(path))
                os.remove(path)

        # *** 2. Write out config ***

        self.log.debug("Converting config data to json and writing to file "
                       "'{}'.".format(config_path))
        with open(config_path, "w") as config_file:
            json.dump(self.metadata, config_file, sort_keys=True, indent=4)
        self.log.debug("Done.")

        # *** 3. Write out data ***

        self.log.debug("Converting data to csv and writing to "
                       "file '{}'.".format(data_path))
        if self.df.empty:
            self.log.error("Dataframe seems to be empty. Still writing "
                           "out anyway.")
        with open(data_path, "w") as data_file:
            self.df.to_csv(data_file)
        self.log.debug("Done")

        # *** 4. Done ***

        self.log.info("Writing out finished.")


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
    c.cluster()
    c.write(args.output_path)

    plt.show()

if __name__ == "__main__":
    # Run command line interface
    cli()
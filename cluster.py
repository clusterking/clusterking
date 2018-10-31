#!/usr/bin/env python3

"""Read the results from scan.py and clusters them.
"""

# standard
import argparse
import atexit
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
    """This class implements the clustering functionality

    Example:
    ```python
    c = Cluster("output/scan/general_output")
    c.build_hierarchy()
    c.dendogram(show=True)
    c.cluster(max_d=0.2)
    c.write("output/scan/general_output)
    ```
    """

    # **************************************************************************
    # A:  Setup
    # **************************************************************************

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

        # should we wait for plots to be shown?
        self.wait_plots = False
        # call self.close() when this script exits
        atexit.register(self.close)

    def _get_scan_data(self):
        """ Read data from scan.py """
        path = Scanner.data_output_path(self.input_path)
        self.log.debug("Loading scanner data from '{}'.".format(path))
        with open(path, 'r') as data_file:
            self.df = pd.read_csv(data_file)
        self.df.set_index("index", inplace=True)
        self.log.debug("Done.")

    def _get_scan_metadata(self):
        """ Read metadata from scan.py """
        path = Scanner.config_output_path(self.input_path)
        self.log.debug("Loading scanner metadata from '{}'.".format(path))
        with open(path, 'r') as metadata_file:
            scan_metadata = json.load(metadata_file)
        self.metadata.update(scan_metadata)
        self.log.debug("Done.")

    # **************************************************************************
    # B:  Cluster
    # **************************************************************************

    def build_hierarchy(self, **kwargs) -> None:
        """ Build the hierarchy object.
        Args:
            **kwargs: keyword arguments to scipy.cluster.hierarchy.linkage
        """
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
        nclusters = len(set(clusters))
        self.log.info("This resulted in {} clusters.".format(nclusters))

        self.df["cluster"] = pd.Series(clusters, index=self.df.index)

        config = self.metadata["cluster"]["cluster"]
        config["max_d"] = max_d
        config["criterion"] = fcluster_config["criterion"]
        config["nclusters"] = nclusters

    # **************************************************************************
    # C:  Built-in plotting methods
    # **************************************************************************

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

        dendrogram(
            self.hierarchy,
            ax=ax,
            **den_config
        )

        if show:
            fig.show()

            # Trigger a plt.show() at the end of this script
            self.wait_plots = True

        if output:
            assert(isinstance(output, str))
            dirname = os.path.dirname(output)
            if dirname and not os.path.exists(dirname):
                self.log.debug("Creating dir '{}'.".format(dirname))
                os.makedirs(dirname)
            fig.savefig(output, bbox_inches="tight")
            self.log.info("Wrote dendogram to '{}'.".format(output))

        return ax

    # **************************************************************************
    # D:  Write out
    # **************************************************************************

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
        """ Write out all results.
        IMPORTANT NOTE: All output files will always be overwritten!

        Args:
            general_output_path: Path to the output file without file extension.
                We will add suffixes and file extensions to this!
        """

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

    # **************************************************************************
    # E:  MISC
    # **************************************************************************

    def close(self):
        """This method is called when this script exits. A corresponding
        hook has been set up in the __init__ method.
        We use that to wait for interactive plots/plotting windows to close
        if we made any. """
        if self.wait_plots:
            # this will block until all plotting windows were closed
            plt.show()


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
    c.dendogram(show=True)
    c.cluster(max_d=0.2)
    c.write(args.output_path)

if __name__ == "__main__":
    # Run command line interface
    cli()

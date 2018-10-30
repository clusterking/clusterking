#!/usr/bin/env python3

""" Scans the NP parameter space in a grid and also q2, producing the
normalized q2 distribution. """

# standard modules
import argparse
import collections
import datetime
import functools
import multiprocessing
import numpy as np
import os
import os.path
import sys
import time
from typing import List, Tuple
import json
import pandas as pd

# internal modules
from modules.inputs import Wilson
import modules.distribution as distribution
from modules.util.cli import yn_prompt
from modules.util.log import get_logger


def nested_dict():
    """ This is very clever and stolen from
    https://stackoverflow.com/questions/16724788/
    Use it to initialize a dictionary-like object which automatically adds
    levels.
    E.g.
        a = nested_dict()
        a['test']['this']['is']['working'] = "yaaay"
    """
    return collections.defaultdict(nested_dict)


def calculate_bpoint(w: Wilson, bin_edges: np.array) -> np.array:
    """Calculates one benchmark point.

    Args:
        w: Wilson coefficients
        bin_edges:

    Returns:
        np.array of the integration results
    """

    return distribution.bin_function(lambda x: distribution.dGq2(w, x),
                                     bin_edges,
                                     normalized=True)


class Scanner(object):
    """ This implements all the functionality. """
    def __init__(self):
        self.log = get_logger("Scanner")

        # benchmark points (i.e. Wilson coefficients)
        # Do NOT directly modify this, but use one of the methods below.
        self._bpoints = []

        # EDGES of the q2 bins
        # Do NOT directly modify this, but use one of the methods below.
        self._q2points = np.array([])

        # This will hold all of the results
        self.df = pd.DataFrame()

        # This will hold all the configuration that we will write out
        self.config = nested_dict()

    def set_q2points_manual(self, q2points: np.array) -> None:
        """ Manually set the edges of the q2 binning. """
        self._q2points = q2points
        self.config["q2points"]["sampling"] = "manual"

    def set_q2points_equidist(
        self,
        no_bins,
        dist_max=distribution.q2max,
        dist_min=distribution.q2min
    ) -> None:
        """ Set the edges of the q2 binning automatically.

        Args:
            no_bins: Number of bins
            dist_max: The right edge of the last bin (=maximal q2 value)
            dist_min: The left edge of the first bin (=minimal q2 value)

        Returns:
            None
        """
        self._q2points = np.linspace(dist_min, dist_max, no_bins+1)
        self.config["q2points"]["sampling"] = "equidistant"

    def set_bpoints_manual(self, bpoints) -> None:
        """ Manually set a list of benchmark points """
        self._bpoints = bpoints
        self.config["bpoints"]["sampling"] = "manual"

    # todo: make method more flexible
    def set_bpoints_equidist(self, sampling=20) -> None:
        """ Set a list of 'equidistant' benchmark points. """
        bpoints = []
        # I set epsR an epsSR to zero  as the observables are only sensitive to
        # linear combinations  L + R
        eps_r = 0.
        eps_sr = 0.
        for eps_l in np.linspace(-0.30, 0.30, sampling):
            for eps_sl in np.linspace(-0.30, 0.30, sampling):
                for eps_t in np.linspace(-0.40, 0.40, sampling):
                    bpoints.append(Wilson(eps_l, eps_r, eps_sr, eps_sl, eps_t))
        self._bpoints = bpoints
        self.config["bpoints"]["sampling"] = "equidistant"

    # todo: docstring
    @staticmethod
    def data_output_path(general_output_path):
        return os.path.join(
            os.path.dirname(general_output_path),
            os.path.basename(general_output_path) + "_data.csv"
        )

    # todo: docstring
    @staticmethod
    def config_output_path(general_output_path):
        return os.path.join(
            os.path.dirname(general_output_path),
            os.path.basename(general_output_path) + "_config.json"
        )

    def write(self, general_output_path) -> None:
        """ Write out all results.
        IMPORTANT NOTE: All output files will always be overwritten!

        Args:
            general_output_path: Path to the output file without file extension.
            We will add suffixes and file extensions to this!
        """
        if self.df.empty:
            self.log.error("Data frame is empty yet attempting to write out. "
                           "Ignore.")
            return

        # *** 1. Clean files and make sure the folders exist ***

        config_path = self.config_output_path(general_output_path)
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
        global_config = {"scan": self.config}
        with open(config_path, "w") as config_file:
            json.dump(global_config, config_file, sort_keys=True, indent=4)
        self.log.debug("Done.")

        # *** 3. Write out data ***

        self.log.debug("Converting data to csv and writing to "
                       "file '{}'.".format(data_path))
        if self.df.empty:
            self.log.error("Dataframe seems to be empty. Still writing "
                           "out anyway.")
        with open(data_path, "w") as data_file:
            self.df.to_csv(data_file)
        self.log.debug("Done.")

        self.log.info("Writing out finished.")


    def run(self, no_workers=4) -> None:
        """Calculate all benchmark points in parallel and saves the result in
        self.df.

        Args:
            no_workers: Number of worker nodes/cores
        """

        if self._bpoints == 0:
            self.log.error("No benchmark points specified. Returning without "
                           "doing anything.")
        if self._q2points.size == 0:
            self.log.error("No q2points specified. Returning without "
                           "doing anything.")

        # pool of worker nodes
        pool = multiprocessing.Pool(processes=no_workers)

        # this is the worker function: calculate_bpoints with additional
        # arguments frozen
        worker = functools.partial(calculate_bpoint,
                                   bin_edges=self._q2points)

        results = pool.imap(worker, self._bpoints)

        # close the queue for new jobs
        pool.close()

        self.log.info("Started queue with {} job(s) distributed over up to {} "
                      "core(s)/worker(s).".format(len(self._bpoints), no_workers))

        starttime = time.time()

        rows = []
        for index, result in enumerate(results):

            bpoint_dict = self._bpoints[index].dict().values()
            rows.append([*bpoint_dict, *result])

            timedelta = time.time() - starttime

            completed = index + 1
            remaining_time = (len(self._bpoints) - completed) * timedelta/completed
            self.log.debug("Progress: {:04}/{:04} ({:04.1f}%) of benchmark points. "
                      "Time/bpoint: {:.1f}s => "
                      "time remaining: {}".format(
                         completed,
                         len(self._bpoints),
                         100*completed/len(self._bpoints),
                         timedelta/completed,
                         datetime.timedelta(seconds=remaining_time)
                         ))

        # Wait for completion of all jobs here
        pool.join()

        self.log.debug("Converting data to pandas dataframe.")
        cols = ["eps_l", "eps_r", "eps_sr", "eps_sl", "eps_t"]
        cols.extend(["bin{}".format(no_bin) for no_bin in range(len(self._q2points)-1)])
        self.df = pd.DataFrame(data=rows, columns=cols)

        self.log.info("Integration done.")


def cli():
    """Command line interface to run the integration jobs from the command
    line with additional options.

    Simply run this script with '--help' to see all options.
    """

    desc = "Build q2 histograms for the different NP benchmark points."

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-o", "--output",
                        help="Output file.",
                        default="output/scan/global_results",
                        dest="output_path")
    parser.add_argument("-p", "--parallel",
                        help="Number of processes to run in parallel",
                        type=int,
                        default=4)
    parser.add_argument("-n", "--np-grid-subdivision",
                        help="Number of points sampled per NP parameter",
                        type=int,
                        default=20,
                        dest="np_grid_subdivision")
    parser.add_argument("-g", "--grid-subdivision",
                        help="Number of bins in q2",
                        type=int,
                        default=15,
                        dest="q2_bins")
    args = parser.parse_args()

    s = Scanner()

    s.set_bpoints_equidist(args.np_grid_subdivision)
    s.log.info("NP parameters will be sampled with {} sampling points.".format(
        args.np_grid_subdivision))

    s.set_q2points_equidist(args.q2_bins)
    s.log.info("q2 will be sampled with {} sampling points.".format(
        args.q2_bins))

    paths = [s.config_output_path(args.output_path),
             s.data_output_path(args.output_path)]
    existing_paths = [ path for path in paths if os.path.exists(path) ]
    if existing_paths:
        agree = yn_prompt("Output paths {} already exist(s) and will be "
                          "overwritten. "
                          "Proceed?".format(', '.join(existing_paths)))
        if not agree:
            s.log.critical("User abort.")
            sys.exit(1)

    # s.log.info("Output file will be: '{}'.".format(args.output_path))

    s.run(no_workers=args.parallel)

    s.write(args.output_path)


if __name__ == "__main__":
    # Check command line arguments and run computation
    cli()

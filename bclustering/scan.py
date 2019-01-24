#!/usr/bin/env python3

""" Scans the NP parameter space in a grid and also q2, producing the
normalized q2 distribution. """

# std
import argparse
import datetime
import functools
import json
import multiprocessing
import pathlib
import sys
import time
from typing import Union

# 3rd party
import numpy as np
import pandas as pd

# ours
import bclustering.physics.bdlnu.distribution as distribution
from bclustering.util.cli import yn_prompt
from bclustering.util.log import get_logger
from bclustering.util.metadata import nested_dict, git_info
from bclustering.wilson import Wilson


# NEEDS TO BE GLOBAL FUNCTION for multithreading
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
    """ Scans the NP parameter space in a grid and also q2, producing the
    normalized q2 distribution.

    Usage example:

    ```python
    # Initialize Scanner object
    s = Scanner()

    # Sample 3 points for each of the 5 Wilson coefficients
    s.set_bpoints_equidist(3)

    # Use 5 bins in q2
    s.set_q2points_equidist(5)

    # Start running with maximally 3 cores
    s.run(no_workers=3)

    # Write out results
    s.write("output/scan/global_output")
    ```

    This is example is equivalent to calling
    ```./scan.py -n 3 -g 5 -o output/scan/global_output -p 3```
    or
    ```./scan.py --np-grid-subdivision 3 --grid-subdivision 5 \
        --output output/scan/global_output --parallel 3 ```
    """

    # **************************************************************************
    # A:  Setup
    # **************************************************************************

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
        self.metadata = nested_dict()
        self.metadata["scan"]["git"] = git_info(self.log)
        self.metadata["scan"]["time"] = time.strftime("%a %_d %b %Y %H:%M", time.gmtime())

    def set_q2points_manual(self, q2points: np.array) -> None:
        """ Manually set the edges of the q2 binning. """
        self._q2points = q2points
        self.metadata["scan"]["q2points"]["sampling"] = "manual"
        self.metadata["scan"]["q2points"]["nbins"] = len(self._q2points)

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
        md = self.metadata["scan"]["q2points"]
        md["sampling"] = "equidistant"
        md["nbins"] = len(self._q2points) - 1
        md["min"] = dist_min
        md["max"] = dist_max

    def set_bpoints_manual(self, bpoints) -> None:
        """ Manually set a list of benchmark points """
        self._bpoints = bpoints
        self.metadata["scan"]["bpoints"]["sampling"] = "manual"
        self.metadata["scan"]["bpoints"]["npoints"] = len(self._bpoints)

    def set_bpoints_equidist(self, sampling=20, minima=None,
                             maxima=None) -> None:
        """ Set a list of 'equidistant' benchmark points.

        Args:
            sampling: If int, this will be taken as the number of points to
                be sampled from all 5 Wilson coefficients.
                If a dictionary with the keys 'l', 'r', 'sl' is supplied, these
                will set the number of sampling pointsn separately for each
                Wilson coefficient.
            minima:
                Minima of the Wilson coefficients. If None, default values are
                chosen, else supply a dictionary similar to the one of the
                sampling parameter.
            maxima:
                Similar to the minimal parameter

        Returns:
            None
        """

        # I set epsR an epsSR to zero  as the observables are only sensitive to
        # linear combinations  L + R

        _min = minima
        if not _min:
            _min = {
                'l': -0.3,
                'r': 0.,
                'sl': -0.3,
                'sr': 0.,
                't': -0.4
            }

        _max = maxima
        if not _max:
            _max = {
                'l': 0.3,
                'r': 0.,
                'sl': 0.3,
                'sr': 0.,
                't': 0.4
            }

        if isinstance(sampling, int):
            _sam = {key: sampling for key in ['l', 'r', 'sl', 'sr', 't']}
        elif isinstance(sampling, dict):
            _sam = sampling
        else:
            raise ValueError("Can't handle that type.")

        def lsp(dmin, dmax, dsam):
            """ Like np.linspace, but throws out identical values"""
            return sorted(list(set(np.linspace(dmin, dmax, dsam))))

        lists = {
            key: lsp(_min[key], _max[key], _sam[key])
            for key in ['l', 'r', 'sr', 'sl', 't']
        }

        bpoints = []
        for l in lists['l']:
            for r in lists['r']:
                for sr in lists['sr']:
                    for sl in lists['sl']:
                        for t in lists['t']:
                            bpoints.append(Wilson(l, r, sr, sl, t))

        self._bpoints = bpoints
        md = self.metadata["scan"]["bpoints"]
        md["sampling"] = "equidistant"
        md["npoints"] = len(self._bpoints)
        md["min"] = _min
        md["max"] = _max
        # do not just take the sampling value (because if start == end, they
        # are incorrect and we only have exactly one point)
        md["sample"] = {
            key: len(value) for key, value in lists.items()
        }

    # **************************************************************************
    # B:  Run
    # **************************************************************************

    def run(self, no_workers=4) -> None:
        """Calculate all benchmark points in parallel and saves the result in
        self.df.

        Args:
            no_workers: Number of worker nodes/cores
        """

        if self._bpoints == 0:
            self.log.error("No benchmark points specified. Returning without "
                           "doing anything.")
            return
        if self._q2points.size == 0:
            self.log.error("No q2points specified. Returning without "
                           "doing anything.")
            return

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
                      "core(s)/worker(s).".format(len(self._bpoints),
                                                  no_workers))

        starttime = time.time()

        rows = []
        for index, result in enumerate(results):

            bpoint_dict = self._bpoints[index].dict().values()
            rows.append([*bpoint_dict, *result])

            timedelta = time.time() - starttime

            completed = index + 1
            rem_time = (len(self._bpoints) - completed) * timedelta/completed
            self.log.debug("Progress: {}/{} ({:04.1f}%) of bpoints. "
                           "Time/bpoint: {:.1f}s => "
                           "time remaining: {} "
                           "(total elapsed: {})".format(
                                str(completed).zfill(
                                    len(str(len(self._bpoints)))),
                                len(self._bpoints),
                                100*completed/len(self._bpoints),
                                timedelta/completed,
                                datetime.timedelta(seconds=int(rem_time)),
                                datetime.timedelta(seconds=int(timedelta))
                                ))

        # Wait for completion of all jobs here
        pool.join()

        self.log.debug("Converting data to pandas dataframe.")
        cols = list(Wilson(0, 0, 0, 0, 0).dict().keys())
        cols.extend(
            ["bin{}".format(no_bin) for no_bin in range(len(self._q2points)-1)]
        )
        self.df = pd.DataFrame(data=rows, columns=cols)
        self.df.index.name = "index"

        self.log.info("Integration done.")

    # **************************************************************************
    # C:  Write out
    # **************************************************************************

    @staticmethod
    def data_output_path(general_output_path: Union[pathlib.Path, str]) \
            -> pathlib.Path:
        """ Taking the general output path, return the path to the data file.
        """
        path = pathlib.Path(general_output_path)
        return path.parent / (path.name + "_data.csv")

    @staticmethod
    def metadata_output_path(general_output_path: Union[pathlib.Path, str]) \
            -> pathlib.Path:
        """ Taking the general output path, return the path to the metadata file.
        """
        path = pathlib.Path(general_output_path)
        return path.parent / (path.name + "_metadata.json")

    def write(self, general_output_path: Union[pathlib.Path, str]) -> None:
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

        metadata_path = self.metadata_output_path(general_output_path)
        data_path = self.data_output_path(general_output_path)

        self.log.info("Will write metadata to '{}'.".format(metadata_path))
        self.log.info("Will write data to '{}'.".format(data_path))

        paths = [metadata_path, data_path]
        for path in paths:
            if not path.parent.is_dir():
                self.log.debug("Creating directory '{}'.".format(path.parent))
                path.parent.mkdir(parents=True)
            if path.exists():
                self.log.debug("Removing file '{}'.".format(path))
                path.unlink()

        # *** 2. Write out metadata ***

        self.log.debug("Converting metadata data to json and writing to file "
                       "'{}'.".format(metadata_path))
        with metadata_path.open("w") as metadata_file:
            json.dump(self.metadata, metadata_file, sort_keys=True, indent=4)
        self.log.debug("Done.")

        # *** 3. Write out data ***

        self.log.debug("Converting data to csv and writing to "
                       "file '{}'.".format(data_path))
        if self.df.empty:
            self.log.error("Dataframe seems to be empty. Still writing "
                           "out anyway.")
        self.df.index.name = "index"
        with data_path.open("w") as data_file:
            self.df.to_csv(data_file)
        self.log.debug("Done")

        # *** 4. Done ***

        self.log.info("Writing out finished.")


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

    paths = [s.metadata_output_path(args.output_path),
             s.data_output_path(args.output_path)]
    existing_paths = [path for path in paths if path.exists()]
    if existing_paths:
        agree = yn_prompt("Output paths {} already exist(s) and will be "
                          "overwritten. "
                          "Proceed?".format(', '.join(map(str, existing_paths))))
        if not agree:
            s.log.critical("User abort.")
            sys.exit(1)

    # s.log.info("Output file will be: '{}'.".format(args.output_path))

    s.run(no_workers=args.parallel)

    s.write(args.output_path)


if __name__ == "__main__":
    # Check command line arguments and run computation
    cli()

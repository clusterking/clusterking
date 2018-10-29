#!/usr/bin/env python3

""" Scans the NP parameter space in a grid and also q2, producing the
normalized q2 distribution. """

# standard modules
import argparse
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


log = get_logger("Scan")


def get_bpoints(np_grid_subdivisions=20) -> List[Wilson]:
    """Get a list of all benchmark points.

    Args:
        np_grid_subdivisions: Number of subdivision/sample points for the NP
            parameter grid that is sampled

    Returns:
        a list of all benchmark points as tuples in the form
        (epsL, epsR, epsSR, epsSL, epsT)
    """

    bps = []  # list of of benchmark points

    # I set epsR an epsSR to zero  as the observables are only sensitive to
    # linear combinations  L + R
    epsR = 0
    epsSR = 0
    for epsL in np.linspace(-0.30, 0.30, np_grid_subdivisions):
        for epsSL in np.linspace(-0.30, 0.30, np_grid_subdivisions):
            for epsT in np.linspace(-0.40, 0.40, np_grid_subdivisions):
                bps.append(Wilson(epsL, epsR, epsSR, epsSL, epsT))

    return bps


def calculate_bpoint(w: Wilson, bin_edges: np.array) -> np.array:
    """Calculates one benchmark point.

    Args:
        w: Wilson coefficients
        bin_edges:

    Returns:
        Resulting q2 histogram as a list of tuples (q2, distribution at this q2)
    """

    return distribution.bin_function(lambda x: distribution.dGq2(w, x),
                                     bin_edges,
                                     normalized=True)


# todo: writeout should be different method
def run(bpoints: List[Wilson], no_workers=4, output_path="global_results.out",
        no_bins=15):
    """Calculate all benchmark points in parallel.

    Args:
        bpoints: Benchmark points
        no_workers: Number of worker nodes/cores
        output_path: Output path. Will be overwritten if existing!
        no_bins: q2 grid spacing

    Returns:
        None
    """

    # pool of worker nodes
    pool = multiprocessing.Pool(processes=no_workers)

    bin_edges = np.linspace(distribution.q2min, distribution.q2max, no_bins+1)

    # this is the worker function: calculate_bpoints with additional
    # arguments frozen
    worker = functools.partial(calculate_bpoint,
                               bin_edges=bin_edges)

    results = pool.imap(worker, bpoints)

    # close the queue for new jobs
    pool.close()

    log.info("Started queue with {} job(s) distributed over up to {} "
             "core(s)/worker(s).".format(len(bpoints), no_workers))

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_path):
        os.remove(output_path)

    starttime = time.time()

    rows = []
    for index, result in enumerate(results):

        bpoint_dict = bpoints[index].dict().values()
        rows.append([*bpoint_dict, *result])

        timedelta = time.time() - starttime

        completed = index + 1
        remaining_time = (len(bpoints) - completed) * timedelta/completed
        log.debug("Progress: {:04}/{:04} ({:04.1f}%) of benchmark points. "
                  "Time/bpoint: {:.1f}s => "
                  "time remaining: {}".format(
                     completed,
                     len(bpoints),
                     100*completed/len(bpoints),
                     timedelta/completed,
                     datetime.timedelta(seconds=remaining_time)
                     ))

    # Wait for completion of all jobs here
    pool.join()

    log.debug("Converting data to pandas dataframe.")
    cols = ["eps_l", "eps_r", "eps_sr", "eps_sl", "eps_t"]
    cols.extend(["bin{}".format(no_bin) for no_bin in range(no_bins)])
    df = pd.DataFrame(data=rows, columns=cols)

    log.debug("Converting data to json.")

    json_data = {}
    json_data["data"] = df  #.to_json(orient="split")
    json_data["config"] = {}
    json_data["config"]["bin_edges"] = list(bin_edges)

    log.debug("Writing out data.")
    with open(output_path, "w") as outfile:
        outfile.write(pd.io.json.dumps(json_data))

    log.info("Finished")


def cli():
    """Command line interface to run the integration jobs from the command
    line with additional options.

    Simply run this script with '--help' to see all options.
    """

    desc = "Build q2 histograms for the different NP benchmark points."

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-o", "--output",
                        help="Output file.",
                        default="output/scan/global_results.out",
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
                        help="Number of sample points between minimal and "
                             "maximal q2",
                        type=int,
                        default=15,
                        dest="grid_subdivision")
    args = parser.parse_args()

    log.info("NP parameters will be sampled with {} sampling points.".format(
        args.np_grid_subdivision))
    log.info("q2 will be sampled with {} sampling points.".format(
        args.grid_subdivision))

    bpoints = get_bpoints(args.np_grid_subdivision)
    log.info("Total integrations to be performed: {}.".format(
        len(bpoints) * args.grid_subdivision))

    if os.path.exists(args.output_path):
        agree = yn_prompt("Output path '{}' already exists and will be "
                          "overwritten. Proceed?".format(args.output_path))
        if not agree:
            log.critical("User abort.")
            sys.exit(1)

    log.info("Output file: '{}'.".format(args.output_path))

    run(bpoints,
        no_workers=args.parallel,
        output_path=args.output_path,
        no_bins=args.grid_subdivision)


if __name__ == "__main__":
    # Check command line arguments and run computation
    cli()

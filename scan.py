#!/usr/bin/env python3

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

# internal modules
import modules.distribution as distribution
from modules.util.misc import yn_prompt

###
### scans the NP parameter space in a grid and also q2, producing the normalized q2 distribution
###   I guess this table can then be used for the clustering algorithm, any.  

## q2 distribution normalized by total,  integral of this would be 1 by definition
## dGq2normtot(epsL, epsR, epsSR, epsSL, epsT,q2)


def get_bpoints(np_grid_subdivisions = 20):
    """ Get a list of all benchmark points.

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
                bps.append((epsL, epsR, epsSR, epsSL, epsT))

    return bps


def calculate_bpoint(bpoint, grid_subdivision):
    """ Calculates one benchmark point and returns the string to be written
    to the output file.

    Args:
        bpoint: epsL, epsR, epsSR, epsSL, epsT
        grid_subdivision: q2 grid spacing

    Returns:
        Result string to be written in the output file
    """

    result_list = []
    for q2 in np.linspace(distribution.q2min, distribution.q2max, grid_subdivision):
        dist_tmp = distribution.dGq2normtot(*bpoint, q2)
        result_list.append((q2, dist_tmp))

    result_string = ""
    for q2, dist_tmp in result_list:
            for param in bpoint:
                result_string += "{:.5f}    ".format(param)
            result_string += '{:.5f}     {:.10f}'.format(q2 , dist_tmp)

    return result_string


def run_parallel(bpoints, no_workers=4, output_path="global_results.out",
                 grid_subdivision=15):
    """
    Run integrations in parallel (main function).

    Args:
        bpoints: Benchmark points
        no_workers: Number of worker nodes/cores
        output_path: Output path. Will be overwritten if existing!
        grid_subdivision: q2 grid spacing

    Returns:
        None
    """

    # pool of worker nodes
    pool = multiprocessing.Pool(processes=no_workers)

    # this is the worker function: calculate_bpoints with additional
    # arguments frozen
    worker = functools.partial(calculate_bpoint,
                               grid_subdivision=grid_subdivision)

    # submit jobs (use imap_unordered if we do not care for the order)
    results = pool.imap(worker, bpoints)

    # close the queue for new jobs
    pool.close()

    print("Started queue with {} jobs.".format(len(bpoints)))

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    starttime = time.time()
    # Note: this will overwrite the output path! Ask user in interface whether
    # he's ok with that.
    # buffering = 1: Write out every line
    with open(output_path, "w", buffering=1) as outfile:
        for index, result in enumerate(results):

            outfile.write(result + "\n")

            timedelta = time.time() - starttime

            completed = index + 1
            remaining_time = (len(bpoints) - completed) * timedelta/completed
            print("Progress: {:04}/{:04} ({:04.1f}%) of benchmark points. "
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
    print("Finished")


def cli():
    """ Command line interface to run the integration jobs from the command
    line with additional options.
    Simply run this script with '--help' to see all options.
    """
    parser = argparse.ArgumentParser()
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

    print("NP parameters will be sampled with {} sampling points.".format(
        args.np_grid_subdivision))
    print("q2 will be sampled with {} sampling points.".format(args.grid_subdivision))

    if os.path.exists(args.output_path):
        agree = yn_prompt("Output path '{}' already exists and will be "
                          "overwritten. Proceed?".format(args.output_path))
        if not agree:
            print("User abort.")
            sys.exit(1)

    print("Output file: '{}'.".format(args.output_path))

    bpoints = get_bpoints(args.np_grid_subdivision)
    print("Total integrations to be performed: {}.".format(
        len(bpoints) * args.grid_subdivision))
    run_parallel(bpoints,
                 no_workers=args.parallel,
                 output_path=args.output_path,
                 grid_subdivision=args.grid_subdivision)


if __name__ == "__main__":
    # Check command line arguments and run computation
    cli()

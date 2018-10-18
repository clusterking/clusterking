#!/usr/bin/env python3

import numpy as np
import distribution
import time
import datetime
import multiprocessing
import functools
import os

###
### scans the NP parameter space in a grid and also q2, producing the normalized q2 distribution
###   I guess this table can then be used for the clustering algorithm, any.  

## q2 distribution normalized by total,  integral of this would be 1 by definition
## dGq2normtot(epsL, epsR, epsSR, epsSL, epsT,q2):


# Output file
output_path = "global_results.out"


def get_bpoints(np_grid_subdivisions = 20):
    """ Get a list of all benchmark points.

    Args:
        np_grid_subdivisions:

    Returns:
    """

    bpoints = []

    # todo: This setting should be implemented differently perhaps
    #  I set epsR an epsSR to zero  as the observables are only sensitive to
    # linear combinations  L + R
    epsR = 0
    epsSR = 0
    for epsL in np.linspace(-0.30, 0.30, np_grid_subdivisions):
        for epsSL in np.linspace(-0.30, 0.30, np_grid_subdivisions):
            for epsT in np.linspace(-0.40, 0.40, np_grid_subdivisions):
                bpoints.append((epsL, epsR, epsSR, epsSL, epsT))

    return bpoints



def calculate_bpoint(lock, bpoint):
    """ Calculates one benchmark point and writes the result to the output
    file. This method is designed to be thread save.

    Args:
        lock: multithreading.lock instance
        bpoint: epsL, epsR, epsSR, epsSL, epsT

    Returns:
        None

    """

    # todo: This setting should be implemented differently perhaps
    grid_subdivisions = 15

    result_list = []
    for q2 in np.linspace(distribution.q2min, distribution.q2max, grid_subdivisions):
        dist_tmp = distribution.dGq2normtot(*bpoint, q2)
        result_list.append((q2, dist_tmp))

    # Acquire and release lock before writing into file to make sure
    # this works with multithreading
    global output_path
    lock.acquire()
    with open(output_path, "a") as outfile:
        for q2, dist_tmp in result_list:
            # todo: switch to <str>.format
            for param in bpoint:
                outfile.write("%.5f    "  % param)
            outfile.write('%.5f   ' % q2  + '%.10f       ' % dist_tmp + '\n')
    lock.release()


if __name__ == "__main__":

    # remove old output file
    if os.path.exists(output_path):
        os.remove(output_path)

    no_workers = 2  # number of cores used
    pool = multiprocessing.Pool(processes=no_workers)

    manager = multiprocessing.Manager()
    lock = manager.Lock()

    bpoints = get_bpoints()

    worker = functools.partial(calculate_bpoint, lock)
    results = pool.imap_unordered(worker, bpoints)

    pool.close()  # no more work will be added

    # ** Everything below here is just a status monitor **

    completed = 0
    starttime = time.time()

    while True:
        # results._index holds the number of the completed results
        if completed == results._index:
            # Wait till we have a new result
            time.sleep(0.5)
            continue

        completed = results._index

        if completed == len(bpoints):
            break

        timedelta = time.time() - starttime

        if completed > 0:
            remaining_time = (len(bpoints) - completed) * timedelta/completed
            print("Progress: {:04}/{} ({:04.1f}%) of benchmark points. "
                  "Time/bpoint: {:.1f}s => "
                  "time remaining: {}".format(
                     completed,
                     len(bpoints),
                     100*completed/len(bpoints),
                     timedelta/completed,
                     datetime.timedelta(seconds=remaining_time)
                 ))

    pool.join()

# print("***** scan finalized *****")


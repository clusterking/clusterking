#!/usr/bin/env python3

""" Scans the NP parameter space in a grid and also q2, producing the
normalized q2 distribution. """

# std
import functools
import itertools
import multiprocessing
import os
import shutil
import time
from typing import Callable, List, Sized

# 3rd party
import numpy as np
import pandas as pd
import tqdm

# ours
from clusterking.data.data import Data
import clusterking.maths.binning
from clusterking.util.metadata import git_info, failsafe_serialize, nested_dict
from clusterking.util.log import get_logger


class SpointCalculator(object):
    """ A class that holds the function with which we calculate each
    point in wilson space. Note that this has to be a separate class from
    Scanner to avoid problems related to multiprocessing's use of the pickle
    library, which are described here:
    https://stackoverflow.com/questions/1412787/
    """
    def __init__(self, func: Callable, binning: Sized, normalize, kwargs):
        self.dfunc = func
        self.dfunc_binning = binning
        self.dfunc_normalize = normalize
        self.dfunc_kwargs = kwargs

    def calc(self, spoint) -> np.array:
        """Calculates one point in wilson space.

        Args:
            spoint: Wilson coefficients

        Returns:
            np.array of the integration results
        """

        if self.dfunc_binning is not None:
            return clusterking.maths.binning.bin_function(
                functools.partial(self.dfunc, spoint, **self.dfunc_kwargs),
                self.dfunc_binning,
                normalize=self.dfunc_normalize
            )
        else:
            return self.dfunc(spoint, **self.dfunc_kwargs)


class Scanner(object):

    # **************************************************************************
    # A:  Setup
    # **************************************************************************

    def __init__(self):
        self.log = get_logger("Scanner")

        #: Points in wilson space
        #:  Use self.spoints to access this
        self._spoints = []

        #: Instance of SpointCalculator to perform the claculations of
        #:  the wilson space points.
        self._spoint_calculator = None  # type: SpointCalculator

        self.md = nested_dict()
        self.md["git"] = git_info(self.log)
        self.md["time"] = time.strftime(
            "%a %_d %b %Y %H:%M", time.gmtime()
        )

    # Write only access
    @property
    def spoints(self):
        """ Points in parameter space that are sampled."""
        return self._spoints

    def set_dfunction(
            self,
            func: Callable,
            binning: Sized = None,
            normalize=False,
            **kwargs
    ):
        """ Set the function that generates the distributions that are later
        clustered (e.g. a differential cross section).

        Args:
            func: A function that takes the point in parameter space
                as the first argument.
                It should either return a float (if the binning
                option is specified), or a np.array elsewise.
            binning: If this parameter is not set (None), we will use the
                function as is. If it is set to an array-like object, we will
                integrate the function over the bins specified by this
                parameter.
            normalize: If a binning is specified, normalize the resulting
                distribution
            **kwargs: All other keyword arguments are passed to the function.

        Returns:
            None

        """
        md = self.md["dfunction"]
        try:
            md["name"] = func.__name__
            md["doc"] = func.__doc__
        except AttributeError:
            try:
                # For functools.partial objects
                # noinspection PyUnresolvedReferences
                md["name"] = "functools.partial({})".format(func.func.__name__)
                # noinspection PyUnresolvedReferences
                md["doc"] = func.func.__doc__
            except AttributeError:
                pass

        md["kwargs"] = failsafe_serialize(kwargs)
        if binning is not None:
            md["nbins"] = len(binning) - 1

        # global spoint calculator
        self._spoint_calculator = SpointCalculator(
            func, binning, normalize, kwargs
        )

    # todo: implement set_spoints in a more general way here!

    # **************************************************************************
    # B:  Run
    # **************************************************************************

    def run(self, data: Data, no_workers=None) -> None:
        """Calculate all sample points in parallel and saves the result in
        self.df.

        Args:
            data: Data object.
            no_workers: Number of worker nodes/cores. Default: Total number of
                cores.
        """

        if not self._spoints:
            self.log.error(
                "No sample points specified. Returning without doing "
                "anything."
            )
            return
        if not self._spoint_calculator:
            self.log.error(
                "No function specified. Please set it "
                "using ``Scanner.set_dfunction``. Returning without doing "
                "anything."
            )
            return

        if not no_workers:
            no_workers = os.cpu_count()
        if not no_workers:
            # os.cpu_count() didn't work
            self.log.warn(
                "os.cpu_count() not determine number of cores. Fallling "
                "back to no_workser = 1."
            )
            no_workers = 1

        # pool of worker nodes
        pool = multiprocessing.Pool(processes=no_workers)

        # this is the worker function.
        worker = self._spoint_calculator.calc

        results = pool.imap(worker, self._spoints)

        # close the queue for new jobs
        pool.close()

        self.log.info(
            "Started queue with {} job(s) distributed over up to {} "
            "core(s)/worker(s).".format(len(self._spoints), no_workers)
        )

        rows = []
        for index, result in tqdm.tqdm(
            enumerate(results),
            desc="Scanning: ",
            unit=" spoint",
            total=len(self._spoints),
            ncols=min(100, shutil.get_terminal_size((80, 20)).columns)
        ):
            md = self.md["dfunction"]
            if "nbins" not in md:
                md["nbins"] = len(result) - 1

            coeff_values = list(self._spoints[index].wc.values.values())
            rows.append([*coeff_values, *result])

        # Wait for completion of all jobs here
        pool.join()

        self.log.debug("Converting data to pandas dataframe.")
        # todo: check that there isn't any trouble with sorting.
        cols = self.md["spoints"]["coeffs"].copy()
        cols.extend([
            "bin{}".format(no_bin)
            for no_bin in range(self.md["dfunction"]["nbins"])
        ])

        # Now we finally write everything to data
        data.df = pd.DataFrame(data=rows, columns=cols)
        data.df.index.name = "index"
        data.md["scan"] = self.md

        self.log.info("Integration done.")

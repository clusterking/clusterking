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
import wilson
import tqdm

# ours
from bclustering.data.data import Data
import bclustering.maths.binning
from bclustering.util.metadata import git_info, failsafe_serialize, nested_dict
from bclustering.util.log import get_logger


class WpointCalculator(object):
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

    def calc(self, w: wilson.Wilson) -> np.array:
        """Calculates one point in wilson space.

        Args:
            w: Wilson coefficients

        Returns:
            np.array of the integration results
        """

        if self.dfunc_binning is not None:
            return bclustering.maths.binning.bin_function(
                functools.partial(self.dfunc, w, **self.dfunc_kwargs),
                self.dfunc_binning,
                normalize=self.dfunc_normalize
            )
        else:
            return self.dfunc(w, **self.dfunc_kwargs)


class Scanner(object):
    """ Scans the NP parameter space in a grid and also q2, producing the
    normalized q2 distribution.

    See bclustering.dfmd.DFMD for how to initialize this class from output
    files or existing instances.

    Usage example:

    .. code-block:: python

        import flavio
        from bclustering import Scanner, Data

        # Initialize Scanner object
        s = Scanner()
    
        # Sample 4 points for each of the 5 Wilson coefficients
        s.set_wpoints_equidist(
            {
                "CVL_bctaunutau": (-1, 1, 4),
                "CSL_bctaunutau": (-1, 1, 4),
                "CT_bctaunutau": (-1, 1, 4)
            },
            scale=5,
            eft='WET',
            basis='flavio'
        )
    
        # Set function and binning
        s.set_dfunction(
            functools.partial(flavio.np_prediction, "dBR/dq2(B+->Dtaunu)"),
            binning=np.linspace(3.15, bdlnu.q2max, 11.66),
            normalize=True
        )

        # Initialize a Data objects to write to
        d = Data()

        # Start running with maximally 3 cores and write the results to Data
        s.run(d)

    """

    # **************************************************************************
    # A:  Setup
    # **************************************************************************

    def __init__(self):
        self.log = get_logger("Scanner")

        #: Points in wilson space
        #:  Use self.wpoints to access this
        self._wpoints = []

        #: Instance of WpointCalculator to perform the claculations of
        #:  the wilson space points.
        self._wpoint_calculator = None  # type: WpointCalculator

        self.md = nested_dict()
        self.md["git"] = git_info(self.log)
        self.md["time"] = time.strftime(
            "%a %_d %b %Y %H:%M", time.gmtime()
        )

    # Write only access
    @property
    def wpoints(self) -> List[wilson.Wilson]:
        """ Points in wilson space that are sampled."""
        return self._wpoints

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
            func: A function that takes the wilson coefficient as the first
                argument. It should either return a float (if the binning
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

        # global wpoint_calculator
        self._wpoint_calculator = WpointCalculator(
            func, binning, normalize, kwargs
        )

    def set_wpoints_grid(self, values, scale, eft, basis) -> None:
        """ Set a grid of points in wilson space.

        Args:
            values: A dictionary of the following form:

                .. code-block:: python

                    {
                        <wilson coeff name>: [
                            value1,
                            value2,
                            ...
                        ]
                    }

            scale: Wilson coeff input scale in GeV
            eft: Wilson coeff input eft
            basis: Wilson coeff input basis
        """

        # Important to remember the order now, because of what we do next.
        # Dicts are NOT ordered
        coeffs = list(values.keys())
        # It's very important to sort the coefficient names here, because when
        # calling wilson.Wilson(...).wc.values() later, these will also
        # be alphabetically ordered.
        coeffs.sort()
        # Nowe we collect all lists of values.
        values_lists = [
            values[coeff] for coeff in coeffs
        ]
        # Now we build the cartesian product, i.e.
        # [a1, a2, ...] x [b1, b2, ...] x ... x [z1, z2, ...] =
        # [(a1, b1, ..., z1), ..., (a2, b2, ..., z2)]
        cartesians = list(itertools.product(*values_lists))

        # And build wilson coefficients from this
        self._wpoints = [
            wilson.Wilson(
                wcdict={
                    coeffs[icoeff]: cartesian[icoeff]
                    for icoeff in range(len(coeffs))
                },
                scale=scale,
                eft=eft,
                basis=basis
            )
            for cartesian in cartesians
        ]

        md = self.md["wpoints"]
        md["coeffs"] = list(values.keys())
        md["values"] = values
        md["scale"] = scale
        md["eft"] = eft
        md["basis"] = basis

    def set_wpoints_equidist(self, ranges, scale, eft, basis) -> None:
        """ Set a list of 'equidistant' points in wilson space.

        Args:
            ranges: A dictionary of the following form:

                .. code-block:: python

                    {
                        <wilson coeff name>: (
                            <Minimum of wilson coeff>,
                            <Maximum of wilson coeff>,
                            <Number of bins between min and max>,
                        )
                    }

            scale: <Wilson coeff input scale in GeV>,
            eft: <Wilson coeff input eft>,
            basis: <Wilson coeff input basis>

        Returns:
            None
        """

        grid_config = {
            coeff: list(np.linspace(*ranges[coeff]))
            for coeff in ranges
        }
        self.set_wpoints_grid(
            grid_config,
            scale=scale,
            eft=eft,
            basis=basis,
        )
        # Make sure to do this after set_wpoints_grid, so we overwrite
        # the relevant parts.
        md = self.md["wpoints"]
        md["sampling"] = "equidistant"
        md["ranges"] = ranges

    # **************************************************************************
    # B:  Run
    # **************************************************************************

    def run(self, data: Data, no_workers=None) -> None:
        """Calculate all wilson points in parallel and saves the result in
        self.df.

        Args:
            data: Data object.
            no_workers: Number of worker nodes/cores. Default: Total number of
                cores.
        """

        if not self._wpoints:
            self.log.error(
                "No wilson points specified. Returning without doing "
                "anything."
            )
            return
        if not self._wpoint_calculator:
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
        worker = self._wpoint_calculator.calc

        results = pool.imap(worker, self._wpoints)

        # close the queue for new jobs
        pool.close()

        self.log.info(
            "Started queue with {} job(s) distributed over up to {} "
            "core(s)/worker(s).".format(len(self._wpoints), no_workers)
        )

        rows = []
        for index, result in tqdm.tqdm(
            enumerate(results),
            desc="Scanning: ",
            unit=" wpoint",
            total=len(self._wpoints),
            ncols=min(100, shutil.get_terminal_size((80, 20)).columns)
        ):
            md = self.md["dfunction"]
            if "nbins" not in md:
                md["nbins"] = len(result) - 1

            coeff_values = list(self._wpoints[index].wc.values.values())
            rows.append([*coeff_values, *result])

        # Wait for completion of all jobs here
        pool.join()

        self.log.debug("Converting data to pandas dataframe.")
        # todo: check that there isn't any trouble with sorting.
        cols = self.md["wpoints"]["coeffs"].copy()
        cols.extend([
            "bin{}".format(no_bin)
            for no_bin in range(self.md["dfunction"]["nbins"])
        ])

        # Now we finally write everything to data
        data.df = pd.DataFrame(data=rows, columns=cols)
        data.df.index.name = "index"
        data.md["scan"] = self.md

        self.log.info("Integration done.")

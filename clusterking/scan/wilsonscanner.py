#!/usr/bin/env python3

# 3rd
import wilson
import numpy as np
import itertools

# ours
from clusterking.scan.scanner import Scanner


class WilsonScanner(Scanner):
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
        s.set_spoints_equidist(
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
    def __init__(self):
        super().__init__()

    def set_spoints_grid(self, values, scale, eft, basis) -> None:
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
        self._spoints = [
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

        md = self.md["spoints"]
        md["coeffs"] = list(values.keys())
        md["values"] = values
        md["scale"] = scale
        md["eft"] = eft
        md["basis"] = basis

    def set_spoints_equidist(self, ranges, scale, eft, basis) -> None:
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
        self.set_spoints_grid(
            grid_config,
            scale=scale,
            eft=eft,
            basis=basis,
        )
        # Make sure to do this after set_spoints_grid, so we overwrite
        # the relevant parts.
        md = self.md["spoints"]
        md["sampling"] = "equidistant"
        md["ranges"] = ranges

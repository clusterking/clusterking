#!/usr/bin/env python3

# 3rd
import wilson

# ours
from clusterking.scan.scanner import Scanner, SpointCalculator


class WpointCalculator(SpointCalculator):
    """ A class that holds the function with which we calculate each
    point in wilson space.
    """
    def __init__(self):
        super().__init__()
        # All of these have to be set to work!
        self.coeffs = None
        self.scale = None
        self.eft = None
        self.basis = None

    def _prepare_spoint(self, spoint):
        return wilson.Wilson(
            wcdict={
                self.coeffs[icoeff]: spoint[icoeff]
                for icoeff in range(len(self.coeffs))
            },
            scale=self.scale,
            eft=self.eft,
            basis=self.basis
        )


class WilsonScanner(Scanner):
    """ Scans the NP parameter space in a grid and also in the kinematic
    variable.

    Usage example:

    .. code-block:: python

        import flavio
        import functools
        import numpy as np
        import clusterking as ck

        # Initialize Scanner object
        s = ck.scan.WilsonScanner(scale=5, eft='WET', basis='flavio')

        # Sample 4 points for each of the 5 Wilson coefficients
        s.set_spoints_equidist(
            {
                "CVL_bctaunutau": (-1, 1, 4),
                "CSL_bctaunutau": (-1, 1, 4),
                "CT_bctaunutau": (-1, 1, 4)
            }
        )

        # Set function and binning
        s.set_dfunction(
            functools.partial(flavio.np_prediction, "dBR/dq2(B+->Dtaunu)"),
            binning=np.linspace(3.15, 11.66, 10),
            normalize=True
        )

        # Initialize a Data objects to write to
        d = ck.Data()

        # Start running with maximally 3 cores and write the results to Data
        s.run(d)

    """
    def __init__(self, scale, eft, basis):
        """ Initializes the :class:`clusterking.scan.WilsonScanner` class.

        Args:
            scale: Wilson coeff input scale in GeV
            eft: Wilson coeff input eft
            basis: Wilson coeff input basis

        .. note::

            A list of applicable bases and EFTs can be found at
            https://wcxf.github.io/bases.html
        """
        super().__init__()
        self._set_wilson_format(scale, eft, basis)
        self._spoint_calculator = WpointCalculator()

    def _set_wilson_format(self, scale, eft, basis):
        """ Set scale, eft and basis of input wilson coefficients

        Args:
            scale: Wilson coeff input scale in GeV
            eft: Wilson coeff input eft
            basis: Wilson coeff input basis
        """
        self.md["spoints"]["scale"] = scale
        self.md["spoints"]["eft"] = eft
        self.md["spoints"]["basis"] = basis

    def set_dfunction(self, *args, **kwargs):
        super().set_dfunction(*args, **kwargs)
        self._spoint_calculator.coeffs = self.coeffs
        self._spoint_calculator.scale = self.scale
        self._spoint_calculator.eft = self.eft
        self._spoint_calculator.basis = self.basis

    def set_spoints_grid(self, *args, **kwargs):
        super().set_spoints_grid(*args, **kwargs)
        self._spoint_calculator.coeffs = self.coeffs

    @property
    def scale(self):
        """ Scale of the input wilson coefficients in GeV (read-only). """
        return self.md["spoints"]["scale"]

    @property
    def eft(self):
        """  Wilson coefficient input EFT (read-only) """
        return self.md["spoints"]["eft"]

    @property
    def basis(self):
        """ Wilson coefficient input basis (read-only)"""
        return self.md["spoints"]["basis"]

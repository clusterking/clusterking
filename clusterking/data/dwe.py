#!/usr/bin/env python3

# 3rd
import numpy as np

# ours
from clusterking.data.data import Data
from clusterking.maths.statistics import cov2err, cov2corr, abs2rel_cov, \
    ensure_array, corr2cov


class DataWithErrors(Data):
    def __init__(self, *args, **kwargs):
        """
        This class gets initialized with an array of n x nbins data points,
        corresponding to n histograms with nbins bins.

        Methods offer convenient and performant ways to add errors to this
        dataset.

        Args:
            data: n x nbins matrix
        """
        super().__init__(*args, **kwargs)

        # Initialize some values to their default
        self.rel_cov = None
        self.abs_cov = None
        self.poisson_errors = False

    # **************************************************************************
    # A: Additional shortcuts
    # **************************************************************************

    @property
    def rel_cov(self):
        value = self.md["errors"]["rel_cov"]
        if value is None:
            if self.nbins:
                return np.zeros((self.nbins, self.nbins))
            else:
                return None
        return np.array(value)

    @rel_cov.setter
    def rel_cov(self, value):
        if isinstance(value, np.ndarray):
            # convert to list in order to serialize
            value = value.tolist()
        self.md["errors"]["rel_cov"] = value

    @property
    def abs_cov(self):
        value = self.md["errors"]["abs_cov"]
        if value is None:
            if self.nbins:
                return np.zeros((self.nbins, self.nbins))
            else:
                return None
        return np.array(value)

    @abs_cov.setter
    def abs_cov(self, value):
        if isinstance(value, np.ndarray):
            # convert to list in order to serialize
            value = value.tolist()
        self.md["errors"]["abs_cov"] = value

    @property
    def poisson_errors(self):
        return self.md["errors"]["poisson"]

    @poisson_errors.setter
    def poisson_errors(self, value):
        self.md["errors"]["poisson"] = value

    # **************************************************************************
    # B: Helper functions
    # **************************************************************************

    def _interpret_input(self, inpt, what: str) -> np.ndarray:
        """ Interpret user input

        Args:
            inpt: User input
            what: 'err', 'corr', 'cov'

        Returns:
            correctly shaped numpy array
        """
        inpt = ensure_array(inpt)
        if what.lower() in ["err"]:
            if inpt.ndim == 0:
                return np.tile(inpt, self.nbins)
            if inpt.ndim == 1:
                return inpt
            else:
                raise ValueError(
                    "Wrong dimension ({}) of {} array.".format(
                        inpt.ndim, what
                    )
                )
        elif what.lower() in ["corr", "cov"]:
            if inpt.ndim == 2:
                return inpt
            else:
                raise ValueError(
                    "Wrong dimension ({}) of {} array.".format(inpt.ndim, what)
                )
        else:
            raise ValueError("Unknown value what='{}'.".format(what))

    # **************************************************************************
    # C: Doing the actual calculations
    # **************************************************************************

    # Note: Overrides inherited method from data.
    def data(self, decorrelate=False, **kwargs):
        """ Return data matrix

        Args:
            decorrelate: Unrotate the correlation matrix to return uncorrelated
                data entries
            **kwargs: Any keyword argument to Data.data()

        Returns:
            self.n x self.nbins array
        """
        ret = super().data(**kwargs)

        if decorrelate:
            inverses = np.linalg.inv(self.corr())
            ret = np.einsum("kij,kj->ki", inverses, ret)

        return ret

    def cov(self, relative=False):
        """ Return covariance matrix

        Args:
            relative: "Relative to data", i.e. Cov_ij / (data_i * data_j)

        Returns:
            self.n x self.nbins x self.nbins array
        """

        data = self.data()
        cov = np.tile(self.abs_cov, (self.n, 1, 1))
        cov += np.tile(self.rel_cov, (self.n, 1, 1)) * data

        if not relative:
            return cov
        else:
            return abs2rel_cov(cov, data)

    def corr(self):
        """ Return correlation matrix

        Returns:
            self.n x self.nbins x self.nbins array
        """
        return cov2corr(self.cov())

    def err(self, relative=False):
        """ Return errors per bin

        Args:
            relative: Relative errors

        Returns:
            self.n x self.nbins array
        """
        if not relative:
            return cov2err(self.cov())
        else:
            return cov2err(self.cov()) / self.data()

    # **************************************************************************
    # D: Configuration
    # **************************************************************************

    # -------------------------------------------------------------------------
    # Add absolute errors
    # -------------------------------------------------------------------------

    def add_err_cov(self, cov) -> None:
        """ Add error from covariance matrix.

        Args:
            cov: self.n x self.nbins x self.nbins array of covariance matrices
                or self.nbins x self.nbins covariance matrix (if equal for
                all data points)
        """
        cov = self._interpret_input(cov, "cov")
        self.abs_cov += cov

    def add_err_corr(self, err, corr) -> None:
        """ Add error from errors vector and correlation matrix.

        Args:
            err: self.n x self.nbins vector of errors for each data point and
                bin or self.nbins vector of uniform errors per data point or
                float (uniform error per bin and datapoint)
            corr: self.n x self.nbins x self.nbins correlation matrices
                or self.nbins x self.nbins correlation matrix
        """
        err = self._interpret_input(err, "err")
        corr = self._interpret_input(corr, "corr")
        self.add_err_cov(corr2cov(corr, err))

    def add_err_uncorr(self, err) -> None:
        """
        Add uncorrelated error.

        Args:
            err: see argument of add_err_corr
        """
        err = self._interpret_input(err, "err")
        corr = np.identity(self.nbins)
        self.add_err_corr(err, corr)

    def add_err_maxcorr(self, err) -> None:
        """
        Add maximally correlated error.

        Args:
            err: see argument of add_err_corr
        """
        err = self._interpret_input(err, "err")
        corr = np.ones((self.nbins, self.nbins))
        self.add_err_corr(err, corr)

    # -------------------------------------------------------------------------
    # Add relative errors
    # -------------------------------------------------------------------------

    def add_rel_err_cov(self, cov: np.array) -> None:
        """
        Add error from "relative" covariance matrix

        Args:
            cov: see argument of add_err_cov
        """
        cov = self._interpret_input(cov, "cov")
        self.rel_cov += cov

    def add_rel_err_corr(self, err, corr) -> None:
        """
        Add error from relative errors and correlation matrix.

        Args:
            err: see argument of add_err_corr
            corr: see argument of add_err_corr
        """
        err = self._interpret_input(err, "err")
        corr = self._interpret_input(corr, "corr")
        self.add_rel_err_cov(corr2cov(corr, err))

    def add_rel_err_uncorr(self, err: np.array) -> None:
        """
        Add uncorrelated relative error.

        Args:
            err: see argument of add_err_corr
        """
        err = self._interpret_input(err, "err")
        corr = np.identity(self.nbins)
        self.add_rel_err_corr(err, corr)

    def add_rel_err_maxcorr(self, err: np.array) -> None:
        """
        Add maximally correlated relative error.

        Args:
            err: see argument of add_err_corr
        """
        err = self._interpret_input(err, "err")
        corr = np.ones((self.nbins, self.nbins))
        self.add_rel_err_corr(err, corr)

    # -------------------------------------------------------------------------
    # Other forms of errors
    # -------------------------------------------------------------------------

    def add_err_poisson(self):
        self.poisson_errors = True

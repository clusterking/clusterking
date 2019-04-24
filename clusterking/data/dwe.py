#!/usr/bin/env python3

# 3rd
import numpy as np

# ours
from clusterking.data.data import Data
from clusterking.maths.statistics import cov2err, cov2corr, abs2rel_cov, \
    corr2cov


class DataWithErrors(Data):
    """ This class extends the :py:class:`~clusterking.data.Data` class by
    convenient and performant ways to add errors to the distributions.

    See the description of the :py:class:`~clusterking.data.Data` class for more
    information about the data structure itself.

    There are three basic ways to add errors:

    1. Add relative errors (with correlation) relative to the bin content of
       each bin in the distribution: ``add_rel_err_...``
    2. Add absolute errors (with correlation): ``add_err_...``
    3. Add poisson errors:
       :py:meth:`.add_err_poisson`

    .. note::
        All of these methods, add the errors in a consistent way for all sample
        points/distributions, i.e. it is impossible to add a certain error
        specifically to one sample point only!

    Afterwards, you can get errors, correlation and covariance matrices for
    every data point by using one of the methods such as
    :meth:`.cov`, :meth:`.corr`, :meth:`err`.

    .. note::
        When saving your dataset, your error configuration is saved as well,
        so you can reload it like any other :class:`~clusterking.data.Data`
        or :class:`~clusterking.data.DFMD` object.

    Args:
        data: n x nbins matrix
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize some values to their default
        self.rel_cov = None
        self.abs_cov = None
        self.poisson_errors = 0

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
        inpt = np.array(inpt, float)
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
    def data(self, decorrelate=False, **kwargs) -> np.ndarray:
        """ Return data matrix

        Args:
            decorrelate: Unrotate the correlation matrix to return uncorrelated
                data entries
            **kwargs: Any keyword argument to
                :meth:`clusterking.data.Data.data()`

        Returns:
            self.n x self.nbins array
        """
        ret = super().data(**kwargs)

        if decorrelate:
            inverses = np.linalg.inv(self.corr())
            ret = np.einsum("kij,kj->ki", inverses, ret)

        return ret

    def cov(self, relative=False) -> np.ndarray:
        """ Return covariance matrix

        Args:
            relative: "Relative to data", i.e.
                :math:`\\mathrm{Cov}_{ij} / (\\mathrm{data}_i \cdot \\mathrm{data}_j)`

        Returns:
            self.n x self.nbins x self.nbins array
        """

        data = self.data()
        cov = np.tile(self.abs_cov, (self.n, 1, 1))
        cov += np.einsum("ij,ki,kj->kij", self.rel_cov, data, data)
        if self.poisson_errors:
            cov += corr2cov(
                np.tile(np.eye(self.nbins), (self.n, 1, 1)),
                np.sqrt(self.poisson_errors) * np.sqrt(data)
            )

        if not relative:
            return cov
        else:
            return abs2rel_cov(cov, data)

    def corr(self) -> np.ndarray:
        """ Return correlation matrix

        Returns:
            self.n x self.nbins x self.nbins array
        """
        return cov2corr(self.cov())

    def err(self, relative=False) -> np.ndarray:
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
            err: see argument of :py:meth:`.add_err_corr`
        """
        err = self._interpret_input(err, "err")
        corr = np.identity(self.nbins)
        self.add_err_corr(err, corr)

    def add_err_maxcorr(self, err) -> None:
        """
        Add maximally correlated error.

        Args:
            err: see argument of :py:meth:`.add_err_corr`
        """
        err = self._interpret_input(err, "err")
        corr = np.ones((self.nbins, self.nbins))
        self.add_err_corr(err, corr)

    # -------------------------------------------------------------------------
    # Add relative errors
    # -------------------------------------------------------------------------

    def add_rel_err_cov(self, cov) -> None:
        """
        Add error from "relative" covariance matrix

        Args:
            cov: see argument of :py:meth:`.add_err_cov`
        """
        cov = self._interpret_input(cov, "cov")
        self.rel_cov += cov

    def add_rel_err_corr(self, err, corr) -> None:
        """
        Add error from relative errors and correlation matrix.

        Args:
            err: see argument of :py:meth:`.add_err_corr`
            corr: see argument of :py:meth:`.add_err_corr`
        """
        err = self._interpret_input(err, "err")
        corr = self._interpret_input(corr, "corr")
        self.add_rel_err_cov(corr2cov(corr, err))

    def add_rel_err_uncorr(self, err) -> None:
        """
        Add uncorrelated relative error.

        Args:
            err: see argument of
                :py:meth:`.add_err_corr`
        """
        err = self._interpret_input(err, "err")
        corr = np.identity(self.nbins)
        self.add_rel_err_corr(err, corr)

    def add_rel_err_maxcorr(self, err) -> None:
        """
        Add maximally correlated relative error.

        Args:
            err: see argument of :py:meth:`.add_err_corr`
        """
        err = self._interpret_input(err, "err")
        corr = np.ones((self.nbins, self.nbins))
        self.add_rel_err_corr(err, corr)

    # -------------------------------------------------------------------------
    # Other forms of errors
    # -------------------------------------------------------------------------

    def add_err_poisson(self, normalization_scale=1) -> None:
        """
        Add poisson errors/statistical errors.

        Args:
            normalization_scale: Apply poisson errors corresponding to data
                normalization scaled up by this factor. For example, if your
                data is normalized to 1 and you still want to apply Poisson
                errors that correspond to a yield of 200, you can call
                ``add_err_poisson(200)``. Your data will stay normalized, but
                the poisson errors are appropriate for a total yield of 200.

        Returns:
            None
        """
        self.poisson_errors = normalization_scale

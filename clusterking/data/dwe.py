#!/usr/bin/env python3

# 3rd
import numpy as np
from typing import List, Optional, Dict, Any

# ours
from clusterking.data.data import Data
from clusterking.maths.statistics import (
    cov2err,
    cov2corr,
    abs2rel_cov,
    corr2cov,
)


class DataWithErrors(Data):
    """ This class extends the :py:class:`~clusterking.data.Data` class by
    convenient and performant ways to add errors to the distributions.

    See the description of the :py:class:`~clusterking.data.Data` class for more
    information about the data structure itself.

    There are three basic ways to add errors:

    1. Add relative errors (with correlation) relative to the bin content of
       each bin in the distribution: ``add_rel_err_...``
       (:math:`\\mathrm{Cov}^{(k)}_{\\text{rel}}(i, j)`)
    2. Add absolute errors (with correlation): ``add_err_...``
       (:math:`\\mathrm{Cov}^{(k)}_{\\text{abs}}(i, j)`)
    3. Add poisson errors:
       :py:meth:`.add_err_poisson`

    The covariance matrix for bin i and j of distribution n
    (with contents :math:`d^{(n)}_i`) will then
    be

    .. math::

        \\mathrm{Cov}(d^{(n)}_i, d^{(n)}_j)  =
                &\\sum_{k}\\mathrm{Cov}_{\\text{rel}}^{(k)}(i, j)
                \\cdot d^{(n)}_i d^{(n)}_j + \\\\
            + &\\sum_k\\mathrm{Cov}_{\\text{abs}}^{(k)}(i, j) + \\\\
            + &\\delta_{ij} \\sqrt{d^{(n)}_i d^{(n)}_j} / \\sqrt{s}

    .. note::
        All of these methods add the errors in a consistent way for all sample
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
        # [Defined as properties, documented below]
        self.rel_cov = None
        self.abs_cov = None
        self.poisson_errors = False
        self.poisson_errors_scale = 1.0

    # **************************************************************************
    # Properties
    # **************************************************************************

    @property
    def rel_cov(self):
        """ Relative covariance matrix that will be later applied to the data
        (see class documentation).

        .. math::

            \\mathrm{Cov}_{\\text{rel}}(i, j)
            = \\sum_k\\mathrm{Cov}_{\\text{rel}}^{(k)}(i, j)

        If no errors have been added, this is defined to be a zero matrix.

        Returns:
            ``self.nbins * self.nbins`` matrix
        """
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
        """ Absolute covariance matrix that will be later applied to the data
        (see class documentation).

        .. math::

            \\mathrm{Cov}_{\\text{abs}}(i, j)
            = \\sum_k\\mathrm{Cov}_{\\text{abs}}^{(k)}(i, j)

        If no errors have been added, this is defined to be a zero matrix.

        Returns:
            ``self.nbins * self.nbins`` matrix
        """
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
    def poisson_errors(self) -> bool:
        """ Should poisson errors be added? """
        return self.md["errors"]["poisson"]

    @poisson_errors.setter
    def poisson_errors(self, value):
        self.md["errors"]["poisson"] = value

    @property
    def poisson_errors_scale(self) -> float:
        """ Scale poisson errors. See documentation of :meth:`add_err_poisson`.
        """
        return self.md["errors"]["poisson_scale"]

    @poisson_errors_scale.setter
    def poisson_errors_scale(self, value):
        self.md["errors"]["poisson_scale"] = value

    # **************************************************************************
    # Internal helper functions
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
                    "Wrong dimension ({}) of {} array.".format(inpt.ndim, what)
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
    # Actual calculations
    # **************************************************************************

    def cov(self, relative=False) -> np.ndarray:
        """ Return covariance matrix :math:`\\mathrm{Cov}(d^{(n)}_i, d^{(n)}_j)`

        If no errors have been added, a zero matrix is returned.

        Args:
            relative: "Relative to data", i.e.
                :math:`\\mathrm{Cov}(d^{(n)}_i, d^{(n)}_j) /
                (d^{(n)}_i \\cdot d^{(n)}_j)`

        Returns:
            ``self.n x self.nbins x self.nbins`` array
        """

        data = self.data()
        cov = np.tile(self.abs_cov, (self.n, 1, 1))
        cov += np.einsum("ij,ki,kj->kij", self.rel_cov, data, data)
        if self.poisson_errors:
            cov += corr2cov(
                np.tile(np.eye(self.nbins), (self.n, 1, 1)),
                # Normal poisson errors are sqrt(data_i). What happens if
                # data is normalized from N to N', i.e.
                #   Sum(data_normalized) = N'?
                # Let N/N' = scale
                # Then the errors should be
                #   sqrt(data) / scale = sqrt(data/scale) / sqrt(scale) =
                #   = sqrt(data_normalized) / sqrt(scale)
                # Hence:
                np.sqrt(data) / np.sqrt(self.poisson_errors_scale),
            )

        if not relative:
            return cov
        else:
            return abs2rel_cov(cov, data)

    def corr(self) -> np.ndarray:
        """ Return correlation matrix. If covariance matrix is empty (because
        no errors have been added), a unit matrix is returned.

        Returns:
            ``self.n x self.nbins x self.nbins`` array
        """
        if np.sum(np.abs(self.cov())) == 0.0:
            return np.tile(np.eye(self.nbins), (self.n, 1, 1))
        return cov2corr(self.cov())

    def err(self, relative=False) -> np.ndarray:
        """ Return errors per bin, i.e.
        :math:`e_i^{(n)} = \\sqrt{\\mathrm{Cov}(d^{(n)}_i, d^{(n)}_i)}`

        Args:
            relative: Relative errors, i.e. :math:`e_i^{(n)}/d_i^{(n)}`

        Returns:
            ``self.n x self.nbins`` array
        """
        if not relative:
            return cov2err(self.cov())
        else:
            return cov2err(self.cov()) / self.data()

    # **************************************************************************
    # Configuration
    # **************************************************************************

    def reset_errors(self) -> None:
        """ Set all errors back to 0.

        Returns:
            None
        """
        self.rel_cov = None
        self.abs_cov = None
        self.poisson_errors = False
        self.poisson_errors_scale = 1

    # -------------------------------------------------------------------------
    # Add absolute errors
    # -------------------------------------------------------------------------

    def add_err_cov(self, cov) -> None:
        """ Add error from covariance matrix.

        Args:
            cov: ``self.n x self.nbins x self.nbins`` array of covariance
                matrices or self.nbins x self.nbins covariance matrix (if equal
                for all data points)
        """
        cov = self._interpret_input(cov, "cov")
        self.abs_cov += cov

    def add_err_corr(self, err, corr) -> None:
        """ Add error from errors vector and correlation matrix.

        Args:
            err: ``self.n x self.nbins`` vector of errors for each data point
                and bin or self.nbins vector of uniform errors per data point or
                float (uniform error per bin and datapoint)
            corr: ``self.n x self.nbins x self.nbins`` correlation matrices
                or ``self.nbins x self.nbins`` correlation matrix
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
        if self.poisson_errors:
            self.log.warning("Poisson errors had already been added before.")
            if normalization_scale != self.poisson_errors_scale:
                self.log.warning(
                    "However we CHANGED (not added to) the scaling of the "
                    "Poisson errors."
                )
        self.poisson_errors = True
        self.poisson_errors_scale = normalization_scale

    # **************************************************************************
    # Quick plots
    # **************************************************************************

    def plot_dist_err(
        self,
        cluster_column="cluster",
        bpoint_column="bpoint",
        title: Optional[str] = None,
        clusters: Optional[List[int]] = None,
        bpoints=True,
        legend=True,
        hist_kwargs: Optional[Dict[str, Any]] = None,
        hist_fill_kwargs: Optional[Dict[str, Any]] = None,
        ax=None,
    ):
        """Plot distribution with errors.

        Args:
            cluster_column: Column with the cluster names (default 'cluster')
            bpoint_column: Column with bpoints (default 'bpoint')
            title: Plot title (``None``: automatic)
            clusters: List of clusters to selected or single cluster.
                If ``None`` (default), all clusters are chosen.
            bpoints: Draw benchmark points if available (default True). If
                false or not benchmark points are available, pick a random
                sample point for each cluster.
            legend: Draw legend? (default True)
            hist_kwargs: Keyword arguments to
                :meth:`~clusterking.plots.plot_histogram.plot_histogram`
            hist_fill_kwargs: Keyword arguments to
                :meth:`~clusterking.plots.plot_histogram.plot_histogram_fill`
            ax: Instance of `matplotlib.axes.Axes` to plot on. If ``None``, a
                new one is instantiated.

        Note: To customize these kind of plots further, check the
        :py:class:`~clusterking.plots.plot_bundles.BundlePlot` class and the
        :py:meth:`~clusterking.plots.plot_bundles.BundlePlot.err_plot`
        method thereof.

        Returns:
            Figure
        """
        from clusterking.plots.plot_bundles import BundlePlot

        bp = BundlePlot(self)
        bp.cluster_column = cluster_column
        bp.bpoint_column = bpoint_column
        bp.title = title
        bp.draw_legend = legend
        bp.err_plot(
            clusters=clusters,
            bpoints=bpoints,
            hist_kwargs=hist_kwargs,
            hist_fill_kwargs=hist_fill_kwargs,
            ax=ax,
        )
        return bp.fig

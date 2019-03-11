#!/usr/bin/env python3

# from .histogram import Histogram

import numpy as np


def ensure_array(x):
    if not isinstance(x, np.ndarray):
        return np.array(x)
    else:
        return x

def cov2err(cov):
    """ Convert covariance matrix (or array of covariance matrices of equal
    shape) to error array (or array thereof).

    Args:
        cov: [n x ] nbins x nbins array

    Returns
        [n x ] nbins array
    """
    cov = ensure_array(cov)
    if cov.ndim == 2:
        return np.sqrt(cov.diagonal())
    elif cov.ndim == 3:
        return np.sqrt(cov.diagonal(axis1=1, axis2=2))
    else:
        raise ValueError("Wrong dimensions.")


def cov2corr(cov):
    """ Convert covariance matrix (or array of covariance matrices of equal
    shape) to correlation matrix (or array thereof).

    Args:
        cov: [n x ] nbins x nbins array

    Returns
        [n x ] nbins x nbins array
    """
    cov = ensure_array(cov)
    err = cov2err(cov)
    if cov.ndim == 2:
        return cov / np.outer(err, err)
    elif cov.ndim == 3:
        return cov / np.einsum("ki,kj->kij", err, err)
    else:
        raise ValueError("Wrong dimensions")


def corr2cov(corr, err):
    """ Convert correlation matrix (or array of covariance matrices of equal
    shape) together with error array (or array thereof) to covariance
    matrix (or array thereof).

    Args:
        corr: [n x ] nbins x nbins array
        err: [n x ] nbins array

    Returns
        [n x ] nbins x nbins array
    """
    corr = ensure_array(corr)
    err = ensure_array(err)
    if corr.ndim == 2:
        return np.einsum("ij,i,j->ij", corr, err, err)
    elif corr.ndim == 3:
        return np.einsum("kij,ki,kj->kij", corr, err, err)
    else:
        raise ValueError("Wrong dimensions")


def rel2abs_cov(cov, data):
    """ Convert relative covariance matrix to absolute covariance matrix

    Args:
        cov: n x nbins x nbins array
        data: n x nbins array

    Returns:
        n x nbins x nbins array
    """
    cov = ensure_array(cov)
    data = ensure_array(data)
    assert cov.ndim == data.ndim + 1
    if data.ndim == 1:
        return np.einsum("ij,i,j->ij", cov, data, data)
    elif data.ndim == 2:
        return np.einsum("kij,ki,kj->kij", cov, data, data)
    else:
        raise ValueError("Wrong dimensions")


def abs2rel_cov(cov, data):
    """ Convert covariance matrix to relative covariance matrix

    Args:
        cov: n x nbins x nbins array
        data: n x nbins array

    Returns:
        n x nbins x nbins array
    """
    cov = ensure_array(cov)
    data = ensure_array(data)
    assert cov.ndim == data.ndim + 1
    if data.ndim == 1:
        nbins = len(data)
        return cov / data.reshape((nbins, 1)) / data.reshape((1, nbins))
    elif data.ndim == 2:
        n, nbins = data.shape
        return cov / data.reshape((n, nbins, 1)) / data.reshape((n, 1, nbins))
    else:
        raise ValueError("Wrong dimensions")



# todo: add metadata?
class DataWithErrors(object):
    def __init__(self, data):
        """
        This class gets initialized with an array of n x nbins data points,
        corresponding to n histograms with nbins bins.

        Methods offer convenient and performant ways to add errors to this
        dataset.

        Args:
            data: n x nbins matrix
        """
        #: A self.n x self.nbins array
        self._data = ensure_array(data).astype(float)
        self.n, self.nbins = self._data.shape
        self._cov = np.zeros((self.n, self.nbins, self.nbins))

    def norms(self):
        """ Return the histogram
        normalizations.

        Returns:
            array of length self.n
        """
        return np.sum(self._data, axis=1)

    def data(self, normalize=False, decorrelate=False):
        """ Return data matrix

        Args:
            normalize: Normalize data before returning it
            decorrelate: Unrotate the correlation matrix to return uncorrelated
                data entries

        Returns:
            self.n x self.nbins array
        """
        ret = np.array(self._data)
        if decorrelate:
            inverses = np.linalg.inv(self.corr())
            ret = np.einsum("kij,kj->ki", inverses, ret)
        # todo: does that work after decorrelate as well?
        if normalize:
            ret /= self.norms().reshape((self.n, 1))
        return ret

    def cov(self, relative=False):
        """ Return covariance matrix

        Args:
            relative: "Relative to data", i.e. Cov_ij / (data_i * data_j)

        Returns:
            self.n x self.nbins x self.nbins array
        """
        if not relative:
            return self._cov
        else:
            return abs2rel_cov(self._cov, self._data)

    def corr(self):
        """ Return correlation matrix

        Returns:
            self.n x self.nbins x self.nbins array
        """
        return cov2corr(self._cov)

    def err(self, relative=False):
        """ Return errors per bin

        Args:
            relative: Relative errors

        Returns:
            self.n x self.nbins array
        """
        if not relative:
            return cov2err(self._cov)
        else:
            return cov2err(self._cov) / self.norms().reshape((self.n, 1))

    # -------------------------------------------------------------------------

    def add_err_cov(self, cov: np.array) -> None:
        """ Add error from covariance matrix.

        Args:
            cov: self.n x self.nbins x self.nbins array of covariance matrices
                or self.nbins x self.nbins covariance matrix (if equal for
                all data points)
        """
        if not isinstance(cov, np.ndarray):
            cov = np.array(cov)
        if len(cov.shape) == 2:
            self._cov += np.tile(cov, (self.n, 1, 1))
        elif len(cov.shape) == 3:
            self._cov += cov
        else:
            raise ValueError("Wrong dimensionality of covariance matrix.")

    def add_err_corr(self, err: np.array, corr: np.array) -> None:
        """ Add error from errors vector and correlation matrix.

        Args:
            err: self.n x self.nbins vector of errors for each data point and
                bin or self.nbins vector of uniform errors per data point or
                float (uniform error per bin and datapoint)
            corr: self.n x self.nbins x self.nbins correlation matrices
                or self.nbins x self.nbins correlation matrix
        """
        if not isinstance(err, np.ndarray):
            err = np.array(err)
        if not isinstance(corr, np.ndarray):
            corr = np.array(corr)

        if len(err.shape) == 0:
            err = np.tile(err, (self.n, self.nbins))
        if len(err.shape) == 1:
            err = np.tile(err, (self.n, 1))
        elif len(err.shape) == 2:
            pass
        else:
            raise ValueError("Wrong dimension of error array.")

        if len(corr.shape) == 2:
            corr = np.tile(corr, (self.n, 1, 1))
        elif len(corr.shape) == 3:
            pass
        else:
            raise ValueError("Wrong dimension of correlation matrix")

        self.add_err_cov(corr2cov(corr, err))

    def add_err_uncorr(self, err: np.array) -> None:
        """
        Add uncorrelated error.

        Args:
            err: see argument of add_err_corr
        """
        corr = np.tile(np.identity(self.nbins), (self.n, 1, 1))
        self.add_err_corr(err, corr)

    def add_err_maxcorr(self, err) -> None:
        """
        Add maximally correlated error.

        Args:
            err: see argument of add_err_corr
        """
        corr = np.ones(self.n, self.nbins, self.nbins)
        self.add_err_corr(err, corr)

    # -------------------------------------------------------------------------

    def add_rel_err_cov(self, cov: np.array) -> None:
        """
        Add error from "relative" covariance matrix

        Args:
            cov: see argument of add_err_cov
        """
        self.add_err_cov(rel2abs_cov(cov, self._data))

    def add_rel_err_corr(self, err: np.array, corr: np.array) -> None:
        """
        Add error from relative errors and correlation matrix.

        Args:
            err: see argument of add_err_corr
            corr: see argument of add_err_corr
        """
        self.add_rel_err_cov(corr2cov(corr, err))

    def add_rel_err_uncorr(self, err: np.array) -> None:
        """
        Add uncorrelated relative error.

        Args:
            err: see argument of add_err_corr
        """
        corr = np.identity(self.nbins)
        self.add_rel_err_corr(err, corr)

    def add_rel_err_maxcorr(self, err: np.array) -> None:
        """
        Add maximally correlated relative error.

        Args:
            err: see argument of add_err_corr
        """
        corr = np.ones((self.nbins, self.nbins))
        self.add_rel_err_corr(err, corr)

    # -------------------------------------------------------------------------

    def add_poisson_error(self):
        self.add_err_uncorr(np.sqrt(self._data))


def chi2_metric(dwe: DataWithErrors):
    """
    Returns the chi2/ndf values of the comparison of a datasets.

    Args:
        dwe:

    Returns:

    """
    # https://root.cern.ch/doc/master/classTH1.html#a6c281eebc0c0a848e7a0d620425090a5

    # todo: in principle this could still be a factor of 2 faster, because we only need the upper triangular matrix

    # n vector
    n = dwe.norms()  # todo: this stays untouched by decorrelation, right?
    # n x nbins
    d = dwe.data(decorrelate=True)
    # n x nbins
    e = dwe.err()

    # n x n x nbins
    nom1 = np.einsum("k,li->kli", n, d)
    nom2 = np.transpose(nom1, (1, 0, 2))
    nominator = np.square(nom1 - nom2)

    # n x n x nbins
    den1 = np.einsum("k,li->kli", n, e)
    den2 = np.transpose(den1, (1, 0, 2))
    denominator = np.square(den1) + np.square(den2)

    # n x n x nbins
    summand = nominator / denominator

    # n x n
    chi2 = np.einsum("kli->kl", summand)

    return chi2 / dwe.nbins


def condense_distance_matrix(matrix):
    return matrix[np.triu_indices(len(matrix), k=1)]


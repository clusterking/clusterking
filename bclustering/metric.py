#!/usr/bin/env python3

# from .histogram import Histogram

import numpy as np


def cov2err(cov):
    """ Convert covariance matrix (or array of covariance matrices of equal
    shape) to error array (or array thereof).

    Args:
        cov: [n x ] nbins x nbins array

    Returns
        [n x ] nbins array
    """
    if len(cov.shape) == 2:
        return np.sqrt(cov.diagonal())
    elif len(cov.shape) == 3:
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
    err = cov2err(cov)
    if len(cov.shape) == 2:
        return cov / np.outer(err, err)
    elif len(cov.shape) == 3:
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
    if len(corr.shape) == 2:
        return np.einsum("ij,i,j->ij", corr, err, err)
    elif len(corr.shape) == 3:
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
    return np.einsum("ij,ki,kj->kij", cov, data, data)


def abs2rel_cov(cov, data):
    raise NotImplementedError


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
        self._data = data
        self.n, self.nbins = self._data.shape
        self._cov = np.zeros((self.n, self.nbins, self.nbins))

    def norms(self):
        """ Return the n vector of sums of bin contents, alias the histogram
        normalizations.
        """
        return np.sum(self._data, axis=1)

    def data(self, normalize=False, decorrelate=False):
        ret = np.array(self._data)
        if normalize:
            ret /= self.norms()
        if decorrelate:
            inverses = np.linalg.inv(self.corr())
            ret = np.einsum("kij,kj->ki", inverses, ret)
        return ret

    def normalized(self):
        return self._data / self.norms()

    def cov(self, relative=False):
        """ self.n x self.nbins x self.nbins array of covariance matrices """
        if not relative:
            return self._cov
        else:
            return abs2rel_cov(self._cov, self.data)

    def corr(self):
        """ self.n x self.nbins x self.nbins array of correlation matrices """
        return cov2corr(self._cov)

    def err(self, relative=False):
        """ self.n x self.nbins array of errors """
        if not relative:
            return cov2err(self._cov)
        else:
            return cov2err(self._cov) / self.norms()

    # -------------------------------------------------------------------------

    def add_err_cov(self, cov):
        """ Add error from self.n x self.nbins x self.nbins covariance
        matrix. """
        self._cov += cov

    def add_err_uncorr(self, err):
        """
        Add uncorrelated error.

        Args:
            err: A self.n x self.nbins array

        Returns:
            None
        """
        corr = np.identity((self.n, self.nbins, self.nbins))
        self.add_err_corr(err, corr)

    def add_err_corr(self, err, corr):
        self.add_err_cov(corr2cov(corr, err))

    # -------------------------------------------------------------------------

    def add_rel_err_cov(self, cov):
        """
        Add relative error from covariance matrix

        Args:
            cov: self.nbins x self.nbins array

        Returns:
            None
        """
        self.add_err_cov(rel2abs_cov(cov, self._data))

    def add_rel_err_uncorr(self, err):
        corr = np.identity((self.nbins, self.nbins))
        self.add_rel_err_corr(err, corr)

    def add_rel_err_corr(self, err, corr):
        self.add_rel_err_cov(corr2cov(corr, err))

    # -------------------------------------------------------------------------

    def add_poisson_error(self, norm):
        pass


def chi2_metric(dwe1: DataWithErrors):
    """
    Returns the chi2 values of the comparison of a datasets.

    Args:
        dwe1:

    Returns:

    """
    # https://root.cern.ch/doc/master/classTH1.html#a6c281eebc0c0a848e7a0d620425090a5

    # todo: in principle this could still be a factor of 2 faster, because we only need the upper triangular matrix

    # n vector
    n = dwe1.norms()  # todo: this stays untouched by decorrelation, right?
    # n x nbins
    d = dwe1.data(decorrelate=True)
    # n x nbins
    e = dwe1.err()

    # n x n x nbins
    nom1 = np.einsum("k,li->kli", n, d)
    nom2 = np.transpose(nom1, (0, 1))
    nominator = np.square(nom1 - nom2)

    # n x n x nbins
    den1 = np.einsum("k,li->kli", n, e)
    den2 = np.transpose(den1, (0, 1))
    denominator = np.square(den1) + np.square(den2)

    # n x n x nbins
    summand = nominator / denominator

    # n x n
    chi2 = np.einsum("kli->kl", summand)

    return chi2

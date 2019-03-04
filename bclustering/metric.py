#!/usr/bin/env python3

# from .histogram import Histogram

import numpy as np


def cov2err(cov):
    if len(cov.shape) == 2:
        return np.sqrt(cov.diagonal())
    elif len(cov.shape) == 3:
        return np.sqrt(cov.diagonal(axis1=1, axis2=2))
    else:
        raise ValueError("Wrong dimensions.")


def cov2corr(cov):
    err = cov2err(cov)
    if len(cov.shape) == 2:
        return cov / np.outer(err, err)
    elif len(cov.shape) == 3:
        return cov / np.einsum("ki,kj->kij", err, err)
    else:
        raise ValueError("Wrong dimensions")


def corr2cov(corr, err):
    if len(corr.shape) == 2:
        return np.einsum("ij,i,j->ij", corr, err, err)
    elif len(corr.shape) == 3:
        return np.einsum("kij,ki,kj->kij", corr, err, err)
    else:
        raise ValueError("Wrong dimensions")


def rel2abs_err(err, data):
    return err * data


def rel2abs_cov(cov, data):
    return np.einsum("kij,ki,kj->kij", cov, data, data)


class DataWithErrors(object):
    def __init__(self, data):
        #: A self.n x self.nbins array
        self.data = data
        self.n, self.nbins = self.data.shape
        self._cov = np.zeros((self.n, self.nbins, self.nbins))

    def cov(self):
        return self._cov

    def corr(self):
        return cov2corr(self._cov)

    def err(self):
        return cov2err(self._cov)

    # -------------------------------------------------------------------------

    def add_err_cov(self, cov):
        self._cov += cov

    def add_err_uncorr(self, err):
        """
        Add uncorrelated error.

        Args:
            err: A self.n x self.nbins array

        Returns:
            None
        """
        corr = np.ones((self.n, self.nbins, self.nbins))
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
        self.add_err_cov(rel2abs_cov(cov, self.data))

    def add_rel_err_uncorr(self, err):
        corr = np.ones((self.nbins, self.nbins))
        self.add_rel_err_corr(err, corr)

    def add_rel_err_corr(self, err, corr):
        self.add_err_cov(corr2cov(corr, err))

    # -------------------------------------------------------------------------

    def add_poisson_error(self, norm):
        pass

    # -------------------------------------------------------------------------

    def get_uncorr_data(self):
        # return histogram with uncorrelated bins
        raise NotImplementedError

        # return unrot, err

    def normalize(self):
        # normalize histogram
        raise NotImplementedError


def chi2_comparison(dwe1, dwe2):
    pass

#!/usr/bin/env python3

import numpy as np


def cov2err(cov):
    # sqrt(covii)
    raise NotImplementedError


def cov2corr(cov):
    # err = cov2err
    # corr = covij / (erri * errj)
    raise NotImplementedError

def corr2cov(corr, err):
    # corrij * erri * errj
    raise NotImplementedError


class Histogram(object):
    def __init__(self, nbins):
        self.content = np.zeros(nbins)
        self.covariance = np.ones((nbins, nbins))

    def add_err_cov(self, cove):
        raise NotImplementedError

    def add_err_uncorr(self, errFalse):
        # cov = np.ones(...)
        # self.add_err_cov(.....)
        raise NotImplementedError
    # def

    def add_err_corr(self, err, cor):
        raise NotImplementedError

    def get_uncorr_hist(self):
        # return histogram with uncorrelated bins
        raise NotImplementedError

    def normalize(self):
        # normalize histogram
        raise NotImplementedError


def joint_hist(content_a, content_b, cov):
    """ Return joint histogram with proper """
    raise NotImplementedError


def combine_hist(hist_a, hist_b, cov_ab=None):
    """ """
    if not cov_ab:
        # cov_ab = np.ones(....)
        pass
    raise NotImplementedError
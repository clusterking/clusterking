#!/usr/bin/env python3

# 3rd
import numpy as np


def cov2err(cov):
    """ Convert covariance matrix (or array of covariance matrices of equal
    shape) to error array (or array thereof).

    Args:
        cov: [n x ] nbins x nbins array

    Returns
        [n x ] nbins array
    """
    cov = np.array(cov)
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
    cov = np.array(cov)
    err = cov2err(cov)
    if cov.ndim == 2:
        return cov / np.einsum("i,j->ij", err, err)
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
    corr = np.array(corr)
    err = np.array(err)
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
    cov = np.array(cov)
    data = np.array(data)
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
    cov = np.array(cov)
    data = np.array(data)
    assert cov.ndim == data.ndim + 1
    if data.ndim == 1:
        nbins = len(data)
        return cov / data.reshape((nbins, 1)) / data.reshape((1, nbins))
    elif data.ndim == 2:
        n, nbins = data.shape
        return cov / data.reshape((n, nbins, 1)) / data.reshape((n, 1, nbins))
    else:
        raise ValueError("Wrong dimensions")

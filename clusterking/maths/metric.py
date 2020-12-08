#!/usr/bin/env python3

# 3rd
import numpy as np

# ours
from clusterking.maths.metric_utils import condense_distance_matrix
from clusterking.data.dwe import DataWithErrors


def chi2(
    n1: np.ndarray,
    n2: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
    normalize=False,
) -> np.ndarray:
    """

    Args:
        n1: n_obs x n_bins
        n2: Either n_obs x n_bins or just nbins if we're testing against a constant histogram
        cov1: Either n_obs x n_bins x n_bins or n_bins x n_bins
        cov2: Either n_obs x n_bins x n_bins or n_bins x n_bins
        normalize:

    Returns:
        n_obs vector of chi2 test results (degrees of freedom not yet divided out)
    """
    assert n1.ndim == 2
    n_obs, n_bins = n1.shape
    if n2.shape == (n_obs, n_bins):
        pass
    elif n2.shape == (n_bins,):
        n2 = n2.reshape((1, n_bins))
    else:
        raise ValueError("Invalid shape of n2: {}.".format(n2.shape))
    for _cov in [cov1, cov2]:
        if _cov.shape == (n_obs, n_bins, n_bins):
            pass
        elif _cov.shape == (n_bins, n_bins):
            pass
        else:
            raise ValueError(
                "Invalid shape of covariance matrix: {}".format(_cov.shape)
            )
    if normalize:
        if cov1.ndim == 2:
            cov1 = np.tile(cov1, (n_obs, 1, 1))
        if cov2.ndim == 2:
            cov2 = np.tile(cov2, (n_obs, 1, 1))
        norm1 = n1.sum(axis=1)
        norm2 = n2.sum(axis=1)
        n1 = n1.copy() / norm1.reshape((norm1.size, 1))
        n2 = n2.copy() / norm2.reshape((norm2.size, 1))
        cov1 = cov1.copy() / np.square(norm1).reshape((norm1.size, 1, 1))
        cov2 = cov2.copy() / np.square(norm2).reshape((norm2.size, 1, 1))
    diff = n1 - n2
    cov = cov1 + cov2
    if cov.ndim == 3:
        return np.einsum("ni,nij,nj->n", diff, np.linalg.inv(cov), diff)
    elif cov.ndim == 2:
        return np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov), diff)
    else:
        raise ValueError(
            "Invalid dimensionality of covariance matrix."
            " This is likely a bug in the package. Please"
            " report it."
        )


# todo: unittest
def chi2_metric(dwe: DataWithErrors, output="condensed"):
    """
    Returns the chi2/ndf values of the comparison of a datasets.

    Args:
        dwe: :py:class:`clusterking.data.dwe.DataWithErrors` object
        output: 'condensed' (condensed distance matrix) or 'full' (full distance
            matrix)

    Returns:
        Condensed distance matrix or full distance matrix

    """
    if not isinstance(dwe, DataWithErrors):
        raise TypeError(
            "In order to use chi2 metric, you have to use a DataWithErrors "
            "object with added errors, however you supplied an object of type "
            "{type}. ".format(type=type(dwe))
        )

    d = dwe.data()
    n_obs, n_bins = d.shape

    cov = dwe.cov(relative=False)
    assert cov.shape == (n_obs, n_bins, n_bins)

    # n x n
    chi2s = np.full((n_obs, n_obs), np.nan)
    # todo: this calculates the full n x n matrix, even though it's symmetric
    #    so we could likely optimize this if we wanted
    for i in range(n_obs):
        chi2s[i, :] = chi2(d, d[i], cov, cov[i], normalize=True)

    # todo: check for symmetry and vanishing diagonal of matrix here

    ndf = n_bins - 1
    chi2ndf = chi2s / np.full((1, 1), ndf)

    if output == "condensed":
        return condense_distance_matrix(chi2ndf)
    elif output == "full":
        return chi2ndf
    else:
        raise ValueError("Unknown argument '{}'.".format(output))

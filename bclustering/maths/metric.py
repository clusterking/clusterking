#!/usr/bin/env python3

# 3rd
import numpy as np

# ours
from bclustering.data.dwe import DataWithErrors


# todo: unittest
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


# todo: unittest
def condense_distance_matrix(matrix):
    return matrix[np.triu_indices(len(matrix), k=1)]

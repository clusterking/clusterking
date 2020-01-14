#!/usr/bin/env python3

# 3rd
import numpy as np

# ours
from clusterking.maths.metric_utils import *
from clusterking.data.dwe import DataWithErrors


# todo: unittest
def chi2_metric(dwe: DataWithErrors, output="condensed"):
    """
    Returns the chi2/ndf values of the comparison of a datasets.

    Args:
        dwe: :py:class:`clusterking.data.dwe.DataWithErrors` object
        output: 'condensed' (condensed distance matrix) or 'full' (full distance
            matrix)

    Returns:
        Condensed distance matrix

    """
    # If people try to use chi2 metric with a mere Data object, the decorrelate
    # option doesn't exist and we'd get an ununderstandable error, so we rather
    # cache this here.
    if not isinstance(dwe, DataWithErrors):
        raise TypeError(
            "In order to use chi2 metric, you have to use a DataWithErrors "
            "object with added errors, however you supplied an object of type "
            "{type}. ".format(type=type(dwe))
        )

    # https://root.cern.ch/doc/master/classTH1.html#a6c281eebc0c0a848e7a0d620425090a5

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
    chi2ndf = np.einsum("kli->kl", summand) / dwe.nbins

    if output == "condensed":
        return condense_distance_matrix(chi2ndf)
    elif output == "full":
        return chi2ndf
    else:
        raise ValueError("Unknown argument '{}'.".format(output))

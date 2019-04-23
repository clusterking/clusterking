#!/usr/bin/env python3

# 3rd
import numpy as np
import scipy.spatial
import functools
from typing import Callable

# ours
from clusterking.data.dwe import DataWithErrors


def condense_distance_matrix(matrix):
    """ Convert a square-form distance matrix  to a vector-form distance vector

    Args:
        matrix: n x n symmetric matrix with 0 diagonal

    Returns:
        n choose 2 vector
    """
    return scipy.spatial.distance.squareform(matrix)


def uncondense_distance_matrix(vector):
    """ Convert a vector-form distance vector to a square-form distance matrix

    Args:
        vector: n choose 2 vector

    Returns:
        n x n symmetric matrix with 0 diagonal
    """
    return scipy.spatial.distance.squareform(vector)


def metric_selection(*args, **kwargs) -> Callable:
    """ Select a metric in one of the following ways:

    1. If no positional arguments are given, we choose the euclidean metric.
    2. If the first positional argument is string, we pick one of the metrics
       that are defined in ``scipy.spatical.distance.pdist`` by that name (all
       additional arguments will be past to this function).
    3. If the first positional argument is a function, we take this function
       (and add all additional arguments to it).

    Examples:

    * ``...()``: Euclidean metric
    * ``...("euclidean")``: Also Euclidean metric
    * ``...(lambda data: scipy.spatial.distance.pdist(data.data(),
      'euclidean')``: Also Euclidean metric
    * ``...("minkowski", p=2)``: Minkowsky distance with ``p=2``.

    See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    for more information.

    Args:
        *args: see description above
        **kwargs:  see description above

    Returns:
        Function that takes Data object as only parameter and returns a
        reduced distance matrix.
    """
    if len(args) == 0:
        # default
        args = ['euclidean']
    if isinstance(args[0], str):
        # The user can specify any of the metrics from
        # scipy.spatial.distance.pdist by name and supply additional
        # values
        return lambda data: scipy.spatial.distance.pdist(
            data.data(),
            args[0],
            *args[1:],
            **kwargs
        )
    elif isinstance(args[0], Callable):
        # Assume that this is a function that takes DWE or Data as first
        # argument
        return functools.partial(args[0], *args[1:], **kwargs)
    else:
        raise ValueError(
            "Invalid type of first argument: {}".format(type(args[0]))
        )


# todo: unittest
def chi2_metric(dwe: DataWithErrors, output='condensed'):
    """
    Returns the chi2/ndf values of the comparison of a datasets.

    Args:
        dwe: :py:class:`clusterking.data.dwe.DataWithErrors` object
        output: 'condensed' (condensed distance matrix) or 'full' (full distance
            matrix)

    Returns:
        Condensed distance matrix

    """
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

    if output == 'condensed':
        return condense_distance_matrix(chi2ndf)
    elif output == 'full':
        return chi2ndf
    else:
        raise ValueError("Unknown argument '{}'.".format(output))

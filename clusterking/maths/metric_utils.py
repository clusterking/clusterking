#!/usr/bin/env python3

""" This additional file is needed to avoid circular imports, because metric.py
has dependencies on the DWE class.
"""

# 3rd
import scipy.spatial
import functools
from typing import Callable
import numpy as np

# ours


def condense_distance_matrix(matrix):
    """ Convert a square-form distance matrix  to a vector-form distance vector

    Args:
        matrix: n x n symmetric matrix with 0 diagonal

    Returns:
        n choose 2 vector
    """
    assert matrix.ndim == 2
    # Let's do the checks ourselves, because scipy checks for exact symmetry,
    # which we won't achieve due to rounding errors.
    assert np.isclose(matrix - matrix.T, 0).all()
    assert np.isclose(np.diag(matrix), 0.0).all()
    return scipy.spatial.distance.squareform(matrix, checks=False)


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
        args = ["euclidean"]
    if isinstance(args[0], str):
        # The user can specify any of the metrics from
        # scipy.spatial.distance.pdist by name and supply additional
        # values
        return lambda data: scipy.spatial.distance.pdist(
            data.data(), args[0], *args[1:], **kwargs
        )
    elif isinstance(args[0], Callable):
        # Assume that this is a function that takes DWE or Data as first
        # argument
        return functools.partial(args[0], *args[1:], **kwargs)
    else:
        raise ValueError(
            "Invalid type of first argument: {}".format(type(args[0]))
        )

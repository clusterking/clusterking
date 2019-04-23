#!/usr/bin/env python3

from scipy import integrate as integrate
import numpy as np


# todo: unittest
def bin_function(fct, binning: np.array, normalize=False) -> np.array:
    """Bin function, i.e. calculate the integrals of a function for each bin.

    Args:
        fct: Function to be integrated per bin
        binning:  Array of bin edge points.
        normalize: If true, we will normalize the distribution, i.e. divide
            by the sum of all bins in the end.

    Returns:
        Array of bin contents
    """
    binning = np.array(binning)
    assert len(binning.shape) == 1
    assert binning.shape[0] >= 2
    binning = np.sort(binning)

    bins = list(zip(binning[:-1], binning[1:]))

    bin_contents = []
    for this_bin in bins:
        bin_contents.append(integrate.quad(fct, this_bin[0], this_bin[1])[0])

    bin_contents = np.array(bin_contents)

    if normalize:
        bin_contents = bin_contents / sum(bin_contents)

    return bin_contents

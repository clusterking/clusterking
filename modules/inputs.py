#!/usr/bin/env python3

""" Input parameters/physical constants. """

from collections import OrderedDict


# todo: Perhaps implement that similar to flavio or use values from flavio in the first place?

inputs = {
    'GF': 1.1663787*10**(-5),
    'Vcb': 41.2*10**(-3),
    'new': 1,
    'mB': 5.27942,
    'mD': 1.86723,
    'Btaul': 0.178,  # Tau branching fraction taken from Alonso et al paper
    'mtau': 1.7768,
    'mb': 4.02,
    'mc': 0.946,
    'hbar': 6.58212*10**(-25),
}

# B meson lifetime in natural units
inputs['tauBp'] = 1.638*10**(-12)/inputs['hbar']


class Wilson(object):
    """Class to hold the wilson coefficients/NP parameters."""
    def __init__(self, l, r, sr, sl, t):
        self.l = l
        self.r = r
        self.sr = sr
        self.sl = sl
        self.t = t

    def dict(self):
        params = ["l", "r", "sr", "sl", "t"]
        d = OrderedDict()
        for p in params:
            d[p] = getattr(self, p)
        return d

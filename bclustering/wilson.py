#!/usr/bin/env python3

from collections import OrderedDict


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

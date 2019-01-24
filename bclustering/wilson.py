#!/usr/bin/env python3

from collections import OrderedDict


class Wilson(object):
    """Class to hold the wilson coefficients/NP parameters."""
    def __init__(self, l, r, sr, sl, t):
        #: Left-handed vector current
        self.l = l
        #: Right-handed vector current
        self.r = r
        #: Right-handed scalar current
        self.sr = sr
        #: Left-handed scalar current
        self.sl = sl
        #: Tensor current
        self.t = t

    def dict(self) -> OrderedDict:
        """ Returns an ordered dict of the wilson coefficients in the 
        order l, r, sr, sl and t. """
        params = ["l", "r", "sr", "sl", "t"]
        d = OrderedDict()
        for p in params:
            d[p] = getattr(self, p)
        return d

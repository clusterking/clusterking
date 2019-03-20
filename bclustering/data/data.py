#!/usr/bin/env python3

# 3d
import numpy as np

# ours
from bclustering.data.dfmd import DFMD

# todo: docstrings
class Data(DFMD):
    """ A class which adds more convenience methods to DFMD. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # **************************************************************************
    # Property shortcuts
    # **************************************************************************

    @property
    def was_scanned(self):
        return "scan" in self.md

    @property
    def was_clustered(self):
        return "cluster" in self.md

    @property
    def bin_cols(self):
        columns = list(self.df.columns)
        # todo: more general?
        return [c for c in columns if c.startswith("bin")]

    @property
    def par_cols(self):
        return self.md["scan"]["wpoints"]["coeffs"]

    @property
    def n(self):
        return len(self.df)

    @property
    def nbins(self):
        return len(self.bin_cols)

    @property
    def npars(self):
        return len(self.par_cols)

    # **************************************************************************
    # Returning things
    # **************************************************************************

    def data(self, normalize=False):
        data = self.df[self.bin_cols].values
        if normalize:
            return data / np.sum(data, axis=1)
        else:
            return data

    def norms(self):
        return np.sum(self.data(), axis=1)

#!/usr/bin/env python3

# std

# 3rd
import numpy as np
import functools
from typing import Callable

# ours
from bclustering.data.data import Data
from bclustering.util.metadata import nested_dict, failsafe_serialize
from bclustering.util.log import get_logger
from bclustering.maths.metric import uncondense_distance_matrix, metric_selection


class AbstractBenchmark(object):
    def __init__(self, data: Data):
        self.data = data
        self.bpoints = None
        self.md = nested_dict()
        self.log = get_logger("Benchmark")

    @property
    def cluster_column(self):
        value = self.md["cluster_column"]
        if not value:
            value = "cluster"
        return value

    @cluster_column.setter
    def cluster_column(self, value):
        self.md["cluster_column"] = value

    @property
    def _clusters(self):
        return self.data.df[self.cluster_column]

    def select_bpoints(self, *args, **kwargs):
        self.md["select_bpoints_args"] = failsafe_serialize(args)
        self.md["select_bpoints_kwargs"] = failsafe_serialize(kwargs)
        self.bpoints = self._select_bpoints(*args, **kwargs)

    def _select_bpoints(self, *args, **kwargs):
        raise NotImplementedError

    def write(self, bpoint_column="bpoint"):
        self.data.df[bpoint_column] = self.bpoints
        self.data.md["bpoint"][bpoint_column] = self.md


class Benchmark(AbstractBenchmark):
    def __init__(self, data):
        super().__init__(data)
        self.metric = None
        self.fom = lambda x: np.sum(x, axis=1)

    def set_metric(self, *args, **kwargs) -> None:
        self.md["metric"]["args"] = failsafe_serialize(args)
        self.md["metric"]["kwargs"] = failsafe_serialize(kwargs)
        self.metric = metric_selection(*args, **kwargs)

    def set_fom(self, fct: Callable, *args, **kwargs):
        self.fom = functools.partial(fct, *args, **kwargs)

    def _select_bpoints(self, **kwargs):
        """ Select one benchmark point for each cluster.

        Args:
            data: Data object
            column: Column to write to (True if is benchmark point, False other
                sise)
        """
        if self.metric is None:
            self.log.error(
                "Metric not set. please run self.set_metric or set "
                "self.metric manually before running this method. "
                "Returning without doing anything."
            )
            return

        result = np.full(self.data.n, False)
        for cluster in set(self._clusters):
            # The indizes of all wpoints that are in the current cluster
            indizes = np.argwhere(self._clusters == cluster).squeeze()
            # A data object with only these wpoints
            d_cut = type(self.data)(
                df=self.data.df.iloc[indizes],
                md=self.data.md
            )
            m = self.fom(uncondense_distance_matrix(self.metric(d_cut)))
            # The index of the wpoint of the current cluster that has the lowest
            # sum of distances to all other elements in the same cluster
            index_minimal = indizes[np.argmin(m)]
            result[index_minimal] = True
        return result

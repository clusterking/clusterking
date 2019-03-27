#!/usr/bin/env python3

# std

# 3rd
import numpy as np
from typing import Callable

# ours
from clusterking.data.data import Data
from clusterking.util.metadata import nested_dict, failsafe_serialize
from clusterking.util.log import get_logger
from clusterking.maths.metric import uncondense_distance_matrix, metric_selection


class AbstractBenchmark(object):
    def __init__(self, data: Data):
        """Subclass this class to implement algorithms to choose benchmark
        points from all the points (in parameter space) that correspond to one
        cluster.

        Args:
            data: Data object
        """
        self.data = data
        self.bpoints = None
        self.md = nested_dict()
        self.log = get_logger("Benchmark")
        self.cluster_column = "cluster"

    @property
    def cluster_column(self):
        """ The column from which we read the cluster information.
        Defaults to 'cluster'. """
        return self.md["cluster_column"]

    @cluster_column.setter
    def cluster_column(self, value):
        self.md["cluster_column"] = value

    @property
    def _clusters(self):
        return self.data.df[self.cluster_column]

    def select_bpoints(self) -> None:
        """ Select one benchmark point for each cluster.
        """
        self.bpoints = self._select_bpoints()

    def _select_bpoints(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def write(self, bpoint_column="bpoint") -> None:
        """ Write benchmark points to a column in the dataframe of the data
        object.

        Args:
            bpoint_column: Column to write to

        Returns:
            None
        """
        self.data.df[bpoint_column] = self.bpoints
        self.data.md["bpoint"][bpoint_column] = self.md


# todo: test this
class Benchmark(AbstractBenchmark):
    """ Selecting benchmarks based on a figure of merit that is calculated
    with the metric. You have to use ``Benchmark.set_metric`` to specify
    the metric (as for the ``HierarchyCluster`` class).
    The default case for the figure of merit ("sum") chooses the point as
    benchmark point that minimizes the sum of all distances to all other
    points in the same cluster (where "distance" of course is with respect
    to the metric).
    """
    def __init__(self, data):
        """
        Args:
            data: Data object
        """
        super().__init__(data)
        self.metric = None
        self.fom = lambda x: np.sum(x, axis=1)

    # Docstring set below
    def set_metric(self, *args, **kwargs) -> None:
        self.md["metric"]["args"] = failsafe_serialize(args)
        self.md["metric"]["kwargs"] = failsafe_serialize(kwargs)
        self.metric = metric_selection(*args, **kwargs)

    set_metric.__doc__ = metric_selection.__doc__

    def set_fom(self, fct: Callable, *args, **kwargs) -> None:
        """ Set a figure of merit. The default case for the figure of merit (
        "sum") chooses the point as benchmark point that minimizes the sum of
        all distances to all other points in the same cluster (where
        "distance" of course is with respect to the metric). In general we
        choose the point that minimizes ``self.fom(<metric>)``, i.e. the default
        case corresponds to ``self.fom = lambda x: np.sum(x, axis=1)``, which
        you could have also set by calling ``self.set_com(np.sum, axis=1)``.

        Args:
            fct: Function that takes the metric as first argument
            *args: Positional arguments that are added to the positional
                arguments of ``fct`` after the metric
            **kwargs: Keyword arguments for the function

        Returns:
            None
        """
        self.fom = lambda metric: fct(metric, *args, **kwargs)

    def _select_bpoints(self):
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

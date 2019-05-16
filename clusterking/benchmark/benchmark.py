#!/usr/bin/env python3

# std

# 3rd
import copy
import numpy as np
from typing import Callable

# ours
from clusterking.benchmark.abstract_benchmark import AbstractBenchmark
from clusterking.util.metadata import failsafe_serialize
from clusterking.maths.metric import uncondense_distance_matrix, \
    metric_selection


# todo: test this
class Benchmark(AbstractBenchmark):
    """ Selecting benchmarks based on a figure of merit that is calculated
    with the metric. You have to use
    :py:meth:`~clusterking.bpoints.Benchmark.set_metric` to specify
    the metric (as for the :py:class:`~clusterking.cluster.HierarchyCluster`
    class).
    The default case for the figure of merit ("sum") chooses the point as
    benchmark point that minimizes the sum of all distances to all other
    points in the same cluster (where "distance" of course is with respect
    to the metric).
    """
    def __init__(self, data, cluster_column="cluster"):
        """
        Args:
            data: :py:class:`~clusterking.data.data.Data` object
            cluster_column: Column name of the clusters
        """
        super().__init__(data=data, cluster_column=cluster_column)
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

        result = np.full(self.data.n, False, bool)
        for cluster in set(self._clusters):
            # The indizes of all spoints that are in the current cluster
            indizes = np.squeeze(np.argwhere(self._clusters == cluster), axis=1)
            # A data object with only these spoints
            # todo: Can't we somehow implement this nicer?
            d_cut = type(self.data)()
            d_cut.df = self.data.df.iloc[indizes]
            d_cut.md = copy.deepcopy(self.data.md)
            m = self.fom(uncondense_distance_matrix(self.metric(d_cut)))
            # The index of the wpoint of the current cluster that has the lowest
            # sum of distances to all other elements in the same cluster
            index_minimal = indizes[np.argmin(m)]
            result[index_minimal] = True
        return result

#!/usr/bin/env python3

# std
from typing import Iterable, Optional
import collections
import copy

# 3rd
import tqdm.auto
import pandas as pd

# ours
from clusterking.stability.stabilitytester import (
    AbstractStabilityTester,
    SimpleStabilityTesterResult,
)
from clusterking.data.data import Data
from clusterking.cluster import Cluster
from clusterking.benchmark import AbstractBenchmark


class SubSampleStabilityTesterResult(SimpleStabilityTesterResult):
    pass


class SubSampleStabilityTester(AbstractStabilityTester):
    """ Test the stability of clustering algorithms by repeatedly
    clustering subsamples of data.

    Example:

    .. code-block:: python

        ssst = SubSampleStabilityTester()
        ssst.set_sampling(frac=0.99)
        ssst.set_repeat(50)

        d = ck.Data(path)

        c = ck.cluster.HierarchyCluster()
        c.set_metric("euclidean")
        c.set_max_d(0.2)
        c.run(data=d).write()

        b = Benchmark()
        b.set_metric("euclidean")
        b.run(data=d).write()

        ssstr = ssst.run(data=d, cluster=c, benchmark=b)
    """

    def __init__(self):
        super().__init__()
        #: Fraction of sample points to be contained in the subsamples.
        #: Set using :meth:`set_basic_config`.
        self._sample_kwargs = {}
        #: Number of subsamples to consider.
        #: Set using :meth:`set_basic_config`.
        self._repeat = None
        #: Display a progress bar?
        self._progress_bar = True

        # Set default values:
        self.set_repeat()
        self.set_progress_bar()

    # **************************************************************************
    # Config
    # **************************************************************************

    def set_sampling(self, **kwargs) -> None:
        """ Configure the subsampling of the data. If performing
        benchmarking, it is ensured that none of the benchmark points of the
        original dataframe are removed during subsampling (to allow to
        compare the benchmarking results).

        Args:
            **kwargs: Keyword arguments to
                :meth:`clusterking.data.Data.sample_param_random`, in particular
                keyword arguments to :meth:`pandas.DataFrame.sample`.

        Returns:
            None

        Example:

        .. code-block:: python

            ssst.set_sampling(n=100)     # Sample 100 points
            ssst.set_sampling(frac=0.9)  # Sample 90% of the points

        """
        self._sample_kwargs = kwargs

    def set_repeat(self, repeat=100) -> None:
        """

        Args:
            repeat: Number of subsamples to test

        Returns:
            None
        """
        self._repeat = repeat

    def set_progress_bar(self, state=True) -> None:
        """ Set or unset progress bar.

        Args:
            state: Bool: Display progress bar?

        Returns:
            None
        """
        self._progress_bar = state

    # **************************************************************************
    # Run
    # **************************************************************************

    def run(
        self,
        data: Data,
        cluster: Cluster,
        benchmark: Optional[AbstractBenchmark] = None,
    ) -> SubSampleStabilityTesterResult:
        """ Run test.

        Args:
            data: :class:`~clusterking.data.Data` object
            cluster: Pre-configured :class:`~clusterking.cluster.Cluster`
                object
            benchmark: Optional: :class:`~clusterking.cluster.cluster.Cluster`
                object

        Returns:
            :class:`SubSampleStabilityTesterResult` object
        """
        if not self._sample_kwargs:
            msg = (
                "You need to configure sampling with set_sampling before "
                "you can run this method."
            )
            raise ValueError(msg)

        original_data = data.copy(deep=True)
        cluster.run(original_data).write()
        if self._progress_bar:
            iterator = tqdm.auto.tqdm(range(self._repeat))
        else:
            iterator = range(self._repeat)
        fom_results = collections.defaultdict(list)

        sample_kwargs = copy.deepcopy(self._sample_kwargs)
        if benchmark is not None and "bpoints" not in self._sample_kwargs:
            sample_kwargs["bpoints"] = True

        for _ in iterator:
            this_data = data.sample_param_random(**sample_kwargs)
            cluster.run(this_data).write()
            if benchmark is not None:
                benchmark.run(this_data).write()
            for fom_name, fom in self._foms.items():
                try:
                    fom = fom.run(original_data, this_data).fom
                except ValueError:
                    fom = -1
                fom_results[fom_name].append(fom)

        df = pd.DataFrame(fom_results)
        return SubSampleStabilityTesterResult(df=df)


class SubSampleStabilityVsFractionResult(SimpleStabilityTesterResult):
    pass


class SubSampleStabilityVsFraction(object):
    """ Repeatedly run :class:`SubSampleStabilityTester` for different
    fractions.
    """

    def __init__(self):
        pass

    def run(
        self,
        data: Data,
        cluster: Cluster,
        ssst: SubSampleStabilityTester,
        fractions: Iterable[float],
    ):
        results = collections.defaultdict(list)
        ssst.set_progress_bar(False)
        for fract in tqdm.auto.tqdm(fractions):
            ssst.set_sampling(frac=fract)
            r = ssst.run(data, cluster)
            foms = r.df.mean().to_dict()
            results["fraction"].append(fract)
            for key, value in foms.items():
                results[key].append(value)
        df = pd.DataFrame(results)
        return SubSampleStabilityVsFractionResult(df=df)

#!/usr/bin/env python3

# std
import copy
import collections

# 3rd
import pandas as pd
import tqdm

# ours
from clusterking.stability.stabilitytester import (
    AbstractStabilityTester,
    StabilityTesterResult,
)
from clusterking.data.data import Data
from clusterking.scan.scanner import Scanner
from clusterking.cluster.cluster import Cluster


class NoisySampleStabilityTesterResult(StabilityTesterResult):
    """ Result of :class:`NoisySampleStabilityTester`"""

    def __init__(self, df: pd.DataFrame, cached_data=None):
        super().__init__()
        #: :class:`pd.DataFrame` containing figures of merits for each result
        self.df = df
        self._cached_data = cached_data


class NoisySampleStabilityTester(AbstractStabilityTester):
    """ This stability test generates data samples with slightly varied
    sample points (by adding :meth:`clusterking.scan.Scanner.add_spoints_noise`
    to a pre-configured :class:`clusterking.scan.Scanner` object) and compares
    the resulting clusters and benchmark points.


    """

    def __init__(self):
        super().__init__()
        self._noise_kwargs = {}
        self._noise_args = []
        self._repeat = 10
        self._cache_data = True
        self.set_repeat()

    # **************************************************************************
    # Config
    # **************************************************************************

    def set_repeat(self, repeat=10) -> None:
        """ Set number of experiments.

        Args:
            repeat: Number of experiments

        Returns:
            None
        """
        self._repeat = repeat

    def set_noise(self, *args, **kwargs) -> None:
        """ Configure noise, applied to the spoints in each experiment. See
        :meth:`clusterking.scan.Scanner.add_spoints_noise`.

        Args:
            *args: Positional arguments to
                :meth:`clusterking.scan.Scanner.add_spoints_noise`.
            **kwargs: Keyword argumnets to
                :meth:`clusterking.scan.Scanner.add_spoints_noise`.

        Returns:
            None
        """
        self._noise_args = args
        self._noise_kwargs = kwargs

    def set_cache_data(self, value: bool) -> None:
        """ Do we cache the :class:`~clusterking.data.Data` objects of each
        experimnt?

        Args:
            value: bool. Cache the data?

        Returns:
            None
        """
        self._cache_data = value

    # **************************************************************************
    # Run
    # **************************************************************************

    def run(
        self, data: Data, scanner: Scanner, cluster: Cluster
    ) -> NoisySampleStabilityTesterResult:
        """ Run stability test.

        Args:
            data: :class:`~clusterking.data.data.Data` object
            scanner: :class:`~clusterking.scan.scan.Scanner` object
            cluster: :class:`~clusterking.cluster.cluster.Cluster` object

        Returns:
            :class:`~NoisySampleStabilityTesterResult` object
        """
        scanner.set_progress_bar(False)
        datas = []
        original_data = None
        fom_results = collections.defaultdict(list)
        for _ in tqdm.tqdm(range(self._repeat + 1)):
            noisy_scanner = copy.copy(scanner)
            if _ >= 1:
                noisy_scanner.add_spoints_noise(
                    *self._noise_args, **self._noise_kwargs
                )
            this_data = data.copy(deep=True)
            if self._cache_data:
                datas.append(this_data)
            rs = noisy_scanner.run(this_data)
            rs.write()
            rc = cluster.run(this_data)
            rc.write()
            if _ == 0:
                original_data = this_data.copy(deep=True)
                continue
            for fom_name, fom in self._foms.items():
                fom = fom.run(original_data, this_data).fom
                fom_results[fom_name].append(fom)
        return NoisySampleStabilityTesterResult(
            df=pd.DataFrame(fom_results), cached_data=datas
        )

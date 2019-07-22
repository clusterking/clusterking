#!/usr/bin/env python3

# std
import copy
import collections
from typing import Optional, Union, List
from pathlib import PurePath, Path

# 3rd
import pandas as pd
import tqdm.auto

# ours
from clusterking.stability.stabilitytester import (
    AbstractStabilityTester,
    StabilityTesterResult,
)
from clusterking.data.data import Data
from clusterking.scan.scanner import Scanner
from clusterking.cluster.cluster import Cluster
from clusterking.benchmark.benchmark import AbstractBenchmark
from clusterking.worker import AbstractWorker
from clusterking.result import AbstractResult


class NoisySampleStabilityTesterResult(StabilityTesterResult):
    """ Result of :class:`NoisySampleStabilityTester`"""

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        #: :class:`pd.DataFrame` containing figures of merits for each result
        self.df = df


class NoisySampleResult(AbstractResult):
    def __init__(self, samples: Optional[List[Data]] = None):
        super().__init__()
        if samples is None:
            samples = []
        self.samples = samples

    def write(
        self, directory: Union[str, PurePath], ignore_non_empty=False
    ) -> None:
        """ Write to output directory

        Args:
            directory: Path to directory
            ignore_non_empty: Ignore any files that exist in the directory.
                These might be overwritten.

        Returns:
            None
        """
        directory = Path(directory)
        if directory.exists() and not directory.is_dir():
            raise FileExistsError(
                "{} exists but is not a directory.".format(directory.resolve())
            )
        if not directory.exists():
            directory.mkdir(parents=True)
        if len(list(directory.iterdir())) >= 1 and not ignore_non_empty:
            raise FileExistsError(
                "{} is not an empty directory".format(directory.resolve())
            )
        for i, data in enumerate(self.samples):
            path = directory / "data_{:04d}.sql".format(i)
            data.write(path, overwrite="overwrite")

    def load(self, directory: Union[str, PurePath]) -> None:
        """ Load from output directory

        Args:
            directory: Path to directory to load from
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(
                "{} does not exist or is not a directory".format(directory)
            )
        for path in directory.glob("data_*.sql"):
            self.samples.append(Data(path))


class NoisySample(AbstractWorker):
    def __init__(self):
        """ This stability test generates data samples with slightly varied
        sample points (by adding :meth:`clusterking.scan.Scanner.add_spoints_noise`
        to a pre-configured :class:`clusterking.scan.Scanner` object)
        """
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

    # **************************************************************************
    # Run
    # **************************************************************************

    def run(
        self, scanner: Scanner, data: Optional[Data] = None
    ) -> NoisySampleResult:
        """

        Args:
            scanner: :class:`~clusterking.scan.scan.Scanner` object
            data: data:  :class:`~clusterking.data.data.Data` object. This does not
                have to contain any actual sample points, but is used so that
                you can use data with errors by passing a
                :class:`~clusterking.data.DataWithErrors` object.

        Returns:

        """
        scanner.set_progress_bar(False)
        datas = []
        for _ in tqdm.auto.tqdm(range(self._repeat + 1)):
            noisy_scanner = copy.copy(scanner)
            noisy_scanner.add_spoints_noise(
                *self._noise_args, **self._noise_kwargs
            )
            this_data = data.copy(deep=True)
            noisy_scanner.run(this_data).write()
            datas.append(this_data)
        return NoisySampleResult(datas)


class NoisySampleStabilityTester(AbstractStabilityTester):
    """ This stability test generates data samples with slightly varied
    sample points (by adding :meth:`clusterking.scan.Scanner.add_spoints_noise`
    to a pre-configured :class:`clusterking.scan.Scanner` object) and compares
    the resulting clusters and benchmark points.


    """

    def __init__(self):
        super().__init__()

    # **************************************************************************
    # Run
    # **************************************************************************

    def run(
        self,
        sample: NoisySampleResult,
        cluster: Optional[Cluster] = None,
        benchmark: Optional[AbstractBenchmark] = None,
    ) -> NoisySampleStabilityTesterResult:
        """ Run stability test.

        Args:

            cluster: :class:`~clusterking.cluster.cluster.Cluster` object
            benchmark: Optional: :class:`~clusterking.cluster.cluster.Cluster`
                object

        Returns:
            :class:`~NoisySampleStabilityTesterResult` object
        """
        reference_data = None
        fom_results = collections.defaultdict(list)
        for isample, data in enumerate(sample.samples):
            if cluster is not None:
                cluster.run(data).write()
            if benchmark is not None:
                benchmark.run(data).write()
            if isample == 0:
                reference_data = data.copy(deep=True)
                continue
            for fom_name, fom in self._foms.items():
                fom = fom.run(reference_data, data).fom
                fom_results[fom_name].append(fom)
        return NoisySampleStabilityTesterResult(df=pd.DataFrame(fom_results))

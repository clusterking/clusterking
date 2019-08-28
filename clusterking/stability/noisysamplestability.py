#!/usr/bin/env python3

# std
import copy
import collections
from typing import Optional, Union, List, Callable
from pathlib import PurePath, Path

# 3rd
import pandas as pd
import tqdm.auto

# ours
from clusterking.stability.stabilitytester import (
    AbstractStabilityTester,
    SimpleStabilityTesterResult,
)
from clusterking.data.data import Data
from clusterking.scan.scanner import Scanner
from clusterking.cluster.cluster import Cluster
from clusterking.benchmark.benchmark import AbstractBenchmark
from clusterking.worker import AbstractWorker
from clusterking.result import AbstractResult
from clusterking.util.log import get_logger


class NoisySampleStabilityTesterResult(SimpleStabilityTesterResult):
    """ Result of :class:`NoisySampleStabilityTester`"""

    pass


class NoisySampleResult(AbstractResult):
    def __init__(self, samples: Optional[List[Data]] = None):
        super().__init__()
        if samples is None:
            samples = []
        self.samples = samples  # type: List[Data]

    def write(self, directory: Union[str, PurePath], non_empty="add") -> None:
        """ Write to output directory

        Args:
            directory: Path to directory
            non_empty: What to do if directory is not empty: ``raise`` (raise
                :py:class:`FileExistsError`), ``ignore`` (do nothing and
                potentially overwrite files), ``add`` (add files with new name).
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
        if len(list(directory.iterdir())) >= 1 and not non_empty == "raise":
            raise FileExistsError(
                "{} is not an empty directory".format(directory.resolve())
            )
        if non_empty in ["ignore", "raise"]:
            for i, data in enumerate(self.samples):
                path = directory / "data_{:04d}.sql".format(i)
                data.write(path, overwrite="overwrite")
        elif non_empty == "add":
            i = 0
            for data in self.samples:
                while True:
                    path = directory / "data_{:04d}.sql".format(i)
                    if not path.is_file():
                        data.write(path, overwrite="raise")
                        break
                    i += 1
        else:
            raise ValueError(
                "Unknown option '{}' for non_empty.".format(non_empty)
            )

    def load(
        self, directory: Union[str, PurePath], loader: Optional[Callable] = None
    ) -> None:
        """ Load from output directory

        Args:
            directory: Path to directory to load from
            loader: Function used to load data (optional).

        Example:

        .. code-block:: python

            def loader(path):
                d = clusterking.DataWithError(path)
                d.add_rel_err_uncorr(0.01)
                return d

            nsr = NoisySampleResult()
            nsr.load("/path/to/dir/", loader=loader)

        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(
                "{} does not exist or is not a directory".format(directory)
            )
        for path in directory.glob("data_*.sql"):
            if loader is not None:
                d = loader(path)
            else:
                d = Data(path)
            self.samples.append(d)


class NoisySample(AbstractWorker):
    def __init__(self):
        """ This stability test generates data samples with slightly varied
        sample points (by adding
        :meth:`clusterking.scan.Scanner.add_spoints_noise`
        to a pre-configured :class:`clusterking.scan.Scanner` object)
        """
        super().__init__()
        self._noise_kwargs = {}
        self._noise_args = []
        self._repeat = 10
        self._cache_data = True
        self.set_repeat()
        self.log = get_logger("NoisySample")

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
        .. note::
            This method will handle keyboard interrupts and still return the
            so far collected data.

        Args:
            scanner: :class:`~clusterking.scan.scan.Scanner` object
            data: data:  :class:`~clusterking.data.data.Data` object. This does
                not have to contain any actual sample points, but is used so
                that you can use data with errors by passing a
                :class:`~clusterking.data.DataWithErrors` object.

        Returns:
            :class:`NoisySampleResult`.
        """
        datas = []
        for _ in tqdm.auto.tqdm(range(self._repeat + 1), desc="NoisySample"):
            try:
                noisy_scanner = copy.copy(scanner)
                noisy_scanner.set_progress_bar(True, leave=False, position=1)
                noisy_scanner.add_spoints_noise(
                    *self._noise_args, **self._noise_kwargs
                )
                this_data = data.copy(deep=True)
                noisy_scanner.run(this_data).write()
                datas.append(this_data)
            except KeyboardInterrupt:
                self.log.critical(
                    "Keyboard interrupt: Will still return "
                    "so far collected samples"
                )
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

            sample: :class:`~NoisySampleResult`
            cluster: :class:`~clusterking.cluster.cluster.Cluster` object
            benchmark: Optional: :class:`~clusterking.cluster.cluster.Cluster`
                object

        Returns:
            :class:`~NoisySampleStabilityTesterResult` object
        """
        reference_data = None
        fom_results = collections.defaultdict(list)
        for isample, data in tqdm.auto.tqdm(list(enumerate(sample.samples))):
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

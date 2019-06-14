#!/usr/bin/env python3

# std
from abc import abstractmethod
from typing import Optional

# ours
from clusterking.stability.preprocessor import Preprocessor
from clusterking.worker import AbstractWorker
from clusterking.result import AbstractResult
from clusterking.data.data import Data


class FOMResult(AbstractResult):
    """ Object containing the result of a Figure of Merit (FOM), represented
    by a :class:`FOM` object. """

    def __init__(self, fom, name):
        super().__init__()
        self.fom = fom
        self.name = name


class FOM(AbstractWorker):
    """ Figure of Merit, comparing the outcome of two experiments (e.g. the
    clusters of two very similar datasets). """

    def __init__(
        self,
        name: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        """ Initialize the FOM worker.

        Args:
            name: Name of the FOM
        """
        super().__init__()
        self._name = name
        self._preprocessor = preprocessor

    @property
    def name(self):
        """ Name of the FOM """
        if self._name is None:
            return str(type(self).__name__) + "_" + self._preprocessor.name
        return self._name

    def set_name(self, value: str):
        self._name = value

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            self._preprocessor = Preprocessor()
        return self._preprocessor

    def set_preprocessor(self, preprocessor: Preprocessor):
        self._preprocessor = preprocessor

    def run(self, data1: Data, data2: Data) -> FOMResult:
        """ Calculate figure of merit.

        Args:
            data1: "original" :class:`~clusterking.data.data.Data` object
            data2: "other" :class:`~clusterking.data.data.Data` object

        Returns:
            :class:`FOMResult` object
        """
        preprocessed = self.preprocessor.run(data1, data2)
        return FOMResult(
            fom=self._fom(preprocessed.data1, preprocessed.data2),
            name=self.name,
        )

    @abstractmethod
    def _fom(self, data1: Data, data2: Data):
        pass


class CCFOM(FOM):
    """ Cluster Comparison figure of merit (CCFOM), comparing whether the
    clusters of two experiments match. """


class MatchingClusters(CCFOM):
    """ Fraction of sample points (spoints) that lie in the same cluster, when
    comparing two clustered datasets with the same number of sample points.
    """

    def _fom(self, data1, data2) -> float:
        clustered1 = data1.df["cluster"]
        clustered2 = data2.df["cluster"]
        assert len(clustered1) == len(clustered2)
        return sum(clustered1 == clustered2) / len(clustered1)


class DeltaNClusters(CCFOM):
    """ Difference of number of clusters between two experiments
    (number of clusters in experiment 1 - number of lcusters in experiment 2).
    """

    def _fom(self, data1, data2) -> int:
        clustered1 = data1.df["cluster"]
        clustered2 = data2.df["cluster"]
        return len(set(clustered1)) - len(set(clustered2))


class BMFOM(FOM):
    """ Benchmark Figure of Merit (BMFOM), comparing whether the benchmark
    points of two experiments match. """

    pass

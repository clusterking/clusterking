#!/usr/bin/env python3

from abc import abstractmethod

# ours
from clusterking.stability.ccpreprocessor import CCPreprocessor
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

    def __init__(self, name=None):
        """ Initialize the FOM worker.

        Args:
            name: Name of the FOM
        """
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        """ Name of the FOM """
        if self._name is None:
            return str(type(self).__name__)
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @abstractmethod
    def run(self, data1: Data, data2: Data) -> FOMResult:
        """ Calculate figure of merit.

        Args:
            data1: "original" :class:`~clusterking.data.data.Data` object
            data2: "other" :class:`~clusterking.data.data.Data` object

        Returns:
            :class:`FOMResult` object
        """
        pass


class CCFOM(FOM):
    """ Cluster comparison figure of merit (CCFOM). """

    def __init__(self, preprocessor=None, name=None):
        super().__init__(name=name)
        self._preprocessor = preprocessor

    @property
    def name(self):
        """ Name of the FOM """
        if self._name is None:
            return str(type(self).__name__) + "_" + self._preprocessor.name
        return self._name

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            self._preprocessor = CCPreprocessor()
        return self._preprocessor

    def set_preprocessor(self, preprocessor):
        self._preprocessor = preprocessor

    def run(self, data1: Data, data2: Data) -> FOMResult:
        clustered1 = data1.df["cluster"]
        clustered2 = data2.df["cluster"]
        preprocessed = self.preprocessor.run(clustered1, clustered2)
        return FOMResult(
            fom=self._fom(preprocessed.clustered1, preprocessed.clustered2),
            name=self.name,
        )

    @abstractmethod
    def _fom(self, clustered1, clustered2):
        pass


class MatchingClusters(CCFOM):
    """ Fraction of sample points (spoints) that lie in the same cluster, when
    comparing two clustered datasets with the same number of sample points.
    """

    def _fom(self, clustered1, clustered2) -> float:
        assert len(clustered1) == len(clustered2)
        return sum(clustered1 == clustered2) / len(clustered1)


class DeltaNClusters(CCFOM):
    """ Difference of number of clusters between two experiments
    (number of clusters in experiment 1 - number of lcusters in experiment 2).
    """

    def _fom(self, clustered1, clustered2) -> int:
        return len(set(clustered1)) - len(set(clustered2))

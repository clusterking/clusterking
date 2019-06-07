#!/usr/bin/env python3

from abc import ABC, abstractmethod

# ours
from clusterking.stability.ccpreprocessor import CCPreprocessor


class CCFOMResult(object):
    def __init__(self, fom, name):
        self.fom = fom
        self.name = name


class CCFOM(ABC):
    """ Cluster comparison figure of merit (CCFOM). """

    def __init__(self, preprocessor=None, name=None):
        self._preprocessor = preprocessor
        self._name = name

    @property
    def name(self):
        if self._name is None:
            return str(type(self).__name__) + "_" + self._preprocessor.name
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            self._preprocessor = CCPreprocessor()
        return self._preprocessor

    def set_preprocessor(self, preprocessor):
        self._preprocessor = preprocessor

    def run(self, clustered1, clustered2):
        preprocessed = self.preprocessor.run(clustered1, clustered2)
        return CCFOMResult(
            fom=self._fom(preprocessed.clustered1, preprocessed.clustered2),
            name=self.name,
        )

    @abstractmethod
    def _fom(self, clustered1, clustered2):
        pass


class MatchingClusters(CCFOM):
    def _fom(self, clustered1, clustered2):
        assert len(clustered1) == len(clustered2)
        return sum(clustered1 == clustered2) / len(clustered1)


class DeltaNClusters(CCFOM):
    def _fom(self, clustered1, clustered2):
        return len(set(clustered1)) - len(set(clustered2))

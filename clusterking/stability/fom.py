#!/usr/bin/env python3

# std
from abc import abstractmethod
from typing import Optional, Callable, Dict, Tuple, Any, Union

# 3rd
import numpy as np

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
        exceptions="raise",
    ):
        """ Initialize the FOM worker.

        Args:
            name: Name of the FOM
            preprocessor:
                :class:`~clusterking.stability.preprocessor.Preprocessor`
                object
            exceptions: When calculating the FOM, what should we do if an
                exception arises. 'raise': Raise exception, 'print': Return
                None and print exception information
        """
        super().__init__()
        self._name = name
        self._preprocessor = preprocessor
        self._exceptions_handling = exceptions

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
        try:
            fom = self._fom(preprocessed.data1, preprocessed.data2)
        except Exception as e:
            if self._exceptions_handling == "raise":
                raise e
            elif self._exceptions_handling == "print":
                fom = None
                print(e)
            else:
                raise ValueError(
                    "Invalid value for exception "
                    "handling: {}".format(self._exceptions_handling)
                )
        return FOMResult(fom=fom, name=self.name)

    @abstractmethod
    def _fom(self, data1: Data, data2: Data):
        pass


class CCFOM(FOM):
    """ Cluster Comparison figure of merit (CCFOM), comparing whether the
    clusters of two experiments match. """

    @abstractmethod
    def _fom(self, data1: Data, data2: Data):
        pass


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

    def _fom(self, data1: Data, data2: Data) -> float:
        clusters1 = set(data1.df["cluster"].unique())
        clusters2 = set(data2.df["cluster"].unique())
        if not clusters1 == clusters2:
            return np.nan
        clusters = clusters1
        cluster2bpoint = {}
        for cluster in clusters:
            bpoints1 = data1.df[
                (data1.df["cluster"] == cluster) & data1.df["bpoint"]
            ]
            bpoints2 = data2.df[
                (data2.df["cluster"] == cluster) & data2.df["bpoint"]
            ]
            msg = "Found {} bpoints instead of 1 for dataset {}."
            if len(bpoints1) != 1:
                raise ValueError(msg.format(len(bpoints1), 1))
            if len(bpoints2) != 1:
                raise ValueError(msg.format(len(bpoints2), 2))
            bpoint1 = bpoints1.iloc[0][data1.par_cols]
            bpoint2 = bpoints2.iloc[0][data2.par_cols]
            cluster2bpoint[cluster] = (bpoint1, bpoint2)
        return self._fom2(cluster2bpoint)

    @abstractmethod
    def _fom2(self, cluster2bpoint: Dict[int, Tuple[Any, Any]]) -> float:
        pass


class AverageBMProximityFOM(BMFOM):
    """ Returns the average distance of benchmark points in parameter space
    between two experiments.
    """

    _named_averaging_fcts = {
        "max": lambda it: max(it),
        "arithmetic": lambda it: sum(it) / len(it),
    }
    _named_metric_fcts = {
        "euclidean": lambda x: np.sqrt(np.sum(np.square(x[0] - x[1])))
    }

    named_averaging_fcts = _named_averaging_fcts.keys()
    named_metric_fcts = _named_metric_fcts.keys()

    def __init__(self, *args, **kwargs):
        """ Initialize the FOM worker.

        Args:
            See :meth:`~clusterking.stability.fom.FOM.__init__`
        """
        super().__init__(*args, **kwargs)
        self._averaging = self._named_averaging_fcts["arithmetic"]
        self._metric = self._named_metric_fcts["euclidean"]

    def set_averaging(self, fct: Union[str, Callable]) -> None:
        """ Set averaging mode

        Args:
            fct: Function of the distances between benchmark points of the same
                cluster or name of pre-implemented functions (check
                :attr:`named_averaging_fcts` for a list)

        Returns:
            None
        """
        if isinstance(fct, str):
            self._averaging = self._named_averaging_fcts[fct]
        else:
            self._averaging = fct

    def set_metric(self, fct: Union[str, Callable]) -> None:
        """ Set metric in parameter space

        Args:
            fct: Functino of a tuple of two points in parameter space or name
                of pre-implemented functions (check
                :attr:`named_metric_fcts` for a list)

        Returns:
            None
        """
        if isinstance(fct, str):
            self._metric = self._named_metric_fcts[fct]
        else:
            self._metric = fct

    def _fom2(self, cluster2bpoint: Dict[int, Tuple[Any, Any]]) -> float:
        return self._averaging(list(map(self._metric, cluster2bpoint.values())))

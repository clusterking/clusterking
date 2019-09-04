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
    ):
        """ Initialize the FOM worker.

        Args:
            name: Name of the FOM
            preprocessor:
                :class:`~clusterking.stability.preprocessor.Preprocessor`
                object

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
        fom = self._fom(preprocessed.data1, preprocessed.data2)
        return FOMResult(fom=fom, name=self.name)

    @abstractmethod
    def _fom(self, data1: Data, data2: Data):
        pass


# todo: add cluster column setting in init
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


class NClusters(CCFOM):
    """ Number of clusters in dataset 1 or 2"""

    def __init__(self, which, **kwargs):
        """

        Args:
            which: 1 or 2 for dataset 1 or dataset 2
            **kwargs: Keyword argumnets for :class:`CCFOM``
        """
        super().__init__(**kwargs)
        self.which = which
        if self.which not in [1, 2]:
            raise ValueError(
                "Invalid value of which, must be 1 or 2, but is {}".format(
                    self.which
                )
            )

    def _fom(self, data1, data2) -> int:
        if self.which == 1:
            return len(set(data1.df["cluster"]))
        elif self.which == 2:
            return len(set(data2.df["cluster"]))
        else:
            raise ValueError("Invalid which value.")


class BpointList(FOM):
    """ Adds array of bpoint coordinates of data2 """

    def _fom(self, data1, data2) -> np.ndarray:
        return data2.df[data2.df["bpoint"] == True][data2.par_cols].to_numpy()


# todo: configure bpoint column
class BMFOM(FOM):
    """ **Abstract class**:
    Benchmark Figure of Merit (BMFOM), comparing whether the benchmark
    points of two experiments match.
    """

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
            msg = "Found {} bpoints instead of 1 for dataset {}: "
            if len(bpoints1) != 1:
                raise ValueError(msg.format(len(bpoints1), 1) + str(bpoints1))
            if len(bpoints2) != 1:
                raise ValueError(msg.format(len(bpoints2), 2) + str(bpoints2))
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

    # todo: no, set this in __init__
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
            fct: Function of a tuple of two points in parameter space or name
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
        ret = self._averaging(list(map(self._metric, cluster2bpoint.values())))
        if not isinstance(ret, (float, int)):
            raise ValueError("Not float")
        return ret

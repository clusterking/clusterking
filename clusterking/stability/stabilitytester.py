#!/usr/bin/env python3

# std
from abc import abstractmethod
from typing import Union
from pathlib import Path, PurePath

# 3rd
import pandas as pd

# ours
from clusterking.worker import AbstractWorker
from clusterking.result import AbstractResult
from clusterking.stability.fom import FOM


class StabilityTesterResult(AbstractResult):
    """ Result of a :class:`AbstractStabilityTester` """


class SimpleStabilityTesterResult(AbstractResult):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def write(self, path: Union[str, PurePath]) -> None:
        """ Save to file. """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path)

    @classmethod
    def load(cls, path: Union[str, PurePath]) -> "SimpleStabilityTesterResult":
        """ Load :class:`SimpleStabilityTesterResult` from file.

        Args:
            path: Path to result file

        Returns:
            :class:`SimpleStabilityTesterResult` object

        Example:

            sstr = SimpleStabilityTesterResult.load("path/to/file")
        """
        return SimpleStabilityTesterResult(df=pd.read_csv(Path(path)))


class AbstractStabilityTester(AbstractWorker):
    """ Abstract baseclass to perform stability tests. This baseclass is
    a subclass of :class:`clusterking.worker.AbstractWorker` and thereby
    adheres to the Command design pattern: After initialization, several
    methods can be called to modify internal settings. Finally, the
    :meth:`run` method is called to perform the actual test.

    All current stability tests perform the task at hand (clustering,
    benchmarking, etc.) for multiple, slightly varied datasets or worker
    parameters (these runs are called 'experiments'). For each of these (for
    each experiment), figures of merit (FOMs) are calculated that compare the
    outcome with the original outcome (e.g. how many points still lie in the
    same cluster, or how far the benchmark points are diverging). These FOMs
    are then written out to a :class:`StabilityTesterResult` object,
    which provides methods for visualization and further analyses (e.g.
    histograms, etc.).
    """

    def __init__(self, exceptions="raise"):
        """ Initialize :class:`AbstractStabilityTester`

        Args:
            exceptions: When calculating the FOM, what should we do if an
                exception arises. 'raise': Raise exception, 'print': Return
                None and print exception information.
        """
        super().__init__()
        self._foms = {}
        self._exceptions_handling = exceptions

    def add_fom(self, fom: FOM) -> None:
        """ Add a figure of merit (FOM).

        Args:
            fom: :class:`~clusterking.stability.fom.FOM` object

        Returns:
            None
        """
        if fom.name in self._foms:
            # todo: do with log
            print(
                "Warning: FOM with name {} already existed. Replacing.".format(
                    fom.name
                )
            )
        self._foms[fom.name] = fom

    @abstractmethod
    def run(self, *args, **kwargs) -> StabilityTesterResult:
        """ Run the stability test.

        Args:
            *args: Positional arguments
            **kwargs: Key word arguments

        Returns:
            :class:`~StabilityTesterResult`
            object
        """
        pass

#!/usr/bin/env python3

# std
from abc import abstractmethod

# ours
from clusterking.worker import AbstractWorker


class AbstractStabilityTester(AbstractWorker):
    def __init__(self):
        super().__init__()
        self._foms = []

    def add_fom(self, fom) -> None:
        """
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
    def run(self, *args, **kwargs):
        pass

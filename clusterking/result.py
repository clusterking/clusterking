#!/usr/bin/env python3

# std
from abc import ABC, abstractmethod

# ours
from clusterking.util.log import get_logger
from clusterking.data.data import Data


# todo: doc
class Result(ABC):
    def __init__(self, data: Data):
        self._data = data  # type: Data
        self.log = get_logger(str(type(self)))

    def write(self):
        return self._write()

    @abstractmethod
    def _write(self):
        pass

#!/usr/bin/env python3

from abc import ABC, abstractmethod
from clusterking.util.log import get_logger


class Result(ABC):
    def __init__(self, data):
        self._data = data
        self.log = get_logger(str(type(self)))

    def write(self):
        return self._write()

    @abstractmethod
    def _write(self):
        pass

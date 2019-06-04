#!/usr/bin/env python3

# std
from abc import ABC, abstractmethod

# ours
from clusterking.data.data import Data


class Worker(ABC):
    def __init__(self):
        pass

    def run(self, data):
        return self._run(data)

    @abstractmethod
    def _run(self, data: Data):
        pass

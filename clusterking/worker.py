#!/usr/bin/env python3

from abc import ABC, abstractmethod


class Worker(ABC):
    def __init__(self):
        pass

    def run(self, data):
        return self._run(data)

    @abstractmethod
    def _run(self, data):
        pass

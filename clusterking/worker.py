#!/usr/bin/env python3

# std
from abc import ABC, abstractmethod

# ours
from clusterking.data.data import Data


class Worker(ABC):
    """ The worker class represents an operation on the data.

    It provides a number of methods to allow for configuration.

    After configuration, :meth:`run` can be called on a
    :class:`~clusterking.data.Data` object.

    The underlying design patterns of this class are therefore the
    `template method pattern <https://en.wikipedia.org/wiki/Template_method_pattern>`_
    and the `command pattern <https://en.wikipedia.org/wiki/Command_pattern>`_.
    """

    def __init__(self):
        pass

    def run(self, data):
        """ Run the operation on the data.

        Args:
            data: :class:`~clusterking.data.Data` object

        Returns:
            :class:`~clusterking.result.Result` object
        """
        return self._run(data)

    @abstractmethod
    def _run(self, data: Data):
        pass

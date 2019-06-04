#!/usr/bin/env python3

# std
from abc import ABC, abstractmethod

# ours
from clusterking.util.log import get_logger
from clusterking.data.data import Data


# todo: doc
class Result(ABC):
    """ The result object represents the result of the execution of a
    :class:`~clusterking.worker.Worker` object on the
    :class:`~clusterking.data.Data` object.
    """

    def __init__(self, data: Data):
        """ Initializer of the object.

        .. note::

            The :class:`~clusterking.result.Result` is not meant to be
            initialized by the user. Rather it is a return object of the
            :meth:`clusterking.worker.Worker.run` method.
        """
        self._data = data  # type: Data
        self.log = get_logger(str(type(self)))

    def write(self, *args, **kwargs):
        """ Write relevant data back to the :class:`~clusterking.data.Data`
        object that was passed to :meth:`clusterking.worker.Worker.run`.
        """
        return self._write(*args, **kwargs)

    @abstractmethod
    def _write(self, *args, **kwargs):
        pass

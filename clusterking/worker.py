#!/usr/bin/env python3

# std
from abc import ABC, abstractmethod


class AbstractWorker(ABC):
    """ The AbstractWorker class represents an abstract operation on some data.

    It provides a number of methods to allow for configuration.

    After configuration, :meth:`run` can be called.

    The underlying design patterns of this class are therefore the
    `template method pattern <https://en.wikipedia.org/wiki/Template_method_pattern>`_
    and the `command pattern <https://en.wikipedia.org/wiki/Command_pattern>`_.
    """

    def __init__(self):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """ Run the operation. Must be implemented in subclass. """
        pass


class DataWorker(AbstractWorker):
    """ The worker class represents an operation on some data.

    It provides a number of methods to allow for configuration.

    After configuration, :meth:`run` can be called.

    The underlying design patterns of this class are therefore the
    `template method pattern <https://en.wikipedia.org/wiki/Template_method_pattern>`_
    and the `command pattern <https://en.wikipedia.org/wiki/Command_pattern>`_.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def run(self, *args, **kwargs):
        """ Run the operation. Must be implemented in subclass. """
        pass

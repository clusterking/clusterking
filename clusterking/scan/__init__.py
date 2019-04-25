#!/usr/bin/env python3

""" This module generates a parameter dependent distributions for a selection
of sample points (points in parameter space), called ``spoints`` throughout
the code.

Two classes are defined:

* :class:`~clusterking.scan.Scanner`: A general class, set up with a function
  (specified in :meth:`~clusterking.scan.Scanner.set_dfunction`) that depends on
  points in parameter space and a set of sample points in this parameter space
  (specified via one of the ``set_spoints_...`` methods). The function is then
  run for every sample point and the results are written to a
  :class:`~clusterking.data.Data`-like object.
* :class:`~clusterking.scan.WilsonScanner`: This is a subclass of
  :class:`~clusterking.scan.Scanner` that takes a wilson coefficient in the form
  of a :class:`wilson.Wilson` object as first argument.
"""

from clusterking.scan.scanner import Scanner
from clusterking.scan.wilsonscanner import WilsonScanner

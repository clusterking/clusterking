``Data``
=====================

This page describes the main data object that are used by ClusterKinG.
If you do not need to include errors in your analysis, use
:py:class:`~clusterking.data.data.Data`, else
:py:class:`~clusterking.data.dwe.DataWithErrors` (which inherits from
:py:class:`~clusterking.data.data.Data` but adds additional methods to it).

Both classes inherit from a very basic class,
:py:class:`~clusterking.data.dfmd.DFMD`, which provides basic input and output
methods.


``DFMD``
--------

.. autoclass:: clusterking.data.DFMD
  :members:

``Data``
--------

.. autoclass:: clusterking.data.Data
  :members:

``DataWithErrors``
------------------

.. autoclass:: clusterking.data.DataWithErrors
  :members:

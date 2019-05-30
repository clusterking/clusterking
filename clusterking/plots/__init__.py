#!/usr/bin/env python3

"""
Implementation of different plots.

.. note::

    Most plots are now directly available as methods of the :class:`.data.Data`,
    e.g. :meth:`~clusterking.data.Data.plot_clusters_scatter()` is equivalent to


    .. code-block:: python

        cp = ClusterPlot(data)
        cp.scatter()

.. warning::

    These implementations are still subject to change in the near future, so
    it is recommended to use the methods of the :class:`.data.Data` class as
    advertised above.

"""
try:
    import matplotlib
except ImportError:
    from clusterking.util.log import get_logger

    log = get_logger()
    msg = (
        "Could not import matplotlib. Perhaps you didn't install ClusterKinG "
        "with the 'plotting' option? Please install matplotlib to use "
        "ClusterKinG's plotting funcionality. "
    )

from clusterking.plots.plot_bundles import BundlePlot
from clusterking.plots.plot_clusters import ClusterPlot
from clusterking.plots.plot_histogram import plot_histogram
from clusterking.plots.colors import ColorScheme

#!/usr/bin/env python3


def import_matplotlib():
    """ Tries to import matplotlib, printing help message if it fails. """
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
        log.critical(msg)
        raise ImportError(msg)


def check_matplot_inline():
    """ Return true, if running matplotlib inline."""
    import matplotlib

    return "inline" in matplotlib.get_backend()

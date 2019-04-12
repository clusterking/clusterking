#!/usr/bin/env python3


def check_matplot_inline():
    """ Return true, if running matplotlib inline."""
    import matplotlib
    return 'inline' in matplotlib.get_backend()

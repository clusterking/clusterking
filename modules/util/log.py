#!/usr/bin/env python3

"""Defines an easy function to set up a logger. """

import logging

try:
    import colorlog
except ImportError:
    colorlog = None


def get_logger(name="Logger"):
    """Sets up a logging.Logger.

    If the colorlog module is available, the logger will use colors,
    otherwise it will be in b/w. The colorlog module is available at
    https://github.com/borntyping/python-colorlog but can also easily be
    installed with e.g. 'sudo pip3 colorlog' or similar commands.
    
    Arguments:
        name: name of the logger
    Returns:
        Logger
    """
    if colorlog:
        _logger = colorlog.getLogger(name)
    else:
        _logger = logging.getLogger(name)

    if _logger.handlers:
        # the logger already has handlers attached to it, even though
        # we didn't add it ==> logging.get_logger got us an existing
        # logger ==> we don't need to do anything
        return _logger

    _logger.setLevel(logging.DEBUG)
    if colorlog is not None:
        sh = colorlog.StreamHandler()
        log_colors = {'DEBUG':    'cyan',
                      'INFO':     'green',
                      'WARNING':  'yellow',
                      'ERROR':    'red',
                      'CRITICAL': 'red'}
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(name)s:%(levelname)s:%(message)s',
            log_colors=log_colors)
    else:
        # no colorlog available:
        sh = logging.StreamHandler()
        formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)
    _logger.addHandler(sh)

    if colorlog is None:
        _logger.debug("Module colorlog not available. Log will be b/w.")

    return _logger


if __name__ == "__main__":
    # Test the color scheme for the logger.
    l = get_logger("test")
    l.debug("Test debug message")
    l.info("Test info message")
    l.warning("Test warning message")
    l.error("Test error message")
    l.critical("Test critical message")

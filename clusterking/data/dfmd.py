#!/usr/bin/env python3

# std
import copy
import json
import logging
import pandas as pd
from pathlib import PurePath, Path
from typing import Union

import sqlalchemy

# ours
from clusterking.util.metadata import nested_dict
from clusterking.util.log import get_logger


# fixme @caveat below: perhaps we should simply do that ourselves then?
#   Unused objects should be garbage collected anyhow
class DFMD(object):
    """ This class bundles a pandas dataframe together with metadata and
    provides methods to load from and write these two to files.
    """
    def __init__(self, path=None, log=None):
        """
        There are five different ways to initialize this class:

        1. Initialize it empty: ``DFMD()``.
        2. Load from a path: ``DFMD("path/to/io.sql")``

        Args:
            log: instance of :py:class:`logging.Logger` or name of logger to be
                created
            *args: See above
            **kwargs: See above
        """
        # These are the three attributes of this class
        #: This will hold all the configuration that we will write out
        self.md = None
        #: :py:class:`pandas.DataFrame` to hold all of the results
        self.df = None  # type: pd.DataFrame
        #: Instance of :py:class:`logging.Logger`
        self.log = None

        # todo: remember path?
        if not path:
            # Initialize blank
            self.md = nested_dict()
            self.df = pd.DataFrame()
            self.log = None
        else:
            self.load(path)

        # Overwrite log if user wants that.
        if isinstance(log, logging.Logger):
            self.log = log
        elif isinstance(log, str):
            self.log = get_logger(log)
        elif log is None:
            if not self.log:
                self.log = get_logger("DFMD")
        else:
            raise ValueError(
                "Unsupported type '{}' for 'log' argument.".format(
                    type(log)
                )
            )

    # **************************************************************************
    # Loading
    # **************************************************************************

    def load(self, path: Union[str, PurePath]) -> None:
        """ Load from input files which have been generated from
        :py:meth:`~clusterking.data.DFMD.write`.

        Args:
            path: Path to input/output directory

        Returns:
            None
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("File '{}' doesn't exist.".format(path))
        engine = sqlalchemy.create_engine('sqlite:///' + str(path.resolve()))
        self.df = pd.read_sql_table("df", engine)
        self.df.set_index("index", inplace=True)
        md_json = pd.read_sql_table("md", engine)["md"][0]
        self.md = nested_dict()
        self.md.update(json.loads(md_json))

    # **************************************************************************
    # Writing
    # **************************************************************************

    def write(self, path: Union[str, PurePath]):
        # fixme: handle overwrite?
        path = Path(path)
        # fixme: Do we have to care about non existing directories?
        engine = sqlalchemy.create_engine('sqlite:///' + str(path))
        self.df.to_sql("df", engine, if_exists="replace")
        # todo: perhaps it's better to use pickle in the future?
        md_json = json.dumps(self.md, sort_keys=True, indent=4)
        md_df = pd.DataFrame({"md": [md_json]})
        md_df.to_sql("md", engine, if_exists="replace")

    def copy(self, deep=True):
        """ Make a copy of this object.

        Args:
            deep: Make a deep copy (default True). If this is disabled, any
                change to the copy will also affect the original.

        Returns:
            New object.
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    # **************************************************************************
    # Magic methods
    # **************************************************************************

    def __copy__(self):
        new = type(self)()
        new.df = copy.copy(self.df)
        new.md = copy.copy(self.md)
        new.log = self.log
        return new

    def __deepcopy__(self, memo):
        # Pycharm doesn't seem to recognize the memo argument:
        # noinspection PyArgumentList
        new = type(self)()
        new.df = copy.deepcopy(self.df, memo)
        new.md = copy.deepcopy(self.md, memo)
        new.log = self.log
        memo[id(self)] = new
        return new

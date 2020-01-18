#!/usr/bin/env python3

# std
import copy
import json
import logging
import pandas as pd
from pathlib import PurePath, Path
from typing import Union, Optional

# 3rd
import sqlalchemy

# ours
from clusterking.util.metadata import turn_into_nested_dict, nested_dict
from clusterking.util.log import get_logger
from clusterking.util.cli import handle_overwrite


class DFMD(object):
    """ DFMD = DataFrame with MetaData.
    This class bundles a pandas dataframe together with metadata and
    provides methods to save and load such an object.
    """

    def __init__(
        self,
        path: Optional[Union[str, PurePath]] = None,
        log: Optional[Union[str, logging.Logger]] = None,
    ):
        """
        Initialize a DFMD object.

        Args:
            path: Optional: load from this file (specified as string or
                :class:`pathlib.PurePath`)
            log: Optional: instance of :py:class:`logging.Logger` or name of
                logger to be created
        """
        # These are the three attributes of this class
        #: This will hold all the configuration that we will write out
        self.md = None
        #: :py:class:`pandas.DataFrame` to hold all of the results
        self.df = None  # type: Optional[pd.DataFrame]
        #: Instance of :py:class:`logging.Logger`
        self.log = None

        # todo: remember path?
        if not path:
            # Initialize blank
            self.md = nested_dict()
            self.df = pd.DataFrame()
            self.log = None
        else:
            self._load(path)

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
                "Unsupported type '{}' for 'log' argument.".format(type(log))
            )

    # **************************************************************************
    # Loading
    # **************************************************************************

    def _load(self, path: Union[str, PurePath]) -> None:
        """ Load input file as created by
        :py:meth:`~clusterking.data.DFMD.write`.

        Args:
            path: Path to input file

        Returns:
            None
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("File '{}' doesn't exist.".format(path))
        engine = sqlalchemy.create_engine("sqlite:///" + str(path.resolve()))
        self.df = pd.read_sql_table("df", engine)
        self.df.set_index("index", inplace=True)
        md_json = pd.read_sql_table("md", engine)["md"][0]
        self.md = turn_into_nested_dict(json.loads(md_json))

    # **************************************************************************
    # Writing
    # **************************************************************************

    def write(self, path: Union[str, PurePath], overwrite="ask"):
        """ Write output files.

        Args:
            path: Path to output file
            overwrite: How to proceed if output file already exists:
                'ask' (ask interactively for approval if we have to overwrite),
                'overwrite' (overwrite without asking), 'raise'
                (raise Exception if file exists).
                Default is 'ask'.

        Returns:
            None
        """
        path = Path(path)
        handle_overwrite([path], behavior=overwrite, log=self.log)
        if not path.parent.is_dir():
            self.log.debug("Creating directory '{}'.".format(path.parent))
            path.parent.mkdir(parents=True)

        engine = sqlalchemy.create_engine("sqlite:///" + str(path))
        self.df.to_sql("df", engine, if_exists="replace")
        # todo: perhaps it's better to use pickle in the future?
        md_json = json.dumps(self.md, sort_keys=True, indent=4)
        md_df = pd.DataFrame({"md": [md_json]})
        md_df.to_sql("md", engine, if_exists="replace")

    def copy(self, deep=True, data=True, memo=None):
        """ Make a copy of this object.

        Args:
            deep: Make a deep copy (default True). If this is disabled, any
                change to the copy will also affect the original.
            data: Also copy data
            memo:

        Returns:
            New object.
        """
        new = type(self)()
        if data:
            if deep:
                # Pycharm doesn't seem to recognize the memo argument:
                # noinspection PyArgumentList
                new.df = copy.deepcopy(self.df, memo)
            else:
                new.df = copy.copy(self.df)
        if deep:
            # noinspection PyArgumentList
            new.md = copy.deepcopy(self.md, memo)
        else:
            new.md = copy.copy(self.md)
        new.log = copy.copy(self.log)
        if deep and memo is not None:
            memo[id(self)] = new
        return new

    # **************************************************************************
    # Magic methods
    # **************************************************************************

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(deep=True)

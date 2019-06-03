#!/usr/bin/env python3

# todo: why is this docstring not recognized in sphinx?
"""
This module bundles mostly technical utilities that might not be all this
interesting for users.
"""

import clusterking.cluster
import clusterking.data
import clusterking.maths
import clusterking.scan
import clusterking.util
from clusterking.data import Data, DataWithErrors
from clusterking.benchmark.benchmark import Benchmark
from clusterking.util.metadata import get_version as _get_version

version = _get_version()

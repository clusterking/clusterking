#!/usr/bin/env python3

# std
import sys

# ours
import clusterking.cluster
import clusterking.data
import clusterking.maths
import clusterking.scan
import clusterking.util
from clusterking.data import Data, DataWithErrors
from clusterking.benchmark.benchmark import Benchmark
from clusterking.util.metadata import get_version as _get_version
from clusterking.util.log import get_logger

version = _get_version()


if sys.version_info < (3, 6):
    get_logger().warning(
        "The newer versions of ClusterKinG will require "
        "python >= 3.6. Anything below that has reached"
        " its end of life. Please consider upgrading."
    )

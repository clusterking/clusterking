#!/usr/bin/env python3

# 3rd
import numpy as np

# ours
from clusterking.scan.wilsonscanner import WilsonScanner
from clusterking.data.data import Data
from clusterking.data.dwe import DataWithErrors
from clusterking.cluster.hierarchy_cluster import HierarchyCluster
from clusterking.maths.metric import chi2_metric
from clusterking.benchmark.benchmark import Benchmark


# noinspection PyUnusedLocal
def random_kinematics(w, q):
    # We can't pull sqrts from negative values for floats
    return max(0.0, np.random.normal(loc=10))


def test_dress_rehearsal(tmp_path):
    s = WilsonScanner(scale=5, eft="WET", basis="flavio")

    s.set_dfunction(
        random_kinematics, sampling=np.linspace(0.0, 1.0, 10), normalize=True
    )
    s.set_no_workers(no_workers=1)

    s.set_spoints_equidist(
        {
            "CVL_bctaunutau": (-0.5, 0.5, 3),
            "CSL_bctaunutau": (-0.5, 0.5, 3),
            "CT_bctaunutau": (-0.1, 0.1, 3),
        }
    )
    d = Data()
    r = s.run(d)
    r.write()
    # Can remove str casting once we remove py3.5 support
    d.write(str(tmp_path / "dress_rehearsal.sql"), overwrite="overwrite")

    d = DataWithErrors(str(tmp_path / "dress_rehearsal.sql"))

    d.add_rel_err_uncorr(0.01)
    d.add_err_poisson(1000)

    c = HierarchyCluster()
    c.set_metric(chi2_metric)
    b = Benchmark()
    b.set_metric(chi2_metric)

    c.set_max_d(1)
    c.run(d).write()
    b.run(d).write()

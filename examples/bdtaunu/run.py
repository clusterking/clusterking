#!/usr/bin/env python3

import sys
sys.path = ["../../"] + sys.path
import numpy as np
import bclustering.physics.models.bdlnu.distribution as bdlnu
from bclustering.scan import Scanner
from bclustering.cluster import HierarchyCluster


def main():
    s = Scanner()
    s.set_dfunction(
        bdlnu.dGq2,
        binning=np.linspace(bdlnu.q2min, bdlnu.q2max, 10),
        normalize=True
    )
    s.set_bpoints_equidist(
        {
            "CVL_bctaunutau": (-0.3, 0.3, 10),
            "CSL_bctaunutau": (-0.3, 0.3, 10),
            "CT_bctaunutau": (-0.4, 0.4, 10)
        },
        scale=5,
        eft='WET',
        basis='flavio'
    )
    s.run()

    directory = "output/scan/bdtaunu/our_implementation/q2"
    name = "q2_10_wilson_1000"
    s.write(directory, name, overwrite="ask")

    for max_d in np.linspace(0.01, 1, 20):
        cluster_name = name + "_max_d_{:.3f}".format(max_d)
        c = HierarchyCluster("output/scan", "tutorial_basics")
        c.build_hierarchy()
        c.cluster(max_d=max_d)
        c.write(directory, cluster_name, overwrite="overwrite")


if __name__ == "__main__":
    main()

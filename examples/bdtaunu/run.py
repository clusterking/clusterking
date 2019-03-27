#!/usr/bin/env python3

import sys
sys.path = ["../../"] + sys.path
import numpy as np
import clusterking.physics.models.bdlnu.distribution as bdlnu
from clusterking.scan import WilsonScanner
from clusterking.cluster import HierarchyCluster
from clusterking import Data


def main():
    d = Data()
    s = WilsonScanner()
    s.set_dfunction(
        bdlnu.dGq2,
        binning=np.linspace(bdlnu.q2min, bdlnu.q2max, 10),
        normalize=True
    )
    s.set_spoints_equidist(
        {
            "CVL_bctaunutau": (-0.3, 0.3, 10),
            "CSL_bctaunutau": (-0.3, 0.3, 10),
            "CT_bctaunutau": (-0.4, 0.4, 10)
        },
        scale=5,
        eft='WET',
        basis='flavio'
    )
    s.run(d)

    for max_d in np.linspace(0.01, 1, 20):
        cluster_name = "q2_10_wilson_1000_max_d_{:.3f}".format(max_d)
        c = HierarchyCluster(d)
        c.build_hierarchy()
        c.cluster(max_d=max_d)
        d.write("output/cluster/bdtaunu", cluster_name, overwrite="overwrite")


if __name__ == "__main__":
    main()

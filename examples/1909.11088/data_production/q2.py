#!/usr/bin/env python3

import numpy as np
import clusterking as ck
import clusterking_physics.models.bdlnu.distribution as bdlnu
import flavio

s = ck.scan.WilsonScanner(scale=5, eft="WET", basis="flavio")


def dBrdq2(w, q):
    return flavio.np_prediction("dBR/dq2(B+->Dtaunu)", w, q)


s.set_dfunction(
    dBrdq2, binning=np.linspace(bdlnu.q2min, bdlnu.q2max, 10), normalize=True
)


s.set_spoints_equidist(
    {
        "CVL_bctaunutau": (-0.5, 0.5, 10),
        "CSL_bctaunutau": (-0.5, 0.5, 10),
        "CT_bctaunutau": (-0.1, 0.1, 10),
    }
)
d = ck.Data()
r = s.run(d)
r.write()
d.write("output/q2.sql", overwrite="overwrite")

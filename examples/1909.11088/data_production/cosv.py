#!/usr/bin/env python3

import numpy as np
import clusterking as ck
import flavio

s = ck.scan.WilsonScanner(scale=5, eft="WET", basis="flavio")


def dBRcV(w, cV):
    return flavio.np_prediction("dBR/dcV(B+->D*taunu)", w, cV)


s.set_dfunction(dBRcV, binning=np.linspace(-1, 1, 10), normalize=True)


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
d.write("output/cosv.sql", overwrite="overwrite")

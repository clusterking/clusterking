#!/usr/bin/env python3

import clusterking as ck
import clusterking_physics.models.bdlnu.distribution as bdlnu
import numpy as np
from numpy import sqrt
from wilson.run.smeft.smpar import p
from wilson import Wilson

v = sqrt(1 / (sqrt(2) * p["GF"]))
Yb = 4.2 / v
Ytau = 1.776 / v


def dGEl(bm, el):
    w = Wilson(
        wcdict={"CSR_bctaunutau": -sqrt(2) * Yb * Ytau / 4 / p["GF"] * bm**2},
        scale=5,
        eft="WET",
        basis="flavio",
    )
    return bdlnu.dGEl(w, el)


s = ck.scan.Scanner()
s.set_dfunction(
    dGEl, binning=np.linspace(-bdlnu.Elmin, bdlnu.Elmaxval, 11), normalize=True
)
s.set_spoints_equidist(
    {
        "p": (0, 1, 100),
    }
)
s.set_no_workers(20)
d = ck.Data()
r = s.run(d)
r.write()

d.write("output/el.sql", overwrite="overwrite")

#!/usr/bin/env python3

# std
from pathlib import Path

# 3rd
import matplotlib.pyplot as plt
import matplotlib

# ours
import clusterking as ck
from clusterking.maths.metric import chi2_metric

# Plot style

matplotlib.use("Agg")
plt.rcParams.update({"text.latex.preamble": [r"\usepackage{amsmath}"]})
plt.rc("xtick", **{"top": True, "direction": "in"})
plt.rc("ytick", **{"right": True, "direction": "in"})

# Configure directories

data_dir = Path(__file__).resolve().parent.parent / "data_production" / "output"
output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True, parents=True)

# Load data

s = ck.DataWithErrors(data_dir / "cosv.sql")
s.configure_variable("CVL_bctaunutau", r"$C_{VL}$")
s.configure_variable("CSL_bctaunutau", r"$C_{SL}$")
s.configure_variable("CT_bctaunutau", r"$C_T$")
s.configure_variable("xvar", r"$\cos{(\theta_V)}$")
s.configure_variable(
    "yvar", r"$\mathrm{dBR}(B\to D^{0*}\tau\nu)/\mathrm{d}\cos{(\theta_V)}$"
)

# Add errors

s.add_rel_err_uncorr(0.01)
s.add_err_poisson(700)

# Cluster

c = ck.cluster.HierarchyCluster()
c.set_metric(chi2_metric)
c.set_max_d(1)
c.run(s).write()

# Benchmark

b = ck.Benchmark()
b.set_metric(chi2_metric)
b.run(s).write()

# Plots

s.plot_clusters_scatter(
    ["CT_bctaunutau", "CVL_bctaunutau", "CSL_bctaunutau"]
).savefig(output_dir / "cosV_3D.pdf")

plt.rc("xtick", **{"minor.visible": True})
plt.rc("ytick", **{"minor.visible": True})
s.plot_dist(title="", legend=False, nlines=5).savefig(
    output_dir / "cosV_dist.pdf"
)

bpoints = s.df.loc[s.df["bpoint"] == True][
    ["CVL_bctaunutau", "CSL_bctaunutau", "CT_bctaunutau"]
].round(2)
with (output_dir / "cosl_bpoint_table.txt").open("w") as tf:
    tf.write(bpoints.to_latex())

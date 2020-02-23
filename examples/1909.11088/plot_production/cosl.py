#!/usr/bin/env python3

# std
from pathlib import Path

# 3rd
import matplotlib.pyplot as plt

# ours
import clusterking as ck
from clusterking.maths.metric import chi2_metric

# Plot style

plt.rcParams.update({"text.latex.preamble": [r"\usepackage{amsmath}"]})
plt.rc("xtick", **{"top": True, "direction": "in"})
plt.rc("ytick", **{"right": True, "direction": "in"})

# Configure directories

data_dir = Path(__file__).resolve().parent.parent / "data_production" / "output"
output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True, parents=True)

# Load data

d = ck.DataWithErrors(data_dir / "cosl.sql")
d.configure_variable("CVL_bctaunutau", r"$C_{VL}$")
d.configure_variable("CSL_bctaunutau", r"$C_{SL}$")
d.configure_variable("CT_bctaunutau", r"$C_T$")
d.configure_variable("xvar", r"$\cos{(\theta_\tau)}$")
d.configure_variable(
    "yvar", r"$\mathrm{dBR}(B\to D^{0*}\tau\nu)/\mathrm{d}\cos(\theta_\tau)$"
)

d.add_rel_err_uncorr(0.01)
d.add_err_poisson(700)

c = ck.cluster.HierarchyCluster()
c.set_metric(chi2_metric)

c.set_max_d(1)
c.run(d).write()

b = ck.Benchmark()
b.set_metric(chi2_metric)
b.run(d).write()

# Table
bpoints = d.df.loc[d.df["bpoint"] == True][
    ["CVL_bctaunutau", "CSL_bctaunutau", "CT_bctaunutau"]
].round(2)
with (output_dir / "cosl_bpoint_table.txt").open("w") as tf:
    tf.write(bpoints.to_latex())

# 2D scatter plot
d.plot_clusters_scatter(
    ["CVL_bctaunutau", "CSL_bctaunutau"], max_cols=4, figsize=3
).savefig(output_dir / "cosl_clust2D.pdf")

# Plotting of kinematic distribution
plt.rc("xtick", **{"minor.visible": True})
plt.rc("ytick", **{"minor.visible": True})
d.plot_dist_box(title="", legend=False).savefig(output_dir / "cosl_box.pdf")

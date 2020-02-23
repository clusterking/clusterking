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

# Configure directories

data_dir = Path(__file__).resolve().parent.parent / "data_production" / "output"
output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True, parents=True)

# Load data

d = ck.DataWithErrors(data_dir / "el.sql")
d.configure_variable("CVL_bctaunutau", r"$C_{VL}$")
d.configure_variable("CSR_bctaunutau", r"$C_{SR}$")
d.configure_variable("CT_bctaunutau", r"$C_T$")
d.configure_variable("p", r"$p=\tan(\beta)/{m_{H^\pm}}$ (GeV$^{-1})$")
d.configure_variable("xvar", r"$E_\ell$ (GeV)")
d.configure_variable(
    "yvar", r"$\mathrm{dBR}(B\to D^{0*}\tau\nu)/\mathrm{d}E_\ell}$"
)

# Set up clustering, benchmarking & plotting

c = ck.cluster.HierarchyCluster()
c.set_metric(chi2_metric)
c.set_max_d(1)

b = ck.Benchmark()
b.set_metric(chi2_metric)


def cluster_benchmark_plot(d, iplot):
    c.run(d).write()
    b.run(d).write()

    plt.rc("xtick", **{"top": False, "direction": "out", "minor.visible": True})
    plt.rc(
        "ytick", **{"right": False, "direction": "out", "minor.visible": True}
    )
    d.plot_clusters_scatter(["p"], figsize=8).savefig(
        f"output/El_tanbeta_err{iplot}.pdf"
    )

    plt.rc("xtick", **{"top": True, "direction": "in"})
    plt.rc("ytick", **{"right": True, "direction": "in"})
    d.plot_dist(title="", legend=True, nlines=0).savefig(
        f"output/El_tanbeta_dist{iplot}.pdf"
    )

    plt.rc("xtick", **{"minor.visible": False})
    plt.rc("ytick", **{"minor.visible": False})
    d.plot_bpoint_distance_matrix(metric=chi2_metric).savefig(
        f"output/El_tanbeta_dist{iplot}_bpoint_distances.pdf"
    )


# Run for different scenarios

d.reset_errors()
d.add_rel_err_uncorr(0.01)
d.add_err_poisson(1000)
cluster_benchmark_plot(d, 1)

d.reset_errors()
d.add_rel_err_uncorr(0.01)
d.add_err_poisson(1800)
cluster_benchmark_plot(d, 2)

d.reset_errors()
d.add_rel_err_uncorr(0.01)
d.add_err_poisson(2000)
cluster_benchmark_plot(d, 3)

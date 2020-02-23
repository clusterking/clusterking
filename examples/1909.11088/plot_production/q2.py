#!/usr/bin/env python3

# std
from pathlib import Path

# 3rd
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

# ours
import clusterking as ck
from clusterking.maths.metric import chi2_metric

# Plot style

plt.rc("xtick", **{"top": True, "minor.visible": True, "direction": "in"})
plt.rc("ytick", **{"right": True, "minor.visible": True, "direction": "in"})

# Configure directories

data_dir = Path(__file__).resolve().parent.parent / "data_production" / "output"
output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True, parents=True)


d = ck.DataWithErrors(data_dir / "q2.sql")
c = ck.cluster.HierarchyCluster()
c.set_metric(chi2_metric)
c.set_max_d(1)


def plot(max_yield, rel_err=0.0):
    x_axis = []
    y_axis = []
    avg_cluster_size = []
    for i in tqdm(range(1, max_yield, 1)):
        d.reset_errors()
        d.add_rel_err_uncorr(rel_err)
        d.add_err_poisson(i)

        r = c.run(d)
        r.write()

        x_axis.append(i)
        y_axis.append(len(d.clusters()))
    return x_axis, y_axis


# Plot

if (output_dir / "q2_clust_yield_x.npy").is_file():
    print("Warning: Reloading q2 results!")
    x = np.load(output_dir / "q2_clust_yield_x.npy")
    y1 = np.load(output_dir / "q2_clust_yield_y1.npy")
    y2 = np.load(output_dir / "q2_clust_yield_y2.npy")
    y3 = np.load(output_dir / "q2_clust_yield_y3.npy")
else:
    x, y1 = plot(5000,)
    p.save(output_dir / "q2_clust_yield_x", x)
    np.save(output_dir / "q2_clust_yield_y1", y1)
    _, y2 = plot(5000, rel_err=0.015)
    np.save(output_dir / "q2_clust_yield_y2", y2)
    _, y3 = plot(5000, rel_err=0.1)
    np.save(output_dir / "q2_clust_yield_y3", y3)


plt.plot(x, y1, "-", color="xkcd:light red", label=r"$\sigma_r$ = 0% ")
plt.plot(x, y2, "-", color="xkcd:apple green", label=r"$\sigma_r$ = 1.5% ")
plt.plot(x, y3, "-", color="xkcd:bright blue", label=r"$\sigma_r$ = 10% ")

plt.xlabel("Yield")
plt.ylabel("Number of clusters")
plt.legend(frameon=False)

plt.savefig(output_dir / "clust_yield.pdf")

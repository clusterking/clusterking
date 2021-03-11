#!/usr/bin/env python3

"""
This plots the results of jsd_all_points.py

Usage: python3 jsd_all_points.py
"""

# std
from pathlib import Path

# 3rd
import numpy as np
import matplotlib.pyplot as plt


try:
    plt.style.use("myscience")
except FileNotFoundError:
    pass


exp_path = Path(__file__).resolve().parent

results = {}
for f in exp_path.glob("*.npy"):
    results[f.stem] = np.load(str(f))


def _set_labels(ax):
    ax.set_xlabel(r"JSD(toy $\chi^2_r/r$ || expected $\chi^2_r/r$)")
    ax.set_ylabel("Number of generated distributions")


key2label = {
    "cosl_std": r"$\cos\,\theta_\ell$ (700/1%)",
    "cosv_std": r"$\cos\,\theta_V$  (700/1%)",
    "el_std1": r"$E_\ell$ (1000/1%)",
    "el_std2": r"$E_\ell$ (1800/1%)",
    "el_std3": r"$E_\ell$ (2000/1%)",
    "q2_basic": r"$q^2$ (700/1%)",
}

linestyles = ["solid", "dashed", "dashdot", (0, (3, 1, 1, 1, 1, 1))]

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
for i, key in enumerate(["cosl_std", "cosv_std", "q2_basic"]):
    ax[0].hist(results[key], label=key2label[key], density=False, histtype="step", lw=1.5, bins=20, ls=linestyles[i])
_set_labels(ax[0])
ax[0].set_xlim(None, 0.037)
ax[0].legend(loc="upper right")
for i, key in enumerate(["el_std1", "el_std2", "el_std3"]):
    ax[1].hist(results[key], label=key2label[key], density=False, histtype="step", lw=1.5, bins=8, ls=linestyles[i])
_set_labels(ax[1])
ax[1].legend(loc="upper right")
fig.suptitle(
    r"Jenson-Shannon divergence (JSD) between generated distribution of $\chi^2_r/r$ and theory expectation")

fig.savefig("jsd_all.pdf")

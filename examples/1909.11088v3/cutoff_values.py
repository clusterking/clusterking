#!/usr/bin/env python3

"""
Usage: python3 cutoff_values.py

Generates the figure of cutoff value vs bins vs p value that corresponds
to distinguishability.
"""

import matplotlib.pyplot as plt
import scipy.stats
import numpy as np


try:
    plt.style.use("myscience")
except FileNotFoundError:
    pass


def create_cutoff_plot(filename="cutoff_values.pdf"):
    dfs = np.arange(1, 30)
    fig, ax = plt.subplots()
    shapes = [
        "-",
        "-.",
        "--",
        ":",
    ]
    for i, fp in enumerate([0.05, 0.1, 0.2, 0.35]):
        cutoffs = [scipy.stats.chi2.ppf(1 - fp, df=df) / df for df in dfs]
        ax.plot(dfs, cutoffs, label=rf"$p={100*fp:.0f}\%$", ls=shapes[i])
    ax.legend()
    ax.set_xlabel("Degrees of freedom $r$")
    ax.set_ylabel("Cutoff $c$")
    ax.set_title(
        r"Cutoff values $c$ for fixed $p$ values $P(\chi^2_r/r > c\, |\, H_0)$"
    )
    if filename:
        fig.savefig(filename)


if __name__ == "__main__":
    create_cutoff_plot()

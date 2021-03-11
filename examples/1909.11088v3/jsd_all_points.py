#!/usr/bin/env python3

""" This script takes the output files from ClusterKinG that corresponded to
the results shown in the paper and performs toy experiments as described in
Appendix C of 1909.11088v3.

You will need to generate these output files first and put them in this
directory.
"""

# std
from pathlib import Path

# 3rd
import scipy.stats
import scipy.spatial
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ours
import clusterking as ck
from clusterking.maths.metric import chi2


def generate_toy_dataset(n, cov, n_toys=1000):
    assert n.ndim == 1
    n_bins = n.size
    assert cov.shape == (n_bins, n_bins)
    return scipy.stats.multivariate_normal.rvs(mean=n, cov=cov, size=n_toys)


def _get_binned_theoretical_chi2_distribution(
    dof: int, bins: np.ndarray, normalize=True
) -> np.ndarray:
    """

    Args:
        dof: Degrees of freedom
        bins: Binning of chi2 distribution
        normalize

    Returns:

    """
    vals = scipy.stats.chi2.cdf(bins, df=dof)
    bin_contents = vals[1:] - vals[:-1]
    if normalize:
        bin_contents /= bin_contents.sum()
    return bin_contents


def validate_point(
    dwe, index, n_toys=10000, hist_bins=40, plot=False, normalize=True
):
    n = dwe.data()[index, :]
    cov = dwe.cov(relative=False)[index, :, :]
    toys = generate_toy_dataset(n=n, cov=cov, n_toys=n_toys)

    chi2s = chi2(toys, n, cov, np.zeros_like(cov), normalize=normalize)
    n_bins = n.size
    dof = n_bins - 1 if normalize else n_bins
    bins = np.linspace(0, 4, hist_bins)
    ourvals, _ = np.histogram(chi2s / dof, bins=bins,)
    ourvals = ourvals / ourvals.sum()

    theo_expect = _get_binned_theoretical_chi2_distribution(
        dof=dof, bins=bins * dof
    )
    pv = scipy.spatial.distance.jensenshannon(ourvals, theo_expect)

    assert theo_expect.size == ourvals.size

    if plot:
        fig, ax = plt.subplots()
        ax.step(
            bins,
            np.hstack((0.0, theo_expect)),
            color="black",
            lw=1.5,
            label=rf"Theo $\chi^2_r/r$ with $r={dof}$",
        )
        ax.step(
            bins, np.hstack((0.0, ourvals)), color="red", lw=1.5, label="Ours"
        )
        ax.legend()

    return pv


def validate_all_points(dwe, **kwargs):
    return [
        validate_point(dwe, index, **kwargs)
        for index in tqdm(range(len(dwe.df)))
    ]


def validate_and_save(dwe, path, force=False, **kwargs):
    path = Path(path)
    if not path.suffix == ".npy":
        path = Path(path.name + ".npy")
    if path.is_file() and not force:
        print(f"Loaded from {path}")
        return np.load(path)
    else:
        to_save = np.array(validate_all_points(dwe, **kwargs))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, to_save)


if __name__ == "__main__":
    # Adapt this if your output files live somewhere else
    data_dir = Path(".").resolve()
    assert data_dir.is_dir()

    d = ck.DataWithErrors(data_dir / "cosl.sql")

    d.reset_errors()
    d.add_rel_err_uncorr(0.01)
    d.add_err_poisson(700)
    validate_and_save(d, "cosl_std")

    d.reset_errors()
    d.add_rel_err_uncorr(0.01)
    d.add_rel_err_maxcorr(0.01)
    d.add_err_poisson(700)
    validate_and_save(d, "cosl_std_plus_mc")

    d = ck.DataWithErrors(data_dir / "cosv.sql")

    d.reset_errors()
    d.add_rel_err_uncorr(0.01)
    d.add_err_poisson(700)
    validate_and_save(d, "cosv_std")

    d.reset_errors()
    d.add_rel_err_uncorr(0.01)
    d.add_rel_err_maxcorr(0.01)
    d.add_err_poisson(700)
    validate_and_save(d, "cosv_std_plus_mc")

    d = ck.DataWithErrors(data_dir / "el.sql")

    d.reset_errors()
    d.add_rel_err_uncorr(0.01)
    d.add_err_poisson(700)
    validate_and_save(d, "el_std", force=True)

    d.reset_errors()
    d.add_rel_err_uncorr(0.01)
    d.add_rel_err_maxcorr(0.01)
    d.add_err_poisson(700)
    validate_and_save(d, "el_std_plus_mc", force=True)

    d.reset_errors()
    d.add_rel_err_uncorr(0.01)
    d.add_err_poisson(1000)
    validate_and_save(d, "el_std1")

    d.reset_errors()
    d.add_rel_err_uncorr(0.01)
    d.add_err_poisson(1800)
    validate_and_save(d, "el_std2")

    d.reset_errors()
    d.add_rel_err_uncorr(0.01)
    d.add_err_poisson(2000)
    validate_and_save(d, "el_std3")

    d = ck.DataWithErrors(data_dir / "q2.sql")

    d.reset_errors()
    d.add_rel_err_uncorr(0.01)
    d.add_err_poisson(700)
    validate_and_save(d, "q2_basic")

    d.reset_errors()
    d.add_rel_err_uncorr(0.01)
    d.add_rel_err_maxcorr(0.01)
    d.add_err_poisson(700)
    validate_and_save(d, "q2_basic_plus_mc")

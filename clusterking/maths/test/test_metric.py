#!/usr/bin/env python3

# std
from functools import partial

# 3rd
import pytest
import numpy as np
import scipy.stats

# ours
from clusterking.maths.metric import chi2


_metrics_to_test = [
    partial(chi2, normalize=False),
    partial(chi2, normalize=True),
]


def random_correlation_matrix(n):
    evs = np.random.random(n)
    evs /= evs.sum()
    evs *= n
    return scipy.stats.random_correlation.rvs(evs)


@pytest.mark.parametrize("metric", _metrics_to_test)
def test_metric_vanishes_identical(metric):
    n_obs = 10
    for n_bins in [2, 5, 10]:
        n1 = np.random.random(size=(n_obs, n_bins))
        rele1 = np.random.random(size=(n_obs, n_bins))
        e1 = rele1 * n1
        corr = np.full((n_obs, n_bins, n_bins), np.nan)
        for i in range(n_obs):
            corr[i, :, :] = random_correlation_matrix(n_bins)
        cov1 = np.einsum("ni,nij,nj->nij", e1, corr, e1)
        # This test also tests the different signatures
        assert np.isclose(metric(n1=n1, n2=n1, cov1=cov1, cov2=cov1), 0.0).all()
        assert np.isclose(
            metric(n1=n1, n2=n1, cov1=cov1, cov2=np.zeros_like(cov1)), 0.0
        ).all()
        assert np.isclose(
            metric(n1=n1, n2=n1, cov1=cov1, cov2=cov1[0]), 0.0
        ).all()
        assert np.isclose(
            metric(n1=n1, n2=n1, cov1=cov1[0], cov2=cov1[0]), 0.0
        ).all()
        assert np.isclose(
            metric(n1=n1, n2=n1, cov1=cov1[0], cov2=cov1), 0.0
        ).all()
        assert np.isclose(
            metric(
                n1=n1[0].reshape((1, n_bins)),
                n2=n1[0],
                cov1=cov1[0],
                cov2=cov1[0],
            ),
            0.0,
        ).all()


@pytest.mark.parametrize("metric", _metrics_to_test)
def test_metric_symmetric(metric):
    n_experiments = 10
    n_obs = 10
    n_bins = 10
    for iexp in range(n_experiments):
        nj = []
        covj = []
        for i in range(2):
            n = np.random.random(size=(n_obs, n_bins))
            rele = np.random.random(size=(n_obs, n_bins))
            e = rele * n
            corr = random_correlation_matrix(n_bins)
            cov = np.einsum("ni,ij,nj->nij", e, corr, e)
            nj.append(n)
            covj.append(cov)
        chi2s1 = chi2(nj[0], nj[1], covj[0], covj[1])
        chi2s2 = chi2(nj[1], nj[0], covj[1], covj[0])
        assert np.isclose(chi2s1, chi2s2).all()


def generate_toy_dataset(
    base_hist: np.ndarray, cov: np.ndarray, n_toys=1000
) -> np.ndarray:
    """ Generate toy dataset around base_hist

    Args:
        base_hist: Expectation value
        cov: Covariance matrix
        n_toys: Number of toys to generate

    Returns:
        (n_obs, n_bins) array
    """
    assert base_hist.ndim == 1
    n_bins = base_hist.size
    assert cov.shape == (n_bins, n_bins)
    return scipy.stats.multivariate_normal.rvs(
        mean=base_hist, cov=cov, size=n_toys
    )


def _get_binned_theoretical_chi2_distribution(
    dof: int, bins: np.ndarray
) -> np.ndarray:
    """

    Args:
        dof: Degrees of freedom
        bins: Binning of chi2 distribution

    Returns:

    """
    vals = scipy.stats.chi2.cdf(bins, df=dof)
    bin_contents = vals[1:] - vals[:-1]
    bin_contents /= bin_contents.sum()
    return bin_contents


@pytest.mark.slow
@pytest.mark.parametrize("errors", ("statonly", "uncorrrel", "statrel"))
@pytest.mark.parametrize("normalize", [True, False])
def test_chi2_distribution(normalize, errors):
    n_toys = 10000
    n_bins = 10
    base_hist = 100 * np.array([1, 2, 3, 4, 5, 6, 4, 3, 2, 1])
    assert base_hist.size == n_bins

    n_experiments = 1 if not errors == "statrel" else 4

    for i_exp in range(n_experiments):
        if errors == "statonly":
            e = np.sqrt(base_hist)
            corr = np.eye(n_bins)
            cov = np.einsum("i,ij,j->ij", e, corr, e)
        elif errors == "uncorrrel":
            e = 0.01 * base_hist
            corr = np.eye(n_bins)
            cov = np.einsum("i,ij,j->ij", e, corr, e)
        elif errors == "statrel":
            e1 = 0.01 * base_hist
            cov1 = np.einsum("i,j->ij", e1, e1)
            e2 = 0.01 * base_hist
            corr = random_correlation_matrix(n_bins)
            cov2 = np.einsum("i,ij,j->ij", e2, corr, e2)
            cov = cov1 + cov2
        else:
            raise ValueError("Invalid test parameter")

        toys = generate_toy_dataset(base_hist, cov, n_toys=n_toys)

        chi2s = chi2(
            toys, base_hist, cov, np.zeros_like(cov), normalize=normalize
        )

        bins = np.linspace(0, 20, 30)
        ourvals, _ = np.histogram(chi2s, bins=bins,)
        ourvals = ourvals / ourvals.sum()

        dof = n_bins - 1 if normalize else n_bins
        theo_expect = _get_binned_theoretical_chi2_distribution(
            dof=dof, bins=bins
        )

        ks, pv = scipy.stats.ks_2samp(ourvals, theo_expect)

        assert pv > 0.95, (ourvals, theo_expect)

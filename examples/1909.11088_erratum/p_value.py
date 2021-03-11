#!/usr/bin/env python3

import scipy.stats


def calc_p_value_paper_version(bins, cutoff=1.):
    return 1-scipy.stats.chi2.cdf(bins*cutoff, df=bins-1)

def calc_cutoff_vs_bins_pval(bins, pval):
    df = bins - 1
    return scipy.stats.chi2.ppf(1-pval, df=df)/df

nbins = 9
print(f"{nbins} bins")
p_val = calc_p_value_paper_version(nbins)
print(p_val)
cutoff = calc_cutoff_vs_bins_pval(nbins, p_val)
print(f"Cutoff value for corrected metric: {cutoff}")

#!/usr/bin/env python3

from .histogram import Histogram


class Metric(object):
    def __init__(self):
        # config values etc. etc.
        pass

    def calc(self, hist_a: Histogram, hist_b: Histogram) -> float:
        # should be implemented in subclass
        raise NotImplementedError


class Chi2Metric(Metric):
    def __init__(self):
        super().__init__()

    def calc(self, hist_a, hist_b):
        hist_a = hist_a.get_uncorr_hist()
        hist_b = hist_b.get_uncorr_hist()
        return self._calc_uncorr(hist_a, hist_b)

    def _calc_uncorr(self, hist_a: Histogram, hist_b: Histogram):
        # maybe like https://root.cern.ch/doc/master/classTH1.html#a6c281eebc0c0a848e7a0d620425090a5
        # (Two weighted histograms comparison)
        pass


# class MetricFrom2HDMPaper(....)


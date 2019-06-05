#!/usr/bin/env python3

# std
from typing import Optional, Callable

# 3rd
import tqdm
import pandas as pd

# ours
from clusterking.stability.clustermatcher import (
    ClusterMatcher,
    TrivialClusterMatcher,
)


class SubSampleStabilityTesterResult(object):
    def __init__(self, df: pd.DataFrame):
        #: Results as :class:`pandas.DataFrame`
        self.df = df


def default_fom(clustered1, clustered2):
    assert len(clustered1) == len(clustered2)
    return sum(clustered1 == clustered2) / len(clustered1)


# todo: allow adding several foms and cluster matchers
class SubSampleStabilityTester(object):
    """ Test the stability of clustering algorithms by repeatedly
    clustering subsamples of data and then comparing if the clusters match.

    """

    def __init__(self):
        #: Fraction of sample points to be contained in the subsamples.
        #: Set using :meth:`set_basic_config`.
        self._fraction = None
        #: Number of subsamples to consider.
        #: Set using :meth:`set_basic_config`.
        self._repeat = None
        #: Cluster matcher object to be used
        self._cluster_matcher = None
        #: Figure of merit to calculate
        self._fom = None
        #: Display a progress bar?
        self._progress_bar = True

        # Set default values:
        self.set_basic_config()
        self.set_cluster_matcher()
        self.set_fom()
        self.set_progress_bar()

    # **************************************************************************
    # Config
    # **************************************************************************

    def set_basic_config(self, fraction=0.75, repeat=100) -> None:
        """ Basic configuration

        Args:
            fraction: Fraction of sample points to be contained in the subsamples
            repeat: Number of subsamples to test

        Returns:
            None
        """
        assert 0 < fraction < 1
        self._fraction = fraction
        self._repeat = repeat

    def set_cluster_matcher(
        self, matcher: Optional[ClusterMatcher] = None
    ) -> None:
        """ Set cluster matcher (matches cluster names of two different
        clusterings

        Args:
            matcher: :class:`~clusterking.stability.clustermatcher.ClusterMatcher`
                object. Default: TrivialClusterMatcher.

        Returns:
            None
        """
        if matcher is None:
            matcher = TrivialClusterMatcher
        self._cluster_matcher = matcher

    def set_fom(self, fct: Optional[Callable] = None) -> None:
        """ Set figure of merit to be calculated.

        Args:
            fct: Function that takes two clusterings and returns a float.

        Returns:
            None
        """
        if fct is None:
            fct = default_fom
        self._fom = fct

    def set_progress_bar(self, state=True) -> None:
        """ Set or unset progress bar.

        Args:
            state: Bool: Display progress bar?

        Returns:
            None
        """
        self._progress_bar = state

    # **************************************************************************
    # Run
    # **************************************************************************

    def run(self, data, cluster_worker):
        """ Run test.

        Args:
            data: `~clusterking.data.Data` object
            cluster_worker: Pre-configured `~clusterking.cluster.Cluster`
                object

        Returns:
            :class:`~clusterking.stability.subsamplestability.SubSampleStabilityTesterResult` object
        """
        foms = []
        nclusters = []
        match_losts = []
        default_clusters = cluster_worker.run(data).get_clusters(indexed=True)
        matcher = self._cluster_matcher()
        if self._progress_bar:
            iterator = tqdm.tqdm(range(self._repeat))
        else:
            iterator = range(self._repeat)
        for i in iterator:
            r = cluster_worker.run(
                data.sample_param_random(frac=self._fraction)
            )
            clusters = r.get_clusters(indexed=True)
            relevant_default_clusters = default_clusters[clusters.index]
            rename_dct = matcher.run(clusters, relevant_default_clusters).dct
            clusters_renamed = clusters.map(rename_dct)
            fom = self._fom(clusters_renamed, relevant_default_clusters)
            foms.append(fom)
            ncluster = len(set(clusters))
            nclusters.append(ncluster)
            match_lost = len(rename_dct.keys()) - len(set(rename_dct.values()))
            match_losts.append(match_lost)
        df = pd.DataFrame()
        df["fom"] = foms
        df["nclusters"] = nclusters
        df["match_lost"] = match_losts
        return SubSampleStabilityTesterResult(df=df)


class SubSampleStabilityVsFractionResult(object):
    def __init__(self, df: pd.DataFrame):
        #: Results as :class:`pandas.DataFrame`
        self.df = df


class SubSampleStabilityVsFraction(object):
    """ Repeatedly run :class:`SubSampleStabilityTester` for different
    fractions.
    """

    def __init__(self):
        pass

    def run(self, data, cluster, ssst, fractions):
        x = []
        fom = []
        nclusters = []
        ssst.set_progress_bar(False)
        for fract in tqdm.tqdm(fractions):
            x.append(fract)
            ssst.set_basic_config(fraction=fract, repeat=200)
            r = ssst.run(data, cluster)
            fom.append(r.df["fom"].mean())
            nclusters.append(r.df["nclusters"].mean())
        df = pd.DataFrame()
        df["fract"] = x
        df["fom"] = fom
        df["nclusters"] = nclusters
        return SubSampleStabilityVsFractionResult(df=df)

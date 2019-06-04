#!/usr/bin/env python3

# 3rd
import tqdm
import pandas as pd

# ours
from clusterking.stability.clustermatcher import TrivialClusterMatcher


class SubSampleStabilityTesterResult(object):
    def __init__(self, df):
        self.df = df


def default_fom(clustered1, clustered2):
    assert len(clustered1) == len(clustered2)
    return sum(clustered1 == clustered2) / len(clustered1)


class SubSampleStabilityTester(object):
    def __init__(self):
        self._fraction = None
        self._repeat = None
        self.set_basic_config()
        self._cluster_matcher = None
        self.set_cluster_matcher()
        self._fom = None
        self.set_fom()
        self._progress_bar = None
        self.set_progress_bar()

    # **************************************************************************
    # Config
    # **************************************************************************

    def set_basic_config(self, fraction=0.75, repeat=100):
        assert 0 < fraction < 1
        self._fraction = fraction
        self._repeat = repeat

    def set_cluster_matcher(self, matcher=TrivialClusterMatcher):
        self._cluster_matcher = matcher

    def set_fom(self, fct=default_fom):
        self._fom = fct

    def set_progress_bar(self, state=True):
        self._progress_bar = state

    # **************************************************************************
    # Run
    # **************************************************************************

    def run(self, data, worker):
        foms = []
        nclusters = []
        match_losts = []
        default_clusters = worker.run(data).get_clusters(indexed=True)
        matcher = self._cluster_matcher()
        if self._progress_bar:
            iterator = tqdm.tqdm(range(self._repeat))
        else:
            iterator = range(self._repeat)
        for i in iterator:
            r = worker.run(data.sample_param_random(frac=self._fraction))
            clusters = r.get_clusters(indexed=True)
            relevant_default_clusters = default_clusters[clusters.index]
            rename_dct = matcher.run(clusters, relevant_default_clusters)
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
    def __init__(self, df):
        self.df = df


class SubSampleStabilityVsFraction(object):
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

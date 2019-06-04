#!/usr/bin/env python3

# 3rd
import tqdm

# ours
from clusterking.stability.clustermatcher import TrivialClusterMatcher
from clusterking.util.matplotlib_utils import import_matplotlib


class SubSampleStabilityTesterResult(object):
    def __init__(self, foms, nclusters):
        self.foms = foms
        self.nclusters = nclusters

    def hist_foms(self):
        import_matplotlib()
        import matplotlib.pyplot as plt

        plt.hist(self.foms, density=True)

    def hist_nclusters(self):
        import_matplotlib()
        import matplotlib.pyplot as plt

        plt.hist(self.nclusters, density=True)


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

    def set_basic_config(self, fraction=0.75, repeat=100):
        assert 0 < fraction < 1
        self._fraction = fraction
        self._repeat = repeat

    def set_cluster_matcher(self, matcher=TrivialClusterMatcher):
        self._cluster_matcher = matcher

    def set_fom(self, fct=default_fom):
        self._fom = fct

    def run(self, data, worker):
        foms = []
        nclusters = []
        default_clusters = worker.run(data).get_clusters(indexed=True)
        matcher = self._cluster_matcher()
        for i in tqdm.tqdm(range(self._repeat)):
            r = worker.run(data.sample_param_random(frac=self._fraction))
            clusters = r.get_clusters(indexed=True)
            relevant_default_clusters = default_clusters[clusters.index]
            rename_dct = matcher.run(clusters, relevant_default_clusters)
            clusters_renamed = clusters.map(rename_dct)
            fom = self._fom(clusters_renamed, relevant_default_clusters)
            foms.append(fom)
            ncluster = len(set(clusters))
            nclusters.append(ncluster)
        return SubSampleStabilityTesterResult(foms=foms, nclusters=nclusters)

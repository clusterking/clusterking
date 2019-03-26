#!/usr/bin/env python3

# 3rd
import sklearn

# ours
from clusterking.cluster.cluster import Cluster


# todo: document!
class KmeansCluster(Cluster):
    def __init__(self, data):
        super().__init__(data)

    def _cluster(self, **kwargs):
        kmeans = sklearn.cluster.KMeans(**kwargs)
        matrix = self.data.data()
        kmeans.fit(matrix)
        return kmeans.predict(matrix)

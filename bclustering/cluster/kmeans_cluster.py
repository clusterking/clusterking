#!/usr/bin/env python3

# 3rd
import numpy as np
import sklearn

# ours
from bclustering.cluster.cluster import Cluster


# todo: document!
class KmeansCluster(Cluster):
    def __init__(self):
        super().__init__()

    def _cluster(self, data, **kwargs):
        kmeans = sklearn.cluster.KMeans(**kwargs)
        bin_columns = [col for col in data.df.columns if col.startswith("bin")]
        x_matrix = np.array(data.df[bin_columns].astype(float))
        kmeans.fit(x_matrix)
        return kmeans.predict(x_matrix)

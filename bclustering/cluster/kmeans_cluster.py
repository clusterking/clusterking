#!/usr/bin/env python3

# 3rd
import numpy as np
import sklearn

# ours
from bclustering.cluster.cluster import Cluster


# todo: document!
class KmeansCluster(Cluster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _cluster(self, **kwargs):
        kmeans = sklearn.cluster.KMeans(**kwargs)
        bin_columns = [col for col in self.df.columns if col.startswith("bin")]
        x_matrix = np.array(self.df[bin_columns].astype(float))
        kmeans.fit(x_matrix)
        return kmeans.predict(x_matrix)

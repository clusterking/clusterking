#!/usr/bin/env python3

# 3rd
import sklearn.cluster

# ours
from clusterking.cluster.cluster import Cluster, ClusterResult
from clusterking.util.metadata import failsafe_serialize, nested_dict


class KmeansClusterResult(ClusterResult):
    pass


class KmeansCluster(Cluster):
    """ Kmeans clustering
    (`wikipedia <https://en.wikipedia.org/wiki/K-means_clustering>`_) as
    implemented in :mod:`sklearn.cluster`.

    Example:

    .. code-block:: python

        import clusterking as ck
        d = ck.Data("/path/to/data.sql")    # Load some data
        c = ck.cluster.KmeansCluster()      # Init worker class
        c.set_kmeans_options(n_clusters=5)  # Set options for clustering
        r = c.run(d)                        # Perform clustering on data
        r.write()                           # Write results back to data

    """

    def __init__(self):
        super().__init__()
        self._kmeans_kwargs = {}
        self.md = nested_dict()

    def set_kmeans_options(self, **kwargs) -> None:
        """ Configure clustering algorithms.

        Args:
            **kwargs: Keyword arguments to :func:`sklearn.cluster.KMeans`.
        """
        self._kmeans_kwargs = kwargs
        self.md["kmeans"]["kwargs"] = failsafe_serialize(kwargs)

    def run(self, data) -> KmeansClusterResult:
        kmeans = sklearn.cluster.KMeans(**self._kmeans_kwargs)
        matrix = data.data()
        kmeans.fit(matrix)
        return KmeansClusterResult(
            data=data, md=self.md, clusters=kmeans.predict(matrix)
        )

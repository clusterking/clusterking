#!/usr/bin/env python3

# std
import pathlib
from typing import Union, Callable
import functools

# 3rd
import scipy.cluster
import matplotlib.pyplot as plt
import scipy.spatial
import numpy as np

# ours
from bclustering.cluster.cluster import Cluster
from bclustering.util.metadata import failsafe_serialize
from bclustering.maths.metric import uncondense_distance_matrix


# todo: Function to save/load hierarchy?

# todo: document
class HierarchyCluster(Cluster):
    def __init__(self, data):
        super().__init__(data)

        self.hierarchy = None
        #: Function that, applied to Data or DWE object returns the metric as
        #: a condensed distance matrix.
        self.metric = None  # type: Callable

    def set_metric(self, *args, **kwargs) -> None:
        self.md["metric"]["args"] = failsafe_serialize(args)
        self.md["metric"]["kwargs"] = failsafe_serialize(kwargs)
        if len(args) == 0:
            # default
            args = ['euclidean']
        if isinstance(args[0], str):
            # The user can specify any of the metrics from
            # scipy.spatial.distance.pdist by name and supply additional
            # values
            self.metric = lambda data: scipy.spatial.distance.pdist(
                data.data(),
                args[0],
                *args[1:],
                **kwargs
            )
        elif isinstance(args[0], Callable):
            # Assume that this is a function that takes DWE or Data as first
            # argument
            self.metric = functools.partial(args[0], *args[1:], **kwargs)
        else:
            raise ValueError(
                "Invalid type of first argument: {}".format(type(args[0]))
            )

    def build_hierarchy(self, method="complete", optimal_ordering=False) -> None:
        """ Build the hierarchy object.

        Args:
            method: See reference on scipy.cluster.hierarchy.linkage
            optimal_ordering: See reference on scipy.cluster.hierarchy.linkage
        """
        if self.metric is None:
            self.log.error(
                "Metric not set. please run self.set_metric or set "
                "self.metric manually before running this method. "
                "Returning without doing anything."
            )
            return

        self.log.debug("Building hierarchy.")

        md = self.md["hierarchy"]
        md["method"] = method
        md["optimal_ordering"] = optimal_ordering

        self.hierarchy = scipy.cluster.hierarchy.linkage(
            self.metric(self.data),
            method=method,
            optimal_ordering=optimal_ordering
        )

        self.log.debug("Done")

    def _cluster(self, max_d=0.2, **kwargs):
        """Performs the actual clustering
        Args:
            max_d:
            **kwargs:

        Returns:
            None
        """

        if self.hierarchy is None:
            msg = "Please run build_hierarchy first to set self.hierarchy or" \
                  "manually set HieararchyCluster.hierachy."
            self.log.critical(msg)
            raise ValueError(msg)

        # set up defaults for clustering here
        # (this way we can overwrite them with additional arguments)
        fcluster_config = {
            "criterion": "distance"
        }
        fcluster_config.update(kwargs)
        # noinspection PyTypeChecker
        clusters = scipy.cluster.hierarchy.fcluster(
            self.hierarchy,
            max_d,
            **fcluster_config
        )

        return clusters

    def _select_bpoints(self, **kwargs):
        """ Select one benchmark point for each cluster.

        Args:
            data: Data object
            column: Column to write to (True if is benchmark point, False other
                sise)
        """
        m = np.sum(uncondense_distance_matrix(self.metric(self.data)), axis=1)
        result = np.full(self.data.n, False)
        for cluster in set(self.clusters):
            # The indizes of all wpoints that are in the current cluster
            indizes = np.argwhere(self.clusters == cluster).squeeze()
            # The index of the wpoint of the current cluster that has the lowest
            # sum of distances to all other elements in the same cluster
            index_minimal = indizes[np.argmin(m[indizes])]
            result[index_minimal] = True
        return result

    def dendrogram(
            self,
            output: Union[None, str, pathlib.Path] = None,
            ax=None,
            show=False,
            **kwargs
    ) -> Union[plt.Axes, None]:
        """Creates dendrogram

        Args:
            output: If supplied, we save the dendrogram there
            ax: An axes object if you want to add the dendrogram to an existing
                axes rather than creating a new one
            show: If true, the dendrogram is shown in a viewer.
            **kwargs: Additional keyword options to
                scipy.cluster.hierarchy.dendrogram

        Returns:
            The matplotlib.pyplot.Axes object
        """
        self.log.debug("Plotting dendrogram.")
        if self.hierarchy is None:
            self.log.error("Hierarchy not yet set up. Returning without "
                           "doing anything.")
            return

        # do we add to a plot or generate a whole new figure?
        if ax:
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots()

        labelsize = 20
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=labelsize)
        ax.set_xlabel('ID', fontsize=labelsize)
        ax.set_ylabel('Distance', fontsize=labelsize)

        # set defaults for dendrogram plotting options here
        # (this way we can overwrite them with additional arguments)
        den_config = {
            "color_threshold": "default",
            "leaf_rotation": 90.,  # rotates the x axis labels
            "leaf_font_size": 8,   # font size for the x axis labels
        }
        den_config.update(kwargs)

        scipy.cluster.hierarchy.dendrogram(
            self.hierarchy,
            ax=ax,
            **den_config
        )

        if show:
            fig.show()

        if output:
            output = pathlib.Path(output)
            if not output.parent.is_dir():
                self.log.debug("Creating dir '{}'.".format(output.parent))
                output.parent.mkdir(parents=True)
            fig.savefig(output, bbox_inches="tight")
            self.log.info("Wrote dendrogram to '{}'.".format(output))

        return ax

#!/usr/bin/env python3

# std
import pathlib
from typing import Union, Callable

# 3rd
import scipy.cluster
import matplotlib.pyplot as plt
import scipy.spatial

# ours
from clusterking.cluster.cluster import Cluster
from clusterking.util.metadata import failsafe_serialize
from clusterking.maths.metric import metric_selection


# todo: document
class HierarchyCluster(Cluster):
    def __init__(self, data):
        """
        Args:
            data: :py:class:`~clusterking.data.data.Data` object
        """
        super().__init__(data)

        self.hierarchy = None
        #: Function that, applied to Data or DWE object returns the metric as
        #: a condensed distance matrix.
        self.metric = None  # type: Callable

    # Docstring set below
    def set_metric(self, *args, **kwargs) -> None:
        self.md["metric"]["args"] = failsafe_serialize(args)
        self.md["metric"]["kwargs"] = failsafe_serialize(kwargs)
        self.metric = metric_selection(*args, **kwargs)

    set_metric.__doc__ = metric_selection.__doc__

    def build_hierarchy(self, method="complete", optimal_ordering=False) \
            -> None:
        """ Build the hierarchy object.

        Args:
            method: See reference on scipy.cluster.hierarchy.linkage
            optimal_ordering: See reference on scipy.cluster.hierarchy.linkage

        """
        if self.metric is None:
            msg = "Metric not set. please run self.set_metric or set " \
                  " self.metric manually before running this method. " \
                  "Returning without doing anything."
            self.log.critical(msg)
            raise ValueError(msg)

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

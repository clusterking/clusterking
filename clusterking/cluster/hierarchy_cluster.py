#!/usr/bin/env python3

# std
import pathlib
from typing import Union, Callable, Optional

# 3rd
import scipy.cluster
import scipy.spatial

# ours
from clusterking.cluster.cluster import Cluster, ClusterResult
from clusterking.util.metadata import failsafe_serialize
from clusterking.maths.metric import metric_selection
from clusterking.util.matplotlib_utils import import_matplotlib


class HierarchyClusterResult(ClusterResult):
    def __init__(self, data, md, clusters, hierarchy):
        super().__init__(data=data, md=md, clusters=clusters)
        self._hierarchy = hierarchy

    @property
    def hierarchy(self):
        return self._hierarchy

    def dendrogram(
        self,
        output: Optional[Union[None, str, pathlib.Path]] = None,
        ax=None,
        show=False,
        **kwargs
    ):
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
        import_matplotlib()
        import matplotlib.pyplot as plt

        if self.hierarchy is None:
            self.log.error(
                "Hierarchy not yet set up. Returning without " "doing anything."
            )
            return

        # do we add to a plot or generate a whole new figure?
        if ax:
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots()

        labelsize = 20
        ax.set_title("Hierarchical Clustering Dendrogram", fontsize=labelsize)
        ax.set_xlabel("ID", fontsize=labelsize)
        ax.set_ylabel("Distance", fontsize=labelsize)

        # set defaults for dendrogram plotting options here
        # (this way we can overwrite them with additional arguments)
        den_config = {
            "color_threshold": "default",
            "leaf_rotation": 90.0,  # rotates the x axis labels
            "leaf_font_size": 8,  # font size for the x axis labels
        }
        den_config.update(kwargs)

        scipy.cluster.hierarchy.dendrogram(self.hierarchy, ax=ax, **den_config)

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


# todo: document
class HierarchyCluster(Cluster):
    def __init__(self):
        super().__init__()

        #: Function that, applied to Data or DWE object returns the metric as
        #: a condensed distance matrix.
        self._metric = None  # type: Callable
        self._fcluster_kwargs = {}

        self.set_metric()
        self.set_hierarchy_options()
        self.set_fcluster_options()

    # Docstring set below
    def set_metric(self, *args, **kwargs) -> None:
        self.md["metric"]["args"] = failsafe_serialize(args)
        self.md["metric"]["kwargs"] = failsafe_serialize(kwargs)
        self._metric = metric_selection(*args, **kwargs)

    set_metric.__doc__ = metric_selection.__doc__

    # todo: should be at least properties
    def set_hierarchy_options(self, method="complete", optimal_ordering=False):
        """ Configure hierarchy building

        Args:
            method: See reference on scipy.cluster.hierarchy.linkage
            optimal_ordering: See reference on scipy.cluster.hierarchy.linkage

        """
        md = self.md["hierarchy"]
        md["method"] = method
        md["optimal_ordering"] = optimal_ordering

    def _build_hierarchy(self, data):

        if self._metric is None:
            msg = (
                "Metric not set. please run self.set_metric or set "
                " self.metric manually before running this method. "
                "Returning without doing anything."
            )
            self.log.critical(msg)
            raise ValueError(msg)

        self.log.debug("Building hierarchy.")

        hierarchy = scipy.cluster.hierarchy.linkage(
            self._metric(data),
            method=self.md["hierarchy"]["method"],
            optimal_ordering=self.md["hierarchy"]["optimal_ordering"],
        )

        self.log.debug("Done")

        return hierarchy

    def set_max_d(self, max_d):
        # todo: make prop
        self.md["max_d"] = max_d

    def set_fcluster_options(self, **kwargs):
        # set up defaults for clustering here
        # (this way we can overwrite them with additional arguments)
        self._fcluster_kwargs = {"criterion": "distance"}
        self._fcluster_kwargs.update(kwargs)
        self.md["fcluster"]["kwargs"] = failsafe_serialize(
            self._fcluster_kwargs
        )

    # todo: Allow reusing of hierarchy
    def _run(
        self,
        data,
        reuse_hierarchy_from: Optional[HierarchyClusterResult] = None,
    ):
        """

        Args:
            data:
            reuse_hierarchy_from: Reuse the hierarchy from a
                :class:`HierarchyClusterResult` object.

        Returns:

        """
        if reuse_hierarchy_from:
            # todo: Perhaps add some consistency checks here to ensure that
            #   only hierarchies are reused that belong to the same data and
            #   to the same worker
            hierarchy = reuse_hierarchy_from.hierarchy
        else:
            hierarchy = self._build_hierarchy(data)

        # noinspection PyTypeChecker
        clusters = scipy.cluster.hierarchy.fcluster(
            hierarchy, self.md["max_d"], **self._fcluster_kwargs
        )

        return HierarchyClusterResult(
            data=data, md=self.md, clusters=clusters, hierarchy=hierarchy
        )

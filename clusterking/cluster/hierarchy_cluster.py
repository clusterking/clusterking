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
    def __init__(self, data, md, clusters, hierarchy, worker_id):
        super().__init__(data=data, md=md, clusters=clusters)
        self._hierarchy = hierarchy
        self._worker_id = worker_id

    @property
    def hierarchy(self):
        return self._hierarchy

    @property
    def worker_id(self):
        """ ID of the HierarchyCluster worker that generated this object. """
        return self._worker_id

    @property
    def data_id(self) -> int:
        """ ID of the data object that the HierarchyCluster worker was run on.
        """
        return id(self._data)

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


class HierarchyCluster(Cluster):
    def __init__(self):
        super().__init__()

        #: Function that, applied to Data or DWE object returns the metric as
        #: a condensed distance matrix.
        self._metric = None  # type: Callable
        #: Keyword arguments to the call of fcluster
        self._fcluster_kwargs = {}

        self.set_metric()
        self.set_hierarchy_options()
        self.set_fcluster_options()

    @property
    def max_d(self) -> Optional[float]:
        """ Cutoff value set in :meth:`set_max_d`. """
        return self.md["max_d"]

    @property
    def metric(self) -> Callable:
        """ Metric that was set in :meth:`set_metric`
        (Function that takes Data object as only parameter and
        returns a reduced distance matrix.) """
        return self._metric

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
            method: See reference on :class:`scipy.cluster.hierarchy.linkage`
            optimal_ordering: See reference on
                :class:`scipy.cluster.hierarchy.linkage`

        """
        md = self.md["hierarchy"]
        md["method"] = method
        md["optimal_ordering"] = optimal_ordering

    def _build_hierarchy(self, data):
        """ Builds hierarchy using :class:`scipy.cluster.hierarchy.linkage` """

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

    def set_max_d(self, max_d) -> None:
        """ Set the cutoff value of the hierarchy that then gives the clusters.
        This corresponds to the ``t`` argument of
        :class:`scipy.cluster.hierarchy.fcluster`.

        Args:
            max_d: float

        Returns:
            None
        """
        self.md["max_d"] = max_d

    def set_fcluster_options(self, **kwargs) -> None:
        """ Set additional keyword options for our call to
        ``scipy.cluster.hierarchy.fcluster``.

        Args:
            kwargs: Keyword arguments

        Returns:
            None
        """
        # set up defaults for clustering here
        # (this way we can overwrite them with additional arguments)
        self._fcluster_kwargs = {"criterion": "distance"}
        self._fcluster_kwargs.update(kwargs)
        self.md["fcluster"]["kwargs"] = failsafe_serialize(
            self._fcluster_kwargs
        )

    def run(
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
        if not self.max_d:
            raise ValueError(
                "Please use set the cutoff value using set_max_d before"
                "running this worker."
            )

        if reuse_hierarchy_from:
            if not id(self) == reuse_hierarchy_from.worker_id:
                raise ValueError(
                    "It seems like the hierarchy you passed comes from a"
                    " different HierarchyCluster object than this one: IDs "
                    "don't match (self: {} vs reuse_hierarchy_from: {})".format(
                        id(self), reuse_hierarchy_from.worker_id
                    )
                )
            if not id(data) == reuse_hierarchy_from.data_id:
                raise ValueError(
                    "It seems like the hierarchy you passed corresponds to a"
                    " different data object than the one you gave me now. "
                    "IDs don't match (passed to me: {} vs "
                    "reuse_hierarchy_from: {})".format(
                        id(data), reuse_hierarchy_from.data_id
                    )
                )
            # Without caching properties of data and cluster class, we can't
            # really check that they weren't modified in place, so this is
            # about all we can do right now.
            hierarchy = reuse_hierarchy_from.hierarchy
        else:
            hierarchy = self._build_hierarchy(data)

        # noinspection PyTypeChecker
        clusters = scipy.cluster.hierarchy.fcluster(
            hierarchy, self.max_d, **self._fcluster_kwargs
        )

        return HierarchyClusterResult(
            data=data,
            md=self.md,
            clusters=clusters,
            hierarchy=hierarchy,
            worker_id=id(self),
        )

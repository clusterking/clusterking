#!/usr/bin/env python3

# std
import pathlib
from typing import Union

# 3rd
import scipy.cluster
import matplotlib.pyplot as plt

# ours
from bclustering.maths.metric import condense_distance_matrix
from bclustering.cluster.cluster import Cluster


# todo: document
class HierarchyCluster(Cluster):
    def __init__(self):
        super().__init__()

        self.hierarchy = None

    # todo: Save hierarchy and give option to load again?
    def build_hierarchy(self, data, metric="euclidean",
                        method="complete", optimal_ordering=False) -> None:
        """ Build the hierarchy object.

        Args:
            metric: Either any of the keywords described on
                scipy.cluster.hierarchy.linkage, or a condensed distance
                matrix as described on scipy.cluster.hierarchy.linkage
            method: See reference on scipy.cluster.hierarchy.linkage
            optimal_ordering: See reference on scipy.cluster.hierarchy.linkage
        """
        self.log.debug("Building hierarchy.")

        md = self.md["hierarchy"]
        if isinstance(metric, str):
            md["metric"] = metric
        else:
            md["metric"] = "custom supplied"
        md["method"] = method
        md["optimal_ordering"] = optimal_ordering

        if isinstance(metric, str):
            # only the q2 bins without any other information in the dataframe
            self.hierarchy = scipy.cluster.hierarchy.linkage(
                data.data(),
                metric=metric,
                method=method,
                optimal_ordering=optimal_ordering
            )
        else:
            if len(metric.shape) == 1:
                pass
            elif len(metric.shape) == 2:
                metric = condense_distance_matrix(metric)
            else:
                raise ValueError("Strange metric matrix dimensions >= 3")

            self.hierarchy = scipy.cluster.hierarchy.linkage(
                metric,
                method=method,
                optimal_ordering=optimal_ordering
            )

        self.log.debug("Done")

    def _cluster(self, data, max_d=0.2, **kwargs):
        """Performs the actual clustering
        Args:
            max_d:
            **kwargs:

        Returns:
            None
        """

        if self.hierarchy is None:
            msg = "Please run build_hierarchy first to set self.hierarchy!"
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

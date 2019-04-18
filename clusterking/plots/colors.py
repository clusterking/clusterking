#!/usr/bin/env python3

# std
import logging

# 3rd party
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

# ours
from clusterking.util.log import get_logger


class ColorScheme(object):
    """ Class holding color scheme. We want to assign a unique color to every
    cluster and keep it consistent accross different plots.
    Subclass and overwrite color lists to implement different schemes.
    """
    def __init__(self, clusters, colors=None):
        self.log = get_logger("Colors", sh_level=logging.WARNING)

        self.clusters = list(clusters)

        self._cluster_colors = None

        self.cluster_colors = [
            "xkcd:light red",
            "xkcd:apple green",
            "xkcd:bright blue",
            "xkcd:charcoal grey",
            "xkcd:orange",
            "xkcd:hot pink"
        ]
        if colors:
            self.cluster_colors = colors

        if len(clusters) > len(self.cluster_colors):
            self.log.warning(
                "Not enough colors for all clusters. Some clusters might end up"
                " with identical colors."
            )

    @property
    def cluster_colors(self):
        return self._cluster_colors

    @cluster_colors.setter
    def cluster_colors(self, value):
        self._cluster_colors = list(map(
            matplotlib.colors.to_rgba,
            value
        ))

    def get_cluster_color(self, cluster):
        """ Try to pick a unique element of a list corresponding to cluster.
        
        Args:
            cluster: Name of cluster

        Returns:
            Element of that list
        """
        if cluster in self.clusters:
            index = self.clusters.index(cluster)
        else:
            self.log.error(
                "Cluster {} is not in the list of clusters. ".format(
                    cluster
                ))
            index = 0

        return self.cluster_colors[index % len(self.cluster_colors)]

    def to_colormap(self, name="MyColorMap"):
        return matplotlib.colors.LinearSegmentedColormap.from_list(
            name,
            list(map(self.get_cluster_color, self.clusters))
        )

    def faded_colormap(self, cluster, nlines, name="MyFadedColorMap", **kwargs):
        colors = self.get_cluster_colors_faded(cluster, nlines, **kwargs)
        return matplotlib.colors.LinearSegmentedColormap.from_list(
            name,
            colors
        )

    def demo(self):
        z = np.array(self.clusters).reshape((1, len(self.clusters)))
        plt.imshow(z, cmap=self.to_colormap())

    def demo_faded(self, cluster, nlines, **kwargs):
        z = np.array(range(nlines)).reshape((1, nlines))
        plt.imshow(z, cmap=self.faded_colormap(cluster, nlines, **kwargs))

    def get_cluster_colors_faded(self, cluster, nlines, max_alpha=0.7,
                                 min_alpha=0.3):
        base_color = self.get_cluster_color(cluster)
        alphas = np.linspace(min_alpha, max_alpha, nlines)
        colors = [
            matplotlib.colors.to_rgba(base_color, alpha)
            for alpha in alphas
        ]
        return colors

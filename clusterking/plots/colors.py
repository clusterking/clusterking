#!/usr/bin/env python3

# std
import logging

# 3rd party
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

# ours
from clusterking.util.log import get_logger


# todo: docs
class ColorScheme(object):
    """ Class holding color scheme. We want to assign a unique color to every
    cluster and keep it consistent accross different plots.
    Subclass and overwrite color lists to implement different schemes.
    """
    def __init__(self, clusters=None, colors=None):
        self.log = get_logger("Colors", sh_level=logging.WARNING)

        self._cluster_colors = None

        # todo: perhaps use this: https://personal.sron.nl/~pault/

        self.cluster_colors = [
            "xkcd:light red",
            "xkcd:apple green",
            "xkcd:bright blue",
            "xkcd:charcoal grey",
            "xkcd:orange",
            "xkcd:purple",
            "xkcd:brick red",
            "xkcd:hunter green",
            "xkcd:marigold",
            "xkcd:darkish blue",
            "xkcd:dirt brown",
            "xkcd:vivid green",
            "xkcd:periwinkle"
        ]
        if colors:
            self.cluster_colors = colors

        if not clusters:
            self.clusters = range(len(self.cluster_colors))
        else:
            self.clusters = list(clusters)

        if len(self.clusters) > len(self.cluster_colors):
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

    def demo_faded(self, cluster=None, nlines=10, **kwargs):
        z = np.array(range(nlines)).reshape((1, nlines))
        plt.imshow(z, cmap=self.faded_colormap(cluster, nlines, **kwargs))

    # todo: perhaps this should just be done in a different way, the faded
    #   colors add little value as far as distinguishability is concerned
    #   and make picking color schemes much harder...
    def get_cluster_colors_faded(self, cluster, nlines, max_alpha=0.7,
                                 min_alpha=0.3):
        base_color = self.get_cluster_color(cluster)
        alphas = np.linspace(min_alpha, max_alpha, nlines)
        colors = [
            matplotlib.colors.to_rgba(base_color, alpha)
            for alpha in alphas
        ]
        return colors

#!/usr/bin/env python3

# std
import logging
from typing import List, Optional

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

    def __init__(
        self,
        clusters: Optional[List[int]] = None,
        colors: Optional[List[str]] = None,
    ):
        """ Initialize `ColorScheme` object.

        Args:
            clusters: List of cluster names
            colors: List of colors
        """
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
            "xkcd:periwinkle",
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
        """ List of colors """
        return self._cluster_colors

    @cluster_colors.setter
    def cluster_colors(self, value):
        self._cluster_colors = list(map(matplotlib.colors.to_rgba, value))

    def get_cluster_color(self, cluster: int):
        """ Returns base color for cluster.

        Args:
            cluster: Name of cluster. Has to be in :attr:`clusters`

        Returns:
            Color
        """
        if cluster in self.clusters:
            index = self.clusters.index(cluster)
        else:
            self.log.error(
                "Cluster {} is not in the list of clusters. ".format(cluster)
            )
            index = 0

        return self.cluster_colors[index % len(self.cluster_colors)]

    def to_colormap(self, name="MyColorMap"):
        """ Returns colormap with color for each cluster. """
        return matplotlib.colors.LinearSegmentedColormap.from_list(
            name, list(map(self.get_cluster_color, self.clusters))
        )

    def faded_colormap(
        self, cluster: int, nlines: int, name="MyFadedColorMap", **kwargs
    ):
        """ Returns colormap for one cluster, including the faded colors.

        Args:
            cluster: Name of cluster
            nlines: Number of shades
            name: Name of colormap
            **kwargs: Arguments for :meth:`get_cluster_colors_faded`

        Returns:
            Colormap
        """
        colors = self.get_cluster_colors_faded(cluster, nlines, **kwargs)
        return matplotlib.colors.LinearSegmentedColormap.from_list(name, colors)

    def demo(self):
        """ Plot the colors for all clusters.

        Returns:
            figure
        """
        z = np.array(self.clusters).reshape((1, len(self.clusters)))
        return plt.imshow(z, cmap=self.to_colormap())

    def demo_faded(self, cluster: Optional[int] = None, nlines=10, **kwargs):
        """ Plot the color shades for different lines corresponding to the same
        cluster

        Args:
            cluster: Name of cluster
            nlines: Number of shades
            **kwargs: Arguments for :meth:`get_cluster_colors_faded`

        Returns:
            figure
        """
        z = np.array(range(nlines)).reshape((1, nlines))
        return plt.imshow(
            z, cmap=self.faded_colormap(cluster, nlines, **kwargs)
        )

    # todo: perhaps this should just be done in a different way, the faded
    #   colors add little value as far as distinguishability is concerned
    #   and make picking color schemes much harder...
    def get_cluster_colors_faded(
        self, cluster: int, nlines: int, max_alpha=0.7, min_alpha=0.3
    ):
        """ Shades of the base color, for cases where we want to draw multiple
        lines for one cluster

        Args:
            cluster: Name of cluster
            nlines: Number of shades
            max_alpha: Maximum alpha value
            min_alpha: Minimum alpha value

        Returns:
            List of colors
        """
        base_color = self.get_cluster_color(cluster)
        alphas = np.linspace(min_alpha, max_alpha, nlines)
        colors = [
            matplotlib.colors.to_rgba(base_color, alpha) for alpha in alphas
        ]
        return colors

    def get_err_color(self, cluster: int):
        """ Get color for error shades.

        Args:
            cluster: Cluster name

        Returns:
            color
        """
        base_color = self.get_cluster_color(cluster)
        return matplotlib.colors.to_rgba(base_color, 0.3)

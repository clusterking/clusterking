#!/usr/bin/env python3

"""Read the results from scan.py and get_clusters them.
"""

# standard
import atexit
import json
import pathlib
import time
from typing import Union, Any

# 3rd party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster
import scipy.cluster.hierarchy

# us
from bclustering.scan import Scanner
from bclustering.util.log import get_logger
from bclustering.util.metadata import nested_dict, git_info
from bclustering.maths.metric import condense_distance_matrix


# todo: allow initializing from either file or directly the dataframe and the metadata
class Cluster(object):
    # **************************************************************************
    # A:  Setup
    # **************************************************************************

    def __init__(self, input_path):
        """ This class is subclassed to implement specific clustering
        algorithms and defines common functions.

        Args:
            input_path: Input path (without extensions) for metadata and
                output from 'scan'
        """
        #: Instance of logging.Logger to write out log messages
        self.log = get_logger("Cluster")

        self.log.info("Input file basename: '{}'.".format(input_path))

        #: The input path
        self.input_path = pathlib.Path(input_path)

        #: Metadata
        self.metadata = nested_dict()
        self._get_scan_metadata()
        self.metadata["cluster"]["git"] = git_info(self.log)
        self.metadata["cluster"]["time"] = time.strftime("%a %_d %b %Y %H:%M",
                                                         time.gmtime())

        #: Dataframe from scanner
        self.df = None
        self._get_scan_data()

        #: Should we wait for plots to be shown?
        self._wait_plots = False
        # call self.close() when this script exits
        atexit.register(self.close)

    def _get_scan_data(self):
        """ Read data from scan.py """
        path = Scanner.data_output_path(self.input_path)
        self.log.debug("Loading scanner data from '{}'.".format(
            path.resolve()))
        with path.open() as data_file:
            self.df = pd.read_csv(data_file)
        self.df.set_index("index", inplace=True)
        self.log.debug("Done.")

    def _get_scan_metadata(self):
        """ Read metadata from scan.py """
        path = Scanner.metadata_output_path(self.input_path)
        self.log.debug("Loading scanner metadata from '{}'.".format(
            path.resolve()))
        with path.open() as metadata_file:
            scan_metadata = json.load(metadata_file)
        self.metadata.update(scan_metadata)
        self.log.debug("Done.")

    # **************************************************************************
    # B:  Cluster
    # **************************************************************************

    def cluster(self, column="cluster", **kwargs):
        """ Performs the clustering. 
        This method is a wrapper around the _cluster implementation in the 
        subclasses. See there for additional arguments. 
        
        Args:
            column: Column to which the get_clusters should be appended.
            
        Returns:
            None
        """
        self.log.info("Performing clustering.")

        # Taking the column name as additional key means that we can easily
        # save the configuration values of different clusterings
        md = self.metadata["cluster"][column]

        for key, value in kwargs.items():
            md[key] = value

        clusters = self._cluster(**kwargs)

        n_clusters = len(set(clusters))
        self.log.info(
            "Clustering resulted in {} get_clusters.".format(n_clusters)
        )
        md["n_clusters"] = n_clusters

        self.df[column] = clusters

        self.rename_clusters_auto(column)

        self.log.info("Done")

    def _cluster(self, **kwargs):
        """ Implmentation of the clustering. Should return an array-like object
        with the cluster number.
        """
        raise NotImplementedError

    # **************************************************************************
    # C:  Utility
    # **************************************************************************

    def data_matrix(self):
        # todo: make flexible
        return self.df[["bin{}".format(i) for i in range(self.metadata["scan"]["dfunction"]["nbins"])]].values

    def rename_clusters(self, old2new, column="cluster", new_column=None):
        """Renames the get_clusters. This also allows to merge several get_clusters 
        by assigning them the same name. 
        
        Args:
            old2new: Dictionary old name -> new name. If no mapping is defined
                for a key, it remains unchanged.
            column: The column with the original cluster numbers. 
            new_column: Write out as a new column with name `new_columns`, 
                e.g. when merging get_clusters with this method
        """
        clusters_old_unique = self.df[column].unique()
        # If a key doesn't appear in old2new, this means we don't change it.
        for cluster in clusters_old_unique:
            if cluster not in old2new:
                old2new[cluster] = cluster
        self.rename_clusters_apply(
            lambda name: old2new[name],
            column,
            new_column
        )

    def rename_clusters_apply(self, funct, column="cluster", new_column=None):
        """Apply method to cluster names. 
        
        Example:  Suppose your get_clusters are numbered from 1 to 10, but you
        want to start counting at 0:
        
        .. code-block:: python
            
            self.rename_clusters_apply(lambda i: i-1)
        
        Args:
            funct: Function to be applied to each cluster name (taking one 
                argument)
            column: The column with the original cluster numbers. 
            new_column: Write out as a new column with new name
            
        Returns:
            None
        """
        if not new_column:
            new_column = column
        self.df[new_column] = \
            [funct(cluster) for cluster in self.df[column].values]

    def rename_clusters_auto(self, column="cluster", new_column=None):
        """Try to name get_clusters in a way that doesn't depend on the 
        clustering algorithm (e.g. hierarchy clustering assigns names from 1 
        to n, whereas other cluster methods assign names from 0, etc.). 
        Right now, we simply change the names of the get_clusters in such a 
        way, that they are numbered from 0 to n-1 in an 'ascending' way with 
        respect to the order of rows in the dataframe. 
        
        Args:
            column: Column containing the cluster names
            new_column: Write out as a new column with new name
            
        Returns:
            None
        """
        old_cluster_names = self.df[column].unique()
        new_cluster_names = range(len(old_cluster_names))
        old2new = dict(zip(old_cluster_names, new_cluster_names))
        self.rename_clusters(old2new, column, new_column)

    # **************************************************************************
    # D:  Write out
    # **************************************************************************

    @staticmethod
    def data_output_path(general_output_path: Union[pathlib.Path, str]) \
            -> pathlib.Path:
        """ Taking the general output path, return the path to the data file.
        """
        path = pathlib.Path(general_output_path)
        # noinspection PyTypeChecker
        return path.parent / (path.name + "_data.csv")

    @staticmethod
    def metadata_output_path(general_output_path: Union[pathlib.Path, str]) \
            -> pathlib.Path:
        """ Taking the general output path, return the path to the 
        metadata file.
        """
        path = pathlib.Path(general_output_path)
        # noinspection PyTypeChecker
        return path.parent / (path.name + "_metadata.json")

    def write(self, general_output_path):
        """ Write out all results.
        IMPORTANT NOTE: All output files will always be overwritten!

        Args:
            general_output_path: Path to the output file without file 
                extension. We will add suffixes and file extensions to this!
        """

        # *** 1. Clean files and make sure the folders exist ***

        metadata_path = self.metadata_output_path(general_output_path)
        data_path = self.data_output_path(general_output_path)

        self.log.info("Will write metadata to '{}'.".format(metadata_path))
        self.log.info("Will write data to '{}'.".format(data_path))

        paths = [metadata_path, data_path]
        for path in paths:
            if not path.parent.is_dir():
                self.log.debug("Creating directory '{}'.".format(path.parent))
                path.parent.mkdir(parents=True)
            if path.exists():
                self.log.debug("Removing file '{}'.".format(path))
                path.unlink()

        # *** 2. Write out metadata ***

        self.log.debug("Converting metadata data to json and writing to file "
                       "'{}'.".format(metadata_path))
        with metadata_path.open("w") as metadata_file:
            json.dump(self.metadata, metadata_file, sort_keys=True, indent=4)
        self.log.debug("Done.")

        # *** 3. Write out data ***

        self.log.debug("Converting data to csv and writing to "
                       "file '{}'.".format(data_path))
        if self.df.empty:
            self.log.error("Dataframe seems to be empty. Still writing "
                           "out anyway.")
        with data_path.open("w") as data_file:
            self.df.to_csv(data_file)
        self.log.debug("Done")

        # *** 4. Done ***

        self.log.info("Writing out finished.")

    # **************************************************************************
    # E:  MISC
    # **************************************************************************

    def close(self):
        """This method is called when this script exits. A corresponding
        hook has been set up in the __init__ method.
        We use that to wait for interactive plots/plotting windows to close
        if we made any. """
        if self._wait_plots:
            # this will block until all plotting windows were closed
            plt.show()


class HierarchyCluster(Cluster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hierarchy = None

    # todo: Save hierarchy and give option to load again?
    def build_hierarchy(self, metric: Any = "euclidean",
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

        md = self.metadata["cluster"]["hierarchy"]
        if isinstance(metric, str):
            md["metric"] = metric
        else:
            md["metric"] = "custom supplied"
        md["method"] = method
        md["optimal_ordering"] = optimal_ordering

        if isinstance(metric, str):
            nbins = self.metadata["scan"]["dfunction"]["nbins"]
            # only the q2 bins without any other information in the dataframe
            data = self.df[["bin{}".format(i) for i in range(nbins)]]

            self.hierarchy = scipy.cluster.hierarchy.linkage(
                data,
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

    def _cluster(self, max_d=0.2, **kwargs):
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
            output: Union[None, str, pathlib.Path]=None,
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

            # Trigger a plt.show() at the end of this script
            self._wait_plots = True

        if output:
            output = pathlib.Path(output)
            if not output.parent.is_dir():
                self.log.debug("Creating dir '{}'.".format(output.parent))
                output.parent.mkdir(parents=True)
            fig.savefig(output, bbox_inches="tight")
            self.log.info("Wrote dendrogram to '{}'.".format(output))

        return ax


class KmeansCluster(Cluster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _cluster(self, **kwargs):
        kmeans = sklearn.cluster.KMeans(**kwargs)
        bin_columns = [col for col in self.df.columns if col.startswith("bin")]
        x_matrix = np.array(self.df[bin_columns].astype(float))
        kmeans.fit(x_matrix)
        return kmeans.predict(x_matrix)


# paths = [c.metadata_output_path(args.output_path),
#          c.data_output_path(args.output_path)]
# existing_paths = [path for path in paths if path.exists()]
# if existing_paths:
#     agree = yn_prompt(
#         "Output paths {} already exist(s) and will be "
#         "overwritten. "
#         "Proceed?".format(', '.join(map(str, existing_paths)))
#     )
#     if not agree:
#         c.log.critical("User abort.")
#         sys.exit(15)

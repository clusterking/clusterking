#!/usr/bin/env python3

# std
from typing import Optional, Callable, Dict
import collections

# 3rd
import tqdm
import pandas as pd

# ours
from clusterking.stability.ccfom import CCFOM


class SubSampleStabilityTesterResult(object):
    def __init__(self, df: pd.DataFrame):
        #: Results as :class:`pandas.DataFrame`
        self.df = df


class SubSampleStabilityTester(object):
    """ Test the stability of clustering algorithms by repeatedly
    clustering subsamples of data and then comparing if the clusters match.

    """

    def __init__(self):
        #: Fraction of sample points to be contained in the subsamples.
        #: Set using :meth:`set_basic_config`.
        self._fraction = None
        #: Number of subsamples to consider.
        #: Set using :meth:`set_basic_config`.
        self._repeat = None
        #: Figure of merits to calculate as dictionary name: CCFOM
        self._foms = {}  # type: Dict[str, CCFOM]
        #: Display a progress bar?
        self._progress_bar = True

        # Set default values:
        self.set_fraction()
        self.set_repeat()
        self.set_progress_bar()

    # **************************************************************************
    # Config
    # **************************************************************************

    def set_fraction(self, fraction=0.75) -> None:
        """ Basic configuration

        Args:
            fraction: Fraction of sample points to be contained in the subsamples

        Returns:
            None
        """
        assert 0 <= fraction <= 1
        self._fraction = fraction

    def set_repeat(self, repeat=100) -> None:
        """

        Args:
            repeat: Number of subsamples to test

        Returns:
            None
        """
        self._repeat = repeat

    def add_fom(self, fom) -> None:
        """
        """
        if fom.name in self._foms:
            # todo: do with log
            print(
                "Warning: FOM with name {} already existed. Replacing.".format(
                    fom.name
                )
            )
        self._foms[fom.name] = fom

    def set_progress_bar(self, state=True) -> None:
        """ Set or unset progress bar.

        Args:
            state: Bool: Display progress bar?

        Returns:
            None
        """
        self._progress_bar = state

    # **************************************************************************
    # Run
    # **************************************************************************

    def run(self, data, cluster):
        """ Run test.

        Args:
            data: `~clusterking.data.Data` object
            cluster: Pre-configured `~clusterking.cluster.Cluster`
                object

        Returns:
            :class:`~clusterking.stability.subsamplestability.SubSampleStabilityTesterResult` object
        """
        original_clusters = cluster.run(data).get_clusters(indexed=True)
        if self._progress_bar:
            iterator = tqdm.tqdm(range(self._repeat))
        else:
            iterator = range(self._repeat)
        fom_results = collections.defaultdict(list)
        for i in iterator:
            r = cluster.run(data.sample_param_random(frac=self._fraction))
            subsample_clusters = r.get_clusters(indexed=True)
            for fom_name, fom in self._foms.items():
                fom = fom.run(original_clusters, subsample_clusters).fom
                fom_results[fom_name].append(fom)

        df = pd.DataFrame(fom_results)
        return SubSampleStabilityTesterResult(df=df)


class SubSampleStabilityVsFractionResult(object):
    def __init__(self, df: pd.DataFrame):
        #: Results as :class:`pandas.DataFrame`
        self.df = df


class SubSampleStabilityVsFraction(object):
    """ Repeatedly run :class:`SubSampleStabilityTester` for different
    fractions.
    """

    def __init__(self):
        pass

    def run(self, data, cluster, ssst, fractions):
        results = collections.defaultdict(list)
        ssst.set_progress_bar(False)
        for fract in tqdm.tqdm(fractions):
            ssst.set_fraction(fract)
            r = ssst.run(data, cluster)
            foms = r.df.mean().to_dict()
            results["fraction"].append(fract)
            for key, value in foms.items():
                results[key].append(value)
        df = pd.DataFrame(results)
        return SubSampleStabilityVsFractionResult(df=df)

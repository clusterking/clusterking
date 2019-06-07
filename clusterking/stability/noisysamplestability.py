#!/usr/bin/env python3

# std
import copy
import collections

# 3rd
import pandas as pd
import tqdm

# ours
from clusterking.worker import AbstractWorker
from clusterking.result import AbstractResult


class NoisySampleStabilityTesterResult(AbstractResult):
    def __init__(self, df, cached_data=None):
        super().__init__()
        self.df = df
        self._cached_data = cached_data


class NoisySampleStabilityTester(AbstractWorker):
    def __init__(self):
        super().__init__()
        self._noise_kwargs = {}
        self._noise_args = []
        self._repeat = 10
        self._cache_data = True
        self.set_repeat()
        self._foms = {}

    # **************************************************************************
    # Config
    # **************************************************************************

    def set_repeat(self, repeat=10):
        self._repeat = repeat

    def set_noise(self, *args, **kwargs):
        self._noise_args = args
        self._noise_kwargs = kwargs

    def set_cache_data(self, value):
        self._cache_data = value

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

    # **************************************************************************
    # Run
    # **************************************************************************

    def run(self, data, scanner, cluster):
        scanner.set_progress_bar(False)
        datas = []
        original_data = None
        fom_results = collections.defaultdict(list)
        for _ in tqdm.tqdm(range(self._repeat + 1)):
            noisy_scanner = copy.copy(scanner)
            if _ >= 1:
                noisy_scanner.add_spoints_noise(
                    *self._noise_args, **self._noise_kwargs
                )
            this_data = data.copy(deep=True)
            if self._cache_data:
                datas.append(this_data)
            rs = noisy_scanner.run(this_data)
            rs.write()
            rc = cluster.run(this_data)
            rc.write()
            if _ == 0:
                original_data = this_data.copy(deep=True)
                continue
            for fom_name, fom in self._foms.items():
                fom = fom.run(original_data, this_data).fom
                fom_results[fom_name].append(fom)
        return NoisySampleStabilityTesterResult(
            df=pd.DataFrame(fom_results), cached_data=datas
        )

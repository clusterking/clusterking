#!/usr/bin/env python3

# std
from pathlib import Path
import pytest

# ours
from clusterking.data.data import Data
from clusterking.cluster.hierarchy_cluster import HierarchyCluster


@pytest.fixture
def _data():
    ddir = Path(__file__).parent / "data"
    dname = "1d.sql"
    d = Data(ddir / dname)
    return d


def test_cluster(_data):
    d = _data.copy()
    c = HierarchyCluster()
    c.set_metric("euclidean")
    c.set_max_d(0.75)
    c.run(d).write()
    c.set_max_d(1.5)
    c.run(d).write(cluster_column="cluster15")
    # The minimal distance between our distributions is 1, so they all
    # end up in different clusters
    assert len(d.clusters()) == d.n
    # This is a bit unfortunate, since we have so many distribution pairs
    # with equal distance (so it's up to the implementation of the algorithm
    # , which clusters develop) but this is what happened so far:
    assert len(d.clusters(cluster_column="cluster15")) == 6


def test_reuse_hierarchy(_data):
    d = _data.copy()
    c = HierarchyCluster()
    c.set_metric("euclidean")
    c.set_max_d(1.5)
    r = c.run(d)
    r.write()
    r2 = c.run(d, reuse_hierarchy_from=r)
    r2.write(cluster_column="reused")
    assert d.df["cluster"].tolist() == d.df["reused"].tolist()


def test_reuse_hierarchy_fail_different_data(_data):
    d = _data.copy()
    e = _data.copy()
    c = HierarchyCluster()
    c.set_metric("euclidean")
    c.set_max_d(1.5)
    r = c.run(d)
    r.write()
    with pytest.raises(ValueError, match=".*different data object.*"):
        c.run(e, reuse_hierarchy_from=r)


def test_reuse_hierarchy_fail_different_cluster(_data):
    d = _data.copy()
    c = HierarchyCluster()
    c2 = HierarchyCluster()
    c.set_metric("euclidean")
    c.set_max_d(1.5)
    c2.set_metric("euclidean")
    c2.set_max_d(1.5)
    r = c.run(d)
    r.write()
    with pytest.raises(
        ValueError, match=".*different HierarchyCluster object.*"
    ):
        c2.run(d, reuse_hierarchy_from=r)


def test_hierarchy_cluster_no_max_d(_data):
    d = _data.copy()
    c = HierarchyCluster()
    with pytest.raises(ValueError, match=".*set_max_d.*"):
        c.run(d)


def test_dendrogram_plot(_data, tmp_path):
    c = HierarchyCluster()
    c.set_metric()
    c.set_max_d(0.2)
    r = c.run(_data)
    r.dendrogram(output=str(tmp_path / "output.pdf"))

#!/usr/bin/env python3

# std
from pathlib import Path
import tempfile
import unittest

# ours
from clusterking.util.testing import MyTestCase
from clusterking.data.dfmd import DFMD


class TestDFMD(MyTestCase):
    def setUp(self):
        self.data_dir = Path(__file__).parent / "data"
        self.test_fname = "test.sql"

    def test_init_empty(self):
        DFMD()

    def _test_dfmd_vs_cached(self, dfmd):
        self.assertListEqual(
            list(dfmd.df.columns),
            [
                "CVL_bctaunutau",
                "CT_bctaunutau",
                "CSL_bctaunutau",
                "bin0",
                "bin1",
                "cluster",
                "other_cluster",
                "bpoint",
                "other_bpoint"
            ]
        )
        self.assertEqual(
            len(dfmd.df),
            2
        )

    def _compare_dfs(self, dfmd1, dfmd2):
        self.assertListEqual(
            sorted(list(dfmd1.md.keys())), sorted(list(dfmd2.md.keys()))
        )
        self.assertListEqual(
            list(dfmd1.df.columns),
            list(dfmd2.df.columns),
        )

    def test_init_dir_name(self):
        dfmd = DFMD(self.data_dir / self.test_fname)
        self._test_dfmd_vs_cached(dfmd)

    # todo: implement working tests for copying
    # def test_shallow_copy(self):
    #     dfmd1 = DFMD(self.data_dir, "test_scan")
    #     dfmd2 = dfmd1.copy(False)
    #     dfmd3 = copy.copy(dfmd1)
    #     self.assertEqual(id(dfmd1.df), id(dfmd2.df))
    #     self.assertEqual(id(dfmd1.md), id(dfmd2.md))
    #     self.assertEqual(id(dfmd1.df), id(dfmd3.df))
    #     self.assertEqual(id(dfmd1.md), id(dfmd3.md))
    #
    # def test_deep_copy(self):
    #     # Note: hash(str(df.values)) only corresponds to comparing some of the
    #     # entries.
    #     dfmd1 = DFMD(self.data_dir, "test_scan")
    #     dfmd2 = dfmd1.copy()
    #     dfmd3 = copy.deepcopy(dfmd1)
    #     self.assertNotEqual(id(dfmd1.df), id(dfmd2.df))
    #     self.assertNotEqual(id(dfmd1.md), id(dfmd2.md))
    #     self.assertNotEqual(id(dfmd1.df), id(dfmd3.df))
    #     self.assertNotEqual(id(dfmd1.md), id(dfmd3.md))

    def test_write_read(self):
        dfmd = DFMD(self.data_dir / self.test_fname)
        with tempfile.TemporaryDirectory() as tmpdir:
            dfmd.write(Path(tmpdir) / "tmp_test.sql")
            dfmd_loaded = DFMD(Path(tmpdir) / "tmp_test.sql")
            self._compare_dfs(dfmd, dfmd_loaded)


if __name__ == "__main__":
    unittest.main()

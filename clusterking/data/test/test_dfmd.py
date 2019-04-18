#!/usr/bin/env python3

# std
from pathlib import Path
import tempfile
import unittest

# ours
from clusterking.util.log import silence_all_logs
from clusterking.data.dfmd import DFMD


class TestDFMD(unittest.TestCase):
    def setUp(self):
        silence_all_logs()
        self.data_dir = Path(__file__).parent / "data"
        self.test_fname = "test"

    def test_init_empty(self):
        DFMD()

    def test_get_paths(self):
        dfmd = DFMD()
        self.assertEqual(
            str(dfmd.get_md_path("dir", "name")),
            str(Path("dir/name_metadata.json"))
        )
        self.assertEqual(
            str(dfmd.get_df_path("dir", "name")),
            str(Path("dir/name_data.csv"))
        )

    def test_init_dfmd(self):
        _dfmd = DFMD()
        _dfmd.md = 1
        _dfmd.df = 2
        _dfmd.log = 3

        dfmd = DFMD(_dfmd)
        self.assertEqual(dfmd.md, 1)
        self.assertEqual(dfmd.df, 2)
        self.assertEqual(dfmd.log, 3)

        # with keyword
        dfmd = DFMD(dfmd=_dfmd)
        self.assertEqual(dfmd.md, 1)
        self.assertEqual(dfmd.df, 2)
        self.assertEqual(dfmd.log, 3)

    def _test_dfmd_vs_cached(self, dfmd):
        self.assertEqual(
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
        self.assertEqual(
            list(dfmd1.md.keys()), list(dfmd2.md.keys())
        )
        self.assertEqual(
            list(dfmd1.df.columns),
            list(dfmd2.df.columns),
        )

    def test_init_dir_name(self):
        dfmd = DFMD(self.data_dir, self.test_fname)
        self._test_dfmd_vs_cached(dfmd)
        dfmd = DFMD(directory=self.data_dir, name=self.test_fname)
        self._test_dfmd_vs_cached(dfmd)

    def test_init_df_md(self):
        _dfmd = DFMD(self.data_dir, self.test_fname)
        dfmd = DFMD(df=_dfmd.df, md=_dfmd.md)
        self._compare_dfs(_dfmd, dfmd)

    def test_init_df_path_md_path(self):
        dfmd = DFMD(
            df=DFMD.get_df_path(self.data_dir, self.test_fname),
            md=DFMD.get_md_path(self.data_dir, self.test_fname)
        )
        self._test_dfmd_vs_cached(dfmd)

    def test_init_mixed(self):
        _dfmd = DFMD(self.data_dir, self.test_fname)
        dfmd = DFMD(
            df=DFMD.get_df_path(self.data_dir, self.test_fname),
            md=_dfmd.md
        )
        self._compare_dfs(_dfmd, dfmd)
        dfmd = DFMD(
            df=_dfmd.df,
            md=DFMD.get_md_path(self.data_dir, self.test_fname)
        )
        self._compare_dfs(_dfmd, dfmd)

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
        dfmd = DFMD(self.data_dir, self.test_fname)
        with tempfile.TemporaryDirectory() as tmpdir:
            dfmd.write(tmpdir, "tmp_test")
            dfmd_loaded = DFMD(tmpdir, "tmp_test")
            self._compare_dfs(dfmd, dfmd_loaded)


if __name__ == "__main__":
    unittest.main()

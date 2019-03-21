#!/usr/bin/env python3

# std
import unittest
from pathlib import Path

# ours
from bclustering.data.dfmd import DFMD


# todo: disable logging output
class TestDFMD(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).parent / "data"
        pass

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
                "bin1"
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
        dfmd = DFMD(self.data_dir, "test_scan")
        self._test_dfmd_vs_cached(dfmd)
        dfmd = DFMD(directory=self.data_dir, name="test_scan")
        self._test_dfmd_vs_cached(dfmd)

    def test_init_df_md(self):
        _dfmd = DFMD(self.data_dir, "test_scan")
        dfmd = DFMD(df=_dfmd.df, md=_dfmd.md)
        self._compare_dfs(_dfmd, dfmd)

    def test_init_df_path_md_path(self):
        dfmd = DFMD(
            df=self.data_dir / "test_scan_data.csv",
            md=self.data_dir / "test_scan_metadata.json"
        )
        self._test_dfmd_vs_cached(dfmd)

    def test_init_mixed(self):
        _dfmd = DFMD(self.data_dir, "test_scan")
        dfmd = DFMD(
            df=self.data_dir / "test_scan_data.csv",
            md=_dfmd.md
        )
        self._compare_dfs(_dfmd, dfmd)
        dfmd = DFMD(
            df=_dfmd.df,
            md=self.data_dir / "test_scan_metadata.json"
        )
        self._compare_dfs(_dfmd, dfmd)


if __name__ == "__main__":
    unittest.main()
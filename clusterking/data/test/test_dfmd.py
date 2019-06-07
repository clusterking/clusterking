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
        path = Path(__file__).parent / "data" / "test.sql"
        self.dfmd = DFMD(path)

    def ndfmd(self):
        return self.dfmd.copy(deep=True)

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
                "other_bpoint",
            ],
        )
        self.assertEqual(len(dfmd.df), 2)

    def _compare_dfs(self, dfmd1, dfmd2):
        self.assertListEqual(
            sorted(list(dfmd1.md.keys())), sorted(list(dfmd2.md.keys()))
        )
        self.assertListEqual(list(dfmd1.df.columns), list(dfmd2.df.columns))

    def test_init_dir_name(self):
        dfmd = self.ndfmd()
        self._test_dfmd_vs_cached(dfmd)

    def test_shallow_copy(self):
        dfmd1 = self.ndfmd()
        dfmd2 = dfmd1.copy(deep=False)
        self.assertTrue(dfmd1.df.equals(dfmd2.df))
        self.assertDictEqual(dfmd1.md, dfmd2.md)

    def test_deep_copy(self):
        dfmd1 = self.ndfmd()
        dfmd2 = dfmd1.copy(deep=True)
        self.assertTrue(dfmd1.df.equals(dfmd2.df))
        self.assertDictEqual(dfmd1.md, dfmd2.md)
        dfmd2.md["TESTTEST"] = "modified"
        self.assertFalse(dfmd1.md["TESTTEST"])

    def test_write_read(self):
        dfmd = self.ndfmd()
        with tempfile.TemporaryDirectory() as tmpdir:
            dfmd.write(Path(tmpdir) / "tmp_test.sql")
            dfmd_loaded = DFMD(Path(tmpdir) / "tmp_test.sql")
            self._compare_dfs(dfmd, dfmd_loaded)

    def test_handle_overwrite(self):
        dfmd = DFMD()
        dfmd2 = self.ndfmd()
        with tempfile.TemporaryDirectory() as tmpdir:
            dfmd.write(Path(tmpdir) / "test.sql")
            with self.assertRaises(FileExistsError):
                dfmd.write(Path(tmpdir) / "test.sql", overwrite="raise")
            dfmd2.write(Path(tmpdir) / "test.sql", overwrite="overwrite")
            dfmdx = DFMD(Path(tmpdir) / "test.sql")
            self.assertGreaterEqual(len(dfmdx.df), 2)

    def test_write_new_dir(self):
        dfmd = DFMD()
        with tempfile.TemporaryDirectory() as tmpdir:
            fn = Path(tmpdir) / "a" / "b" / "c"
            dfmd.write(fn)
            self.assertTrue(fn.is_file())


if __name__ == "__main__":
    unittest.main()
